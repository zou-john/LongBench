"""
Transformer Parameter Explorer for LLaMA 3.1 8B Instruct
=========================================================
10 tests that show everything you can inspect and tweak during inference.
Each test is independent — comment out what you don't need.

Requirements:
    pip install transformers torch matplotlib seaborn numpy accelerate

Note: Needs GPU with ~16GB+ VRAM. Set attn_implementation="eager" to get
      raw attention weights (slower but lets you see everything).
"""

from linecache import cache
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings, os, time
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
OUTPUT_DIR = "explorer_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT = "Explain how memory works in the human brain."

# Dark plot style
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",
    "text.color": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "axes.labelcolor": "#e6edf3",
    "figure.dpi": 120,
    "font.family": "monospace",
})


# ============================================================================
# HELPERS
# ============================================================================
def load_model():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto",
        # "eager" gives us raw attention weights we can inspect.
        # "sdpa" or "flash_attention_2" are faster but hide the weights.
        attn_implementation="eager",
    )
    model.eval()
    print(f"Loaded: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_attention_heads} Q-heads, "
          f"{model.config.num_key_value_heads} KV-heads, "
          f"{model.config.hidden_size} hidden dim\n")
    return model, tokenizer


def prepare_input(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {k: v.to(DEVICE) for k, v in inputs.items()}


def decode_tokens(tokenizer, token_ids):
    return [tokenizer.decode(t, skip_special_tokens=False).strip() or "·"
            for t in token_ids]


def save_fig(name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(path, bbox_inches="tight", dpi=150, facecolor="#0d1117")
    plt.close()
    print(f"  → Saved: {path}")


# ============================================================================
# TEST 1: ARCHITECTURE — what's fixed after training
# ============================================================================
def test_architecture(model, tokenizer):
    print("=" * 70)
    print("TEST 1: ARCHITECTURE (all fixed — cannot change at inference)")
    print("=" * 70)

    c = model.config
    kv_ratio = c.num_attention_heads // c.num_key_value_heads
    head_dim = c.hidden_size // c.num_attention_heads

    print(f"""
  vocab_size:             {c.vocab_size:,}      ← size of token dictionary
  hidden_size:            {c.hidden_size}       ← dimension of each token's representation
  num_hidden_layers:      {c.num_hidden_layers}         ← depth of the network
  num_attention_heads:    {c.num_attention_heads}         ← query heads per layer
  num_key_value_heads:    {c.num_key_value_heads}          ← KV heads per layer (GQA)
  head_dim:               {head_dim}        ← hidden_size / num_attention_heads
  intermediate_size:      {c.intermediate_size:,}     ← FFN hidden size (usually ~4x hidden)
  max_position_embeddings:{c.max_position_embeddings:,}   ← hard context window limit
  rope_theta:             {getattr(c, 'rope_theta', 'N/A')}  ← RoPE base frequency

  GQA: {kv_ratio} query heads share each KV head.
       This means the KV cache is {kv_ratio}x smaller than if every head had its own KV.
       But attention patterns differ per query head even with shared KV.
""")


# ============================================================================
# TEST 2: KV CACHE — shape, memory, growth
# ============================================================================
def test_kv_cache(model, tokenizer):
    print("=" * 70)
    print("TEST 2: KV CACHE — the effective context window at inference")
    print("=" * 70)

    inputs = prepare_input(tokenizer, PROMPT)

    seq_len = inputs["input_ids"].shape[1]
    print(f"\n  Input tokens: {seq_len}")

    input_ids = inputs['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    for i, (id, tok) in enumerate(zip(input_ids.tolist(), tokens)):
        print(f"  {i:3d}  {id:6d}  {repr(tok)}")

    with torch.no_grad():
        out = model(**inputs, use_cache=True, return_dict_in_generate=True)
    cache = out.past_key_values
    layer0 = cache.layers[0]
    print("\n  Class:", type(layer0))
    print(f"\n  Attributes and Methods of {type(layer0)}:", dir(layer0))

    # NOTE:
    # torch.Size([1, 8, N, 128])
    #   - 1 — batch size (one sequence)
    #   - 8 — number of KV heads (Llama 3.1 8B uses GQA with 8 KV heads vs 32
    #   query heads)                                                             
    #   - N — max_cache_len (input tokens + new tokens)                                                    
    #   - 128 — head dimension (4096 hidden_size / 32 heads = 128)
    print(layer0.keys.shape)
    print(layer0.values.shape)

    # NOTE:
    # Current KV cache for KV Cache Size of 179 tokens
    # 1 × 8 × 179 × 128 = 183,296  elements per tensor
    # × 2 (keys + values)  = 366,592
    # × 32 layers          = 11,730,944 elements
    # × 2 bytes (bfloat16) = 23,461,888 bytes
    # ÷ 1024 ÷ 1024        = 22.375 MB
    total_bytes = sum(
        layer.keys.nelement() * layer.keys.element_size() +
        layer.values.nelement() * layer.values.element_size()                
        for layer in cache.layers
    )                                                                        
    print(f"Current KV cache: {total_bytes / 1024 / 1024:.2f} MB for {seq_len} tokens")

    # watch it grow during generation
    print(f"Cache growth during generation:")

    past = cache
    print("Output Logit Shape:", out.logits.shape)
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    for step in range(5):
        with torch.no_grad():                                            
            out2 = model(next_id, past_key_values=past, use_cache=True) # context with the current word
        past = out2.past_key_values                                      
        next_id = out2.logits[:, -1, :].argmax(dim=-1, keepdim=True)     
        cl = past.get_seq_length()                                       
        tok = tokenizer.decode(next_id[0])
        print(f"    step {step+1}: cache has {cl} KV pairs | generated: '{tok}'")



# ============================================================================
# TEST 3: ATTENTION HEATMAPS — what tokens attend to what
# ============================================================================
def test_attention_heatmaps(model, tokenizer):
    print("=" * 70)
    print("TEST 3: ATTENTION HEATMAPS")
    print("=" * 70)

    inputs = prepare_input(tokenizer, PROMPT)
    token_ids = inputs["input_ids"][0].tolist()
    labels = decode_tokens(tokenizer, token_ids)
    show_labels = len(labels) <= 35

    with torch.no_grad():
        out = model(**inputs, output_attentions=True, use_cache=False)

    attns = out.attentions
    n_layers = len(attns) # 32 layers
    print(attns[0].shape) # each layer has shape (batch=1, num_heads=32, seq_len, seq_len)
    n_heads = attns[0].shape[1]
    print(f"  {n_layers} layers × {n_heads} heads, seq_len={len(token_ids)}")

    # --- Plot A: 4 layers × 4 heads grid ---
    layer_picks = [0, n_layers // 4, n_layers // 2, n_layers - 1]
    head_picks = [0, 1, n_heads // 2, n_heads - 1]

    fig, axes = plt.subplots(4, 4, figsize=(20, 18), facecolor="#0d1117")
    fig.suptitle("Attention Patterns: Layers × Heads\n(row=query token, col=key token)",
                 color="#e6edf3", fontsize=14, y=0.99)

    for r, li in enumerate(layer_picks):
        for c, hi in enumerate(head_picks):
            ax = axes[r][c]
            a = attns[li][0, hi].float().cpu().numpy() # extracts a single seq_len x seq_len attention matrix
            sns.heatmap(a, ax=ax, cmap="magma", vmin=0, cbar=False, square=True,
                        xticklabels=labels if show_labels else False,
                        yticklabels=labels if show_labels else False)
            ax.set_title(f"L{li} H{hi}", color="#e6edf3", fontsize=10)
            ax.tick_params(labelsize=5, colors="#8b949e")
            ax.set_facecolor("#0d1117")

    plt.tight_layout()
    save_fig("01_attention_layers_heads")
    
    # --- Plot B: Head-averaged attention at 8 layers ---
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), facecolor="#0d1117")
    samples = np.linspace(0, n_layers - 1, 8, dtype=int)

    for idx, li in enumerate(samples):
        ax = axes[idx // 4][idx % 4]
        avg = attns[li][0].float().mean(dim=0).cpu().numpy()
        sns.heatmap(avg, ax=ax, cmap="inferno", vmin=0, cbar=False, square=True,
                    xticklabels=labels if show_labels else False,
                    yticklabels=labels if show_labels else False)
        ax.set_title(f"Layer {li} (head-avg)", color="#e6edf3", fontsize=10)
        ax.tick_params(labelsize=5, colors="#8b949e")
        ax.set_facecolor("#0d1117")

    fig.suptitle("Head-Averaged Attention Per Layer", color="#e6edf3", fontsize=14, y=0.99)
    plt.tight_layout()
    save_fig("02_attention_avg_layers")
    
    # --- Plot C: Attention sinks bar chart ---
    received = torch.zeros(len(token_ids))
    for la in attns:
        received += la[0].float().sum(dim=(0, 1)).cpu()
    # NOTE:
    # [seq_len_key] — one number per key token, representing
    # the total attention poured into it from every head and every query
    # position combined.
    
    fig, ax = plt.subplots(figsize=(14, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    colors = plt.cm.magma(received.numpy() / received.max().item())
    ax.bar(range(len(token_ids)), received.numpy(), color=colors, width=1.0)
    if show_labels:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("Total Attention Received")
    ax.set_title("Attention Sinks (summed across all layers & heads)", color="#e6edf3")
    plt.tight_layout()
    save_fig("03_attention_sinks")

    top5 = received.argsort(descending=True)[:5]
    print(f"\n  Top 5 attention sinks:")
    for i, idx in enumerate(top5):
        print(f"    {i+1}. '{labels[idx]}' (pos {idx.item()}) → {received[idx]:.1f}")
    print()


# ============================================================================
# TEST 4: KV CACHE MANIPULATION — modify cache, see output change
# ============================================================================
def test_kv_cache_manipulation(model, tokenizer):
    print("=" * 70)
    print("TEST 4: KV CACHE MANIPULATION — evict tokens, see what happens")
    print("  This is exactly what BumbleBee does, just with smarter selection.")
    print("=" * 70)

    inputs = prepare_input(tokenizer, PROMPT)
    MAX_GEN_TOK = 40

    def gen_from_cache(cache, label):
        gen_id = inputs["input_ids"][:, -1:]
        past = cache
        ids = []
        for _ in range(MAX_GEN_TOK):
            with torch.no_grad():
                o = model(gen_id, past_key_values=past, use_cache=True)
            past = o.past_key_values
            gen_id = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids.append(gen_id.item())
            if gen_id.item() == tokenizer.eos_token_id:
                break
        txt = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"  [{label}]")
        print(f"    {txt[:200]}\n")

    with torch.no_grad():
        out = model(**inputs, use_cache=True, output_attentions=True)
    full_cache = out.past_key_values
    n = full_cache.layers[0].keys.shape[2] # (batch_size x num of KV heads x max_cache_len x head dimension)
    budget = n // 2
    print(f"\n  Full cache: {n} tokens → compressing to {budget}\n")

    # A) Baseline
    gen_from_cache(full_cache, f"FULL CACHE ({n} tokens)")

    # B) Local only — keep last tokens (used sliding and streaming attention)
    from transformers import DynamicCache                        
    # update in place
    local_cache = DynamicCache()
    for layer_idx, layer in enumerate(full_cache.layers):
        local_cache.update(
            layer.keys[:, :, -budget:, :],
            layer.values[:, :, -budget:, :],
            layer_idx=layer_idx
        )
    gen_from_cache(local_cache, f"LOCAL ONLY (last {budget})")

    # C) First tokens only — attention sinks
    sink_cache = DynamicCache()
    for layer_idx, layer in enumerate(full_cache.layers):
        sink_cache.update(
            layer.keys[:, :, :budget, :], 
            layer.values[:, :, :budget, :], 
            layer_idx=layer_idx
        )
    gen_from_cache(sink_cache, f"FIRST TOKENS (sinks, first {budget})")

    # D) Random
    random = sorted(np.random.choice(n, budget, replace=False).tolist())
    random_tensors = torch.tensor(random, device=DEVICE)
    random_cache = DynamicCache()
    for layer_idx, layer in enumerate(full_cache.layers):
        random_cache.update(
            layer.keys[:, :, random_tensors, :], 
            layer.values[:, :, random_tensors, :], 
            layer_idx=layer_idx
        )
    gen_from_cache(random_cache, f"RANDOM ({budget} tokens)")

    # E) Attention-based (H2O-style)
    scores = torch.zeros(n, device=DEVICE)
    for la in out.attentions:
        scores += la[0].float().sum(dim=(0, 1)) # element-wise sum of attention received by each key token across all heads and query positions

    local_n = max(4, n // 10) # the most recent 10 percent of previous tokens

    global_n = budget - local_n

    global_scores = scores.clone()

    global_scores[-local_n:] = -float("inf") # suppress the last local_n positions so topk won't pick them

    global_indices = global_scores.topk(global_n).indices

    local_indices = torch.arange(n - local_n, n, device=DEVICE)
    h2o_idx = torch.cat([global_indices, local_indices]).sort().values
    h2o_cache = DynamicCache()
    for layer_idx, layer in enumerate(full_cache.layers):
        h2o_cache.update(layer.keys[:, :, h2o_idx, :], layer.values[:, :, h2o_idx, :], layer_idx=layer_idx)
    gen_from_cache(h2o_cache, f"H2O-STYLE (top-attention + local, {budget} tokens)")

# ============================================================================
# TEST 5: GENERATION PARAMETERS — sampling strategies
# ============================================================================
def test_generation_params(model, tokenizer):
    print("=" * 70)
    print("TEST 5: GENERATION PARAMETERS")
    print("  These control token SELECTION, not the model itself.")
    print("  The model computes the same logits; these decide what to pick.")
    print("=" * 70)

    inputs = prepare_input(tokenizer, PROMPT)

    configs = {
        "Greedy (deterministic)": dict(
            # NOTE: always picks the highest-probability token; fully deterministic
            max_new_tokens=60, do_sample=False,
        ),
        "Temp=0.3 (focused sampling)": dict(
            # NOTE: sharpens distribution — model sticks to high-confidence tokens, less variety
            max_new_tokens=60, do_sample=True, temperature=0.3,
        ),
        "Temp=1.5 (creative sampling)": dict(
            # NOTE: flattens distribution — lower-probability tokens get more chances, more creative but less coherent
            max_new_tokens=60, do_sample=True, temperature=1.5,
        ),
        "Top-k=10 (narrow vocab)": dict(
            # NOTE: only sample from top 10 tokens at each step — very constrained, repetitive but safe
            max_new_tokens=60, do_sample=True, top_k=10, temperature=0.7,
        ),
        "Top-k=100 (wider vocab)": dict(
            # NOTE: sample from top 100 tokens — more variety while still excluding very unlikely tokens
            max_new_tokens=60, do_sample=True, top_k=100, temperature=0.7,
        ),
        "Top-p=0.5 (nucleus narrow)": dict(
            # NOTE: sample from smallest set of tokens whose cumulative prob >= 0.5 — tight nucleus, conservative
            max_new_tokens=60, do_sample=True, top_p=0.5, temperature=0.7,
        ),
        "Top-p=0.95 (nucleus wide)": dict(
            # NOTE: sample from tokens covering 95% of probability mass — wide nucleus, allows long-tail tokens
            max_new_tokens=60, do_sample=True, top_p=0.95, temperature=0.7,
        ),
        "Repetition penalty=1.5": dict(
            # NOTE: divides logits of already-seen tokens by 1.5, discouraging repetition
            max_new_tokens=60, do_sample=True, temperature=0.7, repetition_penalty=1.5,
        ),
        "Beam search (4 beams)": dict(
            # NOTE: maintains 4 candidate sequences in parallel, picks highest overall probability — deterministic
            max_new_tokens=60, num_beams=4, early_stopping=True,
        ),
    }

    for name, params in configs.items():
        out = model.generate(**inputs, **params)
        txt = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        print(f"\n  [{name}]")
        print(f"    {txt[:200]}{'...' if len(txt) > 200 else ''}")
    print()


# ============================================================================
# TEST 6: LOGITS — raw model output before sampling
# ============================================================================
def test_logits(model, tokenizer):
    print("=" * 70)
    print("TEST 6: LOGITS — the raw scores the model outputs")
    print("  Every test above operates on these. This is ground truth.")
    print("=" * 70)

    inputs = prepare_input(tokenizer, PROMPT)
    with torch.no_grad():
        out = model(**inputs)

    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)

    print(f"\n  Logits shape: {logits.shape} (one score per vocab token)")
    print(f"  Range: [{logits.min():.2f}, {logits.max():.2f}]")

    top_k = 15
    top_p, top_i = probs.topk(top_k)
    cum = 0
    print(f"\n  Top {top_k} next-token predictions:")
    for j in range(top_k):
        tok = tokenizer.decode(top_i[j])
        p = top_p[j].item()
        cum += p
        bar = "█" * int(p * 60)
        print(f"    {j+1:>2}. '{tok:<12}' p={p:.4f}  cum={cum:.4f}  {bar}")

    print(f"\n  Distribution shape:")
    print(f"    Tokens with p > 0.01:  {(probs > 0.01).sum().item()}")
    print(f"    Tokens with p > 0.001: {(probs > 0.001).sum().item()}")
    print(f"    Top-10 covers:         {probs.topk(10).values.sum():.4f} of probability mass")
    print(f"    Top-100 covers:        {probs.topk(100).values.sum():.4f} of probability mass")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), facecolor="#0d1117")

    ax1.set_facecolor("#0d1117")
    top_labels = [tokenizer.decode(top_i[j]) for j in range(top_k)]
    ax1.barh(range(top_k), top_p[:top_k].cpu().numpy(), color="#f97583")
    ax1.set_yticks(range(top_k))
    ax1.set_yticklabels(top_labels, fontsize=9, color="#e6edf3")
    ax1.invert_yaxis()
    ax1.set_xlabel("Probability")
    ax1.set_title("Top 15 Predictions", color="#e6edf3")

    ax2.set_facecolor("#0d1117")
    sorted_p = probs.sort(descending=True).values.cpu().numpy()
    ax2.plot(sorted_p[:500], color="#79c0ff", linewidth=1.5)
    ax2.set_yscale("log")
    ax2.set_xlabel("Rank")
    ax2.set_ylabel("Probability (log)")
    ax2.set_title("Full Distribution (top 500)", color="#e6edf3")
    ax2.grid(True, alpha=0.1)

    plt.tight_layout()
    save_fig("04_logits_distribution")
    print()


# ============================================================================
# TEST 7: TEMPERATURE — how it reshapes the distribution
# ============================================================================
def test_temperature(model, tokenizer):
    print("=" * 70)
    print("TEST 7: TEMPERATURE VISUALIZATION")
    print("  softmax(logits / T): lower T = peakier, higher T = flatter")
    print("=" * 70)

    inputs = prepare_input(tokenizer, PROMPT)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :].float()

    temps = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), facecolor="#0d1117")
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(temps)))

    for i, t in enumerate(temps):
        ax = axes[i // 4][i % 4]
        ax.set_facecolor("#0d1117")
        p = torch.softmax(logits / t, dim=-1)
        sp = p.sort(descending=True).values.cpu().numpy()

        ax.plot(sp[:100], color=cmap[i], linewidth=2)
        ax.fill_between(range(100), sp[:100], alpha=0.3, color=cmap[i])
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1)

        ent = -(p * (p + 1e-10).log()).sum().item()
        ax.set_title(f"T={t}", color="#e6edf3", fontsize=12)
        ax.text(0.95, 0.95,
                f"top1={sp[0]:.3f}\ntop10={sp[:10].sum():.3f}\nentropy={ent:.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color="#8b949e",
                bbox=dict(boxstyle="round", facecolor="#161b22", alpha=0.8))
        ax.grid(True, alpha=0.1)

    fig.suptitle("Temperature Effect on Next-Token Distribution",
                 color="#e6edf3", fontsize=14)
    plt.tight_layout()
    save_fig("05_temperature_effect")
    print("  Low T → model is very confident (picks top token almost always)")
    print("  High T → model spreads probability (more random/creative)\n")


# ============================================================================
# TEST 8: CACHE vs NO CACHE — prove identical output
# ============================================================================
def test_cache_speed(model, tokenizer):
    print("=" * 70)
    print("TEST 8: WITH vs WITHOUT KV CACHE")
    print("  Outputs are identical. Cache is purely a speed optimization.")
    print("=" * 70)
    inputs = prepare_input(tokenizer, PROMPT)
    N = 40

    # With cache
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        o = model(**inputs, use_cache=True) # setting up the cache

    past = o.past_key_values
    nid = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    ids_cached = [nid.item()]
    for _ in range(N - 1):
        with torch.no_grad():
            o = model(nid, past_key_values=past, use_cache=True)
        past = o.past_key_values
        nid = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids_cached.append(nid.item())
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_cached = time.time() - t0

    # Without cache
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time()
    gen = inputs["input_ids"] # starts as the prompt
    for _ in range(N):
        with torch.no_grad():
            o = model(gen, use_cache=False) # re-proces ALL tokens every step
        nid = o.logits[:, -1, :].argmax(dim=-1, keepdim=True) # append new token, sequence grows
        gen = torch.cat([gen, nid], dim=1)
    if torch.cuda.is_available(): torch.cuda.synchronize() 
    t_nocache = time.time() - t0

    ids_nocache = gen[0, inputs["input_ids"].shape[1]:].tolist()

    print(f"\n  With cache:    {t_cached:.3f}s → {tokenizer.decode(ids_cached, skip_special_tokens=True)}")
    print(f"  Without cache: {t_nocache:.3f}s → {tokenizer.decode(ids_nocache, skip_special_tokens=True)}")
    print(f"  Speedup:       {t_nocache / t_cached:.1f}x")
    print(f"  Same output:   {ids_cached == ids_nocache}")
    print(f"\n  The cache trades STORAGE for COMPUTE. Nothing else changes.\n")


# ============================================================================
# TEST 9: HIDDEN STATES — internal representations per layer
# ============================================================================
def test_hidden_states(model, tokenizer):
    print("=" * 70)
    print("TEST 9: HIDDEN STATES — how representations evolve through layers")
    print("=" * 70)

    inputs = prepare_input(tokenizer, PROMPT)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    hs = out.hidden_states  # (num_layers+1,) each (batch, seq, hidden)
    print(f"\n  Snapshots: {len(hs)} (1 embedding + {len(hs)-1} layer outputs)")
    print(f"  Shape: {hs[0].shape}")

    norms = [h[0].float().norm(dim=-1).mean().item() for h in hs]
    diffs = [(hs[i] - hs[i-1]).float().norm(dim=-1).mean().item()
             for i in range(1, len(hs))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")

    ax1.set_facecolor("#0d1117")
    ax1.plot(norms, color="#f97583", linewidth=2, marker="o", markersize=3)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean L2 Norm")
    ax1.set_title("Hidden State Magnitude", color="#e6edf3")
    ax1.grid(True, alpha=0.1)

    ax2.set_facecolor("#0d1117")
    ax2.plot(diffs, color="#79c0ff", linewidth=2, marker="o", markersize=3)
    ax2.set_xlabel("Layer Transition")
    ax2.set_ylabel("Mean Change (L2)")
    ax2.set_title("How Much Each Layer Modifies the Representation", color="#e6edf3")
    ax2.grid(True, alpha=0.1)

    plt.tight_layout()
    save_fig("06_hidden_states")

    cos = torch.nn.functional.cosine_similarity(
        hs[1][0].float().mean(0, keepdim=True),
        hs[-1][0].float().mean(0, keepdim=True)
    ).item()
    print(f"  Cosine sim (first layer → last layer): {cos:.4f}")
    print(f"  Lower = the network transforms representations more.\n")


# ============================================================================
# TEST 10: RoPE — why the context window is limited
# ============================================================================
def test_rope(model, tokenizer):
    print("=" * 70)
    print("TEST 10: RoPE POSITION ENCODING")
    print("  This is the hard constraint on context window size.")
    print("  The model only learned positions up to max_position_embeddings.")
    print("=" * 70)

    c = model.config
    theta = getattr(c, "rope_theta", 10000.0)
    head_dim = c.hidden_size // c.num_attention_heads
    max_pos = c.max_position_embeddings

    print(f"\n  RoPE theta: {theta}")
    print(f"  Max positions: {max_pos:,}")
    print(f"  Head dim: {head_dim}")

    dim_pairs = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(0, min(max_pos, 8192)).float()
    angles = torch.outer(positions, freqs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0d1117")

    ax1.set_facecolor("#0d1117")
    ax1.plot(freqs.numpy(), color="#79c0ff", linewidth=1.5)
    ax1.set_yscale("log")
    ax1.set_xlabel("Dimension pair")
    ax1.set_ylabel("Frequency (log)")
    ax1.set_title("RoPE Frequency Spectrum", color="#e6edf3")
    ax1.grid(True, alpha=0.1)

    ax2.set_facecolor("#0d1117")
    for d in [0, 1, dim_pairs // 4, dim_pairs // 2, dim_pairs - 1]:
        ax2.plot(torch.cos(angles[:200, d]).numpy(),
                 label=f"dim {d}", linewidth=1, alpha=0.8)
    ax2.set_xlabel("Position")
    ax2.set_ylabel("cos(pos × freq)")
    ax2.set_title("RoPE Phases (first 200 positions)", color="#e6edf3")
    ax2.legend(fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    ax2.grid(True, alpha=0.1)

    plt.tight_layout()
    save_fig("07_rope_positions")

    print(f"\n  Low-freq dims → coarse position (long-range patterns)")
    print(f"  High-freq dims → fine position (nearby token distinctions)")
    print(f"  Beyond position {max_pos:,}, encodings are out of distribution → garbage.\n")


# ============================================================================
# MAIN — run everything
# ============================================================================
if __name__ == "__main__":
    model, tokenizer = load_model()

    print("\n" + "=" * 70)
    print("  TRANSFORMER PARAMETER EXPLORER")
    print(f"  Model: {MODEL_ID}")
    print(f"  Outputs → {OUTPUT_DIR}/")
    print("=" * 70 + "\n")

    test_architecture(model, tokenizer)          # 1. What's fixed
    # test_kv_cache(model, tokenizer)              # 2. KV cache = context
    # test_attention_heatmaps(model, tokenizer)    # 3. Attention visualization
    # test_kv_cache_manipulation(model, tokenizer) # 4. Modify cache → see effect
    # test_generation_params(model, tokenizer)     # 5. Sampling strategies
    # test_logits(model, tokenizer)                # 6. Raw probabilities
    # test_temperature(model, tokenizer)           # 7. Temperature effect
    # test_cache_speed(model, tokenizer)           # 8. Cache = same output, faster
    # test_hidden_states(model, tokenizer)         # 9. Layer representations
    # test_rope(model, tokenizer)                  # 10. Position encoding limits

    print("\n" + "=" * 70)
    print("  DONE — generated plots:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".png"):
            print(f"    {OUTPUT_DIR}/{f}")
    print("=" * 70)