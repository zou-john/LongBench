import statistics
import textwrap

import click
from datasets import load_dataset


@click.command()
def main():
    click.echo(click.style("\n=== Loading zai-org/LongBench-v2 ===", fg="cyan", bold=True))
    ds = load_dataset("zai-org/LongBench-v2", split="train")

    # --- Metadata ---
    click.echo(click.style("\n[Metadata]", fg="yellow", bold=True))
    click.echo(f"  {click.style('Split size:', fg='white', bold=True)} {len(ds):,} examples")
    click.echo(f"  {click.style('Features:', fg='white', bold=True)}")
    for name, feat in ds.features.items():
        click.echo(f"    {click.style(name, fg='green')}: {feat}")

    # --- Example 0 ---
    click.echo(click.style("\n[Example 0 — all fields except context]", fg="yellow", bold=True))
    ex = ds[1]
    for key, val in ex.items():
        if key == "context":
            continue
        click.echo(f"  {click.style(key, fg='green')}: {val!r}")

    # --- Context preview ---
    context = ex["context"]
    # Strip leading UUID (36 chars + space) if present
    if len(context) > 37 and context[36] == " ":
        context_clean = context[37:]
    else:
        context_clean = context
    click.echo(click.style("\n[Example 0 — context preview (first 500 chars)]", fg="yellow", bold=True))
    preview = textwrap.fill(context_clean[:500], width=100)
    click.echo(click.style(preview, fg="white"))
    click.echo(click.style("  ... (truncated)", fg="bright_black"))

    # --- Context length stats ---
    click.echo(click.style("\n[Context Length Distribution (chars)]", fg="yellow", bold=True))
    lengths = [
        len(row["context"][37:]) if len(row["context"]) > 37 and row["context"][36] == " " else len(row["context"])
        for row in ds
    ]
    stats = {
        "min":    min(lengths),
        "max":    max(lengths),
        "mean":   statistics.mean(lengths),
        "median": statistics.median(lengths),
        "stdev":  statistics.stdev(lengths),
    }
    for label, val in stats.items():
        bar_len = int(val / max(lengths) * 40)
        bar = click.style("█" * bar_len, fg="magenta")
        click.echo(f"  {click.style(label, fg='green'):>10}  {bar}  {val:>12,.0f}")

    click.echo(click.style("\nDone!\n", fg="cyan", bold=True))


if __name__ == "__main__":
    main()
