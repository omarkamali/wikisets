"""Dataset card generation utilities."""

from datetime import datetime
from typing import Any, Optional


def generate_dataset_card(
    config: Any,
    language_stats: list[dict[str, Any]],
    warnings: list[str],
    total_size: int,
    pretrain_config: Optional[dict[str, Any]] = None,
) -> str:
    """Generate markdown dataset card.

    Args:
        config: WikisetConfig instance.
        language_stats: List of per-language statistics.
        warnings: List of warning messages.
        total_size: Total dataset size.
        pretrain_config: Optional pretraining configuration.

    Returns:
        Markdown dataset card.
    """
    lines = []

    # Header
    lines.append("# Wikiset Dataset Card")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")

    # Configuration summary
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Date:** {config.date}")
    lines.append(f"- **Total Size:** {total_size:,} items")
    lines.append(f"- **Languages:** {len(config.languages)}")
    lines.append(f"- **Seed:** {config.seed}")
    lines.append(f"- **Shuffle:** {config.shuffle}")
    lines.append(f"- **Use Train Split:** {config.use_train_split}")
    lines.append("")

    # Language breakdown
    lines.append("## Language Composition")
    lines.append("")
    lines.append(
        "| Language | Requested Size | Split Used | Actual Size | Percentage |"
    )
    lines.append("|----------|---------------|------------|-------------|------------|")

    for stat in language_stats:
        lang = stat["language"]
        req_size = stat["requested_size"]
        split = stat["split_used"]
        actual = stat["actual_size"]
        pct = (actual / total_size * 100) if total_size > 0 else 0
        lines.append(f"| {lang} | {req_size} | {split} | {actual:,} | {pct:.2f}% |")

    lines.append("")

    # Sampling methodology
    lines.append("## Sampling Methodology")
    lines.append("")
    lines.append("### Split Selection Rules")
    lines.append("")
    lines.append("- **Exact matches (1k, 5k, 10k):** Use corresponding sample split")
    lines.append(
        "- **Sizes ≤10k:** Use smallest sample split that fits (ceil strategy)"
    )
    lines.append("- **Sizes >10k or percentages:** Reservoir sampling from train split")
    lines.append("- **100% or 1.0:** Full train split without sampling")
    lines.append("- **Missing sample splits:** Automatic fallback to train split")
    lines.append("")

    if config.shuffle:
        lines.append("### Language Mixing")
        lines.append("")
        lines.append(
            "Languages are proportionally interleaved based on their selected sizes "
        )
        lines.append("to provide fair representation in batches.")
        lines.append("")

    # Pretraining config
    if pretrain_config:
        lines.append("## Pretraining Configuration")
        lines.append("")
        lines.append(
            f"- **Split Token Length:** {pretrain_config.get('split_token_len', 'None')}"
        )
        lines.append(f"- **Tokenizer:** {pretrain_config.get('tokenizer', 'N/A')}")
        lines.append(
            f"- **Delimiter:** {pretrain_config.get('nearest_delimiter', 'newline')}"
        )
        lines.append("")
        lines.append("### Chunking Logic")
        lines.append("")
        lines.append(
            "Articles are split into chunks with token counts up to the specified limit. "
        )
        lines.append(
            "Text is cut at the nearest delimiter (newline by default) before the token "
        )
        lines.append(
            "boundary. If no delimiter is found in the last 20% of the target length, "
        )
        lines.append("the text is cut at the token boundary with a warning.")
        lines.append("")
        lines.append(
            "Each chunk preserves the original article metadata (id, url, title, lang) "
        )
        lines.append("and adds chunk_index and total_chunks fields.")
        lines.append("")

    # Warnings
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    # Source attribution
    lines.append("## Source")
    lines.append("")
    lines.append("This dataset is built from [omarkamali/wikipedia-monthly]")
    lines.append("(https://huggingface.co/datasets/omarkamali/wikipedia-monthly), ")
    lines.append("which provides fresh, clean Wikipedia dumps updated monthly.")
    lines.append("")

    # Citation
    lines.append("## Citation")
    lines.append("")
    lines.append("```bibtex")
    lines.append("@software{wikisets2025,")
    lines.append("  author = {Omar Kamali},")
    lines.append("  title = {Wikisets: Flexible Wikipedia Dataset Builder},")
    lines.append("  year = {2025},")
    lines.append("  url = {https://github.com/omarkamali/wikisets}")
    lines.append("}")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)
