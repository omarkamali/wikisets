"""Dataset card generation utilities."""

from datetime import datetime
from typing import Any, Optional


def generate_dataset_card(
    config: Any,
    language_stats: list[dict[str, Any]],
    warnings: list[str],
    total_size: int,
    pretrain_config: Optional[dict[str, Any]] = None,
    token_stats: Optional[dict[str, Any]] = None,
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
    # Header
    header = (
        "# Wikiset Dataset Card\n\n"
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
    )

    # Configuration summary
    config_section = (
        "## Configuration\n\n"
        f"- **Date:** {config.date}\n"
        f"- **Total Size:** {total_size:,} items\n"
        f"- **Languages:** {len(config.languages)}\n"
        f"- **Seed:** {config.seed}\n"
        f"- **Shuffle:** {config.shuffle}\n"
        f"- **Use Train Split:** {config.use_train_split}\n\n"
    )

    # Language breakdown table
    table_header = (
        "## Language Composition\n\n"
        "| Language | Requested Size | Split Used | Actual Size | Percentage |\n"
        "|----------|---------------|------------|-------------|------------|\n"
    )
    table_rows_list: list[str] = []
    for stat in language_stats:
        lang = stat["language"]
        req_size = stat["requested_size"]
        split = stat["split_used"]
        actual = stat["actual_size"]
        pct = (actual / total_size * 100) if total_size > 0 else 0
        table_rows_list.append(
            f"| {lang} | {req_size} | {split} | {actual:,} | {pct:.2f}% |"
        )
    language_section = table_header + "\n".join(table_rows_list) + "\n\n"

    # Sampling methodology
    sampling_section = (
        "## Sampling Methodology\n\n"
        "### Split Selection Rules\n\n"
        "- **Exact matches (1k, 5k, 10k):** Use corresponding sample split\n"
        "- **Sizes ≤10k:** Use smallest sample split that fits (ceil strategy)\n"
        "- **Sizes >10k or percentages:** Reservoir sampling from train split\n"
        "- **100% or 1.0:** Full train split without sampling\n"
        "- **Missing sample splits:** Automatic fallback to train split\n\n"
    )

    language_mixing_section = ""
    if config.shuffle:
        language_mixing_section = (
            "### Language Mixing\n\n"
            "Languages are proportionally interleaved based on their selected sizes \n"
            "to provide fair representation in batches.\n\n"
        )

    # Pretraining config
    pretrain_section = ""
    if pretrain_config:
        pretrain_section = (
            "## Pretraining Configuration\n\n"
            f"- **Split Token Length:** {pretrain_config.get('split_token_len', 'None')}\n"
            f"- **Tokenizer:** {pretrain_config.get('tokenizer', 'N/A')}\n"
            f"- **Delimiter:** {pretrain_config.get('nearest_delimiter', 'newline')}\n\n"
            "### Chunking Logic\n\n"
            "Articles are split into chunks with token counts up to the specified limit. \n"
            "Text is cut at the nearest delimiter (newline by default) before the token \n"
            "boundary. If no delimiter is found in the last 20% of the target length, \n"
            "the text is cut at the token boundary with a warning.\n\n"
            "Each chunk preserves the original article metadata (id, url, title, lang) \n"
            "and adds chunk_index and total_chunks fields.\n\n"
        )

    # Pretraining token statistics
    token_stats_section = ""
    if token_stats:
        per_lang_rows = "\n".join(
            f"| {row['language']} | {row['tokens']:,} |" for row in token_stats.get("per_language", [])
        )
        token_stats_section = (
            "## Pretraining Token Statistics\n\n"
            f"- **Total Tokens:** {token_stats.get('total_tokens', 0):,}\n\n"
            "| Language | Tokens |\n"
            "|----------|--------|\n"
            f"{per_lang_rows}\n\n"
        )

    # Warnings
    warnings_section = ""
    if warnings:
        warnings_list = "\n".join(f"- {w}" for w in warnings)
        warnings_section = f"## Warnings\n\n{warnings_list}\n\n"

    # Source attribution
    source_section = (
        "## Source\n\n"
        "This dataset is built from [omarkamali/wikipedia-monthly]\n"
        "(https://huggingface.co/datasets/omarkamali/wikipedia-monthly), \n"
        "which provides fresh, clean Wikipedia dumps updated monthly.\n\n"
    )

    # Citation
    citation_section = (
        "## Citation\n\n"
        "```bibtex\n"
        "@software{wikisets2025,\n"
        "  author = {Omar Kamali},\n"
        "  title = {Wikisets: Flexible Wikipedia Dataset Builder},\n"
        "  year = {2025},\n"
        "  url = {https://github.com/omarkamali/wikisets}\n"
        "}\n"
        "```\n\n"
    )

    return (
        header
        + config_section
        + language_section
        + sampling_section
        + language_mixing_section
        + pretrain_section
        + token_stats_section
        + warnings_section
        + source_section
        + citation_section
    )
