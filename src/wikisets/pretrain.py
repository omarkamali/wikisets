"""Pretraining utilities for text chunking."""

import re
import warnings
from typing import Any, Optional, Union

from datasets import Dataset


def load_tokenizer(tokenizer: Union[str, Any]) -> Any:
    """Load tokenizer from name or return existing tokenizer.

    Args:
        tokenizer: Tokenizer instance or HuggingFace model name.

    Returns:
        Tokenizer instance.
    """
    if isinstance(tokenizer, str):
        from transformers import AutoTokenizer

        print(f"Loading tokenizer: {tokenizer}")
        return AutoTokenizer.from_pretrained(tokenizer)
    return tokenizer


def find_split_point(
    text: str, max_tokens: int, tokenizer: Any, delimiter: str = "newline"
) -> tuple[int, bool]:
    """Find character position to split text based on token limit and delimiter.

    Args:
        text: Input text.
        max_tokens: Maximum token count.
        tokenizer: Tokenizer instance.
        delimiter: Delimiter type ("space", "newline", or regex pattern).

    Returns:
        Tuple of (character_position, found_delimiter).
    """
    # Encode to get token count
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # If within limit, return full text
    if len(tokens) <= max_tokens:
        return len(text), True

    # Binary search for character position that fits max_tokens
    # This works with any tokenizer, even without offset mapping
    left, right = 0, len(text)
    best_pos = 0

    while left < right:
        mid = (left + right + 1) // 2
        chunk_tokens = tokenizer.encode(text[:mid], add_special_tokens=False)

        if len(chunk_tokens) <= max_tokens:
            best_pos = mid
            left = mid
        else:
            right = mid - 1

    # Now find nearest delimiter before best_pos
    if delimiter == "space":
        pattern = r"\s"
    elif delimiter == "newline":
        pattern = r"\n"
    else:
        pattern = delimiter

    # Search in last 20% of best_pos
    search_start = max(0, int(best_pos * 0.8))
    search_region = text[search_start:best_pos]

    # Find all delimiter positions
    matches = list(re.finditer(pattern, search_region))

    if matches:
        # Use last delimiter position
        last_match = matches[-1]
        return search_start + last_match.end(), True
    else:
        # No delimiter found in search region
        warnings.warn(
            f"No delimiter found in last 20% of text (length {best_pos}). "
            f"Cutting at token boundary.",
            stacklevel=2,
        )
        return best_pos, False


def chunk_article(
    article: dict[str, Any],
    max_tokens: int,
    tokenizer: Any,
    delimiter: str = "newline",
    tracker: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Chunk a single article into multiple parts.

    Args:
        article: Article dictionary with id, url, title, text, lang.
        max_tokens: Maximum tokens per chunk.
        tokenizer: Tokenizer instance.
        delimiter: Delimiter for splitting.
        tracker: Optional warning tracker.

    Returns:
        List of chunk dictionaries.
    """
    text = article["text"]
    chunks = []
    offset = 0
    chunk_idx = 0

    while offset < len(text):
        remaining = text[offset:]

        # Find split point
        split_pos, found_delim = find_split_point(
            remaining, max_tokens, tokenizer, delimiter
        )

        # Track warnings
        if not found_delim and tracker is not None:
            tracker.warn(
                f"Article {article['id']} chunk {chunk_idx}: "
                f"No delimiter in last 20%, cut at token boundary"
            )

        # Create chunk
        chunk_text = remaining[:split_pos]
        # Compute token length for this chunk
        token_len = len(tokenizer.encode(chunk_text, add_special_tokens=False))
        chunks.append(
            {
                "id": article["id"],
                "url": article["url"],
                "title": article["title"],
                "text": chunk_text,
                "lang": article["lang"],
                "chunk_index": chunk_idx,
                "total_chunks": -1,  # Will update after
                "token_len": token_len,
            }
        )

        offset += split_pos
        chunk_idx += 1

    # Update total_chunks
    for chunk in chunks:
        chunk["total_chunks"] = len(chunks)

    return chunks


def apply_pretrain_chunking(
    dataset: Dataset,
    split_token_len: Optional[int],
    tokenizer: Union[str, Any],
    nearest_delimiter: str = "newline",
    num_proc: Optional[int] = None,
    batch_size: int = 1000,
    tracker: Optional[Any] = None,
) -> Dataset:
    """Apply pretraining chunking to dataset.

    Args:
        dataset: Input dataset.
        split_token_len: Maximum tokens per chunk (None for no chunking).
        tokenizer: Tokenizer instance or name.
        nearest_delimiter: Delimiter type.
        num_proc: Number of processes.
        batch_size: Batch size for processing.
        tracker: Warning tracker.

    Returns:
        Chunked dataset.
    """
    if split_token_len is None:
        # No chunking, just ensure schema
        if "chunk_index" not in dataset.column_names:
            print("Adding chunk metadata (no splitting)...")
            dataset = dataset.map(
                lambda x: {
                    "chunk_index": 0,
                    "total_chunks": 1,
                },
                num_proc=num_proc,
                batch_size=batch_size,
                desc="Adding metadata",
            )
        return dataset

    # Load tokenizer
    tok = load_tokenizer(tokenizer)

    # Process articles into chunks
    def process_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        all_chunks: dict[str, list[Any]] = {
            "id": [],
            "url": [],
            "title": [],
            "text": [],
            "lang": [],
            "chunk_index": [],
            "total_chunks": [],
            "token_len": [],
        }

        for i in range(len(batch["id"])):
            article = {
                "id": batch["id"][i],
                "url": batch["url"][i],
                "title": batch["title"][i],
                "text": batch["text"][i],
                "lang": batch["lang"][i],
            }

            chunks = chunk_article(
                article, split_token_len, tok, nearest_delimiter, tracker
            )

            for chunk in chunks:
                for key in all_chunks:
                    all_chunks[key].append(chunk[key])

        return all_chunks

    print(f"Chunking {len(dataset):,} articles (max {split_token_len} tokens)...")

    chunked = dataset.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Chunking articles",
    )

    return chunked
