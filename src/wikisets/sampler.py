"""Sampling utilities for dataset construction."""

from collections.abc import Iterable
from typing import Any, Optional

import numpy as np
from datasets import Dataset


def reservoir_sample(
    dataset: Dataset, k: int, seed: int = 42, total_size: Optional[int] = None
) -> Dataset:
    """Perform reservoir sampling to select k items from dataset.

    Uses a two-pass approach when total_size is known for memory efficiency.
    Falls back to single-pass reservoir sampling for unknown sizes.

    Args:
        dataset: Input dataset.
        k: Number of items to sample.
        seed: Random seed.
        total_size: Total size of dataset (for optimization).

    Returns:
        Sampled dataset with k items.
    """
    rng = np.random.default_rng(seed)

    # Get actual size
    if total_size is None:
        total_size = len(dataset)

    # If k >= total, return full dataset
    if k >= total_size:
        return dataset

    # Two-pass approach: generate indices, then select
    # This is more memory-efficient than loading all data
    print(f"  Sampling {k:,} from {total_size:,} items...")
    indices = rng.choice(total_size, size=k, replace=False)
    indices = sorted(indices)  # Sort for cache-friendly access

    return dataset.select(indices)


def reservoir_sample_streaming(
    iterable: Iterable[dict[str, Any]], k: int, seed: int = 42
) -> Dataset:
    """Single-pass reservoir sampling for streaming datasets.

    Works with an iterable of examples (e.g., IterableDataset) without requiring
    random access or knowing the total length in advance. Materializes only k
    sampled examples into a standard in-memory ``Dataset``.

    Args:
        iterable: An iterable yielding dict-like examples.
        k: Number of items to sample.
        seed: Random seed.

    Returns:
        A ``Dataset`` containing k sampled items (or fewer if the iterable has < k items).
    """
    rng = np.random.default_rng(seed)

    reservoir: list[dict[str, Any]] = []
    n = 0
    for ex in iterable:
        n += 1
        if n <= k:
            reservoir.append(ex)
        else:
            j = rng.integers(0, n)
            if j < k:
                reservoir[j] = ex

    # If the stream had fewer than k items, just return what we collected
    return Dataset.from_list(reservoir)


def compute_interleave_probabilities(sizes: list[int]) -> list[float]:
    """Compute proportional probabilities for interleaving.

    Args:
        sizes: List of dataset sizes.

    Returns:
        List of probabilities summing to 1.0.
    """
    total = sum(sizes)
    if total == 0:
        return [1.0 / len(sizes)] * len(sizes)
    return [size / total for size in sizes]
