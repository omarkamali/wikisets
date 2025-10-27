"""Sampling utilities for dataset construction."""

from typing import Optional

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
