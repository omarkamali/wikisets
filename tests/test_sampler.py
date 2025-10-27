"""Tests for sampling utilities."""

from wikisets.sampler import compute_interleave_probabilities


def test_compute_interleave_probabilities():
    """Test probability computation for interleaving."""
    probs = compute_interleave_probabilities([100, 200, 300])
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 0.001
