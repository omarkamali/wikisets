"""Tests for Wikiset class."""

from wikisets.utils import select_split_for_size


def test_select_split_for_size():
    """Test split selection logic."""
    assert select_split_for_size(1000, False) == "1000"
    assert select_split_for_size(5000, False) == "5000"
    assert select_split_for_size(10000, False) == "10000"
    assert select_split_for_size(3000, False) == "5000"
    assert select_split_for_size(15000, False) == "train"
