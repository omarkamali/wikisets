"""Tests for utils: WarningTracker, parse_size, select_split_for_size."""

import pytest

from wikisets.utils import WarningTracker, parse_size, select_split_for_size


def test_warning_tracker_records_and_emits_warning(recwarn):
    tracker = WarningTracker()
    tracker.warn("test message")

    # Captured by warnings
    assert any("test message" in str(w.message) for w in recwarn)
    # Stored in tracker
    assert tracker.get_warnings() == ["test message"]

    tracker.clear()
    assert tracker.get_warnings() == []


def test_parse_size_with_percentage_requires_total():
    with pytest.raises(ValueError, match="total required for percentage"):
        parse_size("10%", None)


def test_parse_size_with_fraction_requires_total():
    with pytest.raises(ValueError, match="total required for fractional"):
        parse_size(0.1, None)


def test_parse_size_percentage_and_fraction_and_int():
    # Percentage rounds up and minimum 1
    target, desc = parse_size("10%", total=9)
    assert target == 1 and "10.0" in desc

    target, desc = parse_size("50%", total=9)
    assert target == 5 and "%" in desc

    # Fraction
    target, desc = parse_size(0.25, total=9)
    assert target == 3 and "25.0%" in desc

    # Integer
    target, desc = parse_size(123, total=None)
    assert target == 123 and "123 items" == desc


def test_select_split_for_size_cases():
    # Force train overrides
    assert select_split_for_size(1, True) == "train"

    # Exact sample sizes
    assert select_split_for_size(1000, False) == "1000"
    assert select_split_for_size(5000, False) == "5000"
    assert select_split_for_size(10000, False) == "10000"

    # Ceiling strategy
    assert select_split_for_size(999, False) == "1000"
    assert select_split_for_size(3000, False) == "5000"
    assert select_split_for_size(7000, False) == "10000"

    # Train for >10k
    assert select_split_for_size(15000, False) == "train"
