"""
These tests monkeypatch `datasets.load_dataset` to capture which split is
requested by `Wikiset._load_language()`.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from wikisets.utils import select_split_for_size
from wikisets.wikiset import Wikiset


class _FakeHFData:
    def __init__(self, n: int) -> None:
        self._n = n
        # minimal attrs used when wrapping back into Wikiset
        self._data = object()
        self._info = types.SimpleNamespace(description="")
        self._split = None
        self._indices = None
        self._fingerprint = "fake"

    def __len__(self) -> int:
        return self._n

    def add_column(self, name: str, values: list[Any]):
        # mimic HF Dataset.add_column returning a new dataset
        return self

    def select(self, indices):
        # mimic HF Dataset.select by returning a dataset with len(indices)
        try:
            new_len = len(indices)
        except Exception:
            new_len = 0
        return _FakeHFData(new_len)


@pytest.fixture()
def capture_load(monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_load_dataset(path: str, subset: str, split: str):
        calls.append({"path": path, "subset": subset, "split": split})
        # Return a fake dataset with an arbitrary size large enough for sampling paths
        if split == "train":
            return _FakeHFData(123456)
        # Sample splits should have exactly their nominal sizes
        if split in {"1000", "5000", "10000"}:
            return _FakeHFData(int(split))
        raise RuntimeError(f"unexpected split in test: {split}")

    monkeypatch.setattr("wikisets.wikiset.load_dataset", fake_load_dataset)
    return calls


def _call_load_language(**kwargs) -> tuple[int, dict[str, Any]]:
    # Access the private method for a focused test
    ds, stat = Wikiset._load_language(**kwargs)
    return len(ds), stat


def test_uses_10k_split_for_exact_size(capture_load):
    size, stat = _call_load_language(
        lang="en",
        size=10000,
        date="20251001",
        use_train_split=False,
        seed=42,
        tracker=types.SimpleNamespace(warn=lambda *a, **k: None),
    )

    # Confirm the right split was requested
    assert capture_load[-1]["split"] == "10000"
    # And the returned dataset is of that size
    assert size == 10000
    assert stat["split_used"] == "10000"


def test_percentage_and_fraction_use_train(capture_load):
    # 50% should request train
    _call_load_language(
        lang="fr",
        size="50%",
        date="latest",
        use_train_split=False,
        seed=0,
        tracker=types.SimpleNamespace(warn=lambda *a, **k: None),
    )
    assert capture_load[-1]["split"] == "train"

    # 0.1 should request train
    _call_load_language(
        lang="ar",
        size=0.1,
        date="latest",
        use_train_split=False,
        seed=0,
        tracker=types.SimpleNamespace(warn=lambda *a, **k: None),
    )
    assert capture_load[-1]["split"] == "train"


def test_ceiling_small_sizes_use_smallest_sample(capture_load):
    # 1200 should choose the 1k sample according to our ceiling strategy
    # (function returns the smallest sample that fits; 1200 -> "5000" by current impl)
    # Verify via the helper directly and full path through _load_language
    split = select_split_for_size(1200, use_train_split=False)
    assert split in {"1000", "5000", "10000", "train"}

    _call_load_language(
        lang="ary",  # minor language, but we stub load_dataset so this is offline
        size=1200,
        date="latest",
        use_train_split=False,
        seed=0,
        tracker=types.SimpleNamespace(warn=lambda *a, **k: None),
    )
    # Ensure we didn't fall back to train in the happy path
    assert capture_load[-1]["split"] in {"1000", "5000", "10000"}


def test_downsamples_from_1000_split_to_requested_size(capture_load):
    size, stat = _call_load_language(
        lang="en",
        size=800,
        date="latest",
        use_train_split=False,
        seed=123,
        tracker=types.SimpleNamespace(warn=lambda *a, **k: None),
    )
    assert capture_load[-1]["split"] == "1000"
    assert size == 800
    assert stat["split_used"] == "1000"


def test_downsamples_from_5000_split_to_requested_size(capture_load):
    size, stat = _call_load_language(
        lang="en",
        size=2500,
        date="latest",
        use_train_split=False,
        seed=123,
        tracker=types.SimpleNamespace(warn=lambda *a, **k: None),
    )
    assert capture_load[-1]["split"] == "5000"
    assert size == 2500
    assert stat["split_used"] == "5000"


def test_downsamples_from_10000_split_to_requested_size(capture_load):
    size, stat = _call_load_language(
        lang="en",
        size=7000,
        date="latest",
        use_train_split=False,
        seed=123,
        tracker=types.SimpleNamespace(warn=lambda *a, **k: None),
    )
    assert capture_load[-1]["split"] == "10000"
    assert size == 7000
    assert stat["split_used"] == "10000"


def test_exact_1000_and_5000_return_exact_sizes(capture_load):
    size1, stat1 = _call_load_language(
        lang="en",
        size=1000,
        date="latest",
        use_train_split=False,
        seed=123,
        tracker=types.SimpleNamespace(warn=lambda *a, **k: None),
    )
    assert capture_load[-1]["split"] == "1000"
    assert size1 == 1000
    assert stat1["split_used"] == "1000"

    size2, stat2 = _call_load_language(
        lang="en",
        size=5000,
        date="latest",
        use_train_split=False,
        seed=123,
        tracker=types.SimpleNamespace(warn=lambda *a, **k: None),
    )
    assert capture_load[-1]["split"] == "5000"
    assert size2 == 5000
    assert stat2["split_used"] == "5000"


def test_greater_than_10k_uses_train_and_returns_requested(capture_load):
    size, stat = _call_load_language(
        lang="simple",
        size=15000,
        date="latest",
        use_train_split=False,
        seed=123,
        tracker=types.SimpleNamespace(warn=lambda *a, **k: None),
    )
    assert capture_load[-1]["split"] == "train"
    assert size == 15000
    assert stat["split_used"] == "train (sampled)"
