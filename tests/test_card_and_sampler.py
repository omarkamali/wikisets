"""Tests for dataset card generation and sampler behavior."""

from datasets import Dataset

from wikisets.card_generator import generate_dataset_card
from wikisets.config import WikisetConfig
from wikisets.sampler import reservoir_sample


def test_generate_dataset_card_basic_sections():
    config = WikisetConfig(
        languages=[{"lang": "en", "size": 1000}, {"lang": "ary", "size": "10%"}],
        date="20251001",
        seed=123,
        shuffle=True,
        use_train_split=False,
    )
    language_stats = [
        {
            "language": "en",
            "requested_size": "1000 items",
            "split_used": "1000",
            "actual_size": 1000,
        },
        {
            "language": "ary",
            "requested_size": "10.0% (~200 items)",
            "split_used": "train (sampled)",
            "actual_size": 200,
        },
    ]
    warnings = ["Split '10000' not found for ary, falling back to train"]
    card = generate_dataset_card(
        config,
        language_stats,
        warnings,
        total_size=1200,
        pretrain_config={
            "split_token_len": 2048,
            "tokenizer": "test-tokenizer",
            "nearest_delimiter": "newline",
        },
    )

    # Header and Configuration
    assert "# Wikiset Dataset Card" in card
    assert "## Configuration" in card
    assert "Total Size:" in card
    assert "Languages:" in card

    # Language Composition table rows
    assert (
        "| Language | Requested Size | Split Used | Actual Size | Percentage |" in card
    )
    assert "| en | 1000 items | 1000 | 1,000 |" in card
    assert "| ary | 10.0% (~200 items) | train (sampled) | 200 |" in card

    # Sampling Methodology and Language Mixing
    assert "## Sampling Methodology" in card
    assert "### Language Mixing" in card

    # Pretraining Configuration
    assert "## Pretraining Configuration" in card
    assert "Split Token Length:" in card

    # Warnings section
    assert "## Warnings" in card
    assert any(w in card for w in warnings)

    # Source and Citation
    assert "## Source" in card and "## Citation" in card


def test_reservoir_sample_returns_full_when_k_ge_total():
    ds = Dataset.from_dict({"a": list(range(5))})
    out = reservoir_sample(ds, k=10, seed=0, total_size=None)
    assert len(out) == 5


def test_reservoir_sample_returns_k_items_deterministic():
    ds = Dataset.from_dict({"a": list(range(100))})
    out1 = reservoir_sample(ds, k=10, seed=123, total_size=100)
    out2 = reservoir_sample(ds, k=10, seed=123, total_size=100)
    assert len(out1) == 10
    assert len(out2) == 10
    # Same seed -> same selection
    assert out1[:]["a"] == out2[:]["a"]
