"""Tests for WikisetConfig."""

import pytest

from wikisets.config import WikisetConfig


def test_valid_config():
    """Test valid configuration."""
    config = WikisetConfig(
        languages=[
            {"lang": "en", "size": 1000},
            {"lang": "fr", "size": "50%"},
            {"lang": "de", "size": 0.3},
        ]
    )
    assert len(config.languages) == 3


def test_from_dict():
    """Test creating config from dictionary."""
    config_dict = {
        "languages": [{"lang": "en", "size": 1000}],
        "date": "20250101",
    }
    config = WikisetConfig.from_dict(config_dict)
    assert config.date == "20250101"


def test_empty_languages():
    """Test that empty languages raises error."""
    with pytest.raises(ValueError, match="cannot be empty"):
        WikisetConfig(languages=[])
