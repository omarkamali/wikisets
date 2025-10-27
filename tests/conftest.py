"""Pytest configuration and fixtures."""

import pytest
from datasets import Dataset


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset.from_dict(
        {
            "id": ["1", "2", "3", "4", "5"],
            "url": ["url1", "url2", "url3", "url4", "url5"],
            "title": ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"],
            "text": [
                "This is a short text.",
                "This is a longer text with multiple sentences. It has more content.",
                "Short.",
                "Medium length text here.\nWith a newline.",
                "Another piece of text for testing purposes.",
            ],
        }
    )


@pytest.fixture
def sample_config():
    """Create a sample WikisetConfig for testing."""
    return {
        "languages": [
            {"lang": "en", "size": 1000},
            {"lang": "fr", "size": "50%"},
        ],
        "date": "latest",
        "seed": 42,
    }
