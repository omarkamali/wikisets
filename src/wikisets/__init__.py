"""Wikisets: Flexible Wikipedia dataset builder."""

__version__ = "0.1.2"

from .config import WikisetConfig
from .wikiset import Wikiset

__all__ = ["Wikiset", "WikisetConfig"]
