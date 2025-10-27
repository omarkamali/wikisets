"""Configuration classes for Wikiset dataset construction."""

from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class WikisetConfig:
    """Configuration for building a Wikiset dataset.

    Attributes:
        languages: List of {lang: str, size: Union[int, float, str]} dictionaries.
        date: Wikipedia dump date in yyyymmdd format (default: "latest").
        use_train_split: Force sampling from "train" split only (default: False).
        shuffle: Whether to interleave languages proportionally (default: False).
        seed: Random seed for reproducibility (default: 42).
        num_proc: Number of processes for parallel operations (default: None).
    """

    languages: list[dict[str, Union[str, int, float]]]
    date: str = "latest"
    use_train_split: bool = False
    shuffle: bool = False
    seed: int = 42
    num_proc: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.languages:
            raise ValueError("languages list cannot be empty")

        for entry in self.languages:
            if "lang" not in entry:
                raise ValueError(f"Missing 'lang' key in entry: {entry}")
            if "size" not in entry:
                raise ValueError(f"Missing 'size' key in entry: {entry}")

            # Validate size type
            size = entry["size"]
            if not isinstance(size, (int, float, str)):
                raise ValueError(
                    f"size must be int, float, or str with %, got {type(size).__name__}"
                )

            # Validate percentage strings
            if isinstance(size, str):
                if not size.endswith("%"):
                    raise ValueError(f"String size must end with '%', got: {size}")
                try:
                    percent = float(size.rstrip("%"))
                    if not 0 <= percent <= 100:
                        raise ValueError(f"Percentage must be 0-100, got: {percent}")
                except ValueError as e:
                    raise ValueError(f"Invalid percentage string: {size}") from e

            # Validate numeric ranges
            if isinstance(size, (int, float)) and not isinstance(size, bool):
                if size < 0:
                    raise ValueError(f"size cannot be negative, got: {size}")
                if isinstance(size, float) and not 0 <= size <= 1:
                    raise ValueError(f"Float size must be 0-1, got: {size}")

        # Validate date format for explicit dates
        if self.date != "latest":
            if not self.date.isdigit() or len(self.date) != 8:
                raise ValueError(
                    f"date must be 'latest' or yyyymmdd format, got: {self.date}"
                )

        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "WikisetConfig":
        """Create WikisetConfig from dictionary.

        Args:
            config_dict: Dictionary with configuration parameters.

        Returns:
            WikisetConfig instance.
        """
        return cls(**config_dict)
