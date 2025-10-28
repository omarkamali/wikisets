"""Utility functions for Wikiset."""

import warnings
from typing import Optional, Union

import numpy as np


class WarningTracker:
    """Track warnings for inclusion in dataset cards."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warn(self, message: str, category: type = UserWarning) -> None:
        """Issue a warning and track it.

        Args:
            message: Warning message.
            category: Warning category.
        """
        self.warnings.append(message)
        warnings.warn(message, category, stacklevel=2)

    def get_warnings(self) -> list[str]:
        """Get all tracked warnings.

        Returns:
            List of warning messages.
        """
        return self.warnings.copy()

    def clear(self) -> None:
        """Clear tracked warnings."""
        self.warnings.clear()


def parse_size(size: Union[str, int, float], total: Optional[int] = None) -> tuple[int, str]:
    """Parse size parameter to get target count and description.

    Args:
        size: Size specification (int, float 0-1, or str with %).
        total: Total available items (needed for percentages).

    Returns:
        Tuple of (target_count, description_string).
    """
    if isinstance(size, str):
        # Percentage string
        percent = float(size.rstrip("%"))
        if total is None:
            raise ValueError("total required for percentage size")
        target = max(1, int(np.ceil(percent / 100.0 * total)))
        return target, f"{percent}% (~{target} items)"

    elif isinstance(size, float):
        # Fraction 0-1
        if total is None:
            raise ValueError("total required for fractional size")
        target = max(1, int(np.ceil(size * total)))
        return target, f"{size * 100:.1f}% (~{target} items)"

    else:
        # Absolute integer
        return int(size), f"{size} items"


def select_split_for_size(size: int, use_train_split: bool) -> str:
    """Determine which split to use based on requested size.

    Args:
        size: Target number of items.
        use_train_split: If True, always return "train".

    Returns:
        Split name to use.
    """
    if use_train_split:
        return "train"

    # Exact matches for sample splits
    if size == 10000:
        return "10000"
    elif size == 5000:
        return "5000"
    elif size == 1000:
        return "1000"

    # For sizes <= 10k, use smallest sample that fits
    if size <= 1000:
        return "1000"
    elif size <= 5000:
        return "5000"
    elif size <= 10000:
        return "10000"
    else:
        # Sizes > 10k always use train
        return "train"
