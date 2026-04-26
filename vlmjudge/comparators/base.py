"""
Comparator interfaces.

In Phase 1, comparators are only structured; no comparison logic is implemented yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseComparator(ABC):
    """
    Base interface for comparing two images given a prompt.

    The return type is intentionally left open for future extensibility (e.g., winner label,
    probability, margin, or richer diagnostics).
    """

    @abstractmethod
    def compare(self, imgA: Any, imgB: Any, prompt: str):
        """Compare two images under a prompt. Logic intentionally not implemented in Phase 1."""
        raise NotImplementedError

