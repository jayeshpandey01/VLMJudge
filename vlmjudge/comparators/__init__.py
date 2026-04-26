"""Comparators: pairwise image comparison interfaces (logic not implemented in Phase 1)."""

from __future__ import annotations

from .base import BaseComparator
from .pairwise import PairwiseComparator, PairwiseComparatorConfig

__all__ = ["BaseComparator", "PairwiseComparator", "PairwiseComparatorConfig"]
