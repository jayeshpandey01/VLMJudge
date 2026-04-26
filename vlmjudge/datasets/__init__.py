"""Dataset utilities for generating training-ready preference data."""

from __future__ import annotations

from .builder import build_preference
from .quality import QualityEvaluator, QualityConfig, filter_samples, split_dataset, quality_report

__all__ = [
    "build_preference",
    "QualityEvaluator",
    "QualityConfig",
    "filter_samples",
    "split_dataset",
    "quality_report",
]
