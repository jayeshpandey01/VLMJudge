# Name: Jayesh Pandey
# Summary: Pipelines: orchestration layers for multi-step scoring/comparison workflows.

"""Pipelines: orchestration layers for multi-step scoring/comparison workflows."""

from __future__ import annotations

from .base_pipeline import BasePipeline
from .compare_pipeline import ComparePipeline

__all__ = ["BasePipeline", "ComparePipeline"]
