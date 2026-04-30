# Name: Jayesh Pandey
# Summary: Scorers: prompt-conditioned image scoring backends (reward models, metrics, etc.).

"""Scorers: prompt-conditioned image scoring backends (reward models, metrics, etc.)."""

from __future__ import annotations

from .base import BaseScorer, ImageInput
from .image_reward import ImageRewardScorer, ImageRewardScorerConfig
from .clip_score import OpenCLIPScorer, OpenCLIPScorerConfig
from .aesthetic import AestheticScorer, AestheticScorerConfig
from .lpips import LPIPSScorer, LPIPSScorerConfig
from .distilled_scorer import DistilledScorer, DistilledScorerConfig
from .reasoning_score import ReasoningScorer, ReasoningScorerConfig

try:
    from .dino_score import DINOv2Scorer, DINOv2ScorerConfig
except Exception:  # optional
    DINOv2Scorer = None  # type: ignore
    DINOv2ScorerConfig = None  # type: ignore

__all__ = [
    "BaseScorer",
    "ImageInput",
    "ImageRewardScorer",
    "ImageRewardScorerConfig",
    "OpenCLIPScorer",
    "OpenCLIPScorerConfig",
    "AestheticScorer",
    "AestheticScorerConfig",
    "LPIPSScorer",
    "LPIPSScorerConfig",
    "DistilledScorer",
    "DistilledScorerConfig",
    "ReasoningScorer",
    "ReasoningScorerConfig",
    "DINOv2Scorer",
    "DINOv2ScorerConfig",
]
