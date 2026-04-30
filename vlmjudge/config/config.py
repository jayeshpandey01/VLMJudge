# Name: Jayesh Pandey
# Summary: Simple config system.

"""
Simple config system.

Phase 1 intentionally keeps this minimal. It provides:
- model selection placeholder
- weights placeholder for future extensibility
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    """
    Global configuration for future pipelines.

    Attributes:
        scorer_name: Which scorer backend to use (placeholder for future selection logic).
        weights_path: Placeholder for custom weights/checkpoints.
        device: Placeholder for device selection (e.g., "cpu", "cuda", "cuda:0").
        download_root: Placeholder for model download/cache directory.
        med_config: Placeholder for ImageReward's BLIP med_config path override.
    """

    scorer_name: str = "image_reward"
    weights_path: Optional[str] = None
    device: Optional[str] = None
    download_root: Optional[str] = None
    med_config: Optional[str] = None

