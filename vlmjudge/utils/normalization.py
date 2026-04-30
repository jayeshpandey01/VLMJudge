# Name: Jayesh Pandey
# Summary: Score normalization utilities.

"""
Score normalization utilities.

These helpers are intentionally small and dependency-light so they can be used across
scorers without pulling in heavy frameworks at import time.
"""

from __future__ import annotations

from typing import Optional
import math


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp `value` into the closed interval [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def min_max_normalize(
    value: float,
    min_value: float,
    max_value: float,
    *,
    clamp_to_unit: bool = True,
) -> Optional[float]:
    """
    Min-max normalize a scalar to [0, 1].

    Returns None if `max_value <= min_value` to avoid division by zero.
    """
    denom = max_value - min_value
    if denom <= 0:
        return None
    normalized = (value - min_value) / denom
    return clamp(normalized, 0.0, 1.0) if clamp_to_unit else normalized


def cosine_to_unit_interval(cosine_similarity: float) -> float:
    """
    Map cosine similarity in [-1, 1] to [0, 1] and clamp for numerical safety.
    """
    return clamp((cosine_similarity + 1.0) * 0.5, 0.0, 1.0)


def sigmoid(x: float) -> float:
    """
    Numerically-stable sigmoid for scalar floats.

    Returns a value in (0, 1), clamped to [0, 1] for safety.
    """
    # Stable branches to avoid overflow for large |x|.
    if x >= 0:
        z = math.exp(-x)
        s = 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        s = z / (1.0 + z)
    return clamp(s, 0.0, 1.0)

