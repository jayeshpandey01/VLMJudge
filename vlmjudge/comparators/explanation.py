"""
Basic explanation generator (no external APIs).

Produces a short natural-language explanation based on which scorers most contributed
to the final decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Mapping, Optional, Tuple, List

from vlmjudge.utils.normalization import clamp

logger = logging.getLogger(__name__)


_DEFAULT_PHRASES = {
    "openclip": "higher CLIP alignment to the prompt",
    "clip": "higher CLIP alignment to the prompt",
    "image_reward": "higher ImageReward preference score",
    "aesthetic": "better aesthetic quality",
    "lpips": "closer perceptual match (LPIPS)",
    "dinov2": "richer semantic features (DINOv2)",
}


@dataclass(frozen=True)
class ExplanationConfig:
    top_k: int = 3


def _contribution(
    score_a: float,
    conf_a: float,
    score_b: float,
    conf_b: float,
    weight: float,
) -> float:
    # Contribution magnitude: how much this scorer could have moved the decision.
    avg_conf = 0.5 * (clamp(conf_a, 0.0, 1.0) + clamp(conf_b, 0.0, 1.0))
    return abs(score_a - score_b) * max(weight, 0.0) * avg_conf


def generate_explanation(
    *,
    prompt: str,
    winner: str,
    scoresA: Mapping[str, Mapping[str, float]],
    scoresB: Mapping[str, Mapping[str, float]],
    weights: Optional[Mapping[str, float]] = None,
    config: Optional[ExplanationConfig] = None,
) -> str:
    """
    Generate a short explanation mentioning 2-3 key factors.
    """
    weights = weights or {}
    cfg = config or ExplanationConfig()

    contribs: List[Tuple[str, float, float, float]] = []
    for name in scoresA.keys():
        a = scoresA.get(name, {})
        b = scoresB.get(name, {})
        sa = float(a.get("score", 0.5))
        ca = float(a.get("confidence", 0.0))
        sb = float(b.get("score", 0.5))
        cb = float(b.get("confidence", 0.0))
        w = float(weights.get(name, 1.0))
        c = _contribution(sa, ca, sb, cb, w)
        contribs.append((name, c, sa, sb))

    contribs.sort(key=lambda x: x[1], reverse=True)
    top = [t for t in contribs[: max(1, cfg.top_k)] if t[1] > 0]

    if not top:
        if winner == "tie":
            return "The two images are close in overall quality under the prompt, so the system returned a tie."
        return f"Image {winner} is preferred overall based on the combined scorer signals."

    factors = []
    for name, _, sa, sb in top[:3]:
        phrase = _DEFAULT_PHRASES.get(name, f"stronger signal from {name}")
        if winner == "A":
            direction = "A" if sa >= sb else "B"
        elif winner == "B":
            direction = "B" if sb >= sa else "A"
        else:
            direction = None

        if direction is None:
            factors.append(phrase)
        else:
            factors.append(phrase)

    factors = factors[:3]
    if winner == "tie":
        return "The two images are close; the decision is driven mainly by " + ", ".join(factors) + "."
    return f"Image {winner} is preferred due to " + ", ".join(factors) + "."

