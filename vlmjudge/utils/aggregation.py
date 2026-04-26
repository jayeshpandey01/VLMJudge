"""
Score aggregation utilities (prep for Phase 3).

Aggregates multiple scorer outputs using:
  contribution = weight * confidence
  weighted_score = sum(score * contribution) / sum(contribution)

If all contributions are zero, returns a neutral default.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

from vlmjudge.utils.normalization import clamp

ScoreOutput = Dict[str, float]


def aggregate_scores(
    score_dict: Mapping[str, Mapping[str, float]],
    weights: Optional[Mapping[str, float]] = None,
) -> ScoreOutput:
    """
    Aggregate multiple scorer outputs into a single normalized score.

    Args:
        score_dict: Mapping like {"clip": {"score": 0.7, "confidence": 0.9}, ...}.
        weights: Optional mapping of scorer_name -> weight. Missing keys default to 1.0.

    Returns:
        {"score": float in [0,1], "confidence": float in [0,1]} where confidence is the
        normalized total contribution relative to total possible weight.
    """
    weights = weights or {}

    num = 0.0
    denom = 0.0
    total_weight = 0.0

    for name, out in score_dict.items():
        w = float(weights.get(name, 1.0))
        if w < 0:
            continue
        total_weight += w

        score = float(out.get("score", 0.5))
        conf = float(out.get("confidence", 0.0))
        conf = clamp(conf, 0.0, 1.0)
        score = clamp(score, 0.0, 1.0)

        contribution = w * conf
        num += score * contribution
        denom += contribution

    if denom <= 1e-12:
        return {"score": 0.5, "confidence": 0.0}

    agg_score = num / denom
    agg_score = clamp(agg_score, 0.0, 1.0)

    # Confidence reflects how much "trusted weight" participated.
    agg_conf = denom / max(total_weight, 1e-12)
    agg_conf = clamp(agg_conf, 0.0, 1.0)

    return {"score": agg_score, "confidence": agg_conf}

