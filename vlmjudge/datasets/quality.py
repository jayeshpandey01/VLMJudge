# Name: Jayesh Pandey
# Summary: Dataset quality control utilities.

"""
Dataset quality control utilities.

This module evaluates the reliability of pairwise comparison results and classifies
samples into quality tiers for preference-learning datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from vlmjudge.utils.normalization import clamp

logger = logging.getLogger(__name__)


def _variance(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _extract_scores(scores: Mapping[str, Mapping[str, float]]) -> List[float]:
    out: List[float] = []
    for _, d in scores.items():
        try:
            out.append(float(d.get("score", 0.5)))
        except Exception:
            out.append(0.5)
    return out


def _extract_confidences(scores: Mapping[str, Mapping[str, float]]) -> List[float]:
    out: List[float] = []
    for _, d in scores.items():
        try:
            out.append(float(d.get("confidence", 0.0)))
        except Exception:
            out.append(0.0)
    return out


def _get_scores_views(comparison_result: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """
    Support both result shapes:
    - ComparePipeline output: {"scores": {"A": {...}, "B": {...}}}
    - PairwiseComparator output: {"scoresA": {...}, "scoresB": {...}}
    """
    scores = comparison_result.get("scores", {})
    if isinstance(scores, dict) and "A" in scores and "B" in scores:
        scoresA = scores.get("A", {})
        scoresB = scores.get("B", {})
        if isinstance(scoresA, dict) and isinstance(scoresB, dict):
            return scoresA, scoresB

    scoresA = comparison_result.get("scoresA", {})
    scoresB = comparison_result.get("scoresB", {})
    if isinstance(scoresA, dict) and isinstance(scoresB, dict):
        return scoresA, scoresB

    return {}, {}


@dataclass(frozen=True)
class QualityConfig:
    # Classification thresholds
    high_confidence: float = 0.7
    high_disagreement_max: float = 0.2
    medium_confidence: float = 0.5


class QualityEvaluator:
    """
    Evaluates comparison result quality.

    Expected `comparison_result` format is the Phase 3 output dict produced by
    `ComparePipeline.run(...)` (or `PairwiseComparator.compare(...)`).
    """

    def __init__(self, config: Optional[QualityConfig] = None) -> None:
        self._config = config or QualityConfig()

    def evaluate(self, comparison_result: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Compute quality metrics and classification.

        Returns:
            {
              "quality": "high"|"medium"|"low",
              "label": "A"|"B"|"tie",
              "confidence": float[0,1],
              "delta": float[0,1],
              "disagreement": float[0,1],
            }
        """
        winner = str(comparison_result.get("winner", "tie"))
        confidence = float(comparison_result.get("confidence", 0.0))
        confidence = float(clamp(confidence, 0.0, 1.0))

        # Delta from aggregate scores when available, else fall back to `delta` field.
        delta = comparison_result.get("delta", None)
        if isinstance(delta, (int, float)):
            delta_f = abs(float(delta))
        else:
            agg = comparison_result.get("aggregate", {})
            try:
                a = float(agg.get("A", {}).get("score", 0.5))
                b = float(agg.get("B", {}).get("score", 0.5))
                delta_f = abs(a - b)
            except Exception:
                delta_f = 0.0
        delta_f = float(clamp(delta_f, 0.0, 1.0))

        # Disagreement: variance across scorer outputs on A and B.
        # Scores are normalized in [0,1], so max variance is 0.25; sum(A,B) max 0.5.
        scoresA, scoresB = _get_scores_views(comparison_result)

        valsA = _extract_scores(scoresA) if isinstance(scoresA, dict) else []
        valsB = _extract_scores(scoresB) if isinstance(scoresB, dict) else []

        var_sum = _variance(valsA) + _variance(valsB)
        disagreement = float(clamp(var_sum / 0.5, 0.0, 1.0)) if var_sum > 0 else 0.0

        # Coverage: fraction of scorers that actually produced a non-zero-confidence signal.
        confsA = _extract_confidences(scoresA) if isinstance(scoresA, dict) else []
        confsB = _extract_confidences(scoresB) if isinstance(scoresB, dict) else []
        confs = confsA + confsB
        if confs:
            coverage = sum(1 for c in confs if c > 0.0) / len(confs)
        else:
            coverage = 0.0
        coverage = float(clamp(coverage, 0.0, 1.0))

        # Tie handling: always low quality.
        if winner == "tie":
            return {
                "quality": "low",
                "label": "tie",
                "confidence": confidence,
                "delta": delta_f,
                "disagreement": disagreement,
                "coverage": coverage,
            }

        # Classification
        if confidence > self._config.high_confidence and disagreement < self._config.high_disagreement_max:
            quality = "high"
        elif confidence > self._config.medium_confidence:
            quality = "medium"
        else:
            quality = "low"

        return {
            "quality": quality,
            "label": winner,
            "confidence": confidence,
            "delta": delta_f,
            "disagreement": disagreement,
            "coverage": coverage,
        }


_QUALITY_ORDER = {"low": 0, "medium": 1, "high": 2}


def filter_samples(
    samples: Iterable[Mapping[str, Any]],
    min_quality: str = "medium",
    *,
    min_coverage: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Filter preference samples by quality tier.

    Args:
        samples: preference records, each containing a "quality" field.
        min_quality: "low"|"medium"|"high"
        min_coverage: Minimum required scorer coverage in [0,1]. Coverage is the fraction
            of scorer entries with confidence > 0. Samples below this are dropped.
    """
    mq = _QUALITY_ORDER.get(str(min_quality).lower(), 1)
    out: List[Dict[str, Any]] = []
    for s in samples:
        q = str(s.get("quality", "low")).lower()
        cov = 0.0
        try:
            cov = float(s.get("coverage", 0.0))
        except Exception:
            cov = 0.0
        cov = float(clamp(cov, 0.0, 1.0))

        if _QUALITY_ORDER.get(q, 0) >= mq and cov >= float(min_coverage):
            out.append(dict(s))
    return out


def split_dataset(samples: Iterable[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split preference samples into (high, medium, low).
    """
    high: List[Dict[str, Any]] = []
    medium: List[Dict[str, Any]] = []
    low: List[Dict[str, Any]] = []
    for s in samples:
        q = str(s.get("quality", "low")).lower()
        if q == "high":
            high.append(dict(s))
        elif q == "medium":
            medium.append(dict(s))
        else:
            low.append(dict(s))
    return high, medium, low


def quality_report(samples: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary stats for a set of preference records.
    """
    total = 0
    counts = {"high": 0, "medium": 0, "low": 0}
    conf_sum = 0.0
    delta_sum = 0.0
    disagree_sum = 0.0
    ties = 0

    for s in samples:
        total += 1
        q = str(s.get("quality", "low")).lower()
        if q not in counts:
            q = "low"
        counts[q] += 1
        try:
            conf_sum += float(s.get("confidence", 0.0))
        except Exception:
            conf_sum += 0.0
        try:
            delta_sum += float(s.get("delta", 0.0))
        except Exception:
            delta_sum += 0.0
        try:
            disagree_sum += float(s.get("disagreement", 0.0))
        except Exception:
            disagree_sum += 0.0
        if str(s.get("winner", "")).lower() == "tie" or str(s.get("label", "")).lower() == "tie":
            ties += 1

    avg_conf = conf_sum / total if total else 0.0
    avg_delta = delta_sum / total if total else 0.0
    avg_disagree = disagree_sum / total if total else 0.0
    pct = {k: (counts[k] / total if total else 0.0) for k in counts}

    return {
        "total": total,
        "counts": counts,
        "pct": pct,
        "avg_confidence": float(clamp(avg_conf, 0.0, 1.0)),
        "avg_delta": float(clamp(avg_delta, 0.0, 1.0)),
        "avg_disagreement": float(clamp(avg_disagree, 0.0, 1.0)),
        "tie_rate": float(ties / total if total else 0.0),
        "summary": f"Total: {total} | High: {pct['high']*100:.0f}% | Medium: {pct['medium']*100:.0f}% | Low: {pct['low']*100:.0f}% | Ties: {(ties/total*100 if total else 0):.0f}%",
    }
