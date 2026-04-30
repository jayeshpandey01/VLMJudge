# Name: Jayesh Pandey
# Summary: Pairwise comparison engine.

"""
Pairwise comparison engine.

Compares two images under a prompt by:
1) scoring both images with all configured scorers
2) aggregating per-image scores
3) taking a score delta and applying a configurable threshold to decide winner/tie
4) producing a final confidence
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from vlmjudge.comparators.base import BaseComparator
from vlmjudge.scorers.base import BaseScorer, ImageInput, ScoreOutput
from vlmjudge.utils.aggregation import aggregate_scores
from vlmjudge.utils.normalization import clamp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairwiseComparatorConfig:
    threshold: float = 0.05


class PairwiseComparator(BaseComparator):
    """
    Compare two images under a prompt using multiple scorers + aggregation.

    The output is designed to be training-ready and safe for downstream aggregation.
    """

    def __init__(
        self,
        scorers: Mapping[str, BaseScorer],
        *,
        weights: Optional[Mapping[str, float]] = None,
        config: Optional[PairwiseComparatorConfig] = None,
    ) -> None:
        self._scorers = dict(scorers)
        self._weights = dict(weights or {})
        self._config = config or PairwiseComparatorConfig()

    def _run_all_scorers(self, image: ImageInput, *, prompt: Optional[str], image_b: Optional[ImageInput]) -> Dict[str, ScoreOutput]:
        out: Dict[str, ScoreOutput] = {}
        for name, scorer in self._scorers.items():
            try:
                out[name] = scorer.score(image, prompt=prompt, image_b=image_b)
            except Exception as e:
                logger.warning("event=scorer_crash scorer=%s err=%s", name, e)
                out[name] = {"score": 0.5, "confidence": 0.0}
        return out

    def compare(self, imgA: ImageInput, imgB: ImageInput, prompt: str, weights_override: Optional[Mapping[str, float]] = None):
        t0 = time.perf_counter()

        scoresA = self._run_all_scorers(imgA, prompt=prompt, image_b=imgB)
        scoresB = self._run_all_scorers(imgB, prompt=prompt, image_b=imgA)

        weights = weights_override if weights_override is not None else self._weights

        aggA = aggregate_scores(scoresA, weights)
        aggB = aggregate_scores(scoresB, weights)

        finalA = float(aggA["score"])
        confA = float(aggA["confidence"])
        finalB = float(aggB["score"])
        confB = float(aggB["confidence"])

        delta = finalA - finalB
        thr = float(self._config.threshold)

        if delta > thr:
            winner = "A"
        elif delta < -thr:
            winner = "B"
        else:
            winner = "tie"

        confidence = abs(delta) * (confA + confB) * 0.5
        confidence = float(clamp(confidence, 0.0, 1.0))

        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "event=compare winner=%s delta=%.4f thr=%.4f conf=%.4f ms=%.2f",
            winner,
            delta,
            thr,
            confidence,
            dt_ms,
        )

        return {
            "winner": winner,
            "confidence": confidence,
            "threshold": thr,
            "delta": float(delta),
            "aggregateA": aggA,
            "aggregateB": aggB,
            "scoresA": scoresA,
            "scoresB": scoresB,
            "timing_ms": dt_ms,
        }

