# Name: Jayesh Pandey
# Summary: Multi-image ranking.

"""
Multi-image ranking.

Scores each image with all unary scorers, aggregates, and sorts descending.
Pairwise-only scorers (e.g., LPIPS) are excluded by default.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Mapping, Optional, Sequence

from vlmjudge.scorers.base import BaseScorer, ImageInput
from vlmjudge.utils.aggregation import aggregate_scores

logger = logging.getLogger(__name__)


def rank_images(
    images: Sequence[ImageInput],
    prompt: str,
    *,
    scorers: Mapping[str, BaseScorer],
    weights: Optional[Mapping[str, float]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> List[Dict]:
    """
    Rank images by aggregated score.

    Returns:
        [{"image": ..., "score": ..., "confidence": ..., "rank": 1}, ...]
    """
    weights = dict(weights or {})
    exclude_set = set(exclude or ("lpips",))

    t0 = time.perf_counter()
    results: List[Dict] = []

    for img in images:
        per = {}
        for name, scorer in scorers.items():
            if name in exclude_set:
                continue
            per[name] = scorer.score(img, prompt=prompt, image_b=None)

        agg = aggregate_scores(per, weights)
        results.append({"image": img, "score": float(agg["score"]), "confidence": float(agg["confidence"])})

    results.sort(key=lambda x: x["score"], reverse=True)
    for i, item in enumerate(results):
        item["rank"] = i + 1

    dt_ms = (time.perf_counter() - t0) * 1000.0
    logger.info("event=rank n=%d ms=%.2f", len(images), dt_ms)
    return results

