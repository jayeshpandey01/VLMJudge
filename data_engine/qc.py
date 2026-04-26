from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from vlmjudge.datasets.quality import filter_samples


def quality_filter(
    samples: Iterable[Mapping[str, Any]],
    *,
    min_quality: str = "medium",
    min_coverage: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Lightweight wrapper around existing QC filtering.
    """
    return filter_samples(samples, min_quality=str(min_quality), min_coverage=float(min_coverage))

