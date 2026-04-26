from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    n_bootstrap: int


def bootstrap_mean(
    values: Sequence[float],
    *,
    n_bootstrap: int = 200,
    seed: int = 123,
) -> BootstrapResult:
    """
    Bootstrap the mean over `values` using sampling with replacement.
    Returns mean/std and 95% percentile CI.
    """
    n = len(values)
    if n == 0:
        return BootstrapResult(mean=0.0, std=0.0, ci95_low=0.0, ci95_high=0.0, n_bootstrap=int(n_bootstrap))

    rnd = random.Random(int(seed))
    means: List[float] = []
    for _ in range(int(n_bootstrap)):
        s = 0.0
        for _j in range(n):
            s += float(values[rnd.randrange(n)])
        means.append(s / n)

    means_sorted = sorted(means)
    m = sum(means) / len(means)
    var = sum((x - m) ** 2 for x in means) / max(1, (len(means) - 1))
    std = var ** 0.5

    def pct(p: float) -> float:
        idx = int(round(p * (len(means_sorted) - 1)))
        idx = 0 if idx < 0 else (len(means_sorted) - 1 if idx >= len(means_sorted) else idx)
        return float(means_sorted[idx])

    return BootstrapResult(
        mean=float(m),
        std=float(std),
        ci95_low=pct(0.025),
        ci95_high=pct(0.975),
        n_bootstrap=int(n_bootstrap),
    )


def prompt_length_words(prompt: str) -> int:
    p = (prompt or "").strip()
    if not p:
        return 0
    return len(p.split())


def bucket_prompt_length(n_words: int) -> str:
    if n_words < 5:
        return "short"
    if n_words <= 10:
        return "medium"
    return "long"


def bucket_confidence(c: float) -> str:
    c = _clamp01(float(c))
    if c < 0.4:
        return "low"
    if c < 0.7:
        return "medium"
    return "high"


def bucket_similarity(sim: Optional[float]) -> str:
    if sim is None:
        return "unknown"
    s = _clamp01(float(sim))
    if s >= 0.8:
        return "very_similar"
    if s >= 0.5:
        return "moderate"
    return "very_different"


def group_indices(labels: Sequence[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for i, lab in enumerate(labels):
        out.setdefault(str(lab), []).append(int(i))
    return out


def indices_to_mask(n: int, idxs: Sequence[int]) -> List[bool]:
    mask = [False] * int(n)
    for i in idxs:
        if 0 <= int(i) < int(n):
            mask[int(i)] = True
    return mask

