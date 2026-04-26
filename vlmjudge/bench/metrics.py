from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _safe_div(num: float, denom: float) -> float:
    return float(num / denom) if denom else 0.0


@dataclass(frozen=True)
class EvalSummary:
    n: int
    accuracy: float
    ece: float
    mae: float
    avg_confidence: float
    avg_margin: float


def accuracy_from_margins(margins: Sequence[float]) -> float:
    if not margins:
        return 0.0
    correct = sum(1 for m in margins if m > 0.0)
    return float(correct / len(margins))


def mean_absolute_error(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    if not confidences or not correct:
        return 0.0
    n = min(len(confidences), len(correct))
    err = 0.0
    for i in range(n):
        y = 1.0 if bool(correct[i]) else 0.0
        err += abs(float(confidences[i]) - y)
    return float(err / n) if n else 0.0


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    n_bins: int = 10,
) -> float:
    """
    Standard ECE with uniform bins over [0,1].

    Args:
        confidences: predicted probability/confidence in [0,1]
        correct: booleans
        n_bins: number of equal-width bins
    """
    if not confidences or not correct:
        return 0.0
    n = min(len(confidences), len(correct))
    if n == 0:
        return 0.0

    bins_total = [0] * n_bins
    bins_correct = [0] * n_bins
    bins_conf_sum = [0.0] * n_bins

    for i in range(n):
        c = float(confidences[i])
        c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)
        b = int(c * n_bins)
        if b == n_bins:
            b = n_bins - 1
        bins_total[b] += 1
        bins_conf_sum[b] += c
        if bool(correct[i]):
            bins_correct[b] += 1

    ece = 0.0
    for b in range(n_bins):
        if bins_total[b] == 0:
            continue
        acc_b = bins_correct[b] / bins_total[b]
        conf_b = bins_conf_sum[b] / bins_total[b]
        ece += abs(acc_b - conf_b) * (bins_total[b] / n)
    return float(ece)


def binned_accuracy(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    n_bins: int = 10,
) -> Tuple[List[float], List[float], List[int]]:
    if not confidences or not correct:
        return [], [], []
    n = min(len(confidences), len(correct))
    bins_total = [0] * n_bins
    bins_correct = [0] * n_bins
    bins_conf_sum = [0.0] * n_bins

    for i in range(n):
        c = float(confidences[i])
        c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)
        b = int(c * n_bins)
        if b == n_bins:
            b = n_bins - 1
        bins_total[b] += 1
        bins_conf_sum[b] += c
        if bool(correct[i]):
            bins_correct[b] += 1

    xs: List[float] = []
    ys: List[float] = []
    ns: List[int] = []
    for b in range(n_bins):
        if bins_total[b] == 0:
            continue
        xs.append(float(bins_conf_sum[b] / bins_total[b]))
        ys.append(float(bins_correct[b] / bins_total[b]))
        ns.append(int(bins_total[b]))
    return xs, ys, ns


def confusion_matrix_binary(margins: Sequence[float]) -> List[List[int]]:
    """
    Confusion matrix for "chosen wins" (positive label) in a chosen/rejected dataset.

    Actual is always positive (chosen should win), so this is effectively:
        [[TN, FP],
         [FN, TP]]
    with TN/FP always 0. Returned for plotting requirement completeness.
    """
    tp = sum(1 for m in margins if m > 0.0)
    fn = sum(1 for m in margins if m <= 0.0)
    return [[0, 0], [fn, tp]]


def summarize(
    *,
    margins: Sequence[float],
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> EvalSummary:
    n = min(len(margins), len(confidences), len(correct))
    if n == 0:
        return EvalSummary(n=0, accuracy=0.0, ece=0.0, mae=0.0, avg_confidence=0.0, avg_margin=0.0)
    acc = accuracy_from_margins(margins[:n])
    ece = expected_calibration_error(confidences[:n], correct[:n], n_bins=n_bins)
    mae = mean_absolute_error(confidences[:n], correct[:n])
    avg_conf = float(sum(float(c) for c in confidences[:n]) / n)
    avg_margin = float(sum(float(m) for m in margins[:n]) / n)
    return EvalSummary(
        n=int(n),
        accuracy=float(acc),
        ece=float(ece),
        mae=float(mae),
        avg_confidence=float(avg_conf),
        avg_margin=float(avg_margin),
    )

