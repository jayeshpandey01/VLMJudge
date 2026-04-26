from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")  # safe for colab/headless
import matplotlib.pyplot as plt

from vlmjudge.bench.metrics import binned_accuracy, confusion_matrix_binary


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_accuracy_vs_confidence(
    out_dir: str,
    *,
    confidences: Sequence[float],
    correct: Sequence[bool],
    title: str,
    n_bins: int = 10,
    filename: str = "accuracy_vs_confidence.png",
) -> str:
    out_path = Path(out_dir) / filename
    _ensure_dir(out_path.parent)

    xs, ys, ns = binned_accuracy(confidences, correct, n_bins=n_bins)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel("Predicted confidence (bin mean)")
    plt.ylabel("Empirical accuracy")
    for x, y, n in zip(xs, ys, ns):
        plt.annotate(str(n), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path)


def plot_calibration_curve(
    out_dir: str,
    *,
    confidences: Sequence[float],
    correct: Sequence[bool],
    title: str,
    n_bins: int = 10,
    filename: str = "calibration_curve.png",
) -> str:
    out_path = Path(out_dir) / filename
    _ensure_dir(out_path.parent)

    xs, ys, _ = binned_accuracy(confidences, correct, n_bins=n_bins)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.plot(xs, ys, marker="o")
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path)


def plot_margin_histogram(
    out_dir: str,
    *,
    margins: Sequence[float],
    title: str,
    filename: str = "margin_hist.png",
) -> str:
    out_path = Path(out_dir) / filename
    _ensure_dir(out_path.parent)

    plt.figure(figsize=(6, 4))
    plt.hist(list(margins), bins=40, alpha=0.85)
    plt.title(title)
    plt.xlabel("score(chosen) - score(rejected)")
    plt.ylabel("count")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path)


def plot_confusion_matrix(
    out_dir: str,
    *,
    margins: Sequence[float],
    title: str,
    filename: str = "confusion_matrix.png",
) -> str:
    out_path = Path(out_dir) / filename
    _ensure_dir(out_path.parent)

    cm = confusion_matrix_binary(margins)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xticks([0, 1], ["pred B", "pred A"])
    plt.yticks([0, 1], ["true B", "true A"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path)


def plot_accuracy_latency(
    out_dir: str,
    *,
    labels: Sequence[str],
    accuracies: Sequence[float],
    latencies_ms: Sequence[float],
    title: str,
    filename: str = "accuracy_latency.png",
) -> str:
    out_path = Path(out_dir) / filename
    _ensure_dir(out_path.parent)

    xs = [float(x) for x in latencies_ms]
    ys = [float(y) for y in accuracies]

    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys)
    for lab, x, y in zip(labels, xs, ys):
        plt.annotate(str(lab), (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel("Latency (ms / sample)")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path)
