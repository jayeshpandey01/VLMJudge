from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
    return out


def _window_stats(compare: List[Dict[str, Any]]) -> Dict[str, float]:
    confs = []
    disagreements = 0
    teacher_used = 0
    for r in compare:
        try:
            confs.append(float(r.get("confidence", 0.0)))
        except Exception:
            pass
        if str(r.get("method", "")) in ("teacher", "hybrid"):
            teacher_used += 1
            if r.get("agreement", None) is False:
                disagreements += 1
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    disagree_rate = disagreements / teacher_used if teacher_used else 0.0
    return {"avg_confidence": float(avg_conf), "disagreement_rate": float(disagree_rate), "teacher_used": float(teacher_used)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight monitoring dashboard (Phase 8).")
    parser.add_argument("--requests", default="logs/requests.jsonl", type=str)
    parser.add_argument("--feedback", default="logs/feedback.jsonl", type=str)
    parser.add_argument("--registry", default="models/registry.json", type=str)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    args = parser.parse_args()

    reqs = _read_jsonl(args.requests)
    fb = _read_jsonl(args.feedback)

    compare = [r for r in reqs if str(r.get("type", "")) in ("compare", "batch_compare")]
    scores = [r for r in reqs if str(r.get("type", "")) == "score"]
    flagged_fb = _read_jsonl("logs/flagged_feedback.jsonl")

    stats_all = _window_stats(compare)
    avg_conf = float(stats_all["avg_confidence"])
    disagree_rate = float(stats_all["disagreement_rate"])
    teacher_used = int(stats_all["teacher_used"])

    # Trend windows
    w = 200
    last = compare[-w:] if len(compare) >= w else compare
    prev = compare[-2 * w : -w] if len(compare) >= 2 * w else []
    last_stats = _window_stats(last)
    prev_stats = _window_stats(prev) if prev else {"avg_confidence": avg_conf, "disagreement_rate": disagree_rate, "teacher_used": 0.0}

    current_version = None
    dataset_size = None
    new_ratio = None
    if Path(args.registry).exists():
        try:
            reg = json.loads(Path(args.registry).read_text(encoding="utf-8"))
            cur = reg.get("current", None)
            if isinstance(cur, dict):
                current_version = cur.get("version", None)
                dataset_size = cur.get("dataset_size", None)
                nnew = cur.get("new_samples_kept", None)
                if isinstance(dataset_size, int) and isinstance(nnew, int) and dataset_size > 0:
                    new_ratio = float(nnew / dataset_size)
        except Exception:
            current_version = None

    feedback_ratio = float(len(fb) / max(1, len(compare))) if compare else 0.0

    summary = {
        "total_requests": len(reqs),
        "compare_requests": len(compare),
        "score_requests": len(scores),
        "avg_confidence": avg_conf,
        "teacher_used": teacher_used,
        "disagreement_rate": disagree_rate,
        "feedback_entries": len(fb),
        "flagged_feedback_entries": len(flagged_fb),
        "feedback_ratio": feedback_ratio,
        "model_version": current_version or "unknown",
        "dataset_size_last_train": dataset_size,
        "new_ratio_last_train": new_ratio,
        "trend": {
            "last_window": last_stats,
            "prev_window": prev_stats,
        },
        "warnings": [],
    }

    if new_ratio is not None and new_ratio > 0.60:
        summary["warnings"].append("new_data_ratio_gt_60pct")
    if prev and float(last_stats["disagreement_rate"]) > max(0.10, float(prev_stats["disagreement_rate"]) * 1.5):
        summary["warnings"].append("disagreement_spike")
    if prev and float(last_stats["avg_confidence"]) < float(prev_stats["avg_confidence"]) - 0.05:
        summary["warnings"].append("confidence_drop")

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print("=== MONITOR ===")
    print(f"Total requests: {summary['total_requests']}")
    print(f"Compare requests: {summary['compare_requests']} | Score requests: {summary['score_requests']}")
    print(f"Avg confidence (compare): {summary['avg_confidence']:.3f}")
    print(f"Teacher used: {summary['teacher_used']} | Disagreement rate (when teacher used): {summary['disagreement_rate']*100:.2f}%")
    print(
        f"Feedback entries: {summary['feedback_entries']} | Flagged feedback: {summary['flagged_feedback_entries']} | Feedback ratio: {summary['feedback_ratio']*100:.2f}%"
    )
    print(f"Model version: {summary['model_version']}")
    if summary["dataset_size_last_train"] is not None:
        print(f"Dataset size (last train): {summary['dataset_size_last_train']}")
    if summary["new_ratio_last_train"] is not None:
        print(f"New/total ratio (last train): {summary['new_ratio_last_train']*100:.1f}%")
    for w in summary["warnings"]:
        print(f"WARNING: {w}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
