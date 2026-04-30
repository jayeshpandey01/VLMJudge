# Name: Jayesh Pandey
# Summary: Simple drift heuristics:

from __future__ import annotations

import argparse
import json
import math
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


def _safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "agreement_rate": 0.0, "avg_confidence_gap": 0.0, "disagreement_rate": 0.0, "high_conf_disagreement_pct": 0.0}

    agree = 0
    gaps: List[float] = []
    high_conf_dis = 0
    dis = 0

    by_variant: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        a = bool(r.get("agreement", False))
        if a:
            agree += 1
        else:
            dis += 1

        try:
            gaps.append(float(r.get("confidence_gap", 0.0)))
        except Exception:
            gaps.append(0.0)

        try:
            sconf = float((r.get("student", {}) or {}).get("confidence", 0.0))
        except Exception:
            sconf = 0.0
        if (not a) and sconf >= 0.8:
            high_conf_dis += 1

        variant = str((r.get("student", {}) or {}).get("variant", "unknown"))
        v = by_variant.setdefault(variant, {"n": 0, "agree": 0, "gap_sum": 0.0, "high_conf_dis": 0})
        v["n"] += 1
        v["agree"] += 1 if a else 0
        v["gap_sum"] += float(gaps[-1])
        v["high_conf_dis"] += 1 if ((not a) and sconf >= 0.8) else 0

    out = {
        "n": n,
        "agreement_rate": float(agree / n),
        "avg_confidence_gap": float(_safe_mean(gaps)),
        "disagreement_rate": float(dis / n),
        "high_conf_disagreement_pct": float(high_conf_dis / n),
        "by_variant": {},
    }
    for k, v in by_variant.items():
        nn = int(v["n"]) or 1
        out["by_variant"][k] = {
            "n": int(v["n"]),
            "agreement_rate": float(v["agree"] / nn),
            "avg_confidence_gap": float(v["gap_sum"] / nn),
            "high_conf_disagreement_pct": float(v["high_conf_dis"] / nn),
        }
    return out


def drift_detect(rows: List[Dict[str, Any]], *, window: int = 500) -> List[str]:
    """
    Simple drift heuristics:
      - disagreement rate increases > 1.5x vs previous window (and prev >= 5%)
      - avg confidence gap increases > +0.10 vs previous
    """
    if len(rows) < 2 * window:
        return []
    cur = rows[-window:]
    prev = rows[-2 * window : -window]
    m_cur = compute_metrics(cur)
    m_prev = compute_metrics(prev)

    warn: List[str] = []
    if float(m_prev["disagreement_rate"]) >= 0.05 and float(m_cur["disagreement_rate"]) > 1.5 * float(m_prev["disagreement_rate"]):
        warn.append("Possible model drift detected: disagreement_rate_spike")
    if float(m_cur["avg_confidence_gap"]) > float(m_prev["avg_confidence_gap"]) + 0.10:
        warn.append("Possible model drift detected: confidence_gap_increase")
    return warn


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate shadow teacher evaluation metrics.")
    parser.add_argument("--shadow-log", default="logs/shadow_eval.jsonl", type=str)
    parser.add_argument("--window", default=500, type=int)
    parser.add_argument("--out", default=None, type=str, help="Optional JSON output path.")
    args = parser.parse_args()

    rows = _read_jsonl(args.shadow_log)
    metrics = compute_metrics(rows)
    warnings = drift_detect(rows, window=int(args.window))
    metrics["warnings"] = warnings

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    for w in warnings:
        print(w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

