from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot shadow evaluation metrics.")
    parser.add_argument("--shadow-log", default="logs/shadow_eval.jsonl", type=str)
    parser.add_argument("--out-dir", default="analysis_out", type=str)
    parser.add_argument("--window", default=200, type=int)
    args = parser.parse_args()

    rows = _read_jsonl(args.shadow_log)
    if not rows:
        raise ValueError("No shadow rows found.")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    win = max(10, int(args.window))

    agrees = [1.0 if bool(r.get("agreement", False)) else 0.0 for r in rows]
    gaps = [float(r.get("confidence_gap", 0.0)) for r in rows]

    # Moving average agreement over time
    ma = []
    for i in range(len(agrees)):
        j0 = max(0, i - win + 1)
        ma.append(sum(agrees[j0 : i + 1]) / (i - j0 + 1))

    plt.figure(figsize=(7, 3))
    plt.plot(ma)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.title("Shadow agreement (moving average)")
    plt.xlabel("request index")
    plt.ylabel("agreement rate")
    p1 = Path(args.out_dir) / "agreement_over_time.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()

    # Gap histogram
    plt.figure(figsize=(6, 3))
    plt.hist(gaps, bins=40, alpha=0.85)
    plt.grid(True, alpha=0.3)
    plt.title("Confidence gap distribution")
    plt.xlabel("|student_conf - teacher_conf|")
    plt.ylabel("count")
    p2 = Path(args.out_dir) / "confidence_gap_hist.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=150)
    plt.close()

    print(str(p1))
    print(str(p2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

