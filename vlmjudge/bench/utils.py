from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def read_json(path: str) -> Any:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8-sig"))


def normalize_preference_record(item: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize various dataset shapes into:
        {prompt, chosen, rejected, meta...}
    """
    prompt = item.get("prompt", None)
    if not isinstance(prompt, str) or not prompt.strip():
        return None

    if item.get("chosen") is not None and item.get("rejected") is not None:
        chosen = item.get("chosen")
        rejected = item.get("rejected")
        if not isinstance(chosen, str) or not isinstance(rejected, str):
            return None
        out = dict(item)
        out["prompt"] = prompt
        out["chosen"] = chosen
        out["rejected"] = rejected
        return out

    # Support "imgA/imgB + winner" format.
    imgA = item.get("imgA", None)
    imgB = item.get("imgB", None)
    winner = str(item.get("winner", item.get("final_winner", "tie")))
    if isinstance(imgA, str) and isinstance(imgB, str) and winner in ("A", "B"):
        chosen = imgA if winner == "A" else imgB
        rejected = imgB if winner == "A" else imgA
        out = dict(item)
        out["prompt"] = prompt
        out["chosen"] = chosen
        out["rejected"] = rejected
        return out
    return None


def load_preference_dataset(path: str, *, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    raw = read_json(path)
    if not isinstance(raw, list):
        raise ValueError("Dataset must be a JSON list.")

    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        rec = normalize_preference_record(item)
        if rec is None:
            continue
        out.append(rec)
        if max_samples is not None and len(out) >= int(max_samples):
            break
    return out


def get_teacher_fields(item: Mapping[str, Any]) -> Tuple[float, float, float]:
    """
    Returns (confidence, coverage, delta) in [0,1] when available.
    """
    def _clamp01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    conf_raw = item.get("fusion_confidence", None)
    if conf_raw is None:
        conf_raw = item.get("final_confidence", None)
    if conf_raw is None:
        conf_raw = item.get("confidence", 0.0)
    try:
        conf = _clamp01(float(conf_raw))
    except Exception:
        conf = 0.0

    try:
        cov = _clamp01(float(item.get("coverage", 0.0)))
    except Exception:
        cov = 0.0

    delta_raw = item.get("delta", None)
    if delta_raw is None:
        delta_raw = (item.get("metadata", {}) or {}).get("delta", 0.0)
    try:
        delta = _clamp01(abs(float(delta_raw)))
    except Exception:
        delta = 0.0

    return float(conf), float(cov), float(delta)

