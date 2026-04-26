from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@dataclass(frozen=True)
class SelectionConfig:
    low_conf_threshold: float = 0.6
    high_conf_threshold: float = 0.8
    max_samples: Optional[int] = 5000


def _key(prompt: str, image_a: str, image_b: str) -> str:
    return f"{prompt.strip()}||{image_a.strip()}||{image_b.strip()}"


def select_samples(
    *,
    requests_path: str = "logs/requests.jsonl",
    feedback_path: str = "logs/feedback.jsonl",
    flagged_feedback_path: str = "logs/flagged_feedback.jsonl",
    shadow_path: str = "logs/shadow_eval.jsonl",
    config: Optional[SelectionConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Select "valuable" samples from API logs.

    Rules:
      - Low confidence: confidence < threshold
      - Disagreement: student_winner != teacher_winner when teacher used
      - High-impact wrong: high-confidence predictions later corrected by feedback
    """
    cfg = config or SelectionConfig()
    reqs = _read_jsonl(requests_path)
    fb = _read_jsonl(feedback_path)
    flagged = _read_jsonl(flagged_feedback_path)
    shadow = _read_jsonl(shadow_path)

    flagged_keys: set[str] = set()
    for item in flagged:
        try:
            prompt = str(item.get("prompt", ""))
            image_a = str(item.get("imageA", ""))
            image_b = str(item.get("imageB", ""))
        except Exception:
            continue
        if prompt and image_a and image_b:
            flagged_keys.add(_key(prompt, image_a, image_b))

    # Index feedback by (prompt,imageA,imageB) for fast lookup.
    fb_map: Dict[str, str] = {}
    for item in fb:
        try:
            prompt = str(item.get("prompt", ""))
            image_a = str(item.get("imageA", ""))
            image_b = str(item.get("imageB", ""))
            cw = str(item.get("correct_winner", ""))
        except Exception:
            continue
        if cw not in ("A", "B", "tie"):
            continue
        k = _key(prompt, image_a, image_b)
        if k in flagged_keys or bool(item.get("flagged", False)):
            continue
        fb_map[k] = cw

    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for r in reqs:
        if str(r.get("type", "")) not in ("compare", "batch_compare"):
            continue
        prompt = str(r.get("prompt", ""))
        image_a = str(r.get("imageA", ""))
        image_b = str(r.get("imageB", ""))
        if not prompt or not image_a or not image_b:
            continue
        k = _key(prompt, image_a, image_b)
        if k in seen:
            continue

        winner = str(r.get("winner", "tie"))
        if winner not in ("A", "B", "tie"):
            winner = "tie"

        try:
            conf = _clamp01(float(r.get("confidence", 0.0)))
        except Exception:
            conf = 0.0

        method = str(r.get("method", "student"))
        student_w = r.get("student_winner", None)
        teacher_w = r.get("teacher_winner", None)
        used_teacher = teacher_w in ("A", "B", "tie") and method in ("teacher", "hybrid")

        reason: Optional[str] = None
        if conf < float(cfg.low_conf_threshold):
            reason = "low_confidence"
        if used_teacher and student_w in ("A", "B", "tie") and teacher_w in ("A", "B", "tie") and student_w != teacher_w:
            reason = "disagreement"

        fb_key = fb_map.get(k)
        if fb_key is not None:
            # High-impact wrong: model was confident but feedback says otherwise.
            if conf >= float(cfg.high_conf_threshold) and winner in ("A", "B") and fb_key in ("A", "B") and winner != fb_key:
                reason = "high_impact_wrong"

        if reason is None:
            continue

        seen.add(k)
        selected.append(
            {
                "prompt": prompt,
                "imageA": image_a,
                "imageB": image_b,
                "winner": winner,
                "confidence": conf,
                "method": method,
                "student_winner": student_w,
                "teacher_winner": teacher_w,
                "agreement": r.get("agreement", None),
                "latency_ms": r.get("latency_ms", None),
                "feedback_correct_winner": fb_key,
                "flagged": False,
                "selection_reason": reason,
                "timestamp": r.get("timestamp", None),
            }
        )

        if cfg.max_samples is not None and len(selected) >= int(cfg.max_samples):
            break

    # Shadow disagreement mining (prioritized)
    for r in shadow:
        try:
            prompt = str(r.get("prompt", ""))
            image_a = str(r.get("imageA", ""))
            image_b = str(r.get("imageB", ""))
        except Exception:
            continue
        if not prompt or not image_a or not image_b:
            continue
        k = _key(prompt, image_a, image_b)
        if k in seen or k in flagged_keys:
            continue

        agree = bool(r.get("shadow_agreement", r.get("agreement", True)))
        if agree:
            continue

        student = r.get("student", {}) or {}
        teacher = r.get("teacher", {}) or {}
        s_w = str(student.get("winner", "tie"))
        t_w = str(teacher.get("winner", "tie"))
        if t_w not in ("A", "B"):
            continue

        try:
            conf = _clamp01(float(student.get("confidence", 0.0)))
        except Exception:
            conf = 0.0

        seen.add(k)
        selected.append(
            {
                "prompt": prompt,
                "imageA": image_a,
                "imageB": image_b,
                "winner": t_w,  # prefer teacher label in shadow
                "confidence": conf,
                "method": "shadow",
                "student_winner": s_w,
                "teacher_winner": t_w,
                "agreement": False,
                "latency_ms": None,
                "feedback_correct_winner": fb_map.get(k),
                "flagged": False,
                "selection_reason": "shadow_disagreement",
                "timestamp": r.get("timestamp", None),
            }
        )

        if cfg.max_samples is not None and len(selected) >= int(cfg.max_samples):
            break

    return selected
