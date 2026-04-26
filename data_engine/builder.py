from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@dataclass(frozen=True)
class BuildConfig:
    # Convert confidence to quality tier for downstream QC.
    high_conf: float = 0.7
    medium_conf: float = 0.5


def _quality_from_conf(conf: float, cfg: BuildConfig) -> str:
    c = _clamp01(float(conf))
    if c >= float(cfg.high_conf):
        return "high"
    if c >= float(cfg.medium_conf):
        return "medium"
    return "low"


def build_preferences(
    selected: List[Mapping[str, Any]],
    *,
    config: Optional[BuildConfig] = None,
    source: str = "api",
) -> List[Dict[str, Any]]:
    """
    Convert selected API logs into Phase-5 style preference records.
    """
    cfg = config or BuildConfig()
    out: List[Dict[str, Any]] = []

    for s in selected:
        if bool(s.get("flagged", False)):
            continue
        prompt = str(s.get("prompt", "")).strip()
        a = str(s.get("imageA", "")).strip()
        b = str(s.get("imageB", "")).strip()
        if not prompt or not a or not b:
            continue

        winner = str(s.get("feedback_correct_winner") or s.get("winner") or "tie")
        if winner not in ("A", "B"):
            continue

        chosen = a if winner == "A" else b
        rejected = b if winner == "A" else a

        try:
            conf = _clamp01(float(s.get("confidence", 0.0)))
        except Exception:
            conf = 0.0

        quality = _quality_from_conf(conf, cfg)
        agreement = bool(s.get("agreement", True)) if s.get("agreement", None) is not None else True

        inferred_source = "feedback" if s.get("feedback_correct_winner", None) is not None else str(source)
        out.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "winner": winner,
                "confidence": conf,
                "coverage": 1.0,
                "quality": quality,
                "agreement": agreement,
                "delta": float(s.get("delta", 0.0) or 0.0),
                "source": str(inferred_source),
                "metadata": {
                    "selection_reason": s.get("selection_reason", None),
                    "method": s.get("method", None),
                    "student_winner": s.get("student_winner", None),
                    "teacher_winner": s.get("teacher_winner", None),
                    "timestamp": s.get("timestamp", None),
                },
            }
        )

    return out
