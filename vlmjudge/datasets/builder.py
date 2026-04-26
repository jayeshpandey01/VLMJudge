"""
Preference dataset builder.

Converts a pairwise comparison result into a training-ready preference record.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from vlmjudge.datasets.quality import QualityEvaluator


def build_preference(prompt: str, imgA: Any, imgB: Any, result: Mapping[str, Any]) -> Dict[str, Any]:
    winner = result.get("winner", "tie")
    confidence = float(result.get("confidence", 0.0))

    if winner == "A":
        chosen, rejected = imgA, imgB
    elif winner == "B":
        chosen, rejected = imgB, imgA
    else:
        chosen, rejected = None, None

    # Quality metadata (computed from the full compare output when available).
    evaluator = QualityEvaluator()
    quality_info = evaluator.evaluate(result)

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "winner": winner,
        "quality": quality_info["quality"],
        "confidence": float(quality_info["confidence"]),
        "disagreement": float(quality_info["disagreement"]),
        "delta": float(quality_info["delta"]),
        "coverage": float(quality_info.get("coverage", 0.0)),
        "metadata": {
            "scoresA": result.get("scoresA", {}),
            "scoresB": result.get("scoresB", {}),
            "aggregateA": result.get("aggregateA", {}),
            "aggregateB": result.get("aggregateB", {}),
            "delta": result.get("delta", 0.0),
            "threshold": result.get("threshold", 0.0),
        },
    }
