"""
author: Jayesh Pandey
summary: Orchestrates the image comparison pipeline, integrating multiple scorers and VLM-based fusion judging with confidence calibration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from vlmjudge.comparators.pairwise import PairwiseComparator, PairwiseComparatorConfig
from vlmjudge.comparators.explanation import generate_explanation, ExplanationConfig
from vlmjudge.datasets.builder import build_preference
from vlmjudge.scorers.base import BaseScorer, ImageInput
from vlmjudge.utils.normalization import clamp
from vlmjudge.scorers.reasoning_score import ReasoningScorer

logger = logging.getLogger(__name__)

def calibrate_confidence(base_score: float, disagreement: float, reasoning_score: float) -> float:
    """
    Calibrates confidence combining base score, disagreement penalty, and reasoning quality.
    """
    calibrated = base_score * (1.0 - disagreement) * (0.5 + 0.5 * reasoning_score)
    return float(clamp(calibrated, 0.0, 1.0))


@dataclass(frozen=True)
class ComparePipelineConfig:
    threshold: float = 0.05
    vlm_runs: int = 3


class ComparePipeline:
    def __init__(
        self,
        scorers: Mapping[str, BaseScorer],
        *,
        weights: Optional[Mapping[str, float]] = None,
        config: Optional[ComparePipelineConfig] = None,
        vlm_judge: Optional[Any] = None,
    ) -> None:
        self._scorers = dict(scorers)
        self._weights = dict(weights or {})
        self._config = config or ComparePipelineConfig()
        self._reasoning_scorer = ReasoningScorer()

        self._comparator = PairwiseComparator(
            self._scorers,
            weights=self._weights,
            config=PairwiseComparatorConfig(threshold=self._config.threshold),
        )
        self._vlm_judge = vlm_judge

    def run(self, imgA: ImageInput, imgB: ImageInput, prompt: str) -> Dict[str, Any]:
        t0 = time.perf_counter()

        structured_result = self._comparator.compare(imgA, imgB, prompt)
        
        # Dynamic Weighting
        prompt_length = len(prompt.split())
        needs_reweight = False
        dynamic_weights = dict(self._weights)
        
        if prompt_length > 15:
            dynamic_weights["image_reward"] = dynamic_weights.get("image_reward", 1.0) * 1.5
            needs_reweight = True
            
        try:
            lpips_score = structured_result["scoresA"].get("lpips", {}).get("score", 0.0)
            if lpips_score > 0.8:
                dynamic_weights["lpips"] = dynamic_weights.get("lpips", 1.0) * 2.0
                needs_reweight = True
        except KeyError:
            pass
            
        if needs_reweight:
            from vlmjudge.utils.aggregation import aggregate_scores
            aggA = aggregate_scores(structured_result["scoresA"], dynamic_weights)
            aggB = aggregate_scores(structured_result["scoresB"], dynamic_weights)
            finalA = float(aggA["score"])
            finalB = float(aggB["score"])
            delta = finalA - finalB
            thr = float(self._config.threshold)
            if delta > thr:
                winner = "A"
            elif delta < -thr:
                winner = "B"
            else:
                winner = "tie"
            confidence = abs(delta) * (float(aggA["confidence"]) + float(aggB["confidence"])) * 0.5
            confidence = float(clamp(confidence, 0.0, 1.0))
            
            structured_result["winner"] = winner
            structured_result["confidence"] = confidence
            structured_result["delta"] = float(delta)
            structured_result["aggregateA"] = aggA
            structured_result["aggregateB"] = aggB

        structured_explanation = generate_explanation(
            prompt=prompt,
            winner=structured_result["winner"],
            scoresA=structured_result["scoresA"],
            scoresB=structured_result["scoresB"],
            weights=dynamic_weights if needs_reweight else self._weights,
            config=ExplanationConfig(top_k=3),
        )
        dataset_entry = build_preference(prompt, imgA, imgB, structured_result)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        
        structured_winner = structured_result["winner"]
        structured_conf = float(structured_result["confidence"])
        structured_conf = float(clamp(structured_conf, 0.0, 1.0))

        final_winner = structured_winner
        final_confidence = structured_conf
        vlm_result = None
        agreement: bool = False
        disagreement_score: float = 0.0
        explanation = structured_explanation
        
        if self._vlm_judge is not None:
            # Phase 4.5: multi-run voting + calibrated confidence
            is_low_conf = structured_conf < 0.2
            vlm_runs = int(self._config.vlm_runs)
            if is_low_conf:
                vlm_runs = max(vlm_runs, 3)
            try:
                vlm_result = self._vlm_judge.compare_multi(
                    imgA, imgB, prompt, runs=vlm_runs
                )
            except Exception as e:
                logger.warning("event=vlm_error err=%s", e)
                vlm_result = {
                    "winner": "tie",
                    "confidence": 0.0,
                    "reason": "VLM judge error.",
                    "votes": [],
                    "confidences": [],
                    "reasons": [],
                    "confidence_raw": 0.0,
                    "confidence_calibrated": 0.0,
                    "count_majority": 0,
                    "runs": vlm_runs,
                }

            vlm_winner = str(vlm_result.get("winner", "tie"))
            vlm_conf = float(vlm_result.get("confidence", 0.0))
            vlm_conf = float(clamp(vlm_conf, 0.0, 1.0))

            vlm_votes = vlm_result.get("votes", [])
            vlm_actual_runs = int(vlm_result.get("runs", 0) or 0)
            vlm_has_signal = bool(vlm_actual_runs > 0 and isinstance(vlm_votes, list) and len(vlm_votes) > 0)

            if vlm_has_signal:
                agreement = bool(vlm_winner == structured_winner)
                disagreement_score = float(clamp(abs(vlm_conf - structured_conf), 0.0, 1.0))

                is_hard_case = structured_conf < 0.2 or disagreement_score > 0.5
                
                vlm_weight = 0.5
                if prompt_length > 15:
                    vlm_weight = 0.7

                # Hard Case Specialist Mode and Fusion rules
                if is_hard_case:
                    final_winner = vlm_winner
                    final_confidence = vlm_conf
                elif vlm_winner == structured_winner:
                    final_winner = vlm_winner
                    final_confidence = (vlm_conf + structured_conf) / 2.0
                else:
                    final_winner = structured_winner
                    final_confidence = min(vlm_conf, structured_conf) * 0.7

                # Final explanation: prefer VLM reason when available, otherwise fall back.
                vlm_reason = str(vlm_result.get("reason", "")).strip()
                reasoning_score = self._reasoning_scorer.score_reasoning(vlm_reason) if vlm_reason else 0.5
                
                final_confidence = calibrate_confidence(final_confidence, disagreement_score, reasoning_score)

                if vlm_reason:
                    if agreement or is_hard_case:
                        explanation = vlm_reason
                    else:
                        explanation = (structured_explanation + " | VLM (disagreed): " + vlm_reason)[:512]
            else:
                # VLM unavailable/disabled: keep structured result and mark as no agreement signal.
                agreement = False
                disagreement_score = 0.0
                final_winner = structured_winner
                final_confidence = structured_conf
                explanation = structured_explanation
                vlm_reason = str(vlm_result.get("reason", "")).strip()

            # Dataset enrichment
            dataset_entry["vlm"] = {
                "winner": vlm_winner,
                "confidence_raw": float(vlm_result.get("confidence_raw", vlm_conf)),
                "confidence_calibrated": vlm_conf,
                "votes": vlm_votes,
                "reason": vlm_reason,
                "reasoning_short": vlm_result.get("reasoning_short", ""),
                "reasoning_inconsistent": vlm_result.get("reasoning_inconsistent", False),
                "disagreement_score": vlm_result.get("disagreement_score", 0.0),
                "runs": vlm_actual_runs if vlm_actual_runs else vlm_runs,
            }
            dataset_entry["agreement"] = agreement
            dataset_entry["disagreement_score"] = disagreement_score
            dataset_entry["final_winner"] = final_winner
            dataset_entry["final_confidence"] = final_confidence

        logger.info(
            "event=pipeline_compare winner=%s delta=%.4f conf=%.4f ms=%.2f",
            final_winner,
            float(structured_result.get("delta", 0.0)),
            float(final_confidence),
            dt_ms,
        )

        out: Dict[str, Any] = {
            "winner": final_winner,
            "confidence": final_confidence,
            "vlm": None,
            "structured": {
                "winner": structured_winner,
                "confidence": structured_conf,
                "delta": float(structured_result.get("delta", 0.0)),
                "threshold": float(structured_result.get("threshold", self._config.threshold)),
                "scores": {"A": structured_result["scoresA"], "B": structured_result["scoresB"]},
                "aggregate": {"A": structured_result["aggregateA"], "B": structured_result["aggregateB"]},
            },
            "agreement": agreement,
            "disagreement_score": disagreement_score,
            "explanation": explanation,
            "dataset_entry": dataset_entry,
            "timing_ms": dt_ms,
        }
        
        if vlm_result is not None:
            out["vlm"] = {
                "votes": vlm_result.get("votes", []),
                "confidence_raw": float(vlm_result.get("confidence_raw", 0.0)),
                "confidence_calibrated": float(vlm_result.get("confidence_calibrated", vlm_result.get("confidence", 0.0))),
                "winner": vlm_result.get("winner", "tie"),
                "reason": vlm_result.get("reason", ""),
                "runs": int(vlm_result.get("runs", self._config.vlm_runs)),
                "count_majority": int(vlm_result.get("count_majority", 0)),
            }
            
        return out
