"""
author: Jayesh Pandey
summary: Implements a multi-VLM ensemble judge that aggregates decisions from multiple vision-language models using confidence-weighted voting and reasoning fusion.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Union
from PIL import Image

from vlmjudge.vlm.qwen_judge import QwenJudge, QwenJudgeConfig
from vlmjudge.vlm.cache import VLMCache
from vlmjudge.vlm.reasoning_utils import compress_reasoning
from vlmjudge.vlm.reasoning_consistency import check_consistency

logger = logging.getLogger(__name__)

class VLMEnsemble:
    def __init__(self, config: Dict[str, Any] = None, strict: bool = False):
        self.config = config or {}
        self.models = {}
        self.enabled = False
        self.cache = VLMCache()
        
        # Initialize primary judge
        self._init_qwen(strict=strict)
        
        # Can be extended to load other models based on config (LLaVA, CogVLM, etc.)
        
        if self.models:
            self.enabled = True

    def _init_qwen(self, strict: bool):
        device = self.config.get("device")
        runs = self.config.get("vlm_runs", 3)
        max_tokens = self.config.get("vlm_max_new_tokens", 192)
        
        qwen = QwenJudge(
            device=device,
            config=QwenJudgeConfig(runs=int(runs), max_new_tokens=int(max_tokens)),
            strict=strict
        )
        if qwen.enabled:
            self.models["qwen2.5-vl"] = qwen

    def compare_multi(
        self,
        imageA: Union[str, Image.Image],
        imageB: Union[str, Image.Image],
        prompt: str,
        *,
        runs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs multi-vote on available models and aggregates the results.
        """
        if not self.enabled:
            return self._fallback()

        runs = runs if runs is not None else int(self.config.get("vlm_runs", 3))

        cached_res = self.cache.get(prompt, imageA, imageB, runs)
        if cached_res:
            return cached_res

        all_decisions = []
        for name, model in self.models.items():
            try:
                res = model.compare_multi(imageA, imageB, prompt, runs=runs)
                res["model_name"] = name
                all_decisions.append(res)
            except Exception as e:
                logger.warning(f"VLM model {name} failed: {e}")

        if not all_decisions:
            return self._fallback()

        aggregated = self.aggregate_vlm_decisions(all_decisions)
        
        # Calculate reasoning_short
        aggregated["reasoning_full"] = aggregated.get("reason", "")
        aggregated["reasoning_short"] = compress_reasoning(aggregated["reasoning_full"])
        
        self.cache.set(prompt, imageA, imageB, runs, aggregated)
        
        return aggregated

    def aggregate_vlm_decisions(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combines outputs from multiple VLMs using confidence-weighted averaging
        and reasoning fusion.
        """
        if len(decisions) == 1:
            return decisions[0]

        counts = {"A": 0.0, "B": 0.0, "tie": 0.0}
        total_weight = 0.0
        
        all_reasons = []
        all_votes = []
        all_confs = []
        all_individual_reasons = []
        
        for d in decisions:
            winner = d.get("winner", "tie")
            conf = d.get("confidence_calibrated", d.get("confidence", 0.0))
            reason = d.get("reason", "")
            name = d.get("model_name", "unknown")
            
            # Weight is based on confidence, minimum weight of 0.1 to give everyone a voice
            weight = max(0.1, conf)
            counts[winner] += weight
            total_weight += weight
            
            if reason:
                all_reasons.append(f"[{name}] {reason}")
                
            all_votes.extend(d.get("votes", []))
            all_confs.extend(d.get("confidences", []))
            all_individual_reasons.extend(d.get("reasons", []))

        # Majority voting (weighted)
        final_winner = max(counts.items(), key=lambda x: x[1])[0]
        
        # Disagreement score calculation
        total_votes = len(all_votes)
        majority_count = len([v for v in all_votes if v == final_winner])
        disagreement_score = 1.0 - (majority_count / total_votes) if total_votes > 0 else 0.0
        
        # Confidence is the normalized weight of the winning class
        winner_weight = counts[final_winner]
        if total_weight > 0:
            final_confidence = winner_weight / total_weight
        else:
            final_confidence = 0.0
            
        # Cross-VLM Disagreement Calibration
        final_confidence *= (1.0 - disagreement_score)
            
        # Reason fusion
        reasoning_summary = " | ".join(all_reasons)
        if len(reasoning_summary) > 512:
            reasoning_summary = reasoning_summary[:509] + "..."
            
        # Reasoning Consistency Check
        is_consistent = check_consistency(reasoning_summary, final_winner)
        if not is_consistent:
            final_confidence *= 0.7

        return {
            "winner": final_winner,
            "confidence": final_confidence,
            "reason": reasoning_summary,
            "votes": all_votes,
            "confidences": all_confs,
            "reasons": all_individual_reasons,
            "confidence_raw": final_confidence,
            "confidence_calibrated": final_confidence,
            "count_majority": majority_count,
            "disagreement_score": disagreement_score,
            "reasoning_inconsistent": not is_consistent,
            "runs": sum(d.get("runs", 0) for d in decisions),
            "ensemble_decisions": decisions
        }

    def _fallback(self):
        return {
            "winner": "tie",
            "confidence": 0.0,
            "reason": "VLM ensemble unavailable.",
            "votes": [],
            "confidences": [],
            "reasons": [],
            "confidence_raw": 0.0,
            "confidence_calibrated": 0.0,
            "count_majority": 0,
            "runs": 0,
        }
