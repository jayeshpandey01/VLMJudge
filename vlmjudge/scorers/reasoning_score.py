"""
Reasoning Scorer.

Evaluates the quality/strength of VLM explanations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ReasoningScorerConfig:
    min_length: int = 10
    max_length: int = 250

class ReasoningScorer:
    """
    Assigns a score to an explanation based on heuristic strength markers.
    """
    def __init__(self, config: Optional[ReasoningScorerConfig] = None):
        self.config = config or ReasoningScorerConfig()
        # Keywords indicating comparative reasoning
        self.strong_keywords = [
            "because", "however", "whereas", "although", "compare",
            "more", "less", "better", "worse", "clearly", "specifically",
            "matches", "fails", "missing", "present", "details"
        ]
        # Keywords indicating uncertainty
        self.weak_keywords = [
            "maybe", "perhaps", "unsure", "not clear", "hard to tell",
            "similar", "ambiguous", "might", "could be", "guess", "tie"
        ]

    def score_reasoning(self, reasoning: str) -> float:
        """
        Returns a score in [0, 1] estimating the quality of the reasoning.
        """
        if not reasoning or not isinstance(reasoning, str):
            return 0.0

        text = reasoning.lower()
        words = text.split()
        length = len(words)

        if length < self.config.min_length:
            return 0.2  # Too short to be strong reasoning

        score = 0.5 # Base score

        # Length bonus
        if length > 30:
            score += 0.1
        if length > 50:
            score += 0.1
            
        # Keyword analysis
        strong_count = sum(1 for kw in self.strong_keywords if kw in text)
        weak_count = sum(1 for kw in self.weak_keywords if kw in text)

        score += 0.05 * strong_count
        score -= 0.1 * weak_count

        # Formatting bonus (e.g. mentions 'Image A' or 'Image B')
        if "image a" in text or "image b" in text:
            score += 0.1

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
