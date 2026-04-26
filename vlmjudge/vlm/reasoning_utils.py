"""
Reasoning Compression Utility.
"""

from __future__ import annotations

import re

def compress_reasoning(text: str) -> str:
    """
    Compresses reasoning text into a 1-2 sentence summary.
    Heuristic: Extracts the first sentence and the final conclusive sentence.
    """
    if not text or not isinstance(text, str):
        return ""
        
    text = text.strip()
    
    # Split into sentences using simple regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 2:
        return " ".join(sentences)
        
    first_sentence = sentences[0]
    
    # Look for conclusive sentences at the end
    conclusion_keywords = ["therefore", "in conclusion", "overall", "thus", "clearly", "better", "winner"]
    
    last_sentence = sentences[-1]
    for s in reversed(sentences[1:]):
        lower_s = s.lower()
        if any(kw in lower_s for kw in conclusion_keywords):
            last_sentence = s
            break
            
    if first_sentence == last_sentence:
        return first_sentence
        
    return f"{first_sentence} ... {last_sentence}"
