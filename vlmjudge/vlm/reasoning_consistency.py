# Name: Jayesh Pandey
# Summary: Reasoning Consistency Check.

"""
Reasoning Consistency Check.
"""

from __future__ import annotations

def check_consistency(reasoning: str, predicted_winner: str) -> bool:
    """
    Checks if the reasoning text supports the predicted winner.
    Returns True if consistent, False if a contradiction is detected.
    """
    if not reasoning or not isinstance(reasoning, str):
        return True # Default to true if no reasoning
        
    reasoning = reasoning.lower()
    predicted_winner = predicted_winner.upper()
    
    if predicted_winner == "TIE":
        return True

    opposite_winner = "B" if predicted_winner == "A" else "A"
    
    opposite_phrases = [
        f"image {opposite_winner.lower()} is better",
        f"image {opposite_winner.lower()} is clearly better",
        f"winner is image {opposite_winner.lower()}",
        f"winner: image {opposite_winner.lower()}",
        f"image {opposite_winner.lower()} matches the prompt better",
        f"image {opposite_winner.lower()} is more accurate",
        f"prefer image {opposite_winner.lower()}"
    ]
    
    for phrase in opposite_phrases:
        if phrase in reasoning:
            return False
            
    negative_predicted_phrases = [
        f"image {predicted_winner.lower()} is worse",
        f"image {predicted_winner.lower()} fails",
        f"image {predicted_winner.lower()} is completely wrong",
        f"image {predicted_winner.lower()} does not match"
    ]
    
    for phrase in negative_predicted_phrases:
        if phrase in reasoning:
            if "although " + phrase in reasoning or "while " + phrase in reasoning:
                continue
            return False

    return True
