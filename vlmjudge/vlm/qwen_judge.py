# Name: Jayesh Pandey
# Summary: VLM Judge module using Qwen2.5-VL for reasoning-based image comparison.

"""
VLM Judge module using Qwen2.5-VL for reasoning-based image comparison.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    HAVE_QWEN = True
except ImportError:
    HAVE_QWEN = False

logger = logging.getLogger(__name__)

_FALLBACK = {"winner": "tie", "confidence": 0.0, "reason": "VLM judge unavailable."}


@dataclass(frozen=True)
class QwenJudgeConfig:
    runs: int = 3
    max_new_tokens: int = 192
    confidence_scale: float = 0.8
    confidence_cap: float = 0.85


class QwenJudge:
    """
    Evaluates two images given a prompt using Qwen2.5-VL and returns a structured JSON
    decision with winner, confidence, and reasoning.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str | None = None,
        *,
        config: Optional[QwenJudgeConfig] = None,
        strict: bool = False,
    ) -> None:
        """
        Args:
            strict: If True, raises on missing deps/model load failure. If False,
                disables the judge and allows the structured pipeline to proceed.
        """
        self.config = config or QwenJudgeConfig()
        self.enabled = True

        if not HAVE_QWEN:
            msg = (
                "Missing requirements for QwenJudge. Please install with: "
                "pip install transformers>=4.45.0 qwen-vl-utils"
            )
            if strict:
                raise ImportError(msg)
            logger.warning(msg)
            self.enabled = False
            self.model = None
            self.processor = None
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            return

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading Qwen2.5-VL model: {model_name} on {self.device}")
        
        # Load model with automatic device mapping if cuda is available
        device_map = "auto" if self.device.startswith("cuda") else None
        
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map=device_map,
            )
            if device_map is None:
                self.model = self.model.to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
        except Exception as e:
            if strict:
                raise
            logger.warning("Failed to load Qwen2.5-VL model/processor: %s", e)
            self.enabled = False
            self.model = None
            self.processor = None

    def _ensure_pil(self, img: Union[str, Image.Image]) -> Image.Image:
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        return img.convert("RGB")

    def compare(self, imageA: Union[str, Image.Image], imageB: Union[str, Image.Image], prompt: str) -> Dict[str, Any]:
        """
        Compare two images using a structured prompt.
        
        Returns:
            Dict containing: winner ("A", "B", "tie"), confidence (float 0-1), reason (str)
        """
        if not self.enabled or self.model is None or self.processor is None:
            return dict(_FALLBACK)

        imageA_pil = self._ensure_pil(imageA)
        imageB_pil = self._ensure_pil(imageB)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an expert AI image quality evaluator.\n"
                            "Be conservative in your judgment.\n"
                            "Only mention clearly visible differences.\n"
                            "If unsure, return \"tie\" with low confidence."
                        ),
                    },
                    {"type": "text", "text": f"Given:\n- Prompt: {prompt}"},
                    {"type": "image", "image": imageA_pil, "label": "Image A"},
                    {"type": "image", "image": imageB_pil, "label": "Image B"},
                    {
                        "type": "text", 
                        "text": (
                            "Task:\n"
                            "1. Compare both images based on the prompt\n"
                            "2. Decide which image better matches the prompt\n"
                            "3. Be precise and objective\n\n"
                            "4. If differences are small or ambiguous, choose \"tie\"\n\n"
                            "Output format (STRICT JSON):\n"
                            "{\n"
                            "  \"winner\": \"A\" or \"B\" or \"tie\",\n"
                            "  \"confidence\": float (0 to 1),\n"
                            "  \"reason\": \"short explanation\"\n"
                            "}"
                        )
                    }
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(self.config.max_new_tokens),
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        out = self._parse_output(output_text)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "event=vlm_infer model=qwen2.5-vl batch=1 ms=%.2f winner=%s conf=%.3f",
            dt_ms,
            out["winner"],
            out["confidence"],
        )
        return out

    def compare_multi(
        self,
        imageA: Union[str, Image.Image],
        imageB: Union[str, Image.Image],
        prompt: str,
        *,
        runs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Multi-run voting to improve stability.

        Returns:
            {
              "winner": "A"|"B"|"tie",
              "confidence": float[0,1],
              "reason": str,
              "votes": [...],
              "confidence_raw": float,
              "confidence_calibrated": float
            }
        """
        if not self.enabled:
            return {
                "winner": "tie",
                "confidence": 0.0,
                "reason": "VLM judge unavailable.",
                "votes": [],
                "confidences": [],
                "reasons": [],
                "confidence_raw": 0.0,
                "confidence_calibrated": 0.0,
                "count_majority": 0,
                "runs": 0,
            }

        n = int(runs or self.config.runs)
        n = max(1, min(9, n))

        votes: List[str] = []
        confs: List[float] = []
        reasons: List[str] = []

        t0 = time.perf_counter()
        for _ in range(n):
            r = self.compare(imageA, imageB, prompt)
            v = str(r.get("winner", "tie"))
            if v not in ("A", "B", "tie"):
                v = "tie"
            c = float(r.get("confidence", 0.0))
            c = max(0.0, min(1.0, c))
            reason = str(r.get("reason", "")).strip()
            votes.append(v)
            confs.append(c)
            reasons.append(reason)

        counts = {"A": 0, "B": 0, "tie": 0}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        max_count = max(counts.values()) if counts else 0
        winners = [k for k, c in counts.items() if c == max_count]
        final_winner = winners[0] if len(winners) == 1 else "tie"

        confidence_raw = (sum(confs) / len(confs) if confs else 0.0) * (max_count / n if n else 0.0)
        confidence_raw = max(0.0, min(1.0, confidence_raw))

        confidence_cal = confidence_raw * float(self.config.confidence_scale)
        confidence_cal = min(confidence_cal, float(self.config.confidence_cap))
        confidence_cal = max(0.0, min(1.0, confidence_cal))

        maj_reasons = []
        for v, c, reason in zip(votes, confs, reasons):
            if v == final_winner and reason:
                maj_reasons.append((c, reason))
        maj_reasons.sort(key=lambda x: x[0], reverse=True)
        selected = [r for _, r in maj_reasons[:2]]
        if not selected:
            reason = "No clear visible differences; returning tie." if final_winner == "tie" else "Decision based on overall match to the prompt."
        elif len(selected) == 1:
            reason = selected[0]
        else:
            reason = (selected[0] + " / " + selected[1]).strip()
        reason = reason[:320]

        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "event=vlm_vote model=qwen2.5-vl runs=%d winner=%s maj=%d conf_raw=%.3f conf_cal=%.3f ms=%.2f",
            n,
            final_winner,
            max_count,
            confidence_raw,
            confidence_cal,
            dt_ms,
        )

        return {
            "winner": final_winner,
            "confidence": confidence_cal,
            "reason": reason,
            "votes": votes,
            "confidences": confs,
            "reasons": reasons,
            "confidence_raw": confidence_raw,
            "confidence_calibrated": confidence_cal,
            "count_majority": max_count,
            "runs": n,
        }

    def _parse_output(self, output_text: str) -> Dict[str, Any]:
        """Robustly parses the JSON output from the model."""
        try:
            # Try to extract JSON if there's surrounding text
            start = output_text.find('{')
            end = output_text.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = output_text[start:end]
                result = json.loads(json_str)
            else:
                result = json.loads(output_text)
                
            # Validate fields
            winner = str(result.get("winner", "tie")).upper()
            if winner not in ["A", "B", "TIE"]:
                winner = "tie"
                
            # Normalize TIE to tie
            if winner == "TIE":
                winner = "tie"
                
            confidence = float(result.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence)) # Clamp to [0,1]
            reason = str(result.get("reason", "No reason provided."))
            
            return {
                "winner": winner,
                "confidence": confidence,
                "reason": reason
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse VLM output: {output_text}. Error: {e}")
            return {
                "winner": "tie",
                "confidence": 0.0,
                "reason": f"Parsing error: {str(e)}"
            }
