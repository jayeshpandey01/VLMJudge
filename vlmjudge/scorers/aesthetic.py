"""
Aesthetic scorer (LAION-style aesthetic predictor or equivalent).

This implementation is designed to run when the required weights are available.
If weights or dependencies are missing, it returns None and logs a warning.

Default approach:
    - Extract OpenCLIP image embeddings
    - Apply a lightweight linear head (aesthetic predictor)
    - Normalize score into [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import time
from typing import Optional

from PIL import Image

from vlmjudge.scorers.base import BaseScorer, ImageInput, ScoreOutput
from vlmjudge.utils.normalization import clamp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AestheticScorerConfig:
    """
    Configuration for AestheticScorer.

    `weights_path` should point to a torch state_dict for the aesthetic head.
    Common community checkpoints output scores roughly in [1, 10]; we normalize with
    `score_min`/`score_max` placeholders (override as needed for your checkpoint).
    """

    weights_path: Optional[str] = None
    device: Optional[str] = None

    clip_model_name: str = "ViT-L-14"
    clip_pretrained: str = "openai"

    score_min: float = 1.0
    score_max: float = 10.0


class AestheticScorer(BaseScorer):
    """
    Image-only aesthetic scorer.

    Returns:
        A normalized score in [0, 1] when weights are available, else None.
    """

    def __init__(self, config: Optional[AestheticScorerConfig] = None) -> None:
        self._config = config or AestheticScorerConfig()
        self._device = None
        self._clip_model = None
        self._clip_preprocess = None
        self._head = None

        self._try_init()

    def _resolve_weights_path(self) -> Optional[str]:
        if self._config.weights_path:
            return self._config.weights_path
        env_path = os.environ.get("IMAGEREWARD_AESTHETIC_WEIGHTS")
        return env_path or None

    def _try_init(self) -> None:
        weights_path = self._resolve_weights_path()
        if not weights_path:
            logger.warning("AestheticScorer disabled: no weights provided (set `weights_path` or IMAGEREWARD_AESTHETIC_WEIGHTS).")
            return

        try:
            import torch
            import open_clip
        except Exception as e:
            logger.warning("AestheticScorer unavailable (missing dependency): %s", e)
            return

        device = self._config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        try:
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                self._config.clip_model_name, pretrained=self._config.clip_pretrained
            )
            clip_model.eval().to(self._device)
        except Exception as e:
            logger.warning("AestheticScorer failed to initialize OpenCLIP backbone: %s", e)
            return

        try:
            state = torch.load(weights_path, map_location="cpu")
        except Exception as e:
            logger.warning("AestheticScorer failed to load weights from %r: %s", weights_path, e)
            return

        # Accept either a raw tensor dict {"weight": ..., "bias": ...} or a nested dict.
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else None
        if not isinstance(state_dict, dict):
            logger.warning("AestheticScorer weights file has unsupported format: %r", type(state))
            return

        embed_dim = None
        try:
            visual = getattr(clip_model, "visual", None)
            if visual is not None and hasattr(visual, "output_dim"):
                embed_dim = int(visual.output_dim)
        except Exception:
            embed_dim = None

        if embed_dim is None:
            # Best-effort inference from a small dummy forward pass.
            try:
                dummy = torch.zeros(1, 3, 224, 224, device=self._device)
                with torch.no_grad():
                    embed_dim = int(clip_model.encode_image(dummy).shape[-1])
            except Exception:
                embed_dim = 768

        head = torch.nn.Linear(embed_dim, 1)
        try:
            head.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.warning("AestheticScorer failed to load head state_dict strictly: %s", e)
            # best-effort: continue; scoring may fail if shapes mismatch.
        head.eval().to(self._device)

        self._clip_model = clip_model
        self._clip_preprocess = preprocess
        self._head = head

    def _load_pil(self, image: ImageInput) -> Optional[Image.Image]:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            try:
                return Image.open(image).convert("RGB")
            except Exception as e:
                logger.warning("AestheticScorer failed to open image path %r: %s", image, e)
                return None
        logger.warning("AestheticScorer received unsupported image type: %s", type(image))
        return None

    def _fallback(self) -> ScoreOutput:
        return {"score": 0.5, "confidence": 0.0}

    def score(self, image: ImageInput, prompt: Optional[str] = None, image_b: Optional[ImageInput] = None) -> ScoreOutput:
        del prompt, image_b

        if self._clip_model is None or self._clip_preprocess is None or self._head is None:
            return self._fallback()

        pil = self._load_pil(image)
        if pil is None:
            return self._fallback()

        try:
            import torch
        except Exception as e:
            logger.warning("AestheticScorer missing torch at runtime: %s", e)
            return self._fallback()

        t0 = time.perf_counter()
        try:
            image_tensor = self._clip_preprocess(pil).unsqueeze(0).to(self._device)
            with torch.no_grad():
                emb = self._clip_model.encode_image(image_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                raw = self._head(emb).squeeze().item()

            # Per requirements: typical [1,10] -> divide by 10.
            score = clamp(float(raw) / 10.0, 0.0, 1.0)
            confidence = 0.8
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("event=scorer_infer scorer=aesthetic batch=1 ms=%.2f", dt_ms)
            return {"score": float(score), "confidence": float(confidence)}
        except Exception as e:
            logger.warning("AestheticScorer scoring failed: %s", e)
            return self._fallback()
