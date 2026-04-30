# Name: Jayesh Pandey
# Summary: Optional DINOv2 scorer.

"""
Optional DINOv2 scorer.

This is designed to be best-effort:
    - If DINOv2 cannot be loaded (e.g., no weights / no internet for torch.hub), returns None.
    - Provides a simple "semantic richness" proxy based on embedding norm.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Optional

from PIL import Image

from vlmjudge.scorers.base import BaseScorer, ImageInput, ScoreOutput
from vlmjudge.utils.normalization import clamp, min_max_normalize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DINOv2ScorerConfig:
    device: Optional[str] = None
    hub_repo: str = "facebookresearch/dinov2"
    hub_model: str = "dinov2_vits14"

    # Placeholder normalization range for embedding norm; override for your setup.
    norm_min: float = 5.0
    norm_max: float = 25.0


class DINOv2Scorer(BaseScorer):
    """
    Image-only semantic richness scorer based on DINOv2 embeddings.

    Returns:
        A normalized [0,1] score derived from embedding L2 norm, or None if unavailable.
    """

    def __init__(self, config: Optional[DINOv2ScorerConfig] = None) -> None:
        self._config = config or DINOv2ScorerConfig()
        self._device = None
        self._model = None
        self._try_init()

    def _try_init(self) -> None:
        try:
            import torch
            import torchvision.transforms as T
        except Exception as e:
            logger.warning("DINOv2Scorer unavailable (missing dependency): %s", e)
            return

        device = self._config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        try:
            model = torch.hub.load(self._config.hub_repo, self._config.hub_model)  # may require internet
            model.eval().to(self._device)
        except Exception as e:
            logger.warning("DINOv2Scorer failed to load via torch.hub (%s/%s): %s", self._config.hub_repo, self._config.hub_model, e)
            return

        self._model = model
        self._transform = T.Compose(
            [
                T.Resize(518),
                T.CenterCrop(518),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def _load_pil(self, image: ImageInput) -> Optional[Image.Image]:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            try:
                return Image.open(image).convert("RGB")
            except Exception as e:
                logger.warning("DINOv2Scorer failed to open image path %r: %s", image, e)
                return None
        logger.warning("DINOv2Scorer received unsupported image type: %s", type(image))
        return None

    def _fallback(self) -> ScoreOutput:
        return {"score": 0.5, "confidence": 0.0}

    def score(self, image: ImageInput, prompt: Optional[str] = None, image_b: Optional[ImageInput] = None) -> ScoreOutput:
        del prompt, image_b
        if self._model is None or self._device is None:
            return self._fallback()

        pil = self._load_pil(image)
        if pil is None:
            return self._fallback()

        try:
            import torch
        except Exception as e:
            logger.warning("DINOv2Scorer missing torch at runtime: %s", e)
            return self._fallback()

        t0 = time.perf_counter()
        try:
            x = self._transform(pil).unsqueeze(0).to(self._device)
            with torch.no_grad():
                emb = self._model(x)
                if hasattr(emb, "flatten"):
                    emb_vec = emb.flatten()
                else:
                    emb_vec = torch.as_tensor(emb).flatten()
                norm = float(torch.norm(emb_vec, p=2).item())

            # Per requirements: normalize to [0,1]. Here we treat embedding norm as a proxy signal.
            score = min_max_normalize(norm, self._config.norm_min, self._config.norm_max, clamp_to_unit=True)
            if score is None:
                score = 0.5
            score_f = float(clamp(float(score), 0.0, 1.0))
            confidence = score_f
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("event=scorer_infer scorer=dinov2 batch=1 ms=%.2f", dt_ms)
            return {"score": score_f, "confidence": float(clamp(confidence, 0.0, 1.0))}
        except Exception as e:
            logger.warning("DINOv2Scorer scoring failed: %s", e)
            return self._fallback()

    def score_batch(self, images, prompts=None, image_bs=None):
        del prompts, image_bs
        if self._model is None or self._device is None:
            return [self._fallback() for _ in images]

        try:
            import torch
        except Exception as e:
            logger.warning("DINOv2Scorer missing torch at runtime: %s", e)
            return [self._fallback() for _ in images]

        pils: list[Image.Image] = []
        valid_mask: list[bool] = []
        for img in images:
            pil = self._load_pil(img)
            if pil is None:
                valid_mask.append(False)
                pils.append(Image.new("RGB", (518, 518), color=(0, 0, 0)))
            else:
                valid_mask.append(True)
                pils.append(pil)

        t0 = time.perf_counter()
        try:
            batch = torch.stack([self._transform(p) for p in pils], dim=0).to(self._device)
            with torch.no_grad():
                emb = self._model(batch)
                emb = emb.view(emb.shape[0], -1)
                norms = torch.norm(emb, p=2, dim=1)

            outs: list[ScoreOutput] = []
            for i in range(len(images)):
                if not valid_mask[i]:
                    outs.append(self._fallback())
                    continue
                n = float(norms[i].item())
                s = min_max_normalize(n, self._config.norm_min, self._config.norm_max, clamp_to_unit=True)
                if s is None:
                    s = 0.5
                s_f = float(clamp(float(s), 0.0, 1.0))
                outs.append({"score": s_f, "confidence": s_f})

            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("event=scorer_infer scorer=dinov2 batch=%d ms=%.2f", len(images), dt_ms)
            return outs
        except Exception as e:
            logger.warning("DINOv2Scorer batch scoring failed: %s", e)
            return [self._fallback() for _ in images]
