"""
LPIPS scorer.

LPIPS is a perceptual distance metric between two images. We expose it through the
`BaseScorer` interface by using:
    - `image` as image A
    - `image_b` as image B

If `image_b` is missing, we return None.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Optional

from PIL import Image

from vlmjudge.scorers.base import BaseScorer, ImageInput, ScoreOutput
from vlmjudge.utils.normalization import clamp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LPIPSScorerConfig:
    """
    Configuration for LPIPSScorer.

    `net` options are defined by the `lpips` library (commonly "alex", "vgg", "squeeze").
    """

    net: str = "alex"
    device: Optional[str] = None


class LPIPSScorer(BaseScorer):
    """
    Pairwise perceptual similarity scorer using LPIPS.

    Returns:
        A similarity in [0, 1] computed from distance `d` as `1 / (1 + d)`.
    """

    def __init__(self, config: Optional[LPIPSScorerConfig] = None) -> None:
        self._config = config or LPIPSScorerConfig()
        self._device = None
        self._lpips = None
        self._try_init()

    def _try_init(self) -> None:
        try:
            import torch
            import lpips  # type: ignore
        except Exception as e:
            logger.warning("LPIPSScorer unavailable (missing dependency): %s", e)
            return

        device = self._config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        try:
            model = lpips.LPIPS(net=self._config.net)
            model.eval().to(self._device)
        except Exception as e:
            logger.warning("LPIPSScorer failed to initialize: %s", e)
            return

        self._lpips = model

    def _load_pil(self, image: ImageInput) -> Optional[Image.Image]:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            try:
                return Image.open(image).convert("RGB")
            except Exception as e:
                logger.warning("LPIPSScorer failed to open image path %r: %s", image, e)
                return None
        logger.warning("LPIPSScorer received unsupported image type: %s", type(image))
        return None

    def _pil_to_tensor(self, pil: Image.Image):
        # LPIPS expects tensors in [-1, 1], shape [B,3,H,W]
        import torch
        try:
            import numpy as np
        except Exception as e:
            logger.warning("LPIPSScorer requires numpy to convert PIL images: %s", e)
            return None

        arr = np.asarray(pil).astype("float32") / 255.0  # [H,W,3] in [0,1]
        arr = (arr * 2.0) - 1.0  # [-1,1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        return tensor

    def _fallback(self) -> ScoreOutput:
        return {"score": 0.5, "confidence": 0.0}

    def score(self, image: ImageInput, prompt: Optional[str] = None, image_b: Optional[ImageInput] = None) -> ScoreOutput:
        del prompt

        if self._lpips is None or self._device is None:
            return self._fallback()
        if image_b is None:
            return self._fallback()

        pil_a = self._load_pil(image)
        pil_b = self._load_pil(image_b)
        if pil_a is None or pil_b is None:
            return self._fallback()

        try:
            import torch
        except Exception as e:
            logger.warning("LPIPSScorer missing torch at runtime: %s", e)
            return self._fallback()

        t0 = time.perf_counter()
        try:
            ta = self._pil_to_tensor(pil_a)
            tb = self._pil_to_tensor(pil_b)
            if ta is None or tb is None:
                return self._fallback()
            ta = ta.to(self._device)
            tb = tb.to(self._device)
            with torch.no_grad():
                dist = self._lpips(ta, tb).squeeze().item()

            # Per requirements: lower is better -> invert.
            dist_f = float(dist)
            score = clamp(1.0 - dist_f, 0.0, 1.0)
            confidence = clamp(1.0 - dist_f, 0.0, 1.0)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("event=scorer_infer scorer=lpips batch=1 ms=%.2f", dt_ms)
            return {"score": float(score), "confidence": float(confidence)}
        except Exception as e:
            logger.warning("LPIPSScorer scoring failed: %s", e)
            return self._fallback()
