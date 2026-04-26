"""
ImageReward scorer wrapper.

This wraps the existing `ImageReward` package into the new `BaseScorer` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Optional, Union

from PIL import Image

from vlmjudge.scorers.base import BaseScorer, ImageInput, ScoreOutput
from vlmjudge.utils.normalization import sigmoid

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageRewardScorerConfig:
    """Configuration for ImageRewardScorer."""

    model_name: str = "ImageReward-v1.0"
    device: Optional[Union[str, Any]] = None
    download_root: Optional[str] = None
    med_config: Optional[str] = None


class ImageRewardScorer(BaseScorer):
    """
    A `BaseScorer` implementation backed by the original ImageReward model.

    Notes:
        The underlying model supports:
        - image path strings pointing to a valid file
        - PIL.Image.Image objects
    """

    def __init__(self, config: Optional[ImageRewardScorerConfig] = None) -> None:
        self._config = config or ImageRewardScorerConfig()
        self._model = self._load_model()

    def _load_model(self):
        try:
            import ImageReward as RM
        except Exception as e:  # pragma: no cover
            logger.warning("ImageRewardScorer disabled: failed to import `ImageReward`: %s", e)
            return None

        try:
            kwargs = {
                "download_root": self._config.download_root,
                "med_config": self._config.med_config,
            }
            if self._config.device is None:
                return RM.load(self._config.model_name, **kwargs)
            return RM.load(self._config.model_name, device=self._config.device, **kwargs)
        except Exception as e:
            logger.warning("ImageRewardScorer disabled: failed to load model '%s': %s", self._config.model_name, e)
            return None

    def score(
        self,
        image: ImageInput,
        prompt: Optional[str] = None,
        image_b: Optional[ImageInput] = None,
    ) -> ScoreOutput:
        del image_b  # not used

        if self._model is None:
            return {"score": 0.5, "confidence": 0.0}

        if prompt is None or not isinstance(prompt, str) or not prompt.strip():
            # Prompt is required for ImageReward; return None instead of crashing.
            return {"score": 0.5, "confidence": 0.0}

        try:
            image_obj: Union[str, Image.Image]
            if isinstance(image, Image.Image):
                image_obj = image
            elif isinstance(image, str):
                image_obj = image
            else:
                logger.warning("ImageRewardScorer received unsupported image type: %s", type(image))
                return {"score": 0.5, "confidence": 0.0}

            # The original API signature is score(prompt, image)
            t0 = time.perf_counter()
            score_value = self._model.score(prompt, image_obj)
        except Exception as e:
            logger.warning("ImageRewardScorer scoring failed: %s", e)
            return {"score": 0.5, "confidence": 0.0}

        try:
            raw = float(score_value)
        except Exception as e:
            logger.warning("ImageRewardScorer returned non-numeric score %r: %s", score_value, e)
            return {"score": 0.5, "confidence": 0.0}

        # Per requirements: normalize using sigmoid.
        score = sigmoid(raw)
        confidence = sigmoid(raw)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("event=scorer_infer scorer=image_reward batch=1 ms=%.2f", dt_ms)
        return {"score": float(score), "confidence": float(confidence)}
