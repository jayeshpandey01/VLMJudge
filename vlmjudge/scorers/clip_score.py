# Name: Jayesh Pandey
# Summary: OpenCLIP-based scorer.

"""
OpenCLIP-based scorer.

Computes cosine similarity between image and prompt embeddings, normalized into [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Optional, Union

from PIL import Image

from vlmjudge.scorers.base import BaseScorer, ImageInput, ScoreOutput
from vlmjudge.utils.normalization import clamp, cosine_to_unit_interval

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenCLIPScorerConfig:
    """
    Configuration for OpenCLIPScorer.

    Defaults target a strong modern model. If the chosen pretrained tag is not available
    in your installed `open_clip` build, set `pretrained` accordingly.
    """

    model_name: str = "ViT-H-14"
    pretrained: str = "laion2b_s32b_b79k"
    device: Optional[str] = None  # e.g. "cpu", "cuda", "cuda:0"


class OpenCLIPScorer(BaseScorer):
    """
    Scores image↔prompt alignment using OpenCLIP.

    Returns:
        Similarity mapped to [0, 1] via `(cos + 1) / 2`.
    """

    def __init__(self, config: Optional[OpenCLIPScorerConfig] = None) -> None:
        self._config = config or OpenCLIPScorerConfig()
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None

        self._try_init()

    def _try_init(self) -> None:
        try:
            import torch
            import open_clip
        except Exception as e:
            logger.warning("OpenCLIPScorer unavailable (missing dependency): %s", e)
            return

        device = self._config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                self._config.model_name, pretrained=self._config.pretrained
            )
            tokenizer = open_clip.get_tokenizer(self._config.model_name)
            model.eval()
            model.to(self._device)
        except Exception as e:
            logger.warning("OpenCLIPScorer failed to initialize model '%s/%s': %s", self._config.model_name, self._config.pretrained, e)
            return

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer

    def _load_pil(self, image: ImageInput) -> Optional[Image.Image]:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            try:
                return Image.open(image).convert("RGB")
            except Exception as e:
                logger.warning("OpenCLIPScorer failed to open image path %r: %s", image, e)
                return None
        logger.warning("OpenCLIPScorer received unsupported image type: %s", type(image))
        return None

    def _fallback(self) -> ScoreOutput:
        return {"score": 0.5, "confidence": 0.0}

    def score(
        self,
        image: ImageInput,
        prompt: Optional[str] = None,
        image_b: Optional[ImageInput] = None,
    ) -> ScoreOutput:
        del image_b  # not used

        if self._model is None or self._preprocess is None or self._tokenizer is None:
            return self._fallback()

        if prompt is None or not isinstance(prompt, str) or not prompt.strip():
            logger.warning("OpenCLIPScorer requires a non-empty `prompt`.")
            return self._fallback()

        pil = self._load_pil(image)
        if pil is None:
            return self._fallback()

        try:
            import torch
        except Exception as e:
            logger.warning("OpenCLIPScorer missing torch at runtime: %s", e)
            return self._fallback()

        t0 = time.perf_counter()
        try:
            image_tensor = self._preprocess(pil).unsqueeze(0).to(self._device)
            text_tokens = self._tokenizer([prompt]).to(self._device)

            with torch.no_grad():
                image_features = self._model.encode_image(image_tensor)
                text_features = self._model.encode_text(text_tokens)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                cosine = (image_features * text_features).sum(dim=-1).item()

            score = float(cosine_to_unit_interval(float(cosine)))
            confidence = float(clamp(abs(float(cosine)), 0.0, 1.0))
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("event=scorer_infer scorer=openclip batch=1 ms=%.2f", dt_ms)
            return {"score": score, "confidence": confidence}
        except Exception as e:
            logger.warning("OpenCLIPScorer scoring failed: %s", e)
            return self._fallback()

    def score_batch(
        self,
        images,
        prompts=None,
        image_bs=None,
    ):
        del image_bs
        if self._model is None or self._preprocess is None or self._tokenizer is None:
            return [self._fallback() for _ in images]

        if prompts is None:
            logger.warning("OpenCLIPScorer.score_batch requires prompts.")
            return [self._fallback() for _ in images]
        if len(prompts) != len(images):
            raise ValueError("`prompts` length must match `images` length.")

        try:
            import torch
        except Exception as e:
            logger.warning("OpenCLIPScorer missing torch at runtime: %s", e)
            return [self._fallback() for _ in images]

        pils: list[Image.Image] = []
        valid_mask: list[bool] = []
        for img in images:
            pil = self._load_pil(img)
            if pil is None:
                valid_mask.append(False)
                pils.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))
            else:
                valid_mask.append(True)
                pils.append(pil)

        texts: list[str] = []
        for p in prompts:
            if p is None or not isinstance(p, str) or not p.strip():
                texts.append("")
            else:
                texts.append(p)

        t0 = time.perf_counter()
        try:
            image_tensor = torch.stack([self._preprocess(p) for p in pils], dim=0).to(self._device)
            text_tokens = self._tokenizer(texts).to(self._device)

            with torch.no_grad():
                image_features = self._model.encode_image(image_tensor)
                text_features = self._model.encode_text(text_tokens)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                cosines = (image_features * text_features).sum(dim=-1)  # [B]

            outs: list[ScoreOutput] = []
            for i in range(len(images)):
                if not valid_mask[i] or texts[i] == "":
                    outs.append(self._fallback())
                    continue
                c = float(cosines[i].item())
                outs.append(
                    {
                        "score": float(cosine_to_unit_interval(c)),
                        "confidence": float(clamp(abs(c), 0.0, 1.0)),
                    }
                )

            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("event=scorer_infer scorer=openclip batch=%d ms=%.2f", len(images), dt_ms)
            return outs
        except Exception as e:
            logger.warning("OpenCLIPScorer batch scoring failed: %s", e)
            return [self._fallback() for _ in images]
