"""
Scorer interfaces.

A scorer produces a scalar score for a single (image, prompt) pair.
This is the primary building block for future comparison and ranking pipelines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Sequence, Union

from PIL import Image

ImageInput = Union[str, Image.Image]
ScoreOutput = Dict[str, float]


class BaseScorer(ABC):
    """
    Abstract scorer interface.

    Implementations should be safe to call repeatedly and should be stateless from the
    perspective of the caller (internal caching is allowed).
    """

    @abstractmethod
    def score(
        self,
        image: ImageInput,
        prompt: Optional[str] = None,
        image_b: Optional[ImageInput] = None,
    ) -> ScoreOutput:
        """
        Score an image (or image pair) with an optional prompt.

        Args:
            image: Either a filesystem path to an image or a `PIL.Image.Image`.
            prompt: Optional text prompt. Some scorers require it (e.g., CLIP, ImageReward).
            image_b: Optional second image for pairwise scorers (e.g., LPIPS).

        Returns:
            A dict with:
              - "score": float in [0, 1]
              - "confidence": float in [0, 1]

            Implementations must always return this schema. On failure or missing inputs,
            return a safe neutral default with zero confidence:
              {"score": 0.5, "confidence": 0.0}

        Design notes:
            - Promptless scorers should ignore `prompt`.
            - Pairwise scorers should use `image` as A and `image_b` as B.
            - Implementations must not crash on recoverable errors; prefer returning None
              (or a neutral default) and logging a warning.
        """

    def score_batch(
        self,
        images: Sequence[ImageInput],
        prompts: Optional[Sequence[Optional[str]]] = None,
        image_bs: Optional[Sequence[Optional[ImageInput]]] = None,
    ) -> List[ScoreOutput]:
        """
        Batch scoring API.

        Implementations may override this for efficient vectorized inference.
        Default behavior loops over `score(...)` safely.
        """
        if prompts is not None and len(prompts) != len(images):
            raise ValueError("`prompts` length must match `images` length.")
        if image_bs is not None and len(image_bs) != len(images):
            raise ValueError("`image_bs` length must match `images` length.")

        outputs: List[ScoreOutput] = []
        for i, img in enumerate(images):
            prompt = prompts[i] if prompts is not None else None
            image_b = image_bs[i] if image_bs is not None else None
            outputs.append(self.score(img, prompt=prompt, image_b=image_b))
        return outputs
