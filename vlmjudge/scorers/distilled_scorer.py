# Name: Jayesh Pandey
# Summary: Distilled Reward Model Scorer.

"""
Distilled Reward Model Scorer.

Uses a trained MLP head on top of CLIP embeddings to produce reward scores.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Union, List
import logging

from vlmjudge.scorers.base import BaseScorer, ImageInput, ScoreOutput
from vlmjudge.utils.normalization import clamp

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DistilledScorerConfig:
    model_name: str = "ViT-L-14"
    pretrained: str = "openai"
    checkpoint_path: Optional[str] = None
    device: Optional[str] = None
    hidden_dim: int = 1024

class DistilledRewardModel(nn.Module):
    def __init__(self, clip_model, hidden_dim: int = 1024):
        super().__init__()
        self.clip = clip_model
        embed_dim = self.clip.visual.output_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image_features, text_features):
        combined = torch.cat([image_features, text_features], dim=-1)
        return self.mlp(combined).squeeze(-1)

class DistilledScorer(BaseScorer):
    """
    Scorer that uses the distilled Reward Model.
    """
    def __init__(self, config: Optional[DistilledScorerConfig] = None) -> None:
        self._config = config or DistilledScorerConfig()
        self._device = self._config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP
        try:
            import open_clip  # type: ignore

            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                self._config.model_name, pretrained=self._config.pretrained
            )
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self._config.model_name)
        except Exception as e:
            logger.warning("DistilledScorer disabled: failed to init open_clip (%s)", e)
            self._model = None
            self._preprocess = None
            self._tokenizer = None
            return
        
        # Initialize Wrapper
        self._model = DistilledRewardModel(clip_model, hidden_dim=self._config.hidden_dim).to(self._device)
        self._model.eval()
        
        # Load weights if provided
        if self._config.checkpoint_path:
            try:
                state_dict = torch.load(self._config.checkpoint_path, map_location=self._device)
                # Check if it's a full checkpoint or just the MLP head
                if "model_state_dict" in state_dict:
                    try:
                        self._model.load_state_dict(state_dict["model_state_dict"])
                    except Exception:
                        # Older checkpoints may have a 3-layer Sequential head (no dropout).
                        remapped = self._remap_full_state_dict_for_dropout(state_dict["model_state_dict"])
                        self._model.load_state_dict(remapped)
                else:
                    self._load_mlp_state_dict_compat(state_dict)
                logger.info(f"Loaded distilled weights from {self._config.checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load distilled weights: {e}")

    def _remap_full_state_dict_for_dropout(self, state_dict: dict) -> dict:
        if not isinstance(state_dict, dict):
            raise TypeError("model_state_dict must be a dict.")
        keys = set(state_dict.keys())
        has_dropout_idx = any(k.startswith("mlp.3.") for k in keys)
        has_no_dropout_idx = any(k.startswith("mlp.2.") for k in keys)
        if has_dropout_idx or not has_no_dropout_idx:
            return state_dict
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("mlp.2."):
                remapped["mlp.3." + k[len("mlp.2."):]] = v
            else:
                remapped[k] = v
        return remapped

    def _load_mlp_state_dict_compat(self, state_dict: dict) -> None:
        """
        Support loading heads saved from different training script variants.

        Common formats:
        - nn.Sequential without dropout: keys like "0.weight", "2.weight"
        - nn.Sequential with dropout: keys like "0.weight", "3.weight"
        - full model checkpoint already handled by caller.
        """
        if not isinstance(state_dict, dict):
            raise TypeError("Head state_dict must be a dict.")

        keys = set(state_dict.keys())
        has_dropout_idx = any(k.startswith("3.") for k in keys) or any(k.startswith("mlp.3.") for k in keys)
        has_no_dropout_idx = any(k.startswith("2.") for k in keys) or any(k.startswith("mlp.2.") for k in keys)

        if has_dropout_idx or not has_no_dropout_idx:
            self._model.mlp.load_state_dict(state_dict)
            return

        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("2."):
                remapped["3." + k[len("2."):]] = v
            elif k.startswith("mlp.2."):
                remapped["mlp.3." + k[len("mlp.2."):]] = v
            else:
                remapped[k] = v

        self._model.mlp.load_state_dict(remapped)

    def _load_pil(self, image: ImageInput) -> Optional[Image.Image]:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            try:
                return Image.open(image).convert("RGB")
            except Exception:
                return None
        return None

    def score(
        self,
        image: ImageInput,
        prompt: Optional[str] = None,
        image_b: Optional[ImageInput] = None,
    ) -> ScoreOutput:
        if self._model is None or self._preprocess is None or self._tokenizer is None:
            return {"score": 0.5, "confidence": 0.0}
        if prompt is None or not isinstance(prompt, str):
            return {"score": 0.5, "confidence": 0.0}

        pil_a = self._load_pil(image)
        if pil_a is None:
            return {"score": 0.5, "confidence": 0.0}

        try:
            image_tensor_a = self._preprocess(pil_a).unsqueeze(0).to(self._device)
            text_tokens = self._tokenizer([prompt]).to(self._device)
            with torch.no_grad():
                image_features_a = self._model.clip.encode_image(image_tensor_a)
                text_features = self._model.clip.encode_text(text_tokens)
                
                image_features_a = F.normalize(image_features_a, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                raw_logit_a = self._model(image_features_a, text_features).item()

                score_a = float(torch.sigmoid(torch.tensor(raw_logit_a)).item())
                score_a = float(clamp(score_a, 0.0, 1.0))

                # Confidence estimation:
                # - If `image_b` is provided, use abs(score_a - score_b) (Phase 5.5 spec).
                # - Otherwise use distance from 0.5 as a single-image heuristic.
                if image_b is not None:
                    pil_b = self._load_pil(image_b)
                    if pil_b is None:
                        conf = float(clamp(abs(score_a - 0.5) * 2.0, 0.0, 1.0))
                        return {"score": score_a, "confidence": conf}
                    image_tensor_b = self._preprocess(pil_b).unsqueeze(0).to(self._device)
                    image_features_b = self._model.clip.encode_image(image_tensor_b)
                    image_features_b = F.normalize(image_features_b, dim=-1)
                    raw_logit_b = self._model(image_features_b, text_features).item()
                    score_b = float(torch.sigmoid(torch.tensor(raw_logit_b)).item())
                    score_b = float(clamp(score_b, 0.0, 1.0))
                    conf = float(clamp(abs(score_a - score_b), 0.0, 1.0))
                else:
                    conf = float(clamp(abs(score_a - 0.5) * 2.0, 0.0, 1.0))

            return {"score": score_a, "confidence": conf}
        except Exception as e:
            logger.warning(f"DistilledScorer scoring failed: {e}")
            return {"score": 0.5, "confidence": 0.0}


def predict_with_fallback(
    *,
    student: BaseScorer,
    teacher: BaseScorer,
    image: ImageInput,
    prompt: str,
    confidence_threshold: float = 0.6,
) -> ScoreOutput:
    """
    Hybrid inference: fall back to the teacher scorer when student confidence is low.
    """
    student_out = student.score(image, prompt=prompt, image_b=None)
    conf = float(student_out.get("confidence", 0.0))
    conf = float(clamp(conf, 0.0, 1.0))
    if conf < float(confidence_threshold):
        return teacher.score(image, prompt=prompt, image_b=None)
    return student_out
