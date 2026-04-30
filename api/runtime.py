"""
author: Jayesh Pandey
summary: Inference runtime handling model loading, image resolution, and hybrid scoring/comparison logic.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import threading
import time
import base64
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from vlmjudge.pipelines.compare_pipeline import ComparePipeline, ComparePipelineConfig
from vlmjudge.scorers import (
    AestheticScorer,
    AestheticScorerConfig,
    ImageRewardScorer,
    ImageRewardScorerConfig,
    LPIPSScorer,
    LPIPSScorerConfig,
    OpenCLIPScorer,
    OpenCLIPScorerConfig,
)
from vlmjudge.utils.normalization import clamp
from train_reward_model import DistilledRewardModel

logger = logging.getLogger(__name__)
_SHADOW_LOCK = threading.Lock()


def _resolve_path(path_str: str, *, base_dir: Path) -> str:
    """
    Resolve a potentially-relative path against a base directory.

    We do this so configs can use relative paths regardless of the process CWD.
    """
    if not isinstance(path_str, str) or not path_str.strip():
        return path_str
    p = Path(path_str.strip())
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


class MissingStudentEngine:
    """
    Placeholder engine used when the student checkpoint is missing.

    This keeps the API running (e.g. teacher-only mode) instead of crashing on startup.
    """

    def __init__(self, *, checkpoint_path: str, resolved_path: str) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.resolved_path = str(resolved_path)

    def _err(self) -> FileNotFoundError:
        return FileNotFoundError(
            f"student checkpoint not found: {self.checkpoint_path} (resolved: {self.resolved_path}). "
            "Set `student_checkpoint` in config.yaml to an existing .pt file or train/export one to that path."
        )

    def score(self, pil: Image.Image, prompt: str) -> Tuple[float, float]:
        raise self._err()

    def compare(self, pil_a: Image.Image, pil_b: Image.Image, prompt: str) -> Tuple[float, float, float, str]:
        raise self._err()


def _coerce_teacher_winner_and_confidence(
    *,
    score_a: float,
    score_b: float,
    winner: str,
    confidence: float,
    tie_threshold: float,
) -> Tuple[str, float]:
    """
    Teacher pipeline may return "tie" with a very small calibrated confidence even when
    aggregate scores slightly differ. For UI friendliness, promote ties to A/B when the
    aggregate delta exceeds the runtime tie_threshold, and ensure confidence is at least
    the absolute score delta.

    This keeps VLM-driven A/B decisions intact (we only override ties).
    """
    derived_winner = _winner_from_scores(score_a, score_b, tie_threshold)
    derived_conf = float(clamp(abs(float(score_a) - float(score_b)), 0.0, 1.0))

    out_winner = str(winner)
    if out_winner == "tie" and derived_winner != "tie":
        logger.info(
            "event=teacher_tie_promoted scoreA=%.4f scoreB=%.4f tie_threshold=%.4f -> %s",
            float(score_a),
            float(score_b),
            float(tie_threshold),
            derived_winner,
        )
        out_winner = derived_winner

    out_conf = float(clamp(float(confidence), 0.0, 1.0))
    if derived_conf > out_conf:
        out_conf = derived_conf

    return out_winner, out_conf


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    line = __import__("json").dumps(obj, ensure_ascii=False)
    with _SHADOW_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _load_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}

    text = p.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        obj = yaml.safe_load(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        # Fallback: minimal "key: value" parser for simple configs.
        out: Dict[str, Any] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            if v.lower() in ("null", "none", ""):
                out[k] = None
            elif v.lower() in ("true", "false"):
                out[k] = v.lower() == "true"
            else:
                try:
                    if "." in v:
                        out[k] = float(v)
                    else:
                        out[k] = int(v)
                except Exception:
                    out[k] = v.strip("\"'")
        return out


def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key, default)
    return default if v is None else v


def _device_from_config(cfg: Dict[str, Any]) -> str:
    dev = cfg.get("device", None)
    if dev is None or str(dev).lower() in ("", "none", "null"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(dev)


def _load_student_checkpoint(model: DistilledRewardModel, checkpoint_path: str, device: str) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        try:
            model.load_state_dict(state["model_state_dict"])
            return
        except Exception:
            sd = state["model_state_dict"]
            if isinstance(sd, dict):
                remapped = {}
                for k, v in sd.items():
                    if k.startswith("mlp.2."):
                        remapped["mlp.3." + k[len("mlp.2."):]] = v
                    else:
                        remapped[k] = v
                model.load_state_dict(remapped)
                return
            raise
    if isinstance(state, dict):
        keys = set(state.keys())
        has_dropout_idx = any(k.startswith("3.") for k in keys)
        has_no_dropout_idx = any(k.startswith("2.") for k in keys)
        if has_no_dropout_idx and not has_dropout_idx:
            remapped = {}
            for k, v in state.items():
                if k.startswith("2."):
                    remapped["3." + k[len("2."):]] = v
                else:
                    remapped[k] = v
            model.mlp.load_state_dict(remapped)
        else:
            model.mlp.load_state_dict(state)
        return
    raise ValueError("Unsupported student checkpoint format.")


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def resolve_image(
    image_ref: str,
    *,
    cache_dir: str,
    url_timeout_sec: int,
    max_download_mb: int,
) -> Image.Image:
    """
    Supports:
      - local file paths
      - http(s) URLs (downloaded + cached)
      - data URLs (base64-encoded images)
    """
    if not isinstance(image_ref, str) or not image_ref.strip():
        raise ValueError("image reference must be a non-empty string")

    ref = image_ref.strip()
    if ref.startswith("data:"):
        # Minimal RFC 2397 support: only base64-encoded payloads.
        try:
            header, payload = ref.split(",", 1)
        except ValueError as e:
            raise ValueError("invalid data URL") from e

        if ";base64" not in header.lower():
            raise ValueError("unsupported data URL (expected base64)")

        try:
            raw = base64.b64decode(payload.encode("utf-8"), validate=False)
        except Exception as e:
            raise ValueError("failed to decode base64 data URL") from e

        if len(raw) > int(max_download_mb) * 1024 * 1024:
            raise ValueError(f"data URL too large (> {max_download_mb} MB)")

        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise ValueError("failed to decode data URL image") from e

    if ref.startswith("http://") or ref.startswith("https://"):
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        key = _sha256(ref)
        target = Path(cache_dir) / f"{key}.img"
        if target.exists() and target.stat().st_size > 0:
            return Image.open(str(target)).convert("RGB")

        req = urllib.request.Request(ref, headers={"User-Agent": "ImageReward-API/1.0"})
        with urllib.request.urlopen(req, timeout=int(url_timeout_sec)) as resp:
            data = resp.read(int(max_download_mb) * 1024 * 1024 + 1)
            if len(data) > int(max_download_mb) * 1024 * 1024:
                raise ValueError(f"download too large (> {max_download_mb} MB)")
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as e:
                raise ValueError("failed to decode downloaded image") from e

        try:
            target.write_bytes(data)
        except Exception:
            pass
        return img

    # Local file
    p = Path(ref)
    if not p.exists():
        raise FileNotFoundError(f"image path not found: {ref}")
    return Image.open(str(p)).convert("RGB")


def _winner_from_scores(score_a: float, score_b: float, tie_threshold: float) -> str:
    delta = float(score_a) - float(score_b)
    if abs(delta) <= float(tie_threshold):
        return "tie"
    return "A" if delta > 0 else "B"


@dataclass
class StudentEngine:
    model: DistilledRewardModel
    device: str
    tie_threshold: float
    _prompt_cache: "OrderedDict[str, torch.Tensor]"
    _prompt_cache_lock: threading.Lock
    _prompt_cache_max: int = 256

    @classmethod
    def load(cls, *, checkpoint_path: str, device: str, tie_threshold: float) -> "StudentEngine":
        model = DistilledRewardModel().to(device)
        _load_student_checkpoint(model, checkpoint_path, device)
        model.eval()
        return cls(
            model=model,
            device=device,
            tie_threshold=float(tie_threshold),
            _prompt_cache=OrderedDict(),
            _prompt_cache_lock=threading.Lock(),
        )

    def _encode_text_cached(self, prompt: str) -> torch.Tensor:
        key = prompt.strip()
        with self._prompt_cache_lock:
            hit = self._prompt_cache.get(key)
            if hit is not None:
                self._prompt_cache.move_to_end(key)
                return hit

        tokens = self.model.tokenizer([key]).to(self.device)
        with torch.no_grad():
            tf = self.model.clip.encode_text(tokens)
            tf = F.normalize(tf, dim=-1)

        with self._prompt_cache_lock:
            self._prompt_cache[key] = tf
            self._prompt_cache.move_to_end(key)
            while len(self._prompt_cache) > int(self._prompt_cache_max):
                self._prompt_cache.popitem(last=False)
        return tf

    def score(self, pil: Image.Image, prompt: str) -> Tuple[float, float]:
        img_t = self.model.preprocess(pil).unsqueeze(0).to(self.device)
        tf = self._encode_text_cached(prompt)
        with torch.no_grad():
            imf = self.model.clip.encode_image(img_t)
            imf = F.normalize(imf, dim=-1)
            combined = torch.cat([imf, tf], dim=-1)
            logit = self.model.mlp(combined).squeeze(0)
            score = float(torch.sigmoid(logit).item())
            score = float(clamp(score, 0.0, 1.0))
            confidence = float(clamp(abs(score - 0.5) * 2.0, 0.0, 1.0))
        return score, confidence

    def compare(self, pil_a: Image.Image, pil_b: Image.Image, prompt: str) -> Tuple[float, float, float, str]:
        imgs = torch.stack([self.model.preprocess(pil_a), self.model.preprocess(pil_b)], dim=0).to(self.device)
        tf = self._encode_text_cached(prompt)  # [1,D]
        tf2 = tf.repeat(imgs.shape[0], 1)

        with torch.no_grad():
            imf = self.model.clip.encode_image(imgs)
            imf = F.normalize(imf, dim=-1)
            combined = torch.cat([imf, tf2], dim=-1)
            logits = self.model.mlp(combined).squeeze(-1)
            scores = torch.sigmoid(logits).detach().cpu().tolist()

        score_a = float(clamp(float(scores[0]), 0.0, 1.0))
        score_b = float(clamp(float(scores[1]), 0.0, 1.0))
        confidence = float(clamp(abs(score_a - score_b), 0.0, 1.0))
        winner = _winner_from_scores(score_a, score_b, self.tie_threshold)
        return score_a, score_b, confidence, winner


@dataclass
class TeacherEngine:
    cfg: Dict[str, Any]
    device: str
    _lock: threading.Lock
    _pipeline: Optional[ComparePipeline] = None

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                return

            tcfg = self.cfg.get("teacher", {})
            model_name = str(_get(tcfg, "model_name", "ImageReward-v1.0"))
            download_root = tcfg.get("download_root", None)
            med_config = tcfg.get("med_config", None)
            aesthetic_weights = tcfg.get("aesthetic_weights", None)

            scorers: Dict[str, Any] = {}
            scorers["image_reward"] = ImageRewardScorer(
                ImageRewardScorerConfig(
                    model_name=model_name,
                    device=self.device,
                    download_root=download_root,
                    med_config=med_config,
                )
            )
            scorers["openclip"] = OpenCLIPScorer(OpenCLIPScorerConfig(device=self.device))
            scorers["aesthetic"] = AestheticScorer(AestheticScorerConfig(weights_path=aesthetic_weights, device=self.device))
            scorers["lpips"] = LPIPSScorer(LPIPSScorerConfig(device=self.device))

            vlm_judge = None
            if bool(_get(tcfg, "use_vlm", False)) or bool(_get(tcfg, "enable_vlm", False)):
                from vlmjudge.vlm.ensemble import VLMEnsemble
                vlm_judge = VLMEnsemble(
                    config=tcfg,
                    strict=False,
                )

            self._pipeline = ComparePipeline(
                scorers,
                config=ComparePipelineConfig(
                    threshold=float(_get(tcfg, "teacher_threshold", 0.05)),
                    vlm_runs=int(_get(tcfg, "vlm_runs", 3)),
                ),
                vlm_judge=vlm_judge,
            )

    def compare(self, pil_a: Image.Image, pil_b: Image.Image, prompt: str) -> Dict[str, Any]:
        self.ensure_loaded()
        assert self._pipeline is not None
        # Conservative lock: underlying torch models may not be thread-safe under concurrent access.
        with self._lock:
            return self._pipeline.run(pil_a, pil_b, prompt)


@dataclass
class InferenceRuntime:
    cfg: Dict[str, Any]
    device: str
    student_stable: Any
    student_canary: Optional[Any]
    shadow_students: List[Tuple[str, Any]] = field(default_factory=list)
    teacher: Optional[TeacherEngine] = None

    @classmethod
    def from_yaml(cls, path: str) -> "InferenceRuntime":
        cfg = _load_yaml(path)
        cfg_dir = Path(path).resolve().parent
        device = _device_from_config(cfg)
        student_checkpoint = str(_get(cfg, "student_checkpoint", "distilled_model/best.pt")).strip()
        student_checkpoint_resolved = _resolve_path(student_checkpoint, base_dir=cfg_dir)
        tie_threshold = float(_get(cfg, "tie_threshold", 0.02))
        if student_checkpoint and Path(student_checkpoint).exists():
            student_stable = StudentEngine.load(
                checkpoint_path=student_checkpoint, device=device, tie_threshold=tie_threshold
            )
        elif student_checkpoint_resolved and Path(student_checkpoint_resolved).exists():
            student_stable = StudentEngine.load(
                checkpoint_path=student_checkpoint_resolved, device=device, tie_threshold=tie_threshold
            )
        else:
            logger.error(
                "event=student_checkpoint_missing checkpoint=%s resolved=%s",
                student_checkpoint,
                student_checkpoint_resolved,
            )
            student_stable = MissingStudentEngine(
                checkpoint_path=student_checkpoint, resolved_path=student_checkpoint_resolved
            )

        student_canary = None
        deployment_mode = str(_get(cfg, "deployment_mode", "stable")).lower()
        canary_checkpoint = cfg.get("canary_checkpoint", None)
        if deployment_mode == "canary" and isinstance(canary_checkpoint, str) and canary_checkpoint.strip():
            try:
                canary_ck = str(canary_checkpoint).strip()
                canary_resolved = _resolve_path(canary_ck, base_dir=cfg_dir)
                load_path = None
                if canary_ck and Path(canary_ck).exists():
                    load_path = canary_ck
                elif canary_resolved and Path(canary_resolved).exists():
                    load_path = canary_resolved
                if load_path is not None:
                    student_canary = StudentEngine.load(
                        checkpoint_path=load_path, device=device, tie_threshold=tie_threshold
                    )
                    logger.info("event=canary_loaded checkpoint=%s", str(canary_checkpoint))
            except Exception as e:
                logger.warning("event=canary_load_failed err=%s", e)
                student_canary = None

        shadow_students: List[Tuple[str, Any]] = []
        shadow_models = cfg.get("shadow_models", [])
        if isinstance(shadow_models, list):
            for p in shadow_models[:5]:
                if not isinstance(p, str) or not p.strip():
                    continue
                ck = str(p).strip()
                try:
                    ck_resolved = _resolve_path(ck, base_dir=cfg_dir)
                    load_path = None
                    if ck and Path(ck).exists():
                        load_path = ck
                    elif ck_resolved and Path(ck_resolved).exists():
                        load_path = ck_resolved
                    if load_path is not None:
                        shadow_students.append(
                            (ck, StudentEngine.load(checkpoint_path=load_path, device=device, tie_threshold=tie_threshold))
                        )
                        logger.info("event=shadow_model_loaded checkpoint=%s", ck)
                except Exception as e:
                    logger.warning("event=shadow_model_load_failed checkpoint=%s err=%s", ck, e)

        enable_teacher = bool(_get(cfg, "enable_teacher", False))
        teacher = TeacherEngine(cfg=cfg, device=device, _lock=threading.Lock()) if enable_teacher else None
        return cls(
            cfg=cfg,
            device=device,
            student_stable=student_stable,
            student_canary=student_canary,
            shadow_students=shadow_students,
            teacher=teacher,
        )

    def _choose_student(self) -> Tuple[StudentEngine, str]:
        """
        Canary routing: randomly route a portion of requests to the canary student model.
        """
        deployment_mode = str(_get(self.cfg, "deployment_mode", "stable")).lower()
        ratio = float(_get(self.cfg, "canary_ratio", 0.0))
        if deployment_mode == "canary" and self.student_canary is not None:
            try:
                import random as _random

                if _random.random() < float(ratio):
                    return self.student_canary, "canary"
            except Exception:
                pass
        return self.student_stable, "stable"

    def compare(self, *, prompt: str, image_a_ref: str, image_b_ref: str, return_debug: bool = False) -> Dict[str, Any]:
        if not prompt or not isinstance(prompt, str):
            out = {
                "winner": "tie",
                "confidence": 0.0,
                "scoreA": 0.5,
                "scoreB": 0.5,
                "method": "safe_fallback",
                "timing_ms": {"student": 0.0, "teacher": 0.0, "total": 0.0},
            }
            if return_debug:
                out["_debug"] = {"error": "missing_prompt"}
            return out

        cache_dir = str(_get(self.cfg, "cache_dir", ".cache/api_images"))
        url_timeout_sec = int(_get(self.cfg, "url_timeout_sec", 10))
        max_download_mb = int(_get(self.cfg, "max_download_mb", 20))

        try:
            pil_a = resolve_image(
                image_a_ref, cache_dir=cache_dir, url_timeout_sec=url_timeout_sec, max_download_mb=max_download_mb
            )
            pil_b = resolve_image(
                image_b_ref, cache_dir=cache_dir, url_timeout_sec=url_timeout_sec, max_download_mb=max_download_mb
            )
        except Exception as e:
            logger.warning("event=image_resolve_failed err=%s", e)
            out = {
                "winner": "tie",
                "confidence": 0.0,
                "scoreA": 0.5,
                "scoreB": 0.5,
                "method": "safe_fallback",
                "timing_ms": {"student": 0.0, "teacher": 0.0, "total": 0.0},
            }
            if return_debug:
                out["_debug"] = {"error": "image_resolve_failed", "detail": str(e)}
            return out

        conf_threshold = float(_get(self.cfg, "confidence_threshold", 0.6))
        hybrid_combine = bool(_get(self.cfg, "hybrid_combine", False))
        w_s = float(_get(self.cfg, "hybrid_student_weight", 0.7))
        w_t = float(_get(self.cfg, "hybrid_teacher_weight", 0.3))
        tie_threshold = float(_get(self.cfg, "tie_threshold", 0.02))

        student_engine, student_variant = self._choose_student()
        student_checkpoint = str(_get(self.cfg, "student_checkpoint", ""))
        if student_variant == "canary":
            student_checkpoint = str(self.cfg.get("canary_checkpoint", student_checkpoint))

        # If the student checkpoint is missing, don't throw/log on every request.
        # Serve using the teacher (if enabled) or a safe fallback.
        if isinstance(student_engine, MissingStudentEngine):
            if self.teacher is not None:
                try:
                    t1 = _now_ms()
                    teacher_out = self.teacher.compare(pil_a, pil_b, prompt)
                    dt_teacher = _now_ms() - t1
                    winner_t = str(teacher_out.get("winner", "tie"))
                    conf_t = float(clamp(float(teacher_out.get("confidence", 0.0)), 0.0, 1.0))
                    agg = teacher_out.get("structured", {}).get("aggregate", {})
                    score_a_t = float(clamp(float(agg.get("A", {}).get("score", 0.5)), 0.0, 1.0))
                    score_b_t = float(clamp(float(agg.get("B", {}).get("score", 0.5)), 0.0, 1.0))
                    winner_t, conf_t = _coerce_teacher_winner_and_confidence(
                        score_a=score_a_t,
                        score_b=score_b_t,
                        winner=winner_t,
                        confidence=conf_t,
                        tie_threshold=tie_threshold,
                    )
                    explanation = str(teacher_out.get("explanation", "")).strip()
                    out = {
                        "winner": winner_t,
                        "confidence": float(conf_t),
                        "scoreA": float(score_a_t),
                        "scoreB": float(score_b_t),
                        "method": "teacher",
                        "reasoning": explanation,
                        "timing_ms": {"student": 0.0, "teacher": float(dt_teacher), "total": float(dt_teacher)},
                        "scores": {"structured": teacher_out.get("structured", {}), "vlm": teacher_out.get("vlm", {})},
                    }
                    if return_debug:
                        out["_debug"] = {
                            "student_variant": student_variant,
                            "student_checkpoint": student_checkpoint,
                            "student_winner": None,
                            "teacher_winner": winner_t,
                            "agreement": None,
                            "used_teacher": True,
                            "error": "student_checkpoint_missing",
                        }
                    return out
                except Exception as e2:
                    logger.warning("event=teacher_infer_failed err=%s", e2)

            out = {
                "winner": "tie",
                "confidence": 0.0,
                "scoreA": 0.5,
                "scoreB": 0.5,
                "method": "safe_fallback",
                "timing_ms": {"student": 0.0, "teacher": 0.0, "total": 0.0},
            }
            if return_debug:
                out["_debug"] = {
                    "student_variant": student_variant,
                    "student_checkpoint": student_checkpoint,
                    "student_winner": None,
                    "teacher_winner": None,
                    "agreement": None,
                    "used_teacher": False,
                    "error": "student_checkpoint_missing",
                }
            return out

        try:
            t0 = _now_ms()
            score_a_s, score_b_s, conf_s, winner_s = student_engine.compare(pil_a, pil_b, prompt)
            dt_student = _now_ms() - t0
        except Exception as e:
            logger.warning("event=student_infer_failed err=%s", e)
            # Try teacher as last resort.
            if self.teacher is not None:
                try:
                    t1 = _now_ms()
                    teacher_out = self.teacher.compare(pil_a, pil_b, prompt)
                    dt_teacher = _now_ms() - t1
                    winner_t = str(teacher_out.get("winner", "tie"))
                    conf_t = float(clamp(float(teacher_out.get("confidence", 0.0)), 0.0, 1.0))
                    agg = teacher_out.get("structured", {}).get("aggregate", {})
                    score_a_t = float(clamp(float(agg.get("A", {}).get("score", 0.5)), 0.0, 1.0))
                    score_b_t = float(clamp(float(agg.get("B", {}).get("score", 0.5)), 0.0, 1.0))
                    winner_t, conf_t = _coerce_teacher_winner_and_confidence(
                        score_a=score_a_t,
                        score_b=score_b_t,
                        winner=winner_t,
                        confidence=conf_t,
                        tie_threshold=tie_threshold,
                    )
                    explanation = str(teacher_out.get("explanation", "")).strip()
                    out = {
                        "winner": winner_t,
                        "confidence": float(conf_t),
                        "scoreA": float(score_a_t),
                        "scoreB": float(score_b_t),
                        "method": "teacher",
                        "reasoning": explanation,
                        "timing_ms": {"student": 0.0, "teacher": float(dt_teacher), "total": float(dt_teacher)},
                        "scores": {"structured": teacher_out.get("structured", {}), "vlm": teacher_out.get("vlm", {})},
                    }
                    if return_debug:
                        out["_debug"] = {
                            "student_variant": student_variant,
                            "student_checkpoint": student_checkpoint,
                            "student_winner": None,
                            "teacher_winner": winner_t,
                            "agreement": None,
                            "used_teacher": True,
                        }
                    return out
                except Exception as e2:
                    logger.warning("event=teacher_infer_failed err=%s", e2)

            # Final guard: safe fallback
            out = {
                "winner": "tie",
                "confidence": 0.0,
                "scoreA": 0.5,
                "scoreB": 0.5,
                "method": "safe_fallback",
                "timing_ms": {"student": 0.0, "teacher": 0.0, "total": 0.0},
            }
            if return_debug:
                out["_debug"] = {
                    "student_variant": student_variant,
                    "student_checkpoint": student_checkpoint,
                    "student_winner": None,
                    "teacher_winner": None,
                    "agreement": None,
                    "used_teacher": False,
                }
            return out

        method = "student"
        out = {
            "winner": winner_s,
            "confidence": float(conf_s),
            "scoreA": float(score_a_s),
            "scoreB": float(score_b_s),
            "method": method,
            "timing_ms": {"student": float(dt_student), "teacher": 0.0, "total": float(dt_student)},
        }

        # Shadow evaluation trigger (does not affect response).
        enable_shadow = bool(_get(self.cfg, "enable_shadow_teacher", False))
        shadow_rate = float(_get(self.cfg, "shadow_sample_rate", 0.0))
        do_shadow = False
        if enable_shadow and shadow_rate > 0.0:
            try:
                import random as _random

                do_shadow = _random.random() < shadow_rate
            except Exception:
                do_shadow = False

        if conf_s >= conf_threshold or self.teacher is None:
            if return_debug:
                out["_debug"] = {
                    "student_variant": student_variant,
                    "student_checkpoint": student_checkpoint,
                    "student_winner": winner_s,
                    "teacher_winner": None,
                    "agreement": None,
                    "used_teacher": False,
                }
            if do_shadow and self.teacher is not None:
                self._spawn_shadow_compare(
                    prompt=prompt,
                    image_a_ref=image_a_ref,
                    image_b_ref=image_b_ref,
                    pil_a=pil_a,
                    pil_b=pil_b,
                    student_variant=student_variant,
                    student_checkpoint=student_checkpoint,
                    student_out={"winner": winner_s, "confidence": conf_s, "scoreA": score_a_s, "scoreB": score_b_s},
                    method_used="student",
                    teacher_out=None,
                )
            return out

        # Fallback to teacher
        try:
            t1 = _now_ms()
            teacher_out = self.teacher.compare(pil_a, pil_b, prompt)
            dt_teacher = _now_ms() - t1
        except Exception as e:
            logger.warning("event=teacher_infer_failed err=%s", e)
            # Final guard: safe fallback
            out = {
                "winner": "tie",
                "confidence": 0.0,
                "scoreA": 0.5,
                "scoreB": 0.5,
                "method": "safe_fallback",
                "timing_ms": {"student": float(dt_student), "teacher": 0.0, "total": float(dt_student)},
            }
            if return_debug:
                out["_debug"] = {
                    "student_variant": student_variant,
                    "student_checkpoint": student_checkpoint,
                    "student_winner": winner_s,
                    "teacher_winner": None,
                    "agreement": None,
                    "used_teacher": False,
                }
            return out

        try:
            agg = teacher_out.get("structured", {}).get("aggregate", {})
            score_a_t = float(clamp(float(agg.get("A", {}).get("score", 0.5)), 0.0, 1.0))
            score_b_t = float(clamp(float(agg.get("B", {}).get("score", 0.5)), 0.0, 1.0))
        except Exception:
            score_a_t = 0.5
            score_b_t = 0.5
        conf_t = float(clamp(float(teacher_out.get("confidence", 0.0)), 0.0, 1.0))
        winner_t = str(teacher_out.get("winner", "tie"))
        winner_t, conf_t = _coerce_teacher_winner_and_confidence(
            score_a=score_a_t,
            score_b=score_b_t,
            winner=winner_t,
            confidence=conf_t,
            tie_threshold=tie_threshold,
        )

        if hybrid_combine:
            score_a = float(clamp(w_s * score_a_s + w_t * score_a_t, 0.0, 1.0))
            score_b = float(clamp(w_s * score_b_s + w_t * score_b_t, 0.0, 1.0))
            winner = _winner_from_scores(score_a, score_b, tie_threshold)
            confidence = float(clamp(w_s * conf_s + w_t * conf_t, 0.0, 1.0))
            method = "hybrid_v2"
        else:
            score_a = score_a_t
            score_b = score_b_t
            winner = winner_t
            confidence = conf_t
            method = "teacher"

        explanation = teacher_out.get("explanation", "")
        if teacher_out.get("vlm"):
            vlm_data = teacher_out["vlm"]
            
            # Additional values from pipeline
            reasoning_score_val = 0.5
            try:
                from vlmjudge.scorers.reasoning_score import ReasoningScorer
                reasoning_score_val = ReasoningScorer().score_reasoning(vlm_data.get("reason", ""))
            except Exception:
                pass
                
            consistency_flag = vlm_data.get("reasoning_inconsistent", False)
            disag_score = vlm_data.get("disagreement_score", teacher_out.get("disagreement_score", 0.0))
            
            reasoning_entry = {
                "timestamp": float(_now_ms()),
                "prompt": prompt,
                "winner": winner,
                "confidence": confidence,
                "agreement": teacher_out.get("agreement", False),
                "disagreement_score": disag_score,
                "vlm_explanations": vlm_data.get("reasons", [vlm_data.get("reason", "")]),
                "vlm_winner": vlm_data.get("winner", "tie"),
                "consistency_flag": consistency_flag,
                "reasoning_score": reasoning_score_val,
                "final_confidence": confidence
            }
            _append_jsonl(os.path.join("logs", "reasoning.jsonl"), reasoning_entry)
            
            quality_entry = {
                "timestamp": float(_now_ms()),
                "prompt": prompt,
                "reasoning_score": reasoning_score_val,
                "consistency_flag": consistency_flag,
                "confidence": confidence,
                "disagreement_score": disag_score
            }
            _append_jsonl(os.path.join("logs", "reasoning_quality.jsonl"), quality_entry)

        total = float(dt_student + dt_teacher)
        final = {
            "winner": winner,
            "confidence": float(confidence),
            "scoreA": float(score_a),
            "scoreB": float(score_b),
            "method": method,
            "reasoning": explanation,
            "timing_ms": {"student": float(dt_student), "teacher": float(dt_teacher), "total": total},
            "scores": {
                "structured": teacher_out.get("structured", {}),
                "vlm": teacher_out.get("vlm", {})
            }
        }
        if return_debug:
            final["_debug"] = {
                "student_variant": student_variant,
                "student_checkpoint": student_checkpoint,
                "student_winner": winner_s,
                "teacher_winner": winner_t,
                "agreement": bool(winner_s == winner_t),
                "used_teacher": True,
            }
        # Shadow logging: reuse teacher result when we already computed it.
        if do_shadow:
            self._spawn_shadow_compare(
                prompt=prompt,
                image_a_ref=image_a_ref,
                image_b_ref=image_b_ref,
                pil_a=pil_a,
                pil_b=pil_b,
                student_variant=student_variant,
                student_checkpoint=student_checkpoint,
                student_out={"winner": winner_s, "confidence": conf_s, "scoreA": score_a_s, "scoreB": score_b_s},
                method_used=str(method),
                teacher_out={
                    "winner": winner_t,
                    "confidence": conf_t,
                    "scoreA": score_a_t,
                    "scoreB": score_b_t,
                    "timing_ms": float(dt_teacher),
                },
            )
        return final

    def score(self, *, prompt: str, image_ref: str) -> Dict[str, Any]:
        if not prompt or not isinstance(prompt, str):
            return {"score": 0.5, "confidence": 0.0, "timing_ms": 0.0, "_debug": {"error": "missing_prompt"}}

        cache_dir = str(_get(self.cfg, "cache_dir", ".cache/api_images"))
        url_timeout_sec = int(_get(self.cfg, "url_timeout_sec", 10))
        max_download_mb = int(_get(self.cfg, "max_download_mb", 20))

        try:
            pil = resolve_image(
                image_ref, cache_dir=cache_dir, url_timeout_sec=url_timeout_sec, max_download_mb=max_download_mb
            )
        except Exception as e:
            logger.warning("event=image_resolve_failed err=%s", e)
            return {"score": 0.5, "confidence": 0.0, "timing_ms": 0.0, "_debug": {"error": "image_resolve_failed"}}
        student_engine, student_variant = self._choose_student()
        if isinstance(student_engine, MissingStudentEngine):
            return {
                "score": 0.5,
                "confidence": 0.0,
                "timing_ms": 0.0,
                "_debug": {"student_variant": student_variant, "error": "student_checkpoint_missing"},
            }
        try:
            t0 = _now_ms()
            score, confidence = student_engine.score(pil, prompt)
            dt = _now_ms() - t0
            return {
                "score": float(score),
                "confidence": float(confidence),
                "timing_ms": float(dt),
                "_debug": {"student_variant": student_variant},
            }
        except Exception as e:
            logger.warning("event=student_score_failed err=%s", e)
            return {"score": 0.5, "confidence": 0.0, "timing_ms": 0.0, "_debug": {"student_variant": student_variant}}

    def _spawn_shadow_compare(
        self,
        *,
        prompt: str,
        image_a_ref: str,
        image_b_ref: str,
        pil_a: Image.Image,
        pil_b: Image.Image,
        student_variant: str,
        student_checkpoint: str,
        student_out: Dict[str, Any],
        method_used: str,
        teacher_out: Optional[Dict[str, Any]],
    ) -> None:
        if self.teacher is None:
            return

        def _run():
            t0 = time.time()
            teacher_res = teacher_out
            teacher_t_ms = None
            if teacher_res is None:
                try:
                    t1 = _now_ms()
                    tout = self.teacher.compare(pil_a, pil_b, prompt)
                    teacher_t_ms = float(_now_ms() - t1)
                    winner_t = str(tout.get("winner", "tie"))
                    conf_t = float(clamp(float(tout.get("confidence", 0.0)), 0.0, 1.0))
                    agg = tout.get("structured", {}).get("aggregate", {})
                    score_a_t = float(clamp(float(agg.get("A", {}).get("score", 0.5)), 0.0, 1.0))
                    score_b_t = float(clamp(float(agg.get("B", {}).get("score", 0.5)), 0.0, 1.0))
                    teacher_res = {
                        "winner": winner_t,
                        "confidence": conf_t,
                        "scoreA": score_a_t,
                        "scoreB": score_b_t,
                    }
                except Exception as e:
                    logger.warning("event=shadow_teacher_failed err=%s", e)
                    return

            # Shadow student models
            shadow_models_out: List[Dict[str, Any]] = []
            for ckpt, eng in list(self.shadow_students):
                try:
                    sa, sb, sc, sw = eng.compare(pil_a, pil_b, prompt)
                    shadow_models_out.append(
                        {
                            "checkpoint": ckpt,
                            "winner": sw,
                            "confidence": float(sc),
                            "scoreA": float(sa),
                            "scoreB": float(sb),
                        }
                    )
                except Exception:
                    continue

            s_conf = float(student_out.get("confidence", 0.0))
            t_conf = float(teacher_res.get("confidence", 0.0))
            agreement = bool(str(student_out.get("winner", "tie")) == str(teacher_res.get("winner", "tie")))
            gap = abs(s_conf - t_conf)

            entry = {
                "timestamp": float(t0),
                "prompt": prompt,
                "imageA": image_a_ref,
                "imageB": image_b_ref,
                "method_used": method_used,
                "student": {
                    "variant": student_variant,
                    "checkpoint": student_checkpoint,
                    "winner": student_out.get("winner", "tie"),
                    "confidence": s_conf,
                    "scoreA": float(student_out.get("scoreA", 0.5)),
                    "scoreB": float(student_out.get("scoreB", 0.5)),
                },
                "teacher": {
                    "winner": teacher_res.get("winner", "tie"),
                    "confidence": t_conf,
                    "scoreA": float(teacher_res.get("scoreA", 0.5)),
                    "scoreB": float(teacher_res.get("scoreB", 0.5)),
                    "timing_ms": teacher_res.get("timing_ms", teacher_t_ms),
                },
                "agreement": agreement,
                "shadow_agreement": agreement,
                "confidence_gap": float(gap),
                "flags": {
                    "large_gap": bool(gap >= 0.5),
                    "low_conf_disagreement": bool((not agreement) and s_conf < 0.6),
                    "high_conf_disagreement": bool((not agreement) and s_conf >= 0.8),
                },
                "shadow_models": shadow_models_out,
            }
            _append_jsonl(os.path.join("logs", "shadow_eval.jsonl"), entry)

        try:
            th = threading.Thread(target=_run, daemon=True)
            th.start()
        except Exception:
            return
