from __future__ import annotations

import argparse
import csv
import json
import os
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from vlmjudge.bench.advanced import (
    bootstrap_mean,
    bucket_confidence,
    bucket_prompt_length,
    bucket_similarity,
    group_indices,
    indices_to_mask,
    prompt_length_words,
)
from vlmjudge.bench.metrics import summarize
from vlmjudge.bench.plotting import (
    plot_accuracy_vs_confidence,
    plot_accuracy_latency,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_margin_histogram,
)
from vlmjudge.bench.utils import get_teacher_fields, load_preference_dataset
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


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _load_student_checkpoint(model: DistilledRewardModel, checkpoint_path: str, device: str) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        try:
            model.load_state_dict(state["model_state_dict"])
            return
        except Exception:
            # Older checkpoints may have mlp without dropout: remap "mlp.2.*" -> "mlp.3.*"
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
        # Head-only: load into MLP (support with/without dropout).
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


class PairwisePathDataset(Dataset):
    def __init__(self, items: Sequence[Mapping[str, Any]], preprocess: Any) -> None:
        self._items = list(items)
        self._preprocess = preprocess

    def __len__(self) -> int:
        return len(self._items)

    def _load_img(self, path: str) -> torch.Tensor:
        try:
            pil = Image.open(path).convert("RGB")
            return self._preprocess(pil)
        except Exception:
            return torch.zeros(3, 224, 224)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._items[idx]
        prompt = str(item["prompt"])
        chosen = str(item["chosen"])
        rejected = str(item["rejected"])
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "img_chosen": self._load_img(chosen),
            "img_rejected": self._load_img(rejected),
        }


@dataclass(frozen=True)
class SystemResult:
    name: str
    scores_chosen: Optional[List[float]]
    scores_rejected: Optional[List[float]]
    margins: List[float]
    confidences: List[float]
    correct: List[bool]
    ms_total: float


def _pred_from_margin(m: float) -> str:
    return "A" if m > 0.0 else "B"


def _margin_quantiles(margins: Sequence[float]) -> Dict[str, float]:
    if not margins:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
    xs = sorted(float(m) for m in margins)
    n = len(xs)

    def q(p: float) -> float:
        if n == 1:
            return xs[0]
        idx = int(round(p * (n - 1)))
        idx = 0 if idx < 0 else (n - 1 if idx >= n else idx)
        return float(xs[idx])

    return {"p10": q(0.10), "p50": q(0.50), "p90": q(0.90)}


def eval_student(
    items: Sequence[Mapping[str, Any]],
    *,
    checkpoint: str,
    device: str,
    batch_size: int,
) -> Tuple[SystemResult, Dict[str, Any]]:
    model = DistilledRewardModel().to(device)
    _load_student_checkpoint(model, checkpoint, device)
    model.eval()

    ds = PairwisePathDataset(items, model.preprocess)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    scores_chosen: List[float] = []
    scores_rejected: List[float] = []
    margins: List[float] = []
    confidences: List[float] = []
    correct: List[bool] = []

    t0 = _now_ms()
    with torch.no_grad():
        for batch in loader:
            prompts = list(batch["prompt"])
            img_c = batch["img_chosen"].to(device)
            img_r = batch["img_rejected"].to(device)

            logits_c = model(img_c, prompts)
            logits_r = model(img_r, prompts)
            score_c = torch.sigmoid(logits_c)
            score_r = torch.sigmoid(logits_r)

            margin = (score_c - score_r).detach().cpu()
            conf = (score_c - score_r).abs().clamp(0.0, 1.0).detach().cpu()
            sc = score_c.detach().cpu()
            sr = score_r.detach().cpu()

            for i in range(margin.shape[0]):
                m = float(margin[i].item())
                c = float(conf[i].item())
                scores_chosen.append(float(sc[i].item()))
                scores_rejected.append(float(sr[i].item()))
                margins.append(m)
                confidences.append(c)
                correct.append(m > 0.0)
    ms_total = _now_ms() - t0
    summary = summarize(margins=margins, confidences=confidences, correct=correct, n_bins=10)
    extra = {
        "n": summary.n,
        "accuracy": summary.accuracy,
        "ece": summary.ece,
        "mae": summary.mae,
        "avg_confidence": summary.avg_confidence,
        "avg_margin": summary.avg_margin,
        "ms_total": float(ms_total),
        "ms_per_sample": float(ms_total / summary.n) if summary.n else 0.0,
        "margin_quantiles": _margin_quantiles(margins),
    }
    return SystemResult("student", scores_chosen, scores_rejected, margins, confidences, correct, ms_total), extra


class _ImageRewardRaw:
    def __init__(self, *, model_name: str, device: Optional[str], download_root: Optional[str], med_config: Optional[str]) -> None:
        self._model = None
        try:
            import ImageReward as RM
        except Exception:
            self._model = None
            return

        try:
            kwargs = {"download_root": download_root, "med_config": med_config}
            if device is None:
                self._model = RM.load(model_name, **kwargs)
            else:
                self._model = RM.load(model_name, device=device, **kwargs)
        except Exception:
            self._model = None

    def score_raw(self, prompt: str, image: str) -> Optional[float]:
        if self._model is None:
            return None
        try:
            return float(self._model.score(prompt, image))
        except Exception:
            return None


def _build_teacher_pipeline(args) -> ComparePipeline:
    scorers: Dict[str, Any] = {}
    scorers["image_reward"] = ImageRewardScorer(
        ImageRewardScorerConfig(
            model_name=args.model_name,
            device=args.device,
            download_root=args.download_root,
            med_config=args.med_config,
        )
    )
    scorers["openclip"] = OpenCLIPScorer(OpenCLIPScorerConfig(device=args.device))
    scorers["aesthetic"] = AestheticScorer(AestheticScorerConfig(weights_path=args.aesthetic_weights, device=args.device))
    scorers["lpips"] = LPIPSScorer(LPIPSScorerConfig(device=args.device))

    vlm_judge = None
    if args.use_vlm:
        from vlmjudge.vlm.ensemble import VLMEnsemble
        vlm_judge = VLMEnsemble(
            config={"vlm_runs": args.vlm_runs, "vlm_max_new_tokens": args.vlm_max_new_tokens, "device": args.device},
            strict=False,
        )

    return ComparePipeline(
        scorers,
        config=ComparePipelineConfig(threshold=float(args.teacher_threshold), vlm_runs=int(args.vlm_runs)),
        vlm_judge=vlm_judge,
    )


def eval_teacher(items: Sequence[Mapping[str, Any]], *, pipeline: ComparePipeline) -> Tuple[SystemResult, List[Dict[str, Any]], Dict[str, Any]]:
    scores_chosen: List[float] = []
    scores_rejected: List[float] = []
    margins: List[float] = []
    confidences: List[float] = []
    correct: List[bool] = []
    outputs: List[Dict[str, Any]] = []
    reasoning_scores: List[float] = []
    reasoning_inconsistent_count = 0
    vlm_used_count = 0
    
    from vlmjudge.scorers.reasoning_score import ReasoningScorer
    reasoning_scorer = ReasoningScorer()

    t0 = _now_ms()
    for item in items:
        prompt = str(item["prompt"])
        chosen = str(item["chosen"])
        rejected = str(item["rejected"])

        out = pipeline.run(chosen, rejected, prompt)
        outputs.append(out)

        winner = str(out.get("winner", "tie"))
        conf = float(out.get("confidence", 0.0))
        conf = float(clamp(conf, 0.0, 1.0))
        # With chosen as A and rejected as B, A means correct.
        is_correct = bool(winner == "A")
        margin = 0.0
        try:
            margin = float(out.get("structured", {}).get("delta", 0.0))
        except Exception:
            margin = 0.0
        try:
            agg = out.get("structured", {}).get("aggregate", {})
            scores_chosen.append(float(agg.get("A", {}).get("score", 0.5)))
            scores_rejected.append(float(agg.get("B", {}).get("score", 0.5)))
        except Exception:
            scores_chosen.append(0.5)
            scores_rejected.append(0.5)
        margins.append(float(margin))
        confidences.append(conf)
        correct.append(is_correct)
        
        # Calculate reasoning score if VLM reasoning exists
        vlm_data = out.get("vlm", {})
        if vlm_data:
            vlm_used_count += 1
            if vlm_data.get("reasoning_inconsistent", False):
                reasoning_inconsistent_count += 1
            
            vlm_reason = vlm_data.get("reason", "")
            if vlm_reason:
                rs = reasoning_scorer.score_reasoning(vlm_reason)
                reasoning_scores.append(rs)

    ms_total = _now_ms() - t0

    summary = summarize(margins=margins, confidences=confidences, correct=correct, n_bins=10)
    extra = {
        "n": summary.n,
        "accuracy": summary.accuracy,
        "ece": summary.ece,
        "mae": summary.mae,
        "avg_confidence": summary.avg_confidence,
        "avg_margin": summary.avg_margin,
        "ms_total": float(ms_total),
        "ms_per_sample": float(ms_total / summary.n) if summary.n else 0.0,
        "margin_quantiles": _margin_quantiles(margins),
        "avg_reasoning_score": float(sum(reasoning_scores)/len(reasoning_scores)) if reasoning_scores else 0.0,
        "reasoning_consistency_rate": 1.0 - (reasoning_inconsistent_count / max(1, vlm_used_count)),
        "vlm_used_count": vlm_used_count
    }
    
    # Save reasoning report
    try:
        os.makedirs("output", exist_ok=True)
        with open("output/report_reasoning.json", "w", encoding="utf-8") as f:
            json.dump({
                "avg_reasoning_score": extra["avg_reasoning_score"],
                "reasoning_consistency_rate": extra["reasoning_consistency_rate"],
                "vlm_used_count": extra["vlm_used_count"]
            }, f, indent=2)
    except Exception:
        pass

    return SystemResult("teacher", scores_chosen, scores_rejected, margins, confidences, correct, ms_total), outputs, extra


def eval_individual_scorer(items: Sequence[Mapping[str, Any]], *, scorer_name: str, scorer) -> Tuple[SystemResult, Dict[str, Any]]:
    scores_chosen: List[float] = []
    scores_rejected: List[float] = []
    margins: List[float] = []
    confidences: List[float] = []
    correct: List[bool] = []

    t0 = _now_ms()
    for item in items:
        prompt = str(item["prompt"])
        chosen = str(item["chosen"])
        rejected = str(item["rejected"])

        out_c = scorer.score(chosen, prompt=prompt, image_b=None)
        out_r = scorer.score(rejected, prompt=prompt, image_b=None)

        sc = float(out_c.get("score", 0.5))
        sr = float(out_r.get("score", 0.5))
        sc = float(clamp(sc, 0.0, 1.0))
        sr = float(clamp(sr, 0.0, 1.0))
        margin = sc - sr
        conf = float(clamp(abs(margin), 0.0, 1.0))

        scores_chosen.append(sc)
        scores_rejected.append(sr)
        margins.append(float(margin))
        confidences.append(float(conf))
        correct.append(margin > 0.0)

    ms_total = _now_ms() - t0
    summary = summarize(margins=margins, confidences=confidences, correct=correct, n_bins=10)
    extra = {
        "n": summary.n,
        "accuracy": summary.accuracy,
        "ece": summary.ece,
        "mae": summary.mae,
        "avg_confidence": summary.avg_confidence,
        "avg_margin": summary.avg_margin,
        "ms_total": float(ms_total),
        "ms_per_sample": float(ms_total / summary.n) if summary.n else 0.0,
        "margin_quantiles": _margin_quantiles(margins),
    }
    return SystemResult(scorer_name, scores_chosen, scores_rejected, margins, confidences, correct, ms_total), extra


def eval_imagereward_raw(
    items: Sequence[Mapping[str, Any]],
    *,
    model_name: str,
    device: Optional[str],
    download_root: Optional[str],
    med_config: Optional[str],
) -> Tuple[SystemResult, Dict[str, Any]]:
    raw_model = _ImageRewardRaw(model_name=model_name, device=device, download_root=download_root, med_config=med_config)
    scores_chosen: List[float] = []
    scores_rejected: List[float] = []
    margins: List[float] = []
    confidences: List[float] = []
    correct: List[bool] = []

    t0 = _now_ms()
    for item in items:
        prompt = str(item["prompt"])
        chosen = str(item["chosen"])
        rejected = str(item["rejected"])
        rc = raw_model.score_raw(prompt, chosen)
        rr = raw_model.score_raw(prompt, rejected)
        if rc is None or rr is None:
            rc = 0.0
            rr = 0.0
        margin = float(rc - rr)
        scores_chosen.append(float(rc))
        scores_rejected.append(float(rr))
        margins.append(margin)
        # Convert raw margin to a confidence-like signal via sigmoid(|margin|) proxy without extra deps.
        conf = float(clamp(abs(margin) / (1.0 + abs(margin)), 0.0, 1.0))
        confidences.append(conf)
        correct.append(margin > 0.0)
    ms_total = _now_ms() - t0

    summary = summarize(margins=margins, confidences=confidences, correct=correct, n_bins=10)
    extra = {
        "n": summary.n,
        "accuracy": summary.accuracy,
        "ece": summary.ece,
        "mae": summary.mae,
        "avg_confidence": summary.avg_confidence,
        "avg_margin": summary.avg_margin,
        "ms_total": float(ms_total),
        "ms_per_sample": float(ms_total / summary.n) if summary.n else 0.0,
        "margin_quantiles": _margin_quantiles(margins),
    }
    return SystemResult("image_reward_raw", scores_chosen, scores_rejected, margins, confidences, correct, ms_total), extra


def hybrid_accuracy(
    *,
    student: SystemResult,
    teacher: SystemResult,
    threshold: float,
) -> float:
    n = min(len(student.margins), len(student.confidences), len(teacher.correct))
    if n == 0:
        return 0.0
    correct = 0
    for i in range(n):
        use_teacher = float(student.confidences[i]) < float(threshold)
        if use_teacher:
            if bool(teacher.correct[i]):
                correct += 1
        else:
            if bool(student.correct[i]):
                correct += 1
    return float(correct / n)


def evaluate_groups(
    items: Sequence[Mapping[str, Any]],
    *,
    system: SystemResult,
    group_mask: Sequence[bool],
) -> Dict[str, float]:
    idxs = [i for i, ok in enumerate(group_mask) if ok]
    if not idxs:
        return {"n": 0, "accuracy": 0.0}
    correct = sum(1 for i in idxs if bool(system.correct[i]))
    return {"n": int(len(idxs)), "accuracy": float(correct / len(idxs))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6: comprehensive benchmarking for reward models.")
    parser.add_argument("--dataset", required=True, type=str, help="Preference dataset JSON (prompt/chosen/rejected).")
    parser.add_argument("--student-checkpoint", required=True, type=str, help='Student checkpoint ("best.pt" or head state_dict).')
    parser.add_argument("--output-dir", default="bench_out", type=str)
    parser.add_argument("--device", default=None, type=str, help='Device override (e.g. "cpu", "cuda:0").')
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--max-samples", default=None, type=int)

    # Teacher pipeline
    parser.add_argument("--enable-teacher", action="store_true")
    parser.add_argument("--teacher-threshold", default=0.05, type=float)
    parser.add_argument("--model-name", default="ImageReward-v1.0", type=str)
    parser.add_argument("--download-root", default=None, type=str)
    parser.add_argument("--med-config", default=None, type=str)
    parser.add_argument("--aesthetic-weights", default=None, type=str)

    # VLM (teacher fusion)
    parser.add_argument("--use-vlm", action="store_true")
    parser.add_argument("--vlm-runs", default=3, type=int)
    parser.add_argument("--vlm-max-new-tokens", default=192, type=int)

    # Individual scorers
    parser.add_argument("--enable-scorers", action="store_true")
    parser.add_argument("--scorers", default="openclip,imagereward,aesthetic", type=str, help="Comma list.")

    # Hybrid tuning
    parser.add_argument("--hybrid-thresholds", default="0.4,0.5,0.6,0.7", type=str)

    # Outputs
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--bootstrap-samples", default=200, type=int)
    parser.add_argument("--bootstrap-seed", default=123, type=int)
    parser.add_argument("--category-analysis", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_dir(args.output_dir)
    plots_dir = os.path.join(args.output_dir, "plots")

    items = load_preference_dataset(args.dataset, max_samples=args.max_samples)
    if not items:
        raise ValueError("No usable samples found in dataset.")

    # Extract robustness split fields from dataset (not from re-running teacher).
    teacher_meta_conf: List[float] = []
    teacher_meta_cov: List[float] = []
    teacher_meta_delta: List[float] = []
    meta_agreement: List[bool] = []
    for it in items:
        conf, cov, delta = get_teacher_fields(it)
        teacher_meta_conf.append(conf)
        teacher_meta_cov.append(cov)
        teacher_meta_delta.append(delta)
        meta_agreement.append(bool(it.get("agreement", True)))

    # --- Student ---
    student_sys, student_extra = eval_student(
        items,
        checkpoint=args.student_checkpoint,
        device=device,
        batch_size=int(args.batch_size),
    )
    student_pred = [_pred_from_margin(m) for m in student_sys.margins]

    teacher_sys = None
    teacher_outputs: List[Dict[str, Any]] = []
    teacher_extra: Dict[str, Any] = {}

    # --- Teacher ---
    if args.enable_teacher:
        pipeline = _build_teacher_pipeline(args)
        teacher_sys, teacher_outputs, teacher_extra = eval_teacher(items, pipeline=pipeline)

    # --- Individual scorers ---
    scorer_results: Dict[str, Dict[str, Any]] = {}
    scorer_systems: Dict[str, SystemResult] = {}
    if args.enable_scorers:
        names = [s.strip().lower() for s in str(args.scorers).split(",") if s.strip()]
        for name in names:
            if name in ("openclip", "clip"):
                s = OpenCLIPScorer(OpenCLIPScorerConfig(device=args.device))
                sysr, extra = eval_individual_scorer(items, scorer_name="openclip", scorer=s)
            elif name in ("imagereward_raw", "image_reward_raw", "ir_raw"):
                sysr, extra = eval_imagereward_raw(
                    items,
                    model_name=args.model_name,
                    device=args.device,
                    download_root=args.download_root,
                    med_config=args.med_config,
                )
            elif name in ("imagereward", "image_reward"):
                s = ImageRewardScorer(
                    ImageRewardScorerConfig(
                        model_name=args.model_name,
                        device=args.device,
                        download_root=args.download_root,
                        med_config=args.med_config,
                    )
                )
                sysr, extra = eval_individual_scorer(items, scorer_name="image_reward", scorer=s)
            elif name in ("aesthetic",):
                s = AestheticScorer(AestheticScorerConfig(weights_path=args.aesthetic_weights, device=args.device))
                sysr, extra = eval_individual_scorer(items, scorer_name="aesthetic", scorer=s)
            else:
                continue
            scorer_systems[sysr.name] = sysr
            scorer_results[sysr.name] = extra

    # --- Bootstrap significance (accuracy + agreement) ---
    accuracy_ci = None
    agreement_ci = None
    if args.bootstrap and teacher_sys is not None:
        s_acc = [1.0 if b else 0.0 for b in student_sys.correct]
        t_acc = [1.0 if b else 0.0 for b in teacher_sys.correct]

        # Agreement: student winner vs teacher winner (ignore ties as disagreement).
        agree_bools: List[bool] = []
        for i, out in enumerate(teacher_outputs):
            tw = str(out.get("winner", "tie"))
            agree_bools.append(bool(student_pred[i] == tw and tw != "tie"))
        agree_vals = [1.0 if b else 0.0 for b in agree_bools]

        bs_n = int(args.bootstrap_samples)
        bs_seed = int(args.bootstrap_seed)
        s_acc_ci = bootstrap_mean(s_acc, n_bootstrap=bs_n, seed=bs_seed)
        t_acc_ci = bootstrap_mean(t_acc, n_bootstrap=bs_n, seed=bs_seed + 1)
        agree_ci = bootstrap_mean(agree_vals, n_bootstrap=bs_n, seed=bs_seed + 2)
        accuracy_ci = {
            "student": s_acc_ci.__dict__,
            "teacher": t_acc_ci.__dict__,
        }
        agreement_ci = agree_ci.__dict__

    # --- Hybrid threshold tuning ---
    best_thr = None
    best_hybrid_acc = None
    hybrid_accs: Dict[str, float] = {}
    hybrid_meta: Dict[str, Any] = {}
    if teacher_sys is not None:
        thresholds = []
        for tok in str(args.hybrid_thresholds).split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                thresholds.append(float(tok))
            except Exception:
                continue

        for thr in thresholds:
            acc = hybrid_accuracy(student=student_sys, teacher=teacher_sys, threshold=float(thr))
            hybrid_accs[str(thr)] = float(acc)
            if best_hybrid_acc is None or acc > best_hybrid_acc:
                best_hybrid_acc = float(acc)
                best_thr = float(thr)

        if best_thr is not None:
            fallback_n = sum(1 for c in student_sys.confidences if float(c) < float(best_thr))
            fallback_rate = float(fallback_n / max(1, len(student_sys.confidences)))
            student_ms = float(student_extra.get("ms_per_sample", 0.0))
            teacher_ms = float(teacher_extra.get("ms_per_sample", 0.0))
            est_hybrid_ms = float(student_ms + fallback_rate * teacher_ms)
            hybrid_meta = {
                "fallback_rate": fallback_rate,
                "estimated_ms_per_sample": est_hybrid_ms,
                "estimated_speedup_vs_teacher": float(teacher_ms / est_hybrid_ms) if est_hybrid_ms > 1e-9 else 0.0,
            }

    # --- Failures / disagreement logging ---
    failures: List[Dict[str, Any]] = []
    disagreements: List[Dict[str, Any]] = []

    teacher_pred = None
    teacher_conf = None
    teacher_expl = None
    if teacher_sys is not None:
        teacher_pred = []
        teacher_conf = []
        teacher_expl = []
        for out in teacher_outputs:
            teacher_pred.append(str(out.get("winner", "tie")))
            teacher_conf.append(float(clamp(float(out.get("confidence", 0.0)), 0.0, 1.0)))
            teacher_expl.append(str(out.get("explanation", ""))[:512])

    # Agreement metrics for baselines/scorers (external baseline comparison).
    if scorer_systems:
        for name, sysr in scorer_systems.items():
            nn = min(len(sysr.margins), len(student_pred))
            if nn == 0:
                continue
            agree_student = sum(1 for i in range(nn) if _pred_from_margin(sysr.margins[i]) == student_pred[i]) / nn
            scorer_results.setdefault(name, {})
            scorer_results[name]["agreement_with_student"] = float(agree_student)
            if teacher_pred is not None:
                agree_teacher = sum(
                    1 for i in range(nn) if (_pred_from_margin(sysr.margins[i]) == teacher_pred[i] and teacher_pred[i] != "tie")
                ) / nn
                scorer_results[name]["agreement_with_teacher"] = float(agree_teacher)

    n = len(items)
    for i in range(n):
        it = items[i]
        s_corr = bool(student_sys.correct[i])
        s_conf = float(student_sys.confidences[i])
        s_m = float(student_sys.margins[i])
        s_w = student_pred[i]

        t_corr = None
        t_w = None
        t_c = None
        t_e = None
        if teacher_sys is not None and teacher_pred is not None and teacher_conf is not None and teacher_expl is not None:
            t_corr = bool(teacher_sys.correct[i])
            t_w = str(teacher_pred[i])
            t_c = float(teacher_conf[i])
            t_e = str(teacher_expl[i])

        meta = {
            "meta_confidence": teacher_meta_conf[i],
            "meta_coverage": teacher_meta_cov[i],
            "meta_delta": teacher_meta_delta[i],
            "meta_agreement": bool(meta_agreement[i]),
        }

        high_conf_wrong = (not s_corr) and (s_conf >= 0.8)
        low_conf_wrong = (not s_corr) and (s_conf < 0.6)
        student_wrong_teacher_correct = (teacher_sys is not None) and (not s_corr) and bool(t_corr)

        if high_conf_wrong or low_conf_wrong or student_wrong_teacher_correct:
            failures.append(
                {
                    "idx": i,
                    "prompt": it.get("prompt", ""),
                    "chosen": it.get("chosen", ""),
                    "rejected": it.get("rejected", ""),
                    "student": {
                        "winner": s_w,
                        "margin": s_m,
                        "score_chosen": float(student_sys.scores_chosen[i]) if student_sys.scores_chosen is not None else None,
                        "score_rejected": float(student_sys.scores_rejected[i]) if student_sys.scores_rejected is not None else None,
                        "confidence": s_conf,
                        "correct": s_corr,
                    },
                    "teacher": {"winner": t_w, "confidence": t_c, "correct": t_corr, "explanation": t_e},
                    "reason": (
                        "student_wrong_teacher_correct"
                        if student_wrong_teacher_correct
                        else ("high_conf_wrong" if high_conf_wrong else "low_conf_wrong")
                    ),
                    **meta,
                }
            )

        if teacher_sys is not None and t_w is not None:
            if s_w != t_w and t_w != "tie":
                disagreements.append(
                    {
                        "idx": i,
                        "prompt": it.get("prompt", ""),
                        "chosen": it.get("chosen", ""),
                        "rejected": it.get("rejected", ""),
                        "student_winner": s_w,
                        "student_confidence": s_conf,
                        "teacher_winner": t_w,
                        "teacher_confidence": t_c,
                        "meta_confidence": teacher_meta_conf[i],
                        "meta_coverage": teacher_meta_cov[i],
                        "meta_delta": teacher_meta_delta[i],
                        "meta_agreement": bool(meta_agreement[i]),
                    }
                )

    failures_path = os.path.join(args.output_dir, "failures.json")
    Path(failures_path).write_text(json.dumps(failures, indent=2), encoding="utf-8")

    disagree_path = os.path.join(args.output_dir, "disagreements.json")
    Path(disagree_path).write_text(json.dumps(disagreements, indent=2), encoding="utf-8")

    # Simple disagreement analysis summary (no heavy deps).
    if teacher_sys is not None:
        low_delta = sum(1 for d in disagreements if float(d.get("meta_delta", 0.0)) < 0.1)
        low_cov = sum(1 for d in disagreements if float(d.get("meta_coverage", 0.0)) < 0.5)
        by_prefix: Dict[str, int] = {}
        for d in disagreements:
            p = str(d.get("prompt", "")).strip().lower()
            prefix = " ".join(p.split()[:6]) if p else ""
            by_prefix[prefix] = by_prefix.get(prefix, 0) + 1
        top_prefixes = sorted(by_prefix.items(), key=lambda kv: kv[1], reverse=True)[:25]
        summary = {
            "total": len(disagreements),
            "low_delta_count": int(low_delta),
            "low_coverage_count": int(low_cov),
            "top_prompt_prefixes": [{"prefix": k, "count": int(v)} for k, v in top_prefixes if k],
        }
        Path(os.path.join(args.output_dir, "disagreement_analysis.json")).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    # --- Robustness splits (using dataset meta) ---
    high_conf_mask = [c >= 0.7 for c in teacher_meta_conf]
    low_conf_mask = [c < 0.7 for c in teacher_meta_conf]
    high_cov_mask = [c >= 0.5 for c in teacher_meta_cov]
    low_cov_mask = [c < 0.5 for c in teacher_meta_cov]
    agree_mask = [bool(a) for a in meta_agreement]
    disagree_mask = [not bool(a) for a in meta_agreement]

    robustness: Dict[str, Any] = {"student": {}, "teacher": {}, "hybrid": {}}
    robustness["student"]["high_conf"] = evaluate_groups(items, system=student_sys, group_mask=high_conf_mask)
    robustness["student"]["low_conf"] = evaluate_groups(items, system=student_sys, group_mask=low_conf_mask)
    robustness["student"]["high_cov"] = evaluate_groups(items, system=student_sys, group_mask=high_cov_mask)
    robustness["student"]["low_cov"] = evaluate_groups(items, system=student_sys, group_mask=low_cov_mask)
    robustness["student"]["agree"] = evaluate_groups(items, system=student_sys, group_mask=agree_mask)
    robustness["student"]["disagree"] = evaluate_groups(items, system=student_sys, group_mask=disagree_mask)

    if teacher_sys is not None:
        robustness["teacher"]["high_conf"] = evaluate_groups(items, system=teacher_sys, group_mask=high_conf_mask)
        robustness["teacher"]["low_conf"] = evaluate_groups(items, system=teacher_sys, group_mask=low_conf_mask)
        robustness["teacher"]["high_cov"] = evaluate_groups(items, system=teacher_sys, group_mask=high_cov_mask)
        robustness["teacher"]["low_cov"] = evaluate_groups(items, system=teacher_sys, group_mask=low_cov_mask)
        robustness["teacher"]["agree"] = evaluate_groups(items, system=teacher_sys, group_mask=agree_mask)
        robustness["teacher"]["disagree"] = evaluate_groups(items, system=teacher_sys, group_mask=disagree_mask)

        if best_thr is not None:
            # Approximate hybrid robustness using teacher correctness vs student based on confidence.
            hybrid_correct: List[bool] = []
            for i in range(n):
                use_teacher = float(student_sys.confidences[i]) < float(best_thr)
                hybrid_correct.append(bool(teacher_sys.correct[i] if use_teacher else student_sys.correct[i]))
            hybrid_sys = SystemResult(
                name="hybrid",
                scores_chosen=None,
                scores_rejected=None,
                margins=list(student_sys.margins),
                confidences=list(student_sys.confidences),
                correct=hybrid_correct,
                ms_total=float(student_sys.ms_total + teacher_sys.ms_total),
            )
            robustness["hybrid"]["high_conf"] = evaluate_groups(items, system=hybrid_sys, group_mask=high_conf_mask)
            robustness["hybrid"]["low_conf"] = evaluate_groups(items, system=hybrid_sys, group_mask=low_conf_mask)
            robustness["hybrid"]["high_cov"] = evaluate_groups(items, system=hybrid_sys, group_mask=high_cov_mask)
            robustness["hybrid"]["low_cov"] = evaluate_groups(items, system=hybrid_sys, group_mask=low_cov_mask)
            robustness["hybrid"]["agree"] = evaluate_groups(items, system=hybrid_sys, group_mask=agree_mask)
            robustness["hybrid"]["disagree"] = evaluate_groups(items, system=hybrid_sys, group_mask=disagree_mask)

    Path(os.path.join(args.output_dir, "robustness.json")).write_text(json.dumps(robustness, indent=2), encoding="utf-8")

    # --- Failure prioritization (top 100) ---
    prioritized: List[Dict[str, Any]] = []
    for i in range(n):
        err = 0 if bool(student_sys.correct[i]) else 1
        pr = float(student_sys.confidences[i]) * float(err)
        if err:
            prioritized.append(
                {
                    "idx": i,
                    "priority": pr,
                    "prompt": items[i].get("prompt", ""),
                    "chosen": items[i].get("chosen", ""),
                    "rejected": items[i].get("rejected", ""),
                    "student_confidence": float(student_sys.confidences[i]),
                    "student_margin": float(student_sys.margins[i]),
                    "meta_confidence": teacher_meta_conf[i],
                    "meta_coverage": teacher_meta_cov[i],
                    "meta_delta": teacher_meta_delta[i],
                    "meta_agreement": bool(meta_agreement[i]),
                }
            )
    prioritized.sort(key=lambda x: float(x.get("priority", 0.0)), reverse=True)
    top_failures = prioritized[:100]
    Path(os.path.join(args.output_dir, "top_failures.json")).write_text(json.dumps(top_failures, indent=2), encoding="utf-8")

    # --- Category analysis (prompt length, confidence buckets, similarity buckets) ---
    category_analysis: Dict[str, Any] = {}
    similarity_scores: List[Optional[float]] = [None] * n
    if args.category_analysis:
        # Similarity via LPIPS when available.
        lpips = LPIPSScorer(LPIPSScorerConfig(device=args.device))
        for i in range(n):
            out = lpips.score(items[i]["chosen"], prompt=None, image_b=items[i]["rejected"])
            if float(out.get("confidence", 0.0)) <= 0.0:
                similarity_scores[i] = None
            else:
                similarity_scores[i] = float(clamp(float(out.get("score", 0.5)), 0.0, 1.0))

        prompt_labels = [bucket_prompt_length(prompt_length_words(str(it.get("prompt", "")))) for it in items]
        conf_labels = [bucket_confidence(c) for c in teacher_meta_conf]
        sim_labels = [bucket_similarity(similarity_scores[i]) for i in range(n)]

        def _summarize_group(sysr: SystemResult, idxs: List[int]) -> Dict[str, Any]:
            if not idxs:
                return {"n": 0, "accuracy": 0.0, "ece": 0.0, "mae": 0.0}
            margins = [sysr.margins[i] for i in idxs]
            confs = [sysr.confidences[i] for i in idxs]
            corr = [sysr.correct[i] for i in idxs]
            s = summarize(margins=margins, confidences=confs, correct=corr, n_bins=10)
            return {"n": s.n, "accuracy": s.accuracy, "ece": s.ece, "mae": s.mae}

        systems: Dict[str, SystemResult] = {"student": student_sys}
        if teacher_sys is not None:
            systems["teacher"] = teacher_sys

        category_analysis["prompt_length"] = {}
        for lab, idxs in group_indices(prompt_labels).items():
            category_analysis["prompt_length"][lab] = {k: _summarize_group(v, idxs) for k, v in systems.items()}

        category_analysis["confidence_bucket"] = {}
        for lab, idxs in group_indices(conf_labels).items():
            category_analysis["confidence_bucket"][lab] = {k: _summarize_group(v, idxs) for k, v in systems.items()}

        category_analysis["image_similarity"] = {}
        for lab, idxs in group_indices(sim_labels).items():
            category_analysis["image_similarity"][lab] = {k: _summarize_group(v, idxs) for k, v in systems.items()}

        Path(os.path.join(args.output_dir, "category_analysis.json")).write_text(
            json.dumps(category_analysis, indent=2), encoding="utf-8"
        )

        # Disagreement clusters (low delta / low coverage / high similarity)
        if teacher_sys is not None:
            clusters: Dict[str, List[Dict[str, Any]]] = {"low_delta": [], "low_coverage": [], "high_similarity": [], "other": []}
            for d in disagreements:
                i = int(d.get("idx", -1))
                if i < 0 or i >= n:
                    continue
                low_delta = float(d.get("meta_delta", 0.0)) < 0.1
                low_cov = float(d.get("meta_coverage", 0.0)) < 0.5
                sim = similarity_scores[i]
                high_sim = (sim is not None) and (float(sim) >= 0.8)
                entry = {**d, "similarity": sim}
                if low_delta:
                    clusters["low_delta"].append(entry)
                elif low_cov:
                    clusters["low_coverage"].append(entry)
                elif high_sim:
                    clusters["high_similarity"].append(entry)
                else:
                    clusters["other"].append(entry)
            out = {k: {"count": len(v), "examples": v[:50]} for k, v in clusters.items()}
            Path(os.path.join(args.output_dir, "disagreement_clusters.json")).write_text(
                json.dumps(out, indent=2), encoding="utf-8"
            )

    # --- Error breakdown ---
    high_err = sum(1 for i in range(n) if (student_sys.confidences[i] >= 0.7 and not student_sys.correct[i]))
    low_err = sum(1 for i in range(n) if (student_sys.confidences[i] < 0.4 and not student_sys.correct[i]))
    disagree_err = 0
    if teacher_sys is not None:
        # Disagreement errors: student wrong on samples where student != teacher.
        disagree_idxs = set(int(d.get("idx", -1)) for d in disagreements)
        disagree_err = sum(1 for i in disagree_idxs if (0 <= i < n and not student_sys.correct[i]))
    error_breakdown = {
        "high_confidence_error_rate": float(high_err / max(1, n)),
        "low_confidence_error_rate": float(low_err / max(1, n)),
        "disagreement_error_rate": float(disagree_err / max(1, n)),
    }

    # --- Save CSV (optional) ---
    if args.save_csv:
        csv_path = os.path.join(args.output_dir, "results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "idx",
                    "prompt",
                    "chosen",
                    "rejected",
                    "student_margin",
                    "student_score_chosen",
                    "student_score_rejected",
                    "student_confidence",
                    "student_correct",
                    "teacher_winner",
                    "teacher_score_chosen",
                    "teacher_score_rejected",
                    "teacher_confidence",
                    "teacher_correct",
                    "meta_confidence",
                    "meta_coverage",
                    "meta_delta",
                    "meta_agreement",
                ]
            )
            for i in range(n):
                w.writerow(
                    [
                        i,
                        items[i].get("prompt", ""),
                        items[i].get("chosen", ""),
                        items[i].get("rejected", ""),
                        float(student_sys.margins[i]),
                        float(student_sys.scores_chosen[i]) if student_sys.scores_chosen is not None else "",
                        float(student_sys.scores_rejected[i]) if student_sys.scores_rejected is not None else "",
                        float(student_sys.confidences[i]),
                        int(bool(student_sys.correct[i])),
                        (teacher_pred[i] if teacher_pred is not None else ""),
                        (float(teacher_sys.scores_chosen[i]) if (teacher_sys is not None and teacher_sys.scores_chosen is not None) else ""),
                        (float(teacher_sys.scores_rejected[i]) if (teacher_sys is not None and teacher_sys.scores_rejected is not None) else ""),
                        (teacher_conf[i] if teacher_conf is not None else ""),
                        (int(bool(teacher_sys.correct[i])) if teacher_sys is not None else ""),
                        teacher_meta_conf[i],
                        teacher_meta_cov[i],
                        teacher_meta_delta[i],
                        int(bool(meta_agreement[i])),
                    ]
                )

    # --- Plots (matplotlib only) ---
    if args.plots:
        _ensure_dir(plots_dir)
        plot_accuracy_vs_confidence(
            plots_dir,
            confidences=student_sys.confidences,
            correct=student_sys.correct,
            title="Student: accuracy vs confidence",
            filename="student_accuracy_vs_confidence.png",
        )
        plot_calibration_curve(
            plots_dir,
            confidences=student_sys.confidences,
            correct=student_sys.correct,
            title="Student: calibration curve",
            filename="student_calibration_curve.png",
        )
        plot_calibration_curve(
            plots_dir,
            confidences=student_sys.confidences,
            correct=student_sys.correct,
            title="Student: reliability diagram",
            filename="reliability_curve.png",
        )
        plot_margin_histogram(
            plots_dir,
            margins=student_sys.margins,
            title="Student: win margin histogram",
            filename="student_margin_hist.png",
        )
        plot_confusion_matrix(
            plots_dir,
            margins=student_sys.margins,
            title="Student: confusion matrix",
            filename="student_confusion_matrix.png",
        )

        if teacher_sys is not None:
            plot_accuracy_vs_confidence(
                plots_dir,
                confidences=teacher_sys.confidences,
                correct=teacher_sys.correct,
                title="Teacher: accuracy vs confidence",
                filename="teacher_accuracy_vs_confidence.png",
            )
            plot_calibration_curve(
                plots_dir,
                confidences=teacher_sys.confidences,
                correct=teacher_sys.correct,
                title="Teacher: calibration curve",
                filename="teacher_calibration_curve.png",
            )
            plot_margin_histogram(
                plots_dir,
                margins=teacher_sys.margins,
                title="Teacher: win margin histogram",
                filename="teacher_margin_hist.png",
            )
            # Accuracy vs latency curve
            labels = ["student", "teacher"]
            accs = [float(student_extra.get("accuracy", 0.0)), float(teacher_extra.get("accuracy", 0.0))]
            lats = [float(student_extra.get("ms_per_sample", 0.0)), float(teacher_extra.get("ms_per_sample", 0.0))]
            if best_thr is not None and best_hybrid_acc is not None:
                labels.append("hybrid")
                accs.append(float(best_hybrid_acc))
                lats.append(float(hybrid_meta.get("estimated_ms_per_sample", 0.0)) if hybrid_meta else 0.0)
            plot_accuracy_latency(
                plots_dir,
                labels=labels,
                accuracies=accs,
                latencies_ms=lats,
                title="Accuracy vs Latency",
                filename="accuracy_latency.png",
            )

    # --- Write report JSON ---
    replacement_score = None
    replacement_readiness = None
    if teacher_sys is not None and float(teacher_extra.get("accuracy", 0.0)) > 1e-12:
        replacement_score = float(student_extra.get("accuracy", 0.0)) / float(teacher_extra.get("accuracy", 1.0))
        replacement_readiness = float(clamp(replacement_score, 0.0, 1.5) * 100.0)
    report: Dict[str, Any] = {
        "student": student_extra,
        "teacher": teacher_extra if teacher_sys is not None else None,
        "scorers": scorer_results,
        "hybrid": {
            "threshold_acc": hybrid_accs,
            "best_threshold": best_thr,
            "best_accuracy": best_hybrid_acc,
            **hybrid_meta,
        },
        "accuracy_ci": accuracy_ci,
        "agreement_ci": agreement_ci,
        "replacement_score": replacement_score,
        "replacement_readiness_pct": replacement_readiness,
        "category_analysis": category_analysis if category_analysis else None,
        "top_failures": top_failures,
        "latency_analysis": {
            "student_ms_per_sample": float(student_extra.get("ms_per_sample", 0.0)),
            "teacher_ms_per_sample": float(teacher_extra.get("ms_per_sample", 0.0)) if teacher_sys is not None else None,
            "hybrid_estimated_ms_per_sample": float(hybrid_meta.get("estimated_ms_per_sample", 0.0)) if hybrid_meta else None,
        },
        "error_breakdown": error_breakdown,
        "failures_path": failures_path,
        "disagreements_path": disagree_path,
    }
    Path(os.path.join(args.output_dir, "report.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- Write leaderboard ---
    leaderboard = {
        "student": {
            "accuracy": float(student_extra.get("accuracy", 0.0)),
            "ms_per_sample": float(student_extra.get("ms_per_sample", 0.0))
        }
    }
    if teacher_sys is not None:
        leaderboard["teacher (VLM-only)"] = {
            "accuracy": float(teacher_extra.get("accuracy", 0.0)),
            "ms_per_sample": float(teacher_extra.get("ms_per_sample", 0.0)),
            "reasoning_consistency_rate": float(teacher_extra.get("reasoning_consistency_rate", 0.0)),
            "avg_reasoning_score": float(teacher_extra.get("avg_reasoning_score", 0.0))
        }
    if best_hybrid_acc is not None:
        leaderboard["hybrid (student+VLM)"] = {
            "accuracy": float(best_hybrid_acc),
            "ms_per_sample": float(hybrid_meta.get("estimated_ms_per_sample", 0.0)) if hybrid_meta else 0.0
        }
        
    try:
        os.makedirs("analysis", exist_ok=True)
        with open("analysis/model_leaderboard.json", "w", encoding="utf-8") as f:
            json.dump(leaderboard, f, indent=2)
    except Exception:
        pass

    # --- Summary report (paper-style) ---
    print("=== FINAL REPORT ===")
    print(f"Samples: {len(items)}")
    print(f"Student Accuracy: {student_extra['accuracy']:.4f}")
    if teacher_sys is not None:
        print(f"Teacher Accuracy: {teacher_extra['accuracy']:.4f}")
        agree = 1.0 - (len(disagreements) / max(1, len(items)))
        print(f"Teacher Agreement: {agree:.4f}")
        if accuracy_ci and agreement_ci:
            s = accuracy_ci["student"]
            a = agreement_ci
            print(f"Student Accuracy (bootstrap): {s['mean']:.4f} ± {s['std']:.4f}")
            print(f"Agreement (bootstrap): {a['mean']:.4f} ± {a['std']:.4f}")
    if best_thr is not None and best_hybrid_acc is not None:
        print(f"Hybrid Accuracy: {best_hybrid_acc:.4f}")
        print(f"Best Threshold: {best_thr:.2f}")
        if hybrid_meta:
            print(f"Hybrid fallback rate: {hybrid_meta.get('fallback_rate', 0.0):.3f}")
    print(f"Student ECE: {student_extra['ece']:.4f} | MAE: {student_extra['mae']:.4f}")
    if teacher_sys is not None:
        print(f"Teacher ECE: {teacher_extra.get('ece', 0.0):.4f} | MAE: {teacher_extra.get('mae', 0.0):.4f}")
    if teacher_sys is not None:
        sp = float(teacher_extra.get("ms_per_sample", 0.0)) / max(float(student_extra.get("ms_per_sample", 1e-9)), 1e-9)
        print(f"Inference ms/sample: student={student_extra['ms_per_sample']:.2f} teacher={teacher_extra['ms_per_sample']:.2f} speedup={sp:.2f}x")
        if replacement_score is not None:
            readiness = float(replacement_score) * 100.0
            tag = "near replacement" if replacement_score >= 0.9 else "not ready"
            if replacement_score >= 0.95:
                tag = "production ready"
            print(f"Replacement readiness: {readiness:.1f}% ({tag})")
    print(f"High-confidence errors: {error_breakdown['high_confidence_error_rate']*100:.2f}%")
    print(f"Uncertain cases (low-conf errors): {error_breakdown['low_confidence_error_rate']*100:.2f}%")
    if teacher_sys is not None:
        print(f"Disagreement errors: {error_breakdown['disagreement_error_rate']*100:.2f}%")
    print(f"Failures saved: {failures_path}")
    print(f"Disagreements saved: {disagree_path}")
    if args.plots:
        print(f"Plots: {plots_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
