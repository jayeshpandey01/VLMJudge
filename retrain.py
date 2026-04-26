from __future__ import annotations

import argparse
import json
import os
import re
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from data_engine.builder import build_preferences
from data_engine.merge import merge_datasets
from data_engine.qc import quality_filter
from data_engine.selector import SelectionConfig, select_samples
from train_reward_model import DistilledRewardModel, PreferenceDataset, _train_val_split, evaluate, main as train_main


def _read_json(path: str) -> Any:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8-sig"))


def _write_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _next_version(models_dir: str) -> str:
    base = Path(models_dir)
    base.mkdir(parents=True, exist_ok=True)
    vs = []
    for p in base.iterdir():
        if p.is_dir() and re.match(r"^v\d+$", p.name):
            try:
                vs.append(int(p.name[1:]))
            except Exception:
                pass
    n = max(vs) + 1 if vs else 1
    return f"v{n}"


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _write_yaml(path: str, obj: Dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except Exception:
        # Best-effort: only update student_checkpoint via regex.
        return
    Path(path).write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def _get_student_checkpoint_from_config(config_path: str) -> Optional[str]:
    cfg = _load_yaml(config_path)
    cp = cfg.get("student_checkpoint", None)
    if isinstance(cp, str) and cp.strip():
        return cp.strip()
    return None


def _set_student_checkpoint_in_config(config_path: str, new_checkpoint: str) -> None:
    cfg = _load_yaml(config_path)
    if cfg:
        cfg["student_checkpoint"] = str(new_checkpoint)
        _write_yaml(config_path, cfg)
        return
    # Fallback if YAML unavailable: line-replace.
    p = Path(config_path)
    if not p.exists():
        return
    text = p.read_text(encoding="utf-8")
    if "student_checkpoint:" in text:
        text = re.sub(r"^student_checkpoint:\s*.*$", f"student_checkpoint: {new_checkpoint}", text, flags=re.M)
        p.write_text(text, encoding="utf-8")


def _set_canary_checkpoint_in_config(config_path: str, new_checkpoint: str) -> None:
    cfg = _load_yaml(config_path)
    if cfg:
        cfg["canary_checkpoint"] = str(new_checkpoint)
        _write_yaml(config_path, cfg)
        return
    p = Path(config_path)
    if not p.exists():
        return
    text = p.read_text(encoding="utf-8")
    if "canary_checkpoint:" in text:
        text = re.sub(r"^canary_checkpoint:\s*.*$", f"canary_checkpoint: {new_checkpoint}", text, flags=re.M)
    else:
        text = text + f"\ncanary_checkpoint: {new_checkpoint}\n"
    p.write_text(text, encoding="utf-8")


def _evaluate_checkpoint(
    checkpoint_path: str,
    dataset_path: str,
    *,
    split_indices: Optional[Dict[str, Any]] = None,
    device: str,
) -> Dict[str, Any]:
    model = DistilledRewardModel().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        # head-only
        if isinstance(state, dict):
            try:
                model.mlp.load_state_dict(state)
            except Exception:
                # non-dropout -> dropout mapping
                remapped = {}
                for k, v in state.items():
                    if k.startswith("2."):
                        remapped["3." + k[len("2."):]] = v
                    else:
                        remapped[k] = v
                model.mlp.load_state_dict(remapped)
        else:
            raise ValueError("unsupported checkpoint")
    model.eval()

    dataset = PreferenceDataset(dataset_path, model.preprocess)
    if split_indices and isinstance(split_indices.get("val_idx", None), list):
        val_idx = split_indices["val_idx"]
    else:
        _, val_idx = _train_val_split(len(dataset), 0.2, 42)
    from torch.utils.data import DataLoader, Subset

    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=16, shuffle=False, num_workers=0)
    metrics = evaluate(model, val_loader, device)
    return metrics


def _sha256_file(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_audit(event: Dict[str, Any]) -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    p = Path("logs") / "audit.jsonl"
    event = dict(event)
    event["timestamp"] = float(event.get("timestamp", time.time()))
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _load_registry(models_dir: str) -> Dict[str, Any]:
    p = Path(models_dir) / "registry.json"
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _build_teacher_pipeline(cfg: Dict[str, Any], *, device: str):
    tcfg = cfg.get("teacher", {}) if isinstance(cfg.get("teacher", {}), dict) else {}
    teacher_threshold = float(tcfg.get("teacher_threshold", 0.05))
    model_name = str(tcfg.get("model_name", "ImageReward-v1.0"))
    download_root = tcfg.get("download_root", None)
    med_config = tcfg.get("med_config", None)
    aesthetic_weights = tcfg.get("aesthetic_weights", None)

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

    scorers: Dict[str, Any] = {}
    scorers["image_reward"] = ImageRewardScorer(
        ImageRewardScorerConfig(model_name=model_name, device=device, download_root=download_root, med_config=med_config)
    )
    scorers["openclip"] = OpenCLIPScorer(OpenCLIPScorerConfig(device=device))
    scorers["aesthetic"] = AestheticScorer(AestheticScorerConfig(weights_path=aesthetic_weights, device=device))
    scorers["lpips"] = LPIPSScorer(LPIPSScorerConfig(device=device))

    vlm_judge = None
    if bool(tcfg.get("use_vlm", False)):
        from vlmjudge.vlm import QwenJudge
        from vlmjudge.vlm.qwen_judge import QwenJudgeConfig

        vlm_judge = QwenJudge(
            device=device,
            config=QwenJudgeConfig(
                runs=int(tcfg.get("vlm_runs", 3)),
                max_new_tokens=int(tcfg.get("vlm_max_new_tokens", 192)),
            ),
            strict=False,
        )

    return ComparePipeline(
        scorers,
        config=ComparePipelineConfig(threshold=float(teacher_threshold), vlm_runs=int(tcfg.get("vlm_runs", 3))),
        vlm_judge=vlm_judge,
    )


def _agreement_with_teacher(
    *,
    checkpoint_path: str,
    dataset_items: List[Dict[str, Any]],
    val_idx: List[int],
    teacher_pipeline,
    device: str,
) -> float:
    model = DistilledRewardModel().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict):
        try:
            model.mlp.load_state_dict(state)
        except Exception:
            remapped = {}
            for k, v in state.items():
                if k.startswith("2."):
                    remapped["3." + k[len("2."):]] = v
                else:
                    remapped[k] = v
            model.mlp.load_state_dict(remapped)
    else:
        raise ValueError("unsupported checkpoint")
    model.eval()

    agree = 0
    total = 0
    with torch.no_grad():
        for i in val_idx:
            it = dataset_items[i]
            prompt = str(it.get("prompt", ""))
            chosen = str(it.get("chosen", ""))
            rejected = str(it.get("rejected", ""))
            if not prompt or not chosen or not rejected:
                continue

            # Student winner
            from PIL import Image

            pil_c = Image.open(chosen).convert("RGB")
            pil_r = Image.open(rejected).convert("RGB")
            img = torch.stack([model.preprocess(pil_c), model.preprocess(pil_r)], dim=0).to(device)
            tf = model.tokenizer([prompt]).to(device)
            imf = model.clip.encode_image(img)
            txf = model.clip.encode_text(tf).repeat(2, 1)
            imf = imf / imf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            txf = txf / txf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            logits = model.mlp(torch.cat([imf, txf], dim=-1)).squeeze(-1)
            scores = torch.sigmoid(logits).detach().cpu().tolist()
            s_w = "A" if float(scores[0]) > float(scores[1]) else "B"

            # Teacher winner (chosen is A, rejected is B)
            tout = teacher_pipeline.run(chosen, rejected, prompt)
            t_w = str(tout.get("winner", "tie"))
            if t_w == "tie":
                continue
            total += 1
            if s_w == t_w:
                agree += 1
    return float(agree / total) if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 8: continuous learning retrain loop.")
    parser.add_argument("--base-dataset", required=True, type=str, help="Existing preference dataset JSON.")
    parser.add_argument("--config", default="config.yaml", type=str, help="API config.yaml (for current checkpoint update).")
    parser.add_argument("--models-dir", default="models", type=str)
    parser.add_argument("--min-new-samples", default=50, type=int)
    parser.add_argument("--min-total-samples", default=500, type=int)
    parser.add_argument("--min-quality", default="medium", type=str)
    parser.add_argument("--min-coverage", default=0.0, type=float)
    parser.add_argument("--max-total", default=None, type=int)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--loop", action="store_true", help="Run periodically.")
    parser.add_argument("--interval-hours", default=24, type=float)
    parser.add_argument("--min-hours-between-training", default=12, type=float)
    parser.add_argument("--semantic-threshold", default=0.95, type=float)

    # Selection config
    parser.add_argument("--low-conf", default=0.6, type=float)
    parser.add_argument("--high-conf", default=0.8, type=float)
    parser.add_argument("--max-selected", default=5000, type=int)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _run_once() -> None:
        # Retraining cooldown
        registry = _load_registry(args.models_dir)
        history = registry.get("history", [])
        last_ts = None
        if isinstance(history, list) and history:
            try:
                last_ts = float(history[-1].get("timestamp", None))
            except Exception:
                last_ts = None
        if last_ts is not None:
            hours_since = (time.time() - last_ts) / 3600.0
            if hours_since < float(args.min_hours_between_training):
                _append_audit(
                    {
                        "event": "retrain_skipped_cooldown",
                        "models_dir": args.models_dir,
                        "hours_since_last": hours_since,
                        "cooldown_hours": float(args.min_hours_between_training),
                    }
                )
                print(f"Cooldown active: {hours_since:.2f}h < {float(args.min_hours_between_training):.2f}h")
                return

        _append_audit({"event": "retrain_start", "base_dataset": args.base_dataset})
        base = _read_json(args.base_dataset)
        if not isinstance(base, list):
            raise ValueError("--base-dataset must be a JSON list")

        selected = select_samples(
            requests_path=os.path.join("logs", "requests.jsonl"),
            feedback_path=os.path.join("logs", "feedback.jsonl"),
            config=SelectionConfig(
                low_conf_threshold=float(args.low_conf),
                high_conf_threshold=float(args.high_conf),
                max_samples=int(args.max_selected) if args.max_selected else None,
            ),
        )
        built = build_preferences(selected, source="api")
        filtered = quality_filter(built, min_quality=str(args.min_quality), min_coverage=float(args.min_coverage))

        if len(filtered) < int(args.min_new_samples):
            print(f"Not enough new samples after QC: {len(filtered)} < {int(args.min_new_samples)}")
            _append_audit({"event": "retrain_abort", "reason": "not_enough_new_samples", "new_samples": len(filtered)})
            return

        merged, stats = merge_datasets(
            base,
            filtered,
            max_total=args.max_total,
            max_new_ratio=0.5,
            seed=42,
            semantic_dedup=True,
            semantic_threshold=float(args.semantic_threshold),
        )
        if len(merged) < int(args.min_total_samples):
            print(f"Merged dataset too small: {len(merged)} < {int(args.min_total_samples)}")
            _append_audit({"event": "retrain_abort", "reason": "merged_too_small", "merged": len(merged)})
            return

        version = _next_version(args.models_dir)
        version_dir = Path(args.models_dir) / version
        version_dir.mkdir(parents=True, exist_ok=True)

        merged_path = str(version_dir / "dataset_merged.json")
        _write_json(merged_path, merged)
        _write_json(str(version_dir / "selection_stats.json"), {"selected": len(selected), "built": len(built), "filtered": len(filtered), **stats})

        # Data snapshot (reproducibility)
        snapshot = {
            "timestamp": time.time(),
            "base_dataset": args.base_dataset,
            "merged_dataset": merged_path,
            "base_sha256": _sha256_file(args.base_dataset),
            "merged_sha256": _sha256_file(merged_path),
            "requests_sha256": _sha256_file(os.path.join("logs", "requests.jsonl")),
            "feedback_sha256": _sha256_file(os.path.join("logs", "feedback.jsonl")),
            "flagged_feedback_sha256": _sha256_file(os.path.join("logs", "flagged_feedback.jsonl")),
            "n_base": len(base),
            "n_selected": len(selected),
            "n_new_filtered": len(filtered),
            "n_merged": len(merged),
        }
        _write_json(str(version_dir / "data_snapshot.json"), snapshot)

        # Train new model into version dir.
        print(f"Training {version} on {len(merged)} samples...")
        train_main(merged_path, output_dir=str(version_dir), epochs=5, batch_size=16)

        best_new = str(version_dir / "best.pt")
        split_path = version_dir / "split_indices.json"
        split_indices = json.loads(split_path.read_text(encoding="utf-8")) if split_path.exists() else None
        new_metrics = _evaluate_checkpoint(best_new, merged_path, split_indices=split_indices, device=device)
        _write_json(str(version_dir / "metrics.json"), new_metrics)

        current_ckpt = _get_student_checkpoint_from_config(args.config)
        current_metrics = None
        if current_ckpt and Path(current_ckpt).exists():
            try:
                current_metrics = _evaluate_checkpoint(current_ckpt, merged_path, split_indices=split_indices, device=device)
            except Exception as e:
                print(f"Failed to evaluate current checkpoint: {e}")

        # Safe promotion rule:
        # new_acc > old_acc AND new_calibration_error <= old_calibration_error AND new_agreement >= old_agreement
        cfg = _load_yaml(args.config)
        enable_teacher = bool(cfg.get("enable_teacher", False))
        deployment_mode = str(cfg.get("deployment_mode", "stable")).lower()
        dataset_items = merged if isinstance(merged, list) else []
        val_idx = split_indices.get("val_idx", []) if isinstance(split_indices, dict) else []
        if not isinstance(val_idx, list) or not val_idx:
            _, val_idx = _train_val_split(len(dataset_items), 0.2, 42)

        new_agreement = None
        old_agreement = None
        if enable_teacher:
            try:
                teacher_pipeline = _build_teacher_pipeline(cfg, device=device)
                new_agreement = _agreement_with_teacher(
                    checkpoint_path=best_new,
                    dataset_items=dataset_items,
                    val_idx=val_idx,
                    teacher_pipeline=teacher_pipeline,
                    device=device,
                )
                if current_ckpt and Path(current_ckpt).exists():
                    old_agreement = _agreement_with_teacher(
                        checkpoint_path=current_ckpt,
                        dataset_items=dataset_items,
                        val_idx=val_idx,
                        teacher_pipeline=teacher_pipeline,
                        device=device,
                    )
            except Exception as e:
                print(f"Teacher agreement eval failed: {e}")

        def _f(x, d=0.0) -> float:
            try:
                return float(x)
            except Exception:
                return float(d)

        promote = False
        if current_metrics is None:
            promote = True
        else:
            new_acc = _f(new_metrics.get("acc", 0.0))
            old_acc = _f(current_metrics.get("acc", 0.0))
            new_cal = _f(new_metrics.get("calibration_error", 1e9))
            old_cal = _f(current_metrics.get("calibration_error", 1e9))
            if new_agreement is None or old_agreement is None:
                # If teacher is unavailable, fall back to accuracy+calibration safety.
                promote = (new_acc > old_acc) and (new_cal <= old_cal)
            else:
                promote = (new_acc > old_acc) and (new_cal <= old_cal) and (float(new_agreement) >= float(old_agreement))

        registry_path = Path(args.models_dir) / "registry.json"
        registry = _load_registry(args.models_dir)

        entry = {
            "version": version,
            "timestamp": time.time(),
            "dataset_size": len(merged),
            "new_samples_kept": len(filtered),
            "metrics": new_metrics,
            "agreement": new_agreement,
            "promoted": promote,
            "checkpoint": best_new,
        }
        history = list(registry.get("history", [])) if isinstance(registry.get("history", []), list) else []
        history.append(entry)
        registry["history"] = history[-50:]

        if promote:
            if deployment_mode == "canary":
                registry["canary"] = entry
                _set_canary_checkpoint_in_config(args.config, best_new)
                print(f"Canary set {version}: acc={float(new_metrics.get('acc', 0.0)):.4f}")
                _append_audit(
                    {
                        "event": "model_promoted_canary",
                        "version": version,
                        "checkpoint": best_new,
                        "metrics": new_metrics,
                        "agreement": new_agreement,
                    }
                )
            else:
                registry["current"] = entry
                _set_student_checkpoint_in_config(args.config, best_new)
                print(f"Promoted {version}: acc={float(new_metrics.get('acc', 0.0)):.4f}")
                _append_audit(
                    {"event": "model_promoted", "version": version, "checkpoint": best_new, "metrics": new_metrics, "agreement": new_agreement}
                )
        else:
            print(f"Kept current: new acc={float(new_metrics.get('acc', 0.0)):.4f}")
            _append_audit(
                {
                    "event": "model_rejected",
                    "version": version,
                    "checkpoint": best_new,
                    "new_metrics": new_metrics,
                    "old_metrics": current_metrics,
                    "new_agreement": new_agreement,
                    "old_agreement": old_agreement,
                }
            )

        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        _append_audit({"event": "retrain_end", "version": version, "promoted": promote, "dataset_size": len(merged)})

    if not args.loop:
        _run_once()
        return 0

    while True:
        _run_once()
        sleep_s = float(args.interval_hours) * 3600.0
        print(f"Sleeping {sleep_s:.0f}s...")
        time.sleep(sleep_s)


if __name__ == "__main__":
    raise SystemExit(main())
