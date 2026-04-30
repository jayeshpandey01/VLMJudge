"""
author: Jayesh Pandey
summary: Evaluates a distilled reward model on a validation split, computing accuracy, agreement, and calibration metrics.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from train_reward_model import PreferenceDataset, DistilledRewardModel, evaluate, _train_val_split


def _load_split_indices(output_dir: Optional[str]) -> Optional[Tuple[List[int], List[int]]]:
    if not output_dir:
        return None
    p = Path(output_dir) / "split_indices.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        train_idx = obj.get("train_idx", None)
        val_idx = obj.get("val_idx", None)
        
        # Both train_idx and val_idx must be present and be lists
        if not isinstance(train_idx, list):
            return None
        if not isinstance(val_idx, list):
            return None
        
        return train_idx, val_idx
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to load split indices from {p}: {e}")
        return None


def _load_checkpoint(model: DistilledRewardModel, checkpoint_path: str, device: str) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        return
    if isinstance(state, dict):
        # Head-only: load into MLP (support both dropout and non-dropout sequentials).
        keys = set(state.keys())
        has_dropout_idx = any(k.startswith("3.") for k in keys)
        has_no_dropout_idx = any(k.startswith("2.") for k in keys)
        if has_no_dropout_idx and not has_dropout_idx:
            remapped: Dict[str, Any] = {}
            for k, v in state.items():
                if k.startswith("2."):
                    remapped["3." + k[len("2."):]] = v
                else:
                    remapped[k] = v
            model.mlp.load_state_dict(remapped)
        else:
            model.mlp.load_state_dict(state)
        return
    raise ValueError("Unsupported checkpoint format.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate distilled reward model on a validation split.")
    parser.add_argument("--data-path", required=True, type=str, help="Path to preference dataset JSON.")
    parser.add_argument("--checkpoint", required=True, type=str, help='Path to "best.pt" or "reward_head.pt".')
    parser.add_argument("--output-dir", default=None, type=str, help="Training output dir containing split_indices.json.")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--val-fraction", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default=None, type=str, help='Device override, e.g. "cpu" or "cuda:0".')
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DistilledRewardModel().to(device)
    _load_checkpoint(model, args.checkpoint, device)
    model.eval()

    dataset = PreferenceDataset(args.data_path, model.preprocess)
    split = _load_split_indices(args.output_dir)
    if split is None:
        train_idx, val_idx = _train_val_split(len(dataset), float(args.val_fraction), int(args.seed))
    else:
        _, val_idx = split

    val_ds = Subset(dataset, val_idx)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    metrics = evaluate(model, val_loader, device)

    print("Validation metrics:")
    print(f"  accuracy: {metrics['acc']:.4f}")
    print(f"  agreement: {metrics['teacher_agreement']:.4f}")
    print(f"  calibration_error: {metrics['calibration_error']:.4f}")
    print(f"  loss: {metrics['loss']:.4f} (pair={metrics['pair_loss']:.4f} reg={metrics['reg_loss']:.4f})")
    print(f"  avg_confidence: {metrics['avg_confidence']:.4f}")

    if args.output_dir:
        out_path = Path(args.output_dir) / "eval_metrics.json"
        try:
            out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(f"Wrote {str(out_path)}")
        except Exception as e:
            print(f"Failed to write eval_metrics.json: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

