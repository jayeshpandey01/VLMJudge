"""
author: Jayesh Pandey
summary: Core training script for the distilled student reward model, defining the dataset, architecture, and training loop.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm import tqdm
import logging
from typing import List, Dict, Any, Optional, Tuple
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Dataset Loader ---

class PreferenceDataset(Dataset):
    """
    Dataset for pairwise preference data.
    Format: [{"prompt": "...", "chosen": "path/to/A.png", "rejected": "path/to/B.png", ...}]
    """
    def __init__(
        self, 
        data_path: str, 
        preprocess: Any, 
        quality_filter: List[str] = ["high", "medium"],
        include_medium_weight: float = 0.5,
        log_disagreements_path: Optional[str] = None,
    ):
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        self.samples = []
        self.preprocess = preprocess
        self._disagreements: List[Dict[str, Any]] = []
        
        # Try to load ReasoningScorer for VLM reasoning weighting
        self.reasoning_scorer = None
        try:
            from vlmjudge.scorers.reasoning_score import ReasoningScorer
            self.reasoning_scorer = ReasoningScorer()
        except ImportError as e:
            logger.warning(f"ReasoningScorer not available: {e}. Disabling reasoning-weighted loss.")
        except Exception as e:
            logger.warning(f"Failed to initialize ReasoningScorer: {e}. Disabling reasoning-weighted loss.")
        
        for item in raw_data:
            quality = item.get("quality", "low").lower()
            if quality not in quality_filter:
                continue
            
            # Base weight from quality
            base_weight = 1.0 if quality == "high" else include_medium_weight
            
            # Distillation weights: confidence * coverage
            # Prefer fusion confidence when available.
            conf_raw = item.get("fusion_confidence", None)
            if conf_raw is None:
                conf_raw = item.get("final_confidence", None)
            if conf_raw is None:
                conf_raw = item.get("confidence", 1.0)
            try:
                conf = float(conf_raw)
            except Exception:
                conf = 1.0
            conf = float(max(0.0, min(1.0, conf)))
            cov = float(item.get("coverage", 1.0))
            cov = float(max(0.0, min(1.0, cov)))
            
            # Extract reasoning score if VLM was used
            vlm_data = item.get("vlm", {})
            vlm_reason = vlm_data.get("reason", "")
            reasoning_score = 1.0
            if vlm_reason and self.reasoning_scorer is not None:
                reasoning_score = self.reasoning_scorer.score_reasoning(vlm_reason)
                
            # Distillation v2: reasoning-weighted loss
            # Weight = base * confidence * coverage * reasoning_factor
            weight = base_weight * conf * cov * (0.5 + 0.5 * reasoning_score)

            # Feedback weighting (Phase 8 hardening):
            # - feedback: 1.5x
            # - api: 1.0x (default)
            source = str(item.get("source", "api")).lower()
            if source == "feedback":
                weight *= 1.5

            # Hard negative mining from teacher delta (small delta => hard).
            # Prefer top-level delta; fall back to metadata delta if present.
            delta_raw = item.get("delta", None)
            if delta_raw is None:
                delta_raw = (item.get("metadata", {}) or {}).get("delta", 0.0)
            try:
                teacher_delta = abs(float(delta_raw))
            except Exception:
                teacher_delta = 0.0
            teacher_delta = float(max(0.0, min(1.0, teacher_delta)))
            if teacher_delta < 0.1:
                weight *= 1.5
            elif teacher_delta > 0.5:
                weight *= 0.7

            agreement = bool(item.get("agreement", True))

            # Disagreement handling: keep sample, but use a soft target and down-weight.
            pair_target = 1.0
            if not agreement:
                pair_target = 0.5
                weight *= 0.5
                self._disagreements.append(
                    {
                        "prompt": item.get("prompt", ""),
                        "chosen": item.get("chosen", ""),
                        "rejected": item.get("rejected", ""),
                        "confidence": conf,
                        "delta": teacher_delta,
                    }
                )

            self.samples.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
                "weight": float(weight),
                "pair_target": float(pair_target),
                "target_confidence": float(conf),
                "agreement": bool(agreement),
                "teacher_delta": float(teacher_delta),
            })
            
        logger.info(f"Loaded {len(self.samples)} samples after filtering (filter={quality_filter})")
        if log_disagreements_path and self._disagreements:
            try:
                os.makedirs(os.path.dirname(log_disagreements_path), exist_ok=True)
            except OSError as e:
                logger.warning(f"Failed to create disagreements log directory: {e}. Skipping disagreement logging.")
                return
            try:
                with open(log_disagreements_path, "w", encoding="utf-8") as f:
                    for row in self._disagreements:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                logger.info(f"Wrote {len(self._disagreements)} disagreement samples to {log_disagreements_path}")
            except Exception as e:
                logger.warning(f"Failed to write disagreements log: {e}")

    def __len__(self):
        return len(self.samples)

    def _load_img(self, path: str):
        try:
            return self.preprocess(Image.open(path).convert("RGB"))
        except Exception as e:
            logger.warning(f"Failed to load image {path}: {e}")
            return torch.zeros(3, 224, 224) # Placeholder

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_chosen = self._load_img(item["chosen"])
        img_rejected = self._load_img(item["rejected"])
        return {
            "prompt": item["prompt"],
            "img_chosen": img_chosen,
            "img_rejected": img_rejected,
            "weight": torch.tensor(item["weight"], dtype=torch.float32),
            "pair_target": torch.tensor(item["pair_target"], dtype=torch.float32),
            "target_confidence": torch.tensor(item["target_confidence"], dtype=torch.float32),
            "agreement": torch.tensor(1.0 if item["agreement"] else 0.0, dtype=torch.float32),
            "teacher_delta": torch.tensor(item["teacher_delta"], dtype=torch.float32),
        }

# --- 2. Model Definition ---

class DistilledRewardModel(nn.Module):
    """
    Lightweight Reward Model: Frozen CLIP + Trainable MLP Head.
    """
    def __init__(
        self, 
        clip_model_name: str = "ViT-L-14", 
        pretrained: str = "openai",
        hidden_dim: int = 1024,
        unfreeze_last: bool = False
    ):
        super().__init__()
        try:
            import open_clip  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing dependency: open_clip. Install open_clip to train the student model.") from e
        # Load CLIP
        model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=pretrained)
        self.clip = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)
        
        # Freeze encoder
        for param in self.clip.parameters():
            param.requires_grad = False
            
        if unfreeze_last:
            # Unfreeze last transformer block (optional Advanced Option B)
            for param in self.clip.visual.transformer.resblocks[-1].parameters():
                param.requires_grad = True
            for param in self.clip.transformer.resblocks[-1].parameters():
                param.requires_grad = True

        # MLP Head: embedding -> Linear -> ReLU -> Linear -> score
        embed_dim = self.clip.visual.output_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        self._unfreeze_last = bool(unfreeze_last)

    def forward(self, images: torch.Tensor, prompts: List[str]):
        # Feature extraction
        # Keep CLIP mostly frozen: use no_grad unless explicitly unfreezing.
        with torch.no_grad() if not self._unfreeze_last else torch.enable_grad():
            image_features = self.clip.encode_image(images)
            text_tokens = self.tokenizer(prompts).to(images.device)
            text_features = self.clip.encode_text(text_tokens)
            
            # Normalize features
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
        
        # Concatenate: [batch, embed_dim * 2]
        combined = torch.cat([image_features, text_features], dim=-1)
        
        # Scalar score
        return self.mlp(combined).squeeze(-1)

# --- 3. Training Logic ---

def _compute_losses(
    logits_chosen: torch.Tensor,
    logits_rejected: torch.Tensor,
    *,
    pair_target: torch.Tensor,
    target_confidence: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Pairwise probability that chosen beats rejected.
    diff = logits_chosen - logits_rejected

    # Weighted Bradley-Terry with soft labels for disagreement (target=0.5).
    pair_loss = F.binary_cross_entropy_with_logits(diff, pair_target, reduction="none")

    # Calibration regression on the pairwise probability vs teacher confidence.
    pred_prob = torch.sigmoid(diff)
    reg_loss = (pred_prob - target_confidence) ** 2

    # Final (spec): loss = pairwise_loss + 0.3 * reg_loss
    pair_loss_w = (pair_loss * weight).mean()
    reg_loss_w = (reg_loss * weight).mean()
    total = pair_loss_w + 0.3 * reg_loss_w
    return total, pair_loss_w.detach(), reg_loss_w.detach()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_pair = 0
    total_reg = 0
    correct = 0
    total = 0
    disagreement_n = 0
    
    for batch in tqdm(loader, desc="Training"):
        prompts = batch["prompt"]
        img_chosen = batch["img_chosen"].to(device)
        img_rejected = batch["img_rejected"].to(device)
        weights = batch["weight"].to(device)
        pair_target = batch["pair_target"].to(device)
        target_confidence = batch["target_confidence"].to(device)
        agreement = batch["agreement"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass for both
        scores_chosen = model(img_chosen, prompts)
        scores_rejected = model(img_rejected, prompts)
        
        loss, pair_l, reg_l = _compute_losses(
            scores_chosen,
            scores_rejected,
            pair_target=pair_target,
            target_confidence=target_confidence,
            weight=weights,
        )

        loss.backward()
        optimizer.step()
        
        total_loss += float(loss.item())
        total_pair += float(pair_l.item())
        total_reg += float(reg_l.item())
        
        # Accuracy tracking: chosen > rejected
        correct += (scores_chosen > scores_rejected).sum().item()
        total += len(prompts)
        disagreement_n += int((agreement < 0.5).sum().item())
        
    return {
        "loss": total_loss / max(1, len(loader)),
        "pair_loss": total_pair / max(1, len(loader)),
        "reg_loss": total_reg / max(1, len(loader)),
        "acc": correct / max(1, total),
        "disagreement_n": disagreement_n,
        "n": total,
    }

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_pair = 0.0
    total_reg = 0.0
    correct = 0
    total = 0
    calib_err_sum = 0.0
    agreement_n = 0
    disagreement_n = 0
    avg_conf_sum = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            prompts = batch["prompt"]
            img_chosen = batch["img_chosen"].to(device)
            img_rejected = batch["img_rejected"].to(device)
            weights = batch["weight"].to(device)
            pair_target = batch["pair_target"].to(device)
            target_confidence = batch["target_confidence"].to(device)
            agreement = batch["agreement"].to(device)

            logits_chosen = model(img_chosen, prompts)
            logits_rejected = model(img_rejected, prompts)

            loss, pair_l, reg_l = _compute_losses(
                logits_chosen,
                logits_rejected,
                pair_target=pair_target,
                target_confidence=target_confidence,
                weight=weights,
            )

            total_loss += float(loss.item())
            total_pair += float(pair_l.item())
            total_reg += float(reg_l.item())

            # Metrics
            correct += (logits_chosen > logits_rejected).sum().item()
            total += len(prompts)

            diff = logits_chosen - logits_rejected
            pred_prob = torch.sigmoid(diff)
            calib_err_sum += float((pred_prob - target_confidence).abs().sum().item())
            avg_conf_sum += float(target_confidence.sum().item())

            disagreement_n += int((agreement < 0.5).sum().item())
            agreement_n += int((agreement >= 0.5).sum().item())

    return {
        "loss": total_loss / max(1, len(loader)),
        "pair_loss": total_pair / max(1, len(loader)),
        "reg_loss": total_reg / max(1, len(loader)),
        "acc": correct / max(1, total),
        "teacher_agreement": correct / max(1, total),
        "calibration_error": calib_err_sum / max(1, total),
        "avg_confidence": avg_conf_sum / max(1, total),
        "agreement_n": agreement_n,
        "disagreement_n": disagreement_n,
        "n": total,
    }

# --- 4. Main Training Script ---

def _train_val_split(n: int, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    idxs = list(range(n))
    rnd = random.Random(int(seed))
    rnd.shuffle(idxs)
    val_n = int(round(n * float(val_fraction)))
    val_n = max(1, min(n - 1, val_n)) if n >= 2 else 0
    val_idx = idxs[:val_n]
    train_idx = idxs[val_n:]
    return train_idx, val_idx


def main(
    data_path: str,
    output_dir: str = "distilled_model",
    batch_size: int = 16,
    epochs: int = 5,
    lr_head: float = 1e-4,
    lr_unfreeze: float = 1e-5,
    unfreeze_last: bool = False,
    val_fraction: float = 0.2,
    seed: int = 42,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = DistilledRewardModel(unfreeze_last=unfreeze_last).to(device)
    
    # Dataset
    disagreements_path = os.path.join(output_dir, "disagreements.jsonl")
    dataset = PreferenceDataset(data_path, model.preprocess, log_disagreements_path=disagreements_path)

    train_idx, val_idx = _train_val_split(len(dataset), float(val_fraction), int(seed))
    with open(os.path.join(output_dir, "split_indices.json"), "w", encoding="utf-8") as f:
        json.dump({"train_idx": train_idx, "val_idx": val_idx, "val_fraction": val_fraction, "seed": seed}, f, indent=2)

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Optimizer
    params = [
        {'params': model.mlp.parameters(), 'lr': lr_head},
    ]
    if unfreeze_last:
        params.append({'params': model.clip.parameters(), 'lr': lr_unfreeze})
        
    optimizer = optim.AdamW(params, weight_decay=0.01)
    
    # Loop
    best_val_acc = -1.0
    history: List[Dict[str, Any]] = []
    for epoch in range(epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        history.append(
            {
                "epoch": int(epoch + 1),
                "train": dict(train_metrics),
                "val": dict(val_metrics),
            }
        )

        logger.info(
            "Epoch %d:\n  loss: %.4f (pair=%.4f reg=%.4f)\n  val_loss: %.4f (pair=%.4f reg=%.4f)\n  acc: %.4f\n  agreement: %.4f\n  calibration_error: %.4f\n  avg_confidence: %.4f",
            epoch + 1,
            float(train_metrics["loss"]),
            float(train_metrics["pair_loss"]),
            float(train_metrics["reg_loss"]),
            float(val_metrics["loss"]),
            float(val_metrics["pair_loss"]),
            float(val_metrics["reg_loss"]),
            float(val_metrics["acc"]),
            float(val_metrics["teacher_agreement"]),
            float(val_metrics["calibration_error"]),
            float(val_metrics["avg_confidence"]),
        )
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train': train_metrics,
            'val': val_metrics,
        }, os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt"))

        # Save best model by validation accuracy
        val_acc = float(val_metrics["acc"])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": int(epoch + 1),
                    "model_state_dict": model.state_dict(),
                    "val": val_metrics,
                },
                os.path.join(output_dir, "best.pt"),
            )
            torch.save(model.mlp.state_dict(), os.path.join(output_dir, "best_reward_head.pt"))
            logger.info(f"Saved new best model (val_acc={best_val_acc:.4f})")

    # Save final model
    torch.save(model.mlp.state_dict(), os.path.join(output_dir, "reward_head.pt"))
    with open(os.path.join(output_dir, "metrics_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Model weights saved to {output_dir}")

# --- 5. Inference Example ---

class Predictor:
    def __init__(self, model_path: str, clip_model_name: str = "ViT-L-14", device: str = "cuda"):
        self.device = device
        self.model = DistilledRewardModel(clip_model_name=clip_model_name).to(device)
        self.model.mlp.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
    def predict(self, image: Image.Image, prompt: str) -> Dict[str, float]:
        img_tensor = self.model.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(img_tensor, [prompt]).squeeze(0)
            score = float(torch.sigmoid(logit).item())
            confidence = float(max(0.0, min(1.0, abs(score - 0.5) * 2.0)))
        return {"score": score, "confidence": confidence}

if __name__ == "__main__":
    # Example usage (uncomment and adjust paths)
    # main("path/to/preference_data.json")
    pass
