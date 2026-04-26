from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _hash_key(prompt: str, chosen: str, rejected: str) -> str:
    s = f"{prompt.strip()}||{chosen.strip()}||{rejected.strip()}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _as_record(x: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        prompt = str(x.get("prompt", "")).strip()
        chosen = str(x.get("chosen", "")).strip()
        rejected = str(x.get("rejected", "")).strip()
    except Exception:
        return None
    if not prompt or not chosen or not rejected:
        return None
    out = dict(x)
    out["prompt"] = prompt
    out["chosen"] = chosen
    out["rejected"] = rejected
    return out


class _CLIPDeduper:
    def __init__(
        self,
        *,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
        threshold: float = 0.95,
        max_bucket: int = 2000,
        seed: int = 42,
    ) -> None:
        self._enabled = False
        self._threshold = float(threshold)
        self._max_bucket = int(max_bucket)
        self._rnd = random.Random(int(seed))
        self._device = str(device)

        try:
            import torch
            import open_clip
            from PIL import Image
        except Exception:
            return

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            tokenizer = open_clip.get_tokenizer(model_name)
            model.eval().to(self._device)
        except Exception:
            return

        self._torch = torch
        self._Image = Image
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._img_cache: Dict[str, Any] = {}
        self._txt_cache: Dict[str, Any] = {}
        self._enabled = True

        # prompt_bucket -> list[tensor(D)]
        self._buckets: Dict[str, List[Any]] = {}

    def enabled(self) -> bool:
        return bool(self._enabled)

    def _norm_prompt(self, prompt: str) -> str:
        return " ".join(prompt.strip().lower().split())[:256]

    def _embed_text(self, prompt: str):
        key = self._norm_prompt(prompt)
        hit = self._txt_cache.get(key)
        if hit is not None:
            return hit
        tokens = self._tokenizer([key]).to(self._device)
        with self._torch.no_grad():
            tf = self._model.encode_text(tokens)
            tf = tf / tf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        self._txt_cache[key] = tf
        return tf

    def _embed_image(self, path: str):
        hit = self._img_cache.get(path)
        if hit is not None:
            return hit
        try:
            pil = self._Image.open(path).convert("RGB")
        except Exception:
            # fallback: zero vector (will reduce similarity)
            with self._torch.no_grad():
                z = self._torch.zeros(1, int(getattr(self._model.visual, "output_dim", 512)), device=self._device)
                self._img_cache[path] = z
                return z
        img_t = self._preprocess(pil).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            im = self._model.encode_image(img_t)
            im = im / im.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        self._img_cache[path] = im
        return im

    def _embed_sample(self, prompt: str, chosen: str, rejected: str):
        tf = self._embed_text(prompt)
        ic = self._embed_image(chosen)
        ir = self._embed_image(rejected)
        with self._torch.no_grad():
            emb = (tf + ic + ir) / 3.0
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return emb

    def is_near_duplicate(self, prompt: str, chosen: str, rejected: str) -> bool:
        if not self._enabled:
            return False
        bucket_key = self._norm_prompt(prompt)
        emb = self._embed_sample(prompt, chosen, rejected)
        bucket = self._buckets.get(bucket_key, [])
        if not bucket:
            return False

        # Limit comparisons in extremely large buckets.
        if len(bucket) > self._max_bucket:
            idxs = [self._rnd.randrange(len(bucket)) for _ in range(self._max_bucket)]
            cand = [bucket[i] for i in idxs]
        else:
            cand = bucket

        # Cosine similarity = dot since normalized.
        for prev in cand:
            sim = float((emb * prev).sum(dim=-1).item())
            if sim >= self._threshold:
                return True
        return False

    def add(self, prompt: str, chosen: str, rejected: str) -> None:
        if not self._enabled:
            return
        bucket_key = self._norm_prompt(prompt)
        emb = self._embed_sample(prompt, chosen, rejected).detach().cpu()
        self._buckets.setdefault(bucket_key, []).append(emb)


def merge_datasets(
    base: Iterable[Mapping[str, Any]],
    new: Iterable[Mapping[str, Any]],
    *,
    max_total: Optional[int] = None,
    max_new_ratio: float = 0.5,
    seed: int = 42,
    semantic_dedup: bool = True,
    semantic_threshold: float = 0.95,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Merge two preference datasets, avoiding exact duplicates on (prompt,chosen,rejected).
    """
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    stats = {
        "base_kept": 0,
        "new_added": 0,
        "duplicates_skipped": 0,
        "new_downsampled": 0,
        "semantic_skipped": 0,
    }

    rnd = random.Random(int(seed))
    deduper = _CLIPDeduper(threshold=float(semantic_threshold), seed=int(seed)) if semantic_dedup else None

    for x in base:
        rec = _as_record(x)
        if rec is None:
            continue
        k = _hash_key(rec["prompt"], rec["chosen"], rec["rejected"])
        if k in seen:
            stats["duplicates_skipped"] += 1
            continue
        seen.add(k)
        merged.append(rec)
        stats["base_kept"] += 1
        if deduper is not None and deduper.enabled():
            # Populate semantic index from base (best-effort).
            try:
                deduper.add(rec["prompt"], rec["chosen"], rec["rejected"])
            except Exception:
                pass
        if max_total is not None and len(merged) >= int(max_total):
            return merged, stats

    # Collect unique new records first (exact dedup), then apply semantic + drift protection.
    new_unique: List[Dict[str, Any]] = []
    for x in new:
        rec = _as_record(x)
        if rec is None:
            continue
        k = _hash_key(rec["prompt"], rec["chosen"], rec["rejected"])
        if k in seen:
            stats["duplicates_skipped"] += 1
            continue
        seen.add(k)
        new_unique.append(rec)

    # Semantic dedup (prompt-bucketed CLIP cosine) within base+accepted.
    new_filtered: List[Dict[str, Any]] = []
    if deduper is not None and deduper.enabled():
        for rec in new_unique:
            try:
                if deduper.is_near_duplicate(rec["prompt"], rec["chosen"], rec["rejected"]):
                    stats["semantic_skipped"] += 1
                    continue
                deduper.add(rec["prompt"], rec["chosen"], rec["rejected"])
                new_filtered.append(rec)
            except Exception:
                new_filtered.append(rec)
    else:
        new_filtered = new_unique

    # Drift protection: cap proportion of new samples.
    # Enforce: new <= max_new_ratio * (base + new)  => new <= r*base/(1-r)
    base_n = int(stats["base_kept"])
    r = float(max(0.0, min(0.95, max_new_ratio)))
    max_new_allowed = int((r * base_n) / max(1e-9, (1.0 - r))) if base_n > 0 else int(len(new_filtered))
    if max_new_allowed < 0:
        max_new_allowed = 0

    if len(new_filtered) > max_new_allowed:
        rnd.shuffle(new_filtered)
        stats["new_downsampled"] = len(new_filtered) - max_new_allowed
        new_filtered = new_filtered[:max_new_allowed]

    for rec in new_filtered:
        merged.append(rec)
        stats["new_added"] += 1
        if max_total is not None and len(merged) >= int(max_total):
            break

    return merged, stats
