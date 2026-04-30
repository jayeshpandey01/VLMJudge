# Name: Jayesh Pandey
# Summary: VLM Output Caching System.

"""
VLM Output Caching System.
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
from typing import Any, Dict, Optional, Union
from PIL import Image

logger = logging.getLogger(__name__)

class VLMCache:
    def __init__(self, cache_dir: str = "cache/vlm_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_key(self, prompt: str, imageA: Union[str, Image.Image], imageB: Union[str, Image.Image], runs: int) -> str:
        def _hash_input(img: Union[str, Image.Image]) -> str:
            if isinstance(img, str):
                return img
            elif hasattr(img, "filename") and img.filename:
                return img.filename
            else:
                # Fallback to hashing image bytes
                return hashlib.md5(img.tobytes()).hexdigest()
                
        hash_str = f"{prompt}|{_hash_input(imageA)}|{_hash_input(imageB)}|{runs}"
        return hashlib.sha256(hash_str.encode("utf-8")).hexdigest()

    def get(self, prompt: str, imageA: Union[str, Image.Image], imageB: Union[str, Image.Image], runs: int) -> Optional[Dict[str, Any]]:
        key = self._get_key(prompt, imageA, imageB, runs)
        path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.debug(f"VLM cache hit for key {key}")
                return data
            except Exception as e:
                logger.warning(f"Failed to read VLM cache {key}: {e}")
        return None

    def set(self, prompt: str, imageA: Union[str, Image.Image], imageB: Union[str, Image.Image], runs: int, data: Dict[str, Any]) -> None:
        key = self._get_key(prompt, imageA, imageB, runs)
        path = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"VLM cache set for key {key}")
        except Exception as e:
            logger.warning(f"Failed to write VLM cache {key}: {e}")
