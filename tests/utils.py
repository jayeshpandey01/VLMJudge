from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@contextlib.contextmanager
def temp_workdir() -> Iterator[str]:
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        try:
            yield td
        finally:
            os.chdir(old)


def write_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def make_dummy_png(path: str, *, size: int = 32) -> None:
    from PIL import Image

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (size, size), color=(123, 222, 10))
    img.save(str(p), format="PNG")


def run_py(args: List[str], *, timeout: int = 60) -> str:
    cmd = [sys.executable] + args
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout, text=True)


def fastapi_available() -> bool:
    try:
        import fastapi  # noqa: F401

        return True
    except Exception:
        return False

