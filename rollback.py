from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional


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
        return
    Path(path).write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Rollback active model to a previous version.")
    parser.add_argument("--version", required=True, type=str, help="Version folder name, e.g. v2")
    parser.add_argument("--models-dir", default="models", type=str)
    parser.add_argument("--config", default="config.yaml", type=str)
    args = parser.parse_args()

    version = str(args.version).strip()
    if not re.match(r"^v\d+$", version):
        raise ValueError("--version must look like v2, v10, ...")

    ckpt = Path(args.models_dir) / version / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {str(ckpt)}")

    cfg = _load_yaml(args.config)
    if cfg:
        cfg["student_checkpoint"] = str(ckpt)
        # Clear canary on rollback to reduce risk.
        cfg["canary_checkpoint"] = None
        _write_yaml(args.config, cfg)
    else:
        # Best-effort line replace if YAML isn't available.
        p = Path(args.config)
        if p.exists():
            text = p.read_text(encoding="utf-8")
            text = re.sub(r"^student_checkpoint:\s*.*$", f"student_checkpoint: {str(ckpt)}", text, flags=re.M)
            text = re.sub(r"^canary_checkpoint:\s*.*$", "canary_checkpoint: null", text, flags=re.M)
            p.write_text(text, encoding="utf-8")

    reg = _load_registry(args.models_dir)
    entry = None
    hist = reg.get("history", [])
    if isinstance(hist, list):
        for it in reversed(hist):
            if isinstance(it, dict) and it.get("version", None) == version:
                entry = dict(it)
                break
    if entry is None:
        entry = {"version": version, "timestamp": time.time(), "checkpoint": str(ckpt), "metrics": None, "promoted": True}
        if isinstance(hist, list):
            hist.append(entry)
            reg["history"] = hist[-50:]
        else:
            reg["history"] = [entry]

    reg["current"] = entry
    p = Path(args.models_dir) / "registry.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, indent=2), encoding="utf-8")

    _append_audit({"event": "rollback", "version": version, "checkpoint": str(ckpt), "config": args.config})
    print(f"Rolled back to {version} ({str(ckpt)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

