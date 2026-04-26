from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from tests.utils import temp_workdir, write_json, write_jsonl


class RollbackTests(unittest.TestCase):
    def test_rollback_updates_config_and_registry(self):
        with temp_workdir():
            Path("models/v1").mkdir(parents=True, exist_ok=True)
            Path("models/v2").mkdir(parents=True, exist_ok=True)
            Path("models/v1/best.pt").write_bytes(b"x")
            Path("models/v2/best.pt").write_bytes(b"y")

            Path("config.yaml").write_text(
                "student_checkpoint: models/v2/best.pt\ncanary_checkpoint: models/v3/best.pt\n", encoding="utf-8"
            )
            write_json(
                "models/registry.json",
                {"history": [{"version": "v1", "checkpoint": "models/v1/best.pt"}, {"version": "v2", "checkpoint": "models/v2/best.pt"}], "current": {"version": "v2"}},
            )

            import rollback as rb

            with mock.patch("sys.argv", ["rollback.py", "--version", "v1", "--models-dir", "models", "--config", "config.yaml"]):
                rb.main()

            cfg = Path("config.yaml").read_text(encoding="utf-8")
            self.assertTrue(("models/v1/best.pt" in cfg) or ("models\\v1\\best.pt" in cfg))
            self.assertIn("canary_checkpoint", cfg)

            reg = json.loads(Path("models/registry.json").read_text(encoding="utf-8"))
            self.assertEqual(reg["current"]["version"], "v1")
            audit = Path("logs/audit.jsonl").read_text(encoding="utf-8")
            self.assertIn("\"event\": \"rollback\"", audit)

    def test_runtime_uses_rolled_back_checkpoint(self):
        with temp_workdir():
            Path("models/v1").mkdir(parents=True, exist_ok=True)
            Path("models/v1/best.pt").write_bytes(b"x")
            Path("config.yaml").write_text("student_checkpoint: models/v1/best.pt\n", encoding="utf-8")

            from api import runtime as rtmod

            seen = {}

            def _fake_load(*, checkpoint_path: str, device: str, tie_threshold: float):
                seen["checkpoint"] = checkpoint_path

                class _E:
                    pass

                return _E()

            with mock.patch.object(rtmod.StudentEngine, "load", side_effect=_fake_load):
                r = rtmod.InferenceRuntime.from_yaml("config.yaml")
            self.assertEqual(seen["checkpoint"], "models/v1/best.pt")
