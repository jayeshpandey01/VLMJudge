# Name: Jayesh Pandey
# Summary: Source file for test_e2e.py in the tests module.

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from tests.utils import fastapi_available, make_dummy_png, temp_workdir, write_json


@unittest.skipUnless(fastapi_available(), "fastapi not available")
class E2ETests(unittest.TestCase):
    def test_end_to_end_log_to_dataset_to_retrain_artifacts(self):
        from api.app import create_app

        class FakeRuntime:
            device = "cpu"

            def compare(self, **kwargs):
                return {
                    "winner": "A",
                    "confidence": 0.2,
                    "scoreA": 0.7,
                    "scoreB": 0.3,
                    "method": "student",
                    "timing_ms": {"total": 1.0},
                    "_debug": {"agreement": None, "student_winner": "A", "teacher_winner": None},
                }

            def score(self, **kwargs):
                return {"score": 0.5, "confidence": 0.1, "timing_ms": 1.0}

        with temp_workdir():
            make_dummy_png("a.png")
            make_dummy_png("b.png")
            write_json(
                "base.json",
                [{"prompt": "p", "chosen": "a.png", "rejected": "b.png", "quality": "high", "confidence": 0.8, "coverage": 1.0, "agreement": True}],
            )
            Path("cur.pt").write_bytes(b"x")
            Path("config.yaml").write_text("student_checkpoint: cur.pt\nenable_teacher: false\n", encoding="utf-8")

            with mock.patch("api.app.InferenceRuntime.from_yaml", return_value=FakeRuntime()):
                app = create_app(config_path="config.yaml")
                from fastapi.testclient import TestClient

                with TestClient(app) as c:
                    r = c.post("/compare", json={"prompt": "p", "imageA": "a.png", "imageB": "b.png"})
                    self.assertEqual(r.status_code, 200)

            # Build new data
            from data_engine.selector import select_samples
            from data_engine.builder import build_preferences

            sel = select_samples()
            prefs = build_preferences(sel, source="api")
            self.assertTrue(isinstance(prefs, list))

            # Retrain with patched training/eval (dry-ish)
            import retrain as rr
            import data_engine.merge as mmod

            def _fake_train_main(*args, **kwargs):
                outdir = kwargs.get("output_dir")
                Path(outdir).mkdir(parents=True, exist_ok=True)
                Path(outdir, "best.pt").write_bytes(b"y")
                Path(outdir, "split_indices.json").write_text(json.dumps({"train_idx": [0], "val_idx": [0]}), encoding="utf-8")

            def _fake_eval(*args, **kwargs):
                return {"acc": 0.9, "calibration_error": 0.1}

            def _merge_no_semantic(base, new, **kwargs):
                kwargs["semantic_dedup"] = False
                return mmod.merge_datasets(base, new, **kwargs)

            with (
                mock.patch.object(rr, "train_main", side_effect=_fake_train_main),
                mock.patch.object(rr, "_evaluate_checkpoint", side_effect=_fake_eval),
                mock.patch.object(rr, "merge_datasets", side_effect=_merge_no_semantic),
            ):
                with mock.patch(
                    "sys.argv",
                    ["retrain.py", "--base-dataset", "base.json", "--min-new-samples", "0", "--min-total-samples", "1", "--min-quality", "low"],
                ):
                    rr.main()

            self.assertTrue(Path("models/v1/dataset_merged.json").exists())
            self.assertTrue(Path("models/v1/data_snapshot.json").exists())
