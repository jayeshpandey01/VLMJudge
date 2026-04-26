from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from tests.utils import make_dummy_png, temp_workdir, write_json, write_jsonl


class RetrainGuardTests(unittest.TestCase):
    def test_retrain_rejects_worse_calibration(self):
        with temp_workdir():
            # base dataset
            make_dummy_png("a.png")
            make_dummy_png("b.png")
            base = [{"prompt": "p", "chosen": "a.png", "rejected": "b.png", "quality": "high", "confidence": 0.8, "coverage": 1.0, "agreement": True}]
            write_json("base.json", base)

            # logs: one compare
            write_jsonl(
                "logs/requests.jsonl",
                [
                    {
                        "type": "compare",
                        "prompt": "p",
                        "imageA": "a.png",
                        "imageB": "b.png",
                        "winner": "A",
                        "confidence": 0.2,
                        "method": "student",
                    }
                ],
            )
            write_jsonl("logs/feedback.jsonl", [])

            # config with current checkpoint
            Path("cur.pt").write_bytes(b"x")
            Path("config.yaml").write_text("student_checkpoint: cur.pt\nenable_teacher: false\n", encoding="utf-8")

            import retrain as r
            import data_engine.merge as mmod

            def _fake_train_main(*args, **kwargs):
                outdir = kwargs.get("output_dir")
                Path(outdir).mkdir(parents=True, exist_ok=True)
                Path(outdir, "best.pt").write_bytes(b"y")
                Path(outdir, "split_indices.json").write_text(json.dumps({"train_idx": [0], "val_idx": [0]}), encoding="utf-8")

            def _fake_eval(path, dataset_path, **kwargs):
                if str(path).endswith("cur.pt"):
                    return {"acc": 0.80, "calibration_error": 0.10}
                return {"acc": 0.81, "calibration_error": 0.20}

            def _merge_no_semantic(base, new, **kwargs):
                kwargs["semantic_dedup"] = False
                return mmod.merge_datasets(base, new, **kwargs)

            with (
                mock.patch.object(r, "train_main", side_effect=_fake_train_main),
                mock.patch.object(r, "_evaluate_checkpoint", side_effect=_fake_eval),
                mock.patch.object(r, "merge_datasets", side_effect=_merge_no_semantic),
            ):
                with mock.patch(
                    "sys.argv",
                    [
                        "retrain.py",
                        "--base-dataset",
                        "base.json",
                        "--min-new-samples",
                        "1",
                        "--min-total-samples",
                        "1",
                        "--min-quality",
                        "low",
                    ],
                ):
                    r.main()

            audit = Path("logs/audit.jsonl").read_text(encoding="utf-8")
            self.assertIn("\"event\": \"model_rejected\"", audit)
            reg = json.loads(Path("models/registry.json").read_text(encoding="utf-8"))
            # should not set current
            self.assertTrue("current" not in reg or reg.get("current", {}).get("checkpoint") != str(Path("models/v1/best.pt")))
