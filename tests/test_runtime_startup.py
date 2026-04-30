# Name: Jayesh Pandey
# Summary: Source file for test_runtime_startup.py in the tests module.

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from api.runtime import InferenceRuntime, MissingStudentEngine


class TestRuntimeStartup(unittest.TestCase):
    def test_from_yaml_does_not_crash_when_student_checkpoint_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        "enable_teacher: false",
                        "student_checkpoint: distilled_model/best.pt",
                        "tie_threshold: 0.02",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rt = InferenceRuntime.from_yaml(str(cfg_path))
            self.assertIsInstance(rt.student_stable, MissingStudentEngine)

            expected = str((Path(td) / "distilled_model" / "best.pt").resolve())
            self.assertEqual(rt.student_stable.resolved_path, expected)

            img = Image.new("RGB", (1, 1), color=(0, 0, 0))
            with self.assertRaises(FileNotFoundError):
                rt.student_stable.score(img, "x")
            with self.assertRaises(FileNotFoundError):
                rt.student_stable.compare(img, img, "x")

