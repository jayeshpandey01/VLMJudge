from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any, Dict, Optional

from tests.utils import make_dummy_png, temp_workdir


@dataclass
class _FailingStudent:
    def compare(self, *args, **kwargs):
        raise RuntimeError("student failed")

    def score(self, *args, **kwargs):
        raise RuntimeError("student failed")


@dataclass
class _FailingTeacher:
    def compare(self, *args, **kwargs):
        raise RuntimeError("teacher failed")


class FailureTests(unittest.TestCase):
    def test_invalid_image_path_safe_fallback(self):
        from api.runtime import InferenceRuntime

        with temp_workdir():
            cfg = {"confidence_threshold": 0.6, "enable_teacher": False}
            rt = InferenceRuntime(cfg=cfg, device="cpu", student_stable=_FailingStudent(), student_canary=None, teacher=None)
            out = rt.compare(prompt="p", image_a_ref="nope.png", image_b_ref="nope2.png")
            self.assertEqual(out["method"], "safe_fallback")
            self.assertEqual(out["winner"], "tie")
            self.assertEqual(out["confidence"], 0.0)

    def test_corrupted_image_safe_fallback(self):
        from api.runtime import InferenceRuntime

        with temp_workdir():
            with open("bad.png", "wb") as f:
                f.write(b"not a png")
            make_dummy_png("ok.png")
            cfg = {"confidence_threshold": 0.6, "enable_teacher": False}
            rt = InferenceRuntime(cfg=cfg, device="cpu", student_stable=_FailingStudent(), student_canary=None, teacher=None)
            out = rt.compare(prompt="p", image_a_ref="bad.png", image_b_ref="ok.png")
            self.assertEqual(out["method"], "safe_fallback")

    def test_unsupported_format_safe_fallback(self):
        from api.runtime import InferenceRuntime

        with temp_workdir():
            with open("file.txt", "w", encoding="utf-8") as f:
                f.write("hello")
            make_dummy_png("ok.png")
            cfg = {"confidence_threshold": 0.6, "enable_teacher": False}
            rt = InferenceRuntime(cfg=cfg, device="cpu", student_stable=_FailingStudent(), student_canary=None, teacher=None)
            out = rt.compare(prompt="p", image_a_ref="file.txt", image_b_ref="ok.png")
            self.assertEqual(out["method"], "safe_fallback")

    def test_missing_prompt_safe_fallback(self):
        from api.runtime import InferenceRuntime

        with temp_workdir():
            make_dummy_png("a.png")
            make_dummy_png("b.png")
            cfg = {"confidence_threshold": 0.6, "enable_teacher": False}
            rt = InferenceRuntime(cfg=cfg, device="cpu", student_stable=_FailingStudent(), student_canary=None, teacher=None)
            out = rt.compare(prompt="", image_a_ref="a.png", image_b_ref="b.png")
            self.assertEqual(out["method"], "safe_fallback")
            self.assertEqual(out["winner"], "tie")

    def test_both_student_and_teacher_fail(self):
        from api.runtime import InferenceRuntime

        with temp_workdir():
            make_dummy_png("a.png")
            make_dummy_png("b.png")
            cfg = {"confidence_threshold": 0.6, "enable_teacher": True}
            rt = InferenceRuntime(cfg=cfg, device="cpu", student_stable=_FailingStudent(), student_canary=None, teacher=_FailingTeacher())
            out = rt.compare(prompt="p", image_a_ref="a.png", image_b_ref="b.png")
            self.assertEqual(out["method"], "safe_fallback")
            self.assertEqual(out["winner"], "tie")
            self.assertEqual(out["confidence"], 0.0)

