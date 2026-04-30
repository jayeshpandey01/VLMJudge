# Name: Jayesh Pandey
# Summary: Source file for test_teacher_tie_break.py in the tests module.

from __future__ import annotations

import base64
import io
import unittest

from PIL import Image

from api.runtime import InferenceRuntime


def _png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class _StudentRaises:
    def compare(self, pil_a, pil_b, prompt):
        raise FileNotFoundError("no student")

    def score(self, pil, prompt):
        raise FileNotFoundError("no student")


class _TeacherStub:
    def compare(self, pil_a, pil_b, prompt):
        # Simulate pipeline returning tie with tiny calibrated confidence while aggregates differ.
        return {
            "winner": "tie",
            "confidence": 0.001,
            "explanation": "x",
            "vlm": {"reason": "x"},
            "structured": {"aggregate": {"A": {"score": 0.19}, "B": {"score": 0.15}}},
        }


class TeacherTieBreakTests(unittest.TestCase):
    def test_teacher_tie_promoted_using_runtime_tie_threshold(self) -> None:
        rt = InferenceRuntime(
            cfg={"tie_threshold": 0.02, "confidence_threshold": 0.6},
            device="cpu",
            student_stable=_StudentRaises(),
            student_canary=None,
            shadow_students=[],
            teacher=_TeacherStub(),
        )

        img_a = _png_data_url(Image.new("RGB", (2, 2), color=(255, 0, 0)))
        img_b = _png_data_url(Image.new("RGB", (2, 2), color=(0, 0, 255)))
        out = rt.compare(prompt="x", image_a_ref=img_a, image_b_ref=img_b, return_debug=True)

        self.assertEqual(out.get("winner"), "A")
        self.assertAlmostEqual(float(out.get("confidence", 0.0)), 0.04, places=6)
        self.assertAlmostEqual(float(out.get("scoreA", 0.0)), 0.19, places=6)
        self.assertAlmostEqual(float(out.get("scoreB", 0.0)), 0.15, places=6)
        self.assertIn("scores", out)
        self.assertIn("vlm", out.get("scores", {}))
        self.assertIn("structured", out.get("scores", {}))
        self.assertEqual(out.get("reasoning"), "x")
