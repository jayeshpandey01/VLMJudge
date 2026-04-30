# Name: Jayesh Pandey
# Summary: Source file for test_data_url_images.py in the tests module.

from __future__ import annotations

import base64
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


def _png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class DataUrlImageTests(unittest.TestCase):
    def test_compare_accepts_data_urls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.yaml"
            cfg_path.write_text("enable_teacher: false\nstudent_checkpoint: models/v1/best.pt\n", encoding="utf-8")
            (Path(td) / "models" / "v1").mkdir(parents=True, exist_ok=True)
            (Path(td) / "models" / "v1" / "best.pt").write_bytes(b"x")

            from api import runtime as rtmod

            class _Engine:
                def compare(self, pil_a, pil_b, prompt):
                    return 0.6, 0.4, 0.2, "A"

            with mock.patch.object(rtmod.StudentEngine, "load", return_value=_Engine()):
                rt = rtmod.InferenceRuntime.from_yaml(str(cfg_path))

            img_a = _png_data_url(Image.new("RGB", (2, 2), color=(255, 0, 0)))
            img_b = _png_data_url(Image.new("RGB", (2, 2), color=(0, 0, 255)))
            out = rt.compare(prompt="test", image_a_ref=img_a, image_b_ref=img_b, return_debug=True)
            self.assertEqual(out.get("winner"), "A")
            self.assertGreater(float(out.get("confidence", 0.0)), 0.0)

