from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from tests.utils import fastapi_available, temp_workdir, write_jsonl


@unittest.skipUnless(fastapi_available(), "fastapi not available")
class FeedbackValidationTests(unittest.TestCase):
    def test_flagged_feedback_written_and_excluded(self):
        from api.app import create_app

        class FakeRuntime:
            device = "cpu"

            def compare(self, **kwargs):
                return {"winner": "A", "confidence": 0.95, "scoreA": 0.9, "scoreB": 0.1, "method": "student", "timing_ms": {"total": 1.0}}

            def score(self, **kwargs):
                return {"score": 0.5, "confidence": 0.1, "timing_ms": 1.0}

        with temp_workdir():
            with mock.patch("api.app.InferenceRuntime.from_yaml", return_value=FakeRuntime()):
                app = create_app(config_path="config.yaml")
                from fastapi.testclient import TestClient

                with TestClient(app) as c:
                    r = c.post(
                        "/feedback",
                        json={"correct_winner": "B", "prompt": "p", "imageA": "a.png", "imageB": "b.png"},
                    )
                    self.assertEqual(r.status_code, 200)
                    self.assertTrue(r.json().get("flagged", False))

            flagged = Path("logs/flagged_feedback.jsonl").read_text(encoding="utf-8")
            self.assertIn("\"flagged\": true", flagged.lower())

            # Selector should ignore flagged feedback
            write_jsonl(
                "logs/requests.jsonl",
                [{"type": "compare", "prompt": "p", "imageA": "a.png", "imageB": "b.png", "winner": "A", "confidence": 0.95, "method": "student"}],
            )
            from data_engine.selector import select_samples

            sel = select_samples()
            self.assertEqual(sel, [])
