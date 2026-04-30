# Name: Jayesh Pandey
# Summary: Source file for test_logging.py in the tests module.

from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests.utils import read_jsonl, temp_workdir, write_jsonl


class LoggingIntegrityTests(unittest.TestCase):
    def test_requests_jsonl_valid(self):
        with temp_workdir():
            write_jsonl("logs/requests.jsonl", [{"a": 1}, {"b": 2}])
            rows = read_jsonl("logs/requests.jsonl")
            self.assertEqual(len(rows), 2)

    def test_audit_has_required_events(self):
        with temp_workdir():
            Path("logs").mkdir(parents=True, exist_ok=True)
            audit = [
                {"event": "retrain_start"},
                {"event": "model_rejected"},
                {"event": "retrain_end"},
            ]
            with open("logs/audit.jsonl", "w", encoding="utf-8") as f:
                for r in audit:
                    f.write(json.dumps(r) + "\n")
            text = Path("logs/audit.jsonl").read_text(encoding="utf-8")
            self.assertIn("retrain_start", text)
            self.assertIn("retrain_end", text)
            self.assertTrue(("model_promoted" in text) or ("model_rejected" in text))

