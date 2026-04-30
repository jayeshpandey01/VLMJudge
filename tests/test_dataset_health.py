# Name: Jayesh Pandey
# Summary: Source file for test_dataset_health.py in the tests module.

from __future__ import annotations

import json
import unittest

from unittest import mock

from tests.utils import temp_workdir, write_json, write_jsonl


class DatasetHealthTests(unittest.TestCase):
    def test_monitor_health_thresholds(self):
        with temp_workdir():
            # registry: new ratio 0.5
            write_json(
                "models/registry.json",
                {"current": {"version": "v1", "dataset_size": 1000, "new_samples_kept": 500}},
            )
            # logs: stable confidence/disagreement
            rows = []
            for i in range(300):
                rows.append(
                    {
                        "type": "compare",
                        "confidence": 0.7,
                        "method": "teacher",
                        "agreement": True,
                    }
                )
            write_jsonl("logs/requests.jsonl", rows)
            write_jsonl("logs/feedback.jsonl", [])

            import monitor as mon

            with mock.patch("sys.argv", ["monitor.py", "--json"]):
                # monitor prints JSON to stdout; capture via json module by re-running its logic is heavy,
                # so instead call as subprocess-like by redirecting stdout.
                import io
                import sys

                buf = io.StringIO()
                old = sys.stdout
                sys.stdout = buf
                try:
                    mon.main()
                finally:
                    sys.stdout = old
                obj = json.loads(buf.getvalue())
            self.assertLess(obj["new_ratio_last_train"], 0.6)
            self.assertEqual(obj["warnings"], [])
