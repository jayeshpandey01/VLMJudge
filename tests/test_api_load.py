from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from tests.utils import fastapi_available, temp_workdir


@unittest.skipUnless(fastapi_available(), "fastapi not available")
class APILoadTests(unittest.TestCase):
    def test_concurrent_requests_no_crash(self):
        from api.app import create_app

        class FakeRuntime:
            device = "cpu"

            def compare(self, **kwargs):
                return {"winner": "A", "confidence": 0.7, "scoreA": 0.8, "scoreB": 0.2, "method": "student", "timing_ms": {"total": 1.0}}

            def score(self, **kwargs):
                return {"score": 0.5, "confidence": 0.1, "timing_ms": 1.0}

        with temp_workdir():
            with mock.patch("api.app.InferenceRuntime.from_yaml", return_value=FakeRuntime()):
                app = create_app(config_path="config.yaml")
                from fastapi.testclient import TestClient

                errors = []

                def worker():
                    try:
                        r = client.post("/compare", json={"prompt": "p", "imageA": "a", "imageB": "b"})
                        if r.status_code != 200:
                            errors.append(r.status_code)
                    except Exception as e:
                        errors.append(str(e))

                with TestClient(app) as client:
                    t0 = time.perf_counter()
                    threads = [threading.Thread(target=worker) for _ in range(50)]
                    for th in threads:
                        th.start()
                    for th in threads:
                        th.join()
                    dt = time.perf_counter() - t0

                self.assertEqual(errors, [])
                self.assertLess(dt, 5.0)
