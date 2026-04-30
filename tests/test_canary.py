# Name: Jayesh Pandey
# Summary: Source file for test_canary.py in the tests module.

from __future__ import annotations

import random
import unittest


class CanaryRoutingTests(unittest.TestCase):
    def test_canary_ratio_distribution(self):
        from api.runtime import InferenceRuntime

        class _S:
            pass

        rt = InferenceRuntime(
            cfg={"deployment_mode": "canary", "canary_ratio": 0.1},
            device="cpu",
            student_stable=_S(),
            student_canary=_S(),
            teacher=None,
        )

        random.seed(0)
        n = 200
        canary = 0
        for _ in range(n):
            _, variant = rt._choose_student()
            if variant == "canary":
                canary += 1
        ratio = canary / n
        self.assertTrue(0.05 <= ratio <= 0.15, f"ratio={ratio} canary={canary}")

