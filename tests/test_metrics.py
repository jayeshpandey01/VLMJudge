from __future__ import annotations

import math
import unittest


class MetricsConsistencyTests(unittest.TestCase):
    def test_metrics_are_finite_and_in_range(self):
        from vlmjudge.bench.metrics import expected_calibration_error, mean_absolute_error

        confidences = [0.1, 0.9, 0.6, 0.2, 0.8]
        correct = [False, True, True, False, True]

        ece = expected_calibration_error(confidences, correct, n_bins=5)
        mae = mean_absolute_error(confidences, correct)
        self.assertTrue(math.isfinite(ece))
        self.assertTrue(math.isfinite(mae))
        self.assertTrue(0.0 <= ece <= 1.0)
        self.assertTrue(0.0 <= mae <= 1.0)

