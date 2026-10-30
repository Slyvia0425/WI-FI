import unittest

import numpy as np

from backend.modules.breathing_detection.evaluation.analyze_wire_domain_shift import (
    adjacent_correlations,
    distances,
)


class DomainShiftDiagnosticsTest(unittest.TestCase):
    def test_adjacent_correlation_preserves_signed_frequency_relationship(self):
        signal = np.arange(10, dtype=float)
        values = np.stack((signal, signal * 2 + 5, -signal), axis=1)
        np.testing.assert_allclose(adjacent_correlations(values), [1, -1])

    def test_distance_is_zero_at_reference_center_and_increases_outward(self):
        reference = np.random.default_rng(42).normal(size=(100, 4))
        center = reference.mean(axis=0)
        queries = np.stack((center, center + 1, center + 2))
        result = distances(reference, queries)
        self.assertAlmostEqual(result[0], 0)
        self.assertAlmostEqual(result[2], result[1] * 2)


if __name__ == '__main__':
    unittest.main()
