import unittest

import numpy as np

from backend.modules.breathing_detection.evaluation.perturb_wire_frequency import perturb


class FrequencyPerturbationTest(unittest.TestCase):
    def test_only_reorders_frequency_streams(self):
        values = np.arange(24).reshape(2, 3, 4)
        shifted = perturb(values, 1)
        np.testing.assert_array_equal(shifted[..., 0], values[..., -1])
        np.testing.assert_array_equal(perturb(shifted, -1), values)
        np.testing.assert_array_equal(perturb(values, reverse=True)[..., 0], values[..., -1])
        np.testing.assert_array_equal(np.sort(shifted, axis=-1), np.sort(values, axis=-1))

    def test_edge_shift_does_not_wrap_opposite_boundary(self):
        values = np.arange(12).reshape(1, 3, 4)
        shifted = perturb(values, 1, edge=True)
        np.testing.assert_array_equal(shifted[..., 0], values[..., 0])
        np.testing.assert_array_equal(shifted[..., 1:], values[..., :-1])
        shifted = perturb(values, -1, edge=True)
        np.testing.assert_array_equal(shifted[..., -1], values[..., -1])


if __name__ == '__main__':
    unittest.main()
