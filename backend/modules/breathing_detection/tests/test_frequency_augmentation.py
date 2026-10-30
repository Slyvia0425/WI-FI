import unittest

import numpy as np

from backend.modules.breathing_detection.training.train_frequency_robustness import augment_batch


class FrequencyAugmentationTest(unittest.TestCase):
    def test_augmentation_preserves_time_streams_and_is_reproducible(self):
        values = np.arange(8 * 3 * 12, dtype=np.float32).reshape(8, 3, 12)
        actual = augment_batch(values, np.random.default_rng(5))
        repeated = augment_batch(values, np.random.default_rng(5))
        np.testing.assert_array_equal(actual, repeated)
        self.assertEqual(actual.shape, values.shape)
        # Every output stream still has its original temporal increments.
        np.testing.assert_array_equal(np.diff(actual, axis=1), np.diff(values, axis=1))
        np.testing.assert_array_equal(values, np.arange(values.size).reshape(values.shape))


if __name__ == '__main__':
    unittest.main()
