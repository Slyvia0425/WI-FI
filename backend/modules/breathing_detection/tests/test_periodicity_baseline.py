import unittest

import numpy as np

from backend.modules.breathing_detection.preprocessing.features import (
    FEATURE_NAMES,
    PeriodicityFeatureConfig,
    extract_periodicity_features,
)
from backend.modules.breathing_detection.training.periodicity_baseline import (
    LogisticBaseline,
    binary_metrics,
    select_macro_f1_threshold,
)


class PeriodicityFeaturesTest(unittest.TestCase):
    def test_detects_known_periodic_peak_with_variable_subcarriers(self):
        rng = np.random.default_rng(42)
        time = np.arange(300) / 30.0
        for subcarriers in (112, 208, 240, 416):
            phases = rng.uniform(0, 2 * np.pi, size=(subcarriers, 1))
            signal = np.sin(2 * np.pi * 0.3 * time + phases)
            signal += rng.normal(0, 0.15, size=signal.shape)
            features = extract_periodicity_features(signal[None, :, :])
            feature_map = dict(zip(FEATURE_NAMES, features))
            self.assertEqual(features.shape, (len(FEATURE_NAMES),))
            self.assertAlmostEqual(feature_map["peak_frequency_median_hz"], 0.3, places=5)
            self.assertGreater(feature_map["peak_frequency_consistency"], 0.9)

    def test_rejects_wrong_layout(self):
        with self.assertRaises(ValueError):
            extract_periodicity_features(np.zeros((208, 300)))

    def test_configuration_requires_fft_bins_in_band(self):
        config = PeriodicityFeatureConfig(breathing_band_hz=(20.0, 21.0))
        with self.assertRaises(ValueError):
            extract_periodicity_features(np.zeros((1, 10, 300)), config)


class LogisticBaselineTest(unittest.TestCase):
    def test_fits_separable_features_and_selects_validation_threshold(self):
        rng = np.random.default_rng(7)
        negative = rng.normal(-1.0, 0.2, size=(100, 3))
        positive = rng.normal(1.0, 0.2, size=(100, 3))
        x = np.vstack([negative, positive])
        y = np.asarray([0] * 100 + [1] * 100)
        model = LogisticBaseline(epochs=400).fit(x, y)
        probabilities = model.predict_proba(x)
        threshold = select_macro_f1_threshold(y, probabilities)
        metrics = binary_metrics(y, probabilities, threshold)
        self.assertGreater(metrics["f1"], 0.99)
        self.assertEqual(metrics["confusion_matrix"]["fn"], 0)


if __name__ == "__main__":
    unittest.main()
