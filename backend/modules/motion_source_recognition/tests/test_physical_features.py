"""Checks on the physical features of a window.

    backend/modules/motion_source_recognition/.venv/bin/python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import unittest

import numpy as np

from backend.modules.motion_source_recognition.preprocessing.physical import RAW_NAMES as NAMES, raw_numbers as physical_features

T = 500
TIME = np.arange(T) / 100.0


def window(seed=0, tone=None, common=0.0):
    """232 channels of noise; `tone` adds the same oscillation (Hz) to every channel with random gain; `common` adds a shared noise component."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(1, 232, T))
    if tone:
        x = x + 3 * rng.uniform(0.5, 1.5, size=(1, 232, 1)) * np.sin(2 * np.pi * tone * TIME)
    if common:
        x = x + common * rng.normal(size=(1, 1, T))
    return x.astype(np.float32)


def feats(x):
    return dict(zip(NAMES, physical_features(x)[0]))


class PhysicalFeatureTest(unittest.TestCase):
    def test_a_shared_oscillation_is_one_strong_component(self):
        noise, tone = feats(window()), feats(window(tone=1.5))
        self.assertGreater(tone["pca_share_1"], 0.5)
        self.assertGreater(tone["pca_share_1"], 10 * noise["pca_share_1"])
        self.assertLess(tone["pca_effective_rank"], noise["pca_effective_rank"])
        self.assertAlmostEqual(tone["link_coherence_min"], 1.0, delta=0.15)

    def test_periodicity_and_frequency_of_the_strongest_component(self):
        f = feats(window(tone=2.0))
        self.assertGreater(f["ac_peak"], 0.8)
        self.assertAlmostEqual(f["ac_lag"], 0.5, delta=0.06)           # one period of 2 Hz
        self.assertAlmostEqual(f["zero_crossings_per_s"], 4.0, delta=0.7)
        self.assertLess(f["permutation_entropy"], feats(window(common=5.0))["permutation_entropy"])

    def test_high_frequency_content_is_found(self):
        self.assertGreater(feats(window(tone=28.0))["share_20_35"], 0.8)
        self.assertGreater(feats(window(tone=42.0))["global_share_35_50"], 0.5)

    def test_decorrelation_time_and_frequency_variation(self):
        slow, noise = feats(window(tone=0.8)), feats(window(seed=1))
        self.assertGreater(slow["decorrelation_time"], 0.1)
        self.assertLess(feats(window(common=5.0, seed=2))["decorrelation_time"], 0.1)   # shared white noise forgets itself at once
        steady = feats(window(tone=3.0))
        self.assertLess(steady["stft_peak_freq_std"], noise["stft_peak_freq_std"])
        self.assertLess(steady["spectral_flux"], noise["spectral_flux"])

    def test_signal_size_and_level_do_not_matter(self):
        x = window(seed=3, tone=1.0)
        a, b = physical_features(x)[0], physical_features(7.0 * x + 3.0)[0]
        np.testing.assert_allclose(a, b, rtol=2e-3, atol=2e-3)

    def test_a_flat_window_gives_finite_numbers(self):
        self.assertTrue(np.isfinite(physical_features(np.full((1, 232, T), 1.68, dtype=np.float32))).all())


if __name__ == "__main__":
    unittest.main()
