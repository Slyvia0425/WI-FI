"""Inference only for the author's released seed42 spectral ExtraTrees model.

The spectral training runner consumes raw amplitudes. Its CNN window z-score
configuration is not applied to spectral features. No fitting occurs here.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn

from ..experiments.spectral import SpectralEnsemble, extract_spectral_features
from ..preprocessing.quality import check_quality

MODULE = Path(__file__).resolve().parents[1]
LABELS = ("jumping", "running", "seated-breathing", "walking", "wavinghand")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReleasedActivityModel:
    def __init__(self):
        self.config = json.loads((Path(__file__).parent / "frozen_model.json").read_text())
        if sklearn.__version__ != self.config["sklearn_version"]:
            raise RuntimeError("Released B requires scikit-learn==" + self.config["sklearn_version"])
        path = MODULE / self.config["checkpoint"]
        if digest(path) != self.config["checkpoint_sha256"]:
            raise ValueError("Released Module B checkpoint checksum mismatch")
        for name, expected in self.config["feature_sources_sha256"].items():
            if digest(MODULE / name) != expected:
                raise ValueError("Released B feature implementation changed: " + name)
        # Load only the pinned, author-published artifact after checksum verification.
        self.model = joblib.load(path)
        if not isinstance(self.model, SpectralEnsemble):
            raise ValueError("Released B artifact is not the expected spectral ensemble")
        if tuple(self.model.labels) != LABELS or self.model.feature_version != "csi-band-power-v1":
            raise ValueError("Released B label/feature version mismatch")
        if self.model.selector.n_features_in_ != 4195 or self.model.selector.k != 2800:
            raise ValueError("Released B spectral feature dimensions mismatch")
        for tree in self.model.extra_models:
            if list(tree.classes_) != list(range(5)):
                raise ValueError("Released B class output order mismatch")
            tree.n_jobs = 2  # Scheduling only; no model parameters are fitted or changed.

    def predict_many(self, amplitudes):
        arrays = [np.asarray(a, dtype=np.float32) for a in amplitudes]
        qualities = []
        for a in arrays:
            if a.shape != (232, 500) or not np.isfinite(a).all():
                raise ValueError("Released B integration requires finite HP raw amplitude [232,500]")
            q = check_quality(a, expected_feature_size=232, min_frames=20, max_abs_amplitude=100000.0)
            if not q.ok:
                raise ValueError("Released B input quality failed: " + ",".join(q.codes))
            qualities.append(q)
        features = extract_spectral_features(arrays)
        probabilities = self.model._probabilities_from_features(features)
        if probabilities.shape != (len(arrays), 5) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid released B probability output")
        if (
            (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(1), 1, atol=1e-6)
        ):
            raise ValueError("Released B probabilities must sum to one")
        results = []
        for p, q in zip(probabilities, qualities, strict=False):
            index = int(p.argmax())
            confidence = float(p[index])
            results.append(
                {
                    "activity": LABELS[index],
                    "activity_id": index,
                    "confidence": confidence,
                    "stable": confidence >= self.config["confidence_threshold"],
                    "probabilities": dict(zip(LABELS, map(float, p), strict=False)),
                    "stability": float(np.sort(p)[-1] - np.sort(p)[-2]),
                    "quality": q.flag,
                    "quality_codes": list(q.codes),
                    "model_version": self.config["model_version"],
                    "decision_note": (
                        "Top class among five trained activities; stable means confidence >=0.6, "
                        "not temporal stability. Fall is not a trained B class."
                    ),
                }
            )
        return results

    def predict(self, amplitude):
        return self.predict_many([amplitude])[0]
