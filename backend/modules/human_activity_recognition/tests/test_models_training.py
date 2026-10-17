from __future__ import annotations

import unittest

import numpy as np
import torch

from backend.modules.human_activity_recognition.errors import ModelValidationError
from backend.modules.human_activity_recognition.experiments import synthetic_split_bundle
from backend.modules.human_activity_recognition.training import (
    ModelSpec,
    TrainingSettings,
    build_model,
    count_parameters,
    predict_logits,
    train_model,
)


class ModelTrainingTest(unittest.TestCase):
    def test_both_models_have_expected_output_shape_and_budget(self):
        for name in ("cnn_1d", "cnn_transformer"):
            spec = ModelSpec(name=name, feature_size=8, num_classes=4, hidden_dim=16, dropout=0.1, heads=4, layers=1)
            model = build_model(spec)
            output = model(torch.zeros(2, 8, 32))
            self.assertEqual(tuple(output.shape), (2, 4))
            self.assertGreater(count_parameters(model), 0)

    def test_parameter_budget_is_enforced(self):
        model = build_model(
            ModelSpec(name="cnn_1d", feature_size=8, num_classes=4, hidden_dim=16, dropout=0.1, heads=4, layers=1)
        )
        from backend.modules.human_activity_recognition.training.models import assert_parameter_budget

        with self.assertRaises(ModelValidationError):
            assert_parameter_budget(model, 1)

    def test_training_is_deterministic_for_fixed_seed(self):
        bundle = synthetic_split_bundle(feature_size=8, frames=24)
        spec = ModelSpec(name="cnn_1d", feature_size=8, num_classes=4, hidden_dim=16, dropout=0.0, heads=4, layers=1)
        settings = TrainingSettings(
            model=spec,
            batch_size=4,
            epochs=1,
            learning_rate=0.01,
            weight_decay=0.0,
            patience=1,
            max_parameters=1000000,
            seed=9,
        )
        first = train_model(
            train_windows=bundle.train.windows,
            train_labels=bundle.train.labels,
            validation_windows=bundle.validation.windows,
            validation_labels=bundle.validation.labels,
            settings=settings,
            transform=bundle.stats,
        )
        second = train_model(
            train_windows=bundle.train.windows,
            train_labels=bundle.train.labels,
            validation_windows=bundle.validation.windows,
            validation_labels=bundle.validation.labels,
            settings=settings,
            transform=bundle.stats,
        )
        first_logits = predict_logits(first.model, bundle.test.windows)
        second_logits = predict_logits(second.model, bundle.test.windows)
        np.testing.assert_allclose(first_logits, second_logits, rtol=1e-6, atol=1e-6)
