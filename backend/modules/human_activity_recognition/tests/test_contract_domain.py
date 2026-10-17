from __future__ import annotations

import unittest

from backend.modules.human_activity_recognition.contract import (
    FRAME_FIELDS,
    OPTIONAL_FRAME_FIELDS,
    validate_frame_values,
)
from backend.modules.human_activity_recognition.domain import (
    ActivityState,
    Prediction,
    prediction_from_scores,
    validate_probabilities,
)
from backend.modules.human_activity_recognition.errors import InputValidationError
from backend.shared.contract import validate_replay


class DomainAndContractTest(unittest.TestCase):
    def test_required_contract_fields_are_unchanged(self):
        self.assertEqual(FRAME_FIELDS, ["activity", "confidence", "stable"])
        self.assertIn("probabilities", OPTIONAL_FRAME_FIELDS)

    def test_probabilities_validate_and_boundaries(self):
        self.assertEqual(validate_probabilities({"standing": 1.0}), {"standing": 1.0})
        with self.assertRaises(InputValidationError):
            validate_probabilities({"standing": 1.1})
        with self.assertRaises(InputValidationError):
            validate_probabilities({"standing": 0.4, "sitting": 0.4})

    def test_prediction_uses_unknown_for_unstable_state(self):
        prediction = prediction_from_scores(
            {"standing": 0.45, "sitting": 0.35, "walking": 0.15, "lying": 0.05},
            threshold=0.6,
            stability=0.1,
            model_version="test",
        )
        self.assertEqual(prediction.activity, ActivityState.UNKNOWN)
        self.assertFalse(prediction.stable)

    def test_required_replay_contract_accepts_optional_evidence(self):
        payload = {
            "frames": [
                Prediction(
                    activity=ActivityState.STANDING,
                    confidence=0.9,
                    stable=True,
                    probabilities={"standing": 0.9, "sitting": 0.04, "walking": 0.04, "lying": 0.02},
                    stability=0.8,
                    quality="good",
                    model_version="test",
                ).to_frame(0)
            ]
        }
        self.assertEqual(validate_replay("human_activity_recognition", payload), [])

    def test_replay_contract_rejects_missing_required_field(self):
        problems = validate_replay("human_activity_recognition", {"frames": [{"t": 0, "activity": "standing"}]})
        self.assertTrue(any("confidence" in problem for problem in problems))

    def test_module_value_domain_validation(self):
        problems = validate_frame_values(
            {"frames": [{"t": 0, "activity": "jumping", "confidence": 2.0, "stable": "yes"}]}
        )
        self.assertEqual(len(problems), 3)
