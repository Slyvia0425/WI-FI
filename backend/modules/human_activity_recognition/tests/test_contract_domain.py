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
        self.assertEqual(validate_probabilities({"jumping": 1.0}), {"jumping": 1.0})
        with self.assertRaises(InputValidationError):
            validate_probabilities({"jumping": 1.1})
        with self.assertRaises(InputValidationError):
            validate_probabilities({"jumping": 0.4, "running": 0.4})

    def test_prediction_uses_unknown_for_unstable_state(self):
        prediction = prediction_from_scores(
            {"jumping": 0.45, "running": 0.35, "seated-breathing": 0.1, "walking": 0.05, "wavinghand": 0.05},
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
                    activity=ActivityState.JUMPING,
                    confidence=0.9,
                    stable=True,
                    probabilities={
                        "jumping": 0.9,
                        "running": 0.03,
                        "seated-breathing": 0.03,
                        "walking": 0.02,
                        "wavinghand": 0.02,
                    },
                    stability=0.8,
                    quality="good",
                    model_version="test",
                ).to_frame(0)
            ]
        }
        self.assertEqual(validate_replay("human_activity_recognition", payload), [])

    def test_replay_contract_rejects_missing_required_field(self):
        problems = validate_replay("human_activity_recognition", {"frames": [{"t": 0, "activity": "jumping"}]})
        self.assertTrue(any("confidence" in problem for problem in problems))

    def test_module_value_domain_validation(self):
        problems = validate_frame_values(
            {"frames": [{"t": 0, "activity": "spinning", "confidence": 2.0, "stable": "yes"}]}
        )
        self.assertEqual(len(problems), 3)
