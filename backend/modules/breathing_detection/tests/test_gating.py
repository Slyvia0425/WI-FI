import unittest

from backend.modules.breathing_detection.inference.gating import (
    GateConfig,
    GateInputs,
    GateReason,
    decide_breathing_status,
)


class BreathingGateTest(unittest.TestCase):
    def setUp(self):
        self.config = GateConfig(
            breathing_threshold=0.1,
            motion_threshold=0.8,
            fan_threshold=0.7,
            periodic_interference_threshold=0.75,
            temporal_instability_threshold=0.6,
        )

    def decide(self, **kwargs):
        return decide_breathing_status(GateInputs(**kwargs), self.config)

    def test_quality_gate_has_highest_priority(self):
        decision = self.decide(
            breathing_probability=0.99,
            quality_assessable=False,
            quality_flags=("FLAT_SIGNAL",),
            motion_probability=0.99,
        )
        self.assertEqual(decision.status, "not_assessable")
        self.assertEqual(decision.reason, GateReason.LOW_QUALITY)
        self.assertEqual(decision.quality_flags, ("FLAT_SIGNAL",))

    def test_external_gates_follow_declared_priority(self):
        decision = self.decide(
            breathing_probability=0.9,
            motion_probability=0.8,
            fan_probability=0.99,
            periodic_interference_risk=0.99,
        )
        self.assertEqual(decision.reason, GateReason.MOTION)
        self.assertEqual(decision.triggered_score, 0.8)

    def test_each_optional_gate_can_reject(self):
        cases = (
            ({"fan_probability": 0.7}, GateReason.FAN),
            (
                {"periodic_interference_risk": 0.75},
                GateReason.PERIODIC_INTERFERENCE,
            ),
            ({"temporal_instability": 0.6}, GateReason.TEMPORAL_INSTABILITY),
        )
        for extra, expected in cases:
            with self.subTest(reason=expected):
                decision = self.decide(breathing_probability=0.9, **extra)
                self.assertEqual(decision.status, "not_assessable")
                self.assertEqual(decision.reason, expected)

    def test_missing_external_evidence_does_not_reject(self):
        decision = self.decide(breathing_probability=0.8)
        self.assertEqual(decision.status, "breathing")
        self.assertEqual(decision.reason, GateReason.BREATHING_EVIDENCE)

    def test_frozen_threshold_selects_breathing_and_none(self):
        breathing = self.decide(breathing_probability=0.1)
        none = self.decide(breathing_probability=0.05)
        self.assertEqual(breathing.status, "breathing")
        self.assertEqual(none.status, "none")
        self.assertEqual(breathing.confidence, 0.0)
        self.assertAlmostEqual(none.confidence, 0.5)

    def test_frame_preserves_current_public_contract(self):
        frame = self.decide(
            breathing_probability=0.9, fan_probability=0.8
        ).to_frame(12)
        self.assertEqual(
            frame,
            {"t": 12.0, "status": "not_assessable", "confidence": 0.8},
        )

    def test_invalid_scores_are_rejected(self):
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GateInputs(breathing_probability=value)
        with self.assertRaises(TypeError):
            GateInputs(breathing_probability=True)


if __name__ == "__main__":
    unittest.main()
