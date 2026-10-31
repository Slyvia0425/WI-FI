import unittest
from copy import deepcopy

from backend.modules.breathing_detection.evaluation.evaluate_fan_detection import (
    detection_metrics, on_off_difference, report_slices,
)


def row(label, probability, scenario='NLOS_FanOn', device='a'):
    return {'label': label, 'probability': probability, 'scenario': scenario,
            'device_id': device, 'unavailable_reasons': ['bad_quality'] if probability is None else []}


class FanDetectionTests(unittest.TestCase):
    def test_confusion_and_threshold_equality(self):
        rows = [row(1, .5), row(1, .2), row(0, .8), row(0, .1)]
        m = detection_metrics(rows, .5)
        self.assertEqual(m['confusion_matrix'], {'tp': 1, 'fn': 1, 'tn': 1, 'fp': 1})
        self.assertEqual(m['breathing_recall'], .5)
        self.assertEqual(m['empty_false_positive_rate'], .5)
        self.assertEqual(m['macro_f1'], .5)

    def test_positive_only_has_no_false_positive_estimate(self):
        m = detection_metrics([row(1, .9), row(1, .1)], .5)
        self.assertEqual(m['breathing_recall'], .5)
        self.assertIsNone(m['empty_false_positive_rate'])
        self.assertIsNone(m['specificity'])
        self.assertIsNone(m['macro_f1'])

    def test_negative_only_does_not_invent_recall(self):
        m = detection_metrics([row(0, .9), row(0, .1)], .5)
        self.assertIsNone(m['breathing_recall'])
        self.assertEqual(m['empty_false_positive_rate'], .5)
        self.assertIsNone(m['macro_f1'])

    def test_invalid_quality_retained_in_coverage_not_called_empty(self):
        m = detection_metrics([row(1, .9), row(1, None), row(0, None)], .5)
        self.assertEqual(m['available'], 1)
        self.assertEqual(m['coverage'], 1/3)
        self.assertEqual(m['breathing_recall'], 1.)
        self.assertEqual(m['positive_detection_fraction_of_all_labeled'], .5)
        self.assertEqual(m['confusion_matrix']['tn'], 0)
        self.assertEqual(m['unavailable_reasons']['bad_quality'], 2)

    def test_empty_group_metrics_remain_undefined(self):
        m = detection_metrics([], .5)
        for key in ('coverage', 'breathing_recall', 'empty_false_positive_rate', 'accuracy', 'macro_f1'):
            self.assertIsNone(m[key])

    def test_nonfinite_out_of_bounds_and_unknown_label_rejected(self):
        for p in (float('nan'), float('inf'), -1., 2.):
            with self.assertRaises(ValueError):
                detection_metrics([row(1, p)], .5)
        with self.assertRaises(ValueError):
            detection_metrics([row(2, .8)], .5)
        with self.assertRaises(ValueError):
            detection_metrics([], float('nan'))

    def test_comparison_uses_percentage_points_and_missing_stays_missing(self):
        result = on_off_difference({'breathing_recall': .8, 'empty_false_positive_rate': .1},
                                   {'breathing_recall': .9, 'empty_false_positive_rate': None})
        self.assertAlmostEqual(result['breathing_recall_on_minus_off_percentage_points'], 10.)
        self.assertIsNone(result['empty_false_positive_rate_on_minus_off_percentage_points'])

    def test_device_equal_weight_is_not_pooled_sample_weight(self):
        rows = [row(1, .9, 'LOS_FanOff', 'a') for _ in range(9)]
        rows += [row(1, .1, 'LOS_FanOff', 'b')]
        rows += [row(1, .9, 'LOS_FanOn', 'a'), row(1, .9, 'LOS_FanOn', 'b')]
        before = deepcopy(rows)
        report = report_slices(rows, .5)
        self.assertEqual(report['by_scenario']['LOS_FanOff']['breathing_recall'], .9)
        self.assertEqual(report['equal_device_scenario_means']['LOS_FanOff']['breathing_recall'], .5)
        self.assertEqual(report['on_off_comparisons']['LOS']['equal_device_matched']['breathing_recall']['on_minus_off_percentage_points'], 50.)
        self.assertIsNone(report['on_off_comparisons']['LOS']['equal_device_matched']['empty_false_positive_rate']['on_minus_off_percentage_points'])
        self.assertEqual(rows, before)

    def test_device_comparison_requires_relevant_class_in_both_conditions(self):
        rows = [row(0, .1, 'NLOS_FanOff', 'a'), row(0, .1, 'NLOS_FanOn', 'b')]
        report = report_slices(rows, .5)
        comparison = report['on_off_comparisons']['NLOS']['equal_device_matched']['empty_false_positive_rate']
        self.assertEqual(comparison['devices'], [])
        self.assertIsNone(comparison['on_minus_off_percentage_points'])


if __name__ == '__main__':
    unittest.main()
