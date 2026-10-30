import unittest
from copy import deepcopy

from backend.modules.breathing_detection.evaluation.export_fan_dashboard import select_fan_ids


class FanExportTests(unittest.TestCase):
    def test_selection_uses_ids_devices_and_availability_not_score_or_label_match(self):
        rows = [dict(sample_id='z', scenario='NLOS_FanOn', label=0, device_id='a', probability=.01),
                dict(sample_id='a', scenario='NLOS_FanOn', label=0, device_id='a', probability=.99),
                dict(sample_id='b', scenario='NLOS_FanOn', label=0, device_id='b', probability=.5)]
        before = deepcopy(rows)
        ids = select_fan_ids(rows)
        self.assertEqual(ids[('NLOS_FanOn', 0, 'a')], 'a')
        rows[1]['probability'] = .001
        self.assertEqual(select_fan_ids(rows), ids)
        self.assertEqual(before[0], rows[0])

    def test_missing_class_is_not_fabricated_and_unavailable_not_selected(self):
        rows = [dict(sample_id='a', scenario='LOS_FanOn', label=1, device_id='a', probability=None),
                dict(sample_id='b', scenario='LOS_FanOn', label=1, device_id='a', probability=.9)]
        result = select_fan_ids(rows)
        self.assertEqual(list(result.values()), ['b'])
        self.assertTrue(all(key[1] == 1 for key in result))
