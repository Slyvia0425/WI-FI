import unittest
from types import SimpleNamespace
from backend.modules.breathing_detection.evaluation.export_csi_dashboard import select_examples


class Source:
    def __init__(self):
        self._rows = [dict(id=i, label=label) for i, label in [('z','empty'),('b','sleep'),('a','empty'),('c','sleep'),('d','empty')]]

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        row = self._rows[index]
        return SimpleNamespace(sample_id=row['id'], assessable=row['id'] != 'a')


class CSIExportTest(unittest.TestCase):
    def test_selection_is_lexicographic_and_quality_only(self):
        selected = select_examples(Source(), per_class=2)
        self.assertEqual([s.sample_id for s in selected[0]], ['d','z'])
        self.assertEqual([s.sample_id for s in selected[1]], ['b','c'])

    def test_insufficient_eligible_samples_raise(self):
        with self.assertRaises(ValueError):
            select_examples(Source(), per_class=3)
