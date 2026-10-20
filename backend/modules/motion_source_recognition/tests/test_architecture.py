"""The layers of the module import in one direction only:

    preprocessing  <  training  <  experiments  <  evaluation        (a folder may import from the ones to its left, never from the ones to its right)
    calibration.py and inference.py (the live module) use preprocessing and training only. Only the evaluation calls the live module (to show its answers).

    backend/modules/motion_source_recognition/.venv/bin/python -m unittest discover -s backend/modules/motion_source_recognition/tests -t .
"""
import ast
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
ORDER = {"preprocessing": 0, "training": 1, "experiments": 2, "evaluation": 3}
LIVE_MAY_USE = {"calibration": {"preprocessing"}, "inference": {"preprocessing", "training", "calibration"}}


def imported_parts(path: Path) -> set:
    """The first part of every relative import of the file, resolved to a top-level folder or root file of the module."""
    depth = len(path.relative_to(MODULE).parts) - 1   # 0 for a root file
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            up = node.level - 1
            if up > depth:
                continue   # leaves the module (backend.shared and the like)
            parts = list(path.relative_to(MODULE).parts[:-1])[:depth - up] + (node.module or "").split(".")
            first = [p for p in parts if p]
            if first:
                found.add(first[0])
            elif node.names:
                found.update(a.name for a in node.names)
    return found


class ArchitectureTest(unittest.TestCase):
    def test_pipeline_folders_only_import_to_the_left(self):
        problems = []
        for folder, rank in ORDER.items():
            for path in (MODULE / folder).glob("*.py"):
                for part in imported_parts(path):
                    if part in ORDER and ORDER[part] > rank:
                        problems.append(f"{path.relative_to(MODULE)} imports {part}")
                    if part == "inference" and folder != "evaluation":
                        problems.append(f"{path.relative_to(MODULE)} imports the live module")   # only the evaluation calls the detector, to show its answers
        self.assertEqual(problems, [])

    def test_the_live_module_does_not_depend_on_experiments_or_evaluation(self):
        problems = []
        for name, allowed in LIVE_MAY_USE.items():
            for part in imported_parts(MODULE / f"{name}.py"):
                if part in ORDER and part not in allowed:
                    problems.append(f"{name}.py imports {part}")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
