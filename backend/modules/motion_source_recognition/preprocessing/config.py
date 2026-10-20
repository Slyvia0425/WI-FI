"""Paths and constants for module A preprocessing.

Where the dataset is on your machine is never written in code. It is read, in this order, from
1. the environment variable CSI_DATA_ROOT, or
2. the git-ignored file backend/modules/motion_source_recognition/.env.local  (a line: CSI_DATA_ROOT=/your/path)
Copy .env.example to .env.local and edit it once.
"""
import json
import os
from pathlib import Path

from backend.shared.paths import outputs_dir

MODULE = "motion_source_recognition"
MODULE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = MODULE_DIR / ".env.local"

# Every tunable number of the module lives in config.json (next to this folder); the sections are read here once.
SETTINGS = json.loads((MODULE_DIR / "config.json").read_text())
DATA, SPLITS, FEATURES, TRAINING, EXPERIMENT_SETTINGS, EVALUATION, WEBSITE = (SETTINGS[k] for k in ("data", "splits", "features", "training", "experiments", "evaluation", "website"))

# The models of the module: folder name under outputs/models, label, algorithm, feature route; finalist = candidate of the model choice, optimistic = features designed after seeing the data
MODELS = {m["name"]: m for m in SETTINGS["models"]}
ROUTE_LABELS = SETTINGS["route_labels"]
FINALISTS = {name: m["label"] for name, m in MODELS.items() if m.get("finalist")}

CLASSES = DATA["classes"]  # order of metadata/label_mapping.json is Fan, Human, IRobot, Pet
SAMPLE_SHAPE = tuple(DATA["sample_shape"])  # channels x time steps (5 s at 100 Hz) x 1
H5_KEY = DATA["h5_key"]

# Generated files (git-ignored). Everything downstream reads these instead of the raw h5 files.
PREP_DIR = outputs_dir(MODULE) / "prep"
MODELS_DIR = outputs_dir(MODULE) / "models"
EXPERIMENTS_DIR = outputs_dir(MODULE) / "experiments"
EVALUATION_DIR = outputs_dir(MODULE) / "evaluation"


def _from_env_file() -> str | None:
    if not ENV_FILE.exists():
        return None
    for line in ENV_FILE.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "CSI_DATA_ROOT" and value.strip():
            return value.strip().strip("\"'")
    return None


def data_root() -> Path:
    """The MotionSourceRecognition folder of CSI-Bench on this machine."""
    value = os.environ.get("CSI_DATA_ROOT") or _from_env_file()
    if not value:
        raise SystemExit(
            "The dataset location is not set. Either export CSI_DATA_ROOT, or copy\n"
            f"  {MODULE_DIR / '.env.example'}\nto\n  {ENV_FILE}\nand put the path of your MotionSourceRecognition folder in it."
        )
    root = Path(value).expanduser()
    if not (root / "metadata" / "sample_metadata.csv").exists():
        raise SystemExit(f"{root} does not look like the MotionSourceRecognition folder (no metadata/sample_metadata.csv)")
    return root
