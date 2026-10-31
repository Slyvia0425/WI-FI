"""Run the pinned released B spectral model on one raw HP HDF5 segment."""

import argparse
import json
from pathlib import Path

from backend.modules.fall_detection.activity_reference import segment_identity

from ..preprocessing.hdf5_reader import read_csi_array
from .runtime import ReleasedActivityModel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("Choose a new output file")
    a = read_csi_array(args.file, ("CSI_amps",))
    if a.shape != (232, 500, 1):
        raise ValueError("Expected HP CSI_amps [232,500,1]")
    a = a[:, :, 0]
    model = ReleasedActivityModel()
    frame = {"t": 0.0, "input_id": segment_identity(a), "device": "HP", "sample_id": args.file.stem, **model.predict(a)}
    result = {
        "module": "human_activity_recognition",
        "mock": False,
        "model_version": model.config["model_version"],
        "provenance": {"trained_artifact_available": True, "checkpoint_sha256": model.config["checkpoint_sha256"]},
        "frames": [frame],
        "notice": "Five-class activity preference only; no trained Fall class or verified activity label",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
