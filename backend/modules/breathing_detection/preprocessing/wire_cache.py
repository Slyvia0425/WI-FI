"""Resumable model-input cache for timestamp-aligned WiRe-Breath windows."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .temporal import INTERPOLATED_FEATURES, extract_interpolated_subcarriers
from .wire_breath import WireBreathDataset


WIRE_CACHE_VERSION = 1
WIRE_TIME_STEPS = 300


@dataclass(frozen=True)
class WireBreathCache:
    sequences: np.ndarray
    respiratory_references: np.ndarray
    weights: np.ndarray
    manifest: dict


def _paths(cache_dir: Path, count: int, partial: bool = False) -> dict[str, Path]:
    suffix = ".partial" if partial else ""
    stem = f"wire_breath_v{WIRE_CACHE_VERSION}_n{count}"
    return {
        "sequences": cache_dir / f"{stem}.sequences{suffix}.npy",
        "references": cache_dir / f"{stem}.references{suffix}.npy",
        "weights": cache_dir / f"{stem}.weights{suffix}.npy",
        "manifest": cache_dir / f"{stem}.manifest{suffix}.json",
    }


def load_wire_breath_cache(cache_dir: str | Path, count: int) -> WireBreathCache:
    paths = _paths(Path(cache_dir), count)
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"Complete WiRe-Breath cache not found for n={count}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("version") != WIRE_CACHE_VERSION:
        raise ValueError(f"Invalid WiRe-Breath cache manifest: {paths['manifest']}")
    sequences = np.load(paths["sequences"], mmap_mode="r")
    references = np.load(paths["references"], mmap_mode="r")
    weights = np.load(paths["weights"], mmap_mode="r")
    expected = (count, WIRE_TIME_STEPS, INTERPOLATED_FEATURES)
    if sequences.shape != expected:
        raise ValueError(f"Sequence cache shape mismatch: {sequences.shape}, expected {expected}")
    if references.shape != (count, WIRE_TIME_STEPS) or weights.shape != (count,):
        raise ValueError("WiRe-Breath reference or weight cache shape mismatch")
    if len(manifest.get("samples", [])) != count:
        raise ValueError("WiRe-Breath cache sample metadata count mismatch")
    return WireBreathCache(sequences, references, weights, manifest)


def build_wire_breath_cache(
    source: WireBreathDataset,
    cache_dir: str | Path,
    *,
    checkpoint_every: int = 100,
) -> WireBreathCache:
    """Build a float16 cache while retaining flagged windows with zero weight."""
    count = len(source)
    if source.window_points != WIRE_TIME_STEPS:
        raise ValueError(f"Cache requires {WIRE_TIME_STEPS} time points")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    final = _paths(cache_dir, count)
    if all(path.is_file() for path in final.values()):
        return load_wire_breath_cache(cache_dir, count)

    partial = _paths(cache_dir, count, partial=True)
    base_manifest = {
        "version": WIRE_CACHE_VERSION,
        "dataset": "WiRe-Breath",
        "representation": "timestamp_resampled_ordered_interpolation",
        "count": count,
        "shape": [count, WIRE_TIME_STEPS, INTERPOLATED_FEATURES],
        "sequence_dtype": "float16",
        "reference_dtype": "float16",
        "processed": 0,
        "complete": False,
        "samples": [],
    }
    if partial["manifest"].is_file():
        manifest = json.loads(partial["manifest"].read_text(encoding="utf-8"))
        for key in ("version", "count", "shape", "sequence_dtype", "reference_dtype"):
            if manifest.get(key) != base_manifest[key]:
                raise ValueError(f"Incompatible partial WiRe-Breath cache: {partial['manifest']}")
        sequences = np.load(partial["sequences"], mmap_mode="r+")
        references = np.load(partial["references"], mmap_mode="r+")
        weights = np.load(partial["weights"], mmap_mode="r+")
    else:
        manifest = base_manifest
        sequences = np.lib.format.open_memmap(
            partial["sequences"], mode="w+", dtype=np.float16,
            shape=(count, WIRE_TIME_STEPS, INTERPOLATED_FEATURES),
        )
        references = np.lib.format.open_memmap(
            partial["references"], mode="w+", dtype=np.float16,
            shape=(count, WIRE_TIME_STEPS),
        )
        weights = np.lib.format.open_memmap(
            partial["weights"], mode="w+", dtype=np.uint8, shape=(count,)
        )
        partial["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    start = int(manifest["processed"])
    if len(manifest["samples"]) != start:
        raise ValueError("Partial cache metadata does not match processed count")
    for index in range(start, count):
        sample = source[index]
        flags = list(sample.quality_flags)
        try:
            sequences[index] = extract_interpolated_subcarriers(sample.x).astype(np.float16)
            transform_ok = True
        except ValueError:
            sequences[index] = 0
            flags.append("REPRESENTATION_ERROR")
            transform_ok = False
        references[index] = sample.respiratory_reference.astype(np.float16)
        weights[index] = int(sample.assessable and transform_ok)
        manifest["samples"].append(
            {
                "index": index,
                "sample_id": sample.sample_id,
                "recording_id": sample.recording_id,
                "participant_id": sample.metadata["participant_id"],
                "room": sample.metadata["room"],
                "posture": sample.metadata["posture"],
                "environment": sample.metadata["environment"],
                "assessable": bool(weights[index]),
                "quality_flags": flags,
            }
        )
        if (index + 1) % checkpoint_every == 0 or index + 1 == count:
            sequences.flush()
            references.flush()
            weights.flush()
            manifest["processed"] = index + 1
            partial["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    del sequences, references, weights
    for key in ("sequences", "references", "weights"):
        os.replace(partial[key], final[key])
    manifest["complete"] = True
    partial["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(partial["manifest"], final["manifest"])
    return load_wire_breath_cache(cache_dir, count)


def subject_holdout_folds(manifest: dict) -> dict[str, dict[str, list[int]]]:
    """Return leakage-safe outer folds with one complete participant held out."""
    samples = manifest.get("samples", [])
    participants = sorted({sample["participant_id"] for sample in samples})
    folds: dict[str, dict[str, list[int]]] = {}
    for participant in participants:
        test = [sample["index"] for sample in samples if sample["participant_id"] == participant]
        train = [sample["index"] for sample in samples if sample["participant_id"] != participant]
        folds[f"subject_{participant}"] = {"train": train, "test": test}
    return folds
