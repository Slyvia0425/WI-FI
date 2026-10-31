"""Versioned, resumable cache for interpolated breathing sequences."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import BreathingDataset
from .temporal import INTERPOLATED_FEATURES, extract_interpolated_subcarriers


CACHE_VERSION = 1
TIME_STEPS = 300


@dataclass(frozen=True)
class TemporalCache:
    sequences: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    manifest: dict


def _cache_stem(split: str, count: int) -> str:
    return f"interpolated_v{CACHE_VERSION}_{split}_n{count}"


def _paths(cache_dir: Path, split: str, count: int, partial: bool = False) -> dict[str, Path]:
    suffix = ".partial" if partial else ""
    stem = _cache_stem(split, count)
    return {
        "sequences": cache_dir / f"{stem}.sequences{suffix}.npy",
        "labels": cache_dir / f"{stem}.labels{suffix}.npy",
        "weights": cache_dir / f"{stem}.weights{suffix}.npy",
        "manifest": cache_dir / f"{stem}.manifest{suffix}.json",
    }


def load_temporal_cache(cache_dir: str | Path, split: str, count: int) -> TemporalCache:
    paths = _paths(Path(cache_dir), split, count)
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"Complete temporal cache not found for {split} n={count}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("version") != CACHE_VERSION:
        raise ValueError(f"Invalid temporal cache manifest: {paths['manifest']}")
    sequences = np.load(paths["sequences"], mmap_mode="r")
    labels = np.load(paths["labels"], mmap_mode="r")
    weights = np.load(paths["weights"], mmap_mode="r")
    expected = (count, TIME_STEPS, INTERPOLATED_FEATURES)
    if sequences.shape != expected or labels.shape != (count,) or weights.shape != (count,):
        raise ValueError(f"Temporal cache shape mismatch: {sequences.shape}, expected {expected}")
    return TemporalCache(sequences, labels, weights, manifest)


def build_temporal_cache(
    source: BreathingDataset,
    cache_dir: str | Path,
    *,
    limit: int | None = None,
    checkpoint_every: int = 250,
) -> TemporalCache:
    """Build or resume a float16 cache without loading it into RAM."""
    count = min(len(source), limit) if limit is not None else len(source)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    final = _paths(cache_dir, source.split, count)
    if all(path.is_file() for path in final.values()):
        return load_temporal_cache(cache_dir, source.split, count)

    partial = _paths(cache_dir, source.split, count, partial=True)
    manifest = {
        "version": CACHE_VERSION,
        "representation": "ordered_interpolation",
        "split": source.split,
        "count": count,
        "shape": [count, TIME_STEPS, INTERPOLATED_FEATURES],
        "dtype": "float16",
        "processed": 0,
        "complete": False,
    }
    if partial["manifest"].is_file():
        existing = json.loads(partial["manifest"].read_text(encoding="utf-8"))
        compatible = all(existing.get(key) == manifest[key] for key in ("version", "split", "count", "shape", "dtype"))
        if not compatible:
            raise ValueError(f"Incompatible partial cache: {partial['manifest']}")
        manifest = existing
        sequences = np.load(partial["sequences"], mmap_mode="r+")
        labels = np.load(partial["labels"], mmap_mode="r+")
        weights = np.load(partial["weights"], mmap_mode="r+")
    else:
        sequences = np.lib.format.open_memmap(
            partial["sequences"], mode="w+", dtype=np.float16,
            shape=(count, TIME_STEPS, INTERPOLATED_FEATURES),
        )
        labels = np.lib.format.open_memmap(partial["labels"], mode="w+", dtype=np.uint8, shape=(count,))
        weights = np.lib.format.open_memmap(partial["weights"], mode="w+", dtype=np.uint8, shape=(count,))
        partial["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    start = int(manifest["processed"])
    for index in range(start, count):
        sample = source[index]
        labels[index] = sample.label
        try:
            if not sample.assessable:
                raise ValueError("not assessable")
            sequences[index] = extract_interpolated_subcarriers(sample.x).astype(np.float16)
            weights[index] = 1
        except ValueError:
            sequences[index] = 0
            weights[index] = 0
        if (index + 1) % checkpoint_every == 0 or index + 1 == count:
            sequences.flush()
            labels.flush()
            weights.flush()
            manifest["processed"] = index + 1
            partial["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    del sequences, labels, weights
    for key in ("sequences", "labels", "weights"):
        os.replace(partial[key], final[key])
    manifest["complete"] = True
    partial["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(partial["manifest"], final["manifest"])
    return load_temporal_cache(cache_dir, source.split, count)
