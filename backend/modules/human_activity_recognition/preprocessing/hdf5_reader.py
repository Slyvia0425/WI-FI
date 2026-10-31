"""HDF5 access is isolated here so other layers do not know file-format details."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..errors import DataNotFoundError, DataValidationError


def _h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise DataValidationError("h5py is required to read CSI-Bench HDF5 files") from exc
    return h5py


def list_datasets(path: Path) -> list[str]:
    def visit(name: str, obj: object) -> None:
        if hasattr(obj, "shape"):
            found.append(name)

    h5py = _h5py()
    found: list[str] = []
    with h5py.File(path, "r") as handle:
        handle.visititems(visit)
    return sorted(found)


def read_csi_array(path: Path, key_candidates: Iterable[str]) -> np.ndarray:
    if not path.exists():
        raise DataNotFoundError("HDF5 file not found", {"path": str(path)})
    h5py = _h5py()
    try:
        with h5py.File(path, "r") as handle:
            candidates = [str(key) for key in key_candidates]
            dataset = next((handle[key] for key in candidates if key in handle), None)
            if dataset is None:
                names = list_datasets(path)
                if len(names) != 1:
                    raise DataValidationError(
                        "no configured CSI key and the file does not contain exactly one dataset",
                        {"path": str(path), "keys": candidates, "datasets": names},
                    )
                dataset = handle[names[0]]
            array = np.asarray(dataset[()])
    except OSError as exc:
        raise DataValidationError("HDF5 file could not be read", {"path": str(path), "reason": str(exc)}) from exc
    if array.size == 0:
        raise DataValidationError("HDF5 dataset is empty", {"path": str(path)})
    if not np.isfinite(array).all():
        raise DataValidationError("HDF5 dataset contains non-finite values", {"path": str(path)})
    return array.astype(np.float32, copy=False)
