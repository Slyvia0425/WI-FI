"""Reading windows from the HDF5 files of the dataset: one function for the whole module."""
import h5py
import numpy as np

from .config import H5_KEY, SAMPLE_SHAPE


def read_windows(paths: list[str]) -> np.ndarray:
    """The windows in the files, as float32 with shape (files, channels, time steps)."""
    out = np.empty((len(paths), SAMPLE_SHAPE[0], SAMPLE_SHAPE[1]), dtype=np.float32)
    for i, p in enumerate(paths):
        with h5py.File(p, "r") as f:
            out[i] = f[H5_KEY][...][:, :, 0]
    return out
