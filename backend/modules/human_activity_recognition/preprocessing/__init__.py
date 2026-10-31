"""Public preprocessing surface for Module B."""

from .dataset import SampleRecord, assert_no_split_leakage, discover_records, stable_sample_id, write_manifest
from .hdf5_reader import read_csi_array
from .labels import canonical_activity, read_label_mapping
from .normalization import FeatureStats
from .quality import QualityReport, check_quality, orient_window, require_valid

__all__ = [
    "FeatureStats",
    "QualityReport",
    "SampleRecord",
    "assert_no_split_leakage",
    "canonical_activity",
    "check_quality",
    "discover_records",
    "orient_window",
    "read_csi_array",
    "read_label_mapping",
    "require_valid",
    "stable_sample_id",
    "write_manifest",
]
