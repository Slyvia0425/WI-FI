"""Dataset discovery, stable ids, split guards and manifests."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..config import AppConfig
from ..errors import DataNotFoundError, DataValidationError, SplitLeakageError
from ..security import AccessPolicy
from .labels import canonical_activity

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "id": "test",
    "dev": "val",
}

META_FIELDS = {
    "sample_id": ("sample_id", "id", "record_id"),
    "path": ("path", "file", "file_path", "relative_path", "h5_path"),
    "session_id": ("session_id", "session", "recording_id"),
    "user_id": ("user_id", "user", "subject_id", "participant_id"),
    "environment_id": ("environment_id", "environment", "env", "room_id"),
    "device_id": ("device_id", "device", "hardware_id"),
    "activity": ("activity", "activity_name", "label", "class"),
    "split": ("split", "partition", "set"),
}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    path: Path
    session_id: str
    user_id: str
    environment_id: str
    device_id: str
    activity: str
    split: str = "unknown"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "path": str(self.path),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "environment_id": self.environment_id,
            "device_id": self.device_id,
            "activity": self.activity,
            "split": self.split,
        }


def stable_sample_id(*parts: object) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _pick(row: Mapping[str, str], names: Iterable[str]) -> str:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for name in names:
        if name in lowered and str(lowered[name]).strip():
            return str(lowered[name]).strip()
    return ""


def _split_name(value: str) -> str:
    return SPLIT_ALIASES.get(value.strip().lower(), "unknown")


def _path_ids(path: Path) -> dict[str, str]:
    ids = {"user_id": "unknown", "environment_id": "unknown", "device_id": "unknown", "session_id": path.stem}
    for part in path.parts:
        lowered = part.lower()
        if lowered.startswith("user_"):
            ids["user_id"] = part[5:] or "unknown"
        elif lowered.startswith("env_"):
            ids["environment_id"] = part[4:] or "unknown"
        elif lowered.startswith("device_"):
            ids["device_id"] = part[7:] or "unknown"
        elif "session" in lowered:
            ids["session_id"] = part
    return ids


def read_metadata(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _resolve_sample_path(
    *,
    relative: str,
    task_root: Path,
    metadata_path: Path,
    data_root: Path | None,
    sample_root: Path,
) -> Path | None:
    candidate = Path(relative)
    bases = [metadata_path.parent, task_root, sample_root]
    if data_root is not None:
        bases.append(data_root)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    for base in bases:
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    # Metadata may use a path relative to the dataset root while the H5 files
    # are kept in the shared sample root, e.g. Multitask/sub_Human_h5/...
    parts = candidate.parts
    for anchor in ("sub_Human_h5", "sub_Human_mat"):
        if anchor in parts:
            tail = Path(*parts[parts.index(anchor) + 1 :])
            resolved = (sample_root / tail).resolve()
            if resolved.exists():
                return resolved
    return None


def read_split_files(task_root: Path, split_dir: str) -> dict[str, str]:
    directory = task_root / split_dir
    mapping: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for name in (f"{split}_id.json", f"test_{split}.json" if split == "test" else ""):
            if not name:
                continue
            path = directory / name
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise DataValidationError("split file is not valid JSON", {"path": str(path)}) from exc
            if not isinstance(payload, list):
                raise DataValidationError("split file must contain a JSON list", {"path": str(path)})
            for value in payload:
                text = str(value)
                mapping[text] = split
                mapping[Path(text).name] = split
    return mapping


def discover_records(config: AppConfig, policy: AccessPolicy, *, actor: str = "system") -> list[SampleRecord]:
    if config.task_root is None:
        raise DataNotFoundError("dataset root is not configured", {"env_names": ["HAR_DATA_ROOT", "CSI_DATA_ROOT"]})
    task_root = policy.authorize_read(actor, config.task_root)
    if not task_root.exists():
        raise DataNotFoundError("task directory does not exist", {"path": str(task_root)})
    sample_root = (task_root / config.dataset.sample_root).resolve() if config.dataset.sample_root else task_root
    if not sample_root.exists():
        raise DataNotFoundError("shared sample directory does not exist", {"path": str(sample_root)})
    metadata_path = task_root / config.dataset.metadata_file
    split_map = read_split_files(task_root, config.dataset.split_dir)
    rows = read_metadata(policy.authorize_read(actor, metadata_path)) if metadata_path.exists() else []
    records: list[SampleRecord] = []
    if rows:
        for index, row in enumerate(rows):
            relative = _pick(row, META_FIELDS["path"])
            if not relative:
                continue
            resolved_sample = _resolve_sample_path(
                relative=relative,
                task_root=task_root,
                metadata_path=metadata_path,
                data_root=config.data_root,
                sample_root=sample_root,
            )
            if resolved_sample is None:
                raise DataNotFoundError("metadata sample path does not exist", {"path": relative})
            path = policy.authorize_read(actor, resolved_sample)
            activity = canonical_activity(_pick(row, META_FIELDS["activity"]), allow_decision_state=False).value
            if path.is_relative_to(task_root):
                source_relative = path.relative_to(task_root)
            elif path.is_relative_to(sample_root):
                source_relative = path.relative_to(sample_root)
            else:
                raise DataValidationError("sample is outside task and sample roots", {"path": str(path)})
            ids = _path_ids(source_relative)
            record_id = _pick(row, META_FIELDS["sample_id"]) or stable_sample_id(source_relative, index)
            declared_split = _split_name(_pick(row, META_FIELDS["split"]))
            records.append(
                SampleRecord(
                    sample_id=stable_sample_id(record_id, source_relative, index),
                    path=path,
                    session_id=_pick(row, META_FIELDS["session_id"]) or ids["session_id"],
                    user_id=_pick(row, META_FIELDS["user_id"]) or ids["user_id"],
                    environment_id=_pick(row, META_FIELDS["environment_id"]) or ids["environment_id"],
                    device_id=_pick(row, META_FIELDS["device_id"]) or ids["device_id"],
                    activity=activity,
                    split=(
                        declared_split
                        if declared_split != "unknown"
                        else split_map.get(record_id)
                        or split_map.get(Path(relative).name)
                        or split_map.get(relative)
                        or "unknown"
                    ),
                    metadata={str(k): str(v) for k, v in row.items()},
                )
            )
    if not records:
        for path in sorted(sample_root.glob(config.dataset.h5_glob)):
            resolved = policy.authorize_read(actor, path)
            relative_path = (
                resolved.relative_to(task_root)
                if resolved.is_relative_to(task_root)
                else resolved.relative_to(sample_root)
            )
            ids = _path_ids(relative_path)
            activity_text = next((part for part in relative_path.parts if part.lower().startswith("act_")), "")
            if not activity_text:
                raise DataValidationError("cannot infer activity from sample path", {"path": str(relative_path)})
            records.append(
                SampleRecord(
                    sample_id=stable_sample_id(relative_path),
                    path=resolved,
                    session_id=ids["session_id"],
                    user_id=ids["user_id"],
                    environment_id=ids["environment_id"],
                    device_id=ids["device_id"],
                    activity=canonical_activity(activity_text[4:], allow_decision_state=False).value,
                    split="unknown",
                    metadata={},
                )
            )
    if not records:
        raise DataNotFoundError("no HDF5 samples were discovered", {"path": str(task_root)})
    return records


def assert_no_split_leakage(
    records: Iterable[SampleRecord],
    *,
    group_fields: tuple[str, ...] = ("user_id", "environment_id", "device_id", "session_id"),
) -> None:
    groups: dict[str, set[str]] = {}
    for record in records:
        if record.split not in {"train", "val", "test"}:
            continue
        for field_name in group_fields:
            key = str(getattr(record, field_name))
            groups.setdefault(f"{field_name}:{key}", set()).add(record.split)
    leaked = {key: sorted(value) for key, value in groups.items() if len(value) > 1}
    if leaked:
        raise SplitLeakageError("group identifiers cross split boundaries", {"leaked": leaked})


def write_manifest(records: Iterable[SampleRecord], path: Path) -> int:
    rows = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return len(rows)


def iter_train_val_test(records: Iterable[SampleRecord]) -> Iterator[tuple[str, SampleRecord]]:
    for record in records:
        if record.split in {"train", "val", "test"}:
            yield record.split, record
