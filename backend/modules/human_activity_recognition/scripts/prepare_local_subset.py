#!/usr/bin/env python3
"""Extract only the csi-bench directories used by Module B.

The script never downloads the complete 81 GB dataset. It accepts an already downloaded Kaggle archive or an
already extracted dataset root and copies only:

  Multitask/HumanActivityRecognition/
  Multitask/sub_Human_h5/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DESTINATION = ROOT / "backend/modules/human_activity_recognition/local_data/csi_bench_module_b"
SELECTED_PREFIXES = (
    "Multitask/HumanActivityRecognition/",
    "Multitask/sub_Human_h5/",
)


def _inside(destination: Path, relative: str) -> Path:
    target = (destination / PurePosixPath(relative)).resolve()
    if target != destination.resolve() and destination.resolve() not in target.parents:
        raise ValueError(f"archive member escapes destination: {relative}")
    return target


def _prepare_from_root(source_root: Path, destination: Path, *, dry_run: bool) -> list[dict]:
    entries: list[dict] = []
    for prefix in SELECTED_PREFIXES:
        source = source_root / prefix.rstrip("/")
        if not source.exists():
            raise FileNotFoundError(f"required subset directory not found: {source}")
        for file in sorted(path for path in source.rglob("*") if path.is_file()):
            relative = file.relative_to(source_root).as_posix()
            entries.append({"path": relative, "bytes": file.stat().st_size})
            if not dry_run:
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target)
    return entries


def _prepare_from_archive(archive: Path, destination: Path, *, dry_run: bool) -> list[dict]:
    entries: list[dict] = []
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            if member.is_dir():
                continue
            normalised = member.filename.lstrip("./")
            if not normalised.startswith(SELECTED_PREFIXES):
                continue
            target = _inside(destination, normalised)
            entries.append({"path": normalised, "bytes": member.file_size})
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a Module B-only local csi-bench subset")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="Kaggle csi-bench ZIP already downloaded")
    source.add_argument("--source-root", type=Path, help="already extracted csi-bench root")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    destination = args.destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not args.dry_run:
        print(json.dumps({"ok": False, "error": f"destination is not empty: {destination}"}, indent=2))
        return 2
    if not args.dry_run:
        destination.mkdir(parents=True, exist_ok=True)
    try:
        if args.archive:
            entries = _prepare_from_archive(args.archive.expanduser().resolve(), destination, dry_run=args.dry_run)
        else:
            entries = _prepare_from_root(args.source_root.expanduser().resolve(), destination, dry_run=args.dry_run)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__, "message": str(exc)}, indent=2))
        return 2
    if not entries:
        print(json.dumps({"ok": False, "error": "no selected files were found"}, indent=2))
        return 3
    manifest = {
        "ok": True,
        "dry_run": args.dry_run,
        "created_at_unix": time.time(),
        "source": str(args.archive or args.source_root),
        "destination": str(destination),
        "selected_prefixes": list(SELECTED_PREFIXES),
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    if not args.dry_run:
        (destination / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: manifest[key] for key in ("ok", "dry_run", "destination", "file_count", "total_bytes")}, indent=2
        )
    )
    print(f"export HAR_DATA_ROOT={destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
