"""Verify bundled WiRe-Breath files and optionally copy them to the legacy data root.

Uses only the Python standard library. Never overwrites different existing data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "WiReBreath"
MANIFEST = HERE / "wire_manifest.json"
PROJECT = HERE.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entries() -> list[dict]:
    return [
        {"path": path.relative_to(SOURCE).as_posix(),
         "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(SOURCE.rglob("*")) if path.is_file()
    ]


def check_inventory(files: list[dict]) -> None:
    mats = [row for row in files if row["path"].endswith(".mat")]
    if len(mats) != 467:
        raise ValueError(f"Expected 467 MAT files, found {len(mats)}")
    for room, expected in (("Room1", 170), ("Room2", 297)):
        if sum(row["path"].startswith(room + "/") for row in mats) != expected:
            raise ValueError(f"Unexpected file count in {room}")


def install(files: list[dict], destination: Path) -> None:
    destination = destination.resolve()
    if destination == SOURCE or SOURCE in destination.parents or destination in SOURCE.parents:
        raise ValueError("Destination must be separate from the bundled source")
    # Accept both normal paths and the flat Windows-style extraction used locally.
    existing = {}
    if destination.exists():
        for path in destination.rglob("*.mat"):
            name = path.name.replace("\\", "/").split("/")[-1]
            if name in existing:
                raise ValueError(f"Duplicate segment already exists: {name}")
            existing[name] = path
    mats = [row for row in files if row["path"].endswith(".mat")]
    expected_names = {Path(row["path"]).name for row in mats}
    if set(existing) - expected_names:
        raise ValueError("Destination contains MAT files outside this release")
    # Preflight every collision before copying anything.
    for row in mats:
        current = existing.get(Path(row["path"]).name)
        if current is not None and sha256(current) != row["sha256"]:
            raise ValueError(f"Refusing to overwrite different data: {current}")
    copied = 0
    for row in mats:
        if Path(row["path"]).name in existing:
            continue
        target = destination / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / row["path"], target)
        if sha256(target) != row["sha256"]:
            raise ValueError(f"Copy verification failed: {target}")
        copied += 1
    print(f"Ready: {destination} ({len(mats)} MAT files; {copied} copied, {len(existing)} reused)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="copy verified MAT files to --destination")
    parser.add_argument("--destination", type=Path, default=PROJECT / "data/WiReBreath")
    parser.add_argument("--write-manifest", action="store_true", help="maintainer: create the initial manifest")
    args = parser.parse_args()
    actual = entries()
    check_inventory(actual)
    if args.write_manifest:
        if MANIFEST.exists():
            raise FileExistsError("Preserve the published manifest; refusing to replace it")
        payload = {
            "dataset": "WiRe-Breath", "release": "v1.0.0",
            "source": "https://github.com/Yunpeng-Ge/WiRe-Breath/releases/tag/v1.0.0",
            "license": "CC-BY-4.0", "mat_files": 467,
            "verification": "SHA-256 of the local original files used in this project; not an upstream signature",
            "files": actual,
        }
        MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    if actual != expected:
        raise ValueError("Bundled files differ from wire_manifest.json (missing, extra, or modified file)")
    print(f"Verified {len(actual)} files, including 467 MAT segments ({sum(r['bytes'] for r in actual):,} bytes)")
    if args.install:
        install(expected, args.destination)


if __name__ == "__main__":
    main()
