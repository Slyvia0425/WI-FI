"""Download and verify the B/C model release without replacing different local files."""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def download(url, destination, expected=None):
    if expected and destination.exists() and digest(destination) == expected:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(
        url, headers={"User-Agent": "WiFi-Sense-Model-Restore/1"}
    )
    with (
        urllib.request.urlopen(request, timeout=60) as response,
        partial.open("wb") as output,
    ):
        shutil.copyfileobj(response, output, length=1024**2)
    if expected and digest(partial) != expected:
        raise ValueError("Downloaded archive checksum mismatch: " + destination.name)
    partial.replace(destination)


def restore(archive, root, manifest, asset):
    expected = {
        name: value
        for name, value in manifest["files"].items()
        if value["asset"] == asset
    }
    root = root.resolve()
    with tarfile.open(archive, "r:gz") as package:
        members = package.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or set(names) != set(expected) | {
            "MODEL_ASSET_MANIFEST.json"
        }:
            raise ValueError("Unexpected archive file list")
        metadata = json.load(package.extractfile("MODEL_ASSET_MANIFEST.json"))
        if metadata["asset"] != asset or metadata["files"] != expected:
            raise ValueError("Archive manifest does not match the code manifest")
        # Check every destination before restoring anything from this archive.
        for name, value in expected.items():
            parts = PurePosixPath(name).parts
            target = root.joinpath(*parts)
            if (
                name.startswith("/")
                or ".." in parts
                or not target.resolve().is_relative_to(root)
            ):
                raise ValueError("Unsafe model destination")
            if target.is_symlink():
                raise ValueError("Model destination is a symlink: " + name)
            if target.exists() and digest(target) != value["sha256"]:
                raise FileExistsError(
                    "Existing file differs; use an empty --output-root: " + str(target)
                )
        restored = 0
        for member in members:
            if member.name == "MODEL_ASSET_MANIFEST.json":
                continue
            value = expected[member.name]
            if not member.isfile() or member.size != value["bytes"]:
                raise ValueError("Invalid model archive member: " + member.name)
            target = root.joinpath(*PurePosixPath(member.name).parts)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix="." + target.name + ".",
                suffix=".restore-partial",
                dir=target.parent,
                delete=False,
            ) as output:
                temporary = Path(output.name)
                with package.extractfile(member) as source:
                    shutil.copyfileobj(source, output, length=1024**2)
            try:
                if digest(temporary) != value["sha256"]:
                    raise ValueError("Model file checksum mismatch: " + member.name)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            restored += 1
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="Also restore all historical/resume models"
    )
    parser.add_argument(
        "--archives-dir", type=Path, default=ROOT / "model_artifacts/downloads"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument(
        "--offline", action="store_true", help="Use already downloaded release assets"
    )
    args = parser.parse_args()
    manifest = json.loads((ROOT / "model_artifacts/manifest.json").read_text())
    base = (
        "https://github.com/"
        + manifest["repository"]
        + "/releases/download/"
        + manifest["release_tag"]
        + "/"
    )
    sums = args.archives_dir / "SHA256SUMS"
    if not args.offline:
        download(base + "SHA256SUMS", sums)
    checksums = {
        line.split()[1]: line.split()[0]
        for line in sums.read_text().splitlines()
        if line.strip()
    }
    assets = ["module_bc_runtime_models.tar.gz"]
    if args.all:
        assets.append("module_bc_research_models.tar.gz")
    for asset in assets:
        path = args.archives_dir / asset
        if not args.offline:
            download(base + asset, path, checksums[asset])
        if digest(path) != checksums[asset]:
            raise ValueError("Archive checksum mismatch: " + asset)
        count = restore(path, args.output_root, manifest, asset)
        print(
            json.dumps(
                {
                    "asset": asset,
                    "verified_files": sum(
                        v["asset"] == asset for v in manifest["files"].values()
                    ),
                    "restored_files": count,
                }
            )
        )
    print(
        "Model files restored. The application model and routing policy were not changed."
    )


if __name__ == "__main__":
    main()
