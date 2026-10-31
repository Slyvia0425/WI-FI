"""Download and verify tasks of CSI-Bench (official Kaggle version 12) without downloading the whole dataset.

The official archive holds six tasks, 352,000 files, 81.5 GB. This script fetches only the tasks you name, by HTTP range requests on the
archive, so no Kaggle account and no copy of the archive are needed. Every file is checked against the size and the CRC32 checksum stored
in the archive, so the files are byte for byte the official ones; nothing is converted or cleaned.

    pip install remotezip requests
    python3 download_csi_bench.py --destination CSI-Bench-v12 --task MotionSourceRecognition
    python3 download_csi_bench.py --destination CSI-Bench-v12 --task FallDetection --task BreathingDetection
    python3 download_csi_bench.py --destination CSI-Bench-v12 --task Multitask --only Multitask/sub_Human_h5
    python3 download_csi_bench.py --destination CSI-Bench-v12 --task MotionSourceRecognition --verify-only
    python3 download_csi_bench.py --destination sample --all --sample

--task can be repeated; --all means every task (81.5 GB). --only <task>/<folder> (repeatable) narrows a download to some folders.
Each task goes to <destination>/<task>, the folder the module code reads
(for module A, CSI_DATA_ROOT is <destination>/MotionSourceRecognition). The dataset's own README.md goes to <destination>/README.md.
--verify-only downloads nothing: it checks the files already there. --sample fetches 3 files from each large subfolder of a task, plus its
small metadata files, for a quick look at the data format; it is not enough to run a pipeline. Writes download_receipt.json in <destination>.
"""
import argparse
import concurrent.futures
import json
import struct
import time
import zlib
from pathlib import Path

import requests
from remotezip import RemoteZip

URL = "https://www.kaggle.com/api/v1/datasets/download/guozhenjennzhu/csi-bench?datasetVersionNumber=12"
DATASET, VERSION = "guozhenjennzhu/csi-bench", 12
TASK_FILES = {                     # number of files per task in version 12, to notice a different version
    "MotionSourceRecognition": 60867, "FallDetection": 6708, "BreathingDetection": 100008,
    "Localization": 10740, "Multitask": 83354, "RawContinuousRecording": 90357,
}
README = "README.md"
SAMPLE_PER_SUBFOLDER, SAMPLE_MIN_FILES, SAMPLE_MAX_SUBFOLDERS, SAMPLE_META_BYTES = 3, 20, 8, 1024**2
MAX_GAP = 16 * 1024**2            # neighbouring files closer than this are fetched in one request
LOCAL_HEADER_EXTRA = 65535        # a ZIP local header can carry up to 65535 bytes of extra field


def official_index(tasks: list[str]) -> dict[str, list[dict]]:
    """Name, size, checksum and position of every file of the tasks, read from the archive's table of contents."""
    with RemoteZip(URL, timeout=120) as archive:
        entries = [dict(name=i.filename, size=i.file_size, compressed=i.compress_size, offset=i.header_offset, method=i.compress_type, crc=i.CRC)
                   for i in archive.infolist() if not i.is_dir() and (i.filename == README or i.filename.split("/")[0] in tasks)]
    index = {task: [e for e in entries if e["name"].startswith(task + "/")] for task in tasks}
    for task, files in index.items():
        if len(files) != TASK_FILES[task]:
            raise SystemExit(f"The archive lists {len(files)} files in {task}, expected {TASK_FILES[task]}. Is it still version {VERSION}?")
    index[README] = [e for e in entries if e["name"] == README]
    return index


def pick_sample(files: list[dict]) -> list[dict]:
    """Small metadata files, and a few evenly spread files from each large subfolder (for MotionSourceRecognition: 3 per class)."""
    chosen = [e for e in files if "/metadata/" in e["name"] and e["size"] <= SAMPLE_META_BYTES]
    groups: dict[str, list[dict]] = {}
    for e in files:
        parts = e["name"].split("/")
        if len(parts) > 2 and parts[1] not in ("metadata", "splits"):
            groups.setdefault(parts[1], []).append(e)
    for key in sorted(k for k, v in groups.items() if len(v) >= SAMPLE_MIN_FILES)[:SAMPLE_MAX_SUBFOLDERS]:
        ordered = sorted(groups[key], key=lambda e: e["name"])
        chosen += [ordered[i * len(ordered) // SAMPLE_PER_SUBFOLDER] for i in range(SAMPLE_PER_SUBFOLDER)]
    return chosen


def is_valid(path: Path, entry: dict) -> bool:
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    crc = 0
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            crc = zlib.crc32(block, crc)
    return crc == entry["crc"]


def fetch_block(block: list[dict], root: Path) -> int:
    if all(is_valid(root / e["name"], e) for e in block):
        return len(block)
    start, last = block[0]["offset"], block[-1]
    end = last["offset"] + 30 + len(last["name"].encode()) + LOCAL_HEADER_EXTRA + last["compressed"] - 1
    for attempt in range(5):
        try:
            with requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=120, stream=True) as response:
                response.raise_for_status()
                if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                    raise ValueError("The server did not honour the range request; refusing to download the whole dataset")
                chunk = response.content
            for e in block:
                pos = e["offset"] - start
                if chunk[pos:pos + 4] != b"PK\x03\x04":
                    raise ValueError("Invalid ZIP local header")
                name_length, extra_length = struct.unpack_from("<HH", chunk, pos + 26)
                data_start = pos + 30 + name_length + extra_length
                raw = chunk[data_start:data_start + e["compressed"]]
                if e["method"] == 8:
                    raw = zlib.decompress(raw, -15)
                elif e["method"] != 0:
                    raise ValueError("Unsupported ZIP compression")
                if len(raw) != e["size"] or zlib.crc32(raw) != e["crc"]:
                    raise ValueError(f"Size or checksum mismatch: {e['name']}")
                target = (root / e["name"]).resolve()
                if not target.is_relative_to(root):
                    raise ValueError("Unsafe path in the archive")
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(target.suffix + ".partial")
                partial.write_bytes(raw)
                partial.replace(target)
            return len(block)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def group_blocks(entries: list[dict]) -> list[list[dict]]:
    blocks: list[list[dict]] = []
    for e in sorted(entries, key=lambda e: e["offset"]):
        if not blocks or e["offset"] - blocks[-1][-1]["offset"] > MAX_GAP:
            blocks.append([])
        blocks[-1].append(e)
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--destination", type=Path, required=True, help="folder that will hold <task>/ for every task")
    parser.add_argument("--task", action="append", choices=sorted(TASK_FILES), default=[], help="a task to fetch; can be repeated")
    parser.add_argument("--only", action="append", default=[], metavar="TASK/FOLDER", help="only files under this path; can be repeated")
    parser.add_argument("--all", action="store_true", help="every task (81.5 GB)")
    parser.add_argument("--verify-only", action="store_true", help="check the files already there, download nothing")
    parser.add_argument("--sample", action="store_true", help="fetch a few files per large subfolder only")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    tasks = sorted(TASK_FILES) if args.all else sorted(set(args.task))
    if not tasks:
        parser.error("name at least one --task, or use --all")
    root = args.destination.resolve()
    index = official_index(tasks)
    chosen = {task: [e for e in index[task] if not args.only or any(e["name"].startswith(prefix.rstrip("/") + "/") for prefix in args.only)] for task in tasks}
    entries = [e for task in tasks for e in (pick_sample(chosen[task]) if args.sample else chosen[task])] + index[README]

    if args.verify_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            bad = [e["name"] for e, ok in zip(entries, pool.map(lambda e: is_valid(root / e["name"], e), entries)) if not ok]
        print(f"{len(entries) - len(bad)} of {len(entries)} files match the official version {VERSION} byte for byte")
        for name in bad[:20]:
            print("  differs or missing:", name)
        raise SystemExit(1 if bad else 0)

    root.mkdir(parents=True, exist_ok=True)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count in pool.map(lambda block: fetch_block(block, root), group_blocks(entries)):
            done += count
            print(f"verified {done}/{len(entries)} files", flush=True)
    receipt = {"source_url": URL, "dataset": DATASET, "version": VERSION, "tasks": tasks, "only": args.only, "files": len(entries),
               "bytes": sum(e["size"] for e in entries), "sample": args.sample, "verification": "size and CRC32 of every file against the archive",
               "downloaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (root / "download_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    for task in tasks:
        print(f"{task}: {root / task}")


if __name__ == "__main__":
    main()
