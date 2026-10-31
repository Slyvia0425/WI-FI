"""Fetch the original CSI-Bench files this module reads (task MotionSourceRecognition, official Kaggle version 12) into datasets/motion_source_recognition/data/.

    python3 datasets/motion_source_recognition/get_data.py                 # download and verify
    python3 datasets/motion_source_recognition/get_data.py --verify-only   # check the files already there
    python3 datasets/motion_source_recognition/get_data.py --sample        # a few files only, to see the format

Needs: pip install remotezip requests. More options: python3 datasets/download_csi_bench.py --help
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import download_csi_bench

if __name__ == "__main__":
    sys.argv[1:1] = ["--destination", str(HERE / ("sample_check" if "--sample" in sys.argv else "data")), "--task", "MotionSourceRecognition"]
    download_csi_bench.main()
