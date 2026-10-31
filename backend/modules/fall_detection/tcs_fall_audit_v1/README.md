# TCS-Fall preliminary source audit

Read-only audit of the pinned official repository, its actual-fall-time workbook
and seven short demonstration recordings. No upstream training code is executed.
No continuous-model inputs, synthetic statistics or alarm metrics are produced.

Source: https://github.com/zhouziyu2050/TCS-Fall

Pinned commit: `3907c3244eb17770ff92e6d0e8575e9dced0f8f0`.

Downloaded sources and raw demos are under the ignored
`outputs/tcs_fall_audit_v1/upstream` directory. Provenance and the original Git
tree are retained beside it. All seven `.dat` files match their official Git blob
hashes. The workbook is unchanged and read with bundled Python/openpyxl.

From the repository root:

```bash
/Users/chenxiaohe/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 backend/modules/fall_detection/tcs_fall_audit_v1/audit.py --source-root backend/modules/fall_detection/outputs/tcs_fall_audit_v1/upstream --git-root backend/modules/fall_detection/outputs/tcs_fall_audit_v1/source_git --output backend/modules/fall_detection/reports/tcs_fall_audit_v1
```

Reports are `audit.json`, `annotations_extracted.json` and `report_zh.md`.
Annotations are not yet linked to raw continuous files. Header structure and
timestamps are checked across all demonstration packets; raw complex CSI is
decoded for the first 128 valid packets per file only, without scaling or antenna
permutation. These short demos do not supply continuous-alarm ground truth.

The complete raw download requires separate access. The user chose to download
through the author's Baidu Netdisk link and provide local archive paths. Inspect
`dataset/1.zip` and `TCS-Fall-main.zip` first; the public Git tree lacks the
`preprocess.ipynb` named in the official instructions. Do not assume the second
archive contains that missing notebook until it is actually inspected.

Intel5300 input cannot inherit HP's trained normalization through zero padding.
Future TCS training must remain an independent experiment with subject/recording
splits established before window generation. Existing HP weights remain intact.

## Downloaded local archives

The user supplied `/Users/chenxiaohe/Downloads/1.zip` and
`/Users/chenxiaohe/Downloads/TCS-Fall-main.zip`. The newer local audit is separate
from the earlier public-demo audit and is recorded in
`reports/tcs_fall_local_v1/`. It reads original ZIP members without extracting
the complete 2.4 GB raw archive. The source package is retained unchanged under
`outputs/tcs_fall_local_v1/source_package/TCS-Fall-main/`.

```bash
.venv/bin/python backend/modules/fall_detection/tcs_fall_audit_v1/audit_local_archives.py --raw-zip /Users/chenxiaohe/Downloads/1.zip --code-zip /Users/chenxiaohe/Downloads/TCS-Fall-main.zip --output backend/modules/fall_detection/reports/tcs_fall_local_v1_repeat --pilot-output backend/modules/fall_detection/outputs/tcs_fall_local_v1/pilot_repeat
.venv/bin/python backend/modules/fall_detection/tcs_fall_audit_v1/report_local_archives.py --report-dir backend/modules/fall_detection/reports/tcs_fall_local_v1_repeat --pilot-dir backend/modules/fall_detection/outputs/tcs_fall_local_v1/pilot_repeat
```

An audit exit status of 2 means real source-data integrity failures were recorded,
not that unsuccessful records were silently skipped. Existing report directories
are preserved; choose a new output path when rechecking a redownloaded archive.

Actual local result: 226 members attempted, 223 readable, three corrupt members
(`a_01_03_22.dat`, `a_01_03_23.dat`, `a_01_03_24.dat`). The errors were independently
confirmed with `unzip -t`. All 12 structurally annotated continuous recordings are
readable. Four activity-16/17 recordings are retained with unknown semantics and
excluded from labels and exposure-time denominators. Only nominal subject 01 is
available. The preprocessing notebook remains absent from the downloaded code ZIP.

All successful records have their complete native CSI decoded and RF-scaled in
bounded blocks. No filling, filtering, resampling, normalization fit, train/test
split, CNN inference, training, threshold change or alarm evaluation is performed.
One recording's native Tx0 amplitude `[155389,90]` and elapsed packet timestamps
are retained for inspection; they are not passed to HP model weights.
