# Module C 数据预检查

状态：第一阶段尚未验收；CNN、训练、测试、推理尚未运行。

本地数据位置：`/Users/chenxiaohe/Desktop/nus-iss/CSI-Bench-inspection`。
本报告仅统计实际读取的 `fall_metadata.csv` 和两个 HDF5，不视为最新官方完整数据审计。

## 实际检查结果

metadata 共 6700 条，无重复 ID。Fall 2770（41.34%），Nonfall 3930（58.66%）。

|设备|Fall|Nonfall|合计|
|---|---:|---:|---:|
|HP|2493|3713|6206|
|ESP32|277|217|494|

Difficulty 原始字段：Easy 6082、Medium 124、Hard 494。这是 metadata 难度统计，不是官方测试集合数量。

字段对应：样本 ID 为 `id`，路径为 `file_path`，原始标签为 `label`，设备为 `device`，用户为 `user`，环境为 `environment`，会话为 `session`。另有 `subject`、`activity`、`is_multi_subject`、`Difficulty`。标签实际是字符串 Fall / Nonfall；官方数字映射尚未取得，不猜测数字含义。

路径解析示例：任务根目录 `FallDetection` + metadata 的 `./sub_Human/.../session_2168000__freq232.h5`。对应行的 ID 是 `Human_U07_Fall_E21_HP_2168000`，原始标签 Fall。所属集合目前未知，必须再与官方 split 文件匹配。

|本地样本|HDF5 key|shape|dtype|finite_ratio|
|---|---|---|---|---:|
|U07 / Fall / HP / session 2168000|CSI_amps|(232, 500, 1)|float32|1.0|
|U08 / Nonfall / ESP32 / session 10000|CSI_amps|(64, 500, 1)|float32|1.0|

两个文件没有根属性或 dataset 属性。轴语义尚未得到生成代码或数据规范的佐证，不将形状猜测写入加载器。以上形状只代表两个实测文件，不代表全部 6700 条记录。

## 缺失与未验证

- metadata 引用的 6700 个路径中，6698 个文件缺失，2 个可读。
- 未找到 `metadata/sample_metadata.csv`；本地只有另存的 `fall_metadata.csv`，尚未核实是否与目标版本一致。
- 未找到 `metadata/label_mapping.json` 和 `splits/*.json`。
- 所以官方 split 的结构、集合总数、类别比例、集合间 ID/内容重复、同源录制跨集合及难度测试集重叠均尚未验证。
- 全数据 shape、NaN/Inf 与内容哈希检查尚未完成；两个现有文件无 NaN/Inf。
- 目前不生成正式 manifest，避免把未知 split、未知轴语义或部分数据误标成正式审计通过。

## 重复运行

从仓库根目录执行（已准备的临时环境）：

```bash
/tmp/wifi-module-c-audit-venv/bin/python backend/modules/fall_detection/preflight_data.py \
  --inspection-root /Users/chenxiaohe/Desktop/nus-iss/CSI-Bench-inspection
```

重新建立环境时：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install h5py numpy
.venv/bin/python backend/modules/fall_detection/preflight_data.py \
  --inspection-root /path/to/CSI-Bench-inspection
```

输入是含 `fall_metadata.csv` 与 `FallDetection/` 的检查目录；输出是标准输出中的 JSON，不修改原始数据。解析失败直接报错；HDF5 错误逐文件明确记录。

正常预检查应复现当前数量和两个文件的形状。missing_files 非零、error 非空、finite_ratio 小于 1 均需要处理；即使为零，也不能单凭此脚本证明官方划分与轴语义正确。

下一步需要完整且同版本的 FallDetection（含 metadata、label_mapping、splits 和 HDF5），以及能佐证轴语义的官方加载/生成代码或说明。获得后先展示小样本证据，再实现正式 audit_data.py 并扫描全部记录。
