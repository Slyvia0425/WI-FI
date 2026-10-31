# Fall detection (backend)

Output: fall or non-fall score and alert state.

Everything for this module on the backend lives in this folder. Work on your own branch and edit only here.

| Folder | Put here |
|---|---|
| `preprocessing/` | Data reading, quality checks, splits and features. |
| `training/` | Model definitions and training scripts. |
| `experiments/` | Model comparison, ablations and threshold studies. |
| `evaluation/` | Metrics, confusion matrices, error analysis, and the export of outputs/replay.json. |
| `outputs/` | Generated files: `replay.json` (read by the server), checkpoints, figures. Large files stay out of git. |
| `api.py` | What the server returns for this module |
| `contract.py` | The frame fields this module promises |
| `requirements.txt` | Optional, packages only this module needs |

Frontend counterpart: `frontend/src/modules/fall_detection/`.
Export a result for the website with `backend.shared.export.write_replay("fall_detection", payload, bundle=True)`. One call updates the backend output and the copy bundled with the website.
Develop alone with `python -m backend.api --only fall_detection`.

## Frame fields

Defined in `contract.py`, besides `t` (seconds).

| Field | Meaning |
|---|---|
| `fall_score` | 0 to 1 |
| `state` | normal, suspected or alert |

## Module C 第一阶段 数据下载与审计

此阶段只核查完整分段、标签和官方划分。尚未实现 Dataset、CNN、训练或推理；网站示例评分不属于本次实验结果。

从仓库根目录运行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/modules/fall_detection/requirements.txt

# 从官方 Kaggle v12 ZIP 只取 FallDetection，并核对每个文件 CRC/大小。
# 数据放仓库外；已有且 CRC 正确的文件可复用。
.venv/bin/python backend/modules/fall_detection/download_data.py \
  --destination /Users/chenxiaohe/Documents/CSI-Bench-v12

# --data-root 必须指向含 metadata/、splits/、sub_Human/ 的任务目录。
.venv/bin/python backend/modules/fall_detection/audit_data.py \
  --data-root /Users/chenxiaohe/Documents/CSI-Bench-v12/FallDetection \
  --data-version kaggle-guozhenjennzhu-csi-bench-v12 \
  --output-dir backend/modules/fall_detection/reports
```

`download_data.py` 输入目标目录，输出数据及父目录下载凭据、ZIP 索引；需要网络。`audit_data.py` 输入实际数据目录，输出：

- `reports/sample_manifest.csv`：一条 metadata 记录一行；split 用分号保存所有成员关系。
- `reports/data_audit.md`：集合、设备、类别、形状、重复统计。
- `reports/audit_details.json`：异常、内容重复组、来源和版本哈希。
- `reports/hdf5_inventory.json`：每个文件的 dataset key、形状、dtype 和属性。
- `reports/inspection_notes.md`：本次人工小样本检查、官方代码证据与轴语义限制。

正常情况应完整扫描 6700 条，读取错误、未匹配标签、非有限数值等必须为零。遇到关键异常仍生成报告，再以退出码 2 结束；不会自动跳过坏数据或另做划分。缺失官方 metadata/split 会直接报错。重复内容是否构成泄漏需结合来源检查；报告不自动删除任何样本。

原始标签字符串 Fall 对应官方数字 0、Nonfall 对应官方数字 1；内部 binary_label 则为 Fall=1、Nonfall=0。官方测试难度子集与 test_id 重叠是预期成员关系，不能相加当作更多独立测试样本。

旧 `preflight_data.py` 和 `reports/data_preflight.md` 只记录此前仅有两个本地样本时的检查，不代表本次完整 v12 审计结果。当前以 `data_audit.md` 为准。

训练、验证阈值选择、测试评价和单文件推理命令：尚未实现／尚未运行，后续阶段再提供。

## 第二阶段 完整分段 Dataset

输入适配基于官方加载器的操作约定 `(频域特征,500,单例维)`；物理子载波编号和链路含义仍未确认。代码只执行 `data[:,:,0]`，不 reshape、不切窗、不重采样。HP 有 232 个原始特征，ESP32 有 64 个；两设备特征位置不宣称物理对齐。

对每台设备、每个原始特征，用该设备 train_id 的全部时间点计算 population mean/std：`z=(x-mean)/std`。验证、测试和后续推理复用保存的参数。标准差小于 1e-8 时 scale=1，保留并记录该特征；没有删除恒定位置。只在归一化完成后把 ESP32 右侧补 168 行零，因此填充不参与统计。参数是逐设备的，推理必须提供已知设备，未知设备会报错。

```bash
# 使用上面的 .venv 安装 requirements 后，从仓库根目录执行：
.venv/bin/python backend/modules/fall_detection/preprocess.py \
  --manifest backend/modules/fall_detection/reports/sample_manifest.csv \
  --output backend/modules/fall_detection/outputs/preprocess_v1.json

.venv/bin/python backend/modules/fall_detection/check_dataset.py \
  --manifest backend/modules/fall_detection/reports/sample_manifest.csv \
  --config backend/modules/fall_detection/outputs/preprocess_v1.json \
  --output-dir backend/modules/fall_detection/reports/stage2
```

第一个命令只读取训练 HDF5 估计参数，保存归一化数值、训练 ID、manifest 哈希和适配配置；不会拿验证/测试计算参数。第二个命令在独立进程复用 JSON，扫描三大集合的有限数值和填充，核对训练输出的整体均值/标准差，保存实际检查 JSON 与曲线。测试数据只做输入质量检查，不选择归一化方案、模型或阈值。

```python
from backend.modules.fall_detection.dataset import FallDataset
from torch.utils.data import DataLoader

dataset = FallDataset(
    'backend/modules/fall_detection/reports/sample_manifest.csv',
    'backend/modules/fall_detection/outputs/preprocess_v1.json',
    'train_id',  # val_id / test_id / test_easy / test_medium / test_hard
)
sample = dataset[0]
batch = next(iter(DataLoader(dataset, batch_size=32, shuffle=True)))
print(sample['x'].shape, sample['y'])  # [232,500] and scalar float32 0 or 1
print(batch['x'].shape, batch['y'].shape)  # [32,232,500] and [32]
```

Dataset 返回字典：`x` 为 float32，`y` 为 float32 标量；`feature_mask` 标明原文件存在的位置，`variable_feature_mask` 标明其中在训练中有变化的位置；`valid_features` 只是原文件特征数（HP=232、ESP32=64），不代表物理有效子载波数量；另有 `sample_id` 和 `device`。ESP32 训练数据中 12 个恒定位置保留在前 64 行，后 168 行才是填充。普通 CNN 不会自动使用掩码，后续模型需明确处理方式。

检查 batch 是特意包含两类、两设备的展示 batch，不是随机平衡采样器，也不改变训练分布。全部信号图使用训练样本，横轴为采样点，不猜测秒数；曲线只能检查读数与预处理，不能证明分类能力。

重复内容及设备/难度混杂风险继续存在。官方划分未改动，也没有去重；逐设备归一化和补零不等于跨设备校准，模型可能从填充模式识别设备。详见 reports/stage2/stage2_report.md。

## 第三阶段 最小 1D-CNN 与训练流程检查

`models/cnn.py` 定义 `FallCNN`，输入 `[B,232,500]`，输出 `[B,1]` 原始 logit。结构为 Conv1d(232,32,5,padding=2) → ReLU → MaxPool1d(2) → Conv1d(32,64,3,padding=1) → ReLU → MaxPool1d(2) → AdaptiveAvgPool1d(1) → Flatten → Dropout(0.3) → Linear(64,1)。共 43425 个可训练参数。没有 Sigmoid 层，损失前只用 `squeeze(-1)` 将 `[B,1]` 对齐到标签 `[B]`；查看评分时才调用 sigmoid。

从仓库根目录，完成上述依赖安装及第二阶段参数拟合后执行：

```bash
.venv/bin/python backend/modules/fall_detection/check_cnn.py \
  --manifest backend/modules/fall_detection/reports/sample_manifest.csv \
  --preprocess-config backend/modules/fall_detection/outputs/preprocess_v1.json \
  --output-dir backend/modules/fall_detection/reports/stage3 \
  --seed 42 --max-overfit-steps 500
```

输入为审计清单及固定预处理 JSON；输出为 stage3/cnn_check.json、overfit_history.csv、overfit_curve.png。脚本只从官方 train_id 取样；没有验证/测试评价、模型选择、阈值优化、正式训练或模型 checkpoint。运行时打印实际逐层形状和每个检查的进度。

- 冒烟：64 个完整分段，每个 device/label 组合 16 个；batch=32，2 轮，共 4 次 Adam 更新，Dropout=0.3。
- 过拟合：重新初始化模型和 Adam，16 个固定完整分段，每个组合 4 个；batch=16、Dropout=0，每次更新后在同一 16 条上 eval。预设通过条件为连续 5 次正确率 100% 且 BCE<0.02，上限 500 次更新；不通过则返回退出码 2，并保留实际结果。
- 两项检查：seed=42、CPU、2 线程、Adam lr=0.001、BCEWithLogitsLoss。诊断子集两类平衡，所以 pos_weight=1；这不是正式训练集类别权重的决定。
- 每次更新检查有限损失、所有参数梯度和参数数值；冒烟另外确认首层权重发生改变。评价使用 model.eval() 与 torch.no_grad()。
- 子集按固定种子、设备/标签分层抽取，内部内容哈希不重复；不修改官方清单，也不去除官方集合间已有的重复风险。两个诊断子集允许交叠，因它们不是独立评估集。
- 模型直接使用已填零的 x，不另外消费 feature_mask/variable_feature_mask；恒定原始位置保留，填充模式仍可能透露设备。

本次实测：冒烟通过；过拟合第 44 次更新通过，16/16 正确、BCE=0.011374859。具体解释见 reports/stage3/stage3_report.md。这些只证明固定训练样本可拟合，不是泛化指标；正式训练尚未运行。

## 第四阶段 正式训练与验证（不评价测试集）

`train.py` 只构建官方 train_id/val_id 的 Dataset，不构建 test Dataset。全量训练使用种子42、batch size 32、Adam lr=0.001、Dropout=0.3，最多100轮。每轮遍历4690条训练记录（147批，最后18条），用训练样本梯度更新参数；再用eval/no_grad遍历1005条验证记录（32批，最后13条）。不丢弃末批，不重新拟合预处理参数。

训练集有Fall=1922、Nonfall=2768，所以 BCEWithLogitsLoss 的 `pos_weight=2768/1922=1.4401664932362124`，只加权正类（Fall）的损失项，不是重采样。训练与验证均使用这一固定加权损失，按每批样本数加权汇总；正式训练损失和第三阶段未加权诊断损失不能直接比较。

验证 Precision、Recall、F1 使用固定阈值0.5（等价于logit>=0），分母为零时记0；它们只作为学习过程记录。本阶段不优化阈值。checkpoint只由严格最低验证损失选择：改善即保存、等待计数清零；相等也算未改善，连续10轮未改善就早停。早停时保留的best.pt可能来自更早轮次，不能把最后轮次当作最佳模型。

从仓库根目录，在已有依赖和第二阶段参数的基础上运行：

```bash
.venv/bin/python backend/modules/fall_detection/train.py \
  --manifest backend/modules/fall_detection/reports/sample_manifest.csv \
  --preprocess-config backend/modules/fall_detection/outputs/preprocess_v1.json \
  --audit-details backend/modules/fall_detection/reports/audit_details.json \
  --checkpoint-dir backend/modules/fall_detection/outputs/cnn_seed42_mps_v1 \
  --report-dir backend/modules/fall_detection/reports/stage4_seed42_mps_v1 \
  --device mps --seed 42 --max-epochs 100 --patience 10 \
  --batch-size 32 --learning-rate 0.001
```

`--device mps` 使用Apple GPU；其他机器可改 `--device cpu`，并使用新的输出目录。固定种子不保证跨硬件/软件版本逐位相同。为保护已有结果，检测到相同目录内已有best.pt或history.csv会拒绝覆盖；重复实验请指定新目录。

输入：审计manifest、固定预处理JSON、审计版本信息。输出：

- `outputs/cnn_seed42_mps_v1/best.pt`：最佳权重、模型配置、完整预处理与归一化参数、标签映射、种子、最佳轮次、数据版本/哈希、train/val ID、优化器状态与验证指标。权重保存为CPU张量以便重载。selected_threshold保持None，尚未选择最终阈值。
- `reports/stage4_seed42_mps_v1/history.csv`：每轮损失、验证Precision/Recall/F1及混淆计数、等待轮数、耗时。
- 同目录的 `run_config.json`、`progress.json`、`training_summary.json`：运行设置、进度、最终早停原因与最佳轮次。
- `best_validation_predictions.csv`：最佳模型的验证评分，供后续阶段使用；当前没有据此选择阈值。
- `training_curves.png`：训练/验证损失与固定阈值验证指标曲线。

每次最佳权重先写临时文件再替换，避免半写checkpoint。非有限损失、梯度或覆盖不全会报错，不静默跳样本。本次先进行CPU尝试，因每轮约56秒而中止并记录在reports/stage4_seed42_v1/interrupted_run.json；随后以相同种子从头初始化MPS实验。CPU尝试不是最终实验，切换依据为执行速度。

正常结果是逐轮打印EPOCH、更新best.pt，达到早停或100轮后打印DONE并生成曲线。单轮验证损失上升很常见；持续训练损失下降而验证损失上升提示可能过拟合。训练启用Dropout、验证关闭，且训练损失汇总了更新过程中的模型，因此两条曲线不能简单按高低判断数据错误。

已知官方train/val内容重复和设备混杂风险仍保留并披露。验证结果只表示当前官方划分上的表现；不能称为完全独立泛化评价。测试集未加载/评价，最终阈值未选择。

本次第四阶段最终实测：MPS正式实验共44轮，在第35—44轮连续10轮不改善后早停；最佳为第34轮。最佳加权验证loss=0.142527366，固定0.5下Fall Precision=93.0804%、Recall=94.9886%、F1=94.0248%，TP/FP/FN/TN=417/31/22/535。完整报告见 reports/stage4_seed42_mps_v1/stage4_report.md。以上都是验证结果，测试集没有评价。

## 第五阶段 验证阈值冻结、独立评价与单文件推理

`freeze_model.py` 只用第34轮最佳模型的验证预测选择最大Fall F1，精确F1并列时优先Recall，再按更高阈值确定唯一结果。候选阈值使用相邻不同评分的中点及0，覆盖所有可能成为最优的非空正类预测集合；验证集有Fall时，全部判Nonfall的F1=0不可能胜出。相同评分作为整体处理，不把它们拆开。中点避免直接把阈值放在某条样本的评分上。没有使用测试评分。

本次冻结阈值为 **0.38938893377780914**，判断规则是 `fall_score >= threshold`。CPU是冻结后的评分后端；复用原来的模型与归一化参数，没有重新训练。分数不是校准后的真实事故概率。

从仓库根目录运行（先按前文创建.venv并安装requirements）：

```bash
# 1. 只用验证集冻结；现有frozen.pt/frozen.json会拒绝覆盖。
.venv/bin/python backend/modules/fall_detection/freeze_model.py \
  --checkpoint backend/modules/fall_detection/outputs/cnn_seed42_mps_v1/best.pt \
  --manifest backend/modules/fall_detection/reports/sample_manifest.csv \
  --output-checkpoint backend/modules/fall_detection/outputs/cnn_seed42_mps_v1/frozen.pt \
  --report-dir backend/modules/fall_detection/reports/stage5

# 2. 新Python进程，先在验证集检查冻结模型重载。
.venv/bin/python backend/modules/fall_detection/evaluate.py \
  --checkpoint backend/modules/fall_detection/outputs/cnn_seed42_mps_v1/frozen.pt \
  --manifest backend/modules/fall_detection/reports/sample_manifest.csv \
  --splits val_id \
  --output-dir backend/modules/fall_detection/reports/stage5/reload_validation

# 3. 配置已冻结后，再评价官方测试集合；此命令不会重新选阈值。
.venv/bin/python backend/modules/fall_detection/evaluate.py \
  --checkpoint backend/modules/fall_detection/outputs/cnn_seed42_mps_v1/frozen.pt \
  --manifest backend/modules/fall_detection/reports/sample_manifest.csv \
  --splits test_id,test_easy,test_medium,test_hard \
  --output-dir backend/modules/fall_detection/reports/stage5/official_test

# 4. 一个完整HDF5独立推理；无须manifest或外部归一化JSON。
# --device是采集设备HP/ESP32，不是计算后端；本阶段推理使用CPU。
.venv/bin/python backend/modules/fall_detection/infer.py \
  --checkpoint backend/modules/fall_detection/outputs/cnn_seed42_mps_v1/frozen.pt \
  --file /path/to/complete_segment.h5 \
  --device HP
```

评价输出目录已有evaluation.json时拒绝覆盖，重跑请换目录。infer.py可加 `--output /path/to/result.json` 保存结果。模型部署需要同时保留frozen.pt和同名frozen.json；后者包含checkpoint SHA256与冻结记录，脚本检查它们的一致性。原best.pt保持不变，正式训练不需重跑。

脚本输入输出：

- `freeze_model.py`：输入best.pt+manifest，只读取val_id信号；输出frozen.pt/frozen.json、所有候选阈值与验证预测CSV、threshold_selection.json。模型配置、预处理/归一化、标签映射、训练种子、最佳轮次、数据版本与划分标识均随checkpoint保存。
- `evaluate.py`：输入冻结checkpoint+manifest+集合名；输出各集合/设备的样本数、类别数、Fall Precision/Recall/F1、TP/FP/FN/TN，以及逐样本评分CSV。阈值不可通过命令行覆盖，不训练、不选择阈值。test_id的预测可直接复用于嵌套难度子集，保持评分完全一致。
- `infer.py`：输入冻结checkpoint+一个HDF5+明确的采集设备；直接调用checkpoint内的相同Preprocessor，输出评分、冻结阈值、Fall/Nonfall判断和模型版本。没有标签输入，也不依赖train.py内存。未知设备、形状不符、非有限数值或冻结文件哈希不符会明确报错。
- `inference_runtime.py`：三者共享的模型重载与评分/统计实现；不导入train.py。

正常重载应在合理浮点误差内复现评分；实际新进程CPU验证评分与冻结前最大差异0，CPU与训练时MPS最大差异4.768e-7。文件哈希不一致、预处理manifest不匹配、漏读记录、NaN/Inf均应停止检查。全套实际结果及风险说明见reports/stage5/stage5_report.md。

官方train/test有两组内容重复、train/val有一组，设备与难度也混杂；本次保留官方结果，未按测试表现修改阈值或去重。这是完整分段分类，不是连续事件检测：没有滑动窗口、LSTM、报警规则、事件召回或每小时误报评价。

## ESP32误报的只读诊断

详见 reports/esp32_diagnosis/esp32_diagnosis.md。模型/阈值没有变化，没有重新训练。固定阈值下ESP32在训练/验证/测试的Nonfall误报率分别为134/152=88.16%、26/30=86.67%、29/35=82.86%；问题已存在于训练记录上。

核查全部6700条原始内容哈希、有限值及补零；以独立sum/sum_sq方法重新核算训练均值/std，与checkpoint值误差约1e-14。所有新增填充全零，12个ESP32原始恒定位置在全部494条记录也为零。补零实现通过不代表跨设备特征位置已物理对齐；数据/全局目标不均衡、共享特征适配仍是设计假设，没有控制实验定因。

```bash
.venv/bin/python backend/modules/fall_detection/diagnose_esp32.py \
  --checkpoint backend/modules/fall_detection/outputs/cnn_seed42_mps_v1/frozen.pt \
  --manifest backend/modules/fall_detection/reports/sample_manifest.csv \
  --stage5-dir backend/modules/fall_detection/reports/stage5 \
  --output-dir backend/modules/fall_detection/reports/esp32_diagnosis_repeat

.venv/bin/python backend/modules/fall_detection/plot_esp32_diagnosis.py \
  --diagnostic-dir backend/modules/fall_detection/reports/esp32_diagnosis_repeat \
  --manifest backend/modules/fall_detection/reports/sample_manifest.csv \
  --checkpoint backend/modules/fall_detection/outputs/cnn_seed42_mps_v1/frozen.pt
```

诊断脚本输入冻结模型、manifest和原评价目录，只计算评分与检查统计；输出diagnostics.json、全样本诊断CSV及全部ESP32测试错误清单。绘图脚本输出评分/误报率图、固定规则选择的FP/TN/TP/FN曲线与实际补零输入图。本地.venv已安装相同核心版本依赖。已有诊断目录拒绝覆盖，复现请用新目录。

本分析是看过测试结果后的探索性解释；未改阈值、未扫描新的测试工作点，不能把随后依据这些错误设计的方案在同一测试集上的提升当作从未查看过测试集的独立验证。

## 官方论文与代码的特征语义核查

详见 [带页码和固定代码链接的核查记录](reports/official_feature_semantics/official_feature_semantics.md)。论文跌倒任务描述NXP 2×2、4条空间链路、每条58个子载波，支持HP的232按4×58组合特征解释；ESP32-S3为1×1、64个频域位置。具体展平顺序、HP显式别名映射和逐位置物理频率仍未取得导出脚本确认。

官方允许频域补零/裁剪，实际加载器在样本归一化后将原始特征放在前面、尾部补零到232。因此当前尾部补零有官方基线依据，但它不建立物理对齐。另需明确：官方是逐样本整体mean/std，当前是训练集拟合的逐设备逐特征参数，整个预处理并非逐值复现官方。本次仅查证和只读算例，没有改变冻结模型、预处理或阈值，没有重新训练。


### Independent SenseFi-style temporal CNN diagnostics

`models/sensefi_style.py` adds device-specific 1x1 Conv1d adapters (HP 232→32,
ESP32 64→32), three shared temporal convolutions, adaptive average pooling and
an independent binary classifier. This is a SenseFi-inspired adaptation, not
original SenseFi reproduction or pretrained-weight transfer. Existing model,
preprocessing, checkpoints and threshold are unchanged.

The collator reuses train-fitted normalization, verifies the original-feature
mask and zero trailing padding, then removes only added ESP32 padding before
stacking separate device tensors. Original 12 constant ESP32 positions remain.
Mixed batches are ordered HP then ESP32, with labels/IDs reordered together.
This interface consistency does not establish physical domain alignment.

From the repository root (use a new output directory on each run):

```bash
.venv/bin/python backend/modules/fall_detection/check_sensefi_style.py \
  --manifest backend/modules/fall_detection/reports/sample_manifest.csv \
  --preprocess-config backend/modules/fall_detection/outputs/preprocess_v1.json \
  --output-dir backend/modules/fall_detection/reports/sensefi_style_stage3_repeat
```

Recorded run: `reports/sensefi_style_stage3/sensefi_style_check.json`.
Seed 42, CPU, Adam 0.001, balanced diagnostic subsets, BCEWithLogitsLoss,
threshold 0.5. Smoke: 64 train segments, two epochs/four updates, dropout 0.3.
Fresh overfit model: 16 fixed train segments, dropout 0; passed at update 27,
BCE 0.000756537, HP 8/8 and ESP32 8/8. Criterion: BCE <0.02 and all 16 correct
for five consecutive post-update eval checks. Sample IDs/content hashes and
all losses are recorded. This is trainability evidence only; no validation or
test data, formal training, model selection, pretrained weights or production
checkpoint. These checks do not establish reduced ESP32 false alarms.


### SenseFi-style formal training (seed 42; validation only)

Independent entry point: `train_sensefi_style.py`. This run trains from scratch,
with no SenseFi pretrained weights. Original baseline files remain untouched.
Run from the repository root, using fresh output directories:

```bash
.venv/bin/python backend/modules/fall_detection/train_sensefi_style.py \
  --manifest backend/modules/fall_detection/reports/sample_manifest.csv \
  --preprocess-config backend/modules/fall_detection/outputs/preprocess_v1.json \
  --audit-details backend/modules/fall_detection/reports/audit_details.json \
  --checkpoint-dir backend/modules/fall_detection/outputs/sensefi_style_seed42_mps_repeat \
  --report-dir backend/modules/fall_detection/reports/sensefi_style_seed42_mps_repeat \
  --device mps --seed 42 --max-epochs 100 --patience 10 --batch-size 32
```

Recorded run: `reports/sensefi_style_seed42_mps_v1/`; best model:
`outputs/sensefi_style_seed42_mps_v1/best.pt`. Adam lr 0.001, dropout 0.3,
train-derived positive weight 2768/1922 = 1.4401664932, same weighted BCE
for training/validation. Global validation loss selects checkpoints; per-device
metrics never select separate checkpoints. Fixed metric threshold 0.5; no tuning.
Only official train_id (4690) and val_id (1005) are constructed. Existing
train-fitted normalization is reused; trailing added ESP32 padding is removed.

Stopped at epoch 35 after epochs 26–35 did not beat epoch 25. Best train loss
0.0953027521; best validation loss 0.1731588852. Duration 890.44 seconds.
Best overall validation P/R/F1: 0.937931/0.929385/0.933638.
HP: 926 records (390 Fall, 536 Nonfall), TP378 FP20 FN12 TN516;
P/R/F1=0.949749/0.969231/0.959391, FPR20/536=3.7313%.
ESP32: 79 records (49 Fall, 30 Nonfall), TP30 FP7 FN19 TN23;
P/R/F1=0.810811/0.612245/0.697674, FPR7/30=23.3333%.

A new Python process safely loaded the checkpoint (weights_only=True), rebuilt
SenseFiStyleCNN, strictly loaded its state, and replayed all validation records
on MPS. Maximum score difference from saved best-epoch predictions: 0.0;
validation loss identical. See `checkpoint_check.json`. Hashes confirm original
baseline source, preprocessing and best/frozen models remained unchanged.

Inspect `training_summary.json`, `history.csv`, `best_validation_predictions.csv`,
`training_curves.png`, and `device_validation_curves.png`. The latter plots
both devices at the same epoch/threshold; dashed line marks selected epoch 25.
Reduced training loss with fluctuating validation loss indicates limited
continued validation improvement. ESP32 still misses 19/49 falls at the selected
checkpoint: this run does not establish resolution of the device problem.
Official split content duplicates remain; results are single-seed validation
observations. No test evaluation, threshold optimization or frozen deployment
artifact was produced. Existing baseline evaluation/inference tools expect the
old architecture and must not be used unchanged with this new checkpoint.


### Original vs SenseFi-style validation-only comparison

`compare_validation_models.py` strictly reloads both best checkpoints on CPU,
constructs only val_id, asserts identical preprocessing, manifest and train/val
membership, and joins scores by sample ID. Threshold is exactly 0.5 for both;
the original model's frozen 0.3893889 threshold is not used. No optimizer,
threshold search, training, or test evaluation. Run from repository root:

```bash
.venv/bin/python backend/modules/fall_detection/compare_validation_models.py
```

Evidence is in `reports/validation_model_comparison_v2/` (script refuses to
overwrite this completed directory; choose a new OUT path for a repeat).
The v1 directory is an explicitly marked incomplete CSV-export attempt.
Fresh CPU scores differ from original MPS records by at most 4.7684e-7 for the
baseline and 8.3447e-7 for the new model; no class decisions change.

At threshold 0.5:
- Overall F1: original 0.940248, new 0.933638.
- HP: original TP382 FP15 FN8 TN521; new TP378 FP20 FN12 TN516.
- ESP32: original TP35 FP16 FN14 TN14; new TP30 FP7 FN19 TN23.
- ESP32 error overlap: both wrong 19 (6 FP, 13 FN); original-only errors 11
  (10 FP corrected, 1 FN corrected); new-only errors 7 (1 new FP, 6 new FN);
  both correct 42. Union of erroneous segments: 37, all listed in esp32_errors.csv.

All 79 ESP32 validation HDF5s were reopened: finite 64x500 amplitude, all 12
training-constant positions remain zero. None matches a training content hash.
All belong to E22, so hardware and environment effects cannot be separated.
There are 7 users among shared errors and 6 among new-only errors; errors are
not confined to one user. Examples are the alphabetically first ID per true
class/error group, not chosen for attractive traces. See comparison.json,
esp32_all_records.csv, esp32_comparison.png and esp32_example_traces.png.

Additional descriptive score analysis: score_descriptions.json. ESP32 ROC-AUC
original 0.691837, new 0.772109; Fall score medians 0.674431→0.713590 and
Nonfall medians 0.527881→0.438690. These indicate better observed ranking, not
proof of reliable fall detection. New-model 26 errors include 19 outside the
0.45–0.55 band, so errors are not exclusively near threshold. A common missed
Fall U11/99000 scores 0.251328→0.080136; a newly missed Fall U22/271000 scores
0.732343→0.169036. Do not infer true action timing or annotation correctness
from three amplitude traces alone. No timestamped event/video ground truth
was available. No causal attribution to adapters, pooling or normalization is
established because multiple architectural factors differ and only one seed
has been compared. Original/new model/config/source hashes remained unchanged.


### Independent CNN+LSTM diagnostics (no formal training)

`models/cnn_lstm.py` adds OriginalCNNLSTM and SenseFiStyleCNNLSTM without
modifying the existing two models. Both use a single unidirectional LSTM,
hidden size 64, batch_first=True, no recurrent state carried between calls.
Conv features [B,C,125] transpose to [B,125,C]. The last hidden state [B,64]
feeds external Dropout(0.3) and Linear(64,1), returning logits (no sigmoid).
No global average pooling remains. The SenseFi-style 128-unit classifier is
replaced, not retained. Input normalization, complete 500-point segments and
original device adapters remain; old variant retains 232-wide padded ESP32
input, new variant removes only added trailing padding. Original 12 constant
positions remain. Both are randomly initialized, without loading any weights.

From repository root (choose a fresh output directory for repeat runs):

```bash
.venv/bin/python backend/modules/fall_detection/check_cnn_lstm.py \
  --output-dir backend/modules/fall_detection/reports/cnn_lstm_stage3_seed42_repeat \
  --seed 42 --max-overfit-steps 500
```

Recorded results: reports/cnn_lstm_stage3_seed42/diagnostics.json. Only train_id
was loaded. Both models use the same selected unique-content sample IDs:
shape=4 (one per device/class), smoke=64 (16 per device/class), overfit=16
(4 per device/class). Sorting in mixed-device collation keeps IDs/labels with
inputs. The report records scores in the explicitly specified sample-ID order.
CPU 2 threads, deterministic algorithms, Adam lr=0.001, balanced BCE (weight1),
fixed threshold0.5. Smoke: dropout0.3, batch32, 2epochs/4updates. Overfit: fresh
model/optimizer, dropout0, batch16, at most500updates. Passing requires all16
correct and BCE<0.02 for five consecutive post-update eval checks.

OriginalCNNLSTM: 76,705 parameters; smoke train BCE 0.694496→0.679434,
same-subset eval final0.663957; overfit initial0.689682→0.0163409 at update47,
HP8/8 and ESP328/8. SenseFiStyleCNNLSTM: 80,961 parameters; smoke train BCE
0.696985→0.691415, same-subset eval final0.688910; overfit initial0.694347→
0.00832533 at update30, HP8/8 and ESP328/8. All parameters received finite
nonzero total gradients and changed during smoke. Stateless repeated eval and
new-variant separate-device vs mixed-batch outputs agree within tolerance.
Wrong feature/time dimensions, unknown device and nonfinite inputs are rejected
(guard_checks.json). Existing model/source/preprocessing/checkpoint hashes
were unchanged. No val/test inference, formal training, threshold optimization,
production checkpoint, or claim of better generalization. The original variant
has a transient overfit-loss increase near update28, then meets the declared
criterion; the full history is preserved in CSV and overfit_curves.png.


### CNN+LSTM formal training and four-model validation comparison

Independent entry point: train_cnn_lstm.py --variant original|sensefi_style.
Only official train_id/val_id are constructed. Each model starts from random
parameters, seed42, Adam lr0.001, batch32, dropout0.3, at most100epochs,
patience10, strict lower global validation loss (min_delta0). Training-derived
positive weight2768/1922=1.4401664932 is used in training and validation BCE.
Validation is eval/no_grad. Per-device metrics at0.5 do not select models.
No new threshold selection, test evaluation, or deployment freeze.

From repository root, use fresh directories for repeats:

```bash
.venv/bin/python backend/modules/fall_detection/train_cnn_lstm.py \
  --variant original \
  --manifest backend/modules/fall_detection/reports/sample_manifest.csv \
  --preprocess-config backend/modules/fall_detection/outputs/preprocess_v1.json \
  --audit-details backend/modules/fall_detection/reports/audit_details.json \
  --checkpoint-dir backend/modules/fall_detection/outputs/original_cnn_lstm_seed42_mps_repeat \
  --report-dir backend/modules/fall_detection/reports/original_cnn_lstm_seed42_mps_repeat \
  --device mps --seed 42 --max-epochs 100 --patience 10
```

For the second variant, use --variant sensefi_style and different checkpoint/
report directories. Recorded original run: original_cnn_lstm_seed42_mps_v1;
recorded new run: sensefi_style_cnn_lstm_seed42_mps_v1, under outputs/ and
reports/. Never overwrite the two earlier CNN checkpoints. Existing baseline
inference scripts do not support these LSTM architectures unchanged.

Original+LSTM stopped at37, best27, weighted validation BCE0.196924452,
best training BCE0.139167304, duration1506.17s. New+LSTM stopped at40, best30,
weighted validation BCE0.184399859, best training BCE0.114010739,
duration1624.02s. Best models include CPU state_dict, architecture/LSTM config,
preprocessing/normparameters, label mapping, seed/best_epoch, data version and
content/manifest/split identifiers, optimizer state and training configuration.
Metric threshold0.5 is not an optimized deployment threshold.

Run the independent four-model replay from repository root:

```bash
.venv/bin/python backend/modules/fall_detection/compare_four_validation_models.py
```

The completed output directory reports/four_model_validation_seed42_v1 must
be changed to repeat; the script refuses overwrite. It safely loads best.pt
using weights_only=True, strict state_dict loading, asserts identical manifest,
train/val membership and preprocessing across all4, evaluates all1005 val
records on CPU, and joins predictions by sampleID. Maximum score differences
from saved MPS predictions: oldCNN4.7684e-7,newCNN8.3447e-7,old+LSTM8.29995e-6,
new+LSTM1.37091e-6. No classifications changed. All preserved source/config/
checkpoint hashes match pre-run records. Separate checkpoint_check.json files
also live in each new run's report directory.

At fixed threshold0.5, model order is oldCNN,newCNN,oldCNN+LSTM,newCNN+LSTM:
- Overall F1: .940248, .933638, .924779, .929705.
- HP F1: .970775, .959391, .950125, .952261.
- ESP32 F1: .700000, .697674, .725490, .720930.
- HP (TP,FP,FN,TN): (382,15,8,521); (378,20,12,516);
  (381,31,9,505); (379,27,11,509).
- ESP32 (TP,FP,FN,TN): (35,16,14,14); (30,7,19,23);
  (37,16,12,14); (31,6,18,24).

LSTM modestly improves observed ESP32 metrics relative to each corresponding
CNN but worsens HP/global F1. Old+LSTM fixes net2 ESP32 misses with unchanged
FP; new+LSTM has net1 fewer FP and1 fewer FN. These single-seed, small-subgroup
results do not prove reliable generalization or that longer temporal reasoning
caused improvement: pooling/classifier/parameter count changed too. Official
train/val content duplicates remain. No test records were opened or evaluated.
See comparison.csv/json, validation_predictions.csv, four_model_validation.png,
four_training_curves.png and per-run device_validation_curves.png. Training
logs and all epoch histories are retained; no additional hyperparameter runs
were launched based on validation outcomes.


### Official CSI-Bench source reproduction

See `official_reproduction/README.md` for pinned upstream source, exact run
commands, input checks, implementation discrepancies and platform differences.
Completed: official MLP seed42, train/validation only, batch128, epochs47/best32,
patience15, unchanged official normalization/MLP/AdamW/CE/trainer/scheduler math.
Fall F1 overall .886689, HP .911980, ESP32 .659341; weighted F1 overall .897817.
This is not complete reproduction of all seven models, three seeds or paper
test tables. ResNet18 batch128 probe was interrupted (completion unconfirmed);
no formal ResNet training claimed. All earlier custom models remain unchanged.
Official MLP truncates input to first10000 flattened values, and batch-count
warmup is advanced once/epoch in the source. Both were preserved and disclosed,
not silently fixed. Checkpoint, validation metrics, learning rates and new-
process replay are under outputs/reports/official_mlp_seed42_sourcefaithful_v1.


## 四步最终研究基线选择（已完成）

完整证据见 [四步结果报告](reports/final_selection_v1_report_zh.md)，执行协议和新脚本位于 [final_selection_v1](final_selection_v1/README.md)。比较原 1D-CNN 与固定官方源码 ResNet18，种子均为 42、43、44；只用同一 val_id 选择全局阈值及方案，最终交付种子预先固定为 42。

选中原 1D-CNN，三种子验证 Fall F1 均值 94.08%（样本标准差 0.64 个百分点）；ResNet18 为 93.27%（0.51 个百分点）。冻结阈值为 0.38938893377780914。新模型包是 `outputs/final_selection_v1/final_research_model.pt` 及同名 `.json` 回执，旧模型均保留。

冻结并通过新进程单文件推理检查后，test_id Fall Precision/Recall/F1 为 89.71%/98.04%/93.69%。ESP32 测试误报为 29/35（82.86%），因此该交付定位为研究基线，尚未验证实际连续报警可用性。项目早期已查看过原 CNN 测试结果；不能声称测试集历史上从未被查看。本轮最终选回同一原模型和阈值，测试结果相同属于预期。

单文件推理（完整原始 HDF5 分段；替换为真实文件路径）：

```bash
.venv/bin/python backend/modules/fall_detection/final_selection_v1/infer.py \
  --checkpoint backend/modules/fall_detection/outputs/final_selection_v1/final_research_model.pt \
  --file /absolute/path/to/actual_segment.h5
```

所有新增训练、报告、冻结文件位于独立目录；原有模型、预处理参数和既有源码哈希检查通过。Google Drive 持久结果目录：[CSI-Bench_FallDetection_FinalSelection_v1](https://drive.google.com/drive/u/0/folders/1vL5ih_6yhkO3ct6eAq-9wxFI_Z6xx5IU)。


## HP latency and TCS-Fall source audit (2026-10-26)

Current scope: HP input only; ESP32 deferred. Existing weights were trained under the earlier mixed-device protocol and were not retrained. No threshold changes or new test evaluation were performed.

Batch-one CPU/MPS measurements are complete on Apple M2 (8 GiB), with 30 warmups and 600 measurements per scope. The in-memory adapter exactly matches established preprocessing.

- [HP latency report](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/reports/hp_latency_v1/report_zh.md)
- [TCS-Fall preliminary audit](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/reports/tcs_fall_audit_v1/report_zh.md)
- [Timing scripts and commands](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/hp_latency_v1/README.md)
- [Audit scripts and commands](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/tcs_fall_audit_v1/README.md)

TCS-Fall: all seven public short demonstrations and the actual-fall-time workbook were inspected. Raw continuous recordings have not been obtained. The user will download through the author's Baidu Netdisk link and provide local paths; begin with dataset/1.zip and TCS-Fall-main.zip. No event-level recall, false alarms/hour, or alert latency has been measured. TCS-Fall Intel5300 data must not inherit HP normalization through padding.


## TCS-Fall local archives inspected (2026-10-28)

The user supplied Downloads/1.zip and Downloads/TCS-Fall-main.zip. All 226 raw members were attempted: 223 readable, three corrupt (a_01_03_22.dat, a_01_03_23.dat, a_01_03_24.dat), independently confirmed with unzip. The raw archive fails integrity acceptance. All 12 structurally annotated continuous recordings are readable (about 31.5 minutes). Four activity-16/17 files remain semantically unknown and excluded. Only nominal subject 01 is available.

Full native packet headers, canonical antenna-corrected I/Q, RF-scaled finite values and timing were checked for readable records. No model training/inference, threshold adjustment or test/continuous-alarm evaluation was performed. Existing frozen artifacts remain unchanged. Timing-zero semantics and the missing preprocessing notebook remain unresolved.

- [Local archive audit report](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/reports/tcs_fall_local_v1/report_zh.md)
- [All 226 manifest rows, including failures](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/reports/tcs_fall_local_v1/sample_manifest.csv)
- [Independent ZIP integrity check](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/reports/tcs_fall_local_v1/independent_zip_check.json)


## TCS-Fall within-subject pilot trained (2026-10-28)

User authorized starting with the207 intact single-action recordings while deferring archive repair. The independent GCV + official2D CNN pilot uses165train/42validation whole recordings, seed42, threshold0.5, and at training time no test/continuous evaluation. It completed100epochs, selected epoch100 at validation loss0.23128879, and obtained TP18/FP0/FN0/TN24 on this small within-subject validation split. This does not establish cross-subject or alarm generalization. The3corrupt and4unknown-activity recordings were excluded explicitly. Old HP artifacts were hash-checked unchanged.

- [Training report and curves](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/reports/tcs_subject01_gcv_cnn_seed42_v1/report_zh.md)
- [Independent model and commands](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/tcs_withinsubject_v1/README.md)


## TCS-Fall continuous replay diagnostic (2026-10-28)

The user requested replay of the 12 reserved continuous recordings without retraining or threshold changes. The frozen within-subject 2D CNN and threshold0.5 produced 1,791 rolling-window scores (8,000 packets per window, 1,000-packet stride). Using the official code's event matching interval, all12 nominal events were detected. The fixed adjacent-positive-window episode count yielded25 false-alarm episodes across0.450 non-event monitor hours, or55.6/hour. All25 were before the event tolerance interval and concentrated in four activity12/13 trials. Nominal alert offsets to the integer-second workbook annotations had median+0.86s and range-2.39..+1.71s; annotation onset/impact semantics and time-zero alignment are not verified, so these are not validated true alert latencies. No compute-to-notification latency was measured. This is one nominal subject and a diagnostic, not evidence of deployment readiness. The checkpoint, threshold, HP models and test evaluation remain unchanged.

- [Continuous replay report and score plot](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/reports/tcs_continuous_replay_v1/report_zh.md)
- [Per-window predictions](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/outputs/tcs_continuous_replay_v1/window_predictions.csv)
- [Per-alarm episodes](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/reports/tcs_continuous_replay_v1/alarm_episodes.csv)
- [Replay implementation](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/tcs_withinsubject_v1/replay_continuous.py)

## Four-subject TCS-Fall data preparation (2026-10-29)

Archives1..4 are present and audited. Subjects01/02 are assigned to training,03 to validation,04 to a held-out test. An explicit input eligibility review excludes one6973-packet training SA record and accepts an independently verified8980-frame prefix of one validation record whose trailing frame contains only a2-byte header. Original archives are untouched. Earlier3corrupt SA files and16unknown-activity recordings remain excluded.

Prepared356training SA plus2688continuous background windows, and140validation SA plus1344background windows. All background inputs end within120seconds on the actual NIC clock. Test features/predictions are not produced. Two training-only per-feature normalizers and a memory-mapped Dataset are saved. All inputs are finite;207existing SA feature arrays match exactly; no exact cross-role raw CSI or prepared-feature duplicates were found. Complete features occupy about622MiB. Four-arm comparison sampling and validation-loss protocol is saved before any new model predictions. Training initially paused because local disk free space was about0.56GiB. After the user freed space, approximately10GiB was available and the four-arm seed42 run started. Live epoch progress and per-arm best/last checkpoints are saved in the separate comparison_seed42_v1 directories; subject04 remains unevaluated.

See [preparation report](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/reports/tcs_cross_subject_v1/preparation_checked_20261029/report_zh.md) and [scripts/commands](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/tcs_cross_subject_v1/README.md).


## Four-arm TCS-Fall training and validation completed (2026-10-29)

The seed42 arms ran100/14/42/25epochs with best epochs100/4/32/15. All four checkpoints were reloaded in a fresh process; each1484-score validation comparison matched exactly. At threshold0.5, all four detected12/12 subject03 continuous events. Background alarm episodes were21/17/28/2 over22.4minutes of eligible background. Adding both CT negatives and training-fitted standardization reduced background alarms most, but its separate SA recall was90% (6misses), with nominal continuous alert-delay median2.62s. This is one-person validation evidence, not final deployment validation or a full paper reproduction. Subject04 was not evaluated; original HP/TCS models were preserved.

[Full results, curves, error records, and limitations](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/reports/tcs_cross_subject_v1/comparison_seed42_v1/evaluation/report_zh.md).


## ResNet18 + Module B integration (2026-10-30)

The primary website output now uses the existing CSI-Bench official ResNet18 seed42 checkpoint and its previously selected global validation threshold0.574872225522995. All926 HP validation segments are shown: TP384/FP23/FN6/TN513. No retraining, TCS transfer, or new test inference occurred. B context is attached only through exact raw-input identity or explicit same-clock overlapping windows; display time is never an alignment key. Missing/fixture/broken B output cannot veto or amplify C. The new author release is integrated through a pinned spectral ExtraTrees runtime. B runs on exactly the same raw HP input as C. It has no Fall class; these C inputs have no five-class activity labels, so no joint accuracy is claimed.

The earlier D/TCS website replay is preserved in outputs/tcs_D_replay.json and frontend data/archive_tcs.json, displayed only as a separate archived experiment. The prior1D research-selection artifact is unchanged.

Commands and limitations: [ResNet18+B handoff](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/docs/RESNET_B_HANDOFF.md).
