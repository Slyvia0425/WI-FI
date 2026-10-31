# HP 活动 + 跌倒的独立 ResNet18 实验

本目录不修改线上模型、B 低置信度门控或已有训练产物。新模型随机初始化，结构和 seed42 初始张量已经与固定提交的官方实现逐项核对。

## 训练前冻结的方案

- 数据版本：官方 Kaggle CSI-Bench v12。只使用 HP、232 特征、500 时间点、1 链路完整分段。
- FallDetection：官方 HP train 4348 条、val 926 条。训练内部去重及剔除与保留验证内容相同的记录后，训练剩 4341 条。
- HAR：从官方 HP train 7117 条中预先确定每类 200 条，共 1000 条；从官方 HP val 中每类取 50 条，共 250 条。按 user/environment/activity 分组轮流取样，组内由种子 42 的 SHA256 排序。选样文件先于训练生成，不根据模型误判挑训练样本。
- 不纳入先前 25 条活动诊断验证样本的相同内容。训练与验证按原始数值内容哈希检查，保持验证集固定，不重新随机划分。
- 训练 5341 条：Fall1727、Nonfall3614。验证 1176 条：Fall390、原 Nonfall536、HAR 弱 Nonfall250。
- 网络标签为 Fall=0、Nonfall=1（与早期 1D-CNN 的内部标签不同）。五类普通活动作为分段级弱 Nonfall 标签，不声称逐帧证明无跌倒。
- 输入 `[B,1,500,232]`，与官方相同的每分段均值/标准差归一化（PyTorch std correction=1），无补零、重采样、滤波或增强。此方案没有需要用验证集拟合的归一化参数。
- seed42，AdamW(lr=0.001, weight_decay=1e-5)，FP32，batch64，最多100轮，验证损失连续10轮不改善早停，最低损失保存最佳权重。不使用学习率调度器。
- CrossEntropyLoss，训练计数决定类别权重 `N/(2*N_class)`；验证沿用相同权重，选取混合来源加权验证损失最低的模型。
- 两模型独立推理，不经过 B 门控；共同固定阈值 0.5，补充共同固定原阈值 0.574872225522995，不寻找新阈值。
- 不评价测试集，不自动部署新模型。

## 命令（仓库根目录）

数据选择文件已生成；不要重新选样覆盖它。数据准备支持重试失败下载，并明确保留失败信息：

```bash
.venv/bin/python backend/modules/fall_detection/mixed_hp_resnet_v1/prepare.py prepare --workers 8
```

本机训练需要能使用 Metal 的终端，并将 device 改为 mps、batch-size 改为16。下方为 CUDA 环境命令；output 使用全新目录。训练脚本只访问 cache/train.npy 进行反向传播，只用 cache/val.npy 计算验证损失：

```bash
python backend/modules/fall_detection/mixed_hp_resnet_v1/train.py \
  --cache backend/modules/fall_detection/local_data/mixed_hp_resnet_v1/cache \
  --audit backend/modules/fall_detection/reports/mixed_hp_resnet_v1/data_audit.json \
  --output backend/modules/fall_detection/outputs/hp_mixed_resnet18_v1 \
  --device cuda --batch-size 64
```

中断后同样命令加 `--resume`。配置必须保持完全一致；每轮原子保存 last.pt（含优化器和 RNG）、best.pt、history.json、status.json 和训练曲线。

训练结束后在新的 Python 进程评价。原模型 checkpoint 不改写：

```bash
python backend/modules/fall_detection/mixed_hp_resnet_v1/compare.py \
  --cache backend/modules/fall_detection/local_data/mixed_hp_resnet_v1/cache \
  --audit backend/modules/fall_detection/reports/mixed_hp_resnet_v1/data_audit.json \
  --candidate backend/modules/fall_detection/outputs/hp_mixed_resnet18_v1/best.pt \
  --original backend/modules/fall_detection/outputs/official_resnet18_seed42_recovery_v1/reproduction_bundle.pt \
  --output backend/modules/fall_detection/reports/mixed_hp_resnet_v1/comparison \
  --device cuda --batch-size 64
```

也可先用 baseline.py 在本机独立跑原模型，compare.py 用 `--original-predictions` 替换 `--original`。复用基线要求 checkpoint 哈希、全部验证 ID 和原始内容哈希一致；若平台归约产生归一化舍入差异，还须核对每条样本的归一化统计。

独立单文件推理（将 file 换成实际 HP HDF5 路径，固定阈值0.5，不经过B门控）：

```bash
.venv/bin/python backend/modules/fall_detection/mixed_hp_resnet_v1/infer.py \
  --checkpoint backend/modules/fall_detection/outputs/hp_mixed_resnet18_v1/best.pt \
  --file /绝对路径/HP分段.h5 --device cpu
```

它输出 Fall score、阈值、Fall/Nonfall 判断、模型版本和权重哈希。仅支持实际核实的 `[232,500,1]` 实数幅度，不适配其他设备或强行改变尺寸。

## 如何阅读结果

FallDetection 验证集报告 TP/FP/FN/TN、Fall Precision/Recall/F1。HAR 单列各活动触发数量及比例，它不是连续监控中的每小时误报。查看逐样本表中的 fixed/regressed/both_wrong，检查减少误报是否同时增加漏检。

这些验证数据参与早停，结果不是独立测试估计。原模型训练使用 HP+ESP32，而新模型仅 HP 且增加 HAR、去重、采用类别权重及不同训练配置，因此不是只改变一个因素的消融实验。来源与场景也有关联；新增数据不保证真实环境泛化。

原训练数据有一条与验证内容相同；新训练已剔除对应内容。比较保留完整共同验证集，并补充排除原模型此重复内容后的共同子集。
