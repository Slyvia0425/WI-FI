# TCS-Fall 跨人四组对照

先保留五层 CNN，比较连续负样本和标准化的影响。腾出空间后，四组训练和验证回放均已完成，最终结果见 comparison_seed42_v1/evaluation/report_zh.md。四人方案是小规模可行性实验，不能称为论文二十人交叉验证复现。

## 下载清单

官方仓库：https://github.com/zhouziyu2050/TCS-Fall

作者百度网盘：https://pan.baidu.com/s/1IT5KZEhvhNjuBjQ3EKxMnw?pwd=qitg ，提取码 `qitg`。
进入 `TCS-Fall/dataset`，依次下载 `2.zip`、`3.zip`、`4.zip`，保留完整 ZIP。
备选官方入口：https://ieee-dataport.org/documents/tcs-fall 。该入口可能要求登录，文件分卷方式与百度不同，不直接按文件名假定覆盖等价。

| 包 | 本轮作用 | 说明 |
|---|---|---|
| 1.zip | 训练／开发 | 已有；207 条完整 SA、12 条连续录制可用；3 条损坏 SA 继续显式排除；16/17 活动语义未核实排除 |
| 2.zip | 训练 | 需要完整 SA 和连续录制，尤其正常行走及行走伴随日常活动 |
| 3.zip | 验证 | 独立受试者，用于验证损失、连续误报和模型选择 |
| 4.zip | 最后测试 | 先检查数据完整性，冻结方案后才运行模型评价 |

编号与实际受试者身份通过包内 `a_XX_YY_ZZ.dat` 和论文命名规则核实，不能只相信 ZIP 名称。现有 `TCS-Fall-main.zip` 与完整时间标注表已取得，无需重复下载。完整二十人实验以后再补 `5.zip` 至 `20.zip`。

2026-10-29 本机剩余约 12 GiB，已有 1.zip 约 2.2 GiB。其他包大小需以下载页面为准；先下 2.zip，再安排 3.zip、4.zip。空间不足时放外置盘并提供实际目录；不要全部解压，也不要删除现有唯一原始数据。脚本直接读取 ZIP，不复制完整原始数据。

## 已确认的方法差异

论文 Methods/Dataset split 和官方 `butter-cv-cnn.py` 均使用 SA 加训练受试者连续录制早期的正常活动。当前试跑只训练 SA，缺少持续行走负样本；其连续误报集中在活动12/13，这是优先检查的数据覆盖差异，不是已证明的因果结论。

论文 Methods/Standardization of data 描述逐子载波零均值、单位标准差，但没有明确统计量的样本范围。当前可用代码缺 `preprocess.ipynb`，不能声称已恢复其细节。后续标准化分支明确选择“仅训练集拟合每个特征均值/标准差”；这是我们自己的记录清楚的实验方案，不冒充官方原实现。现有未标准化 checkpoint 的输入不能直接改变。

连续负样本窗口必须整体位于已记录的早期非跌倒段内，用真实包时间检查结束时间不超过120秒，并交叉核对事件标注。这比官方代码仅检查窗口起始时间0..119秒更保守，要单独披露。所有录制先按人归属，再切窗口；本次已分析的受试者01不能当作新的独立最终测试受试者。

固定 CNN、初始阈值0.5、种子42，预定四个对照分支：SA/SA+连续负样本，分别采用原GCV/训练拟合标准化GCV。完整数据审计后、看到新模型验证结果前，再确定共同验证集合、类别/录制采样与优化预算，避免添加大量负窗口时把类别比例变化混作方法收益。

## 入库命令

从仓库根目录运行；每次使用新的输出目录：

```bash
.venv/bin/python backend/modules/fall_detection/tcs_cross_subject_v1/preflight_archives.py \
  --data-dir /Users/chenxiaohe/Downloads \
  --output backend/modules/fall_detection/reports/tcs_cross_subject_v1/intake_next
```

输入：指定目录的1.zip至4.zip、固定角色协议、已有审计及时间表提取结果。
输出：`data_readiness.json` 与 `archive_members.csv`。
完整读取新 ZIP 成员进行 CRC 和 SHA 检查，但不解码模型特征、不运行模型。对已审计1.zip只在整包哈希完全相同时复用逐文件审计。
缺包或新完整性问题时退出码2，报告实际缺项；不会用假数据补齐，不会自动重新分配受试者。

入库完整性通过以后仍需完成新受试者的原生 CSI、时间轴、标注和跨集合内容重复审计，才进入特征生成及训练。

## 2026-10-29 已完成数据准备

四个包已到齐。新包2/3/4逐文件CRC通过，全部新录制完成原生审计。当前输入规格下：训练01/02有356条SA（160Fall/196Nonfall）及24条连续录制；验证03有140条SA及12条连续录制；测试04有140条SA及12条连续录制。

两条新发现已单独审阅：02/a_02_03_05.dat只有6973包，因固定8000包规格而排除；03/a_03_05_04.dat有8980个完整帧及末尾两字节未完成帧头，完整前缀重新审计并与官方解析函数核对后，保留其未受影响的前8000包。原文件未修改。3条原有ZIP损坏与16条未知16/17活动仍明确排除。

训练新增2688个连续非跌倒窗口，验证新增1344个；整个窗口均结束于真实包时间120秒以内。特征缓存共4528×1×400×90，652032000字节，未生成测试特征。SA-only及SA+CT两套标准化均仅拟合训练01/02；全部输出有限。207条旧SA特征逐条差异为0，训练/验证完整GCV窗口哈希未发现跨集合精确重复。

数据准备结束时磁盘仅余约0.56GiB，曾暂停训练。用户腾出空间后检查约有10GiB余量，现已启动四组训练。启动要求至少5GiB，运行时低于1GiB会保存已有检查点并退出；不要删除唯一原始副本。四组采样与共同验证损失已在新模型预测前记录。

- [完整准备报告](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/reports/tcs_cross_subject_v1/preparation_checked_20261029/report_zh.md)
- [训练协议](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/reports/tcs_cross_subject_v1/preparation_checked_20261029/training_protocol.json)
- [输入资格与边界情况](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/reports/tcs_cross_subject_v1/eligibility_20261029/eligibility.json)

已实际执行的后续阶段命令（重跑时给输出目录换新名字）：

```bash
.venv/bin/python backend/modules/fall_detection/tcs_cross_subject_v1/audit_native.py --intake backend/modules/fall_detection/reports/tcs_cross_subject_v1/intake_downloaded_20261029 --output backend/modules/fall_detection/reports/tcs_cross_subject_v1/native_20261029
.venv/bin/python backend/modules/fall_detection/tcs_cross_subject_v1/resolve_native_findings.py --audit backend/modules/fall_detection/reports/tcs_cross_subject_v1/native_20261029 --output backend/modules/fall_detection/reports/tcs_cross_subject_v1/eligibility_20261029
.venv/bin/python backend/modules/fall_detection/tcs_cross_subject_v1/prepare_features.py --eligibility backend/modules/fall_detection/reports/tcs_cross_subject_v1/eligibility_20261029 --output backend/modules/fall_detection/outputs/tcs_cross_subject_v1/features_v1
.venv/bin/python backend/modules/fall_detection/tcs_cross_subject_v1/check_preparation.py --features backend/modules/fall_detection/outputs/tcs_cross_subject_v1/features_v1 --eligibility backend/modules/fall_detection/reports/tcs_cross_subject_v1/eligibility_20261029 --report backend/modules/fall_detection/reports/tcs_cross_subject_v1/preparation_checked_20261029
```

原生严格审计退出码2表示发现两条输入异常，原报告保留；后续资格审阅明确记录了如何处理这两条，未伪装成原始数据全部无问题。四组训练的采样和共同验证损失为本实验设计，不是原论文训练协议的精确复刻。


## 四组训练与验证回放

当前运行目录为 `outputs/tcs_cross_subject_v1/comparison_seed42_v1`，报告目录为 `reports/tcs_cross_subject_v1/comparison_seed42_v1`。训练运行时不要重复执行下面的命令；完成后结果见 `training_comparison.json`。每组的 `best.pt` 保存共同验证损失最低的模型，`last.pt` 保存最近一轮及优化器和随机状态。

```bash
caffeinate -i .venv/bin/python backend/modules/fall_detection/tcs_cross_subject_v1/train_comparison.py \
  --features backend/modules/fall_detection/outputs/tcs_cross_subject_v1/features_v1 \
  --protocol backend/modules/fall_detection/reports/tcs_cross_subject_v1/preparation_checked_20261029/training_protocol.json \
  --output backend/modules/fall_detection/outputs/tcs_cross_subject_v1/comparison_seed42_v1 \
  --report backend/modules/fall_detection/reports/tcs_cross_subject_v1/comparison_seed42_v1
```

中断后先确认没有同一训练进程，再在相同命令末尾加 `--resume`；源代码、协议、数据和配置必须与原运行一致。脚本跳过已完成分支，未完成分支从保存的完整轮次继续。`plot_progress.py --report <报告目录>` 只读取历史生成进度曲线，不改变训练。

四组区别：A只用单动作SA；B增加连续CT背景；C为A加入训练拟合的逐特征标准化；D为B加入标准化。每轮45步，每batch8条，4个Fall和4个Nonfall；B/D的负例为2个SA与2个CT，CT先均匀抽录制再抽窗口。全部使用同一初始权重、种子42、Adam学习率0.0001、最多100轮和连续10轮共同验证损失不降低早停。共同验证损失为0.5×Fall SA平均CE +0.25×Nonfall SA平均CE +0.25×CT背景平均CE。阈值固定0.5。

验证人员03的12条完整连续录制已准备1742个窗口，其中1344个早期背景窗口与已有验证缓存逐项一致。评价协议已在连续分数产生前固定。四组训练全部完成后，使用新Python进程运行：

```bash
.venv/bin/python backend/modules/fall_detection/tcs_cross_subject_v1/evaluate_comparison.py \
  --features backend/modules/fall_detection/outputs/tcs_cross_subject_v1/features_v1 \
  --replay backend/modules/fall_detection/outputs/tcs_cross_subject_v1/validation_replay_features_v1 \
  --models backend/modules/fall_detection/outputs/tcs_cross_subject_v1/comparison_seed42_v1 \
  --training-report backend/modules/fall_detection/reports/tcs_cross_subject_v1/comparison_seed42_v1 \
  --output backend/modules/fall_detection/reports/tcs_cross_subject_v1/comparison_seed42_v1/evaluation
```

输出目录必须尚不存在，避免覆盖历史。评价先核对每组1484个保存前后的共同验证分数，再做完整连续回放。新报警按连续阳性段的起点计数，报告跌倒时刻起11秒内检出及允许提前3秒的辅助口径。误报只统计初始120秒背景，扣除首个完整窗口前的预热时间，并同时报告背景阳性状态时间占比。延迟是整数秒事件标注至输出窗口端点的名义偏移，尚不是包含计算与通知耗时的端到端延迟。测试人员04不参与这一步。


## 本轮已完成结果

A/B/C/D实际训练100/14/42/25轮，最佳100/4/32/15轮。四组均在独立进程中重新加载，每组1484个验证分数与保存前完全一致。人员03的12次连续跌倒均被四组检出；A/B/C/D背景误报警段为21/17/28/2，来自约22.4分钟有效背景。D的连续误报最低，但其SA跌倒召回为90%，仍漏6条，不能直接定为部署模型。阈值保持0.5，测试04未评价，原HP和TCS模型哈希保留。

[实际结果、训练曲线、延迟与限制](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/reports/tcs_cross_subject_v1/comparison_seed42_v1/evaluation/report_zh.md)。原始分数和每次报警记录也在该目录。


## D模型误判诊断已完成

重新解码验证人员03的23条SA和2条连续录制，原始输入与缓存逐值一致；训练集标准化经独立batch mean/M2算法重算通过。六条SA漏检集中在活动2。冻结模型和阈值0.5时，将六条记录的完整8000包窗口后移1000包，评分均跨过0.5；这是诊断性窗口位置敏感性证据，不替换原90%的分段召回。三条SA非跌倒误报没有因此消除。两次CT背景误报警发生在33/35秒，标注跌倒均137秒，且相关窗口也被另外三组误报。

[完整诊断报告与复现命令](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/reports/tcs_cross_subject_v1/diagnosis_D_v1/report_zh.md)。没有重新训练、调整阈值或评价人员04。下一步应预先规定统一窗口与记录级评价策略，覆盖全部非跌倒活动，避免对已知漏检挑选最有利窗口。


## 统一窗口鲁棒性对照已完成

覆盖训练/验证全部496条合格SA、1649个窗口、四个冻结模型。规则在新预测前固定：开头、居中、末尾、固定1000包步长网格的最高评分，阈值0.5。D的验证Fall检出54/60→居中60/60，非跌倒误判保持3/80，Fall F1为97.56%；网格最大值检出60/60但误判4/80。末尾窗口新增两条活动0漏检。训练记录上居中也新增误判，不能视作完整解决。B网格在这140条离线分段上全对，但此前连续误报警仍是独立问题。

[实际结果、逐条变化、限制与准确命令](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/reports/tcs_cross_subject_v1/window_robustness_v1/report_zh.md)。没有重新训练、调阈值或评价04，未自动改生产推理规则。
