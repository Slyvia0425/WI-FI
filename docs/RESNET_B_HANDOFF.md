> 当前默认路由已更新为B置信度<0.6才调用C。以下是此前并行版说明；以[B低置信度门控说明](B_LOW_CONFIDENCE_GATE.md)为准。

# ResNet18 + 已发布 Module B：同输入活动参考

本版使用 CSI-Bench 官方 ResNet18 seed42 作为 C，沿用原检查点和阈值 `0.574872225522995`。B 使用作者发布的频谱 ExtraTrees seed42 参数。两个模型处理同一份 HP 原始幅度，B 只附加活动参考，C 保留独立评分。

原1D-CNN研究产物和TCS五层CNN均保留。没有重新训练、微调TCS、重选阈值或新增C测试集预测。

## 参数来源

- 作者代码分支：`Module-B`，提交 `2963e7dd69891b2d3f2bfa0a8aa318b974e3ed91`。
- 新参数索引分支：[`Module-B-artifacts`](https://github.com/Bartley007/WI-FI/tree/Module-B-artifacts)，提交 `96b387daa08c3f393bb265e3bb490d80c28ee859`。B业务代码与已导入版本逐文件一致。
- 实际参数在 [GitHub Release module-b-data-models-v1](https://github.com/Bartley007/WI-FI/releases/tag/module-b-data-models-v1) 的 `module_b_models_and_evaluation.tar.zst`，不是普通分支文件。
- 发布包大小62,945,936字节，作者SHA256：`46f1cc9efc73426e87e06cca50fb184ce8074cb7c18962639a6eeb5e94037665`。
- 使用 `optimized_final/spectral/seed-42-stable/model.joblib`；固定seed42与C一致，没有按测试结果挑选B种子。
- 模型、标签和特征源码固定在 `backend/modules/human_activity_recognition/integration_v1/frozen_model.json`。导入证据见该模块 `docs/RELEASE_IMPORT_20261030.json`。

本次使用现有C原始HP文件完成同输入推理，不需要下载约950MB的另一份HAR数据包。作者发布包中的HAR测试分数只是历史作者结果，不能当成本次重新评价。

## 启动与命令

便携包包含构建好的网站、B ExtraTrees、C ResNet18以及独立归档的TCS模型。解压后：

```bash
python3 scripts/serve_demo.py
```

打开 http://localhost:5173/#/module/fall_detection 。网站读取保存的真实预测JSON，回放服务只需Python标准库；它不是每次切换页面都执行模型。

从源码开发：

```bash
npm ci --prefix frontend
npm run dev
# 修改前端后构建
VITE_BACKEND_URL=http://localhost:8000 npm run build
```

真实单文件推理需先执行 `python -m pip install -r backend/modules/fall_detection/requirements.txt`；其中scikit-learn固定为作者保存模型的1.7.2。网站标准库回放不需要这些ML依赖。随后从仓库根目录执行：

```bash
# 默认新进程重载C和已发布B，读取同一个原始HP分段
python -m backend.modules.fall_detection.resnet_b_integration.infer --file /path/to/HP_segment.h5

# 单独检查B，输出也可以供C的--b-output使用
python -m backend.modules.human_activity_recognition.integration_v1.infer \
  --file /path/to/HP_segment.h5 --output /path/to/new_B_output.json

# 接收已有同输入B结果，或显式仅运行C
python -m backend.modules.fall_detection.resnet_b_integration.infer \
  --file /path/to/HP_segment.h5 --b-output /path/to/new_B_output.json
python -m backend.modules.fall_detection.resnet_b_integration.infer \
  --file /path/to/HP_segment.h5 --no-b
```

输出中的 `activity_reference` 包含B活动偏好、置信度、匹配方式和质量；B缺失、读取失败或不匹配时，C仍输出原分数，参考状态为unavailable并给出原因。原B `service.py` 是CNN的PyTorch加载服务；发布的ExtraTrees必须走新 `integration_v1` 适配器，不能把joblib文件当作`.pt`。

## 同一输入，各自预处理

HDF5：`CSI_amps [232,500,1]`，去掉最后单例维得到 `[232,500]`。这里只接HP，不给ESP32或TCS强凑尺寸。

- **C**：固定官方逐记录整体均值/标准差归一化，转为 `[1,1,500,232]`，ResNet softmax第0列表示Fall。原始标签为Fall0/Nonfall1。
- **B**：依据作者 `experiments/runner.py` 的原始窗口加载及 `experiments/spectral.py` 的训练路径，直接对原始幅度执行逐特征去均值、时间方向FFT、频带功率和统计特征，形成4195维，再用已经训练的selector保留2800维，900棵ExtraTrees输出5类概率。没有重新拟合selector。
- B的 `config.yaml` 中 `window_zscore` 用于其CNN训练路径；频谱训练函数直接读取`bundle.*.windows`，没有调用该z-score。两条分支不能混用。

`stable`沿用作者频谱导出规则：最大五类置信度≥0.6。它不是连续时间上的稳定性判断，也不是跌倒评分阈值。C阈值继续固定为0.574872。

## 配对与边界

主要配对方式是相同 `input_id`：对归一化前的 `[features,time]` 原始幅度，转小端float32、C连续、统一正负零，对紧凑JSON形状和字节做SHA256，前缀 `csi-amplitude-f32-v1:`。由 `activity_reference.segment_identity()` 生成。

接口也支持有明确 `recording_id`、`clock_id`、实际时间窗的重叠过去窗口。当前独立分段不含物理时间戳，所以只用内容身份；页面序号`t`不能用于录制时间对齐，也不能把列表拼成连续监测。

B五类是jumping、running、seated-breathing、walking、wavinghand，**没有Fall类**。它在跌倒分段上也必须五选一，可能表现为某个动作的偏好；即使置信度高，也不是该动作的真实标签。它不能否决、降低或抬高C评分，不能确认事故。数据质量不足、不匹配、未来窗口、冲突结果或fixture均不能成为参考。

C的FallDetection分段只有Fall/Nonfall真值，没有B五类动作真值。本次不能计算B动作准确率或B/C联合提升；B训练集合与C验证集合的同源关系也没有完成跨任务审计。当前交付证明的是模型可加载、同输入可配对、接口能运行。

## 导出网页数据与证据

在原始C数据及旧研究CSV仍在时：

```bash
# 复用既有C验证分数，并核对原始输入与标签
python -m backend.modules.fall_detection.resnet_b_integration.export --bundle
# 实际运行冻结B，附加到同一批C输入；不训练或重选阈值
python -m backend.modules.fall_detection.resnet_b_integration.pair_released_b --bundle
```

C显示官方val_id中的926条HP分段（390Fall/536Nonfall），原指标：TP384/FP23/FN6/TN513，Precision94.35%、Recall98.46%、Fall F196.36%。这些始终是C单模型HP验证分数，不是联合效果或测试结果。页面可筛选6条漏检、23条非跌倒误报。

作者包中的历史缓存 `spectral_features_2000.npz` 为3963维，而最终模型selector要求4195维。该缓存保留作历史资料，没有补零/裁剪或用于最终模型复现；本次由原始HP分段按已固定的作者代码提取4195维特征。未宣称独立复现作者HAR测试指标。

本次参数导入、真实配对、独立进程核对和测试日志位于 `backend/modules/fall_detection/reports/resnet_b_integration_v2/`；旧版接口验收保留在v1目录。作者历史HAR评价文件在B的`outputs/released_v1/`，与当前无动作标签的配对结果分开。

C原检查点SHA256：`e3aabc34810aee4a647555eb7b1970bbf19a063b9d76f31d4ff59bfc5521a0a0`。此前同一检查点CPU batch1前向中位数42.6ms/P9548.8ms，内存预处理加模型中位数49.7ms/P95139.4ms；这些是此前C单模型测量，不包含B或采集等待，不能当作本次联合延迟。
