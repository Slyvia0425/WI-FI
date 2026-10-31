> 历史版本：本页记录2026-10-29的TCS D演示。当前主线已切换为CSI-Bench ResNet18，见 [最新交接](RESNET_B_HANDOFF.md)。

# Module B + C：一天内提交的集成交接

当前本地分支：`C-Falling-Detection`。本轮优先冻结现有方案、完成接口和演示；未重新训练跌倒模型、未调整阈值、未评价TCS人员04。

## 直接启动演示

交付ZIP包含构建好的网站、两模块回放和冻结D检查点。解压后进入文件夹，运行：

```bash
python3 scripts/serve_demo.py
```

此回放启动方式只需要Python3，不需要安装PyTorch、Node或原始数据。若8000/5173端口已有旧服务，先在旧终端按Ctrl+C。

- 跌倒页：http://localhost:5173/#/module/fall_detection
- 活动页：http://localhost:5173/#/module/human_activity_recognition
- 融合页：http://localhost:5173/#/fusion
- 模块接口：http://localhost:8000/api/modules

源代码开发方式仍为 `npm ci --prefix frontend`，然后 `npm run dev`。修改前端后运行 `VITE_BACKEND_URL=http://localhost:8000 npm run build` 重新生成静态演示。

## 现在实际完成了什么

| 部分 | 当前交付 | 限制 |
|---|---|---|
| B人体活动识别 | 同学的后端、前端、配置、演示fixture、测试已合入 | 没有同学训练好的权重及真实回放，显示PLACEHOLDER RESULTS |
| C跌倒检测 | 冻结D模型、12条真实验证录制、1742个窗口评分、可选录制回放、实际阈值0.5 | 研究候选，未做最终独立测试或部署验证 |
| B+C接口 | 同一后端模块发现、统一JSON契约、同一网站导航和融合说明 | 各模块录制与设备不同，没有共同时间真值 |
| 模型推理 | 新进程加载D，完整归一化，正常/误报警/事件窗口复核 | 输入限Intel5300 native .dat，不接受HAR的232特征HP张量直接拼接 |
| 性能 | CPU batch1模型前向和预处理计算耗时实测 | 不包含采集等待、文件读取、通知投递 |

B来源：`https://github.com/Bartley007/WI-FI/tree/Module-B`，下载提交 `2963e7dd69891b2d3f2bfa0a8aa318b974e3ed91`。只合入该模块拥有的两个目录，记录见 `backend/modules/human_activity_recognition/docs/IMPORT_PROVENANCE.json`。原始B文档中的真实数据分数是作者报告，不能当成本机重新验证的结果。网站演示fixture的`mock=true`保留。

## C当前冻结什么

演示候选为D：官方五层2D-CNN，GCV输入`[B,1,400,90]`，种子42，最佳第15轮；每窗口8000包，训练人员01/02拟合的逐特征标准化，阈值0.5。

检查点：`backend/modules/fall_detection/outputs/tcs_cross_subject_v1/comparison_seed42_v1/SA_CT_standardized_GCV/best.pt`。

SHA256：`77a87c8b6da3552dab1cd827429ee08e878d8f9d7d5daefcc8b21c577c40190e`。

标准化必须读取checkpoint的`normalization_parameters`。基础`preprocess_config`中复制的旧描述字段不能替代真实参数。新推理适配器已检查模型和预处理源码哈希。

本次不把居中截取实验直接改成线上方案：居中截取需要事先知道录制结束点。此前居中实验验证F1为97.56%，仍属开发对照；网站保留冻结方案原始SA验证F1 **92.31%**、Recall **90%**。

## 演示顺序

1. 打开活动页，指出代码已接入，但当前时间线是明确标注的fixture，不能宣称本机B模型已推理。
2. 打开跌倒页，点击Evaluation，默认 `3/a_03_13_00.dat`。
3. 点Full trace查看整条真实分数曲线，橙线为0.5阈值，红线为137秒名义跌倒标注。
4. 用滑块到第26个窗口：33.022秒出现0.509分的误报警，诚实展示局限。
5. 点Near annotated event，再播放录制，观察跌倒附近的评分升高。可切换全部12条验证录制。
6. 打开融合页，说明它演示模块接口与规则流程，各输入不是同一次同步录制，不能报告联合准确率。

页面`normal/suspected`表达模型分数；没有添加新的连续确认、冷却、短信或真实事故确认规则。真实标注不会用于生成预测。

## 当前可以报告的C结果

- 单动作验证：60Fall/80Nonfall，TP54、FP3、FN6、TN77；Fall Precision94.74%、Recall90%、F192.31%。
- 12条连续验证录制：按固定新报警起点口径检出12/12，初始背景共2次误报警，约22.4分钟有效背景。折算5.36次/小时是短时率估计，不是全天实测。
- 名义标注至报警窗口端点中位数2.62秒；标注的onset/impact与时间零点尚未独立核对，不能称为完整端到端报警延迟。
- 本机CPU、float32、batch1、2线程，预热3次/测量30次：前向中位数28.2ms、P9532.1ms；内存包解码+滤波+GCV+归一化+模型中位数222.3ms、P95241.5ms。排除模型加载、文件/ZIP读取、包收集等待与通知。

没有独立人员04的最终测试结果，不要在展示或报告中写成已经测试通过。原HP模型、原TCS单人模型和其余三个新模型均保留。

## 独立原始文件推理与复现耗时

只看网站不需要以下依赖。要运行模型，先建立Python虚拟环境并安装 `backend/modules/fall_detection/requirements.txt`。

```bash
# 对一个已下载的完整原始文件推理
python backend/modules/fall_detection/tcs_cross_subject_v1/infer_frozen.py --file /path/to/example.dat

# 本机原3.zip中一个正常窗口
python backend/modules/fall_detection/tcs_cross_subject_v1/infer_frozen.py \
  --archive /Users/chenxiaohe/Downloads/3.zip --member 3/a_03_13_00.dat --start-packet 0

# 同一录制中跌倒附近的窗口
python backend/modules/fall_detection/tcs_cross_subject_v1/infer_frozen.py \
  --archive /Users/chenxiaohe/Downloads/3.zip --member 3/a_03_13_00.dat --start-packet 132000

# 包含预处理的计算耗时；不重新训练
python backend/modules/fall_detection/tcs_cross_subject_v1/infer_frozen.py \
  --archive /Users/chenxiaohe/Downloads/3.zip --member 3/a_03_13_00.dat --start-packet 25000 --benchmark-repeats 30
```

标准化、标签、模型哈希不匹配时明确报错，不补零、不隐式接受不完整帧。该命令的输出评分不是经过校准的事故概率。

## 验证与剩余事项

本次已通过现有后端13项、B模块45项和新增B/C集成5项，共63项测试；前端构建通过。新增测试核对所有1742个真实回放分数、12条录制的时间线、B/C数据来源区别、API与离线包一致性，保留原始混淆矩阵。

```bash
python -m unittest discover -s backend/tests -t .
OMP_NUM_THREADS=2 python -m unittest discover -s backend/modules/human_activity_recognition/tests -t .
npm run build
```

剩余优先级：先在提交用电脑走一遍上述演示并确认能重启，再完成报告与截图。B若需要真实模型推理，仍需同学提供训练权重、预处理参数、标签映射和一条可核对的输入/输出样例；不凭文档分数伪造真实回放。若课程要求独立测试，再先明确冻结最终实验协议后执行一次人员04测试；当前不自动开展新的结构搜索或阈值搜索。
