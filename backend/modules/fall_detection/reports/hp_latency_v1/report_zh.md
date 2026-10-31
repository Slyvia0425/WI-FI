# HP 设备延迟与资源测量

本次只使用 HP 验证集中的 16 条完整分段（Fall/Nonfall 各 8 条）；每条反复用于计时，并非 600 条独立样本。沿用先前混合设备训练得到的权重，没有新训练 HP 专用模型。没有使用测试集、重训、调阈值或改变现有模型。

机器为 Apple M2、8 GiB 内存，macOS 14.0；CPU 固定 2 个计算线程、1 个 inter-op 线程。PyTorch 2.14.1，FP32，batch=1，eval + inference_mode。CPU/MPS 按顺序执行，无量化、编译或自动 CPU 回退。

每种范围先预热 30 次，再执行 3 轮 × 200 次；表中的分位数由全部 600 次计时计算。MPS 每次计时前后同步。输入适配与既有程序逐元素一致，最大差值为 0；两后端在选定记录上的评分与既有 CPU 验证评分最大差值均小于 1e-4。

## 实际结果

| 模型 | 后端 | 测量范围 | 中位数 ms | P95 ms | P99 ms |
|---|---|---|---:|---:|---:|
| 1D-CNN | CPU | 纯模型前向 | 0.377 | 0.515 | 0.658 |
| 1D-CNN | CPU | 内存预处理到本地判断 | 0.743 | 0.843 | 1.038 |
| 1D-CNN | CPU | HDF5读取到本地判断（缓存已热） | 5.183 | 5.871 | 8.250 |
| ResNet18 | CPU | 纯模型前向 | 42.597 | 48.779 | 55.718 |
| ResNet18 | CPU | 内存预处理到本地判断 | 49.700 | 139.368 | 288.202 |
| ResNet18 | CPU | HDF5读取到本地判断（缓存已热） | 56.646 | 78.666 | 109.461 |
| 1D-CNN | MPS | 纯模型前向 | 0.842 | 1.137 | 1.232 |
| 1D-CNN | MPS | 内存预处理到本地判断 | 1.395 | 1.691 | 1.911 |
| 1D-CNN | MPS | HDF5读取到本地判断（缓存已热） | 6.671 | 7.215 | 8.980 |
| ResNet18 | MPS | 纯模型前向 | 12.909 | 15.944 | 16.485 |
| ResNet18 | MPS | 内存预处理到本地判断 | 13.452 | 16.531 | 19.073 |
| ResNet18 | MPS | HDF5读取到本地判断（缓存已热） | 14.586 | 19.970 | 21.497 |

![HP 延迟比较](/Users/chenxiaohe/Documents/ChatGPT/csi-bench/backend/modules/fall_detection/reports/hp_latency_v1/latency_comparison.png)

P95 表示本次记录中约 95% 的耗时不超过该数值。P99 估计依赖有限重复次数，不能视为绝对最坏耗时保证。

## 怎样解释

1D-CNN 在本机 CPU 上比 MPS 快。对于这个小模型、batch=1，GPU 调度及同步成本可能超过并行计算收益；这是与实际结果一致的解释，没有用 profiler 单独证明各项成本。ResNet18 在 MPS 上明显快于本机 CPU。两者都应在最终部署设备复测。

内存流程从已解析的幅度数组开始，计入原有数学预处理、传输、模型、评分回传和本地阈值判断；它不包含原始无线包解析。使用与既有文件预处理一致的独立内存适配器，逐元素核对通过。纯前向输入已驻留设备，不含预处理、传输或 Sigmoid/Softmax。

文件流程直接调用原有 runtime，包括其中的检查与读取。文件已被多次读取，OS 缓存已热，不等于冷磁盘性能，也不包括 Python 启动或模型加载。原 1D-CNN 的既有 runtime 会先检查读取，再由 Preprocessor 读取，计时保留这一行为。

这是日常桌面环境中的观测，未隔离后台进程、锁定频率或控制温度。ResNet18 CPU 流程出现明显长尾，原始 CSV 未删异常值。各范围分开计时，时间段不同，不能把不同范围的中位数/P95 相减或相加来推算某个步骤的成本。

## 资源与启动

| 模型 | 参数数 | 参数及 buffer MiB | 保存包 MiB | CPU 进程峰值 RSS MiB | MPS 进程峰值 RSS MiB |
|---|---:|---:|---:|---:|---:|
| 1D-CNN | 43,425 | 0.166 | 0.585 | 251.6 | 314.6 |
| ResNet18 | 11,171,266 | 42.652 | 42.913 | 511.1 | 471.4 |

保存包包含配置等元数据，不能全部当成网络权重。RSS 是 Python 进程整体峰值，包含依赖导入、加载、预处理和模型，不能当作模型自身或 GPU 显存的独占用量。

| 模型 | 后端 | 导入至首个结果中位数 ms | 模型加载与校验中位数 ms | 首次文件预测中位数 ms |
|---|---|---:|---:|---:|
| 1D-CNN | CPU | 1160.7 | 40.1 | 6.8 |
| 1D-CNN | MPS | 1467.2 | 44.6 | 300.6 |
| ResNet18 | CPU | 2622.0 | 1398.6 | 55.5 |
| ResNet18 | MPS | 4180.5 | 1539.2 | 1315.2 |

每组为 3 次独立 Python 进程的中位数。这里的“导入至首个结果”从脚本开始执行计时，不包含 Python 解释器开始执行脚本前的开销；父进程记录的完整启动至退出耗时保存在 results.json。新进程不等于电脑刚开机或磁盘缓存为空。

## 对连续监测的含义

当前证明：本机能够较快处理已经形成的一条 HP 分段。没有测量收集窗口、移动间隔、无线数据解析、排队、连续确认或通知发送，更没有测量跌倒发生到报警的真实延迟。500 个时间点仍不能未经核实直接换算成秒。

待确定连续窗口更新间隔后，再比较实际处理耗时与更新间隔，并检查长时间运行是否积压。当前不宣称满足某个实际报警时限。

## 复现命令

从仓库根目录运行，使用新的输出目录：

```bash
.venv/bin/python backend/modules/fall_detection/hp_latency_v1/benchmark.py --output backend/modules/fall_detection/reports/hp_latency_v1_repeat
.venv/bin/python backend/modules/fall_detection/hp_latency_v1/report.py --results backend/modules/fall_detection/reports/hp_latency_v1_repeat/results.json
```

模型及受保护推理文件的运行前后哈希一致。详细协议、原始逐次计时和新进程启动记录保存在同目录。

计时方法参考：[PyTorch Benchmark 官方文档](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)。
