"""Render measured HP timing results; no additional model inference."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.results.read_text())
    out = args.results.parent
    names = {'original_1d_cnn': '1D-CNN', 'official_resnet18': 'ResNet18'}
    scopes = ['forward_only', 'memory_pipeline', 'file_pipeline_warm_os_cache']
    scope_titles = ['Model forward only', 'Preprocess + transfer + prediction', 'HDF5 + exact saved runtime']
    rows = []
    for run in data['runs']:
        for item in run['scopes']:
            rows.append({'model': run['kind'], 'backend': run['backend'], **item})
    with (out / 'timing_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True)
    labels = [names[r['kind']] + '\n' + r['backend'].upper() for r in data['runs']]
    for ax, scope, title in zip(axes, scopes, scope_titles):
        values = [next(s for s in r['scopes'] if s['scope'] == scope) for r in data['runs']]
        x = np.arange(len(values))
        median = [v['median_ms'] for v in values]
        p95 = [v['p95_ms'] for v in values]
        ax.bar(x - 0.18, median, width=0.36, color='#2563eb', label='Median')
        ax.bar(x + 0.18, p95, width=0.36, color='#e08b28', label='P95')
        ax.set_yscale('log'); ax.set_ylim(0.15, 400)
        ax.set_xticks(x, labels); ax.set_title(title, fontsize=11)
        ax.set_ylabel('Milliseconds (log scale)'); ax.grid(axis='y', alpha=0.2)
        for xx, yy in zip(x - 0.18, median):
            ax.text(xx, yy * 1.13, f'{yy:.2f}', ha='center', va='bottom', fontsize=8)
        for xx, yy in zip(x + 0.18, p95):
            ax.text(xx, yy * 1.13, f'{yy:.2f}', ha='center', va='bottom', fontsize=8)
    axes[0].legend(loc='upper left', frameon=False)
    fig.suptitle('HP validation segments | Apple M2, 8 GiB | batch=1 | 600 timings per bar', fontsize=13)
    fig.savefig(out / 'latency_comparison.png', dpi=170)
    plt.close(fig)
    lines = [
        '# HP 设备延迟与资源测量', '',
        '本次只使用 HP 验证集中的 16 条完整分段（Fall/Nonfall 各 8 条）；每条反复用于计时，并非 600 条独立样本。沿用先前混合设备训练得到的权重，没有新训练 HP 专用模型。没有使用测试集、重训、调阈值或改变现有模型。', '',
        '机器为 Apple M2、8 GiB 内存，macOS 14.0；CPU 固定 2 个计算线程、1 个 inter-op 线程。PyTorch 2.14.1，FP32，batch=1，eval + inference_mode。CPU/MPS 按顺序执行，无量化、编译或自动 CPU 回退。', '',
        '每种范围先预热 30 次，再执行 3 轮 × 200 次；表中的分位数由全部 600 次计时计算。MPS 每次计时前后同步。输入适配与既有程序逐元素一致，最大差值为 0；两后端在选定记录上的评分与既有 CPU 验证评分最大差值均小于 1e-4。', '',
        '## 实际结果', '',
        '| 模型 | 后端 | 测量范围 | 中位数 ms | P95 ms | P99 ms |',
        '|---|---|---|---:|---:|---:|',
    ]
    cn_scopes = dict(zip(scopes, ['纯模型前向', '内存预处理到本地判断', 'HDF5读取到本地判断（缓存已热）']))
    for row in rows:
        lines.append(f"| {names[row['model']]} | {row['backend'].upper()} | {cn_scopes[row['scope']]} | {row['median_ms']:.3f} | {row['p95_ms']:.3f} | {row['p99_ms']:.3f} |")
    lines += ['', f"![HP 延迟比较]({(out / 'latency_comparison.png').resolve()})", '',
              'P95 表示本次记录中约 95% 的耗时不超过该数值。P99 估计依赖有限重复次数，不能视为绝对最坏耗时保证。', '',
              '## 怎样解释', '',
              '1D-CNN 在本机 CPU 上比 MPS 快。对于这个小模型、batch=1，GPU 调度及同步成本可能超过并行计算收益；这是与实际结果一致的解释，没有用 profiler 单独证明各项成本。ResNet18 在 MPS 上明显快于本机 CPU。两者都应在最终部署设备复测。', '',
              '内存流程从已解析的幅度数组开始，计入原有数学预处理、传输、模型、评分回传和本地阈值判断；它不包含原始无线包解析。使用与既有文件预处理一致的独立内存适配器，逐元素核对通过。纯前向输入已驻留设备，不含预处理、传输或 Sigmoid/Softmax。', '',
              '文件流程直接调用原有 runtime，包括其中的检查与读取。文件已被多次读取，OS 缓存已热，不等于冷磁盘性能，也不包括 Python 启动或模型加载。原 1D-CNN 的既有 runtime 会先检查读取，再由 Preprocessor 读取，计时保留这一行为。', '',
              '这是日常桌面环境中的观测，未隔离后台进程、锁定频率或控制温度。ResNet18 CPU 流程出现明显长尾，原始 CSV 未删异常值。各范围分开计时，时间段不同，不能把不同范围的中位数/P95 相减或相加来推算某个步骤的成本。', '',
              '## 资源与启动', '',
              '| 模型 | 参数数 | 参数及 buffer MiB | 保存包 MiB | CPU 进程峰值 RSS MiB | MPS 进程峰值 RSS MiB |',
              '|---|---:|---:|---:|---:|---:|']
    for kind in names:
        cpu = next(r for r in data['runs'] if r['kind'] == kind and r['backend'] == 'cpu')
        mps = next(r for r in data['runs'] if r['kind'] == kind and r['backend'] == 'mps')
        lines.append(f"| {names[kind]} | {cpu['parameter_count']:,} | {cpu['parameter_and_buffer_bytes']/2**20:.3f} | {cpu['checkpoint_file_bytes']/2**20:.3f} | {cpu['process_peak_rss_bytes']/2**20:.1f} | {mps['process_peak_rss_bytes']/2**20:.1f} |")
    lines += ['', '保存包包含配置等元数据，不能全部当成网络权重。RSS 是 Python 进程整体峰值，包含依赖导入、加载、预处理和模型，不能当作模型自身或 GPU 显存的独占用量。', '',
              '| 模型 | 后端 | 导入至首个结果中位数 ms | 模型加载与校验中位数 ms | 首次文件预测中位数 ms |',
              '|---|---|---:|---:|---:|']
    for kind in names:
        for backend in ('cpu', 'mps'):
            cold = [r for r in data['startup'] if r['kind'] == kind and r['backend'] == backend]
            values = [np.median([r[key] for r in cold]) for key in
                      ('process_code_to_first_result_ms', 'load_model_and_integrity_check_ms', 'first_file_pipeline_ms')]
            lines.append(f"| {names[kind]} | {backend.upper()} | {values[0]:.1f} | {values[1]:.1f} | {values[2]:.1f} |")
    lines += ['', '每组为 3 次独立 Python 进程的中位数。这里的“导入至首个结果”从脚本开始执行计时，不包含 Python 解释器开始执行脚本前的开销；父进程记录的完整启动至退出耗时保存在 results.json。新进程不等于电脑刚开机或磁盘缓存为空。', '',
              '## 对连续监测的含义', '',
              '当前证明：本机能够较快处理已经形成的一条 HP 分段。没有测量收集窗口、移动间隔、无线数据解析、排队、连续确认或通知发送，更没有测量跌倒发生到报警的真实延迟。500 个时间点仍不能未经核实直接换算成秒。', '',
              '待确定连续窗口更新间隔后，再比较实际处理耗时与更新间隔，并检查长时间运行是否积压。当前不宣称满足某个实际报警时限。', '',
              '## 复现命令', '', '从仓库根目录运行，使用新的输出目录：', '', '```bash',
              '.venv/bin/python backend/modules/fall_detection/hp_latency_v1/benchmark.py --output backend/modules/fall_detection/reports/hp_latency_v1_repeat',
              '.venv/bin/python backend/modules/fall_detection/hp_latency_v1/report.py --results backend/modules/fall_detection/reports/hp_latency_v1_repeat/results.json',
              '```', '', '模型及受保护推理文件的运行前后哈希一致。详细协议、原始逐次计时和新进程启动记录保存在同目录。', '',
              '计时方法参考：[PyTorch Benchmark 官方文档](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)。']
    (out / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    print(out / 'report_zh.md')


if __name__ == '__main__':
    main()
