"""Present actual downloaded-archive audit evidence; no training or inference."""
import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODULE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report-dir', type=Path, required=True)
    ap.add_argument('--pilot-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.report_dir
    audit = json.loads((out / 'audit.json').read_text())
    rows = json.loads((out / 'records.json').read_text())
    inventory = json.loads((out / 'archive_inventory.json').read_text())
    annotated = sorted((r for r in rows if r.get('annotation_available')), key=lambda r: r['archive_member'])
    errors = audit['records_with_errors']
    pilot_row = next(r for r in rows if r['archive_member'] == '1/a_01_10_00.dat')
    amplitude = np.load(args.pilot_dir / 'pilot_native_tx0_amplitude_90.npy', mmap_mode='r')
    elapsed = np.load(args.pilot_dir / 'pilot_packet_elapsed_seconds.npy', mmap_mode='r')
    fall = pilot_row['annotation_record']['recorded_fall_time_seconds']
    traces = [0, 15, 45, 75]
    names = ['Tx0 / Rx0 / SC0', 'Tx0 / Rx0 / SC15', 'Tx0 / Rx1 / SC15', 'Tx0 / Rx2 / SC15']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for feature, label in zip(traces, names):
        axes[0, 0].plot(elapsed, amplitude[:, feature], lw=0.45, alpha=0.8, label=label)
        mask = (elapsed >= fall - 10) & (elapsed <= fall + 10)
        axes[0, 1].plot(elapsed[mask], amplitude[mask, feature], lw=0.6, label=label)
    for ax in axes[0]:
        ax.axvline(fall, color='#dc2626', ls='--', lw=1.4, label='Nominal fall annotation')
        ax.set_xlabel('Seconds since first received packet'); ax.set_ylabel('RF-scaled CSI amplitude')
        ax.grid(alpha=0.18)
    axes[0, 0].set_title('One full native recording (no model predictions)')
    axes[0, 0].legend(fontsize=8, loc='upper left')
    axes[0, 1].set_title('Annotation region; time-zero alignment is unverified')
    axes[1, 0].plot(elapsed, elapsed - np.arange(len(elapsed)) / 1000, color='#2563eb')
    axes[1, 0].set_xlabel('Seconds since first received packet')
    axes[1, 0].set_ylabel('Timestamp minus packet-index / 1000 (s)')
    axes[1, 0].set_title('Packet count is not an exact clock')
    axes[1, 0].grid(alpha=0.18)
    ids = [f"{r['activity_id']}/{r['repeat_id']}" for r in annotated]
    diff = [r['raw_duration_minus_annotation_seconds'] for r in annotated]
    axes[1, 1].bar(ids, diff, color='#e08b28')
    axes[1, 1].set_xlabel('Activity ID / repetition index')
    axes[1, 1].set_ylabel('Raw span minus stated duration (s)')
    axes[1, 1].set_title('Twelve structural annotation matches')
    axes[1, 1].grid(axis='y', alpha=0.18)
    fig.suptitle('TCS-Fall subject-01 archive audit | Native Intel5300 data | No training or alarm evaluation', fontsize=13)
    plot = out / 'continuous_data_checks.png'
    fig.savefig(plot, dpi=160); plt.close(fig)

    manifest = []
    archive = audit['source_archives'][0]['file']
    for row in sorted(rows, key=lambda r: r['archive_member']):
        parsed = re.fullmatch(r'\d+/a_(\d+)_(\d+)_(\d+)\.dat', row['archive_member'])
        subject, activity, repeat = map(int, parsed.groups())
        ann = row.get('annotation_record') or {}
        manifest.append({
            'sample_id': Path(row['archive_member']).stem, 'archive_path': archive,
            'archive_member': row['archive_member'], 'nominal_subject_id': subject,
            'activity_id': activity, 'repeat_id': repeat,
            'sample_type': row.get('source_kind', 'unreadable_record'),
            'binary_label': row.get('single_action_binary_label'),
            'annotation_fall_seconds': ann.get('recorded_fall_time_seconds'),
            'annotation_total_seconds': ann.get('recorded_duration_seconds'),
            'annotation_time_origin_verified': False,
            'packet_count': row.get('packet_count'), 'packet_csi_shape': row.get('packet_csi_shape'),
            'timestamp_span_seconds': row.get('timestamp_span_seconds'),
            'finite_ratio_rf_scaled_amplitude': row.get('rf_scaled_amplitude_finite_ratio'),
            'raw_file_sha256': row.get('raw_file_sha256'),
            'canonical_csi_sha256': row.get('canonical_csi_sha256'),
            'crc_ok': row.get('zip_crc32_verified', False),
            'split': 'unassigned; no training/validation/test split created',
            'status': 'error' if row['errors'] else ('unknown_activity_excluded' if activity in (16, 17) else 'audited'),
            'errors': '; '.join(row['errors']),
        })
    with (out / 'sample_manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)
    unknown = [r for r in rows if r.get('source_kind') == 'unknown_activity_excluded']
    (out / 'records_needing_review.json').write_text(json.dumps({'archive_corruption': errors,
        'unknown_activity_records': [r['archive_member'] for r in unknown],
        'all_12_continuous_time_zero_alignment_requires_verification': True}, indent=2))

    workbook = MODULE / 'outputs/tcs_fall_local_v1/source_package/TCS-Fall-main/actual fall time.xlsx'
    source = MODULE / 'outputs/tcs_fall_local_v1/source_package/TCS-Fall-main/code'
    lines = ['# TCS-Fall 本地下载数据审计', '',
        '审计已完成，但 1.zip 尚未通过完整性验收：226 条记录中，223 条可读并完成检查，3 条损坏。全部 12 条有时间标注的连续录制均可读。没有重新训练、调整阈值、执行现有 HP 模型或评价测试集。', '',
        '## 实际取得哪些数据', '',
        '| 压缩包 | 字节数 | 内容 |', '|---|---:|---|',
        f"| 1.zip | {inventory[0]['bytes']:,} | 226 条原始 .dat，名义受试者编号均为 01 |",
        f"| TCS-Fall-main.zip | {inventory[1]['bytes']:,} | 代码、时间表、7 条旧演示数据；完整 CRC 检查通过 |", '',
        '这两个包就是上一阶段请求先下载的小样本资料。当前原始数据覆盖名义受试者 1，尚未取得完整 20 人数据。代码包里的演示文件位于 dataset/1，但文件名包含 a_05；它们单独保留，不将其目录名当作受试者身份，也不混入新的 1.zip 统计。', '',
        '原始压缩包均保持在 Downloads；没有解压复制整个 2.4 GB 数据包。审计逐条从 ZIP 读取并校验，时间曲线使用的一个原生幅度矩阵单独保存在忽略目录。归档 SHA256 见 archive_inventory.json。', '',
        '## 完整性发现', '', '| 记录 | Python ZIP 检查 | 独立 unzip 检查 |', '|---|---|---|',
        '| a_01_03_22.dat | CRC-32 失败 | 无法正常解压压缩数据 |',
        '| a_01_03_23.dat | 本地文件头签名错误 | ZIP 本地文件头偏移无效 |',
        '| a_01_03_24.dat | 本地文件头签名错误 | ZIP 本地文件头偏移无效 |', '',
        '两种工具独立确认这三条有问题。现在无法判断是下载过程还是发布包本身损坏。未修复原包、填造内容或静默跳过；3 条记录仍在清单中，状态为 error。审计命令以退出码 2 表示数据验收失败，同时保存其他记录的实际检查证据。', '',
        '建议重新下载 1.zip，保留旧包以便比对。可把新包命名为 1_redownload.zip，再检查相同三条。如果新旧 SHA256 完全相同且仍损坏，应考虑源包问题或改用另一官方渠道。', '',
        '## 样本和标签统计', '',
        '| 类型 | 包目录声明 | 实际可读 | 用途与限制 |', '|---|---:|---:|---|',
        '| 单动作 ID 0–2 | 90 | 90 | 按官方训练代码映射 Fall=1 |',
        '| 单动作 ID 3–6 | 120 | 117 | 按官方训练代码映射 Nonfall=0；损坏的三条不计入可用数 |',
        '| 连续录制 ID 10–15 | 12 | 12 | 每条有一项名义跌倒时间，不能把整条录制所有时间都标成 Fall |',
        '| 活动 ID 16、17 | 4 | 4 | 语义和事件标注未确认，单独排除，不当作非跌倒或误报分母 |', '',
        '标签依据是训练脚本对活动编号的明确映射；文件名字段解释仍是目录/文件名、官方 metadata 顺序和时间表时长的交叉推断，未取得缺失预处理程序的命名函数。受试者真实身份与标注事件含义未独立验证。', '',
        '## 连续录制与时间表', '',
        f"已把 12 条记录按名义受试者/活动/重复编号结构匹配到[原始时间表](<{workbook.resolve()}>)(Sheet2 F2:K2)。下载时间表 SHA256 与上一阶段只读审计完全一致。12 条原始时间戳跨度合计 {audit['nominal_annotated_continuous_raw_hours']*60:.3f} 分钟；时间表所记总时长合计 {audit['annotation_stated_hours_for_these_records']*60:.3f} 分钟。", '',
        '| 活动/重复 | 时间表总时长 s | 原始时间戳跨度 s | 差值 s | 名义跌倒时间 s |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in annotated:
        ann = row['annotation_record']
        lines.append(f"| {row['activity_id']}/{row['repeat_id']} | {ann['recorded_duration_seconds']} | {row['timestamp_span_seconds']:.6f} | {row['raw_duration_minus_annotation_seconds']:+.6f} | {ann['recorded_fall_time_seconds']} |")
    align = [r['packet_index_event_alignment_error_seconds'] for r in annotated]
    lines += ['',
        '全部名义跌倒时间均落在相应原始跨度之内，但这只证明时间值可落入录制范围，不能证明标注已经精确对齐。原始跨度比表中时长多约 0.81–1.58 秒，可能与记录起止、人工计时和取整有关，当前没有依据选定其中某个解释，也未擅自平移标注。', '',
        f"若直接把第 fall_time×1000 个包当作跌倒时刻，这 12 条记录的实际时间戳会比名义时间晚约 {min(align):.3f}–{max(align):.3f} 秒。最大观测包间隔为 {audit['max_observed_packet_gap_ms']:.3f} 毫秒。因此包序号除以 1000 不是准确时间轴。", '',
        '可读记录没有非递增时间戳。bfee_count 的连续性只能检查驱动测量到用户空间这段计数定义内的缺失，不能保证无线链路每个计划发送都被接收。所有间隔及计数器检查保存在逐条 records.json。', '',
        '标注只有整数秒，官方评价代码另外使用 ±3 秒容差。当前没有证据把标注解释为精确的跌倒开始或触地时刻，不能报告毫秒级真实报警延迟。', '',
        '## 输入和数值检查', '',
        f"223 条可读记录共有 {sum(r.get('packet_count',0) for r in rows):,} 个 CSI 包，全部检查了帧长度、CSI 结构、天线排列、时间戳、计数器，并分块解码和计算 RF 缩放幅度。检查了 {audit['decoded_and_rf_scaled_values_checked']:,} 个缩放幅度值，未发现 NaN/Inf。", '',
        '每包原始结构为 `[Tx=3,Rx=3,报告子载波位置=30]`。按官方可视化代码的 reshape 后取前 90 项，含义是第一个 Tx 流 × 3 个 Rx × 30 个位置，形成 `[time,90]`。这是已核实的本机解析轴序；不能等同于 HP 的 232 个物理特征位置。', '',
        f"试验性保留的一条连续录制原生幅度矩阵为 `{list(amplitude.shape)}`，dtype 为 `{amplitude.dtype}`；没有补到 232，没有重采样、训练归一化、滤波或切训练窗口。图中时间来自包时间戳，RF 幅度已经进行接收天线顺序校正。", '',
        '新实现与已审阅官方解析函数在首条连续录制的 16 个实际包上交叉核对：复数值和天线排列完全一致；缩放幅度最大绝对差为 1.52587890625e-5，小于 1e-4 检查容差。另对活动 12 的 21 个实际包检查，覆盖全部 6 种已观测天线排列编码，复数值完全一致，幅度差仍为 1.52587890625e-5；证据见 reference_all_antenna_codes_check.json。全量解码复用同一实现。', '',
        '原始文件哈希和规范化 CSI 内容哈希均未发现可读记录之间的完全相同组。内容哈希使用统一形状、校正 Rx 顺序后的 int8 I/Q，不包含时间戳/RSSI；它只检查精确相同的原生量化 CSI，不代表排除了近似重复或同源录制。没有创建任何训练/验证/测试划分。', '',
        f'![连续数据及时间轴检查]({plot.resolve()})', '',
        '红虚线只是原时间表中的名义跌倒时间，时间零点关系尚未独立确认。曲线检查读取、数值和时间轴，不证明模型能识别跌倒；图中没有模型预测或报警。', '',
        '## 代码包补齐了什么', '',
        f"实际查看了[可视化笔记本](<{(source/'csi-view.ipynb').resolve()}>)的代码单元，而未执行整个笔记本。它提供 RF 缩放后选第一个 Tx、每 20 点计算变异系数，以及 Butterworth/filtfilt 示例。8 秒、1000 Hz、每 20 点分组时，尺寸可得到 `[400,90]`。这一尺寸来自代码和采样设计，尚未替代缺失的正式预处理实现。", '',
        'preprocess.ipynb 在下载代码包中仍然不存在；data_xx_1x3_butter0.2_cv20.npy、data_yys.npy 和可直接加载的训练权重也未提供。下载 README 是较旧版本，滤波描述与当前训练/可视化代码不完全一致。因此不能声称已经完整复现了 TCS-Fall。', '',
        'filtfilt 是双向滤波：如果对完整录制先滤波再声称逐时在线输出，会使用未来数据。后续必须明确离线复现与只使用当前/历史数据的在线流程；本次没有把双向滤波用于在线预测。', '',
        '## 当前可做与下一步', '',
        '已可以检查这 12 条连续录制、做不含模型预测的回放、验证时间缓冲与输入适配。尚不能执行可信的跨受试者训练/验证/测试：只有一个名义受试者，不能把其重叠窗口随机拆开来替代跨人划分。', '',
        '下一步先处理 1.zip 的三条损坏记录，并取得其他受试者原始包。之后约定独立 TCS-Fall 实验的数据划分和预处理；保留 HP 冻结基线，不将其 232 特征归一化或权重硬套到 Intel5300。窗口、阈值和报警规则只在训练/验证阶段确定，最后再做事件级评价。', '',
        '事件 Recall、误报/小时、真实报警延迟：尚未运行。14 项旧代码、归一化和模型文件的哈希与此前保护记录一致。', '',
        '## 复现命令', '',
        '从仓库根目录运行，并给审计使用一个新的输出目录：', '', '```bash',
        '.venv/bin/python backend/modules/fall_detection/tcs_fall_audit_v1/audit_local_archives.py --raw-zip /Users/chenxiaohe/Downloads/1.zip --code-zip /Users/chenxiaohe/Downloads/TCS-Fall-main.zip --output backend/modules/fall_detection/reports/tcs_fall_local_v1_repeat --pilot-output backend/modules/fall_detection/outputs/tcs_fall_local_v1/pilot_repeat',
        '.venv/bin/python backend/modules/fall_detection/tcs_fall_audit_v1/report_local_archives.py --report-dir backend/modules/fall_detection/reports/tcs_fall_local_v1_repeat --pilot-dir backend/modules/fall_detection/outputs/tcs_fall_local_v1/pilot_repeat',
        '```', '',
        '原数据完整性失败时审计程序退出码为 2，仍会输出诊断报告；修复后的新压缩包通过则为 0。sample_manifest.csv 一条记录一行，包含损坏与未知活动记录；连续录制没有被赋予整个录制统一的二分类标签。', '',
        '来源：[TCS-Fall 官方仓库](https://github.com/zhouziyu2050/TCS-Fall)、本地下载的原始 ZIP 和代码 ZIP；来源哈希、逐条证据和独立 unzip 输出均保存在同目录。']
    (out / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    print(out / 'report_zh.md')


if __name__ == '__main__':
    main()
