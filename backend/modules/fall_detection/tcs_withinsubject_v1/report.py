"""Plot and explain actual TCS within-subject training results."""
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
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    out = args.report
    summary = json.loads((out / 'training_summary.json').read_text())
    config = json.loads((out / 'run_config.json').read_text())
    prep = json.loads((args.data_dir / 'preparation.json').read_text())
    checks = json.loads((out / 'shape_smoke_check.json').read_text())
    reload_check = json.loads((out / 'independent_reload_check.json').read_text())
    with (out / 'history.csv').open(newline='') as stream:
        history = list(csv.DictReader(stream))
    epochs = np.array([int(r['epoch']) for r in history])
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax in axes[:2]:
        ax.plot(epochs, [float(r['train_loss']) for r in history], label='Train loss', color='#2563eb')
        ax.plot(epochs, [float(r['val_loss']) for r in history], label='Validation loss', color='#e08b28')
        ax.axvline(summary['best_epoch'], ls='--', color='#64748b', label='Selected checkpoint')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-entropy'); ax.grid(alpha=0.2)
    axes[0].set_title('Training and validation loss'); axes[0].legend(fontsize=8)
    axes[1].set_yscale('log'); axes[1].set_title('Same losses, logarithmic scale')
    for key, name, color in [('precision', 'Fall Precision', '#2563eb'), ('recall', 'Fall Recall', '#e08b28'),
                             ('f1', 'Fall F1', '#16a34a')]:
        axes[2].plot(epochs, [float(r['val_' + key]) for r in history], label=name, color=color)
    axes[2].set_ylim(0, 1.04); axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Validation score at threshold 0.5')
    axes[2].set_title('Only 42 within-subject validation recordings'); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.2)
    fig.suptitle('TCS-Fall GCV + official CNN | subject01 pilot | seed42 | no test or continuous evaluation', fontsize=13)
    figure = out / 'training_curves.png'
    fig.savefig(figure, dpi=170); plt.close(fig)
    metrics = summary['best_validation_metrics_at_0_5']
    lines = ['# TCS-Fall 受试者内 GCV＋CNN 训练结果', '',
        '已按用户要求，在等待损坏文件重新下载期间，用已通过审计的可读单动作录制完成一个独立试验。现有 HP/CSI-Bench 模型及阈值保留。此结果只表示名义受试者1、单动作片段的受试者内验证，不能当作跨用户、连续监测或严格原论文复现结果。', '',
        '## 数据如何使用', '',
        '| 集合 | 总数 | Fall | Nonfall |', '|---|---:|---:|---:|',
        '| 训练 | 165 | 72 | 93 |', '| 验证 | 42 | 18 | 24 |', '',
        '原始记录先按活动编号分层：每个活动固定6条完整录制作为验证，其余作为训练，种子42。每个文件只产生一个模型样本，没有滑动窗口。3条损坏记录的排除是用户明确要求先训练后的已披露选择；4条活动16/17不明记录排除。12条连续录制保留，本次准备特征和训练均未读取其 CSI。', '',
        '训练/验证之间没有相同录制 ID，也没有已审计的精确相同规范化 I/Q 内容。它们来自同一个人的重复动作，录制会话、邻近重复的独立性尚未建立，不能把这种划分解释为跨人或跨会话泛化。未生成随机测试集。', '',
        '## 输入怎样构建', '',
        '公开 csi-view.ipynb 提供的代码依据：RF缩放幅度按 `[Tx,Rx,30]` 排列，取第一个Tx的90项，保留前8000个包，使用8阶 Butterworth、归一化截止0.2的 filtfilt，再每20包计算 std/mean。这里只在当前8000包内部执行双向滤波，不读取窗口外的未来数据，也不声称已经完成逐包在线报警。', '',
        '输入变换为：`[8000,90] → [400,20,90] → GCV[400,90] → CNN[batch,1,400,90]`。std使用总体标准差ddof=0；均值严格为0的位置按公开代码返回0。实际特征全部有限，没有NaN填充、HP归一化、补到232、重采样或增强。', '',
        f"207条特征缓存的形状为 `{prep['shape']}`，范围为 {prep['gcv_range'][0]:.8f} 至 {prep['gcv_range'][1]:.6f}。前8000包的实际时间戳跨度为 {prep['selected_timestamp_span_range_seconds'][0]:.6f}–{prep['selected_timestamp_span_range_seconds'][1]:.6f} 秒，并非严格均匀的8秒。", '',
        '每条原始短录制尾部被舍弃的包数已逐条记录在 preparation.json。完整 preprocess.ipynb 缺失，单动作记录没有事件时刻标注，因此不能证明每条前8000包都保留了完整跌倒过程，也不能声称与官方完整预处理逐项等价。本次是公开代码参考下的明确重建方案，未擅自改变原始时间轴或补造标注。', '',
        '## 模型、形状和冒烟检查', '',
        '保留公开代码CNN类的全部层和参数：二维卷积在 GCV时间组和特征位置两个方向移动。这里的二维矩阵是信号特征，不是相机图片。随机初始化，无预训练权重。', '',
        '| 层组 | 输出形状，示例batch=1 |', '|---|---|']
    for name, shape in checks['layer_outputs'].items():
        lines.append(f'| {name} | `{shape}` |')
    lines += ['',
        f"参数量 {checks['parameter_count']:,}。五组卷积后展平为1280项，最后输出2个logit，对应 Nonfall=0、Fall=1。推理用Softmax第1列作为Fall评分，固定阈值0.5。评分没有做概率校准。", '',
        f"冒烟训练用8条固定训练样本、两类各4条，更新2步，损失实际为 {checks['smoke_losses'][0]:.6f}、{checks['smoke_losses'][1]:.6f}，前向输出和反向梯度均有限。Dropout下两次损失不要求单调下降。正式训练重新初始化模型和随机种子，没有沿用冒烟更新。", '',
        '## 正式训练与实际验证结果', '',
        'Adam，学习率0.0001，不带类别权重的交叉熵；batch=8（为本机M2、8GiB内存保留空间，官方脚本原设32）。种子42，最多100轮，验证损失连续10轮不改善就早停，最低验证损失保存best.pt。验证采用eval和禁用梯度。', '',
        f"本次共运行 {summary['epochs_run']} 轮，最佳模型在第 {summary['best_epoch']} 轮，最佳验证损失为 {summary['best_val_loss']:.8f}，总耗时 {summary['elapsed_seconds']:.1f} 秒。是否因耐心值早停：{summary['stopped_by_patience']}。", '',
        '| 指标 | 最佳模型验证结果，阈值0.5 |', '|---|---:|',
        f"| Fall Precision | {metrics['precision']:.2%} |", f"| Fall Recall | {metrics['recall']:.2%} |",
        f"| Fall F1 | {metrics['f1']:.2%} |", f"| Accuracy | {metrics['accuracy']:.2%} |",
        f"| TP / FP / FN / TN | {metrics['tp']} / {metrics['fp']} / {metrics['fn']} / {metrics['tn']} |", '',
        f'![训练与验证曲线]({figure.resolve()})', '',
        '验证集只有42条同一人的重复动作，分数很高时也不能推出新受试者表现，更不能与CSI-Bench的独立测试指标直接比较。训练损失在train模式下含Dropout，验证损失在eval模式下计算；单凭两条损失曲线的差值无法确定差异原因。没有据此调阈值、换划分或选择另一套预处理。', '',
        '## 保存与独立加载', '',
        f"最佳模型：[best.pt]({summary['checkpoint']})。保存了模型权重、模型/预处理配置、标签映射、种子、最佳轮次、数据版本、排除项、划分 ID 和阈值。SHA256：`{summary['checkpoint_sha256']}`。", '',
        f"新的Python进程用CPU加载最佳checkpoint并重新计算全部42条验证记录。与训练期间MPS保存评分的最大差异为 {reload_check['max_score_difference_vs_saved_mps_best']:.9g}，阈值0.5下判断变化为 {reload_check['prediction_changes']}。", '',
        '另外从ZIP原始记录开始的独立进程单条推理结果保存于本目录 single_record_*.json，并与相同验证记录的缓存评分比较。该检查证明预处理和保存模型能够独立运行，不代表新的泛化试验。', '',
        '## 复现和推理', '', '从仓库根目录运行；新训练必须使用新输出路径，避免覆盖本次结果。', '', '```bash',
        '.venv/bin/python backend/modules/fall_detection/tcs_withinsubject_v1/prepare.py --output backend/modules/fall_detection/outputs/tcs_subject01_gcv_cnn_seed42_repeat',
        '.venv/bin/python backend/modules/fall_detection/tcs_withinsubject_v1/train.py --data-dir backend/modules/fall_detection/outputs/tcs_subject01_gcv_cnn_seed42_repeat --output backend/modules/fall_detection/outputs/tcs_subject01_gcv_cnn_seed42_repeat/run --report backend/modules/fall_detection/reports/tcs_subject01_gcv_cnn_seed42_repeat',
        '```', '',
        '单条原始记录推理命令会在README中给出本次实际验证过的文件名。当前infer命令限制为已审计单动作 ID；连续录制的窗口/报警规则尚未配置，不自动套用。', '',
        '后续仍需更多受试者、独立划分、原始数据修复和连续事件评价。现阶段不报告测试、误报/小时或报警延迟。', '',
        '代码来源：下载包中 TCS-Fall-main/code/models.py 的CNN类，语法树未改动；GPL-3.0许可证保留为LICENSE.upstream。预处理依据已审阅的原始可视化代码和本项目已核对的Intel5300解析实现。']
    (out / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    print(out / 'report_zh.md')


if __name__ == '__main__':
    main()
