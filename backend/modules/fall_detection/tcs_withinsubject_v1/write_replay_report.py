"""Derive alarm-episode table and a Chinese report from frozen replay outputs."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay-dir', type=Path, required=True)
    ap.add_argument('--report-dir', type=Path, required=True)
    args = ap.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.replay_dir / 'replay_summary.json').read_text())
    with (args.replay_dir / 'window_predictions.csv').open(newline='') as f:
        records = defaultdict(list)
        for row in csv.DictReader(f):
            for key in ('window_index', 'activity_id', 'repeat_id', 'start_packet', 'end_packet'):
                row[key] = int(row[key])
            for key in ('window_start_elapsed_seconds', 'window_end_elapsed_seconds', 'actual_window_span_seconds',
                        'fall_score', 'fall_annotation_seconds', 'official_event_interval_start_seconds',
                        'official_event_interval_end_seconds'):
                row[key] = float(row[key])
            for key in ('prediction', 'within_official_event_interval'):
                row[key] = row[key] == 'True'
            records[row['member']].append(row)
    episode_rows = []
    rederived = {}
    for member, rows in records.items():
        rows.sort(key=lambda r: r['window_index'])
        episodes, active = [], []
        for row in rows:
            if row['prediction']:
                if active and row['window_index'] != active[-1]['window_index'] + 1:
                    episodes.append(active); active = []
                active.append(row)
            elif active:
                episodes.append(active); active = []
        if active:
            episodes.append(active)
        detected = any(r['prediction'] and r['within_official_event_interval'] for r in rows)
        matching = [r for r in rows if r['prediction'] and r['within_official_event_interval']]
        latency = matching[0]['window_end_elapsed_seconds'] - rows[0]['fall_annotation_seconds'] if matching else None
        false_count = 0
        for i, ep in enumerate(episodes, 1):
            event_episode = any(r['within_official_event_interval'] for r in ep)
            if not event_episode:
                false_count += 1
            episode_rows.append({
                'member': member, 'episode_index': i,
                'episode_type': 'event-associated' if event_episode else 'false-alarm',
                'start_elapsed_seconds': ep[0]['window_end_elapsed_seconds'],
                'end_elapsed_seconds': ep[-1]['window_end_elapsed_seconds'],
                'duration_between_first_last_positive_seconds': ep[-1]['window_end_elapsed_seconds'] - ep[0]['window_end_elapsed_seconds'],
                'positive_window_count': len(ep), 'max_fall_score': max(r['fall_score'] for r in ep),
                'fall_annotation_seconds': ep[0]['fall_annotation_seconds'],
            })
        rederived[member] = {'event_detected': detected, 'latency': latency,
                             'positive_windows': sum(r['prediction'] for r in rows),
                             'outside_windows': sum(r['prediction'] and not r['within_official_event_interval'] for r in rows),
                             'false_alarm_episodes': false_count}

    # Make sure report generation independently reproduces replay_summary.
    by_member = {r['member']: r for r in summary['per_record']}
    for member, actual in rederived.items():
        expected = by_member[member]
        assert actual['event_detected'] == expected['event_detected_official_tolerance']
        assert actual['positive_windows'] == expected['positive_windows_total']
        assert actual['outside_windows'] == expected['positive_windows_outside_event_interval']
        assert actual['false_alarm_episodes'] == expected['false_alarm_episodes']
        if actual['latency'] is not None:
            assert abs(actual['latency'] - expected['nominal_event_to_alert_latency_seconds']) < 1e-8
    assert len(records) == 12

    with (args.report_dir / 'alarm_episodes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0]) if episode_rows else ['member'])
        writer.writeheader(); writer.writerows(episode_rows)
    (args.report_dir / 'replay_summary.json').write_text(json.dumps(summary, indent=2))
    metric = summary['metrics']
    rows_md = []
    for r in summary['per_record']:
        latency = r['nominal_event_to_alert_latency_seconds']
        rows_md.append('| {activity_id}/{repeat_id} | {n} | {hit} | {lat} | {false} | {outside} |'.format(
            activity_id=r['activity_id'], repeat_id=r['repeat_id'], n=r['window_count'],
            hit='是' if r['event_detected_official_tolerance'] else '否',
            lat='—' if latency is None else f'{latency:+.2f}',
            false=r['false_alarm_episodes'], outside=r['positive_windows_outside_event_interval']))
    false_alarm_rows = [r for r in episode_rows if r['episode_type'] == 'false-alarm']
    false_alarm_rows.sort(key=lambda r: (r['member'], r['start_elapsed_seconds']))
    fa_table = ['| 录制 | 报警起点 s | 报警终点 s | 连续阳性窗口数 | 最高分 |', '|---|---:|---:|---:|---:|']
    for r in false_alarm_rows:
        fa_table.append(f"| {r['member']} | {r['start_elapsed_seconds']:.2f} | {r['end_elapsed_seconds']:.2f} | {r['positive_window_count']} | {r['max_fall_score']:.3f} |")
    minlat = min(metric['latency_detected_events_only_seconds'])
    maxlat = max(metric['latency_detected_events_only_seconds'])
    text = f'''# TCS-Fall 连续录制回放（冻结模型）

## 本次做了什么

使用已训练完成的 `{summary['protocol']['model_version']}`，checkpoint SHA-256 `{summary['protocol']['checkpoint_sha256']}`，阈值固定为 **0.5**。没有重新训练、没有改阈值，也没有新增报警确认规则。回放了 12 条带时间标注的连续录制；压缩包每条录制均通过 ZIP CRC 和此前审计的原始文件 SHA 校验。

一个模型输入是连续 **8,000 个包位置 × 90 个特征**，输出形状 `[1,1,400,90]`；窗口每隔 **1,000 个包位置**前进一次（名义约 1 秒，统计使用真实 CSI 时间戳）。每条记录产生 143–163 个窗口。每个窗口仍按训练时流程做 RF 幅度、Butterworth(8, 0.2) 双向滤波及 20 包一组的 GCV。双向滤波只处理当前已收齐的 8 秒窗口，没有使用窗口结束之后的数据；因此这是需要缓存完整窗口的回放方式，不是逐包零延迟流式推理。

## 事件、报警和延迟的定义

沿用官方 TCS-Fall 评价代码的事件容差：模型对一个窗口报 Fall，且窗口结束时间落在表格名义跌倒时间 **−3 到 +11 秒**内，就算该录制检出一次事件。官方仓库将窗口预测时间记为 `start+8`，并使用 ±3 秒容差。连续相邻采样点均为阳性时合并成一个报警段；若报警段与事件容差区间没有交集，则计作一次误报警。误报/小时的分母是从首个可输出窗口到末个可输出窗口的时长，减去每次事件容差区间。没有设置额外滞回或冷却时间。

“名义报警延迟”是首个区间内阳性窗口的真实结束时间戳，减去表格中的整数秒跌倒时间。原始录制起点与表格计时起点未独立同步，且标注没有确认表示动作开始还是触地，所以它不是可信的真实动作到报警延迟。负值表示模型在表格名义时刻之前报警。

## 实际结果

- **事件检出：{metric['events_detected']}/{metric['events_total']}，回放事件 Recall = {metric['event_recall']:.1%}。** 这只是在这 12 条同一名义受试者录制上、按官方宽容区间得到的回放结果。
- **误报警：{metric['false_alarm_episodes']} 段**，覆盖非事件监控时间 **{metric['non_event_monitor_hours']:.3f} 小时**，折算 **{metric['false_alarms_per_hour']:.1f} 段/小时**。区间外阳性窗口共 {metric['positive_windows_outside_event_interval']} 个；重叠窗口高度相关，不能把窗口数当成独立误报次数。
- **名义报警偏移：中位数 {metric['median_nominal_event_to_alert_latency_seconds']:+.2f} 秒，均值 {metric['mean_nominal_event_to_alert_latency_seconds']:+.2f} 秒，范围 {minlat:+.2f} 到 {maxlat:+.2f} 秒。** 其中负值是相对未同步的整数秒标注提前，不代表模型真正预知跌倒。
- 12 条录制的总 CSI 时间戳跨度约 **31.5 分钟**（此前逐条时间戳审计结果）；回放可用输出时间为 **{metric['eligible_monitor_hours_including_event_interval']*60:.1f} 分钟**；误报率分母另排除了事件容差时间。
- 回放脚本处理全部 12 条的墙钟耗时为 {summary['elapsed_wall_seconds']:.1f} 秒。这是离线批量回放时间，**不是**单次报警计算延迟或通知送达延迟。

## 每条录制

| 活动/重复 | 窗口数 | 官方容差内检出 | 名义报警偏移 s | 误报警段 | 区间外阳性窗口 |
|---|---:|:---:|---:|---:|---:|
{chr(10).join(rows_md)}

明显的误报集中在活动 12 和 13 的四条录制：25 段误报全部来自这四条；活动 13 两条分别有 102 和 127 个阳性窗口。核查报警时间后，25 段都早于对应事件容差区间，最晚一段也比容差起点早约 2.51 秒。这说明全局的 12/12 事件命中掩盖了较强的录制间差异，也说明固定 0.5 阈值下模型在部分日常活动阶段会持续误触发。

## 误报警段

| 录制 | 报警起点 s | 报警终点 s | 连续阳性窗口数 | 最高分 |
|---|---:|---:|---:|---:|
{chr(10).join(fa_table[2:])}

## 图与逐窗口证据

![12 条录制逐窗口跌倒分数，橙色为官方事件容差区间]({(args.replay_dir / 'continuous_scores.png').resolve()})

逐窗口 CSI 时间、窗口跨度、分数、阈值结果见 [`window_predictions.csv`]({(args.replay_dir / 'window_predictions.csv').resolve()})；逐报警段见 [`alarm_episodes.csv`]({(args.report_dir / 'alarm_episodes.csv').resolve()})；完整逐录制摘要见 [`replay_summary.json`]({(args.report_dir / 'replay_summary.json').resolve()})。

## 结论边界

这次回放证明冻结模型能够在这 12 条记录里对名义跌倒时间附近给出阳性输出，但**不能据此认定模型已经适合连续监护**：误报率高，而且录制都来自同一名义受试者，试验之间也不是独立用户。由于时间零点、精确跌倒 onset/impact 未核实，报警延迟只能称为对表格整数秒的偏移；它不能支持毫秒级或真实现场时延结论。计算与通知延迟没有测量。数据压缩包仍有 3 条无关单动作记录损坏，不过这 12 条连续记录全部逐条通过 CRC 和 SHA 校验。

结果只用于当前已授权的连续录制诊断。阈值与 checkpoint 均保持不变；原 HP 及 CSI-Bench 模型、阈值和推理文件保持原样。
'''
    (args.report_dir / 'report_zh.md').write_text(text)
    print(json.dumps({'records': len(records), 'window_rows': sum(len(v) for v in records.values()),
                      'alarm_episodes': len(episode_rows), 'false_alarm_episodes': len(false_alarm_rows),
                      'independent_recheck_matches': True,
                      'report': str((args.report_dir / 'report_zh.md').resolve())}, indent=2))


if __name__ == '__main__':
    main()
