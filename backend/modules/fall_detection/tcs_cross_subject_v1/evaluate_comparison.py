"""Reload completed arms in a fresh process and evaluate subject03 only."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
MODULE=HERE.parent
sys.path.insert(0,str(HERE))
from train_comparison import CNN, batch, validate, sha, protect, save_json


def event_metrics(rows,scores):
    scores=np.asarray(scores)
    if not rows or len(rows)!=len(scores) or not np.isfinite(scores).all():
        raise ValueError('Missing or nonfinite replay predictions')
    if any(b['window_end_seconds']<=a['window_end_seconds'] for a,b in zip(rows,rows[1:])):
        raise ValueError('Replay windows must be strictly time ordered')
    fall=rows[0]['annotation_seconds'];episodes=[];active=None
    for row,score in zip(rows,scores):
        if score>=.5:
            if active is None:
                active={'start_seconds':row['window_end_seconds'],'last_positive_seconds':row['window_end_seconds'],'positive_windows':0}
                episodes.append(active)
            active['last_positive_seconds']=row['window_end_seconds'];active['positive_windows']+=1
        else:active=None
    strict=[e for e in episodes if fall<=e['start_seconds']<=fall+11]
    tolerant=[e for e in episodes if fall-3<=e['start_seconds']<=fall+11]
    background_alarms=[e for e in episodes if e['start_seconds']<=120]
    exposure=120-rows[0]['window_end_seconds'];positive_seconds=0.;positive_windows=0;background_windows=0
    for i,(r,s) in enumerate(zip(rows,scores)):
        if r['window_end_seconds']>120:continue
        background_windows+=1;positive_windows+=int(s>=.5)
        right=min(120,rows[i+1]['window_end_seconds'] if i+1<len(rows) else 120)
        if s>=.5:positive_seconds+=max(0,right-r['window_end_seconds'])
    overlap=[i for i,(r,s) in enumerate(zip(rows,scores)) if s>=.5 and r['window_start_seconds']<=fall+3 and r['window_end_seconds']>=fall-3]
    outside=[i for i,(r,s) in enumerate(zip(rows,scores)) if s>=.5 and (r['window_end_seconds']<fall-3 or r['window_start_seconds']>fall+3)]
    return {'member':rows[0]['member'],'activity':rows[0]['activity'],'repeat':rows[0]['repeat'],'annotation_seconds':fall,
            'strict_new_alert_detected':bool(strict),'tolerant_new_alert_detected':bool(tolerant),
            'strict_nominal_delay_seconds':strict[0]['start_seconds']-fall if strict else None,
            'tolerant_nominal_delay_seconds':tolerant[0]['start_seconds']-fall if tolerant else None,
            'paper_proxy_cwr_hit':bool(overlap),'paper_proxy_newr_success':not bool(outside),
            'background_false_alarm_episodes':len(background_alarms),'background_exposure_seconds':exposure,
            'background_positive_seconds':positive_seconds,'background_positive_windows':positive_windows,
            'background_window_count':background_windows,'episodes':episodes,'total_positive_windows':int((scores>=.5).sum())}


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features',type=Path,required=True);ap.add_argument('--replay',type=Path,required=True)
    ap.add_argument('--models',type=Path,required=True);ap.add_argument('--training-report',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing comparison evaluation')
    training=json.loads((args.training_report/'training_comparison.json').read_text())
    prep=json.loads((args.features/'preparation.json').read_text());rows=json.loads((args.features/'samples.json').read_text())
    replay_prep=json.loads((args.replay/'preparation.json').read_text());windows=json.loads((args.replay/'windows.json').read_text())
    if sha(args.features/'samples.json')!=prep['sample_manifest_sha256']:raise ValueError('Validation sample metadata changed')
    if sha(args.replay/'protocol.json')!=replay_prep['protocol_sha256']:raise ValueError('Frozen replay protocol changed')
    for root,meta in ((args.features,prep),(args.replay,replay_prep)):
        if sha(root/'features.f32')!=meta['feature_cache_sha256']:raise ValueError('Feature cache hash mismatch')
    if sha(args.replay/'windows.json')!=replay_prep['windows_sha256']:raise ValueError('Replay time metadata changed')
    if {r['subject'] for r in windows}!={3}:raise ValueError('Unexpected replay subjects')
    expected=['SA_raw_GCV','SA_CT_raw_GCV','SA_standardized_GCV','SA_CT_standardized_GCV']
    if [r['arm'] for r in training['arms']]!=expected:raise ValueError('Four completed comparison arms required')
    args.output.mkdir(parents=True)
    save_json(args.output/'replay_protocol.json',json.loads((args.replay/'protocol.json').read_text()))
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    cache=np.memmap(args.features/'features.f32',dtype=np.float32,mode='r',shape=tuple(prep['shape']))
    replay_cache=np.memmap(args.replay/'features.f32',dtype=np.float32,mode='r',shape=tuple(replay_prep['shape']))
    indices=[i for i,r in enumerate(rows) if r['role']=='validation']
    by_member=defaultdict(list)
    for i,r in enumerate(windows):by_member[r['member']].append(i)
    results=[];all_scores={};reload_checks=[]
    for run in training['arms']:
        name=run['arm'];path=args.models/name/'best.pt'
        if sha(path)!=run['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['data_version']['feature_cache_sha256']!=prep['feature_cache_sha256'] or state['data_version']['sample_manifest_sha256']!=prep['sample_manifest_sha256']:
            raise ValueError('Checkpoint and validation data provenance differ')
        if state['label_mapping']!={'Nonfall':0,'Fall':1}:raise ValueError('Unexpected label mapping')
        if state['threshold']!=.5 or state['split_identifiers']['validation_subjects']!=[3]:raise ValueError('Frozen model protocol mismatch')
        if sha(MODULE/'tcs_withinsubject_v1/model.py')!=state['run_config']['model_source_sha256']:raise ValueError('Model source changed')
        model=CNN().to('mps').eval();model.load_state_dict(state['model_state_dict'],strict=True)
        norm=state['normalization_parameters'];mean=scale=None
        if norm is not None:mean=np.asarray(norm['mean'],dtype=np.float32);scale=np.asarray(norm['scale'],dtype=np.float32)
        validation,scores=validate(model,cache,rows,indices,mean,scale,'mps')
        with (args.training_report/name/'best_validation_predictions.csv').open() as f:before=list(csv.DictReader(f))
        if [int(r['cache_index']) for r in before]!=indices:raise ValueError('Validation score order mismatch')
        old_scores=np.array([float(r['fall_score']) for r in before]);difference=float(np.max(np.abs(old_scores-scores)))
        flips=int(((old_scores>=.5)!=(scores>=.5)).sum())
        if difference>5e-5 or flips:raise ValueError('New-process validation predictions differ unexpectedly')
        reload_checks.append({'arm':name,'samples_compared':len(scores),'max_abs_score_difference':difference,'class_changes':flips})
        full_scores=[]
        for start in range(0,len(windows),32):
            ix=np.arange(start,min(start+32,len(windows)))
            logits=model(batch(replay_cache,ix,mean,scale,'mps'))
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite continuous logits')
            full_scores.extend(logits.softmax(1)[:,1].cpu().tolist())
        full_scores=np.asarray(full_scores);all_scores[name]=full_scores
        record_results=[event_metrics([windows[i] for i in ix],full_scores[ix]) for ix in by_member.values()]
        exposure=sum(r['background_exposure_seconds'] for r in record_results)
        alarms=sum(r['background_false_alarm_episodes'] for r in record_results)
        strict_delays=[r['strict_nominal_delay_seconds'] for r in record_results if r['strict_nominal_delay_seconds'] is not None]
        continuous={'events':len(record_results),'strict_new_alert_hits':sum(r['strict_new_alert_detected'] for r in record_results),
                    'tolerant_new_alert_hits':sum(r['tolerant_new_alert_detected'] for r in record_results),
                    'paper_proxy_cwr_hits':sum(r['paper_proxy_cwr_hit'] for r in record_results),
                    'paper_proxy_newr_successes':sum(r['paper_proxy_newr_success'] for r in record_results),
                    'background_false_alarm_episodes':alarms,'background_exposure_hours':exposure/3600,
                    'background_false_alarms_per_hour':alarms/(exposure/3600),
                    'background_positive_state_fraction':sum(r['background_positive_seconds'] for r in record_results)/exposure,
                    'background_positive_windows':sum(r['background_positive_windows'] for r in record_results),
                    'background_window_count':sum(r['background_window_count'] for r in record_results),
                    'strict_detected_only_nominal_delay_median_seconds':float(np.median(strict_delays)) if strict_delays else None,
                    'strict_detected_only_nominal_delays_seconds':strict_delays,'per_record':record_results}
        if continuous['background_positive_windows']!=validation['background_positive_windows']:
            raise ValueError('Background classifications differ between training validation and continuous replay')
        result={'arm':name,'best_epoch':run['best_epoch'],'epochs_run':run['epochs_run'],'stop_reason':run['stop_reason'],
                'threshold':.5,'single_action':validation['single_action'],'validation_selection_loss':validation['selection_loss'],
                'continuous':continuous,'checkpoint_sha256':run['checkpoint_sha256'],'test_evaluated':False}
        results.append(result);save_json(args.output/(name+'.json'),result)
        with (args.output/(name+'_windows.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(windows[0])+['fall_score','prediction']);writer.writeheader()
            writer.writerows({**r,'fall_score':float(s),'prediction':int(s>=.5)} for r,s in zip(windows,full_scores))
        print(json.dumps({'evaluated':name,'SA':validation['single_action'],
                          'continuous':{k:v for k,v in continuous.items() if k not in ('per_record','strict_detected_only_nominal_delays_seconds')}}),flush=True)
        del model,state;torch.mps.empty_cache()
    save_json(args.output/'reload_checks.json',reload_checks)
    activity_rows=[]
    for result in results:
        groups=defaultdict(list)
        for record in result['continuous']['per_record']:groups[record['activity']].append(record)
        for activity,group in sorted(groups.items()):
            exposure=sum(r['background_exposure_seconds'] for r in group)
            alarms=sum(r['background_false_alarm_episodes'] for r in group)
            activity_rows.append({'arm':result['arm'],'activity':activity,'recordings':len(group),
                'strict_new_alert_hits':sum(r['strict_new_alert_detected'] for r in group),
                'tolerant_new_alert_hits':sum(r['tolerant_new_alert_detected'] for r in group),
                'background_false_alarm_episodes':alarms,'background_exposure_seconds':exposure,
                'background_false_alarms_per_hour':alarms/exposure*3600,
                'background_positive_state_fraction':sum(r['background_positive_seconds'] for r in group)/exposure,
                'background_positive_windows':sum(r['background_positive_windows'] for r in group),
                'background_windows':sum(r['background_window_count'] for r in group)})
    with (args.output/'activity_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(activity_rows[0]));writer.writeheader();writer.writerows(activity_rows)
    histories={}
    for r in results:
        with (args.training_report/r['arm']/'history.csv').open() as f:histories[r['arm']]=list(csv.DictReader(f))
    for left,right in (('SA_raw_GCV','SA_standardized_GCV'),('SA_CT_raw_GCV','SA_CT_standardized_GCV')):
        for a,b in zip(histories[left],histories[right]):
            if a['draws_sha256']!=b['draws_sha256']:raise ValueError('Paired normalization arms saw different sampling schedules')
    save_json(args.output/'comparison.json',{'arms':results,'same_initial_weights':training['same_initial_weights_verified'],
              'paired_sampling_schedules_verified':True,'fresh_process_reload_checks':reload_checks,
              'protected_artifacts':protect(),'test_evaluated':False})
    short=['A: SA','B: SA+CT','C: SA+Z','D: SA+CT+Z']
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,r,label in zip(axes.flat,results,short):
        h=histories[r['arm']];x=[int(v['epoch']) for v in h]
        ax.plot(x,[max(float(v['train_loss']),1e-10) for v in h],label='training CE')
        ax.plot(x,[float(v['val_loss']) for v in h],label='fixed grouped validation CE')
        ax.axvline(r['best_epoch'],ls='--',color='gray',label=f"best epoch {r['best_epoch']}")
        ax.set_yscale('log');ax.set_xlabel('Epoch');ax.set_ylabel('Loss (log scale)');ax.set_title(label);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Same CNN, seed42 | train subjects01/02, validation03')
    fig.savefig(args.output/'training_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,r,label in zip(axes.flat,results,short):
        h=histories[r['arm']];x=[int(v['epoch']) for v in h]
        for field,name in [('val_sa_recall','SA Fall recall'),('val_sa_f1','SA Fall F1'),
                           ('val_background_positive_window_rate','Background positive windows')]:
            ax.plot(x,[100*float(v[field]) for v in h],label=name)
        ax.axvline(r['best_epoch'],ls='--',color='gray');ax.set_ylim(0,103)
        ax.set_xlabel('Epoch');ax.set_ylabel('Percent');ax.set_title(label);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Validation metrics over training | fixed threshold0.5')
    fig.savefig(args.output/'validation_metric_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True);x=np.arange(4)
    panels=[('Single-action Fall F1 (%)',[100*r['single_action']['f1'] for r in results]),
            ('New-alert recall, no early tolerance (%)',[100*r['continuous']['strict_new_alert_hits']/12 for r in results]),
            ('Background false-alarm episodes/hour',[r['continuous']['background_false_alarms_per_hour'] for r in results]),
            ('Background time in positive state (%)',[100*r['continuous']['background_positive_state_fraction'] for r in results])]
    for ax,(title,values) in zip(axes.flat,panels):
        bars=ax.bar(x,values,color=['#476b8e','#2b9872','#b98330','#8b609f']);ax.set_xticks(x,short,fontsize=9);ax.set_title(title);ax.bar_label(bars,fmt='%.1f');ax.set_ylim(0,max(max(values)*1.18,1));ax.grid(axis='y',alpha=.2)
    fig.suptitle('Validation subject03 only | fixed threshold0.5 | one seed')
    fig.savefig(args.output/'validation_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(17,12),sharey=True,constrained_layout=True)
    fig.get_layout_engine().set(rect=(0,0,1,.96))
    for ax,(member,ix) in zip(axes.flat,by_member.items()):
        t=np.array([windows[i]['window_end_seconds'] for i in ix]);fall=windows[ix[0]]['annotation_seconds']
        for r,label in zip(results,short):ax.plot(t,all_scores[r['arm']][ix],lw=.9,label=label)
        ax.axhline(.5,c='black',ls='--',lw=.6);ax.axvline(fall,c='red',ls=':',lw=.8)
        ax.axvspan(fall-3,fall+11,color='orange',alpha=.12);ax.set_ylim(-.03,1.03)
        ax.set_title(member);ax.set_xlabel('CSI elapsed seconds');ax.set_ylabel('Fall score');ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4)
    fig.savefig(args.output/'continuous_validation_scores.png',dpi=150);plt.close(fig)
    table=[]
    for r,label in zip(results,short):
        c=r['continuous'];s=r['single_action']
        table.append(f"| {label} | {r['best_epoch']}/{r['epochs_run']} | {s['f1']:.1%} | {c['strict_new_alert_hits']}/12 | {c['tolerant_new_alert_hits']}/12 | {c['background_false_alarm_episodes']} | {c['background_false_alarms_per_hour']:.1f} | {c['background_positive_state_fraction']:.1%} |")
    report=f'''# TCS-Fall 四组跨人对照：训练与验证

四组都使用同一个五层CNN、相同初始化、种子42、阈值0.5。训练人员01/02，验证人员03，测试04未评价。A仅SA原GCV；B加入连续背景；C仅SA加训练拟合标准化Z；D同时加入背景和Z。每batch4个Fall与4个Nonfall，加入背景时负样本分为2个SA/2个CT；每轮45步。Adam学习率0.0001，最多100轮，固定共同验证损失连续10轮不改善早停，保存最小损失checkpoint。

共同验证损失是0.5×Fall SA平均CE +0.25×Nonfall SA平均CE +0.25×连续背景平均CE。验证单动作共140条（60Fall/80Nonfall），背景1344个重叠窗口。这是预先记录的受控实验，不能称为论文二十人交叉验证复现；旧的一人模型也不作为同划分对照。

| 组 | 最佳/总轮数 | 单动作Fall F1 | 新报警检出 | 含提前3秒容差检出 | 背景误报警段 | 段/小时 | 背景阳性状态时间占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

新报警指连续阳性段的首次阳性输出。主检出区间是名义跌倒时刻至其后11秒，另报告提前3秒容差；早已持续报警的状态不会自动算作新事件检出。背景误报只统计每条录制初始120秒，扣除首个完整窗口前的预热时间。阳性状态占比同时揭示持续误报，避免一直报警的模型仅因“只产生一次报警”而显得很好。分母、逐窗口分数、每次报警起点和每条录制结果全部保存在同目录JSON/CSV。

![训练曲线]({(args.output/'training_curves.png').resolve()})

![验证召回与误判变化]({(args.output/'validation_metric_curves.png').resolve()})

![同一验证集比较]({(args.output/'validation_comparison.png').resolve()})

![12条验证连续录制分数]({(args.output/'continuous_validation_scores.png').resolve()})

所有模型均在新的Python进程中从best.pt重新加载，逐一复核1484个共同验证分数。结果见reload_checks.json。标准化参数只从对应训练池拟合；未搜索阈值。两组原GCV/标准化配对在共同训练轮次内的采样计划哈希完全相同，初始化哈希也一致。原HP/TCS模型及受保护源码保留。

连续评价只有一位验证人员的12条脚本化录制，重叠窗口不独立，不能据此声称跨人泛化已经充分验证。事件标注为整数秒，时间零点和onset/impact尚未独立对齐；保存的延迟是名义标注到新报警窗口端点的偏移，未包括计算或通知时延。详细JSON同时保留论文NEWR/CWR的时间戳适配代理口径，不能与真实现场指标混同。测试受试者04继续保留，尚未评价。
'''
    (args.output/'report_zh.md').write_text(report)
    print(json.dumps({'evaluation_complete':True,'report':str((args.output/'report_zh.md').resolve()),'test_evaluated':False}),flush=True)


if __name__=='__main__':main()
