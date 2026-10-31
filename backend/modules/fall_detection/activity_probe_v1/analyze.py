"""Read-only diagnosis and report for the frozen 25-example activity probe."""
import csv,json,sys,hashlib,html
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,h5py,torch
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from backend.modules.fall_detection.audit_data import digest
from backend.modules.fall_detection.activity_probe_v1.probe import protected,read,save,MODULE,B_MODULE,REPORT,LABELS
from backend.modules.fall_detection.final_selection_v1.runtime import load_candidate,predict_files


def describe(a):
 a=np.asarray(a[:,:,0],dtype=np.float64)
 z=(a-a.mean())/max(a.std(ddof=1),1e-8)
 return {'raw_mean':float(a.mean()),'raw_std':float(a.std()),'normalized_temporal_variance_fraction':float(z.var(axis=1).mean()),'normalized_frequency_mean_variance':float(z.mean(axis=1).var()),'mean_adjacent_time_difference':float(np.abs(np.diff(z,axis=1)).mean())}

def main():
 rows=read('records.json');s=read('summary.json');selection=read('selection.json')
 assert len(rows)==25 and all(r['status']=='ok' for r in rows)
 with (MODULE/'reports/sample_manifest.csv').open() as f:manifest=list(csv.DictReader(f))
 byhash=defaultdict(list)
 for r in manifest:
  if r['status']=='ok':byhash[r['content_hash']].append(r)
 pairs=[];desc=[]
 for r in rows:
  with h5py.File(r['source']['local_path'],'r') as h:a=h['CSI_amps'][()]
  h=digest(a)
  pairs.append({'HAR_sample_id':r['sample_id'],'audit_canonical_hash':h,'C_matches':[{k:m[k] for k in ('sample_id','split','original_label','environment','user')} for m in byhash.get(h,[])]})
  desc.append({'sample_id':r['sample_id'],'cohort':'HAR_'+r['source']['label'],'environment':r['source']['environment'],**describe(a)})
 # One C training record per label/environment for descriptive signal inspection only.
 # No model/normalizer fitting; these are not an evaluation cohort.
 ref=[]
 for label in ('Fall','Nonfall'):
  candidates=[r for r in manifest if r['split']=='train_id' and r['device']=='HP' and r['original_label']==label]
  for env in sorted({r['environment'] for r in candidates}):
   row=sorted([r for r in candidates if r['environment']==env],key=lambda r:r['sample_id'])[0]
   with h5py.File(row['file_path'],'r') as h:a=h[row['h5_key']][()]
   ref.append({'sample_id':row['sample_id'],'cohort':'C_train_'+label,'environment':env,'role':'descriptive input reference only; not model evaluation',**describe(a)})
 # New-process model runtime sanity control on two existing C-validation samples
 # whose original predictions were already saved. No C test-set data is accessed.
 old=MODULE/'reports/resnet_b_integration_v1'
 cfg=json.loads((MODULE/'resnet_b_integration/frozen_model.json').read_text());model,state=load_candidate(MODULE/cfg['checkpoint'],'official_resnet18')
 controls=[]
 for case in ('true_positive','true_negative'):
  oldp=json.loads((old/(case+'_new_process.json')).read_text())
  r={'sample_id':case,'file_path':oldp['file'],'device':'HP','binary_label':1 if case=='true_positive' else 0}
  got=predict_files(model,state,'official_resnet18',[r])[0]['fall_score']
  controls.append({'case':case,'saved_score':oldp['fall_score'],'new_score':got,'abs_difference':abs(got-oldp['fall_score']),'same_classification':(got>=cfg['threshold'])==(oldp['fall_score']>=cfg['threshold'])})
  assert abs(got-oldp['fall_score'])<1e-6
 # Author validation session identities can be matched, but their train raw files
 # were not downloaded/audited. Do not claim full B train independence.
 ev=json.loads((B_MODULE/'outputs/released_v1/optimized_final/spectral/seed-42-stable/evaluation.json').read_text())
 sourcekeys={(x['user'],x['environment'],x['device'],str(x['session'])) for x in ev['observations']['validation']}
 overlaps=[r['sample_id'] for r in rows if (r['source']['user'],r['source']['environment'],r['source']['device'],str(r['source']['session'])) in sourcekeys]
 env={}
 for e in sorted({r['source']['environment'] for r in rows}):
  group=[r for r in rows if r['source']['environment']==e]
  env[e]={'n':len(group),'C_Fall_triggers':sum(r['C_triggers_on_HAR_activity'] for r in group),'B_activity_agreement':sum(r['B_matches_official_activity'] for r in group)}
 metrics={}
 for cohort in sorted({r['cohort'] for r in desc+ref}):
  group=[r for r in desc+ref if r['cohort']==cohort]
  metrics[cohort]={'n':len(group),**{key:{'min':min(r[key] for r in group),'median':float(np.median([r[key] for r in group])),'max':max(r[key] for r in group)} for key in ('raw_mean','raw_std','normalized_temporal_variance_fraction','mean_adjacent_time_difference')}}
 details={'by_environment':env,'C_runtime_controls':controls,'HAR_vs_C_raw_content_matches':pairs,'HAR_records_with_any_exact_C_content_match':sum(bool(r['C_matches']) for r in pairs),'hash_comparison':'Reused unchanged C audit manifest numeric content hashes; computed HAR hashes using the same shape+little-endian float64 convention. Equal content indicates duplication needing provenance review; absence does not establish no same-source overlap.','B_author_validation_same_session_count':len(overlaps),'B_author_validation_same_session_ids':overlaps,'descriptive_HAR_signals':desc,'descriptive_C_training_references':ref,'signal_summary':metrics,'confirmed':['All25 official HAR val HP/232/5G samples have finite[232,500,1] arrays; no padding/clipping needed.','Every B and C raw input identity matches;25 raw and normalized-C inputs are distinct.','All25 C tensors match the pinned official __getitem__ normalization/axis code exactly; zero max absolute difference.','C Fall/Nonfall output indices are verified from pinned checkpoint; existing positive/negative validation controls reproduce.','HAR probe environments E01/E02/E03/E04/E06 do not appear in C HP training E21-E26.','C training metadata provides only Fall/Nonfall labels; it does not establish coverage of running/walking/jumping negatives.','C/B weights, runtime/config, training manifest and production replay files have unchanged SHA256 values.'],
'hypotheses_not_confirmed':['Cross-task/environment/acquisition distribution shift may drive C triggers; the cohort confounds task, environment and subjects, so environment causality is not established.','Insufficient diversity of C Nonfall/hard-negative training examples is plausible, but original action-level provenance is unavailable.','Acquisition cadence, exact physical subcarrier correspondence or signal scaling differences may remain despite equal tensor shape; metadata here does not establish physical equivalence.','High softmax scores may reflect overconfidence under shift; no probability calibration or calibration-quality estimate was performed.'],
'next_experiments':['Verify recording-level no-fall intervals with annotations/video or original collection records before calling these strict false positives.','Reserve a larger cross-activity, cross-environment/person holdout before any tuning; report trigger counts by activity, device, environment and recording groups.','To assess fall recall, add a separately annotated positive-fall holdout; do not infer it from this probe.','If failures persist, verify physical time/subcarrier conventions, then compare frozen C and training-only hard-negative/domain adaptation candidates on the same untouched holdout. Use only designated training HAR records after annotation/provenance/duplicate checks, not these25 validation examples.','Keep official C experiment and cross-task/adapted experiment separate; do not silently replace the official split.','Any threshold/rule candidate must be chosen on designated validation data and assessed jointly for false alerts, missed falls and latency, then frozen before a true held-out test.','B activity preferences can be shown for review, but running/walking is not logically incompatible with a fall. Do not suppress C simply because B returns a trained ordinary activity.'],'recovery_notes':['Python requests returned404; all original selections downloaded with curl, initial failures retained.','A test-harness import of the archived TCS protection helper caused a model namespace collision before inference. The helper now executes in its own process; production runtime was not modified. No samples were replaced.']}

 details['confirmed_zh']=[
 '25条均为官方HAR验证划分的HP/232/5G数据，形状[232,500,1]，无NaN/Inf；没有补零或裁剪。',
 '每条B/C原始幅度内容身份一致；25条原始输入及归一化输入均无精确重复。',
 'C逐值匹配固定官方__getitem__的轴转换和归一化，最大绝对差为0；该测试只替代了元数据/路径/标签查询，不测试官方构造器。',
 '检查点确认Fall为softmax第0列；原有一条跌倒、一条非跌倒验证样本重算分数与保存值一致。',
 '本队列环境E01/E02/E03/E04/E06与C的HP训练环境E21–E26不重叠。用户与环境绑定，不能据此分离环境与用户因果。',
 'C训练metadata只有Fall/Nonfall，无法证明其Nonfall包含了哪些跑步、行走、跳跃等具体动作。',
 '生产权重、阈值/运行配置、训练manifest、B/C运行代码及生产回放SHA256均保持不变。'
 ]
 details['hypotheses_zh']=[
 '跨任务、环境、用户或采集条件的分布偏移可能导致触发；目前同时存在多种差异，尚未证明某一差异是根因。',
 'C的Nonfall困难负样本覆盖不足是合理假设，但缺少原录制动作级来源，不能断言训练中从未包含某动作。',
 '相同232×500尺寸不能证明实际采样间隔、物理子载波排列和射频条件相同；这些仍需采集记录核实。',
 '极高softmax分数在本队列中出现，不等于真实事故概率高；目前未做概率校准或校准质量评价。'
 ]
 details['next_experiments_zh']=[
 '先通过原始标注、视频或采集记录确认无跌倒区间，再把触发称为严格误报。',
 '在调参前固定更大、多用户/环境/录制组的跨活动保留集，按活动、设备、环境分别报告触发；补充独立、已标注的跌倒阳性队列评价召回。',
 '持续失败时先核对真实时间/子载波约定，再比较冻结C与加入困难负样本或域适配的候选；训练只使用另行指定并审核的训练样本，不能把这25条验证样本悄悄放入训练。',
 '官方C实验和跨任务/适配实验分别记录，保留原官方划分；检查原始内容重复及同源录制组泄漏。',
 '阈值或报警规则未来只在指定验证集选择，再冻结到独立保留集，同时比较误触发、漏检与延迟；本次没有做这些选择。',
 'B并列展示动作与置信度，供复核；running/walking与跌倒并非逻辑互斥，不能根据B的普通活动偏好自动关闭C。'
 ]
 details['coverage_zh']=[
 '仅HP设备、5G、单人完整分段；没有ESP32、其他设备或多人场景。',
 '每类5条、确定性选择5个用户/环境组，样本很小且非随机代表性抽样。',
 '只有五类HAR活动标签；没有逐帧无跌倒标注，也没有本次跌倒阳性队列。',
 '没有连续录制、报警事件匹配、每小时误报或报警延迟评价。',
 '25条均能匹配作者B既有验证session，因此B标签一致结果不是全新独立泛化基准。',
 '没有在C现有全部样本中找到这些25条的精确数字内容重复；这不能排除非完全相同的同源录制。'
 ]
 assert protected()==read('preserved_before.json')
 save('diagnosis.json',details)
 # Compact static scientific plots, using only actual outputs.
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
 x=np.arange(5);counts=[s['groups'][k]['C_Fall_triggers'] for k in LABELS];matches=[s['groups'][k]['B_top1_matches_activity'] for k in LABELS]
 axes[0].bar(x-.18,counts,.36,label='C Fall triggers (not strict FPR)',color='#ef6351');axes[0].bar(x+.18,matches,.36,label='B activity-label agreement',color='#208a9c');axes[0].set_xticks(x,['jump','run','seated','walk','wave']);axes[0].set_ylim(0,6);axes[0].set_ylabel('Number out of5');axes[0].legend(fontsize=8);axes[0].set_title('Five official HAR validation clips per class')
 for e,j in zip(sorted(env),range(len(env))):
  group=[r for r in rows if r['source']['environment']==e];axes[1].scatter([LABELS.index(r['source']['label'])+(j-2)*.07 for r in group],[r['C_fall_score'] for r in group],label=e,s=30)
 axes[1].axhline(cfg['threshold'],color='#555',linestyle='--',label='Frozen C threshold');axes[1].set_xticks(x,['jump','run','seated','walk','wave']);axes[1].set_ylim(-.04,1.05);axes[1].set_ylabel('C Fall score (uncalibrated)');axes[1].legend(fontsize=7,ncol=2);axes[1].set_title('Actual C scores by activity and environment')
 fig.savefig(REPORT/'activity_results.png',dpi=160);plt.close(fig)
 total=sum(counts);bcorrect=sum(matches)
 envtr=''.join(f"<tr><td>{e}</td><td>{v['n']}</td><td>{v['C_Fall_triggers']}</td><td>{v['B_activity_agreement']}</td></tr>" for e,v in env.items())
 tr=''.join(f"<tr><td>{html.escape(k)}</td><td>5</td><td>{s['groups'][k]['B_top1_matches_activity']}/5</td><td>{s['groups'][k]['C_Fall_triggers']}/5</td><td>{s['groups'][k]['C_score_min']:.6f}–{s['groups'][k]['C_score_max']:.6f}</td></tr>" for k in LABELS)
 def lis(values):return '<ul>'+''.join('<li>'+html.escape(v)+'</li>' for v in values)+'</ul>'
 body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>B/C 多活动诊断</title><style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1080px;margin:40px auto;padding:0 22px;color:#203044;background:#f6f8fb}}section{{background:white;padding:22px;border-radius:14px;margin:20px 0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}.warn{{border-left:5px solid #df713d}}code{{word-break:break-all}}a{{color:#066e91}}</style><h1>B/C 多活动真实数据诊断</h1><p>CSI-Bench v12 · 官方 HAR val_id · HP / 232 / 5G · 五类各5条 · 固定B/C seed42</p><section class="warn"><strong>25条均完成：B活动标签一致 {bcorrect}/25；C触发Fall {total}/25。</strong><p>C计数是HAR活动片段上的触发／疑似误报，不能作为严格误报率；单一活动标签不是逐帧无跌倒真值。本次没有跌倒阳性队列，不能评价跌倒召回率、事件报警率或报警延迟。</p></section><section><h2>实际逐类结果</h2><table><tr><th>官方活动</th><th>数量</th><th>B最高分与标签一致</th><th>C输出Fall</th><th>C评分范围</th></tr>{tr}</table><img src="activity_results.png" alt="25条实际预测统计与散点图"></section><section><h2>已证实的检查结果</h2>{lis(details['confirmed_zh'])}<p>按同一审计内容哈希查到 C 集合精确重复：{details['HAR_records_with_any_exact_C_content_match']}条。与作者B验证记录同一用户/环境/设备/session：{len(overlaps)}条；B结果不是新的独立泛化基准。</p></section><section><h2>环境／用户分组</h2><table><tr><th>环境</th><th>数量</th><th>C输出Fall</th><th>B标签一致</th></tr>{envtr}</table><p>每个环境对应本队列一个用户，每类仅一条，不能把环境与用户影响分开。不是环境总体误报率。</p></section><section><h2>尚未证实的原因</h2>{lis(details['hypotheses_zh'])}</section><section><h2>若持续出现，建议怎样做</h2>{lis(details['next_experiments_zh'])}<p>当前阶段保持模型、阈值、训练数据和生产回放不变。B只作参考并列展示，辅助人工复核。</p></section><section><h2>覆盖与限制</h2>{lis(details['coverage_zh'])}<p>每类5个用户/环境组合为确定性诊断选择，非随机代表性样本，不给出总体性能置信区间。用户与环境在本队列中绑定，不能分离二者因果影响。无ESP32，无连续真实报警验证。</p></section><section><h2>可复查证据</h2><p><a href="results.csv">逐条CSV</a> · <a href="records.json">详细输入与输出</a> · <a href="selection.json">推理前选择协议</a> · <a href="diagnosis.json">诊断与检查证据</a> · <a href="download_status.json">下载核验</a> · <a href="summary.json">统计汇总</a></p><p>独立脚本：backend/modules/fall_detection/activity_probe_v1/probe.py 与 analyze.py。先select、再download、最后evaluate；分析不训练模型。既有选择文件不可覆盖。</p></section></html>'''
 (REPORT/'report_zh.html').write_text(body)
 print(json.dumps({'C_Fall_triggers':total,'B_activity_agreement':bcorrect,'environment_groups':env,'content_overlap_records':details['HAR_records_with_any_exact_C_content_match'],'B_author_val_session_overlap':len(overlaps),'signal_summary':metrics,'controls':controls},indent=2))

if __name__=='__main__':main()
