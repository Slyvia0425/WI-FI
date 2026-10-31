"""Fresh-process comparison on fixed validation cohorts; no B gate, no test input."""
import argparse,csv,html,json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from common import *

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--original',type=Path);ap.add_argument('--original-predictions',type=Path);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda');ap.add_argument('--batch-size',type=int,default=64);a=ap.parse_args()
 torch.set_num_threads(2);device=torch.device(a.device);a.output.mkdir(parents=True,exist_ok=True)
 audit=json.loads(a.audit.read_text());val=CachedDataset(a.cache,'val');assert sha(a.cache/'val.npy')==audit['cache_sha256']['val']
 loader=torch.utils.data.DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=0)
 assert bool(a.original)!=bool(a.original_predictions),'Choose checkpoint OR verified baseline predictions'
 if a.original:assert sha(a.original)==OLD_SHA,'Original checkpoint changed'
 model,state=load_model(a.candidate,device)
 assert state['config']['selection_sha256']==audit['selection_sha256']
 assert state['split_identifiers']['val']==[r['id'] for r in val.rows]
 assert not set(state['split_identifiers']['train'])&set(state['split_identifiers']['val'])
 new_scores,_=predict(model,loader,device)
 ref=json.loads(a.candidate.with_name('best_reference.json').read_text())
 assert ref['id']==val.rows[0]['id'] and ref['checkpoint_sha256']==sha(a.candidate)
 delta=abs(ref['score']-new_scores[0]);assert delta<5e-5,('Reload score mismatch',delta)
 del model
 if device.type=='cuda':torch.cuda.empty_cache()
 baseline_receipt=None
 if a.original:
  model,old_state=load_model(a.original,device);old_scores,_=predict(model,loader,device)
 else:
  baseline_receipt=json.loads(a.original_predictions.read_text())
  assert baseline_receipt['original_sha256']==OLD_SHA
  assert baseline_receipt['ids']==[r['id'] for r in val.rows]
  assert baseline_receipt['content_hashes']==[r['content_hash'] for r in val.rows]
  if baseline_receipt['cache_sha256']!=audit['cache_sha256']['val']:
   assert baseline_receipt['preprocess_rule']=='torch.float32 permute(2,1,0); per-record mean/std(dim=(1,2),keepdim=True,correction=1),clamp1e-8; no padding'
   stats=[[float(val.x[i].mean()),float(val.x[i].std()),float(val.x[i].min()),float(val.x[i].max())] for i in range(len(val))]
   assert np.allclose(baseline_receipt['normalized_statistics'],stats,rtol=1e-5,atol=1e-5),'Normalization discrepancy beyond platform rounding'
   print('CACHE_PLATFORM_ROUNDOFF: identical raw hashes and rule; float summary checks passed',flush=True)
  old_scores=baseline_receipt['scores'];assert len(old_scores)==len(val) and np.isfinite(old_scores).all()
 results={}
 for t in (.5,OLD_THRESHOLD):
  table={}
  for name,scores in [('original',old_scores),('mixed_HP',new_scores)]:
   sources={}
   for source in ('FallDetection','HAR'):
    ix=[i for i,r in enumerate(val.rows) if r['source']==source]
    sources[source]=metrics([val.rows[i] for i in ix],[scores[i] for i in ix],t)
   sources['HAR_by_activity']={}
   for label in sorted({r['original_label'] for r in val.rows if r['source']=='HAR'}):
    ix=[i for i,r in enumerate(val.rows) if r['source']=='HAR' and r['original_label']==label]
    sources['HAR_by_activity'][label]=metrics([val.rows[i] for i in ix],[scores[i] for i in ix],t)
   table[name]=sources
  results[str(t)]=table
 errors=[]
 for r,o,n in zip(val.rows,old_scores,new_scores):
  correct=lambda s:bool(s>=.5)==(r['label']==0)
  errors.append({**r,'original_fall_score':o,'mixed_fall_score':n,'original_prediction_05':'Fall' if o>=.5 else 'Nonfall','mixed_prediction_05':'Fall' if n>=.5 else 'Nonfall','transition_05':('both_correct' if correct(o) and correct(n) else 'fixed' if not correct(o) and correct(n) else 'regressed' if correct(o) else 'both_wrong')})
 dump(a.output/'validation_predictions.json',errors)
 with (a.output/'validation_predictions.csv').open('w',newline='') as f:
  keys=['id','source','original_label','user','environment','session','label','original_fall_score','mixed_fall_score','original_prediction_05','mixed_prediction_05','transition_05'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(errors)
 # The original run's audit reported a train/validation raw duplicate. Keep the full
 # validation cohort fixed, and additionally show the common subset excluding it.
 raw_path=a.audit.with_name('raw_audit.json')
 clean=None
 if raw_path.exists():
  original_train_hashes={r['content_hash'] for r in json.loads(raw_path.read_text())['records'] if r['source']=='FallDetection' and r['role']=='train'}
  leaked=[i for i,r in enumerate(val.rows) if r['content_hash'] in original_train_hashes]
  ix=[i for i,r in enumerate(val.rows) if r['source']=='FallDetection' and i not in leaked]
  clean={'excluded_ids':[val.rows[i]['id'] for i in leaked],'threshold_05':{name:metrics([val.rows[i] for i in ix],[ss[i] for i in ix],.5) for name,ss in [('original',old_scores),('mixed_HP',new_scores)]}}
 result={'status':'complete','candidate_sha256':sha(a.candidate),'original_sha256':OLD_SHA,'baseline_receipt_sha256':sha(a.original_predictions) if a.original_predictions else None,'best_epoch':state['best_epoch'],'seed':42,'reload_check':{'reference_score':ref['score'],'fresh_process_score':new_scores[0],'absolute_error':delta,'tolerance':5e-5},'results':results,'transition_05':dict(Counter(r['source']+'_'+r['transition_05'] for r in errors)),'common_subset_without_original_train_duplicates':clean,'interpretation':'Validation comparison after early stopping; not an unseen test estimate. HAR labels are weak clip-level Nonfall labels. Not continuous false alarms/hour. Old model HP+ESP32; candidate HP-only mixed-source and class-weighted. B gate bypassed; no deployment or threshold optimization.'}
 dump(a.output/'comparison.json',result)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 f,axes=plt.subplots(1,3,figsize=(13,3.8))
 for ax,(source,label,title) in zip(axes,[('FallDetection',0,'Fall validation'),('FallDetection',1,'Nonfall validation'),('HAR',1,'Ordinary activities (weak Nonfall)')]):
  ix=[i for i,r in enumerate(val.rows) if r['source']==source and r['label']==label]
  for name,ss in [('Original',old_scores),('HP mixed',new_scores)]:ax.hist([ss[i] for i in ix],bins=np.linspace(0,1,21),alpha=.5,label=name)
  ax.axvline(.5,c='black',ls='--');ax.set_title(title);ax.set_xlabel('Fall score');ax.legend()
 f.tight_layout();f.savefig(a.output/'score_distributions.png',dpi=160);plt.close(f)
 lines=['# HP 混合数据 ResNet18 验证比较','',f"新模型最佳轮次：{state['best_epoch']}。重新加载误差：{delta:.3g}。",'', '原模型与新模型均独立推理，未使用 B 门控；没有评价测试集或修改线上模型。','', '## 相同阈值 0.5','', '|模型|Fall检出/总数|漏检|Nonfall误报/总数|Fall Precision|Fall Recall|Fall F1|普通活动触发/总数|','|---|---|---|---|---|---|---|---|']
 for name,data in results['0.5'].items():
  c=data['FallDetection'];h=data['HAR'];lines.append(f"|{name}|{c['TP']}/{c['Fall']}|{c['FN']}|{c['FP']}/{c['Nonfall']}|{c['precision']:.4f}|{c['recall']:.4f}|{c['F1']:.4f}|{h['FP']}/{h['Nonfall']}|")
 lines+=['','## 五类活动（阈值 0.5）','','|活动|原模型触发|新模型触发|样本数|','|---|---|---|---|']
 for label,rr in results['0.5']['original']['HAR_by_activity'].items():lines.append(f"|{label}|{rr['FP']}|{results['0.5']['mixed_HP']['HAR_by_activity'][label]['FP']}|{rr['n']}|")
 lines+=['','![评分分布](score_distributions.png)','','## 结论边界','','- 新训练集没有纳入官方验证 ID、与保留验证集内容相同的训练记录或先前 25 条诊断样本的相同内容。','- 验证集用于早停选模型；这些指标不能当作独立测试表现。','- HAR 的普通活动标签是分段级弱 Nonfall 标注，未提供逐帧跌倒缺席标注；这里不能估计每小时误报。','- 原模型使用 HP+ESP32，新模型使用 HP 混合来源且加类别权重；此比较不能单独证明改善只由新增活动数据造成。','- 两来源的环境与任务有关联；仍需新的独立录制检验泛化。', '- 原训练数据存在一条与验证内容相同的记录；comparison.json 额外提供排除此记录后的共同验证子集指标。','', '完整固定阈值、逐样本结果与哈希见 comparison.json 和 validation_predictions.csv。']
 (a.output/'report_zh.md').write_text('\n'.join(lines))
 body='<pre>'+html.escape('\n'.join(lines))+'</pre><img style="max-width:100%" src="score_distributions.png">'
 (a.output/'report_zh.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>HP 混合数据 ResNet18 验证比较</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;background:#f5f7fb;color:#152438}pre{white-space:pre-wrap;line-height:1.7}</style>'+body)
 print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
