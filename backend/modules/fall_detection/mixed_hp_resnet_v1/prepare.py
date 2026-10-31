"""Build a predeclared HP mixed-data manifest without using validation for training.
HAR ordinary-activity labels are explicitly treated as clip-level weak Nonfall labels.
"""
import argparse,csv,hashlib,json,os,shutil,subprocess,time
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path,PurePosixPath
from urllib.parse import quote
import h5py,numpy as np
HERE=Path(__file__).resolve().parent
M=HERE.parent
REPORT=M/'reports/mixed_hp_resnet_v1'
LABELS=['jumping','running','seated-breathing','walking','wavinghand']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def content_hash(a):
 a=np.asarray(a,dtype='<f8').copy(order='C');a[a==0]=0
 return hashlib.sha256(json.dumps(list(a.shape)).encode()+a.dtype.str.encode()+a.tobytes()).hexdigest()
def url(remote):return 'https://www.kaggle.com/api/v1/datasets/download/guozhenjennzhu/csi-bench/'+quote(remote,safe='')+'?datasetVersionNumber=12'
def select():
 p=REPORT/'selection.json'
 if p.exists():raise FileExistsError('Preserve predeclared selection')
 rows=list(csv.DictReader((M/'reports/sample_manifest.csv').open()));chosen=[]
 for split in ('train_id','val_id'):
  for r in rows:
   if r['device']!='HP' or r['split']!=split:continue
   remote='FallDetection/'+Path(r['file_path']).relative_to('/Users/chenxiaohe/Documents/CSI-Bench-v12/FallDetection').as_posix()
   chosen.append({'id':'FallDetection::'+r['sample_id'],'sample_id':r['sample_id'],'source':'FallDetection','role':'train' if split=='train_id' else 'val','official_split':split,'label':0 if r['original_label']=='Fall' else 1,'original_label':r['original_label'],'activity':r['original_label'],'user':r['user'],'environment':r['environment'],'session':r['session'],'device':'HP','local_path':r['file_path'],'remote_path':remote,'source_url':url(remote),'expected_content_hash':r['content_hash']})
 meta=list(csv.DictReader(open('/tmp/har_official_metadata_20261030.download')))
 splits={s:set(json.load(open('/tmp/har_official_'+s+'_id.json'))) for s in ('train','val')}
 assert not splits['train']&splits['val']
 for split,n in [('train',200),('val',50)]:
  for label in LABELS:
   groups=defaultdict(list)
   for r in meta:
    if r['id'] in splits[split] and r['device']=='HP' and r['num_sub']=='232' and r['label']==label:
     groups[(r['user'],r['environment'],r['activity'])].append(r)
   for key in groups:groups[key].sort(key=lambda r:hashlib.sha256(('42:'+r['id']).encode()).hexdigest())
   selected=[];depth=0
   while len(selected)<n:
    before=len(selected)
    for key in sorted(groups):
     if depth<len(groups[key]) and len(selected)<n:selected.append(groups[key][depth])
    assert len(selected)>before,'Insufficient class candidates'
    depth+=1
   for r in selected:
    parts=PurePosixPath(r['file_path']).parts;remote='/'.join(('Multitask','sub_Human_h5',*parts[parts.index('sub_Human_h5')+1:]))
    chosen.append({'id':'HAR::'+r['id'],'sample_id':r['id'],'source':'HAR','role':split,'official_split':split+'_id','label':1,'original_label':label,'activity':r['activity'],'user':r['user'],'environment':r['environment'],'session':r['session'],'device':'HP','local_path':str((M/'local_data/mixed_hp_resnet_v1/HAR'/r['id']/PurePosixPath(remote).name).resolve()),'remote_path':remote,'source_url':url(remote),'expected_content_hash':None})
 train={r['id'] for r in chosen if r['role']=='train'};val={r['id'] for r in chosen if r['role']=='val'};assert not train&val
 assert not {r['sample_id'] for r in chosen if r['source']=='HAR' and r['role']=='train'} & splits['val']
 prior=json.loads((M/'reports/activity_probe_v1/diagnosis.json').read_text())
 protected_prior_hashes=[x['audit_canonical_hash'] for x in prior['HAR_vs_C_raw_content_matches']]
 protocol={'seed':42,'selection_rule':'HP official source train/val IDs; HAR200/class train,50/class val; round-robin user/environment/activity groups with deterministic SHA256 seed42 order. Selected before training.','HAR_label_assumption':'Five ordinary activity labels are weak clip-level Nonfall labels; no framewise fall-absence claim.','dedup_rule':'Keep validation records fixed. Exclude any train content matching current validation or prior25 HAR diagnostic clips; deduplicate train content, stop on conflicting binary labels. No randomized resplit.','validation_selection_rule':'Best weighted combined-source validation CE; patience10,max100. No test records used.','training':'New randomly initialized official ResNet18 architecture; no old C weights or ImageNet weights.','comparison':'B/gate bypassed. Both models on exactly same HP FallDetection val and selected HAR val. Fixed thresholds0.5 and original0.574872225522995; no threshold optimization.','scope_caveat':'Old model used HP+ESP32; new model is HP-only mixed-source. This is not a controlled data-only ablation. Tasks/environments remain confounded.','known_prior25_validation_hashes':protected_prior_hashes,'records':chosen,'counts':dict(Counter((r['source']+'_'+r['role']) for r in chosen)),'metadata_sources_sha256':{n:sha(p) for n,p in [('HAR_metadata','/tmp/har_official_metadata_20261030.download'),('HAR_train_ids','/tmp/har_official_train_id.json'),('HAR_val_ids','/tmp/har_official_val_id.json'),('Fall_manifest',M/'reports/sample_manifest.csv')]}}
 dump(p,protocol)
 for src,name in [('/tmp/har_official_metadata_20261030.download','HAR_metadata.csv'),('/tmp/har_official_train_id.json','HAR_train_ids.json'),('/tmp/har_official_val_id.json','HAR_val_ids.json'),('/tmp/har_official_label_mapping.json','HAR_label_mapping.json')]:shutil.copy2(src,REPORT/name)
 print(protocol['counts'],flush=True)

def fetch(records,workers):
 def one(r):
  p=(Path(os.environ['MIXED_HP_DATA_ROOT'])/r['remote_path']) if os.environ.get('MIXED_HP_DATA_ROOT') else Path(r['local_path'])
  if not p.exists():
   p.parent.mkdir(parents=True,exist_ok=True)
   for attempt in range(3):
    t=p.with_suffix('.partial');cmd=['curl','-sS','-L','--connect-timeout','15','--max-time','120','--fail',r['source_url'],'-o',str(t)]
    z=subprocess.run(cmd,capture_output=True,text=True)
    if z.returncode==0 and t.exists() and t.read_bytes().startswith(b'\x89HDF\r\n\x1a\n'):t.replace(p);break
    if attempt==2:raise RuntimeError(z.stderr or 'Not an HDF5 response')
    time.sleep(1)
  with h5py.File(p,'r') as h:a=h['CSI_amps'][()]
  if a.shape!=(232,500,1) or not np.isfinite(a).all():raise ValueError('Invalid HP input')
  digest=content_hash(a)
  if r['expected_content_hash'] and r['expected_content_hash']!=digest:raise ValueError('C data changed after original audit')
  return {**r,'local_path':str(p),'shape':list(a.shape),'dtype':str(a.dtype),'finite_ratio':1.,'file_sha256':sha(p),'content_hash':digest}
 out=[];fail=[]
 with ThreadPoolExecutor(max_workers=workers) as pool:
  futures={pool.submit(one,r):r for r in records}
  for f in as_completed(futures):
   r=futures[f]
   try:out.append(f.result())
   except Exception as exc:fail.append({'id':r['id'],'error':repr(exc)})
   if (len(out)+len(fail))%100==0:
    dump(REPORT/'download_progress.json',{'ok':len(out),'failed':len(fail),'total':len(records),'failures':fail});print('Audited',len(out),'failed',len(fail),flush=True)
 dump(REPORT/'raw_audit.json',{'records':out,'failures':fail})
 if fail:raise RuntimeError('Explicit failures retained; no silent replacement')
 return out

def audit_cache(workers):
 protocol=json.loads((REPORT/'selection.json').read_text());raw=fetch(protocol['records'],workers)
 order={r['id']:i for i,r in enumerate(protocol['records'])};raw.sort(key=lambda r:order[r['id']])
 val=[r for r in raw if r['role']=='val'];held_hashes={r['content_hash'] for r in val}|set(protocol['known_prior25_validation_hashes'])
 train0=[r for r in raw if r['role']=='train'];train=[];removed=[];seen={}
 for r in train0:
  h=r['content_hash']
  if h in held_hashes:removed.append({'id':r['id'],'reason':'Exact content matches held validation/previous diagnostic'});continue
  if h in seen:
   if r['label']!=seen[h]['label']:raise ValueError('Conflicting train labels for identical content')
   removed.append({'id':r['id'],'reason':'Exact duplicate within train','kept':seen[h]['id']});continue
  train.append(r);seen[h]=r
 assert not {r['id'] for r in train}&{r['id'] for r in val}
 assert not {r['content_hash'] for r in train}&{r['content_hash'] for r in val}
 groups=lambda rr:{(r['source'],r['user'],r['environment'],r['session']) for r in rr}
 exact_session_overlap=sorted(groups(train)&groups(val))
 if exact_session_overlap:raise ValueError('Exact source/user/environment/session overlap across train/val: '+str(exact_session_overlap[:5]))
 cache=M/'local_data/mixed_hp_resnet_v1/cache';cache.mkdir(parents=True,exist_ok=True)
 import torch
 for role,rr in [('train',train),('val',val)]:
  target=cache/(role+'.npy');x=np.lib.format.open_memmap(target,mode='w+',dtype='float32',shape=(len(rr),1,500,232))
  for i,r in enumerate(rr):
   with h5py.File(r['local_path'],'r') as h:a=h['CSI_amps'][()]
   t=torch.from_numpy(a).float().permute(2,1,0);t=(t-t.mean(dim=(1,2),keepdim=True))/t.std(dim=(1,2),keepdim=True).clamp_min(1e-8)
   assert t.shape==(1,500,232) and torch.isfinite(t).all();x[i]=t.numpy()
  x.flush();del x
  dump(cache/(role+'_records.json'),rr)
 config={'train_n':len(train),'val_n':len(val),'train_class_counts':dict(Counter(str(r['label']) for r in train)),'val_source_class_counts':dict(Counter(r['source']+'_'+str(r['label']) for r in val)),'label_mapping':{'Fall':0,'Nonfall':1},'input_shape':[1,500,232],'normalization':'Pinned official per-record scalar mean/std correction1; no training-fit stats, padding, filtering or augmentation','removed_training_records':removed,'exact_session_overlap':exact_session_overlap,'train_content_hashes_sha256':hashlib.sha256(json.dumps(sorted(r['content_hash'] for r in train)).encode()).hexdigest(),'val_content_hashes_sha256':hashlib.sha256(json.dumps(sorted(r['content_hash'] for r in val)).encode()).hexdigest(),'cache_dir':str(cache.resolve()),'cache_sha256':{role:sha(cache/(role+'.npy')) for role in ('train','val')},'selection_sha256':sha(REPORT/'selection.json'),'validation_kept_fixed':True,'no_test_access':True,'weak_HAR_label_assumption':protocol['HAR_label_assumption']}
 dump(REPORT/'data_audit.json',config);print(json.dumps(config,indent=2),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['select','prepare']);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args()
 select() if a.stage=='select' else audit_cache(a.workers)
