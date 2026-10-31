"""Independent, frozen-model HAR activity probe. Never trains or edits production outputs.
Run select, download, then evaluate as separate stages. Selection precedes inference.
"""
import argparse,ast,csv,hashlib,json,sys,time,subprocess
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
MODULE=ROOT/'backend/modules/fall_detection'
B_MODULE=ROOT/'backend/modules/human_activity_recognition'
REPORT=MODULE/'reports/activity_probe_v1'
DATA=B_MODULE/'local_data/activity_probe_v1'
LABELS=['jumping','running','seated-breathing','walking','wavinghand']
META=Path('/tmp/har_official_metadata_20261030.download')
SPLIT=Path('/tmp/har_official_val_id.json')
MAP=Path('/tmp/har_official_label_mapping.json')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,value):
 REPORT.mkdir(parents=True,exist_ok=True);(REPORT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False))
def read(name):return json.loads((REPORT/name).read_text())
def protected():
 paths=[MODULE/'resnet_b_integration/frozen_model.json',B_MODULE/'integration_v1/frozen_model.json',MODULE/'final_selection_v1/runtime.py',B_MODULE/'integration_v1/runtime.py',MODULE/'outputs/replay.json',B_MODULE/'outputs/replay.json',MODULE/'reports/sample_manifest.csv']
 for cfgp,base in [(paths[0],MODULE),(paths[1],B_MODULE)]:paths.append(base/json.loads(cfgp.read_text())['checkpoint'])
 return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def select():
 if (REPORT/'selection.json').exists():raise FileExistsError('Preserve original pre-inference selection')
 rows=list(csv.DictReader(META.open()));ids=set(json.loads(SPLIT.read_text()));mapping=json.loads(MAP.read_text())
 assert list(mapping['label_to_idx'])==LABELS
 chosen=[]
 for label in LABELS:
  candidates=[r for r in rows if r['id'] in ids and r['label']==label and r['device']=='HP' and r['num_sub']=='232' and r['c_freq']=='5G']
  groups=defaultdict(list)
  for r in sorted(candidates,key=lambda r:r['id']):groups[(r['environment'],r['user'])].append(r)
  # Five distinct environment/user groups, deterministic order; no score-based selection.
  for key in sorted(groups)[:5]:
   r=groups[key][0].copy();tail=PurePosixPath(r['file_path']).parts
   assert 'sub_Human_h5' in tail
   remote='/'.join(('Multitask','sub_Human_h5',*tail[tail.index('sub_Human_h5')+1:]))
   r.update(remote_path=remote,source_url='https://www.kaggle.com/api/v1/datasets/download/guozhenjennzhu/csi-bench/'+quote(remote,safe='')+'?datasetVersionNumber=12',official_split='HumanActivityRecognition/val_id',local_path=str((DATA/r['id']/PurePosixPath(remote).name).resolve()))
   chosen.append(r)
  assert sum(r['label']==label for r in chosen)==5
 assert len({r['id'] for r in chosen})==25
 save('selection.json',{'dataset':'guozhenjennzhu/csi-bench','version':12,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'rule':'Five deterministic distinct environment/user groups per official HAR class, HP232/5G/val_id; no inference before selection. No replacement on download failure.','selected':chosen,'metadata_sha256':sha(META),'split_sha256':sha(SPLIT),'label_mapping_sha256':sha(MAP),'interpretation':'HAR activity labels are not framewise absence-of-fall annotations. C Fall outputs are triggers/suspected false positives, not official FallDetection test metrics. No fall-positive cohort or recall estimate.'})
 save('preserved_before.json',protected())
 for source,name in [(META,'official_metadata.csv'),(SPLIT,'official_val_ids.json'),(MAP,'official_label_mapping.json')]: (REPORT/name).write_bytes(source.read_bytes())
 print('Selected25 before inference',flush=True)

def download():
 import requests,h5py,numpy as np
 def one(r):
  path=Path(r['local_path']);path.parent.mkdir(parents=True,exist_ok=True)
  if not path.exists():
   # Reuse the previously downloaded official running case only if the exact sample matches.
   reuse=B_MODULE/'local_data/running_demo_v1/session_187013__freq232.h5'
   if r['id']=='Human_U01_running_E01_HP_232_187013' and reuse.exists():path.write_bytes(reuse.read_bytes())
   else:
    for attempt in range(2):
     try:
      tmp=path.with_suffix('.partial')
      request=subprocess.run(['curl','-sS','-L','--connect-timeout','15','--max-time','100','--fail',r['source_url'],'-o',str(tmp)],capture_output=True,text=True)
      if request.returncode:raise RuntimeError(request.stderr)
      if not tmp.read_bytes().startswith(b'\x89HDF\r\n\x1a\n'):raise ValueError('Official endpoint did not return HDF5')
      tmp.replace(path);break
     except Exception:
      if attempt==1:raise
      time.sleep(1)
  with h5py.File(path,'r') as h:
   a=h['CSI_amps'][()];assert a.shape==(232,500,1) and np.isfinite(a).all()
  return {'sample_id':r['id'],'ok':True,'file_sha256':sha(path),'bytes':path.stat().st_size,'keys':['CSI_amps'],'shape':list(a.shape),'dtype':str(a.dtype),'finite_ratio':float(np.isfinite(a).mean())}
 results=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  fut={pool.submit(one,r):r for r in read('selection.json')['selected']}
  for f in as_completed(fut):
   r=fut[f]
   try:v=f.result()
   except Exception as e:v={'sample_id':r['id'],'ok':False,'error':repr(e)}
   results.append(v);save('download_status.json',results);print(f"{len(results)}/25 {r['id']} {'OK' if v['ok'] else 'FAILED'}",flush=True)
 if not all(r['ok'] for r in results):raise RuntimeError('Download failures are reported; no samples silently replaced')

def upstream_reader():
 # Execute the pinned official __getitem__ function unchanged. The test facade
 # supplies only metadata lookup/path/label mapping, avoiding unrelated imports.
 import numpy as np,torch,h5py
 p=MODULE/'official_reproduction/upstream/load/supervised/benchmark_dataset.py'
 tree=ast.parse(p.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BenchmarkCSIDataset')
 fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__getitem__')
 ns={'np':np,'torch':torch,'h5py':h5py};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
 return ns['__getitem__'],p

def evaluate():
 import numpy as np,torch,h5py
 from backend.modules.fall_detection.final_selection_v1.runtime import load_candidate,transform_file,read_amplitude
 from backend.modules.human_activity_recognition.integration_v1.runtime import ReleasedActivityModel
 from backend.modules.human_activity_recognition.preprocessing.hdf5_reader import read_csi_array
 from backend.modules.human_activity_recognition.experiments.spectral import extract_spectral_features
 from backend.modules.fall_detection.activity_reference import segment_identity,activity_reference
 torch.set_num_threads(2)
 cfg=json.loads((MODULE/'resnet_b_integration/frozen_model.json').read_text())
 assert sha(MODULE/cfg['checkpoint'])==cfg['checkpoint_sha256']
 model,state=load_candidate(MODULE/cfg['checkpoint'],'official_resnet18');b=ReleasedActivityModel()
 upstream,source=upstream_reader();selected=read('selection.json')['selected'];downloaded={r['sample_id']:r for r in read('download_status.json')}
 records=[];tensor_hashes=defaultdict(list)
 class ILoc:
  def __init__(self,row):self.row=row
  def __getitem__(self,i):return self.row
 for row in selected:
  if not downloaded.get(row['id'],{}).get('ok'):
   records.append({'sample_id':row['id'],'status':'not_run_download_failed'});continue
  try:
   path=Path(row['local_path'])
   with h5py.File(path,'r') as h:raw=h['CSI_amps'][()]
   c_raw=read_amplitude(path);b_raw=read_csi_array(path,('CSI_amps',))[:,:,0]
   c_id=segment_identity(c_raw);b_id=segment_identity(b_raw);assert c_id==b_id
   x=transform_file(path,'HP',state,'official_resnet18')
   facade=SimpleNamespace(split_metadata=SimpleNamespace(iloc=ILoc({'file_path':str(path),'label':row['label']})),data_column='file_path',label_column='label',_resolve_filepath=lambda p:(p,[p]),file_format='h5',data_key='CSI_amps',transform=None,target_transform=None,label_mapper=SimpleNamespace(transform=lambda label:LABELS.index(label)))
   official=upstream(facade,0);assert official is not None
   parity=float((x-official[0]).abs().max());assert parity==0
   assert tuple(x.shape)==(1,500,232) and torch.isfinite(x).all()
   with torch.inference_mode():logits=model(x.unsqueeze(0));score=float(logits.softmax(1)[0,0])
   bp=b.predict(b_raw);features=extract_spectral_features([b_raw]);assert features.shape==(1,4195)
   ref=activity_reference({'input_id':c_id,'device':'HP'},{'mock':False,'frames':[{**bp,'input_id':b_id,'device':'HP'}]})
   tensor_hash=hashlib.sha256(x.numpy().astype('<f4').tobytes()).hexdigest();tensor_hashes[tensor_hash].append(row['id'])
   record={'sample_id':row['id'],'status':'ok','source':row,'raw_file_sha256':sha(path),'h5_key':'CSI_amps','raw_shape':list(raw.shape),'raw_dtype':str(raw.dtype),'finite_ratio':float(np.isfinite(raw).mean()),'raw_min':float(raw.min()),'raw_max':float(raw.max()),'raw_mean':float(raw.mean()),'raw_std':float(raw.std()),'time_feature_mean_std':float(raw[:,:,0].std(axis=1).mean()),'time_feature_constant_count':int((raw[:,:,0].std(axis=1)<1e-8).sum()),'raw_identity_B':b_id,'raw_identity_C':c_id,'aligned':ref['status']=='aligned','C_input_shape':list(x.unsqueeze(0).shape),'C_normalized_mean':float(x.mean()),'C_normalized_std':float(x.std()),'C_normalized_sha256':tensor_hash,'C_upstream_getitem_max_abs_difference':parity,'C_padding_count':0,'C_output_logits':logits[0].tolist(),'C_fall_score':score,'C_threshold':cfg['threshold'],'C_prediction':'Fall' if score>=cfg['threshold'] else 'Nonfall','C_triggers_on_HAR_activity':score>=cfg['threshold'],'B_feature_shape':list(features.shape),'B_feature_sha256':hashlib.sha256(features.astype('<f4').tobytes()).hexdigest(),'B_prediction':bp,'B_matches_official_activity':bp['activity']==row['label'],'activity_reference':ref}
   records.append(record);print(row['label'],row['environment'],bp['activity'],f"B={bp['confidence']:.3f} C={score:.6f}",flush=True)
  except Exception as e:
   records.append({'sample_id':row['id'],'status':'failed','error':repr(e),'source':row});print('FAIL',row['id'],repr(e),flush=True)
  save('records.json',records)
 groups={}
 for label in LABELS:
  rr=[r for r in records if r['status']=='ok' and r['source']['label']==label]
  scores=[r['C_fall_score'] for r in rr]
  groups[label]={'n':len(rr),'C_Fall_triggers':sum(r['C_triggers_on_HAR_activity'] for r in rr),'C_Nonfall':sum(not r['C_triggers_on_HAR_activity'] for r in rr),'B_top1_matches_activity':sum(r['B_matches_official_activity'] for r in rr),'B_confidence_ge_0_6':sum(r['B_prediction']['stable'] for r in rr),'C_score_min':min(scores) if scores else None,'C_score_max':max(scores) if scores else None,'B_activity_counts':dict(Counter(r['B_prediction']['activity'] for r in rr)),'users':sorted({r['source']['user'] for r in rr}),'environments':sorted({r['source']['environment'] for r in rr})}
 with (MODULE/'reports/sample_manifest.csv').open() as f:train=[r for r in csv.DictReader(f) if r['split']=='train_id']
 with (MODULE/'reports/sample_manifest.csv').open() as f:hpval=[r for r in csv.DictReader(f) if r['split']=='val_id' and r['device']=='HP']
 failure=[r for r in records if r['status']!='ok']
 assert protected()==read('preserved_before.json'),'Production weights/config/runtime/replays/manifest changed during probe'
 integrity=json.loads(subprocess.check_output([sys.executable,"-c","import json; from backend.modules.fall_detection.continuous_replay_v1.replay import protect; print(json.dumps(protect()))"],text=True,cwd=ROOT))
 summary={'n_selected':len(selected),'n_run':sum(r['status']=='ok' for r in records),'failures':failure,'groups':groups,'all_same_raw_B_C_identity':all(r['raw_identity_B']==r['raw_identity_C'] for r in records if r['status']=='ok'),'all_C_upstream_input_parity_exact':all(r['C_upstream_getitem_max_abs_difference']==0 for r in records if r['status']=='ok'),'duplicate_raw_inputs':{k:v for k,v in dict(_raw_duplicates(records)).items() if len(v)>1},'duplicate_normalized_C_inputs':{k:v for k,v in tensor_hashes.items() if len(v)>1},'C_checkpoint_sha256':cfg['checkpoint_sha256'],'B_checkpoint_sha256':b.config['checkpoint_sha256'],'upstream_getitem_source_sha256':sha(source),'upstream_parity_method':'Pinned unmodified __getitem__ function executed with test metadata/path/label facade. Constructor and metadata search are not under test.','C_training_environments_all_devices':sorted({r['environment'] for r in train}),'C_training_HP_environments':sorted({r['environment'] for r in train if r['device']=='HP'}),'HAR_probe_environments':sorted({r['environment'] for r in selected}),'C_HP_val_count':len(hpval),'old_artifacts':integrity,'production_artifacts_unchanged':True,'interpretation':'C Fall counts are triggers on activity-labeled HAR examples; suspected false positives only. No framewise fall annotations, no Fall-positive cohort, no fall recall/specificity claim or official C test evaluation. B top1 agreement is exploratory on25 nonrandom diagnostic examples, not a new benchmark estimate.','coverage_limits':['HP only; no ESP32/other devices.','Five users/environments chosen deterministically per class, not random representative sampling.','Single-person segmented clips; no continuous recordings/event timestamps.','No positive falls or multi-person/unsupported activities.','HAR official val samples may overlap the B-author validation subset; not a fresh held-out benchmark.','Cross-task source/raw-overlap relationships with C training require further audit before any future fine-tuning.'],'no_training_no_threshold_change_no_deployment':True}
 save('summary.json',summary)
 flat=[]
 for r in records:
  if r['status']=='ok':flat.append({'sample_id':r['sample_id'],'label':r['source']['label'],'split':r['source']['official_split'],'user':r['source']['user'],'environment':r['source']['environment'],'device':r['source']['device'],'shape':str(r['raw_shape']),'finite_ratio':r['finite_ratio'],'input_id':r['raw_identity_C'],'C_score':r['C_fall_score'],'C_prediction':r['C_prediction'],'B_activity':r['B_prediction']['activity'],'B_confidence':r['B_prediction']['confidence'],'B_activity_agreement':r['B_matches_official_activity']})
 if flat:
  with (REPORT/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 print(json.dumps(summary,indent=2),flush=True)
 if failure:raise RuntimeError('Some selected samples failed; all failures retained in report')

def _raw_duplicates(records):
 d=defaultdict(list)
 for r in records:
  if r['status']=='ok':d[r['raw_identity_C']].append(r['sample_id'])
 return d

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['select','download','evaluate']);args=ap.parse_args()
 {'select':select,'download':download,'evaluate':evaluate}[args.stage]()
