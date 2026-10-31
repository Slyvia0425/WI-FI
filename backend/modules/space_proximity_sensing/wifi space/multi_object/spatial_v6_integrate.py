
from pathlib import Path
import json,shutil
ROOT=Path(__file__).resolve().parent;O=ROOT/'outputs/spatial_v6'
def replace(s,a,b):
 assert a in s,a[:100]
 return s.replace(a,b,1)
def main():
 eligible=[lid for lid in ['default','t1_v1'] if (O/lid/'results.json').exists() and json.loads((O/lid/'results.json').read_text())['release_check']['eligible']]
 if not eligible:raise RuntimeError('No eligible V6 model; integration refused')
 for lid in eligible:
  f=O/lid
  from spatial_v6_registry import sha
  (f/'CANDIDATE.json').write_text(json.dumps(dict(model_sha256=sha(f/'model.joblib'),frozen_sha256=sha(f/'frozen.json'),results_sha256=sha(f/'results.json')),indent=2),encoding='utf-8')
 backup=O/'integration_backup';backup.mkdir(exist_ok=True)
 for name in ['app.py','layout_registry.py','web/app.js']:
  target=backup/name;target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():shutil.copy2(ROOT/name,target)
 p=ROOT/'layout_registry.py';s=p.read_text(encoding='utf-8')
 s=replace(s,"def match(tx):\n if np.allclose(tx,TX", "def match(tx):\n from spatial_v6_registry import model as v6_model\n if np.allclose(tx,NEW_TX,rtol=0,atol=1e-7):\n  upgraded=v6_model('t1_v1')\n  if upgraded is not None:return 't1_v1',upgraded\n if np.allclose(tx,TX")
 s=replace(s,"return 'default',None","return 'default',v6_model('default')")
 a=" return [dict(id='default',name=";s=replace(s,a," es=[dict(id='default',name=")
 end="metrics=None if m is None else {k:m[k] for k in ['n','occupied_zone_macro_f1','count_accuracy','exact_joint_accuracy']})]"
 s=replace(s,end,end+"\n from spatial_v6_registry import evaluation as v6_evaluation\n for e in es:\n  upgraded=v6_evaluation(e['id'])\n  if upgraded is not None:\n   e.update(ready=True,revision='spatial_count_v6',evaluation=upgraded,metrics={k:upgraded['splits']['test'][k] for k in ['n','occupied_zone_macro_f1','count_accuracy','exact_joint_accuracy']})\n return es")
 s=replace(s," if not approved():return {}\n p=DATA/current_folder().name/'manifest.json';return new_catalog(str(p),p.stat().st_mtime_ns)"," from spatial_v6_registry import catalog as v6_catalog\n old={}\n if approved():\n  p=DATA/current_folder().name/'manifest.json';old=new_catalog(str(p),p.stat().st_mtime_ns)\n return dict(old,**v6_catalog())")
 s=replace(s,"def by_id(id):\n if id=='default':", "def by_id(id):\n from spatial_v6_registry import model as v6_model\n if id in ['default','t1_v1']:\n  upgraded=v6_model(id)\n  if upgraded is not None:return upgraded,TX if id=='default' else NEW_TX\n if id=='default':")
 p.write_text(s,encoding='utf-8')
 p=ROOT/'app.py';s=p.read_text(encoding='utf-8')
 s=replace(s,"def record_path(row):return DATA/", "def record_path(row):\n    if row['id'].startswith('v6_'):return DATA/'spatial_v6'/row['layout_id']/'records'/(row['raw_id']+'.npz')\n    return DATA/")
 s=replace(s,"    return row['id'].startswith(('h5_','t1_','t3_')) if version==5 else", "    if version==5:\n        from spatial_v6_registry import approved as v6_approved\n        lid=row.get('layout_id','t1_v1' if row['id'].startswith(('t1_','t3_')) else 'default')\n        if v6_approved(lid):return row['id'].startswith('v6_')\n        return row['id'].startswith(('h5_','t1_','t3_'))\n    return")
 s=replace(s,"            if selected_bundle.get('count_refiner'):", "            if selected_bundle.get('ordinal_refiner'):\n                from spatial_v6_infer import infer_v5 as infer_layout\n            elif selected_bundle.get('count_refiner'):")
 s=replace(s,"if e['ready'] and e['id']!='default':", "if e['ready']:")
 s=replace(s,"                                if preset.get('count_refiner'):", "                                if preset is None:continue\n                                if preset.get('ordinal_refiner'):\n                                    from spatial_v6_infer import infer_v5 as warm_preset\n                                elif preset.get('count_refiner'):")
 s=s.replace("layout_id='t1_v1' if r['id'].startswith(('t1_','t3_')) else 'default'", "layout_id=r.get('layout_id','t1_v1' if r['id'].startswith(('t1_','t3_')) else 'default')")
 s=replace(s,"(version!=5 or r['id'].startswith(('t1_','t3_') if lid=='t1_v1' else 'h5_'))", "(version!=5 or r.get('layout_id','t1_v1' if r['id'].startswith(('t1_','t3_')) else 'default')==lid)")
 s=replace(s,"lid='t1_v1' if row['id'].startswith(('t1_','t3_')) else 'default';selected,tx=by_id(lid)", "lid=row.get('layout_id','t1_v1' if row['id'].startswith(('t1_','t3_')) else 'default');selected,tx=by_id(lid)")
 p.write_text(s,encoding='utf-8')
 p=ROOT/'web/app.js';s=p.read_text(encoding='utf-8')
 s=s.replace("e.id==='default'?t('默认布局','Default layout'):e.revision==='t1_count_v3'?", "e.revision==='spatial_count_v6'?t(e.id==='default'?'默认 · 计数分类版':'T1 · 计数分类版',e.id==='default'?'Default · count classifier':'T1 · count classifier'):e.id==='default'?t('默认布局','Default layout'):e.revision==='t1_count_v3'?")
 s=s.replace("d?.inference_model_revision==='t1_count_v3'?", "d?.inference_model_revision==='spatial_count_v6'?' · '+t('计数分类版','Count classifier'):d?.inference_model_revision==='t1_count_v3'?")
 s=s.replace("o.feature_view==='count_refiner'?", "o.feature_view==='ordinal_count_v6'?t('区域件数分类','Ordinal zone counts'):o.feature_view==='count_refiner'?")
 p.write_text(s,encoding='utf-8')
 print('Optional integration applied; only candidate server sees unapproved V6 models.')
if __name__=='__main__':main()
