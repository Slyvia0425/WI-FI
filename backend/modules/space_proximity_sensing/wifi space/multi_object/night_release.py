
import argparse,json,hashlib
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from multi_common import OUT,features,target
from multi_infer import predict_probabilities
from multi_train import metrics as typed_metrics,seed
from opt_v2_common import window
from night_common import NIGHT,atomic,now,log
from night_cache import jobs_for

def prepare():
    report=json.loads((NIGHT/'results.json').read_text(encoding='utf-8'));fr=json.loads((NIGHT/'candidate_frozen.json').read_text(encoding='utf-8'));path=NIGHT/'model.joblib'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==fr['model_sha256']
    audit=json.loads((NIGHT/'data_audit_full.json').read_text(encoding='utf-8'));assert audit['files_passed'] and not any(audit['train_eval_raw_channel_overlap'].values())
    aux=joblib.load(OUT/'model.joblib');X=[];Y=[]
    for folder,row in jobs_for('test'):
        with np.load(folder/'records'/(row['id']+'.npz')) as a:h=window(a['h'],np.random.default_rng(seed('night-view-v1:'+row['group_id']+':0')))
        X.append(features(h.mean(0),aux['reference']));Y.append(target(row['objects']))
    y=np.array(Y)
    with threadpool_limits(limits=4):p=predict_probabilities(aux,np.array(X)).argmax(2)
    ar=json.loads((OUT/'results.json').read_text(encoding='utf-8'));m=typed_metrics(y,p);m['by_true_count']={str(n):typed_metrics(y[(y>0).sum(1)==n],p[(y>0).sum(1)==n]) for n in [1,2,3]};ar['splits']={'test':m};ar['evaluation_note']='Original experimental size model, not retrained. Re-evaluated on the same NEW v3 test1500 windows as primary. Its scores cannot be used as occupancy/shape reconstruction evidence.'
    atomic('auxiliary_results.json',ar)
    b=joblib.load(path);selection=b['selection_reports'][b['primary_view']]
    selected=b['experts'][b['primary_view']]['decoder']
    entries=selection['comparison']
    if not any(r['config']==selected for r in entries):entries=[selection['best']]+entries
    report['comparison']=[dict(name=json.dumps(r['config'],sort_keys=True),validation=r['validation'],selected=r['config']==selected) for r in entries];atomic('results.json',report)
    atomic('profile_ready.json',dict(ready=True,prepared_at=now(),model_sha256=fr['model_sha256'],release_eligible=report['release_check']['eligible'],source='new independent validation/calibration/test, baseline evaluated on same signals'))
    log('v3 versioned profile prepared; active pointer unchanged.')

def activate(version):
    if version==3:
        ready=json.loads((NIGHT/'profile_ready.json').read_text(encoding='utf-8'));assert ready['ready'] and ready['release_eligible'],'Candidate did not pass release rule'
        v=json.loads((NIGHT/'verification.json').read_text(encoding='utf-8'));assert v['passed'] and v['model_sha256']==ready['model_sha256'],'Run candidate API verification before activation'
    pointer=OUT/'active_profile.json';old=json.loads(pointer.read_text(encoding='utf-8')) if pointer.exists() else dict(version=2,implicit_flag=True)
    history=OUT/'profile_history.json';changes=json.loads(history.read_text(encoding='utf-8')) if history.exists() else [];changes.append(dict(at=now(),previous=old,version=version))
    tmp=history.with_suffix('.partial');tmp.write_text(json.dumps(changes,indent=2),encoding='utf-8');tmp.replace(history)
    tmp=pointer.with_suffix('.partial');tmp.write_text(json.dumps(dict(version=version,activated_at=now()),indent=2),encoding='utf-8');tmp.replace(pointer)
    log('Active profile set to v'+str(version)+'; v2 weights retained unchanged.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','activate','rollback']);args=ap.parse_args()
    if args.action=='prepare':prepare()
    else:activate(3 if args.action=='activate' else 2)
