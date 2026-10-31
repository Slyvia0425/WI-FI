
import argparse,json,hashlib,time,os
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from multi_common import DATA,validate,target
from opt_v2_common import DATA2
from night_data import D3
from night_common import atomic,stage_lock,log,now

def main(files=False):
    with stage_lock('data_audit'):
        started=time.time();rows=json.loads((D3/'manifest.json').read_text());groups=defaultdict(list);descriptions=set();counts=Counter();available=0
        for r in rows:
            checked=validate(r['objects']);assert r['count']==len(checked);assert r['target']==target(checked).tolist();assert [p['zone'] for p in checked]==[p['zone'] for p in r['objects']]
            description=json.dumps(sorted(checked,key=lambda p:p['kind']),sort_keys=True);assert description not in descriptions;descriptions.add(description);groups[r['group_id']].append(r);counts[r['split']]+=1
            available+=(D3/'records'/(r['id']+'.npz')).exists()
        for name,rr in groups.items():
            assert len({r['split'] for r in rr})==1
            if len(rr)>1:
                assert len(rr)==3 and all(r['split']=='train' for r in rr)
                base=next(r for r in rr if r['mode'].endswith('_base'));base_objects={p['kind']:p for p in base['objects']};assert base['count']==2
                for r in rr:
                    if r is base:continue
                    assert r['count']==3 and all(next(p for p in r['objects'] if p['kind']==k)==o for k,o in base_objects.items())
        result=dict(metadata_passed=True,split_counts=dict(counts),physical_scenes=len(rows),group_count=len(groups),paired_background_groups=sum(len(rr)>1 for rr in groups.values()),existing_files=available,all_files_generated=available==len(rows),final_test_scored=False)
        if files:
            if available!=len(rows):raise RuntimeError('Generation not complete')
            seen=defaultdict(set);label_by_hash=defaultdict(set);examples=defaultdict(list)
            for folder,rr,prefix in [(DATA,[r for r in json.loads((DATA/'manifest.json').read_text()) if r['split']=='train'],'old_train'),(DATA2,[r for r in json.loads((DATA2/'manifest.json').read_text()) if r['split']=='train'],'v2_train'),(D3,rows,'v3')]:
                for r in rr:
                    with np.load(folder/'records'/(r['id']+'.npz'),allow_pickle=False) as a:h=a['h']
                    assert h.shape==(32,64) and np.isfinite(h).all() and np.linalg.norm(h)>0
                    digest=hashlib.sha256(h.tobytes()).hexdigest();split=prefix if prefix!='v3' else r['split'];seen[split].add(digest);label=tuple(sorted(p['zone'] for p in r['objects']));label_by_hash[digest].add(label)
                    if len(examples[digest])<4:examples[digest].append(r['id'])
            training=seen['old_train']|seen['v2_train']|seen['train'];overlap={s:len(hs&training) for s,hs in seen.items() if s not in ['old_train','v2_train','train']};assert not any(overlap.values()),overlap
            ambiguous=[dict(ids=examples[h],zone_labels=[list(y) for y in ys]) for h,ys in label_by_hash.items() if len(ys)>1]
            result.update(files_passed=True,train_eval_raw_channel_overlap=overlap,ambiguous_exact_channels=len(ambiguous),ambiguous_examples=ambiguous[:20],unique_channels_by_split={s:len(v) for s,v in seen.items()})
        result['elapsed_s']=time.time()-started;atomic('data_audit_full.json' if files else 'data_audit_metadata.json',result);log('Night data '+('full signal' if files else 'metadata')+' audit passed: '+str(len(rows))+' scenes; group splits intact.')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--files',action='store_true');args=ap.parse_args();main(args.files)
