
import argparse,hashlib,time
from collections import defaultdict,Counter
import numpy as np
from layout_t1_v1_physics import check
from layout_t1_v1_common import D5,O5,read,atomic,now,labels,zone_counts,STATE_COUNTS,state_ids,STATE_MASK,STATE_COUNT,decode
from layout_t1_v1_cache import jobs_for

def main(files):
    rows=read(D5/'manifest.json');ids=set();groups=defaultdict(set)
    for r in rows:
        assert r['id'] not in ids;ids.add(r['id']);groups[r['group_id']].add(r['split']);ps=check(r['objects']);assert len(ps)==r['count'];assert len({p['zone'] for p in ps})==r['region_count'];assert r['shared']==(r['region_count']<r['count'])
    assert all(len(s)==1 for s in groups.values());zc=zone_counts(rows);assert np.array_equal(zc.sum(1),[r['count'] for r in rows]);assert np.array_equal(STATE_COUNTS[state_ids(zc)],zc);assert len(STATE_COUNTS)==219

    wanted=np.zeros(9,int);wanted[4]=2;ix=state_ids(wanted[None])[0];p=np.full((1,9),.01);p[0,4]=.99;lp=np.full((1,219),1e-8);lp[0,ix]=.99
    pr,pc,pp=decode(p,np.array([[.99,.005,.005]]),lp,np.array([[.005,.99,.005]]),dict(odds_bias=0,region_weight=1,layout_weight=1,object_weight=1))
    assert pc[0]==2 and pr[0].sum()==1 and np.array_equal(STATE_COUNTS[pp.argmax(1)[0]],wanted)
    result=dict(metadata_passed=True,at=now(),new_scenes=len(rows),splits=dict(Counter(r['split'] for r in rows)),shared_by_split={s:sum(r['shared'] and r['split']==s for r in rows) for s in sorted({r['split'] for r in rows})},joint_states=219,decoder_same_zone_two_count_passed=True,all_files_generated=all((D5/'records'/(r['id']+'.npz')).exists() for r in rows))
    if files:
        assert result['all_files_generated'];seen=defaultdict(set);channel_labels=defaultdict(set)
        for folder,r in jobs_for('train')+[(D5,r) for r in rows if r['split']!='train']:
            with np.load(folder/'records'/(r['id']+'.npz'),allow_pickle=False) as a:h=a['h']
            assert h.shape==(32,64) and np.isfinite(h).all() and np.linalg.norm(h)>0
            digest=hashlib.sha256(h.tobytes()).hexdigest();seen[r['split']].add(digest);c=tuple(zone_counts([r])[0]);channel_labels[digest].add(c)
        overlap={s:len(v&seen['train']) for s,v in seen.items() if s!='train'};conflicting=sum(len(v)>1 for v in channel_labels.values());result.update(files_passed=True,train_eval_exact_channel_overlap=overlap,conflicting_exact_channels=conflicting)

        result['clean_split_audit']=not any(overlap.values()) and conflicting==0
    atomic('data_audit_full.json' if files else 'data_audit_metadata.json',result);print(result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--files',action='store_true');a=p.parse_args();main(a.files)
