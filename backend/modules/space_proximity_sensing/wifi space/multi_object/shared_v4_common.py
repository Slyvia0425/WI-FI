
import os,json,hashlib
from pathlib import Path
from contextlib import contextmanager
import numpy as np
from itertools import combinations_with_replacement
from multi_common import ROOT,DATA,OUT
from opt_v2_common import MASKS,layout_ids
from night_common import now,alive
D4=DATA/'shared_v4';O4=OUT/'shared_v4';O4.mkdir(exist_ok=True,parents=True)
STATE_COUNTS=np.array([np.bincount(z,minlength=9) for n in [1,2,3] for z in combinations_with_replacement(range(9),n)],np.int64)
STATE_MASK=(STATE_COUNTS>0).astype(np.float32)
STATE_LAYOUT=layout_ids(STATE_MASK)
STATE_COUNT=STATE_COUNTS.sum(1)
STATE_INDEX={tuple(c):i for i,c in enumerate(STATE_COUNTS)}

def atomic(name,d):
    p=O4/name;p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.partial');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(p)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for a in iter(lambda:f.read(8*1024*1024),b''):h.update(a)
    return h.hexdigest()
@contextmanager
def lock(name):
    p=O4/(name+'.lock')
    if p.exists():
        if alive(read(p)['pid']):raise RuntimeError('Stage already running: '+name)
        p.unlink()
    with p.open('x') as f:json.dump(dict(pid=os.getpid(),started=now()),f)
    try:yield
    finally:p.unlink(missing_ok=True)

def labels(rows):
    y=np.zeros((len(rows),9),np.float32)
    for i,r in enumerate(rows):
        for o in r['objects']:y[i,o['zone']]=1
    return y,np.array([len(r['objects']) for r in rows],np.int64)
def zone_counts(rows):
    a=np.zeros((len(rows),9),np.int64)
    for i,r in enumerate(rows):
        for o in r['objects']:a[i,o['zone']]+=1
    return a
def state_ids(counts_by_zone):return np.array([STATE_INDEX[tuple(c)] for c in counts_by_zone])

def decode(p,region_count,layout,object_count,cfg):
    p=np.clip(p,1e-7,1-1e-7);p=1/(1+np.exp(-(np.log(p/(1-p))+cfg['odds_bias'])))
    score=np.log(p)@STATE_MASK.T+np.log1p(-p)@(1-STATE_MASK).T
    score+=cfg['region_weight']*np.log(np.maximum(region_count[:,STATE_MASK.sum(1).astype(int)-1],1e-12))
    score+=cfg['layout_weight']*np.log(np.maximum(layout,1e-12))
    score+=cfg['object_weight']*np.log(np.maximum(object_count[:,STATE_COUNT-1],1e-12))
    score-=score.max(1,keepdims=True);post=np.exp(score);post/=post.sum(1,keepdims=True);ix=post.argmax(1)
    return STATE_MASK[ix],STATE_COUNT[ix],post
