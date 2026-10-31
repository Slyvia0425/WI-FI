
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
DATA=ROOT/'data';OUT=ROOT/'outputs'
NAMES=['Small','Medium','Large']
SIZES=np.array([[.6,.5,.8],[1.1,.8,1.1],[1.6,1.,1.4]],float)
TX=[[.1,1.,.45],[.1,5.,.65],[5.9,1.,.65],[5.9,5.,.45]]
RX=[[.1,2.6,.45],[.1,4.2,.65],[1.8,5.9,.45],[4.2,5.9,.65],[5.9,3.4,.45],[5.9,1.8,.65],[4.2,.1,.45],[1.8,.1,.65]]
CONFIG=dict(version='multi-object-v1',room_m=[6,6,3],size_names=NAMES,canonical_sizes_m=SIZES.tolist(),size_variation=.08,tx_m=TX,rx_m=RX,links=32,subcarriers=64,carrier_hz=5.24e9,bandwidth_hz=20e6,max_depth=3,samples_per_src=100000,clearance_m=.08,notes='Up to one object per size category, distinct center zones, axis-aligned solid wood cuboids. Size category is learned, canonical drawn dimensions are a prior, not exact measurements.')

def zone(x,y):return min(2,int(y//2))*3+min(2,int(x//2))
def center(z):return [1+2*(z%3),1+2*(z//3)]

def validate(objects,allow_shared_zones=False):
    if not isinstance(objects,list) or not 1<=len(objects)<=3:raise ValueError('Use 1–3 objects. / 物体数量须为1–3。')
    kinds=[];zones=[];checked=[]
    for o in objects:
        k=int(o['kind']);x=float(o['x']);y=float(o['y'])
        if k not in range(3) or k in kinds:raise ValueError('At most one Small, one Medium and one Large object. / 每档尺寸目前最多一件。')
        size=np.asarray(o.get('size',SIZES[k]),float)
        if size.shape!=(3,) or not np.isfinite(size).all() or np.any(np.abs(size/SIZES[k]-1)>.081):raise ValueError('Dimensions must be within ±8% of the selected size band. / 尺寸须在所选类别的±8%范围内。')
        if not np.isfinite([x,y]).all() or not(size[0]/2+.01-1e-9<=x<=6-size[0]/2-.01+1e-9 and size[1]/2+.01-1e-9<=y<=6-size[1]/2-.01+1e-9):raise ValueError('Keep every object inside the room. / 物体不能超出房间。')
        d=np.abs(np.array(TX+RX)[:,:2]-[x,y]);z=zone(x,y)
        if np.any(np.all(d<=size[:2]/2+.08,axis=1)):raise ValueError('Object overlaps a radio device clearance. / 物体与设备禁放范围重叠。')
        if z in zones and not allow_shared_zones:raise ValueError('This prototype requires distinct center zones. / 当前版本要求各物体中心位于不同区域。')
        for p in checked:
            if np.all(np.abs(np.array([x,y])-[p['x'],p['y']])<=(size[:2]+np.array(p['size'][:2]))/2+.08):raise ValueError('Objects overlap or are too close. / 物体之间重叠或过近。')
        kinds.append(k);zones.append(z);checked.append(dict(kind=k,x=x,y=y,size=size.tolist(),zone=z))
    return checked

def noisy(h,rng,snr_db=None):
    if snr_db is None:snr_db=rng.uniform(27,40)
    rms=np.sqrt(np.mean(np.abs(h)**2,axis=1,keepdims=True))
    noise=(rng.normal(size=h.shape)+1j*rng.normal(size=h.shape))*np.maximum(rms,1e-6)*10**(-snr_db/20)/np.sqrt(2)
    return ((h+noise)*10**(rng.normal(0,.25,(32,1))/20)).astype(np.complex64)

def features(h,ref):
    h=np.asarray(h,dtype=np.complex64)
    if h.shape!=(32,64) or not np.isfinite(h).all() or not np.any(np.abs(h)>1e-12):raise ValueError('Expected finite, nonzero complex CSI [32,64]. / 需要有效的32×64复数信号。')
    db=20*np.log10(np.maximum(np.abs(h),1e-10));rd=20*np.log10(np.maximum(np.abs(ref),1e-10));delta=np.clip(db-rd,-60,60)
    parts=[delta.reshape(32,8,8).mean(2).ravel(),db.mean(1),db.std(1),np.abs(np.diff(db,axis=1)).mean(1)]
    for lag in [1,8,24]:
        a=h[:,lag:];b=h[:,:-lag];v=np.sum(a*np.conj(b),axis=1)/(np.sum(np.abs(a)*np.abs(b),axis=1)+1e-18);parts.extend([v.real,v.imag])
    return np.concatenate(parts).astype(np.float32)

def target(objects):
    y=np.zeros(3,dtype=np.int64)
    for o in objects:y[o['kind']]=o['zone']+1
    return y
