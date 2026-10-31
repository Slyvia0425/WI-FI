
import numpy as np
from multi_common import TX,RX,SIZES,zone
from home_v5_physics import FIXED

def positions(value):
 a=np.asarray(TX if value is None else value,dtype=float)
 if a.shape!=(4,3) or not np.isfinite(a).all():raise ValueError('Expected four finite TX xyz positions / 需要4个有效信号源坐标')
 if np.any(a[:,:2]<.08) or np.any(a[:,:2]>5.92) or np.any(a[:,2]<.1) or np.any(a[:,2]>2.9):raise ValueError('TX must stay inside room / 信号源须位于房间内')
 if any(np.linalg.norm(a[i]-p)<.08 for i in range(4) for p in list(a[:i])+list(np.asarray(RX))):raise ValueError('Radio devices too close / 设备位置过近')
 return a.tolist()

def validate_targets(objects,tx,allow_shared_zones=True):
    if not isinstance(objects,list) or not 1<=len(objects)<=3:raise ValueError('Use 1–3 objects. / 物体数量须为1–3。')
    kinds=[];zones=[];checked=[]
    for o in objects:
        k=int(o['kind']);x=float(o['x']);y=float(o['y'])
        if k not in range(3) or k in kinds:raise ValueError('At most one Small, one Medium and one Large object. / 每档尺寸目前最多一件。')
        size=np.asarray(o.get('size',SIZES[k]),float)
        if size.shape!=(3,) or not np.isfinite(size).all() or np.any(np.abs(size/SIZES[k]-1)>.081):raise ValueError('Dimensions must be within ±8% of the selected size band. / 尺寸须在所选类别的±8%范围内。')
        if not np.isfinite([x,y]).all() or not(size[0]/2+.01-1e-9<=x<=6-size[0]/2-.01+1e-9 and size[1]/2+.01-1e-9<=y<=6-size[1]/2-.01+1e-9):raise ValueError('Keep every object inside the room. / 物体不能超出房间。')
        d=np.abs(np.array(tx+RX)[:,:2]-[x,y]);z=zone(x,y)
        if np.any(np.all(d<=size[:2]/2+.08,axis=1)):raise ValueError('Object overlaps a radio device clearance. / 物体与设备禁放范围重叠。')
        if z in zones and not allow_shared_zones:raise ValueError('This prototype requires distinct center zones. / 当前版本要求各物体中心位于不同区域。')
        for p in checked:
            if np.all(np.abs(np.array([x,y])-[p['x'],p['y']])<=(size[:2]+np.array(p['size'][:2]))/2+.08):raise ValueError('Objects overlap or are too close. / 物体之间重叠或过近。')
        kinds.append(k);zones.append(z);checked.append(dict(kind=k,x=x,y=y,size=size.tolist(),zone=z))
    return checked


def check(objects,tx):
 ps=validate_targets(objects,tx)
 for o in ps:
  for b in FIXED:
   if all(abs(o[k]-b[k])<=(o['size'][i]+b['size'][i])/2+.08 for i,k in enumerate(['x','y'])):raise ValueError('Object overlaps fixed cabinet / 物体与柜体重叠')
 for p in tx:
  for b in FIXED:
   if all(abs(p[i]-b[k])<=b['size'][i]/2+.08 for i,k in enumerate(['x','y'])):raise ValueError('TX overlaps cabinet / 信号源与柜体重叠')
 return ps
