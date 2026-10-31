
from copy import deepcopy
from multi_common import TX as ORIGINAL_TX,RX
from home_v5_physics import build as original_build,channel,SPEC as ORIGINAL_SPEC
from tx_simulation import positions,check as validate
TX=deepcopy(ORIGINAL_TX);TX[0]=[1.1,1.2,.45]
SPEC=dict(deepcopy(ORIGINAL_SPEC),layout_id='t1_v1',tx_m=TX,rx_m=RX)
def check(objects):return validate(objects,TX)
def build():
 scene,objects,solver=original_build()
 for i,p in enumerate(TX):scene.transmitters[f'tx{i}'].position=p
 return scene,objects,solver
