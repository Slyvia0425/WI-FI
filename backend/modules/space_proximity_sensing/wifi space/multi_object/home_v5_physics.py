
import sys,tempfile,time,json
from pathlib import Path
import numpy as np
from multi_common import TX,RX,SIZES,validate
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from simulate import write_ply
FIXED=[dict(name='cabinet',x=3.,y=5.72,size=[1.2,.38,.9],material='wood')]
SPEC=dict(room_m=[6,6,3],carrier_hz=5.24e9,bandwidth_hz=20e6,subcarriers=64,max_depth=3,samples_per_src=100000,los=True,specular_reflection=True,diffuse_reflection=True,refraction=True,antenna='dipole',wall_scattering=.12,wood_scattering=.08,glass_scattering=.02,fixed_objects=FIXED,notes='Fixed furnished-room simulation, calibrated background, original distributed4TX/8RX. Slab material approximation, not full-wave measured furniture.')

def check(objects):
 ps=validate(objects,allow_shared_zones=True)
 for p in ps:
  for b in FIXED:
   if all(abs(p[a]-b[a]) <= (p['size'][i]+b['size'][i])/2+.08 for i,a in enumerate(['x','y'])):raise ValueError('Keep objects clear of the fixed cabinet. / 请避开固定柜体。')
 return ps

def build():
 import sionna.rt as rt
 folder=Path(tempfile.gettempdir())/'wifi_home_v5';folder.mkdir(exist_ok=True)
 mats={'concrete':(.12,.12),'wood':(.08,.08),'glass':(.008,.02)}

 boxes=[('floor',[-.15,-.15,-.15],[6.15,6.15,0],'concrete'),('ceiling',[-.15,-.15,3],[6.15,6.15,3.15],'concrete'),('west',[-.15,0,0],[0,6,3],'concrete'),('east',[6,0,0],[6.15,6,3],'concrete'),('south_left',[-.15,-.15,0],[2.5,0,3],'concrete'),('south_right',[3.4,-.15,0],[6.15,0,3],'concrete'),('south_top',[2.5,-.15,2.1],[3.4,0,3],'concrete'),('door',[2.5,-.05,0],[3.4,0,2.1],'wood'),('north_left',[-.15,6,0],[2.1,6.15,3],'concrete'),('north_right',[3.9,6,0],[6.15,6.15,3],'concrete'),('north_below',[2.1,6,0],[3.9,6.15,1.15],'concrete'),('north_above',[2.1,6,2.5],[3.9,6.15,3],'concrete'),('window',[2.1,6.02,1.15],[3.9,6.03,2.5],'glass')]
 for b in FIXED:
  xyz=np.array([b['x'],b['y'],b['size'][2]/2]);d=np.array(b['size'])/2;boxes.append((b['name'],xyz-d,xyz+d,b['material']))
 xml='<scene version="3.0.0">'+''.join(f'<bsdf type="diffuse" id="itu_{m}"><rgb name="reflectance" value="0.5,0.5,0.5"/></bsdf>' for m in mats)
 for name,lo,hi,mat in boxes:
  p=folder/(name+'.ply')
  if not p.exists():write_ply(p,lo,hi)
  xml+=f'<shape type="ply" id="{name}"><string name="filename" value="{p.as_posix()}"/><ref id="itu_{mat}"/></shape>'
 scene=rt.load_scene_from_string(xml+'</scene>',merge_shapes=False);scene.frequency=SPEC['carrier_hz']
 for m,(thick,scat) in mats.items():
  material=scene.get('floor' if m=='concrete' else 'door' if m=='wood' else 'window').radio_material;material.thickness=thick;material.scattering_coefficient=scat
 scene.tx_array=rt.PlanarArray(num_rows=1,num_cols=1,pattern='dipole',polarization='V')
 scene.rx_array=rt.PlanarArray(num_rows=1,num_cols=1,pattern='dipole',polarization='V')
 for kind,positions,cls in [('tx',TX,rt.Transmitter),('rx',RX,rt.Receiver)]:
  for i,p in enumerate(positions):scene.add(cls(f'{kind}{i}',position=p))
 cube=folder/'unit_cube.ply'
 if not cube.exists():write_ply(cube,[-.5,-.5,-.5],[.5,.5,.5])
 mat=rt.ITURadioMaterial('home_target_wood','wood',thickness=.1,scattering_coefficient=.08)
 objects=[rt.SceneObject(fname=str(cube),name=f'target_{i}',radio_material=mat) for i in range(3)];scene.edit(add=objects)
 for o in objects:o.position=[3,3,-10]
 return scene,objects,rt.PathSolver()

def channel(scene,solver,seed=17):
 p=solver(scene,max_depth=SPEC['max_depth'],samples_per_src=SPEC['samples_per_src'],seed=seed,los=True,specular_reflection=True,diffuse_reflection=True,refraction=True)
 f=np.linspace(-10e6,10e6,64,endpoint=False,dtype=np.float32)
 h=np.asarray(p.cfr(frequencies=f,normalize_delays=False,normalize=False,out_type='numpy')).reshape(32,64)
 if not np.isfinite(h).all():raise ValueError('Nonfinite channel')
 return h.astype(np.complex64)

if __name__=='__main__':
 from multi_simulate import place
 t=time.perf_counter();scene,objects,solver=build();ref=channel(scene,solver);print('BUILD_REFERENCE',round(time.perf_counter()-t,3),flush=True)
 times=[]
 for i in range(5):
  ps=check([dict(kind=0,x=2.6+i*.08,y=2.7,size=SIZES[0].tolist()),dict(kind=2,x=4.6,y=4.9,size=SIZES[2].tolist())]);place(objects,ps);t=time.perf_counter();h=channel(scene,solver);times.append(time.perf_counter()-t)
  print(json.dumps(dict(i=i,seconds=times[-1],response_change=float(np.linalg.norm(h-ref)/np.linalg.norm(ref)))),flush=True)
 print('MEDIAN_SECONDS',float(np.median(times)),flush=True)
