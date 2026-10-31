# Run after the experiment completes. Downloads artifacts; does not require Drive.
import hashlib,json,sys,zipfile
from pathlib import Path
import h5py,matplotlib,numpy,torch,torchvision
from google.colab import files

assert json.loads((out/'status.json').read_text())['status']=='complete'
assert (comparison/'comparison.json').exists()
environment={'python':sys.version,'torch':str(torch.__version__),'torchvision':torchvision.__version__,'numpy':numpy.__version__,'h5py':h5py.__version__,'matplotlib':matplotlib.__version__,'gpu':torch.cuda.get_device_name(0)}
(out/'environment.json').write_text(json.dumps(environment,indent=2))
package=Path('/content/drive/MyDrive/CSI-Bench_FallDetection_FinalSelection_v1/hp_mixed_resnet18_v1/CSI_Bench_HP_Mixed_ResNet18_v1.zip')
entries=[]
saved_reports=out.parent/'reports'
if not saved_reports.is_dir():saved_reports=report
for folder,prefix in [(out,'training'),(saved_reports,'reports'),(work/'mixed_hp_resnet_v1','source')]:
 for p in sorted(folder.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:entries.append((p,prefix+'/'+p.relative_to(folder).as_posix()))
startup=out.with_name('hp_mixed_resnet18_v1_startup_check')
for name in ['training_config.json','history.json','train_log.txt','best_reference.json']:
 if (startup/name).exists():entries.append((startup/name,'startup_check/'+name))
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
receipt={name:{'sha256':digest(p),'bytes':p.stat().st_size} for p,name in entries}
with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for p,name in entries:z.write(p,name)
 z.writestr('artifact_manifest.json',json.dumps(receipt,indent=2))
print('DOWNLOAD_READY',package.name,package.stat().st_size,digest(package))
print('PACKAGE_PERSISTED',str(package))
# Download through Drive's file UI after confirming this package is present.
