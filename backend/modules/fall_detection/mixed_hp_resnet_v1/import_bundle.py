"""Verify a self-generated Colab bundle and restore only this experiment's artifacts."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path,PurePosixPath
from common import sha,dump
M=Path(__file__).resolve().parent.parent
R=M/'reports/mixed_hp_resnet_v1'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('bundle',type=Path);a=ap.parse_args()
 protected=json.loads((R/'protected_artifacts.json').read_text())
 assert all(sha(p)==h for p,h in protected.items()),'Original artifacts changed'
 roots={'training':M/'outputs/hp_mixed_resnet18_v1','reports':R/'colab','source':R/'colab/source','startup_check':R/'colab/startup_check'}
 with zipfile.ZipFile(a.bundle) as z:
  manifest=json.loads(z.read('artifact_manifest.json'))
  assert sum(i.file_size for i in z.infolist())<1024**3
  assert set(z.namelist())==set(manifest)|{'artifact_manifest.json'}
  for name,info in manifest.items():
   parts=PurePosixPath(name).parts
   assert parts[0] in roots and '..' not in parts and not name.startswith('/')
   target=roots[parts[0]].joinpath(*parts[1:])
   assert target.resolve().is_relative_to(roots[parts[0]].resolve())
   h=hashlib.sha256();size=0
   with z.open(name) as f:
    for block in iter(lambda:f.read(8*1024**2),b''):h.update(block);size+=len(block)
   assert h.hexdigest()==info['sha256'] and size==info['bytes'],name
   if target.exists():assert sha(target)==info['sha256'],'Refusing to overwrite different prior artifact: '+str(target)
   else:
    target.parent.mkdir(parents=True,exist_ok=True)
    with z.open(name) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
  dump(R/'colab/artifact_manifest.json',manifest)
 status=json.loads((roots['training']/'status.json').read_text())
 assert status['status']=='complete'
 assert sha(roots['training']/'best.pt')==status['checkpoint_sha256']
 assert all(sha(p)==h for p,h in protected.items()),'Original artifacts changed'
 bundle_copy=M/'outputs/hp_mixed_resnet18_v1_bundle.zip'
 if bundle_copy.exists():assert sha(bundle_copy)==sha(a.bundle)
 else:shutil.copyfile(a.bundle,bundle_copy)
 dump(R/'artifact_import_receipt.json',{'bundle_sha256':sha(bundle_copy),'best_sha256':status['checkpoint_sha256'],'files_verified':len(manifest),'old_artifacts_unchanged':True,'training_status':status})
 print(json.dumps(status,indent=2));print('VERIFIED_AND_RESTORED',len(manifest),'files')

if __name__=='__main__':main()
