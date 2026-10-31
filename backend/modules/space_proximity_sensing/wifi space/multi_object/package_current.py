
import hashlib,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent

def include(p):
    if not p.is_file() or '.venv' in p.parts or '__pycache__' in p.parts or p.suffix in ['.partial','.lock']:return False
    if p.suffix=='.zip' or p.name.endswith('.zip.partial'):return False
    if any(v in p.parts for v in ['layout_t1_v1','layout_t1_v2','layout_t1_v3','spatial_v6']):
        if 'cache' in p.parts or p.name.startswith(('features_','predictions_')):return False
        if p.suffix=='.joblib' and (p.name!='model.joblib' or 'components' in p.parts):return False
    if 'optimized_v2' in p.parts and (p.name.startswith('features_') or p.name.startswith('pilot_') and p.suffix=='.joblib' or p.name in ['occupancy_73.joblib','occupancy_91.joblib','count.joblib','layout.joblib','selected.joblib']):return False
    if 'overnight' in p.parts:
        if 'cache' in p.parts:return False
        if p.suffix=='.joblib' and ('models' in p.parts or 'pilot' in p.name or p.name=='candidate_uncalibrated.joblib'):return False
        if p.name in ['STOP_AWAKE','awake.json']:return False
    if 'home_v5' in p.parts:
        if 'cache' in p.parts:return False
        if p.suffix=='.joblib' and ('components' in p.parts or p.name=='selected.joblib'):return False
        if p.name in ['STOP_AWAKE','awake.json']:return False
    if 'shared_v4' in p.parts:
        if 'cache' in p.parts:return False
        if p.suffix=='.joblib' and ('components' in p.parts or p.name=='selected.joblib'):return False
        if p.name in ['STOP_AWAKE','awake.json']:return False
    return True

def sha_file(p):
    hh=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):hh.update(chunk)
    return hh.hexdigest()

def main():
    files=[p for p in ROOT.rglob('*') if include(p)];prov=ROOT/'outputs'/'provenance.json'
    evidence={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha_file(p)) for p in files if p!=prov}
    pointer=ROOT/'multi_object'/'outputs'/'active_profile.json';active=json.loads(pointer.read_text())['version'] if pointer.exists() else 2
    prov.write_text(json.dumps(dict(generated_at=datetime.now(timezone.utc).isoformat(),scope=f'Current multi-object active v{active} with approved optional default V6 and T1 V3; historical single-object and experimental reports retained and separately labeled.',excluded='Python environment, bytecode, partial writes/locks, derived feature caches, duplicate component checkpoints/pilot/uncalibrated bundles. All raw signals, final models, source, metrics and verification included.',files=evidence),indent=2),encoding='utf-8')
    if prov not in files:files.append(prov)
    target=ROOT.parent/('static_obstacle_roomsense_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip');tmp=target.with_suffix('.zip.partial')
    if target.exists() or tmp.exists():raise FileExistsError('Refusing to overwrite an existing package')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in files:z.write(p,'static_obstacle/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(tmp) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('CRC error '+bad)
        count=len(z.namelist())
        for rel,entry in evidence.items():
            hh=hashlib.sha256()
            with z.open('static_obstacle/'+rel) as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):hh.update(chunk)
            if hh.hexdigest()!=entry['sha256']:raise RuntimeError('Source changed during packaging: '+rel)
    tmp.replace(target);digest=sha_file(target)
    target.with_suffix('.zip.sha256').write_text(digest+'  '+target.name+'\n',encoding='utf-8')
    result=dict(path=str(target),bytes=target.stat().st_size,files=count,crc='passed',all_embedded_provenance_hashes='passed',sha256=digest,active_multi_version=active)
    target.with_suffix('.package_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
