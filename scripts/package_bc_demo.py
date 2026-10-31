"""Package source, prebuilt UI and pinned B/C models; exclude raw data/caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PREFIX='WiFi_Sense_BC_Demo'
SKIP={'node_modules','.git','.venv','__pycache__','outputs','reports','local_data','.cache','.pytest_cache','.mypy_cache','.ruff_cache','.agents','.codex','.aws','.claude','.idea','.vscode'}
ALLOWED={'.py','.md','.json','.js','.jsx','.css','.html','.yaml','.yml','.toml','.txt','.sh','.mjs','.h5','.csv','.png','.svg','.ipynb'}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior artifact; choose a new output')
    files={}
    for folder in ('backend','frontend','docs','scripts'):
        for here,dirs,names in os.walk(ROOT/folder):
            dirs[:]=sorted(d for d in dirs if d not in SKIP and not (Path(here).name=='frontend' and d=='dist'))
            for name in sorted(names):
                path=Path(here)/name
                if path.is_symlink() or name.endswith(('.pyc','.zip','.pt','.joblib','.pkl')):continue
                if name.startswith('.env') and not name.endswith('.example'):continue
                if path.suffix not in ALLOWED and name not in ('.gitignore','.gitkeep','.env.example','LICENSE','LICENSE.upstream'):continue
                if path.stat().st_size>8*1024**2:raise ValueError('Review unexpectedly large source file: '+str(path))
                files[path.relative_to(ROOT).as_posix()]=path
    for name in ('README.md','package.json','package-lock.json','.gitignore','.env.example'):
        path=ROOT/name
        if path.exists():files[name]=path
    module=ROOT/'backend/modules/fall_detection'
    frozen=json.loads((module/'tcs_cross_subject_v1/frozen_candidate.json').read_text())
    checkpoint=module/frozen['checkpoint']
    if digest(checkpoint)!=frozen['checkpoint_sha256']:raise ValueError('Candidate checkpoint changed')
    for name,h in frozen['sources_sha256'].items():
        if digest(module/name)!=h:raise ValueError('Frozen source changed: '+name)
    primary=json.loads((module/'resnet_b_integration/frozen_model.json').read_text())
    resnet_checkpoint=module/primary['checkpoint']
    if digest(resnet_checkpoint)!=primary['checkpoint_sha256']:raise ValueError('Primary ResNet18 checkpoint changed')
    extras=[checkpoint,resnet_checkpoint,ROOT/'backend/modules/fall_detection/outputs/replay.json',
            ROOT/'backend/modules/fall_detection/outputs/tcs_D_replay.json',ROOT/'backend/modules/human_activity_recognition/outputs/replay.json']
    extras+=[p for p in (module/'reports/resnet_b_integration_v1').glob('*') if p.name!='inputs_manifest.json']
    extras+=[module/'reports/hp_latency_v1/official_resnet18_cpu.json']
    if (module/'outputs/before_confidence_gate_v1.json').exists():
        extras+=[module/'outputs/before_confidence_gate_v1.json']
        extras+=[p for p in (module/'reports/confidence_gate_v1').glob('*') if p.is_file() and p.suffix in {'.json','.txt','.py'}]
    b_module=ROOT/'backend/modules/human_activity_recognition'
    b_config_file=b_module/'integration_v1/frozen_model.json'
    b_config=None
    if b_config_file.exists():
        b_config=json.loads(b_config_file.read_text())
        b_checkpoint=b_module/b_config['checkpoint']
        if digest(b_checkpoint)!=b_config['checkpoint_sha256']:raise ValueError('Released B checkpoint changed')
        extras+=[b_checkpoint,b_module/'outputs/before_real_B_v1.json',module/'outputs/before_real_B_v1.json']
        extras+=[p for p in (module/'reports/resnet_b_integration_v2').glob('*') if p.suffix in {'.json','.txt','.md','.csv'}]

    extras+=list((ROOT/'frontend/dist').rglob('*'))
    extras+=list((module/'reports/bc_integration_v1').glob('*'))
    evaluation=module/'reports/tcs_cross_subject_v1/comparison_seed42_v1/evaluation'
    extras+=[evaluation/'SA_CT_standardized_GCV.json',evaluation/'SA_CT_standardized_GCV_windows.csv']
    for path in extras:
        if path.is_file():files[path.relative_to(ROOT).as_posix()]=path
    if 'frontend/dist/index.html' not in files:raise ValueError('Build UI before packaging')
    built=(ROOT/'frontend/dist/index.html').stat().st_mtime
    for name in ('human_activity_recognition','fall_detection'):
        if (ROOT/f'frontend/src/modules/{name}/data/replay.json').stat().st_mtime>built:
            raise ValueError('Rebuild the UI after exporting the latest B/C data')
    manifest={'scope':'B low confidence (<0.6) selects frozen CSI-Bench ResNet18 C; skipped C has no decision; real paired B; TCS D archived separately; no real raw dataset',
        'primary_checkpoint_sha256':primary['checkpoint_sha256'],'module_B_checkpoint_sha256':b_config['checkpoint_sha256'] if b_config else None,
        'archived_TCS_checkpoint_sha256':frozen['checkpoint_sha256'],'files':[{'path':n,'bytes':p.stat().st_size,'sha256':digest(p)} for n,p in sorted(files.items())]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(files.items()):z.write(path,PREFIX+'/'+name)
        z.writestr(PREFIX+'/PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(args.output) as z:
        if z.testzip() is not None:raise ValueError('Package CRC failure')
    print(json.dumps({'archive':str(args.output.resolve()),'bytes':args.output.stat().st_size,'sha256':digest(args.output),'files':len(files),'raw_dataset_included':False},indent=2))

if __name__=='__main__':main()
