"""Package frozen inference and evidence; verify the extracted package in a new process."""
import csv,hashlib,json,os,subprocess,sys,tempfile,zipfile
from pathlib import Path
from runtime import MODULE,sha

completion=MODULE/'reports/final_selection_v1_completion.json'
result=json.loads(completion.read_text());assert result['completed']
frozen=Path(result['frozen_checkpoint'])
assert sha(frozen)==result['checkpoint_sha256']
receipt=json.loads(frozen.with_suffix('.json').read_text())
output=frozen.parent/'module_c_final_research_v1.zip'
if output.exists():raise FileExistsError('Final package exists; do not overwrite')
files={frozen,frozen.with_suffix('.json'),completion,MODULE/'preprocess.py',MODULE/'models/__init__.py',MODULE/'models/cnn.py'}
files.update(MODULE/n for n in receipt['inference_source_sha256'])
files.update((MODULE/'final_selection_v1').glob('*.py'))
files.update([MODULE/'final_selection_v1/protocol.json',MODULE/'final_selection_v1/preserved_artifacts_before.json'])
files.add(MODULE/'official_reproduction/requirements.lock.txt')
if result['selected_kind']=='official_resnet18':
    upstream=MODULE/'official_reproduction/upstream'
    provenance=json.loads((upstream.parent/'upstream_provenance.json').read_text())
    files.update(upstream/n for n in provenance['files_sha256'])
for folder in ['final_selection_v1_seed42_comparison_v2','final_selection_v1_three_seeds','final_selection_v1_independent_inference','final_selection_v1_test']:
    files.update(p for p in (MODULE/'reports'/folder).rglob('*') if p.is_file())
for family in ('original','resnet'):
    for seed in (42,43,44):files.update(p for p in (MODULE/'reports'/f'final_selection_v1_{family}_seed{seed}_cpu').glob('*') if p.is_file())
for name in ('final_selection_v1_report_zh.md','final_selection_v1_original_training_curves.png','final_selection_v1_resnet_training_curves.png'):
    p=MODULE/'reports'/name
    if p.exists():files.add(p)
readme='''# Frozen Module C research-model package

The selected model is a research baseline. Its device-specific errors and prior test exposure are disclosed in reports. It is not a validated continuous alarm system.

Install the tested dependencies from official_reproduction/requirements.lock.txt, or use the original project virtual environment. From this extracted folder:

```bash
python final_selection_v1/infer.py --checkpoint outputs/final_selection_v1/final_research_model.pt --file /absolute/path/to/one_actual_segment.h5
```

The file must contain finite nonnegative float amplitude CSI_amps with verified layout [features,time,1]: HP [232,500,1] or ESP32 [64,500,1]. No resizing or temporal windows are applied. The model loads saved normalization/configuration and the frozen threshold. Fall scores are not asserted to be calibrated real-world accident probabilities.

No raw data is included. Re-running all training experiments additionally requires the original project data/manifest and source directories described in the project README. The package supports independent single-file inference without train.py or training memory. Frozen source hashes are checked before inference; edit sources only as a new version.
'''
with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as z:
    z.writestr('fall_detection_final_v1/README.md',readme)
    for p in sorted(files):z.write(p,'fall_detection_final_v1/'+str(p.relative_to(MODULE)))
with zipfile.ZipFile(output) as z:assert z.testzip() is None
# The archive must work from its own extracted directory, not by importing project files.
manifest=list(csv.DictReader((MODULE/'reports/sample_manifest.csv').open()))
row=next(r for r in manifest if 'val_id' in r['split'].split(';') and r['device']=='HP' and r['binary_label']=='0')
expected=json.loads((MODULE/'reports/final_selection_v1_independent_inference/HP_label0.json').read_text())
with tempfile.TemporaryDirectory(prefix='module_c_package_check_') as temp:
    with zipfile.ZipFile(output) as z:z.extractall(temp)
    root=Path(temp)/'fall_detection_final_v1'
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    proc=subprocess.run([sys.executable,str(root/'final_selection_v1/infer.py'),'--checkpoint',str(root/'outputs/final_selection_v1/final_research_model.pt'),'--file',row['file_path']],cwd=root,env=env,text=True,capture_output=True,check=True)
    observed=json.loads(proc.stdout)
    difference=abs(observed['fall_score']-expected['fall_score'])
    assert difference<1e-4 and observed['prediction']==expected['prediction']
check={'archive':str(output),'bytes':output.stat().st_size,'sha256':sha(output),'zip_crc_valid':True,
       'extracted_package_new_process_inference_passed':True,'max_score_difference':difference,
       'source_training_memory_required':False,'raw_data_included':False}
(MODULE/'reports/final_selection_v1_package_check.json').write_text(json.dumps(check,indent=2))
print(json.dumps(check,indent=2))
