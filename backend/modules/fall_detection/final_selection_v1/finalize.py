"""Run only after all six runs exist: select, freeze, independent inference, then test."""
import csv,json,subprocess,sys
from pathlib import Path
from runtime import MODULE,sha

HERE=Path(__file__).resolve().parent
protocol=HERE/'protocol.json'
manifest=MODULE/'reports/sample_manifest.csv'
runs=[]
for kind in ('original_1d_cnn','official_resnet18'):
    for seed in (42,43,44):
        if kind=='original_1d_cnn':
            train_name='cnn_seed42_mps_v1' if seed==42 else f'final_original_cnn_seed{seed}_mps_v1'
            checkpoint=MODULE/'outputs'/train_name/'best.pt'
            report=MODULE/'reports'/f'final_selection_v1_original_seed{seed}_cpu'
        else:
            train_name='official_resnet18_seed42_recovery_v1' if seed==42 else f'official_resnet18_seed{seed}_v1'
            checkpoint=MODULE/'outputs'/train_name/'reproduction_bundle.pt'
            report=MODULE/'reports'/f'final_selection_v1_resnet_seed{seed}_cpu'
        if not checkpoint.is_file() or not (report/'validation_predictions.csv').is_file():raise FileNotFoundError(f'Incomplete candidate: {kind} seed {seed}; do not finalize')
        receipt=json.loads((report/'receipt.json').read_text())
        assert receipt['checkpoint_sha256']==sha(checkpoint) and receipt['seed']==seed and receipt['n']==1005
        runs.append((kind,seed,checkpoint,report/'validation_predictions.csv'))
selection_dir=MODULE/'reports/final_selection_v1_three_seeds'
selection_file=selection_dir/'selection.json'
selection_already_exists=selection_file.exists()
command=[sys.executable,str(HERE/'select_across_seeds.py'),'--protocol',str(protocol),'--manifest',str(manifest),'--output-dir',str(selection_dir)]
for kind,seed,checkpoint,predictions in runs:command+=['--run',f'{kind}:{seed}:{predictions}:{checkpoint}']
if not selection_already_exists:subprocess.run(command,check=True)
selection=json.loads(selection_file.read_text())
assert selection['protocol_sha256']==sha(protocol) and selection['manifest_sha256']==sha(manifest)
assert {(r['kind'],r['seed'],r['checkpoint_sha256'],r['predictions_sha256']) for r in selection['runs']}=={(kind,seed,sha(checkpoint),sha(predictions)) for kind,seed,checkpoint,predictions in runs}
frozen=MODULE/'outputs/final_selection_v1/final_research_model.pt'
subprocess.run([sys.executable,str(HERE/'freeze_final.py'),'--selection',str(selection_file),'--protocol',str(protocol),'--output',str(frozen)],check=True)
# Four real validation records: both devices and both labels. Do not cherry-pick correct examples.
all_rows=list(csv.DictReader(manifest.open()))
val=[r for r in all_rows if 'val_id' in r['split'].split(';')]
reference={r['sample_id']:r for r in csv.DictReader(Path(selection['selected_run']['prediction_file']).open())}
inf_dir=MODULE/'reports/final_selection_v1_independent_inference'
inf_dir.mkdir(parents=True,exist_ok=False)
checks=[]
for device in ('HP','ESP32'):
    for label in (0,1):
        row=next(r for r in val if r['device']==device and int(r['binary_label'])==label)
        out=inf_dir/f'{device}_label{label}.json'
        subprocess.run([sys.executable,str(HERE/'infer.py'),'--checkpoint',str(frozen),'--file',row['file_path'],'--output',str(out)],check=True)
        result=json.loads(out.read_text());ref=float(reference[row['sample_id']]['fall_score'])
        difference=abs(result['fall_score']-ref)
        assert difference<1e-4
        assert (result['fall_score']>=result['threshold'])==(ref>=result['threshold'])
        checks.append({'sample_id':row['sample_id'],'device':device,'true_label':label,'new_process_score':result['fall_score'],'validation_reference_score':ref,'absolute_difference':difference,'prediction':result['prediction']})
check_report={'new_python_process_per_file':True,'sample_n':4,'max_score_difference':max(r['absolute_difference'] for r in checks),'changed_predictions':0,'records':checks,'test_used':False,'checkpoint_sha256':sha(frozen)}
(inf_dir/'reload_check.json').write_text(json.dumps(check_report,indent=2))
# Test inference starts only after the frozen receipt and independent inference have passed.
subprocess.run([sys.executable,str(HERE/'evaluate_final.py'),'--checkpoint',str(frozen),'--manifest',str(manifest),'--output-dir',str(MODULE/'reports/final_selection_v1_test')],check=True)
before=json.loads((HERE/'preserved_artifacts_before.json').read_text())
assert before=={n:sha(MODULE/n) for n in before},'An earlier model/source artifact changed'
receipt={'completed':True,'selected_kind':selection['selected_kind'],'seed':42,'selected_threshold':selection['selected_run']['selected_threshold'],'frozen_checkpoint':str(frozen),'checkpoint_sha256':sha(frozen),'independent_inference':check_report,'earlier_artifacts_unchanged':True,'test_evaluated_after_freeze':True,'protocol_sha256':sha(protocol),'selection_sha256':sha(selection_file),'deployment_claim':selection['deployment_claim']}
(MODULE/'reports/final_selection_v1_completion.json').write_text(json.dumps(receipt,indent=2))
print('FOUR_STAGES_COMPLETE',json.dumps(receipt),flush=True)
