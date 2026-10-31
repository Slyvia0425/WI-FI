"""Freeze the preselected seed-42 pipeline, threshold and inference source before test."""
import argparse,json,importlib.metadata
from datetime import datetime,timezone
from pathlib import Path
import torch
from runtime import MODULE,load_candidate,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();selection=json.loads(args.selection.read_text())
    if args.output.exists() or args.output.with_suffix('.json').exists():raise FileExistsError('Frozen artifacts must not be overwritten')
    assert selection['protocol_sha256']==sha(args.protocol) and selection['test_used_for_selection'] is False
    selected=selection['selected_run'];kind=selected['kind'];source=Path(selected['checkpoint'])
    assert selected['seed']==selection['final_artifact_seed']==42 and sha(source)==selected['checkpoint_sha256']
    _,state=load_candidate(source,kind)
    assert state['seed']==42
    state.pop('optimizer_state_dict',None)
    if kind=='official_resnet18':
        state['model_config']={'win_len':500,'feature_size':232,'num_classes':2,'in_channels':1}
        state['preprocess_config']={
            'version':'pinned_official_amplitude_per_record_v1',
            'h5_key':'CSI_amps','source_axes':['feature_position','time_index','singleton'],
            'source_shapes_by_device':{'HP':[232,500,1],'ESP32':[64,500,1]},
            'output_axes':['channel','time_index','feature_position'],'output_shape':[1,500,232],
            'dtype':'float32','mode':'per_record_scalar_zscore_over_original_time_and_features_then_right_zero_pad',
            'std_correction':1,'epsilon':1e-8,'padding_in_statistics':False,
            'raw_constant_positions_in_statistics':True,'time_resampling':False,'windowing':False,
            'filtering':False,'augmentation':False,'sample_rate_hz':None,
            'label_mapping':{'Fall':0,'Nonfall':1},'fall_score':'softmax(logits)[0]'}
        state['normalization_parameters']={'mode':'per_record','train_fitted_parameters':None,
            'explanation':'Each complete record computes its own mean and sample std over original positions, as in pinned official code; fixed algorithm/configuration is reused in inference.'}
    state['inference_versions']={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','h5py','scikit-learn')}
    state.update(frozen=True,candidate_kind=kind,selected_threshold=selected['selected_threshold'],
                 manifest_sha256=selection['manifest_sha256'],
                 model_version='module_c_final_research_v1',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                 selection_receipt_sha256=sha(args.selection),protocol_sha256=sha(args.protocol),
                 threshold_selection_status='validation_global_max_fall_f1_frozen',
                 deployment_claim=selection['deployment_claim'],known_limitations=selection['known_limitations'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix('.pt.tmp');torch.save(state,tmp);tmp.replace(args.output)
    sources=['models/cnn.py','preprocess.py','final_selection_v1/runtime.py','final_selection_v1/infer.py','final_selection_v1/evaluate_final.py','final_selection_v1/compare_thresholds.py']
    if kind=='official_resnet18':sources+=['official_reproduction/upstream/model/supervised/models.py','official_reproduction/upstream_provenance.json']
    receipt={'candidate_kind':kind,'seed':42,'selected_threshold':state['selected_threshold'],
             'checkpoint_sha256':sha(args.output),'source_checkpoint_sha256':sha(source),
             'selection_receipt_sha256':sha(args.selection),'protocol_sha256':sha(args.protocol),
             'frozen_at_utc':state['frozen_at_utc'],'test_used_for_selection':False,
             'inference_source_sha256':{n:sha(MODULE/n) for n in sources},
             'prior_test_exposure':selection['prior_test_exposure'],'deployment_claim':selection['deployment_claim']}
    args.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
