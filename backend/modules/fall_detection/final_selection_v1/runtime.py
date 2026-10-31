"""Frozen inference runtime for the two predeclared candidate pipelines."""
import hashlib,json,sys
from pathlib import Path
import h5py
import numpy as np
import torch

MODULE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(MODULE))
from models.cnn import FallCNN
from preprocess import Preprocessor,read_amplitude


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_candidate(path,kind,device='cpu'):
    torch.set_num_threads(2)
    state=torch.load(path,map_location='cpu',weights_only=True)
    if kind=='original_1d_cnn':
        assert state['label_mapping']=={'Fall':1,'Nonfall':0}
        model=FallCNN(**state['model_config'])
    elif kind=='official_resnet18':
        upstream=MODULE/'official_reproduction/upstream'
        provenance=json.loads((upstream.parent/'upstream_provenance.json').read_text())
        assert provenance['commit']==state['source_commit']=='b1beac7b0fa46e01748901195ecfbe309dfc0180'
        assert all(sha(upstream/n)==v for n,v in provenance['files_sha256'].items())
        sys.path.insert(0,str(upstream))
        from model.supervised.models import ResNet18Classifier
        assert state['label_mapping']=={'Fall':0,'Nonfall':1}
        expected_config={'win_len':500,'feature_size':232,'num_classes':2,'in_channels':1}
        assert state.get('model_config',expected_config)==expected_config
        model=ResNet18Classifier(**expected_config)
    else:raise ValueError('Unknown candidate kind')
    model.load_state_dict(state['model_state_dict'],strict=True)
    model.to(device).eval()
    return model,state


def transform_file(path,device_name,state,kind):
    # Validate actual raw data before any normalization, padding, or inference.
    amplitude=read_amplitude(path)
    if amplitude.shape[0]!={'HP':232,'ESP32':64}[device_name]:raise ValueError('Device/shape mismatch')
    if kind=='original_1d_cnn':
        x,_=Preprocessor(state['preprocess_config']).transform_file(path,device_name)
        return torch.from_numpy(x)
    if kind=='official_resnet18':
        raw=torch.from_numpy(amplitude.astype(np.float32)).transpose(0,1).unsqueeze(0)
        normalized=(raw-raw.mean(dim=(1,2),keepdim=True))/raw.std(dim=(1,2),keepdim=True).clamp_min(1e-8)
        result=torch.zeros((1,500,232),dtype=torch.float32)
        result[:,:,:amplitude.shape[0]]=normalized
        assert torch.isfinite(result).all()
        return result
    raise ValueError('Unknown kind')


@torch.inference_mode()
def predict_files(model,state,kind,rows,device='cpu',batch_size=8):
    records=[]
    for start in range(0,len(rows),batch_size):
        subset=rows[start:start+batch_size]
        x=torch.stack([transform_file(r['file_path'],r['device'],state,kind) for r in subset]).to(device)
        out=model(x)
        if not torch.isfinite(out).all():raise ValueError('Nonfinite logits')
        scores=out.squeeze(-1).sigmoid() if kind=='original_1d_cnn' else out.softmax(1)[:,0]
        for r,s in zip(subset,scores.cpu().tolist()):
            records.append({'sample_id':r['sample_id'],'device':r['device'],'label':int(r['binary_label']),'fall_score':s})
    assert len(records)==len(rows)
    return records


def load_frozen(path,device='cpu'):
    path=Path(path);receipt=json.loads(path.with_suffix('.json').read_text())
    if sha(path)!=receipt['checkpoint_sha256']:raise ValueError('Frozen checkpoint hash mismatch')
    kind=receipt['candidate_kind'];model,state=load_candidate(path,kind,device)
    assert state['frozen'] and state['selected_threshold']==receipt['selected_threshold']
    assert 0<=state['selected_threshold']<=1
    for name,digest in receipt['inference_source_sha256'].items():
        if sha(MODULE/name)!=digest:raise ValueError('Inference source changed after freeze: '+name)
    return model,state,kind
