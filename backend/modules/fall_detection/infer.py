"""Standalone inference for one complete HDF5 segment, using frozen embedded preprocessing."""
import argparse
import json
from pathlib import Path

import torch

if __package__:
    from .preprocess import Preprocessor
    from .inference_runtime import load_model, sha256
else:
    from preprocess import Preprocessor
    from inference_runtime import load_model, sha256


def infer_file(checkpoint,file,device):
    before=sha256(checkpoint)
    model,state=load_model(checkpoint,'cpu')
    x,mask=Preprocessor(state['preprocess_config']).transform_file(file,device)
    with torch.no_grad():
        logit=model(torch.from_numpy(x).unsqueeze(0)).squeeze()
        if not torch.isfinite(logit):
            raise ValueError('Nonfinite logit')
        score=float(logit.sigmoid())
    threshold=state['selected_threshold']
    if sha256(checkpoint)!=before:
        raise ValueError('Checkpoint changed during inference')
    return {'file':str(Path(file).resolve()),'device':device,'fall_score':score,
            'threshold':threshold,'prediction':'Fall' if score>=threshold else 'Nonfall',
            'model_version':state['model_version'],'best_epoch':state['best_epoch'],
            'present_features':int(mask.sum()),'checkpoint_sha256':before}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--file',type=Path,required=True)
    parser.add_argument('--device',required=True,help='Acquisition device HP or ESP32, not compute backend')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=infer_file(args.checkpoint,args.file,args.device)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
