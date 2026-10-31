"""Independent single-segment inference for the new HP candidate, without B gating."""
import argparse,json
from pathlib import Path
import h5py,numpy as np,torch
from common import load_model,sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--file',type=Path,required=True);ap.add_argument('--device',default='cpu');a=ap.parse_args()
 torch.set_num_threads(2)
 with h5py.File(a.file) as h:raw=h['CSI_amps'][()]
 if raw.shape!=(232,500,1) or np.iscomplexobj(raw) or not np.isfinite(raw).all():raise ValueError('Only finite, real HP CSI amplitude [232,500,1] is supported')
 x=torch.from_numpy(raw).float().permute(2,1,0)
 x=(x-x.mean((1,2),keepdim=True))/x.std((1,2),keepdim=True).clamp_min(1e-8)
 assert torch.isfinite(x).all()
 model,state=load_model(a.checkpoint,a.device)
 with torch.inference_mode():score=float(model(x.unsqueeze(0).to(a.device)).softmax(1)[0,0].cpu())
 threshold=.5
 print(json.dumps({'file':str(a.file),'fall_score':score,'threshold':threshold,'prediction':'Fall' if score>=threshold else 'Nonfall','model_version':state['model_version'],'checkpoint_sha256':sha(a.checkpoint),'input_shape':[1,1,500,232],'interpretation':'Uncalibrated classification score; independent C inference, no B gate'},indent=2))

if __name__=='__main__':main()
