"""Real original-checkpoint inference, reusable with cache/ID/hash verification."""
import argparse,json
from pathlib import Path
import torch
from common import *

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--original',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cpu');ap.add_argument('--batch-size',type=int,default=16);a=ap.parse_args()
 torch.set_num_threads(2);assert sha(a.original)==OLD_SHA
 val=CachedDataset(a.cache,'val');model,state=load_model(a.original,a.device)
 loader=torch.utils.data.DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=0)
 scores,_=predict(model,loader,a.device)
 stats=[[float(val.x[i].mean()),float(val.x[i].std()),float(val.x[i].min()),float(val.x[i].max())] for i in range(len(val))]
 dump(a.output,{'original_sha256':sha(a.original),'cache_sha256':sha(a.cache/'val.npy'),'ids':[r['id'] for r in val.rows],'content_hashes':[r['content_hash'] for r in val.rows],'normalized_statistics':stats,'preprocess_rule':'torch.float32 permute(2,1,0); per-record mean/std(dim=(1,2),keepdim=True,correction=1),clamp1e-8; no padding','scores':scores,'device':a.device,'torch':str(torch.__version__),'inference_source_sha256':{n:sha(Path(__file__).parent/n) for n in ('baseline.py','common.py')}})
 print('BASELINE_COMPLETE',len(scores),str(a.output),flush=True)

if __name__=='__main__':main()
