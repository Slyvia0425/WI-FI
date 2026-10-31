"""Portable CPU inference and optional latency check for the frozen D candidate.

Accept an Intel5300 .dat file or ZIP member. Each prediction uses a complete
8000-packet window and the checkpoint's actual normalization_parameters.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import zipfile
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;MODULE=HERE.parent
sys.path.insert(0,str(MODULE/'tcs_withinsubject_v1'))
from model import CNN
from prepare import transform_dat_bytes,RECIPE


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


class FrozenFallDetector:
    def __init__(self, manifest=HERE/'frozen_candidate.json'):
        self.manifest=json.loads(Path(manifest).read_text())
        path=MODULE/self.manifest['checkpoint']
        if not path.is_file():raise FileNotFoundError('Frozen D checkpoint missing; obtain the demo artifact package: '+str(path))
        if sha(path)!=self.manifest['checkpoint_sha256']:raise ValueError('Frozen checkpoint hash mismatch')
        for name,digest in self.manifest['sources_sha256'].items():
            if sha(MODULE/name)!=digest:raise ValueError('Frozen source changed: '+name)
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['preprocess_config']!=RECIPE or state['label_mapping']!={'Nonfall':0,'Fall':1}:raise ValueError('Input/label contract differs')
        if state['model_version']!=self.manifest['model_version'] or state['threshold']!=self.manifest['threshold']:raise ValueError('Candidate identity differs')
        norm=state['normalization_parameters']
        if norm is None or norm['fit_role']!='train' or norm['fit_subjects']!=[1,2]:raise ValueError('Required training normalization missing')
        self.mean=np.array(norm['mean'],np.float32);self.scale=np.array(norm['scale'],np.float32)
        if self.mean.shape!=(90,) or self.scale.shape!=(90,) or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale<=0):raise ValueError('Invalid normalization')
        self.threshold=state['threshold'];self.model=CNN().eval();self.model.load_state_dict(state['model_state_dict'],strict=True)

    def prepare(self, raw, start_packet=0):
        if len(raw)%575:raise ValueError('Incomplete native frame; no implicit truncation or padding')
        n=len(raw)//575
        if start_packet<0 or start_packet+8000>n:raise ValueError('Requested complete8000-packet window unavailable')
        x,details=transform_dat_bytes(raw[start_packet*575:(start_packet+8000)*575])
        x=(x-self.mean)/self.scale
        if x.shape!=(1,400,90) or not np.isfinite(x).all():raise ValueError('Invalid transformed input')
        return torch.from_numpy(x[None].copy()),details

    @torch.inference_mode()
    def score(self,x):
        logits=self.model(x)
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite logits')
        return float(logits.softmax(1)[0,1])

    def predict(self,raw,start_packet=0):
        x,details=self.prepare(raw,start_packet);s=self.score(x)
        return {'fall_score':s,'threshold':self.threshold,'prediction':'Fall' if s>=self.threshold else 'Nonfall',
            'state':'suspected' if s>=self.threshold else 'normal','model_version':self.manifest['model_version'],
            'checkpoint_sha256':self.manifest['checkpoint_sha256'],'normalization_applied':True,
            'window_start_packet':start_packet,'window_end_packet':start_packet+7999,'input_shape':list(x.shape),
            'preprocessing':details,'backend':'cpu','scope':self.manifest['scope']}


def main():
    ap=argparse.ArgumentParser(description=__doc__);group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--file',type=Path);group.add_argument('--archive',type=Path)
    ap.add_argument('--member');ap.add_argument('--start-packet',type=int,default=0)
    ap.add_argument('--benchmark-repeats',type=int,default=0);ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    if args.archive and not args.member:ap.error('--archive requires --member')
    if args.benchmark_repeats<0 or args.benchmark_repeats>1000:ap.error('benchmark repeats must be0..1000')
    if args.output and args.output.exists():raise FileExistsError('Preserve existing result; choose a new output')
    if args.archive:
        with zipfile.ZipFile(args.archive) as z:raw=z.read(args.member)
    else:raw=args.file.read_bytes()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    detector=FrozenFallDetector();result=detector.predict(raw,args.start_packet)
    result['source']={'file':str(args.file) if args.file else None,'archive':str(args.archive) if args.archive else None,'member':args.member,'raw_sha256':hashlib.sha256(raw).hexdigest()}
    if args.benchmark_repeats:
        x,_=detector.prepare(raw,args.start_packet)
        for _ in range(3):detector.score(x);detector.predict(raw,args.start_packet)
        forward=[];pipeline=[]
        for _ in range(args.benchmark_repeats):
            begin=time.perf_counter();detector.score(x);forward.append((time.perf_counter()-begin)*1000)
            begin=time.perf_counter();detector.predict(raw,args.start_packet);pipeline.append((time.perf_counter()-begin)*1000)
        stats=lambda values:{'median_ms':float(np.median(values)),'p95_ms':float(np.quantile(values,.95))}
        result['latency']={'repeats':args.benchmark_repeats,'warmup':3,'batch_size':1,'threads':2,'dtype':'float32','backend':'cpu',
            'platform':platform.platform(),'machine':platform.machine(),'torch':str(torch.__version__),
            'forward_only':stats(forward),'memory_packet_decode_filter_gcv_normalize_forward':stats(pipeline),
            'excludes':['checkpoint loading','file/ZIP I/O','packet collection wait','network','alert delivery'],'not_event_alert_latency':True}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
