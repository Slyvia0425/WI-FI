# Public upstream code only; no local checkpoint or private file is uploaded.
import os, sys, subprocess, random, json, math, gc
from pathlib import Path
import numpy as np
import torch
COMMIT='b1beac7b0fa46e01748901195ecfbe309dfc0180'
UPSTREAM=Path('/content/csi_bench_official')
if not UPSTREAM.exists():
    subprocess.run(['git','clone','https://github.com/guozhen-jenn-zhu/CSI-Bench-Real-WiFi-Sensing-Benchmark.git',str(UPSTREAM)],check=True)
subprocess.run(['git','-C',str(UPSTREAM),'checkout','--detach',COMMIT],check=True)
assert subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()==COMMIT
subprocess.run([sys.executable,'-m','pip','-q','install','remotezip','h5py','pandas','scipy','scikit-learn','seaborn','tqdm'],check=True)
sys.path.insert(0,str(UPSTREAM))
from model.supervised.models import ResNet18Classifier
assert torch.cuda.is_available()
probe_model=None; probe_x=None; probe_loss=None; probe_opt=None
try:
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    probe_model=ResNet18Classifier(win_len=500,feature_size=232,num_classes=2,in_channels=1).cuda().train()
    probe_opt=torch.optim.AdamW(probe_model.parameters(),lr=.001,weight_decay=1e-5)
    probe_x=torch.randn(128,1,500,232,device='cuda')
    probe_y=torch.arange(128,device='cuda')%2
    probe_loss=torch.nn.functional.cross_entropy(probe_model(probe_x),probe_y)
    probe_loss.backward();probe_opt.step();torch.cuda.synchronize()
    print('BATCH_128_PROBE_PASS',{'loss':float(probe_loss.detach()),'peak_GiB':torch.cuda.max_memory_allocated()/2**30},flush=True)
    COLAB_BATCH_SIZE=128
except torch.cuda.OutOfMemoryError:
    COLAB_BATCH_SIZE=None
    print('BATCH_128_PROBE_OOM: formal training has NOT started; do not silently change batch size.',flush=True)
finally:
    del probe_model,probe_x,probe_loss,probe_opt
    gc.collect();torch.cuda.empty_cache()
print('SOURCE_COMMIT',COMMIT)
print('FORMAL_BATCH_SIZE',COLAB_BATCH_SIZE)
