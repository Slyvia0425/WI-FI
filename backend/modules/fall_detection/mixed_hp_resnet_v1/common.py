"""Independent HP experiment. No production model or runtime is imported/changed."""
import hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

LABELS={'Fall':0,'Nonfall':1}
MODEL_CONFIG={'win_len':500,'feature_size':232,'num_classes':2,'in_channels':1}
OLD_THRESHOLD=0.574872225522995
OLD_SHA='e3aabc34810aee4a647555eb7b1970bbf19a063b9d76f31d4ff59bfc5521a0a0'

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()

def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

class ResNet18Classifier(nn.Module):
 """Same construction order and layers as pinned CSI-Bench ResNet18Classifier."""
 def __init__(self,win_len=500,feature_size=232,num_classes=2,in_channels=1):
  super().__init__()
  self.resnet=resnet18(weights=None)
  self.resnet.conv1=nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3,bias=False)
  self.resnet.fc=nn.Linear(512,num_classes)
 def forward(self,x):return self.resnet(x)

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True,warn_only=True)

class CachedDataset(torch.utils.data.Dataset):
 def __init__(self,cache,role):
  self.x=np.load(Path(cache)/(role+'.npy'),mmap_mode='r')
  self.rows=json.loads((Path(cache)/(role+'_records.json')).read_text())
  assert self.x.shape==(len(self.rows),1,500,232)
  assert all(r['role']==role and r['device']=='HP' for r in self.rows)
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):return torch.from_numpy(self.x[i].copy()),self.rows[i]['label'],i

def verify_isolation(train,val):
 assert not {r['id'] for r in train}&{r['id'] for r in val}
 assert not {r['content_hash'] for r in train}&{r['content_hash'] for r in val}
 assert len({r['content_hash'] for r in train})==len(train)
 assert all(r['official_split']=='train_id' for r in train)
 assert all(r['official_split']=='val_id' for r in val)

def metrics(rows,scores,threshold):
 y=np.array([r['label']==0 for r in rows]);p=np.array(scores)>=threshold
 tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
 precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
 return {'n':len(y),'Fall':int(y.sum()),'Nonfall':int((~y).sum()),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':precision,'recall':recall,'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'nonfall_trigger_rate':fp/(fp+tn) if fp+tn else None}

@torch.inference_mode()
def predict(model,loader,device,weights=None):
 model.eval();scores=[];loss_num=0.;loss_den=0.;indices=[]
 for x,y,idx in loader:
  x=x.to(device);y=y.to(device);out=model(x)
  if not torch.isfinite(out).all():raise ValueError('Nonfinite logits')
  scores.extend(out.softmax(1)[:,0].cpu().tolist());indices.extend(idx.tolist())
  loss_num+=torch.nn.functional.cross_entropy(out,y,weight=weights,reduction='sum').item()
  loss_den+=weights[y].sum().item() if weights is not None else len(y)
 assert indices==list(range(len(loader.dataset)))
 return scores,loss_num/loss_den

def load_model(path,device):
 state=torch.load(path,map_location='cpu',weights_only=True)
 assert state['label_mapping']==LABELS
 assert state.get('model_config',MODEL_CONFIG)==MODEL_CONFIG
 model=ResNet18Classifier(**MODEL_CONFIG);model.load_state_dict(state['model_state_dict'],strict=True)
 return model.to(device).eval(),state
