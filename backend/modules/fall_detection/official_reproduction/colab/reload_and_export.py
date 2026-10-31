# Run this cell AFTER training succeeds. New Python process; validation only.
import subprocess, sys, textwrap
from pathlib import Path
reload_code=r'''
import sys, csv, json, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,'/content')
from train_resnet18 import ResNet18Classifier, BenchmarkCSIDataset, strict_collate
root=Path('/content/fall_detection_colab')
name='official_resnet18_seed42_colab_v1'
report=root/'reports'/name
checkpoint=root/'outputs'/name/'best_model.pt'
summary=json.loads((report/'training_summary.json').read_text())
assert summary['completed'] and not summary['test_evaluated']
ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
model=ResNet18Classifier(win_len=500,feature_size=232,num_classes=2,in_channels=1).cuda().eval()
model.load_state_dict(ck['model_state_dict'],strict=True)
val=BenchmarkCSIDataset('/content/CSI-Bench-v12','FallDetection',split_name='val_id',file_format='h5',data_key='CSI_amps')
loader=DataLoader(val,batch_size=128,shuffle=False,num_workers=0,collate_fn=strict_collate)
scores=[]
with torch.inference_mode():
    for x,y in loader:
        scores.extend(model(x.cuda()).softmax(1)[:,0].cpu().tolist())
reference=list(csv.DictReader((report/'best_validation_predictions.csv').open()))
assert [r['sample_id'] for r in reference]==val.split_metadata['id'].astype(str).tolist()
a=np.array(scores);b=np.array([float(r['fall_score']) for r in reference])
diff=float(np.max(np.abs(a-b)))
changed=int(np.sum((a>.5)!=(b>.5)))
assert diff<1e-4 and changed==0
result={'new_python_process':True,'validation_n':len(a),'max_score_difference':diff,'changed_predictions':changed,'best_epoch':int(ck['epoch'])+1,'test_evaluated':False,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
(report/'checkpoint_check.json').write_text(json.dumps(result,indent=2))
print('INDEPENDENT_RELOAD_PASS',json.dumps(result),flush=True)
'''
Path('/content/check_colab_checkpoint.py').write_text(textwrap.dedent(reload_code))
subprocess.run([sys.executable,'-u','/content/check_colab_checkpoint.py'],check=True)
import json, zipfile
run_name='official_resnet18_seed42_colab_v1'
root=Path('/content/fall_detection_colab')
print(json.dumps(json.loads((root/'reports'/run_name/'training_summary.json').read_text()),indent=2))
from IPython.display import Image, display
display(Image(filename=str(root/'reports'/run_name/'training_curves.png')))
archive=Path('/content/CSI-Bench_ResNet18_seed42_results.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for group in ('reports','outputs'):
        for path in (root/group/run_name).rglob('*'):
            if path.is_file():z.write(path,path.relative_to(root))
    for path in (Path('/content/train_resnet18.py'),Path('/content/check_colab_checkpoint.py'),Path('/content/official_resnet18_training.log'),Path('/content/CSI-Bench-v12/download_receipt.json'),Path('/content/CSI-Bench-v12/official_fall_zip_index.json')):
        z.write(path,path.name)
print('RESULT_ARCHIVE',str(archive),'bytes',archive.stat().st_size)
from google.colab import files
files.download(str(archive))
