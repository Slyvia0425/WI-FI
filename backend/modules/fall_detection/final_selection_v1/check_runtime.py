"""Verify new single-file preprocessing/inference against the established Dataset runtime."""
import csv,json,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
from runtime import MODULE,load_candidate,predict_files
sys.path.insert(0,str(MODULE))
from dataset import FallDataset
from inference_runtime import collect_predictions

manifest=MODULE/'reports/sample_manifest.csv'
checkpoint=MODULE/'outputs/cnn_seed42_mps_v1/best.pt'
model,state=load_candidate(checkpoint,'original_1d_cnn')
dataset=FallDataset(manifest,state['preprocess_config'],'val_id')
chosen=[]
for device in ('HP','ESP32'):
    for label in (0,1):
        chosen.append(next(i for i,r in enumerate(dataset.rows) if r['device']==device and int(r['binary_label'])==label))
rows=[dataset.rows[i] for i in chosen]
from_files=predict_files(model,state,'original_1d_cnn',rows,batch_size=4)
from_dataset=collect_predictions(model,DataLoader(Subset(dataset,chosen),batch_size=4,shuffle=False),'cpu')
max_diff=max(abs(a['fall_score']-b['fall_score']) for a,b in zip(from_files,from_dataset))
assert [r['sample_id'] for r in from_files]==[r['sample_id'] for r in from_dataset]
assert max_diff==0
report={'sample_n':len(rows),'coverage':'HP and ESP32, Fall and Nonfall','max_score_difference':max_diff,
        'reference':'established FallDataset + inference_runtime.collect_predictions',
        'new_runtime':'raw H5 file + saved per-device train statistics + model',
        'test_evaluated':False,'training_performed':False,'records':from_files}
out=MODULE/'reports/final_selection_v1_runtime_check'
out.mkdir(exist_ok=True)
(out/'runtime_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
