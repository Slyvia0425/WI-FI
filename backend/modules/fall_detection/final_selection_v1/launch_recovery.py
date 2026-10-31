import os, sys, json, subprocess
from pathlib import Path
assert COLAB_BATCH_SIZE==128
assert (PERSIST_ROOT/'storage_check.txt').is_file()
source=Path('/content/train_official_resnet18.py')
persist_source=PERSIST_ROOT/'source'
persist_source.mkdir(exist_ok=True)
source_copy=persist_source/source.name
if source_copy.exists():assert source_copy.read_bytes()==source.read_bytes()
else:source_copy.write_bytes(source.read_bytes())
log_dir=PERSIST_ROOT/'logs';log_dir.mkdir(exist_ok=True)
log_path=log_dir/'official_resnet18_seed42_recovery_v1.log'
env=os.environ.copy();env['FALL_EXPERIMENT_ROOT']=str(PERSIST_ROOT)
with log_path.open('x') as log:
    process=subprocess.Popen([sys.executable,'-u',str(source),'--seed','42','--run-name','official_resnet18_seed42_recovery_v1'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
    for line in process.stdout:
        log.write(line);log.flush()
        if line.startswith('OBSERVED '):
            row=json.loads(line[len('OBSERVED '):])
            print(f"Epoch {row['epoch']:03d} | val_loss={row['val_loss']:.6f} | Fall F1={row['all']['fall_f1']:.4f} | HP F1={row['by_device']['HP']['fall_f1']:.4f} | ESP32 F1={row['by_device']['ESP32']['fall_f1']:.4f}",flush=True)
        elif line.startswith('CONFIG '):
            cfg=json.loads(line[7:]);print('CONFIG', {k:cfg[k] for k in ('seed','batch_size','max_epochs','patience','gpu','persistent_root','test_evaluated')},flush=True)
        elif line.startswith(('Epoch ','DURABLE_BEST','Best model saved','Early stopping','DONE ','Traceback','Error','RuntimeError','ValueError','AssertionError')):
            print(line,end='',flush=True)
    returncode=process.wait()
assert returncode==0, f'Training failed with code {returncode}; full log: {log_path}'
print('STAGE1_TRAINING_FINISHED: persistent files are ready for independent verification',flush=True)
