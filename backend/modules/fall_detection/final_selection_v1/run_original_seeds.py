"""Stage 3: unchanged original CNN training for predeclared seeds 43 and 44."""
import json,subprocess,sys
from pathlib import Path
MODULE=Path(__file__).resolve().parent.parent
for seed in (43,44):
    name=f'final_original_cnn_seed{seed}_mps_v1'
    report=MODULE/'reports'/name
    checkpoint=MODULE/'outputs'/name
    summary=report/'training_summary.json'
    if summary.exists():
        old=json.loads(summary.read_text())
        assert old['completed'] and old['config']['seed']==seed and (checkpoint/'best.pt').is_file()
        print('VERIFIED_COMPLETED_RUN',name,flush=True);continue
    logpath=MODULE/'reports'/f'{name}.log'
    with logpath.open('x') as log:
        command=[sys.executable,'-u',str(MODULE/'train.py'),'--manifest',str(MODULE/'reports/sample_manifest.csv'),
                 '--preprocess-config',str(MODULE/'outputs/preprocess_v1.json'),'--audit-details',str(MODULE/'reports/audit_details.json'),
                 '--checkpoint-dir',str(checkpoint),'--report-dir',str(report),'--seed',str(seed),
                 '--max-epochs','100','--patience','10','--batch-size','32','--learning-rate','0.001','--device','mps']
        print('START_ORIGINAL_SEED',seed,flush=True)
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            log.write(line);log.flush()
            if line.startswith('EPOCH '):
                r=json.loads(line[6:]);print(f"Seed {seed} Epoch {r['epoch']:03d} | train_loss={r['train_loss']:.6f} | val_loss={r['val_loss']:.6f} | F1@0.5={r['f1']:.4f} | wait={r['wait']}",flush=True)
            else:print(line,end='',flush=True)
        code=process.wait()
    assert code==0,f'Seed {seed} failed; see {logpath}'
print('ORIGINAL_TWO_SEED_TRAINING_COMPLETE',flush=True)
