# Stage 3 only: execute AFTER the seed-42 validation comparison has completed.
import os,sys,json,subprocess,base64,gzip,hashlib
from pathlib import Path
assert (PERSIST_ROOT/'protocol.json').is_file()
assert json.loads((PERSIST_ROOT/'protocol.json').read_text())['seeds']==[42,43,44]
env=os.environ.copy();env['FALL_EXPERIMENT_ROOT']=str(PERSIST_ROOT)
for seed in (43,44):
    name=f'official_resnet18_seed{seed}_v1'
    log_path=PERSIST_ROOT/'logs'/f'{name}.log'
    print('STAGE3_BEGIN_SEED',seed,flush=True)
    with log_path.open('x') as log:
        process=subprocess.Popen([sys.executable,'-u','/content/train_official_resnet18.py','--seed',str(seed),'--run-name',name],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
        for line in process.stdout:
            log.write(line);log.flush()
            if line.startswith('OBSERVED '):
                r=json.loads(line[len('OBSERVED '):])
                print(f"Seed {seed} Epoch {r['epoch']:03d} | val_loss={r['val_loss']:.6f} | Fall F1={r['all']['fall_f1']:.4f} | HP F1={r['by_device']['HP']['fall_f1']:.4f} | ESP32 F1={r['by_device']['ESP32']['fall_f1']:.4f}",flush=True)
            elif line.startswith('CONFIG '):
                cfg=json.loads(line[7:]);print('CONFIG',{k:cfg[k] for k in ('seed','batch_size','max_epochs','patience','gpu','persistent_root','test_evaluated')},flush=True)
            elif line.startswith(('Epoch ','DURABLE_BEST','Early stopping','DONE ','Traceback','Error','RuntimeError','ValueError','AssertionError')):print(line,end='',flush=True)
        code=process.wait()
    assert code==0,f'Seed {seed} failed; inspect persistent log'
    verification=subprocess.run([sys.executable,'-u','/content/verify_official_checkpoint.py','--experiment-root',str(PERSIST_ROOT),'--run-name',name,'--upstream','/content/csi_bench_official','--data-root','/content/CSI-Bench-v12','--device','cuda'],capture_output=True,text=True)
    print(verification.stdout,flush=True)
    if verification.returncode:print(verification.stderr,flush=True)
    assert verification.returncode==0
    report=PERSIST_ROOT/'reports'/name
    export={'seed':seed,'run_name':name,**{k:json.loads((report/f).read_text()) for k,f in [('summary','training_summary.json'),('config','run_config.json'),('reload','independent_reload_check.json')]}}
    export['validation_csv_gzip_base64']=base64.b64encode(gzip.compress((report/'best_validation_predictions.csv').read_bytes())).decode()
    print(f'SEED_{seed}_EXPORT_BASE64 '+base64.b64encode(gzip.compress(json.dumps(export).encode())).decode(),flush=True)
print('STAGE3_RESNET_COMPLETED: all files remain in persistent Drive storage',flush=True)
