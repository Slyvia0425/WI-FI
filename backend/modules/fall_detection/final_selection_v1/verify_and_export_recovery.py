# Runs after the recovery training cell. Validation only, in a NEW Python process.
import subprocess,sys,json,base64,gzip,zipfile,io,hashlib
from pathlib import Path
from IPython.display import display,HTML
name='official_resnet18_seed42_recovery_v1'
result=subprocess.run([sys.executable,'-u','/content/verify_official_checkpoint.py',
    '--experiment-root',str(PERSIST_ROOT),'--run-name',name,
    '--upstream','/content/csi_bench_official','--data-root','/content/CSI-Bench-v12','--device','cuda'],
    capture_output=True,text=True)
print(result.stdout)
if result.returncode:print(result.stderr)
assert result.returncode==0, 'Independent model reload failed; do not proceed to selection.'
report=PERSIST_ROOT/'reports'/name
export={key:json.loads((report/filename).read_text()) for key,filename in [
    ('summary','training_summary.json'),('config','run_config.json'),('reload','independent_reload_check.json')]}
export['validation_csv_gzip_base64']=base64.b64encode(gzip.compress((report/'best_validation_predictions.csv').read_bytes())).decode()
print('STAGE1_EXPORT_JSON_BEGIN')
print(json.dumps(export))
print('STAGE1_EXPORT_JSON_END')
small_zip=io.BytesIO()
with zipfile.ZipFile(small_zip,'w',zipfile.ZIP_DEFLATED) as z:
    for p in report.iterdir():
        if p.is_file():z.writestr(p.name,p.read_bytes())
raw=small_zip.getvalue()
print('REPORT_ZIP_BYTES',len(raw),'SHA256',hashlib.sha256(raw).hexdigest())
display(HTML('<a download="resnet42_recovery_reports.zip" href="data:application/zip;base64,'+base64.b64encode(raw).decode()+'">Download recovery reports ZIP (no model weights)</a>'))
print('STAGE1_VERIFIED_AND_PERSISTENT: next step is validation-only model/threshold comparison')
