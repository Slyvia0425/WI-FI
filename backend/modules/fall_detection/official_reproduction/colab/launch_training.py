assert COLAB_BATCH_SIZE == 128, "Resolve memory probe before formal training"
import subprocess
from pathlib import Path
log_path=Path('/content/official_resnet18_training.log')
with log_path.open('w') as log:
    process=subprocess.Popen([sys.executable,'-u','/content/train_resnet18.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in process.stdout:
        print(line,end='',flush=True);log.write(line);log.flush()
    returncode=process.wait()
assert returncode==0, f"Training exited with {returncode}; inspect the log, do not fabricate results."
