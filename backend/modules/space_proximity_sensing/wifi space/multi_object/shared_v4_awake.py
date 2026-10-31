
import ctypes,time,os
from datetime import datetime,timedelta
from shared_v4_common import O4,atomic,now
end=time.monotonic()+4*60*60
try:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    while time.monotonic()<end and not (O4/'STOP_AWAKE').exists():
        atomic('awake.json',dict(state='active',pid=os.getpid(),updated=now(),scope='Temporary system-only wake request for this training; screen can sleep; no permanent settings changed.'));time.sleep(10)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000);atomic('awake.json',dict(state='released',pid=os.getpid(),updated=now()))
