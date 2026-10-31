
import ctypes,os,time
from datetime import datetime
from night_common import NIGHT,DEADLINE,TZ,atomic,stage_lock,now

def main():
    with stage_lock('awake'):
        k=ctypes.windll.kernel32;k.SetThreadExecutionState.argtypes=[ctypes.c_uint];k.SetThreadExecutionState.restype=ctypes.c_uint
        if not k.SetThreadExecutionState(0x80000001):raise ctypes.WinError()
        try:
            while datetime.now(TZ)<DEADLINE and not (NIGHT/'STOP_AWAKE').exists():
                atomic('awake.json',dict(state='active',pid=os.getpid(),updated=now(),until=DEADLINE.isoformat(),scope='System awake only; screen may turn off; no persistent power-plan changes.'));time.sleep(40)
        finally:
            k.SetThreadExecutionState(0x80000000);atomic('awake.json',dict(state='released',pid=os.getpid(),updated=now()))
if __name__=='__main__':main()
