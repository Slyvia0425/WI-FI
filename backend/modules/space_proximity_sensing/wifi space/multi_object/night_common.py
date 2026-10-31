
import json,os,time,ctypes
from pathlib import Path
from datetime import datetime,timezone,timedelta
from contextlib import contextmanager
from multi_common import ROOT,OUT,DATA
NIGHT=OUT/'overnight';NIGHT.mkdir(exist_ok=True,parents=True)
TZ=timezone(timedelta(hours=8));DEADLINE=datetime.fromisoformat(os.environ.get('ROOMSENSE_DEADLINE','2026-10-30T09:00:00+08:00'))
if DEADLINE.tzinfo is None:raise ValueError('ROOMSENSE_DEADLINE must include timezone, e.g. +08:00')

def now():return datetime.now(TZ).isoformat()
def atomic(name,value):
    p=NIGHT/name;tmp=p.with_name(p.name+'.'+str(os.getpid())+'.partial');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(p)
def alive(pid):

    k=ctypes.windll.kernel32;k.OpenProcess.restype=ctypes.c_void_p
    h=k.OpenProcess(0x1000,False,int(pid))
    if not h:return False
    code=ctypes.c_ulong();k.GetExitCodeProcess(ctypes.c_void_p(h),ctypes.byref(code));k.CloseHandle(ctypes.c_void_p(h));return code.value==259
@contextmanager
def stage_lock(name):
    p=NIGHT/(name+'.lock')
    if p.exists():
        old=json.loads(p.read_text())
        if alive(old['pid']):raise RuntimeError('Stage already running: '+name)
        p.unlink()
    with p.open('x') as f:json.dump(dict(pid=os.getpid(),started=now()),f)
    try:yield
    finally:p.unlink(missing_ok=True)
def log(message):
    with (ROOT/'OVERNIGHT_LOG.md').open('a',encoding='utf-8') as f:f.write('\n- '+now()+' '+message+'\n')
    print(message,flush=True)
