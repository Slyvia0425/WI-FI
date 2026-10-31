
import argparse,json,os,hashlib,subprocess,sys,time,urllib.request,urllib.error,webbrowser
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SERVICE='roomsense-multi'

def expected_identity():
    p=ROOT/'outputs'/'active_profile.json'
    version=json.loads(p.read_text(encoding='utf-8'))['version'] if p.exists() else (2 if (ROOT/'outputs'/'optimized_v2'/'ACTIVE').exists() else 1)
    revisions={}
    if version==5:
        from layout_registry import entries
        revisions={e['id']:e['revision'] for e in entries() if e['ready']}
    return dict(service=SERVICE,project_id=hashlib.sha256(str(ROOT.resolve()).casefold().encode()).hexdigest()[:16],model_version=version,layout_revisions=revisions)

def validate_identity(health,expected):
    if any(health.get(k)!=v for k,v in expected.items()):
        raise RuntimeError('Port has an older/different RoomSense service. Restart that service or choose another --port. / 端口上的服务版本或项目路径不符，请重启该服务或更换端口。')


def get(url,timeout=3):
    with urllib.request.urlopen(url,timeout=timeout) as response:return json.load(response)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8769);ap.add_argument('--no-browser',action='store_true');args=ap.parse_args()
    if not 1024<=args.port<=65535:ap.error('Port must be 1024–65535')
    python=ROOT.parent/'.venv'/'Scripts'/'python.exe'
    if not python.exists():raise RuntimeError('Python environment missing. See SETUP.md / 请先按照 SETUP.md 配置环境。')
    expected=expected_identity()
    base=f'http://127.0.0.1:{args.port}';proc=None;out=ROOT/'outputs'/'launcher';out.mkdir(exist_ok=True,parents=True)
    try:
        health=get(base+'/api/health')
        validate_identity(health,expected)
        print('Using existing RoomSense service / 使用已启动的服务',flush=True)
    except urllib.error.HTTPError as e:
        raise RuntimeError('Port is occupied by a different or older service; close its console or choose --port 8768 / 当前端口被旧服务占用') from e
    except (urllib.error.URLError,TimeoutError,ConnectionError):
        flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
        with (out/'server.stdout.log').open('ab') as stdout,(out/'server.stderr.log').open('ab') as stderr:
            proc=subprocess.Popen([str(python),'-u',str(ROOT/'app.py'),'--port',str(args.port)],cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=flags)
        (out/'last_launch.json').write_text(json.dumps(dict(pid=proc.pid,port=args.port,started=time.time()),indent=2),encoding='utf-8')
        print('Starting RoomSense / 正在启动…',flush=True)
    deadline=time.monotonic()+120;last_error=''
    while time.monotonic()<deadline:
        if proc is not None and proc.poll() is not None:raise RuntimeError('Service exited; see outputs/launcher/server.stderr.log / 服务启动失败，请查看日志')
        try:
            health=get(base+'/api/ready',timeout=45)
            validate_identity(health,expected)
            if health.get('ready'):break
        except (OSError,ValueError) as e:last_error=str(e)
        time.sleep(.5)
    else:raise RuntimeError('Service not ready: '+last_error)
    print(f'Ready / 已就绪: {base}/ (model v{health["model_version"]})',flush=True)
    if not args.no_browser:webbrowser.open(base+'/')

if __name__=='__main__':
    try:main()
    except Exception as e:print('ERROR / 错误: '+str(e),file=sys.stderr);raise SystemExit(1)
