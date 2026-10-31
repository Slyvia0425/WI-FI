import hashlib,json,socket,subprocess,sys,time,urllib.request,webbrowser
from pathlib import Path
root=Path(__file__).resolve().parent
app=root/'multi_object/app.py'
identity=hashlib.sha256(str(app.parent.resolve()).casefold().encode()).hexdigest()[:16]
for port in range(8770,8780):
    try:
        with socket.socket() as s:s.bind(('127.0.0.1',port))
        break
    except OSError:continue
else:raise SystemExit('No available local port')
proc=subprocess.Popen([sys.executable,'-u',str(app),'--port',str(port)],cwd=app.parent,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
url=f'http://127.0.0.1:{port}/'
for i in range(180):
    if proc.poll() is not None:raise SystemExit('Server stopped')
    try:
        with urllib.request.urlopen(url+'api/ready',timeout=2) as response:state=json.load(response)
        if state.get('ready') and state.get('project_id')==identity:
            webbrowser.open(url)
            break
    except Exception:pass
    time.sleep(1)
print(url)
try:proc.wait()
except KeyboardInterrupt:proc.terminate();proc.wait()
