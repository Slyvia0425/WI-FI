
import argparse,json,time,threading,io,csv
import socket
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
from functools import lru_cache
import numpy as np
import joblib
from learning import ROOT,CFG,classify,observe,zone_center

LOCK=threading.Lock()
SIM=None

def profile():
    p=ROOT/'active_experiment.json'
    if p.exists() and json.loads(p.read_text(encoding='utf-8')).get('experiment')=='coverage_v3':
        return ROOT/'data/coverage_v3',ROOT/'outputs/coverage_v3'
    return ROOT/'data',ROOT/'outputs'

@lru_cache(maxsize=2)
def _bundle(path,stamp):return joblib.load(path)
def bundle():
    p=profile()[1]/'model.joblib';return _bundle(str(p),p.stat().st_mtime_ns)

@lru_cache(maxsize=2)
def _records(path,stamp):return {r['id']:r for r in json.loads(Path(path).read_text())}
def records():
    p=profile()[0]/'manifest.json';return _records(str(p),p.stat().st_mtime_ns)

def package(h,truth=None,extra=None):
    result=classify(bundle(),h)
    result['signal_db']=(20*np.log10(np.maximum(np.abs(h),1e-10))).tolist()
    result['reference_delta_db']=(20*np.log10(np.maximum(np.abs(h),1e-10))-20*np.log10(np.maximum(np.abs(bundle()['reference']),1e-10))).mean(1).tolist()
    if truth is not None:
        result['truth']=truth
        result['correct']=result['prediction']==truth['zone'] if result['prediction'] is not None else None
        result['error_m']=float(np.linalg.norm(np.array(result['center_m'])-truth['xy'])) if result['center_m'] is not None else None
    result.update(extra or {})
    return result

def record_prediction(recordid,seed=42,snr=None):
    row=records().get(recordid)
    if not row or row['split']=='train':raise ValueError('Choose a held-out record.')
    with np.load(profile()[0]/'records'/(recordid+'.npz'),allow_pickle=False) as a:h=a['h']
    h=observe(h,np.random.default_rng(seed),snr_db=snr)
    devices={};idx=0
    for kind in ['tx','rx']:
        devices[kind]=[]
        for position in CFG[kind+'_m']:
            q=np.array(position,dtype=float)
            if 'device_offset_xy' in row:q[:2]+=row['device_offset_xy'][idx]
            elif row['split']=='diagnostic_original_device_shift':q[:2]+=[.06,-.04] if len(devices[kind])%2==0 else [-.06,.04]
            idx+=1;devices[kind].append(q.tolist())
    return package(h,dict(zone=row['zone'],xy=row['xy'],id=recordid,split=row['split']),dict(source='held_out_ray_traced_record',noise_seed=seed,snr_db=snr,device_positions=devices))

def simulate_prediction(x,y,seed=42,snr=None):
    global SIM
    if not(.61<=x<=5.39 and .41<=y<=5.59):raise ValueError('Keep the known box inside the room.')
    from coverage_common import valid_placement,near_boundary
    if not valid_placement([x,y]):raise ValueError('The box overlaps a TX/RX device or its 8 cm clearance. Move it away from the shaded area. / 箱体与设备或其8厘米安全间距重叠，请移开。')
    with LOCK:
        t=time.perf_counter()
        if SIM is None:
            from simulate import build_scene,response
            SIM=(build_scene(),response)
        (scene,obj,solver),response=SIM
        obj.position=[x,y,.6]
        h=response(scene,solver,17)
        rt_ms=(time.perf_counter()-t)*1000
    h=observe(h,np.random.default_rng(seed),snr_db=snr)
    zone=min(2,int(y//2))*3+min(2,int(x//2))
    center=zone_center(zone)
    supported=True if bundle().get('protocol')==3 else abs(x-center[0])<=CFG['position_jitter_m'][0] and abs(y-center[1])<=CFG['position_jitter_m'][1]

    return package(h,dict(zone=zone,xy=[x,y],id='fresh_scene',split='fresh_simulation',near_boundary=bool(near_boundary([x,y]))),dict(source='fresh_sionna_ray_trace',simulation_ms=rt_ms,in_training_placement_range=supported,noise_seed=seed,snr_db=snr))

class Handler(BaseHTTPRequestHandler):
    def send(self,value,code=200):
        data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        u=urlparse(self.path)
        if u.path=='/api/state':
            p=profile()[1]/'results.json'
            if not p.exists():return self.send(dict(ready=False,progress=json.loads((ROOT/'data/progress.json').read_text()) if (ROOT/'data/progress.json').exists() else {}))
            result=json.loads(p.read_text(encoding='utf-8'))
            if sum(result['split_counts'].values())!=len(records()):return self.send(dict(ready=False,progress=json.loads((ROOT/'data/progress.json').read_text())))
            config=dict(CFG,coverage_mode='full_valid' if result.get('protocol')==3 else 'central_ranges',device_clearance_m=.08,valid_bounds_m=[.61,5.39,.41,5.59])
            return self.send(dict(ready=True,results=result,config=config,records=[r for r in records().values() if r['split'] in ['test','boundary_test','device_shift','material_shift']]))
        if u.path=='/api/download-example':
            row=next(r for r in records().values() if r['split']=='test')
            with np.load(profile()[0]/'records'/(row['id']+'.npz'),allow_pickle=False) as a:h=a['h']

            data=json.dumps(dict(real=h.real.tolist(),imag=h.imag.tolist())).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Disposition','attachment; filename="signal_only.json"');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
        assets={'/':'index.html','/index.html':'index.html','/app.js':'app.js','/style.css':'style.css'}
        if u.path not in assets:return self.send({'error':'Not found'},404)
        p=ROOT/'web'/assets[u.path]
        data=p.read_bytes();self.send_response(200);self.send_header('Content-Type',{'html':'text/html; charset=utf-8','js':'text/javascript; charset=utf-8','css':'text/css; charset=utf-8'}[p.suffix[1:]]);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_POST(self):
        try:
            route=urlparse(self.path).path
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=256000:raise ValueError('Invalid body size.')
            d=json.loads(self.rfile.read(n))
            if not isinstance(d,dict):raise ValueError('JSON object required.')
            seed=int(d.get('seed',42));snr=d.get('snr_db')
            if snr is not None:
                snr=float(snr)
                if not np.isfinite(snr) or not -10<=snr<=60:raise ValueError('SNR must be -10 to 60 dB.')
            if route=='/api/record':return self.send(record_prediction(str(d.get('id','')),seed,snr))
            if route=='/api/simulate':return self.send(simulate_prediction(float(d['x']),float(d['y']),seed,snr))
            if route=='/api/predict':
                if set(d)!={'real','imag'}:raise ValueError('Only real and imag signal arrays are accepted. Coordinates/labels are forbidden.')
                real=np.asarray(d['real'],dtype=np.float32);imag=np.asarray(d['imag'],dtype=np.float32)
                if real.shape!=(12,64) or imag.shape!=(12,64):raise ValueError('Expected real and imag, each [12,64].')
                if not np.isfinite(real).all() or not np.isfinite(imag).all():raise ValueError('Non-finite signal.')
                return self.send(package(real+1j*imag,extra=dict(source='uploaded_signal_only',truth=None)))
            return self.send({'error':'Not found'},404)
        except (ValueError,KeyError,TypeError,OverflowError) as e:return self.send(dict(error=str(e),prediction=None),400)
        except Exception as e:
            import traceback;traceback.print_exc()
            return self.send(dict(error=str(e),prediction=None),500)
    def log_message(self,*args):pass

class LocalHTTPServer(ThreadingHTTPServer):

    allow_reuse_address=False
    def server_bind(self):
        if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8766);args=parser.parse_args()
    print(f'Static obstacle simulation demo: http://127.0.0.1:{args.port}',flush=True)
    try:server=LocalHTTPServer(('127.0.0.1',args.port),Handler)
    except OSError:
        import urllib.request
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/state',timeout=3) as f:existing=json.load(f)
            if existing.get('config',{}).get('version')==CFG['version']:
                print('This demo is already running. Use the existing browser page. / 已有服务运行，直接使用页面即可。',flush=True)
                raise SystemExit(0)
        except (OSError,ValueError):pass
        raise
    server.serve_forever()
