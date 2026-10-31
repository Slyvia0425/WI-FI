
import sys,json,threading,time,hashlib
from pathlib import Path
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from functools import lru_cache
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse
from multi_common import ROOT,DATA,OUT,CONFIG,NAMES,SIZES,validate,noisy,target
from multi_infer import infer
from app import LocalHTTPServer

THREAD_LIMITER=threadpool_limits(limits=2)
LOCK=threading.Lock();SIM=None;SIM_USES=0
PROFILE_OVERRIDE=None
from collections import OrderedDict
TX_CHANNEL_CACHE=OrderedDict()

def active_profile():
    if PROFILE_OVERRIDE is not None:return PROFILE_OVERRIDE,{1:OUT,2:OUT/'optimized_v2',3:OUT/'overnight',4:OUT/'shared_v4',5:OUT/'home_v5'}[PROFILE_OVERRIDE]
    p=OUT/'active_profile.json'
    if p.exists():
        choice=json.loads(p.read_text(encoding='utf-8'));version=choice.get('version')
        if version not in [1,2,3,4,5]:raise ValueError('Invalid active model version')
        folder={1:OUT,2:OUT/'optimized_v2',3:OUT/'overnight',4:OUT/'shared_v4',5:OUT/'home_v5'}[version]
        if not (folder/('occupancy.joblib' if version==1 else 'model.joblib')).exists():raise RuntimeError('Active model missing')
        return version,folder
    return (2,OUT/'optimized_v2') if (OUT/'optimized_v2'/'ACTIVE').exists() else (1,OUT)

def optimized():return active_profile()[0]>=2
def occupancy_path():
    version,folder=active_profile();return folder/('model.joblib' if version>=2 else 'occupancy.joblib')
def observation(h,rng,snr=30):
    if active_profile()[0]==5:
        from home_v5_window import window
        return window(h,rng,snr)
    if optimized():
        from opt_v2_common import window
        return window(h,rng,snr)
    return noisy(h,rng,snr)
def record_path(row):
    if row['id'].startswith('v6_'):return DATA/'spatial_v6'/row['layout_id']/'records'/(row['raw_id']+'.npz')
    return DATA/('layout_t1_v3' if row['id'].startswith('t3_') else 'layout_t1_v1' if row['id'].startswith(('t1_','t3_')) else 'home_v5' if row['id'].startswith('h5_') else 'shared_v4' if row['id'].startswith('s4_') else 'overnight_v3' if row['id'].startswith('v3_') else 'optimized_v2' if row['id'].startswith('v2_') else '')/'records'/(row['id']+'.npz')

def active_record(row,version=None):
    if version is None:version=active_profile()[0]
    if version==5:
        from spatial_v6_registry import approved as v6_approved
        lid=row.get('layout_id','t1_v1' if row['id'].startswith(('t1_','t3_')) else 'default')
        if v6_approved(lid):return row['id'].startswith('v6_')
        return row['id'].startswith(('h5_','t1_','t3_'))
    return row['id'].startswith('s4_') if version==4 else row['id'].startswith('v3_') if version==3 else row['id'].startswith('v2_') if version==2 else not row['id'].startswith(('v2_','v3_','s4_'))

def runtime_config():
    version,_=active_profile()
    extra={}
    if version==5:
        from home_v5_physics import SPEC
        from layout_registry import entries
        extra=dict(available_layouts=entries(),home_environment=True,fixed_objects=SPEC['fixed_objects'],background_in_target_count=False,physics=SPEC)
    return dict(CONFIG,**extra,supports_shared_zones=version>=4,separate_object_count=version>=4,legacy_size_enabled=version<4,notes='Shared center zones allowed; up to one object per size band. Objects remain physically separated from each other, walls and radio clearances.' if version>=4 else CONFIG['notes'])

@lru_cache(maxsize=2)
def _bundle(stamp):return joblib.load(OUT/'model.joblib')
def bundle():return _bundle((OUT/'model.joblib').stat().st_mtime_ns)
@lru_cache(maxsize=2)
def _occupancy_bundle(path,stamp):return joblib.load(path)
def occupancy_bundle():
    p=occupancy_path();return _occupancy_bundle(str(p),p.stat().st_mtime_ns)
@lru_cache(maxsize=2)
def _catalog(version):
    if version==5:return {r['id']:r for r in json.loads((DATA/'home_v5'/'manifest.json').read_text(encoding='utf-8'))}
    rows=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    if version>=2:rows+=json.loads((DATA/'optimized_v2'/'manifest.json').read_text(encoding='utf-8'))
    if version>=3:rows+=json.loads((DATA/'overnight_v3'/'manifest.json').read_text(encoding='utf-8'))
    if version>=4:rows+=json.loads((DATA/'shared_v4'/'manifest.json').read_text(encoding='utf-8'))
    return {r['id']:r for r in rows}
def catalog():
    rows=_catalog(active_profile()[0])
    if active_profile()[0]==5:
        from layout_registry import catalog as extra_catalog
        return dict(rows,**extra_catalog())
    return rows

def package(h,truth=None,extra=None,selected_bundle=None):
    from opt_v2_common import collapse
    average,frames=collapse(h)
    version,_=active_profile()
    d=infer(bundle(),average) if version<4 else dict(status='Unknown',reason='legacy_size_model_not_enabled_for_shared_zones',objects=[],heads=[],estimated_count=None,confirmed_count=0,elapsed_ms=0.)
    d['truth']=truth;d['frames']=frames;d['inference_model_revision']=selected_bundle.get('revision','t1_v1') if selected_bundle is not None else 'default_v'+str(version)
    if version>=4 or (OUT/'occupancy.joblib').exists():
        from occupancy import infer_occupancy
        if selected_bundle is not None:
            if selected_bundle.get('ordinal_refiner'):
                from spatial_v6_infer import infer_v5 as infer_layout
            elif selected_bundle.get('count_refiner'):
                from layout_t1_v3_infer import infer_v5 as infer_layout
            else:
                from layout_t1_v1_infer import infer_v5 as infer_layout
            d['occupancy']=infer_layout(selected_bundle,h)
        else:d['occupancy']=infer_occupancy(occupancy_bundle(),h)
    if version>=4:
        primary=d['occupancy'];d.update(status=primary['status'],reason=primary['reason'],estimated_count=primary['estimated_object_count'],confirmed_count=primary['confirmed_objects'],elapsed_ms=primary['elapsed_ms'],output_kind='zone_object_counts',legacy_size_enabled=False)
    reference=(selected_bundle or occupancy_bundle())['reference'] if version>=4 else bundle()['reference']
    d['link_delta_db']=(20*np.log10(np.maximum(np.abs(average),1e-10))-20*np.log10(np.maximum(np.abs(reference),1e-10))).mean(1).tolist()
    if truth is not None:
        actual=target(truth);raw=np.array([a['raw_label'] for a in d['heads']]) if d['heads'] else np.zeros(3,int)
        d['exact_layout_correct']=bool(np.all(actual==raw));d['true_count']=len(truth)
        if version>=4:
            occ=d['occupancy'];actual_counts=np.bincount([o['zone'] for o in truth],minlength=9);d['exact_layout_correct']=bool(np.array_equal(actual_counts,occ.get('per_zone_object_count',[])))
            d['true_region_count']=len({o['zone'] for o in truth})
        for o in d['objects']:
            match=next((p for p in truth if p['kind']==o['kind']),None);o['correct_zone_and_size_band']=bool(match is not None and match['zone']==o['zone'])
    d.update(extra or {});return d

def simulate(objects,seed=42,snr_db=30,tx_positions=None):
    global SIM,SIM_USES
    if active_profile()[0]==5:
        from tx_simulation import positions,check
        tx_positions=positions(tx_positions);ps=check(objects,tx_positions)
    else:ps=validate(objects,allow_shared_zones=active_profile()[0]>=4)
    with LOCK:
        from multi_simulate import place
        if active_profile()[0]==5:from home_v5_physics import build,channel
        else:from multi_simulate import build,channel
        start=time.perf_counter()
        if SIM is None or SIM_USES>=100:SIM=build();SIM_USES=0
        scene,os,solver=SIM
        if active_profile()[0]==5:
            for i,p in enumerate(tx_positions):scene.transmitters[f'tx{i}'].position=p
        cache_key=json.dumps([ps,tx_positions],sort_keys=True);cached=cache_key in TX_CHANNEL_CACHE
        if cached:h=TX_CHANNEL_CACHE[cache_key];TX_CHANNEL_CACHE.move_to_end(cache_key)
        else:
            place(os,ps);h=channel(scene,solver);SIM_USES+=1;TX_CHANNEL_CACHE[cache_key]=h
            if len(TX_CHANNEL_CACHE)>16:TX_CHANNEL_CACHE.popitem(last=False)
        elapsed=(time.perf_counter()-start)*1000
    import hashlib
    from layout_registry import match
    layout_id,selected_bundle=match(tx_positions or CONFIG['tx_m']) if active_profile()[0]==5 else ('default',None)
    moved=active_profile()[0]==5 and not np.allclose(tx_positions,CONFIG['tx_m'],rtol=0,atol=1e-7)
    d=package(observation(h,np.random.default_rng(seed),snr_db),ps,dict(source='cached_joint_ray_trace' if cached else 'fresh_joint_ray_trace',channel_cached=cached,simulation_ms=elapsed,noise_seed=seed,snr_db=snr_db,tx_positions=tx_positions or CONFIG['tx_m'],source_positions_changed=bool(moved),signal_sha256=hashlib.sha256(h.tobytes()).hexdigest(),model_layout_id=layout_id),selected_bundle=selected_bundle)
    if moved and selected_bundle is None:
        o=d['occupancy']
        if o['status']!='Unknown':o['status']='Uncertain';o['reason']='transmitter_positions_not_validated'
        o['confirmed_regions']=0;o['confirmed_objects']=0
        for r in o['regions']:r['accepted']=False
        d.update(status=o['status'],reason=o['reason'],confirmed_count=0,model_position_validated=False)
    else:d['model_position_validated']=True
    return d

class Handler(BaseHTTPRequestHandler):
    def send(self,d,code=200):
        b=json.dumps(d,ensure_ascii=False,allow_nan=False).encode();self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        path=urlparse(self.path).path
        if path=='/api/layouts':
            from layout_registry import entries
            return self.send(dict(layouts=[{k:e[k] for k in ['id','ready','revision']} for e in entries()]))
        if path in ['/api/health','/api/ready']:
            if path=='/api/ready':
                try:
                    from occupancy import infer_occupancy
                    b=occupancy_bundle();warm=np.repeat(np.asarray(b['reference'])[None],16,axis=0) if optimized() else np.asarray(b['reference'])
                    infer_occupancy(b,warm)
                    if active_profile()[0]<4:bundle()
                    if active_profile()[0]==5:
                        from layout_registry import entries,by_id
                        for e in entries():
                            if e['ready']:
                                preset,_=by_id(e['id'])
                                if preset is None:continue
                                if preset.get('ordinal_refiner'):
                                    from spatial_v6_infer import infer_v5 as warm_preset
                                elif preset.get('count_refiner'):
                                    from layout_t1_v3_infer import infer_v5 as warm_preset
                                else:
                                    from layout_t1_v1_infer import infer_v5 as warm_preset
                                warm_preset(preset,np.repeat(np.asarray(preset['reference'])[None],16,axis=0))
                except Exception as e:return self.send(dict(service='roomsense-multi',ready=False,error=str(e)),503)
            from layout_registry import entries
            return self.send(dict(service='roomsense-multi',ready=True,project_id=hashlib.sha256(str(ROOT.resolve()).casefold().encode()).hexdigest()[:16],model_version=active_profile()[0],layout_revisions={e['id']:e['revision'] for e in entries() if e['ready']} if active_profile()[0]==5 else {}))
        if path=='/api/state':
            if active_profile()[0]<4 and (not(OUT/'results.json').exists() or not(OUT/'occupancy_results.json').exists()):return self.send(dict(ready=False,progress=json.loads((OUT/'progress.json').read_text(encoding='utf-8')) if (OUT/'progress.json').exists() else {}))
            version,folder=active_profile();active=version>=2
            auxiliary=dict(selected_model='Legacy size model disabled for shared zones',comparison=[],splits={}) if version>=4 else json.loads((folder/('auxiliary_results.json' if active else 'results.json')).read_text(encoding='utf-8'))
            primary_results=json.loads((folder/('results.json' if active else 'occupancy_results.json')).read_text(encoding='utf-8'))
            if version==5:
                from spatial_v6_registry import evaluation as v6_evaluation
                primary_results=v6_evaluation('default') or primary_results
            return self.send(dict(ready=True,model_version=version,results=auxiliary,occupancy_results=primary_results,config=runtime_config(),records=[dict(id=r['id'],count=r['count'],split=r['split'],layout_id=r.get('layout_id','t1_v1' if r['id'].startswith(('t1_','t3_')) else 'default'),shared=r.get('shared',False),region_count=r.get('region_count',r['count'])) for r in catalog().values() if r['split'] in ['test','boundary_test'] and active_record(r,version)]))
        if path in ['/api/example','/api/example-json']:
            version=active_profile()[0]
            from urllib.parse import parse_qs
            lid=parse_qs(urlparse(self.path).query).get('layout_id',['default'])[0]
            from layout_registry import by_id
            try:by_id(lid)
            except ValueError as e:return self.send(dict(error=str(e)),400)
            if version!=5 and lid!='default':return self.send(dict(error='Layout presets require model v5'),400)
            r=next(r for r in catalog().values() if r['split']=='test' and active_record(r,version) and (version!=5 or r.get('layout_id','t1_v1' if r['id'].startswith(('t1_','t3_')) else 'default')==lid))
            with np.load(record_path(r),allow_pickle=False) as a:h=observation(a['h'],np.random.default_rng(42),30) if optimized() else a['h']
            payload=dict(real=h.real.tolist(),imag=h.imag.tolist())
            if lid!='default':payload['layout_id']=lid
            b=json.dumps(payload).encode();self.send_response(200);self.send_header('Content-Type','application/json')
            if path=='/api/example':self.send_header('Content-Disposition','attachment; filename="multi_signal_only.json"')
            self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);return
        assets={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
        if path not in assets:return self.send({'error':'Not found'},404)
        p=ROOT/'web'/assets[path];b=p.read_bytes();self.send_response(200);self.send_header('Content-Type',{'html':'text/html','js':'text/javascript','css':'text/css'}[p.suffix[1:]]+'; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def do_POST(self):
        try:
            path=urlparse(self.path).path;n=int(self.headers.get('Content-Length',0))
            if not 0<n<8000000:raise ValueError('Invalid request size')
            d=json.loads(self.rfile.read(n))
            if not isinstance(d,dict):raise ValueError('JSON object required')
            seed=int(d.get('seed',42));snr=float(d.get('snr_db',30))
            if not np.isfinite(snr) or not 0<=snr<=50:raise ValueError('SNR must be 0–50 dB')
            if path=='/api/simulate':return self.send(simulate(d['objects'],seed,snr,d.get('tx_positions')))
            if path=='/api/record':
                row=catalog().get(str(d.get('id','')))
                if row is None or row['split'] not in ['test','boundary_test']:raise ValueError('Choose a held-out test scene')
                with np.load(record_path(row),allow_pickle=False) as a:h=a['h']
                from layout_registry import by_id
                lid=row.get('layout_id','t1_v1' if row['id'].startswith(('t1_','t3_')) else 'default');selected,tx=by_id(lid)
                if d.get('layout_id',lid)!=lid:raise ValueError('Record/model layout mismatch')
                return self.send(package(observation(h,np.random.default_rng(seed),snr),row['objects'],dict(source='held_out_multi_scene',id=row['id'],noise_seed=seed,snr_db=snr,model_layout_id=lid,tx_positions=tx,model_position_validated=True),selected_bundle=selected))
            if path=='/api/predict':
                if not {'real','imag'}<=set(d) or not set(d)<={'real','imag','layout_id'}:raise ValueError('Only real/imag signal arrays accepted; no object metadata.')
                real=np.asarray(d['real'],float);imag=np.asarray(d['imag'],float)
                valid_shape=real.shape==(32,64) or (real.ndim==3 and real.shape[1:]==(32,64) and 1<=real.shape[0]<=64)
                if not valid_shape or imag.shape!=real.shape or not np.isfinite(real).all() or not np.isfinite(imag).all():raise ValueError('Expected finite real/imag [32,64] or [frames,32,64], 1–64 frames')
                from layout_registry import by_id
                lid=d.get('layout_id','default');selected,tx=by_id(lid)
                return self.send(package(real+1j*imag,extra=dict(source='uploaded_signal_only',model_layout_id=lid,tx_positions=tx,model_position_validated=True),selected_bundle=selected))
            return self.send({'error':'Not found'},404)
        except (ValueError,TypeError,KeyError,OverflowError,IndexError) as e:return self.send({'error':str(e),'status':'Unknown','objects':[]},400)
        except Exception as e:
            import traceback;traceback.print_exc();return self.send({'error':str(e),'status':'Unknown','objects':[]},500)
    def log_message(self,*args):pass

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8769);parser.add_argument('--model-version',type=int,choices=[1,2,3,4,5]);args=parser.parse_args()
    PROFILE_OVERRIDE=args.model_version
    if not 1024<=args.port<=65535:parser.error('Port must be 1024–65535')
    print(f'Multi-object experiment: http://127.0.0.1:{args.port}/',flush=True)
    try:server=LocalHTTPServer(('127.0.0.1',args.port),Handler)
    except OSError:
        import urllib.request
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/state',timeout=3) as f:existing=json.load(f)
            if existing.get('config',{}).get('version')=='multi-object-v1':
                print('Multi-object service is already running. Use the existing page.',flush=True);raise SystemExit(0)
        except (OSError,ValueError):pass
        raise
    server.serve_forever()
