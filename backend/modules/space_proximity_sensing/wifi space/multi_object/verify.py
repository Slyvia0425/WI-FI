
import json,unittest,hashlib,urllib.request,urllib.error,time
import numpy as np
import joblib
from multi_common import ROOT,DATA,OUT,validate,target,features,noisy
from multi_infer import infer
from multi_train import seed

def post(path,d):
    q=urllib.request.Request('http://127.0.0.1:8767'+path,data=json.dumps(d).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(q,timeout=90) as r:return json.load(r)

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=json.loads((DATA/'manifest.json').read_text());cls.b=joblib.load(OUT/'model.joblib');cls.r=next(r for r in cls.rows if r['split']=='test' and r['count']==3)
        with np.load(DATA/'records'/(cls.r['id']+'.npz'),allow_pickle=False) as a:cls.h=a['h']
    def test_01_all_files_and_placements(self):
        hashes={};descriptions=set()
        for r in self.rows:
            ps=validate(r['objects']);self.assertEqual(target(ps).tolist(),r['target']);self.assertEqual(len(ps),r['count'])
            desc=json.dumps(sorted(ps,key=lambda p:p['kind']),sort_keys=True);self.assertNotIn(desc,descriptions);descriptions.add(desc)
            with np.load(DATA/'records'/(r['id']+'.npz'),allow_pickle=False) as a:h=a['h']
            self.assertEqual(h.shape,(32,64));self.assertTrue(np.isfinite(h).all());self.assertGreater(np.linalg.norm(h),0)
            hashes.setdefault(r['split'],set()).add(hashlib.sha256(h.tobytes()).hexdigest())
        for split,values in hashes.items():
            if split!='train':self.assertFalse(values&hashes['train'],split)
    def test_02_features(self):self.assertEqual(features(self.h,self.b['reference']).shape,(544,))
    def test_03_unknown_invalid(self):
        for h in [np.zeros((32,64)),np.full((32,64),np.nan)]:self.assertEqual(infer(self.b,h)['status'],'Unknown')
    def test_04_signal_only_api(self):
        d=post('/api/predict',dict(real=self.h.real.tolist(),imag=self.h.imag.tolist()));expected=infer(self.b,self.h)
        self.assertIsNone(d['truth']);self.assertEqual(d['estimated_count'],expected['estimated_count'])
        for a,b in zip(d['heads'],expected['heads']):np.testing.assert_allclose(a['probabilities'],b['probabilities'],rtol=1e-12,atol=1e-14)
    def test_05_rejects_target_metadata(self):
        with self.assertRaises(urllib.error.HTTPError) as e:post('/api/predict',dict(real=self.h.real.tolist(),imag=self.h.imag.tolist(),objects=self.r['objects']))
        self.assertEqual(e.exception.code,400)
    def test_06_record_is_actual_inference(self):
        if (OUT/'optimized_v2'/'ACTIVE').exists():
            from opt_v2_common import window
            observation=window(self.h,np.random.default_rng(12345),30).mean(0)
        else:observation=noisy(self.h,np.random.default_rng(12345),30)
        d=post('/api/record',dict(id=self.r['id'],seed=12345,snr_db=30));expected=infer(self.b,observation)
        for a,b in zip(d['heads'],expected['heads']):np.testing.assert_allclose(a['probabilities'],b['probabilities'],rtol=1e-12,atol=1e-14)
    def test_07_training_record_forbidden(self):
        with self.assertRaises(urllib.error.HTTPError):post('/api/record',dict(id=next(r['id'] for r in self.rows if r['split']=='train')))
    def test_08_invalid_object_placements(self):
        for ps in [[dict(kind=0,x=-1,y=1)],[dict(kind=0,x=1,y=1),dict(kind=0,x=5,y=5)],[dict(kind=0,x=1,y=1),dict(kind=2,x=1.1,y=1.1)]]:
            with self.assertRaises(urllib.error.HTTPError):post('/api/simulate',dict(objects=ps))
    def test_09_fresh_two_objects(self):
        ps=[dict(kind=0,x=1.12,y=1.18),dict(kind=2,x=4.85,y=4.9)]
        d=post('/api/simulate',dict(objects=ps,seed=42,snr_db=30));self.assertEqual(d['source'],'fresh_joint_ray_trace');self.assertEqual(len(d['truth']),2);self.assertEqual(len(d['heads']),3);self.assertEqual(len(d['link_delta_db']),32)
    def test_10_fresh_three_objects(self):
        ps=[dict(kind=0,x=1.12,y=1.18),dict(kind=1,x=3.05,y=3.12),dict(kind=2,x=4.85,y=4.9)]
        d=post('/api/simulate',dict(objects=ps,seed=43,snr_db=30));self.assertEqual(len(d['truth']),3);self.assertEqual(len(d['heads']),3);self.assertGreater(d['simulation_ms'],0)
    def test_11_truth_does_not_enter_inference(self):

        import importlib.util
        spec=importlib.util.spec_from_file_location('multi_server_for_check',ROOT/'app.py');server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
        a=server.package(self.h,self.r['objects']);b=server.package(self.h,[dict(kind=1,x=3,y=3,size=[1.1,.8,1.1],zone=4)])
        np.testing.assert_allclose([h['probabilities'] for h in a['heads']],[h['probabilities'] for h in b['heads']],rtol=1e-12,atol=1e-14);self.assertEqual(a['estimated_count'],b['estimated_count'])
        np.testing.assert_allclose(a['occupancy']['probabilities'],b['occupancy']['probabilities'],rtol=1e-6,atol=1e-7)
    def test_12_report(self):
        r=json.loads((OUT/'results.json').read_text());self.assertEqual(sum(r['split_counts'].values()),4800);self.assertEqual(r['splits']['test']['n'],900);self.assertEqual(r['feature_dimension'],544)
    def test_13_occupancy_signal_only(self):
        from occupancy import infer_occupancy
        path=OUT/'optimized_v2'/'model.joblib' if (OUT/'optimized_v2'/'ACTIVE').exists() else OUT/'occupancy.joblib'
        b=joblib.load(path);expected=infer_occupancy(b,self.h);d=post('/api/predict',dict(real=self.h.real.tolist(),imag=self.h.imag.tolist()))
        np.testing.assert_allclose(d['occupancy']['probabilities'],expected['probabilities'],rtol=1e-6,atol=1e-7);self.assertEqual(len(expected['probabilities']),9)
    def test_14_independent_variable_meshes(self):
        from multi_simulate import build,place
        scene,os,solver=build();ps=validate([dict(kind=0,x=1,y=1),dict(kind=1,x=3,y=3),dict(kind=2,x=5,y=5)]);place(os,ps)
        for o,p in zip(os,ps):
            bbox=o.mi_mesh.bbox();np.testing.assert_allclose(np.array(bbox.max-bbox.min).ravel(),p['size'],atol=1e-4);np.testing.assert_allclose(np.array(o.position).ravel(),[p['x'],p['y'],p['size'][2]/2],atol=1e-4)
    def test_15_inline_signal_download(self):
        with urllib.request.urlopen('http://127.0.0.1:8767/api/example-json') as response:
            self.assertIsNone(response.headers.get('Content-Disposition'));d=json.load(response)
        expected=(16,32,64) if (OUT/'optimized_v2'/'ACTIVE').exists() else (32,64)
        self.assertEqual(set(d),{'real','imag'});self.assertEqual(np.array(d['real']).shape,expected)

if __name__=='__main__':
    start=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks);r=unittest.TextTestRunner(verbosity=2).run(suite)
    (OUT/'verification.json').write_text(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),failures=[(str(t),m) for t,m in r.failures],errors=[(str(t),m) for t,m in r.errors],elapsed_s=time.perf_counter()-start),indent=2));raise SystemExit(not r.wasSuccessful())
