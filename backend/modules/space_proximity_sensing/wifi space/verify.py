
import json,unittest,hashlib,urllib.request,urllib.error,time
import numpy as np
import joblib
from learning import ROOT,features,classify,observe
from app import package,profile

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.out=profile();cls.rows=json.loads((cls.data/'manifest.json').read_text());cls.model=joblib.load(cls.out/'model.joblib')
        cls.testrow=next(r for r in cls.rows if r['split']=='test')
        with np.load(cls.data/'records'/(cls.testrow['id']+'.npz'),allow_pickle=False) as a:cls.h=a['h']
    def post(self,path,body):
        req=urllib.request.Request('http://127.0.0.1:8766'+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=60) as r:return json.load(r)
    def test_01_manifest(self):
        self.assertEqual(len(self.rows),len({r['id'] for r in self.rows}))
        self.assertEqual(len(self.rows),len({tuple(r['xy']) for r in self.rows}))
        for r in self.rows:
            self.assertEqual(r['zone'],int(r['xy'][1]//2)*3+int(r['xy'][0]//2))
    def test_02_all_channels_and_no_leakage(self):
        hashes={};hashsets={}
        for r in self.rows:
            with np.load(self.data/'records'/(r['id']+'.npz'),allow_pickle=False) as a:h=a['h']
            self.assertEqual(h.shape,(12,64));self.assertTrue(np.isfinite(h).all());self.assertGreater(np.linalg.norm(h),0)
            sha=hashlib.sha256(h.tobytes()).hexdigest();hashsets.setdefault(r['split'],set()).add(sha);hashes[r['id']]=sha
        for s,hs in hashsets.items():
            if s!='train':self.assertFalse(hs&hashsets['train'],s)
        (self.out/'channel_sha256.json').write_text(json.dumps(hashes,indent=2))
    def test_03_feature_contract(self):
        fn=features
        if self.model.get('protocol')==3:
            from coverage_common import signal_features
            fn=signal_features
        self.assertEqual(fn(self.h,self.model['reference']).shape,(204 if self.model.get('protocol')==3 else 132,))
        for shape in [(12,63),(1,64),(12,64,1)]:
            with self.assertRaises(ValueError):fn(np.ones(shape),self.model['reference'])
    def test_04_nonfinite_and_zero(self):
        for h in [np.full((12,64),np.nan),np.zeros((12,64))]:self.assertIsNone(classify(self.model,h)['prediction'])
    def test_05_determinism(self):
        a=classify(self.model,self.h);b=classify(self.model,self.h)
        self.assertEqual(a['probabilities'],b['probabilities']);self.assertAlmostEqual(sum(a['probabilities']),1)
    def test_06_truth_not_used(self):
        a=package(self.h,{'zone':0,'xy':[1,1]});b=package(self.h,{'zone':8,'xy':[5,5]})
        self.assertEqual(a['probabilities'],b['probabilities']);self.assertEqual(a['prediction'],b['prediction']);self.assertEqual(a['center_m'],b['center_m'])
    def test_07_api_signal_only(self):
        d=self.post('/api/predict',dict(real=self.h.real.tolist(),imag=self.h.imag.tolist()))
        self.assertEqual(d['probabilities'],classify(self.model,self.h)['probabilities']);self.assertIsNone(d['truth'])
    def test_08_api_forbids_truth(self):
        with self.assertRaises(urllib.error.HTTPError) as e:self.post('/api/predict',dict(real=self.h.real.tolist(),imag=self.h.imag.tolist(),zone=8))
        self.assertEqual(e.exception.code,400)
    def test_09_api_forbids_train(self):
        with self.assertRaises(urllib.error.HTTPError):self.post('/api/record',{'id':next(r['id'] for r in self.rows if r['split']=='train')})
    def test_10_record_seed(self):
        d=self.post('/api/record',dict(id=self.testrow['id'],seed=13579,snr_db=30))
        h=observe(self.h,np.random.default_rng(13579),snr_db=30)
        self.assertEqual(d['probabilities'],classify(self.model,h)['probabilities']);self.assertEqual(d['truth']['xy'],self.testrow['xy'])
    def test_11_bad_shape_api(self):
        with self.assertRaises(urllib.error.HTTPError) as e:self.post('/api/predict',dict(real=[[1]],imag=[[0]]))
        self.assertEqual(e.exception.code,400)
    def test_12_new_scene(self):
        xy=[3.1234,2.8765];self.assertNotIn(tuple(xy),{tuple(r['xy']) for r in self.rows})
        d=self.post('/api/simulate',dict(x=xy[0],y=xy[1],seed=42,snr_db=30))
        self.assertEqual(d['source'],'fresh_sionna_ray_trace');self.assertEqual(d['truth']['xy'],xy);self.assertGreater(d['simulation_ms'],0);self.assertEqual(len(d['signal_db']),12)
    def test_13_invalid_scene(self):
        with self.assertRaises(urllib.error.HTTPError) as e:self.post('/api/simulate',dict(x=-1,y=2))
        self.assertEqual(e.exception.code,400)
    def test_14_report_counts(self):
        r=json.loads((self.out/'results.json').read_text())
        self.assertEqual(sum(r['split_counts'].values()),len(self.rows));self.assertEqual(r['splits']['test']['n'],sum(a['split']=='test' for a in self.rows))
        self.assertTrue(all(v==0 for v in r['duplicate_train_raw_channels'].values()))
    def test_15_noise_rejection(self):
        clean=classify(self.model,observe(self.h,np.random.default_rng(99),snr_db=30))
        bad=classify(self.model,observe(self.h,np.random.default_rng(99),snr_db=0))
        self.assertIsNotNone(clean['raw_prediction']);self.assertIsNone(bad['prediction'])
    def test_16_recompute_stored_physics(self):
        from simulate import build_scene,response,check_generation_config
        check_generation_config();scene,obj,solver=build_scene();obj.position=[*self.testrow['xy'],.6]
        regenerated=response(scene,solver,17)



        relative=float(np.linalg.norm(regenerated-self.h)/np.linalg.norm(self.h))
        per_link=np.linalg.norm(regenerated-self.h,axis=1)/np.maximum(np.linalg.norm(self.h,axis=1),1e-12)
        self.assertLess(relative,.005);self.assertLess(float(per_link.max()),.02)
        self.assertEqual(classify(self.model,regenerated)['raw_prediction'],classify(self.model,self.h)['raw_prediction'])
        (self.out/'rt_recompute.json').write_text(json.dumps(dict(relative_field_l2=relative,max_link_relative_field_l2=float(per_link.max()),global_limit=.005,link_limit=.02,classification_unchanged=True,note='Approximate GPU RT response agreement, not bitwise determinism. Cold-run audit observed 0.00206 global relative L2 and 0.0786 dB maximum amplitude difference.'),indent=2))
    def test_17_valid_full_coverage_placements(self):
        if self.model.get('protocol')!=3:self.skipTest('Historical central-range experiment')
        from coverage_common import valid_placement
        for r in self.rows:self.assertTrue(valid_placement(r['xy'],r['device_offset_xy']),r['id'])
    def test_18_boundary_test_support(self):
        if self.model.get('protocol')!=3:self.skipTest('Historical central-range experiment')
        from coverage_common import near_boundary
        rows=[r for r in self.rows if r['split']=='boundary_test'];self.assertEqual(len(rows),450)
        self.assertTrue(all(near_boundary(r['xy']) for r in rows))
    def test_19_candidate_policy(self):
        if self.model.get('protocol')!=3:self.skipTest('Historical central-range experiment')
        d=classify(self.model,self.h)
        if d['prediction'] is not None:self.assertEqual(d['candidate_zones'],[d['prediction']])
        self.assertTrue(0<self.model['temperature']<7)
        self.assertTrue(0<self.model['set_cutoff']<=1)
    def test_20_collision_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as e:self.post('/api/simulate',dict(x=.65,y=1))
        self.assertEqual(e.exception.code,400)
    def test_21_weak_evidence_keeps_two_candidates(self):
        if self.model.get('protocol')!=3:self.skipTest('Historical experiment')
        row=next(r for r in self.rows if r['id']=='v3_boundary_Z1_0000')
        from coverage_train import noise_seed
        d=self.post('/api/record',dict(id=row['id'],seed=noise_seed(row['id'])))
        self.assertIsNone(d['prediction']);self.assertEqual(d['reason'],'ambiguous_zones');self.assertEqual(len(d['candidate_zones']),2)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks);start=time.perf_counter();result=unittest.TextTestRunner(verbosity=2).run(suite)
    payload=dict(tests=result.testsRun,passed=result.wasSuccessful(),failures=[(str(t),msg) for t,msg in result.failures],errors=[(str(t),msg) for t,msg in result.errors],elapsed_s=time.perf_counter()-start)
    (profile()[1]/'verification.json').write_text(json.dumps(payload,indent=2))
    raise SystemExit(not result.wasSuccessful())
