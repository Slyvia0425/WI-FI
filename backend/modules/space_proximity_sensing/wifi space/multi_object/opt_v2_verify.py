
import hashlib,json,time,urllib.request,urllib.error,unittest
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from opt_v2_common import *
from opt_v2_infer import infer_v2
from multi_common import validate

def post(path,d):
    q=urllib.request.Request('http://127.0.0.1:8767'+path,data=json.dumps(d).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(q,timeout=120) as f:return json.load(f)

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b=joblib.load(OUT2/'model.joblib');cls.rows=json.loads((DATA2/'manifest.json').read_text());cls.row=next(r for r in cls.rows if r['split']=='test' and r['count']==3)
        with np.load(DATA2/'records'/(cls.row['id']+'.npz')) as a:cls.h=a['h']
        cls.obs=window(cls.h,np.random.default_rng(123),30)
    def test_01_all_records_unique_and_valid(self):
        counts={};hashes={};descriptions=set()
        for folder,rows in [(DATA,json.loads((DATA/'manifest.json').read_text())),(DATA2,self.rows)]:
            for r in rows:
                checked=validate(r['objects']);self.assertEqual(len(checked),r['count']);self.assertEqual([p['zone'] for p in checked],[p['zone'] for p in r['objects']]);description=json.dumps(sorted(checked,key=lambda p:p['kind']),sort_keys=True);self.assertNotIn(description,descriptions);descriptions.add(description)
                with np.load(folder/'records'/(r['id']+'.npz'),allow_pickle=False) as a:h=a['h']
                self.assertEqual(h.shape,(32,64));self.assertTrue(np.isfinite(h).all());self.assertGreater(np.linalg.norm(h),0);digest=hashlib.sha256(h.tobytes()).hexdigest();split=('old_' if folder==DATA else 'new_')+r['split'];hashes.setdefault(split,set()).add(digest);counts[split]=counts.get(split,0)+1
        training=hashes['old_train']|hashes['new_train']
        for split,hs in hashes.items():
            if split not in ['old_train','new_train']:self.assertFalse(training&hs,split)
        self.assertEqual(counts['new_test'],1500);self.assertEqual(counts['new_boundary_test'],450)
        self.__class__.counts=counts
    def test_02_distinct_frames_and_feature_shape(self):
        self.assertEqual(self.obs.shape,(16,32,64));self.assertGreater(np.linalg.norm(self.obs[0]-self.obs[1]),0);self.assertEqual(extract(self.obs,self.b['reference']).shape,(1600,))
        np.testing.assert_allclose(extract(self.obs,self.b['reference']),extract(self.obs.mean(0),self.b['reference']),rtol=1e-5,atol=1e-6)
    def test_03_invalid_signal_unknown(self):
        for h in [np.zeros((16,32,64)),np.full((16,32,64),np.nan)]:self.assertEqual(infer_v2(self.b,h)['status'],'Unknown')
    def test_04_no_count_or_truth_input(self):
        d=post('/api/predict',dict(real=self.obs.real.tolist(),imag=self.obs.imag.tolist()));e=infer_v2(self.b,self.obs);self.assertIsNone(d['truth']);np.testing.assert_allclose(d['occupancy']['probabilities'],e['probabilities'],atol=2e-6);self.assertEqual(d['occupancy']['estimated_region_count'],e['estimated_region_count'])
        with self.assertRaises(urllib.error.HTTPError):post('/api/predict',dict(real=self.obs.real.tolist(),imag=self.obs.imag.tolist(),count=3))
    def test_05_real_record_inference(self):
        d=post('/api/record',dict(id=self.row['id'],seed=123,snr_db=30));e=infer_v2(self.b,self.obs);np.testing.assert_allclose(d['occupancy']['probabilities'],e['probabilities'],atol=2e-6);self.assertEqual(d['frames'],16)
    def test_06_legacy_frame_not_faked(self):
        d=post('/api/predict',dict(real=self.h.real.tolist(),imag=self.h.imag.tolist()));self.assertEqual(d['occupancy']['frames'],1);self.assertNotEqual(d['occupancy']['status'],'Accepted regions')
    def test_07_training_records_forbidden(self):
        with self.assertRaises(urllib.error.HTTPError):post('/api/record',dict(id=next(r['id'] for r in self.rows if r['split']=='train')))
    def test_08_download_roundtrip(self):
        with urllib.request.urlopen('http://127.0.0.1:8767/api/example-json') as f:d=json.load(f)
        self.assertEqual(set(d),{'real','imag'});self.assertEqual(np.asarray(d['real']).shape,(16,32,64));a=post('/api/predict',d);self.assertIsNone(a['truth']);self.assertEqual(a['frames'],16)
    def test_09_decoder_consistency(self):
        e=infer_v2(self.b,self.obs);self.assertEqual(len(e['regions']),e['estimated_region_count']);self.assertTrue(1<=len(e['regions'])<=3);self.assertEqual(len({r['zone'] for r in e['regions']}),len(e['regions']))
    def test_10_truth_attachment_cannot_change_model(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('multi_opt_test_server',ROOT/'app.py');server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
        a=server.package(self.obs,self.row['objects']);b=server.package(self.obs,[dict(kind=1,x=3,y=3,size=[1.1,.8,1.1],zone=4)]);np.testing.assert_array_equal(a['occupancy']['probabilities'],b['occupancy']['probabilities'])
    def test_11_frozen_metrics_match_predictions(self):
        from opt_v2_train import metrics
        report=json.loads((OUT2/'results.json').read_text())
        for split in ['test','boundary_test']:
            with np.load(OUT2/f'predictions_{split}.npz') as a:m=metrics(a['truth'],a['pred'])
            for key in ['occupied_zone_macro_f1','count_accuracy','exact_occupied_layout_accuracy']:self.assertAlmostEqual(m[key],report['splits'][split][key],places=12)
    def test_12_active_state_is_new_test(self):
        with urllib.request.urlopen('http://127.0.0.1:8767/api/state') as f:s=json.load(f)
        self.assertEqual(s['model_version'],2);self.assertEqual(len(s['records']),1950);self.assertTrue(all(r['id'].startswith('v2_') for r in s['records']));self.assertEqual(s['results']['splits']['test']['n'],1500)

if __name__=='__main__':
    with threadpool_limits(limits=4):
        started=time.time();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    (OUT2/'verification.json').write_text(json.dumps(dict(tests=r.testsRun,passed=r.wasSuccessful(),failures=[(str(t),m) for t,m in r.failures],errors=[(str(t),m) for t,m in r.errors],records=getattr(Checks,'counts',None),elapsed_s=time.time()-started),indent=2));raise SystemExit(not r.wasSuccessful())
