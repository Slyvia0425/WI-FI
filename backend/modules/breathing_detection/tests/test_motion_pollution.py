import unittest
from copy import deepcopy
import numpy as np

from backend.modules.breathing_detection.preprocessing.motion_pollution import (
    FEATURES, pollution_features, pollution_rules, weighted_quantile,
    calibrate_thresholds, pollution_withhold,
)
from backend.modules.breathing_detection.evaluation.evaluate_motion_pollution import select_pollution_rule, assess


def signal():
    t = np.arange(900)/30
    return np.sin(2*np.pi*.3*t[None,None,:]+np.arange(5)[None,:,None]*.1)


def features(**kwargs):
    return {'available':True,**{key:0. for key in FEATURES},**kwargs}


class MotionPollutionTest(unittest.TestCase):
    def test_features_are_finite_scale_invariant_and_readonly(self):
        x = signal()
        copy = x.copy()
        a,b = pollution_features(x),pollution_features(7*x+500)
        for key in FEATURES:
            self.assertTrue(np.isfinite(a[key]))
            self.assertAlmostEqual(a[key],b[key],places=7)
        np.testing.assert_array_equal(x,copy)
        self.assertLess(a['broadband_fraction'],.001)
        self.assertLess(a['adjacent_incoherence'],.02)

    def test_transient_increases_jump_and_short_energy_proxies(self):
        x = signal()
        t = np.arange(900)/30
        polluted = x+10*np.exp(-((t-10.25)/.03)**2)[None,None,:]
        a,b = pollution_features(x),pollution_features(polluted)
        self.assertGreater(b['jump_ratio'],a['jump_ratio']*3)
        self.assertGreater(b['energy_burst_ratio'],a['energy_burst_ratio']*2)

    def test_independent_noise_has_broadband_and_incoherence_without_physical_label(self):
        a = pollution_features(signal())
        b = pollution_features(np.random.default_rng(123).normal(size=(1,5,900)))
        self.assertGreater(b['broadband_fraction'],.5)
        self.assertGreater(b['adjacent_incoherence'],.8)
        self.assertGreater(b['jump_ratio'],a['jump_ratio'])

    def test_negative_correlation_is_coherent_and_flat_pairs_not_bridged(self):
        x = np.tile(signal()[:,:1],(1,5,1))
        x[:,1::2]*=-1
        self.assertLess(pollution_features(x)['adjacent_incoherence'],1e-10)
        x[:,2]=0
        result = pollution_features(x)
        self.assertEqual(result['valid_subcarriers'],4)
        self.assertEqual(result['valid_adjacent_pairs'],2)
        x[:,1]=0
        x[:,3]=0
        with self.assertRaises(ValueError):
            pollution_features(x)

    def test_invalid_or_short_signals_raise(self):
        x = signal()
        for value in (x[0],x[...,:300],x[:,:1],np.zeros_like(x),np.full_like(x,np.nan)):
            with self.assertRaises(ValueError):
                pollution_features(value)
        with self.assertRaises(ValueError):
            pollution_features(x,50)

    def test_record_balancing_changes_quantile_without_using_reference(self):
        rows = [dict(sample_id=str(i),recording_id='many' if i<10 else 'single') for i in range(11)]
        f = {r['sample_id']:features(**{key:0. if i<10 else 10. for key in FEATURES}) for i,r in enumerate(rows)}
        a = calibrate_thresholds(rows,f,.6)
        self.assertTrue(all(v==10 for v in a.values()))
        self.assertEqual(weighted_quantile([0]*10+[10],[1]*11,.6),0)
        modified = [{**r,'reference_bpm':1000,'participant_id':'not_used'} for r in rows]
        f['held_out']=features(**{key:1e10 for key in FEATURES})
        self.assertEqual(calibrate_thresholds(modified,f,.6),a)
        for weights in ([1,0],[1,-1],[1,np.nan]):
            with self.assertRaises(ValueError):
                weighted_quantile([0,1],weights,.9)

    def test_fixed_families_need_conjunction_and_strict_exceedance(self):
        limits = {key:1. for key in FEATURES}
        rules = {r['family']:r for r in pollution_rules() if r['quantile']==.9}
        self.assertEqual(len(pollution_rules()),8)
        self.assertFalse(pollution_withhold(features(jump_ratio=10),rules['transient'],limits))
        self.assertTrue(pollution_withhold(features(jump_ratio=2,energy_burst_ratio=2),rules['transient'],limits))
        self.assertTrue(pollution_withhold(features(broadband_fraction=2,adjacent_incoherence=2),rules['broadband'],limits))
        self.assertFalse(pollution_withhold(features(**limits),rules['either_pair'],limits))
        self.assertFalse(pollution_withhold({'available':False},rules['either_pair'],limits))
        self.assertFalse(pollution_withhold(features(),None,{}))

    def setup_selection(self):
        rows = [dict(sample_id=str(i),participant_id='01' if i<100 else '02',recording_id='a' if i<100 else 'b',
                     reference_bpm=12,predictions={'unified':12 if i<140 else 30}) for i in range(141)]
        f = {r['sample_id']:features() for r in rows}
        rule = pollution_rules()[0]
        limits = {'0.9':{key:1. for key in FEATURES}}
        return rows,f,rule,limits

    def test_safe_validation_gain_selected_and_bpm_not_changed(self):
        rows,f,rule,limits = self.setup_selection()
        f['140']=features(jump_ratio=2,energy_burst_ratio=2)
        original = deepcopy(rows)
        chosen,audit = select_pollution_rule(rows,f,limits,[rule])
        self.assertEqual(chosen,rule)
        self.assertEqual(audit[0]['metrics']['large_withheld'],1)
        outcomes = assess(rows,f,chosen,limits)
        self.assertIsNone(outcomes[-1]['predictions']['withheld'])
        self.assertEqual(outcomes[-1]['predictions']['unified'],30)
        self.assertTrue(all(not r['breathing_status_changed'] for r in outcomes))
        self.assertEqual(rows,original)

    def test_per_person_harm_cannot_hide_in_pooled_average(self):
        rows,f,rule,limits = self.setup_selection()
        for sample_id in ('100','140'):
            f[sample_id]=features(jump_ratio=2,energy_burst_ratio=2)
        chosen,audit = select_pollution_rule(rows,f,limits,[rule])
        self.assertLess(audit[0]['metrics']['good_withholding_rate'],.01)
        self.assertFalse(audit[0]['eligible'])
        self.assertIsNone(chosen)

    def test_no_benefit_prefers_no_gate(self):
        rows,f,rule,limits = self.setup_selection()
        self.assertIsNone(select_pollution_rule(rows,f,limits,[rule])[0])
