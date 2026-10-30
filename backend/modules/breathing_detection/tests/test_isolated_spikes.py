import unittest
from copy import deepcopy
import numpy as np

from backend.modules.breathing_detection.preprocessing.isolated_spikes import (
    isolated_spike_filter,resample_amplitudes,preprocessing_configs,
)
from backend.modules.breathing_detection.evaluation.evaluate_spike_preprocessing import (
    audit_raw,candidate_window,select_preprocessing,
)
from backend.modules.breathing_detection.evaluation.audit_spike_overlap import prefix_suffix_overlap


def source(rate=50,bpm=12):
    t=np.arange(rate*30)/rate
    x=10+np.sin(2*np.pi*bpm/60*t[:,None]+np.arange(4)[None,:]*.1)
    return t,x


class IsolatedSpikeTest(unittest.TestCase):
    def test_single_spike_replaced_readonly_and_without_smoothing(self):
        t,x=source()
        x[400]+=20
        original=x.copy()
        result,stats=isolated_spike_filter(t,x,8)
        self.assertEqual(stats['replaced_cells'],4)
        np.testing.assert_allclose(result[400],(x[399]+x[401])/2)
        np.testing.assert_array_equal(result[:400],x[:400])
        np.testing.assert_array_equal(result[401:],x[401:])
        np.testing.assert_array_equal(x,original)

    def test_sustained_run_and_unbracketed_boundary_preserved(self):
        t,x=source()
        x[400:405]+=20
        x[0]+=20
        result,_=isolated_spike_filter(t,x)
        np.testing.assert_array_equal(result[400:405],x[400:405])
        np.testing.assert_array_equal(result[0],x[0])

    def test_filter_does_not_bridge_large_time_gap(self):
        t,x=source()
        x[400]+=20
        t[400:]+=.2
        result,_=isolated_spike_filter(t,x)
        np.testing.assert_array_equal(result[400],x[400])

    def test_smooth_and_low_rate_are_not_modified(self):
        for bpm in (6,9,12,30):
            t,x=source(bpm=bpm)
            result,stats=isolated_spike_filter(t,x)
            self.assertEqual(stats['replaced_cells'],0)
            np.testing.assert_array_equal(result,x)

    def test_resampler_no_extrapolation_and_correct_layout(self):
        t=np.arange(10)/10
        x=np.stack((2*t+1,3*t+4),axis=1)
        result=resample_amplitudes(t,x,np.array([.05,.15,.25]))
        self.assertEqual(result.shape,(1,2,3))
        np.testing.assert_allclose(result[0,0],[1.1,1.3,1.5])
        with self.assertRaises(ValueError):
            resample_amplitudes(t,x,[-.1,.1])

    def test_bad_inputs_and_unapproved_threshold_rejected(self):
        t,x=source()
        for bad_t,bad_x in ((t[::-1],x),(t,np.full_like(x,np.nan)),(t[:-1],x),(t,x[:,0])):
            with self.assertRaises(ValueError):
                isolated_spike_filter(bad_t,bad_x)
        with self.assertRaises(ValueError):
            isolated_spike_filter(t,x,3)
        self.assertEqual(len(preprocessing_configs()),4)

    def test_placements_produce_900_points_without_extra_tail_dependence(self):
        t,x=source()
        targets=np.arange(900)/30
        current=resample_amplitudes(t,x,targets)
        for config in preprocessing_configs():
            processed,_=candidate_window(t,x,targets,current,config)
            self.assertEqual(processed.shape,(1,4,900))
            np.testing.assert_allclose(processed,current,atol=1e-6)
            # Additional source-file tail after the observation cannot affect it.
            extended_t=np.r_[t,30+np.arange(50)/50]
            extended_x=np.vstack((x,np.full((50,4),1e6)))
            with_tail,_=candidate_window(extended_t,extended_x,targets,current,config)
            np.testing.assert_array_equal(processed,with_tail)

    def test_raw_audit_verifies_complex_abs_duplicates_and_interpolation(self):
        t=np.arange(10)/30
        x=np.ones((10,3),dtype=np.float32)
        x[2:4]=10
        a=audit_raw(t,x,x.astype(complex),t,x.T[None])
        self.assertTrue(a['complex_abs_close'])
        self.assertTrue(a['resampling_matches_loader'])
        self.assertEqual(a['peak_frame_occurrences_in_file'],2)
        self.assertEqual(a['raw_excursion_frames'],2)
        b=audit_raw(t,x,(x+1).astype(complex),t,x.T[None])
        self.assertFalse(b['complex_abs_close'])

    def selection_rows(self,candidate_good=12):
        return [dict(sample_id='good',recording_id='r',reference_bpm=12,baseline_error=0,
                     predictions={'baseline':12,'candidate':candidate_good}),
                dict(sample_id='bad',recording_id='r',reference_bpm=12,baseline_error=18,
                     predictions={'baseline':30,'candidate':12})]

    def test_selection_requires_zero_harm_and_keeps_input_immutable(self):
        rows=self.selection_rows()
        original=deepcopy(rows)
        self.assertEqual(select_preprocessing(rows,['baseline','candidate'])[0],'candidate')
        self.assertEqual(rows,original)
        self.assertEqual(select_preprocessing(self.selection_rows(15),['baseline','candidate'])[0],'baseline')

    def test_missing_candidate_output_counts_as_harm_and_baseline_fallback(self):
        self.assertEqual(select_preprocessing(self.selection_rows(None),['baseline','candidate'])[0],'baseline')

    def test_source_overlap_requires_exact_contiguous_complex_sequence(self):
        a=(np.arange(30).reshape(10,3)+1)*(1+1j)
        b=np.concatenate((a[6:],a[:6]+100))
        self.assertEqual(prefix_suffix_overlap(a,b),(4,6))

    def test_equal_magnitude_with_different_phase_is_not_exact_source_overlap(self):
        a=(np.arange(30).reshape(10,3)+1)*(1+1j)
        b=a*1j
        np.testing.assert_array_equal(abs(a),abs(b))
        self.assertEqual(prefix_suffix_overlap(a,b),(0,None))

    def test_overlap_checks_full_sequence_not_just_a_matching_first_frame(self):
        a=np.array([[0],[1],[0],[2],[3]],dtype=complex)
        b=np.array([[0],[2],[3],[4],[5]],dtype=complex)
        self.assertEqual(prefix_suffix_overlap(a,b),(3,2))
