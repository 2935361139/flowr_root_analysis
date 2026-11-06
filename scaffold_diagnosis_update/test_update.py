import sys, unittest, tempfile, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scaffold_geometry_audit_v1'))
from temporal import series, suffix
from scaffold_study_v1.core import analyze

class TemporalTests(unittest.TestCase):
    def points(self, distances, classes=None):
        n=len(distances); xyz=np.zeros((n,1,3));xyz[:,0,0]=distances
        return series(xyz,np.array(classes or [0]*n)[:,None],np.ones((n,1),bool),
            ['C','N'],0,'C',np.zeros(3),2.,.75,
            dict(indices=list(range(n)),times=[None]*n,stages=['saved_state']*n))
    def test_recovers_after_early_clash(self):
        w=suffix(self.points([1.,2.,1.,1.]));self.assertEqual(w['first_observed']['index'],0)
        self.assertEqual(w['first_persistent_observation']['index'],2)
        self.assertTrue(w['boundary_is_confirmed_change'])
    def test_element_change_is_unknown(self):
        w=suffix(self.points([1.,1.,1.],[0,1,0]))
        self.assertEqual(w['first_persistent_observation']['index'],2)
        self.assertFalse(w['boundary_is_confirmed_change'])
    def test_no_terminal_clash(self):
        self.assertIsNone(suffix(self.points([1.,2.]))['first_persistent_observation'])
    def test_threshold_uncertainty(self):
        self.assertIsNone(self.points([1.5])[0]['value'])
    def test_coordinate_translation(self):
        p=self.points([1.,2.]);self.assertEqual([x['value'] for x in p],[True,False])
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):self.points([float('nan')])
    def test_unknown_does_not_bridge(self):
        p=self.points([1.,1.5,1.]);self.assertEqual(suffix(p)['first_persistent_observation']['index'],2)
    def test_empty_series_rejected(self):
        with self.assertRaises((IndexError,ValueError)):suffix([])

class StudyTests(unittest.TestCase):
    def dataset(self):
        e=dict(chemical_valid=True,connected=True,fixed_chemistry_preserved=True,
               fixed_substructure_match=True,geometry_valid=False,scaffold_changed=False,
               geometry_failed_checks=['minimum_distance_to_protein'])
        pts=TemporalTests().points([2.,1.])
        timeline=dict(check='minimum_distance_to_protein',stream='network_prediction',entity='slot:1/env:2',
                      signature='final_pair_clash',scope='observation',points=pts)
        r=dict(run_id='a',input_id='b',input_group='c',seed=1,endpoints={'raw':e,'decoded':e},observations=[],
               geometry={'raw':{'clash_pairs':[{'check':'minimum_distance_to_protein'}],'timelines':[timeline]}})
        return dict(schema='scaffold-evidence/1',model='test',runs=[r])
    def test_join_geometry_not_scaffold(self):
        r=analyze(self.dataset())['runs'][0]
        i=next(i for i in r['issues'] if i['problem']=='geometry_failed' and i['endpoint_stage']=='raw')
        self.assertEqual(len(i['temporal_evidence']),1);self.assertIsNone(i['internal_module'])
        for i in r['issues']:
            if i['problem']=='scaffold_unchanged':self.assertEqual(i['temporal_evidence'],[])
    def test_nonconsecutive_rejected(self):
        d=self.dataset();d['runs'][0]['geometry']['raw']['timelines'][0]['points'][1]['index']=3
        with self.assertRaises(ValueError):analyze(d)
    def test_missing_geometry_is_explicit(self):
        d=self.dataset();d['runs'][0]['geometry']={}
        r=analyze(d)['runs'][0]
        self.assertTrue(all(i['localization_status']=='time_and_module_unresolved' for i in r['issues']))

if __name__=='__main__':unittest.main()
