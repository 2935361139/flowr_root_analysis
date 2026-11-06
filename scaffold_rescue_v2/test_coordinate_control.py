import importlib.util, sys, unittest
from pathlib import Path
import numpy as np
# Run these tests from the real project after installation, or provide project root.
root=Path(__file__).resolve().parent
if len(sys.argv)>1 and sys.argv[1]=='--repo':
    sys.path.insert(0,str(Path(sys.argv[2]).resolve()));del sys.argv[1:3]
import scaffold_study_v1
scaffold_study_v1.__path__.insert(0, str(root/'scaffold_study_v1'))
spec=importlib.util.spec_from_file_location('control_under_test',root/'scaffold_study_v1/terminal_control.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class CoordinateControl(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[10,20,30],[11,21,32],[12,20,30]],dtype=np.float32)
        self.goal=self.x.copy();self.goal[0]+=[.2,-.3,.4];self.goal[2]+=[-.4,.2,.3]
    def test_restores_exactly_selected(self):
        y,r=m.coordinate_arm(self.x,self.goal,[0,2],'restore_fixed',4)
        np.testing.assert_array_equal(y[[0,2]],self.goal[[0,2]])
        np.testing.assert_array_equal(y[1],self.x[1])
    def test_input_not_mutated(self):
        before=self.x.copy();g=self.goal.copy();m.coordinate_arm(self.x,self.goal,[0,2],'random_direction',4)
        np.testing.assert_array_equal(self.x,before);np.testing.assert_array_equal(self.goal,g)
    def test_per_atom_norm_and_orthogonality(self):
        for seed in range(30):
            y,r=m.coordinate_arm(self.x,self.goal,[0,2],'random_direction',seed)
            d=y[[0,2]].astype(float)-self.x[[0,2]];target=self.goal[[0,2]].astype(float)-self.x[[0,2]]
            np.testing.assert_allclose(np.linalg.norm(d,axis=1),np.linalg.norm(target,axis=1),atol=5e-6,rtol=0)
            np.testing.assert_allclose(np.sum(d*target,axis=1),0,atol=5e-6,rtol=0)
            np.testing.assert_array_equal(y[1],self.x[1])
    def test_repeatable_local_control(self):
        a,_=m.coordinate_arm(self.x,self.goal,[0,2],'random_direction',4)
        b,_=m.coordinate_arm(self.x,self.goal,[0,2],'random_direction',4)
        np.testing.assert_array_equal(a,b)
    def test_no_global_rng_mutation(self):
        np.random.seed(9);state=np.random.get_state();m.coordinate_arm(self.x,self.goal,[0,2],'random_direction',4)
        other=np.random.get_state();np.testing.assert_array_equal(state[1],other[1]);self.assertEqual(state[2:],other[2:])
    def test_zero_change_is_uninformative(self):
        y,r=m.coordinate_arm(self.x,self.x,[0,2],'random_direction',3)
        np.testing.assert_array_equal(y,self.x);self.assertEqual(r['informative'],[False,False])
    def test_rejects_nonfinite(self):
        self.goal[0,0]=np.nan
        with self.assertRaises(ValueError):m.coordinate_arm(self.x,self.goal,[0,2],'restore_fixed',4)
    def test_rejects_bad_selections(self):
        for ids in [[],[0,0],[-1],[3]]:
            with self.assertRaises(ValueError):m.coordinate_arm(self.x,self.goal,ids,'restore_fixed',4)
    def test_float64_supported(self):
        y,_=m.coordinate_arm(self.x.astype(float),self.goal.astype(float),[0,2],'random_direction',2)
        self.assertEqual(y.dtype,np.float64)
    def test_baseline_gate_rejects_changed_check(self):
        expected={k:True for k in m.BASELINE_LABELS};actual=expected.copy()
        actual['posebusters_checks']={'water':False}
        with self.assertRaises(ValueError):m.compare_endpoint(actual,expected)
    def test_unknown_not_equal_to_false(self):
        expected={k:True for k in m.BASELINE_LABELS};actual=expected.copy()
        expected['posebusters_valid']=False;actual['posebusters_valid']=None
        with self.assertRaises(ValueError):m.compare_endpoint(actual,expected)
    def test_baseline_gate_rejects_coordinate_drift(self):
        expected={k:True for k in m.BASELINE_LABELS};actual=expected.copy()
        expected['fixed_max_displacement_A']=0.;actual['fixed_max_displacement_A']=.1
        with self.assertRaises(ValueError):m.compare_endpoint(actual,expected)
    def test_rejects_unknown_arm(self):
        with self.assertRaises(ValueError):m.coordinate_arm(self.x,self.goal,[0,2],'water_delete',2)
if __name__=='__main__':unittest.main()
