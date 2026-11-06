"""Software checks, including real RDKit UFF. No user data or model required."""
import importlib.util
import unittest
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('geometry_under_test',Path(__file__).parent/'scaffold_study_v1/geometry_intervention.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


class GeometryTests(unittest.TestCase):
    def test_repulsion_gradient(self):
        x=np.array([[1.,.2,.3],[.2,2.,.4]])
        env=np.array([[0.,0.,0.],[1.,1.,2.]])
        cutoff=np.full((2,2),2.4)
        e,grad=g.repulsion(x,env,cutoff,23.)
        for i in range(2):
            for j in range(3):
                plus=x.copy();minus=x.copy();plus[i,j]+=1e-6;minus[i,j]-=1e-6
                numeric=(g.repulsion(plus,env,cutoff,23.)[0]-g.repulsion(minus,env,cutoff,23.)[0])/2e-6
                self.assertAlmostEqual(grad[i,j],numeric,places=5)
        self.assertGreater(e,0.)

    def test_empty_environment(self):
        e,grad=g.repulsion(np.ones((3,3)),np.empty((0,3)),np.empty((3,0)),500.)
        self.assertEqual(e,0.);np.testing.assert_array_equal(grad,np.zeros((3,3)))

    def test_coincident_atoms_finite(self):
        e,grad=g.repulsion(np.zeros((1,3)),np.zeros((1,3)),np.ones((1,1)),500.)
        self.assertTrue(np.isfinite(e) and np.isfinite(grad).all())

    def test_no_repulsion_outside_cutoff(self):
        e,grad=g.repulsion(np.ones((1,3))*10,np.zeros((1,3)),np.ones((1,1)),500.)
        self.assertEqual(e,0.);self.assertFalse(grad.any())

    def test_hard_budget(self):
        count=[0]
        def fun(x):
            count[0]+=1
            return float(np.sum((x-1)**2)),2*(x-1)
        x,r=g.bounded_minimize(fun,np.array([5.]),[(-10.,10.)],80,1)
        self.assertEqual(count[0],1);self.assertEqual(r['objective_calls'],1)
        self.assertFalse(r['converged']);np.testing.assert_array_equal(x,[5.])

    def test_optimization_bounds(self):
        x,r=g.bounded_minimize(lambda x:(float(np.sum((x-10)**2)),2*(x-10)),
                               np.array([0.]),[(-1.,1.)],5,20)
        self.assertLessEqual(x[0],1.);self.assertLessEqual(r['iterations'],5)

    def test_nonfinite_stops(self):
        with self.assertRaises(ValueError):
            g.bounded_minimize(lambda x:(float('nan'),x),np.ones(1),[(-2.,2.)],2,2)

    def test_real_uff_preserves_fixed_and_input(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        h=Chem.AddHs(Chem.MolFromSmiles('CCCO'));self.assertEqual(AllChem.EmbedMolecule(h,randomSeed=42),0)
        mol=Chem.RemoveHs(h);before=mol.GetConformer().GetPositions().copy()
        terms=[dict(coords=np.array([before[3]+[.6,0,0]]),cutoff=np.full((4,1),2.3))]
        result,r=g.relax_generated(mol,[0,1],terms,True)
        np.testing.assert_array_equal(result[:2],before[:2])
        np.testing.assert_array_equal(mol.GetConformer().GetPositions(),before)
        self.assertLessEqual(r['objective_calls'],200);self.assertLessEqual(r['iterations'],80)
        self.assertLessEqual(np.max(np.abs(result-before)),2.+1e-9)
        self.assertGreater(np.linalg.norm(result[3]-terms[0]['coords'][0]),.6)

    def test_uff_only_ignores_environment(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        h=Chem.AddHs(Chem.MolFromSmiles('CCO'));AllChem.EmbedMolecule(h,randomSeed=1)
        mol=Chem.RemoveHs(h)
        a,_=g.relax_generated(mol,[0],[],False)
        b,_=g.relax_generated(mol,[0],[dict(coords=np.zeros((1,3)),cutoff=np.ones((3,1))*100)],False)
        np.testing.assert_array_equal(a,b)


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.ok={k:True for k in ('rdkit_valid','fc_valid','root_fixed_substructure_match','fixed_chemistry_preserved','posebusters_valid')}
        self.bad={**self.ok,'posebusters_valid':False}

    def test_two_independent_causes(self):
        r=g.classify_rescue(self.bad,{'A':self.ok,'B':self.ok,'AB':self.ok})
        self.assertEqual(r['independently_effective_candidates'],['A','B']);self.assertFalse(r['combination_only'])

    def test_joint_only_not_two_independent(self):
        r=g.classify_rescue(self.bad,{'A':self.bad,'B':self.bad,'AB':self.ok})
        self.assertEqual(r['independently_effective_candidates'],[]);self.assertTrue(r['combination_only'])

    def test_partial_pair_repair_is_not_success(self):
        self.assertFalse(g.successful(self.bad))
        for key in self.ok:
            self.assertFalse(g.successful({**self.ok,key:False}))
            self.assertFalse(g.successful({**self.ok,key:None}))

    def test_passing_control_not_rescue(self):
        r=g.classify_rescue(self.ok,{'A':self.ok,'B':self.ok,'AB':self.ok})
        self.assertEqual(r['independently_effective_candidates'],[]);self.assertFalse(r['combination_success'])

    def test_no_success_is_recorded(self):
        r=g.classify_rescue(self.bad,{'A':self.bad,'B':self.bad,'AB':self.bad})
        self.assertEqual(r['independently_effective_candidates'],[]);self.assertFalse(r['combination_success'])

if __name__=='__main__':unittest.main()
