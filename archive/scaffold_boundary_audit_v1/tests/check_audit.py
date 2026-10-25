"""CPU software tests; actual RDKit self-tests run in the user's original environment."""
from pathlib import Path
import importlib.util
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('audit',Path(__file__).resolve().parents[1]/'audit_saved_boundaries.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
TOKENS=['<PAD>','C','O','N','S','<NOATOM>']
CHARGES=['<PAD>',0,1,2,3,-1,-2,-3]


def data(symbols,edges,charges=None):
    n=len(symbols)
    atom=np.zeros((1,n,len(TOKENS)));q=np.zeros((1,n,len(CHARGES)))
    b=np.zeros((1,n,n,5));b[:,:,:,0]=1
    for i,x in enumerate(symbols):atom[0,i,TOKENS.index(x)]=1
    for i,x in enumerate(charges or [0]*n):q[0,i,CHARGES.index(x)]=1
    for i,j,k in edges:b[0,max(i,j),min(i,j)]=np.eye(5)[k]
    return {'coords':np.zeros((1,n,3)), 'mask':np.ones((1,n),bool),'atomics':atom,'charges':q,'bonds':b}


def profile(fixed,elements,bonds=None):
    return {'fixed':fixed,'anchors':set(fixed),
            'expected':{i:{'element':elements[i],'charge':0,'reference_index':i+10} for i in fixed},
            'expected_bonds':bonds or {},'pairs':[]}


class Tests(unittest.TestCase):
    def test_native_lower_triangle_and_virtual_removal(self):
        d=data(['C','O','C','<NOATOM>'],[(0,1,1),(1,2,1),(1,3,3)])
        d['bonds'][0,0,1]=np.eye(5)[3]  # contradicting UPPER triangle must be ignored
        g=a.graph(d,TOKENS,CHARGES)
        self.assertEqual(g['slots'],[0,1,2]);self.assertEqual(g['classes'][0,1],1)
        self.assertEqual(g['classes'][1,3],0)
        self.assertEqual(float(a.ORDERS[g['classes'][1]].sum()),2)

    def test_bad_mask_or_nonfinite_data_is_not_silent(self):
        d=data(['C','O','C'],[]);d['mask'][0]=[True,False,True]
        with self.assertRaisesRegex(RuntimeError,'prefix mask'):a.graph(d,TOKENS,CHARGES)
        d=data(['C','O','C'],[]);d['bonds'][0,1,0,0]=np.nan
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):a.graph(d,TOKENS,CHARGES)

    def test_charge_and_all_fixed_sites_not_carbon_only(self):
        d=data(['C','O','N','S','C'],[(0,1,1),(1,2,1),(1,3,1)],charges=[0,0,1,0,0])
        g=a.graph(d,TOKENS,CHARGES);p=profile([1,2,3],g['elements'])
        chem={'problems':[{'type':'AtomValenceException','sampler_slots':[1,3]}]}
        events,sites=a.site_observations(g,p,chem)
        self.assertIn('fixed_chemistry_problem:AtomValenceException:1',events)
        self.assertIn('fixed_chemistry_problem:AtomValenceException:3',events)
        self.assertIn('fixed_charge_changed:2',events)
        self.assertEqual(sites[0]['explicit_bond_order_sum'],3)
        self.assertEqual(g['charges'][2],1)

    def test_internal_bond_changes_at_different_atom_indices(self):
        d=data(['C','C','C','N','C','O'],[(3,4,1),(4,5,2)])
        g=a.graph(d,TOKENS,CHARGES);p=profile([3,4,5],g['elements'],{(3,4):3,(3,5):0,(4,5):2})
        events,_=a.site_observations(g,p,{'problems':[]})
        self.assertEqual(events,['fixed_internal_bond_changed:3:4'])

    def test_first_appearance_is_not_terminal_interval(self):
        rows=[{'call':i,'time':i/10,'events':(['A'] if i in [0,2,3] else [])+(['B'] if i==0 else [])} for i in range(4)]
        timing={x['event']:x for x in a.event_timing(rows)}
        self.assertEqual(timing['A']['first_observed_call'],0)
        self.assertEqual(timing['A']['terminal_contiguous_start_call'],2)
        self.assertIsNone(timing['B']['terminal_contiguous_start_call'])
        with self.assertRaisesRegex(RuntimeError,'Missing'):a.event_timing(rows[1:])

    def test_geometry_translation_and_lower_triangle_probability(self):
        d=data(['C','O','N'],[(1,2,3)])
        reference=np.array([[0.,0,0],[1.,0,0],[2.,0,0]])
        d['coords'][0]=reference+np.array([3.,5.,2.])
        d['bonds'][0,2,1]=[-5,-4,-3,5,-6]
        d['bonds'][0,1,2]=[6,0,0,0,0]
        g=a.graph(d,TOKENS,CHARGES);p=profile([1,2],g['elements']);p['pairs']=[(1,2,3)]
        row=a.pair_metrics(g,{'self_condition':{'coords':d['coords']}},reference,g['coords'],p,2.)[0]
        self.assertEqual(row['relative_vector_error_A'],0)
        self.assertEqual(row['feedback_vs_previous_prediction_max_A'],0)
        self.assertGreater(row['expected_bond_probability'],.99)
        self.assertEqual(row['reference_length_A'],2)

    def test_decode_element_change_and_mapping_guard(self):
        d=data(['<NOATOM>','O','C'],[(1,2,1)]);g=a.graph(d,TOKENS,CHARGES)
        p=profile([1],g['elements'])
        class Atom:
            def GetSymbol(self):return 'C'
            def GetFormalCharge(self):return 0
        class Mol:
            def GetNumAtoms(self):return 2
            def GetConformer(self):return self
            def GetPositions(self):return np.zeros((2,3))
            def GetAtomWithIdx(self,i):return Atom()
        r=a.compare_decoded(g,Mol(),p)
        self.assertEqual(r['fixed_atom_edits'][0]['slot'],1)
        self.assertEqual(r['fixed_atom_edits'][0]['before_decode']['element'],'O')
        g['coords'][1,0]=1
        with self.assertRaisesRegex(RuntimeError,'coordinates/order'):a.compare_decoded(g,Mol(),p)


if __name__=='__main__':unittest.main(verbosity=2)
