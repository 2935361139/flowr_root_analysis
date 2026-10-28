"""Requires the installed FLOWR environment. Exercises the real native inpaint operation."""
import copy
import tempfile
import unittest
from pathlib import Path
import hashlib

import torch
from flowr.models.mol_builder import MolBuilder
from scaffold_study_v1.terminal_control import (
    compare_prediction, compare_endpoint, restore_frame, changes, aggregate, EvidenceFiles,
)


def tensors():
    return {'coords': torch.tensor([[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]]),
            'atomics':torch.nn.functional.one_hot(torch.tensor([[1,1,1]]),3).float(),
            'charges':torch.nn.functional.one_hot(torch.tensor([[1,1,1]]),3).float(),
            'bonds':torch.nn.functional.one_hot(torch.zeros(1,3,3,dtype=torch.long),5).float(),
            'mask':torch.ones(1,3,dtype=torch.bool)}


class TerminalControlTests(unittest.TestCase):
    def test_exact_reproduction(self):
        a=tensors();self.assertTrue(all(x['passed'] for x in compare_prediction(a,copy.deepcopy(a)).values()))

    def test_probability_drift_rejected_even_if_argmax_same(self):
        a=tensors();b=copy.deepcopy(a);b['atomics'][0,0,1]-=1e-6
        with self.assertRaises(ValueError):compare_prediction(a,b)

    def test_coordinate_absolute_limit_not_relative(self):
        a=tensors();a['coords']+=1000;b=copy.deepcopy(a);b['coords'][0,0,0]+=.01
        with self.assertRaises(ValueError):compare_prediction(a,b)

    def test_nonfinite_rejected(self):
        a=tensors();a['coords'][0,0,0]=float('nan')
        with self.assertRaises(ValueError):compare_prediction(a,a)

    def test_mask_change_rejected(self):
        a=tensors();b=copy.deepcopy(a);b['mask'][0,1]=False
        with self.assertRaises(ValueError):compare_prediction(a,b)

    def test_native_scale_com_mask_and_no_mutation(self):
        b=MolBuilder.__new__(MolBuilder);p=tensors();saved=copy.deepcopy(p);p['mask'][0,2]=False
        i={'coord_scale':2.,'pocket_com':[torch.tensor([[10.,20.,30.]])]}
        result=restore_frame(b,p,i)
        self.assertTrue(torch.equal(result['coords'][0,0],torch.tensor([12.,24.,36.])))
        self.assertTrue(torch.equal(result['coords'][0,2],torch.zeros(3)))
        self.assertTrue(torch.equal(p['coords'],saved['coords']))

    def test_native_inpaint_includes_global_bond_symmetrization(self):
        b=MolBuilder.__new__(MolBuilder);prior=tensors();prior.update(fragment_mask=torch.tensor([[True,False,False]]),fragment_mode=['scaffold_hopping'])
        pred=tensors();pred['coords']+=5
        # Non-fixed atom pair: asymmetric probabilities. Native code chooses upper triangle.
        pred['bonds'][0,1,2]=torch.tensor([.1,.9,0.,0.,0.])
        original=copy.deepcopy(pred)
        on=b.inpaint_molecule(copy.deepcopy(prior),copy.deepcopy(pred))
        self.assertTrue(torch.equal(on['coords'][0,0],prior['coords'][0,0]))
        self.assertTrue(torch.equal(on['coords'][0,1:],pred['coords'][0,1:]))
        self.assertEqual(int(on['bonds'][0,2,1].argmax()),1)
        self.assertEqual(changes(pred,on,prior)['changed_other_bond_count'],1)
        self.assertGreater(changes(pred,on,prior)['all_bond_probability_entries_changed'],0)
        self.assertTrue(torch.equal(pred['bonds'],original['bonds']))

    def test_endpoint_boolean_mismatch_rejected(self):
        with self.assertRaises(ValueError):compare_endpoint({'rdkit_valid':True},{'rdkit_valid':False})

    def test_unknown_count_not_failed(self):
        row={k:None for k in ('rdkit_valid','fc_valid','fixed_chemistry_preserved','root_fixed_substructure_match','posebusters_valid','murcko_scaffold_changed')}
        arm={s:row for s in ('raw_without_repair','run_decode')}
        result=aggregate([{'off':arm,'on':arm}])['raw_without_repair']['posebusters_valid']
        self.assertEqual(result['on_unknown'],1);self.assertEqual(result['on_fail'],0)

    def test_corrupt_lfs_object_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=b'known';oid=hashlib.sha256(data).hexdigest()
            obj=root/'.git/lfs/objects'/oid[:2]/oid[2:4]/oid;obj.parent.mkdir(parents=True);obj.write_bytes(b'wrong')
            pointer=root/'x.pt';pointer.write_text(f'version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize 5\n')
            with self.assertRaises(ValueError):EvidenceFiles(root).resolve(pointer)


if __name__=='__main__':unittest.main()
