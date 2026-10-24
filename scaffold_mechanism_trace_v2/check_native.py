"""Regression checks using real PyTorch; optional original head weight fixture."""
from pathlib import Path
from types import SimpleNamespace
import os
import unittest
import numpy as np
import torch
from native_readout import HeadCapture, validate_head, native_head, evaluate_coalitions, pair_values


class NativeChecks(unittest.TestCase):
    def model(self):
        torch.manual_seed(104)
        mlp=torch.nn.Sequential(torch.nn.Linear(258,512),torch.nn.SiLU(),torch.nn.Linear(512,128))
        dec=SimpleNamespace(bond_refine=SimpleNamespace(message_mlp=mlp),bond_proj=torch.nn.Linear(128,5))
        fixture=os.environ.get('TRACE_TEST_WEIGHTS')
        if fixture:
            with np.load(fixture,allow_pickle=False) as w,torch.no_grad():
                for name,param in [('w0',mlp[0].weight),('b0',mlp[0].bias),('w2',mlp[2].weight),
                                   ('b2',mlp[2].bias),('wp',dec.bond_proj.weight),('bp',dec.bond_proj.bias)]:
                    param.copy_(torch.from_numpy(w[name]).to(dtype=param.dtype))
        return dec

    def state(self,dec,n=29,seed=8):
        generator=torch.Generator().manual_seed(seed)
        x=torch.randn((1,n,n,258),generator=generator)
        adj=torch.ones((1,n,n),dtype=torch.long)
        adj[:,-1,:]=0;adj[:,:,-1]=0
        return dict(features=x,adjacency=adj,network_bonds=native_head(dec,x,adj,torch))

    def test_full_native_capture_and_each_stage(self):
        dec=self.model();state=self.state(dec)
        observer=HeadCapture(dec,torch)
        out=native_head(dec,state['features'],state['adjacency'],torch)
        observer.close()
        result=validate_head(dec,observer,out,state['adjacency'],torch)
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['stages']),7)
        self.assertTrue(all(s['max_abs_difference']==0 for s in result['stages'].values()))

    def test_detects_real_projection_mismatch(self):
        dec=self.model();state=self.state(dec,n=11)
        observer=HeadCapture(dec,torch)
        out=native_head(dec,state['features'],state['adjacency'],torch)
        observer.close()
        with torch.no_grad():dec.bond_proj.bias.add_(.1)
        result=validate_head(dec,observer,out,state['adjacency'],torch)
        self.assertFalse(result['passed'])
        self.assertTrue(result['stages']['mlp_output']['passed'])
        self.assertFalse(result['stages']['projection_output']['passed'])

    def test_all_16_coalitions_and_rng(self):
        dec=self.model();a,b=self.state(dec,n=36,seed=5),self.state(dec,n=36,seed=9)
        pairs=np.array([[7,2],[7,14],[7,23]])
        rows=[]
        val,checks,extra=evaluate_coalitions(dec,a,b,pairs,64,torch,callback=lambda c,z:rows.append(c))
        self.assertEqual(rows,list(range(16)))
        self.assertTrue(checks['previous_pair_logits']['passed'] and checks['after_full_logits']['passed'])
        self.assertTrue(checks['torch_rng_unchanged'])
        np.testing.assert_allclose(val['order_phi'].sum(0),val['expected_order'][15]-val['expected_order'][0],atol=1e-8)

    def test_no_change_exactly_zero_with_live_head(self):
        dec=self.model();a=self.state(dec,n=11)
        val,checks,_=evaluate_coalitions(dec,a,a,np.array([[7,2],[7,4]]),64,torch)
        np.testing.assert_array_equal(val['order_phi'],np.zeros((4,2)))
        self.assertTrue(checks['after_full_logits']['passed'])

    def test_unselected_logits_unchanged(self):
        dec=self.model();a,b=self.state(dec,n=11,seed=1),self.state(dec,n=11,seed=3)
        pairs=np.array([[7,2],[7,4]])
        def check(c,z):
            np.testing.assert_array_equal(z[0,0,1].detach().numpy(),b['network_bonds'][0,0,1].detach().numpy())
            return c
        _,_,rows=evaluate_coalitions(dec,a,b,pairs,64,torch,callback=check)
        self.assertEqual(len(rows),16)

    def test_changed_masks_are_rejected(self):
        dec=self.model();a,b=self.state(dec,n=11,seed=1),self.state(dec,n=11,seed=3)
        b['adjacency'][0,2,7]=0
        with self.assertRaisesRegex(RuntimeError,'Node-mask adjacency'):
            evaluate_coalitions(dec,a,b,np.array([[7,2]]),64,torch)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
