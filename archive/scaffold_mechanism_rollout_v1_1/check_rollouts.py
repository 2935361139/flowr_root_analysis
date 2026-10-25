"""Software regression tests. Constructed activations/molecules are NOT results."""
from __future__ import annotations
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from tensordict import TensorDict
import native_adapter as native
import replay_support as helpers
from head_patch import HeadPatch,modify_features,selected_values
from native_readout import native_head,evaluate_coalitions,pair_values
from protocol import FAMILIES,select_protocol
from test_native_support import native_parts,Vocab


def rng_state():
    state=np.random.get_state()
    return dict(python=random.getstate(),numpy=(state[0],torch.from_numpy(state[1].copy()),*state[2:]),torch_cpu=torch.get_rng_state().clone())


class TinyNativeSampler(torch.nn.Module):
    """Four-call recurrent fixture using real Torch modules and RDKit builder."""
    def __init__(self,builder):
        super().__init__();self.builder=builder;self.gen=torch.nn.Module()
        self.gen.ligand_dec=torch.nn.Module();dec=self.gen.ligand_dec
        dec.bond_refine=torch.nn.Module();dec.bond_refine.node_proj=torch.nn.Linear(2,2)
        dec.bond_refine.message_mlp=torch.nn.Sequential(torch.nn.Linear(7,1),torch.nn.SiLU(),torch.nn.Linear(1,1))
        dec.bond_proj=torch.nn.Linear(1,5)
        self.gen.get_pocket_encoding=lambda *a,**k:(None,None)
        with torch.no_grad():
            for p in dec.parameters():p.zero_()
            dec.bond_refine.message_mlp[0].weight[0,4]=1
            dec.bond_refine.message_mlp[0].weight[0,6]=1
            dec.bond_refine.message_mlp[2].weight[0,0]=1
            dec.bond_proj.weight[1,0]=1;dec.bond_proj.bias.fill_(-5);dec.bond_proj.bias[0]=.5;dec.bond_proj.bias[1]=0
        self.coord_scale=1.;self.self_condition=True;self.sc_recycle_steps=1
        self.auxiliary_outputs=False

    def forward(self,batch,pocket,t,cond_batch=None,**kwargs):
        call=int(t[0]);n=4;x=torch.zeros((1,n,n,7));adj=torch.ones((1,n,n),dtype=torch.long)
        for a,b in [(0,1),(1,2)]:x[0,a,b,4]=x[0,b,a,4]=2.
        level=.2 if call<2 else .5
        feedback=float(cond_batch['bonds'][0,3,1,1])*.05
        x[0,1,3,4]=x[0,3,1,4]=level+feedback
        x[0,1,3,6]=x[0,3,1,6]=level
        bonds=native_head(self.gen.ligand_dec,x,adj,torch)
        coords=batch['coords']+cond_batch['bonds'][...,1].sum()*1e-5
        out=dict(coords=coords,atomics=batch['atomics'].clone(),charges=batch['charges'].clone(),
                 mask=batch['mask'].clone(),bonds=bonds)
        if self.auxiliary_outputs:
            value=batch['coords'].mean(dim=(1,2)).reshape(-1,1)
            # ROOT uses TensorDict without batch_size for these two outputs.
            out['affinity']=TensorDict({k:value.clone() for k in ['pic50','pkd','pki','pec50']})
            out['docking_score']=TensorDict({k:value.clone() for k in ['vina_score','gnina_score']})
            out['interactions']=None
        return out

    def _build_trajectory(self,curr,predicted,pocket_data,iter=0,step=0):
        curr['res_names']=torch.zeros(curr['atomics'].shape[:2],dtype=torch.long)

    def _save_trajectory(self,*a,**k):pass

    def _generate(self,prior,pocket,steps,times,**kwargs):
        batch=helpers.move_copy(prior,'cpu',torch);cond=helpers.move_copy(prior,'cpu',torch)
        for k in range(steps):
            out=self(batch,pocket,torch.tensor([float(k)]),cond_batch=cond)
            cond=helpers.move_copy(out,'cpu',torch);batch=helpers.move_copy(out,'cpu',torch)
            self._build_trajectory(batch,out,pocket,iter=k,step=k)
        return batch


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo=Path(os.environ.get('TRACE_NATIVE_SOURCE','.')).resolve()
        builder_type,cls.rd,cls.fn,modules=native_parts(repo)
        cls.modules_patch=patch.dict(__import__('sys').modules,modules);cls.modules_patch.start()
        cls.builder=builder_type();cls.builder.vocab=Vocab(['C','O','<NOATOM>']);cls.builder.vocab_charges=Vocab([0,1])

    @classmethod
    def tearDownClass(cls):cls.modules_patch.stop()

    def source(self,auxiliary=False):
        model=TinyNativeSampler(self.builder).eval()
        model.auxiliary_outputs=auxiliary
        prior=dict(coords=torch.tensor([[[0.,0.,0.],[1.4,0.,0.],[1.4,1.4,0.],[1.4,0.,1.4]]]),
              atomics=torch.nn.functional.one_hot(torch.tensor([[0,1,0,0]]),3).float(),
              charges=torch.nn.functional.one_hot(torch.zeros((1,4),dtype=torch.long),2).float(),
              mask=torch.ones((1,4),dtype=torch.long),bonds=torch.zeros((1,4,4,5)))
        inputs=dict(prior=prior,pocket_data={},pocket_com=[],rng=rng_state(),times=[],strategy='test',solver='test',final_inpaint=False,coord_scale=1.)
        calls=[]
        def pre(m,args,kwargs):
            calls.append(dict(call_index=len(calls),state_before_forward=helpers.move_copy(args[0],'cpu',torch),
                              self_condition=helpers.move_copy(kwargs['cond_batch'],'cpu',torch),times=args[2].clone(),
                              times_scalar=float(args[2][0]),rng_before_forward=rng_state()))
        def post(m,args,kwargs,out):calls[-1]['network_output_before_postprocessing']=helpers.move_copy(out,'cpu',torch)
        handles=[model.register_forward_pre_hook(pre,with_kwargs=True),model.register_forward_hook(post,with_kwargs=True)]
        with torch.no_grad():final=model._generate(prior,{},steps=4,times=[])
        for h in handles:h.remove()
        return model,dict(inputs=inputs,calls=calls,final=final,encoding=dict(pocket_equis=None,pocket_invs=None),
             manifest=dict(args=dict(integration_steps=4,corrector_iters=0,save_traj=True,coord_noise_scale=0.)),pairs=np.array([[1,2],[1,3]]))

    def run_one(self,model,data,kind='sham',family='none',source=None,expected=None,index=0):
        trial=dict(seed=7,site=1,call=2,kind=kind,family=family,groups=FAMILIES.get(family,[]),control_seed=35,control_index=index)
        holder={}
        def factory(m,t,c,T):
            holder['patch']=HeadPatch(m,t,c,T,data,source=source,expected=expected);return holder['patch']
        with tempfile.TemporaryDirectory() as td:
            result=native.run_rollout(model,data,trial,factory,helpers,torch.device('cpu'),torch,Path(td))
        return result,holder['patch']

    def test_native_full_sham_replay_is_identical(self):
        m,d=self.source();(final,receipt,logs,sampler),observer=self.run_one(m,d)
        self.assertTrue(receipt['baseline_or_sham_full_match']);self.assertTrue(receipt['final_match']['passed'])
        self.assertEqual(len(receipt['checked_calls']),4);self.assertEqual(len(logs),4);self.assertEqual(len(sampler),4)
        self.assertFalse(observer.receipt['applied']);self.assertTrue(logs[2]['site_valence_problem'])

    def test_targeted_native_rollout_is_free_after_write(self):
        m,d=self.source();(_,sham,_,_),observer=self.run_one(m,d)
        source={k:observer.capture[k] for k in ['before','after']}
        (final,receipt,logs,_),target=self.run_one(m,d,'targeted','geometry',source)
        self.assertTrue(receipt['random_streams_matched_all_calls']);self.assertEqual([r['call'] for r in receipt['checked_calls']],[0,1])
        self.assertFalse(logs[2]['site_valence_problem']);self.assertFalse(logs[2]['connected'])
        self.assertFalse(torch.equal(final['coords'],d['final']['coords']))
        self.assertTrue(target.receipt['nonbond_outputs_unchanged_at_intervention'])
        self.assertEqual(target.visits,1)

    def test_random_control_native_rollout_keeps_rng_and_norm(self):
        m,d=self.source();(_,_,_,_),observer=self.run_one(m,d)
        source={k:observer.capture[k] for k in ['before','after']}
        (_,receipt,_,_),random_patch=self.run_one(m,d,'random_direction','geometry_and_edge',source)
        self.assertTrue(receipt['random_streams_matched_all_calls'])
        for g in random_patch.receipt['per_group']:
            self.assertTrue(g['magnitude_matched']);self.assertLess(abs(g['cosine_to_target']),1e-3)

    def test_wrong_same_seed_source_is_rejected(self):
        m,d=self.source();(_,_,_,_),observer=self.run_one(m,d)
        source={k:observer.capture[k].clone() for k in ['before','after']};source['after']+=1
        with self.assertRaisesRegex(ValueError,'head features differ'):
            self.run_one(m,d,'targeted','edge_context',source)

    def test_rollout_reproduces_local_coalition_logits(self):
        m,d=self.source();(_,_,_,_),observer=self.run_one(m,d)
        source={k:observer.capture[k] for k in ['before','after']};adj=torch.ones((1,4,4),dtype=torch.long)
        states=[dict(features=source[name],adjacency=adj,network_bonds=native_head(m.gen.ligand_dec,source[name],adj,torch)) for name in ['before','after']]
        val,_,_=evaluate_coalitions(m.gen.ligand_dec,*states,d['pairs'],2,torch)
        expected=dict(pairs=d['pairs'],logits=val['logits'])
        for family in FAMILIES:
            (_,_,_,_),target=self.run_one(m,d,'targeted',family,source,expected)
            self.assertTrue(target.receipt['previous_local_experiment_match']['passed'])

    def test_geometry_random_control_symmetry_norm_and_local_rng(self):
        rng=np.random.default_rng(34);x=torch.from_numpy(rng.normal(size=(1,9,9,258)).astype(np.float32));before=x.clone()
        pairs=np.array([[2,4],[2,7]]);f,g=pairs.T
        for i,j in pairs:
            x[0,j,i,128:130]=x[0,i,j,128:130]
            before[0,i,j,128:130]=x[0,i,j,128:130]-.2;before[0,j,i,128:130]=before[0,i,j,128:130]
        torch_state=torch.get_rng_state().clone();np_state=np.random.get_state();python_state=random.getstate()
        out,info=modify_features(x,before,pairs,[2],64,'random_direction','fixed_key',torch)
        out2,_=modify_features(x,before,pairs,[2],64,'random_direction','fixed_key',torch)
        self.assertTrue(torch.equal(out,out2));self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
        self.assertEqual(python_state,random.getstate());np.testing.assert_array_equal(np_state[1],np.random.get_state()[1])
        self.assertTrue(torch.equal(out[0,f,g,128:130],out[0,g,f,128:130]))
        self.assertTrue(info['per_group'][0]['magnitude_matched']);self.assertLess(abs(info['per_group'][0]['cosine_to_target']),1e-3)

    def test_zero_change_is_uninformative(self):
        x=torch.zeros((1,4,4,258));out,info=modify_features(x,x,np.array([[1,3]]),[2,3],64,'random_direction','zero',torch)
        self.assertTrue(torch.equal(out,x));self.assertFalse(info['applied'])
        self.assertFalse(any(g['magnitude_informative'] for g in info['per_group']))

    def test_original_weight_head_matches_all_targeted_groups(self):
        fixture=os.environ.get('TRACE_TEST_WEIGHTS');self.assertTrue(fixture,'Set TRACE_TEST_WEIGHTS to the uploaded head fixture')
        mlp=torch.nn.Sequential(torch.nn.Linear(258,512),torch.nn.SiLU(),torch.nn.Linear(512,128))
        dec=SimpleNamespace(bond_refine=SimpleNamespace(message_mlp=mlp),bond_proj=torch.nn.Linear(128,5))
        with np.load(fixture) as w,torch.no_grad():
            for name,param in [('w0',mlp[0].weight),('b0',mlp[0].bias),('w2',mlp[2].weight),('b2',mlp[2].bias),('wp',dec.bond_proj.weight),('bp',dec.bond_proj.bias)]:param.copy_(torch.from_numpy(w[name]).float())
        generator=torch.Generator().manual_seed(8);states=[];adj=torch.ones((1,12,12),dtype=torch.long)
        for i in range(2):
            x=torch.randn((1,12,12,258),generator=generator)
            states.append(dict(features=x,adjacency=adj,network_bonds=native_head(dec,x,adj,torch)))
        pairs=np.array([[7,2],[7,9],[7,4]]);val,_,_=evaluate_coalitions(dec,*states,pairs,64,torch)
        for groups in FAMILIES.values():
            changed,_=modify_features(states[1]['features'],states[0]['features'],pairs,groups,64,'targeted','fixture',torch)
            actual=pair_values(native_head(dec,changed,adj,torch),pairs).numpy();coalition=15-sum(1<<g for g in groups)
            np.testing.assert_array_equal(actual,val['logits'][coalition])

    def test_unknown_chemistry_not_normal(self):
        m,d=self.source();prediction={k:v.clone() for k,v in d['final'].items()};m.builder.vocab=Vocab(['Xx','O','<NOATOM>'])
        try:metrics=native.graph_metrics(prediction,m.builder,dict(site=1),1.)
        finally:m.builder.vocab=Vocab(['C','O','<NOATOM>'])
        self.assertIsNone(metrics['site_valence_problem']);self.assertIsNone(metrics['rdkit_valid'])
        self.assertNotEqual(metrics['chemistry_status'],'assessed')

    def test_fixed_identity_separate_from_external_valence(self):
        from rdkit import Chem
        from run_mechanism_rollouts import raw_fixed_checks
        m,d=self.source();d['task']=dict(fixed_sampler_indices=[0,1],fixed_reference_indices=[0,1])
        d['reference']=Chem.MolFromSmiles('CO.CC')
        result=raw_fixed_checks(d['final'],d,m)
        self.assertTrue(result['fixed_chemistry_preserved'])
        self.assertTrue(native.graph_metrics(d['final'],m.builder,dict(site=1),1.)['site_valence_problem'])
        changed={k:v.clone() for k,v in d['final'].items()};changed['atomics'][0,1]=torch.tensor([1.,0.,0.])
        result=raw_fixed_checks(changed,d,m)
        self.assertFalse(result['fixed_chemistry_preserved']);self.assertEqual(result['fixed_element_changed_sites'],[1])

    def test_real_review_selects_bounded_batch(self):
        root=Path(os.environ['TRACE_REVIEW_ROOT']);audit=Path(os.environ['TRACE_AUDIT_ROOT'])
        protocol,trials,records=select_protocol(root,audit)
        self.assertEqual(protocol['call'],52);self.assertEqual(protocol['reference_atom'],14)
        self.assertEqual(len(records),10);self.assertEqual(len(trials),150)
        self.assertTrue(all(t['kind']=='sham' for t in trials[:10]))
        self.assertEqual(len({t['id'] for t in trials}),150)

    def test_tensordict_plain_dict_roundtrip(self):
        values=[TensorDict({}),TensorDict({'pic50':torch.tensor([[1.]])}),
                TensorDict({'pic50':torch.tensor([[1.]])},batch_size=[1]),
                TensorDict({'nested':TensorDict({'pic50':torch.tensor([[1.]])})})]
        for value in values:
            with self.subTest(value=str(value)):
                copied=helpers.move_copy(value,'cpu',torch)
                self.assertIsInstance(copied,dict)
                native.compare_tree(value,copied,torch,'output.affinity')
                native.compare_tree(copied,value,torch,'output.affinity')
                native.compare_tree(value,value.clone(),torch,'output.affinity')

    def test_tensordict_changed_leaf_is_rejected(self):
        actual=TensorDict({'pic50':torch.tensor([[1.]])})
        wrongs=[({'pic50':torch.tensor([[2.]])},'Native replay diverges'),
                ({'pic50':torch.tensor([1.])},'Shape/dtype mismatch'),
                ({'pic50':torch.tensor([[1.]],dtype=torch.float64)},'Shape/dtype mismatch'),
                ({'pic50':torch.tensor([[float('nan')]])},'Nonfinite computational tensor')]
        for expected,message in wrongs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError,message+'.*output.affinity.pic50'):
                    native.compare_tree(actual,expected,torch,'output.affinity')

    def test_tensordict_missing_key_and_wrong_container_rejected(self):
        actual=TensorDict({'pic50':torch.tensor([[1.]])})
        for expected,message in [({},'Keys differ'),({'pkd':torch.tensor([[1.]])},'Keys differ'),(None,'Mapping type mismatch')]:
            with self.subTest(expected=str(expected)):
                with self.assertRaisesRegex(ValueError,message+'.*output.affinity'):
                    native.compare_tree(actual,expected,torch,'output.affinity')

    def test_tensordict_missing_input_label_policy_unchanged(self):
        x=TensorDict({'pic50':torch.tensor([[float('nan')]])})
        native.compare_tree(x,helpers.move_copy(x,'cpu',torch),torch,'input.affinity')
        with self.assertRaisesRegex(ValueError,'Nonfinite computational tensor'):
            native.compare_tree(x,helpers.move_copy(x,'cpu',torch),torch,'output.affinity')
        with self.assertRaisesRegex(ValueError,'Missing-label locations changed'):
            native.compare_tree(x,{'pic50':torch.tensor([[1.]])},torch,'input.affinity')

    def test_native_full_sham_with_tensordict_auxiliary_outputs(self):
        model,data=self.source(auxiliary=True)
        (final,receipt,logs,_),observer=self.run_one(model,data)
        self.assertTrue(receipt['final_match']['passed'])
        self.assertEqual(len(receipt['checked_calls']),4)
        self.assertTrue(observer.receipt['nonbond_outputs_unchanged_at_intervention'])
        self.assertIn('affinity.pic50',receipt['final_match']['fields'])
        self.assertIn('docking_score.vina_score',receipt['final_match']['fields'])

    def test_targeted_and_random_rollout_with_tensordict_auxiliary_outputs(self):
        model,data=self.source(auxiliary=True)
        (_,_,_,_),observer=self.run_one(model,data)
        source={k:observer.capture[k] for k in ['before','after']}
        for kind in ['targeted','random_direction']:
            with self.subTest(kind=kind):
                (final,receipt,logs,_),intervention=self.run_one(model,data,kind,'geometry_and_edge',source)
                self.assertTrue(receipt['random_streams_matched_all_calls'])
                self.assertTrue(intervention.receipt['nonbond_outputs_unchanged_at_intervention'])
                self.assertEqual(len(logs),4)

    def test_nonbond_mutation_at_patch_call_still_rejected(self):
        model,data=self.source(auxiliary=True)
        (_,_,_,_),observer=self.run_one(model,data)
        source={k:observer.capture[k] for k in ['before','after']}
        def corrupt(module,args,kwargs,out):
            if int(args[2][0])==2:out['affinity']['pic50']=out['affinity']['pic50']+1.
        handle=model.register_forward_hook(corrupt,with_kwargs=True)
        try:
            with self.assertRaisesRegex(ValueError,'Native replay diverges: output.affinity.pic50'):
                self.run_one(model,data,'targeted','geometry_and_edge',source)
        finally:handle.remove()


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
