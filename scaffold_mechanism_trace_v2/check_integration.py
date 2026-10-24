"""Constructed regression fixtures, NOT scientific results or full-model replay.

Run from a FLOWR.ROOT repo, or set TRACE_NATIVE_SOURCE to that repo.
Executes selected unmodified native builder/chemistry functions from its source
with real Torch/RDKit, without importing unrelated training dependencies.
"""
from __future__ import annotations
import ast
import csv
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
import native_chemistry as nc
from native_readout import native_head
import trace_boundary_readout as runner


def native_parts(repo):
    def compile_nodes(nodes, ns, filename):
        prefix=ast.parse('from __future__ import annotations').body
        exec(compile(ast.fix_missing_locations(ast.Module(body=prefix+nodes,type_ignores=[])),str(filename),'exec'),ns)
    rdpath=repo/'flowr/util/rdkit.py'
    rd=ModuleType('flowr.util.rdkit')
    rd.__dict__.update(Chem=Chem,AllChem=AllChem,np=np,threading=threading)
    names={'PeriodicTable','_check_shape_len','_check_dim_shape','mol_is_valid','mol_from_atoms'}
    nodes=[]
    for node in ast.parse(rdpath.read_text()).body:
        if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'PT','IDX_BOND_MAP'} for t in node.targets):
            nodes.append(node)
    compile_nodes(nodes,rd.__dict__,rdpath)
    fpath=repo/'flowr/util/functional.py'
    fn=ModuleType('flowr.util.functional');fn.__dict__.update(torch=torch)
    compile_nodes([n for n in ast.parse(fpath.read_text()).body if isinstance(n,ast.FunctionDef)
                   and n.name in {'pad_tensors','bonds_from_adj','adj_from_node_mask'}],fn.__dict__,fpath)
    bpath=repo/'flowr/models/mol_builder.py'
    classes=[n for n in ast.parse(bpath.read_text()).body if isinstance(n,ast.ClassDef)]
    methods={'_extract_mols','_remove_virtual_atoms','_mol_from_tensors','_mol_extract_atomics',
             '_mol_extract_bonds','_mol_extract_charges','_mol_extract_hybridization','_mol_extract_aromaticity'}
    selected=[n for c in classes for n in c.body if isinstance(n,ast.FunctionDef) and n.name in methods]
    assert {n.name for n in selected}==methods
    cls=ast.ClassDef(name='ExtractedNativeBuilder',bases=[],keywords=[],body=selected,decorator_list=[])
    ns=dict(torch=torch,np=np,smolRD=rd,smolF=fn)
    compile_nodes([cls],ns,bpath)
    flowr=ModuleType('flowr');flowr.constants=SimpleNamespace(NOATOM_TOKEN='<NOATOM>')
    return ns['ExtractedNativeBuilder'],rd,fn,{'flowr':flowr,'flowr.util.rdkit':rd,'flowr.util.functional':fn}


class Vocab:
    def __init__(self,tokens):
        self.size=len(tokens);self.idx_token_map=dict(enumerate(tokens))
        self.token_idx_map={t:i for i,t in enumerate(tokens)}
    def contains(self,token):return token in self.token_idx_map
    def tokens_from_indices(self,ids):return [self.idx_token_map[i] for i in ids]


class IntegrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo=Path(os.environ.get('TRACE_NATIVE_SOURCE','.')).resolve()
        builder_type,cls.rd,cls.fn,modules=native_parts(repo)
        cls.modules_patch=patch.dict(sys.modules,modules);cls.modules_patch.start()
        cls.builder=builder_type();cls.builder.vocab=Vocab(['C','O','<NOATOM>'])
        cls.builder.vocab_charges=Vocab([0,1])

    @classmethod
    def tearDownClass(cls):cls.modules_patch.stop()

    def fixture(self):
        # One genuine (constructed) neutral oxygen valence error. Restoring its
        # previous geometry feature removes a bond but leaves one carbon isolated.
        mlp=torch.nn.Sequential(torch.nn.Linear(9,1),torch.nn.SiLU(),torch.nn.Linear(1,1))
        dec=SimpleNamespace(bond_refine=SimpleNamespace(message_mlp=mlp),bond_proj=torch.nn.Linear(1,5))
        with torch.no_grad():
            for p in list(mlp.parameters())+list(dec.bond_proj.parameters()):p.zero_()
            mlp[0].weight[0,4]=1;mlp[2].weight[0,0]=1
            dec.bond_proj.weight[1,0]=1;dec.bond_proj.bias.fill_(-5);dec.bond_proj.bias[0]=.5;dec.bond_proj.bias[1]=0
        raw=dict(coords=torch.tensor([[[0.,0.,0.],[1.4,0.,0.],[1.4,1.4,0.],[1.4,0.,1.4]]]),
                 atomics=torch.nn.functional.one_hot(torch.tensor([[0,1,0,0]]),3).float(),
                 charges=torch.nn.functional.one_hot(torch.zeros((1,4),dtype=torch.long),2).float(),
                 mask=torch.ones((1,4),dtype=torch.long))
        adj=self.fn.adj_from_node_mask(raw['mask'],self_connect=True)
        states=[]
        for edges in [[(0,1),(1,2)],[(0,1),(1,2),(1,3)]]:
            x=torch.zeros((1,4,4,9))
            for a,b in edges:x[0,a,b,4]=x[0,b,a,4]=1
            z=native_head(dec,x,adj,torch)
            states.append(dict(features=x,adjacency=adj,network_bonds=z,
                               prediction={**{k:v.clone() for k,v in raw.items()},'bonds':z.clone()}))
        model=SimpleNamespace(gen=SimpleNamespace(ligand_dec=dec),builder=self.builder)
        weights={k:p.detach().numpy().astype(np.float64) for k,p in
                 [('w0',mlp[0].weight),('b0',mlp[0].bias),('w2',mlp[2].weight),('b2',mlp[2].bias),
                  ('wp',dec.bond_proj.weight),('bp',dec.bond_proj.bias)]}
        frames=[]
        for i,s in enumerate(states):
            g=nc.graph(s['prediction'],['C','O','<NOATOM>'],[0,1])
            chem,_=nc.chemistry(s['prediction'],self.builder,g,Chem,rdBase,self.rd.mol_is_valid)
            neighbors=[dict(slot=j,**{'class':int(g['classes'][1,j])},fixed=j==0)
                       for j in range(4) if g['classes'][1,j]>0]
            frames.append(dict(call=i,chemistry=chem,fixed_sites=[dict(slot=1,present=True,element='O',charge=0,neighbors=neighbors)]))
        return model,states,weights,frames

    def test_native_chemistry_charge_and_virtual_mapping(self):
        model,states,_,_=self.fixture();raw=states[1]['prediction']
        g=nc.graph(raw,['C','O','<NOATOM>'],[0,1]);chem,mol=nc.chemistry(raw,self.builder,g,Chem,rdBase,self.rd.mol_is_valid)
        self.assertFalse(chem['native_rdkit_valid']);self.assertEqual(chem['problems'][0]['sampler_slots'],[1])
        charged={k:v.clone() for k,v in raw.items()};charged['charges'][0,1]=torch.tensor([0.,1.])
        g=nc.graph(charged,['C','O','<NOATOM>'],[0,1]);chem,_=nc.chemistry(charged,self.builder,g,Chem,rdBase,self.rd.mol_is_valid)
        self.assertTrue(chem['native_rdkit_valid'])
        virtual={k:v.clone() for k,v in raw.items()};virtual['atomics'][0,2]=torch.tensor([0.,0.,1.])
        g=nc.graph(virtual,['C','O','<NOATOM>'],[0,1]);chem,mol=nc.chemistry(virtual,self.builder,g,Chem,rdBase,self.rd.mol_is_valid)
        self.assertEqual(g['slots'],[0,1,3]);self.assertEqual(mol.GetNumAtoms(),3)
        self.assertTrue(chem['native_rdkit_valid'])

    def test_capture_receipts_and_optional_numpy_failure(self):
        model,states,weights,frames=self.fixture();after=states[1]
        def forward(*args,**kwargs):
            return {**after['prediction'],'bonds':native_head(model.gen.ligand_dec,after['features'],after['adjacency'],torch)}
        call=dict(call_index=1,network_output_before_postprocessing=after['prediction'])
        with tempfile.TemporaryDirectory() as td,patch.object(runner,'forward_once',side_effect=forward),patch.object(runner,'readout',side_effect=RuntimeError('optional diagnostic intentionally unavailable')):
            out=Path(td)
            values,head,receipt=runner.capture_state(model,dict(pocket=None,encoding=None),call,np.array([[1,2],[1,3]]),
                        torch,torch.device('cpu'),weights,frames[1],out/'receipt.json',out/'debug.npz')
            self.assertTrue(receipt['accepted']);self.assertTrue(receipt['native_head']['passed'])
            self.assertFalse(receipt['numpy_diagnostic_only']['used_for_attribution'])
            self.assertIn('error',receipt['numpy_diagnostic_only'])
            self.assertEqual(tuple(head['features'].shape),(1,4,4,9))

    def test_capture_mismatch_saves_error_and_tensors(self):
        model,states,weights,frames=self.fixture();after=states[1]
        broken={**after['prediction'],'bonds':after['network_bonds']+.25}
        call=dict(call_index=1,network_output_before_postprocessing=after['prediction'])
        with tempfile.TemporaryDirectory() as td,patch.object(runner,'forward_once',return_value=broken):
            out=Path(td)
            with self.assertRaisesRegex(RuntimeError,'Original forward replay differs'):
                runner.capture_state(model,dict(pocket=None,encoding=None),call,np.array([[1,2],[1,3]]),
                        torch,torch.device('cpu'),weights,frames[1],out/'receipt.json',out/'debug.npz')
            self.assertFalse(json.loads((out/'receipt.json').read_text())['accepted'])
            with np.load(out/'debug.npz') as z:self.assertIn('recorded_bonds',z.files)

    def test_16_path_analysis_preserves_disconnection_and_evidence(self):
        model,states,weights,frames=self.fixture();rid='constructed_case'
        records={rid:dict(report=dict(case='constructed',ligand_idx=0,seed=0,
                      fixed_mapping={'1':{'reference_index':7}}),frames=frames)}
        job=dict(id='contrast_000',run=rid,before=0,after=1,site=1,kind='onset',terminal_segment_onset=True)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);folder=out/'captures'/rid;folder.mkdir(parents=True);(out/'contrasts').mkdir()
            for i,state in enumerate(states):
                pairs=np.array([[1,2],[1,3]])
                np.savez_compressed(folder/f'call_{i:04d}.npz',pairs=pairs,active=np.ones(4,dtype=bool),
                      logits=state['network_bonds'][0,[2,3],[1,1]].numpy())
                torch.save(state,folder/f'call_{i:04d}_head.pt')
            summary,pairs=runner.analyses(dict(jobs=[job]),records,out,weights,2,model,torch,torch.device('cpu'))
            with (out/'path_comparison.csv').open(encoding='utf-8-sig') as file:
                rows=list(csv.DictReader(file))
            self.assertEqual(len(rows),16)
            self.assertEqual(rows[15]['site_valence_problem'],'True');self.assertEqual(rows[15]['connected'],'True')
            self.assertEqual(rows[11]['site_valence_problem'],'False');self.assertEqual(rows[11]['rdkit_valid'],'True')
            self.assertEqual(rows[11]['connected'],'False')
            self.assertEqual(summary[0]['largest_absolute_score_contributor'],'geometry')
            self.assertLess(summary[0]['attribution_sum_error'],1e-8)
            self.assertTrue(all(r['final_rollout_effect']=='not_tested' for r in rows))
            self.assertEqual(summary[0]['final_failure_causality'],'not_tested')
            runner.report_html(out,summary,dict(completed=True,fixture_only=True))
            package=runner.bundle(out)
            import zipfile
            with zipfile.ZipFile(package) as z:
                self.assertIn('path_comparison.csv',z.namelist())
                self.assertFalse(any(p.endswith('.pt') for p in z.namelist()))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
