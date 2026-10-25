"""Test-only native-source loader; constructed regression fixtures, not study data."""
from __future__ import annotations
import ast,threading
from pathlib import Path
from types import ModuleType,SimpleNamespace
import numpy as np
import torch
from rdkit import Chem,rdBase
from rdkit.Chem import AllChem
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
