#!/usr/bin/env python3
"""One-write counterfactuals followed by the complete native ROOT sampler.

Observational findings select candidates; they are not assumed causal. A full
native replay and a zero-write sham must pass before interventions are run.
No model source, weights, training flags, input conditions or original files
are edited. Generated atoms are never aligned across different seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import json
import random
import sys
import traceback
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

VERSION = '2026-10-20.causal.2'
HELPER_SHA = '1d5b6cfef030f85d1b0df939d96b6402d64f68e9b2a8ef0b746c12236a84ff82'
SUBMODULE_SHA = 'c05f5a1c98a7841711decadc18097914084490006cce7c878cf17570e846e360'
INTEGRATOR_SHA = 'e9abd83167f708da4712db2ac49ab24ac44a39c59c46a54c20789c7873c58000'
MODULE_ATTRS = {'self_attention':'self_attention', 'conditional_attention':'cond_attention',
                'equivariant_feedforward':'equi_ff'}
# These are carried-through input labels, not model predictions. The inspected
# ROOT forward reads none of them, and this adapter disallows guidance. Missing
# labels may be NaN, but their masks, dtype, shape and all finite values must be
# identical. No other state, self-condition, prediction or activation is exempt.
MISSING_LABEL_PATHS = frozenset({
    'input.affinity.pic50', 'input.affinity.pkd',
    'input.affinity.pki', 'input.affinity.pec50',
    'input.docking_score.vina_score', 'input.docking_score.gnina_score',
    'input.docking_score.glide_score',
})


def require(ok, message):
    if not bool(ok): raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v
                    for k,v in row.items()} for row in rows)


def import_file(path, name, expected=None):
    path=Path(path).resolve();require(path.is_file(),f'Missing dependency: {path}')
    if expected: require(digest(path)==expected,f'Version mismatch: {path.name}')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def array(x): return x.detach().cpu().numpy().copy()


def compare_tree(a,b,torch,path='root'):
    """Compare recorded sampler inputs, including SC flags; no resets hide drift."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        require(torch.is_tensor(a) and torch.is_tensor(b),f'Tensor type mismatch: {path}')
        x,y=array(a),array(b)
        require(x.shape==y.shape and x.dtype==y.dtype,
                f'Shape/dtype mismatch: {path}; actual={x.shape}/{x.dtype}; original={y.shape}/{y.dtype}')
        if path in MISSING_LABEL_PATHS:
            require(not np.isinf(x).any() and not np.isinf(y).any(),
                    f'Infinite input label is not an allowed missing value: {path}')
            require(np.array_equal(np.isnan(x),np.isnan(y)),
                    f'Missing-label locations changed: {path}')
            require(np.array_equal(x,y,equal_nan=True),
                    f'Input label values changed: {path}')
            return
        require(np.isfinite(x).all() and np.isfinite(y).all(),
                f'Nonfinite computational tensor: {path}')
        maximum=float(np.max(np.abs(x.astype(float)-y.astype(float)))) if x.size else 0.
        require(np.allclose(x,y,atol=1e-5,rtol=1e-5),f'Native replay diverges: {path}; max_abs_difference={maximum}')
        if path.rsplit('.',1)[-1] in {'atomics','bonds','charges','hybridization'}:
            require(np.array_equal(x.argmax(-1),y.argmax(-1)),f'Categorical mismatch: {path}')
    elif isinstance(a,Mapping) or isinstance(b,Mapping):
        require(isinstance(a,Mapping) and isinstance(b,Mapping) and set(a)==set(b),f'Keys differ: {path}')
        for k in a: compare_tree(a[k],b[k],torch,path+'.'+str(k))
    elif isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):
        require(type(a)==type(b) and len(a)==len(b),f'Sequence differs: {path}')
        for i,(x,y) in enumerate(zip(a,b)): compare_tree(x,y,torch,path+'.'+str(i))
    else: require(a==b,f'Scalar differs: {path}')


def rng_matches(saved,device,torch):
    n=np.random.get_state();s=saved['numpy']
    ok=(random.getstate()==saved['python'] and n[0]==s[0]
        and np.array_equal(n[1],array(s[1]).astype(np.uint32))
        and n[2:]==tuple(s[2:]) and torch.equal(torch.get_rng_state(),saved['torch_cpu'].cpu()))
    if 'torch_cuda' in saved:
        ok=ok and device.type=='cuda' and torch.equal(torch.cuda.get_rng_state(device),saved['torch_cuda'].cpu())
    return bool(ok)


def plan_trials(p):
    """Configuration contains sample-specific choices; execution has no seed/atom constants."""
    focus=p['failure'];seeds=[focus,*p['success_controls']]
    require(len({s['seed'] for s in seeds})==len(seeds),'Duplicated sample roles.')
    require(len(p['events'])>0 and len(p['candidates'])>0,'Empty experiment plan.')
    require(1<=int(p['random_direction_replicates'])<=100,'Invalid random control count.')
    trials=[dict(id=f"baseline_s{s['seed']}",kind='baseline',**s) for s in seeds]
    e=p['events'][-1];c=p['candidates'][0]
    trials.append(dict(id=f"sham_s{focus['seed']}",kind='sham',**focus,**e,**c))
    for e in p['events']:
        require(e['donor_call']<e['call'],'Temporal donor must precede intervention.')
        for c in p['candidates']:
            require(c['module'] in MODULE_ATTRS and c['layer']>=1,'Unsupported intervention site.')
            common={**focus,**e,**c,'reference_call':e['call']}
            name=f"s{focus['seed']}_c{e['call']}_L{c['layer']}_{c['module']}"
            trials.append(dict(id=name+'_targeted',kind='targeted',**common))
            for rep in range(p['random_direction_replicates']):
                trials.append(dict(id=name+f'_random{rep}',kind='random_direction',replicate=rep,**common))
    e=next(v for v in p['events'] if v['call']==p['time_control_event_call'])
    require(p['time_control_call']<e['call'],'Time control must be earlier than its source event.')
    for c in p['candidates']:
        trials.append(dict(id=f"s{focus['seed']}_time_c{p['time_control_call']}_L{c['layer']}_{c['module']}",
            kind='time_control',**focus,**c,call=p['time_control_call'],donor_call=e['donor_call'],reference_call=e['call']))
    for s in p['success_controls']:
        for c in p['candidates']:
            e=p['success_control_event']
            trials.append(dict(id=f"s{s['seed']}_success_L{c['layer']}_{c['module']}",kind='success_control',
                **s,**c,**e,reference_call=e['call']))
    require(len({t['id'] for t in trials})==len(trials),'Duplicated trial identifiers.')
    return trials


def matched_direction(delta, control_seed):
    """Random signed channel permutation: exact Frobenius norm, SE(3)-equivariant.

    Channels are scalar copies of 3D vectors. The same channel transform is
    applied to x,y,z; spatial axes are never independently shuffled.
    This is a specific control distribution, not all possible random controls.
    """
    rng=np.random.default_rng(control_seed)
    order=rng.permutation(delta.shape[-1]);sign=rng.choice([-1.,1.],delta.shape[-1])
    result=delta[...,order]*sign
    require(np.isclose(np.linalg.norm(result),np.linalg.norm(delta),rtol=1e-12,atol=1e-12),'Norm control failed.')
    return result


def module_write(record, trial):
    key=f"layer_{trial['layer']-1:02d}__{trial['module']}_update"
    require(key in record,f'Missing recorded module: {key}')
    x=record[key]
    require(x.ndim==3 and x.shape[1]==3 and np.isfinite(x).all(),'Invalid vector-write record.')
    return x


class SingleWritePatch:
    """Change one generated atom's vector update once; invariant/edge writes are kept."""
    def __init__(self,module,trial,clock,reference,delta,torch):
        self.trial,self.clock,self.reference,self.delta,self.torch=trial,clock,reference,delta,torch
        self.visits=0;self.receipt=None
        self.handle=module.register_forward_hook(self.hook)

    def hook(self,module,args,output):
        if self.clock['call']!=self.trial['call']:return None
        self.visits+=1;require(self.visits==1,'Intervention module called more than once at target time.')
        is_tuple=self.trial['module']!='equivariant_feedforward'
        if is_tuple:require(isinstance(output,tuple) and len(output)==3,'Unsupported attention output contract.')
        value=output[0] if is_tuple else output
        require(value.ndim==4 and value.shape[0]==1,'Expected [1,atoms,xyz,channels].')
        original=array(value[0]);require(original.shape==self.reference.shape,'Module shape differs from recording.')
        require(np.array_equal(original,self.reference),'Pre-intervention module write differs from original; stop.')
        partner=self.trial['partner'];before=original[partner].astype(np.float64)
        correction=self.delta.astype(np.float64)
        require(correction.shape==before.shape and np.isfinite(correction).all(),'Invalid patch dimensions.')
        norm=float(np.linalg.norm(correction))
        if self.trial['kind']=='sham':
            self.receipt=dict(applied=False,visits=1,delta_norm=0.,prewrite_exact=True)
            return None
        changed=value.clone()
        # Form the desired value in float64 on CPU and round once. Full temporal
        # replacement consequently equals the recorded donor exactly in float32.
        desired=(before+correction).astype(original.dtype)
        changed[0,partner]=self.torch.as_tensor(desired,device=value.device,dtype=value.dtype)
        after=array(changed[0]);actual=after[partner].astype(np.float64)-before
        untouched=np.arange(len(after))!=partner
        require(np.array_equal(original[untouched],after[untouched]),'An unselected atom was modified.')
        require(np.isfinite(after).all(),'Nonfinite patched write.')
        bound=8*np.finfo(original.dtype).eps*(np.linalg.norm(before)+norm+1.)
        require(np.linalg.norm(actual-correction)<=bound,'Patch rounding exceeds native precision bound.')
        self.receipt=dict(applied=True,visits=1,prewrite_exact=True,delta_norm=norm,
            actual_delta_norm=float(np.linalg.norm(actual)),rounding_error=float(np.linalg.norm(actual-correction)),
            write_norm=float(np.linalg.norm(before)),only_generated_atom=partner,
            relative_write_change=norm/max(float(np.linalg.norm(before)),1e-30))
        return (changed,*output[1:]) if is_tuple else changed

    def close(self):self.handle.remove()


def graph_metrics(prediction, builder, focus, scale):
    """Read native lower-triangle decisions; never equate bond loss with success."""
    coords=array(prediction['coords'][0]).astype(float)*scale
    mask=array(prediction['mask'][0]).astype(bool)
    atoms=builder.vocab.tokens_from_indices(prediction['atomics'][0].argmax(-1).detach().cpu().tolist())
    charges=builder.vocab_charges.tokens_from_indices(prediction['charges'][0].argmax(-1).detach().cpu().tolist())
    d=array(prediction['bonds'][0]);require(d.shape[-1]==5,'Unsupported bond vocabulary.')
    classes=d.argmax(-1);bond=np.tril(classes,-1);bond=bond+bond.T
    keep=mask & np.array([a not in {'<PAD>','<NOATOM>'} for a in atoms])
    weights=np.array([0.,1.,2.,3.,1.5]);order=weights[bond]*keep[:,None]*keep[None,:]
    s,j=focus['site'],focus['partner'];valid=bool(keep[s] and keep[j])
    charge=int(charges[s]) if isinstance(charges[s],(int,np.integer)) else None
    neighbors=[int(k) for k in np.flatnonzero((bond[s]>0)&keep)] if keep[s] else []
    active=set(np.flatnonzero(keep).tolist());components=0
    while active:
        components+=1;todo=[active.pop()]
        while todo:
            i=todo.pop();new=set(np.flatnonzero((bond[i]>0)&keep).tolist()) & active
            active-=new;todo.extend(new)
    return dict(site=s,partner=j,site_element=str(atoms[s]),site_charge=charge,site_charge_token=str(charges[s]),
        site_present=bool(keep[s]),partner_present=bool(keep[j]),bond_class=int(bond[s,j]) if valid else None,
        single_minus_no_bond_value=float(d[max(s,j),min(s,j),1]-d[max(s,j),min(s,j),0]) if valid else None,
        distance_A=float(np.linalg.norm(coords[j]-coords[s])) if valid else None,
        site_explicit_bond_order_sum=float(order[s].sum()),site_neighbors=neighbors,
        site_neutral_carbon_overvalent=(bool(order[s].sum()>4+1e-6) if keep[s] and atoms[s]=='C' and charge==0 else None),
        active_atoms=int(keep.sum()),n_components=components,connected=bool(components==1 and keep.any()))


class ScopedMethods:
    def __init__(self):self.items=[]
    def set(self,obj,name,fn):
        self.items.append((obj,name,name in vars(obj),vars(obj).get(name)))
        setattr(obj,name,MethodType(fn,obj))
    def close(self):
        for obj,name,had,value in reversed(self.items):
            if had:setattr(obj,name,value)
            else:delattr(obj,name)
        self.items=[]


def run_rollout(model,data,trial,write_reference,delta,helpers,device,torch,debug_dir=None):
    """Run the native method from the saved prior. Never feed later saved states back.

    Cached protein encodings and its COM are restored. Native visualization's
    res_names side effect is retained without writing thousands of XYZ files.
    Every baseline/sham input and output must match the original complete run.
    """
    inputs=data['inputs'];calls=data['calls'];cfg=data['manifest']['args']
    move=lambda x:helpers.move_copy(x,device,torch)
    prior=move(inputs['prior']);pocket=move(inputs['pocket_data'])
    if 'complex' in pocket:
        require(len(inputs['pocket_com'])==1,'Expected one pocket COM.')
        pocket['complex']=[SimpleNamespace(com=inputs['pocket_com'][0].cpu().clone())]
    enc=move(data['encoding']);clock={'call':-1};logs=[];sampler=[];checks=[];handles=[]
    methods=ScopedMethods();patch=None
    strict_all=trial['kind'] in {'baseline','sham'}
    def pre(module,args,kwargs):
        clock['call']+=1;k=clock['call'];require(k<len(calls),'Unexpected extra model call.')
        c=calls[k]
        batch=args[0] if args else kwargs['batch'];times=args[2] if len(args)>2 else kwargs['t']
        clock.update(phase='before_forward',last_input=dict(state=batch,times=times,self_condition=kwargs.get('cond_batch')))
        clock.pop('last_output',None)
        if k==0:helpers.restore_rng(c['rng_before_forward'],device,torch,np)
        require(rng_matches(c['rng_before_forward'],device,torch),f'Random streams no longer matched at call {k}.')
        compare_tree(times,c['times'],torch,'times')
        compare_tree(batch['mask'],c['state_before_forward']['mask'],torch,'mask')
        if strict_all or k<=trial['call']:
            compare_tree(batch,c['state_before_forward'],torch,'input')
            compare_tree(kwargs.get('cond_batch'),c['self_condition'],torch,'self_condition')
        return None
    def post(module,args,kwargs,out):
        clock.update(phase='after_forward',last_output=out)
        k=clock['call'];require(all(np.isfinite(array(v)).all() for v in helpers.tensor_leaves(out,torch).values()),'Nonfinite prediction.')
        if strict_all or k<trial['call']:
            check=helpers.compare_predictions(out,calls[k]['network_output_before_postprocessing'],torch,np,1e-5,1e-5)
            failures={k:v for k,v in check['fields'].items() if not v['passed']}
            require(check['passed'],f'Full native replay differs at call {k}: {failures}')
            checks.append({'call':k,'passed':True,'max_difference':max(v['max_abs_difference'] for v in check['fields'].values())})
        logs.append({'call':k,'t':calls[k]['times_scalar'],'categorical_values':'network_logits',
                     **graph_metrics(out,model.builder,trial,inputs['coord_scale'])})
        return None
    def encoding(_self,*args,**kwargs):return enc['pocket_equis'],enc['pocket_invs']
    def build(_self,curr,predicted,pocket_data,iter=0,step=0):
        sampler.append({'step':step,'categorical_values':'native_integrated_state_or_final_probabilities',
                        **graph_metrics(curr,model.builder,trial,inputs['coord_scale'])})
        # The original _build_trajectory mutates this otherwise-unused field.
        curr['res_names']=torch.zeros(curr['atomics'].size(0),curr['atomics'].size(1),device=device).long()
    try:
        methods.set(model.gen,'get_pocket_encoding',encoding)
        methods.set(model,'_build_trajectory',build)
        methods.set(model,'_save_trajectory',lambda _self,*a,**kw:None)
        handles=[model.register_forward_pre_hook(pre,with_kwargs=True),model.register_forward_hook(post,with_kwargs=True)]
        if trial['kind']!='baseline':
            layer=model.gen.ligand_dec.layers[trial['layer']-1]
            module=getattr(layer,MODULE_ATTRS[trial['module']])
            patch=SingleWritePatch(module,trial,clock,write_reference,delta,torch)
        helpers.restore_rng(inputs['rng'],device,torch,np)
        with torch.no_grad():
            final=model._generate(prior,pocket,steps=cfg['integration_steps'],times=move(inputs['times']),
                strategy=inputs['strategy'],solver=inputs['solver'],corr_iters=cfg['corrector_iters'],
                save_traj=bool(cfg['save_traj']),final_inpaint=inputs['final_inpaint'],
                apply_guidance=False,coord_noise_level=cfg['coord_noise_scale'])
        require(clock['call']+1==len(calls),'Incomplete native trajectory.')
        if patch:require(patch.visits==1 and patch.receipt is not None,'Patch was not executed exactly once.')
        final_match=None
        if strict_all:
            final_match=helpers.compare_predictions(final,data['final'],torch,np,1e-5,1e-5)
            require(final_match['passed'],'Final prediction differs from original full trajectory.')
        return final,dict(call_count=len(logs),random_streams_matched_all_calls=True,
            unmodified_prefix_verified=True,baseline_or_sham_full_match=strict_all,checked_calls=checks,
            final_match=final_match,patch=patch.receipt if patch else None),logs,sampler
    except BaseException:
        if debug_dir is not None:
            try:
                diagnostic={'trial':trial,'clock':clock,'original_call':calls[max(0,min(clock['call'],len(calls)-1))]}
                torch.save(helpers.move_copy(diagnostic,'cpu',torch),Path(debug_dir)/'replay_failure.pt')
                write_json(Path(debug_dir)/'partial_network_timeline.json',logs)
            except Exception:
                (Path(debug_dir)/'debug_capture_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        if patch:patch.close()
        for h in handles:h.remove()
        methods.close()


def evaluate_final(final,model,data,trial,folder,quality,torch):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from flowr.util.rdkit import mol_is_valid
    from flowr.gen.utils import check_substructure_match
    from posebusters import PoseBusters
    reference=data['reference'];task=data['task'];rows=[]
    mask=array(final['mask'][0]).astype(bool)
    atoms=model.builder.vocab.tokens_from_indices(final['atomics'][0].argmax(-1).tolist())
    slots=[i for i in range(len(mask)) if mask[i] and atoms[i]!='<NOATOM>']
    for stage,repair in [('raw_without_repair',False),('run_decode',None)]:
        # Both are post-inference evaluations. Repair never changes a rollout.
        mol=model._generate_mols(final,valence_repair=repair)[0]
        row=dict(stage=stage,rdkit_valid=bool(mol_is_valid(mol,connected=False)),
            fc_valid=bool(mol_is_valid(mol,connected=True)),fixed_chemistry_preserved=False,
            root_fixed_substructure_match=None,posebusters_valid=None,errors=[])
        rows.append(row)
        with Chem.SDWriter(str(folder/f'{stage}.sdf')) as w:
            if mol is not None:w.write(mol)
        if not row['rdkit_valid']:continue
        Chem.SanitizeMol(mol)
        try:
            require(mol.GetNumAtoms()==len(slots),'Decoded molecule/slot count differs.')
            require(np.allclose(mol.GetConformer().GetPositions(),array(final['coords'][0])[slots],
                                atol=2e-5,rtol=1e-6),'Decoded coordinate order differs.')
            fixed_missing=[i for i in task['fixed_sampler_indices'] if i not in slots]
            row['fixed_missing_sites']=fixed_missing
            if not fixed_missing:
                row.update(quality.fixed_diagnostics(mol,reference,task,slots,np))
                row['fixed_chemistry_preserved']=not any(row[k] for k in
                    ['fixed_element_changed_sites','fixed_charge_changed_sites','fixed_internal_bond_changed_pairs'])
            row['root_fixed_substructure_match']=bool(check_substructure_match(Chem.Mol(mol),Chem.Mol(reference),'scaffold_hopping'))
            row['molecule_smiles']=Chem.MolToSmiles(mol,isomericSmiles=False)
            row['scaffold_smiles']=Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol),isomericSmiles=False)
            row['murcko_scaffold_changed']=row['scaffold_smiles']!=data['reference_scaffold']
        except Exception as exc:
            row['errors'].append(f'Fixed-fragment/mapping evaluation: {type(exc).__name__}: {exc}')
            row['fixed_chemistry_preserved']=None
        try:row.update(quality.pb_metrics(mol,Path(data['manifest']['args']['pdb_file']),PoseBusters,Chem))
        except Exception as exc:row['errors'].append(f'PoseBusters could not evaluate: {type(exc).__name__}: {exc}')
    raw=rows[0]
    # This is deliberately narrower than overall scaffold-hopping success.
    chemical=(bool(raw['fc_valid'] and raw['fixed_chemistry_preserved'] and raw['root_fixed_substructure_match'])
              if raw['fixed_chemistry_preserved'] is not None else None)
    return dict(raw_valid_connected_fixed_chemistry=chemical,stages=rows,
        scope='Chemical connection/fixed-identity endpoint; 3D checks and scaffold change reported separately. No activity claim.',
        final_graph={'categorical_values':'final_probabilities',**graph_metrics(final,model.builder,trial,1.0)})


def prepare(args,plan,torch,helpers,quality):
    from rdkit import Chem,rdBase
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from flowr.data.interpolate import extract_scaffolds
    sub=read_json(args.submodule_dir/'submodule_capture_summary.json')
    analysis=read_json(args.submodule_dir/'submodule_analysis_summary.json')
    readout=read_json(args.readout_dir/'readout_capture_summary.json')
    require(sub['completed'] and sub['error'] is None and analysis['completed'],'Incomplete submodule observation.')
    require(sub['script_sha256']==SUBMODULE_SHA,'Unexpected submodule capture version.')
    require(analysis['capture_summary_sha256']==digest(args.submodule_dir/'submodule_capture_summary.json'),'Submodule summary changed.')
    require(sub['signature']['readout_summary_sha256']==digest(args.readout_dir/'readout_capture_summary.json'),'Readout provenance differs.')
    require(plan['case']==sub['plan']['case'],'Experiment case differs from captured case.')
    records={}
    for e in sub['states']:
        require(e['accepted'] and e['all_readout_tensors_exact'] and digest(args.submodule_dir/e['file'])==e['sha256'],'Unverified submodule record.')
        require(all(e[k]['passed'] for k in ['replay_vs_original','observation_vs_original','observation_vs_replay']),'Invalid native replay.')
        records[(e['seed'],e['call'])]=args.submodule_dir/e['file']
    source=dict(readout['source_sha256']);source.update(quality.SOURCE_HASHES)
    source['flowr/models/integrator.py']=INTEGRATOR_SHA
    for file,sha in source.items():require(digest(Path.cwd()/file)==sha,f'Source version differs: {file}')
    trials=plan_trials(plan);data={};load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
    for focus in [plan['failure'],*plan['success_controls']]:
        seed=focus['seed'];folder=args.output_root/f"sh_record_{plan['case']}_seed{seed}/research_record/attempt_000"
        manifest=read_json(folder.parent/'manifest.json');task=read_json(folder/'task_definition.json')
        require(manifest==readout['metadata'][str(seed)]['manifest'],'Original generation manifest differs.')
        require(task==readout['metadata'][str(seed)]['task'] and task['mapping_verified'],'Original atom mapping differs.')
        require(manifest['torch_version']==str(torch.__version__) and manifest['rdkit_version']==rdBase.rdkitVersion,'Use original Torch/RDKit environment.')
        for file,sha in manifest['input_sha256'].items():require(digest(Path(file))==sha,f'Original molecule/protein changed: {file}')
        cfg=manifest['args'];inputs=load(folder/'inputs.pt');attempt=read_json(folder/'attempt.json')
        require(cfg['guidance_config'] is None and not cfg.get('optimize_gen_ligs') and not cfg.get('optimize_gen_ligs_hs'),'Unsupported guidance/optimization run.')
        require(attempt['generation_completed'] and cfg['scaffold_hopping'],'Expected a complete scaffold-hopping record.')
        require(inputs['solver']==cfg['solver'] and inputs['strategy']==cfg['ode_sampling_strategy']
                and inputs['final_inpaint']==cfg['final_inpaint'],'Sampler settings differ.')
        require(inputs['prior']['mask'].shape[0]==1,'Only batch-one recordings supported in this adapter.')
        fixed=task['fixed_sampler_indices'];s,j=focus['site'],focus['partner']
        mask=array(inputs['prior']['mask'][0]).astype(bool)
        require(s in fixed and j not in fixed and 0<=j<len(mask) and mask[j],'Invalid fixed/generated focus mapping.')
        reference=quality.read_one_sdf(folder/'reference_preprocessed.sdf',Chem);require(reference is not None,'Missing reference molecule.')
        Chem.SanitizeMol(reference)
        actual_fixed=extract_scaffolds([reference],invert_mask=True)[0].nonzero(as_tuple=True)[0].tolist()
        require(actual_fixed==task['fixed_reference_indices'],'Native fixed-fragment extraction differs.')
        paths=sorted((folder/'forward_calls').glob('call_*.pt'))
        require(len(paths)==attempt['forward_calls'],'Missing native call recordings.')
        calls=[load(p) for p in paths]
        require([c['call_index'] for c in calls]==list(range(len(calls))),'Noncontiguous recorded calls.')
        data[seed]=dict(folder=folder,manifest=manifest,task=task,inputs=inputs,calls=calls,
            encoding=load(folder/'pocket_encoding.pt'),final=load(folder/'raw_final_prediction.pt'),
            reference=reference,reference_scaffold=Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(reference),isomericSmiles=False),
            input_hashes={str(p.relative_to(folder)):digest(p) for p in [folder/'inputs.pt',folder/'pocket_encoding.pt',
                folder/'raw_final_prediction.pt',folder/'task_definition.json',folder/'reference_preprocessed.sdf',*paths]})
    base=data[plan['failure']['seed']]['manifest']
    for d in data.values():require(helpers.condition_key(d['manifest'])==helpers.condition_key(base),'Seeds do not share generation conditions.')
    checkpoint=Path(base['args']['ckpt_path'])
    require(digest(checkpoint)==readout['checkpoint_sha256']==sub['signature']['checkpoint_sha256'],'Checkpoint changed.')
    for t in trials:
        if t['kind']=='baseline':continue
        require(t['layer']<=len(sub['layout']),'Layer out of range.')
        if t['module']=='conditional_attention':require(sub['layout'][t['layer']-1]['has_condition'],'Missing conditional module.')
        for call in [t['call'],t['donor_call'],t.get('reference_call',t['call'])]:
            require((t['seed'],call) in records,'Requested time is outside verified submodule capture.')
    identity=dict(version=VERSION,script_sha256=digest(__file__),plan=plan,
        submodule_summary_sha256=digest(args.submodule_dir/'submodule_capture_summary.json'),
        readout_summary_sha256=digest(args.readout_dir/'readout_capture_summary.json'),
        checkpoint_sha256=readout['checkpoint_sha256'],source_sha256=source,
        input_sha256={str(s):d['input_hashes'] for s,d in data.items()},
        evaluation_helper_sha256=digest(Path(quality.__file__)))
    return data,records,trials,identity,readout


def trial_vectors(t,records,plan):
    if t['kind']=='baseline':return None,None,{}
    def load(call):
        with np.load(records[(t['seed'],call)],allow_pickle=False) as z:
            return module_write(z,t).copy()
    current=load(t['call']);ref=load(t.get('reference_call',t['call']));donor=load(t['donor_call'])
    require(current.shape==ref.shape==donor.shape,'Temporal write shapes differ.')
    j=t['partner'];delta=donor[j].astype(float)-ref[j].astype(float)
    require(np.linalg.norm(delta)>0,'Zero temporal contrast; this trial is uninformative.')
    kind=t['kind']
    if kind=='random_direction':
        token=f"{plan['random_control_seed']}|{t['seed']}|{t['call']}|{t['layer']}|{t['module']}|{t['replicate']}"
        seed=int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'big')
        delta=matched_direction(delta,seed)
    if kind=='sham':delta=np.zeros_like(delta)
    return current,delta,dict(before=current[j],donor=donor[j],reference=ref[j],delta=delta)


def make_report(out,trials,results,completed,error,render_plots=True):
    rows=[]
    for t in trials:
        r=results.get(t['id'])
        if r is None:continue
        end=r['evaluation'];raw=end['stages'][0];repaired=end['stages'][1]
        rows.append({**t,'raw_valid_connected_fixed_chemistry':end['raw_valid_connected_fixed_chemistry'],
            'final_site_bond_order_sum':end['final_graph']['site_explicit_bond_order_sum'],
            'final_focal_bond_class':end['final_graph']['bond_class'],
            'raw_fc_valid':raw['fc_valid'],'raw_fixed_chemistry':raw['fixed_chemistry_preserved'],
            'raw_posebusters_valid':raw['posebusters_valid'],'fixed_max_displacement_A':raw.get('fixed_max_displacement_A'),
            'scaffold_changed':raw.get('murcko_scaffold_changed'),
            'repaired_fc_valid':repaired['fc_valid'],'repaired_fixed_chemistry':repaired['fixed_chemistry_preserved'],
            'delta_norm':(r['replay']['patch'] or {}).get('delta_norm',0.),
            'evaluation_errors':[e for s in end['stages'] for e in s['errors']]})
    write_csv(out/'trial_results.csv',rows)
    titles={'id':'实验','kind':'类型','call':'调用','layer':'层','module':'子模块',
        'raw_valid_connected_fixed_chemistry':'原始产物：有效/连通/固定化学身份',
        'final_site_bond_order_sum':'终态位点键级和','final_focal_bond_class':'终态焦点键',
        'raw_posebusters_valid':'原始产物三维检查','repaired_fixed_chemistry':'修复后固定化学身份',
        'delta_norm':'实际施加前的改动范数'}
    table='<table><thead><tr>'+''.join('<th>'+v+'</th>' for v in titles.values())+'</tr></thead><tbody>'
    for r in rows:
        table+='<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in titles)+'</tr>'
    table+='</tbody></table>'
    images=[]
    if rows and render_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        target=[t for t in trials if t['kind']=='targeted' and t['id'] in results]
        if target:
            fig,axes=plt.subplots(len(target),1,figsize=(11,2.5*len(target)),squeeze=False)
            for ax,t in zip(axes[:,0],target):
                base=next(x for x in trials if x['kind']=='baseline' and x['seed']==t['seed'])
                candidates=[base,t,*[x for x in trials if x['kind']=='random_direction' and all(x[k]==t[k] for k in ['seed','call','layer','module'])]]
                for v in candidates:
                    if v['id'] not in results:continue
                    logs=read_json(out/'trials'/v['id']/'network_timeline.json')
                    if v['kind']=='baseline':color,alpha,label='black',.7,'native baseline'
                    elif v['kind']=='targeted':color,alpha,label='#006bb3',1.,'targeted temporal patch'
                    else:color,alpha,label='#a5a5a5',.6,f"random {v['replicate']}"
                    ax.plot([q['call'] for q in logs],[q['site_explicit_bond_order_sum'] for q in logs],color=color,alpha=alpha,label=label)
                ax.axvline(t['call'],color='#b23d28',ls='--',lw=1)
                ax.set_title(f"seed {t['seed']}, call {t['call']}, layer {t['layer']}, {t['module']}")
                ax.set_ylabel('Site bond-order sum');ax.set_xlabel('Network call (0-based)');ax.legend(fontsize=7,ncol=3)
            fig.tight_layout();fig.savefig(out/'causal_timelines.png',dpi=150);plt.close(fig)
            images=['<img src="causal_timelines.png" style="max-width:100%">']
    page='''<!doctype html><meta charset="utf-8"><title>Scaffold hopping 因果定位初筛</title>
<style>body{font:16px system-ui;max-width:1450px;margin:35px auto;padding:0 20px;color:#223}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccd;padding:6px;vertical-align:top}th{background:#e9f0f5}p{line-height:1.6}</style>
<h1>Scaffold hopping：第一轮因果定位</h1>'''
    page+=f'<p>运行完整：{completed}；已完成实验 {len(rows)} / {len(trials)}。错误：{html.escape(str(error))}</p>'
    page+='<p>每次从同一原始生成条件出发，只在一个调用、一个子模块、一个生成原子的向量更新上修改一次，然后运行到终态。所有实验匹配原始随机数流。原始全轨迹与零改动对照必须先通过。</p>'
    page+='<p><b>解释边界：</b>“原始产物有效/连通/固定化学身份”只是一项化学终态标准。三维检查、固定原子位移、骨架是否变化单列；不能合称全部任务成功。None 表示未能评价。修复后有效但固定碳变硫，不算保留固定片段。</p>'
    page+='<p>靶向修改优于同等大小的随机方向，且影响持续到终态，才支持该内部写入对当前失败具有因果影响。单个成功不证明唯一根因；没有成功也不能排除上游或多模块作用。同轨迹前一时刻是时间参考，不是独立健康轨迹。3 个随机对照只是探索性筛查。</p>'
    page+=table+''.join(images)
    (out/'REPORT.html').write_text(page,encoding='utf-8')


def bundle(out):
    with zipfile.ZipFile(out/'causal_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='causal_review.zip' and not p.name.endswith('.tmp'):
                z.write(p,p.relative_to(out))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,default=Path(__file__).with_name('causal_plan.json'))
    ap.add_argument('--submodule-dir',type=Path,default=Path('outputs/sh_submodule_localization_ptp1b'))
    ap.add_argument('--readout-dir',type=Path,default=Path('outputs/sh_coordinate_source_ptp1b'))
    ap.add_argument('--output-root',type=Path,default=Path('outputs'))
    ap.add_argument('--output-dir',type=Path,default=Path('outputs/sh_causal_localization_ptp1b_v2'))
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();out=args.output_dir.resolve()
    if out.exists() and not args.resume:ap.error('Output exists. Use --resume for an interrupted identical run, or a new output directory.')
    if args.resume and not (out/'run_identity.json').is_file():ap.error('No verified resume identity. Preserve this folder and choose a new output directory.')
    out.mkdir(parents=True,exist_ok=True)
    summary=dict(version=VERSION,completed=False,error=None,completed_trials=0);results={};trials=[]
    try:
        require((Path.cwd()/'flowr/scriptutil.py').is_file(),'Run in the existing FLOWR.ROOT project.')
        sys.path.insert(0,str(Path.cwd()))
        import torch
        from flowr.scriptutil import load_model
        from flowr.util.device import resolve_device
        helpers=import_file(Path.cwd()/'replay_connection_hidden.py','causal_replay_helpers',HELPER_SHA)
        quality=import_file(Path(__file__).with_name('evaluate_scaffold_outputs.py'),'causal_quality_helpers')
        plan=read_json(args.plan)
        print('Verify native sources, prior trajectories and observed submodule records ...',flush=True)
        data,records,trials,identity,readout=prepare(args,plan,torch,helpers,quality)
        if args.resume:require(read_json(out/'run_identity.json')==identity,'Resume identities differ; use a new output directory.')
        else:write_json(out/'run_identity.json',identity)
        write_json(out/'planned_trials.json',trials);summary['planned_trials']=len(trials)
        base=data[plan['failure']['seed']]['manifest']
        torch.set_float32_matmul_precision(readout['precision'])
        model_args=argparse.Namespace(**base['args']);model_args.save_dir=str(out/'model_context')
        device=resolve_device(model_args)
        print(f'Loading frozen pretrained model on {device}. Planned full rollouts: {len(trials)}',flush=True)
        model=load_model(model_args)[0].to(device).eval()
        require(sum(p.requires_grad for p in model.parameters())==readout['parameter_tensors_requiring_grad'],'Native parameter flags differ.')
        for d in data.values():
            require(bool(model.self_condition)==d['inputs']['self_condition'] and int(model.sc_recycle_steps)==d['inputs']['sc_recycle_steps'],'Self-conditioning settings differ.')
            require(float(model.coord_scale)==d['inputs']['coord_scale'],'Coordinate units differ.')
        for n,t in enumerate(trials):
            summary['active_trial']=t['id']
            folder=out/'trials'/t['id'];folder.mkdir(parents=True,exist_ok=True)
            done=folder/'result.json'
            if args.resume and done.is_file():
                saved=read_json(done)
                require(saved['trial']==t and saved['completed'],'Resume trial differs.')
                for f,sha in saved['file_sha256'].items():require(digest(folder/f)==sha,f'Resume output changed: {f}')
                results[t['id']]=saved
                print(f"[{n+1}/{len(trials)}] Reuse verified {t['id']}",flush=True);continue
            # Baselines and sham precede every nontrivial intervention.
            if t['kind'] not in {'baseline','sham'}:
                gates=[v['id'] for v in trials if v['kind'] in {'baseline','sham'}]
                require(all(v in results and results[v]['replay']['baseline_or_sham_full_match'] for v in gates),'Full native baseline/sham gate is incomplete.')
            print(f"[{n+1}/{len(trials)}] {t['id']}: run native sampling to final output",flush=True)
            write_json(folder/'trial.json',t)
            reference,delta,vectors=trial_vectors(t,records,plan)
            if vectors:np.savez_compressed(folder/'patch_vectors.npz',**vectors)
            final,replay,logs,sampler=run_rollout(model,data[t['seed']],t,reference,delta,helpers,device,torch,debug_dir=folder)
            torch.save(final,folder/'final_prediction.pt')
            write_json(folder/'network_timeline.json',logs);write_json(folder/'sampler_timeline.json',sampler)
            write_json(folder/'replay_checks.json',replay)
            evaluation=evaluate_final(final,model,data[t['seed']],t,folder,quality,torch)
            saved=dict(completed=True,trial=t,replay=replay,evaluation=evaluation,
                file_sha256={p.name:digest(p) for p in folder.iterdir() if p.is_file() and p.name!='result.json'})
            write_json(done,saved);results[t['id']]=saved
            summary['completed_trials']=len(results);write_json(out/'causal_summary.json',summary)
            make_report(out,trials,results,False,None,render_plots=False)
            print(f"  Finished; raw chemical endpoint: {evaluation['raw_valid_connected_fixed_chemistry']}",flush=True)
            del final,reference,delta
        summary['completed']=len(results)==len(trials)
        summary['active_trial']=None
        summary['completed_trials']=len(results)
        summary['evaluation_errors']=[{'trial':k,'stage':s['stage'],'errors':s['errors']}
            for k,r in results.items() for s in r['evaluation']['stages'] if s['errors']]
        summary['scope']=plan['scope']
    except Exception as exc:
        summary['error']=f'{type(exc).__name__}: {exc}'
        (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print(f"STOP: {summary['error']}",flush=True)
    finally:
        summary['completed_trials']=len(results)
        write_json(out/'causal_summary.json',summary)
        try:make_report(out,trials,results,summary['completed'],summary['error'])
        except Exception:
            (out/'report_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        bundle(out)
        print(f"Completed: {summary['completed']}; trials {len(results)}/{len(trials)}",flush=True)
        print(f"Review file: {out/'causal_review.zip'}",flush=True)
    return 0 if summary['completed'] else 1


if __name__=='__main__':raise SystemExit(main())
