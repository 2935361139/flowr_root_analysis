"""Native replay/evaluation adapter reused from the verified causal runner."""
from pathlib import Path
from collections.abc import Mapping
from types import MethodType, SimpleNamespace
import hashlib,importlib.util,json,csv,random,traceback
import numpy as np
MISSING_LABEL_PATHS=frozenset({'input.affinity.pic50','input.affinity.pkd','input.affinity.pki','input.affinity.pec50','input.docking_score.vina_score','input.docking_score.gnina_score','input.docking_score.glide_score'})


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
    def scalar(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        raise TypeError(f'Unsupported JSON value: {type(v).__name__}')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=scalar),encoding='utf-8')
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
        pair_margin_value=float(d[max(s,j),min(s,j),focus.get('target_bond_class',1)]-d[max(s,j),min(s,j),focus.get('reference_bond_class',0)]) if valid else None,
        distance_A=float(np.linalg.norm(coords[j]-coords[s])) if valid else None,
        site_explicit_bond_order_sum=float(order[s].sum()),site_neighbors=neighbors,
        site_neutral_carbon_overvalent=(bool(order[s].sum()>4+1e-6) if keep[s] and atoms[s]=='C' and charge==0 and not np.any(bond[s]==4) else None),
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


def run_rollout(model,data,trial,patch_factory,helpers,device,torch,debug_dir=None):
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
        if patch_factory is not None:
            patch=patch_factory(model,trial,clock,torch)
        helpers.restore_rng(inputs['rng'],device,torch,np)
        with torch.no_grad():
            final=model._generate(prior,pocket,steps=cfg['integration_steps'],times=move(inputs['times']),
                strategy=inputs['strategy'],solver=inputs['solver'],corr_iters=cfg['corrector_iters'],
                save_traj=bool(cfg['save_traj']),final_inpaint=inputs['final_inpaint'],
                apply_guidance=False,coord_noise_level=cfg['coord_noise_scale'])
        require(clock['call']+1==len(calls),'Incomplete native trajectory.')
        if patch:patch.validate_complete()
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
    chemical=(False if not raw['fc_valid'] else
              None if raw['root_fixed_substructure_match'] is None else bool(raw['root_fixed_substructure_match']))
    return dict(raw_valid_connected_root_fragment=chemical,stages=rows,
        scope='Native valid+connected+ROOT fixed-fragment match endpoint; slot diagnostics, 3D quality and scaffold change are separate.',
        final_graph={'categorical_values':'final_probabilities',**graph_metrics(final,model.builder,trial,1.0)})
