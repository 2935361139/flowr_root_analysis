#!/usr/bin/env python3
"""Unified finite geometry intervention bank on saved FLOWR predictions.

No new sampling, neural forward passes, weight loading or parameter changes.
All baseline tensor/decoder/quality gates must pass before any active arm runs.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import traceback
import zipfile
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scaffold_study_v1.io_utils import require, sha, write_json, write_csv
from scaffold_study_v1.geometry_intervention import (
    SETTINGS, PRIMARY_ARMS, environment_terms, relax_generated, successful,
    classify_rescue, route_failure, select_result)

FIELDS = ('coords', 'atomics', 'bonds', 'charges', 'mask')
COORD_ATOL_A = 1e-5  # fixed before data processing; no relative tolerance


class EvidenceFiles:
    """Read regular files or checksum-verified, already fetched Git LFS objects."""
    def __init__(self, repo):
        self.repo = Path(repo).resolve()
        self.hashes = {}

    def resolve(self, path):
        path = Path(path).resolve()
        with path.open('rb') as f:
            head = f.read(256)
        target = path
        if head.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            fields = dict(line.split(' ', 1) for line in head.decode().splitlines())
            oid = fields['oid'].removeprefix('sha256:')
            require(len(oid) == 64 and all(c in '0123456789abcdef' for c in oid), 'Invalid LFS object ID')
            target = self.repo/'.git/lfs/objects'/oid[:2]/oid[2:4]/oid
            require(target.is_file(), f'Missing LFS data for {path}. Fetch the actual file first.')
            require(target.stat().st_size == int(fields['size']) and sha(target) == oid, f'LFS checksum mismatch: {path}')
        self.hashes[str(path)] = sha(target)
        return target

    def json(self, path):
        return json.loads(self.resolve(path).read_text(encoding='utf-8-sig'))

    def tensor(self, path):
        import torch
        return torch.load(self.resolve(path), map_location='cpu', weights_only=True)


def compare_prediction(actual, expected):
    """Exact categorical probabilities/mask; absolute-only coordinate comparison."""
    import torch
    details = {}
    for key in FIELDS:
        a, b = actual[key], expected[key]
        require(a.shape == b.shape and a.dtype == b.dtype, f'{key}: shape/dtype mismatch')
        require(torch.isfinite(a).all().item() and torch.isfinite(b).all().item(), f'{key}: nonfinite tensor')
        error = float((a.to(torch.float64)-b.to(torch.float64)).abs().max())
        ok = error <= COORD_ATOL_A if key == 'coords' else torch.equal(a, b)
        require(ok, f'{key}: baseline discrepancy {error}')
        if key in ('atomics', 'bonds', 'charges'):
            require(torch.equal(a.argmax(-1), b.argmax(-1)), f'{key}: changed categorical decision')
        details[key] = {'max_abs_error': error, 'passed': True}
    return details


def restore_frame(builder, prediction, inputs):
    out = copy.deepcopy(prediction)
    out['coords'] = builder.undo_zero_com_batch(out['coords'] * inputs['coord_scale'], out['mask'], inputs['pocket_com'])
    return out


def decode(builder, prediction, repair):
    return builder.mols_from_tensors(prediction['coords'], prediction['atomics'], prediction['mask'],
        bond_dists=prediction['bonds'], charge_dists=prediction['charges'],
        hybridization_dists=prediction.get('hybridization'), sanitise=True,
        add_hs=False, valence_repair=repair)[0]


def evaluate(builder, prediction, ref, task, protein, repair, evaluator):
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from posebusters import PoseBusters
    from flowr.util.rdkit import mol_is_valid
    from flowr.gen.utils import check_substructure_match
    tokens = builder.vocab.tokens_from_indices(list(range(prediction['atomics'].shape[-1])))
    charge_tokens = builder.vocab_charges.tokens_from_indices(list(range(prediction['charges'].shape[-1])))
    charge_tokens = [int(v) if str(v).lstrip('-').isdigit() else str(v) for v in charge_tokens]
    graph = evaluator.inspect_raw_graph(prediction, tokens, charge_tokens, task, ref, np)
    mol = decode(builder, prediction, repair)
    row = {'rdkit_valid': bool(mol_is_valid(mol)), 'fc_valid': bool(mol_is_valid(mol, connected=True)),
           'root_fixed_substructure_match': None, 'fixed_chemistry_preserved': graph['fixed_chemistry_preserved'],
           'posebusters_valid': None, 'posebusters_checks': None, 'posebusters_failed_checks': [],
           'murcko_scaffold_changed': None, 'molecule_smiles': None, 'scaffold_smiles': None,
           'raw_graph': graph, 'repair_stats': copy.deepcopy(builder.repair_stats)}
    fixed = task['fixed_sampler_indices']
    if mol is None or not row['rdkit_valid']:
        row['geometry_status'] = 'not_assessed_invalid_decode'
        return row, mol
    Chem.SanitizeMol(mol)
    slots, _ = evaluator.restore_coordinates(mol, prediction, tokens, np)
    missing = sorted(set(fixed)-set(slots))
    if missing:
        row.update(fixed_missing_sites=missing, fixed_chemistry_preserved=False)
    else:
        row.update(evaluator.fixed_diagnostics(mol, ref, task, slots, np))
        row['fixed_chemistry_preserved'] = not any(row[k] for k in
            ('fixed_element_changed_sites','fixed_charge_changed_sites','fixed_internal_bond_changed_pairs'))
    row['root_fixed_substructure_match'] = bool(check_substructure_match(Chem.Mol(mol), Chem.Mol(ref), 'scaffold_hopping'))
    row['molecule_smiles'] = Chem.MolToSmiles(mol, isomericSmiles=False)
    row['scaffold_smiles'] = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=False)
    row['murcko_scaffold_changed'] = row['scaffold_smiles'] != Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(ref), isomericSmiles=False)
    row.update(evaluator.pb_metrics(mol, protein, PoseBusters, Chem))
    row['geometry_status'] = 'evaluated'
    return row, mol


BASELINE_LABELS = ('rdkit_valid','fc_valid','root_fixed_substructure_match','fixed_chemistry_preserved',
                   'posebusters_valid','posebusters_checks','murcko_scaffold_changed','molecule_smiles','scaffold_smiles')


def compare_endpoint(actual, expected):
    for key in BASELINE_LABELS:
        require(actual.get(key) == expected.get(key), f'Baseline {key} differs: new={actual.get(key)!r}, old={expected.get(key)!r}')
    for key in ('fixed_rms_displacement_A', 'fixed_max_displacement_A'):
        if key in expected:
            require(key in actual and abs(actual[key]-expected[key]) <= COORD_ATOL_A, f'Baseline {key} differs')


def save_mol(path, mol):
    from rdkit import Chem
    with Chem.SDWriter(str(path)) as w:
        if mol is not None:
            w.write(mol)


# A coordinate-only edit on explicitly selected slots; no discrete-channel edit.
def coordinate_arm(coords, target, fixed, kind, seed):
    import numpy as np
    x=np.asarray(coords); goal=np.asarray(target)
    require(x.shape==goal.shape and x.ndim==2 and x.shape[1]==3, 'Invalid coordinate shape')
    require(x.dtype.kind=='f' and np.isfinite(x).all() and np.isfinite(goal).all(), 'Invalid coordinates')
    require(len(fixed)>0 and len(set(fixed))==len(fixed) and all(0<=i<len(x) for i in fixed), 'Invalid fixed selection')
    require(kind in ('restore_fixed','random_direction'), 'Unknown coordinate arm')
    result=x.copy(); delta=goal[fixed].astype(float)-x[fixed].astype(float)
    if kind=='restore_fixed':
        result[fixed]=goal[fixed]
    else:
        rng=np.random.default_rng(seed); change=np.zeros_like(delta)
        for k,d in enumerate(delta):
            length=float(np.linalg.norm(d))
            if length==0:continue
            direction=d/length
            for _ in range(16):
                noise=rng.normal(size=3); noise-=np.dot(noise,direction)*direction
                norm=float(np.linalg.norm(noise))
                if norm>1e-12:break
            else:raise ValueError('Degenerate random direction')
            change[k]=noise*(length/norm)
        result[fixed]=(x[fixed].astype(float)+change).astype(x.dtype)
    actual=result[fixed].astype(float)-x[fixed].astype(float)
    norms=np.linalg.norm(delta,axis=1);actual_norms=np.linalg.norm(actual,axis=1)
    tolerance=16*np.finfo(x.dtype).eps*(np.linalg.norm(x[fixed].astype(float),axis=1)+norms+1.)
    require(np.all(np.abs(norms-actual_norms)<=tolerance), 'Control magnitude mismatch')
    other=np.ones(len(x),dtype=bool);other[fixed]=False
    require(np.array_equal(result[other],x[other]), 'Unselected coordinates changed')
    require(np.isfinite(result).all(), 'Nonfinite patched coordinates')
    informative=norms>tolerance
    cosines=np.divide(np.sum(actual*delta,axis=1),actual_norms*norms,
                      out=np.zeros_like(norms),where=(actual_norms*norms)>0)
    if kind=='random_direction':
        require(np.all(np.abs(cosines[informative])<=tolerance[informative]/actual_norms[informative]),
                'Control direction not orthogonal within numerical precision')
    return result,dict(kind=kind,seed=seed if kind=='random_direction' else None,
        cosines_to_target=cosines.tolist(),
        selected_slots=list(fixed),target_norms_A=norms.tolist(),actual_norms_A=actual_norms.tolist(),
        norm_rounding_tolerance_A=tolerance.tolist(),informative=(norms>tolerance).tolist(),
        only_selected_coordinates_changed=True)


def pair_checks(mol, slots, item, original_pairs, env, config):
    from rdkit import Chem
    from posebusters.modules.intermolecular_distance import check_intermolecular_distance
    require(mol is not None, 'Coordinate-only intervention unexpectedly prevented decoding')
    wanted={p['check'] for p in original_pairs};found={}
    for m in config['modules']:
        if m['function']!='intermolecular_distance':continue
        label=m['rename_outputs'].get('no_clashes','no_clashes').lower().replace(' ','_')
        if label not in wanted:continue
        result=check_intermolecular_distance(Chem.AddHs(Chem.Mol(mol),addCoords=True),env,**m.get('parameters',{}))
        found[label]={(slots[int(r['ligand_atom_id'])],int(r['protein_atom_id']))
                      for r in result['details'].to_dict('records') if bool(r['clash'])}
    require(set(found)==wanted, 'Missing native clash module')
    return [dict(check=p['check'],sampler_atom_index=int(p['sampler_atom_index']),
                 environment_atom_index=int(p['environment_atom_index']),
                 region=p['region'],still_clashes=(int(p['sampler_atom_index']),int(p['environment_atom_index'])) in found[p['check']])
            for p in original_pairs]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path.cwd())
    p.add_argument('--cohort',type=Path,default=Path('outputs/sh_restart_v1'))
    p.add_argument('--output-dir',type=Path)
    a=p.parse_args(argv);repo=a.repo.resolve();batch=(repo/a.cohort).resolve()
    out=(a.output_dir.resolve() if a.output_dir else batch/'rescue_v2')
    if out.exists():p.error('Output already exists. Preserve it; no automatic retry or overwrite.')
    out.mkdir(parents=True)
    files=EvidenceFiles(repo);records=[];conclusions=[];branch_errors=[];coverage=[]
    status=dict(completed=False,phase='preflight',baseline_runs_passed=0,active_arms_completed=0,
                errors=[],new_generations=0,neural_interventions=0)
    try:
        import numpy as np
        import torch
        import rdkit
        import scipy
        from rdkit import Chem
        import posebusters
        from posebusters import PoseBusters
        from posebusters.tools.loading import safe_load_mol
        from flowr.models.mol_builder import MolBuilder
        from flowr.constants import CORE_ATOMS
        from flowr.scriptutil import _build_vocab,_build_vocab_charges,_build_vocab_hybridization
        # Use the installed, already exercised inventory and ID mapping.
        adapter_path=repo/'scaffold_geometry_audit_v1/flowr_adapter.py'
        spec=importlib.util.spec_from_file_location('geometry_inventory',adapter_path)
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        q,items=adapter.inventory(batch,batch/'diagnostics',repo)
        require(0<len(items)<=20, 'This pilot is capped at 20 saved runs; no automatic expansion')
        require(str(torch.__version__)==q['versions']['torch'] and rdkit.__version__==q['versions']['rdkit'],'Use original Torch/RDKit')
        require(Path(posebusters.__file__).resolve().parent==repo/'posebusters','Use original PoseBusters fork')
        evaluator_path=repo/'scaffold_evaluation_v2/evaluate_scaffold_outputs.py'
        require(sha(evaluator_path)==q['script_sha256'], 'Evaluator code changed')
        spec=importlib.util.spec_from_file_location('coordinate_control_evaluator',evaluator_path)
        evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
        geometry=batch/'localization_v2/geometry'
        gs=files.json(geometry/'summary.json');gp=files.json(geometry/'protocol.json')
        require(gs['completed'] and gs['error_count']==0,'Need completed localization')
        require(gp['quality_report_sha256']==sha(batch/'quality/quality_report.json'), 'Localization belongs to another batch')
        for f,h in {**q['source_sha256'],**gp['posebusters_source_sha256']}.items():
            require(sha(repo/f)==h, f'Source changed: {f}')
        import csv
        pair_path=files.resolve(geometry/'clash_pairs.csv')
        with pair_path.open(encoding='utf-8-sig') as f:all_pairs=list(csv.DictReader(f))
        config=PoseBusters('dock').config
        require(config==gp['posebusters_config'], 'Native PB configuration changed')
        # Include every supported failure, irrespective of fixed/generated region.
        routes={item['id']:route_failure(item['rows']['raw_without_repair']) for item in items}
        eligible_inputs={item['input_id'] for item in items if routes[item['id']]['status']=='supported_geometry'}
        selected=[item for item in items if item['input_id'] in eligible_inputs
                  and routes[item['id']]['status']!='unsupported']
        require(len(selected)<=20, 'This pilot permits at most twenty active samples')
        selected_ids={item['id'] for item in selected}
        coverage=[dict(run_id=item['id'],input_id=item['input_id'],**routes[item['id']],
            selected=item['id'] in selected_ids,failed_checks=item['rows']['raw_without_repair'].get('posebusters_failed_checks',[]))
            for item in items]
        write_json(out/'coverage.json',coverage)
        excluded=[dict(run_id=item['id'],posebusters_valid=item['rows']['raw_without_repair']['posebusters_valid'],
            reason=routes[item['id']]['reason'],
            reference_conflicts=[dict(sampler_atom_index=int(r['sampler_atom_index']),
                environment_atom_index=int(r['environment_atom_index']),check=r['check'])
                for r in all_pairs if r['id']==item['id'] and r['stage']=='raw_without_repair'
                and r.get('reference_pair_comparison')=='same_fixed_atom_environment_pair_already_clashes_in_reference'])
            for item in items if item['id'] not in selected_ids]
        arms=['A','B','AB','C','UFF_only','A_UFF_only','C_UFF_only',
              'A_random_0','A_random_1','A_random_2',
              'B_random_0','B_random_1','B_random_2',
              'AB_random_0','AB_random_1','AB_random_2',
              'C_random_0','C_random_1','C_random_2']
        write_json(out/'protocol.json',dict(n_runs=len(items),arms=['baseline',*arms],
            active_runs=[item['id'] for item in selected], excluded=excluded,
            active_arms=len(selected)*len(arms),new_generations=0,neural_interventions=0,
            versions=dict(torch=str(torch.__version__),rdkit=rdkit.__version__,numpy=np.__version__,scipy=scipy.__version__),
            selection='All supported geometry failures and same-input passing controls; no case/atom/region hard coding.',
            interventions=dict(A='Restore only final fixed-atom coordinates from prior in world frame.',
                B='Environment-aware UFF relaxation of generated heavy atoms; fixed heavy atoms remain at baseline.',
                AB='Restore fixed coordinates, then perform B.',
                C='Joint environment-aware optimization of all ligand heavy coordinates, including fixed atoms. Fixed chemistry preserved; fixed-coordinate conditioning is relaxed.',
                UFF_only='B without environment repulsion.',A_UFF_only='Restore fixed coordinates, then UFF-only relaxation.',
                C_UFF_only='C without environment repulsion.'),
            optimizer_settings=SETTINGS,
            controls='Three per-atom norm-matched orthogonal random controls per intervention, plus UFF-only controls. AB controls retain A; C controls move the same heavy atoms as C.',
            primary_endpoint='All original chemistry/fixed-substructure checks and whole-molecule native PoseBusters pass.',
            secondary_endpoint='Persistence of each previously localized raw clash pair. Pair rescue is not whole-molecule rescue.',
            stop_rule='One fixed batch; no new seeds, doses, windows, reruns or hidden-layer search.',
            interpretation='Terminal-coordinate counterfactual, not full neural rollout or proof of a unique upstream cause.',
            quality_sha256=sha(batch/'quality/quality_report.json'),localization_pairs_sha256=sha(pair_path),
            code_sha256={str(f):sha(f) for f in (Path(__file__),evaluator_path,adapter_path,
                repo/'scaffold_study_v1/io_utils.py',repo/'scaffold_study_v1/geometry_intervention.py')}))
        environments={};cache={};previous_successes={}
        prior_output=batch/'rescue_v1'
        if (prior_output/'results.json').is_file():
            prior_protocol=files.json(prior_output/'protocol.json')
            prior_results=files.json(prior_output/'results.json')
            require(prior_protocol['quality_sha256']==sha(batch/'quality/quality_report.json'),
                    'Previous rescue results belong to another batch')
            require(prior_results['status']['completed'], 'Previous rescue results incomplete')
            for old in prior_results['records']:
                if old['arm'] in ('A','B','AB') and successful(old['evaluation']):
                    previous_successes.setdefault(old['run_id'],set()).add(old['arm'])
        status['previous_successes_checked']=0
        torch.set_num_threads(1)

        def prepare(item):
            folder=item['folder'];m=item['manifest'];args=m['args']
            require(args['final_inpaint'] is False and args['scaffold_hopping'] is True, 'Unsupported baseline mode')
            require(not args.get('optimize_gen_ligs') and not args.get('optimize_gen_ligs_hs'), 'Optimized baseline unsupported')
            for f,h in m['source_sha256'].items():require(sha(repo/f)==h,f'Generator source changed: {f}')
            require(sha(files.resolve(folder/'raw_final_prediction.pt'))==item['info']['raw_final_sha256'], 'Final tensor changed')
            inputs=files.tensor(folder/'inputs.pt');raw=files.tensor(folder/'raw_final_prediction.pt')
            task=item['task'];fixed=task['fixed_sampler_indices']
            mask=(inputs['prior']['fragment_mask'][0].bool() & inputs['prior']['mask'][0].bool()).nonzero().flatten().tolist()
            require(mask==fixed and inputs['final_inpaint'] is False,'Fixed mask/settings mismatch')
            require(raw['atomics'].shape[-1] in (len(CORE_ATOMS)+1,len(CORE_ATOMS)+2),'Unknown atom vocabulary')
            builder=MolBuilder(_build_vocab(virtual_nodes=raw['atomics'].shape[-1]==len(CORE_ATOMS)+2),
                _build_vocab_charges(),vocab_hybridization=_build_vocab_hybridization(),n_workers=1,
                ligand_valence_repair=args['ligand_valence_repair'],
                ligand_valence_repair_max_edits=args['ligand_valence_repair_max_edits'],
                ligand_valence_repair_top_k=args['ligand_valence_repair_top_k'],
                ligand_valence_repair_max_states=args['ligand_valence_repair_max_states'],
                ligand_valence_repair_allow_bond_deletion=args['ligand_valence_repair_allow_bond_deletion'])
            target=restore_frame(builder,inputs['prior'],inputs)
            ref=evaluator.read_one_sdf(folder/'reference_preprocessed.sdf',Chem);Chem.SanitizeMol(ref)
            require(np.max(np.abs(target['coords'][0,fixed].numpy()-ref.GetConformer().GetPositions()[task['fixed_reference_indices']]))<1e-3,'Prior/reference coordinate mismatch')
            last=len(item['report']['streams']['sampler_state']['indices'])-1
            frame=files.tensor(folder/'sampler_steps'/f'step_{last:04d}.pt')
            require(frame['stage']=='final_corrector_prediction','Unexpected terminal stage')
            compare_prediction(restore_frame(builder,frame['current'],inputs),raw)
            key=sha(item['protein'])
            if key not in environments:
                environments[key]=safe_load_mol(item['protein'],**config.get('loading',{}).get('mol_cond',{}))
                require(environments[key] is not None,'Native environment loader failed')
            pairs=[r for r in all_pairs if r['id']==item['id'] and r['stage']=='raw_without_repair']
            return builder,raw,target,task,ref,environments[key],pairs

        # Validate every baseline before any treated branch.
        for item in items:
            rid=item['id'];status['phase']='baseline_validation';status['active_run']=rid
            builder,raw,target,task,ref,env,pairs=prepare(item)
            row,mol=evaluate(builder,copy.deepcopy(raw),ref,task,item['protein'],False,evaluator)
            compare_endpoint(row,item['rows']['raw_without_repair'])
            if mol is not None and row['rdkit_valid']:
                slots,_=evaluator.restore_coordinates(mol,raw,builder.vocab.tokens_from_indices(list(range(raw['atomics'].shape[-1]))),np)
                checks=pair_checks(mol,slots,item,pairs,env,config)
            else:
                require(not pairs,'Invalid decoded baseline unexpectedly has localized geometry pairs')
                checks=[]
            require(all(x['still_clashes'] for x in checks),'Original clash pair no longer reproduces')
            dest=out/rid;dest.mkdir();save_mol(dest/'baseline.sdf',mol)
            baseline=dict(run_id=rid,input_id=item['input_id'],seed=item['seed'],arm='baseline',evaluation=row,pairs=checks)
            write_json(dest/'baseline.json',baseline);cache[rid]=baseline;records.append(baseline)
            status['baseline_runs_passed']+=1;write_json(out/'status.json',status)
            print(f"BASELINE {status['baseline_runs_passed']}/{len(items)} {rid}: passed",flush=True)

        for item in selected:
            rid=item['id'];builder,raw,target,task,ref,env,pairs=prepare(item)
            fixed=task['fixed_sampler_indices'];xyz0=raw['coords'][0].numpy()
            base_mol=decode(builder,raw,False)
            slots,_=evaluator.restore_coordinates(base_mol,raw,builder.vocab.tokens_from_indices(list(range(raw['atomics'].shape[-1]))),np)
            fixed_heavy=[i for i,s in enumerate(slots) if s in fixed]
            generated=[s for s in slots if s not in fixed]
            require(generated and len(fixed_heavy)==len(fixed), 'Invalid fixed/generated mapping')
            terms=environment_terms(base_mol,env,config)
            xyzA,receiptA=coordinate_arm(xyz0,target['coords'][0].numpy(),fixed,'restore_fixed',0)
            designs={'A':(xyzA,receiptA)}
            # Exactly six bounded optimizations. C adds movable fixed atoms to B.
            # Endpoints are never consulted while optimizing or to select another dose.
            for name,start,use_env,move_fixed in [('B',xyz0,True,False),('AB',xyzA,True,False),
                  ('UFF_only',xyz0,False,False),('A_UFF_only',xyzA,False,False),
                  ('C',xyz0,True,True),('C_UFF_only',xyz0,False,True)]:
                status.update(phase='bounded_optimization',active_run=rid,active_arm=name)
                write_json(out/'status.json',status)
                mol=Chem.Mol(base_mol)
                for i,s in enumerate(slots):mol.GetConformer().SetAtomPosition(i,tuple(float(v) for v in start[s]))
                try:
                    heavy_xyz,receipt=relax_generated(mol,[] if move_fixed else fixed_heavy,terms,use_environment=use_env)
                except Exception as exc:
                    error=dict(run_id=rid,arm=name,error=f'{type(exc).__name__}: {exc}',status='not_evaluated',
                               traceback=traceback.format_exc())
                    branch_errors.append(error);write_json(out/rid/(name+'_error.json'),error)
                    print(f"NOT EVALUATED {rid} {name}: {exc}",flush=True)
                    continue
                result=start.copy()
                # A/B keep their previous coordinate scopes; C moves all heavy slots.
                for i,s in enumerate(slots):
                    if move_fixed or s in generated:result[s]=heavy_xyz[i]
                receipt['base']='A' if name in ('AB','A_UFF_only') else 'baseline'
                receipt['fixed_coordinate_constraint_relaxed']=move_fixed
                receipt['condition_fixed_heavy_indices']=fixed_heavy
                receipt['moved_heavy_slots']=slots if move_fixed else generated
                receipt['max_fixed_change_from_baseline_A']=float(np.linalg.norm(
                    result[fixed].astype(float)-xyz0[fixed].astype(float),axis=1).max())
                if move_fixed:
                    receipt['only_generated_heavy_coordinates_changed']=False
                    receipt['optimized_heavy_indices']=receipt.pop('generated_heavy_indices')
                    receipt['coordinate_scope']='All ligand heavy atoms, including condition-fixed atoms'
                designs[name]=(result,receipt)
                print(f"OPTIMIZED {rid} {name}: {receipt['iterations']} iterations, converged={receipt['converged']}",flush=True)
            for name in ('A','B','AB','C'):
                if name not in designs:continue
                start=xyzA if name=='AB' else xyz0
                selection=fixed if name=='A' else (slots if name=='C' else generated)
                for number in range(3):
                    label=f'{name}_random_{number}'
                    seed=int.from_bytes(hashlib.sha256(('20261029:'+rid+':'+label).encode()).digest()[:8],'little')
                    result,receipt=coordinate_arm(start,designs[name][0],selection,'random_direction',seed)
                    receipt['matched_arm']=name;receipt['base']='A' if name=='AB' else 'baseline'
                    designs[label]=(result,receipt)
            evaluated={}
            for arm in arms:
                if arm not in designs:
                    if not any(e['run_id']==rid and e['arm']==arm for e in branch_errors):
                        branch_errors.append(dict(run_id=rid,arm=arm,status='not_evaluated',error='Parent intervention unavailable'))
                    continue
                status.update(phase='coordinate_intervention',active_run=rid,active_arm=arm)
                xyz,receipt=designs[arm]
                treated=copy.deepcopy(raw);treated['coords'][0]=torch.as_tensor(xyz,dtype=raw['coords'].dtype)
                for key in FIELDS:
                    if key!='coords':require(torch.equal(treated[key],raw[key]),f'Unselected field changed: {key}')
                builder.reset_repair_stats()
                try:
                    row,mol=evaluate(builder,treated,ref,task,item['protein'],False,evaluator)
                except Exception as exc:
                    error=dict(run_id=rid,arm=arm,error=f'{type(exc).__name__}: {exc}',status='not_evaluated',
                        traceback=traceback.format_exc())
                    branch_errors.append(error);write_json(out/rid/(arm+'_error.json'),error)
                    print(f"NOT EVALUATED {rid} {arm}: {exc}",flush=True)
                    continue
                for key in ('rdkit_valid','fc_valid','root_fixed_substructure_match','fixed_chemistry_preserved','murcko_scaffold_changed','molecule_smiles','scaffold_smiles'):
                    require(row[key]==cache[rid]['evaluation'][key],f'Coordinate-only arm changed chemical result: {key}')
                slots,_=evaluator.restore_coordinates(mol,treated,builder.vocab.tokens_from_indices(list(range(raw['atomics'].shape[-1]))),np)
                check=pair_checks(mol,slots,item,pairs,env,config)
                old_checks=cache[rid]['evaluation']['posebusters_checks'];new_checks=row['posebusters_checks']
                record=dict(run_id=rid,input_id=item['input_id'],seed=item['seed'],arm=arm,
                    receipt=receipt,evaluation=row,pairs=check,
                    checks_improved=[k for k,v in old_checks.items() if v is False and new_checks[k] is True],
                    checks_worsened=[k for k,v in old_checks.items() if v is True and new_checks[k] is False],
                    full_pb_rescue=cache[rid]['evaluation']['posebusters_valid'] is False and row['posebusters_valid'] is True,
                    full_success=successful(row),
                    operational_rescue=not successful(cache[rid]['evaluation']) and successful(row))
                save_mol(out/rid/(arm+'.sdf'),mol);write_json(out/rid/(arm+'.json'),record);records.append(record)
                evaluated[arm]=row
                status['active_arms_completed']+=1;write_json(out/'status.json',status)
                print(f"ARM {status['active_arms_completed']}/{len(selected)*len(arms)} {rid} {arm}: success={successful(row)}",flush=True)
            for old_arm in previous_successes.get(rid,set()):
                require(old_arm in evaluated and successful(evaluated[old_arm]),
                    f'Previously successful branch regressed: {rid} {old_arm}; old rescue_v1 output is preserved')
                status['previous_successes_checked']+=1
            conclusion=dict(run_id=rid,**classify_rescue(cache[rid]['evaluation'],evaluated),
                **select_result(cache[rid]['evaluation'],evaluated),
                C_success='C' in evaluated and successful(evaluated['C']),
                control_success={name:successful(row) for name,row in evaluated.items() if name not in PRIMARY_ARMS},
                original_failed_checks=cache[rid]['evaluation']['posebusters_failed_checks'])
            conclusions.append(conclusion);write_json(out/'conclusions.json',conclusions)
        for original,h in list(files.hashes.items()):require(sha(files.resolve(original))==h,'Evidence changed during experiment')
        status.update(completed=True,phase='finished',active_run=None,active_arm=None,
            active_arms_planned=len(selected)*len(arms),branch_error_count=len(branch_errors),
            all_planned_arms_evaluated=not branch_errors)
    except (Exception, KeyboardInterrupt) as exc:
        status['errors'].append(dict(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
        print(status['errors'][-1]['traceback'],file=sys.stderr)
    write_json(out/'status.json',status);write_json(out/'input_hashes.json',files.hashes)
    write_json(out/'branch_errors.json',branch_errors)
    # Export one selected result per sample, without sacrificing original passes.
    # This is explicitly an intervention portfolio, not one uniformly applied arm.
    import shutil
    selected_dir=out/'selected_outputs';selected_dir.mkdir(exist_ok=True)
    result_summary=[]
    coverage_by_id={r['run_id']:r for r in coverage}
    for baseline in (r for r in records if r['arm']=='baseline'):
        rid=baseline['run_id']
        rows={r['arm']:r['evaluation'] for r in records if r['run_id']==rid and r['arm'] in PRIMARY_ARMS}
        selection=select_result(baseline['evaluation'],rows)
        selection.update(run_id=rid,baseline_success=successful(baseline['evaluation']),
            treatment_scope='Terminal geometry repair; network cause not established',
            coverage=coverage_by_id.get(rid,{}),
            candidate_errors=[e['arm'] for e in branch_errors if e['run_id']==rid])
        if selection['result']=='unresolved' and coverage_by_id.get(rid,{}).get('status')=='unsupported':
            selection['result']='unsupported'
        arm=selection['selected_arm']
        if arm is not None and status['completed']:
            shutil.copy2(out/rid/(arm+'.sdf'),selected_dir/(rid+'.sdf'))
        result_summary.append(selection)
    write_json(out/'selected_results.json',result_summary)
    write_csv(out/'selected_results.csv',result_summary)
    totals=dict(completed=status['completed'],original_samples=len(result_summary),
        original_passes=sum(r['baseline_success'] for r in result_summary),
        original_failures=sum(not r['baseline_success'] for r in result_summary),
        rescued_without_C=sum(not r['baseline_success'] and r['selected_arm'] in ('A','B','AB') for r in result_summary),
        rescued_with_fixed_coordinate_relaxation=sum(not r['baseline_success'] and r['selected_arm']=='C' for r in result_summary),
        unresolved=sum(r['result']=='unresolved' for r in result_summary),
        unsupported=sum(r['result']=='unsupported' for r in result_summary),
        branch_errors=len(branch_errors),
        interpretation='Portfolio rescue count, not a new native generation success rate. C relaxes fixed-coordinate conditioning.')
    write_json(out/'summary.json',totals)
    write_json(out/'results.json',dict(status=status,records=records,conclusions=conclusions,
        selection=result_summary,coverage=coverage,branch_errors=branch_errors,neural_root_cause='not_established'))
    write_csv(out/'candidate_results.csv',[dict(run_id=r['run_id'],
        baseline_success=r['baseline_success'],
        independently_effective_candidates=r['independently_effective_candidates'],
        combination_success=r['combination_success'],combination_only=r['combination_only'],
        C_success=r['C_success'],all_successful_interventions=r['all_successful_interventions'],
        selected_arm=r['selected_arm'],fixed_coordinate_relaxation=r['fixed_coordinate_relaxation'],
        control_success=r['control_success']) for r in conclusions])
    write_csv(out/'paired_summary.csv',[dict(run_id=r['run_id'],input_id=r['input_id'],seed=r['seed'],arm=r['arm'],
        posebusters_valid=r['evaluation']['posebusters_valid'],failed_checks=r['evaluation']['posebusters_failed_checks'],
        original_pairs_remaining=sum(x['still_clashes'] for x in r['pairs']),original_pairs_total=len(r['pairs']),
        checks_improved=r.get('checks_improved',[]),checks_worsened=r.get('checks_worsened',[]),
        full_pb_rescue=r.get('full_pb_rescue',False),full_success=successful(r['evaluation']),
        operational_rescue=r.get('operational_rescue',False)) for r in records])
    with zipfile.ZipFile(out/'rescue_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.suffix in ('.json','.csv','.sdf'):z.write(f,f.relative_to(out))
    print('Completed:',status['completed']);print('Upload:',out/'rescue_review.zip')
    return 0 if status['completed'] else 1


if __name__=='__main__':raise SystemExit(main())
