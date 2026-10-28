#!/usr/bin/env python3
"""Paired native terminal-inpainting control on saved FLOWR predictions.

No new sampling, neural forward passes, weight loading or parameter changes.
All baseline tensor/decoder/quality gates must pass before the ON arm runs.
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
           'root_fixed_substructure_match': None, 'fixed_chemistry_preserved': None,
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


def changes(before, after, prior):
    import torch
    fixed = prior['fragment_mask'][0].bool() & prior['mask'][0].bool()
    valid = prior['mask'][0].bool()
    pair_fixed = fixed[:, None] & fixed[None, :]
    pair_valid = valid[:, None] & valid[None, :] & torch.tril(torch.ones_like(pair_fixed), diagonal=-1)
    edge_change = before['bonds'][0].argmax(-1) != after['bonds'][0].argmax(-1)
    generated = valid & ~fixed
    return {'fixed_atom_indices': fixed.nonzero().flatten().tolist(),
            'changed_fixed_internal_bond_count': int((edge_change & pair_fixed & pair_valid).sum()),
            'changed_other_bond_count': int((edge_change & ~pair_fixed & pair_valid).sum()),
            'all_bond_probability_entries_changed': int((before['bonds'] != after['bonds']).sum()),
            'generated_coordinate_max_change_A': float((before['coords'][0,generated]-after['coords'][0,generated]).abs().max()) if generated.any() else 0,
            'generated_atom_classes_changed': int(((before['atomics'][0].argmax(-1)!=after['atomics'][0].argmax(-1)) & generated).sum()),
            'intervention_scope': 'Native inpaint_molecule including global upper-triangle argmax bond symmetrization; no component ablations.'}


def save_mol(path, mol):
    from rdkit import Chem
    with Chem.SDWriter(str(path)) as w:
        if mol is not None:
            w.write(mol)


def aggregate(records):
    result = {}
    for stage in ('raw_without_repair','run_decode'):
        result[stage] = {}
        for metric in ('rdkit_valid','fc_valid','fixed_chemistry_preserved','root_fixed_substructure_match','posebusters_valid','murcko_scaffold_changed'):
            pairs = [(r['off'][stage][metric], r['on'][stage][metric]) for r in records]
            result[stage][metric] = {'off_pass':sum(a is True for a,b in pairs), 'on_pass':sum(b is True for a,b in pairs),
                 'off_fail':sum(a is False for a,b in pairs), 'on_fail':sum(b is False for a,b in pairs),
                 'off_unknown':sum(a is None for a,b in pairs), 'on_unknown':sum(b is None for a,b in pairs),
                 'fail_to_pass':sum(a is False and b is True for a,b in pairs),
                 'pass_to_fail':sum(a is True and b is False for a,b in pairs), 'n_pairs':len(pairs)}
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path.cwd())
    p.add_argument('--quality-report',type=Path,default=Path('outputs/sh_discovery_cohort_v2/quality/quality_report.json'))
    p.add_argument('--diagnostic-dir',type=Path,default=Path('outputs/sh_discovery_cohort_v2_diagnostics'))
    p.add_argument('--records-root',type=Path,default=Path('outputs/sh_discovery_cohort_v2/runs'))
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv);repo=a.repo.resolve();out=a.output_dir.resolve()
    if out.exists():p.error('Output exists; use a new output directory.')
    sys.path.insert(0,str(repo))
    files=EvidenceFiles(repo)
    relative=lambda x: x if x.is_absolute() else repo/x
    out.mkdir(parents=True)
    status={'completed':False,'phase':'preflight','baseline_runs_passed':0,'paired_runs_completed':0,'errors':[],'new_generations':0,'neural_interventions':0}
    records=[]
    try:
        import numpy as np
        import torch
        import rdkit
        from rdkit import Chem
        from flowr.models.mol_builder import MolBuilder
        from flowr.constants import CORE_ATOMS
        from flowr.scriptutil import _build_vocab, _build_vocab_charges, _build_vocab_hybridization
        torch.set_num_threads(1)
        evaluator_path=repo/'scaffold_evaluation_v2/evaluate_scaffold_outputs.py'
        spec=importlib.util.spec_from_file_location('native_evaluator',evaluator_path)
        evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
        q=files.json(relative(a.quality_report));diag_dir=relative(a.diagnostic_dir)
        ds=files.json(diag_dir/'summary.json')
        require(q.get('completed') is True and not q['errors'] and len(q['runs'])==q['requested_run_count']==100,'Expected the completed 100-run cohort')
        require(ds['completed'] is True and ds['quality_report_sha256']==sha(files.resolve(relative(a.quality_report))),'Diagnostic report does not match quality report')
        require(rdkit.__version__==q['versions']['rdkit'],'RDKit version mismatch')
        for f,h in q['source_sha256'].items():require(sha(repo/f)==h,f'Evaluator dependency changed: {f}')
        manifests={}
        for item in ds['records']:
            require(item['analysis_completed'] is True,'Incomplete diagnosis')
            d=files.json(diag_dir/item['report_directory']/'connection_report.json')
            require(d['run'] not in manifests,'Duplicate diagnostic run')
            manifests[d['run']]=d['manifest']
        require(set(manifests)=={r['run_directory'] for r in q['runs']},'Run inventories disagree')
        rows={}
        for r in q['rows']:
            key=(r['run_directory'],r['stage']);require(key not in rows and not r.get('errors'),'Duplicate/error quality row');rows[key]=r
        # Lock evidence and scientific comparison rules before any branch is executed.
        write_json(out/'protocol.json',{'n_runs':100,'selection':'All existing cohort records, both decodes, no filtering',
            'off':'Saved terminal prediction with native coordinate restoration and decoder',
            'on':'Same prediction passed through native MolBuilder.inpaint_molecule, then identical restoration and decoder',
            'baseline_gate':'All 100 baseline tensors and both endpoint decodes must reproduce before ON runs',
            'coordinate_atol_A':COORD_ATOL_A,'other_molecular_tensors':'exact equality and exact categorical decisions',
            'stop':'Finish these 100 pairs only; no hidden-module scan or added generation',
            'scope':'Finite terminal-processing intervention; not neural mechanistic attribution',
            'code_sha256':{str(f.relative_to(repo)):sha(f) for f in [Path(__file__),evaluator_path,repo/'flowr/models/mol_builder.py',repo/'flowr/models/fm_pocket.py',repo/'flowr/util/functional.py']}})

        def prepare(info):
            run_id=f"{info['case']}_lig{info['ligand_idx']}_seed_{info['seed']}"
            folder=relative(a.records_root)/f"{info['case']}_lig{info['ligand_idx']}"/f"seed_{info['seed']}"/'research_record/attempt_000'
            m=manifests[info['run_directory']];args=m['args']
            require(args['final_inpaint'] is False and args['scaffold_hopping'] is True and args['integration_steps']==100 and args['corrector_iters']==0,'Unsupported recorded generator settings')
            require(not args.get('optimize_gen_ligs') and not args.get('optimize_gen_ligs_hs'),'Optimized baseline unsupported')
            require(args['seed']==info['seed'] and args['ligand_idx']==info['ligand_idx'],'Input identity mismatch')
            for f,h in m['source_sha256'].items():require(sha(repo/f)==h,f'Generator source changed: {f}')
            for name,key in [('raw_final_prediction.pt','raw_final_sha256'),('task_definition.json','task_sha256'),('reference_preprocessed.sdf','reference_sdf_sha256')]:
                require(sha(files.resolve(folder/name))==info[key],f'Record checksum mismatch: {name}')
            inputs=files.tensor(folder/'inputs.pt');raw=files.tensor(folder/'raw_final_prediction.pt')
            frame=files.tensor(folder/'sampler_steps/step_0100.pt');task=files.json(folder/'task_definition.json')
            require(inputs['final_inpaint'] is False and frame['step_index']==100 and frame['stage']=='final_corrector_prediction','Unexpected terminal snapshot semantics')
            require(task['mapping_verified'] is True,'Unverified fixed atom mapping')
            fixed=(inputs['prior']['fragment_mask'][0].bool() & inputs['prior']['mask'][0].bool()).nonzero().flatten().tolist()
            require(fixed==task['fixed_sampler_indices'],'Fixed mask mismatch')
            require(inputs['prior']['fragment_mode']==['scaffold_hopping'],'Unexpected inpainting mode')
            require(raw['atomics'].shape[-1] in (len(CORE_ATOMS)+1,len(CORE_ATOMS)+2),'Unrecognized atom vocabulary')
            builder=MolBuilder(_build_vocab(virtual_nodes=raw['atomics'].shape[-1]==len(CORE_ATOMS)+2),
                _build_vocab_charges(),vocab_hybridization=_build_vocab_hybridization(),n_workers=1,
                ligand_valence_repair=args['ligand_valence_repair'],
                ligand_valence_repair_max_edits=args['ligand_valence_repair_max_edits'],
                ligand_valence_repair_top_k=args['ligand_valence_repair_top_k'],
                ligand_valence_repair_max_states=args['ligand_valence_repair_max_states'],
                ligand_valence_repair_allow_bond_deletion=args['ligand_valence_repair_allow_bond_deletion'])
            ref=evaluator.read_one_sdf(files.resolve(folder/'reference_preprocessed.sdf'),Chem);Chem.SanitizeMol(ref)
            protein=repo/'examples'/Path(args['pdb_file']).name
            require(sha(files.resolve(protein))==m['input_sha256'][args['pdb_file']],'Protein input changed')
            prior_world=restore_frame(builder,inputs['prior'],inputs)
            delta=np.max(np.abs(prior_world['coords'][0,fixed].numpy()-ref.GetConformer().GetPositions()[task['fixed_reference_indices']]))
            require(delta<=1e-3,'Initial fixed coordinates do not match reference mapping')
            return run_id,builder,inputs,raw,frame['current'],task,ref,files.resolve(protein)

        status['phase']='baseline_validation'
        for idx,info in enumerate(q['runs'],1):
            run_id,builder,inputs,raw,pred,task,ref,protein=prepare(info)
            world=restore_frame(builder,pred,inputs)
            tensor_checks=compare_prediction(world,raw)
            dest=out/run_id;dest.mkdir()
            off={}
            for stage,repair in [('raw_without_repair',False),('run_decode',None)]:
                builder.reset_repair_stats()
                row,mol=evaluate(builder,world,ref,task,protein,repair,evaluator)
                compare_endpoint(row,rows[info['run_directory'],stage])
                off[stage]=row;save_mol(dest/f'off_{stage}.sdf',mol)
            write_json(dest/'baseline.json',{'run_id':run_id,'tensor_checks':tensor_checks,'off':off})
            status['baseline_runs_passed']=idx;write_json(out/'status.json',status)
            print(f'BASELINE {idx}/100 {run_id}: passed',flush=True)
        status['phase']='paired_terminal_control'
        for idx,info in enumerate(q['runs'],1):
            run_id,builder,inputs,raw,pred,task,ref,protein=prepare(info)
            before=copy.deepcopy(pred)
            patched=builder.inpaint_molecule(copy.deepcopy(inputs['prior']),copy.deepcopy(pred),keep_interactions='interactions' in pred)
            # Neither the OFF snapshot nor its reusable prior may be mutated.
            for key in FIELDS:require(torch.equal(before[key],pred[key]),'OFF snapshot mutated')
            world=restore_frame(builder,patched,inputs);off_world=restore_frame(builder,pred,inputs)
            prior_world=restore_frame(builder,inputs['prior'],inputs)
            fixed=(inputs['prior']['fragment_mask'] & inputs['prior']['mask']).bool()
            for key in ('coords','atomics','charges'):
                require(torch.equal(world[key][fixed],prior_world[key][fixed]),f'Native inpaint did not restore {key}')
            dest=out/run_id;baseline=json.loads((dest/'baseline.json').read_text());on={}
            for stage,repair in [('raw_without_repair',False),('run_decode',None)]:
                builder.reset_repair_stats()
                row,mol=evaluate(builder,world,ref,task,protein,repair,evaluator)
                on[stage]=row;save_mol(dest/f'on_{stage}.sdf',mol)
            torch.save(world,dest/'on_terminal_prediction.pt')
            record={'run_id':run_id,'input_id':f"{info['case']}_lig{info['ligand_idx']}",'seed':info['seed'],
                    'off':baseline['off'],'on':on,'tensor_changes':changes(off_world,world,inputs['prior'])}
            write_json(dest/'paired_result.json',record);records.append(record)
            status['paired_runs_completed']=idx;write_json(out/'status.json',status)
            print(f'PAIR {idx}/100 {run_id}: PB {record["off"]["raw_without_repair"]["posebusters_valid"]} -> {on["raw_without_repair"]["posebusters_valid"]}',flush=True)
        status.update(completed=True,phase='complete')
        write_json(out/'results.json',{'completed':True,'aggregate':aggregate(records),'records':records,'neural_root_cause':'not_established'})
        write_csv(out/'paired_summary.csv',[{'run_id':r['run_id'],'input_id':r['input_id'],'seed':r['seed'],'decode':stage,
            **{f'{arm}_{k}':r[arm][stage].get(k) for arm in ('off','on') for k in ('rdkit_valid','fc_valid','fixed_chemistry_preserved','fixed_max_displacement_A','posebusters_valid','posebusters_failed_checks','murcko_scaffold_changed')}} for r in records for stage in ('raw_without_repair','run_decode')])
    except Exception as exc:
        status['errors'].append({'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()})
        print(status['errors'][-1]['traceback'],file=sys.stderr)
    write_json(out/'status.json',status)
    write_json(out/'input_hashes.json',files.hashes)
    with zipfile.ZipFile(out/'terminal_control_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.suffix in ('.json','.csv','.sdf'):z.write(f,f.relative_to(out))
    print(f'Completed: {status["completed"]}; baseline {status["baseline_runs_passed"]}/100; pairs {status["paired_runs_completed"]}/100',flush=True)
    return 0 if status['completed'] else 1


if __name__=='__main__':raise SystemExit(main())
