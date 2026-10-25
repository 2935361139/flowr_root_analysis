#!/usr/bin/env python3
"""Prepare/run a bounded NEW-INPUT baseline cohort. No interventions or training.

Use from the FLOWR.ROOT repository. Selection uses input structures and old
input manifests only, never generated outputs. The three helpers are unchanged.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import traceback
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION = '2026-10-23.new-input-baselines.1'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def package_hashes():
    names = ['run_batch.py', 'batch_config.json', 'helpers/record_scaffold_run.py',
             'helpers/inspect_scaffold_trace.py', 'helpers/quality.py']
    return {name: sha(HERE / name) for name in names}


def check_sources(repo, config):
    for name, expected in config['source_sha256'].items():
        require((repo / name).is_file() and sha(repo / name) == expected,
                f'FLOWR source missing or changed: {name}. Do not remove this check.')
    for name, expected in config['helper_sha256'].items():
        require(sha(HERE / name) == expected, f'Packaged helper changed: {name}')


def heavy_mol(mol, Chem):
    require(mol is not None, 'Unreadable molecule')
    result = Chem.RemoveHs(Chem.Mol(mol))
    Chem.SanitizeMol(result)
    return result


def canonical(mol, Chem):
    copy = heavy_mol(mol, Chem)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True)


def input_profile(mol, Chem, extract_scaffolds):
    """Input-only eligibility; preview indices must NEVER be treated as sampler slots."""
    mol = heavy_mol(mol, Chem)
    require(mol.GetNumAtoms() > 0 and len(Chem.GetMolFrags(mol)) == 1, 'Input must be a connected ligand')
    require(mol.GetNumConformers() > 0 and mol.GetConformer().Is3D(), 'Missing 3D conformer')
    require(all(math.isfinite(float(v)) for row in mol.GetConformer().GetPositions() for v in row), 'Nonfinite coordinates')
    mask = extract_scaffolds([mol], invert_mask=True)[0].tolist()
    fixed = {i for i, flag in enumerate(mask) if bool(flag)}
    require(0 < len(fixed) < mol.GetNumAtoms(), 'No fixed fragment or no replaceable scaffold')
    anchors = set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (i in fixed) != (j in fixed):
            anchors.add(i if i in fixed else j)
    pairs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in fixed and j in fixed and (i in anchors or j in anchors):
            pairs.append({'input_atom_indices': [i, j], 'bond_type': str(bond.GetBondType()),
                          'elements': [mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(j).GetSymbol()]})
    require(pairs, 'No fixed internal bond adjacent to a scaffold attachment; outside this mechanism family')
    return {'input_smiles': canonical(mol, Chem), 'n_heavy_atoms': mol.GetNumAtoms(), 'n_fixed_preview': len(fixed),
            'boundary_adjacent_fixed_bonds_preview': pairs,
            'index_note': 'Input-heavy-atom preview only; actual reference/sampler mapping comes from task_definition.json.'}


def previous_inputs(repo, out, config, Chem):
    used = {tuple(v) for v in config['explicit_excluded_inputs']}
    graphs, manifests, errors = set(), [], []
    scan_root = repo / config['existing_records_root']
    if scan_root.exists():
        for path in sorted(scan_root.rglob('research_record/manifest.json')):
            if out in path.resolve().parents:
                continue
            try:
                args = read_json(path)['args']
                case = Path(args['pdb_file']).stem.removesuffix('_protein')
                idx = args['ligand_idx']
                require(type(idx) is int and idx >= 0, 'Unresolved ligand index')
                used.add((case, idx))
                manifests.append({'path': str(path), 'sha256': sha(path), 'case': case, 'ligand_idx': idx})
                ref = path.parent / 'attempt_000/reference_preprocessed.sdf'
                if ref.is_file():
                    supplier = Chem.SDMolSupplier(str(ref), sanitize=True, removeHs=False)
                    require(len(supplier) == 1 and supplier[0] is not None, 'Unreadable old reference input')
                    graphs.add(canonical(supplier[0], Chem))
            except Exception as exc:
                errors.append({'path': str(path), 'error': f'{type(exc).__name__}: {exc}'})
    return used, graphs, manifests, errors


def choose_inputs(candidates, case_order, n_inputs, used, seen_graphs):
    queues = {case: [] for case in case_order}
    for item in sorted(candidates, key=lambda x: (x['case'], x['ligand_idx'])):
        if (item['case'], item['ligand_idx']) not in used and item['input_smiles'] not in seen_graphs:
            queues[item['case']].append(item)
    selected, graphs = [], set(seen_graphs)
    while len(selected) < n_inputs:
        progressed = False
        for case in case_order:
            while queues[case]:
                item = queues[case].pop(0)
                if item['input_smiles'] in graphs:
                    continue
                selected.append(item)
                graphs.add(item['input_smiles'])
                progressed = True
                break
            if len(selected) == n_inputs:
                break
        if not progressed:
            break
    return selected


def prepare(repo, out, config):
    require(not out.exists(), f'Output already exists: {out}. Use --run for a frozen plan; do not overwrite.')
    check_sources(repo, config)
    sys.path.insert(0, str(repo))
    import torch
    from rdkit import Chem, rdBase
    from flowr.data.interpolate import extract_scaffolds
    checkpoint = repo / config['checkpoint']
    require(checkpoint.is_file(), f'Missing checkpoint: {checkpoint}')
    checkpoint_hash = sha(checkpoint)
    require(checkpoint_hash == config['checkpoint_sha256'], 'Checkpoint differs from the discovery experiment')
    used, graphs, manifests, history_errors = previous_inputs(repo, out, config, Chem)
    candidates, inventory, input_hashes = [], [], {}
    for case in config['case_order']:
        sdf, protein = repo / f'examples/{case}_ligands.sdf', repo / f'examples/{case}_protein.pdb'
        if not sdf.is_file() or not protein.is_file():
            inventory.append({'case': case, 'status': 'input_files_missing'})
            continue
        input_hashes[str(sdf)], input_hashes[str(protein)] = sha(sdf), sha(protein)
        for idx, mol in enumerate(Chem.SDMolSupplier(str(sdf), sanitize=True, removeHs=False)):
            item = {'case': case, 'ligand_idx': idx}
            try:
                if (case, idx) in used:
                    if mol is not None:
                        graphs.add(canonical(mol, Chem))
                    item['status'] = 'previously_used_input'
                else:
                    item.update(input_profile(mol, Chem, extract_scaffolds))
                    item['status'] = 'eligible_before_graph_deduplication'
                    candidates.append(dict(item))
            except Exception as exc:
                item.update(status='ineligible_input', reason=str(exc))
            inventory.append(item)
    selected = choose_inputs(candidates, config['case_order'], config['n_inputs'], used, graphs)
    for item in inventory:
        if item.get('status') == 'eligible_before_graph_deduplication' and item['input_smiles'] in graphs:
            item['status'] = 'previously_used_ligand_graph'
    out.mkdir(parents=True)
    write_json(out / 'input_inventory.json', {'version': VERSION, 'inventory': inventory, 'selected_inputs': selected,
               'excluded_input_ids': [list(v) for v in sorted(used)], 'old_reference_graphs': len(graphs),
               'history_manifests': manifests, 'history_errors': history_errors,
               'scope': 'Unused in discoverable local records; not a pretraining-data holdout or necessarily new protein targets.'})
    if history_errors or len(selected) != config['n_inputs']:
        write_json(out / 'batch_status.json', {'phase': 'preparation_blocked', 'gpu_runs': 0,
                   'reason': 'Ambiguous input history' if history_errors else 'Insufficient unused eligible ligand graphs',
                   'requested_inputs': config['n_inputs'], 'available_inputs': len(selected)})
        bundle(out)
        raise RuntimeError(f"Preparation blocked; no GPU runs. Review {out / 'validation_review.zip'}")
    selected_hashes = {}
    for item in selected:
        for suffix in ('_ligands.sdf', '_protein.pdb'):
            path = str(repo / 'examples' / (item['case'] + suffix))
            selected_hashes[path] = input_hashes[path]
    plan = {'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'repo': str(repo), 'output': str(out), 'config': config, 'selected_inputs': selected,
            'seeds': config['seeds'], 'planned_baseline_runs': len(selected) * len(config['seeds']),
            'torch_version': str(torch.__version__), 'rdkit_version': rdBase.rdkitVersion,
            'checkpoint_sha256': checkpoint_hash, 'input_sha256': selected_hashes, 'package_sha256': package_hashes(),
            'selection_rule': 'Input-only eligibility; exclude used inputs/graphs; target round-robin and ascending SDF index.',
            'scope': 'Baseline cohort for later mechanism validation; NO interventions in this program.',
            'stopping_rule': 'Only these inputs/seeds. No extra search when the desired failure is absent.'}
    write_json(out / 'frozen_plan.json', plan)
    (out / 'frozen_plan.sha256').write_text(sha(out / 'frozen_plan.json') + '\n')
    write_json(out / 'batch_status.json', {'phase': 'prepared', 'gpu_runs': 0, 'planned_baseline_runs': plan['planned_baseline_runs']})
    bundle(out)
    for item in selected:
        print(f"INPUT: {item['case']} ligand_idx={item['ligand_idx']}; fixed-atom preview={item['n_fixed_preview']}", flush=True)
    print(f"PREPARED: {len(selected)} inputs x {len(plan['seeds'])} seeds = {plan['planned_baseline_runs']} baseline runs; 0 interventions.", flush=True)


def jobs_for(plan, out):
    return [{'id': f"{x['case']}_lig{x['ligand_idx']}_seed{seed}", 'case': x['case'],
             'ligand_idx': x['ligand_idx'], 'seed': seed,
             'run_directory': str(out / 'runs' / f"{x['case']}_lig{x['ligand_idx']}" / f'seed_{seed}')}
            for x in plan['selected_inputs'] for seed in plan['seeds']]


def validate_record(job, plan):
    root = Path(job['run_directory']) / 'research_record'
    manifest, summary = read_json(root / 'manifest.json'), read_json(root / 'summary.json')
    args = manifest['args']
    require(args['seed'] == job['seed'] and args['ligand_idx'] == job['ligand_idx'], 'Record identity mismatch')
    require(Path(args['pdb_file']).stem == job['case'] + '_protein' and args.get('scaffold_hopping') is True, 'Recorded task mismatch')
    require(manifest['checkpoint_sha256'] == plan['checkpoint_sha256'], 'Recorded checkpoint differs')
    require(manifest['torch_version'] == plan['torch_version'] and manifest['rdkit_version'] == plan['rdkit_version'], 'Recorded environment differs')
    require(manifest['recorder_sha256'] == plan['package_sha256']['helpers/record_scaffold_run.py'], 'Recorded observer differs')
    require(all(plan['config']['source_sha256'].get(k) == v for k, v in manifest['source_sha256'].items()), 'Recorded source differs')
    require(all(plan['input_sha256'].get(k) == v for k, v in manifest['input_sha256'].items()), 'Recorded input differs')
    require(not summary.get('pipeline_error') and summary.get('n_attempts') == 1, 'Pipeline error or incomplete generation')
    attempt = summary['attempts'][0]
    require(attempt.get('generation_completed') is True and not attempt.get('audit_error'), 'Recorder audit incomplete')
    require(read_json(root / 'attempt_000/task_definition.json').get('mapping_verified') is True, 'Unverified atom mapping')
    require(attempt.get('forward_calls') == 101, 'Unexpected native call count')
    require(len(list((root / 'attempt_000/forward_calls').glob('call_*.pt'))) == 101, 'Missing complete native-call tensors')


def stream(command, log_path, repo):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write('\nCOMMAND ' + json.dumps(command, ensure_ascii=False) + '\n')
        with subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors='replace', bufsize=1) as proc:
            try:
                for line in proc.stdout:
                    log.write(line); log.flush()
                    print(line, end='', flush=True)
                require(proc.wait() == 0, f'Command failed; read {log_path}. No automatic retry or sample substitution.')
            except BaseException:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait()
                raise


def three_way_and(values):
    if any(v is False for v in values):
        return False
    return True if all(v is True for v in values) else None


def make_summary(plan, jobs, quality):
    by_run = {str(Path(r['run_directory']).resolve()): r for r in quality.get('rows', []) if r.get('stage') == 'raw_without_repair'}
    rows = []
    for job in jobs:
        q = by_run.get(str(Path(job['run_directory']).resolve()), {})
        rows.append({**job, 'quality_assessed': bool(q) and not q.get('errors'),
                     'valid_connected_root_fragment': three_way_and([q.get('fc_valid'), q.get('root_fixed_substructure_match')]),
                     'fixed_chemistry_preserved': q.get('fixed_chemistry_preserved'),
                     'posebusters_valid': q.get('posebusters_valid'), 'posebusters_failed_checks': q.get('posebusters_failed_checks', []),
                     'fixed_internal_bond_changed_pairs': q.get('fixed_internal_bond_changed_pairs', []),
                     'failure_flags': q.get('failure_flags', []), 'scaffold_changed': q.get('murcko_scaffold_changed')})
    status = Counter('missing_or_error' if not r['quality_assessed'] else
                     'pass' if r['valid_connected_root_fragment'] is True else
                     'fail' if r['valid_connected_root_fragment'] is False else 'not_assessed' for r in rows)
    return {'phase': 'baseline_evaluation_complete' if quality.get('completed') is True else 'baseline_evaluation_incomplete',
            'planned_inputs': len(plan['selected_inputs']), 'planned_seed_runs': len(jobs),
            'distinct_protein_targets': len({x['case'] for x in jobs}), 'chemical_fragment_endpoint_counts': dict(status),
            'rows': rows, 'causal_interventions_performed': 0,
            'scope': 'Baseline preparation, NOT causal generalization. Keep all planned outcomes, including absent matching failures.'}


def bundle(out):
    with zipfile.ZipFile(out / 'validation_review.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name in ('input_inventory.json', 'frozen_plan.json', 'frozen_plan.sha256', 'batch_status.json', 'baseline_summary.json', 'baseline_labels.csv'):
            p = out / name
            if p.is_file():
                z.write(p, name)
        for parent in ('quality', 'runs', 'logs'):
            for p in sorted((out / parent).rglob('*')):
                if p.is_file() and not p.is_symlink() and p.suffix in {'.json', '.csv', '.sdf', '.log'}:
                    z.write(p, p.relative_to(out))
        for name in package_hashes():
            z.write(HERE / name, 'program/' + name)


def run(repo, out, config):
    pf = out / 'frozen_plan.json'
    require(pf.is_file(), 'No frozen plan. Run --prepare first.')
    require(sha(pf) == (out / 'frozen_plan.sha256').read_text().strip(), 'Frozen plan was edited')
    plan = read_json(pf)
    require(plan['config'] == config and plan['repo'] == str(repo) and plan['output'] == str(out), 'Plan/config/workspace mismatch')
    require(plan['package_sha256'] == package_hashes(), 'Program changed after preparation')
    check_sources(repo, config)
    for name, expected in plan['input_sha256'].items():
        require(sha(name) == expected, f'Selected input changed: {name}')
    require(sha(repo / config['checkpoint']) == plan['checkpoint_sha256'], 'Checkpoint changed')
    import torch
    from rdkit import rdBase
    require(str(torch.__version__) == plan['torch_version'] and rdBase.rdkitVersion == plan['rdkit_version'], 'Python environment changed')
    require(torch.cuda.is_available(), 'CUDA unavailable in this Python environment')
    jobs = jobs_for(plan, out)
    state = {'phase': 'running_baselines', 'planned': len(jobs), 'complete': 0, 'error': None}
    try:
        for number, job in enumerate(jobs, 1):
            state['active_job'] = job['id']; write_json(out / 'batch_status.json', state)
            run_dir = Path(job['run_directory'])
            print(f"\n[{number}/{len(jobs)}] {job['id']}", flush=True)
            if not run_dir.exists():
                run_dir.parent.mkdir(parents=True, exist_ok=True)
                stream([sys.executable, '-u', str(HERE / 'helpers/record_scaffold_run.py'),
                        '--case', job['case'], '--ligand_idx', str(job['ligand_idx']), '--seed', str(job['seed']),
                        '--ckpt_path', str(repo / config['checkpoint']), '--save_dir', str(run_dir)],
                       out / 'logs' / (job['id'] + '_record.log'), repo)
            validate_record(job, plan)
            trace = run_dir / 'research_record/attempt_000/trace_check/trace_report.json'
            if not trace.is_file():
                stream([sys.executable, '-u', str(HERE / 'helpers/inspect_scaffold_trace.py'), str(run_dir)],
                       out / 'logs' / (job['id'] + '_trace.log'), repo)
            require(read_json(trace).get('snapshot_counts_match_recorder') is True, 'Incomplete trajectory observation')
            state['complete'] = number; write_json(out / 'batch_status.json', state)
        state['phase'] = 'evaluating_baselines'; write_json(out / 'batch_status.json', state)
        qdir = out / 'quality'
        if not qdir.exists():
            stream([sys.executable, '-u', str(HERE / 'helpers/quality.py'), '--run-dirs',
                    *[job['run_directory'] for job in jobs], '--save-dir', str(qdir)], out / 'logs/quality.log', repo)
        quality = read_json(qdir / 'quality_report.json')
        summary = make_summary(plan, jobs, quality)
        write_json(out / 'baseline_summary.json', summary)
        with (out / 'baseline_labels.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary['rows'][0])); writer.writeheader()
            for row in summary['rows']:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
        require(quality.get('completed') is True and all(r['quality_assessed'] for r in summary['rows']),
                'Quality evaluation incomplete; missing values are not molecular failures')
        state.update(phase='baseline_evaluation_complete', active_job=None)
        print('BASELINES COMPLETE. This batch performed no interventions and does not prove mechanistic generalization.', flush=True)
    except BaseException as exc:
        state.update(phase='stopped_for_review', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        write_json(out / 'batch_status.json', state); bundle(out)
        print(f"REVIEW: {out / 'validation_review.zip'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true', help='Input inventory and frozen plan; no GPU inference')
    action.add_argument('--run', action='store_true', help='Execute/resume only frozen baseline jobs')
    parser.add_argument('--output', type=Path, default=Path('outputs/sh_validation_baselines_v1'))
    args = parser.parse_args()
    repo, out = Path.cwd().resolve(), args.output.resolve()
    config = read_json(HERE / 'batch_config.json')
    require((repo / 'flowr/gen/generate_from_pdb.py').is_file(), 'Run from the FLOWR.ROOT repository')
    require(out != repo and repo in out.parents, 'Output must be a separate directory inside this repository')
    require(type(config['n_inputs']) is int and config['n_inputs'] > 0, 'Invalid input count')
    require(config['seeds'] and len(set(config['seeds'])) == len(config['seeds']) and
            all(type(s) is int and 0 <= s <= 4294967295 for s in config['seeds']), 'Invalid seed list')
    prepare(repo, out, config) if args.prepare else run(repo, out, config)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(2)
