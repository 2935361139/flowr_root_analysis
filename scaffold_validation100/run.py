#!/usr/bin/env python3
"""Bounded validation orchestrator. Calls installed, unchanged research programs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

import existing_input_selection as selection

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = 'outputs/sh_validation100_v1'
METHOD = HERE / 'frozen_method.json'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_hashes(base, expected):
    for name, digest in expected.items():
        p = Path(name) if Path(name).is_absolute() else base / name
        require(p.is_file() and sha(p) == digest,
                f'Missing or changed file: {p}. Preserve files; submit review package.')


def versions(repo):
    sys.path.insert(0, str(repo))
    import torch
    import rdkit
    import numpy
    import scipy
    import posebusters
    require(Path(posebusters.__file__).resolve().parent == repo / 'posebusters',
            'Expected the original local PoseBusters fork')
    require(torch.cuda.is_available(), 'CUDA is unavailable; no generation started')
    return {k: str(m.__version__) for k, m in
            [('torch', torch), ('rdkit', rdkit), ('numpy', numpy), ('scipy', scipy)]}


def package_hashes():
    names = ('run.py','frozen_method.json','existing_input_selection.py',
             'install_compat.py','terminal_control.py','compatibility_change.json')
    return {name:sha(HERE/name) for name in names}


def group_plan(inputs, seeds, out):
    require(len(inputs) == 10 and len(seeds) == 10 and len(set(seeds)) == 10,
            'This protocol requires exactly 10 inputs and 10 seeds')
    require(len({(r['case'], r['ligand_idx']) for r in inputs}) == 10, 'Duplicate input identity')
    require(len({r['input_smiles'] for r in inputs}) == 10, 'Duplicate ligand graphs')
    groups = []
    for k in range(5):
        folder = out / f'group_{k+1:02d}'
        jobs = []
        for item in inputs[2*k:2*k+2]:
            for seed in seeds:
                rid = f"{item['case']}_lig{item['ligand_idx']}_seed_{seed}"
                jobs.append(dict(id=rid, case=item['case'], ligand_idx=item['ligand_idx'], seed=seed,
                                 run_directory=str(folder/'runs'/rid)))
        groups.append(dict(name=folder.name, folder=str(folder), jobs=jobs))
    require(len({j['id'] for g in groups for j in g['jobs']}) == 100, 'Duplicate generation job')
    return groups


def prepare(repo, out):
    method = read(METHOD)
    check_hashes(repo, method['source_sha256'])
    checkpoint = repo / method['checkpoint']
    check_hashes(repo, {method['checkpoint']: method['checkpoint_sha256']})
    observed_versions = versions(repo)
    require(observed_versions == method['versions'], f'Environment differs: {observed_versions}')
    from rdkit import Chem
    from flowr.data.interpolate import extract_scaffolds
    used, old_graphs, history, errors = selection.previous_inputs(repo, out, method, Chem)
    candidates, inventory, inputs_hashes = [], [], {}
    for case in method['case_order']:
        sdf, protein = repo/f'examples/{case}_ligands.sdf', repo/f'examples/{case}_protein.pdb'
        require(sdf.is_file() and protein.is_file(), f'Missing original input files for {case}')
        for p in (sdf, protein):
            require(not p.read_bytes()[:100].startswith(b'version https://git-lfs.github.com/'),
                    f'LFS pointer instead of input bytes: {p}')
            inputs_hashes[str(p)] = sha(p)
        for idx, mol in enumerate(Chem.SDMolSupplier(str(sdf), sanitize=True, removeHs=False)):
            item = dict(case=case, ligand_idx=idx)
            try:
                if (case, idx) in used:
                    if mol is not None:
                        old_graphs.add(selection.canonical(mol, Chem))
                    item['status'] = 'previously_used_input'
                else:
                    item.update(selection.input_profile(mol, Chem, extract_scaffolds))
                    item['status'] = 'eligible_before_graph_deduplication'
                    candidates.append(dict(item))
            except Exception as exc:
                item.update(status='ineligible_input', reason=str(exc))
            inventory.append(item)
    chosen = selection.choose_inputs(candidates, method['case_order'], 10, used, old_graphs)
    write(out/'input_inventory.json', dict(inventory=inventory, selected_inputs=chosen,
          requested_inputs=10, available_selected_inputs=len(chosen), history_errors=errors,
          history_manifests=history, excluded_ids=sorted(used), previous_ligand_graphs=len(old_graphs),
          scope='Unused in retained local history; not a pretraining holdout or new protein targets.'))
    require(not errors, 'Input history is ambiguous; inspect input_inventory.json. No generation started.')
    require(len(chosen) == 10,
            f'Only {len(chosen)}/10 unused eligible ligand graphs. No old-input fallback or GPU sampling.')
    # Pin all implementation sources after checking the known, previously exercised method.
    sources = dict(method['source_sha256'])
    for dirname in ('flowr', 'posebusters', 'scaffold_geometry_audit_v1', 'scaffold_study_v1'):
        for p in (repo/dirname).rglob('*'):
            if p.is_file() and p.suffix in ('.py', '.yml', '.yaml'):
                sources[str(p.relative_to(repo))] = sha(p)
    # Estimate storage using completed original records, when available. Do not delete any evidence.
    sizes = []
    for h in history:
        record = Path(h['path']).parent
        summary = record/'summary.json'
        if summary.is_file() and read(summary).get('n_attempts') == 1:
            sizes.append(sum(p.stat().st_size for p in record.rglob('*') if p.is_file()))
    free = shutil.disk_usage(out).free
    required = int(max(sizes)*100*1.25 + 1024**3) if sizes else None
    write(out/'storage_estimate.json', dict(free_bytes=free, estimated_required_bytes=required,
          basis='100 x largest retained research_record x 1.25 + 1 GiB; estimate only'))
    require(required is None or free >= required, 'Insufficient free space for estimated saved trajectories')
    groups = group_plan(chosen, method['seeds'], out)
    plan = dict(version='2026-10-31.validation100.1', repo=str(repo), output=str(out),
          selected_inputs=chosen, groups=groups, versions=observed_versions,
          source_sha256=sources, input_sha256=inputs_hashes, package_sha256=package_hashes(),
          checkpoint=str(checkpoint), checkpoint_sha256=method['checkpoint_sha256'],
          planned_generations=100, max_active_intervention_control_evaluations=1900,
          group_note='Five execution chunks, each with two distinct inputs and ten seeds; not five treatments.',
          rules=dict(selection='Input-only round-robin by target, ascending SDF index, excluding used ligand graphs',
              endpoint='Unchanged native chemical/connected/fixed-chemistry/substructure and full PoseBusters conjunction',
              repair='Existing A/B/AB/C and 15 controls; fixed geometry settings; no new candidate tuning',
              stop='Only 100 planned attempts; no replacement of failed jobs or additional seeds',
              scope='New local ligand inputs in existing FLOWR example pockets; no cross-model claim'),
          baseline_args=method['baseline_args'])
    write(out/'frozen_plan.json', plan)
    (out/'frozen_plan.sha256').write_text(sha(out/'frozen_plan.json')+'\n')
    write(out/'status.json', dict(completed=False, phase='prepared', planned_generations=100,
                                 actual_generations_started=0))
    print('PREPARED: 10 unused inputs x 10 seeds = 100 generations; 5 groups x 20.', flush=True)
    for item in chosen:
        print(f"INPUT {item['case']} ligand_idx={item['ligand_idx']}", flush=True)


def validate_generation(job, plan):
    root = Path(job['run_directory'])/'research_record'
    manifest, summary = read(root/'manifest.json'), read(root/'summary.json')
    args = manifest['args']
    require(args['seed'] == job['seed'] and args['ligand_idx'] == job['ligand_idx'], 'Generation identity differs')
    require(Path(args['pdb_file']).resolve() == Path(plan['repo'])/f"examples/{job['case']}_protein.pdb", 'Wrong protein input')
    require(Path(args['ligand_file']).resolve() == Path(plan['repo'])/f"examples/{job['case']}_ligands.sdf", 'Wrong ligand input')
    require(Path(args['save_dir']).resolve() == Path(job['run_directory']), 'Wrong output directory')
    variable = {'seed', 'ligand_idx', 'pdb_file', 'ligand_file', 'save_dir', 'ckpt_path'}
    require({k:v for k,v in args.items() if k not in variable} ==
            {k:v for k,v in plan['baseline_args'].items() if k not in variable}, 'Generation settings changed')
    require(manifest['checkpoint_sha256'] == plan['checkpoint_sha256'], 'Checkpoint differs')
    require(manifest['recorder_sha256'] == plan['source_sha256']['record_scaffold_run.py'], 'Recorder changed')
    for key in ('torch', 'rdkit'):
        require(manifest[key+'_version'] == plan['versions'][key], 'Generation environment differs')
    for k,v in manifest['source_sha256'].items():
        require(plan['source_sha256'].get(k) == v, 'Recorded generator source differs')
    for k,v in manifest['input_sha256'].items():
        require(plan['input_sha256'].get(k) == v, 'Recorded input differs')
    require(summary.get('pipeline_error') is None and summary.get('n_attempts') == 1, 'Incomplete generation')
    attempt = summary['attempts'][0]
    require(attempt.get('generation_completed') is True and not attempt.get('audit_error'), 'Recorder audit failed')
    require(attempt.get('forward_calls') == 101, 'Unexpected network-call count')
    folder = root/'attempt_000'
    require(read(folder/'task_definition.json').get('mapping_verified') is True, 'Unverified atom mapping')
    for name, pattern in [('forward_calls','call_*.pt'), ('sampler_steps','step_*.pt')]:
        require(len(list((folder/name).glob(pattern))) == 101, f'Incomplete {name}')
    for name in ('inputs.pt','raw_final_prediction.pt'):
        require((folder/name).is_file(), f'Missing {name}')


def fingerprint(folder):
    return {str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.suffix != '.zip' and '__pycache__' not in p.parts}


def stage(repo, out, key, command, destination, validator):
    receipt = out/'receipts'/(key+'.json')
    if receipt.exists():
        old = read(receipt)
        require(old['state'] == 'completed', f'{key} already started but did not finish; no automatic retry')
        require(old['command'] == command and old['outputs'] == fingerprint(destination), f'{key} result changed')
        validator()
        print('KEEP COMPLETED:', key, flush=True)
        return
    require(not destination.exists(), f'Untracked output exists: {destination}')
    log_path = out/'logs'/(key+'.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write(receipt, dict(state='started', command=command))
    print('RUN:', key, flush=True)
    try:
        with log_path.open('w', encoding='utf-8') as log:
            log.write(json.dumps(command, ensure_ascii=False)+'\n'); log.flush()
            with subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors='replace', bufsize=1) as proc:
                try:
                    for line in proc.stdout:
                        log.write(line); log.flush()
                        print(line, end='', flush=True)
                    require(proc.wait() == 0, f'{key} stopped; see {log_path}')
                except BaseException:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill(); proc.wait()
                    raise
        validator()
        write(receipt, dict(state='completed', command=command, outputs=fingerprint(destination)))
    except BaseException as exc:
        write(receipt, dict(state='stopped', command=command, error=f'{type(exc).__name__}: {exc}'))
        raise


def require_complete(path, count=None):
    d = read(path)
    require(d.get('completed') is True and not d.get('errors') and not d.get('error_count'), f'Incomplete report: {path}')
    if count is not None:
        require(d.get('requested_run_count') == count and len(d.get('runs', [])) == count, 'Incomplete quality inventory')


def commands(repo, group):
    p, g = sys.executable, Path(group['folder'])
    return [
        ('quality', [p, '-u', str(repo/'scaffold_evaluation_v2/evaluate_scaffold_outputs.py'),
            '--run-dirs', *[j['run_directory'] for j in group['jobs']], '--save-dir', str(g/'quality')], g/'quality',
            lambda: require_complete(g/'quality/quality_report.json', 20)),
        ('diagnostics', [p, '-u', str(repo/'scaffold_diagnostics_v1/diagnose_scaffold_runs.py'),
            '--quality-report', str(g/'quality/quality_report.json'), '--output-dir', str(g/'diagnostics')], g/'diagnostics',
            lambda: require_complete(g/'diagnostics/summary.json')),
        ('geometry', [p, '-u', str(repo/'scaffold_geometry_audit_v1/run_audit.py'), '--repo', str(repo),
            '--cohort-dir', str(g), '--diagnostic-dir', str(g/'diagnostics'), '--output-dir', str(g/'localization_v2/geometry')],
            g/'localization_v2/geometry', lambda: require_complete(g/'localization_v2/geometry/summary.json')),
        ('study', [p, '-u', str(repo/'scaffold_study_v1/run.py'), '--quality-report', str(g/'quality/quality_report.json'),
            '--diagnostic-dir', str(g/'diagnostics'), '--geometry-dir', str(g/'localization_v2/geometry'),
            '--output-dir', str(g/'localization_v2/study')], g/'localization_v2/study',
            lambda: require(read(g/'localization_v2/study/diagnosis.json')['n_runs'] == 20, 'Incomplete study')),
        ('rescue', [p, '-u', str(repo/'scaffold_study_v1/terminal_control.py'), '--repo', str(repo), '--cohort', str(g)],
            g/'rescue_v2', lambda: require_complete(g/'rescue_v2/status.json'))]


def aggregate(out, plan):
    keys = ('original_samples','original_passes','original_failures','rescued_without_C',
            'rescued_with_fixed_coordinate_relaxation','unresolved','unsupported','branch_errors')
    rows, samples = [], []
    for group in plan['groups']:
        folder = Path(group['folder'])/'rescue_v2'
        if not (folder/'summary.json').is_file():
            continue
        s = read(folder/'summary.json')
        if not s.get('completed'):
            continue
        require(s['original_samples'] == 20, 'Incomplete rescue inventory')
        require(s['original_passes']+s['original_failures'] == 20, 'Baseline denominator differs')
        require(s['rescued_without_C']+s['rescued_with_fixed_coordinate_relaxation']+
                s['unresolved']+s['unsupported'] == s['original_failures'], 'Rescue categories do not partition failures')
        selected = read(folder/'selected_results.json')
        expected_ids = {j['id'] for j in group['jobs']}
        require(len(selected) == 20 and {x['run_id'] for x in selected} == expected_ids,
                'Missing, duplicate or foreign selected sample identities')
        rows.append(dict(group=group['name'], **{k:s[k] for k in keys}))
        samples.extend(dict(group=group['name'], **r) for r in selected)
    total = {k:sum(r[k] for r in rows) for k in keys}
    failures = total['original_failures']
    summary = dict(completed_groups=len(rows), planned_groups=5, **total,
        rescue_fraction_of_all_failures=(total['rescued_without_C']+total['rescued_with_fixed_coordinate_relaxation'])/failures if failures else None,
        scope='Only completed groups shown; incomplete attempts remain in status/receipts. This is repair coverage, not causal accuracy.')
    write(out/'summary.json', summary)
    write(out/'samples.json', samples)
    with (out/'group_results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, ['group', *keys]); writer.writeheader(); writer.writerows(rows)
    return summary


def run(repo, out):
    require((out/'frozen_plan.json').is_file(), 'No plan: run --prepare first')
    require(sha(out/'frozen_plan.json') == (out/'frozen_plan.sha256').read_text().strip(), 'Frozen plan changed')
    plan = read(out/'frozen_plan.json')
    require(plan['repo'] == str(repo) and plan['output'] == str(out), 'Workspace/output moved')
    require(plan['package_sha256'] == package_hashes(), 'Batch program changed after preparation')
    require(versions(repo) == plan['versions'], 'Environment changed')
    check_hashes(repo, plan['source_sha256']); check_hashes(repo, plan['input_sha256'])
    check_hashes(repo, {plan['checkpoint']: plan['checkpoint_sha256']})
    for group in plan['groups']:
        check_hashes(repo, plan['source_sha256']); check_hashes(repo, plan['input_sha256'])
        g = Path(group['folder'])
        print(f"\nGROUP {group['name']}: 20 generations, then unchanged evaluation/localization/interventions", flush=True)
        g.mkdir(exist_ok=True)
        for job in group['jobs']:
            dest = Path(job['run_directory'])
            dest.parent.mkdir(exist_ok=True)
            cmd = [sys.executable, '-u', str(repo/'record_scaffold_run.py'), '--case', job['case'],
                '--ligand_idx', str(job['ligand_idx']), '--seed', str(job['seed']),
                '--ckpt_path', plan['checkpoint'], '--save_dir', str(dest)]
            stage(repo, out, group['name']+'_'+job['id'], cmd, dest, lambda j=job: validate_generation(j, plan))
        for key, cmd, dest, validator in commands(repo, group):
            stage(repo, out, group['name']+'_'+key, cmd, dest, validator)
        aggregate(out, plan)
        write(out/'status.json', dict(completed=False, phase='between_groups', last_completed_group=group['name']))
        bundle(out)
    check_hashes(repo, plan['source_sha256']); check_hashes(repo, plan['input_sha256'])
    summary = aggregate(out, plan)
    require(summary['completed_groups'] == 5 and summary['original_samples'] == 100, 'Incomplete final dataset')
    write(out/'status.json', dict(completed=True,
          phase='finished_with_branch_errors' if summary['branch_errors'] else 'finished',
          planned_generations=100, actual_generations_started=100,
          all_intervention_branches_evaluated=summary['branch_errors'] == 0))
    print('\nCOMPLETE:', json.dumps(summary, ensure_ascii=False), flush=True)


def bundle(out):
    target = out/'validation100_review.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.csv','.sdf','.log','.txt','.sha256','.npz'):
                z.write(p, p.relative_to(out))
        for p in sorted(HERE.iterdir()):
            if p.is_file() and p.suffix in ('.py','.json','.md'):
                z.write(p, 'batch_program/'+p.name)
    print('SEND THIS FILE:', target, flush=True)


def progress_counts(out):
    receipts = [read(p) for p in (out/'receipts').glob('*.json')]
    generations = [x for x in receipts if any(Path(v).name == 'record_scaffold_run.py'
                   for v in x.get('command', []))]
    return dict(planned_generations=100, generation_jobs_launched=len(generations),
                generation_jobs_completed=sum(x['state'] == 'completed' for x in generations))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    repo = args.repo.resolve(); out = (repo/args.output).resolve()
    require(repo in out.parents and repo != out, 'Output must be a new directory within the repository')
    if args.prepare:
        require(not out.exists(), 'Output already exists. Preserve it; do not prepare again.')
        out.mkdir(parents=True)
    else:
        require(out.is_dir(), 'Run --prepare first')
    lock = out/'RUNNING.lock'
    require(not lock.exists(), 'Run lock exists. Do not launch twice; send the existing report/logs.')
    with lock.open('x') as f:
        f.write('Active validation process; never remove while a process is running.\n')
    try:
        if args.prepare:
            prepare(repo, out)
        else:
            run(repo, out)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        text = traceback.format_exc()
        write(out/'status.json', dict(completed=False, phase='preparation_blocked' if args.prepare else 'stopped',
              error=f'{type(exc).__name__}: {exc}', traceback=text, **progress_counts(out)))
        print(text, file=sys.stderr)
        return 1
    finally:
        try:
            bundle(out)
        finally:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
