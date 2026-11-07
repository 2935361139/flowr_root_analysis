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
from datetime import datetime
from pathlib import Path

import input_selection as selection
from settings import load_config

HERE = Path(__file__).resolve().parent
METHOD = HERE / 'method_reference.json'
CONFIG = HERE / 'config.py'


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
    # config.py is snapshotted into the plan, so changing it affects only the next run.
    names = ('run.py', 'method_reference.json', 'input_selection.py', 'settings.py')
    return {name:sha(HERE/name) for name in names}


def group_plan(inputs, seeds, out, inputs_per_group):
    require(inputs and seeds and len(set(seeds)) == len(seeds), 'Need inputs and unique seeds')
    require(type(inputs_per_group) is int and inputs_per_group > 0, 'Invalid inputs_per_group')
    require(len({(r['case'], r['ligand_idx']) for r in inputs}) == len(inputs), 'Duplicate input identity')
    groups = []
    for offset in range(0, len(inputs), inputs_per_group):
        folder = out / f'group_{len(groups)+1:02d}'
        jobs = []
        for item in inputs[offset:offset+inputs_per_group]:
            for seed in seeds:
                rid = f"{item['case']}_lig{item['ligand_idx']}_seed_{seed}"
                jobs.append(dict(id=rid, case=item['case'], ligand_idx=item['ligand_idx'], seed=seed,
                                 run_directory=str(folder/'runs'/rid)))
        groups.append(dict(name=folder.name, folder=str(folder), jobs=jobs))
    require(len({j['id'] for g in groups for j in g['jobs']}) == len(inputs)*len(seeds), 'Duplicate generation job')
    return groups


def prepare(repo, out, settings):
    method = read(METHOD)
    check_hashes(repo, method['source_sha256'])
    checkpoint = repo / method['checkpoint']
    check_hashes(repo, {method['checkpoint']: method['checkpoint_sha256']})
    observed_versions = versions(repo)
    require(observed_versions == method['versions'], f'Environment differs: {observed_versions}')
    from rdkit import Chem
    from flowr.data.interpolate import extract_scaffolds
    inventory, inputs_hashes, suppliers = [], {}, {}
    for case in dict.fromkeys(x['case'] for x in settings['inputs']):
        sdf, protein = repo/f'examples/{case}_ligands.sdf', repo/f'examples/{case}_protein.pdb'
        require(sdf.is_file() and protein.is_file(), f'Missing original input files for {case}')
        for p in (sdf, protein):
            require(not p.read_bytes()[:100].startswith(b'version https://git-lfs.github.com/'),
                    f'LFS pointer instead of input bytes: {p}')
            inputs_hashes[str(p)] = sha(p)
        suppliers[case] = Chem.SDMolSupplier(str(sdf), sanitize=True, removeHs=False)
    for selected in settings['inputs']:
        item = dict(selected)
        try:
            supplier = suppliers[item['case']]
            require(item['ligand_idx'] < len(supplier), f'ligand_idx exceeds SDF record count {len(supplier)}')
            item.update(selection.input_profile(supplier[item['ligand_idx']], Chem, extract_scaffolds))
            item['status'] = 'eligible'
        except Exception as exc:
            item.update(status='ineligible_input', reason=str(exc))
        inventory.append(item)
    chosen = [dict(x) for x in inventory if x['status'] == 'eligible']
    write(out/'input_inventory.json', dict(inventory=inventory, selected_inputs=chosen,
          requested_inputs=len(settings['inputs']), available_selected_inputs=len(chosen),
          history_scanned=False, selection='Explicit configured inputs; reuse across experiments is allowed',
          scope='No claim of unseen inputs, unseen targets, or a training holdout.'))
    require(len(chosen) == len(settings['inputs']),
            'Some configured inputs are invalid; see input_inventory.json. No replacement or generation.')
    # Pin all implementation sources after checking the known, previously exercised method.
    sources = dict(method['source_sha256'])
    for dirname in ('flowr', 'posebusters', 'scaffold_geometry_audit_v1', 'scaffold_study_v1'):
        for p in (repo/dirname).rglob('*'):
            if p.is_file() and p.suffix in ('.py', '.yml', '.yaml'):
                sources[str(p.relative_to(repo))] = sha(p)
    free = shutil.disk_usage(out).free
    write(out/'storage_estimate.json', dict(free_bytes=free, estimated_required_bytes=None,
          basis='No historical-output dependency; trajectory storage requirement is not estimated.'))
    groups = group_plan(chosen, settings['seeds'], out, settings['inputs_per_group'])
    total = sum(len(g['jobs']) for g in groups)
    plan = dict(version='2026-11-07.configurable.1', repo=str(repo), output=str(out),
          settings=settings,
          selected_inputs=chosen, groups=groups, versions=observed_versions,
          source_sha256=sources, input_sha256=inputs_hashes, package_sha256=package_hashes(),
          checkpoint=str(checkpoint), checkpoint_sha256=method['checkpoint_sha256'],
          planned_generations=total, max_active_intervention_control_evaluations=total*19,
          group_note='Execution chunks preserve all seeds of each input together; not independent treatments.',
          rules=dict(selection='Explicit inputs from config.py; no historical output scanning or novelty claim',
              endpoint='Unchanged native chemical/connected/fixed-chemistry/substructure and full PoseBusters conjunction',
              repair='Existing A/B/AB/C and 15 controls; fixed geometry settings; no new candidate tuning',
              stop=f'Only {total} planned attempts; no replacement of failed jobs or additional seeds',
              scope='Configured local ligand inputs in existing FLOWR example pockets; no cross-model claim'),
          baseline_args=method['baseline_args'])
    write(out/'frozen_plan.json', plan)
    (out/'frozen_plan.sha256').write_text(sha(out/'frozen_plan.json')+'\n')
    write(out/'status.json', dict(completed=False, phase='prepared', planned_generations=total,
                                 actual_generations_started=0))
    print(f'PREPARED: {len(chosen)} inputs x {len(settings["seeds"])} seeds = {total} generations; '
          f'{len(groups)} groups; history scanning disabled. Free space: {free / 1024**3:.1f} GiB.', flush=True)
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
    count = len(group['jobs'])
    return [
        ('quality', [p, '-u', str(repo/'scaffold_evaluation_v2/evaluate_scaffold_outputs.py'),
            '--run-dirs', *[j['run_directory'] for j in group['jobs']], '--save-dir', str(g/'quality')], g/'quality',
            lambda: require_complete(g/'quality/quality_report.json', count)),
        ('diagnostics', [p, '-u', str(repo/'scaffold_diagnostics_v1/diagnose_scaffold_runs.py'),
            '--quality-report', str(g/'quality/quality_report.json'), '--output-dir', str(g/'diagnostics')], g/'diagnostics',
            lambda: require_complete(g/'diagnostics/summary.json')),
        ('geometry', [p, '-u', str(repo/'scaffold_geometry_audit_v1/run_audit.py'), '--repo', str(repo),
            '--cohort-dir', str(g), '--diagnostic-dir', str(g/'diagnostics'), '--output-dir', str(g/'localization_v2/geometry')],
            g/'localization_v2/geometry', lambda: require_complete(g/'localization_v2/geometry/summary.json')),
        ('study', [p, '-u', str(repo/'scaffold_study_v1/run.py'), '--quality-report', str(g/'quality/quality_report.json'),
            '--diagnostic-dir', str(g/'diagnostics'), '--geometry-dir', str(g/'localization_v2/geometry'),
            '--output-dir', str(g/'localization_v2/study')], g/'localization_v2/study',
            lambda: require(read(g/'localization_v2/study/diagnosis.json')['n_runs'] == count, 'Incomplete study')),
        ('rescue', [p, '-u', str(repo/'scaffold_study_v1/terminal_control.py'), '--repo', str(repo), '--cohort', str(g),
            '--max-runs', str(count)],
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
        count = len(group['jobs'])
        require(all(type(s[k]) is int and s[k] >= 0 for k in keys), 'Invalid summary counts')
        require(s['original_samples'] == count, 'Incomplete rescue inventory')
        require(s['original_passes']+s['original_failures'] == count, 'Baseline denominator differs')
        require(s['rescued_without_C']+s['rescued_with_fixed_coordinate_relaxation']+
                s['unresolved']+s['unsupported'] == s['original_failures'], 'Rescue categories do not partition failures')
        selected = read(folder/'selected_results.json')
        expected_ids = {j['id'] for j in group['jobs']}
        require(len(selected) == count and {x['run_id'] for x in selected} == expected_ids,
                'Missing, duplicate or foreign selected sample identities')
        rows.append(dict(group=group['name'], **{k:s[k] for k in keys}))
        samples.extend(dict(group=group['name'], **r) for r in selected)
    total = {k:sum(r[k] for r in rows) for k in keys}
    failures = total['original_failures']
    summary = dict(completed_groups=len(rows), planned_groups=len(plan['groups']),
        planned_samples=plan['planned_generations'], **total,
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
        print(f"\nGROUP {group['name']}: {len(group['jobs'])} generations, then unchanged evaluation/localization/interventions", flush=True)
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
    require(summary['completed_groups'] == len(plan['groups']) and
            summary['original_samples'] == plan['planned_generations'], 'Incomplete final dataset')
    write(out/'status.json', dict(completed=True,
          phase='finished_with_branch_errors' if summary['branch_errors'] else 'finished',
          planned_generations=plan['planned_generations'], actual_generations_started=plan['planned_generations'],
          all_intervention_branches_evaluated=summary['branch_errors'] == 0))
    print('\nCOMPLETE:', json.dumps(summary, ensure_ascii=False), flush=True)


def bundle(out):
    target = out/'validation100_review.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.csv','.sdf','.log','.txt','.sha256','.npz'):
                z.write(p, p.relative_to(out))
        for p in sorted(HERE.iterdir()):
            # The editable page may now describe the next experiment. The actual
            # settings are recorded in experiment_settings.json and frozen_plan.json.
            if p.is_file() and p.name != 'config.py' and p.suffix in ('.py','.json','.md'):
                z.write(p, 'batch_program/'+p.name)
    print('SEND THIS FILE:', target, flush=True)


def progress_counts(out):
    receipts = [read(p) for p in (out/'receipts').glob('*.json')]
    generations = [x for x in receipts if any(Path(v).name == 'record_scaffold_run.py'
                   for v in x.get('command', []))]
    plan_path = out/'frozen_plan.json'
    settings_path = out/'experiment_settings.json'
    planned = (read(plan_path)['planned_generations'] if plan_path.is_file() else
               (len(read(settings_path)['inputs'])*read(settings_path)['samples_per_input'] if settings_path.is_file() else 0))
    return dict(planned_generations=planned, generation_jobs_launched=len(generations),
                generation_jobs_completed=sum(x['state'] == 'completed' for x in generations))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, help='New output folder; required for --run')
    parser.add_argument('--config', type=Path, help='Parameter page; default: config.py next to this runner')
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    if args.run and (args.output is None or args.config is not None):
        parser.error('--run requires --output and uses the saved plan, not --config')
    settings = None if args.run else load_config(args.config or CONFIG)
    if args.output is None:
        args.output = Path('outputs') / (settings['output_prefix'] + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out = (repo/args.output).resolve()
    require(repo in out.parents and repo != out, 'Output must be a new directory within the repository')
    if not args.run:
        require(not out.exists(), 'Output already exists. Preserve it; do not prepare again.')
        out.mkdir(parents=True)
    else:
        require(out.is_dir(), 'Run --prepare first')
    lock = out/'RUNNING.lock'
    require(not lock.exists(), 'Run lock exists. Do not launch twice; send the existing report/logs.')
    with lock.open('x') as f:
        f.write('Active validation process; never remove while a process is running.\n')
    try:
        if not args.run:
            write(out/'experiment_settings.json', settings)
            prepare(repo, out, settings)
        if not args.prepare:
            run(repo, out)
        else:
            import shlex
            print('NEXT:', shlex.join([sys.executable, str(HERE/'run.py'), '--run', '--repo', str(repo), '--output', str(out)]))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        text = traceback.format_exc()
        write(out/'status.json', dict(completed=False, phase='preparation_blocked' if not (out/'frozen_plan.json').is_file() else 'stopped',
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
