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
import path_io

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


def versions(repo, require_cuda=True):
    sys.path.insert(0, str(repo))
    import torch
    import rdkit
    import numpy
    import scipy
    import posebusters
    require(Path(posebusters.__file__).resolve().parent == repo / 'posebusters',
            'Expected the original local PoseBusters fork')
    if require_cuda:
        require(torch.cuda.is_available(), 'CUDA is unavailable; no generation started')
    return {k: str(m.__version__) for k, m in
            [('torch', torch), ('rdkit', rdkit), ('numpy', numpy), ('scipy', scipy)]}


def package_hashes():
    # config.py is snapshotted into the plan, so changing it affects only the next run.
    names = ('run.py', 'method_reference.json', 'input_selection.py', 'settings.py', 'path_io.py')
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
                                 pdb_file=item['pdb_file'], ligand_file=item['ligand_file'],
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
    sdf_original, protein_original = Path(settings['sdf_path']), Path(settings['pdb_path'])
    for p in (sdf_original, protein_original):
        require(p.is_file(), f'Missing input: {p}')
        with p.open('rb') as f:
            require(not f.read(100).startswith(b'version https://git-lfs.github.com/'), f'LFS pointer instead of input: {p}')
    require(any(line.startswith(('ATOM  ', 'HETATM')) for line in protein_original.read_text(errors='replace').splitlines()),
            'PDB has no ATOM/HETATM records')
    # Immutable input copies keep older experiments usable after the user changes a source SDF.
    (out/'inputs').mkdir()
    sdf, protein = out/'inputs/ligands.sdf', out/'inputs/protein.pdb'
    shutil.copy2(sdf_original,sdf);shutil.copy2(protein_original,protein)
    inputs_hashes={str(p):sha(p) for p in (sdf,protein)}
    case='dataset_'+hashlib.sha256((sha(protein)+sha(sdf)).encode()).hexdigest()[:16]
    supplier=Chem.SDMolSupplier(str(sdf),sanitize=True,removeHs=False)
    indices=list(range(len(supplier))) if settings['ligand_indices']=='all' else settings['ligand_indices']
    require(indices, 'Selected SDF has no molecules')
    inventory=[]
    for idx in indices:
        item=dict(case=case,ligand_idx=idx,pdb_file=str(protein),ligand_file=str(sdf))
        try:
            require(idx<len(supplier), f'ligand_idx {idx} exceeds SDF record count {len(supplier)}')
            item.update(selection.input_profile(supplier[idx], Chem, extract_scaffolds))
            item['status']='eligible'
        except Exception as exc:
            item.update(status='ineligible_input',reason=str(exc))
        inventory.append(item)
    chosen = [dict(x) for x in inventory if x['status'] == 'eligible']
    write(out/'input_inventory.json', dict(inventory=inventory, selected_inputs=chosen,
          requested_inputs=len(indices), available_selected_inputs=len(chosen),
          history_scanned=False, selection='Explicit configured inputs; reuse across experiments is allowed',
          scope='No claim of unseen inputs, unseen targets, or a training holdout.'))
    require(len(chosen) == len(indices),
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
    plan = dict(version='2026-11-07.paths.1', repo=str(repo), output=str(out),
          settings=settings, original_input_paths=dict(sdf=str(sdf_original),pdb=str(protein_original)),
          selected_inputs=chosen, groups=groups, versions=observed_versions,
          source_sha256=sources, input_sha256=inputs_hashes, package_sha256=package_hashes(),
          checkpoint=str(checkpoint), checkpoint_sha256=method['checkpoint_sha256'],
          planned_generations=total, max_active_intervention_control_evaluations=total*19,
          group_note='Execution chunks preserve all seeds of each input together; not independent treatments.',
          rules=dict(selection='Explicit inputs from config.py; no historical output scanning or novelty claim',
              endpoint='Unchanged native chemical/connected/fixed-chemistry/substructure and full PoseBusters conjunction',
              repair='Existing A/B/AB/C and 15 controls; fixed geometry settings; no new candidate tuning',
              stop=f'Only {total} planned attempts; no replacement of failed jobs or additional seeds',
              scope='Configured SDF/PDB pair; original scaffold-hopping method; no cross-model claim'),
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
    require(manifest.get('case_label') == job['case'], 'Dataset identity differs')
    require(args['seed'] == job['seed'] and args['ligand_idx'] == job['ligand_idx'], 'Generation identity differs')
    require(Path(args['pdb_file']).resolve() == Path(job['pdb_file']), 'Wrong protein input')
    require(Path(args['ligand_file']).resolve() == Path(job['ligand_file']), 'Wrong ligand input')
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


def verify_plan(repo,out,require_cuda=True):
    require((out/'frozen_plan.json').is_file(), 'No prepared plan at this generation directory')
    require(sha(out/'frozen_plan.json')==(out/'frozen_plan.sha256').read_text().strip(), 'Frozen plan changed')
    plan=read(out/'frozen_plan.json')
    require(plan['repo']==str(repo) and plan['output']==str(out), 'Workspace/output moved')
    require(plan['package_sha256']==package_hashes(), 'Batch program changed after preparation')
    require(versions(repo,require_cuda=require_cuda)==plan['versions'], 'Environment changed')
    check_hashes(repo,plan['source_sha256']);check_hashes(repo,plan['input_sha256'])
    if require_cuda:check_hashes(repo,{plan['checkpoint']:plan['checkpoint_sha256']})
    return plan


def run(repo,out):
    plan=verify_plan(repo,out)
    for group in plan['groups']:
        check_hashes(repo,plan['source_sha256']);check_hashes(repo,plan['input_sha256'])
        g=Path(group['folder']);g.mkdir(exist_ok=True)
        print(f"GROUP {group['name']}: {len(group['jobs'])} generations, evaluation and localization",flush=True)
        for job in group['jobs']:
            dest=Path(job['run_directory']);dest.parent.mkdir(exist_ok=True)
            cmd=[sys.executable,'-u',str(repo/'record_scaffold_run.py'),
                 '--pdb_file',job['pdb_file'],'--ligand_file',job['ligand_file'],'--input_name',job['case'],
                 '--ligand_idx',str(job['ligand_idx']),'--seed',str(job['seed']),
                 '--ckpt_path',plan['checkpoint'],'--save_dir',str(dest)]
            stage(repo,out,group['name']+'_'+job['id'],cmd,dest,lambda j=job:validate_generation(j,plan))
        for key,cmd,dest,validator in commands(repo,group)[:-1]:
            stage(repo,out,group['name']+'_'+key,cmd,dest,validator)
        path_io.export_generation(out,plan)
    summary=path_io.export_generation(out,plan)
    require(summary['completed'] and summary['classified_samples']==plan['planned_generations'],'Incomplete generation inventory')
    check_hashes(repo,plan['input_sha256'])
    write(out/'status.json',dict(phase='generation_finished',**summary))
    print('GENERATION COMPLETE:',json.dumps(summary,ensure_ascii=False),flush=True)
    return plan


def intervene(repo,source,out):
    generation_root,plan,selected=path_io.selected_failures(source)
    verify_plan(repo,generation_root,require_cuda=False)
    require(not (generation_root/'RUNNING.lock').exists(),'Generation is still active')
    require(not path_io.contained(out,generation_root) and not path_io.contained(generation_root,out),
            'Intervention output must be separate from the source generation')
    require(not out.exists(),'Intervention output exists; choose a new directory')
    # Hold a read/use lock so a concurrent installer does not change shared methods.
    lock=generation_root/'RUNNING.lock'
    with lock.open('x') as f:f.write('Intervention using these generation records')
    try:
        out.mkdir(parents=True)
        (out/'RUNNING.lock').write_text('Active intervention')
        write(out/'intervention_plan.json',dict(generation_root=str(generation_root),generation_plan_sha256=sha(generation_root/'frozen_plan.json'),
              input_directory=str(source),output_directory=str(out),requested_ids=[x['run_id'] for x in selected],
              package_sha256=package_hashes(),no_new_generations=True))
        groups=[]
        for group in plan['groups']:
            ids=[x['run_id'] for x in selected if x['group']==group['name']]
            if not ids:continue
            folder=out/group['name'];folder.mkdir();request=folder/'requested_failures.json';write(request,ids)
            dest=folder/'rescue_v2'
            cmd=[sys.executable,'-u',str(repo/'scaffold_study_v1/terminal_control.py'),
                 '--repo',str(repo),'--cohort',group['folder'],'--output-dir',str(dest),
                 '--max-runs',str(len(group['jobs'])),'--failure-ids',str(request)]
            stage(repo,out,group['name']+'_rescue',cmd,dest,lambda d=dest:require_complete(d/'status.json'))
            groups.append((group,dest))
        summary=path_io.export_interventions(out,selected,groups)
        original=read(generation_root/'generation_summary.json')
        summary.update(original_samples=original['classified_samples'],original_passes=original['original_passes'],
                       original_failures=original['original_failures'],
                       unselected_failures=original['original_failures']-len(selected),
                       final_passes_in_source_batch=original['original_passes']+summary['repaired'])
        write(out/'summary.json',summary)
        write(out/'status.json',dict(completed=True,phase='finished_with_branch_errors' if summary['branch_errors'] else 'finished',
              all_intervention_branches_evaluated=summary['branch_errors']==0))
        print('INTERVENTION COMPLETE:',json.dumps(summary,ensure_ascii=False),flush=True)
    except (Exception,KeyboardInterrupt) as exc:
        write(out/'status.json',dict(completed=False,phase='stopped',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
        raise
    finally:
        try:
            if out.is_dir():bundle(out,generation_root)
        finally:
            (out/'RUNNING.lock').unlink(missing_ok=True);lock.unlink(missing_ok=True)


def bundle(out,generation_root=None):
    target=out/'validation100_review.zip'
    roots=[('',out)] if generation_root is None else [('generation',generation_root),('intervention',out)]
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for prefix,root in roots:
            for p in sorted(root.rglob('*')):
                if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.csv','.sdf','.pdb','.log','.txt','.sha256','.npz'):
                    z.write(p,Path(prefix)/p.relative_to(root))
        for p in sorted(HERE.iterdir()):
            if p.is_file() and p.name!='config.py' and p.suffix in ('.py','.json','.md'):
                z.write(p,'batch_program/'+p.name)
    print('SEND THIS FILE:',target,flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--prepare',action='store_true',help='Validate inputs and freeze the generation plan')
    actions.add_argument('--run',action='store_true',help='Run an existing prepared generation plan; requires --output')
    actions.add_argument('--generate-only',action='store_true',help='Generate, evaluate, localize and classify, without intervention')
    actions.add_argument('--intervene-only',action='store_true',help='Read INTERVENTION_INPUT_DIR; do not generate new molecules')
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path,help='Existing prepared generation directory for --run only')
    args=parser.parse_args(argv);repo=args.repo.resolve()
    if args.run:
        if args.output is None or args.config is not None:parser.error('--run requires --output and uses the frozen settings')
        out=(repo/args.output).resolve();settings=read(out/'frozen_plan.json')['settings']
    else:
        if args.output is not None:parser.error('Set paths in config.py; --output is only used with --run')
        settings=load_config(args.config or CONFIG,repo);out=Path(settings['generation_dir'])
    generate=not args.intervene_only
    do_intervene=not (args.prepare or args.generate_only or args.run)
    try:
        if not args.run:path_io.check_layout(settings,generate=generate,intervene=do_intervene)
        if args.intervene_only:
            intervene(repo,Path(settings['intervention_input_dir']),Path(settings['intervention_output_dir']))
            return 0
        if not args.run:out.mkdir(parents=True)
        lock=out/'RUNNING.lock'
        with lock.open('x') as f:f.write('Active generation; do not install updates during this run')
        try:
            if not args.run:
                write(out/'experiment_settings.json',settings);prepare(repo,out,settings)
            if not args.prepare:run(repo,out)
            else:print('NEXT: python scaffold_validation100/run.py --run --repo '+str(repo)+' --output '+str(out))
        except (Exception,KeyboardInterrupt) as exc:
            write(out/'status.json',dict(completed=False,phase='stopped',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
            raise
        finally:
            try:bundle(out)
            finally:lock.unlink(missing_ok=True)
        if do_intervene:intervene(repo,Path(settings['intervention_input_dir']),Path(settings['intervention_output_dir']))
        return 0
    except (Exception,KeyboardInterrupt):
        traceback.print_exc();return 1


if __name__=='__main__':raise SystemExit(main())
