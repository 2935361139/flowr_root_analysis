#!/usr/bin/env python3
"""One bounded native FLOWR replay. Run in the original project with its .venv.

This adds observation/verification only; no intervention, solver rewrite, per-step
state replacement, automatic retry, or second case. Original recorder is reused.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import subprocess
import traceback
import zipfile
from collections.abc import Mapping
from pathlib import Path

VERSION = '2026-10-28.replay-once.1'
CASE = 'ptp1b_lig3_seed_47'
BASE = Path('outputs/sh_discovery_cohort_v2/runs/ptp1b_lig3/seed_47')
OUT = Path('outputs/sh_replay_once_ptp1b_lig3_seed47')
QUALITY = Path('outputs/sh_discovery_cohort_v2/quality/quality_report.json')
MANIFEST_SHA = '9a129929ba2f66308537b8cb5c2300e772f887d62092dc02cd01e22dacbccac3'
QUALITY_SHA = 'd8a9e496cae25307a0bd6fbc78837fba5e4eaa0165dffe0a816e613599c2099f'
EVALUATOR_SHA = '97eda26eca03e1f8dcb3d59b8b33430cbb855e87508eeee20f17905a3fe66f75'
ATOL = 1e-5  # absolute only; fixed before running; never adjusted automatically
CATEGORICAL = {'atomics', 'bonds', 'charges', 'hybridization', 'interactions'}
INPUT_KEYS = ('prior', 'pocket_data', 'pocket_com', 'times', 'rng', 'coord_scale',
              'self_condition', 'sc_recycle_steps', 'final_inpaint', 'solver', 'strategy')
EXTRA_SOURCE = {
    'flowr/models/pocket.py': 'ccc5199b91c46dba0c24b0a98f68085a7b7a9998970485a770d9823ae197a966',
    'flowr/models/pocket_util.py': '8bdc8f3635ece1b8e16eb4e8846e464cb5e6f8dbdd2faa51e75f9adb86830cfd',
    'flowr/util/rdkit.py': '00626d65e097dd1ac6b8f558219320869fef52611f964880a381f35903fda0d5',
    'flowr/util/metrics.py': 'e2ec7666d7c63ccbe164f67063f5d13701bcd11c146f423c86c86b6d723768e3',
    'flowr/gen/utils.py': '2185b671e62bea6012dd38f4dfb8e6ba877285769b8609cc252c28760be440e0',
}
QUALITY_FIELDS = ('rdkit_valid', 'fc_valid', 'raw_graph_connected',
    'root_fixed_substructure_match', 'fixed_chemistry_preserved',
    'chemical_connection_endpoint', 'posebusters_valid', 'posebusters_status',
    'posebusters_checks', 'posebusters_failed_checks', 'murcko_scaffold_changed',
    'molecule_smiles', 'scaffold_smiles', 'fixed_missing_sites',
    'fixed_element_changed_sites', 'fixed_charge_changed_sites',
    'fixed_internal_bond_changed_pairs', 'fixed_rms_displacement_A',
    'fixed_max_displacement_A', 'errors')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


class Evidence:
    """Resolve only real files or already-local verified LFS objects; never fetch."""
    def __init__(self, repo):
        self.repo, self.seen = Path(repo), {}
        p = subprocess.run(['git', 'rev-parse', '--git-path', 'lfs/objects'],
                           cwd=repo, capture_output=True, text=True)
        loc = Path(p.stdout.strip()) if p.returncode == 0 else Path('.git/lfs/objects')
        self.lfs = loc if loc.is_absolute() else self.repo / loc

    def resolve(self, path):
        path = Path(path).resolve()
        if str(path) in self.seen:
            return Path(self.seen[str(path)]['actual_path'])
        require(path.is_file(), f'Missing file: {path}')
        with path.open('rb') as f:
            head = f.read(256)
        actual = path
        if head.startswith(b'version https://git-lfs.github.com/spec/v1'):
            fields = dict(line.split(' ', 1) for line in head.decode().splitlines())
            oid = fields['oid'].removeprefix('sha256:')
            require(len(oid) == 64 and all(c in '0123456789abcdef' for c in oid), 'Invalid LFS pointer')
            actual = self.lfs / oid[:2] / oid[2:4] / oid
            require(actual.is_file(), f'LFS bytes not local: {path}; no GPU run started')
            require(actual.stat().st_size == int(fields['size']) and sha(actual) == oid,
                    f'LFS integrity failure: {path}')
        self.seen[str(path)] = {'actual_path': str(actual), 'sha256': sha(actual)}
        return actual

    def json(self, path):
        return json.loads(self.resolve(path).read_text(encoding='utf-8'))

    def tensor(self, path):
        import torch
        return torch.load(self.resolve(path), map_location='cpu', weights_only=True)


def array(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value, str(value.dtype)
    if hasattr(value, 'detach') and hasattr(value, 'cpu'):
        v = value.detach().cpu()
        dtype = str(v.dtype)
        try:
            return v.numpy(), dtype
        except TypeError:
            return v.double().numpy(), dtype
    return None


def compare(actual, expected, path='', exact=False):
    """Value comparison, not serialized-file comparison. Null is never a pass label."""
    import numpy as np
    result = {'passed': True, 'exact': True, 'max_abs_error': 0.0, 'differences': []}

    def bad(where, reason):
        result['passed'] = result['exact'] = False
        if len(result['differences']) < 20:
            result['differences'].append({'field': where, 'reason': reason})

    def walk(a, b, name, strict):
        av, bv = array(a), array(b)
        if av is not None or bv is not None:
            if av is None or bv is None:
                bad(name, 'tensor/non-tensor mismatch'); return
            x, xd = av; y, yd = bv
            if x.shape != y.shape or xd != yd:
                bad(name, 'shape/dtype mismatch'); return
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                bad(name, 'nonfinite value'); return
            identical = bool(np.array_equal(x, y))
            result['exact'] &= identical
            numeric_float = x.dtype.kind in 'fc'
            if numeric_float:
                err = float(np.max(np.abs(x.astype(np.complex128 if x.dtype.kind == 'c' else np.float64) - y))) if x.size else 0.0
                result['max_abs_error'] = max(result['max_abs_error'], err)
                if (strict and not identical) or (not strict and err > ATOL):
                    bad(name, f'absolute error {err}; allowed {0 if strict else ATOL}')
            elif not identical:
                bad(name, 'integer/mask/RNG differs')
            if name.rsplit('.', 1)[-1] in CATEGORICAL and x.ndim and x.shape[-1]:
                if not np.array_equal(x.argmax(-1), y.argmax(-1)):
                    bad(name, 'categorical decision changed')
            return
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            if set(a) != set(b):
                bad(name, 'dictionary keys differ')
            for key in sorted(set(a) & set(b)):
                walk(a[key], b[key], f'{name}.{key}', strict or key in ('rng', 'rng_before_forward'))
        elif isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            if type(a) is not type(b) or len(a) != len(b):
                bad(name, 'sequence type/length differs'); return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f'{name}[{i}]', strict)
        elif type(a) is float and type(b) is float:
            if not math.isfinite(a) or not math.isfinite(b):
                bad(name, 'nonfinite scalar'); return
            err = abs(a-b)
            result['max_abs_error'] = max(result['max_abs_error'], err)
            result['exact'] &= a == b
            if err > (0 if strict else ATOL):
                bad(name, f'scalar absolute error {err}')
        elif type(a) is not type(b) or a != b:
            bad(name, 'scalar or type differs')
    walk(actual, expected, path, exact)
    return result


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Audit:
    def __init__(self, out):
        self.out, self.rows = out, []
        self.state = {'version': VERSION, 'case': CASE, 'completed': False,
                      'phase': 'preflight', 'sampling_entries': 0, 'forward_calls_verified': 0,
                      'sampler_frames_verified': 0, 'interventions': 0,
                      'automatic_retries': 0, 'errors': []}

    def save(self):
        write_json(self.out / 'status.json', self.state)
        write_json(self.out / 'comparisons.json', self.rows)

    def check(self, label, actual, expected, exact=False):
        r = compare(actual, expected, label, exact)
        self.rows.append({'comparison': label, **r})
        if not r['passed']:
            self.save()
            raise RuntimeError(f'{label}: mismatch; see comparisons.json. No automatic retry.')


def guarded_recorder(original, evidence, expected_dir, audit):
    """Wrap existing observer boundaries; never feed historical tensors into sampling."""
    class Guarded(original.Recorder):
        def attach(self, model):
            native_signature = inspect.signature(model._generate)
            super().attach(model)
            recorded_generate, recorded_build = model._generate, model._build_trajectory

            def generate(_model, *args, **kwargs):
                require(audit.state['sampling_entries'] == 0, 'More than one generation attempt blocked')
                audit.state['sampling_entries'] = 1
                audit.state['phase'] = 'continuous_replay'
                audit.save()
                bound = native_signature.bind(*args, **kwargs)
                bound.apply_defaults()
                a = bound.arguments
                current = {'prior': original.snapshot(a['prior']),
                    'pocket_data': original.snapshot(a['pocket_data']),
                    'pocket_com': original.snapshot([s.com for s in a['pocket_data'].get('complex', [])]),
                    'times': original.snapshot(a['times']), 'rng': original.rng_snapshot(),
                    'coord_scale': float(model.coord_scale), 'self_condition': bool(model.self_condition),
                    'sc_recycle_steps': int(model.sc_recycle_steps), 'final_inpaint': bool(a['final_inpaint']),
                    'solver': a['solver'], 'strategy': a['strategy']}
                old = evidence.tensor(expected_dir / 'inputs.pt')
                audit.check('initial_state_and_rng', current, {k: old[k] for k in INPUT_KEYS}, exact=True)
                # Reconstructed native objects are used. Old snapshots are comparison-only.
                result = recorded_generate(*args, **kwargs)
                audit.check('raw_final_prediction', original.snapshot(result), evidence.tensor(expected_dir / 'raw_final_prediction.pt'))
                audit.state['trajectory_verified'] = True
                audit.save()
                return result

            def build(_model, curr, predicted, pocket_data, iter=0, step=0):
                value = recorded_build(curr, predicted, pocket_data, iter=iter, step=step)
                require(step == audit.state['sampler_frames_verified'] and step <= 100,
                        'Missing/duplicate/out-of-order sampler frame')
                file = Path('sampler_steps') / f'step_{step:04d}.pt'
                audit.check(f'sampler_{step:04d}', evidence.tensor(self.active['folder'] / file), evidence.tensor(expected_dir / file))
                audit.state['sampler_frames_verified'] += 1
                audit.save()
                return value

            self.replace_method(model, '_generate', generate)
            self.replace_method(model, '_build_trajectory', build)

        def before_forward(self, module, args, kwargs):
            super().before_forward(module, args, kwargs)
            if self.active is None:
                return None
            index = self.active['n_forward_calls']
            require(index <= 100, 'Extra network call blocked')
            old = evidence.tensor(expected_dir / 'forward_calls' / f'call_{index:04d}.pt')
            audit.check(f'call_{index:04d}_input', self.pending, {k: old[k] for k in self.pending})
            if index == 0:
                audit.check('pocket_encoding', evidence.tensor(self.active['folder'] / 'pocket_encoding.pt'),
                            evidence.tensor(expected_dir / 'pocket_encoding.pt'))
            return None

        def after_forward(self, module, args, kwargs, output):
            if self.active is None:
                return None
            index = self.active['n_forward_calls']
            super().after_forward(module, args, kwargs, output)
            file = Path('forward_calls') / f'call_{index:04d}.pt'
            audit.check(f'call_{index:04d}_output', evidence.tensor(self.active['folder'] / file), evidence.tensor(expected_dir / file))
            audit.state['forward_calls_verified'] += 1
            audit.save()
            print(f'[Replay] verified network call {index}/100', flush=True)
            return None
    return Guarded


def preflight(repo, evidence):
    record = repo / BASE / 'research_record'
    manifest_path = evidence.resolve(record / 'manifest.json')
    require(sha(manifest_path) == MANIFEST_SHA, 'Original manifest identity changed')
    m = evidence.json(manifest_path)
    require(m['args']['seed'] == 47 and m['args']['ligand_idx'] == 3, 'Wrong case')
    problems = []
    for rel, digest in {**m['source_sha256'], **EXTRA_SOURCE}.items():
        p = repo / rel
        if not p.is_file() or sha(p) != digest:
            problems.append(f'Source missing/changed: {rel}')
    # The original batch used its packaged recorder. The root copy is an allowed
    # fallback only when its bytes have the identical recorded SHA256.
    choices = [repo / 'scaffold_cohort_v2/helpers/record_scaffold_run.py', repo / 'record_scaffold_run.py']
    recorder = next((p for p in choices if p.is_file() and sha(p) == m['recorder_sha256']), None)
    if recorder is None:
        problems.append('No original recorder matching manifest SHA256')
    evaluator = repo / 'scaffold_evaluation_v2/evaluate_scaffold_outputs.py'
    if not evaluator.is_file() or sha(evaluator) != EVALUATOR_SHA:
        problems.append('Quality evaluator missing/changed')
    for key in ('pdb_file', 'ligand_file', 'ckpt_path'):
        # Native recorder reconstructs these paths in the current project.
        p = repo / ('checkpoints' if key == 'ckpt_path' else 'examples') / Path(m['args'][key]).name
        digest = m['checkpoint_sha256'] if key == 'ckpt_path' else m['input_sha256'][m['args'][key]]
        if not p.is_file() or sha(p) != digest:
            problems.append(f'Original input/weight missing or changed: {p}')
    a = record / 'attempt_000'
    names = ['inputs.pt', 'pocket_encoding.pt', 'raw_final_prediction.pt', 'reference_batch.pt',
             'task_definition.json', 'attempt.json', 'decoded_status.json',
             'decoded_before_filter.sdf', 'decoded_without_repair.sdf', 'reference_preprocessed.sdf']
    names += [f'forward_calls/call_{i:04d}.pt' for i in range(101)]
    names += [f'sampler_steps/step_{i:04d}.pt' for i in range(101)]
    for p in [record/'summary.json', repo/QUALITY] + [a/n for n in names]:
        try:
            evidence.resolve(p)
        except Exception as exc:
            problems.append(str(exc))
    require(not problems, '\n'.join(problems))
    for folder, pattern in [('forward_calls', 'call_*.pt'), ('sampler_steps', 'step_*.pt')]:
        require(len(list((a/folder).glob(pattern))) == 101, f'Unexpected {folder} inventory')
    require(evidence.json(a/'attempt.json') == {'generation_completed': True, 'forward_calls': 101,
            'saved_sampler_frames': 101, 'integration_steps_requested': 100}, 'Incomplete original attempt')
    require(evidence.json(record/'summary.json')['pipeline_error'] is None, 'Original pipeline incomplete')
    require(sha(evidence.resolve(repo/QUALITY)) == QUALITY_SHA, 'Quality report identity changed')
    q = evidence.json(repo/QUALITY)
    require(q['completed'] and not q['errors'], 'Original quality evaluation incomplete')
    rows = [r for r in q['rows'] if r['run_directory'] == m['args']['save_dir']]
    require(len(rows) == 2 and {r['stage'] for r in rows} == {'raw_without_repair', 'run_decode'}
            and all(not r['errors'] for r in rows), 'Missing original quality rows')
    return m, recorder, evaluator, rows


def run_native(repo, out, evidence, audit, manifest, recorder_path):
    import flowr.gen.generate_from_pdb as entry
    recorder = load_module(recorder_path, '_original_flowr_recorder')
    old_class, old_evaluate, old_argv = recorder.Recorder, entry.evaluate, sys.argv
    recorder.Recorder = guarded_recorder(recorder, evidence, repo/BASE/'research_record/attempt_000', audit)
    run = out / 'run'
    expected_args = dict(manifest['args'])
    for name in ('pdb_file', 'ligand_file', 'ckpt_path'):
        expected_args[name] = str(repo / ('checkpoints' if name == 'ckpt_path' else 'examples') / Path(expected_args[name]).name)
    expected_args['save_dir'] = str(run)

    def checked_evaluate(args):
        audit.check('generation_arguments', vars(args), expected_args, exact=True)
        return old_evaluate(args)

    entry.evaluate = checked_evaluate
    sys.argv = [str(recorder_path), '--case', 'ptp1b', '--seed', '47', '--ligand_idx', '3',
                '--ckpt_path', expected_args['ckpt_path'], '--save_dir', str(run)]
    try:
        code = recorder.main()
        require(code == 0, f'Native recorder stopped (code {code}); inspect run.log and recorded pipeline_error')
    finally:
        recorder.Recorder, entry.evaluate, sys.argv = old_class, old_evaluate, old_argv
    require(audit.state['forward_calls_verified'] == 101 and audit.state['sampler_frames_verified'] == 101,
            'Full trajectory inventory did not match')
    a, b = run/'research_record/attempt_000', repo/BASE/'research_record/attempt_000'
    for name in ('task_definition.json', 'decoded_status.json', 'attempt.json'):
        audit.check(name, evidence.json(a/name), evidence.json(b/name))
    audit.check('recorder_summary', evidence.json(run/'research_record/summary.json'),
                evidence.json(repo/BASE/'research_record/summary.json'))
    audit.check('reference_batch', evidence.tensor(a/'reference_batch.pt'), evidence.tensor(b/'reference_batch.pt'))
    for name in ('reference_preprocessed.sdf', 'decoded_without_repair.sdf', 'decoded_before_filter.sdf'):
        result = recorder.compare_sdf(evidence.resolve(b/name), a/name)
        audit.rows.append({'comparison': name, 'passed': result.get('matches_at_sdf_precision') is True, **result})
        require(result.get('matches_at_sdf_precision') is True, f'SDF graph/coordinate mismatch: {name}')
    audit.state['phase'] = 'quality_check_one_sample'
    audit.save()
    return run


def quality_check(repo, out, evaluator, run, rows, audit):
    command = [sys.executable, '-B', str(evaluator), '--run-dirs', str(run), '--save-dir', str(out/'quality')]
    result = subprocess.run(command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (out/'quality.log').write_text(result.stdout, encoding='utf-8')
    require(result.returncode == 0, 'Single-sample quality check stopped; see quality.log')
    q = json.loads((out/'quality/quality_report.json').read_text())
    require(q['completed'] is True and not q['errors'] and len(q['rows']) == 2, 'Incomplete new quality report')
    old = {r['stage']: r for r in rows}
    require({r['stage'] for r in q['rows']} == set(old), 'Quality stages differ')
    for r in q['rows']:
        require(not r['errors'], 'Quality measurement error, not a generated-molecule failure')
        audit.check('quality_'+r['stage'], {k: r.get(k) for k in QUALITY_FIELDS},
                    {k: old[r['stage']].get(k) for k in QUALITY_FIELDS})


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
    def write(self, s):
        self.stream.write(s); self.file.write(s); self.file.flush()
        return len(s)
    def flush(self):
        self.stream.flush(); self.file.flush()
    def isatty(self):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()  # no case/seed/tolerance overrides in this bounded task
    repo = Path.cwd().resolve()
    if not (repo/'flowr/gen/generate_from_pdb.py').is_file():
        print('STOP: run from /home/ljl/projects/flowr_root using .venv/bin/python', file=sys.stderr)
        return 2
    out = repo/OUT
    if out.exists():
        print(f'STOP: existing result retained: {out}. No second generation, no overwrite.')
        return 2
    out.mkdir(parents=True)
    audit, evidence = Audit(out), None
    audit.save()
    with (out/'run.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                evidence = Evidence(repo)
                m, recorder, evaluator, rows = preflight(repo, evidence)
                import torch
                from rdkit import rdBase
                require(str(torch.__version__) == m['torch_version'], 'PyTorch version differs from original')
                require(rdBase.rdkitVersion == m['rdkit_version'], 'RDKit version differs from original')
                require(torch.cuda.is_available(), 'Original CUDA execution required; CPU fallback blocked')
                require(os.environ.get('FLOWR_DEVICE', '').lower() not in ('cpu', 'mps'), 'Non-CUDA device override')
                audit.state['environment'] = {'torch': str(torch.__version__), 'rdkit': rdBase.rdkitVersion,
                    'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}
                write_json(out/'protocol.json', {'version': VERSION, 'case': CASE, 'max_sampling_entries': 1,
                    'automatic_retries': 0, 'interventions': 0, 'float_atol': ATOL, 'float_rtol': 0,
                    'initial_state_and_rng': 'exact', 'categorical_decisions': 'exact',
                    'mode': 'native end-to-end rerun; snapshots are comparison-only, never injected',
                    'budget_note': 'One entered sampling attempt counts toward the existing 24-run budget, even if stopped early.',
                    'limitations': ['No hidden-activation equivalence claim.', 'No causal intervention.',
                        'No universal mechanism or cross-model validation.', 'Quality labels reproduce tool results, not biological activity.'],
                    'script_sha256': sha(__file__)})
                sys.path.insert(0, str(repo))
                print('[Replay] Starting ONE native replay. Stop automatically on mismatch.', flush=True)
                run = run_native(repo, out, evidence, audit, m, recorder)
                quality_check(repo, out, evaluator, run, rows, audit)
                # Detect files changed while the experiment was in progress.
                for source, meta in evidence.seen.items():
                    require(sha(meta['actual_path']) == meta['sha256'], f'Evidence changed during replay: {source}')
                audit.state.update(completed=True, phase='passed')
            except BaseException as exc:
                audit.state['phase'] = 'stopped'
                audit.state['errors'].append(f'{type(exc).__name__}: {exc}')
                traceback.print_exc()
            finally:
                if evidence is not None:
                    write_json(out/'evidence_hashes.json', evidence.seen)
                audit.save()
    fields = ['comparison', 'passed', 'exact', 'max_abs_error']
    with (out/'comparison_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(audit.rows)
    passed = audit.state['completed']
    report = ('完整重放通过。可以准备下一轮有限干预；本轮没有执行干预。' if passed else
              '本轮停止，不能进入干预。请查看 status.json、comparisons.json 和 run.log；不要自动重跑或放宽阈值。')
    (out/'RESULT_CN.txt').write_text(report+'\n'+json.dumps(audit.state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with zipfile.ZipFile(out/'replay_review.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json', '.csv', '.log', '.txt', '.sdf'):
                z.write(p, p.relative_to(out))
    print(f'\nReplay passed: {passed}')
    print(f'Upload this file: {out / "replay_review.zip"}')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
