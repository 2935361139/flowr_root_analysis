"""CPU orchestration tests. Synthetic subprocesses are not FLOWR experiments."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import run as r
import install_compat
import ast
import types


def inputs():
    return [dict(case=['cdk2','tyk2','ptp1b','bace'][i%4], ligand_idx=i+10,
                 input_smiles=f'graph_{i}') for i in range(10)]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root/'outputs'; self.out.mkdir()
        self.quiet = contextlib.redirect_stdout(io.StringIO()); self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)
        self.tmp.cleanup()

    def groups(self):
        return r.group_plan(inputs(), list(range(2001,2011)), self.out)

    def test_exact_budget_and_input_grouping(self):
        gs = self.groups()
        self.assertEqual(len(gs), 5)
        self.assertEqual(sum(len(g['jobs']) for g in gs), 100)
        self.assertTrue(all(len(g['jobs']) == 20 for g in gs))
        seen = set()
        for g in gs:
            ids = {(j['case'],j['ligand_idx']) for j in g['jobs']}
            self.assertEqual(len(ids), 2)
            self.assertFalse(seen & ids); seen.update(ids)
        self.assertEqual(len(seen), 10)

    def test_duplicate_graph_blocked(self):
        a = inputs(); a[-1]['input_smiles'] = a[0]['input_smiles']
        with self.assertRaises(RuntimeError):
            r.group_plan(a, list(range(10)), self.out)

    def test_duplicate_identity_blocked(self):
        a = inputs(); a[-1].update(case=a[0]['case'],ligand_idx=a[0]['ligand_idx'])
        with self.assertRaises(RuntimeError):
            r.group_plan(a, list(range(10)), self.out)

    def test_no_old_inputs_or_graphs_selected(self):
        chosen = r.selection.choose_inputs(inputs(), ['cdk2','tyk2','ptp1b','bace'], 10,
                 {('cdk2',10)}, {'graph_1'})
        self.assertEqual(len(chosen), 8)
        self.assertNotIn('graph_0', [x['input_smiles'] for x in chosen])
        self.assertNotIn('graph_1', [x['input_smiles'] for x in chosen])

    def test_selector_deduplicates_candidates(self):
        a = inputs(); a[-1]['input_smiles'] = a[0]['input_smiles']
        chosen = r.selection.choose_inputs(a, ['cdk2','tyk2','ptp1b','bace'], 10, set(), set())
        self.assertEqual(len(chosen), 9)

    def test_commands_use_each_new_group(self):
        for g in self.groups():
            cs = r.commands(self.root, g)
            self.assertEqual([c[0] for c in cs], ['quality','diagnostics','geometry','study','rescue'])
            quality = cs[0][1]
            paths = quality[quality.index('--run-dirs')+1:quality.index('--save-dir')]
            self.assertEqual(paths, [j['run_directory'] for j in g['jobs']])
            rescue = cs[-1][1]
            self.assertEqual(rescue[rescue.index('--cohort')+1], g['folder'])
            self.assertNotIn('sh_restart_v1', ' '.join(rescue))

    def command(self, destination):
        return [sys.executable, '-c',
                'from pathlib import Path; import sys; p=Path(sys.argv[1]); p.mkdir(); (p/"ok.txt").write_text("done")',
                str(destination)]

    def test_completed_stage_is_not_reexecuted(self):
        dest = self.out/'job'; cmd = self.command(dest)
        validate = lambda: r.require((dest/'ok.txt').read_text() == 'done', 'missing')
        r.stage(self.root, self.out, 'job', cmd, dest, validate)
        with mock.patch.object(r.subprocess, 'Popen', side_effect=AssertionError('must not rerun')):
            r.stage(self.root, self.out, 'job', cmd, dest, validate)

    def test_changed_completed_output_is_rejected(self):
        dest = self.out/'job'; cmd = self.command(dest)
        r.stage(self.root, self.out, 'job', cmd, dest, lambda: None)
        (dest/'ok.txt').write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'result changed'):
            r.stage(self.root, self.out, 'job', cmd, dest, lambda: None)

    def test_failed_stage_is_not_retried(self):
        dest = self.out/'job'; cmd = [sys.executable,'-c','raise SystemExit(3)']
        with self.assertRaises(RuntimeError):
            r.stage(self.root, self.out, 'job', cmd, dest, lambda: None)
        with self.assertRaisesRegex(RuntimeError, 'no automatic retry'):
            r.stage(self.root, self.out, 'job', cmd, dest, lambda: None)

    def test_untracked_directory_is_never_overwritten(self):
        dest = self.out/'job'; dest.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'Untracked output'):
            r.stage(self.root, self.out, 'job', self.command(dest), dest, lambda: None)

    def test_analysis_error_is_not_success(self):
        path = self.out/'quality.json'
        r.write(path, dict(completed=False,errors=['error'], requested_run_count=20,runs=[]))
        with self.assertRaises(RuntimeError):
            r.require_complete(path,20)

    def test_report_pack_excludes_tensor_weights_and_nested_zip(self):
        for name in ['result.json','molecule.sdf','run.log','x.pt','x.ckpt','old.zip']:
            (self.out/name).write_text('{}')
        r.bundle(self.out)
        with zipfile.ZipFile(self.out/'validation100_review.zip') as z:
            names = set(z.namelist())
        self.assertTrue({'result.json','molecule.sdf','run.log'} <= names)
        self.assertFalse({'x.pt','x.ckpt','old.zip'} & names)

    def fake_group_results(self, g):
        s = dict(completed=True, original_samples=20,original_passes=12, original_failures=8,
                 rescued_without_C=3,rescued_with_fixed_coordinate_relaxation=2,unresolved=2,unsupported=1,branch_errors=1)
        p = Path(g['folder'])/'rescue_v2'
        r.write(p/'summary.json', s)
        r.write(p/'selected_results.json', [dict(run_id=j['id']) for j in g['jobs']])
        return s

    def test_aggregate_separates_relaxed_constraint_and_unsupported(self):
        gs = self.groups()
        for g in gs:
            self.fake_group_results(g)
        s = r.aggregate(self.out, dict(groups=gs))
        self.assertEqual(s['original_samples'],100)
        self.assertEqual(s['original_failures'],40)
        self.assertEqual(s['rescued_without_C'],15)
        self.assertEqual(s['rescued_with_fixed_coordinate_relaxation'],10)
        self.assertEqual(s['unsupported'],5)
        self.assertEqual(s['branch_errors'],5)
        self.assertEqual(s['rescue_fraction_of_all_failures'],25/40)

    def test_partial_batches_not_reported_as_100(self):
        gs = self.groups(); self.fake_group_results(gs[0])
        s = r.aggregate(self.out, dict(groups=gs))
        self.assertEqual(s['completed_groups'],1)
        self.assertEqual(s['original_samples'],20)

    def test_failure_denominator_cannot_drop_unsupported(self):
        gs = self.groups(); s = self.fake_group_results(gs[0]); s['unsupported'] = 0
        r.write(Path(gs[0]['folder'])/'rescue_v2/summary.json', s)
        with self.assertRaisesRegex(RuntimeError,'partition failures'):
            r.aggregate(self.out,dict(groups=gs))

    def test_source_drift_rejected(self):
        f = self.root/'source.py'; f.write_text('original')
        expected = {'source.py':r.sha(f)}; f.write_text('changed')
        with self.assertRaises(RuntimeError):
            r.check_hashes(self.root,expected)

    def test_attempted_and_completed_jobs_are_separate(self):
        cmd = [sys.executable,'record_scaffold_run.py']
        r.write(self.out/'receipts/a.json', dict(command=cmd,state='completed'))
        r.write(self.out/'receipts/b.json', dict(command=cmd,state='stopped'))
        r.write(self.out/'receipts/c.json', dict(command=['analysis.py'],state='completed'))
        counts = r.progress_counts(self.out)
        self.assertEqual(counts['generation_jobs_launched'],2)
        self.assertEqual(counts['generation_jobs_completed'],1)

    def test_compatibility_installer_backups_and_is_idempotent(self):
        target = self.root/'scaffold_study_v1/terminal_control.py'; target.parent.mkdir()
        fixed = (r.HERE/'terminal_control.py').read_text()
        old = fixed.replace("'fixed_chemistry_preserved': graph['fixed_chemistry_preserved']",
                            "'fixed_chemistry_preserved': None")
        target.write_text(old)
        install_compat.install(self.root)
        self.assertEqual(target.read_text(),fixed)
        backups = list((self.root/'archive').glob('*/terminal_control.py'))
        self.assertEqual(len(backups),1); self.assertEqual(backups[0].read_text(),old)
        install_compat.install(self.root)
        self.assertEqual(len(list((self.root/'archive').glob('*/terminal_control.py'))),1)

    def test_invalid_decode_keeps_raw_fixed_label_for_baseline_comparison(self):
        # Execute the real patched evaluate() early-return path with dependency stubs.
        tree = ast.parse((r.HERE/'terminal_control.py').read_text())
        function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
        modules = {name:types.ModuleType(name) for name in
                   ['numpy','rdkit','rdkit.Chem','rdkit.Chem.Scaffolds','posebusters',
                    'flowr','flowr.util','flowr.util.rdkit','flowr.gen','flowr.gen.utils']}
        modules['rdkit'].Chem = modules['rdkit.Chem']
        modules['rdkit.Chem.Scaffolds'].MurckoScaffold = object()
        modules['posebusters'].PoseBusters = object()
        modules['flowr.util.rdkit'].mol_is_valid = lambda mol,**kwargs:False
        modules['flowr.gen.utils'].check_substructure_match = lambda *args:False
        vocab=types.SimpleNamespace(tokens_from_indices=lambda x:list(x))
        builder=types.SimpleNamespace(vocab=vocab,vocab_charges=vocab,repair_stats={})
        tensor=types.SimpleNamespace(shape=(1,1,3))
        for label in (True,False):
            evaluator=types.SimpleNamespace(inspect_raw_graph=lambda *args, v=label:dict(fixed_chemistry_preserved=v))
            ns={'decode':lambda *args:None,'copy':__import__('copy')}
            with mock.patch.dict(sys.modules,modules):
                exec(compile(ast.Module(body=[function],type_ignores=[]),'evaluate-test','exec'),ns)
                row,mol=ns['evaluate'](builder,dict(atomics=tensor,charges=tensor),None,
                                      dict(fixed_sampler_indices=[0]),None,False,evaluator)
            self.assertIsNone(mol)
            self.assertIs(row['fixed_chemistry_preserved'],label)
            self.assertFalse(row['rdkit_valid'])

    def test_full_synthetic_pipeline_100_jobs_and_25_analysis_stages(self):
        # No molecular data, CUDA or torch: this tests orchestration with fake child tools.
        groups = self.groups()
        (self.root/'checkpoint.ckpt').write_text('synthetic checkpoint')
        core = ['record_scaffold_run.py','scaffold_evaluation_v2/evaluate_scaffold_outputs.py',
                'scaffold_diagnostics_v1/diagnose_scaffold_runs.py',
                'scaffold_geometry_audit_v1/run_audit.py','scaffold_study_v1/run.py',
                'scaffold_study_v1/terminal_control.py']
        common = '''
import argparse,json,sys
from pathlib import Path
repo=Path(__file__).resolve().parent
plan=json.loads((repo/'outputs/frozen_plan.json').read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
p=argparse.ArgumentParser()
p.add_argument('--case');p.add_argument('--ligand_idx',type=int);p.add_argument('--seed',type=int)
p.add_argument('--ckpt_path');p.add_argument('--save_dir');p.add_argument('--run-dirs',nargs='+')
p.add_argument('--save-dir');p.add_argument('--quality-report');p.add_argument('--output-dir')
p.add_argument('--repo');p.add_argument('--cohort-dir');p.add_argument('--diagnostic-dir')
p.add_argument('--geometry-dir');p.add_argument('--cohort')
a=p.parse_args()
name=Path(sys.argv[0]).name
if name=='record_scaffold_run.py':
 root=Path(a.save_dir)/'research_record';folder=root/'attempt_000'
 args=dict(case=None,seed=a.seed,ligand_idx=a.ligand_idx,pdb_file=str(repo/f'examples/{a.case}_protein.pdb'),ligand_file=str(repo/f'examples/{a.case}_ligands.sdf'),save_dir=a.save_dir,ckpt_path=a.ckpt_path)
 args.pop('case')
 write(root/'manifest.json',dict(args=args,checkpoint_sha256=plan['checkpoint_sha256'],recorder_sha256=plan['source_sha256']['record_scaffold_run.py'],torch_version='fake',rdkit_version='fake',source_sha256=plan['source_sha256'],input_sha256={}))
 write(root/'summary.json',dict(pipeline_error=None,n_attempts=1,attempts=[dict(generation_completed=True,forward_calls=101)]))
 write(folder/'task_definition.json',dict(mapping_verified=True))
 for d,base in [('forward_calls','call'),('sampler_steps','step')]:
  (folder/d).mkdir()
  for i in range(101):(folder/d/f'{base}_{i:04d}.pt').write_text('synthetic')
 for n in ['inputs.pt','raw_final_prediction.pt']:(folder/n).write_text('synthetic')
elif name=='evaluate_scaffold_outputs.py':
 write(Path(a.save_dir)/'quality_report.json',dict(completed=True,errors=[],requested_run_count=len(a.run_dirs),runs=a.run_dirs))
elif name=='diagnose_scaffold_runs.py':
 write(Path(a.output_dir)/'summary.json',dict(completed=True))
elif name=='run_audit.py':
 write(Path(a.output_dir)/'summary.json',dict(completed=True,error_count=0))
elif name=='run.py':
 write(Path(a.output_dir)/'diagnosis.json',dict(n_runs=20))
elif name=='terminal_control.py':
 out=Path(a.cohort)/'rescue_v2'
 write(out/'status.json',dict(completed=True))
 write(out/'summary.json',dict(completed=True,original_samples=20,original_passes=12,original_failures=8,rescued_without_C=3,rescued_with_fixed_coordinate_relaxation=2,unresolved=2,unsupported=1,branch_errors=0))
 group=next(g for g in plan['groups'] if g['folder']==str(Path(a.cohort)))
 write(out/'selected_results.json',[dict(run_id=j['id']) for j in group['jobs']])
else:raise RuntimeError(name)
'''
        (self.root/'fixture_impl.py').write_text(common)
        for name in core:
            f = self.root/name; f.parent.mkdir(parents=True,exist_ok=True)
            f.write_text(f'import sys\nsys.path.insert(0,{str(self.root)!r})\nimport fixture_impl\n')
        plan = dict(repo=str(self.root),output=str(self.out),groups=groups,
                    package_sha256=r.package_hashes(),versions={'torch':'fake','rdkit':'fake'},
                    source_sha256={n:r.sha(self.root/n) for n in core},input_sha256={},
                    checkpoint=str(self.root/'checkpoint.ckpt'),checkpoint_sha256=r.sha(self.root/'checkpoint.ckpt'),
                    baseline_args={})
        r.write(self.out/'frozen_plan.json', plan)
        (self.out/'frozen_plan.sha256').write_text(r.sha(self.out/'frozen_plan.json'))
        with mock.patch.object(r,'versions',return_value=plan['versions']):
            r.run(self.root,self.out)
            # Resume must verify/reuse all 125 receipts without launching subprocesses.
            with mock.patch.object(r.subprocess,'Popen',side_effect=AssertionError('must not repeat')):
                r.run(self.root,self.out)
        self.assertEqual(len(list((self.out/'receipts').glob('*.json'))),125)
        self.assertEqual(r.progress_counts(self.out)['generation_jobs_completed'],100)
        self.assertEqual(r.read(self.out/'summary.json')['original_failures'],40)
        self.assertTrue(r.read(self.out/'status.json')['completed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
