"""CPU software tests. Synthetic subprocess outputs are not molecular results."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
PAYLOAD = HERE/'payload'
sys.path.insert(0, str(PAYLOAD/'scaffold_validation100'))
import run as r
from settings import load_config
import install


def inputs(n=10):
    return [dict(case=['cdk2','tyk2','ptp1b','bace'][i%4], ligand_idx=i+10,
                 input_smiles=f'graph_{i}') for i in range(n)]


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def parse(self, suffix='', **changes):
        values = dict(INPUTS=[('cdk2',5)], SAMPLES_PER_INPUT=10, SEED_START=2001,
                      INPUTS_PER_GROUP=2, OUTPUT_PREFIX='test')
        values.update(changes)
        p=self.root/'config.py'
        p.write_text('\n'.join(f'{k} = {v!r}' for k,v in values.items())+suffix)
        return load_config(p)

    def test_default_repeats_previous_100_input_plan(self):
        c=load_config(PAYLOAD/'scaffold_validation100/config.py')
        self.assertEqual([(x['case'],x['ligand_idx']) for x in c['inputs']],
            [('cdk2',5),('tyk2',4),('ptp1b',4),('bace',3),('cdk2',6),
             ('tyk2',5),('ptp1b',5),('bace',4),('cdk2',8),('tyk2',6)])
        self.assertEqual(c['seeds'],list(range(2001,2011)))
        gs=r.group_plan(c['inputs'],c['seeds'],self.root,2)
        self.assertEqual([len(g['jobs']) for g in gs],[20]*5)

    def test_sizes_and_partial_groups(self):
        for n,m,k,expected in [(1,1,2,[1]),(3,7,2,[14,7]),(2,25,1,[25,25]),
                                (3,13,3,[39]),(5,3,2,[6,6,3])]:
            with self.subTest(n=n,m=m,k=k):
                gs=r.group_plan(inputs(n),list(range(m)),self.root,k)
                self.assertEqual([len(g['jobs']) for g in gs],expected)
                for item in inputs(n):
                    matching=[g for g in gs if any(j['case']==item['case'] and j['ligand_idx']==item['ligand_idx'] for j in g['jobs'])]
                    self.assertEqual(len(matching),1)
                self.assertEqual(sum(expected),n*m)

    def test_same_graph_in_different_contexts_is_allowed(self):
        a=inputs(2);a[1]['input_smiles']=a[0]['input_smiles']
        self.assertEqual(len(r.group_plan(a,[1],self.root,2)[0]['jobs']),2)

    def test_duplicate_jobs_rejected(self):
        with self.assertRaises(RuntimeError):r.group_plan(inputs(1),[1,1],self.root,2)
        with self.assertRaises(RuntimeError):r.group_plan([inputs(1)[0]]*2,[1],self.root,2)

    def test_parameter_validation(self):
        for changes in [dict(SAMPLES_PER_INPUT=0),dict(SAMPLES_PER_INPUT=True),
                        dict(INPUTS_PER_GROUP=0),dict(SEED_START=-1),dict(SEED_START=2**32-1),
                        dict(INPUTS=[]),dict(INPUTS=[('cdk2',-1)]),dict(INPUTS=[('cdk2',True)]),
                        dict(INPUTS=[('cdk2',1),('cdk2',1)]),dict(OUTPUT_PREFIX='../oops'),
                        dict(INPUTS=[('unknown',1)])]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.parse(**changes)

    def test_config_does_not_execute_code_or_ignore_misspellings(self):
        for suffix in ['\nimport os','\nSAMPLES_PER_INPUT=3','\nSAMPLE_PER_INPUT=3',
                       '\nprint("unexpected")']:
            with self.subTest(suffix=suffix),self.assertRaises(ValueError):self.parse(suffix)

    def test_seed_boundary(self):
        c=self.parse(SEED_START=2**32-1,SAMPLES_PER_INPUT=1)
        self.assertEqual(c['seeds'],[2**32-1])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.out=self.root/'outputs/new';self.out.mkdir(parents=True)
        self.quiet=contextlib.redirect_stdout(io.StringIO());self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None,None,None);self.tmp.cleanup()

    def prepare_fixture(self, selected=None, invalid=False):
        (self.root/'examples').mkdir()
        for case in ['cdk2','tyk2','ptp1b','bace']:
            for ext in ['ligands.sdf','protein.pdb']:(self.root/f'examples/{case}_{ext}').write_text('fixture input')
        (self.root/'checkpoint').write_text('fixture weights')
        method=self.root/'method.json'
        r.write(method,dict(source_sha256={},checkpoint='checkpoint',checkpoint_sha256=r.sha(self.root/'checkpoint'),
                           versions={'torch':'fake'},baseline_args={}))
        settings=dict(inputs=selected or inputs(3),seeds=[1,2],samples_per_input=2,
                      seed_start=1,inputs_per_group=2,output_prefix='test')
        chem=types.ModuleType('rdkit.Chem');chem.SDMolSupplier=lambda *a,**kw:list(range(100))
        rdkit=types.ModuleType('rdkit');rdkit.Chem=chem
        interp=types.ModuleType('flowr.data.interpolate');interp.extract_scaffolds=object()
        modules={'rdkit':rdkit,'rdkit.Chem':chem,'flowr':types.ModuleType('flowr'),
                 'flowr.data':types.ModuleType('flowr.data'),'flowr.data.interpolate':interp}
        def profile(mol,*args):
            if invalid and mol==11:raise ValueError('Invalid test ligand')
            return dict(input_smiles=f'mol{mol}')
        stack=contextlib.ExitStack()
        stack.enter_context(mock.patch.dict(sys.modules,modules))
        stack.enter_context(mock.patch.object(r,'METHOD',method))
        stack.enter_context(mock.patch.object(r,'versions',return_value={'torch':'fake'}))
        stack.enter_context(mock.patch.object(r.selection,'input_profile',side_effect=profile))
        return settings,stack

    def test_bad_history_does_not_block_prepare_and_repetition(self):
        for i in range(3):
            old=self.root/f'outputs/old{i}/research_record';old.mkdir(parents=True)
            (old/'manifest.json').write_text('version https://git-lfs.github.com/spec/v1\n')
            (old/'summary.json').write_text('not JSON')
        settings,stack=self.prepare_fixture()
        with stack:
            r.prepare(self.root,self.out,settings)
            second=self.root/'outputs/again';second.mkdir()
            r.prepare(self.root,second,settings)
        inv=r.read(self.out/'input_inventory.json')
        self.assertFalse(inv['history_scanned'])
        self.assertEqual(inv['requested_inputs'],3)
        self.assertEqual(r.read(self.out/'frozen_plan.json')['planned_generations'],6)
        self.assertEqual(inv['selected_inputs'],r.read(second/'input_inventory.json')['selected_inputs'])

    def test_explicit_invalid_input_is_not_silently_replaced(self):
        settings,stack=self.prepare_fixture(invalid=True)
        with stack,self.assertRaisesRegex(RuntimeError,'configured inputs are invalid'):
            r.prepare(self.root,self.out,settings)
        self.assertFalse((self.out/'frozen_plan.json').exists())
        inv=r.read(self.out/'input_inventory.json')
        self.assertEqual(inv['inventory'][1]['status'],'ineligible_input')

    def test_group_counts_flow_into_validators_and_rescue_limit(self):
        gs=r.group_plan(inputs(3),list(range(7)),self.out,2)
        for g in gs:
            cs=r.commands(self.root,g);count=len(g['jobs'])
            cmd=cs[-1][1]
            self.assertEqual(cmd[cmd.index('--max-runs')+1],str(count))
            q=Path(g['folder'])/'quality/quality_report.json'
            r.write(q,dict(completed=True,requested_run_count=count,runs=[None]*count))
            cs[0][3]()
            r.write(q,dict(completed=True,requested_run_count=20,runs=[None]*20))
            with self.assertRaises(RuntimeError):cs[0][3]()

    def fake_results(self,g):
        n=len(g['jobs']);p=Path(g['folder'])/'rescue_v2'
        s=dict(completed=True,original_samples=n,original_passes=0,original_failures=n,
               rescued_without_C=0,rescued_with_fixed_coordinate_relaxation=0,unresolved=n-1,unsupported=1,branch_errors=0)
        r.write(p/'summary.json',s)
        r.write(p/'selected_results.json',[dict(run_id=j['id']) for j in g['jobs']])
        return p,s

    def test_partial_aggregate_uses_real_denominators(self):
        gs=r.group_plan(inputs(3),list(range(7)),self.out,2)
        self.fake_results(gs[0]);plan=dict(groups=gs,planned_generations=21)
        s=r.aggregate(self.out,plan)
        self.assertEqual((s['completed_groups'],s['planned_groups'],s['original_samples'],s['planned_samples']),(1,2,14,21))
        self.fake_results(gs[1]);s=r.aggregate(self.out,plan)
        self.assertEqual(s['original_failures'],21);self.assertEqual(s['unsupported'],2)

    def test_missing_samples_and_dropped_failures_rejected(self):
        gs=r.group_plan(inputs(1),[1,2,3],self.out,2);p,s=self.fake_results(gs[0])
        r.write(p/'selected_results.json',[dict(run_id='wrong')]*3)
        with self.assertRaisesRegex(RuntimeError,'identities'):r.aggregate(self.out,dict(groups=gs,planned_generations=3))
        p,s=self.fake_results(gs[0]);s['unsupported']=0;r.write(p/'summary.json',s)
        with self.assertRaisesRegex(RuntimeError,'partition'):r.aggregate(self.out,dict(groups=gs,planned_generations=3))

    def test_receipt_reuse_and_failed_stage_guard(self):
        dest=self.out/'job'
        cmd=[sys.executable,'-c','from pathlib import Path; import sys; p=Path(sys.argv[1]); p.mkdir(); (p/"ok").write_text("done")',str(dest)]
        r.stage(self.root,self.out,'one',cmd,dest,lambda:None)
        with mock.patch.object(r.subprocess,'Popen',side_effect=AssertionError('must not run')):
            r.stage(self.root,self.out,'one',cmd,dest,lambda:None)
        (dest/'ok').write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'changed'):r.stage(self.root,self.out,'one',cmd,dest,lambda:None)
        bad=[sys.executable,'-c','raise SystemExit(3)']
        with self.assertRaises(RuntimeError):r.stage(self.root,self.out,'bad',bad,self.out/'bad',lambda:None)
        with self.assertRaisesRegex(RuntimeError,'no automatic retry'):
            r.stage(self.root,self.out,'bad',bad,self.out/'bad',lambda:None)

    def test_source_and_plan_drift_still_rejected(self):
        p=self.root/'source';p.write_text('original');h=r.sha(p);p.write_text('changed')
        with self.assertRaises(RuntimeError):r.check_hashes(self.root,{'source':h})
        r.write(self.out/'frozen_plan.json',{'test':1});(self.out/'frozen_plan.sha256').write_text('wrong')
        with self.assertRaisesRegex(RuntimeError,'Frozen plan changed'):r.run(self.root,self.out)

    def test_report_contains_actual_settings_not_next_experiment_config(self):
        r.write(self.out/'experiment_settings.json',{'samples_per_input':7})
        (self.out/'raw.pt').write_text('large tensor omitted')
        r.bundle(self.out)
        with zipfile.ZipFile(self.out/'validation100_review.zip') as z:
            self.assertEqual(json.loads(z.read('experiment_settings.json'))['samples_per_input'],7)
            self.assertNotIn('batch_program/config.py',z.namelist())
            self.assertNotIn('raw.pt',z.namelist())

    def test_default_main_prepares_then_runs_once(self):
        def prepare(repo,out,settings):
            r.write(out/'frozen_plan.json',dict(planned_generations=len(settings['inputs'])*settings['samples_per_input']))
        with mock.patch.object(r,'prepare',side_effect=prepare) as p,mock.patch.object(r,'run') as run:
            result=r.main(['--repo',str(self.root)])
        self.assertEqual(result,0);p.assert_called_once();run.assert_called_once()
        out=run.call_args.args[1]
        self.assertTrue((out/'experiment_settings.json').is_file())
        self.assertFalse((out/'RUNNING.lock').exists())

    def test_prepare_error_cannot_start_generation(self):
        with (mock.patch.object(r,'prepare',side_effect=RuntimeError('invalid')),
              mock.patch.object(r,'run') as run, contextlib.redirect_stderr(io.StringIO())):
            result=r.main(['--repo',str(self.root)])
        self.assertEqual(result,1);run.assert_not_called()


FIXTURE = '''
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser()
for key in ['case','seed','ligand_idx','ckpt_path','save_dir','save-dir','quality-report','output-dir','repo','cohort-dir','diagnostic-dir','geometry-dir','cohort','max-runs']:
 p.add_argument('--'+key)
p.add_argument('--run-dirs',nargs='+')
a=p.parse_args()
repo=Path(__file__).resolve().parent
plan=json.loads((repo/'outputs/new/frozen_plan.json').read_text())
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
name=Path(sys.argv[0]).name
if name=='record_scaffold_run.py':
 root=Path(a.save_dir)/'research_record';folder=root/'attempt_000'
 args=dict(seed=int(a.seed),ligand_idx=int(a.ligand_idx),pdb_file=str(repo/f'examples/{a.case}_protein.pdb'),ligand_file=str(repo/f'examples/{a.case}_ligands.sdf'),save_dir=a.save_dir,ckpt_path=a.ckpt_path)
 write(root/'manifest.json',dict(args=args,checkpoint_sha256=plan['checkpoint_sha256'],recorder_sha256=plan['source_sha256']['record_scaffold_run.py'],torch_version='fake',rdkit_version='fake',source_sha256=plan['source_sha256'],input_sha256={}))
 write(root/'summary.json',dict(pipeline_error=None,n_attempts=1,attempts=[dict(generation_completed=True,forward_calls=101)]))
 write(folder/'task_definition.json',dict(mapping_verified=True))
 for d,base in [('forward_calls','call'),('sampler_steps','step')]:
  (folder/d).mkdir()
  for i in range(101):(folder/d/f'{base}_{i:04d}.pt').write_text('synthetic')
 for n in ['inputs.pt','raw_final_prediction.pt']:(folder/n).write_text('synthetic')
elif name=='evaluate_scaffold_outputs.py':
 write(Path(a.save_dir)/'quality_report.json',dict(completed=True,errors=[],requested_run_count=len(a.run_dirs),runs=a.run_dirs))
elif name=='diagnose_scaffold_runs.py' or name=='run_audit.py':
 write(Path(a.output_dir)/'summary.json',dict(completed=True,error_count=0))
elif name=='run.py':
 q=json.loads(Path(a.quality_report).read_text());write(Path(a.output_dir)/'diagnosis.json',dict(n_runs=q['requested_run_count']))
elif name=='terminal_control.py':
 group=next(g for g in plan['groups'] if g['folder']==str(Path(a.cohort)));n=len(group['jobs'])
 assert int(a.max_runs)==n
 out=Path(a.cohort)/'rescue_v2'
 write(out/'status.json',dict(completed=True))
 failed=(n+1)//2;rescued=failed//2
 write(out/'summary.json',dict(completed=True,original_samples=n,original_passes=n-failed,original_failures=failed,rescued_without_C=rescued,rescued_with_fixed_coordinate_relaxation=0,unresolved=failed-rescued,unsupported=0,branch_errors=0))
 write(out/'selected_results.json',[dict(run_id=j['id']) for j in group['jobs']])
else:raise RuntimeError(name)
'''


class PipelineTests(unittest.TestCase):
    setUp = RunnerTests.setUp
    tearDown = RunnerTests.tearDown
    def test_synthetic_pipeline_multiple_sizes_and_resume(self):
        for n,m,k in [(3,7,2),(1,25,1),(10,10,2)]:
            with self.subTest(inputs=n,samples=m,group_inputs=k):
                shutil.rmtree(self.out);self.out.mkdir()
                groups=r.group_plan(inputs(n),list(range(2001,2001+m)),self.out,k)
                (self.root/'checkpoint').write_text('synthetic checkpoint')
                core=['record_scaffold_run.py','scaffold_evaluation_v2/evaluate_scaffold_outputs.py',
                      'scaffold_diagnostics_v1/diagnose_scaffold_runs.py','scaffold_geometry_audit_v1/run_audit.py',
                      'scaffold_study_v1/run.py','scaffold_study_v1/terminal_control.py']
                (self.root/'fixture_impl.py').write_text(FIXTURE)
                for name in core:
                    p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
                    p.write_text(f'import sys\nsys.path.insert(0,{str(self.root)!r})\nimport fixture_impl\n')
                plan=dict(repo=str(self.root),output=str(self.out),groups=groups,planned_generations=n*m,
                          package_sha256=r.package_hashes(),versions={'torch':'fake','rdkit':'fake'},
                          source_sha256={name:r.sha(self.root/name) for name in core},input_sha256={},
                          checkpoint=str(self.root/'checkpoint'),checkpoint_sha256=r.sha(self.root/'checkpoint'),baseline_args={})
                r.write(self.out/'frozen_plan.json',plan)
                (self.out/'frozen_plan.sha256').write_text(r.sha(self.out/'frozen_plan.json'))
                with mock.patch.object(r,'versions',return_value=plan['versions']):
                    r.run(self.root,self.out)
                    with mock.patch.object(r.subprocess,'Popen',side_effect=AssertionError('resume must not generate again')):
                        r.run(self.root,self.out)
                s=r.read(self.out/'summary.json')
                self.assertEqual(s['original_samples'],n*m)
                self.assertEqual(s['completed_groups'],len(groups))
                self.assertEqual(r.progress_counts(self.out)['generation_jobs_completed'],n*m)
                self.assertEqual(len(list((self.out/'receipts').glob('*.json'))),n*m+len(groups)*5)
                self.assertTrue(r.read(self.out/'status.json')['completed'])


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.repo=self.base/'repo';self.repo.mkdir()
        self.update=self.base/'update';self.update.mkdir()
        shutil.copytree(PAYLOAD,self.update/'payload')
        self.original_terminal=b'original terminal file\n'
        (self.repo/'scaffold_study_v1').mkdir()
        (self.repo/install.TERMINAL).write_bytes(self.original_terminal)
        (self.repo/install.BATCH).mkdir()
        (self.repo/install.BATCH/'old.py').write_text('original runner')
        (self.repo/'source.py').write_text('original method')
        payload=self.update/'payload'
        manifest=dict(payload_sha256={str(p.relative_to(payload)):install.sha(p) for p in payload.rglob('*')
                                      if p.is_file() and '__pycache__' not in p.parts},
                      previous_source_sha256={str(install.TERMINAL):install.sha(self.repo/install.TERMINAL),
                                              'source.py':install.sha(self.repo/'source.py')})
        (self.update/'package_manifest.json').write_text(json.dumps(manifest))
        self.patch=mock.patch.object(install,'HERE',self.update);self.patch.start()
        self.quiet=contextlib.redirect_stdout(io.StringIO());self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None,None,None);self.patch.stop();self.tmp.cleanup()

    def test_install_backup_idempotence_and_restore(self):
        backup=install.install(self.repo)
        self.assertEqual((backup/install.TERMINAL).read_bytes(),self.original_terminal)
        self.assertEqual((backup/install.BATCH/'old.py').read_text(),'original runner')
        config=self.repo/install.BATCH/'config.py';config.write_text('user settings')
        self.assertIsNone(install.install(self.repo));self.assertEqual(config.read_text(),'user settings')
        install.restore(self.repo,backup)
        self.assertEqual((self.repo/install.TERMINAL).read_bytes(),self.original_terminal)
        self.assertEqual((self.repo/install.BATCH/'old.py').read_text(),'original runner')
        self.assertTrue(list((self.repo/'archive').glob('validation_before_restore_*/scaffold_validation100/config.py')))

    def test_reinstall_keeps_user_config(self):
        install.install(self.repo)
        config=self.repo/install.BATCH/'config.py';config.write_text('user settings')
        (self.repo/install.BATCH/'run.py').unlink()
        install.install(self.repo)
        self.assertEqual(config.read_text(),'user settings')
        self.assertTrue((self.repo/install.BATCH/'run.py').is_file())

    def test_unknown_source_and_active_lock_block_before_changes(self):
        (self.repo/'source.py').write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'differs'):install.install(self.repo)
        self.assertFalse((self.repo/'archive').exists())
        (self.repo/'source.py').write_text('original method')
        lock=self.repo/'outputs/running/RUNNING.lock';lock.parent.mkdir(parents=True);lock.write_text('active')
        with self.assertRaisesRegex(RuntimeError,'lock exists'):install.install(self.repo)
        self.assertFalse((self.repo/'archive').exists())

    def test_write_failure_restores_original_files(self):
        original=Path.replace
        def fail_terminal(path,target):
            if path.name=='new_terminal.py':raise OSError('simulated filesystem failure')
            return original(path,target)
        with mock.patch.object(Path,'replace',fail_terminal),self.assertRaises(OSError):install.install(self.repo)
        self.assertEqual((self.repo/install.TERMINAL).read_bytes(),self.original_terminal)
        self.assertEqual((self.repo/install.BATCH/'old.py').read_text(),'original runner')
        self.assertFalse(list(self.repo.glob('validation_config_stage_*')))

    def test_package_corruption_is_rejected(self):
        (self.update/'payload'/install.BATCH/'run.py').write_text('bad')
        with self.assertRaisesRegex(RuntimeError,'Incomplete update'):install.install(self.repo)
        self.assertFalse((self.repo/'archive').exists())


if __name__=='__main__':unittest.main(verbosity=2)
