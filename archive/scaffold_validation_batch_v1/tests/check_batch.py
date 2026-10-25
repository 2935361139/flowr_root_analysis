"""Software-only tests with explicitly synthetic backends, not GPU/chemistry tests."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'run_batch.py'
spec = importlib.util.spec_from_file_location('batch_tested', SOURCE)
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

RECORD = r'''
import argparse,json,pathlib,sys
p=argparse.ArgumentParser()
for k in ['case','ligand_idx','seed','ckpt_path','save_dir']:p.add_argument('--'+k)
a=p.parse_args();out=pathlib.Path(a.save_dir);plan=json.loads((out.parents[2]/'frozen_plan.json').read_text())
repo=pathlib.Path(plan['repo']);out.mkdir(parents=True);rec=out/'research_record';rec.mkdir()
if (repo/'inject_record_failure').exists() and int(a.seed)==43:
 print('INTENTIONAL SYNTHETIC FAILURE');sys.exit(3)
args={'seed':int(a.seed),'ligand_idx':int(a.ligand_idx),'pdb_file':str(repo/f'examples/{a.case}_protein.pdb'),'scaffold_hopping':True}
manifest={'args':args,'checkpoint_sha256':plan['checkpoint_sha256'],'torch_version':plan['torch_version'],'rdkit_version':plan['rdkit_version'],'recorder_sha256':plan['package_sha256']['helpers/record_scaffold_run.py'],'source_sha256':plan['config']['source_sha256'],'input_sha256':plan['input_sha256']}
(rec/'manifest.json').write_text(json.dumps(manifest))
(rec/'summary.json').write_text(json.dumps({'n_attempts':1,'pipeline_error':None,'attempts':[{'generation_completed':True,'audit_error':None,'forward_calls':101}]}))
folder=rec/'attempt_000';folder.mkdir();(folder/'task_definition.json').write_text(json.dumps({'mapping_verified':True}))
calls=folder/'forward_calls';calls.mkdir()
for i in range(101):(calls/f'call_{i:04d}.pt').write_bytes(b'SYNTHETIC TEST MARKER; NOT A TENSOR')
'''
TRACE = r'''
import pathlib,json,sys
p=pathlib.Path(sys.argv[1])/'research_record/attempt_000/trace_check';p.mkdir()
(p/'trace_report.json').write_text(json.dumps({'snapshot_counts_match_recorder':True}))
'''
QUALITY = r'''
import pathlib,argparse,json
p=argparse.ArgumentParser();p.add_argument('--run-dirs',nargs='+');p.add_argument('--save-dir');a=p.parse_args()
out=pathlib.Path(a.save_dir);out.mkdir()
rows=[{'run_directory':d,'stage':'raw_without_repair','errors':[],'fc_valid':True,'root_fixed_substructure_match':i%2==0,'posebusters_valid':False,'fixed_chemistry_preserved':i%2==0} for i,d in enumerate(a.run_dirs)]
(out/'quality_report.json').write_text(json.dumps({'completed':True,'rows':rows}))
'''


class Tests(unittest.TestCase):
    def test_input_selection_excludes_history_and_duplicates(self):
        items=[{'case':c,'ligand_idx':i,'input_smiles':f'g{i}' if i==0 else f'{c}{i}'} for c in ['a','b'] for i in range(4)]
        chosen=b.choose_inputs(items,['b','a'],4,{('a',1)},{'b1'})
        self.assertEqual([(x['case'],x['ligand_idx']) for x in chosen],[('b',0),('a',2),('b',2),('a',3)])
        self.assertEqual(len({x['input_smiles'] for x in chosen}),4)
        self.assertEqual(b.choose_inputs(items,['a','b'],99,set(),{x['input_smiles'] for x in items}),[])

    def test_missing_evaluation_is_not_failure(self):
        with tempfile.TemporaryDirectory() as td:
            plan={'selected_inputs':[{'case':'a','ligand_idx':17}],'seeds':[42,43,44,45]}
            jobs=b.jobs_for(plan,Path(td))
            q={'completed':False,'rows':[
                {'run_directory':jobs[0]['run_directory'],'stage':'raw_without_repair','errors':[],'fc_valid':False,'root_fixed_substructure_match':None},
                {'run_directory':jobs[1]['run_directory'],'stage':'raw_without_repair','errors':['PB error'],'fc_valid':True,'root_fixed_substructure_match':True},
                {'run_directory':jobs[2]['run_directory'],'stage':'raw_without_repair','errors':[],'fc_valid':True,'root_fixed_substructure_match':None}]}
            summary=b.make_summary(plan,jobs,q)
            self.assertEqual(summary['chemical_fragment_endpoint_counts'],{'fail':1,'missing_or_error':2,'not_assessed':1})
            self.assertIsNone(summary['rows'][3]['valid_connected_root_fragment'])
            self.assertEqual(summary['causal_interventions_performed'],0)

    def test_history_uses_input_manifests_not_generated_labels(self):
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td);out=repo/'outputs/new'
            old=repo/'outputs/old/research_record';old.mkdir(parents=True)
            b.write_json(old/'manifest.json',{'args':{'pdb_file':'/a/bace_protein.pdb','ligand_idx':7}})
            b.write_json(old/'generated_labels.json',{'poison':'Must not be inspected'})
            config={'explicit_excluded_inputs':[['bace',16]],'existing_records_root':'outputs'}
            used,graphs,manifests,errors=b.previous_inputs(repo,out,config,None)
            self.assertEqual(used,{('bace',7),('bace',16)});self.assertFalse(errors)
            (old/'manifest.json').write_text('bad JSON')
            self.assertEqual(len(b.previous_inputs(repo,out,config,None)[3]),1)

    def fake_run(self, td):
        repo=Path(td);pkg=repo/'package';(pkg/'helpers').mkdir(parents=True)
        out=repo/'outputs/cohort';out.mkdir(parents=True)
        (repo/'flowr/gen').mkdir(parents=True);(repo/'flowr/gen/generate_from_pdb.py').write_text('synthetic source')
        (repo/'checkpoints').mkdir();(repo/'checkpoints/model.ckpt').write_bytes(b'synthetic checkpoint')
        (repo/'examples').mkdir()
        for c in ['a','b']:(repo/f'examples/{c}_protein.pdb').write_text('synthetic input')
        (pkg/'run_batch.py').write_bytes(SOURCE.read_bytes())
        for name,src in [('record_scaffold_run.py',RECORD),('inspect_scaffold_trace.py',TRACE),('quality.py',QUALITY)]:
            (pkg/'helpers'/name).write_text(src)
        config={'source_sha256':{'flowr/gen/generate_from_pdb.py':b.sha(repo/'flowr/gen/generate_from_pdb.py')},
                'helper_sha256':{f'helpers/{x.name}':b.sha(x) for x in (pkg/'helpers').iterdir()},'checkpoint':'checkpoints/model.ckpt'}
        (pkg/'batch_config.json').write_text(json.dumps(config))
        old=b.HERE;b.HERE=pkg;self.addCleanup(setattr,b,'HERE',old)
        plan={'repo':str(repo),'output':str(out),'config':config,
              'selected_inputs':[{'case':'a','ligand_idx':17},{'case':'b','ligand_idx':34}],'seeds':[42,43],
              'checkpoint_sha256':b.sha(repo/config['checkpoint']),'torch_version':'FAKE_TEST','rdkit_version':'FAKE_TEST',
              'input_sha256':{str(x):b.sha(x) for x in (repo/'examples').iterdir()},'package_sha256':b.package_hashes()}
        b.write_json(out/'frozen_plan.json',plan);(out/'frozen_plan.sha256').write_text(b.sha(out/'frozen_plan.json'))
        return repo,out,config,plan

    def fake_modules(self):
        return patch.dict(sys.modules,{'torch':types.SimpleNamespace(__version__='FAKE_TEST',cuda=types.SimpleNamespace(is_available=lambda:True)),
                          'rdkit':types.SimpleNamespace(rdBase=types.SimpleNamespace(rdkitVersion='FAKE_TEST'))})

    def test_complete_and_resume_without_regeneration(self):
        with tempfile.TemporaryDirectory() as td,self.fake_modules():
            repo,out,config,plan=self.fake_run(td);b.run(repo,out,config)
            before={str(p):p.stat().st_mtime_ns for p in (out/'runs').rglob('*.pt')}
            self.assertEqual(len(before),404)
            self.assertEqual(b.read_json(out/'baseline_summary.json')['chemical_fragment_endpoint_counts'],{'pass':2,'fail':2})
            b.run(repo,out,config)
            self.assertEqual(before,{str(p):p.stat().st_mtime_ns for p in (out/'runs').rglob('*.pt')})
            import zipfile
            with zipfile.ZipFile(out/'validation_review.zip') as z:
                self.assertIsNone(z.testzip());self.assertFalse(any(n.endswith('.pt') for n in z.namelist()))

    def test_fail_stop_preserves_partial_record(self):
        with tempfile.TemporaryDirectory() as td,self.fake_modules():
            repo,out,config,plan=self.fake_run(td);(repo/'inject_record_failure').write_text('yes')
            with self.assertRaisesRegex(RuntimeError,'Command failed'):b.run(repo,out,config)
            self.assertEqual(b.read_json(out/'batch_status.json')['complete'],1)
            jobs=b.jobs_for(plan,out);self.assertFalse(Path(jobs[2]['run_directory']).exists())
            (repo/'inject_record_failure').unlink()
            with self.assertRaises(FileNotFoundError):b.run(repo,out,config)
            self.assertFalse(Path(jobs[2]['run_directory']).exists())

    def test_changed_program_or_plan_stops(self):
        with tempfile.TemporaryDirectory() as td,self.fake_modules():
            repo,out,config,plan=self.fake_run(td);(b.HERE/'helpers/quality.py').write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'Program changed'):b.run(repo,out,config)
            self.assertFalse((out/'runs').exists())
            plan['seeds']=[99];b.write_json(out/'frozen_plan.json',plan)
            with self.assertRaisesRegex(RuntimeError,'Frozen plan was edited'):b.run(repo,out,config)


if __name__=='__main__':
    unittest.main(verbosity=2)
