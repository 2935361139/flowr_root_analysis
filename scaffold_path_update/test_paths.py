"""CPU tests for paths, classification, selection, installation and orchestration.
Synthetic subprocesses test software behavior, not molecular repair efficacy.
"""
import ast,contextlib,importlib.util,io,json,shutil,sys,tempfile,types,unittest,zipfile
from pathlib import Path
from unittest import mock

HERE=Path(__file__).resolve().parent;PAYLOAD=HERE/'payload'
sys.path.insert(0,str(PAYLOAD/'scaffold_validation100'))
import run as r
import path_io as io_paths
from settings import load_config
spec=importlib.util.spec_from_file_location('paths_installer',HERE/'install.py');installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)

DEFAULTS=dict(SDF_PATH='inputs/source.sdf',PDB_PATH='inputs/source.pdb',LIGAND_INDICES=[5],SAMPLES_PER_INPUT=5,SEED_START=3001,
 GENERATION_DIR='outputs/new/generated',INTERVENTION_INPUT_DIR='outputs/new/generated/failed',INTERVENTION_OUTPUT_DIR='outputs/new/intervention',INPUTS_PER_GROUP=2)

class Base(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.repo=Path(self.tmp.name);self.quiet=contextlib.redirect_stdout(io.StringIO());self.quiet.__enter__()
 def tearDown(self):self.quiet.__exit__(None,None,None);self.tmp.cleanup()
 def config(self,**changes):
  d=DEFAULTS|changes;p=self.repo/'config.py';p.write_text('\n'.join(f'{k}={v!r}' for k,v in d.items()));return p,load_config(p,self.repo)

class ConfigTests(Base):
 def test_paths_resolved_against_repo_and_total_five(self):
  _,c=self.config();self.assertEqual(c['sdf_path'],str(self.repo/'inputs/source.sdf'));self.assertEqual(c['seeds'],[3001,3002,3003,3004,3005])
 def test_all_indices_and_absolute_paths(self):
  _,c=self.config(SDF_PATH='/tmp/a space.sdf',LIGAND_INDICES='all');self.assertEqual(c['sdf_path'],'/tmp/a space.sdf');self.assertEqual(c['ligand_indices'],'all')
 def test_reject_bad_config_and_execution(self):
  for change in [dict(LIGAND_INDICES=[]),dict(LIGAND_INDICES=[0,0]),dict(LIGAND_INDICES=[True]),dict(LIGAND_INDICES='ALL'),dict(SEED_START=-1),dict(SAMPLES_PER_INPUT=0),dict(INPUTS_PER_GROUP=True),dict(SDF_PATH='')]:
   with self.subTest(change=change),self.assertRaises(ValueError):self.config(**change)
  p,_=self.config();p.write_text(p.read_text()+'\nimport os')
  with self.assertRaises(ValueError):load_config(p,self.repo)
 def test_layout_rejects_overwrite_nested_outputs_and_wrong_input(self):
  for change in [dict(INTERVENTION_OUTPUT_DIR='outputs/new/generated/repair'),dict(INTERVENTION_INPUT_DIR='elsewhere'),dict(SDF_PATH='outputs/new/generated/m.sdf')]:
   _,c=self.config(**change)
   with self.assertRaises(RuntimeError):io_paths.check_layout(c)
  _,c=self.config();Path(c['generation_dir']).mkdir(parents=True)
  with self.assertRaisesRegex(RuntimeError,'already exists'):io_paths.check_layout(c)
 def test_arbitrary_identity_and_last_short_group(self):
  items=[dict(case='any_protein',ligand_idx=i,pdb_file='/a/p.pdb',ligand_file='/a/l.sdf') for i in range(3)]
  groups=r.group_plan(items,list(range(7)),self.repo,2);self.assertEqual([len(g['jobs']) for g in groups],[14,7])
  self.assertEqual(groups[0]['jobs'][0]['pdb_file'],'/a/p.pdb')
 def test_recorder_explicit_pair_and_old_cli(self):
  spec=importlib.util.spec_from_file_location('path_recorder',PAYLOAD/'record_scaffold_run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  _,o=m.parse_options(['--pdb_file','other.pdb','--ligand_file','different.sdf','--input_name','new_1']);self.assertEqual(str(o.ligand_file),'different.sdf')
  _,o=m.parse_options(['--case','cdk2']);self.assertIsNone(o.pdb_file)
  with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):m.parse_options(['--pdb_file','alone.pdb'])

FIXTURE=r'''
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser()
for k in ['pdb_file','ligand_file','input_name','seed','ligand_idx','ckpt_path','save_dir','save-dir','quality-report','output-dir','repo','cohort-dir','diagnostic-dir','geometry-dir','cohort','max-runs','failure-ids']:p.add_argument('--'+k)
p.add_argument('--run-dirs',nargs='+');a=p.parse_args()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
def text(p,t='synthetic'):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(t)
def findplan(p):
 p=Path(p)
 for d in [p,*p.parents]:
  if (d/'frozen_plan.json').is_file():return read(d/'frozen_plan.json')
 raise RuntimeError('missing plan')
keys=['rdkit_valid','fc_valid','root_fixed_substructure_match','fixed_chemistry_preserved','posebusters_valid']
name=Path(sys.argv[0]).name
if name=='record_scaffold_run.py':
 plan=findplan(a.save_dir);root=Path(a.save_dir)/'research_record';folder=root/'attempt_000'
 args=dict(seed=int(a.seed),ligand_idx=int(a.ligand_idx),pdb_file=a.pdb_file,ligand_file=a.ligand_file,save_dir=a.save_dir,ckpt_path=a.ckpt_path)
 write(root/'manifest.json',dict(case_label=a.input_name,args=args,checkpoint_sha256=plan['checkpoint_sha256'],recorder_sha256=plan['source_sha256']['record_scaffold_run.py'],torch_version='fake',rdkit_version='fake',source_sha256=plan['source_sha256'],input_sha256=plan['input_sha256']))
 write(root/'summary.json',dict(pipeline_error=None,n_attempts=1,attempts=[dict(generation_completed=True,forward_calls=101)]));write(folder/'task_definition.json',dict(mapping_verified=True))
 for d,base in [('forward_calls','call'),('sampler_steps','step')]:
  for i in range(101):text(folder/d/f'{base}_{i:04d}.pt')
 for n in ['inputs.pt','raw_final_prediction.pt']:text(folder/n)
 text(folder/'decoded_without_repair.sdf','synthetic SDF for '+a.input_name+' '+a.seed)
elif name=='evaluate_scaffold_outputs.py':
 rows=[];runs=[]
 for path in a.run_dirs:
  m=read(Path(path)/'research_record/manifest.json');seed=m['args']['seed'];e={k:True for k in keys};e['posebusters_valid']=seed%2==0
  if seed%7==0:e['fixed_chemistry_preserved']=False
  for stage in ['raw_without_repair','run_decode']:rows.append(dict(run_directory=path,stage=stage,errors=[],posebusters_failed_checks=[] if e['posebusters_valid'] else ['minimum_distance_to_waters'],**e))
  runs.append(dict(run_directory=path))
 write(Path(a.save_dir)/'quality_report.json',dict(completed=True,errors=[],requested_run_count=len(runs),runs=runs,rows=rows))
elif name in ['diagnose_scaffold_runs.py','run_audit.py']:write(Path(a.output_dir)/'summary.json',dict(completed=True,error_count=0))
elif name=='run.py':write(Path(a.output_dir)/'diagnosis.json',dict(n_runs=read(a.quality_report)['requested_run_count']))
elif name=='terminal_control.py':
 plan=findplan(a.cohort);group=next(g for g in plan['groups'] if g['folder']==a.cohort);assert int(a.max_runs)==len(group['jobs'])
 requested=set(read(a.failure_ids));out=Path(a.output_dir);q=read(Path(a.cohort)/'quality/quality_report.json');choices=[]
 for job in group['jobs']:
  rid=job['id'];row=next(x for x in q['rows'] if x['run_directory']==job['run_directory'] and x['stage']=='raw_without_repair');passed=all(row[k] for k in keys)
  arm='baseline' if passed else None;result='original_pass' if passed else ('unsupported' if not row['fixed_chemistry_preserved'] else 'unresolved')
  if not passed and rid not in requested:result='not_requested'
  if rid in requested and row['fixed_chemistry_preserved'] and job['seed']%5!=0:arm='C' if job['seed']%3 else 'B';result='rescued_under_original_evaluator'
  choices.append(dict(run_id=rid,baseline_success=passed,selected_arm=arm,result=result,candidate_errors=[],all_successful_interventions=[arm] if arm in ['B','C'] else []))
  if arm in ['B','C']:text(out/rid/(arm+'.sdf'),'synthetic treated molecule');write(out/rid/(arm+'.json'),dict(evaluation={k:True for k in keys}))
 write(out/'status.json',dict(completed=True,errors=[]));write(out/'selected_results.json',choices)
 write(out/'summary.json',dict(completed=True,branch_errors=0,original_samples=len(choices)))
else:raise RuntimeError(name)
'''

class PipelineTests(Base):
 def fixture(self,n=1,m=5,all_indices=False):
  config,settings=self.config(LIGAND_INDICES='all' if all_indices else list(range(n)),SAMPLES_PER_INPUT=m)
  src=self.repo/'inputs';src.mkdir();(src/'source.sdf').write_text('synthetic input');(src/'source.pdb').write_text('ATOM  synthetic protein\n')
  (self.repo/'weights').write_text('synthetic weights')
  core=['record_scaffold_run.py','scaffold_evaluation_v2/evaluate_scaffold_outputs.py','scaffold_diagnostics_v1/diagnose_scaffold_runs.py','scaffold_geometry_audit_v1/run_audit.py','scaffold_study_v1/run.py','scaffold_study_v1/terminal_control.py']
  (self.repo/'fixture.py').write_text(FIXTURE)
  for name in core:
   p=self.repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(f'import sys\nsys.path.insert(0,{str(self.repo)!r})\nimport fixture\n')
  method=self.repo/'method.json';r.write(method,dict(source_sha256={name:r.sha(self.repo/name) for name in core},checkpoint='weights',checkpoint_sha256=r.sha(self.repo/'weights'),versions={'torch':'fake','rdkit':'fake'},baseline_args={}))
  chem=types.ModuleType('rdkit.Chem');chem.SDMolSupplier=lambda *a,**kw:list(range(n));rdkit=types.ModuleType('rdkit');rdkit.Chem=chem
  interp=types.ModuleType('flowr.data.interpolate');interp.extract_scaffolds=None
  modules={'rdkit':rdkit,'rdkit.Chem':chem,'flowr':types.ModuleType('flowr'),'flowr.data':types.ModuleType('flowr.data'),'flowr.data.interpolate':interp}
  stack=contextlib.ExitStack();stack.enter_context(mock.patch.dict(sys.modules,modules));stack.enter_context(mock.patch.object(r,'METHOD',method));stack.enter_context(mock.patch.object(r,'versions',return_value={'torch':'fake','rdkit':'fake'}));stack.enter_context(mock.patch.object(r.selection,'input_profile',side_effect=lambda mol,*a:dict(input_smiles=f'graph{mol}')))
  return config,settings,stack
 def test_full_five_samples_and_classified_originals_unchanged(self):
  c,s,stack=self.fixture()
  with stack:self.assertEqual(r.main(['--repo',str(self.repo),'--config',str(c)]),0)
  gen=Path(s['generation_dir']);dest=Path(s['intervention_output_dir']);summary=r.read(dest/'summary.json')
  self.assertEqual(summary['original_samples'],5);self.assertEqual(summary['requested_failures'],3);self.assertEqual(summary['repaired'],1);self.assertEqual(summary['unsupported'],1)
  self.assertEqual(len(list((gen/'success').glob('*/molecule.sdf'))),2);self.assertEqual(len(list((gen/'failed').glob('*/molecule.sdf'))),3)
  self.assertEqual(len(list((dest/'success').glob('*/molecule.sdf'))),1);self.assertEqual(len(list((dest/'failed').glob('*/molecule.sdf'))),2)
  for item in r.read(gen/'sample_index.json'):self.assertEqual(r.sha(item['source_sdf']),item['sdf_sha256'])
  with zipfile.ZipFile(dest/'validation100_review.zip') as z:self.assertIn('generation/inputs/protein.pdb',z.namelist());self.assertIn('intervention/summary.json',z.namelist())
 def test_generate_then_subset_intervene_without_generation(self):
  c,s,stack=self.fixture(n=3,m=7,all_indices=True)
  with stack:
   self.assertEqual(r.main(['--repo',str(self.repo),'--config',str(c),'--generate-only']),0)
   gen=Path(s['generation_dir']);self.assertEqual(r.read(gen/'generation_summary.json')['classified_samples'],21)
   selected=next(x for x in sorted((gen/'failed').iterdir()) if x.is_dir());subset=self.repo/'chosen';subset.mkdir();shutil.copytree(selected,subset/selected.name)
   (self.repo/'inputs/source.sdf').write_text('replaced source: archived copy must remain usable')
   c,_=self.config(INTERVENTION_INPUT_DIR=str(subset),INTERVENTION_OUTPUT_DIR='outputs/subset_repair')
   real_stage=r.stage
   def stage(repo,out,key,cmd,*rest):
    self.assertNotIn('record_scaffold_run.py',[Path(x).name for x in cmd]);return real_stage(repo,out,key,cmd,*rest)
   with mock.patch.object(r,'stage',side_effect=stage):self.assertEqual(r.main(['--repo',str(self.repo),'--config',str(c),'--intervene-only']),0)
  summary=r.read(self.repo/'outputs/subset_repair/summary.json');self.assertEqual(summary['requested_failures'],1);self.assertEqual(summary['unselected_failures'],summary['original_failures']-1)
 def test_prepare_then_run_and_safe_completed_resume(self):
  c,s,stack=self.fixture(n=1,m=25)
  with stack:
   self.assertEqual(r.main(['--repo',str(self.repo),'--config',str(c),'--prepare']),0)
   self.assertFalse((Path(s['generation_dir'])/'generation_summary.json').exists())
   args=['--repo',str(self.repo),'--run','--output',s['generation_dir']]
   self.assertEqual(r.main(args),0)
   with mock.patch.object(r.subprocess,'Popen',side_effect=AssertionError('must not regenerate')):self.assertEqual(r.main(args),0)
  self.assertEqual(r.read(Path(s['generation_dir'])/'generation_summary.json')['classified_samples'],25)
 def test_invalid_selected_input_stops_before_any_generation(self):
  c,s,stack=self.fixture()
  with stack,mock.patch.object(r.selection,'input_profile',side_effect=ValueError('no scaffold')),mock.patch.object(r.subprocess,'Popen') as proc,contextlib.redirect_stderr(io.StringIO()):
   self.assertEqual(r.main(['--repo',str(self.repo),'--config',str(c)]),1);proc.assert_not_called()
  self.assertIn('ineligible_input',(Path(s['generation_dir'])/'input_inventory.json').read_text())
 def test_corrupt_history_does_not_block_and_different_sdf_has_new_identity(self):
  c,s,stack=self.fixture();old=self.repo/'outputs/old';old.mkdir(parents=True);(old/'manifest.json').write_text('broken old log')
  with stack:
   self.assertEqual(r.main(['--repo',str(self.repo),'--config',str(c),'--prepare']),0)
   first=r.read(Path(s['generation_dir'])/'frozen_plan.json')['selected_inputs'][0]['case']
   (self.repo/'inputs/source.sdf').write_text('different content')
   c,s2=self.config(LIGAND_INDICES=[0],GENERATION_DIR='outputs/next/generated',INTERVENTION_INPUT_DIR='outputs/next/generated/failed',INTERVENTION_OUTPUT_DIR='outputs/next/intervention')
   self.assertEqual(r.main(['--repo',str(self.repo),'--config',str(c),'--prepare']),0)
   second=r.read(Path(s2['generation_dir'])/'frozen_plan.json')['selected_inputs'][0]['case'];self.assertNotEqual(first,second)
 def test_absolute_external_output_with_spaces(self):
  c,s,stack=self.fixture(n=2,m=1)
  with tempfile.TemporaryDirectory(prefix='external paths ') as tmp:
   gen=Path(tmp)/'new data/generated';dest=Path(tmp)/'new data/intervention'
   values=DEFAULTS|dict(LIGAND_INDICES=[0,1],SAMPLES_PER_INPUT=1,GENERATION_DIR=str(gen),INTERVENTION_INPUT_DIR=str(gen/'failed'),INTERVENTION_OUTPUT_DIR=str(dest))
   c.write_text('\n'.join(f'{k}={v!r}' for k,v in values.items()))
   with stack:self.assertEqual(r.main(['--repo',str(self.repo),'--config',str(c)]),0)
   self.assertEqual(r.read(dest/'summary.json')['original_samples'],2)
 def test_missing_trajectory_stops_intervention(self):
  c,s,stack=self.fixture()
  with stack:self.assertEqual(r.main(['--repo',str(self.repo),'--config',str(c),'--generate-only']),0)
  source=Path(s['intervention_input_dir']);sample=r.read(next(source.glob('*/sample.json')))
  (Path(sample['run_directory'])/'research_record/attempt_000/inputs.pt').unlink()
  with self.assertRaisesRegex(RuntimeError,'Missing trajectory'):io_paths.selected_failures(source)
 def test_tampered_sdf_and_bare_sdf_rejected(self):
  c,s,stack=self.fixture()
  with stack:self.assertEqual(r.main(['--repo',str(self.repo),'--config',str(c),'--generate-only']),0)
  source=Path(s['intervention_input_dir']);sample=next(source.glob('*/molecule.sdf'));original=sample.read_bytes();sample.write_text('edited')
  with self.assertRaisesRegex(RuntimeError,'edited'):io_paths.selected_failures(source)
  sample.write_bytes(original);(source/'bare.sdf').write_text('not a complete sample')
  with self.assertRaisesRegex(RuntimeError,'Unrecognized'):io_paths.selected_failures(source)
 def test_zero_failures_is_a_valid_empty_intervention(self):
  c,s,stack=self.fixture(n=1,m=1);d=DEFAULTS|dict(LIGAND_INDICES=[0],SAMPLES_PER_INPUT=1,SEED_START=3002);c.write_text('\n'.join(f'{k}={v!r}' for k,v in d.items()))
  with stack:self.assertEqual(r.main(['--repo',str(self.repo),'--config',str(c)]),0)
  summary=r.read(Path(s['intervention_output_dir'])/'summary.json');self.assertEqual(summary['requested_failures'],0);self.assertEqual(summary['repaired'],0)

class SelectionTests(Base):
 def choose(self,items,routes,ids):
  tree=ast.parse((PAYLOAD/'scaffold_study_v1/terminal_control.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='choose_active')
  ns={'require':io_paths.require,'successful':io_paths.successful};exec(compile(ast.Module(body=[node],type_ignores=[]),'choose_active','exec'),ns);return ns['choose_active'](items,routes,ids)
 def test_subset_preserves_controls_and_does_not_treat_unselected_failures(self):
  items=[];routes={}
  for rid,inp,status in [('f1','a','supported_geometry'),('f2','a','supported_geometry'),('p1','a','already_passed'),('p2','b','already_passed'),('u1','a','unsupported')]:
   row={k:True for k in io_paths.ENDPOINT_KEYS};row['posebusters_valid']=status=='already_passed';items.append(dict(id=rid,input_id=inp,rows={'raw_without_repair':row}));routes[rid]={'status':status}
  selected,requested=self.choose(items,routes,['f1','u1']);self.assertEqual([x['id'] for x in selected],['f1','p1']);self.assertEqual(requested,{'f1','u1'})
  for bad in [['p1'],['missing'],['f1','f1']]:
   with self.assertRaises(RuntimeError):self.choose(items,routes,bad)
  selected,_=self.choose(items,routes,None);self.assertEqual([x['id'] for x in selected],['f1','f2','p1'])

class InstallerTests(Base):
 def setup_install(self):
  update=self.repo/'update';shutil.copytree(HERE/'payload',update/'payload',ignore=shutil.ignore_patterns('__pycache__'))
  checkout=self.repo/'checkout';checkout.mkdir()
  for name in installer.TARGETS:
   path=checkout/name
   if name=='scaffold_validation100':path.mkdir();(path/'config.py').write_text('INPUTS = [("cdk2",5)]')
   else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text('old source '+name)
  payload=update/'payload';r.write(update/'package_manifest.json',dict(payload_sha256={str(p.relative_to(payload)):r.sha(p) for p in payload.rglob('*') if p.is_file()},allowed_source_sha256={n:[r.sha(checkout/n),r.sha(payload/n)] for n in installer.TARGETS if n!='scaffold_validation100'}))
  return checkout,update
 def test_install_backup_idempotence_restore(self):
  repo,update=self.setup_install()
  with mock.patch.object(installer,'HERE',update):
   backup=installer.install(repo);self.assertEqual((backup/'scaffold_validation100/config.py').read_text(),'INPUTS = [("cdk2",5)]')
   config=repo/'scaffold_validation100/config.py';config.write_text(config.read_text().replace('SAMPLES_PER_INPUT = 5','SAMPLES_PER_INPUT = 7'))
   self.assertIsNone(installer.install(repo));self.assertIn('SAMPLES_PER_INPUT = 7',config.read_text())
   installer.restore(repo,backup);self.assertEqual(config.read_text(),'INPUTS = [("cdk2",5)]')
 def test_install_failure_rolls_back_all_targets(self):
  repo,update=self.setup_install();before={p.relative_to(repo):p.read_bytes() for p in repo.rglob('*') if p.is_file()};rename=Path.rename
  def fail(source,target):
   if source.name=='terminal_control.py' and 'new' in source.parts:raise OSError('simulated write failure')
   return rename(source,target)
  with mock.patch.object(installer,'HERE',update),mock.patch.object(Path,'rename',fail),self.assertRaises(OSError):installer.install(repo)
  for name,contents in before.items():self.assertEqual((repo/name).read_bytes(),contents)
 def test_unknown_source_or_active_lock_prevents_changes(self):
  repo,update=self.setup_install();(repo/'record_scaffold_run.py').write_text('unexpected edit')
  with mock.patch.object(installer,'HERE',update),self.assertRaisesRegex(RuntimeError,'differs'):installer.install(repo)
  self.assertFalse((repo/'archive').exists())
  lock=repo/'outputs/active/RUNNING.lock';lock.parent.mkdir(parents=True);lock.write_text('running')
  with mock.patch.object(installer,'HERE',update),self.assertRaisesRegex(RuntimeError,'active'):installer.install(repo)

if __name__=='__main__':unittest.main(verbosity=2)
