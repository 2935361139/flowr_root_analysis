import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scaffold_study_v1.core import analyze, terminal_window, METRICS, SCHEMA
from scaffold_study_v1.flowr_adapter import load, observations
from scaffold_study_v1.io_utils import write_json, write_csv, sha, boolean
from scaffold_study_v1.protocol import final_constraint_plan
from scaffold_study_v1.run import main


def dataset():
    endpoints = {m: True for m in METRICS}
    return {'schema': SCHEMA, 'model': 'synthetic_test_model', 'provenance': {}, 'runs': [
        {'run_id': 'a42', 'input_id': 'a', 'input_group': 'group_a', 'seed': 42,
         'endpoints': {'raw': dict(endpoints), 'decoded': dict(endpoints)},
         'observations': [], 'runtime_settings': {'final_inpaint': False}, 'generator_identity': {}}]}


def points(values):
    return [{'index': i, 'native_time': i/10, 'stage': 'synthetic_state', 'value': v} for i,v in enumerate(values)]


class CoreTests(unittest.TestCase):
    def test_model_neutral(self):
        r = analyze(dataset())
        self.assertEqual(r['model'], 'synthetic_test_model')
        self.assertEqual(r['runs'][0]['issues'], [])
        self.assertIsNone(r['runs'][0]['full_scaffold_hopping_success'])

    def test_unknown_not_pass_or_fail(self):
        d = dataset(); d['runs'][0]['endpoints']['raw']['geometry_valid'] = None
        self.assertEqual(analyze(d)['counts']['raw']['geometry_valid'], {'pass':0,'fail':0,'unknown':1,'total':1})

    def test_early_noise_is_not_failure(self):
        d = dataset(); d['runs'][0]['observations'] = [dict(stream='sampler', entity='whole_molecule', metric='disconnected', points=points([True, False]))]
        self.assertEqual(analyze(d)['runs'][0]['issues'], [])

    def test_unknown_predecessor_does_not_confirm_onset(self):
        w = terminal_window(points([False, None, True, True]))
        self.assertEqual(w['first_persistent_observation']['index'], 2)
        self.assertFalse(w['boundary_is_confirmed_change'])
        self.assertIsNone(w['causal_onset'])

    def test_transient_anomaly_not_terminal_signature(self):
        self.assertEqual(terminal_window(points([True, False]))['status'], 'terminal_signature_absent_or_unknown')

    def test_separate_stream_counters(self):
        d = dataset(); r=d['runs'][0]; r['endpoints']['raw']['connected']=False
        r['observations'] = [dict(stream='network', entity='whole_molecule', metric='disconnected', points=points([False, True])),
                             dict(stream='sampler', entity='whole_molecule', metric='disconnected', points=points([False, False, True]))]
        e=analyze(d)['runs'][0]['issues'][0]['temporal_evidence']
        self.assertEqual([x['first_persistent_observation']['index'] for x in e], [1,2])
        self.assertEqual([x['stream'] for x in e], ['network','sampler'])

    def test_missing_frame_rejected(self):
        d=dataset(); p=points([False,True]);p[1]['index']=2
        d['runs'][0]['observations']=[dict(stream='s',entity='e',metric='disconnected',points=p)]
        with self.assertRaises(ValueError):analyze(d)

    def test_duplicate_run_rejected(self):
        d=dataset();d['runs'].append(copy.deepcopy(d['runs'][0]))
        with self.assertRaises(ValueError):analyze(d)

    def test_strings_not_boolean(self):
        d=dataset();d['runs'][0]['endpoints']['raw']['chemical_valid']='False'
        with self.assertRaises(ValueError):analyze(d)

    def test_decode_change_does_not_become_network_cause(self):
        d=dataset();d['runs'][0]['endpoints']['decoded']['fixed_chemistry_preserved']=False
        r=analyze(d)['runs'][0]
        self.assertEqual(len(r['decode_stage_changes']),1)
        self.assertEqual(r['issues'][0]['temporal_evidence'],[])
        self.assertIsNone(r['issues'][0]['internal_module'])

    def test_same_check_reference_not_whole_failure_inherited(self):
        d=dataset();r=d['runs'][0];r['endpoints']['raw']['geometry_valid']=False
        r['endpoints']['raw']['geometry_failed_checks']=['protein_clash','energy']
        r['geometry']={'raw':{'clash_pairs':[{'check':'protein_clash','reference_pair_comparison':'same_fixed_pair'}],
                              'reference_context':{'same_failed_check_in_reference':['protein_clash']}}}
        issue=analyze(d)['runs'][0]['issues'][0]
        self.assertEqual(issue['unlocalized_checks'],['energy'])
        self.assertEqual(issue['temporal_evidence'],[])
        self.assertEqual(issue['causal_status'],'not_tested')

    def test_carbon_signature_not_applied_to_valid_molecule(self):
        d=dataset();d['runs'][0]['observations']=[dict(stream='network',entity='atom:0',metric='neutral_carbon_overvalence',points=points([True]))]
        self.assertEqual(analyze(d)['runs'][0]['issues'],[])

    def test_repeated_seeds_not_independent_inputs(self):
        d=dataset();r=copy.deepcopy(d['runs'][0]);r['run_id']='a43';r['seed']=43;d['runs'].append(r)
        self.assertEqual(analyze(d)['n_input_groups'],1)

    def test_other_atoms_carbon_signature_not_linked_to_failure(self):
        d=dataset();r=d['runs'][0];r['endpoints']['raw']['chemical_valid']=False
        r['endpoints']['raw']['neutral_carbon_overvalent_atoms']=[1]
        r['observations']=[dict(stream='network',entity='fixed_atom:7',metric='neutral_carbon_overvalence',points=points([True]))]
        self.assertEqual(analyze(d)['runs'][0]['issues'][0]['temporal_evidence'],[])

    def test_no_empty_success(self):
        d=dataset();d['runs']=[]
        with self.assertRaises(ValueError):analyze(d)


class PlanTests(unittest.TestCase):
    def plan_data(self):
        d=dataset();d['model']='FLOWR.ROOT'
        r=copy.deepcopy(d['runs'][0]);r['seed']=43;r['run_id']='a43';r['endpoints']['raw']['fixed_chemistry_preserved']=False
        d['runs'].append(r)
        return d

    def test_keeps_both_normal_and_failed_seeds(self):
        p=final_constraint_plan(self.plan_data(),['a'],4)
        self.assertEqual(len(p['jobs']),4)
        self.assertEqual({j['seed'] for j in p['jobs']},{42,43})
        self.assertTrue(all(j['status']=='not_run' for j in p['jobs']))
        self.assertFalse(p['runtime_executor_included'])

    def test_budget_rejected_not_silently_truncated(self):
        with self.assertRaises(ValueError):final_constraint_plan(self.plan_data(),['a'],3)

    def test_unknown_input_rejected(self):
        with self.assertRaises(ValueError):final_constraint_plan(self.plan_data(),['missing'],40)

    def test_missing_setting_not_defaulted(self):
        d=self.plan_data();d['runs'][0]['runtime_settings']={}
        with self.assertRaises(ValueError):final_constraint_plan(d,['a'],4)

    def test_wrong_model_rejected(self):
        with self.assertRaises(ValueError):final_constraint_plan(dataset(),['a'],2)


class AdapterTests(unittest.TestCase):
    def fixture(self, root):
        diag=root/'diag';folder=diag/'run_001';folder.mkdir(parents=True)
        m={'args':{'seed':42,'pdb_file':'p.pdb','final_inpaint':False},'input_sha256':{'p.pdb':'protein'},'checkpoint_sha256':'ckpt','source_sha256':{'x':'s'}}
        rows=[dict(run_directory='/old/a',stage=stage,rdkit_valid=True,fc_valid=True,raw_graph_connected=True,
                   fixed_chemistry_preserved=True,root_fixed_substructure_match=True,posebusters_valid=True,murcko_scaffold_changed=True)
              for stage in ('raw_without_repair','run_decode')]
        q={'completed':True,'errors':[],'requested_run_count':1,'rows':rows,'runs':[
            dict(run_directory='/old/a',case='case',ligand_idx=0,seed=42,checkpoint_sha256='ckpt',recorded_source_sha256={'x':'s'},
                 input_sha256={'p.pdb':'protein'},reference_sdf_sha256='reference')]}
        write_json(root/'q.json',q)
        write_json(diag/'summary.json',{'completed':True,'quality_report_sha256':sha(root/'q.json'),
                   'records':[dict(analysis_completed=True,report_directory='run_001',seed=42)]})
        report=dict(run='/old/a',seed=42,endpoint_evaluation=rows,manifest=m,
                    task_definition={'mapping_verified':True,'fixed_sampler_indices':[7]},
                    streams={s:{'indices':[0,1],'times':[None,None],'stages':['a','b']} for s in ('network_prediction','sampler_state')})
        write_json(folder/'connection_report.json',report)
        common=[dict(stream=s,index=i,continuous_time='',stage='ab'[i]) for s in report['streams'] for i in range(2)]
        write_csv(folder/'connection_timeline.csv',[dict(**r,fixed_sampler_index=7,fixed_chemistry_changed=False,neutral_carbon_overvalent=False,fixed_displacement_A=0) for r in common])
        write_csv(folder/'graph_timeline.csv',[dict(**r,connected=True) for r in common])
        return root/'q.json',diag,folder

    def test_relocated_report_paths_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            q,d,_=self.fixture(Path(tmp));r=analyze(load(q,d))
            self.assertEqual(r['n_runs'],1)

    def test_mixed_batch_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            q,d,_=self.fixture(Path(tmp));q.write_text(q.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'hashes disagree'):load(q,d)

    def test_missing_timeline_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            q,d,f=self.fixture(Path(tmp));(f/'graph_timeline.csv').unlink()
            with self.assertRaises(OSError):load(q,d)

    def test_site_id_permutation_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            q,d,_=self.fixture(Path(tmp));e=load(q,d)
            self.assertTrue(any(o['entity']=='fixed_atom:7' for o in e['runs'][0]['observations']))

    def test_unknown_boolean_never_becomes_false(self):
        self.assertIsNone(boolean(''))
        with self.assertRaises(ValueError):boolean('probably')

    def test_repeated_graph_frame_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            q,d,f=self.fixture(Path(tmp));p=f/'graph_timeline.csv'
            with p.open('a') as h:h.write('network_prediction,0,,a,True\n')
            with self.assertRaises(ValueError):load(q,d)

    def test_unverified_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            q,d,f=self.fixture(Path(tmp));p=f/'connection_report.json'
            r=json.loads(p.read_text());r['task_definition']['mapping_verified']=False;write_json(p,r)
            with self.assertRaisesRegex(ValueError,'Unverified atom'):load(q,d)

    def test_different_endpoint_label_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            q,d,f=self.fixture(Path(tmp));p=f/'connection_report.json'
            r=json.loads(p.read_text());r['endpoint_evaluation'][0]['rdkit_valid']=False;write_json(p,r)
            with self.assertRaisesRegex(ValueError,'Endpoint report changed'):load(q,d)


class CliTests(unittest.TestCase):
    def test_no_overwrite_and_portable_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'data.json',dataset())
            args=['--normalized-input',str(root/'data.json'),'--output-dir',str(root/'out')]
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main(args),0)
            before=(root/'out/diagnosis.json').read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):main(args)
            self.assertEqual(before,(root/'out/diagnosis.json').read_bytes())


if __name__=='__main__':unittest.main()
