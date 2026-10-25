"""Bounded, outcome-explicit protocol; no molecule/slot/seed is hard-coded."""
from pathlib import Path
import csv
import numpy as np
import native_adapter as native

FAMILIES={'geometry':[2], 'edge_context':[3], 'geometry_and_edge':[2,3],
          'fixed_node':[0], 'generated_node':[1]}
CONTROL_FAMILIES=('geometry','edge_context','geometry_and_edge')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as file:
        return list(csv.DictReader(file))


def select_protocol(trace, audit, contrast=None):
    status=native.read_json(trace/'status.json')
    native.require(status.get('completed') and not status.get('errors') and
                   status.get('coalition_backend')=='native_torch_full_shape', 'Need completed V2 native-head results')
    rows=read_csv(trace/'contrast_summary.csv')
    choices=[r for r in rows if r['contrast']==contrast] if contrast else [r for r in rows if r['terminal_segment_onset']=='True']
    native.require(len(choices)==1, 'Need exactly one selected onset; use --contrast to select explicitly if several exist')
    focus=choices[0]
    native.require(focus['kind']=='onset' and int(focus['after'])==int(focus['before'])+1, 'Need an adjacent onset window')
    checks=native.read_json(trace/'contrasts'/f"{focus['contrast']}_checks.json")
    native.require(all(checks[k]['passed'] for k in ['previous_pair_logits','after_full_logits','after_pair_logits'])
                   and checks['torch_rng_unchanged'], 'Selected local contrast failed its checks')
    summary=native.read_json(audit/'summary.json')
    native.require(summary['completed'] and not summary['errors'], 'Existing boundary audit incomplete')
    records={}
    for entry in summary['reports']:
        rid=entry['id'];native.require(Path(rid).name==rid,'Invalid run name')
        folder=audit/'runs'/rid;done=native.read_json(folder/'complete.json')
        native.require(native.digest(folder/'report.json')==done['output_sha256']['report.json'], 'Audit report changed')
        report=native.read_json(folder/'report.json')
        if report['case']==focus['case'] and report['ligand_idx']==int(focus['ligand_idx']):records[rid]=report
    native.require(focus['run'] in records and records, 'Missing source cohort')
    seeds=sorted(r['seed'] for r in records.values())
    native.require(len(seeds)==len(set(seeds)), 'Duplicate seeds in selected input cohort')
    protocol=dict(case=focus['case'],ligand_idx=int(focus['ligand_idx']),focus_contrast=focus['contrast'],
                  focal_run=focus['run'],focal_seed=int(focus['seed']),
                  reference_atom=int(focus['fixed_reference_atom']),call=int(focus['after']),
                  seeds=seeds,random_controls_per_family=3,control_seed=20261024,
                  intervention='One native BondRefine.message_mlp input write, followed by free native rollout',
                  primary_endpoint='Raw valid+connected+ROOT fragment match AND recorded fixed chemistry; also retain every separate metric, 3D quality and scaffold novelty',
                  selection_scope='Discovery: timing/site/families informed by the existing local results; not independent validation',
                  stopping_rule='Execute the fixed batch once. Do not auto-scan times, doses, sites or regenerate failures.')
    trials=[dict(id=f'seed_{s}_sham',seed=s,call=protocol['call'],kind='sham',family='none',groups=[]) for s in seeds]
    for s in seeds:
        for family,groups in FAMILIES.items():
            trials.append(dict(id=f'seed_{s}_{family}',seed=s,call=protocol['call'],kind='targeted',family=family,groups=groups))
        for family in CONTROL_FAMILIES:
            for rep in range(3):
                trials.append(dict(id=f'seed_{s}_{family}_random_{rep}',seed=s,call=protocol['call'],
                                   kind='random_direction',family=family,groups=FAMILIES[family],control_index=rep))
    return protocol,trials,records


def prepare_metadata(batch,trace,audit,protocol,records,torch,rdBase):
    frozen=native.read_json(batch/'frozen_plan.json');identity=native.read_json(trace/'identity.json')
    audit_id=native.read_json(audit/'identity.json')
    native.require(native.digest(batch/'frozen_plan.json')==identity['audit_plan_sha256']==audit_id['plan_sha256'],'Generation plan differs')
    native.require(native.digest(batch/'quality/quality_report.json')==identity['quality_sha256']==audit_id['quality_sha256'],'Terminal evaluation differs')
    native.require(str(torch.__version__)==frozen['torch_version'] and rdBase.rdkitVersion==frozen['rdkit_version'], 'Use original Torch/RDKit environment')
    native.require(frozen['config']['source_sha256']==identity['native_sources'],'Recorded sources differ')
    for name,sha in identity['native_sources'].items():native.require(native.digest(Path.cwd()/name)==sha,f'Native source changed: {name}')
    metadata={};common=None
    for rid,report in sorted(records.items()):
        run=batch/'runs'/f"{report['case']}_lig{report['ligand_idx']}"/f"seed_{report['seed']}"
        folder=run/'research_record/attempt_000'
        manifest=native.read_json(folder.parent/'manifest.json');task=native.read_json(folder/'task_definition.json')
        attempt=native.read_json(folder/'attempt.json');cfg=manifest['args']
        native.require(attempt['generation_completed'] and cfg['scaffold_hopping'] and cfg['seed']==report['seed'], 'Wrong/incomplete native run')
        native.require(task['mapping_verified'],'Unverified atom mapping')
        native.require(cfg['guidance_config'] is None and not cfg.get('optimize_gen_ligs') and not cfg.get('optimize_gen_ligs_hs'),'Guidance/optimization outside protocol')
        native.require(manifest['checkpoint_sha256']==identity['checkpoint']==frozen['checkpoint_sha256'],'Checkpoint identity differs')
        native.require(manifest['torch_version']==str(torch.__version__) and manifest['rdkit_version']==rdBase.rdkitVersion,'Mixed runtime versions')
        for name,sha in manifest['input_sha256'].items():native.require(native.digest(Path(name))==sha,f'Changed input: {name}')
        for entry in report['trajectory_identity']:
            native.require(native.digest(run/entry['name'])==entry['sha256'],f"Changed trajectory: {rid}/{entry['name']}")
        for name,sha in identity['inputs'][rid].items():native.require(native.digest(folder.parent/name)==sha,f'Changed V2 input: {rid}/{name}')
        ref_to_slot=dict(zip(task['fixed_reference_indices'],task['fixed_sampler_indices']))
        native.require(protocol['reference_atom'] in ref_to_slot, 'Reference attachment absent from fixed mask')
        site=int(ref_to_slot[protocol['reference_atom']])
        native.require(protocol['reference_atom'] in {x['fixed_reference_index'] for x in task['reference_attachment_sites']},'Selected atom is not an attachment')
        paths=sorted((folder/'forward_calls').glob('call_*.pt'))
        native.require(len(paths)==attempt['forward_calls'] and protocol['call']<len(paths), 'Missing original forward calls')
        key=dict(args={k:v for k,v in cfg.items() if k not in {'seed','save_dir'}},source=manifest['source_sha256'],inputs=manifest['input_sha256'])
        if common is None:common=key
        native.require(key==common,'Other generation conditions differ across seeds')
        metadata[report['seed']]=dict(rid=rid,folder=folder,manifest=manifest,task=task,site=site,report=report,
                paths=paths,identity_files={**{e['name']:e['sha256'] for e in report['trajectory_identity']},
                'research_record/attempt_000/pocket_encoding.pt':native.digest(folder/'pocket_encoding.pt')})
    first=next(iter(metadata.values()))
    native.require(native.digest(Path(first['manifest']['args']['ckpt_path']))==identity['checkpoint'],'Checkpoint file changed')
    return metadata,identity


def load_data(meta,protocol,torch,quality,model):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from flowr.data.interpolate import extract_scaffolds
    load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
    folder=meta['folder'];inputs=load(folder/'inputs.pt');calls=[load(p) for p in meta['paths']]
    native.require([c['call_index'] for c in calls]==list(range(len(calls))), 'Nonsequential calls')
    cfg=meta['manifest']['args'];task=meta['task'];k=protocol['call']
    native.require(inputs['prior']['mask'].shape[0]==1 and inputs['self_condition'] and int(inputs['sc_recycle_steps'])==1,'Expected recorded batch-one, one-forward-per-step mode')
    native.require(bool(model.self_condition)==inputs['self_condition'] and int(model.sc_recycle_steps)==inputs['sc_recycle_steps'],'Self-conditioning mode differs')
    native.require(float(model.coord_scale)==inputs['coord_scale'],'Coordinate scale differs')
    native.require(inputs['solver']==cfg['solver'] and inputs['strategy']==cfg['ode_sampling_strategy'] and inputs['final_inpaint']==cfg['final_inpaint'],'Sampler settings differ')
    fixed=(inputs['prior']['fragment_mask'][0].bool() & inputs['prior']['mask'][0].bool()).nonzero(as_tuple=True)[0].tolist()
    native.require(fixed==task['fixed_sampler_indices'],'Fixed mask differs')
    present=[]
    for call in calls[k-1:k+1]:
        pred=call['network_output_before_postprocessing']
        tokens=model.builder.vocab.tokens_from_indices(pred['atomics'][0].argmax(-1).tolist())
        present.append([bool(v) and t not in {'<NOATOM>','<PAD>'} for v,t in zip(pred['mask'][0].tolist(),tokens)])
    active=np.asarray(present).all(axis=0);site=meta['site']
    native.require(active[site], 'Fixed site not present in both source states')
    pairs=np.array([[site,j] for j in range(len(active)) if active[j] and j not in fixed],dtype=np.int64)
    native.require(len(pairs)>0,'No shared active boundary pairs')
    reference=quality.read_one_sdf(folder/'reference_preprocessed.sdf',Chem);native.require(reference is not None,'Missing native reference')
    Chem.SanitizeMol(reference)
    native.require(extract_scaffolds([reference],invert_mask=True)[0].nonzero(as_tuple=True)[0].tolist()==task['fixed_reference_indices'],'Native fixed-fragment extraction differs')
    return {**meta,'inputs':inputs,'calls':calls,'pairs':pairs,'encoding':load(folder/'pocket_encoding.pt'),
            'final':load(folder/'raw_final_prediction.pt'),'reference':reference,
            'reference_scaffold':Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(reference),isomericSmiles=False)}
