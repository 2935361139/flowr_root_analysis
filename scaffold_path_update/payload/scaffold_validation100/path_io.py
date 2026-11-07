"""Lossless file classification and explicit failure selection. No chemistry rules changed."""
import hashlib
import json
import shutil
from pathlib import Path

ENDPOINT_KEYS=('rdkit_valid','fc_valid','root_fixed_substructure_match','fixed_chemistry_preserved','posebusters_valid')

def require(ok,message):
    if not ok:raise RuntimeError(message)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def successful(row):return all(row.get(k) is True for k in ENDPOINT_KEYS)

def contained(child,parent):return child==parent or parent in child.parents

def check_layout(settings,generate=True,intervene=True):
    gen=Path(settings['generation_dir']);dest=Path(settings['intervention_output_dir']);source=Path(settings['intervention_input_dir'])
    require(gen!=gen.parent and dest!=dest.parent,'Do not use a filesystem root for results')
    require(not contained(gen,dest) and not contained(dest,gen),'Generation and intervention output directories must be separate, not nested')
    require(not contained(source,dest) and not contained(dest,source),'Intervention output must be separate from its input')
    if generate and intervene:require(source==gen/'failed','For a full run, INTERVENTION_INPUT_DIR must equal GENERATION_DIR/failed')
    for key in ('sdf_path','pdb_path'):
        original=Path(settings[key])
        require(not contained(original,gen) and not contained(original,dest),'Input files must be outside new result directories')
    if generate:require(not gen.exists(),f'Generation directory already exists: {gen}. Use a new name or --run to continue a prepared plan.')
    if intervene:require(not dest.exists(),f'Intervention output already exists: {dest}. Choose a new output directory.')


def copy_checked(source,destination):
    source,destination=Path(source),Path(destination)
    require(source.is_file(),f'Missing source file: {source}')
    if destination.exists():require(sha(source)==sha(destination),f'Existing export changed: {destination}')
    else:
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    return sha(destination)


def export_group(out,plan,group):
    """Copy the exact evaluated RAW SDF, including an empty file for failed decoding."""
    out=Path(out);folder=Path(group['folder']);qpath=folder/'quality/quality_report.json';q=read(qpath)
    require(q.get('completed') and not q.get('errors'),'Incomplete quality report; cannot classify')
    require(len(q['runs'])==len(group['jobs']),'Quality run count differs')
    rows={x['run_directory']:x for x in q['rows'] if x['stage']=='raw_without_repair'}
    require(len(rows)==len(group['jobs']),'Missing/duplicate raw quality rows')
    exported=[]
    for job in group['jobs']:
        row=rows[job['run_directory']];require(not row.get('errors'),'Evaluation error must not be labeled as a molecular failure')
        source=Path(job['run_directory'])/'research_record/attempt_000/decoded_without_repair.sdf'
        passed=successful(row);category='success' if passed else 'failed';target=out/category/job['id']
        digest=copy_checked(source,target/'molecule.sdf')
        item=dict(schema='flowr_path_sample_v1',run_id=job['id'],group=group['name'],generation_root=str(out),
                  generation_plan_sha256=sha(out/'frozen_plan.json'),run_directory=job['run_directory'],
                  source_sdf=str(source),sdf_sha256=digest,quality_report=str(qpath),quality_sha256=sha(qpath),
                  baseline_success=passed,endpoint={k:row.get(k) for k in ENDPOINT_KEYS},
                  failed_checks=row.get('posebusters_failed_checks',[]),sdf_has_content=bool(source.read_bytes().strip()))
        if (target/'sample.json').exists():require(read(target/'sample.json')==item,'Existing sample metadata changed')
        else:write(target/'sample.json',item)
        exported.append(item)
    return exported


def export_generation(out,plan):
    out=Path(out);samples=[]
    for category in ['success','failed']:(out/category).mkdir(exist_ok=True)
    completed=0
    for group in plan['groups']:
        if not (Path(group['folder'])/'localization_v2/study/diagnosis.json').is_file():continue
        samples.extend(export_group(out,plan,group));completed+=1
    require(len({s['run_id'] for s in samples})==len(samples),'Duplicate classified samples')
    for category in ['success','failed']:
        items=[s for s in samples if s['baseline_success']==(category=='success')]
        write(out/category/'index.json',dict(schema='flowr_path_index_v1',generation_root=str(out),samples=items))
    write(out/'sample_index.json',samples)
    summary=dict(completed=completed==len(plan['groups']),planned_samples=plan['planned_generations'],
                 classified_samples=len(samples),original_passes=sum(s['baseline_success'] for s in samples),
                 original_failures=sum(not s['baseline_success'] for s in samples),completed_groups=completed,
                 endpoint=list(ENDPOINT_KEYS),stage='raw_without_repair',note='Before any intervention; originals remain unchanged')
    write(out/'generation_summary.json',summary)
    return summary


def selected_failures(source):
    """A directory of sample folders is a selection, never a source of trusted labels."""
    source=Path(source);require(source.is_dir(),f'Missing intervention input: {source}')
    markers=sorted(source.glob('*/sample.json'));items=[]
    allowed={m.parent.name for m in markers}|{'index.json'}
    unexpected=[p.name for p in source.iterdir() if p.name not in allowed and not p.name.endswith(':Zone.Identifier')]
    require(not unexpected,f'Unrecognized intervention files: {unexpected}. Copy complete failed sample folders, not bare SDFs.')
    for marker in markers:
        item=read(marker)
        require(item.get('schema')=='flowr_path_sample_v1' and item.get('baseline_success') is False,'Only exported original failures can be selected')
        require(sha(marker.parent/'molecule.sdf')==item['sdf_sha256'],'Selected molecule was edited; no silent substitution')
        items.append(item)
    roots={s['generation_root'] for s in items}
    if not items:
        index=read(source/'index.json') if (source/'index.json').is_file() else {}
        require(index.get('schema')=='flowr_path_index_v1','Empty directory has no generation index')
        roots.add(index['generation_root'])
    require(len(roots)==1,'Select failures from one generation experiment at a time')
    root=Path(next(iter(roots))).resolve();plan=read(root/'frozen_plan.json')
    require(sha(root/'frozen_plan.json')==(root/'frozen_plan.sha256').read_text().strip(),'Generation plan changed')
    require(plan['output']==str(root),'Generation directory moved; preserve the recorded paths')
    require(read(root/'generation_summary.json')['completed'],'Generation/diagnosis is not complete')
    originals=read(root/'sample_index.json');indexed={s['run_id']:s for s in originals}
    require(len(indexed)==len(originals),'Duplicate generation identities')
    require(len({s['run_id'] for s in items})==len(items),'Duplicate selected samples')
    for s in items:
        require(indexed.get(s['run_id'])==s,'Selected metadata does not match the recorded generation')
        require(s['generation_plan_sha256']==sha(root/'frozen_plan.json'),'Sample belongs to another plan')
        require(sha(s['source_sdf'])==s['sdf_sha256'] and sha(s['quality_report'])==s['quality_sha256'],'Original evidence changed')
        require((Path(s['run_directory'])/'research_record/attempt_000/inputs.pt').is_file(),'Missing trajectory evidence; keep the generated directory')
    return root,plan,items


def export_interventions(out,selected,groups):
    out=Path(out);indexed={s['run_id']:s for s in selected};results=[];group_stats=[]
    for category in ['success','failed']:(out/category).mkdir(exist_ok=True)
    for group,rescue in groups:
        rescue=Path(rescue);summary=read(rescue/'summary.json');status=read(rescue/'status.json')
        require(status['completed'] and not status.get('errors'),'Incomplete intervention stage')
        choices=read(rescue/'selected_results.json')
        require(len(choices)==len(group['jobs']) and {s['run_id'] for s in choices}=={j['id'] for j in group['jobs']},'Incomplete/duplicate intervention sample identities')
        group_stats.append({'group':group['name'],**summary})
        for s in choices:
            rid=s['run_id']
            if rid not in indexed:continue
            require(s['baseline_success'] is False,'Selected failure turned into an original pass')
            arm=s['selected_arm'];good=arm is not None
            require(arm in (None,'A','B','AB','C'),'Unexpected selected intervention')
            if good:
                record=read(rescue/rid/(arm+'.json'));require(successful(record['evaluation']),'Selected intervention does not pass the full endpoint')
            target=out/('success' if good else 'failed')/rid
            source=rescue/rid/(arm+'.sdf') if good else Path(indexed[rid]['source_sdf'])
            digest=copy_checked(source,target/'molecule.sdf')
            item=dict(s,source_generation_root=indexed[rid]['generation_root'],source_run_directory=indexed[rid]['run_directory'],
                      exported_sdf_sha256=digest,exported_molecule='successful_intervention' if good else 'unchanged_original_failure',
                      branch_results=str(rescue/rid))
            write(target/'result.json',item);results.append(item)
    require(len(results)==len(selected) and {r['run_id'] for r in results}==set(indexed),'Not all requested failures have results')
    summary=dict(completed=True,requested_failures=len(selected),rescued_without_C=sum(r['selected_arm'] in ('A','B','AB') for r in results),
                 rescued_with_fixed_coordinate_relaxation=sum(r['selected_arm']=='C' for r in results),
                 unresolved=sum(r['result']=='unresolved' for r in results),unsupported=sum(r['result']=='unsupported' for r in results),
                 branch_errors=sum(g['branch_errors'] for g in group_stats),
                 samples_with_branch_errors=sum(bool(r['candidate_errors']) for r in results),
                 scope='Only explicitly requested failures. Same-input passing controls are evaluated but not counted as repairs.')
    summary['repaired']=summary['rescued_without_C']+summary['rescued_with_fixed_coordinate_relaxation']
    summary['not_repaired']=len(selected)-summary['repaired']
    require(summary['unresolved']+summary['unsupported']==summary['not_repaired'],'Unaccounted requested failures')
    write(out/'summary.json',summary);write(out/'samples.json',results);write(out/'group_results.json',group_stats)
    return summary
