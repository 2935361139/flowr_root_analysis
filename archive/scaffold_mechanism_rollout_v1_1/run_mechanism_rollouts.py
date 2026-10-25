#!/usr/bin/env python3
"""Test whether observed local head effects survive a complete native rollout."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import sys
import time
import traceback
import zipfile
from pathlib import Path
import numpy as np
import native_adapter as native
import native_chemistry as nc
import replay_support as helpers
import quality
from protocol import select_protocol,prepare_metadata,load_data,read_csv
from head_patch import HeadPatch,selected_values

HERE=Path(__file__).resolve().parent
VERSION='2026-10-25.local-head-rollout.1.1'


def raw_fixed_checks(final,data,model):
    task,ref=data['task'],data['reference'];builder=model.builder
    atoms=[builder.vocab.idx_token_map[i] for i in range(builder.vocab.size)]
    charges=[builder.vocab_charges.idx_token_map[i] for i in range(builder.vocab_charges.size)]
    g=nc.graph(final,atoms,charges)
    mapping=dict(zip(task['fixed_sampler_indices'],task['fixed_reference_indices']))
    missing=[s for s in mapping if not g['keep'][s]]
    elements=[s for s,r in mapping.items() if g['keep'][s] and g['elements'][s]!=ref.GetAtomWithIdx(r).GetSymbol()]
    q=[s for s,r in mapping.items() if g['keep'][s] and g['charges'][s]!=ref.GetAtomWithIdx(r).GetFormalCharge()]
    changed=[]
    for s,r in mapping.items():
        for j,t in mapping.items():
            if j>=s:continue
            b=ref.GetBondBetweenAtoms(r,t);cls=0 if b is None else {1.:1,2.:2,3.:3,1.5:4}[b.GetBondTypeAsDouble()]
            if g['classes'][s,j]!=cls:changed.append([s,j])
    return dict(fixed_chemistry_preserved=not any([missing,elements,q,changed]),fixed_missing_sites=missing,
                fixed_element_changed_sites=elements,fixed_charge_changed_sites=q,fixed_internal_bond_changed_pairs=changed,
                fixed_checks_status='raw_argmax_tensor_mapping')


def local_expectation(trace,trace_plan,rid,call,site,kind):
    if kind=='random_direction':return None
    jobs=[j for j in trace_plan['jobs'] if j['run']==rid and j['after']==call and j['site']==site]
    if not jobs:return None
    native.require(len(jobs)==1,'Ambiguous previous local contrast')
    with np.load(trace/'contrasts'/f"{jobs[0]['id']}.npz",allow_pickle=False) as a:
        return dict(pairs=a['pairs'].copy(),logits=a['coalition_logits'].copy())


def result_row(result,sham):
    t=result['trial'];ev=result['evaluation'];raw,decoded=ev['stages'];base=sham['evaluation']['raw_valid_connected_root_fragment']
    loc=result['local_metrics'];baseline_local=sham['local_metrics']
    controlled=result['replay']['patch']
    return dict(trial=t['id'],seed=t['seed'],kind=t['kind'],family=t['family'],control_index=t.get('control_index'),
                call=t['call'],fixed_slot=t['site'],reference_atom=t['reference_atom'],
                sham_site_valence_at_call=baseline_local['at_intervention']['site_valence_problem'],
                site_valence_at_call=loc['at_intervention']['site_valence_problem'],
                local_rdkit_valid=loc['at_intervention']['rdkit_valid'],local_connected=loc['at_intervention']['connected'],
                first_later_valence_problem_call=loc['first_later_valence_problem_call'],
                later_unassessed_calls=loc['later_unassessed_calls'],
                baseline_raw_chemical_endpoint=base,raw_chemical_endpoint=ev['raw_valid_connected_root_fragment'],
                baseline_raw_connection_and_fixed_identity=sham['evaluation']['raw_connection_and_fixed_identity'],
                raw_connection_and_fixed_identity=ev['raw_connection_and_fixed_identity'],
                raw_valid_connected=raw['fc_valid'],raw_fixed_chemistry_preserved=raw['fixed_chemistry_preserved'],
                raw_root_fragment_match=raw['root_fixed_substructure_match'],raw_posebusters_valid=raw['posebusters_valid'],
                raw_scaffold_changed=raw.get('murcko_scaffold_changed'),
                run_decode_fixed_preserved=decoded['fixed_chemistry_preserved'],
                final_site_valence_problem=ev['final_graph']['site_valence_problem'],
                intervention_applied=controlled['applied'],intervention_norm=controlled['total_norm'],
                all_changed_groups_informative=all(g['magnitude_informative'] for g in controlled['per_group']) if controlled['per_group'] else None,
                all_control_magnitudes_matched=all(g['magnitude_matched'] for g in controlled['per_group']) if controlled['per_group'] else None,
                evaluation_errors=[s['errors'] for s in ev['stages']],elapsed_seconds=result['elapsed_seconds'])


def write_summary(out,state,results,trials):
    rows=[result_row(results[t['id']],results[f"seed_{t['seed']}_sham"]) for t in trials
          if t['id'] in results and f"seed_{t['seed']}_sham" in results]
    native.write_csv(out/'rollout_results.csv',rows)
    state['completed_trials']=len(results)
    native.write_json(out/'status.json',state)
    note=('本实验只在指定的一次网络调用，改变成键头入口的指定内部特征；之后由原生采样器自由生成。'
          '原样回放先复现全部网络调用和最终张量，再允许执行定向修改与同幅度随机对照。'
          '原始化学连接终点、固定原子身份、PoseBusters和骨架变化分别报告。'
          '这些种子属于同一分子输入；方案由当前结果选出，不能据此宣称跨输入泛化或唯一根因。'
          '一次局部修改无最终效果，也不能排除该路径在其他时刻或持续更新中的作用；本批次不会自动追加时刻、强度或样本。')
    fields=['seed','kind','family','control_index','site_valence_at_call','first_later_valence_problem_call',
            'raw_connection_and_fixed_identity','raw_fixed_chemistry_preserved','raw_posebusters_valid','raw_scaffold_changed']
    table='<tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr>'
    table+=''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k)))+'</td>' for k in fields)+'</tr>' for r in rows)
    (out/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>内部路径完整续跑</title>'
       '<style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:0 18px;line-height:1.7}pre{white-space:pre-wrap}'
       'table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccd;padding:6px}th{background:#edf4f6}</style>'
       '<h1>固定片段—新骨架连接：内部路径完整续跑</h1><p>'+html.escape(note)+'</p><pre>'+
       html.escape(json.dumps(state,ensure_ascii=False,indent=2))+'</pre><div style="overflow:auto"><table>'+table+'</table></div></html>',encoding='utf-8')


def bundle(out):
    target=out/'rollout_review.zip';temp=out/'rollout_review.tmp'
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            rel=p.relative_to(out)
            if p.is_file() and p.suffix not in {'.pt','.tmp','.zip','.pyc'} and not {'model_context','__pycache__'} & set(rel.parts):z.write(p,rel)
    temp.replace(target)
    return target


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    ap.add_argument('--trace',type=Path,default=Path('outputs/sh_mechanism_trace_v2'))
    ap.add_argument('--batch',type=Path,default=Path('outputs/sh_validation_baselines_v1'))
    ap.add_argument('--output',type=Path,default=Path('outputs/sh_mechanism_rollout_v1_1'))
    ap.add_argument('--contrast');ap.add_argument('--resume',action='store_true');ap.add_argument('--list-plan',action='store_true')
    args=ap.parse_args(argv);trace,batch,out=args.trace.resolve(),args.batch.resolve(),args.output.resolve()
    audit=batch/'boundary_audit_v1'
    protocol,trials,records=select_protocol(trace,audit,args.contrast)
    print(json.dumps(dict(protocol=protocol,planned_rollouts=len(trials),conditions_per_seed=15),ensure_ascii=False,indent=2),flush=True)
    if args.list_plan:return 0
    native.require((Path.cwd()/'flowr/scriptutil.py').is_file(),'Run from original FLOWR.ROOT repository')
    native.require(not out.exists() or args.resume,'Output exists; preserve it and use --resume for identical inputs/code')
    out.mkdir(parents=True,exist_ok=True);results={}
    state=dict(version=VERSION,completed=False,completed_trials=0,planned_trials=len(trials),errors=[],stage='preflight',protocol=protocol)
    try:
        sys.path.insert(0,str(Path.cwd()))
        import torch
        from rdkit import rdBase
        from flowr.scriptutil import load_model
        from flowr.util.device import resolve_device
        metadata,trace_identity=prepare_metadata(batch,trace,audit,protocol,records,torch,rdBase)
        captures=[json.loads(line) for line in (trace/'capture_checks.jsonl').read_text().splitlines()]
        native.require(len(captures)==native.read_json(trace/'status.json')['captured'] and all(r['accepted'] and r['original_match']['passed'] and r['observation_match']['passed'] and r['native_head']['passed'] for r in captures),'Invalid previous capture receipts')
        precision={r['native_head']['matmul_precision'] for r in captures};native.require(len(precision)==1,'Mixed precision settings')
        precision=next(iter(precision))
        for t in trials:t.update(site=metadata[t['seed']]['site'],reference_atom=protocol['reference_atom'],control_seed=protocol['control_seed'])
        bindings={str(p.relative_to(trace)):native.digest(p) for p in trace.rglob('*') if p.is_file() and
                  ('captures' not in p.relative_to(trace).parts and 'model_context' not in p.relative_to(trace).parts and p.suffix in {'.json','.jsonl','.csv','.npz'})}
        identity=dict(version=VERSION,protocol=protocol,planned_trials=trials,trace_file_hashes=bindings,
                      checkpoint=trace_identity['checkpoint'],native_sources=trace_identity['native_sources'],precision=precision,
                      input_hashes={str(s):d['identity_files'] for s,d in metadata.items()},
                      code_sha256={p.name:native.digest(p) for p in HERE.glob('*.py')})
        if args.resume:native.require((out/'identity.json').is_file() and native.read_json(out/'identity.json')==identity,'Resume identity changed; preserve directory and use a new --output')
        else:native.write_json(out/'identity.json',identity)
        native.write_json(out/'protocol.json',protocol);native.write_json(out/'planned_trials.json',trials)
        (out/'program').mkdir(exist_ok=True)
        for p in HERE.iterdir():
            if p.is_file() and p.suffix in {'.py','.json','.md'}:(out/'program'/p.name).write_bytes(p.read_bytes())
        torch.set_float32_matmul_precision(precision)
        ma=argparse.Namespace(**next(iter(metadata.values()))['manifest']['args']);ma.save_dir=str(out/'model_context')
        device=resolve_device(ma);model=load_model(ma)[0].to(device).eval()
        native.require(model.gen.ligand_dec.bond_refine.message_mlp[0].in_features==2*model.gen.ligand_dec.bond_refine.node_proj.out_features+2+model.gen.ligand_dec.bond_proj.in_features,'Unexpected native feature layout')
        trace_plan=native.read_json(trace/'plan.json');cached_seed=None;data=None
        for index,t in enumerate(trials):
            state.update(stage='native_rollout',active_trial=t['id']);native.write_json(out/'status.json',state)
            folder=out/'trials'/t['id'];folder.mkdir(parents=True,exist_ok=True)
            done=folder/'result.json'
            if args.resume and done.is_file():
                result=native.read_json(done);native.require(result['completed'] and result['trial']==t,'Cached trial differs')
                for name,sha in result['file_sha256'].items():native.require(native.digest(folder/name)==sha,f'Cached output changed: {name}')
                results[t['id']]=result;write_summary(out,state,results,trials)
                print(f"[{index+1}/{len(trials)}] reused {t['id']}",flush=True);continue
            if cached_seed!=t['seed']:
                data=load_data(metadata[t['seed']],protocol,torch,quality,model);cached_seed=t['seed']
            source=None
            if t['kind']!='sham':
                native.require(all(f'seed_{s}_sham' in results and results[f'seed_{s}_sham']['replay']['baseline_or_sham_full_match'] for s in protocol['seeds']),'All same-input identity rollouts must pass before interventions')
                source=torch.load(out/'trials'/f"seed_{t['seed']}_sham"/'source_features.pt',map_location='cpu',weights_only=True)
            expected=local_expectation(trace,trace_plan,data['rid'],t['call'],t['site'],t['kind'])
            holder={}
            def factory(m,trial,clock,torch_module):
                obj=HeadPatch(m,trial,clock,torch_module,data,source=source,expected=expected);holder['patch']=obj;return obj
            print(f"[{index+1}/{len(trials)}] {t['id']}",flush=True);start=time.monotonic()
            native.write_json(folder/'trial.json',t)
            try:
                final,replay,logs,sampler=native.run_rollout(model,data,t,factory,helpers,device,torch,debug_dir=folder)
            except BaseException:
                obj=holder.get('patch')
                if obj is not None:
                    native.write_json(folder/'partial_patch_receipt.json',obj.receipt)
                    np.savez_compressed(folder/'partial_patch_tensors.npz',**{k:native.array(v) for k,v in obj.capture.items()})
                raise
            obj=holder['patch']
            if t['kind']=='sham':torch.save({k:obj.capture[k] for k in ['before','after']},folder/'source_features.pt')
            np.savez_compressed(folder/'intervention_details.npz',pairs=data['pairs'],
                 used_pair_logits=native.array(obj.capture['used_pair_logits']),
                 before_selected=selected_values(obj.capture['before'],data['pairs']),
                 after_selected=selected_values(obj.capture['after'],data['pairs']))
            native.write_json(folder/'network_timeline.json',logs);native.write_json(folder/'sampler_timeline.json',sampler)
            native.write_json(folder/'replay_checks.json',replay)
            torch.save(helpers.move_copy(final,'cpu',torch),folder/'final_prediction.pt')
            evaluation=native.evaluate_final(final,model,data,t,folder,quality,torch)
            evaluation['stages'][0].update(raw_fixed_checks(final,data,model))
            native_ok=evaluation['raw_valid_connected_root_fragment'];fixed_ok=evaluation['stages'][0]['fixed_chemistry_preserved']
            evaluation['raw_connection_and_fixed_identity']=(False if native_ok is False or fixed_ok is False else
                    None if native_ok is None or fixed_ok is None else True)
            for row in evaluation['stages'][1:]:
                if not row['rdkit_valid']:row['fixed_chemistry_preserved']=None
            if t['kind']=='sham':
                old=next(r for r in data['report']['terminal_labels_unchanged'] if r['stage']=='raw_without_repair')
                raw=evaluation['stages'][0]
                native.require(all(raw[k]==old[k] for k in ['rdkit_valid','fc_valid','root_fixed_substructure_match','fixed_chemistry_preserved']),'Identity rollout terminal labels differ from existing audit')
            later=[r for r in logs if r['call']>t['call']]
            metrics=dict(at_intervention=logs[t['call']],first_later_valence_problem_call=next((r['call'] for r in later if r['site_valence_problem'] is True),None),
                         later_unassessed_calls=[r['call'] for r in later if r['site_valence_problem'] is None])
            result=dict(completed=True,trial=t,replay=replay,evaluation=evaluation,local_metrics=metrics,
                        elapsed_seconds=round(time.monotonic()-start,3),
                        file_sha256={p.name:native.digest(p) for p in folder.iterdir() if p.is_file() and p.name!='result.json' and p.suffix!='.tmp'})
            native.write_json(done,result);results[t['id']]=result
            write_summary(out,state,results,trials)
        state.update(completed=True,stage='finished',active_trial=None)
    except (Exception,KeyboardInterrupt) as exc:
        state['errors'].append(f'{type(exc).__name__}: {exc}')
        (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8');print(state['errors'][-1],file=sys.stderr,flush=True)
    finally:
        write_summary(out,state,results,trials);package=bundle(out)
        print(f"Completed: {state['completed']}\nReview file: {package}",flush=True)
    return 0 if state['completed'] else 1


if __name__=='__main__':raise SystemExit(main())
