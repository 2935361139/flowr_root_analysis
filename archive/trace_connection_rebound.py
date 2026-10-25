#!/usr/bin/env python3
"""Observe recurrence in already completed ROOT counterfactuals.

Reuses the verified causal engine without adding or strengthening a patch.
Read-only hooks record the altered trajectories, including sampler handoffs.
The entire replay must match the already completed trial before acceptance.
--analyze-only summarizes existing JSON records without PyTorch or a GPU.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import importlib.util
import json
import sys
import traceback
import zipfile
from pathlib import Path
from types import MethodType
import numpy as np

VERSION = '2026-10-20.rebound.1'
ENGINE_SHA = 'a89728edaca2e8caa0521c247830b5b94a8eabf25930c7b1b25fc686ca4c61e4'

def require(ok, text):
    if not bool(ok): raise ValueError(text)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def csv_write(path, rows):
    names=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=names);w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v
                     for k,v in r.items()} for r in rows)

def import_engine(path):
    require(Path(path).is_file() and sha(path)==ENGINE_SHA,
            'Use the verified causal.2 causal_scaffold_localization.py in the project root.')
    spec=importlib.util.spec_from_file_location('rebound_causal_engine',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def local_relief(before, after):
    # This is only a change in a local diagnostic, never overall task success.
    return before.get('site_neutral_carbon_overvalent') is True and after.get('site_neutral_carbon_overvalent') is False

def load_completed(folder):
    summary=read(folder/'causal_summary.json');planned=read(folder/'planned_trials.json')
    require(summary['completed'] and summary['error'] is None, 'Prior causal run is incomplete.')
    require(len(planned)==len({t['id'] for t in planned})==summary['completed_trials'], 'Missing/duplicate trials.')
    results={};logs={};sampler={};hashed=0
    for t in planned:
        d=(folder/'trials'/t['id']).resolve()
        require(d.is_relative_to(folder.resolve()), 'Invalid trial path.')
        r=read(d/'result.json');require(r['completed'] and r['trial']==t, 'Trial identity differs.')
        required={'network_timeline.json','sampler_timeline.json','replay_checks.json','trial.json','final_prediction.pt'}
        require(required<=set(r['file_sha256']), 'Incomplete integrity record.')
        for name,expected in r['file_sha256'].items():
            p=(d/name).resolve();require(p.is_relative_to(d) and p.is_file() and sha(p)==expected,
                                       f'Changed/missing recorded file: {t["id"]}/{name}')
            hashed+=1
        require(read(d/'trial.json')==t and read(d/'replay_checks.json')==r['replay'], 'Trial checks differ.')
        replay=r['replay'];require(replay['random_streams_matched_all_calls'] and replay['unmodified_prefix_verified'], 'Replay not verified.')
        a=read(d/'network_timeline.json');b=read(d/'sampler_timeline.json')
        require([v['call'] for v in a]==list(range(replay['call_count'])), 'Missing/duplicate network calls.')
        require(all(v['categorical_values']=='network_logits' for v in a), 'Expected denoised network predictions.')
        if t['kind'] in {'baseline','sham'}:
            require(replay['baseline_or_sham_full_match'] and replay['final_match']['passed'], 'Baseline/sham not matched.')
            require(len(replay['checked_calls'])==len(a), 'Incomplete baseline checks.')
        else:
            p=replay['patch'];require(p['applied'] and p['visits']==1 and p['prewrite_exact'], 'Invalid patch receipt.')
        results[t['id']]=r;logs[t['id']]=a;sampler[t['id']]=b
    baselines={}
    for t in planned:
        if t['kind']=='baseline':
            key=(t['seed'],t['site'],t['partner'])
            require(key not in baselines,'Ambiguous matched baseline.');baselines[key]=t['id']
    for t in planned:
        key=(t['seed'],t['site'],t['partner']);require(key in baselines,'Missing same-seed/same-pair baseline.')
        b=logs[baselines[key]];a=logs[t['id']]
        require(len(a)==len(b) and [v['t'] for v in a]==[v['t'] for v in b], 'Time grid differs.')
        stop=t.get('call',len(a))
        require(a[:stop]==b[:stop],'Observed prefix differs before the intervention.')
    return dict(summary=summary,identity=read(folder/'run_identity.json'),planned=planned,
                results=results,logs=logs,sampler=sampler,baselines=baselines,hashed_files=hashed)

def summarize(context, out):
    out.mkdir(parents=True,exist_ok=True);rows=[];groups={}
    for t in context['planned']:
        id=t['id'];a=context['logs'][id]
        b=context['logs'][context['baselines'][(t['seed'],t['site'],t['partner'])]]
        r=context['results'][id];e=r['evaluation'];kind=t['kind']
        row=dict(id=id,kind=kind,seed=t['seed'],site=t['site'],partner=t['partner'],
                 final_raw_chemical_endpoint=e['raw_valid_connected_fixed_chemistry'],
                 final_site_bond_order_sum=e['final_graph']['site_explicit_bond_order_sum'])
        g=groups.setdefault(kind,dict(trials=0,local_relief=0,local_relief_with_disconnection=0,final_chemical_endpoint_pass=0))
        g['trials']+=1;g['final_chemical_endpoint_pass']+=e['raw_valid_connected_fixed_chemistry'] is True
        if 'call' in t:
            k=t['call'];relief=local_relief(b[k],a[k])
            recurrence=next((v['call'] for v in a[k+1:] if v['site_neutral_carbon_overvalent']),None) if relief else None
            row.update(call=k,layer=t['layer'],module=t['module'],local_relief=relief,
                       local_components=a[k]['n_components'],recurrence_call=recurrence,
                       local_distance_delta_A=a[k]['distance_A']-b[k]['distance_A'],
                       local_single_minus_none_logit_delta=a[k]['single_minus_no_bond_value']-b[k]['single_minus_no_bond_value'],
                       changed_focal_bond_calls=[i for i in range(len(a)) if a[i]['bond_class']!=b[i]['bond_class']])
            g['local_relief']+=relief;g['local_relief_with_disconnection']+=relief and not a[k]['connected']
        rows.append(row)
    report=dict(version=VERSION,prior_summary=context['summary'],verified_file_hashes=context['hashed_files'],
                groups=groups,trials=rows,
                scope='Local effects are not final rescues; no unique root cause, input-atom SHAP, or cross-complex generalization is established.')
    write(out/'effect_summary.json',report);csv_write(out/'effect_comparison.csv',rows)
    selected=[r for r in rows if r['kind']=='targeted']
    cells=lambda vals:'<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in vals)+'</tr>'
    table=''.join(cells([r['id'],r.get('local_relief'),r.get('local_components'),r.get('recurrence_call'),r['final_raw_chemical_endpoint']]) for r in selected)
    counts=''.join(cells([kind,g['trials'],g['local_relief'],g['local_relief_with_disconnection'],g['final_chemical_endpoint_pass']]) for kind,g in groups.items())
    doc='''<!doctype html><meta charset="utf-8"><title>因果实验复核</title>
<style>body{font:16px/1.7 system-ui;max-width:1100px;margin:36px auto;padding:0 20px;color:#203047}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #cbd5e1;padding:8px;text-align:left}code{background:#eef2f7;padding:3px}</style>
<h1>第一轮因果实验：运行完成，结论分层</h1>
<p>“实验完成”只说明流程执行完毕。局部预测改变、最终失败修复和唯一根因是三个不同层面的结论。</p>
<table><tr><th>实验类型</th><th>次数</th><th>局部超价消失</th><th>消失但预测断连</th><th>最终化学联合检查通过</th></tr>'''+counts+'''</table>
<p>targeted：定向修改；random_direction：同幅度随机方向；time_control：时机对照；success_control：原成功轨迹上的规则对照。成功轨迹仍通过不能称为失败修复。重复干预同一条轨迹不等于独立分子样本。</p>
<p>表中的“局部超价消失”仅指原先超价的中性碳在该次网络预测中不再超价。预测若变成多个片段，不能算 scaffold hopping 修复成功。采样器的中间噪声状态也不能直接套用终态化学有效性标准。</p>
<table><tr><th>定向实验</th><th>局部超价消失</th><th>此时连通分量</th><th>再次超价的调用索引</th><th>最终化学联合检查</th></tr>'''+table+'''</table>
<p>调用索引从 0 开始。不存在复发值可能表示该次干预没有消除局部超价，不能解释成永久修复。</p>
<p>完整对照和原始数值见 <a href="effect_comparison.csv">effect_comparison.csv</a> 与 <a href="effect_summary.json">effect_summary.json</a>。</p>
<p>下一步记录同一反事实轨迹的当前预测、采样更新、自条件输入及子模块输出，定位局部影响何处减弱或异常重新形成。表示差异本身仍不等于原因。</p>'''
    (out/'REPORT.html').write_text(doc,encoding='utf-8');return report

def feature_summary(a, b):
    require(a.shape==b.shape and a.dtype==b.dtype, 'Compared feature shape/dtype differs.')
    x,y=a.astype(float),b.astype(float);d=x-y
    return dict(max_abs=float(np.max(abs(d))) if d.size else 0.,
                delta_l2=float(np.linalg.norm(d)),baseline_l2=float(np.linalg.norm(y)))

def probe_state(state,site,partner,scale,engine):
    if state is None:return dict(available=False)
    result=dict(available=True)
    if state.get('coords') is not None:
        c=engine.array(state['coords'][0]).astype(float)
        result['distance_A']=float(np.linalg.norm(c[partner]-c[site])*scale)
    if state.get('bonds') is not None:
        b=engine.array(state['bonds'][0]);i,j=max(site,partner),min(site,partner)
        require(b.shape[-1]==5,'Unsupported bond categories.')
        result['focal_bond_class']=int(b[i,j].argmax())
        result['single_minus_none_value']=float(b[i,j,1]-b[i,j,0])
    if state.get('mask') is not None:
        mask=engine.array(state['mask'][0]).astype(bool)
        result['both_slots_unmasked']=bool(mask[site] and mask[partner])
    return result

class ReboundObserver:
    """Read-only, post-patch observation. No synthetic intermediate state is injected."""
    def __init__(self, model, calls, atoms, site, folder, engine, helpers, torch, with_modules=True,partner=None,scale=1.):
        self.model,self.calls,self.atoms,self.site=model,set(calls),list(atoms),site
        self.folder,self.e,self.h,self.torch=folder,engine,helpers,torch
        self.partner,self.scale=partner,scale
        self.k=-1;self.pending=None;self.arrays={};self.handles=[];self.methods=engine.ScopedMethods()
        self.records=[];self.installed=False;self.with_modules=with_modules;self.expected=None
        folder.mkdir(parents=True,exist_ok=True)
        self.handles.append(model.register_forward_pre_hook(self.pre,with_kwargs=True))
        self.handles.append(model.register_forward_hook(self.post,with_kwargs=True))
        original_predictions=model._get_predictions
        def predictions(_self,*a,**kw):
            result=original_predictions(*a,**kw)
            if self.pending is not None:
                require('prediction_conversion' not in self.pending,'Unexpected repeated prediction conversion.')
                self.pending['prediction_conversion']=self.snapshot(result)
            return result
        self.methods.set(model,'_get_predictions',predictions)
        original_step=model.integrator.step
        def step(_self,*a,**kw):
            before=None
            if self.pending is not None:
                require(len(a)==5 and not kw,'Integrator step signature changed.')
                before=dict(current=self.snapshot(a[0]),predicted=self.snapshot(a[1]),
                            times=self.snapshot(a[3]),dt=self.snapshot(a[4]))
            result=original_step(*a,**kw)
            if self.pending is not None:
                require('integration' not in self.pending,'Expected one Euler update per network call.')
                self.pending['integration']=dict(before=before,after=self.snapshot(result))
            return result
        self.methods.set(model.integrator,'step',step)

    def snapshot(self,x):return self.h.move_copy(x,'cpu',self.torch)
    def put(self,name,value,edge=False):
        if self.pending is None or value is None:return
        require(name not in self.arrays and name+'__from_site' not in self.arrays,'Repeated captured feature: '+name)
        require(self.torch.is_tensor(value) and value.shape[0]==1,'Expected batch-one feature: '+name)
        a=self.e.array(value[0]);require(np.isfinite(a).all(),'Nonfinite internal feature: '+name)
        if edge:
            require(a.ndim==3 and a.shape[0]==a.shape[1], 'Expected ligand-pair features: '+name)
            self.arrays[name+'__from_site']=a[self.site,self.atoms].copy()
            self.arrays[name+'__to_site']=a[self.atoms,self.site].copy()
        else:
            require(max(self.atoms)<a.shape[0],'Tracked atom out of range: '+name)
            self.arrays[name]=a[self.atoms].copy()

    def install_modules(self):
        # Called from first model pre-hook: run_rollout has already installed its
        # patch hook. Observers appended here therefore see the effective write.
        dec=self.model.gen.ligand_dec
        def add(module,fn,pre=False,kwargs=False):
            register=module.register_forward_pre_hook if pre else module.register_forward_hook
            self.handles.append(register(fn,with_kwargs=True) if kwargs else register(fn))
        for i,layer in enumerate(dec.layers):
            p=f'layer_{i+1:02d}__'
            def layer_pre(mod,a,kw,p=p):
                if self.pending is None:return
                for name,v in zip(['in_equis','in_invs'],a[:2]):self.put(p+name,v)
                self.put(p+'in_edges',a[2],edge=True);self.put(p+'in_coords',kw.get('coords'))
            def layer_post(mod,a,result,p=p):
                if self.pending is None:return
                require(isinstance(result,tuple) and len(result)==5,'Layer output contract changed.')
                for name,v in zip(['out_equis','out_invs','out_coords'],result[:3]):self.put(p+name,v)
                self.put(p+'out_edges',result[3],edge=True)
            add(layer,layer_pre,pre=True,kwargs=True);add(layer,layer_post)
            for attr in ['self_attention','cond_attention']:
                if not hasattr(layer,attr):continue
                def attention(mod,a,result,p=p,attr=attr):
                    if self.pending is None:return
                    require(isinstance(result,tuple) and len(result)==3,'Attention output contract changed.')
                    self.put(p+attr+'__effective_equi_write',result[0])
                    self.put(p+attr+'__inv_write',result[1])
                    if attr=='self_attention':self.put(p+attr+'__edge_output',result[2],edge=True)
                add(getattr(layer,attr),attention)
            for attr in ['equi_ff','inv_ff']:
                if hasattr(layer,attr):
                    def ff(mod,a,result,p=p,attr=attr):self.put(p+attr+'__write',result)
                    add(getattr(layer,attr),ff)
        def refine_pre(mod,a):
            if self.pending is None:return
            require(len(a)==4,'Bond-refinement signature changed.')
            self.put('bond_head__coords',a[0]);self.put('bond_head__invs',a[1]);self.put('bond_head__edges',a[3],edge=True)
        def refine_post(mod,a,result):self.put('bond_head__refined_edges',result,edge=True)
        def proj_pre(mod,a):self.put('bond_head__projection_input',a[0],edge=True)
        def proj_post(mod,a,result):self.put('bond_head__logits',result,edge=True)
        add(dec.bond_refine,refine_pre,pre=True);add(dec.bond_refine,refine_post)
        add(dec.bond_proj,proj_pre,pre=True);add(dec.bond_proj,proj_post)

    def pre(self,mod,a,kw):
        self.flush();self.k+=1
        if not self.installed:
            if self.with_modules:self.install_modules()
            self.installed=True
        if self.k in self.calls:
            require(len(a)>=3,'Network input signature changed.')
            self.pending=dict(call=self.k,tracked_atoms=self.atoms,site=self.site,
                input_state=self.snapshot(a[0]),times=self.snapshot(a[2]),
                self_condition=self.snapshot(kw.get('cond_batch')),forward_finished=False)
            self.arrays={}

    def post(self,mod,a,kw,result):
        if self.pending is not None:
            self.pending['network_output']=self.snapshot(result);self.pending['forward_finished']=True

    def flush(self):
        if self.pending is None:return
        p=self.folder/f'call_{self.pending["call"]:04d}'
        self.torch.save(self.pending,p.with_suffix('.pt'))
        np.savez_compressed(p.with_suffix('.npz'),**self.arrays)
        keys=sorted(self.arrays)
        if self.pending['forward_finished']:
            if self.expected is None:self.expected=keys
            require(keys==self.expected,'Module coverage changed between calls.')
            if self.with_modules:
                require('bond_head__logits__to_site' in keys,'Bond head was not captured.')
        stages={}
        if self.partner is not None:
            values={'input_noisy_state':self.pending['input_state'],
                    'self_condition_input_probabilities':self.pending['self_condition'],
                    'network_output_logits':self.pending.get('network_output')}
            if 'prediction_conversion' in self.pending:
                pred,cond=self.pending['prediction_conversion']
                values.update(converted_prediction_probabilities=pred,self_condition_before_fragment_restore_probabilities=cond)
            if 'integration' in self.pending:
                values['integrated_state_before_fragment_restore']=self.pending['integration']['after']
            stages={k:probe_state(v,self.site,self.partner,self.scale,self.e) for k,v in values.items()}
        self.records.append(dict(call=self.pending['call'],finished=self.pending['forward_finished'],handoff=stages,
                                 state_file=p.with_suffix('.pt').name,features_file=p.with_suffix('.npz').name,
                                 state_sha256=sha(p.with_suffix('.pt')),features_sha256=sha(p.with_suffix('.npz'))))
        self.pending=None;self.arrays={}

    def close(self):
        try:self.flush()
        finally:
            for handle in self.handles:handle.remove()
            self.handles=[];self.methods.close()

def compare_captures(out, records, selected, context):
    index={t['id']:{r['call']:r for r in records[t['id']]} for t in selected}
    byid={t['id']:t for t in selected};rows=[];handoffs=[]
    for id,items in index.items():
        t=byid[id];baseline=context['baselines'][(t['seed'],t['site'],t['partner'])]
        if id==baseline:continue
        require(baseline in index,'A selected counterfactual needs its same-focus baseline.')
        for call,r in items.items():
            br=index[baseline][call]
            for stage,values in r['handoff'].items():
                reference=br['handoff'][stage]
                hrow=dict(trial=id,call=call,stage=stage,**values)
                for key in ['distance_A','single_minus_none_value']:
                    if key in values and key in reference:
                        hrow['baseline_'+key]=reference[key];hrow['delta_'+key]=values[key]-reference[key]
                handoffs.append(hrow)
            with np.load(out/'trials'/id/'captures'/r['features_file'],allow_pickle=False) as a, np.load(out/'trials'/baseline/'captures'/index[baseline][call]['features_file'],allow_pickle=False) as b:
                require(set(a.files)==set(b.files),'Paired capture coverage differs.')
                for feature in a.files:
                    rows.append(dict(trial=id,call=call,feature=feature,**feature_summary(a[feature],b[feature])))
    csv_write(out/'internal_differences.csv',rows)
    csv_write(out/'handoff_comparison.csv',handoffs)
    return dict(compared_feature_rows=len(rows),interpretation='Difference norms locate changes; they are not causal importance or atom-level contribution scores.')

def run_capture(args, context, out, summary):
    from rdkit import rdBase
    import torch
    from tensordict import TensorDict
    from tensordict._reductions import _make_td
    from flowr.scriptutil import load_model
    from flowr.util.device import resolve_device
    engine=import_engine(Path.cwd()/'causal_scaffold_localization.py')
    helper=engine.import_file(Path.cwd()/'replay_connection_hidden.py','rebound_helper',engine.HELPER_SHA)
    quality=engine.import_file(Path.cwd()/'evaluate_scaffold_outputs.py','rebound_quality')
    plan=read(args.plan);wanted=plan['trial_ids'];calls=plan['capture_calls']
    require(len(wanted)==len(set(wanted)) and len(calls)==len(set(calls)) and calls==sorted(calls),'Duplicate or unsorted selections.')
    selected=[next((t for t in context['planned'] if t['id']==id),None) for id in wanted]
    require(all(t is not None for t in selected),'Selected trial is not in the completed run.')
    require(all(context['baselines'][(t['seed'],t['site'],t['partner'])] in wanted for t in selected),'Select the same-seed/same-pair baseline for each trial.')
    priorplan=context['identity']['plan']
    data,source_records,all_trials,identity,readout=engine.prepare(args,priorplan,torch,helper,quality)
    require(identity==context['identity'] and all_trials==context['planned'],'Model, inputs, code or prior experiment identity changed.')
    for t in selected:
        cfg=data[t['seed']]['manifest']['args'];inputs=data[t['seed']]['inputs']
        require(inputs['solver']=='euler' and int(inputs['sc_recycle_steps'])==1 and cfg['corrector_iters'] in [0,None],
                'This handoff observer currently supports Euler, one forward per step, no extra corrector iterations.')
        require(calls and min(calls)>=0 and max(calls)<len(data[t['seed']]['calls'])-1,'Capture ordinary integration calls; exclude final corrector.')
    metadata=dict(version=VERSION,script_sha256=sha(__file__),prior_identity_sha256=sha(args.causal_dir/'run_identity.json'),
                  engine_sha256=ENGINE_SHA,plan=plan,torch_version=str(torch.__version__),rdkit_version=rdBase.rdkitVersion)
    write(out/'capture_identity.json',metadata)
    torch.set_float32_matmul_precision(readout['precision'])
    base=data[selected[0]['seed']]['manifest'];model_args=argparse.Namespace(**base['args']);model_args.save_dir=str(out/'model_context')
    device=resolve_device(model_args);model=load_model(model_args)[0].to(device).eval()
    require(sum(p.requires_grad for p in model.parameters())==readout['parameter_tensors_requiring_grad'],'Native parameter flags differ.')
    for d in data.values():
        require(bool(model.self_condition)==d['inputs']['self_condition'] and int(model.sc_recycle_steps)==d['inputs']['sc_recycle_steps'],'Self-conditioning settings differ.')
        require(float(model.coord_scale)==d['inputs']['coord_scale'],'Coordinate scale differs.')
    # Use the same tracking set for each matched group; never align slots across seeds.
    tracking={}
    for t in selected:
        key=(t['seed'],t['site'],t['partner']);d=data[t['seed']]
        if plan['track_all_atoms']:atoms=set(range(d['inputs']['prior']['mask'].shape[1]))
        else:
            atoms=set(d['task']['fixed_sampler_indices'])|{t['site'],t['partner']}
            for other in selected:
                if (other['seed'],other['site'],other['partner'])==key:
                    for k in calls:atoms.update(context['logs'][other['id']][k]['site_neighbors'])
                    atoms.update(context['results'][other['id']]['evaluation']['final_graph']['site_neighbors'])
        tracking[key]=sorted(atoms)
    all_records={};summary['selected_trials']=wanted
    for n,t in enumerate(selected):
        print(f'[{n+1}/{len(selected)}] Observe existing {t["id"]}',flush=True)
        summary['active_trial']=t['id'];folder=out/'trials'/t['id'];folder.mkdir(parents=True,exist_ok=True)
        d=data[t['seed']];atoms=tracking[(t['seed'],t['site'],t['partner'])]
        observer=ReboundObserver(model,calls,atoms,t['site'],folder/'captures',engine,helper,torch,
                                 partner=t['partner'],scale=d['inputs']['coord_scale'])
        try:
            reference,delta,vectors=engine.trial_vectors(t,source_records,priorplan)
            final,replay,logs,sampler=engine.run_rollout(model,d,t,reference,delta,helper,device,torch,debug_dir=folder)
        finally:observer.close()
        require([r['call'] for r in observer.records]==calls and all(r['finished'] for r in observer.records),'Incomplete observation window.')
        require(logs==context['logs'][t['id']] and sampler==context['sampler'][t['id']],
                'Observation changed the recorded network/sampler timeline. Stop before interpretation.')
        with torch.serialization.safe_globals([TensorDict,_make_td]):
            expected=torch.load(args.causal_dir/'trials'/t['id']/'final_prediction.pt',map_location='cpu',weights_only=True)
        final_check=helper.compare_predictions(final,expected,torch,np,0.,0.)
        require(final_check['passed'],'Observed final tensor differs from the already completed trial.')
        effective_write=None
        if t['kind']!='baseline' and t['call'] in calls:
            attr=engine.MODULE_ATTRS[t['module']]
            suffix='__write' if attr=='equi_ff' else '__effective_equi_write'
            key=f'layer_{t["layer"]:02d}__'+attr+suffix
            with np.load(folder/'captures'/f'call_{t["call"]:04d}.npz',allow_pickle=False) as z:
                expected_write=(reference[t['partner']].astype(float)+delta).astype(reference.dtype)
                require(np.array_equal(z[key][atoms.index(t['partner'])],expected_write),
                        'Observer did not capture the effective post-patch write.')
            effective_write=True
        result=dict(trial=t,accepted=True,tracked_atoms=atoms,full_network_timeline_exact=True,
                    full_sampler_timeline_exact=True,final_tensor_exact=final_check,
                    effective_patch_write_exact=effective_write,
                    original_intervention_replay=replay,captures=observer.records,
                    evaluation=context['results'][t['id']]['evaluation'],
                    evaluation_provenance='Prior evaluation retained only after exact final-tensor equality; chemical evaluators were not rerun.')
        write(folder/'result.json',result);all_records[t['id']]=observer.records
        summary['completed_trials']=len(all_records);write(out/'rebound_summary.json',summary)
        del final,expected,observer
    summary['internal_comparison']=compare_captures(out,all_records,selected,context)
    summary['capture_completed']=True;summary['active_trial']=None
    summary['scope']='Read-only observation of already tested counterfactuals. No new intervention, no unique root cause, no input-atom SHAP, no cross-complex validation.'

def bundle(out):
    path=out/'rebound_review.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p!=path:z.write(p,p.relative_to(out))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--causal-dir',type=Path,default=Path('outputs/sh_causal_localization_ptp1b_v2'))
    ap.add_argument('--submodule-dir',type=Path,default=Path('outputs/sh_submodule_localization_ptp1b'))
    ap.add_argument('--readout-dir',type=Path,default=Path('outputs/sh_coordinate_source_ptp1b'))
    ap.add_argument('--output-root',type=Path,default=Path('outputs'))
    ap.add_argument('--plan',type=Path,default=Path(__file__).with_name('rebound_plan.json'))
    ap.add_argument('--output-dir',type=Path,default=Path('outputs/sh_connection_rebound_ptp1b_v1'))
    ap.add_argument('--analyze-only',action='store_true')
    args=ap.parse_args();out=args.output_dir.resolve();args.causal_dir=args.causal_dir.resolve()
    if out.exists():ap.error('Choose a new output directory; preserve previous results.')
    out.mkdir(parents=True);summary=dict(version=VERSION,completed=False,capture_completed=False,completed_trials=0,error=None)
    try:
        context=load_completed(args.causal_dir);summarize(context,out/'previous_effects')
        summary['previous_trials_verified']=len(context['planned']);summary['previous_file_hashes_verified']=context['hashed_files']
        if not args.analyze_only:
            require((Path.cwd()/'flowr/scriptutil.py').is_file(),'Run in the existing ROOT project.')
            sys.path.insert(0,str(Path.cwd()));run_capture(args,context,out,summary)
        summary['completed']=True
    except Exception as exc:
        summary['error']=f'{type(exc).__name__}: {exc}'
        (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('STOP:',summary['error'],flush=True)
    finally:
        write(out/'rebound_summary.json',summary);bundle(out)
        print(f'Completed: {summary["completed"]}; captures {summary["completed_trials"]}',flush=True)
        print('Review file:',out/'rebound_review.zip',flush=True)
    return 0 if summary['completed'] else 1

if __name__=='__main__':raise SystemExit(main())
