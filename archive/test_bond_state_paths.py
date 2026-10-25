#!/usr/bin/env python3
"""One-event causal path tests using the complete, unchanged native ROOT sampler.

Separate a noisy bond state's use by the network from its use by the integrator.
This tests a specified candidate, not a unique root cause or an input SHAP score.
Never overwrite the final molecule, reinsert later saved states, or train weights.
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
from types import SimpleNamespace
import numpy as np

VERSION = '2026-10-21.bond-paths.1'
ENGINE_SHA = 'a89728edaca2e8caa0521c247830b5b94a8eabf25930c7b1b25fc686ca4c61e4'
OBSERVER_SHA = 'f7067cd29773b3428fd0c70ea4768e70dd3130fb3807075043f3ea9428851574'
MODES = ('network', 'integration', 'both')

def require(ok, message):
    if not bool(ok): raise ValueError(message)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def import_module(path, name, expected):
    path=Path(path);require(path.is_file(),f'Missing existing helper: {path}')
    require(sha(path)==expected,f'Helper version changed: {path.name}. Preserve outputs and report this error.')
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def as_array(x):
    return x.detach().cpu().numpy().copy() if hasattr(x,'detach') else np.asarray(x).copy()

def pair_vector(state, pair):
    b=as_array(state['bonds']);i,j=pair
    require(b.ndim==4 and b.shape[0]==1 and b.shape[1]==b.shape[2] and b.shape[-1]==5,'Unsupported bond tensor.')
    require(0<=i<b.shape[1] and 0<=j<b.shape[1] and i!=j,'Pair indices out of range.')
    require(as_array(state['mask'])[0,i] and as_array(state['mask'])[0,j],'Inactive pair slot.')
    v=b[0,i,j]
    require(np.array_equal(v,b[0,j,i]),'Intervention requires a symmetric pair.')
    require(np.isfinite(v).all() and np.array_equal(v,np.eye(len(v),dtype=v.dtype)[int(v.argmax())]),
            'Expected a categorical one-hot internal state, not logits/probabilities.')
    return v

def make_trials(plan, states, fixed):
    """Indices/times come from configuration; controls are derived from real states."""
    site=int(plan['site']);pair=list(map(int,plan['focal_pair']));prev=int(plan['previous_call']);k=int(plan['event_call'])
    require(prev+1==k and prev in states and k in states,'Need consecutive, captured event calls.')
    require(site in fixed and site in pair and len(set(pair)&set(fixed))==1,'Focal pair must join a fixed and a generated slot.')
    partner=pair[1] if pair[0]==site else pair[0]
    before=states[prev]['input_state'];after=states[k]['input_state']
    a=pair_vector(before,pair);b=pair_vector(after,pair)
    require(int(a.argmax())==plan['expected_old_class'] and int(b.argmax())==plan['expected_new_class'] and not np.array_equal(a,b),
            'The configured event is not present in the uploaded capture.')
    require(np.array_equal(pair_vector(states[prev]['integration']['after'],pair),b),'Event did not pass through the recorded integrator output.')
    focus=dict(seed=int(plan['seed']),site=site,partner=partner,monitor_partners=list(map(int,plan['monitor_partners'])))
    require(partner in focus['monitor_partners'],'Monitor the focal generated endpoint.')
    trials=[dict(**focus,id='baseline',kind='baseline',mode='none')]
    def add(id,kind,mode,call,p,q,r):
        trials.append(dict(**focus,id=id,kind=kind,mode=mode,call=int(call),pair=list(map(int,p)),
                           from_vector=q.tolist(),to_vector=r.tolist(),from_class=int(q.argmax()),to_class=int(r.argmax())))
    for mode in MODES:add('sham_'+mode,'sham',mode,k,pair,b,b)
    for mode in MODES:add('focal_'+mode,'focal',mode,k,pair,b,a)
    old=as_array(before['bonds'])[0].argmax(-1);new=as_array(after['bonds'])[0].argmax(-1)
    require(old.shape==new.shape,'Atom slots changed between consecutive states.')
    changes=[]
    for i,j in zip(*np.where(np.triu(old!=new,1))):
        p=[int(i),int(j)]
        if i in fixed and j in fixed:continue
        av=pair_vector(before,p);bv=pair_vector(after,p)
        changes.append(dict(pair=p,old_class=int(av.argmax()),new_class=int(bv.argmax())))
        if set(p)==set(pair):continue
        for mode in MODES:add(f'other_{i}_{j}_{mode}','other_pair',mode,k,p,bv,av)
    for call in plan['time_control_calls']:
        require(call in states and call!=k,'Invalid time control call.')
        current=pair_vector(states[call]['input_state'],pair)
        require(np.array_equal(current,b),'Time control must start from the same bond category as the focal intervention.')
        add(f'time_{call}_both','time_control','both',call,pair,current,a)
    if plan.get('include_reverse_timing_control',False):
        add(f'reverse_{prev}_both','reverse_timing','both',prev,pair,a,b)
    require(len({t['id'] for t in trials})==len(trials),'Duplicate trial IDs.')
    return trials,changes

def replace_pair(batch, trial, helper, device, torch, engine):
    """Clone, then replace only one symmetric bond pair; never mutate native curr."""
    before=as_array(batch['bonds']);i,j=trial['pair']
    v=pair_vector(batch,[i,j]);expected=np.asarray(trial['from_vector'],dtype=v.dtype)
    target=np.asarray(trial['to_vector'],dtype=v.dtype)
    require(np.array_equal(v,expected),'Pre-intervention pair differs from its recorded baseline.')
    require(np.array_equal(target,np.eye(len(target),dtype=target.dtype)[int(target.argmax())]),'Target must be one-hot.')
    out=helper.move_copy(batch,device,torch)
    out['bonds'][0,i,j]=torch.as_tensor(target,device=device,dtype=out['bonds'].dtype)
    out['bonds'][0,j,i]=torch.as_tensor(target,device=device,dtype=out['bonds'].dtype)
    wanted=before.copy();wanted[0,i,j]=target;wanted[0,j,i]=target
    require(np.array_equal(as_array(batch['bonds']),before),'The original integration state was mutated.')
    require(np.array_equal(as_array(out['bonds']),wanted),'Unrequested bond entries changed.')
    for key in batch:
        if key!='bonds':engine.compare_tree(out[key],batch[key],torch,'input.'+str(key))
    delta=wanted.astype(float)-before.astype(float)
    return out,dict(pair=[i,j],before=v.tolist(),after=target.tolist(),
                    changed_scalars=int(np.count_nonzero(delta)),delta_norm=float(np.linalg.norm(delta)),
                    original_unchanged=True,only_requested_pair_changed=True)

def run_trial(model,data,trial,helper,device,torch,e,observer_cls=None,folder=None,capture_calls=(),tracked_atoms=(),with_modules=True):
    """Run the actual model._generate; separate reading and integration paths once."""
    inputs=data['inputs'];calls=data['calls'];cfg=data['manifest']['args'];move=lambda x:helper.move_copy(x,device,torch)
    prior=move(inputs['prior']);pocket=move(inputs['pocket_data']);enc=move(data['encoding'])
    if 'complex' in pocket:
        require(len(inputs['pocket_com'])==1,'Expected one pocket COM.')
        pocket['complex']=[SimpleNamespace(com=inputs['pocket_com'][0].cpu().clone())]
    clock={'call':-1};logs=[];sampler=[];checks=[];receipts=[];handles=[];observer=None
    methods=e.ScopedMethods();routes=e.ScopedMethods();strict=trial['kind'] in {'baseline','sham'}
    patch_call=trial.get('call',len(calls));mode=trial['mode']
    def pre(mod,args,kwargs):
        clock['call']+=1;k=clock['call'];require(k<len(calls),'Unexpected extra network call.')
        require(len(args)>=3,'Native network signature changed.')
        batch,times=args[0],args[2];c=calls[k]
        clock.update(phase='before_forward',last_input=move(batch));clock.pop('last_output',None)
        if k==0:helper.restore_rng(c['rng_before_forward'],device,torch,np)
        require(e.rng_matches(c['rng_before_forward'],device,torch),f'Random streams diverged at call {k}.')
        e.compare_tree(times,c['times'],torch,'times');e.compare_tree(batch['mask'],c['state_before_forward']['mask'],torch,'mask')
        if strict or k<=patch_call:
            e.compare_tree(batch,c['state_before_forward'],torch,'input')
            e.compare_tree(kwargs.get('cond_batch'),c['self_condition'],torch,'self_condition')
        if k==patch_call and mode in {'network','both'}:
            replacement,receipt=replace_pair(batch,trial,helper,device,torch,e)
            receipts.append(dict(call=k,path='network',**receipt))
            return (replacement,*args[1:]),kwargs
        return None
    def post(mod,args,kwargs,out):
        k=clock['call'];clock.update(phase='after_forward',last_output=out)
        require(all(np.isfinite(as_array(v)).all() for v in helper.tensor_leaves(out,torch).values()),'Nonfinite network prediction.')
        # Integration-only must leave this call's network output exactly unchanged.
        if strict or k<patch_call or (k==patch_call and mode=='integration'):
            check=helper.compare_predictions(out,calls[k]['network_output_before_postprocessing'],torch,np,0.,0.)
            require(check['passed'],f'Exact replay differs at network call {k}: {check}')
            checks.append(dict(call=k,passed=True,max_difference=0.))
        row=dict(call=k,t=calls[k]['times_scalar'],categorical_values='network_logits',
                 **e.graph_metrics(out,model.builder,trial,inputs['coord_scale']))
        row['monitored_pairs']=[e.graph_metrics(out,model.builder,dict(site=trial['site'],partner=j),inputs['coord_scale']) for j in trial['monitor_partners']]
        logs.append(row)
    def encoding(_self,*a,**kw):return enc['pocket_equis'],enc['pocket_invs']
    def build(_self,curr,predicted,pocket_data,iter=0,step=0):
        sampler.append(dict(step=step,categorical_values='native_integrated_state_or_final_probabilities',
                            **e.graph_metrics(curr,model.builder,trial,inputs['coord_scale'])))
        curr['res_names']=torch.zeros(curr['atomics'].size(0),curr['atomics'].size(1),device=device).long()
    try:
        methods.set(model.gen,'get_pocket_encoding',encoding)
        methods.set(model,'_build_trajectory',build)
        methods.set(model,'_save_trajectory',lambda _self,*a,**kw:None)
        handles=[model.register_forward_pre_hook(pre,with_kwargs=True),model.register_forward_hook(post,with_kwargs=True)]
        if observer_cls is not None:
            require(folder is not None,'Capture folder is required.')
            observer=observer_cls(model,capture_calls,tracked_atoms,trial['site'],folder/'captures',e,helper,torch,
                                  with_modules=with_modules,partner=trial['partner'],scale=inputs['coord_scale'])
        original_step=model.integrator.step
        def step(_self,*args,**kwargs):
            require(len(args)==5 and not kwargs,'Integrator signature changed.')
            k=clock['call']
            if k==patch_call and mode in {'integration','both'}:
                replacement,receipt=replace_pair(args[0],trial,helper,device,torch,e)
                receipts.append(dict(call=k,path='integration',**receipt))
                args=(replacement,*args[1:])
            # The observer wrapper is inside this wrapper: it sees effective inputs.
            return original_step(*args,**kwargs)
        routes.set(model.integrator,'step',step)
        helper.restore_rng(inputs['rng'],device,torch,np)
        with torch.no_grad():
            final=model._generate(prior,pocket,steps=cfg['integration_steps'],times=move(inputs['times']),
                strategy=inputs['strategy'],solver=inputs['solver'],corr_iters=cfg['corrector_iters'],
                save_traj=bool(cfg['save_traj']),final_inpaint=inputs['final_inpaint'],
                apply_guidance=False,coord_noise_level=cfg['coord_noise_scale'])
        require(clock['call']+1==len(calls),'Incomplete native generation.')
        paths=[] if mode=='none' else (['network','integration'] if mode=='both' else [mode])
        require([r['path'] for r in receipts]==paths,'Intervention was not applied exactly once per requested path.')
        final_match=None
        if strict:
            final_match=helper.compare_predictions(final,data['final'],torch,np,0.,0.)
            require(final_match['passed'],'Baseline/sham final prediction differs from the original.')
        replay=dict(call_count=len(logs),random_streams_matched_all_calls=True,unmodified_prefix_verified=True,
                    baseline_or_sham_full_match=strict,checked_calls=checks,final_match=final_match,patch_receipts=receipts)
    except BaseException:
        if folder is not None:
            try:
                torch.save(helper.move_copy(dict(trial=trial,clock=clock),'cpu',torch),folder/'replay_failure.pt')
                write(folder/'partial_network_timeline.json',logs)
            except Exception:(folder/'debug_error.txt').write_text(traceback.format_exc())
        raise
    finally:
        routes.close()
        try:
            if observer is not None:observer.close()
        finally:
            for h in handles:h.remove()
            methods.close()
    captures=observer.records if observer is not None else []
    if observer is not None:
        require([c['call'] for c in captures]==list(capture_calls) and all(c['finished'] for c in captures),'Incomplete internal capture.')
    return final,replay,logs,sampler,captures

def load_earlier(folder, context, torch, e, plan):
    summary=read(folder/'rebound_summary.json');identity=read(folder/'capture_identity.json')
    require(summary['completed'] and summary['capture_completed'] and summary['error'] is None,'Earlier capture is incomplete.')
    require(identity['script_sha256']==OBSERVER_SHA and identity['engine_sha256']==ENGINE_SHA,'Earlier observer/engine version differs.')
    candidates=[t for t in context['planned'] if t['kind']=='baseline' and t['seed']==plan['seed'] and t['site']==plan['site']]
    require(len(candidates)==1,'Need one matching earlier baseline.')
    tid=candidates[0]['id'];d=folder/'trials'/tid;r=read(d/'result.json')
    require(r['accepted'] and r['full_network_timeline_exact'] and r['full_sampler_timeline_exact'] and r['final_tensor_exact']['passed'],
            'Earlier baseline was not verified.')
    require(all(v['max_abs_difference']==0 for v in r['final_tensor_exact']['fields'].values()),'Earlier final tensors are not exact.')
    require(r['evaluation']==context['results'][tid]['evaluation'],'Earlier endpoint provenance differs.')
    states={}
    for c in r['captures']:
        require(c['finished'],'Unfinished capture.')
        for fk,hk in [('state_file','state_sha256'),('features_file','features_sha256')]:
            p=(d/'captures'/c[fk]).resolve()
            require(p.is_relative_to(d.resolve()) and sha(p)==c[hk],'Earlier capture hash mismatch.')
        states[c['call']]=torch.load(d/'captures'/c['state_file'],map_location='cpu',weights_only=True)
    require(sorted(states)==identity['plan']['capture_calls'],'Missing/duplicate earlier calls.')
    return states,identity

def compare_internal(out, results, e):
    rows=[];baseline=results.get('baseline')
    if not baseline:return
    bm={c['call']:c for c in baseline['captures']}
    for tid,r in results.items():
        if r['trial']['kind']!='focal':continue
        for c in r['captures']:
            k=c['call'];a=out/'trials/baseline/captures'/bm[k]['features_file'];b=out/'trials'/tid/'captures'/c['features_file']
            with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
                require(set(x.files)==set(y.files),'Paired internal feature coverage differs.')
                for key in sorted(x.files):
                    u,v=x[key].astype(float),y[key].astype(float);require(u.shape==v.shape,'Paired feature shape differs.')
                    delta=v-u
                    rows.append(dict(trial=tid,call=k,feature=key,l2_difference=float(np.linalg.norm(delta)),
                                     max_abs_difference=float(np.max(np.abs(delta))) if delta.size else 0.,
                                     changed_entries=int(np.count_nonzero(delta)),
                                     interpretation='Matched within-seed response to intervention; not an importance ranking.'))
    e.write_csv(out/'internal_responses.csv',rows)

def report(out,trials,results,summary,e):
    rows=[]
    for t in trials:
        if t['id'] not in results:continue
        r=results[t['id']];v=r['evaluation'];raw=v['stages'][0];g=v['final_graph']
        rows.append(dict(trial=t['id'],kind=t['kind'],mode=t['mode'],call=t.get('call'),pair=t.get('pair'),
                         chemical_endpoint=v['raw_valid_connected_fixed_chemistry'],raw_rdkit_valid=raw['rdkit_valid'],
                         final_connected=g['connected'],fixed_preserved=raw['fixed_chemistry_preserved'],
                         final_site_overvalent=g['site_neutral_carbon_overvalent'],final_site_neighbors=g['site_neighbors'],
                         raw_posebusters=raw.get('posebusters_valid'),scaffold_changed=raw.get('murcko_scaffold_changed')))
    e.write_csv(out/'comparison.csv',rows)
    headings=list(rows[0]) if rows else []
    cell=lambda v:html.escape(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else str(v))
    table='<table><thead><tr>'+''.join('<th>'+cell(k)+'</th>' for k in headings)+'</tr></thead><tbody>'
    for row in rows:table+='<tr>'+''.join('<td>'+cell(row.get(k))+'</td>' for k in headings)+'</tr>'
    table+='</tbody></table>'
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>连接状态路径实验</title>
<style>body{font:16px/1.65 system-ui;margin:30px;color:#182538}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ccd5de}th{background:#edf3f8}p{max-width:1000px}</style>
<h1>连接状态路径实验</h1><p>只检验一个候选事件：内部连接状态分别经过网络读取、积分更新或两者，是否影响后续连接与最终失败。</p>
<p><b>chemical_endpoint 仅表示原始输出的化学有效、连通与固定片段条件保持。</b>三维质量和骨架变化分别报告；不能把少一条键、暂时不超价或修复后的 SDF 当作整体成功。</p>
<p>network：仅替换网络这一次看到的键状态。integration：网络保持原输入，只替换积分更新读取的当前键状态。both：两处各替换一次。其余生成过程自然续行，随机流匹配。</p>
<p>其他键对照来自同一时刻发生变化的键，单路径修改幅度相同；它们不是独立样本，也不是随机方向重复。reverse 是提前加入同类状态的时机测试，不是健康样本诱发失败实验。</p>
<p>internal_responses.csv 记录干预后的内部响应，数值大不等于贡献大。当前只有一个复合物的一个失败种子，不能宣称唯一根因或跨样本泛化。</p>'''
    text+='<p>运行状态：'+cell(summary)+'</p>'+table+'</html>'
    (out/'REPORT.html').write_text(text,encoding='utf-8')

def bundle(out):
    path=out/'bond_paths_review.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p!=path:z.write(p,p.relative_to(out))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--earlier-dir',type=Path,default=Path('outputs/sh_earlier_connection_ptp1b_v1'))
    ap.add_argument('--causal-dir',type=Path,default=Path('outputs/sh_causal_localization_ptp1b_v2'))
    ap.add_argument('--submodule-dir',type=Path,default=Path('outputs/sh_submodule_localization_ptp1b'))
    ap.add_argument('--readout-dir',type=Path,default=Path('outputs/sh_coordinate_source_ptp1b'))
    ap.add_argument('--output-root',type=Path,default=Path('outputs'))
    ap.add_argument('--plan',type=Path,default=Path(__file__).with_name('bond_paths_plan.json'))
    ap.add_argument('--output-dir',type=Path,default=Path('outputs/sh_bond_state_paths_ptp1b_v1'))
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();out=args.output_dir.resolve()
    if out.exists() and not args.resume:ap.error('Output exists. Preserve it; --resume only for an identical interrupted run, or choose a new --output-dir.')
    if args.resume and not (out/'run_identity.json').is_file():ap.error('No resume identity. Use a new output directory.')
    out.mkdir(parents=True,exist_ok=True);summary=dict(version=VERSION,completed=False,completed_trials=0,error=None)
    results={};trials=[];e=None
    try:
        require((Path.cwd()/'flowr/scriptutil.py').is_file(),'Run in the existing FLOWR.ROOT project.')
        sys.path.insert(0,str(Path.cwd()))
        e=import_module(Path.cwd()/'causal_scaffold_localization.py','bond_engine',ENGINE_SHA)
        o=import_module(Path.cwd()/'trace_connection_rebound.py','bond_observer',OBSERVER_SHA)
        import torch
        from tensordict import TensorDict
        from tensordict._reductions import _make_td
        from flowr.scriptutil import load_model
        from flowr.util.device import resolve_device
        helper=e.import_file(Path.cwd()/'replay_connection_hidden.py','bond_helpers',e.HELPER_SHA)
        quality=e.import_file(Path.cwd()/'evaluate_scaffold_outputs.py','bond_quality')
        print('Verify previous results, recorded inputs, source versions and checkpoint ...',flush=True)
        context=o.load_completed(args.causal_dir);plan=read(args.plan)
        data,records,oldtrials,identity,readout=e.prepare(args,context['identity']['plan'],torch,helper,quality)
        require(identity==context['identity'] and oldtrials==context['planned'],'Original model/data/code identity changed.')
        with torch.serialization.safe_globals([TensorDict,_make_td]):
            states,early_identity=load_earlier(args.earlier_dir,context,torch,e,plan)
        require(early_identity['prior_identity_sha256']==sha(args.causal_dir/'run_identity.json'),'Earlier run uses different original data.')
        require(plan['seed'] in data,'Requested seed has no verified original trajectory.')
        d=data[plan['seed']];inputs=d['inputs'];cfg=d['manifest']['args']
        require(inputs['solver']=='euler' and inputs['sc_recycle_steps']==1 and cfg['corrector_iters'] in [0,None],
                'This adapter requires Euler, one forward per step, and no extra corrector iterations.')
        for k,x in states.items():
            require(x['call']==k,'Capture call identity differs.')
            e.compare_tree(x['input_state'],d['calls'][k]['state_before_forward'],torch,'input')
            e.compare_tree(x['self_condition'],d['calls'][k]['self_condition'],torch,'self_condition')
            require(helper.compare_predictions(x['network_output'],d['calls'][k]['network_output_before_postprocessing'],torch,np,0.,0.)['passed'],
                    'Earlier network prediction differs from original.')
        fixed=d['task']['fixed_sampler_indices'];trials,changes=make_trials(plan,states,fixed)
        capture_calls=plan['capture_calls'];require(capture_calls==sorted(set(capture_calls)) and capture_calls and min(capture_calls)>=0 and max(capture_calls)<len(d['calls'])-1,'Invalid capture window.')
        tracked=sorted(set(fixed)|set(plan['monitor_partners'])|set(plan['focal_pair']))
        require(all(0<=j<inputs['prior']['mask'].shape[1] for j in tracked),'Tracked slot out of range.')
        run_identity=dict(version=VERSION,script_sha256=sha(__file__),plan=plan,original=identity,
                          earlier_identity_sha256=sha(args.earlier_dir/'capture_identity.json'),
                          earlier_trial_sha256={p.parent.name:sha(p) for p in (args.earlier_dir/'trials').glob('*/result.json')})
        if args.resume:require(read(out/'run_identity.json')==run_identity,'Resume identities differ; use a new output directory.')
        else:write(out/'run_identity.json',run_identity)
        write(out/'planned_trials.json',trials);write(out/'observed_bond_changes.json',changes)
        summary.update(planned_trials=len(trials),scope=plan['hypothesis'])
        torch.set_float32_matmul_precision(readout['precision'])
        model_args=argparse.Namespace(**d['manifest']['args']);model_args.save_dir=str(out/'model_context')
        device=resolve_device(model_args)
        print(f'Loading pretrained ROOT on {device}; planned complete rollouts: {len(trials)}',flush=True)
        model=load_model(model_args)[0].to(device).eval()
        require(sum(p.requires_grad for p in model.parameters())==readout['parameter_tensors_requiring_grad'],'Native parameter flags differ.')
        require(bool(model.self_condition)==inputs['self_condition'] and int(model.sc_recycle_steps)==inputs['sc_recycle_steps'] and float(model.coord_scale)==inputs['coord_scale'],'Native model settings differ.')
        gates=[t['id'] for t in trials if t['kind'] in {'baseline','sham'}]
        for n,t in enumerate(trials):
            summary['active_trial']=t['id'];folder=out/'trials'/t['id'];folder.mkdir(parents=True,exist_ok=True)
            done=folder/'result.json'
            if args.resume and done.is_file():
                saved=read(done);require(saved['completed'] and saved['trial']==t,'Resume trial differs.')
                for name,h in saved['file_sha256'].items():
                    p=(folder/name).resolve();require(p.is_relative_to(folder.resolve()) and sha(p)==h,'Resume file changed.')
                results[t['id']]=saved;summary['completed_trials']=len(results)
                print(f'[{n+1}/{len(trials)}] Reuse verified {t["id"]}',flush=True);continue
            if t['kind'] not in {'baseline','sham'}:
                require(all(x in results and results[x]['replay']['baseline_or_sham_full_match'] for x in gates),'Baseline/sham gates did not pass.')
            print(f'[{n+1}/{len(trials)}] {t["id"]}',flush=True);write(folder/'trial.json',t)
            capture=t['kind'] in {'baseline','focal'}
            final,replay,logs,sampler,captures=run_trial(model,d,t,helper,device,torch,e,
                observer_cls=o.ReboundObserver if capture else None,folder=folder,capture_calls=capture_calls,tracked_atoms=tracked)
            torch.save(helper.move_copy(final,'cpu',torch),folder/'final_prediction.pt')
            write(folder/'network_timeline.json',logs);write(folder/'sampler_timeline.json',sampler);write(folder/'replay_checks.json',replay)
            evaluation=e.evaluate_final(final,model,d,t,folder,quality,torch)
            saved=dict(completed=True,trial=t,replay=replay,captures=captures,evaluation=evaluation,
                       file_sha256={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p!=done})
            write(done,saved);results[t['id']]=saved;summary['completed_trials']=len(results)
            write(out/'bond_paths_summary.json',summary)
            print('  Raw chemical endpoint:',evaluation['raw_valid_connected_fixed_chemistry'],flush=True)
            del final
        compare_internal(out,results,e)
        summary.update(completed=len(results)==len(trials),active_trial=None)
        summary['evaluation_errors']=[dict(trial=tid,stage=s['stage'],errors=s['errors'])
            for tid,r in results.items() for s in r['evaluation']['stages'] if s['errors']]
    except Exception as exc:
        summary['completed']=False
        summary['error']=f'{type(exc).__name__}: {exc}';(out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('STOP:',summary['error'],flush=True)
    finally:
        summary['completed_trials']=len(results);write(out/'bond_paths_summary.json',summary)
        if e is not None:
            try:report(out,trials,results,summary,e)
            except Exception:(out/'report_error.txt').write_text(traceback.format_exc())
        bundle(out)
        print(f'Completed: {summary["completed"]}; trials {len(results)}/{len(trials)}',flush=True)
        print('Review file:',out/'bond_paths_review.zip',flush=True)
    return 0 if summary['completed'] else 1

if __name__=='__main__':raise SystemExit(main())
