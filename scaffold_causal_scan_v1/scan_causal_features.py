#!/usr/bin/env python3
"""Bounded fixed-pair activation patching with full native ROOT generation.
Layers are one-based, calls zero-based. No training, forced bond edits, or later
recorded states fed back after patching. Donor and recipient seeds differ;
each treated rollout matches its own unmodified recipient random stream.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np
import native_adapter as native
VERSION='2026-10-22.fixed-pair-causal-grid.1'
HELPER_SHA='1d5b6cfef030f85d1b0df939d96b6402d64f68e9b2a8ef0b746c12236a84ff82'
QUALITY_SHA='97eda26eca03e1f8dcb3d59b8b33430cbb855e87508eeee20f17905a3fe66f75'
INTEGRATOR_SHA='e9abd83167f708da4712db2ac49ab24ac44a39c59c46a54c20789c7873c58000'
require=native.require

def read_plan(path):
    import re
    p=native.read_json(path)
    require(isinstance(p.get('case'),str)and re.fullmatch(r'[A-Za-z0-9_-]+',p['case']),'Invalid case.')
    for key in ['recipient_seed','site','partner','target_bond_class','reference_bond_class','random_seed']:
        require(type(p.get(key))is int and p[key]>=0,f'Invalid {key}.')
    require(p['site']!=p['partner'],'Self-pairs unsupported.')
    require(p['target_bond_class']in range(5)and p['reference_bond_class']in range(5)and p['target_bond_class']!=p['reference_bond_class'],'Invalid contrast.')
    for key in ['donor_seeds','calls','layers']:
        v=p.get(key);require(isinstance(v,list)and v and all(type(x)is int and x>=(1 if key=='layers'else 0)for x in v)and len(set(v))==len(v),f'Invalid {key}.')
    require(p['recipient_seed']not in p['donor_seeds'],'Recipient cannot donate to itself.')
    require(p.get('components')==['pairs','nodes'],'Scan both pairs and nodes.')
    require(p.get('random_controls_per_target')==1,'This screening protocol uses one norm-matched control per target.')
    return p

def plan_trials(p):
    focus={k:p[k]for k in ['site','partner','target_bond_class','reference_bond_class']}
    ts=[dict(id=f'baseline_s{s}',kind='baseline',seed=s,**focus)for s in [p['recipient_seed'],*p['donor_seeds']]]
    for component in p['components']:
        ts.append(dict(id=f'sham_{component}',kind='sham',seed=p['recipient_seed'],call=max(p['calls']),layer=max(p['layers']),component=component,**focus))
    for call in p['calls']:
        for layer in p['layers']:
            for component in p['components']:
                for donor in p['donor_seeds']:
                    base=dict(seed=p['recipient_seed'],donor_seed=donor,call=call,layer=layer,component=component,**focus)
                    prefix=f'c{call:03d}_L{layer:02d}_{component}_d{donor}'
                    for kind in ['targeted','random_direction']:ts.append(dict(id=prefix+'_'+kind,kind=kind,**base))
    require(len({t['id']for t in ts})==len(ts),'Duplicate trials.')
    return ts

def cache_key(call,layer,component):return f'c{call:04d}_L{layer:02d}_{component}'

def layer_tensor(args,output,component):
    require(isinstance(output,(tuple,list))and len(output)==5,'Unexpected native layer contract.')
    value=output[1]if component=='nodes'else (output[3]if output[3]is not None else args[2])
    require(value.ndim==(3 if component=='nodes'else 4)and value.shape[0]==1,'Unexpected feature shape.')
    return value

def select_features(value,component,site,partner):
    if component=='nodes':return np.stack([native.array(value[0,site]),native.array(value[0,partner])])
    return np.stack([native.array(value[0,site,partner]),native.array(value[0,partner,site])])

class BaselineCapture:
    def __init__(self,model,plan,clock):
        self.plan,self.clock=plan,clock;self.arrays={};self.handles=[];self.receipt=None
        for layer in plan['layers']:self.handles.append(model.gen.ligand_dec.layers[layer-1].register_forward_hook(self.hook(layer)))
    def hook(self,layer):
        def fn(module,args,output):
            call=self.clock['call']
            if call not in self.plan['calls']:return None
            for component in self.plan['components']:
                key=cache_key(call,layer,component);require(key not in self.arrays,'Repeated baseline capture.')
                value=select_features(layer_tensor(args,output,component),component,self.plan['site'],self.plan['partner'])
                require(value.ndim==2 and value.shape[0]==2 and np.isfinite(value).all(),'Invalid baseline features.')
                self.arrays[key]=value
            return None
        return fn
    def validate_complete(self):
        expected={cache_key(c,l,k)for c in self.plan['calls']for l in self.plan['layers']for k in self.plan['components']}
        require(set(self.arrays)==expected,'Incomplete baseline cache.')
        self.receipt=dict(read_only_capture=True,n_feature_arrays=len(self.arrays))
    def close(self):
        for h in self.handles:h.remove()
        self.handles=[]

class FeaturePatch:
    """Only two mapped scalar node vectors or two directed pair vectors change."""
    def __init__(self,model,trial,clock,reference,desired,torch):
        self.trial,self.clock,self.reference,self.desired,self.torch=trial,clock,reference,desired,torch
        self.visits=0;self.receipt=None
        self.handle=model.gen.ligand_dec.layers[trial['layer']-1].register_forward_hook(self.hook)
    def hook(self,module,args,output):
        t=self.trial
        if self.clock['call']!=t['call']:return None
        self.visits+=1;require(self.visits==1,'Patch executed twice.')
        value=layer_tensor(args,output,t['component']);before=select_features(value,t['component'],t['site'],t['partner'])
        require(before.dtype==np.float32 and np.array_equal(before,self.reference),'Pre-intervention features differ from native baseline.')
        require(self.desired.shape==before.shape and np.isfinite(self.desired).all(),'Invalid desired features.')
        desired=self.desired.astype(before.dtype);delta=desired.astype(float)-before.astype(float)
        if t['kind']=='sham':
            require(np.array_equal(desired,before),'Sham is not identity.')
            self.receipt=dict(applied=False,visits=1,delta_norm=0.,prewrite_exact=True);return None
        changed=value.clone();site,partner=t['site'],t['partner']
        indices=[(0,site),(0,partner)]if t['component']=='nodes'else [(0,site,partner),(0,partner,site)]
        for i,selection in enumerate(indices):changed[selection]=self.torch.as_tensor(desired[i],device=value.device,dtype=value.dtype)
        a,b=native.array(value),native.array(changed);allowed=np.zeros(a.shape[:-1],bool)
        for selection in indices:allowed[selection]=True
        require(np.array_equal(a[~allowed],b[~allowed])and np.isfinite(b).all(),'Unselected/nonfinite changes.')
        require(np.array_equal(select_features(changed,t['component'],site,partner),desired),'Replacement differs in native dtype.')
        self.receipt=dict(applied=True,visits=1,prewrite_exact=True,component=t['component'],delta_norm=float(np.linalg.norm(delta)),
                          reference_norm=float(np.linalg.norm(before.astype(float))),selected_slots=[site,partner],only_selected_feature_entries_changed=True)
        result=list(output);result[1 if t['component']=='nodes'else 3]=changed
        return tuple(result)if isinstance(output,tuple)else result
    def validate_complete(self):require(self.visits==1 and self.receipt is not None,'Patch did not execute once.')
    def close(self):self.handle.remove()

def desired_features(t,caches,plan):
    key=cache_key(t['call'],t['layer'],t['component']);reference=caches[t['seed']][key]
    if t['kind']=='sham':return reference,reference.copy(),dict(control='identity')
    donor=caches[t['donor_seed']][key];require(donor.shape==reference.shape,'Donor feature width differs.')
    delta=donor.astype(float)-reference.astype(float)
    if t['kind']=='targeted':desired=donor.copy();info=dict(control='mapped_fixed_pair_donor')
    else:
        h=hashlib.sha256((str(plan['random_seed'])+':'+t['id']).encode()).digest();seed=int.from_bytes(h[:8],'little');rng=np.random.default_rng(seed)
        order=rng.permutation(delta.shape[-1]);sign=rng.choice([-1.,1.],size=delta.shape[-1]);random_delta=delta[:,order]*sign
        require(np.isclose(np.linalg.norm(random_delta),np.linalg.norm(delta),atol=1e-12,rtol=1e-12),'Norm-matched control failed.')
        desired=(reference.astype(float)+random_delta).astype(reference.dtype);info=dict(control='shared_channel_signed_permutation',control_seed=seed)
    intended=float(np.linalg.norm(delta));actual=float(np.linalg.norm(desired.astype(float)-reference.astype(float)))
    bound=8*np.finfo(reference.dtype).eps*(float(np.linalg.norm(reference.astype(float)))+intended+1.)
    require(abs(actual-intended)<=bound,'Rounding broke norm matching.')
    info.update(donor_delta_norm=intended,actual_delta_norm=actual,zero_donor_delta=bool(intended==0.),norm_rounding_bound=bound)
    return reference,desired,info

def prepare(args,plan,helpers,quality,torch):
    from rdkit import Chem,rdBase
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from flowr.data.interpolate import extract_scaffolds
    head=native.read_json(args.head_dir/'summary.json')
    require(head['capture_completed']and head['analysis_completed']and head['error']is None,'Incomplete head capture.')
    require(head['plan']['case']==plan['case']and head['plan']['site']==plan['site'],'Analysis case/site differs.')
    require(native.digest(args.head_dir/'observer_source.py')==head['script_sha256'],'Head observer source differs.')
    for e in head['states']:require(e['accepted']and e['numerics']['passed']and native.digest(args.head_dir/e['file'])==e['sha256'],'Unverified/changed head capture.')
    sources={**head['source_sha256'],**quality.SOURCE_HASHES,'flowr/models/integrator.py':INTEGRATOR_SHA}
    for name,sha in sources.items():require(native.digest(Path.cwd()/name)==sha,f'Native source changed: {name}')
    data={};load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
    for seed in [plan['recipient_seed'],*plan['donor_seeds']]:
        folder=args.output_root/f"sh_record_{plan['case']}_seed{seed}/research_record/attempt_000"
        manifest=native.read_json(folder.parent/'manifest.json');task=native.read_json(folder/'task_definition.json')
        require(str(seed)in head['metadata']and manifest==head['metadata'][str(seed)]['manifest'],'Recorded generation manifest differs.')
        require(task==head['metadata'][str(seed)]['task']and task['mapping_verified'],'Fixed mapping differs.')
        require(manifest['torch_version']==str(torch.__version__)and manifest['rdkit_version']==rdBase.rdkitVersion,'Use original Torch/RDKit.')
        for name,sha in manifest['input_sha256'].items():require(native.digest(Path(name))==sha,f'Input changed: {name}')
        cfg=manifest['args'];inputs=load(folder/'inputs.pt');attempt=native.read_json(folder/'attempt.json')
        require(cfg['scaffold_hopping']and attempt['generation_completed'],'Incomplete scaffold-hopping record.')
        require(cfg['guidance_config']is None and not cfg.get('optimize_gen_ligs')and not cfg.get('optimize_gen_ligs_hs'),'Guidance/optimization unsupported.')
        require(inputs['solver']==cfg['solver']and inputs['strategy']==cfg['ode_sampling_strategy']and inputs['final_inpaint']==cfg['final_inpaint'],'Sampler settings differ.')
        require(inputs['prior']['mask'].shape[0]==1,'Expected batch-one records.')
        valid=native.array(inputs['prior']['mask'][0]).astype(bool)
        require(all(i in task['fixed_sampler_indices']and 0<=i<len(valid)and valid[i]for i in [plan['site'],plan['partner']]),'Two mapped fixed atoms required; generated slots cannot be aligned by index.')
        reference=quality.read_one_sdf(folder/'reference_preprocessed.sdf',Chem);require(reference is not None,'Missing reference.');Chem.SanitizeMol(reference)
        require(extract_scaffolds([reference],invert_mask=True)[0].nonzero(as_tuple=True)[0].tolist()==task['fixed_reference_indices'],'Native fixed-fragment extraction differs.')
        paths=sorted((folder/'forward_calls').glob('call_*.pt'));require(len(paths)==attempt['forward_calls'],'Missing calls.');calls=[load(p)for p in paths]
        require([c['call_index']for c in calls]==list(range(len(calls)))and max(plan['calls'])<len(calls),'Invalid call indices.')
        identity_files=[folder/'inputs.pt',folder/'pocket_encoding.pt',folder/'raw_final_prediction.pt',folder/'task_definition.json',folder/'reference_preprocessed.sdf',*paths]
        data[seed]=dict(folder=folder,manifest=manifest,task=task,inputs=inputs,calls=calls,encoding=load(folder/'pocket_encoding.pt'),final=load(folder/'raw_final_prediction.pt'),reference=reference,
                        reference_scaffold=Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(reference),isomericSmiles=False),input_hashes={str(p.relative_to(folder)):native.digest(p)for p in identity_files})
    base=data[plan['recipient_seed']]
    for d in data.values():
        require(helpers.condition_key(d['manifest'])==helpers.condition_key(base['manifest']),'Donor conditions differ beyond seed/output.')
        for key in ['fixed_sampler_indices','fixed_reference_indices']:require(d['task'][key]==base['task'][key],'Fixed identities differ.')
    require(native.digest(Path(base['manifest']['args']['ckpt_path']))==head['checkpoint_sha256'],'Checkpoint changed.')
    identity=dict(version=VERSION,plan=plan,head_summary_sha256=native.digest(args.head_dir/'summary.json'),source_sha256=sources,checkpoint_sha256=head['checkpoint_sha256'],
                  code_sha256={p.name:native.digest(p)for p in [Path(__file__),Path(__file__).with_name('native_adapter.py'),Path(__file__).with_name('evaluate_scaffold_outputs.py')]},helper_sha256=HELPER_SHA,input_sha256={str(s):d['input_hashes']for s,d in data.items()})
    return data,head,identity

def summarize(out,trials,results,summary,plot=False):
    rows=[]
    for t in trials:
        if t['id']not in results:continue
        r=results[t['id']];e=r['evaluation'];raw=e['stages'][0]
        rows.append(dict(trial=t['id'],kind=t['kind'],seed=t['seed'],donor_seed=t.get('donor_seed'),call=t.get('call'),layer=t.get('layer'),component=t.get('component'),
            raw_chemical_endpoint=e['raw_valid_connected_root_fragment'],root_fragment_match=raw['root_fixed_substructure_match'],slot_fixed_chemistry_preserved=raw['fixed_chemistry_preserved'],raw_valid_connected=raw['fc_valid'],raw_posebusters_valid=raw['posebusters_valid'],raw_scaffold_changed=raw.get('murcko_scaffold_changed'),
            pair_final_class=e['final_graph']['bond_class'],site_final_bond_order_sum=e['final_graph']['site_explicit_bond_order_sum'],local_margin_change=r.get('local_margin_change'),delta_norm=(r['replay'].get('patch')or {}).get('delta_norm'),
            evaluation_errors=json.dumps([s['errors']for s in e['stages']]),elapsed_seconds=r.get('elapsed_seconds')))
    if rows:native.write_csv(out/'scan_results.csv',rows)
    if plot and any(r['kind']=='targeted'for r in rows):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plan=summary['plan'];fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
        for i,component in enumerate(plan['components']):
            for j,kind in enumerate(['targeted','random_direction']):
                matrix=np.full((len(plan['layers']),len(plan['calls'])),np.nan)
                for li,layer in enumerate(plan['layers']):
                    for ci,call in enumerate(plan['calls']):
                        cell=[r for r in rows if r['kind']==kind and r['component']==component and r['layer']==layer and r['call']==call]
                        if len(cell)==len(plan['donor_seeds'])and all(r['raw_chemical_endpoint']is not None for r in cell):matrix[li,ci]=np.mean([r['raw_chemical_endpoint']for r in cell])
                ax=axes[i,j];ax.imshow(np.ma.masked_invalid(matrix),vmin=0,vmax=1,cmap='Blues')
                ax.set(xticks=range(len(plan['calls'])),xticklabels=plan['calls'],yticks=range(len(plan['layers'])),yticklabels=plan['layers'],xlabel='Call (zero-based)',ylabel='Layer (one-based)',title=f'{component}: {kind}')
                for (li,ci),v in np.ndenumerate(matrix):
                    if np.isfinite(v):ax.text(ci,li,f'{round(v*len(plan["donor_seeds"]))}/{len(plan["donor_seeds"])}',ha='center',va='center',color='white'if v>.6 else 'black')
        fig.suptitle('Valid + connected + native fixed-fragment match\nDonor counts in one recipient; not independent-sample success rates')
        fig.savefig(out/'scan_map.png',dpi=150);plt.close(fig)
    text=f"""# 因果初筛进度\n\n完整结束：{summary.get('completed',False)}；完成 {len(results)}/{len(trials)} 次。\n\n错误：{summary.get('error')}\n\n查看 scan_results.csv；有干预结果时生成 scan_map.png。化学终态是原始解码有效、连通、通过 ROOT 固定片段匹配。PoseBusters 和骨架变化单列，不等于整体任务成功。局部得分改善不等于最终修复。\n\n本轮只有一个失败输入/种子、三个表征参考轨迹。3/3 指三个供体均取得该终态，不是三个独立失败样本。供体和研究轨迹初始噪声不同；每次干预与其原始失败基线使用相同随机流。随机对照检验特定方向是否优于同等幅度改变，不能保证分布内干预。\n\n这是选定时间和层边界的粗扫描，未覆盖所有层、所有时刻、全部模块或组合干预。无修复不能证明表征无因果作用；有修复也不证明该层最早产生错误或是唯一根因。后续需要独立输入验证。\n"""
    (out/'RESULTS_CN.md').write_text(text,encoding='utf-8')

def bundle(out):
    with zipfile.ZipFile(out/'causal_scan_review.zip','w',zipfile.ZIP_DEFLATED)as z:
        for p in sorted(out.rglob('*')):
            if p.is_file()and p.name!='causal_scan_review.zip'and p.suffix not in {'.pt','.tmp'}and 'model_context'not in p.relative_to(out).parts:z.write(p,p.relative_to(out))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,default=Path(__file__).with_name('bace_scan_plan.json'))
    ap.add_argument('--head-dir',type=Path,default=Path('outputs/sh_head_bace_lig16_v3'))
    ap.add_argument('--output-root',type=Path,default=Path('outputs/sh_boundary_screen_v1/bace_lig16'))
    ap.add_argument('--output-dir',type=Path,default=Path('outputs/sh_causal_scan_bace_lig16_v1'))
    ap.add_argument('--resume',action='store_true');ap.add_argument('--list-plan',action='store_true')
    args=ap.parse_args();plan=read_plan(args.plan);trials=plan_trials(plan)
    if args.list_plan:
        from collections import Counter
        print(json.dumps(dict(total=len(trials),counts=dict(Counter(t['kind']for t in trials))),indent=2));return 0
    out=args.output_dir.resolve()
    if out.exists()and not args.resume:ap.error('Preserve existing output; use --resume or a new output directory.')
    if args.resume and not (out/'run_identity.json').is_file():ap.error('No verified resume identity; preserve this output and choose a new directory.')
    out.mkdir(parents=True,exist_ok=True);results={};caches={};summary=dict(version=VERSION,plan=plan,planned_trials=len(trials),completed=False,error=None)
    try:
        require((Path.cwd()/'flowr/scriptutil.py').is_file(),'Run from the original FLOWR.ROOT project/environment.')
        sys.path.insert(0,str(Path.cwd()))
        import torch
        from flowr.scriptutil import load_model
        from flowr.util.device import resolve_device
        helpers=native.import_file(Path.cwd()/'replay_connection_hidden.py','scan_replay_helpers',HELPER_SHA)
        quality=native.import_file(Path(__file__).with_name('evaluate_scaffold_outputs.py'),'scan_quality_helpers',QUALITY_SHA)
        print('Checking recorded data and native source identity ...',flush=True)
        data,head,identity=prepare(args,plan,helpers,quality,torch)
        if args.resume:require(native.read_json(out/'run_identity.json')==identity,'Resume identity differs; use a new output directory.')
        else:native.write_json(out/'run_identity.json',identity)
        for name in ['scan_causal_features.py','native_adapter.py','evaluate_scaffold_outputs.py']:(out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
        native.write_json(out/'planned_trials.json',trials);native.write_json(out/'scan_summary.json',summary)
        precision={e['numerics']['original_precision']for e in head['states']};require(len(precision)==1,'Mixed native precision modes.')
        torch.set_float32_matmul_precision(next(iter(precision)))
        ma=argparse.Namespace(**data[plan['recipient_seed']]['manifest']['args']);ma.save_dir=str(out/'model_context');device=resolve_device(ma)
        print(f'Loading frozen ROOT; {len(trials)} complete native rollouts planned.',flush=True)
        model=load_model(ma)[0].to(device).eval()
        require(sum(p.requires_grad for p in model.parameters())==head['parameter_tensors_requiring_grad'],'Native parameter flags changed.')
        require(max(plan['layers'])<=len(model.gen.ligand_dec.layers),'Requested layer absent.')
        for d in data.values():
            require(bool(model.self_condition)==d['inputs']['self_condition']and int(model.sc_recycle_steps)==d['inputs']['sc_recycle_steps'],'Self-conditioning differs.')
            require(float(model.coord_scale)==d['inputs']['coord_scale'],'Coordinate units differ.')
        for n,t in enumerate(trials):
            summary['active_trial']=t['id'];native.write_json(out/'scan_summary.json',summary)
            folder=out/'trials'/t['id'];folder.mkdir(parents=True,exist_ok=True);done=folder/'result.json'
            if args.resume and done.is_file():
                saved=native.read_json(done);require(saved['completed']and saved['trial']==t,'Resume trial differs.')
                for name,sha in saved['file_sha256'].items():require(native.digest(folder/name)==sha,f'Resume output changed: {name}')
                results[t['id']]=saved
                if t['kind']=='baseline':
                    with np.load(folder/'baseline_features.npz',allow_pickle=False)as a:caches[t['seed']]={k:a[k].copy()for k in a.files}
                print(f'[{n+1}/{len(trials)}] Reuse {t["id"]}',flush=True);continue
            if t['kind']!='baseline':
                require(all(f'baseline_s{s}'in results and results[f'baseline_s{s}']['replay']['baseline_or_sham_full_match']for s in [plan['recipient_seed'],*plan['donor_seeds']]),'Baseline gate incomplete.')
                require(results[f'baseline_s{plan["recipient_seed"]}']['evaluation']['raw_valid_connected_root_fragment']is False,'Recipient no longer has the defined failure.')
                require(all(results[f'baseline_s{s}']['evaluation']['raw_valid_connected_root_fragment']is True for s in plan['donor_seeds']),'Donor is not a chemical fixed-fragment control.')
            if t['kind']not in {'baseline','sham'}:require(all('sham_'+c in results and results['sham_'+c]['replay']['baseline_or_sham_full_match']for c in plan['components']),'Sham gate incomplete.')
            print(f'[{n+1}/{len(trials)}] {t["id"]}',flush=True);start=time.monotonic();captured={};native.write_json(folder/'trial.json',t)
            if t['kind']=='baseline':
                def factory(model,trial,clock,torch):
                    cap=BaselineCapture(model,plan,clock);captured['observer']=cap;return cap
                patch_info={}
            else:
                reference,desired,patch_info=desired_features(t,caches,plan)
                np.savez_compressed(folder/'patch_features.npz',reference=reference,desired=desired)
                def factory(model,trial,clock,torch):return FeaturePatch(model,trial,clock,reference,desired,torch)
            final,replay,logs,sampler=native.run_rollout(model,data[t['seed']],t,factory,helpers,device,torch,debug_dir=folder)
            if t['kind']=='baseline':
                caches[t['seed']]=captured['observer'].arrays;np.savez_compressed(folder/'baseline_features.npz',**caches[t['seed']])
            torch.save(helpers.move_copy(final,'cpu',torch),folder/'final_prediction.pt')
            native.write_json(folder/'network_timeline.json',logs);native.write_json(folder/'sampler_timeline.json',sampler);native.write_json(folder/'replay_checks.json',replay)
            evaluation=native.evaluate_final(final,model,data[t['seed']],t,folder,quality,torch)
            local=None
            if t['kind']!='baseline':
                base_logs=native.read_json(out/'trials'/f'baseline_s{t["seed"]}'/'network_timeline.json')
                current,baseline=logs[t['call']]['pair_margin_value'],base_logs[t['call']]['pair_margin_value']
                local=current-baseline if current is not None and baseline is not None else None
            saved=dict(completed=True,trial=t,replay=replay,evaluation=evaluation,patch_design=patch_info,local_margin_change=local,
                elapsed_seconds=round(time.monotonic()-start,3),file_sha256={p.name:native.digest(p)for p in folder.iterdir()if p.is_file()and p.name!='result.json'})
            native.write_json(done,saved);results[t['id']]=saved;summary['completed_trials']=len(results)
            native.write_json(out/'scan_summary.json',summary);summarize(out,trials,results,summary)
            print(f'  {saved["elapsed_seconds"]:.1f}s; chemical endpoint: {evaluation["raw_valid_connected_root_fragment"]}',flush=True);del final
        summary.update(completed=len(results)==len(trials),active_trial=None)
        summary['evaluation_errors']=[dict(trial=k,stage=s['stage'],errors=s['errors'])for k,r in results.items()for s in r['evaluation']['stages']if s['errors']]
    except Exception as exc:
        summary['error']=f'{type(exc).__name__}: {exc}';(out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8');print('STOP: '+summary['error'],flush=True)
    finally:
        summary['completed_trials']=len(results);native.write_json(out/'scan_summary.json',summary)
        try:summarize(out,trials,results,summary,plot=bool(results))
        except Exception:(out/'report_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        bundle(out);print(f'Completed: {summary["completed"]}; {len(results)}/{len(trials)}',flush=True);print(f'Review file: {out/"causal_scan_review.zip"}',flush=True)
    return 0 if summary['completed']else 1

if __name__=='__main__':raise SystemExit(main())
