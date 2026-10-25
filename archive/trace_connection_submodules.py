#!/usr/bin/env python3
"""Read-only, all-layer submodule localization on recorded ROOT trajectories.

Default: validate prior records, replay with observation hooks, analyze, bundle.
--analyze-only: use this script's completed capture without loading the model.
This is observational localization, not activation patching or input-atom SHAP.
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import html
import sys
import traceback
import zipfile
from pathlib import Path

import numpy as np

VERSION = '2026-10-20.submodules.1'
HEAD_SHA = 'a157c4d133b88db754d7c9e892d7711f1eb2bace793a5865692e863cead9b17a'
ANALYSIS_SHA = 'eaf1bf202a1a3f7abcfeb5a8e95067c7c2a6ad056cab9eab2f1909f95599d968'
MODULES = ['self_attention', 'conditional_attention', 'equivariant_feedforward',
           'coordinate_reset', 'residual_addition_rounding']
MODULE_LABELS = {'self_attention':'配体自注意力', 'conditional_attention':'口袋条件注意力',
    'equivariant_feedforward':'三维隐藏向量前馈更新', 'coordinate_reset':'中间坐标更新后的隐藏重置',
    'residual_addition_rounding':'残差加法舍入', 'initial_hidden':'第一层入口的隐藏表示'}


def require(ok, message):
    if not bool(ok):raise ValueError(message)


def import_file(path, name, expected=None):
    import hashlib
    path=Path(path).resolve()
    require(path.is_file(), f'Missing dependency: {path.name}')
    if expected:require(hashlib.sha256(path.read_bytes()).hexdigest()==expected,
                        f'Use the supplied verified version of {path.name}.')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def layer_layout(decoder):
    layout=[]
    for i,layer in enumerate(decoder.layers):
        require(not layer.fixed_equi, 'This observer requires updating equivariant features.')
        cond=layer.d_inv_cond is not None
        reset=bool(layer.intermediate_coord_updates)
        names=['self_attention','ff_equi_norm','equi_ff']
        if cond:names+=['cond_attention','cond_attn_equi_norm']
        if reset:names+=['coord_down_proj','coord_up_proj']
        require(all(hasattr(layer,n) for n in names), f'Unsupported layer contract at layer {i+1}.')
        layout.append(dict(index=i,number=i+1,has_condition=cond,coordinate_reset=reset,
                           modules={n:type(getattr(layer,n)).__name__ for n in names}))
    require(layout, 'No ligand layers.')
    return layout


class SubmoduleObserver:
    """Copy module writes and raw residual boundaries; never replace a tensor."""
    def __init__(self,decoder,to_array,layout):
        self.arrays,self.handles,self.visits={},{},{}
        self.to_array=to_array
        try:
            for config in layout:
                i=config['index'];layer=decoder.layers[i];prefix=f'layer_{i:02d}__'
                self.add(layer.self_attention,prefix+'self_attention_update',tuple_output=True)
                if config['has_condition']:
                    self.add(layer.cond_attn_equi_norm,prefix+'before_condition',pre=True)
                    self.add(layer.cond_attention,prefix+'conditional_attention_update',tuple_output=True)
                self.add(layer.ff_equi_norm,prefix+'before_feedforward',pre=True)
                self.add(layer.equi_ff,prefix+'equivariant_feedforward_update')
                if config['coordinate_reset']:
                    self.add(layer.coord_down_proj,prefix+'before_coordinate_reset',pre=True)
                    self.add(layer.coord_down_proj,prefix+'coordinate_delta')
                    self.add(layer.coord_up_proj,prefix+'coordinate_up_input',pre=True)
        except Exception:
            self.close();raise

    def add(self,module,name,pre=False,tuple_output=False):
        self.visits[name]=0
        def hook(*args):
            self.visits[name]+=1
            require(self.visits[name]==1,f'Repeated invocation at {name}; adapter needs review.')
            value=args[1][0] if pre else args[2]
            if tuple_output:
                require(isinstance(value,(tuple,list)) and len(value)==3, f'Unexpected output at {name}.')
                value=value[0]
            require(value is not None and value.ndim>=3 and value.shape[0]==1,
                    f'Expected batch-one vector features at {name}.')
            self.arrays[name]=self.to_array(value[0])
            return None
        register=module.register_forward_pre_hook if pre else module.register_forward_hook
        self.handles[name]=register(hook)

    def close(self):
        for h in self.handles.values():h.remove()
        self.handles.clear()


def addition_rounding(before, update, after, name):
    """Bound a float32 addition error without relaxing native replay checks."""
    a,b,c=[np.asarray(v,dtype=np.float64) for v in [before,update,after]]
    residual=c-a-b
    bound=4*np.finfo(np.float32).eps*(abs(a)+abs(b))+np.finfo(np.float32).tiny
    require(np.isfinite(residual).all() and np.all(abs(residual)<=bound),
            f'Residual boundary does not match the recorded module update: {name}.')
    return residual


def branch_features(record,full,layout):
    """Actual module writes + resets + explicit addition roundoff = net E update."""
    expected_keys=set();components={};errors={}
    shape=full['final_norm_input'].shape
    require(len(shape)==3 and shape[1]==3, 'Unexpected equivariant feature axes.')
    for item in layout:
        i=item['index'];p=f'layer_{i:02d}__'
        def get(suffix,required_shape=shape):
            key=p+suffix;expected_keys.add(key)
            require(key in record, f'Missing submodule tensor {key}.')
            value=record[key]
            require(value.shape==required_shape and np.isfinite(value).all(), f'Invalid tensor {key}.')
            require(value.dtype==np.float32, f'Expected native float32 capture: {key}.')
            return value.astype(np.float64)
        e0=full[p+'in__equis'].astype(np.float64)
        e4=full[p+'out__equis'].astype(np.float64)
        e2=get('before_feedforward')
        e1=get('before_condition') if item['has_condition'] else e2
        sa=get('self_attention_update')
        ca=get('conditional_attention_update') if item['has_condition'] else np.zeros_like(e0)
        ff=get('equivariant_feedforward_update')
        e3=get('before_coordinate_reset') if item['coordinate_reset'] else e4
        r1=addition_rounding(e0,sa,e1,p+'self_attention')
        r2=addition_rounding(e1,ca,e2,p+'conditional_attention')
        r3=addition_rounding(e2,ff,e3,p+'feedforward')
        reset=e4-e3
        if item['coordinate_reset']:
            delta=get('coordinate_delta',(shape[0],3,1))[...,0]
            up_input=get('coordinate_up_input',(shape[0],3,1))[...,0]
            coords_before=full[p+'in__coords'].astype(np.float64)
            coords_after=full[p+'out__coords'].astype(np.float64)
            require(np.array_equal(up_input,coords_after), f'Coordinate reset input mismatch at layer {i+1}.')
            mask=full['output_mask'].astype(np.float64)[:,None]
            addition_rounding(coords_before*mask,delta*mask,coords_after,p+'coordinate_addition')
        else:
            require(np.array_equal(full[p+'in__coords'],full[p+'out__coords']), 'Unexpected coordinate update.')
        vals=[sa,ca,ff,reset,r1+r2+r3]
        require(np.allclose(sum(vals),e4-e0,atol=1e-11,rtol=1e-10), 'Submodule-to-layer identity failed.')
        for name,value in zip(MODULES,vals):components[(i+1,name)]=value
        errors[i+1]=float(np.max(abs(r1+r2+r3)))
    require(set(record)==expected_keys, 'Unexpected/unmapped submodule tensors; adapter needs review.')
    total=full['layer_00__in__equis'].astype(np.float64)+sum(components.values())
    require(np.allclose(total,full['final_norm_input'],atol=1e-10,rtol=1e-10),
            'Submodules do not reconstruct final hidden features.')
    return components, errors


def projected_changes(a,b,full_a,full_b,ref_a,ref_b,layout):
    """Allocate only the hidden-vector term of the existing symmetric readout ledger."""
    ca,_=branch_features(a,full_a,layout);cb,_=branch_features(b,full_b,layout)
    require(np.array_equal(ref_a['mask'],ref_b['mask']), 'Mask changed across calls.')
    scale=(ref_a['scale']+ref_b['scale'])/2
    mask=ref_a['mask'][:,None]
    project=lambda delta:(delta*scale).sum(-1)*mask
    out={key:project(cb[key]-value) for key,value in ca.items()}
    out[(0,'initial_hidden')]=project(full_b['layer_00__in__equis'].astype(np.float64)-
                                     full_a['layer_00__in__equis'].astype(np.float64))
    target=project(ref_b['e']-ref_a['e'])
    require(np.allclose(sum(out.values()),target,atol=1e-10,rtol=1e-9),
            'Submodule projection does not match the total hidden-vector term.')
    return out


def allocate_pair(projected,ref_a,ref_b,site,partner,scale):
    require(site!=partner and ref_a['mask'][site]==ref_a['mask'][partner]==1, 'Invalid pair.')
    direction=((ref_b['output'][partner]-ref_b['output'][site])+
               (ref_a['output'][partner]-ref_a['output'][site]))*scale
    rows=[]
    for (layer,name),vectors in projected.items():
        fixed=-float(direction@vectors[site])*scale
        generated=float(direction@vectors[partner])*scale
        rows.append(dict(layer=layer,module=name,term_A2=fixed+generated,
                         fixed_endpoint_term_A2=fixed,generated_endpoint_term_A2=generated))
    return rows


def prepare_inputs(args,ref,obs):
    hd,rd,old=args.head_dir.resolve(),args.readout_dir.resolve(),args.hidden_dir.resolve()
    h=obs.read_json(hd/'summary.json');r=obs.read_json(rd/'readout_capture_summary.json')
    prior=obs.read_json(old/'replay_summary.json')
    require(h['capture_completed'] and h['analysis_completed'] and h['error'] is None, 'Head analysis is incomplete.')
    require(r['completed'] and r['error'] is None and prior['all_requested_calls_verified'] and prior['error'] is None,
            'Earlier readout/hidden capture is incomplete.')
    require(obs.digest(hd/'summary.json')==r['head_summary_sha256'], 'Readout belongs to a different head capture.')
    require(obs.digest(old/'replay_summary.json')==h['previous_summary_sha256'], 'Earlier hidden summary differs.')
    require(h['script_sha256']==HEAD_SHA and r['plan']==h['plan'] and r['metadata']==h['metadata'], 'Analysis plan/metadata mismatch.')
    require(r['checkpoint_sha256']==h['checkpoint_sha256']==prior['checkpoint_sha256'], 'Checkpoint mismatch.')
    require(r['source_sha256']==h['source_sha256'], 'Model source identities differ.')
    require(obs.digest(rd/'coordinate_readout_weights.npz')==r['weights_sha256'] and
            obs.digest(hd/'head_weights.npz')==h['weights_sha256'], 'Exported weights changed.')
    plan=r['plan'];seeds,calls=plan['seeds'],plan['calls']
    require(len(seeds)==len(set(seeds)) and len(calls)>1 and all(y-x==1 for x,y in zip(calls,calls[1:])), 'Invalid seed/time plan.')
    wanted={(s,c) for s in seeds for c in calls}
    for report in [h,r]:
        require(len(report['states'])==len(wanted) and {(e['seed'],e['call']) for e in report['states']}==wanted,
                'Missing or duplicated states.')
    he={(e['seed'],e['call']):e for e in h['states']}
    hw={k:v.astype(np.float64) for k,v in ref.load_npz(hd/'head_weights.npz').items()}
    rw=ref.load_npz(rd/'coordinate_readout_weights.npz');cache={}
    for entry in r['states']:
        key=(entry['seed'],entry['call']);ee=he[key]
        require(entry['accepted'] and ee['accepted'] and ee['numerics']['passed'], 'Unaccepted source state.')
        for e in [entry,ee]:
            require(all(e[n]['passed'] for n in ['replay_vs_original','observation_vs_original','observation_vs_replay']),
                    'Source replay consistency check failed.')
        require(entry['times']==ee['times'], 'Recorded times mismatch.')
        require(obs.digest(rd/entry['file'])==entry['sha256'] and obs.digest(hd/ee['file'])==ee['sha256'], 'Source NPZ hash mismatch.')
        full,head=ref.load_npz(rd/entry['file']),ref.load_npz(hd/ee['file'])
        meta=obs.read_json(old/f'seed_{key[0]}/metadata.json')
        require(meta['task_definition']==r['metadata'][str(key[0])]['task'] and
                meta['recorded_manifest']==r['metadata'][str(key[0])]['manifest'] and
                meta['coord_scale']==r['metadata'][str(key[0])]['coord_scale'], 'Earlier task mapping mismatch.')
        obs.check_readout_arrays(full,r['n_layers'],meta['tracked_atom_indices'],
            ref.load_npz(old/f'seed_{key[0]}/call_{key[1]:04d}_hidden.npz'),head)
        require(full['all_sampler_indices'].tolist()==list(range(len(full['output_mask']))), 'Full-node mapping mismatch.')
        require(full['fixed_sampler_indices'].tolist()==meta['task_definition']['fixed_sampler_indices'], 'Fixed mapping mismatch.')
        require(int(head['site'])==plan['site'] and np.array_equal(head['partners'],
            ref.generated_partners(meta['task_definition'],full['output_mask'],plan['site'])), 'Partner coverage mismatch.')
        require(np.array_equal(head['fixed_indices'],full['fixed_sampler_indices']), 'Head fixed mapping mismatch.')
        ref.validate_layout(head,h['d_message'],h['d_edge'])
        require(ref.compare_arrays(ref.predict_head(head['mlp_input'].astype(np.float64),hw),head['ieee_head_logits'],
                    atol=2e-4,rtol=1e-5)['passed'], 'Head reference verification failed.')
        site=plan['site'];partners=head['partners']
        require(np.array_equal(full['output_bond_logits'][np.maximum(site,partners),np.minimum(site,partners)],head['bond_logits']),
                'Readout/head output mismatch.')
        cache[key]=dict(full=full,head=head,reference=obs.readout_reference(full,rw,r),readout_entry=entry)
    signature={'head_summary_sha256':obs.digest(hd/'summary.json'),
               'readout_summary_sha256':obs.digest(rd/'readout_capture_summary.json'),
               'hidden_summary_sha256':obs.digest(old/'replay_summary.json'),
               'checkpoint_sha256':r['checkpoint_sha256'],'head_script_sha256':HEAD_SHA,'analysis_script_sha256':ANALYSIS_SHA}
    return dict(head=h,readout=r,plan=plan,cache=cache,head_weights=hw,signature=signature)


def capture(args,out,context,ref,obs):
    root=Path.cwd();r=context['readout'];plan=context['plan']
    require((root/'flowr/scriptutil.py').is_file(), 'Run in the existing FLOWR.ROOT project directory.')
    helper=root/'replay_connection_hidden.py'
    require(helper.is_file() and obs.digest(helper)==ref.HELPER_SHA, 'Keep the previously verified replay_connection_hidden.py unchanged.')
    sys.path.insert(0,str(root))
    import torch
    from flowr.scriptutil import load_model
    from flowr.util.device import resolve_device
    helpers=import_file(helper,'submodule_replay_helper',ref.HELPER_SHA)
    base=r['metadata'][str(plan['seeds'][0])]['manifest']
    for name,sha in r['source_sha256'].items():require(obs.digest(root/name)==sha,f'Model source changed: {name}')
    require(str(torch.__version__)==base['torch_version'], 'Use the original recorded PyTorch environment.')
    require(obs.digest(Path(base['args']['ckpt_path']))==r['checkpoint_sha256'], 'Checkpoint changed.')
    torch.set_float32_matmul_precision(r['precision'])
    model_args=argparse.Namespace(**base['args']);model_args.save_dir=str(out/'model_context')
    device=resolve_device(model_args)
    print(f'Loading original model on {device}; read-only submodule capture.',flush=True)
    model=load_model(model_args)[0].to(device).eval();decoder=model.gen.ligand_dec
    layout=layer_layout(decoder)
    require(len(layout)==r['n_layers'] and [v['coordinate_reset'] for v in layout]==r['intermediate_coord_updates'],
            'Decoder layout differs from full-node capture.')
    require(bool(decoder.coord_skip_connect)==r['coord_skip_connect'] and
            (decoder.d_pocket_inv is not None)==r['pocket_conditioned'] and
            bool(decoder.final_coord_norm.zero_com)==r['final_norm_zero_com'] and
            float(decoder.final_coord_norm.eps)==r['final_norm_eps'], 'Readout settings changed.')
    grad_count=sum(p.requires_grad for p in model.parameters())
    require(grad_count==r['parameter_tensors_requiring_grad'], 'Native parameter grad flags changed.')
    summary_path=out/'submodule_capture_summary.json'
    if args.resume and summary_path.is_file():
        summary=obs.read_json(summary_path)
        require(summary['version']==VERSION and summary['signature']==context['signature'] and
                summary['script_sha256']==obs.digest(Path(__file__)) and summary['layout']==layout,
                'Resume identity mismatch; keep this directory and choose a new output directory.')
        entries={(e['seed'],e['call']):e for e in summary['states']}
        require(len(entries)==len(summary['states']), 'Duplicated saved states.')
    else:
        summary=dict(version=VERSION,completed=False,error=None,plan=plan,states=[],layout=layout,
                     signature=context['signature'],script_sha256=obs.digest(Path(__file__)),
                     precision=torch.get_float32_matmul_precision(),parameter_tensors_requiring_grad=grad_count,
                     scope='All-layer read-only equivariant-path observation; no activation replacement or rollout.')
        entries={}
    summary['completed']=False;summary['error']=None
    obs.write_json(summary_path,summary)
    load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
    branch_observer=full_observer=None
    try:
        for seed in plan['seeds']:
            recorded=args.output_root.resolve()/f"sh_record_{plan['case']}_seed{seed}/research_record"
            manifest=obs.read_json(recorded/'manifest.json')
            require(manifest==r['metadata'][str(seed)]['manifest'], 'Original recorded manifest changed.')
            folder=recorded/'attempt_000';task=obs.read_json(folder/'task_definition.json')
            require(task==r['metadata'][str(seed)]['task'], 'Original task definition changed.')
            inputs=load(folder/'inputs.pt')
            require(bool(model.self_condition)==inputs['self_condition'] and int(model.sc_recycle_steps)==inputs['sc_recycle_steps'],
                    'Self-conditioning settings changed.')
            pocket=helpers.move_copy(inputs['pocket_data'],device,torch)
            encoding=helpers.move_copy(load(folder/'pocket_encoding.pt'),device,torch)
            require(encoding['pocket_equis'] is not None and encoding['pocket_invs'] is not None, 'Missing pocket encoding.')
            (out/f'seed_{seed}').mkdir(exist_ok=True)
            for call_index in plan['calls']:
                key=(seed,call_index);old=entries.get(key)
                if old and old['accepted']:
                    require(obs.digest(out/old['file'])==old['sha256'], 'Saved accepted capture changed.')
                    branch_features(ref.load_npz(out/old['file']),context['cache'][key]['full'],layout)
                    print(f'Reuse verified capture: seed {seed}, call {call_index}',flush=True);continue
                entry=dict(seed=seed,call=call_index,accepted=False,file=f'seed_{seed}/call_{call_index:04d}_submodules.npz')
                entries[key]=entry;summary['states']=list(entries.values())
                obs.write_json(summary_path,summary)
                call=load(folder/'forward_calls'/f'call_{call_index:04d}.pt')
                require(call['call_index']==call_index and call['times_scalar']==context['cache'][key]['readout_entry']['times'],
                        'Original call identity changed.')
                print(f'Observe all layers: seed {seed}, call {call_index}',flush=True)
                baseline=helpers.forward_once(model,call,pocket,encoding,device,torch,np)
                entry['replay_vs_original']=helpers.compare_predictions(baseline,call['network_output_before_postprocessing'],torch,np,1e-5,1e-5)
                require(entry['replay_vs_original']['passed'], 'Original replay differs; stop.')
                full_observer=obs.ReadoutObserver(decoder,ref.to_array)
                branch_observer=SubmoduleObserver(decoder,ref.to_array,layout)
                try:observed=helpers.forward_once(model,call,pocket,encoding,device,torch,np)
                finally:
                    branch_observer.close();full_observer.close()
                arrays=branch_observer.arrays
                np.savez_compressed(out/entry['file'],**arrays)
                entry['sha256']=obs.digest(out/entry['file']);entry['times']=call['times_scalar']
                entry['observation_vs_original']=helpers.compare_predictions(observed,call['network_output_before_postprocessing'],torch,np,1e-5,1e-5)
                entry['observation_vs_replay']=helpers.compare_predictions(observed,baseline,torch,np,1e-5,1e-5)
                require(entry['observation_vs_original']['passed'] and entry['observation_vs_replay']['passed'], 'Observation changed prediction.')
                require(all(v==1 for v in branch_observer.visits.values()) and all(v==1 for v in full_observer.visits.values()),
                        'Missing/repeated hook invocation.')
                full=full_observer.arrays
                for name,k in [('output_coords','coords'),('output_mask','mask'),('output_bond_logits','bonds'),
                               ('output_atom_logits','atomics'),('output_charge_logits','charges')]:full[name]=ref.to_array(observed[k][0])
                full['input_coords']=ref.to_array(call['state_before_forward']['coords'][0])
                full['all_sampler_indices']=np.arange(len(full['output_mask']))
                full['fixed_sampler_indices']=np.asarray(task['fixed_sampler_indices'],dtype=np.int64)
                expected=context['cache'][key]['full']
                require(set(full)==set(expected), 'Full readout tensor keys changed.')
                differences=[k for k in expected if not np.array_equal(full[k],expected[k])]
                entry['changed_readout_fields']=differences
                require(not differences, f'Internal capture differs from prior readout: {differences}')
                _,rounding=branch_features(arrays,full,layout)
                entry.update(accepted=True,all_atom_slots_recorded=len(full['output_mask']),
                             submodule_tensors=len(arrays),max_residual_rounding=max(rounding.values()),
                             all_readout_tensors_exact=True)
                obs.write_json(summary_path,summary)
                print('  MATCH: native predictions, earlier internal records, submodule boundaries.',flush=True)
                full_observer=branch_observer=None
                del arrays,full,observed,baseline,call
        summary['completed']=len(entries)==len(context['cache']) and all(e['accepted'] for e in entries.values())
        require(summary['completed'], 'Not all requested states were captured.')
    except Exception as exc:
        summary['error']=f'{type(exc).__name__}: {exc}'
        if branch_observer is not None:np.savez_compressed(out/'failed_submodule_observation.npz',**branch_observer.arrays)
        if full_observer is not None:np.savez_compressed(out/'failed_full_observation.npz',**full_observer.arrays)
        raise
    finally:
        if branch_observer is not None:branch_observer.close()
        if full_observer is not None:full_observer.close()
        obs.write_json(summary_path,summary)
    return summary


def analyze(out,context,ref,obs):
    s=obs.read_json(out/'submodule_capture_summary.json')
    require(s['completed'] and s['error'] is None and s['signature']==context['signature'], 'No complete matching capture.')
    require(s['version']==VERSION and s['script_sha256']==obs.digest(Path(__file__)), 'Capture script version differs.')
    plan=context['plan'];site=plan['site'];layout=s['layout'];n_layers=len(layout)
    require([v['index'] for v in layout]==list(range(n_layers)) and n_layers==context['readout']['n_layers'], 'Layer layout mismatch.')
    require([v['coordinate_reset'] for v in layout]==context['readout']['intermediate_coord_updates'], 'Coordinate reset layout mismatch.')
    require(len(s['states'])==len(context['cache']) and {(e['seed'],e['call']) for e in s['states']}==set(context['cache']),
            'Captured state coverage mismatch.')
    records={}
    for e in s['states']:
        require(e['accepted'] and e['all_readout_tensors_exact'] and obs.digest(out/e['file'])==e['sha256'], 'Unverified submodule file.')
        require(all(e[k]['passed'] for k in ['replay_vs_original','observation_vs_original','observation_vs_replay']), 'Unverified replay.')
        key=(e['seed'],e['call']);a=ref.load_npz(out/e['file'])
        branch_features(a,context['cache'][key]['full'],layout);records[key]=a
    terms,changes=[],[]
    h=context['head'];hw=context['head_weights']
    for seed in plan['seeds']:
        scale=float(context['readout']['metadata'][str(seed)]['coord_scale'])
        require(scale>0 and np.isfinite(scale), 'Invalid coordinate units.')
        for c0,c1 in zip(plan['calls'],plan['calls'][1:]):
            a,b=[context['cache'][(seed,c)] for c in [c0,c1]]
            require(np.array_equal(a['head']['partners'],b['head']['partners']), 'Generated mapping changed across calls.')
            projected=projected_changes(records[(seed,c0)],records[(seed,c1)],a['full'],b['full'],a['reference'],b['reference'],layout)
            for head_row in ref.pair_transition(a['head'],b['head'],hw,h['d_message'],h['d_edge'],plan['target_bond_class']):
                p=head_row['partner'];ident=dict(seed=seed,call_from=c0,call_to=c1,site=site,partner=p)
                actual,_=obs.readout_ledger(a['full'],b['full'],a['reference'],b['reference'],site,p,scale,n_layers)
                rows=allocate_pair(projected,a['reference'],b['reference'],site,p,scale)
                residual=sum(v['term_A2'] for v in rows)-actual['hidden_features_term_A2']
                require(abs(residual)<=1e-9*max(1.,abs(actual['hidden_features_term_A2'])), 'Submodule pair allocation does not close.')
                terms.extend({**ident,**v} for v in rows)
                d0=float(np.linalg.norm(a['reference']['output'][p]-a['reference']['output'][site])*scale)
                d1=float(np.linalg.norm(b['reference']['output'][p]-b['reference']['output'][site])*scale)
                changes.append({**ident,**head_row,**actual,'submodule_accounting_residual_A2':float(residual),
                    'distance_before_A':d0,'distance_after_A':d1,'bond_class_changed':head_row['class_before']!=head_row['class_after'],
                    'submodule_addition_rounding_term_A2':sum(v['term_A2'] for v in rows if v['module']=='residual_addition_rounding')})
    events=[v for v in changes if v['bond_class_changed']]
    candidates=[]
    for event in events:
        pool=[v for v in terms if all(v[k]==event[k] for k in ['seed','call_from','call_to','site','partner'])
              and v['module'] not in ['initial_hidden','residual_addition_rounding'] and v['term_A2']!=0]
        for rank,v in enumerate(sorted(pool,key=lambda x:abs(x['term_A2']),reverse=True)[:6],1):
            candidates.append({**v,'display_rank_by_absolute_allocation':rank,
                               'needs_precision_review':event['needs_precision_review'] or event['readout_needs_precision_review'],
                               'interpretation':'candidate computational location; not causal importance'})
    summary=dict(version=VERSION,completed=True,plan=plan,signature=context['signature'],
        capture_summary_sha256=obs.digest(out/'submodule_capture_summary.json'),script_sha256=obs.digest(Path(__file__)),
        states=len(records),layers=n_layers,pair_transitions=len(changes),submodule_rows=len(terms),events=events,
        max_accounting_residual_A2=max(abs(v['submodule_accounting_residual_A2']) for v in changes),
        max_native_difference=max(f['max_abs_difference'] for e in s['states'] for k in
            ['replay_vs_original','observation_vs_original','observation_vs_replay'] for f in e[k]['fields'].values()),
        head_precision_flags=sum(v['needs_precision_review'] for v in changes),
        readout_precision_flags=sum(v['readout_needs_precision_review'] for v in changes),
        scope='All recorded layers and all generated partners of the configured fixed site in the selected window. Equivariant coordinate pathway only.',
        interpretation='Reference-dependent observational allocation. Not input-atom attribution, causal proof, or generalization validation.')
    obs.write_csv(out/'submodule_terms.csv',terms)
    obs.write_csv(out/'transition_summary.csv',changes)
    obs.write_csv(out/'event_candidates.csv',candidates)
    obs.write_json(out/'submodule_analysis_summary.json',summary)
    make_report(out,summary,terms,changes,candidates)
    return summary


def make_report(out,summary,terms,changes,candidates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    events=summary['events'];figures=[]
    plan=summary['plan']
    available={(v['seed'],v['partner']) for v in changes}
    display=[(s,p) for s in plan['seeds'] for p in plan.get('focus_partners',{}).get(str(s),[]) if (s,p) in available]
    if not display:display=sorted(available)[:4]
    # Whole-window maps include non-events and configured comparison trajectories.
    semantic=MODULES[:4]
    cells=[(i,name) for i in range(1,summary['layers']+1) for name in semantic]
    transitions=list(zip(plan['calls'],plan['calls'][1:]))
    lookup={(v['seed'],v['partner'],v['call_from'],v['call_to'],v['layer'],v['module']):v['term_A2'] for v in terms}
    matrices=[np.asarray([[lookup[(s,p,c0,c1,i,name)] for c0,c1 in transitions] for i,name in cells]) for s,p in display]
    limit=max((float(np.max(abs(v))) for v in matrices),default=1.) or 1.
    abbreviations=['SA','CA','FF','Reset']
    for start in range(0,len(display),2):
        subset=display[start:start+2];data=matrices[start:start+2]
        fig,axes=plt.subplots(1,len(subset),figsize=(4.7*len(subset),max(6.,.19*len(cells))),squeeze=False,layout='constrained')
        for ax,(seed,partner),grid in zip(axes[0],subset,data):
            im=ax.imshow(grid,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
            ax.set_yticks(range(len(cells)),[f'L{i} {abbreviations[semantic.index(name)]}' for i,name in cells],fontsize=7)
            ax.set_xticks(range(len(transitions)),[f'{a}>{b}' for a,b in transitions],rotation=60,fontsize=7)
            ax.set_title(f'seed {seed} / pair {plan["site"]}-{partner}')
            ax.set_xlabel('Adjacent network calls (zero-based)')
            for y in range(4,len(cells),4):ax.axhline(y-.5,color='gray',lw=.25)
        fig.colorbar(im,ax=list(axes[0]),label='Allocation to squared-distance change (A²)',shrink=.7)
        path=out/f'submodule_time_map_{start//2+1:02d}.png';fig.savefig(path,dpi=160);plt.close(fig);figures.append(path)
    if events:
        # Plot all layers. Use a common colour scale across event panels.
        grids=[]
        for e in events:
            selected={(v['layer'],v['module']):v['term_A2'] for v in terms if all(v[k]==e[k] for k in
                       ['seed','call_from','call_to','site','partner'])}
            grids.append(np.asarray([[selected[(i,m)] for m in semantic] for i in range(1,summary['layers']+1)]))
        limit=max(float(np.max(abs(v))) for v in grids) or 1.
        for start in range(0,len(events),4):
            subset=events[start:start+4];data=grids[start:start+4]
            fig,axes=plt.subplots(1,len(subset),figsize=(4.5*len(subset),max(5.,.36*summary['layers'])),squeeze=False,layout='constrained')
            for ax,e,grid in zip(axes[0],subset,data):
                im=ax.imshow(grid,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
                ax.set_xticks(range(4),['Ligand SA','Pocket CA','Equi FF','Coord reset'],rotation=35,ha='right')
                ax.set_yticks(range(summary['layers']),range(1,summary['layers']+1))
                ax.set_ylabel('Layer (one-based)')
                ax.set_title(f"seed {e['seed']} / pair {e['site']}-{e['partner']}\ncalls {e['call_from']} -> {e['call_to']}")
            fig.colorbar(im,ax=list(axes[0]),label='Allocation to squared-distance change (A²)',shrink=.85)
            path=out/f'submodule_events_{start//4+1:02d}.png';fig.savefig(path,dpi=160);plt.close(fig);figures.append(path)
    def table(headers,rows):
        def fmt(v):return html.escape(f'{v:.6g}' if isinstance(v,float) else str(v))
        return '<div class="scroll"><table><tr>'+''.join('<th>'+fmt(v)+'</th>' for v in headers)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+fmt(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
    event_table=table(['种子','调用','原子对','键类别','距离 Å：前→后','隐藏向量项 Å²','归一化尺度项 Å²'],[
        [e['seed'],f"{e['call_from']}→{e['call_to']}",f"{e['site']}–{e['partner']}",f"{e['class_before']}→{e['class_after']}",
         f"{e['distance_before_A']:.4f}→{e['distance_after_A']:.4f}",e['hidden_features_term_A2'],e['norm_scale_term_A2']] for e in events])
    candidate_table=table(['种子','调用','原子对','层','子模块','分项 Å²','精度复查'],[
        [v['seed'],f"{v['call_from']}→{v['call_to']}",f"{v['site']}–{v['partner']}",v['layer'],MODULE_LABELS[v['module']],v['term_A2'],
         '需要' if v['needs_precision_review'] else '无标记'] for v in candidates])
    focus=summary['plan'].get('focus_partners',{})
    controls=[]
    for seed in summary['plan']['seeds']:
        for p in focus.get(str(seed),[]):
            rs=[v for v in changes if v['seed']==seed and v['partner']==p]
            if rs:controls.append([seed,p,len(rs),sum(v['bond_class_changed'] for v in rs),
                 max(abs(v['distance_after_A']-v['distance_before_A']) for v in rs)])
    control_table=table(['种子','生成原子','相邻转换数','键类别变化次数','最大相邻距离变化 Å'],controls)
    images=''.join('<img alt="全部层的子模块观察分项" src="data:image/png;base64,'+base64.b64encode(f.read_bytes()).decode()+'">' for f in figures)
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>连接异常：时间与子模块定位</title>
<style>body{{font:16px/1.75 sans-serif;max-width:1250px;margin:32px auto;padding:0 20px;color:#243746}}.note{{padding:14px;background:#edf5fa;border-left:4px solid #227c9d}}.limit{{padding:14px;background:#fff2db}}table{{font-size:13px;border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{text-align:right;padding:8px;border-bottom:1px solid #ddd}}th{{background:#eef3f7}}.scroll{{overflow:auto}}img{{width:100%;height:auto}}</style>
<h1>Scaffold hopping：时间、层与子模块定位</h1>
<p>案例 {html.escape(summary['plan']['case'])}；{summary['states']} 个真实调用状态；全部 {summary['layers']} 层；固定位点 {summary['plan']['site']} 对全部有效生成原子的 {summary['pair_transitions']} 条相邻转换。</p>
<p class="note"><strong>本轮完成的是更细的观察定位。</strong>借鉴 Meltdown 按时间和模块系统扫描、结合对照缩小候选范围的思路；本轮没有执行其 activation patching。候选项将用于设计后续因果检验。</p>
<h2>1. 先看何时发生键类别变化</h2>{event_table}
<p>调用从 0 编号；层从 1 编号。键类别 0=无键、1=单键、2=双键、3=三键、4=芳香键。类别改变、距离缩短本身都不是失败标准；仍须结合价态、片段条件和终态评价。这里不要求生成骨架恢复参考结构。</p>
<h2>2. 再看哪个子模块的记录变化对应这条路径</h2>{images}{candidate_table}
<p>时间图覆盖完整窗口与配置的对照原子对；事件图与表列出键类别变化处的候选项。每次事件最多显示绝对分配值最大的六个非零项，完整结果在 submodule_terms.csv。相同类型的图共用色阶。负值对应距离平方缩短，正值对应增加。它们是实际记录更新在既定分解下的分项，不是错误概率、百分比或因果重要性。第 11、12 层没有被预先指定为答案。</p>
<h2>3. 子模块的含义</h2><ul><li>Ligand SA：配体自注意力，在固定片段和生成原子所在的配体图中交换信息。</li><li>Pocket CA：条件注意力，读取蛋白质口袋信息；不能直接当作固定片段原子贡献。</li><li>Equi FF：三维隐藏向量的前馈更新。</li><li>Coord reset：中间坐标更新后重建隐藏向量，可能抵消上游分项。</li></ul>
<h2>4. 对照窗口</h2>{control_table}<p>展示对象来自既有配置；计算覆盖全部生成原子。成功轨迹、异常出现和暂时恢复应一起考察。不同种子的生成原子编号不代表同一个化学原子；状态数也不是独立复合物数。</p>
<h2>5. 数值核对</h2><p>新记录的全部层输入输出、归一化和坐标投影与之前全原子记录逐项相同。原始预测一致性检查的最大差异为 {summary['max_native_difference']}。子模块分项与上一阶段隐藏向量总项的最大核算残差为 {summary['max_accounting_residual_A2']:.3g} Å²。头部精度提示 {summary['head_precision_flags']} 条、读出精度提示 {summary['readout_precision_flags']} 条，继续保留在 transition_summary.csv。</p>
<p>每层使用真实记录的自注意力输出、条件注意力输出和前馈输出；残差加法舍入独立保留。层内重置按重置前后差值记录。所有分项加上第一层入口差异，严格还原最终隐藏向量变化，再沿上一阶段的平均读出系数核算。归一化尺度、读出前坐标及最终数值差异仍在上一阶段总账中，不硬分给某个子模块。</p>
<p class="limit"><strong>这不是独立因果来源分解。</strong>一个子模块的输出变化可能来自它的上游输入。读出沿途的归一化、标量信息、原子对信息等也可能参与形成变化。当前只细分三维隐藏向量通路，没有穷尽所有内部变量。已定位的首个观测异常不等于根因首次形成的时刻；若候选指向更早调用，需要扩展窗口。</p>
<h2>6. 下一步怎样用这些结果</h2><p>先核对异常出现和恢复时，候选子模块的分项方向是否相应变化，再比较成功轨迹中的对应通路。根据结果提出明确的时间、模块和作用方向预测。之后的受控干预要继续运行至终态并加入非定向与时机对照，才能评估候选机制；本页不宣称根因已确定或泛化已验证。</p></html>'''
    (out/'REPORT.html').write_text(doc,encoding='utf-8')


def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--head-dir',type=Path,required=True)
    p.add_argument('--hidden-dir',type=Path,required=True)
    p.add_argument('--readout-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--output-root',type=Path,default=Path('outputs'))
    p.add_argument('--head-script',type=Path,default=here/'trace_connection_head_v2.py')
    p.add_argument('--analysis-script',type=Path,default=here/'analyze_connection_window.py')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--resume',action='store_true',help='Resume the same script/plan after a partial capture.')
    mode.add_argument('--analyze-only',action='store_true',help='Analyze this output directory without a model/GPU.')
    args=p.parse_args();out=args.output_dir.resolve()
    if out.exists() and not (args.resume or args.analyze_only):p.error('Keep existing results; use --resume or choose a new output directory.')
    if (args.resume or args.analyze_only) and not (out/'submodule_capture_summary.json').is_file():p.error('No capture summary to resume/analyze.')
    out.mkdir(parents=True,exist_ok=True);success=False
    try:
        ref=import_file(args.head_script,'verified_submodule_head',HEAD_SHA)
        obs=import_file(args.analysis_script,'verified_submodule_readout',ANALYSIS_SHA)
        context=prepare_inputs(args,ref,obs)
        obs.write_json(out/'source_identity.json',context['signature'])
        print(f"Source records verified: {len(context['cache'])} states; all layers and generated partners.",flush=True)
        if not args.analyze_only:capture(args,out,context,ref,obs)
        summary=analyze(out,context,ref,obs)
        print(f"Submodule states verified: {summary['states']}")
        print(f"Layer coverage: {summary['layers']} / {summary['layers']}")
        print(f"Pair transitions analyzed: {summary['pair_transitions']}")
        print(f"Bond-class changes: {len(summary['events'])}")
        print(f"Report: {out/'REPORT.html'}")
        success=True
        if (out/'error.txt').exists():(out/'error.txt').unlink()
    except Exception as exc:
        (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print(f'STOP: {type(exc).__name__}: {exc}',flush=True)
    finally:
        bundle=out/'submodule_review.zip'
        with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.rglob('*')):
                if f.is_file() and f!=bundle and 'model_context' not in f.relative_to(out).parts:z.write(f,f.relative_to(out))
        print(f'Review file: {bundle}')
        print(f'Completed: {success}; observational localization, not causal validation')
    return 0 if success else 1


if __name__=='__main__':
    raise SystemExit(main())
