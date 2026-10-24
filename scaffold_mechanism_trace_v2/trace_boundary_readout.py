#!/usr/bin/env python3
"""Trace the actual BondRefine -> bond_proj calculation on saved ROOT runs.

One reusable entry point. No new molecule generation, training, source edits,
cross-seed activation transplantation, or claim of final-failure causality.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import sys
import traceback
import zipfile
from pathlib import Path
import numpy as np
from trace_core import GROUPS, build_plan, readout, require
from native_readout import HeadCapture, validate_head, pair_values, select_pair_features, evaluate_coalitions
import native_chemistry
from replay_support import move_copy, compare_predictions, forward_once

ROOT = Path(__file__).resolve().parent
VERSION = '2026-10-24.boundary-readout.2'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def records_from_audit(audit):
    summary = read_json(audit/'summary.json')
    require(summary['completed'] and not summary['errors'], 'The existing boundary audit is incomplete')
    records = {}
    for entry in summary['reports']:
        rid = entry['id']
        require(Path(rid).name == rid, 'Invalid audit run name')
        folder = audit/'runs'/rid
        completion = read_json(folder/'complete.json')
        for name in ('report.json', 'network_timeline.json'):
            require(digest(folder/name) == completion['output_sha256'][name], f'Changed audit output: {rid}/{name}')
        records[rid] = dict(report=read_json(folder/'report.json'), frames=read_json(folder/'network_timeline.json'))
    require(len(records) == summary['planned'], 'Missing audit runs')
    return records


def run_folder(batch, report):
    return batch/'runs'/f"{report['case']}_lig{report['ligand_idx']}"/f"seed_{report['seed']}"/'research_record'/'attempt_000'


def array(tensor):
    return tensor.detach().cpu().float().numpy().copy()





def capture_state(model, data, call, pairs, torch, device, weights, audit_frame, receipt_path, debug_path):
    from flowr.util.functional import adj_from_node_mask
    receipt = dict(call=call['call_index'], stage='original_replay', accepted=False)
    observer, observed, baseline = None, None, None
    try:
        original = call['network_output_before_postprocessing']
        baseline = forward_once(model, call, data['pocket'], data['encoding'], device, torch, np)
        receipt['original_match'] = compare_predictions(baseline, original, torch, np, 1e-5, 1e-5)
        write_json(receipt_path, receipt)
        require(receipt['original_match']['passed'], 'Original forward replay differs; see detailed capture receipt')
        observer = HeadCapture(model.gen.ligand_dec, torch)
        try:
            observed = forward_once(model, call, data['pocket'], data['encoding'], device, torch, np)
        finally:
            observer.close()
        receipt['observation_match'] = compare_predictions(observed, original, torch, np, 1e-5, 1e-5)
        receipt['stage'] = 'native_head_validation'
        write_json(receipt_path, receipt)
        require(receipt['observation_match']['passed'], 'Observation changed the original forward output')
        adjacency = adj_from_node_mask(observed['mask'], self_connect=True)
        receipt['native_head'] = validate_head(model.gen.ligand_dec, observer, observed['bonds'], adjacency, torch)
        write_json(receipt_path, receipt)
        require(receipt['native_head']['passed'], 'Native head stage mismatch; see native_head.stages and saved debug tensors')
        features = select_pair_features(observer.tensors['mlp_input'], pairs).astype(np.float64)
        native = array(pair_values(observed['bonds'], pairs)).astype(np.float64)
        # Preserve the old failure's diagnostic, but do not use its backend for attribution.
        try:
            offline = readout(features, weights)
            require(np.isfinite(offline).all(), 'Nonfinite optional NumPy diagnostic')
            receipt['numpy_diagnostic_only'] = dict(
                max_abs_difference=float(np.max(np.abs(offline-native))),
                tolerance_violations=int(np.count_nonzero(np.abs(offline-native)>2e-4+2e-4*np.abs(native))),
                argmax_changes=int(np.count_nonzero(offline.argmax(-1)!=native.argmax(-1))),
                used_for_attribution=False)
        except Exception as exc:
            receipt['numpy_diagnostic_only'] = dict(error=f'{type(exc).__name__}: {exc}', used_for_attribution=False)
        atom_classes = observed['atomics'][0].argmax(-1).tolist()
        tokens = model.builder.vocab.tokens_from_indices(atom_classes)
        active = np.array([bool(v) and t not in {'<PAD>', '<NOATOM>'}
                           for v,t in zip(observed['mask'][0].tolist(),tokens)])
        lookup = {(int(a),int(c)):i for i,(a,c) in enumerate(pairs)}
        for site in audit_frame['fixed_sites']:
            expected = {n['slot']:n['class'] for n in site.get('neighbors',[]) if not n['fixed']}
            for (fixed,other),idx in lookup.items():
                if fixed==site['slot'] and active[fixed] and active[other]:
                    require(int(native[idx].argmax())==expected.get(other,0), 'Native head decisions disagree with the existing audit')
        raw = {k:v.detach().cpu().clone() for k,v in observed.items()
               if k in {'coords','atomics','bonds','charges','hybridization','mask'} and torch.is_tensor(v)}
        head = dict(features=observer.tensors['mlp_input'].detach().cpu().clone(),
                    adjacency=adjacency.detach().cpu().clone(), network_bonds=raw['bonds'], prediction=raw)
        receipt.update(accepted=True,stage='accepted',module='native BondRefine.message_mlp -> bond_proj -> adjacency mask')
        write_json(receipt_path, receipt)
        return dict(logits=native.astype(np.float32),pairs=pairs,active=active,atom_classes=np.array(atom_classes)), head, receipt
    except BaseException as exc:
        receipt.update(error=f'{type(exc).__name__}: {exc}', accepted=False)
        write_json(receipt_path, receipt)
        debug_path.parent.mkdir(parents=True,exist_ok=True)
        arrays={}
        if observer is not None:
            arrays.update({k:array(v) for k,v in observer.tensors.items()})
        if observed is not None:
            arrays['network_bonds']=array(observed['bonds'])
            arrays['mask']=array(observed['mask'])
        if baseline is not None:
            arrays['replayed_bonds']=array(baseline['bonds'])
        arrays['recorded_bonds']=array(call['network_output_before_postprocessing']['bonds'])
        np.savez_compressed(debug_path,**arrays)
        raise



def analyses(plan, records, out, weights, d_message, model, torch, device):
    summaries, rows, skips, path_rows = [], [], [], []
    from rdkit import Chem, rdBase
    from flowr.util.rdkit import mol_is_valid
    atom_tokens = [model.builder.vocab.idx_token_map[i] for i in range(model.builder.vocab.size)]
    charge_tokens = [model.builder.vocab_charges.idx_token_map[i] for i in range(model.builder.vocab_charges.size)]
    load_head = lambda p: move_copy(torch.load(p,map_location='cpu',weights_only=True),device,torch)
    for j in plan['jobs']:
        folder = out/'captures'/j['run']
        with np.load(folder/f"call_{j['before']:04d}.npz", allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        with np.load(folder/f"call_{j['after']:04d}.npz", allow_pickle=False) as z:
            b = {k: z[k] for k in z.files}
        require(np.array_equal(a['pairs'], b['pairs']), 'Pair indexing changed within one trajectory')
        pairs = a['pairs']
        chosen = np.flatnonzero((pairs[:, 0] == j['site']) &
                              a['active'][pairs[:, 0]] & a['active'][pairs[:, 1]] &
                              b['active'][pairs[:, 0]] & b['active'][pairs[:, 1]])
        if not len(chosen):
            skips.append(dict(id=j['id'], reason='No shared active boundary pair; no attribution.'))
            continue
        unknown = np.flatnonzero((pairs[:, 0] == j['site']) & ~(
            a['active'][pairs[:, 0]] & a['active'][pairs[:, 1]] &
            b['active'][pairs[:, 0]] & b['active'][pairs[:, 1]]))
        head_before=load_head(folder/f"call_{j['before']:04d}_head.pt")
        head_after=load_head(folder/f"call_{j['after']:04d}_head.pt")
        def inspect_coalition(coalition,bonds):
            raw={k:v.detach().cpu().clone() for k,v in head_after['prediction'].items()}
            raw['bonds']=bonds.detach().cpu().clone()
            g=native_chemistry.graph(raw,atom_tokens,charge_tokens)
            chem,mol=native_chemistry.chemistry(raw,model.builder,g,Chem,rdBase,mol_is_valid)
            assessed=chem['status']=='assessed'
            problems=chem['problems']
            site_valence=(any(p['type']=='AtomValenceException' and j['site'] in p['sampler_slots'] for p in problems)
                          if assessed else None)
            row=dict(contrast=j['id'],run=j['run'],before=j['before'],after=j['after'],fixed_slot=j['site'],
                     kind=j['kind'],coalition=coalition,
                     groups_from_previous='|'.join(name for k,name in enumerate(GROUPS) if not (coalition & (1<<k))) or 'none_identity',
                     chemistry_status=chem['status'],site_valence_problem=site_valence,
                     rdkit_valid=chem['native_rdkit_valid'] if assessed else None,
                     connected=bool(len(Chem.GetMolFrags(mol))==1 and mol.GetNumAtoms()>0) if mol is not None else None,
                     other_problem_slots='|'.join(map(str,sorted({s for p in problems for s in p['sampler_slots'] if s!=j['site']}))),
                     site_explicit_order_sum=float(np.array([0.,1.,2.,3.,1.5])[g['classes'][j['site']]].sum()) if g['keep'][j['site']] else None,
                     final_rollout_effect='not_tested')
            if coalition==15:
                old_chem=records[j['run']]['frames'][j['after']]['chemistry']
                require(chem['status']==old_chem['status'] and chem['native_rdkit_valid']==old_chem['native_rdkit_valid'],
                        'Identity coalition chemistry differs from the existing audit')
                old_site=any(p['type']=='AtomValenceException' and j['site'] in p['sampler_slots'] for p in old_chem['problems'])
                require(site_valence is None or site_valence==old_site,'Identity coalition site event differs')
            return row
        check_path=out/'contrasts'/f"{j['id']}_checks.json"
        try:
            val,checks,local_rows=evaluate_coalitions(model.gen.ligand_dec,head_before,head_after,pairs[chosen],
                       d_message,torch,callback=inspect_coalition,receipt_callback=lambda r:write_json(check_path,r))
        except BaseException:
            debug=out/'debug';debug.mkdir(exist_ok=True)
            np.savez_compressed(debug/f"{j['id']}_endpoints.npz",before_features=array(head_before['features']),
                                after_features=array(head_after['features']),pairs=pairs[chosen],
                                before_bonds=array(head_before['network_bonds']),after_bonds=array(head_after['network_bonds']))
            raise
        path_rows.extend(local_rows)
        write_csv(out/'path_comparison.csv',path_rows)
        print(f"Compared {j['id']}: {j['run']} calls {j['before']}->{j['after']}; all 16 native combinations",flush=True)
        del head_before,head_after
        rec = records[j['run']]['report']
        timeline = records[j['run']]['frames']
        site_before = next(s for s in timeline[j['before']]['fixed_sites'] if s['slot'] == j['site'])
        site_after = next(s for s in timeline[j['after']]['fixed_sites'] if s['slot'] == j['site'])
        identity_changed = any(site_before.get(k) != site_after.get(k) for k in ('present', 'element', 'charge'))
        internal_changed = ({n['slot']: n['class'] for n in site_before.get('neighbors', []) if n['fixed']} !=
                            {n['slot']: n['class'] for n in site_after.get('neighbors', []) if n['fixed']})
        total_phi = val['order_phi'].sum(axis=1)
        summary = dict(contrast=j['id'], run=j['run'], case=rec['case'], ligand_idx=rec['ligand_idx'], seed=rec['seed'],
                       before=j['before'], after=j['after'], fixed_slot=j['site'],
                       fixed_reference_atom=rec['fixed_mapping'][str(j['site'])]['reference_index'],
                       kind=j['kind'], terminal_segment_onset=j['terminal_segment_onset'],
                       assessed_boundary_pairs=len(chosen), excluded_virtual_or_missing_pairs=len(unknown),
                       fixed_atom_identity_changed=identity_changed, fixed_internal_bonds_changed=internal_changed,
                       expected_boundary_order_before=float(val['expected_order'][0].sum()),
                       expected_boundary_order_after=float(val['expected_order'][15].sum()),
                       expected_boundary_order_change=float((val['expected_order'][15]-val['expected_order'][0]).sum()),
                       attribution_sum_error=float(abs(total_phi.sum()-(val['expected_order'][15]-val['expected_order'][0]).sum())),
                       **{f'phi_{name}': float(total_phi[k]) for k, name in enumerate(GROUPS)},
                       largest_absolute_score_contributor=(GROUPS[int(np.abs(total_phi).argmax())]
                                                           if np.max(np.abs(total_phi)) > 1e-12 else 'none_zero_change'),
                       final_failure_causality='not_tested', input_atom_importance='not_tested')
        summaries.append(summary)
        for local, idx in enumerate(chosen):
            before_cls, after_cls = int(a['logits'][idx].argmax()), int(b['logits'][idx].argmax())
            row = dict(contrast=j['id'], run=j['run'], before=j['before'], after=j['after'],
                       fixed_slot=int(pairs[idx, 0]), generated_slot=int(pairs[idx, 1]),
                       before_class=before_cls, after_class=after_cls, class_changed=before_cls != after_cls,
                       compared_bond_class=int(val['target_class'][local]),
                       margin_before=float(val['margin'][0, local]), margin_after=float(val['margin'][15, local]),
                       expected_order_before=float(val['expected_order'][0, local]), expected_order_after=float(val['expected_order'][15, local]),
                       **{f'margin_phi_{name}': float(val['margin_phi'][k, local]) for k, name in enumerate(GROUPS)},
                       **{f'order_phi_{name}': float(val['order_phi'][k, local]) for k, name in enumerate(GROUPS)})
            rows.append(row)
        # Retain every mixed head result, not just a ranked feature explanation.
        np.savez_compressed(out/'contrasts'/f"{j['id']}.npz", pairs=pairs[chosen],
                            coalition_logits=val['logits'], margin_phi=val['margin_phi'], order_phi=val['order_phi'])
        write_csv(out/'contrast_summary.csv', summaries)
        write_csv(out/'pair_scores.csv', rows)
    write_csv(out/'contrast_summary.csv', summaries)
    write_csv(out/'pair_scores.csv', rows)
    write_json(out/'analysis_skips.json', skips)
    write_csv(out/'path_comparison.csv', path_rows)
    return summaries, rows


def report_html(out, summary_rows, status):
    labels = {'fixed_node': '固定原子表征', 'generated_node': '生成原子表征', 'geometry': '几何特征', 'edge_context': '累积边表征'}
    body = []
    for r in summary_rows:
        vals = [r['run'], f"{r['before']} → {r['after']}", r['fixed_slot'], r['kind'],
                f"{r['expected_boundary_order_change']:.6g}"]
        vals += [f"{r['phi_'+g]:.6g}" for g in GROUPS]
        body.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in vals)+'</tr>')
    headers = ['轨迹', '调用号（从0开始）', '固定槽位', '窗口', '预期边界键级总和变化']+[labels[g] for g in GROUPS]
    text = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>成键计算来源</title>'
            '<style>body{font:16px system-ui;max-width:1350px;margin:32px auto;padding:0 20px;line-height:1.7;color:#18262f}'
            'table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccd6dc;padding:7px}th{background:#e9f1f4}pre{white-space:pre-wrap}</style>'
            '<h1>固定片段—新骨架：成键计算来源</h1>'
            '<p>对已有原始轨迹进行模型内部回放。先复现完整单次前向输出，再读取 BondRefine 的真实输入；'
            '用原生PyTorch成键模块计算四组特征的全部16种前后组合，平均分摊交互项。四项相加必须等于实际分数变化。</p>'
            '<p><b>这是局部成键分数的来源分解，不是“根因概率”，也不是输入原子的贡献度。'
            '它尚未检验后续自由生成与最终失败。混合表征可能偏离自然分布，不能凭排名给最终失败定因。</b></p>'
            '<p>正值表示该组对本次分数上升有贡献；负值表示抵消。预期键级总和仅是连续观察量，不能替代RDKit或ROOT判别。'
            '正常对照也会发生分数变化；须比较持续异常、自行恢复和无该异常轨迹。不同种子的生成槽位不互相对应。</p>'
            '<p>path_comparison.csv 同时列出全部16种组合对当步固定连接位点价态、全图化学有效性和连通性的影响。使用原生构建器及已有RDKit规则；未进行后续自由生成，不能称最终失败已被修复。</p>'
            '<p>现有终态评价保持原样。全部组合由原生PyTorch模块、原设备/精度与完整张量形状计算；NumPy重算只记录差异，不用于归因。</p>'
            '<pre>'+html.escape(json.dumps(status, ensure_ascii=False, indent=2))+'</pre>'
            '<div style="overflow:auto"><table><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr>'+''.join(body)+'</table></div>'
            '<p>pair_scores.csv 提供每条边的类别、分数及来源；contrast_summary.csv 为位点汇总；'
            'capture_checks.jsonl 为原始回放和读出重建校验。没有满足回放校验的结果不会进入分析。</p></html>')
    (out/'REPORT.html').write_text(text, encoding='utf-8')


def bundle(out):
    path = out/'mechanism_review.zip'
    temp = out/'mechanism_review.tmp'
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix not in {'.pt', '.tmp', '.pyc', '.zip'} and not {'captures', 'model_context', '__pycache__'} & set(p.relative_to(out).parts):
                z.write(p, p.relative_to(out))
    temp.replace(path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument('--batch', type=Path, default=Path('outputs/sh_validation_baselines_v1'))
    ap.add_argument('--audit', type=Path)
    ap.add_argument('--output', type=Path, default=Path('outputs/sh_mechanism_trace_v2'))
    ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args(argv)
    batch = args.batch.resolve()
    audit = (args.audit or batch/'boundary_audit_v1').resolve()
    records = records_from_audit(audit)
    plan = build_plan(records)
    count = sum(map(len, plan['captures'].values()))
    print(json.dumps(dict(runs=len(records), local_contrasts=len(plan['jobs']), saved_states_to_replay=count,
                          new_generations=0, full_rollout_interventions=0, scope=plan['scope']), ensure_ascii=False, indent=2), flush=True)
    if args.plan_only:
        return 0
    out, repo = args.output.resolve(), Path.cwd().resolve()
    require((repo/'flowr/scriptutil.py').is_file(), 'Run inside your original FLOWR.ROOT repository')
    require(not out.exists() or args.resume, 'Output exists; preserve it and use --resume for the identical analysis')
    out.mkdir(parents=True, exist_ok=True)
    (out/'contrasts').mkdir(exist_ok=True)
    (out/'debug').mkdir(exist_ok=True)
    state = dict(version=VERSION, completed=False, captured=0, planned_captures=count,
                 errors=[], stage='preflight', scope=plan['scope'])
    summary_rows = []
    try:
        frozen = read_json(batch/'frozen_plan.json')
        audit_identity = read_json(audit/'identity.json')
        require(digest(batch/'frozen_plan.json') == audit_identity['plan_sha256'], 'Audit and generation plan differ')
        require(digest(batch/'quality/quality_report.json') == audit_identity['quality_sha256'], 'Existing terminal evaluation changed')
        expected = read_json(ROOT/'source_audit.json')['source_checks']
        for name, sha in frozen['config']['source_sha256'].items():
            require(digest(repo/name) == sha == expected[name]['actual'], f'Native source changed: {name}; do not bypass')
        import torch
        from rdkit import rdBase
        sys.path.insert(0, str(repo))
        from flowr.scriptutil import load_model
        from flowr.util.device import resolve_device
        require(str(torch.__version__) == frozen['torch_version'] and rdBase.rdkitVersion == frozen['rdkit_version'], 'Use the original Torch/RDKit environment')
        metadata, file_hashes = {}, {}
        for rid, rec in records.items():
            folder = run_folder(batch, rec['report'])
            manifest, task = read_json(folder.parent/'manifest.json'), read_json(folder/'task_definition.json')
            require(task['mapping_verified'] and manifest['args']['scaffold_hopping'], f'{rid}: invalid mapping/task')
            require(manifest['checkpoint_sha256'] == frozen['checkpoint_sha256'], 'Mixed model checkpoints')
            require(bool(manifest['source_sha256']) and
                    all(frozen['config']['source_sha256'].get(k) == v for k, v in manifest['source_sha256'].items()), 'Source identities differ')
            files = [folder/'inputs.pt', folder/'pocket_encoding.pt', folder/'task_definition.json', folder.parent/'manifest.json']
            files += [folder/'forward_calls'/f'call_{k:04d}.pt' for k in plan['captures'][rid]]
            file_hashes[rid] = {str(p.relative_to(folder.parent)): digest(p) for p in files}
            metadata[rid] = dict(folder=folder, manifest=manifest, task=task)
        first = next(iter(metadata.values()))
        ckpt = Path(first['manifest']['args']['ckpt_path'])
        require(digest(ckpt) == frozen['checkpoint_sha256'], 'Checkpoint changed')
        identity = dict(version=VERSION, plan=plan, native_sources=frozen['config']['source_sha256'],
                        checkpoint=frozen['checkpoint_sha256'], inputs=file_hashes,
                        audit_plan_sha256=audit_identity['plan_sha256'], quality_sha256=audit_identity['quality_sha256'],
                        code_sha256={p.name: digest(p) for p in ROOT.glob('*.py')})
        if args.resume:
            require((out/'identity.json').is_file() and read_json(out/'identity.json') == identity, 'Resume identity differs; use a new output directory')
        else:
            write_json(out/'identity.json', identity)
        write_json(out/'plan.json', plan)
        (out/'program').mkdir(exist_ok=True)
        for p in ROOT.iterdir():
            if p.is_file() and p.suffix in {'.py', '.json', '.md'}:
                (out/'program'/p.name).write_bytes(p.read_bytes())
        torch.set_float32_matmul_precision('high')
        ma = argparse.Namespace(**first['manifest']['args'])
        ma.save_dir = str(out/'model_context')
        device = resolve_device(ma)
        model = load_model(ma)[0].to(device).eval()
        # Do not call requires_grad_(False): preserve original inference kernel behavior.
        dec = model.gen.ligand_dec
        mlp = dec.bond_refine.message_mlp
        require(len(mlp) == 3 and isinstance(mlp[0], torch.nn.Linear) and isinstance(mlp[1], torch.nn.SiLU)
                and isinstance(mlp[2], torch.nn.Linear) and isinstance(dec.bond_proj, torch.nn.Linear), 'Unexpected readout architecture')
        weights = {k: array(v).astype(np.float64) for k, v in
                   dict(w0=mlp[0].weight, b0=mlp[0].bias, w2=mlp[2].weight, b2=mlp[2].bias,
                        wp=dec.bond_proj.weight, bp=dec.bond_proj.bias).items()}
        d_message = dec.bond_refine.node_proj.out_features
        np.savez_compressed(out/'readout_weights.npz', **weights, d_message=np.array(d_message))
        load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
        state['stage'] = 'capture'
        for rid, meta in metadata.items():
            folder, task = meta['folder'], meta['task']
            inputs = load(folder/'inputs.pt')
            require(inputs['self_condition'] == bool(model.self_condition) and
                    int(inputs['sc_recycle_steps']) == int(model.sc_recycle_steps), 'Self-conditioning configuration differs')
            fixed = list(task['fixed_sampler_indices'])
            active = inputs['prior']['mask'][0].bool().tolist()
            got_fixed = (inputs['prior']['fragment_mask'][0].bool() & inputs['prior']['mask'][0].bool()).nonzero(as_tuple=True)[0].tolist()
            require(got_fixed == fixed, 'Fixed mask differs from recorded atom mapping')
            generated = [i for i, yes in enumerate(active) if yes and i not in set(fixed)]
            pairs = np.array([(f, g) for f in fixed for g in generated], dtype=np.int64)
            require(len(pairs) > 0, 'No fixed-to-generated pairs in this task')
            data = dict(pocket=move_copy(inputs['pocket_data'], device, torch),
                        encoding=move_copy(load(folder/'pocket_encoding.pt'), device, torch))
            dest = out/'captures'/rid
            dest.mkdir(parents=True, exist_ok=True)
            for k in plan['captures'][rid]:
                target = dest/f'call_{k:04d}.npz'
                receipt_path = dest/f'call_{k:04d}.json'
                head_path = dest/f'call_{k:04d}_head.pt'
                state['active_capture'] = dict(run=rid,call=k)
                write_json(out/'status.json',state)
                if args.resume and target.exists() and receipt_path.exists() and head_path.exists():
                    receipt = read_json(receipt_path)
                    require(receipt.get('accepted') and digest(target) == receipt['sha256'] and digest(head_path)==receipt['head_sha256'], 'Cached capture changed')
                else:
                    call = load(folder/'forward_calls'/f'call_{k:04d}.pt')
                    require(call['call_index'] == k, 'Call file index differs')
                    values, head, receipt = capture_state(model, data, call, pairs, torch, device, weights, records[rid]['frames'][k], receipt_path, out/'debug'/f'{rid}_call_{k:04d}.npz')
                    np.savez_compressed(target, **values)
                    torch.save(head,head_path)
                    receipt['head_sha256'] = digest(head_path)
                    receipt['sha256'] = digest(target)
                    del head
                    write_json(receipt_path, receipt)
                state['captured'] += 1
                print(f"[{state['captured']}/{count}] verified {rid} call {k}", flush=True)
                write_json(out/'status.json', state)
            del data, inputs
        receipts = []
        for rid, calls in plan['captures'].items():
            for k in calls:
                receipts.append(dict(run=rid, **read_json(out/'captures'/rid/f'call_{k:04d}.json')))
        (out/'capture_checks.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in receipts), encoding='utf-8')
        state['stage'] = 'local_readout_attribution'
        write_json(out/'status.json', state)
        summary_rows, pair_rows = analyses(plan, records, out, weights, d_message, model, torch, device)
        state.update(completed=True, stage='finished', active_capture=None, assessed_contrasts=len(summary_rows), pair_rows=len(pair_rows),
                     coalition_backend='native_torch_full_shape', local_graph_comparisons=16*len(summary_rows))
    except (Exception, KeyboardInterrupt) as exc:
        state['errors'].append(f'{type(exc).__name__}: {exc}')
        (out/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print(state['errors'][-1], file=sys.stderr, flush=True)
    finally:
        write_json(out/'status.json', state)
        with (out/'capture_checks.jsonl').open('w',encoding='utf-8') as receipt_log:
            for file in sorted((out/'captures').glob('*/*.json')):
                receipt_log.write(json.dumps(dict(run=file.parent.name,**read_json(file)),ensure_ascii=False,allow_nan=False)+'\n')
        report_html(out, summary_rows, state)
        path = bundle(out)
        print(f'Completed: {state["completed"]}\nReview file: {path}', flush=True)
    return 0 if state['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
