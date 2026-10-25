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
from trace_core import GROUPS, build_plan, decompose, probabilities, readout, require
from replay_support import move_copy, compare_predictions, forward_once

ROOT = Path(__file__).resolve().parent
VERSION = '2026-10-24.boundary-readout.1'


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


class Capture:
    def __init__(self, decoder, pairs):
        self.pairs = pairs
        self.arrays, self.visits = {}, 0
        self.handle = decoder.bond_refine.message_mlp.register_forward_pre_hook(self.hook)

    def hook(self, module, args):
        self.visits += 1
        x = args[0]
        require(x.ndim == 4 and x.shape[0] == 1, 'Unexpected bond refinement input shape')
        f, g = self.pairs[:, 0].tolist(), self.pairs[:, 1].tolist()
        self.arrays['features'] = np.stack((array(x[0, f, g]), array(x[0, g, f])), axis=1)
        return None

    def close(self):
        self.handle.remove()


def capture_state(model, data, call, pairs, torch, device, weights, audit_frame):
    # Preserve original parameter flags; forward_once restores each saved random state.
    original = call['network_output_before_postprocessing']
    baseline = forward_once(model, call, data['pocket'], data['encoding'], device, torch, np)
    check1 = compare_predictions(baseline, original, torch, np, 1e-5, 1e-5)
    require(check1['passed'], f"Original replay differs at call {call['call_index']}: {check1}")
    observer = Capture(model.gen.ligand_dec, pairs)
    try:
        observed = forward_once(model, call, data['pocket'], data['encoding'], device, torch, np)
    finally:
        observer.close()
    check2 = compare_predictions(observed, original, torch, np, 1e-5, 1e-5)
    require(check2['passed'] and observer.visits == 1, 'Observation changed the result or readout was not called once')
    x = observer.arrays['features'].astype(np.float64)
    b = array(observed['bonds'][0]).astype(np.float64)
    f, g = pairs[:, 0], pairs[:, 1]
    native = b[np.maximum(f, g), np.minimum(f, g)]
    reconstructed = readout(x, weights)
    require(np.isfinite(x).all() and np.isfinite(native).all(), 'Nonfinite head values')
    require(np.allclose(reconstructed, native, atol=2e-4, rtol=2e-4)
            and np.array_equal(reconstructed.argmax(-1), native.argmax(-1)),
            'Offline head reconstruction does not reproduce the native bond decision; no attribution is accepted')
    atom_classes = observed['atomics'][0].argmax(-1).tolist()
    tokens = model.builder.vocab.tokens_from_indices(atom_classes)
    active = np.array([bool(v) and t not in {'<PAD>', '<NOATOM>'}
                       for v, t in zip(observed['mask'][0].tolist(), tokens)])
    # Independently compare the audited boundary graph to the reconstructed head.
    lookup = {(int(a), int(c)): i for i, (a, c) in enumerate(pairs)}
    for site in audit_frame['fixed_sites']:
        expected = {n['slot']: n['class'] for n in site.get('neighbors', []) if not n['fixed']}
        for (s, other), idx in lookup.items():
            if s == site['slot'] and active[s] and active[other]:
                require(int(native[idx].argmax()) == expected.get(other, 0), 'Head classes disagree with the existing audit')
    receipt = dict(call=call['call_index'], original_match=check1, observation_match=check2,
                   offline_head_max_abs_error=float(np.max(np.abs(reconstructed-native))),
                   offline_argmax_equal=True, module='gen.ligand_dec.bond_refine.message_mlp -> bond_proj')
    return dict(features=x.astype(np.float32), logits=native.astype(np.float32), pairs=pairs,
                active=active, atom_classes=np.array(atom_classes)), receipt


def analyses(plan, records, out, weights, d_message):
    summaries, rows, skips = [], [], []
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
        val = decompose(a['features'][chosen].astype(np.float64), b['features'][chosen].astype(np.float64), weights, d_message)
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
    write_json(out/'analysis_skips.json', skips)
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
            '用原成键网络计算四组特征的全部16种前后组合，平均分摊交互项。四项相加必须等于实际分数变化。</p>'
            '<p><b>这是局部成键分数的来源分解，不是“根因概率”，也不是输入原子的贡献度。'
            '它尚未检验后续自由生成与最终失败。混合表征可能偏离自然分布，不能凭排名给最终失败定因。</b></p>'
            '<p>正值表示该组对本次分数上升有贡献；负值表示抵消。预期键级总和仅是连续观察量，不能替代RDKit或ROOT判别。'
            '正常对照也会发生分数变化；须比较持续异常、自行恢复和无该异常轨迹。不同种子的生成槽位不互相对应。</p>'
            '<p>现有化学、片段保持、PoseBusters、骨架变化评价完全沿用，彼此分开。当前批次属于机制探索。</p>'
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
    ap.add_argument('--output', type=Path, default=Path('outputs/sh_mechanism_trace_v1'))
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
                if args.resume and target.exists() and receipt_path.exists():
                    receipt = read_json(receipt_path)
                    require(digest(target) == receipt['sha256'], 'Cached capture changed')
                else:
                    call = load(folder/'forward_calls'/f'call_{k:04d}.pt')
                    require(call['call_index'] == k, 'Call file index differs')
                    values, receipt = capture_state(model, data, call, pairs, torch, device, weights, records[rid]['frames'][k])
                    np.savez_compressed(target, **values)
                    receipt['sha256'] = digest(target)
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
        summary_rows, pair_rows = analyses(plan, records, out, weights, d_message)
        state.update(completed=True, stage='finished', assessed_contrasts=len(summary_rows), pair_rows=len(pair_rows))
    except (Exception, KeyboardInterrupt) as exc:
        state['errors'].append(f'{type(exc).__name__}: {exc}')
        (out/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print(state['errors'][-1], file=sys.stderr, flush=True)
    finally:
        write_json(out/'status.json', state)
        report_html(out, summary_rows, state)
        path = bundle(out)
        print(f'Completed: {state["completed"]}\nReview file: {path}', flush=True)
    return 0 if state['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
