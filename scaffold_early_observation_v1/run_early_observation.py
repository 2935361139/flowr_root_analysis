#!/usr/bin/env python3
"""Read-only ROOT replay: early geometry, fixed-node/pair features and bond head.

Run from the original ROOT repo with its original environment. No generation,
training or interventions. Both native and observed forwards must reproduce the
saved prediction before any record is accepted. Indices are zero based.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import html
import importlib.util
import json
import shutil
import sys
import traceback
import zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
VERSION = '2026-10-23.early-observation.1'
ADAPTER_HASHES = {
    'replay.py': '1d5b6cfef030f85d1b0df939d96b6402d64f68e9b2a8ef0b746c12236a84ff82',
    'bond_head.py': 'b1adadd8fccaad431d95632e77da5646c3de2815f9ece1ceac1e217fa747268e',
    'coordinate_readout.py': 'eaf1bf202a1a3f7abcfeb5a8e95067c7c2a6ad056cab9eab2f1909f95599d968',
    'coordinate_branches.py': 'c05f5a1c98a7841711decadc18097914084490006cce7c878cf17570e846e360'}


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def adapters():
    modules = []
    for name, expected in ADAPTER_HASHES.items():
        path = HERE / 'adapters' / name
        require(digest(path) == expected, f'Supplied adapter changed: {name}')
        spec = importlib.util.spec_from_file_location('early_' + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def read_plan(path):
    p = read_json(path)
    require(isinstance(p['case'], str) and p['case'] and '/' not in p['case'], 'Invalid case name.')
    for key in ['seeds', 'calls']:
        require(p[key] and all(type(v) is int and v >= 0 for v in p[key]), f'Invalid {key}.')
        require(len(set(p[key])) == len(p[key]), f'Duplicated {key}.')
    require(len(p['calls']) >= 2 and all(b - a == 1 for a, b in zip(p['calls'], p['calls'][1:])),
            'Use consecutive ascending call indices.')
    require(all(type(p[k]) is int and p[k] >= 0 for k in ['site', 'partner']), 'Invalid focus indices.')
    require(p['site'] != p['partner'] and p['expected_bond_class'] in [1, 2, 3, 4], 'Invalid focus pair.')
    return p


class ScalarBranchObserver:
    """Observe fixed-node residual writes and self-attention pair features."""
    def __init__(self, decoder, tracked, site, to_array, layout):
        self.arrays, self.handles, self.visits = {}, [], {}
        self.tracked, self.site, self.to_array = tracked, site, to_array
        try:
            for item in layout:
                i = item['index']; layer = decoder.layers[i]; p = f'layer_{i:02d}__'
                self.add(layer.self_attention, p + 'self_attention_update', output=True, attention=True)
                if item['has_condition']:
                    self.add(layer.cond_attn_self_inv_norm, p + 'before_condition')
                    self.add(layer.cond_attention, p + 'conditional_attention_update', output=True, attention=True)
                self.add(layer.ff_inv_norm, p + 'before_feedforward')
                self.add(layer.inv_ff, p + 'invariant_feedforward_update', output=True)
        except Exception:
            self.close()
            raise

    def add(self, module, name, output=False, attention=False):
        self.visits[name] = 0
        def hook(*args):
            self.visits[name] += 1
            require(self.visits[name] == 1, f'Repeated scalar hook: {name}')
            value = args[2] if output else args[1][0]
            if attention:
                require(isinstance(value, (tuple, list)) and len(value) == 3, 'Attention contract changed.')
                if 'self_attention' in name and value[2] is not None:
                    edge = self.to_array(value[2][0])
                    self.arrays[name + '__pairs_from_site'] = edge[self.site].copy()
                    self.arrays[name + '__pairs_to_site'] = edge[:, self.site].copy()
                value = value[1]
            require(value.ndim == 3 and value.shape[0] == 1, 'Expected batch-one scalar features.')
            self.arrays[name] = self.to_array(value[0])[self.tracked].copy()
            return None
        register = module.register_forward_hook if output else module.register_forward_pre_hook
        self.handles.append(register(hook))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def scalar_components(record, hidden, layout, branches):
    components, rounding = {}, {}
    for item in layout:
        i = item['index']; p = f'layer_{i:02d}__'
        start, end = hidden[p + 'in__invs'], hidden[p + 'out__invs']
        pre_ff = record[p + 'before_feedforward']
        pre_ca = record[p + 'before_condition'] if item['has_condition'] else pre_ff
        sa = record[p + 'self_attention_update']
        ca = record[p + 'conditional_attention_update'] if item['has_condition'] else np.zeros_like(sa)
        ff = record[p + 'invariant_feedforward_update']
        r = (branches.addition_rounding(start, sa, pre_ca, p + 'scalar_self_attention')
             + branches.addition_rounding(pre_ca, ca, pre_ff, p + 'scalar_conditional_attention')
             + branches.addition_rounding(pre_ff, ff, end, p + 'scalar_feedforward'))
        for name, value in [('self_attention', sa), ('conditional_attention', ca), ('invariant_feedforward', ff),
                            ('residual_addition_rounding', r)]:
            components[(i + 1, name)] = np.asarray(value, dtype=np.float64)
        rounding[i + 1] = float(np.max(abs(r)))
        if i:
            require(np.array_equal(start, hidden[f'layer_{i-1:02d}__out__invs']), 'Scalar layer continuity failed.')
    require(np.allclose(hidden['layer_00__in__invs'].astype(np.float64) + sum(components.values()),
                        hidden[f'layer_{len(layout)-1:02d}__out__invs'], atol=1e-9, rtol=1e-9),
            'Scalar residual decomposition does not close.')
    return components, rounding


def save_state_arrays(arrays, role, state, to_array):
    if state is None:
        return
    for key in ['coords', 'atomics', 'bonds', 'charges', 'mask']:
        if key in state:
            arrays[f'd__{role}__{key}'] = to_array(state[key][0])


def capture(p, out, summary, modules):
    replay, head, coord, branches = modules
    root = Path.cwd()
    require((root / 'flowr/scriptutil.py').is_file(), 'Run from /home/ljl/projects/flowr_root with .venv active.')
    sys.path.insert(0, str(root))
    import torch
    import rdkit
    from flowr.scriptutil import load_model
    from flowr.util.device import resolve_device
    prior_dir = Path(p['reference_head_dir']).resolve()
    prior = read_json(prior_dir / 'summary.json')
    require(prior['capture_completed'] and prior['analysis_completed'] and prior['error'] is None,
            'Reference head analysis must be complete.')
    require(prior['script_sha256'] == ADAPTER_HASHES['bond_head.py'], 'Reference head adapter differs.')
    runs = Path(p['record_root']).resolve()
    records = {s: runs / f"sh_record_{p['case']}_seed{s}" / 'research_record' for s in p['seeds']}
    manifests = {s: read_json(path / 'manifest.json') for s, path in records.items()}
    base = manifests[p['seeds'][0]]
    require(base['args']['scaffold_hopping'], 'Recorded task must be scaffold hopping.')
    for seed, manifest in manifests.items():
        require(manifest['args']['seed'] == seed and replay.condition_key(manifest) == replay.condition_key(base),
                'Generation conditions differ beyond seed/output.')
        require(manifest == prior['metadata'][str(seed)]['manifest'], 'Original manifest differs from verified head run.')
    sources = {**base['source_sha256'], **replay.HOOK_SOURCE_HASHES, 'flowr/models/semla.py': head.SEMLA_SHA}
    for name, sha in sources.items():
        require(digest(root / name) == sha, f'Model source changed: {name}')
    for name, sha in base['input_sha256'].items():
        require(digest(name) == sha, f'Original input changed: {name}')
    require(digest(base['args']['ckpt_path']) == base['checkpoint_sha256'], 'Checkpoint differs.')
    require(str(torch.__version__) == base['torch_version'], 'Use the original PyTorch environment.')
    require(str(rdkit.__version__) == base['rdkit_version'], 'Use the original RDKit environment.')
    summary.update(source_sha256=sources, checkpoint_sha256=base['checkpoint_sha256'],
                   reference_summary_sha256=digest(prior_dir / 'summary.json'), metadata={})
    torch.set_float32_matmul_precision('high')
    args = argparse.Namespace(**base['args']); args.save_dir = str(out / 'model_context')
    device = resolve_device(args)
    print(f'Loading original model once on {device}; read-only replay.', flush=True)
    model = load_model(args)[0].to(device).eval()
    # Native requires_grad flags are intentional: no_grad already prevents training.
    decoder = model.gen.ligand_dec
    layout = branches.layer_layout(decoder)
    dm, de, hw = head.inspect_architecture(decoder, torch)
    config = dict(n_layers=len(layout), coord_skip_connect=bool(decoder.coord_skip_connect),
                  pocket_conditioned=decoder.d_pocket_inv is not None,
                  final_norm_zero_com=bool(decoder.final_coord_norm.zero_com),
                  final_norm_eps=float(decoder.final_coord_norm.eps))
    require(config['coord_skip_connect'] and config['pocket_conditioned'] and not config['final_norm_zero_com'],
            'This coordinate adapter does not support the current architecture.')
    cw = dict(coordinate_projection_weight=head.to_array(decoder.coord_out_proj.weight),
              final_norm_set_weights=head.to_array(decoder.final_coord_norm.set_weights))
    if decoder.coord_out_proj.bias is not None:
        cw['coordinate_projection_bias'] = head.to_array(decoder.coord_out_proj.bias)
    for name, weights in [('head_weights.npz', hw), ('coordinate_weights.npz', cw)]:
        np.savez_compressed(out / name, **weights)
    summary.update(layout=layout, coordinate_config=config, d_message=dm, d_edge=de,
                   weights_sha256={n: digest(out / n) for n in ['head_weights.npz', 'coordinate_weights.npz']},
                   device=str(device), torch_version=str(torch.__version__), rdkit_version=str(rdkit.__version__),
                   precision=torch.get_float32_matmul_precision(),
                   parameter_tensors_requiring_grad=sum(v.requires_grad for v in model.parameters()),
                   inpaint_self_condition=bool(model._inpaint_self_condition))
    load = lambda path: torch.load(path, map_location='cpu', weights_only=True)
    tracked = [p['site'], p['partner']]
    for seed in p['seeds']:
        folder = records[seed] / 'attempt_000'
        task, inputs = read_json(folder / 'task_definition.json'), load(folder / 'inputs.pt')
        require(task == prior['metadata'][str(seed)]['task'] and task['mapping_verified'], 'Task mapping changed/unverified.')
        require(set(tracked).issubset(task['fixed_sampler_indices']), 'This plan compares two verified fixed atoms.')
        require(p['site'] in [r['fixed_sampler_index'] for r in task['reference_attachment_sites']],
                'Focus site must be a verified scaffold attachment endpoint.')
        require(inputs['solver'] == 'euler' and int(inputs['sc_recycle_steps']) == 1,
                'Call/frame alignment requires Euler and one forward per integration step.')
        require(max(p['calls']) < int(manifests[seed]['args']['integration_steps']),
                'Window must be inside integration steps, excluding the final corrector.')
        require(bool(model.self_condition) == inputs['self_condition']
                and int(model.sc_recycle_steps) == inputs['sc_recycle_steps'], 'Conditioning settings differ.')
        require(float(inputs['coord_scale']) == prior['metadata'][str(seed)]['coord_scale'], 'Coordinate scale changed.')
        pocket = replay.move_copy(inputs['pocket_data'], device, torch)
        encoding = replay.move_copy(load(folder / 'pocket_encoding.pt'), device, torch)
        require(encoding['pocket_equis'] is not None and encoding['pocket_invs'] is not None, 'Missing original pocket encoding.')
        destination = out / f'seed_{seed}'; destination.mkdir()
        summary['metadata'][str(seed)] = dict(task=task, manifest=manifests[seed], coord_scale=inputs['coord_scale'],
            tracked_atom_indices=tracked, final_inpaint=inputs['final_inpaint'],
            input_sha256=digest(folder / 'inputs.pt'), pocket_encoding_sha256=digest(folder / 'pocket_encoding.pt'))
        final_arrays = {}; save_state_arrays(final_arrays, 'final', load(folder / 'raw_final_prediction.pt'), head.to_array)
        np.savez_compressed(destination / 'final_state.npz', **final_arrays)
        summary['metadata'][str(seed)]['final_state_sha256'] = digest(destination / 'final_state.npz')
        expected_partners = None
        for index in p['calls']:
            print(f'Observe seed {seed}, call {index}, all {len(layout)} layers...', flush=True)
            callpath = folder / 'forward_calls' / f'call_{index:04d}.pt'
            framepath = folder / 'sampler_steps' / f'step_{index:04d}.pt'
            call, frame = load(callpath), load(framepath)
            require(call['call_index'] == index and frame['step_index'] == index
                    and frame['stage'] == 'integrated_state_after_inpainting', 'Call/frame identity mismatch.')
            original = call['network_output_before_postprocessing']
            mask = head.to_array(original['mask'][0])
            partners = [int(i) for i in np.flatnonzero(mask) if i != p['site']]
            require(p['partner'] in partners, 'Focus partner is masked out.')
            if expected_partners is not None:
                require(partners == expected_partners, 'Active partner mapping changed within trajectory.')
            expected_partners = partners
            entry = dict(seed=seed, call=index, times=call['times_scalar'], accepted=False,
                         file=f'seed_{seed}/call_{index:04d}.npz',
                         original_call_sha256=digest(callpath), original_frame_sha256=digest(framepath))
            summary['states'].append(entry)
            baseline = replay.forward_once(model, call, pocket, encoding, device, torch, np)
            entry['replay_vs_original'] = replay.compare_predictions(baseline, original, torch, np, 1e-5, 1e-5)
            require(entry['replay_vs_original']['passed'], 'Native replay differs; stop before interpretation.')
            observers = []
            try:
                ho = head.HeadObserver(decoder, p['site'], partners); observers.append(ho)
                fo = coord.ReadoutObserver(decoder, head.to_array); observers.append(fo)
                bo = branches.SubmoduleObserver(decoder, head.to_array, layout); observers.append(bo)
                xo = replay.HiddenObserver(model, tracked, [p['site']], torch); observers.append(xo)
                so = ScalarBranchObserver(decoder, tracked, p['site'], head.to_array, layout); observers.append(so)
                observed = replay.forward_once(model, call, pocket, encoding, device, torch, np)
            finally:
                for observer in reversed(observers):
                    observer.close()
            logits = head.to_array(observed['bonds'][0])
            ho.arrays.update(site=np.array(p['site']), partners=np.asarray(partners),
                             fixed_indices=np.asarray(task['fixed_sampler_indices']),
                             bond_logits=logits[np.maximum(p['site'], partners), np.minimum(p['site'], partners)])
            for name, key in [('output_coords', 'coords'), ('output_mask', 'mask'), ('output_bond_logits', 'bonds'),
                              ('output_atom_logits', 'atomics'), ('output_charge_logits', 'charges')]:
                fo.arrays[name] = head.to_array(observed[key][0])
            fo.arrays['input_coords'] = head.to_array(call['state_before_forward']['coords'][0])
            extra = {}
            for role, state in [('input', call['state_before_forward']), ('self_condition', call['self_condition']),
                                ('network', observed), ('sampler_after', frame['current'])]:
                save_state_arrays(extra, role, state, head.to_array)
            def persist():
                arrays = dict(extra)
                for prefix, observer in [('h', ho), ('f', fo), ('b', bo), ('x', xo), ('s', so)]:
                    arrays.update({prefix + '__' + k: v for k, v in observer.arrays.items()})
                np.savez_compressed(out / entry['file'], **arrays)
                entry['sha256'] = digest(out / entry['file'])
            persist()
            entry['observation_vs_original'] = replay.compare_predictions(observed, original, torch, np, 1e-5, 1e-5)
            entry['observation_vs_replay'] = replay.compare_predictions(observed, baseline, torch, np, 1e-5, 1e-5)
            require(entry['observation_vs_original']['passed'] and entry['observation_vs_replay']['passed'],
                    'Observation changed prediction; stop.')
            require(all(v == 1 for ob in [ho, fo, bo, so] for v in ob.visits.values())
                    and xo.layer_visits == {i: 1 for i in range(len(layout))}, 'Missing/repeated hooks.')
            head.validate_layout(ho.arrays, dm, de)
            coord.check_readout_arrays(fo.arrays, len(layout), tracked, xo.arrays, ho.arrays)
            _, er = branches.branch_features(bo.arrays, fo.arrays, layout)
            _, sr = scalar_components(so.arrays, xo.arrays, layout, branches)
            coord.readout_reference(fo.arrays, cw, config)
            entry['max_equivariant_addition_rounding'] = max(er.values())
            entry['max_scalar_addition_rounding'] = max(sr.values())
            entry['numerics'] = head.native_head_check(ho, decoder, dm, hw, torch)
            persist()
            require(entry['numerics']['passed'], 'Head numerical reproduction failed; captured data retained.')
            entry['accepted'] = True
            write_json(out / 'summary.json', summary)
            print('  MATCH: saved prediction, passive hooks, internal identities and head numerics.', flush=True)
            del observed, baseline, call, frame, ho, fo, bo, xo, so, observers, extra
    summary['capture_completed'] = len(summary['states']) == len(p['seeds']) * len(p['calls']) and all(e['accepted'] for e in summary['states'])
    require(summary['capture_completed'], 'Incomplete capture.')


def unpack(path, prefix):
    with np.load(path, allow_pickle=False) as z:
        return {k[len(prefix) + 2:]: z[k].copy() for k in z.files if k.startswith(prefix + '__')}


def probabilities(values, logits, head):
    x = np.asarray(values, dtype=np.float64)
    require(np.isfinite(x).all(), 'Non-finite category values.')
    if logits:
        return head.softmax(x), np.ones(x.shape[:-1], dtype=bool)
    require((x >= -1e-7).all(), 'Expected state probabilities/one-hot values, not logits.')
    mass = x.sum(-1)
    known = mass > 1e-8
    require(np.allclose(mass[known], 1., atol=1e-5, rtol=1e-5), 'State categories are not normalized.')
    # Initial all-zero self-conditioning is unknown, never class zero/no-bond.
    return np.divide(x, mass[..., None], out=np.zeros_like(x), where=known[..., None]), known


def state_rows(data, role, seed, call, p, meta, head, default_mask):
    prefix = role + '__'
    if prefix + 'bonds' not in data:
        return [], None
    state = {k[len(prefix):]: v for k, v in data.items() if k.startswith(prefix)}
    mask = state.get('mask', default_mask)
    require(np.isin(mask, [0, 1]).all(), 'Invalid state mask.')
    site = p['site']; fixed = set(meta['task']['fixed_sampler_indices'])
    bp, bk = probabilities(state['bonds'], role == 'network', head)
    ap, ak = probabilities(state['atomics'], role == 'network', head)
    orders = np.asarray([0., 1., 2., 3., 1.5])
    require(bp.shape[-1] == len(orders), 'Unsupported bond classes.')
    rows, hard_orders, expected_orders, external, unknown = [], [], [], [], 0
    for partner in np.flatnonzero(mask):
        partner = int(partner)
        if partner == site:
            continue
        a, b = max(site, partner), min(site, partner)
        known = bool(bk[a, b]); prob = bp[a, b]
        cls = int(prob.argmax()) if known else None
        row = dict(seed=seed, call=call, role=role, site=site, partner=partner,
                   partner_is_fixed=partner in fixed, bond_known=known, bond_class=cls,
                   site_atom_class=int(ap[site].argmax()) if ak[site] else None,
                   partner_atom_class=int(ap[partner].argmax()) if ak[partner] else None,
                   distance_A=float(np.linalg.norm(state['coords'][site] - state['coords'][partner]) * meta['coord_scale']))
        row.update({f'prob_class_{i}': float(prob[i]) if known else None for i in range(5)})
        rows.append(row)
        if partner not in fixed:
            if known:
                hard_orders.append(float(orders[cls])); expected_orders.append(float(prob @ orders))
                if cls:
                    external.append({'slot': partner, 'class': cls, 'atom_class': row['partner_atom_class']})
            else:
                unknown += 1
    summary = dict(seed=seed, call=call, role=role, site=site, unknown_external_pairs=unknown,
                   external_hard_order_sum=sum(hard_orders), external_expected_order_sum=sum(expected_orders),
                   external_bonds=json.dumps(external, separators=(',', ':')),
                   interpretation='Order sums describe recorded states; they are not a chemical validity test.')
    return rows, summary


def analyze(out, summary, modules):
    replay, head, coord, branches = modules
    require(summary['capture_completed'], 'No completed capture to analyze.')
    require(summary['script_sha256'] == digest(Path(__file__)) and summary['adapter_sha256'] == ADAPTER_HASHES,
            'Capture code identity changed.')
    p = summary['plan']; layout = summary['layout']; config = summary['coordinate_config']
    for name, sha in summary['weights_sha256'].items():
        require(digest(out / name) == sha, 'Exported readout weights changed.')
    with np.load(out / 'head_weights.npz', allow_pickle=False) as z:
        hw = {k: z[k].astype(np.float64) for k in z.files}
    with np.load(out / 'coordinate_weights.npz', allow_pickle=False) as z:
        cw = dict(z)
    expected = {(s, c) for s in p['seeds'] for c in p['calls']}
    require(len(summary['states']) == len(expected)
            and {(e['seed'], e['call']) for e in summary['states']} == expected, 'Missing/duplicated states.')
    cache, pair_rows, site_rows, scalar_rows, layer_rows = {}, [], [], [], []
    for entry in summary['states']:
        require(entry['accepted'] and entry['numerics']['passed'], 'Unaccepted state.')
        require(all(entry[k]['passed'] for k in ['replay_vs_original', 'observation_vs_original', 'observation_vs_replay']),
                'Unverified replay.')
        path = out / entry['file']; require(digest(path) == entry['sha256'], 'Captured data changed.')
        seed, call = entry['seed'], entry['call']; meta = summary['metadata'][str(seed)]
        a = {k: unpack(path, k) for k in ['h', 'f', 'b', 'x', 's', 'd']}
        require(int(a['h']['site']) == p['site'], 'Captured site mismatch.')
        valid = [int(i) for i in np.flatnonzero(a['f']['output_mask']) if i != p['site']]
        require(a['h']['partners'].tolist() == valid, 'Partner coverage differs.')
        require(a['h']['fixed_indices'].tolist() == meta['task']['fixed_sampler_indices'], 'Fixed mapping differs.')
        head.validate_layout(a['h'], summary['d_message'], summary['d_edge'])
        require(head.compare_arrays(head.predict_head(a['h']['mlp_input'].astype(np.float64), hw),
                                    a['h']['ieee_head_logits'], atol=2e-4, rtol=1e-5)['passed'],
                'Head reference differs.')
        coord.check_readout_arrays(a['f'], len(layout), meta['tracked_atom_indices'], a['x'], a['h'])
        branches.branch_features(a['b'], a['f'], layout)
        components, _ = scalar_components(a['s'], a['x'], layout, branches)
        a['reference'] = coord.readout_reference(a['f'], cw, config)
        cache[(seed, call)] = a
        for role in ['input', 'self_condition', 'network', 'sampler_after']:
            rows, row = state_rows(a['d'], role, seed, call, p, meta, head, a['f']['output_mask'])
            pair_rows.extend(rows)
            if row is not None:
                site_rows.append(row)
        for (layer, module), value in components.items():
            for j, atom in enumerate(meta['tracked_atom_indices']):
                scalar_rows.append(dict(seed=seed, call=call, layer=layer, module=module, atom=atom,
                                        write_l2_norm=float(np.linalg.norm(value[j])),
                                        interpretation='Descriptive write magnitude; not causal importance.'))
        for item in layout:
            prefix = f"layer_{item['index']:02d}__"
            for partner in valid:
                row = dict(seed=seed, call=call, layer=item['number'], site=p['site'], partner=partner)
                for side in ['in', 'out']:
                    xyz = a['f'][prefix + side + '__coords']
                    row[side + '_distance_A'] = float(np.linalg.norm(xyz[partner] - xyz[p['site']]) * meta['coord_scale'])
                    for direction in ['from', 'to']:
                        edge = a['x'][prefix + side + '__edges__' + direction + '_sites'][0, partner]
                        row[side + '_pair_' + direction + '_l2_norm'] = float(np.linalg.norm(edge))
                row['interpretation'] = 'Coordinates and pair-feature magnitudes are observations, not failure labels.'
                layer_rows.append(row)
    finals = []
    for seed in p['seeds']:
        meta = summary['metadata'][str(seed)]; path = out / f'seed_{seed}/final_state.npz'
        require(digest(path) == meta['final_state_sha256'], 'Final state changed.')
        rows, row = state_rows(unpack(path, 'd'), 'final', seed, -1, p, meta, head,
                               cache[(seed, p['calls'][0])]['f']['output_mask'])
        pair_rows.extend(rows); site_rows.append(row)
        focus = next(r for r in rows if r['partner'] == p['partner'])
        finals.append(dict(seed=seed, final_focus_bond_class=focus['bond_class'],
                           final_focus_bond_matches_expected=focus['bond_class'] == p['expected_bond_class'],
                           final_site_atom_class=focus['site_atom_class'], final_partner_atom_class=focus['partner_atom_class'],
                           interpretation='Focus-bond endpoint only; not overall scaffold-hopping success.'))
    head_rows, coordinate_rows, branch_rows = [], [], []
    for seed in p['seeds']:
        scale = float(summary['metadata'][str(seed)]['coord_scale'])
        for c0, c1 in zip(p['calls'], p['calls'][1:]):
            a, b = cache[(seed, c0)], cache[(seed, c1)]
            require(np.array_equal(a['h']['partners'], b['h']['partners']), 'Partner mapping changed across calls.')
            ident = dict(seed=seed, call_from=c0, call_to=c1, site=p['site'])
            comparisons = [(k, 0) for k in [1, 2, 3, 4]]
            comparisons.append((p['expected_bond_class'], 1 if p['expected_bond_class'] != 1 else 0))
            for target, reference in dict.fromkeys(comparisons):
                rows = head.pair_transition(a['h'], b['h'], hw, summary['d_message'], summary['d_edge'], target, reference)
                for row in rows:
                    if reference == 0 or row['partner'] == p['partner']:
                        head_rows.append({**ident, 'target_class': target, 'reference_class': reference, **row})
            projected = branches.projected_changes(a['b'], b['b'], a['f'], b['f'], a['reference'], b['reference'], layout)
            for partner in a['h']['partners'].tolist():
                ledger, _ = coord.readout_ledger(a['f'], b['f'], a['reference'], b['reference'],
                                                p['site'], partner, scale, len(layout))
                rows = branches.allocate_pair(projected, a['reference'], b['reference'], p['site'], partner, scale)
                residual = sum(v['term_A2'] for v in rows) - ledger['hidden_features_term_A2']
                require(abs(residual) <= 1e-9 * max(1., abs(ledger['hidden_features_term_A2'])), 'Branch ledger does not close.')
                # The legacy helper's second endpoint label is generic here: the focus partner may also be fixed.
                for row in rows:
                    row['partner_endpoint_term_A2'] = row.pop('generated_endpoint_term_A2')
                    row['site_endpoint_term_A2'] = row.pop('fixed_endpoint_term_A2')
                coordinate_rows.append({**ident, 'partner': partner, **ledger, 'branch_residual_A2': float(residual)})
                branch_rows.extend({**ident, 'partner': partner, **row} for row in rows)
    for filename, rows in [('pair_states.csv', pair_rows), ('site_states.csv', site_rows), ('final_focus_status.csv', finals),
                            ('scalar_branch_writes.csv', scalar_rows), ('layer_pair_observations.csv', layer_rows),
                            ('bond_head_changes.csv', head_rows), ('coordinate_changes.csv', coordinate_rows),
                            ('coordinate_submodule_changes.csv', branch_rows)]:
        require(rows, f'Empty analysis table: {filename}')
        head.write_csv(out / filename, rows)
    summary['analysis'] = dict(accepted_states=len(cache), final_focus_status=finals,
        pair_state_rows=len(pair_rows), head_change_rows=len(head_rows), coordinate_change_rows=len(coordinate_rows),
        max_head_accounting_residual=max(abs(r['accounting_residual']) for r in head_rows),
        max_coordinate_accounting_residual=max(abs(r['readout_accounting_residual_A2']) for r in coordinate_rows),
        max_branch_accounting_residual=max(abs(r['branch_residual_A2']) for r in coordinate_rows),
        head_precision_review_rows=sum(r['needs_precision_review'] for r in head_rows))
    make_report(out, p, summary, pair_rows, site_rows)
    summary['analysis_completed'] = True


def make_report(out, p, summary, pairs, sites):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    finals = {r['seed']: r for r in summary['analysis']['final_focus_status']}
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for seed in p['seeds']:
        match = finals[seed]['final_focus_bond_matches_expected']
        color = 'tab:blue' if match else 'tab:red'
        style = dict(color=color, alpha=.45 if match else .95, linewidth=1.2 if match else 2.)
        net = [next(r for r in pairs if r['seed'] == seed and r['call'] == c and r['role'] == 'network'
                    and r['partner'] == p['partner']) for c in p['calls']]
        inp = [next(r for r in pairs if r['seed'] == seed and r['call'] == c and r['role'] == 'input'
                    and r['partner'] == p['partner']) for c in p['calls']]
        ext = [next(r for r in sites if r['seed'] == seed and r['call'] == c and r['role'] == 'network') for c in p['calls']]
        axs[0, 0].plot(p['calls'], [r['distance_A'] for r in net], label=str(seed), **style)
        axs[0, 1].plot(p['calls'], [r['distance_A'] for r in inp], **style)
        axs[1, 0].plot(p['calls'], [r[f"prob_class_{p['expected_bond_class']}"] for r in net], **style)
        axs[1, 1].plot(p['calls'], [r['external_expected_order_sum'] for r in ext], **style)
    titles = ['Network-predicted fixed-pair distance', 'Actual input fixed-pair distance',
              'Network probability of expected fixed bond', 'Network external expected bond-order sum']
    labels = ['Distance (A)', 'Distance (A)', 'Probability', 'Expected order sum (not valence validity)']
    for ax, title, label in zip(axs.flat, titles, labels):
        ax.set(title=title, xlabel='Forward call index (zero based)', ylabel=label)
        ax.set_xticks(p['calls']); ax.grid(alpha=.2)
    axs[0, 0].legend(title='Seed', fontsize=8, ncol=2)
    fig.suptitle('Blue: final focus bond matches; red: differs. This is not overall validity.', fontsize=11)
    fig.savefig(out / 'early_window.png', dpi=150); plt.close(fig)
    pic = base64.b64encode((out / 'early_window.png').read_bytes()).decode('ascii')
    counts = summary['analysis']
    rows = ''.join('<tr><td>' + str(r['seed']) + '</td><td>' + str(r['final_focus_bond_class'])
                   + '</td><td>' + str(r['final_focus_bond_matches_expected']) + '</td></tr>'
                   for r in counts['final_focus_status'])
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>早期内部过程观察</title>
<style>body{{max-width:1120px;margin:36px auto;padding:0 20px;font:16px/1.65 system-ui;color:#213547}}img{{width:100%}}
table{{border-collapse:collapse}}td,th{{padding:7px 18px;border:1px solid #ccd5dd}}code{{background:#edf2f6}}</style>
<h1>固定连接区域：早期内部过程观察</h1>
<p>已有 {len(p['seeds'])} 条轨迹 × {len(p['calls'])} 次调用；接受 {counts['accepted_states']} 个记录。
范围：{html.escape(p['case'])}，固定原子 {p['site']} 与 {p['partner']}，全部解码层和连接位点的全部有效配对。</p>
<p><strong>这是观察与定位，尚不是根因证明，也不是输入原子的 SHAP 贡献。</strong>
已有终态评价标准没有改变。蓝色仅表示最终这条键保持，不能解释为整个分子合格。</p>
<img src="data:image/png;base64,{pic}" alt="早期距离、键概率和外接键趋势">
<p>先比较模型预测与实际输入/约束后状态，再查看几何和成键得分变化对应的层与子模块。
call 0 是第一次网络调用。采样初期的临时图不能套用终态化学有效性来判错。</p>
<table><tr><th>种子</th><th>最终焦点键类别</th><th>与计划要求一致</th></tr>{rows}</table>
<p>键类别：0 无键、1 单键、2 双键、3 三键、4 芳香键。初始全零自条件标为未知，绝不当作无键。</p>
<ul><li><code>pair_states.csv</code>：输入、自条件、模型原始预测、约束后采样状态及终态；原子类别、距离、所有键型概率。</li>
<li><code>site_states.csv</code>：固定连接位点向生成部分接出的全部键。键级总和仅为描述量。</li>
<li><code>bond_head_changes.csv</code>：各成键类别相对无键的得分变化，及焦点键的附加对照；分开记录节点、几何、原子对和数值差异。</li>
<li><code>coordinate_submodule_changes.csv</code>：几何读出中隐藏向量项的层/子模块分解；其余项见 <code>coordinate_changes.csv</code>。</li>
<li><code>scalar_branch_writes.csv</code> 与 <code>layer_pair_observations.csv</code>：标量更新、配对表征和层内坐标观察。范数大不等于贡献大。</li></ul>
<p>分解能加回观测变化，只说明数值核算成立。它不说明这些项是独立原因；不能用这一批数据证明跨分子泛化。
不同种子的生成原子槽位没有身份对应，不应直接按相同编号比较。固定原子对应由原任务映射检查。</p>
<p>本批结束后先审阅统一结果。只有出现可重复、与成功对照区分开的候选现象，才设计下一轮受控因果检验。</p></html>'''
    (out / 'REPORT.html').write_text(text, encoding='utf-8')


def make_bundle(out):
    source = out / 'program'; source.mkdir(exist_ok=True)
    for path in [Path(__file__), *[HERE / 'adapters' / n for n in ADAPTER_HASHES]]:
        dest = source / path.relative_to(HERE); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    bundle = out / 'early_observation_review.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path != bundle and 'model_context' not in path.relative_to(out).parts and '__pycache__' not in path.parts:
                z.write(path, path.relative_to(out))
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=HERE / 'bace_early_plan.json')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/sh_early_bace_lig16_v1'))
    parser.add_argument('--analyze-only', action='store_true', help='Analyze an already accepted capture without loading ROOT.')
    args = parser.parse_args(); out = args.output_dir.resolve()
    if out.exists() and not args.analyze_only:
        parser.error('Output exists; preserve it and choose a new --output-dir.')
    if args.analyze_only and not (out / 'summary.json').is_file():
        parser.error('Missing captured summary for --analyze-only.')
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(version=VERSION, capture_completed=False, analysis_completed=False, completed=False, error=None,
                   states=[], script_sha256=digest(Path(__file__)), adapter_sha256=ADAPTER_HASHES,
                   scope='Read-only saved-state replay; observational decomposition, not a causal/root-cause detector.')
    try:
        modules = adapters()
        if args.analyze_only:
            summary = read_json(out / 'summary.json')
            summary.update(analysis_completed=False, completed=False, error=None)
        else:
            summary['plan'] = read_plan(args.plan)
            write_json(out / 'plan.json', summary['plan'])
            capture(summary['plan'], out, summary, modules)
        analyze(out, summary, modules)
        summary['completed'] = summary['capture_completed'] and summary['analysis_completed']
        if (out / 'error.txt').is_file():
            (out / 'error.txt').unlink()
    except Exception as exc:
        summary['error'] = f'{type(exc).__name__}: {exc}'
        (out / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('STOP:', summary['error'], flush=True)
    finally:
        write_json(out / 'summary.json', summary)
        bundle = make_bundle(out)
        total = len(summary.get('plan', {}).get('seeds', [])) * len(summary.get('plan', {}).get('calls', []))
        print(f"Capture completed: {summary['capture_completed']}")
        print(f"Analysis completed: {summary['analysis_completed']}")
        print(f"Completed: {summary['completed']}")
        print(f"Accepted calls: {sum(bool(e.get('accepted')) for e in summary['states'])}/{total}")
        print(f'Upload: {bundle}')
    return 0 if summary['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
