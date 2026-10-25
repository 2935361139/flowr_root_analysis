"""Protocol and exact four-group decomposition of an observed ROOT bond readout.

The attributed quantity is a head score change, not final failure causation.
No reference-identical scaffold, atom valence threshold, or sample ID is coded here.
"""
from __future__ import annotations
import math
import numpy as np

GROUPS = ('fixed_node', 'generated_node', 'geometry', 'edge_context')
PREFIX = 'fixed_chemistry_problem:AtomValenceException:'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def status(frame, event):
    # A failed assessment is never a negative observation.
    if frame['chemistry']['status'] != 'assessed':
        return None
    return event in frame['events']


def build_plan(records):
    """All observed onset/recovery transitions, plus a fixed-rule control.

    Controls are the lowest recorded seed with no occurrence of that site event
    under the same input. Generated atom identities are never aligned across seeds.
    Selection uses outcomes: this is DISCOVERY, not independent validation.
    """
    jobs, skipped = [], []
    for rid, rec in sorted(records.items()):
        frames, report = rec['frames'], rec['report']
        require([f['call'] for f in frames] == list(range(len(frames))), f'{rid}: nonsequential calls')
        events = sorted({e for f in frames for e in f['events'] if e.startswith(PREFIX)})
        for event in events:
            site = int(event.rsplit(':', 1)[1])
            require(str(site) in report['fixed_mapping'], f'{rid}: event is not on a fixed site')
            first = status(frames[0], event)
            if first:
                skipped.append(dict(run=rid, event=event, call=0, reason='No previous call; no temporal contrast is possible.'))
            for k in range(1, len(frames)):
                a, b = status(frames[k-1], event), status(frames[k], event)
                if a is None or b is None:
                    if a != b:
                        skipped.append(dict(run=rid, event=event, call=k, reason='Unknown assessment; not a normal/abnormal transition.'))
                    continue
                if a == b:
                    continue
                terminal = b and all(status(f, event) is True for f in frames[k:])
                jobs.append(dict(run=rid, before=k-1, after=k, site=site, event=event,
                                 kind='onset' if b else 'recovery', terminal_segment_onset=bool(terminal)))
    controls = {}
    for job in jobs:
        src = records[job['run']]['report']
        ref = src['fixed_mapping'][str(job['site'])]['reference_index']
        candidates = []
        for rid, rec in records.items():
            r = rec['report']
            if (r['case'], r['ligand_idx']) != (src['case'], src['ligand_idx']) or rid == job['run']:
                continue
            sites = [int(s) for s, v in r['fixed_mapping'].items() if v['reference_index'] == ref]
            if len(sites) != 1:
                continue
            event = PREFIX + str(sites[0])
            if all(status(f, event) is False for f in rec['frames']):
                candidates.append((r['seed'], rid, sites[0], event))
        if candidates:
            _, rid, site, event = min(candidates)
            key = (rid, job['before'], job['after'], site)
            controls.setdefault(key, dict(run=rid, before=job['before'], after=job['after'], site=site,
                                         event=event, kind='same_input_no_event_control',
                                         terminal_segment_onset=False, matched_events=[]))
            controls[key]['matched_events'].append(dict(run=job['run'], after=job['after'], kind=job['kind']))
        else:
            skipped.append(dict(run=job['run'], event=job['event'], call=job['after'], reason='No assessed same-input event-free control.'))
    jobs.extend(controls.values())
    for i, j in enumerate(jobs):
        j['id'] = f'contrast_{i:03d}'
    captures = {rid: set() for rid in records}
    for j in jobs:
        captures[j['run']].update((j['before'], j['after']))
    # All terminal captures provide coverage, including inputs with no target event.
    for rid, rec in records.items():
        captures[rid].add(len(rec['frames']) - 1)
    return dict(jobs=jobs, captures={k: sorted(v) for k, v in sorted(captures.items())}, skipped=skipped,
                scope='Exploratory local bond-readout attribution; no full-rollout intervention and no root-cause label.')


def group_mask(d_message, feature_dim):
    require(feature_dim > 2*d_message+2, 'Unexpected BondRefine feature dimensions')
    out = np.empty((2, feature_dim), dtype=np.int64)
    # direction 0: fixed -> generated; direction 1: generated -> fixed.
    out[0, :d_message], out[1, :d_message] = 0, 1
    out[0, d_message:2*d_message], out[1, d_message:2*d_message] = 1, 0
    out[:, 2*d_message:2*d_message+2] = 2
    out[:, 2*d_message+2:] = 3
    return out


def silu(x):
    # Stable sigmoid avoids spurious overflow in offline float64 reconstruction.
    pos = x >= 0
    s = np.empty_like(x)
    s[pos] = 1/(1+np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    s[~pos] = ex/(1+ex)
    return x*s


def readout(x, weights):
    """x: [pairs, two directions, features], output: [pairs, bond classes]."""
    h = silu(x @ weights['w0'].T + weights['b0'])
    h = h @ weights['w2'].T + weights['b2']
    return h.mean(axis=1) @ weights['wp'].T + weights['bp']


def probabilities(logits):
    v = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return v/v.sum(axis=-1, keepdims=True)


def exact_shapley(values):
    """16 coalitions, four groups. Interactions shared by all insertion orders."""
    require(values.shape[0] == 16, 'Four groups require all 16 coalitions')
    phi = np.zeros((4, *values.shape[1:]), dtype=np.float64)
    for g in range(4):
        for s in range(16):
            if s & (1 << g):
                continue
            n = s.bit_count()
            weight = math.factorial(n)*math.factorial(3-n)/math.factorial(4)
            phi[g] += weight*(values[s | (1 << g)] - values[s])
    require(np.allclose(phi.sum(axis=0), values[15]-values[0], atol=1e-8, rtol=1e-8),
            'Attributions do not sum to the actual score change')
    return phi


def decompose(before, after, weights, d_message):
    require(before.shape == after.shape and before.ndim == 3 and before.shape[1] == 2,
            'Need corresponding within-trajectory pairs in both directions')
    masks = group_mask(d_message, before.shape[-1])
    logits = []
    for coalition in range(16):
        choose_after = ((coalition >> masks) & 1).astype(bool)
        x = np.where(choose_after[None], after, before)
        logits.append(readout(x, weights))
    return summarize_coalitions(np.stack(logits))


def summarize_coalitions(logits):
    """Shared score accounting; real experiments supply native Torch logits."""
    require(logits.shape[-1] == 5 and np.isfinite(logits).all(), 'Unexpected/nonfinite bond scores')
    probs = probabilities(logits)
    expected_order = probs @ np.array([0., 1., 2., 3., 1.5])
    # A fixed contrast: the after-state's strongest nonzero bond class vs no-bond.
    target = logits[15, :, 1:].argmax(-1)+1
    margin = np.take_along_axis(logits, np.broadcast_to(target[None, :, None], (16, len(target), 1)), -1)[..., 0]-logits[..., 0]
    return dict(logits=logits, expected_order=expected_order,
                order_phi=exact_shapley(expected_order), margin=margin,
                margin_phi=exact_shapley(margin), target_class=target)
