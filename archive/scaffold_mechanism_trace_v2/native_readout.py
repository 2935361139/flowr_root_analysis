"""Use the live ROOT head, its device/dtype and full tensor shape.

NumPy is used only for statistics/optional diagnostics, never to compute the
counterfactual head logits used for attribution.
"""
from __future__ import annotations
import numpy as np
from trace_core import require, group_mask, summarize_coalitions


def to_numpy(t):
    return t.detach().cpu().float().numpy().copy()


def compare(actual, expected, categorical=False, atol=1e-5, rtol=1e-5):
    a, b = to_numpy(actual).astype(np.float64), to_numpy(expected).astype(np.float64)
    if a.shape != b.shape:
        return dict(passed=False, actual_shape=list(a.shape), expected_shape=list(b.shape))
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    diff = np.abs(a-b) if finite else None
    bad = diff > atol+rtol*np.abs(b) if finite else None
    changes = int(np.count_nonzero(a.argmax(-1) != b.argmax(-1))) if finite and categorical else None
    return dict(passed=bool(finite and not bad.any() and (changes is None or changes == 0)),
                finite=finite, shape=list(a.shape), actual_dtype=str(actual.dtype), expected_dtype=str(expected.dtype),
                max_abs_difference=float(diff.max()) if finite and diff.size else (0. if finite else None),
                tolerance_violations=int(bad.sum()) if finite else None, argmax_changes=changes,
                atol=atol, rtol=rtol)


def pair_values(t, pairs):
    f, g = pairs[:, 0].tolist(), pairs[:, 1].tolist()
    return t[0, [max(a,b) for a,b in zip(f,g)], [min(a,b) for a,b in zip(f,g)]]


class HeadCapture:
    def __init__(self, decoder, torch):
        self.torch, self.tensors, self.visits, self.handles = torch, {}, {}, []
        mlp = decoder.bond_refine.message_mlp
        for name, mod, is_pre in [('mlp_input', mlp, True), ('linear0', mlp[0], False),
                                  ('activation', mlp[1], False), ('mlp_output', mlp, False),
                                  ('projection_input', decoder.bond_proj, True),
                                  ('projection_output', decoder.bond_proj, False)]:
            def pre(module, args, key=name):
                self.put(key, args[0])
                return None
            def post(module, args, output, key=name):
                self.put(key, output)
                return None
            self.handles.append(mod.register_forward_pre_hook(pre) if is_pre else mod.register_forward_hook(post))

    def put(self, name, value):
        self.visits[name] = self.visits.get(name, 0)+1
        self.tensors[name] = value.detach().clone(memory_format=self.torch.preserve_format)

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def native_head(decoder, x, adjacency, torch):
    """Exactly the three native calls/order in the checked ROOT source."""
    with torch.no_grad():
        hidden = decoder.bond_refine.message_mlp(x)
        symmetric = 0.5*(hidden+hidden.transpose(1, 2))
        projected = decoder.bond_proj(symmetric)
        return projected*adjacency.unsqueeze(-1)


def validate_head(decoder, capture, original_bonds, adjacency, torch):
    require(capture.visits == {k:1 for k in ('mlp_input', 'linear0', 'activation', 'mlp_output',
                                          'projection_input', 'projection_output')}, 'Head call counts changed')
    x = capture.tensors['mlp_input']
    require(x.ndim == 4 and x.shape[0] == 1, 'Expected full [1,N,N,F] head input')
    stages = {}
    with torch.no_grad():
        h0 = decoder.bond_refine.message_mlp[0](x)
        stages['linear0'] = compare(h0, capture.tensors['linear0'])
        h1 = decoder.bond_refine.message_mlp[1](h0)
        stages['activation'] = compare(h1, capture.tensors['activation'])
        h2 = decoder.bond_refine.message_mlp[2](h1)
        stages['mlp_output'] = compare(h2, capture.tensors['mlp_output'])
        sym = 0.5*(h2+h2.transpose(1, 2))
        stages['projection_input'] = compare(sym, capture.tensors['projection_input'])
        proj = decoder.bond_proj(sym)
        stages['projection_output'] = compare(proj, capture.tensors['projection_output'], categorical=True)
        stages['mask_and_network_output'] = compare(proj*adjacency.unsqueeze(-1), original_bonds, categorical=True)
        stages['full_native_head'] = compare(native_head(decoder, x, adjacency, torch), original_bonds, categorical=True)
    return dict(passed=all(v['passed'] for v in stages.values()), stages=stages,
                input_shape=list(x.shape), input_stride=list(x.stride()), input_dtype=str(x.dtype),
                device=str(x.device), matmul_precision=torch.get_float32_matmul_precision(),
                autocast_enabled=bool(torch.is_autocast_enabled(x.device.type)))


def select_pair_features(x, pairs):
    f, g = pairs[:, 0].tolist(), pairs[:, 1].tolist()
    return np.stack((to_numpy(x[0, f, g]), to_numpy(x[0, g, f])), axis=1)


def evaluate_coalitions(decoder, before, after, pairs, d_message, torch,
                        callback=None, receipt_callback=None):
    """Modify selected message features; other rows stay at the after-state.

    Each coalition is evaluated separately at the original full [1,N,N,F]
    shape. This avoids changing GEMM dimensions by packing/slicing pairs.
    """
    x0, x1 = before['features'], after['features']
    require(x0.shape == x1.shape and x0.dtype == x1.dtype and x0.device == x1.device,
            'Head tensor identity differs between adjacent states')
    require(torch.equal(before['adjacency'], after['adjacency']), 'Node-mask adjacency changed; unsupported temporal comparison')
    f, g = pairs[:, 0].tolist(), pairs[:, 1].tolist()
    masks = torch.as_tensor(group_mask(d_message, x1.shape[-1]), device=x1.device)
    before_selected = pair_values(before['network_bonds'], pairs)
    after_selected = pair_values(after['network_bonds'], pairs)
    values, receipts, extra = [], {}, []
    state_cpu = torch.get_rng_state().clone()
    state_gpu = torch.cuda.get_rng_state(x1.device).clone() if x1.device.type == 'cuda' else None
    with torch.no_grad():
        for coalition in range(16):
            x = x1.detach().clone(memory_format=torch.preserve_format)
            choose = ((coalition >> masks) & 1).bool()
            x[0, f, g] = torch.where(choose[0], x1[0, f, g], x0[0, f, g])
            x[0, g, f] = torch.where(choose[1], x1[0, g, f], x0[0, g, f])
            bonds = native_head(decoder, x, after['adjacency'], torch)
            require(bool(torch.isfinite(bonds).all()), 'Nonfinite native coalition logits')
            selected = pair_values(bonds, pairs)
            if coalition == 0:
                receipts['previous_pair_logits'] = compare(selected, before_selected, categorical=True)
                if receipt_callback: receipt_callback(receipts)
                require(receipts['previous_pair_logits']['passed'], 'Previous-state selected scores do not reproduce; inspect coalition_checks.json')
            elif coalition == 15:
                receipts['after_full_logits'] = compare(bonds, after['network_bonds'], categorical=True)
                receipts['after_pair_logits'] = compare(selected, after_selected, categorical=True)
                if receipt_callback: receipt_callback(receipts)
                require(receipts['after_full_logits']['passed'] and receipts['after_pair_logits']['passed'],
                        'Identity coalition does not reproduce the original full bond logits')
            values.append(to_numpy(selected).astype(np.float64))
            if callback is not None:
                extra.append(callback(coalition, bonds))
            del x, bonds
    receipts['torch_rng_unchanged'] = bool(torch.equal(state_cpu, torch.get_rng_state()))
    if state_gpu is not None:
        receipts['torch_rng_unchanged'] &= bool(torch.equal(state_gpu, torch.cuda.get_rng_state(x1.device)))
    require(receipts['torch_rng_unchanged'], 'Deterministic head evaluation consumed random state')
    if receipt_callback: receipt_callback(receipts)
    return summarize_coalitions(np.stack(values)), receipts, extra
