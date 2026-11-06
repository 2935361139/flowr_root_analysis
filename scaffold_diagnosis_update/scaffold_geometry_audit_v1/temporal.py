"""Track FINAL native clash pairs through saved states; no causal-onset claim."""
import numpy as np
from flowr_adapter import require, sha


def suffix(points):
    require(bool(points), 'Empty observation series')
    first = next((p for p in points if p['value'] is True), None)
    if points[-1]['value'] is not True:
        return dict(status='terminal_signature_absent_or_unknown', first_observed=first,
                    first_persistent_observation=None, causal_onset=None)
    i = len(points)-1
    while i and points[i-1]['value'] is True:
        i -= 1
    return dict(status='observed', first_observed=first,
                first_persistent_observation=points[i], last_observation=points[-1],
                preceding_observation=points[i-1] if i else None,
                boundary_is_confirmed_change=i > 0 and points[i-1]['value'] is False,
                causal_onset=None)


def series(xyz, classes, valid, vocab, slot, element, environment_xyz,
           radius_sum_scaled, cutoff, metadata):
    n = len(xyz)
    require(metadata['indices'] == list(range(n)), 'Nonconsecutive temporal indices')
    require(len(metadata['times']) == len(metadata['stages']) == n, 'Incomplete time metadata')
    require(classes.shape == valid.shape == xyz.shape[:2] and xyz.shape[-1] == 3,
            'Trajectory array shape mismatch')
    require(0 <= slot < xyz.shape[1], 'Unmapped clash slot')
    require(np.isfinite(xyz).all() and radius_sum_scaled > 0, 'Invalid coordinates/radii')
    points = []
    for k in range(n):
        cls = int(classes[k, slot])
        require(0 <= cls < len(vocab), 'Invalid atom vocabulary index')
        same = bool(valid[k, slot]) and vocab[cls] == element
        distance = float(np.linalg.norm(xyz[k, slot] - environment_xyz))
        # Numerical ambiguity is unknown, never a new molecular failure threshold.
        margin = distance - radius_sum_scaled * cutoff
        value = bool(margin < 0) if same and abs(margin) > 1e-5 else None
        points.append(dict(index=k, native_time=metadata['times'][k], stage=metadata['stages'][k],
                           value=value, distance_A=distance,
                           relative_distance=distance/radius_sum_scaled,
                           endpoint_element_present=same))
    return points


def track_pairs(item, pairs, env, modules):
    import torch
    require(item['manifest']['torch_version'] == str(torch.__version__), 'Use recorded Torch version')
    path = item['folder'] / 'inputs.pt'
    inputs = torch.load(path, map_location='cpu', weights_only=True)
    require(len(inputs['pocket_com']) == 1, 'Expected exactly one recorded pocket origin')
    origin = inputs['pocket_com'][0].detach().cpu().numpy().reshape(3).astype(float)
    require(np.isfinite(origin).all(), 'Invalid pocket origin')
    from rdkit import Chem
    table = Chem.GetPeriodicTable()
    config = {m['rename_outputs']['no_clashes'].lower().replace(' ', '_'):m['parameters']
              for m in modules if m['function'] == 'intermolecular_distance'}
    path_arrays = item['diag_dir']/'connection_arrays.npz'
    output = []
    with np.load(path_arrays, allow_pickle=False) as a:
        valid = a['raw_final_valid'].astype(bool)
        require(np.allclose(a['sampler_state_coords'][-1][valid].astype(float)+origin,
                            a['raw_final_coords'][valid], atol=2e-5, rtol=0),
                'Final trajectory cannot reproduce original coordinate frame')
        require(np.array_equal(a['sampler_state_atom_classes'][-1][valid],
                               a['raw_final_atom_classes'][valid]), 'Terminal atom identities differ')
        for pair in pairs:
            if pair['stage'] != 'raw_without_repair':
                continue
            slot = pair['sampler_atom_index']; other = pair['environment_atom_index']
            params = config[pair['check']]
            radius = table.GetRvdw if params.get('radius_type','vdw') == 'vdw' else table.GetRcovalent
            denom = (radius(pair['element']) + radius(pair['environment_element'])) * params.get('radius_scale',1.)
            env_xyz = env.GetConformer().GetPositions()[other]
            distance = float(np.linalg.norm(a['raw_final_coords'][slot].astype(float)-env_xyz))
            require(abs(distance-pair['distance_A']) < 2e-5, 'Endpoint pair distance mismatch')
            require(abs(distance/denom-pair['relative_distance']) < 2e-5, 'Native radius formula mismatch')
            for stream, metadata in item['report']['streams'].items():
                points = series(a[stream+'_coords'].astype(float)+origin,
                                a[stream+'_atom_classes'], a[stream+'_valid'],
                                item['report']['atom_vocabulary'], slot, pair['element'], env_xyz,
                                denom, params['clash_cutoff'], metadata)
                output.append(dict(id=item['id'], stage='raw_without_repair', check=pair['check'],
                    stream=stream, entity=f"slot:{slot}/environment:{other}",
                    sampler_atom_index=slot, environment_atom_index=other,
                    signature='final_pair_clash', points=points, **suffix(points),
                    scope='Same final atom pair and element; not the earliest clash anywhere in the molecule. Intermediate noisy states are observations, not failed molecules.',
                    internal_module=None, causal_status='not_tested'))
    return output, {str(path):sha(path), str(path_arrays):sha(path_arrays)}
