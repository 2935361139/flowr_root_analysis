"""Bounded terminal coordinate intervention; independent of a neural architecture.

This is a rescue experiment, not an inference-time network intervention. Native
PoseBusters still decides success; this objective is never used as its substitute.
"""
import numpy as np

SETTINGS = dict(max_iterations=80, max_objective_calls=200,
                max_component_shift_A=2.0, repulsion_weight=500.0,
                tether_weight=2.0, clearance_margin_A=0.1)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def repulsion(xyz, env_xyz, cutoffs, weight):
    """Squared overlap penalty with an analytic Cartesian gradient."""
    delta = xyz[:, None, :] - env_xyz[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=-1) + 1e-24)
    overlap = np.maximum(cutoffs - distances, 0.0)
    energy = 0.5 * weight * np.sum(overlap * overlap)
    gradient = -weight * np.sum((overlap / distances)[:, :, None] * delta, axis=1)
    return float(energy), gradient


def environment_terms(mol, environment, config):
    """Match this checkout's native PB radius and environment selections."""
    from posebusters.tools.protein import get_atom_type_mask
    from posebusters.modules.intermolecular_distance import _get_radii
    xyz = np.asarray(environment.GetConformer().GetPositions(), dtype=float)
    require(np.isfinite(xyz).all(), 'Nonfinite environment coordinates')
    ligand_elements = np.array([a.GetSymbol() for a in mol.GetAtoms()])
    env_elements = np.array([a.GetSymbol() for a in environment.GetAtoms()])
    terms = []
    for module in config['modules']:
        if module['function'] != 'intermolecular_distance':
            continue
        p = module.get('parameters', {})
        mask = np.asarray(get_atom_type_mask(environment, p.get('ignore_types', ['hydrogens'])), dtype=bool)
        radius_type = p.get('radius_type', 'vdw')
        radius = _get_radii(ligand_elements, radius_type)[:, None] + _get_radii(env_elements[mask], radius_type)[None, :]
        # The margin shapes the intervention only. Evaluation thresholds stay unchanged.
        cutoff = radius * p.get('radius_scale', 1.0) * p.get('clash_cutoff', 0.75)
        terms.append(dict(label=module['name'], coords=xyz[mask],
                          cutoff=cutoff + SETTINGS['clearance_margin_A']))
    require(len(terms) > 0, 'Missing native intermolecular-distance configuration')
    return terms


def bounded_minimize(objective, x0, bounds, max_iterations, max_calls):
    """One attempt. Budget exhaustion returns the last accepted iterate."""
    from scipy.optimize import minimize
    require(max_iterations > 0 and max_calls > 0, 'Invalid optimization budget')
    accepted = np.asarray(x0, dtype=float).copy()
    calls = 0
    iterations = 0

    class BudgetReached(Exception):
        pass

    def fun(x):
        nonlocal calls
        if calls >= max_calls:
            raise BudgetReached()
        calls += 1
        e, g = objective(x)
        require(np.isfinite(e) and np.isfinite(g).all(), 'Nonfinite optimization objective/gradient')
        return e, g

    def callback(x):
        nonlocal accepted, iterations
        accepted = x.copy()
        iterations += 1

    try:
        result = minimize(fun, accepted, jac=True, method='L-BFGS-B', bounds=bounds,
                          callback=callback, options=dict(maxiter=max_iterations,
                          maxfun=max_calls, maxls=20, ftol=1e-9, gtol=1e-4))
        accepted = np.asarray(result.x, dtype=float)
        converged = bool(result.success)
        message = str(result.message)
    except BudgetReached:
        converged = False
        message = 'Hard objective-call budget reached; using last accepted iterate'
    require(np.isfinite(accepted).all(), 'Nonfinite optimizer output')
    require(calls <= max_calls and iterations <= max_iterations, 'Optimization exceeded preset budget')
    return accepted, dict(objective_calls=calls, iterations=iterations,
                           converged=converged, message=message)


def relax_generated(mol, fixed_heavy_indices, terms, use_environment=True):
    """Hold every fixed heavy atom exact; optimize generated atoms and auxiliary Hs.

    Hydrogens are discarded afterwards and recreated by the unchanged evaluator.
    No atom/bond/charge changes, protein movement, retries, or outcome-based tuning.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    heavy = Chem.Mol(mol)
    require(heavy.GetNumConformers() == 1, 'Expected one conformer')
    require(all(a.GetAtomicNum() > 1 for a in heavy.GetAtoms()), 'Expected heavy-atom input')
    n = heavy.GetNumAtoms()
    fixed = sorted(fixed_heavy_indices)
    require(len(set(fixed)) == len(fixed) and all(0 <= i < n for i in fixed), 'Invalid fixed indices')
    generated = [i for i in range(n) if i not in fixed]
    require(len(generated) > 0, 'No generated heavy atoms')
    with_h = Chem.AddHs(heavy, addCoords=True)
    require(AllChem.UFFHasAllMoleculeParams(with_h), 'UFF parameters unavailable; no substitute or retry')
    ff = AllChem.UFFGetMoleculeForceField(with_h)
    require(ff is not None, 'UFF initialization failed')
    ff.Initialize()
    initial = np.asarray(with_h.GetConformer().GetPositions(), dtype=float)
    require(np.isfinite(initial).all(), 'Nonfinite ligand coordinates')
    variable = generated + list(range(n, len(initial)))
    x0 = initial[variable].reshape(-1).copy()

    def objective(flat):
        x = initial.copy()
        x[variable] = flat.reshape(-1, 3)
        packed = x.reshape(-1).tolist()
        energy = float(ff.CalcEnergy(packed))
        grad = np.asarray(ff.CalcGrad(packed), dtype=float).reshape(-1, 3)
        if use_environment:
            for term in terms:
                e, g = repulsion(x[:n], term['coords'], term['cutoff'], SETTINGS['repulsion_weight'])
                energy += e
                grad[:n] += g
        displacement = x[generated] - initial[generated]
        energy += 0.5 * SETTINGS['tether_weight'] * float(np.sum(displacement * displacement))
        grad[generated] += SETTINGS['tether_weight'] * displacement
        return energy, grad[variable].reshape(-1)

    shift = SETTINGS['max_component_shift_A']
    bounds = list(zip(x0 - shift, x0 + shift))
    final, receipt = bounded_minimize(objective, x0, bounds,
        SETTINGS['max_iterations'], SETTINGS['max_objective_calls'])
    result = initial.copy()
    result[variable] = final.reshape(-1, 3)
    require(np.array_equal(result[fixed], initial[fixed]), 'Fixed coordinates changed')
    require(np.max(np.abs(result[generated] - initial[generated])) <= shift + 1e-9, 'Shift bound exceeded')
    receipt.update(settings=SETTINGS.copy(), use_environment=bool(use_environment),
        fixed_heavy_indices=fixed, generated_heavy_indices=generated,
        only_generated_heavy_coordinates_changed=True,
        max_generated_displacement_A=float(np.linalg.norm(result[generated] - initial[generated], axis=1).max()),
        auxiliary_hydrogens='Optimized internally; discarded and rebuilt by native evaluator')
    return result[:n].copy(), receipt


def successful(row):
    return all(row.get(k) is True for k in ('rdkit_valid', 'fc_valid',
        'root_fixed_substructure_match', 'fixed_chemistry_preserved', 'posebusters_valid'))


GEOMETRY_CHECKS = frozenset(('bond_lengths','bond_angles','internal_steric_clash',
    'aromatic_ring_flatness','double_bond_flatness','internal_energy',
    'minimum_distance_to_protein','minimum_distance_to_waters',
    'minimum_distance_to_organic_cofactors','minimum_distance_to_inorganic_cofactors',
    'volume_overlap_with_protein','volume_overlap_with_waters',
    'volume_overlap_with_organic_cofactors','volume_overlap_with_inorganic_cofactors'))
PRIMARY_ARMS = ('A','B','AB','C')


def route_failure(row):
    """Route by original checks, never by a case name or one chosen atom/water."""
    if successful(row):
        return dict(status='already_passed', reason='Keep original result; eligible as a same-input control')
    chemical = ('rdkit_valid','fc_valid','root_fixed_substructure_match','fixed_chemistry_preserved')
    if not all(row.get(k) is True for k in chemical):
        return dict(status='unsupported', reason='Chemical/connectivity/fixed-identity failure needs a discrete or model intervention; coordinate code cannot claim to repair it')
    failed = set(row.get('posebusters_failed_checks') or [])
    if row.get('posebusters_valid') is False and failed and failed <= GEOMETRY_CHECKS:
        return dict(status='supported_geometry', reason='Run frozen geometry intervention bank, including fixed-region conflicts')
    return dict(status='unsupported', reason='Missing evaluation or failure check outside this intervention bank')


def select_result(baseline, arms):
    """Keep good original samples; choose first passing primary by fixed priority.

    Controls never supply a repaired output. All passing interventions are reported.
    C changes fixed coordinates, so mark its conditioning constraint relaxation.
    """
    passing = [name for name in PRIMARY_ARMS if name in arms and successful(arms[name])]
    if successful(baseline):
        return dict(selected_arm='baseline', result='original_pass',
                    all_successful_interventions=passing, fixed_coordinate_relaxation=False)
    if passing:
        selected = passing[0]
        return dict(selected_arm=selected, result='rescued_under_original_evaluator',
                    all_successful_interventions=passing, fixed_coordinate_relaxation=selected=='C')
    return dict(selected_arm=None, result='unresolved',
                all_successful_interventions=[], fixed_coordinate_relaxation=False)


def classify_rescue(baseline, arms):
    """Keep multiple independent rescuers; never split a joint-only rescue."""
    failed = not successful(baseline)
    independent = [name for name in ('A', 'B') if failed and name in arms and successful(arms[name])]
    joint = failed and 'AB' in arms and successful(arms['AB'])
    return dict(baseline_success=successful(baseline),
        independently_effective_candidates=independent,
        combination_success=bool(joint), combination_only=bool(joint and not independent),
        interpretation='Operational terminal rescue; unique upstream/neural cause not established')
