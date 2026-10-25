"""Reused native graph/chemistry checks from the existing boundary audit."""
import numpy as np

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()

def graph(data, atom_tokens, charge_tokens):
    """Match native prefix masks, virtual-atom removal and LOWER-triangle argmax."""
    mask = array(data['mask'])[0].astype(bool)
    n = len(mask)
    require(np.array_equal(mask, np.arange(n) < int(mask.sum())), 'Native builder expects a prefix mask; unsupported record')
    atomic = array(data['atomics'])[0]
    charges = array(data['charges'])[0]
    bonds = array(data['bonds'])[0]
    coords = array(data['coords'])[0]
    require(bonds.shape == (n, n, 5) and coords.shape == (n, 3), 'Unsupported tensor dimensions')
    require(all(np.isfinite(x).all() for x in [atomic, charges, bonds, coords]), 'Nonfinite saved computational tensor')
    require(atomic.shape[-1] == len(atom_tokens) and charges.shape[-1] == len(charge_tokens), 'Vocabulary mismatch')
    elements = [atom_tokens[int(i)] for i in atomic.argmax(-1)]
    q = [charge_tokens[int(i)] for i in charges.argmax(-1)]
    keep = mask & np.array([a != '<NOATOM>' for a in elements])
    lower = np.tril(bonds.argmax(-1), -1)
    classes = lower + lower.T
    classes = classes * keep[:, None] * keep[None, :]
    return {'elements': elements, 'charges': q, 'keep': keep, 'slots': np.flatnonzero(keep).tolist(),
            'classes': classes, 'coords': coords, 'bond_values': bonds}

def native_unsanitized(data, builder):
    parts = list(builder._extract_mols(data['coords'], data['atomics'], data['mask'],
                 bond_dists=data['bonds'], charge_dists=data['charges'],
                 hybridization_dists=data.get('hybridization')))
    require(len(parts) == 1, 'Expected one molecule')
    return builder._mol_from_tensors(*parts[0], sanitise=False, add_hs=False, valence_repair=False)

def chemistry(data, builder, g, Chem, rdBase, mol_is_valid):
    # Block normal RDKit error messages from flooding the console; retain messages below.
    with rdBase.BlockLogs():
        mol = native_unsanitized(data, builder)
        if mol is None:
            return {'native_buildable': False, 'native_rdkit_valid': False, 'problems': [],
                    'status': 'native_unsanitized_builder_returned_none'}, None
        require(mol.GetNumAtoms() == len(g['slots']), 'Native molecule-to-sampler atom mapping differs')
        for i, slot in enumerate(g['slots']):
            require(mol.GetAtomWithIdx(i).GetSymbol() == g['elements'][slot], 'Native atom order differs')
        problems = []
        for problem in Chem.DetectChemistryProblems(Chem.Mol(mol)):
            if hasattr(problem, 'GetAtomIdx'):
                ids = [int(problem.GetAtomIdx())]
            elif hasattr(problem, 'GetAtomIndices'):
                ids = [int(i) for i in problem.GetAtomIndices()]
            else:
                ids = []
            require(all(0 <= i < len(g['slots']) for i in ids), 'Unmapped RDKit problem atom')
            problems.append({'type': str(problem.GetType()), 'message': str(problem.Message()),
                             'sampler_slots': [g['slots'][i] for i in ids]})
        return {'native_buildable': True, 'native_rdkit_valid': bool(mol_is_valid(mol, connected=False)),
                'problems': problems, 'status': 'assessed'}, mol
