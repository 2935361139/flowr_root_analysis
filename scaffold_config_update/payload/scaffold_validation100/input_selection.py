"""Input eligibility only. Does not read old outputs or exclude past experiments."""

import math

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def heavy_mol(mol, Chem):
    require(mol is not None, 'Unreadable molecule')
    result = Chem.RemoveHs(Chem.Mol(mol))
    Chem.SanitizeMol(result)
    return result

def canonical(mol, Chem):
    copy = heavy_mol(mol, Chem)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True)

def input_profile(mol, Chem, extract_scaffolds):
    """Input-only eligibility; preview indices must NEVER be treated as sampler slots."""
    mol = heavy_mol(mol, Chem)
    require(mol.GetNumAtoms() > 0 and len(Chem.GetMolFrags(mol)) == 1, 'Input must be a connected ligand')
    require(mol.GetNumConformers() > 0 and mol.GetConformer().Is3D(), 'Missing 3D conformer')
    require(all(math.isfinite(float(v)) for row in mol.GetConformer().GetPositions() for v in row), 'Nonfinite coordinates')
    mask = extract_scaffolds([mol], invert_mask=True)[0].tolist()
    fixed = {i for i, flag in enumerate(mask) if bool(flag)}
    require(0 < len(fixed) < mol.GetNumAtoms(), 'No fixed fragment or no replaceable scaffold')
    anchors = set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (i in fixed) != (j in fixed):
            anchors.add(i if i in fixed else j)
    pairs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in fixed and j in fixed and (i in anchors or j in anchors):
            pairs.append({'input_atom_indices': [i, j], 'bond_type': str(bond.GetBondType()),
                          'elements': [mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(j).GetSymbol()]})
    require(anchors, 'No fixed-to-scaffold attachment in the input')
    return {'input_smiles': canonical(mol, Chem), 'n_heavy_atoms': mol.GetNumAtoms(), 'n_fixed_preview': len(fixed),
            'n_attachment_sites_preview': len(anchors), 'boundary_adjacent_fixed_bonds_preview': pairs,
            'index_note': 'Input-heavy-atom preview only; actual reference/sampler mapping comes from task_definition.json.'}
