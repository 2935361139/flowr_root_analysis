"""Unmodified native-evaluation helpers already used in the project."""
from pathlib import Path
def read_one_sdf(path, Chem):
    if not path.is_file():
        raise FileNotFoundError(path)
    if not path.read_text(encoding="utf-8").strip():
        return None
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    if len(supplier) != 1 or supplier[0] is None:
        raise ValueError(f"Expected exactly one readable recorded molecule: {path}")
    return supplier[0]

def fixed_diagnostics(mol, ref, task, slots, np):
    fixed = list(task["fixed_sampler_indices"])
    ref_map = dict(zip(fixed, task["fixed_reference_indices"]))
    gen_map = {slot: index for index, slot in enumerate(slots)}
    if any(i not in gen_map for i in fixed):
        raise ValueError("A recorded fixed atom is absent; requires a separate mapping audit.")
    expected = {i: [] for i in fixed}
    for entry in task["reference_attachment_sites"]:
        expected[entry["fixed_sampler_index"]].append(entry["reference_bond_type"])
    actual = {i: [] for i in fixed}
    neighbors = {i: [] for i in fixed}
    for bond in mol.GetBonds():
        a, b = slots[bond.GetBeginAtomIdx()], slots[bond.GetEndAtomIdx()]
        for i, j in [(a, b), (b, a)]:
            if i in actual and j not in actual:
                actual[i].append(str(bond.GetBondType()))
                neighbors[i].append(j)
    violations = [i for i in fixed if sorted(actual[i]) != sorted(expected[i])]
    elements = [i for i in fixed if mol.GetAtomWithIdx(gen_map[i]).GetAtomicNum()
                != ref.GetAtomWithIdx(ref_map[i]).GetAtomicNum()]
    charges = [i for i in fixed if mol.GetAtomWithIdx(gen_map[i]).GetFormalCharge()
               != ref.GetAtomWithIdx(ref_map[i]).GetFormalCharge()]
    def bond_type(m, i, j):
        bond = m.GetBondBetweenAtoms(i, j)
        return str(bond.GetBondType()) if bond else None
    internal = [[i, j] for k, i in enumerate(fixed) for j in fixed[:k]
                if bond_type(mol, gen_map[i], gen_map[j]) != bond_type(ref, ref_map[i], ref_map[j])]
    delta = np.asarray([mol.GetConformer().GetPositions()[gen_map[i]]
                        - ref.GetConformer().GetPositions()[ref_map[i]] for i in fixed])
    distances = np.linalg.norm(delta, axis=1)
    return {
        "fixed_element_changed_sites": elements,
        "fixed_charge_changed_sites": charges,
        "fixed_internal_bond_changed_pairs": internal,
        "fixed_rms_displacement_A": float(np.sqrt(np.mean(distances**2))),
        "fixed_max_displacement_A": float(distances.max()),
        "strict_boundary_reference_match": not violations,
        "strict_boundary_differing_sites": violations,
        "attachment_sites_without_generated_neighbor": [i for i in fixed if expected[i] and not actual[i]],
        "nonattachment_sites_with_generated_neighbors": [i for i in fixed if not expected[i] and actual[i]],
        "boundary_details": [{"fixed_site": i, "reference_bond_types": sorted(expected[i]),
                              "generated_bond_types": sorted(actual[i]), "generated_neighbors": neighbors[i]}
                             for i in fixed],
    }

def pb_metrics(mol, protein, PoseBusters, Chem):
    # Matches generate_from_pdb's unoptimized heavy-atom model path: sanitize,
    # add H coordinates, then evaluate_pb_validity's PoseBusters(config="dock").
    import numpy as np
    with_hs = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    buster = PoseBusters(config="dock")
    table = buster.bust([with_hs], None, str(protein))
    if len(table) != 1 or not len(table.columns):
        raise RuntimeError("PoseBusters returned no unique, nonempty result row.")
    row = table.iloc[0]
    if row.isna().any():
        raise RuntimeError("PoseBusters returned missing check results; not a molecular failure label.")
    if any(not isinstance(v, (bool, np.bool_)) for v in row.tolist()):
        raise RuntimeError("Unexpected nonboolean PoseBusters result columns.")
    checks = {str(k): bool(v) for k, v in row.items()}
    return {"posebusters_valid": bool(table.all(axis=1).iloc[0]),
            "posebusters_checks": checks,
            "posebusters_failed_checks": [k for k, v in checks.items() if not v]}
