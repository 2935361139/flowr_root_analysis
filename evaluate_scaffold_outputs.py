#!/usr/bin/env python3
"""Evaluate existing FLOWR.ROOT scaffold-hopping records across complexes.

No generation, training, checkpoint loading, intervention, or filtering.
Uses the installed FLOWR validity and conditional-substructure functions and
its PoseBusters dock configuration. Keeps strict reference-boundary agreement
as a separate diagnostic, never as a universal scaffold-hopping failure label.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

VERSION = "2026-10-21.multicase.2"

SOURCE_HASHES = {
    "flowr/util/rdkit.py": "00626d65e097dd1ac6b8f558219320869fef52611f964880a381f35903fda0d5",
    "flowr/util/metrics.py": "e2ec7666d7c63ccbe164f67063f5d13701bcd11c146f423c86c86b6d723768e3",
    "flowr/gen/utils.py": "2185b671e62bea6012dd38f4dfb8e6ba877285769b8609cc252c28760be440e0",
    "flowr/data/interpolate.py": "171af7162fe2b7fc37e8ffdcbf90f6ea5c99bebed0b17a3d825ef24dee7cdd96",
    "flowr/scriptutil.py": "94422ce28c701966cd54f1d46b4bbb22438204a38789a3e897301f4fb71da6ff",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_one_sdf(path, Chem):
    if not path.is_file():
        raise FileNotFoundError(path)
    if not path.read_text(encoding="utf-8").strip():
        return None
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    if len(supplier) != 1 or supplier[0] is None:
        raise ValueError(f"Expected exactly one readable recorded molecule: {path}")
    return supplier[0]


def restore_coordinates(mol, raw, tokens, np):
    """Verify recorded atom order, then restore pre-SDF-rounding coordinates."""
    mask = raw["mask"][0].cpu().numpy().astype(bool)
    types = raw["atomics"][0].argmax(-1).cpu().numpy()
    slots = [i for i in range(len(mask)) if mask[i] and tokens[int(types[i])] != "<NOATOM>"]
    coords = raw["coords"][0].cpu().numpy()[slots]
    if mol.GetNumAtoms() != len(slots):
        raise ValueError("Decoded atom count differs from recorded retained slots.")
    conf = mol.GetConformer()
    error = float(np.max(np.abs(conf.GetPositions()-coords)))
    if not np.isfinite(coords).all() or error > 0.002:
        raise ValueError("SDF atom order/frame differs from raw final prediction.")
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, tuple(float(x) for x in xyz))
    return slots, error


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


def summarize(rows):
    result = {}
    for stage in ["raw_without_repair", "run_decode"]:
        group = [r for r in rows if r.get("stage") == stage]
        stats = {"n_records": len(group)}
        for metric in ["rdkit_valid", "fc_valid", "root_fixed_substructure_match",
                       "fixed_chemistry_preserved", "chemical_connection_endpoint",
                       "posebusters_valid", "strict_boundary_reference_match", "murcko_scaffold_changed"]:
            observed = [r[metric] for r in group if isinstance(r.get(metric), bool)]
            stats[metric] = {"n_evaluated": len(observed), "n_pass_or_true": sum(observed),
                             "fraction_among_evaluated": sum(observed)/len(observed) if observed else None}
        valid = [r for r in group if r.get("fc_valid") is True and "molecule_smiles" in r and "scaffold_smiles" in r]
        stats["n_fc_valid_with_scaffold_descriptor"] = len(valid)
        stats["unique_molecule_count_among_fc_valid"] = len({r["molecule_smiles"] for r in valid})
        stats["unique_murcko_scaffold_count_among_fc_valid"] = len({r["scaffold_smiles"] for r in valid})
        result[stage] = stats
    return result


def all_three_valued(values):
    """A known failure makes a conjunction false; otherwise unknown stays unknown."""
    if any(v is False for v in values):
        return False
    return True if all(v is True for v in values) else None


def task_profile(ref, task):
    """Describe the retained subgraph without selecting on generated outcomes."""
    fixed_ref = list(task["fixed_reference_indices"])
    fixed_slots = list(task["fixed_sampler_indices"])
    if (not fixed_ref or len(fixed_ref) != len(fixed_slots)
            or len(set(fixed_ref)) != len(fixed_ref)
            or len(set(fixed_slots)) != len(fixed_slots)
            or any(i < 0 or i >= ref.GetNumAtoms() for i in fixed_ref)):
        raise ValueError("Invalid fixed/reference mapping.")
    ref_to_slot = dict(zip(fixed_ref, fixed_slots))
    adjacency = {i: set() for i in fixed_ref}
    internal = []
    for bond in ref.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)
            internal.append([ref_to_slot[a], ref_to_slot[b], str(bond.GetBondType())])
    remaining = set(adjacency)
    components = []
    while remaining:
        stack = [min(remaining)]
        seen = set()
        while stack:
            i = stack.pop()
            if i not in seen:
                seen.add(i)
                stack.extend(adjacency[i] - seen)
        remaining -= seen
        components.append(sorted(seen))
    sites = []
    anchors = sorted({x["fixed_sampler_index"] for x in task["reference_attachment_sites"]})
    for i, slot in zip(fixed_ref, fixed_slots):
        atom = ref.GetAtomWithIdx(i)
        internal_sum = sum(float(ref.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble())
                           for j in adjacency[i])
        sites.append({"sampler_index": slot, "reference_index": i,
                      "element": atom.GetSymbol(), "formal_charge": atom.GetFormalCharge(),
                      "reference_fixed_internal_bond_order_sum": internal_sum,
                      "reference_attachment_site": slot in anchors})
    return {"n_fixed_atoms": len(fixed_ref), "n_generated_slots": task["n_generated_slots"],
            "n_fixed_components": len(components),
            "fixed_component_sizes": sorted([len(c) for c in components], reverse=True),
            "all_fixed_components_are_single_atoms": all(len(c) == 1 for c in components),
            "n_reference_attachment_sites": len(anchors), "attachment_sampler_indices": anchors,
            "fixed_internal_bonds": internal, "fixed_sites": sites,
            "note": "Task descriptors, not a ranking of difficulty or an outcome-based exclusion rule."}


def inspect_raw_graph(raw, tokens, charge_tokens, task, ref, np):
    """Inspect all retained sites in final argmax tensors, even if SDF decode failed.

    The carbon-specific flag detects the previously observed failure family only.
    General chemical validity still comes from the installed FLOWR/RDKit check.
    """
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    atomics, charges, bonds, coords, mask = [array(raw[k]) for k in
                                           ["atomics", "charges", "bonds", "coords", "mask"]]
    if atomics.ndim != 3 or atomics.shape[0] != 1:
        raise ValueError("Expected one molecule in raw final tensors.")
    n = atomics.shape[1]
    if (atomics.shape[-1] != len(tokens) or charges.shape != (1, n, len(charge_tokens))
            or bonds.shape[:3] != (1, n, n) or bonds.shape[-1] not in (5, 6)
            or coords.shape != (1, n, 3) or mask.shape != (1, n)):
        raise ValueError("Unsupported final tensor shape/vocabulary.")
    if not all(np.isfinite(x).all() for x in [atomics, charges, bonds, coords, mask]):
        raise ValueError("Nonfinite final tensor values; graph diagnostics not assessed.")
    elements = [tokens[int(i)] for i in atomics[0].argmax(-1)]
    charge_values = [charge_tokens[int(i)] for i in charges[0].argmax(-1)]
    active = [i for i in range(n) if bool(mask[0, i]) and elements[i] != "<NOATOM>"]
    if any(elements[i].startswith("<") or not isinstance(charge_values[i], int) for i in active):
        raise ValueError("A special atom/charge token remains active; graph diagnostics unsupported.")
    classes = bonds[0].argmax(-1)
    if any(classes[i, j] > 4 or classes[i, j] != classes[j, i]
           for k, i in enumerate(active) for j in active[:k]):
        raise ValueError("Asymmetric or masked final bond classes; do not infer a decoded graph.")
    bond_names = [None, "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
    orders = np.asarray([0.0, 1.0, 2.0, 3.0, 1.5])
    adjacency = {i: [j for j in active if i != j and classes[i, j] != 0] for i in active}
    remaining = set(active)
    n_components = 0
    while remaining:
        stack = [min(remaining)]
        seen = set()
        while stack:
            i = stack.pop()
            if i not in seen:
                seen.add(i)
                stack.extend(set(adjacency[i]) - seen)
        remaining -= seen
        n_components += 1
    fixed = list(task["fixed_sampler_indices"])
    if any(i < 0 or i >= n for i in fixed):
        raise ValueError("A fixed sampler index is outside the saved tensor.")
    ref_map = dict(zip(fixed, task["fixed_reference_indices"]))
    missing = [i for i in fixed if i not in active]
    element_changes = [i for i in fixed if i in active and
                       elements[i] != ref.GetAtomWithIdx(ref_map[i]).GetSymbol()]
    charge_changes = [i for i in fixed if i in active and
                      charge_values[i] != ref.GetAtomWithIdx(ref_map[i]).GetFormalCharge()]
    internal_changes = []
    for k, i in enumerate(fixed):
        for j in fixed[:k]:
            if i not in active or j not in active:
                continue
            b = ref.GetBondBetweenAtoms(ref_map[i], ref_map[j])
            if bond_names[int(classes[i, j])] != (str(b.GetBondType()) if b else None):
                internal_changes.append([i, j])
    sites = []
    for i in fixed:
        if i not in active:
            continue
        neighbors = adjacency[i]
        valence = float(sum(orders[int(classes[i, j])] for j in neighbors))
        sites.append({"sampler_index": i, "element": elements[i], "formal_charge": charge_values[i],
                      "explicit_bond_order_sum": valence, "neighbors": neighbors,
                      "generated_neighbors": [j for j in neighbors if j not in fixed],
                      "neutral_carbon_overvalent": elements[i] == "C" and charge_values[i] == 0
                                                   and valence > 4.0 + 1e-6})
    return {"status": "evaluated", "active_atoms": len(active), "n_components": n_components,
            "connected": bool(active) and n_components == 1,
            "fixed_missing_sites": missing, "fixed_element_changed_sites": element_changes,
            "fixed_charge_changed_sites": charge_changes,
            "fixed_internal_bond_changed_pairs": internal_changes,
            "fixed_chemistry_preserved": not any([missing, element_changes, charge_changes, internal_changes]),
            "fixed_neutral_carbon_overvalent_sites": [s["sampler_index"] for s in sites
                                                      if s["neutral_carbon_overvalent"]],
            "fixed_sites": sites,
            "scope": "Final argmax graph only; carbon-overvalence is one specific failure family, not a complete validity criterion."}


def row_labels(row):
    """Independent flags; scaffold identity and strict boundary are not connection errors."""
    flags = []
    if row.get("rdkit_valid") is False:
        flags.append("chemical_decode_invalid")
    if row.get("raw_graph_connected") is False and row.get("stage") == "raw_without_repair":
        flags.append("raw_graph_disconnected")
    if row.get("fixed_chemistry_preserved") is False:
        flags.append("fixed_chemistry_changed")
    if row.get("fixed_neutral_carbon_overvalent_sites"):
        flags.append("fixed_neutral_carbon_overvalence")
    if row.get("posebusters_valid") is False:
        flags.append("posebusters_failed")
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--save-dir", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cases", nargs="+", help="Case names in sh_record_CASE_seedSEED; defaults to ptp1b.")
    group.add_argument("--run-dirs", type=Path, nargs="+", help="Explicit recorder run directories for other inputs.")
    args = parser.parse_args()
    root = Path.cwd()
    if not (root/"flowr/gen/utils.py").is_file():
        parser.error("Run in /home/ljl/projects/flowr_root using the existing .venv.")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seed list contains duplicates.")
    if args.run_dirs:
        if len(set(p.resolve() for p in args.run_dirs)) != len(args.run_dirs):
            parser.error("Duplicate run directories.")
        requested = [(None, None, p.resolve()) for p in args.run_dirs]
    else:
        cases = args.cases or ["ptp1b"]
        if len(set(cases)) != len(cases) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", c) for c in cases):
            parser.error("Use unique case names containing letters, numbers, underscores or hyphens.")
        requested = [(case, seed, (args.output_root/f"sh_record_{case}_seed{seed}").resolve())
                     for case in cases for seed in args.seeds]
    out = (args.save_dir or args.output_root/(
        "sh_quality_multicase_"+datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if out.exists():
        parser.error(f"Output already exists: {out}. Choose a new --save-dir.")
    out.mkdir(parents=True)
    sys.path.insert(0, str(root))
    report = {"version": VERSION, "completed": False, "rows": [], "errors": [],
              "requested_run_count": len(requested), "runs": [],
              "scope": "Same rules across existing recorded complexes; no full scaffold-hopping success, causal mechanism or bioactivity claim.",
              "rule": "Strict reference-boundary equality is a diagnostic of a narrower task, not a universal failure criterion. New generated atom identities, core topology and coordinates are not required to match the reference.",
              "geometry": "Fixed displacement is continuous, in the original protein frame without alignment. No arbitrary fixed-displacement pass/fail cutoff. PoseBusters uses the installed FLOWR dock configuration after adding H, without geometry optimization.",
              "diversity": "Canonical molecule/scaffold counts only. Murcko change is a structural descriptor, not proof of activity retention. Training-set novelty is NOT assessed.",
              "denominators": "Raw and run-setting decodes reported separately. No samples dropped. Substructure/PB/scaffold checks are not evaluated on invalid decodes; their denominators are explicit.",
              "versions": {}, "source_sha256": {}, "script_sha256": digest(Path(__file__))}
    try:
        for name, expected in SOURCE_HASHES.items():
            observed = digest(root/name)
            report["source_sha256"][name] = observed
            if observed != expected:
                raise RuntimeError(f"Evaluation source differs from the audited copy: {name}. Do not change the source to bypass this check.")
        import numpy as np
        import torch
        import rdkit
        import posebusters
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from posebusters import PoseBusters
        from flowr.constants import CORE_ATOMS
        from flowr.util.rdkit import mol_is_valid
        from flowr.gen.utils import check_substructure_match
        from flowr.data.interpolate import extract_scaffolds
        from flowr.scriptutil import _build_vocab_charges
        report["versions"] = {"rdkit": rdkit.__version__, "torch": str(torch.__version__),
                              "numpy": np.__version__, "posebusters": str(getattr(posebusters, "__version__", "unknown")),
                              "posebusters_module": str(posebusters.__file__)}
        for case, seed, run_directory in requested:
            print(f"Evaluating existing {run_directory.name} ...", flush=True)
            try:
                record = run_directory/"research_record"
                manifest, summary = read_json(record/"manifest.json"), read_json(record/"summary.json")
                actual_seed = manifest["args"]["seed"]
                if seed is None:
                    seed = actual_seed
                if case is None:
                    case = Path(manifest["args"]["pdb_file"]).stem.removesuffix("_protein")
                folder = record/"attempt_000"
                task = read_json(folder/"task_definition.json")
                if (summary["n_attempts"] != 1 or actual_seed != seed or not task["mapping_verified"]
                        or summary.get("pipeline_error") or summary["attempts"][0].get("audit_error")
                        or summary["attempts"][0].get("generation_completed") is not True):
                    raise ValueError("Pilot record count, seed, or fixed-atom mapping mismatch.")
                if not manifest["args"].get("scaffold_hopping") or task.get("mode") != "scaffold_hopping":
                    raise ValueError("This evaluator requires a recorded scaffold-hopping task.")
                if rdkit.__version__ != manifest["rdkit_version"]:
                    raise ValueError("RDKit version changed since recording; preserve the working environment.")
                if manifest["args"].get("optimize_gen_ligs") or manifest["args"].get("optimize_gen_ligs_hs"):
                    raise ValueError("This evaluator targets the recorded unoptimized pilot.")
                protein = Path(manifest["args"]["pdb_file"])
                if digest(protein) != manifest["input_sha256"][str(protein)]:
                    raise ValueError("Protein input changed since generation.")
                ref = read_one_sdf(folder/"reference_preprocessed.sdf", Chem)
                Chem.SanitizeMol(ref)
                if any(a.GetAtomicNum() == 1 for a in ref.GetAtoms()):
                    raise ValueError("This pilot evaluator expects the recorded heavy-atom representation.")
                ref_fixed = extract_scaffolds([ref], invert_mask=True)[0].nonzero(as_tuple=True)[0].tolist()
                if ref_fixed != task["fixed_reference_indices"]:
                    raise ValueError("Official scaffold extraction does not match the recorded task.")
                profile = task_profile(ref, task)
                ref_smi = Chem.MolToSmiles(ref, isomericSmiles=False)
                ref_scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(ref), isomericSmiles=False)
                raw = torch.load(folder/"raw_final_prediction.pt", map_location="cpu", weights_only=True)
                inputs = torch.load(folder/"inputs.pt", map_location="cpu", weights_only=True)
                prior = inputs["prior"]
                prior_fixed = (prior["fragment_mask"][0].bool() & prior["mask"][0].bool()).nonzero(as_tuple=True)[0].tolist()
                if prior_fixed != task["fixed_sampler_indices"]:
                    raise ValueError("Saved prior mask disagrees with the recorded task mapping.")
                width = raw["atomics"].shape[-1]
                tokens = ["<PAD>"]+CORE_ATOMS
                if width == len(tokens)+1:
                    tokens.append("<NOATOM>")
                if width != len(tokens):
                    raise ValueError("Unexpected atom vocabulary.")
                charge_tokens = _build_vocab_charges().tokens_from_indices(list(range(raw["charges"].shape[-1])))
                charge_tokens = [int(v) if str(v).lstrip("-").isdigit() else str(v) for v in charge_tokens]
                raw_graph = inspect_raw_graph(raw, tokens, charge_tokens, task, ref, np)
                run_info = {"case": case, "seed": seed, "run_directory": str(run_directory),
                            "task_profile": profile, "raw_graph": raw_graph,
                            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
                            "recorded_source_sha256": manifest.get("source_sha256"),
                            "input_sha256": manifest.get("input_sha256"),
                            "ligand_idx": manifest["args"].get("ligand_idx"),
                            "reference_sdf_sha256": digest(folder/"reference_preprocessed.sdf"),
                            "raw_final_sha256": digest(folder/"raw_final_prediction.pt"),
                            "task_sha256": digest(folder/"task_definition.json")}
                report["runs"].append(run_info)
                saved = summary["attempts"][0]
                for stage, name, status_key in [
                    ("raw_without_repair", "decoded_without_repair.sdf", "decode_without_valence_repair"),
                    ("run_decode", "decoded_before_filter.sdf", "decode_with_run_setting")]:
                    row = {"case": case, "seed": seed, "run_directory": str(run_directory),
                           "stage": stage, "errors": [], "rdkit_valid": None,
                           "fc_valid": None, "root_fixed_substructure_match": None, "posebusters_valid": None,
                           "strict_boundary_reference_match": None, "murcko_scaffold_changed": None,
                           "fixed_chemistry_preserved": None, "chemical_connection_endpoint": None,
                           "fixed_checks_status": "not_assessed", "posebusters_status": "not_assessed",
                           "raw_graph_connected": raw_graph["connected"] if stage == "raw_without_repair" else None,
                           "input_sdf_sha256": digest(folder/name)}
                    report["rows"].append(row)
                    if stage == "raw_without_repair":
                        row["fixed_neutral_carbon_overvalent_sites"] = raw_graph["fixed_neutral_carbon_overvalent_sites"]
                        row["fixed_chemistry_preserved"] = raw_graph["fixed_chemistry_preserved"]
                        row["fixed_checks_status"] = "evaluated_on_raw_argmax_tensors"
                        for key in ["fixed_missing_sites", "fixed_element_changed_sites", "fixed_charge_changed_sites",
                                    "fixed_internal_bond_changed_pairs"]:
                            row[key] = raw_graph[key]
                    mol = read_one_sdf(folder/name, Chem)
                    status = saved[status_key][0]
                    if (mol is not None) != bool(status["built"]):
                        raise ValueError("Saved SDF availability contradicts recorded decode status.")
                    row["rdkit_valid"] = bool(mol_is_valid(mol, connected=False))
                    row["fc_valid"] = bool(mol_is_valid(mol, connected=True))
                    if row["rdkit_valid"] != status["sanitizable"] or row["fc_valid"] != bool(status["sanitizable"] and status["connected"]):
                        raise ValueError("Saved decode and official validity recheck disagree.")
                    if not row["rdkit_valid"]:
                        row["conditional_checks_status"] = "not_evaluated_invalid_decode"
                        row["posebusters_status"] = "not_assessed_invalid_decode"
                        row["chemical_connection_endpoint"] = False
                        row["failure_flags"] = row_labels(row)
                        continue
                    Chem.SanitizeMol(mol)
                    slots, error = restore_coordinates(mol, raw, tokens, np)
                    row["sdf_coordinate_rounding_max_A"] = error
                    missing = [i for i in task["fixed_sampler_indices"] if i not in slots]
                    row["fixed_missing_sites"] = missing
                    if missing:
                        row["fixed_chemistry_preserved"] = False
                        row["fixed_checks_status"] = "fixed_atoms_absent"
                    else:
                        row.update(fixed_diagnostics(mol, ref, task, slots, np))
                        row["fixed_chemistry_preserved"] = not any(row[key] for key in
                            ["fixed_element_changed_sites", "fixed_charge_changed_sites", "fixed_internal_bond_changed_pairs"])
                        row["fixed_checks_status"] = "evaluated_on_coordinate_verified_decoded_slots"
                    if stage == "raw_without_repair" and row["fixed_chemistry_preserved"] != raw_graph["fixed_chemistry_preserved"]:
                        raise ValueError("Raw tensor and decoded fixed-identity checks disagree.")
                    row["root_fixed_substructure_match"] = bool(check_substructure_match(
                        Chem.Mol(mol), Chem.Mol(ref), "scaffold_hopping"))
                    row["molecule_smiles"] = Chem.MolToSmiles(mol, isomericSmiles=False)
                    row["scaffold_smiles"] = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=False)
                    row["same_molecule_graph_as_reference"] = row["molecule_smiles"] == ref_smi
                    row["murcko_scaffold_changed"] = row["scaffold_smiles"] != ref_scaffold
                    row["chemical_connection_endpoint"] = all_three_valued([
                        row["fc_valid"], row["fixed_chemistry_preserved"], row["root_fixed_substructure_match"]])
                    try:
                        row.update(pb_metrics(mol, protein, PoseBusters, Chem))
                        row["posebusters_status"] = "evaluated"
                    except Exception as exc:
                        row["errors"].append(f"PoseBusters evaluation error (not a molecule failure): {type(exc).__name__}: {exc}")
                        row["posebusters_status"] = "evaluation_error"
                    row["conditional_checks_status"] = "evaluated" if not row["errors"] else "partially_evaluated"
                    row["failure_flags"] = row_labels(row)
            except Exception as exc:
                report["errors"].append({"case": case, "seed": seed, "run_directory": str(run_directory),
                                         "message": f"{type(exc).__name__}: {exc}",
                                         "traceback": traceback.format_exc()})
                print(f"  Evaluation issue: {exc}", flush=True)
        report["completed"] = (len(report["rows"]) == 2*len(requested) and not report["errors"]
                               and not any(r["errors"] for r in report["rows"]))
    except Exception as exc:
        report["errors"].append({"message": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        print(f"Evaluation issue: {exc}", flush=True)
    report["aggregate"] = summarize(report["rows"])
    report["aggregate_by_case"] = {case: summarize([r for r in report["rows"] if r["case"] == case])
                                   for case in sorted({r["case"] for r in report["rows"]})}
    identities = {json.dumps({"checkpoint": r["checkpoint_sha256"], "source": r["recorded_source_sha256"]},
                             sort_keys=True) for r in report["runs"]}
    report["same_recorded_generator_identity_across_runs"] = len(identities) == 1 if identities else None
    report["failure_family_note"] = "Flags can overlap; do not sum them as disjoint categories. No causal conclusions or held-out generalization are established by evaluation."
    for run in report["runs"]:
        stage_rows = [r for r in report["rows"] if r["run_directory"] == run["run_directory"]]
        if len(stage_rows) == 2:
            raw_row, decoded_row = stage_rows
            run["repair_comparison"] = {
                "raw_chemical_valid": raw_row["rdkit_valid"], "run_decode_chemical_valid": decoded_row["rdkit_valid"],
                "run_decode_fixed_chemistry_preserved": decoded_row["fixed_chemistry_preserved"],
                "note": "These are two decodes of the same final network output, not separate generated trajectories."}
    (out/"quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    fields = ["case", "seed", "stage", "rdkit_valid", "fc_valid", "raw_graph_connected",
              "root_fixed_substructure_match", "fixed_chemistry_preserved", "fixed_checks_status",
              "chemical_connection_endpoint", "posebusters_valid", "posebusters_status",
              "posebusters_failed_checks", "fixed_element_changed_sites", "fixed_charge_changed_sites",
              "fixed_internal_bond_changed_pairs", "fixed_rms_displacement_A", "fixed_max_displacement_A",
              "strict_boundary_reference_match", "strict_boundary_differing_sites", "murcko_scaffold_changed",
              "fixed_neutral_carbon_overvalent_sites", "failure_flags", "errors"]
    with (out/"quality_labels.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row.get(k), (dict,list))
                             else row.get(k) for k in fields})
    (out/"task_profiles.json").write_text(json.dumps([
        {"case": r["case"], "seed": r["seed"], "run_directory": r["run_directory"], **r["task_profile"]}
        for r in report["runs"]], ensure_ascii=False, indent=2), encoding="utf-8")
    (out/"evaluator_source.py").write_bytes(Path(__file__).read_bytes())
    bundle = out/"quality_review.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ["quality_report.json", "quality_labels.csv", "task_profiles.json", "evaluator_source.py"]:
            archive.write(out/name, name)
    print(f"\nEvaluation completed: {report['completed']}")
    print(f"Upload this file: {bundle}")
    print("Completion means evaluation ran; it does not mean all molecules passed.")
    for case, stages in report["aggregate_by_case"].items():
        raw_stage = stages["raw_without_repair"]
        print(f"{case}: raw valid+connected {raw_stage['fc_valid']['n_pass_or_true']}/{raw_stage['fc_valid']['n_evaluated']}; "
              f"scaffold changed {raw_stage['murcko_scaffold_changed']['n_pass_or_true']}/{raw_stage['murcko_scaffold_changed']['n_evaluated']}")
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
