#!/usr/bin/env python3
"""Read existing FLOWR recordings and trace fixed-to-generated bonds.

Run from the FLOWR.ROOT repository: python trace_connection_sites.py
Uses the existing PyTorch and NumPy environment; no model loading or generation.
The boundary rule preserves the reference attachment sites, bond counts and
bond orders. It is a task-specific diagnostic, not a full 3D success metric.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path

BOND_CODES = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_v2000(path):
    """Read literal atoms/bonds/charges; RDKit validity comes from the recorder."""
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 4 or "V2000" not in lines[3]:
        raise ValueError(f"Expected a V2000 SDF: {path}")
    if sum(line.strip() == "$$$$" for line in lines) > 1:
        raise ValueError("This pilot reader expects one molecule per SDF.")
    n, b = int(lines[3][:3]), int(lines[3][3:6])
    atoms = []
    charge_codes = {0: 0, 1: 3, 2: 2, 3: 1, 4: 0, 5: -1, 6: -2, 7: -3}
    for line in lines[4:4+n]:
        atoms.append({"element": line[31:34].strip(),
                      "coords": [float(line[k:k+10]) for k in (0, 10, 20)],
                      "charge": charge_codes[int(line[36:39].strip() or 0)]})
    bonds = {}
    for line in lines[4+n:4+n+b]:
        i, j, kind = int(line[:3])-1, int(line[3:6])-1, int(line[6:9])
        bonds[tuple(sorted((i, j)))] = kind
    for line in lines[4+n+b:]:
        if line.startswith("M  CHG"):
            fields = line.split()
            for k in range(int(fields[2])):
                atoms[int(fields[3+2*k])-1]["charge"] = int(fields[4+2*k])
    return {"atoms": atoms, "bonds": bonds}


def expected_boundary(task):
    result = {int(i): [] for i in task["fixed_sampler_indices"]}
    for site in task["reference_attachment_sites"]:
        result[int(site["fixed_sampler_index"])].append(BOND_CODES[site["reference_bond_type"]])
    return {i: sorted(v) for i, v in result.items()}


def endpoint_rows(task, reference, molecule, decoded_to_sampler):
    """Atom ordering must be checked against raw tensor coordinates by caller."""
    expected = expected_boundary(task)
    fixed = set(expected)
    refs = dict(zip(task["fixed_sampler_indices"], task["fixed_reference_indices"]))
    if len(molecule["atoms"]) != len(decoded_to_sampler):
        raise ValueError("Decoded atom count does not match retained sampler slots.")
    atoms = {slot: atom for slot, atom in zip(decoded_to_sampler, molecule["atoms"])}
    bonds = {tuple(sorted((decoded_to_sampler[i], decoded_to_sampler[j]))): kind
             for (i, j), kind in molecule["bonds"].items()}
    rows = []
    for i in sorted(fixed):
        ref = reference["atoms"][refs[i]]
        atom = atoms.get(i)
        edges = sorted((j if i == k else k, kind) for (k, j), kind in bonds.items()
                       if i in (k, j) and (j if i == k else k) not in fixed)
        actual = sorted(kind for _, kind in edges)
        rows.append({"fixed_sampler_index": i, "fixed_reference_index": refs[i],
                     "reference_element": ref["element"], "decoded_element": atom["element"] if atom else None,
                     "fixed_element_preserved": bool(atom and atom["element"] == ref["element"]),
                     "fixed_charge_preserved": bool(atom and atom["charge"] == ref["charge"]),
                     "expected_bond_classes": expected[i], "actual_bond_classes": actual,
                     "boundary_rule_satisfied": actual == expected[i],
                     "generated_neighbors": [{"sampler_index": j, "element": atoms[j]["element"], "bond_class": k}
                                             for j, k in edges]})
    return rows


def source_vocab(root, width):
    tree = ast.parse((root / "flowr/constants.py").read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name) and name.id in {"CORE_ATOMS", "NOATOM_TOKEN"}:
                    values[name.id] = ast.literal_eval(node.value)
    tokens = ["<PAD>"] + values["CORE_ATOMS"]
    if width == len(tokens)+1:
        tokens.append(values["NOATOM_TOKEN"])
    if width != len(tokens):
        raise ValueError("Unexpected atom vocabulary width; do not guess token identities.")
    return tokens


def canonical_boundary(values, fixed, np):
    """Match FLOWR's lower-triangle bond decoder, including asymmetric inputs."""
    n = values.shape[0]
    a, b = np.asarray(fixed)[:, None], np.arange(n)[None, :]
    return values[np.maximum(a, b), np.minimum(a, b)]


def snapshot_arrays(data, fixed, valid, scale, np, logits=False):
    def array(key):
        return data[key][0].detach().cpu().float().numpy()
    bonds = canonical_boundary(array("bonds"), fixed, np)
    if not np.isfinite(bonds).all():
        raise ValueError("Nonfinite bond predictions; the original records are retained.")
    probabilities = None
    if logits:
        e = np.exp(bonds - bonds.max(axis=-1, keepdims=True))
        probabilities = e / e.sum(axis=-1, keepdims=True)
    return {"coords": array("coords") * scale,
            "atom_classes": array("atomics").argmax(-1),
            "charge_classes": array("charges").argmax(-1),
            "bond_classes": bonds.argmax(-1),
            "bond_probabilities": probabilities,
            "valid": array("mask").astype(bool) if "mask" in data else valid.copy()}


def site_rows(state, initial, task, tokens, stream, index, time, np):
    fixed = task["fixed_sampler_indices"]
    fixed_set = set(fixed)
    expected = expected_boundary(task)
    refs = dict(zip(fixed, task["fixed_reference_indices"]))
    active = [i for i, ok in enumerate(state["valid"]) if ok and tokens[int(state["atom_classes"][i])] != "<NOATOM>"]
    rows = []
    for k, i in enumerate(fixed):
        neighbors = [j for j in active if j not in fixed_set and int(state["bond_classes"][k, j]) != 0]
        classes = [int(state["bond_classes"][k, j]) for j in neighbors]
        distances = [float(np.linalg.norm(state["coords"][i]-state["coords"][j])) for j in neighbors]
        if not all(math.isfinite(x) for x in distances):
            distances = [x if math.isfinite(x) else None for x in distances]
        probabilities = (None if state["bond_probabilities"] is None else
                         [float(1-state["bond_probabilities"][k, j, 0]) for j in neighbors])
        drift = float(np.linalg.norm(state["coords"][i]-initial["coords"][i]))
        rows.append({"stream": stream, "index": index, "continuous_time": time,
                     "fixed_sampler_index": i, "fixed_reference_index": refs[i],
                     "reference_attachment_site": bool(expected[i]),
                     "expected_external_bonds": len(expected[i]), "actual_external_bonds": len(neighbors),
                     "boundary_rule_satisfied": i in active and sorted(classes) == expected[i],
                     "expected_bond_classes": json.dumps(expected[i]),
                     "generated_neighbors": json.dumps(neighbors),
                     "neighbor_elements": json.dumps([tokens[int(state["atom_classes"][j])] for j in neighbors]),
                     "bond_classes": json.dumps(classes), "bond_lengths_A": json.dumps(distances),
                     "predicted_nonzero_bond_probabilities": json.dumps(probabilities),
                     "fixed_element": tokens[int(state["atom_classes"][i])],
                     "fixed_element_changed": int(state["atom_classes"][i]) != int(initial["atom_classes"][i]),
                     "fixed_charge_class_changed": int(state["charge_classes"][i]) != int(initial["charge_classes"][i]),
                     "fixed_displacement_A": drift if math.isfinite(drift) else None})
    return rows


def terminal_intervals(rows, fixed):
    """Suffix summaries avoid interpreting normal early sampling noise as onset."""
    output = []
    for i in fixed:
        site = [r for r in rows if r["fixed_sampler_index"] == i]
        if not site:
            continue
        last = site[-1]
        signature = lambda r: (r["generated_neighbors"], r["bond_classes"], r["neighbor_elements"])
        start = len(site)-1
        while start > 0 and signature(site[start-1]) == signature(last):
            start -= 1
        violation_start = None
        if not last["boundary_rule_satisfied"]:
            k = len(site)-1
            while k > 0 and not site[k-1]["boundary_rule_satisfied"]:
                k -= 1
            violation_start = site[k]["index"]
        output.append({"fixed_sampler_index": i, "terminal_boundary_rule_satisfied": last["boundary_rule_satisfied"],
                       "terminal_neighbors": json.loads(last["generated_neighbors"]),
                       "terminal_neighbor_elements": json.loads(last["neighbor_elements"]),
                       "terminal_signature_persistent_from_index": site[start]["index"],
                       "terminal_boundary_violation_persistent_from_index": violation_start})
    return output


def analyze_run(root, run, output, torch, np):
    folder = run / "research_record/attempt_000"
    manifest = read_json(run / "research_record/manifest.json")
    for relative in ("flowr/scriptutil.py", "flowr/models/mol_builder.py", "flowr/models/fm_pocket.py"):
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != manifest["source_sha256"][relative]:
            raise ValueError(f"Source changed since recording: {relative}")
    task = read_json(folder / "task_definition.json")
    if not task.get("mapping_verified"):
        raise ValueError("Fixed-atom mapping was not verified.")
    load = lambda p: torch.load(p, map_location="cpu", weights_only=True)
    inputs = load(folder / "inputs.pt")
    prior = inputs["prior"]
    if prior["coords"].shape[0] != 1:
        raise ValueError("Expected one molecule per attempt.")
    valid = prior["mask"][0].bool().numpy()
    fixed = (prior["fragment_mask"][0].bool() & prior["mask"][0].bool()).nonzero(as_tuple=True)[0].tolist()
    if fixed != task["fixed_sampler_indices"]:
        raise ValueError("Saved mask and task definition disagree.")
    tokens = source_vocab(root, prior["atomics"].shape[-1])
    scale = float(inputs["coord_scale"])
    initial = snapshot_arrays(prior, fixed, valid, scale, np)
    reference = read_v2000(folder / "reference_preprocessed.sdf")
    for f, r in zip(fixed, task["fixed_reference_indices"]):
        if tokens[int(initial["atom_classes"][f])] != reference["atoms"][r]["element"]:
            raise ValueError("Atom vocabulary disagrees with verified reference mapping.")
    arrays = {"fixed_sampler_indices": np.asarray(fixed), "initial_coords_A": initial["coords"],
              "initial_atom_classes": initial["atom_classes"], "initial_charge_classes": initial["charge_classes"],
              "initial_valid_mask": initial["valid"]}
    all_rows, stream_reports = [], {}
    for stream, dirname, pattern in (("network_prediction", "forward_calls", "call_*.pt"),
                                     ("sampler_state", "sampler_steps", "step_*.pt")):
        states, rows, indices, times, stages = [], [], [], [], []
        for path in sorted((folder / dirname).glob(pattern)):
            item = load(path)
            network = stream == "network_prediction"
            index = int(item["call_index"] if network else item["step_index"])
            time = float(item["times_scalar"][0]) if network else None
            stage = "raw_network_prediction" if network else item["stage"]
            data = item["network_output_before_postprocessing"] if network else item["current"]
            state = snapshot_arrays(data, fixed, valid, scale, np, logits=network)
            states.append(state); indices.append(index); times.append(time); stages.append(stage)
            for row in site_rows(state, initial, task, tokens, stream, index, time, np):
                row["stage"] = stage
                rows.append(row)
        if not states:
            raise ValueError(f"Missing {dirname}; use the full local run, not the review bundle.")
        expected = read_json(folder / "attempt.json")
        count = expected["forward_calls" if stream == "network_prediction" else "saved_sampler_frames"]
        if len(states) != count:
            raise ValueError(f"Incomplete {dirname}: {len(states)} of {count} snapshots")
        for key in ("coords", "atom_classes", "charge_classes", "bond_classes", "valid"):
            arrays[f"{stream}_{key}"] = np.stack([s[key] for s in states])
        if stream == "network_prediction":
            arrays[f"{stream}_bond_probabilities"] = np.stack([s["bond_probabilities"] for s in states])
        arrays[f"{stream}_indices"] = np.asarray(indices)
        stream_reports[stream] = {"indices": indices, "times": times, "stages": stages,
                                  "terminal_intervals": terminal_intervals(rows, fixed)}
        all_rows.extend(rows)
    raw_final = snapshot_arrays(load(folder / "raw_final_prediction.pt"), fixed, valid, 1.0, np)
    retained = [i for i, ok in enumerate(raw_final["valid"]) if ok and tokens[int(raw_final["atom_classes"][i])] != "<NOATOM>"]
    decoded = {}
    for filename in ("decoded_without_repair.sdf", "decoded_before_filter.sdf"):
        mol = read_v2000(folder / filename)
        if mol is None:
            decoded[filename] = {"molecule_available": False}
            continue
        if len(mol["atoms"]) != len(retained):
            raise ValueError("Decoded atom count changed; index mapping requires separate verification.")
        error = float(np.max(np.abs(np.asarray([a["coords"] for a in mol["atoms"]]) - raw_final["coords"][retained])))
        if error > 0.002:
            raise ValueError("SDF atom order/coordinates do not match raw final tensor.")
        decoded[filename] = {"molecule_available": True, "coordinate_mapping_max_abs_error_A": error,
                             "fixed_site_checks": endpoint_rows(task, reference, mol, retained)}
    for key in ("coords", "atom_classes", "charge_classes", "bond_classes", "valid"):
        arrays[f"raw_final_{key}"] = raw_final[key]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "connection_timeline.csv", all_rows)
    np.savez_compressed(output / "connection_arrays.npz", **arrays)
    report = {"run": str(run), "seed": manifest["args"]["seed"], "atom_vocabulary": tokens,
              "task_definition": task, "streams": stream_reports, "decoded_checks": decoded,
              "rule": "For each fixed atom, preserve the reference count and bond classes of edges to generated atoms; non-attachment fixed atoms have zero such edges.",
              "index_note": "All sampler indices are zero-based. Generated slot identities are meaningful within a trajectory; do not match equal indices across random seeds.",
              "coordinate_note": "Initial/network/sampler coordinates are in the same centered frame in angstroms. raw_final coordinates are in the original input frame.",
              "array_note": "Bond arrays have axes (frame, fixed-atom-list index, sampler slot[, bond class]); lower-triangle decoding matches FLOWR. Atom class <NOATOM> is excluded from site counts. PAD/unknown bond classes are retained for diagnostics, not called valid chemistry.",
              "timing_note": "Early noisy bond choices are expected. Persistent terminal patterns describe observations, not causal failure onset. Call and sampler indices are different counters.",
              "scope": "Boundary topology and fixed-atom diagnostics; no full geometry/scaffold-change assessment, hidden-layer analysis or causal intervention.",
              "manifest": manifest}
    write_json(output / "connection_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="ptp1b", choices=["ptp1b", "bace", "cdk2", "tyk2"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    import numpy as np
    import torch
    root = Path.cwd()
    runs = args.output_root.resolve()
    output = runs / f"sh_connections_{args.case}"
    output.mkdir(parents=True, exist_ok=True)
    records, completed = [], []
    for seed in dict.fromkeys(args.seeds):
        print(f"Reading {args.case}, seed {seed}...", flush=True)
        destination = output / f"seed_{seed}"
        try:
            report = analyze_run(root, runs / f"sh_record_{args.case}_seed{seed}", destination, torch, np)
            completed.append(destination)
            checks = report["decoded_checks"]["decoded_before_filter.sdf"].get("fixed_site_checks", [])
            issues = [x for x in checks if not x["boundary_rule_satisfied"] or not x["fixed_element_preserved"] or not x["fixed_charge_preserved"]]
            records.append({"seed": seed, "analysis_completed": True, "decoded_site_issues": issues})
            print(f"  Recorded site issues: {len(issues)}", flush=True)
        except Exception as exc:
            records.append({"seed": seed, "analysis_completed": False, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  CHECK: {type(exc).__name__}: {exc}", flush=True)
    write_json(output / "summary.json", {"records": records, "full_scaffold_hopping_success": "NOT_YET_ASSESSED"})
    bundle = output / "connection_trace_review.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(output / "summary.json", "summary.json")
        for folder in completed:
            for name in ("connection_report.json", "connection_timeline.csv", "connection_arrays.npz"):
                z.write(folder / name, (folder / name).relative_to(output))
    print(f"\nReview file: {bundle}", flush=True)
    return 0 if all(r["analysis_completed"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
