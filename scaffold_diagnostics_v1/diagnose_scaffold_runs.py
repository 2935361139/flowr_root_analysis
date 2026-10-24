#!/usr/bin/env python3
"""Batch observational diagnosis of existing FLOWR scaffold-hopping recordings.

Input: a completed multicase quality_report.json and original recorder folders.
Uses saved trajectories; no model loading, generation or intervention.
Official endpoint labels are inherited, not redefined by intermediate states.
Supported temporal observations: fixed neutral-carbon overvalence, fixed atom
identity/internal-bond change, and graph disconnection. Other problems remain
explicitly unlocalized. This is not causal localization or input attribution.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import traceback
import zipfile
from pathlib import Path

BOND_CODES = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
VERSION = "2026-10-21.batch-diagnostics.1"


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


def source_charge_vocab(root):
    tree = ast.parse((root/"flowr/scriptutil.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_vocab_charges")
    values = {}
    for node in function.body:
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name) and name.id in {"special_tokens", "charge_tokens"}:
                    values[name.id] = ast.literal_eval(node.value)
    return values["special_tokens"] + values["charge_tokens"]


def canonical_boundary(values, fixed, np):
    """Match FLOWR's lower-triangle bond decoder, including asymmetric inputs."""
    n = values.shape[0]
    a, b = np.asarray(fixed)[:, None], np.arange(n)[None, :]
    return values[np.maximum(a, b), np.minimum(a, b)]


def snapshot_arrays(data, fixed, valid, scale, np, logits=False):
    def array(key):
        return data[key][0].detach().cpu().float().numpy()
    full_bonds = array("bonds")
    bonds = canonical_boundary(full_bonds, fixed, np)
    if not np.isfinite(full_bonds).all():
        raise ValueError("Nonfinite bond predictions; the original records are retained.")
    for name in ("coords", "atomics", "charges"):
        if not np.isfinite(array(name)).all():
            raise ValueError(f"Nonfinite {name}; this record cannot be interpreted reliably.")
    probabilities = None
    if logits:
        e = np.exp(bonds - bonds.max(axis=-1, keepdims=True))
        probabilities = e / e.sum(axis=-1, keepdims=True)
    return {"coords": array("coords") * scale,
            "atom_classes": array("atomics").argmax(-1),
            "charge_classes": array("charges").argmax(-1),
            "bond_classes": bonds.argmax(-1),
            "full_bond_classes": full_bonds.argmax(-1),
            "bond_probabilities": probabilities,
            "valid": array("mask").astype(bool) if "mask" in data else valid.copy()}


def site_rows(state, initial, task, tokens, stream, index, time, np, charge_tokens=None):
    fixed = task["fixed_sampler_indices"]
    fixed_set = set(fixed)
    expected = expected_boundary(task)
    refs = dict(zip(fixed, task["fixed_reference_indices"]))
    active = [i for i, ok in enumerate(state["valid"]) if ok and tokens[int(state["atom_classes"][i])] != "<NOATOM>"]
    rows = []
    for k, i in enumerate(fixed):
        all_neighbors = [j for j in active if j != i and 1 <= int(state["bond_classes"][k, j]) <= 4]
        neighbors = [j for j in all_neighbors if j not in fixed_set]
        classes = [int(state["bond_classes"][k, j]) for j in neighbors]
        distances = [float(np.linalg.norm(state["coords"][i]-state["coords"][j])) for j in neighbors]
        if not all(math.isfinite(x) for x in distances):
            distances = [x if math.isfinite(x) else None for x in distances]
        probabilities = (None if state["bond_probabilities"] is None else
                         [float(1-state["bond_probabilities"][k, j, 0]) for j in neighbors])
        # The final tensor has already been restored to the protein frame.
        drift = (None if stream == "final_tensor" else
                 float(np.linalg.norm(state["coords"][i]-initial["coords"][i])))
        known_bonds = all(0 <= int(state["bond_classes"][k, j]) <= 4 for j in active if j != i)
        order_sum = (sum({1:1.0, 2:2.0, 3:3.0, 4:1.5}[int(state["bond_classes"][k,j])]
                         for j in all_neighbors) if i in active and known_bonds else None)
        element = tokens[int(state["atom_classes"][i])]
        charge_index = int(state["charge_classes"][i])
        charge = charge_tokens[charge_index] if charge_tokens is not None and charge_index < len(charge_tokens) else None
        known_charge = isinstance(charge, int)
        aromatic_incident = any(int(state["bond_classes"][k,j]) == 4 for j in all_neighbors)
        # A 1.5-per-aromatic-bond sum is not a universal RDKit valence rule
        # (e.g. fused aromatic carbon). Leave those cases to chemical decoding.
        carbon_overvalent = (bool(element == "C" and charge == 0 and order_sum > 4.0 + 1e-6)
                            if order_sum is not None and known_charge and not element.startswith("<") and not aromatic_incident else None)
        internal_known = all(0 <= int(state["bond_classes"][k,j]) <= 4 for j in fixed if j != i)
        internal_changed = (any(int(state["bond_classes"][k,j]) != int(initial["bond_classes"][k,j])
                                for j in fixed if j != i) if internal_known else None)
        element_changed = (element != tokens[int(initial["atom_classes"][i])]
                           if not element.startswith("<") else None)
        charge_changed = (charge_index != int(initial["charge_classes"][i]) if known_charge else None)
        changes = [element_changed, charge_changed, internal_changed]
        fixed_changed = (True if i not in active or any(v is True for v in changes)
                         else False if all(v is False for v in changes) else None)
        rows.append({"stream": stream, "index": index, "continuous_time": time,
                     "fixed_sampler_index": i, "fixed_reference_index": refs[i],
                     "reference_attachment_site": bool(expected[i]),
                     "expected_external_bonds": len(expected[i]), "actual_external_bonds": len(neighbors),
                     "boundary_rule_satisfied": i in active and sorted(classes) == expected[i],
                     "strict_boundary_is_diagnostic_only": True,
                     "expected_bond_classes": json.dumps(expected[i]),
                     "generated_neighbors": json.dumps(neighbors),
                     "neighbor_elements": json.dumps([tokens[int(state["atom_classes"][j])] for j in neighbors]),
                     "bond_classes": json.dumps(classes), "bond_lengths_A": json.dumps(distances),
                     "predicted_nonzero_bond_probabilities": json.dumps(probabilities),
                     "fixed_element": element, "formal_charge": charge if known_charge else None,
                     "fixed_element_changed": element_changed, "fixed_charge_class_changed": charge_changed,
                     "fixed_internal_bonds_changed": internal_changed, "fixed_chemistry_changed": fixed_changed,
                     "fixed_site_present": i in active, "explicit_bond_order_sum": order_sum,
                     "neutral_carbon_overvalent": carbon_overvalent,
                     "carbon_signature_scope":"Nonaromatic explicit bonds on a neutral carbon only; not a complete validity rule.",
                     "all_neighbors": json.dumps(all_neighbors), "incident_bond_classes_known": known_bonds,
                     "fixed_displacement_A": drift if drift is not None and math.isfinite(drift) else None})
    return rows


def graph_observation(state, tokens, np):
    active = [i for i, ok in enumerate(state["valid"]) if ok and tokens[int(state["atom_classes"][i])] != "<NOATOM>"]
    classes = state["full_bond_classes"]
    edges = {}
    for i in active:
        edges[i] = []
        for j in active:
            if i == j:
                continue
            value = int(classes[max(i,j), min(i,j)])
            if value not in range(5):
                return {"connected": None, "n_components": None, "status": "masked_or_unknown_bonds"}
            if value:
                edges[i].append(j)
    remaining = set(active)
    count = 0
    while remaining:
        stack, seen = [min(remaining)], set()
        while stack:
            i = stack.pop()
            if i not in seen:
                seen.add(i)
                stack.extend(set(edges[i]) - seen)
        remaining -= seen
        count += 1
    return {"connected": bool(active) and count == 1, "n_components": count, "status": "evaluated"}


def persistence_interval(rows, flag):
    """Describe a final suffix of observations, never infer a causal start time."""
    ordered = sorted(rows, key=lambda r:r["index"])
    if not ordered or ordered[-1].get(flag) is not True:
        return {"status": "terminal_signature_not_present_in_this_stream", "persistent_from_index": None}
    start = len(ordered)-1
    while start > 0 and ordered[start-1].get(flag) is True:
        start -= 1
    first = ordered[start]
    return {"status": "observed", "persistent_from_index": first["index"],
            "continuous_time": first.get("continuous_time"), "saved_stage": first.get("stage"),
            "last_index": ordered[-1]["index"], "n_suffix_records": len(ordered)-start,
            "preceding_index": ordered[start-1]["index"] if start else None,
            "preceding_value": ordered[start-1].get(flag) if start else None,
            "interpretation": "Start of the final uninterrupted observed signature; not the earliest causal computation."}


def diagnosis_from_rows(task, rows, graph_rows, evaluation_rows, raw_graph, radius):
    by_stage = {r["stage"]:r for r in evaluation_rows}
    raw, decoded = by_stage["raw_without_repair"], by_stage["run_decode"]
    temporal, candidates, unlocalized = [], [], []
    fixed_cases = [(i, "fixed_neutral_carbon_overvalence", "neutral_carbon_overvalent")
                   for i in raw.get("fixed_neutral_carbon_overvalent_sites", []) if raw.get("rdkit_valid") is False]
    if raw.get("fixed_neutral_carbon_overvalent_sites") and raw.get("rdkit_valid") is True:
        unlocalized.append({"problem":"carbon_heuristic_requires_review","status":"chemical_decode_valid_do_not_call_connection_failure",
                            "reason":"Explicit aromatic bond-order sums can exceed four without RDKit invalidity; validity takes precedence over this narrow heuristic."})
    if raw.get("fixed_chemistry_preserved") is False:
        changed = set(raw_graph.get("fixed_missing_sites", []))
        changed.update(raw_graph.get("fixed_element_changed_sites", []))
        changed.update(raw_graph.get("fixed_charge_changed_sites", []))
        for pair in raw_graph.get("fixed_internal_bond_changed_pairs", []):
            changed.update(pair)
        fixed_cases += [(i,"fixed_chemistry_changed","fixed_chemistry_changed") for i in sorted(changed)]
    for site, problem, flag in fixed_cases:
        intervals = {}
        for stream in ["network_prediction", "sampler_state"]:
            selected = [r for r in rows if r["stream"] == stream and r["fixed_sampler_index"] == site]
            interval = persistence_interval(selected,flag)
            intervals[stream] = interval
            if stream == "network_prediction" and interval["status"] == "observed":
                center = interval["persistent_from_index"]
                indices = [r["index"] for r in selected if abs(r["index"]-center) <= radius]
                candidates.append({"problem":problem,"fixed_sampler_index":site,"center_call":center,
                                   "observation_calls":indices,
                                   "not_a_causal_scan_plan":True,
                                   "note":"A later causal scan must also include earlier and out-of-window controls; do not restrict possible causes to this window."})
        temporal.append({"problem":problem,"fixed_sampler_index":site,"streams":intervals,
                         "evidence_level":"observational","causal_status":"not_tested"})
    if raw.get("raw_graph_connected") is False:
        intervals = {}
        for stream in ["network_prediction","sampler_state"]:
            selected = []
            for r in graph_rows:
                if r['stream'] == stream:
                    selected.append({**r,"disconnected":not r['connected'] if isinstance(r['connected'],bool) else None})
            intervals[stream] = persistence_interval(selected,"disconnected")
        temporal.append({"problem":"graph_disconnection","fixed_sampler_index":None,
                         "streams":intervals,"evidence_level":"observational","causal_status":"not_tested"})
    if raw.get("rdkit_valid") is False and not fixed_cases:
        unlocalized.append({"problem":"other_chemical_decode_invalid","status":"detected_not_temporally_localized",
                            "reason":"No supported final-site signature; do not invent a cause or drop the sample."})
    for check in sorted(set((raw.get("posebusters_failed_checks") or []) + (decoded.get("posebusters_failed_checks") or []))):
        stages = [r["stage"] for r in (raw, decoded) if check in (r.get("posebusters_failed_checks") or [])]
        unlocalized.append({"problem":check,"endpoint_stages":stages,
                            "status":"endpoint_failure_detected_not_locally_or_temporally_localized",
                            "reason":"Atom-level geometry/environment tracing requires its own adapter; a PoseBusters failure alone does not establish a scaffold-fragment connection failure."})
    for r in (raw, decoded):
        if r.get("root_fixed_substructure_match") is False and r.get("fixed_chemistry_preserved") is not False:
            unlocalized.append({"problem":"root_fixed_substructure_mismatch","endpoint_stages":[r["stage"]],
                                "status":"detected_not_temporally_localized",
                                "reason":"Substructure matching and slot-wise preservation differ; inspect this case before attributing it to a network computation."})
        if r["stage"] == "run_decode" and r.get("rdkit_valid") is False and raw.get("rdkit_valid") is not False:
            unlocalized.append({"problem":"decode_stage_chemical_invalidity","endpoint_stages":[r["stage"]],
                                "status":"detected_not_locally_localized"})
    decode_changes = []
    if raw.get("fixed_chemistry_preserved") is True and decoded.get("fixed_chemistry_preserved") is False:
        changed_sites = set(decoded.get("fixed_element_changed_sites", []))
        for key in ("fixed_missing_sites", "fixed_charge_changed_sites"):
            changed_sites.update(decoded.get(key, []))
        for pair in decoded.get("fixed_internal_bond_changed_pairs", []):
            changed_sites.update(pair)
        decode_changes.append({"observed_stage":"run_setting_decode","problem":"fixed_chemistry_changed_during_decode",
                               "changed_sites":sorted(changed_sites),
                               "note":"Observed before/after decode change, not a neural-network root-cause claim."})
    return {"temporal_observations":temporal,"candidate_observation_windows":candidates,
            "unlocalized_endpoint_problems":unlocalized,"decode_stage_observations":decode_changes,
            "scaffold_change":raw.get("murcko_scaffold_changed"),
            "scaffold_change_note":"Unchanged Murcko scaffold is reported separately, not as a connection error.",
            "causal_status":"not_tested","input_atom_contributions":"not_assessed"}


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


def analyze_run(root, run, output, torch, np, evaluation_rows, run_info, radius):
    folder = run / "research_record/attempt_000"
    manifest = read_json(run / "research_record/manifest.json")
    if manifest["args"]["seed"] != run_info["seed"]:
        raise ValueError("Recorded seed differs from the completed quality report.")
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
    charge_tokens = source_charge_vocab(root)
    if prior["charges"].shape[-1] != len(charge_tokens):
        raise ValueError("Charge vocabulary width differs; do not guess formal charges.")
    for name, expected in [("raw_final_prediction.pt",run_info["raw_final_sha256"]),
                           ("task_definition.json",run_info["task_sha256"]),
                           ("reference_preprocessed.sdf",run_info["reference_sdf_sha256"])]:
        if hashlib.sha256((folder/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Record changed since the completed quality evaluation: {name}")
    scale = float(inputs["coord_scale"])
    initial = snapshot_arrays(prior, fixed, valid, scale, np)
    reference = read_v2000(folder / "reference_preprocessed.sdf")
    for f, r in zip(fixed, task["fixed_reference_indices"]):
        if tokens[int(initial["atom_classes"][f])] != reference["atoms"][r]["element"]:
            raise ValueError("Atom vocabulary disagrees with verified reference mapping.")
    arrays = {"fixed_sampler_indices": np.asarray(fixed), "initial_coords_A": initial["coords"],
              "initial_atom_classes": initial["atom_classes"], "initial_charge_classes": initial["charge_classes"],
              "initial_valid_mask": initial["valid"]}
    all_rows, all_graph_rows, stream_reports = [], [], {}
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
            all_graph_rows.append({"stream":stream,"index":index,"continuous_time":time,"stage":stage,
                                   **graph_observation(state,tokens,np)})
            states.append(state); indices.append(index); times.append(time); stages.append(stage)
            for row in site_rows(state, initial, task, tokens, stream, index, time, np,charge_tokens):
                row["stage"] = stage
                rows.append(row)
        if not states:
            raise ValueError(f"Missing {dirname}; use the full local run, not the review bundle.")
        expected = read_json(folder / "attempt.json")
        count = expected["forward_calls" if stream == "network_prediction" else "saved_sampler_frames"]
        if len(states) != count:
            raise ValueError(f"Incomplete {dirname}: {len(states)} of {count} snapshots")
        if indices != list(range(count)):
            raise ValueError(f"Nonconsecutive/duplicated indices in {dirname}; temporal continuity is unverified.")
        for key in ("coords", "atom_classes", "charge_classes", "bond_classes", "valid"):
            arrays[f"{stream}_{key}"] = np.stack([s[key] for s in states])
        if stream == "network_prediction":
            arrays[f"{stream}_bond_probabilities"] = np.stack([s["bond_probabilities"] for s in states])
        arrays[f"{stream}_indices"] = np.asarray(indices)
        stream_reports[stream] = {"indices": indices, "times": times, "stages": stages,
                                  "strict_reference_boundary_intervals": terminal_intervals(rows, fixed),
                                  "strict_boundary_note":"Descriptive only; disagreement with the reference is not used to label a failed sample."}
        all_rows.extend(rows)
    raw_final = snapshot_arrays(load(folder / "raw_final_prediction.pt"), fixed, valid, 1.0, np)
    terminal_graph = graph_observation(raw_final,tokens,np)
    if terminal_graph["connected"] != run_info["raw_graph"]["connected"]:
        raise ValueError("Final graph connectivity differs from completed quality evaluation.")
    raw_sites = site_rows(raw_final,initial,task,tokens,"final_tensor",0,None,np,charge_tokens)
    expected_overvalent = set(run_info["raw_graph"]["fixed_neutral_carbon_overvalent_sites"])
    for row in raw_sites:
        if row["neutral_carbon_overvalent"] is not None and row["neutral_carbon_overvalent"] != (row["fixed_sampler_index"] in expected_overvalent):
            raise ValueError("Final carbon-overvalence sites differ from completed quality evaluation.")
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
    write_csv(output / "graph_timeline.csv", all_graph_rows)
    np.savez_compressed(output / "connection_arrays.npz", **arrays)
    report = {"version":VERSION,"run": str(run), "seed": manifest["args"]["seed"], "atom_vocabulary": tokens,
              "task_definition": task, "streams": stream_reports, "decoded_checks": decoded,
              "rule": "Endpoint labels are inherited from the completed multicase evaluator. Reference-boundary equality is retained only as a strict optional diagnostic and never selects failure cases.",
              "index_note": "All sampler indices are zero-based. Generated slot identities are meaningful within a trajectory; do not match equal indices across random seeds.",
              "coordinate_note": "Initial/network/sampler coordinates are in the same centered frame in angstroms. raw_final coordinates are in the original input frame.",
              "array_note": "Bond arrays have axes (frame, fixed-atom-list index, sampler slot[, bond class]); lower-triangle decoding matches FLOWR. Atom class <NOATOM> is excluded from site counts. PAD/unknown bond classes are retained for diagnostics, not called valid chemistry.",
              "timing_note": "Early noisy bond choices are expected. Persistent terminal patterns describe observations, not causal failure onset. Call and sampler indices are different counters.",
              "scope": "Boundary topology and fixed-atom diagnostics; no full geometry/scaffold-change assessment, hidden-layer analysis or causal intervention.",
              "endpoint_evaluation":evaluation_rows,
              "diagnosis":diagnosis_from_rows(task,all_rows,all_graph_rows,evaluation_rows,run_info["raw_graph"],radius),
              "manifest": manifest}
    write_json(output / "connection_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-report",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--records-root",type=Path,help="Optional parent of sh_record_CASE_seedSEED folders if records moved.")
    parser.add_argument("--window-radius",type=int,default=2,help="Observational window only; not a causal-search restriction.")
    args = parser.parse_args()
    if args.window_radius < 0:parser.error("window-radius must be nonnegative")
    quality=read_json(args.quality_report)
    if quality.get("completed") is not True or quality.get("errors"):
        parser.error("Use the completed multicase evaluation report.")
    runs=quality.get("runs",[])
    if not runs or len(runs)!=quality.get("requested_run_count"):
        parser.error("Incomplete run inventory in quality report.")
    if len({r['run_directory'] for r in runs}) != len(runs):
        parser.error("Duplicated run directories in quality report.")
    output=args.output_dir.resolve()
    if output.exists():parser.error("Output already exists; choose a new directory to preserve prior results.")
    import numpy as np
    import torch
    root = Path.cwd()
    output.mkdir(parents=True)
    records, completed = [], []
    for number,info in enumerate(runs,1):
        case,seed=info["case"],info["seed"]
        print(f"[{number}/{len(runs)}] Reading {case}, seed {seed}...", flush=True)
        destination = output / f"run_{number:03d}"
        try:
            run=Path(info['run_directory']) if args.records_root is None else args.records_root/f"sh_record_{case}_seed{seed}"
            evaluation_rows=[r for r in quality['rows'] if r['run_directory']==info['run_directory']]
            if len(evaluation_rows)!=2 or {r['stage'] for r in evaluation_rows}!={'raw_without_repair','run_decode'}:
                raise ValueError("Missing/duplicated endpoint stages for this run.")
            report = analyze_run(root,run,destination,torch,np,evaluation_rows,info,args.window_radius)
            completed.append(destination)
            records.append({"case":case,"seed":seed,"analysis_completed":True,
                            "report_directory":destination.name,**report['diagnosis']})
            print(f"  Temporal observations: {len(report['diagnosis']['temporal_observations'])}; "
                  f"endpoint problems awaiting localization: {len(report['diagnosis']['unlocalized_endpoint_problems'])}",flush=True)
        except Exception as exc:
            records.append({"case":case,"seed":seed,"analysis_completed":False,
                            "error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc()})
            print(f"  CHECK: {type(exc).__name__}: {exc}", flush=True)
    success=all(r["analysis_completed"] for r in records)
    write_json(output / "summary.json", {"version":VERSION,"completed":success,"records":records,
               "quality_report_sha256":hashlib.sha256(args.quality_report.read_bytes()).hexdigest(),
               "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scope":"Batch observational diagnosis. Completion does not mean all error types have been localized or any cause established.",
               "index_note":"Zero-based network calls and saved sampler frame indices are separate counters.",
               "unknown_policy":"Unsupported problems remain explicitly unlocalized; they are not dropped, guessed, or treated as normal."})
    write_json(output / "candidate_observation_windows.json",[
        {"case":r['case'],"seed":r['seed'],**w} for r in records for w in r.get('candidate_observation_windows',[])])
    (output/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    bundle = output / "diagnostic_review.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(output / "summary.json", "summary.json")
        z.write(output/'candidate_observation_windows.json','candidate_observation_windows.json')
        z.write(output/'diagnostic_source.py','diagnostic_source.py')
        for folder in completed:
            for name in ("connection_report.json", "connection_timeline.csv", "graph_timeline.csv", "connection_arrays.npz"):
                z.write(folder / name, (folder / name).relative_to(output))
    print(f"\nAnalysis completed: {success}\nReview file: {bundle}", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
