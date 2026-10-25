#!/usr/bin/env python3
"""Observe recorded ROOT bond heads with native GPU verification and numeric accounting.

Run beside the verified replay_connection_hidden.py in the existing ROOT repo.
No parameter updates, feature replacements, gradients or sampling continuation.
This is a local, reference-dependent decomposition, not input-atom SHAP or a
causal explanation of an entire generated trajectory.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import sys
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

HELPER_SHA = "1d5b6cfef030f85d1b0df939d96b6402d64f68e9b2a8ef0b746c12236a84ff82"
SEMLA_SHA = "e7c409d398a0cbc0332e88314d06b293a042ae36e97face937d5c17c2b4d8433"
DETAIL_GROUPS = ["fixed_node", "generated_node", "squared_distance", "coordinate_dot", "pair_features"]
MAIN_GROUPS = ["geometry", "nodes", "pairs"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def to_array(tensor):
    return tensor.detach().cpu().float().numpy().copy()


def sigmoid(x):
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e))


def silu(x):
    return x * sigmoid(x)


def softmax(x):
    p = np.exp(x - x.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def predict_head(x, w):
    """Float64 algebra of Linear -> SiLU -> Linear -> symmetry -> Linear."""
    z = x @ w["w1"].T + w["b1"]
    edge = silu(z) @ w["w2"].T + w["b2"]
    return edge.mean(axis=-2) @ w["wp"].T + w["bp"]


def group_masks(d_message, d_edge):
    """Axis -2 is direction: fixed->generated, then generated->fixed."""
    d = 2 * d_message + 2 + d_edge
    masks = {g: np.zeros((2, d), dtype=np.float64) for g in DETAIL_GROUPS}
    masks["fixed_node"][0, :d_message] = 1
    masks["fixed_node"][1, d_message:2*d_message] = 1
    masks["generated_node"][0, d_message:2*d_message] = 1
    masks["generated_node"][1, :d_message] = 1
    masks["squared_distance"][:, 2*d_message] = 1
    masks["coordinate_dot"][:, 2*d_message+1] = 1
    masks["pair_features"][:, 2*d_message+2:] = 1
    if not np.array_equal(sum(masks.values()), np.ones((2, d))):
        raise ValueError("Feature groups must form a disjoint complete partition.")
    return masks


def decompose_change(before, after, weights, d_message, d_edge, target_class=1):
    """Exact finite-change decomposition for this one-hidden-layer head.

    SiLU secants allocate each hidden-unit change over its linear input changes.
    Equivalently, this integrates head gradients along the straight line between
    the two observed MLP inputs. No intermediate input is fed into the model.
    The fixed/generator node groups are internal features, not original atoms'
    exclusive causal contributions. Interactions are allocated by this path.
    """
    a, b = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    u = a @ weights["w1"].T + weights["b1"]
    v = b @ weights["w1"].T + weights["b1"]
    delta_u = v - u
    mid = (u + v) * 0.5
    s = sigmoid(mid)
    slope = s * (1 + mid * (1-s))
    np.divide(silu(v)-silu(u), delta_u, out=slope, where=np.abs(delta_u) > 1e-8)
    margin_weight = (weights["wp"][target_class]-weights["wp"][0]) @ weights["w2"]
    contributions = {}
    for name, mask in group_masks(d_message, d_edge).items():
        branch_delta = ((b-a)*mask) @ weights["w1"].T
        contributions[name] = (branch_delta * slope * margin_weight).sum(axis=-1).mean(axis=-1)
    pa, pb = predict_head(a, weights), predict_head(b, weights)
    delta_margin = (pb[..., target_class]-pb[..., 0])-(pa[..., target_class]-pa[..., 0])
    residual = delta_margin - sum(contributions.values())
    if not np.isfinite(residual).all() or not np.allclose(residual, 0, atol=1e-7, rtol=0):
        raise RuntimeError("Head decomposition does not reconstruct the score change.")
    contributions["geometry"] = contributions["squared_distance"] + contributions["coordinate_dot"]
    contributions["nodes"] = contributions["fixed_node"] + contributions["generated_node"]
    contributions["pairs"] = contributions["pair_features"]
    return contributions, delta_margin, residual


def read_plan(path):
    p = read_json(path)
    import re
    if not isinstance(p.get("case"), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", p["case"]):
        raise ValueError("plan.case must be a recorded case name, not a path.")
    for name in ["seeds", "calls"]:
        values = p.get(name)
        if not isinstance(values, list) or not values or any(type(v) is not int or v < 0 for v in values) or len(set(values)) != len(values):
            raise ValueError(f"plan.{name} must be a nonempty list of distinct nonnegative integers.")
    if len(p["calls"]) < 2 or any(b-a != 1 for a, b in zip(p["calls"], p["calls"][1:])):
        raise ValueError("Use consecutive recorded calls in increasing order for this temporal analysis.")
    if type(p.get("site")) is not int or p["site"] < 0:
        raise ValueError("plan.site must name one fixed sampler atom.")
    if type(p.get("target_bond_class", 1)) is not int or p.get("target_bond_class", 1) not in [1, 2, 3, 4]:
        raise ValueError("target_bond_class must be 1/2/3/4; no-bond class 0 is the comparator.")
    p.setdefault("target_bond_class", 1)
    p.setdefault("focus_partners", {})
    p.setdefault("hidden_dir", f"outputs/sh_hidden_{p['case']}")
    if not isinstance(p["focus_partners"], dict):
        raise ValueError("focus_partners must map seeds to generated sampler indices.")
    for key, values in p["focus_partners"].items():
        if key not in {str(s) for s in p["seeds"]} or not isinstance(values, list) or any(type(v) is not int or v < 0 for v in values) or len(values) != len(set(values)):
            raise ValueError("Invalid focus_partners entry.")
    return p


def generated_partners(task, mask, site):
    """Read each trajectory's own mapping; no assumption of contiguous fixed IDs."""
    fixed = list(task["fixed_sampler_indices"])
    valid = np.asarray(mask).astype(bool)
    if not task.get("mapping_verified") or len(fixed) != len(set(fixed)) or any(i < 0 or i >= len(valid) for i in fixed):
        raise ValueError("Invalid fixed-atom mapping.")
    if site not in fixed or not valid[site] or not valid[fixed].all():
        raise ValueError("Requested site/fixed fragment is not valid in this trajectory.")
    partners = [int(i) for i in np.flatnonzero(valid) if i not in set(fixed)]
    if not partners:
        raise ValueError("No valid generated slots to analyze.")
    return partners


class HeadObserver:
    """Copies selected arrays plus full GPU tensors for same-shape native checks."""
    def __init__(self, decoder, site, partners):
        self.site, self.partners = site, list(partners)
        self.arrays, self.full, self.visits, self.handles = {}, {}, {}, []
        self.add(decoder.bond_refine, "refine", self.refine, pre=True)
        self.add(decoder.bond_refine.node_proj, "node_projection", self.node_projection)
        self.add(decoder.bond_refine.message_mlp[0], "mlp_linear1", self.linear1)
        self.add(decoder.bond_refine.message_mlp[1], "mlp_silu", self.activation)
        self.add(decoder.bond_refine.message_mlp[2], "mlp_linear2", self.linear2)
        self.add(decoder.bond_proj, "bond_projection", self.projection)

    def add(self, module, name, fn, pre=False):
        self.visits[name] = 0
        def hook(*args):
            self.visits[name] += 1
            if self.visits[name] != 1:
                raise RuntimeError(f"Unexpected repeated invocation: {name}")
            fn(*args)
            return None
        register = module.register_forward_pre_hook if pre else module.register_forward_hook
        self.handles.append(register(hook))

    def pairs(self, tensor):
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError("Expected batch-one pair tensor.")
        return np.stack([to_array(tensor[0, self.site, self.partners]),
                         to_array(tensor[0, self.partners, self.site])], axis=1)

    def pair_save(self, name, value):
        self.full[name] = value.detach().clone()
        self.arrays[name] = self.pairs(value)

    def refine(self, module, inputs):
        if len(inputs) != 4:
            raise ValueError("Unexpected BondRefine arguments.")
        self.full["head_coords"] = inputs[0].detach().clone()
        self.arrays["head_coords"] = to_array(inputs[0][0])
        self.arrays["head_mask"] = to_array(inputs[2][0])

    def node_projection(self, module, inputs, output):
        self.arrays["projected_nodes"] = to_array(output[0])

    def linear1(self, module, inputs, output):
        self.pair_save("mlp_input", inputs[0])
        self.pair_save("linear1_output", output)

    def activation(self, module, inputs, output):
        self.pair_save("silu_output", output)

    def linear2(self, module, inputs, output):
        self.pair_save("linear2_output", output)

    def projection(self, module, inputs, output):
        self.pair_save("projection_input", inputs[0])
        self.pair_save("projection_output", output)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def inspect_architecture(decoder, torch):
    br, nn = decoder.bond_refine, torch.nn
    mlp = br.message_mlp
    if br.norm_feats or len(mlp) != 3 or not isinstance(mlp[0], nn.Linear) or not isinstance(mlp[1], nn.SiLU) or not isinstance(mlp[2], nn.Linear) or not isinstance(decoder.bond_proj, nn.Linear):
        raise ValueError("Unsupported bond head; this adapter is specific to inspected ROOT.")
    dm, de = br.node_proj.out_features, mlp[2].out_features
    if mlp[0].in_features != 2*dm+2+de or decoder.bond_proj.out_features != 5:
        raise ValueError("Unexpected feature layout or bond vocabulary.")
    w = {}
    for label, module in {"1": mlp[0], "2": mlp[2], "p": decoder.bond_proj}.items():
        if module.bias is None or module.weight.dtype != torch.float32:
            raise ValueError("Expected recorded float32 linear weights and biases.")
        w["w"+label] = to_array(module.weight).astype(np.float64)
        w["b"+label] = to_array(module.bias).astype(np.float64)
    return dm, de, w


def compare_arrays(a, b, atol=1e-5, rtol=1e-5, classes=False):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {"passed": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    result = {"passed": finite and bool(np.allclose(a, b, atol=atol, rtol=rtol)),
              "max_abs_error": float(np.max(np.abs(a-b))) if finite and a.size else None,
              "atol": atol, "rtol": rtol}
    if classes:
        result["argmax_changes"] = int(np.count_nonzero(a.argmax(-1) != b.argmax(-1)))
        result["passed"] = result["passed"] and result["argmax_changes"] == 0
    return result


def validate_layout(a, dm, de):
    site, partners = int(a["site"]), a["partners"].astype(int)
    x, nodes = a["mlp_input"], a["projected_nodes"]
    if x.shape != (len(partners), 2, 2*dm+2+de) or a["head_coords"].shape != (len(a["head_mask"]), 3):
        raise ValueError("Unexpected observation shapes.")
    if not all(np.isfinite(v).all() for v in a.values()):
        raise ValueError("Non-finite captured data.")
    if not np.all(a["head_mask"][[site, *partners]] == 1):
        raise ValueError("A selected atom slot is masked out.")
    for direction in [0, 1]:
        fixed = np.broadcast_to(nodes[site], (len(partners), dm))
        left, right = (fixed, nodes[partners]) if direction == 0 else (nodes[partners], fixed)
        if not (np.array_equal(x[:, direction, :dm], left) and np.array_equal(x[:, direction, dm:2*dm], right)):
            raise ValueError("Fixed/generated node feature groups are incorrectly mapped.")
    if not np.array_equal(a["projection_output"][:, 0], a["projection_output"][:, 1]):
        raise ValueError("Observed projection is unexpectedly asymmetric.")
    if not compare_arrays(a["projection_output"][:, 1], a["bond_logits"], classes=True)["passed"]:
        raise ValueError("Head outputs do not match canonical molecular pair logits.")
    return True


def native_head_check(observer, decoder, dm, weights, torch):
    """Never rerun the generator with changed precision. Reference head only."""
    f = torch.nn.functional
    full = observer.full
    if any(t.dtype != torch.float32 for t in full.values()):
        raise ValueError("Native checks currently support the recorded float32 head only.")
    prior = torch.get_float32_matmul_precision()
    report = {"original_precision": prior, "native": {}, "ieee_reference": {},
              "native_passed": False, "reference_passed": False, "precision_restored": False}
    def run_head(x):
        u = f.linear(x, decoder.bond_refine.message_mlp[0].weight, decoder.bond_refine.message_mlp[0].bias)
        a = f.silu(u, inplace=False)
        v = f.linear(a, decoder.bond_refine.message_mlp[2].weight, decoder.bond_refine.message_mlp[2].bias)
        symmetric = .5*(v+v.transpose(1, 2))
        logits = f.linear(symmetric, decoder.bond_proj.weight, decoder.bond_proj.bias)
        return {"linear1_output": u, "silu_output": a, "linear2_output": v,
                "projection_input": symmetric, "projection_output": logits}
    try:
        with torch.no_grad():
            # Same shape, device, weights and precision as the true forward.
            native = run_head(full["mlp_input"])
            for name, value in native.items():
                report["native"][name] = compare_arrays(to_array(value), to_array(full[name]), classes=name=="projection_output")
            xyz = full["head_coords"]
            diffs = xyz.unsqueeze(2)-xyz.unsqueeze(1)
            dist = (diffs*diffs).sum(-1).unsqueeze(-1)
            dots = torch.bmm(xyz, xyz.transpose(1, 2)).unsqueeze(-1)
            geo = torch.cat([dist, dots], dim=-1)
            report["native"]["geometry_columns"] = compare_arrays(to_array(geo), to_array(full["mlp_input"][..., 2*dm:2*dm+2]))
            report["native_passed"] = all(v["passed"] for v in report["native"].values())
            del native, xyz, diffs, dist, dots, geo
            # An independent numerical reference on recorded head inputs only.
            # Restore the original mode before any further model forward.
            torch.set_float32_matmul_precision("highest")
            reference = run_head(full["mlp_input"])
            x64 = to_array(full["mlp_input"]).astype(np.float64)
            u64 = x64@weights["w1"].T+weights["b1"]
            a64 = silu(u64)
            v64 = a64@weights["w2"].T+weights["b2"]
            s64 = .5*(v64+v64.swapaxes(1, 2))
            p64 = s64@weights["wp"].T+weights["bp"]
            expected = {"linear1_output": u64, "silu_output": a64, "linear2_output": v64,
                        "projection_input": s64, "projection_output": p64}
            for name, value in reference.items():
                report["ieee_reference"][name] = compare_arrays(to_array(value), expected[name], atol=2e-4, rtol=1e-5)
            report["reference_passed"] = all(v["passed"] for v in report["ieee_reference"].values())
            report["original_vs_ieee_first_linear"] = compare_arrays(to_array(full["linear1_output"]), to_array(reference["linear1_output"]), atol=2e-4, rtol=1e-5)
            report["original_vs_ieee_logits"] = compare_arrays(to_array(full["projection_output"]), to_array(reference["projection_output"]), atol=2e-4, rtol=1e-5, classes=True)
            observer.arrays["ieee_head_logits"] = observer.pairs(reference["projection_output"])[:, 1]
    finally:
        torch.set_float32_matmul_precision(prior)
        report["precision_restored"] = torch.get_float32_matmul_precision() == prior
    report["precision_mode_explains_difference"] = bool(report["native_passed"] and report["reference_passed"] and not report["original_vs_ieee_first_linear"]["passed"])
    report["passed"] = report["native_passed"] and report["reference_passed"] and report["precision_restored"]
    return report


def save_observation(out, entry, observer):
    path = out/entry["file"]
    np.savez_compressed(path, **observer.arrays)
    entry["sha256"] = sha256(path)


def capture(args, plan, out):
    root = Path.cwd()
    helper = root/"replay_connection_hidden.py"
    if not helper.is_file() or sha256(helper) != HELPER_SHA:
        raise ValueError("Keep the previously verified replay_connection_hidden.py v2 beside this script.")
    sys.path.insert(0, str(root))
    import torch
    import replay_connection_hidden as helpers
    from flowr.scriptutil import load_model
    from flowr.util.device import resolve_device
    hidden = Path(args.hidden_dir or plan["hidden_dir"]).resolve()
    prior = read_json(hidden/"replay_summary.json")
    accepted = {(r["seed"], r["call_index"]) for r in prior["call_results"] if r["accepted"]}
    requested = {(s, c) for s in plan["seeds"] for c in plan["calls"]}
    if not prior.get("all_requested_calls_verified") or not requested.issubset(accepted):
        raise ValueError("Selected states must already have verified hidden replays.")
    runs, case, site = args.output_root.resolve(), plan["case"], plan["site"]
    manifests = {s: read_json(runs/f"sh_record_{case}_seed{s}/research_record/manifest.json") for s in plan["seeds"]}
    base = manifests[plan["seeds"][0]]
    for seed, manifest in manifests.items():
        meta = read_json(hidden/f"seed_{seed}/metadata.json")
        if manifest["args"]["seed"] != seed or helpers.condition_key(manifest) != helpers.condition_key(base) or meta["recorded_manifest"] != manifest:
            raise ValueError("Recorded conditions or earlier metadata differ.")
    sources = {**base["source_sha256"], **helpers.HOOK_SOURCE_HASHES, "flowr/models/semla.py": SEMLA_SHA}
    for filename, expected in sources.items():
        if sha256(root/filename) != expected:
            raise ValueError(f"Source differs from inspected ROOT: {filename}")
    if sha256(Path(base["args"]["ckpt_path"])) != base["checkpoint_sha256"] or str(torch.__version__) != base["torch_version"]:
        raise ValueError("Use the recorded checkpoint and PyTorch environment.")
    report = {"version": "2026-10-19.2", "plan": plan, "capture_completed": False,
              "analysis_completed": False, "states": [], "metadata": {}, "error": None,
              "checkpoint_sha256": base["checkpoint_sha256"], "source_sha256": sources,
              "script_sha256": sha256(Path(__file__)), "helper_sha256": sha256(helper),
              "previous_summary_sha256": sha256(hidden/"replay_summary.json"),
              "scope": "Local score-change decomposition of a float64 head reference with an explicit observed-minus-reference numerical term. Not exact attribution of the GPU numerical program, not input SHAP or trajectory causality.",
              "generalization": "Configurable cases/seeds/calls/site on this ROOT adapter. Configuration portability is not independently demonstrated explanation generalization. This is not an automatic failure detector."}
    write_json(out/"summary.json", report)
    load = lambda p: torch.load(p, map_location="cpu", weights_only=True)
    observer = None
    try:
        torch.set_float32_matmul_precision("high")  # Original recording mode, unchanged.
        model_args = argparse.Namespace(**base["args"])
        model_args.save_dir = str(out/"model_context")
        device = resolve_device(model_args)
        print(f"Loading recorded model on {device}; native replay remains in original precision.", flush=True)
        model = load_model(model_args)[0].to(device).eval()
        decoder = model.gen.ligand_dec
        dm, de, weights = inspect_architecture(decoder, torch)
        report.update(d_message=dm, d_edge=de, device=str(device), torch_version=str(torch.__version__),
                      parameter_tensors_requiring_grad=sum(p.requires_grad for p in model.parameters()))
        np.savez_compressed(out/"head_weights.npz", **weights)
        report["weights_sha256"] = sha256(out/"head_weights.npz")
        for seed in plan["seeds"]:
            folder = runs/f"sh_record_{case}_seed{seed}/research_record/attempt_000"
            task, inputs = read_json(folder/"task_definition.json"), load(folder/"inputs.pt")
            if bool(model.self_condition) != inputs["self_condition"] or int(model.sc_recycle_steps) != inputs["sc_recycle_steps"]:
                raise ValueError("Self-conditioning settings changed.")
            pocket = helpers.move_copy(inputs["pocket_data"], device, torch)
            encoding = helpers.move_copy(load(folder/"pocket_encoding.pt"), device, torch)
            if encoding["pocket_equis"] is None or encoding["pocket_invs"] is None:
                raise ValueError("Missing original pocket encoding.")
            seed_out = out/f"seed_{seed}"
            seed_out.mkdir()
            report["metadata"][str(seed)] = {"task": task, "coord_scale": inputs["coord_scale"], "manifest": manifests[seed]}
            expected_partners = None
            for index in plan["calls"]:
                print(f"Observe head: seed {seed}, call {index}, fixed site {site}", flush=True)
                entry = {"seed": seed, "call": index, "accepted": False,
                         "file": str(Path(f"seed_{seed}")/f"call_{index:04d}_head.npz")}
                report["states"].append(entry)
                call = load(folder/"forward_calls"/f"call_{index:04d}.pt")
                if call["call_index"] != index:
                    raise ValueError("Recorded call index does not match filename.")
                original = call["network_output_before_postprocessing"]
                mask = to_array(original["mask"])[0]
                partners = generated_partners(task, mask, site)
                if expected_partners is not None and expected_partners != partners:
                    raise ValueError("Valid generated slot mapping changed within the trajectory.")
                expected_partners = partners
                if not set(plan["focus_partners"].get(str(seed), [])).issubset(partners):
                    raise ValueError("A requested display partner is not a valid generated slot.")
                baseline = helpers.forward_once(model, call, pocket, encoding, device, torch, np)
                entry["times"] = call["times_scalar"]
                entry["replay_vs_original"] = helpers.compare_predictions(baseline, original, torch, np, 1e-5, 1e-5)
                if not entry["replay_vs_original"]["passed"]:
                    raise RuntimeError("Unmodified replay differs from original; no attribution accepted.")
                observer = HeadObserver(decoder, site, partners)
                try:
                    observed = helpers.forward_once(model, call, pocket, encoding, device, torch, np)
                finally:
                    observer.close()
                a = observer.arrays
                a.update(site=np.array(site), partners=np.array(partners, dtype=np.int64),
                         fixed_indices=np.array(task["fixed_sampler_indices"], dtype=np.int64))
                logits = to_array(observed["bonds"])[0]
                slots = np.arange(len(mask))
                a["bond_logits"] = logits[np.maximum(site, partners), np.minimum(site, partners)]
                a["site_bond_logits"] = logits[np.maximum(site, slots), np.minimum(site, slots)]
                a["atom_classes"] = to_array(observed["atomics"])[0].argmax(-1)
                a["charge_classes"] = to_array(observed["charges"])[0].argmax(-1)
                a["output_mask"] = to_array(observed["mask"])[0]
                # Persist the observation BEFORE any further validation can fail.
                save_observation(out, entry, observer)
                entry["observation_vs_replay"] = helpers.compare_predictions(observed, baseline, torch, np, 1e-5, 1e-5)
                entry["observation_vs_original"] = helpers.compare_predictions(observed, original, torch, np, 1e-5, 1e-5)
                if not entry["observation_vs_replay"]["passed"] or not entry["observation_vs_original"]["passed"] or any(v != 1 for v in observer.visits.values()):
                    raise RuntimeError("Read-only observations differ from original output.")
                if not np.array_equal(a["head_coords"], to_array(observed["coords"])[0]):
                    raise ValueError("Head coordinates differ from raw model output coordinates.")
                validate_layout(a, dm, de)
                entry["numerics"] = native_head_check(observer, decoder, dm, weights, torch)
                save_observation(out, entry, observer)
                if not entry["numerics"]["passed"]:
                    raise RuntimeError("Native head or independent IEEE reference did not reproduce. Diagnostic state has been saved.")
                entry["accepted"] = True
                write_json(out/"summary.json", report)
                print("  MATCH: original replay and native head verified; numeric reference recorded.", flush=True)
                observer = None
                del a, observed, baseline, call
        report["capture_completed"] = len(report["states"]) == len(requested) and all(s["accepted"] for s in report["states"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if observer is not None:
            try:
                np.savez_compressed(out/"failed_full_head.npz", **{k: to_array(v) for k, v in observer.full.items()})
            except Exception as saving_exc:
                report["failure_save_error"] = str(saving_exc)
        raise
    finally:
        if observer is not None:
            observer.close()
        write_json(out/"summary.json", report)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_transition(a, b, w, dm, de, target):
    values, delta, residual = decompose_change(a["mlp_input"], b["mlp_input"], w, dm, de, target)
    la, lb = a["bond_logits"].astype(np.float64), b["bond_logits"].astype(np.float64)
    ra, rb = predict_head(a["mlp_input"].astype(np.float64), w), predict_head(b["mlp_input"].astype(np.float64), w)
    observed = (lb[:, target]-lb[:, 0])-(la[:, target]-la[:, 0])
    numerical = observed-delta
    accounting = observed-(values["geometry"]+values["nodes"]+values["pairs"]+numerical)
    if not np.allclose(accounting, 0, atol=1e-7, rtol=0):
        raise RuntimeError("Explicit numerical accounting failed.")
    rows = []
    for i, partner in enumerate(a["partners"].tolist()):
        stable = bool(la[i].argmax()==ra[i].argmax() and lb[i].argmax()==rb[i].argmax())
        dominates = bool(abs(numerical[i]) >= abs(observed[i]) and abs(numerical[i]) > 1e-8)
        row = {"partner": partner, "class_before": int(la[i].argmax()), "class_after": int(lb[i].argmax()),
               "observed_score_change": float(observed[i]), "reference_score_change": float(delta[i])}
        row.update({k: float(values[k][i]) for k in [*MAIN_GROUPS, *DETAIL_GROUPS]})
        row.update(numerical_difference=float(numerical[i]), reference_decomposition_residual=float(residual[i]),
                   accounting_residual=float(accounting[i]), endpoint_class_agreement=stable,
                   numerical_difference_at_least_total=dominates,
                   needs_precision_review=not stable or dominates)
        rows.append(row)
    return rows


def analyze(out):
    report = read_json(out/"summary.json")
    plan = report["plan"]
    requested = {(s, c) for s in plan["seeds"] for c in plan["calls"]}
    if not report.get("capture_completed") or len(report["states"]) != len(requested) or {(e["seed"], e["call"]) for e in report["states"]} != requested:
        raise ValueError("All selected states must pass native capture validation.")
    if sha256(out/"head_weights.npz") != report["weights_sha256"]:
        raise ValueError("Exported weights changed.")
    w = {k: v.astype(np.float64) for k, v in load_npz(out/"head_weights.npz").items()}
    dm, de, site, target = report["d_message"], report["d_edge"], plan["site"], plan["target_bond_class"]
    states, changes, cache = [], [], {}
    for entry in report["states"]:
        if not entry.get("accepted") or not entry["numerics"]["passed"]:
            raise ValueError("An unverified state cannot enter the analysis.")
        path = out/entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Captured file changed: {entry['file']}")
        a = load_npz(path)
        validate_layout(a, dm, de)
        if int(a["site"]) != site:
            raise ValueError("Captured site differs from analysis plan.")
        reference = predict_head(a["mlp_input"].astype(np.float64), w)
        if not compare_arrays(reference, a["ieee_head_logits"], atol=2e-4, rtol=1e-5)["passed"]:
            raise ValueError("Offline head does not match its independent IEEE reference.")
        seed, call = entry["seed"], entry["call"]
        cache[(seed, call)] = a
        logits = a["bond_logits"].astype(np.float64)
        probs = softmax(logits)
        for i, partner in enumerate(a["partners"].tolist()):
            states.append({"seed": seed, "call": call, "site": site, "partner": partner,
                           "class_observed": int(logits[i].argmax()), "class_reference": int(reference[i].argmax()),
                           "margin_observed": float(logits[i,target]-logits[i,0]),
                           "margin_reference": float(reference[i,target]-reference[i,0]),
                           "p_any_bond_observed": float(1-probs[i,0]),
                           "distance_A": float(np.linalg.norm(a["head_coords"][site]-a["head_coords"][partner])*report["metadata"][str(seed)]["coord_scale"])})
    display = {}
    for seed in plan["seeds"]:
        # Display selection only; every valid generated partner is analyzed.
        explicit = plan["focus_partners"].get(str(seed))
        if explicit:
            display[str(seed)] = explicit
        else:
            pool = [r for r in states if r["seed"]==seed]
            peak = {p: max(r["p_any_bond_observed"] for r in pool if r["partner"]==p) for p in {r["partner"] for r in pool}}
            display[str(seed)] = sorted(peak, key=lambda p:(-peak[p],p))[:2]
        for c0,c1 in zip(plan["calls"][:-1],plan["calls"][1:]):
            a,b=cache[(seed,c0)],cache[(seed,c1)]
            if not np.array_equal(a["partners"],b["partners"]):
                raise ValueError("Generated-slot mapping changed within a trajectory.")
            for r in pair_transition(a,b,w,dm,de,target):
                changes.append({"seed":seed,"call_from":c0,"call_to":c1,"site":site,**r})
    report.update(display_partners=display, n_pair_transitions=len(changes), analysis_completed=True,
                  precision_sensitive_transitions=sum(r["needs_precision_review"] for r in changes),
                  max_numerical_difference=max(abs(r["numerical_difference"]) for r in changes),
                  max_accounting_residual=max(abs(r["accounting_residual"]) for r in changes))
    write_csv(out/"head_states.csv", states)
    write_csv(out/"head_changes.csv", changes)
    make_report(out,changes,report)
    write_json(out/"summary.json",report)


def make_report(out,changes,report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plan=report["plan"]
    panels=[(s,p) for s in plan["seeds"] for p in report["display_partners"][str(s)]]
    columns=min(2,len(panels)); rows=(len(panels)+columns-1)//columns
    fig,axes=plt.subplots(rows,columns,figsize=(7*columns,3.8*rows),squeeze=False,constrained_layout=True)
    for ax,(seed,partner) in zip(axes.flat,panels):
        data=[r for r in changes if r["seed"]==seed and r["partner"]==partner]
        x=np.arange(len(data))
        for j,(key,color) in enumerate(zip([*MAIN_GROUPS,"numerical_difference"],["#287b96","#df842f","#7755a3","#777777"])):
            ax.bar(x+(j-1.5)*.20,[r[key] for r in data],width=.18,color=color,label=key)
        ax.plot(x,[r["observed_score_change"] for r in data],"ko-",ms=3,lw=1,label="observed total")
        flagged=[i for i,r in enumerate(data) if r["needs_precision_review"]]
        if flagged:
            ax.scatter(flagged,[data[i]["observed_score_change"] for i in flagged],marker="x",s=70,color="red",label="precision review",zorder=5)
        ax.axhline(0,color="#777",lw=.6)
        ax.set_xticks(x,[f"{r['call_from']}>{r['call_to']}" for r in data],rotation=45)
        ax.set_title(f"Seed {seed}: fixed {plan['site']} - generated {partner}")
        ax.set_ylabel(f"Change in class-{plan['target_bond_class']} minus no-bond logit")
        ax.grid(axis="y",alpha=.15)
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(panels):]:ax.set_visible(False)
    title="Head reference contributions + separate numerical difference"
    if report.get("synthetic_test_only"):title="SYNTHETIC TEST ONLY - NOT USER EXPERIMENT DATA"
    fig.suptitle(title)
    picture=out/"head_change_decomposition.png"
    fig.savefig(picture,dpi=150);plt.close(fig)
    focus=[r for r in changes if r["partner"] in report["display_partners"][str(r["seed"])]]
    cells="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in [r["seed"],f"{r['call_from']}→{r['call_to']}",f"{r['site']}–{r['partner']}",f"{r['observed_score_change']:.5f}",f"{r['geometry']:.5f}",f"{r['nodes']:.5f}",f"{r['pairs']:.5f}",f"{r['numerical_difference']:.5f}","需检查精度" if r["needs_precision_review"] else "查看数值差异列"])+"</tr>" for r in focus)
    png=base64.b64encode(picture.read_bytes()).decode("ascii")
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>成键头分析 v2</title>
<style>body{{max-width:1200px;margin:30px auto;padding:0 20px;font:17px/1.7 sans-serif;color:#24313b}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}th{{background:#edf3f6}}img{{width:100%}}.box{{background:#fff0d9;padding:16px}}</style>
<h1>成键头分析：真实回放与数值误差分开核对</h1>
<p class="box">三路贡献分解的是双精度数学参考计算。实际 GPU 得分变化 = 三路参考贡献之和 + 数值差异。灰色数值差异不能当作分子机制贡献，也不能省略。这里没有证明失败原因。</p>
<p>已验证 {len(report['states'])} 次原始状态回放和只读观察，并按原始 GPU 形状与精度复现成键头；之后只在已记录的成键头输入上进行 IEEE 数值核对，未修改原始生成轨迹。</p>
<p>共 {report['n_pair_transitions']} 条原子对转换记录；其中 {report['precision_sensitive_transitions']} 条存在类别判断不一致，或数值差异不小于实际得分变化，必须优先检查。这些记录不等于独立分子样本；未标红也不代表因果归因已经可靠。</p>
<img alt="三路参考贡献和单独列出的数值差异" src="data:image/png;base64,{png}">
<h2>指定连接的变化</h2><p>正贡献推动目标键类相对无键得分上升。多路贡献可相互抵消，不能把大小直接理解为原因或重要性百分比。</p>
<table><tr><th>种子</th><th>调用</th><th>原子对</th><th>实际变化</th><th>几何</th><th>节点</th><th>原子对</th><th>数值差异</th><th>提示</th></tr>{cells}</table>
<h2>解释边界与泛化性</h2><p>每条轨迹均分析所选固定原子到所有有效生成槽位的连接。展示位点来自配置；未指定时按窗口内最大连接概率选前两个，仅用于展示。没有跨种子对齐生成原子。原子数和固定片段映射从记录读取，不能将可改配置理解为泛化效果已经得到验证。</p>
<p>本工具分析给定连接和时间窗口，不会自动发现所有失败。质量检查负责筛查已定义的失败；定位其它位点需要选择相应映射，解释其它错误可能需要新的指标和机制假设。不同模型需新的接口适配。</p>
<p>固定节点表征已混合环境信息，不等于固定输入原子的 Shapley 贡献。原子和原子对表征也可能编码几何，三路不是独立信息来源。贡献依赖参考调用和分配路径。</p>
<p>数值差异最大值：{report['max_numerical_difference']:.6g}；包含数值项后的最大核算残差：{report['max_accounting_residual']:.6g}。残差小只表示账目相符，不表示解释已被证明。</p></html>'''
    (out/"REPORT.html").write_text(doc,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,default=Path(__file__).with_name("ptp1b_plan.json"))
    parser.add_argument("--output-root",type=Path,default=Path("outputs"))
    parser.add_argument("--hidden-dir",type=Path)
    parser.add_argument("--save-dir",type=Path)
    parser.add_argument("--analyze-dir",type=Path,help="Analyze an already verified v2 capture offline.")
    args=parser.parse_args()
    if args.analyze_dir:
        analyze(args.analyze_dir.resolve());return 0
    plan=read_plan(args.plan)
    out=(args.save_dir or Path("outputs")/f"sh_head_{plan['case']}_v2_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    if out.exists():parser.error(f"Keep existing output and choose a new --save-dir: {out}")
    if not (Path.cwd()/"flowr/scriptutil.py").is_file():parser.error("Run from the existing FLOWR.ROOT repo with its .venv active.")
    out.mkdir(parents=True)
    write_json(out/"analysis_plan.json",plan)
    success=False
    try:
        capture(args,plan,out)
        analyze(out)
        report=read_json(out/"summary.json")
        success=True
        print("Native replay verified: True",flush=True)
        print("Contribution tables written: True",flush=True)
        print(f"Precision-sensitive transitions: {report['precision_sensitive_transitions']}",flush=True)
        print(f"Report: {out/'REPORT.html'}",flush=True)
    except Exception as exc:
        (out/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        if (out/"summary.json").is_file():
            report=read_json(out/"summary.json");report.update(error=f"{type(exc).__name__}: {exc}",analysis_completed=False);write_json(out/"summary.json",report)
        print(f"STOP: {type(exc).__name__}: {exc}",flush=True)
    finally:
        bundle=out/"head_review.zip"
        with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
            for path in sorted(out.rglob("*")):
                if path.is_file() and path!=bundle and "model_context" not in path.relative_to(out).parts:z.write(path,path.relative_to(out))
        print(f"Review file: {bundle}",flush=True)
        print(f"Completed: {success} (completion is not causal validation)",flush=True)
    return 0 if success else 1


if __name__=="__main__":
    raise SystemExit(main())


