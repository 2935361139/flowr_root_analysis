#!/usr/bin/env python3
"""Replay recorded FLOWR forward calls and observe selected hidden activations.

Run in the existing FLOWR.ROOT environment: python replay_connection_hidden.py
Defaults: two preliminary controls (42,43), two violations (47,49), selected calls.
Every recorded call is replayed without hooks first, then with observation hooks.
Both must match the original molecular predictions before activations are accepted.
No training, parameter edits, interventions, or sampling rollout is performed.
Preserves the checkpoint's native requires_grad flags; every forward uses no_grad.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import traceback
import zipfile
from collections.abc import Mapping
from pathlib import Path

DEFAULT_CALLS = [0, 1, 2, 20, 30, 37, 38, 39, 50, 83, 84, 99, 100]
HOOK_SOURCE_HASHES = {
    "flowr/models/pocket.py": "ccc5199b91c46dba0c24b0a98f68085a7b7a9998970485a770d9823ae197a966",
    "flowr/models/pocket_util.py": "8bdc8f3635ece1b8e16eb4e8846e464cb5e6f8dbdd2faa51e75f9adb86830cfd",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def condition_key(manifest):
    return {"args": {k: v for k, v in manifest["args"].items() if k not in {"seed", "save_dir"}},
            "source": manifest["source_sha256"], "checkpoint": manifest["checkpoint_sha256"],
            "inputs": manifest["input_sha256"], "torch_version": manifest["torch_version"]}


def move_copy(value, device, torch):
    if torch.is_tensor(value):
        return value.to(device).clone()
    if isinstance(value, Mapping):
        return {k: move_copy(v, device, torch) for k, v in value.items()}
    if isinstance(value, list):
        return [move_copy(v, device, torch) for v in value]
    if isinstance(value, tuple):
        return tuple(move_copy(v, device, torch) for v in value)
    return value


def restore_rng(saved, device, torch, np):
    random.setstate(saved["python"])
    n = saved["numpy"]
    np.random.set_state((n[0], n[1].cpu().numpy().astype(np.uint32), int(n[2]), int(n[3]), float(n[4])))
    torch.set_rng_state(saved["torch_cpu"].cpu())
    if "torch_cuda" in saved:
        if device.type != "cuda":
            raise RuntimeError("Original forward ran on CUDA; use the same working GPU environment for replay.")
        torch.cuda.set_rng_state(saved["torch_cuda"].cpu(), device=device)


def tensor_leaves(value, torch, prefix=""):
    if torch.is_tensor(value):
        return {prefix: value}
    result = {}
    if isinstance(value, Mapping):
        for k, v in value.items():
            result.update(tensor_leaves(v, torch, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(value, (list, tuple)):
        for k, v in enumerate(value):
            result.update(tensor_leaves(v, torch, f"{prefix}.{k}"))
    return result


def compare_predictions(actual, expected, torch, np, atol, rtol):
    a, b = tensor_leaves(actual, torch), tensor_leaves(expected, torch)
    fields = {}
    required = {"coords", "atomics", "bonds", "charges", "mask"}
    passed = required.issubset(a) and required.issubset(b) and set(a) == set(b)
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b or tuple(a[name].shape) != tuple(b[name].shape):
            fields[name] = {"passed": False, "reason": "missing tensor or shape mismatch"}
            passed = False
            continue
        x = a[name].detach().cpu().float().numpy().astype(np.float64)
        y = b[name].detach().cpu().float().numpy().astype(np.float64)
        finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
        close = finite and bool(np.allclose(x, y, atol=atol, rtol=rtol))
        changes = None
        if name in {"atomics", "bonds", "charges", "hybridization"}:
            changes = int(np.count_nonzero(x.argmax(-1) != y.argmax(-1)))
            close = close and changes == 0
        maximum = float(np.max(np.abs(x-y))) if x.size and finite else (0.0 if not x.size else None)
        fields[name] = {"passed": bool(close), "max_abs_difference": maximum,
                        "argmax_changes": changes, "finite": finite}
        passed = passed and close
    return {"passed": bool(passed), "atol": atol, "rtol": rtol,
            "note": "Coordinates are in model units; all tensor outputs are compared. Molecular categorical argmax decisions must also be identical.",
            "fields": fields}


class HiddenObserver:
    """Copies observations immediately and always returns None from hooks."""
    def __init__(self, model, tracked, sites, torch):
        self.tracked, self.sites, self.torch = tracked, sites, torch
        self.arrays, self.handles = {}, []
        self.layer_visits = {}
        decoder = model.gen.ligand_dec
        self.n_layers = len(decoder.layers)
        for index, layer in enumerate(decoder.layers):
            self.handles.append(layer.register_forward_pre_hook(self.pre(index), with_kwargs=True))
            self.handles.append(layer.register_forward_hook(self.post(index), with_kwargs=True))
        self.handles.append(decoder.bond_refine.register_forward_hook(self.bond_refine, with_kwargs=True))
        self.handles.append(decoder.bond_proj.register_forward_hook(self.bond_proj, with_kwargs=True))

    def node(self, name, value):
        if self.torch.is_tensor(value):
            if value.shape[0] != 1:
                raise ValueError("Observer expects batch size one.")
            self.arrays[name] = value[0, self.tracked].detach().cpu().float().numpy().copy()

    def pairs(self, name, value):
        if self.torch.is_tensor(value):
            self.arrays[name+"__from_sites"] = value[0, self.sites, :, :].detach().cpu().float().numpy().copy()
            self.arrays[name+"__to_sites"] = value[0].transpose(0, 1)[self.sites, :, :].detach().cpu().float().numpy().copy()

    def pre(self, index):
        def hook(module, args, kwargs):
            prefix = f"layer_{index:02d}__in"
            self.node(prefix+"__equis", args[0])
            self.node(prefix+"__invs", args[1])
            self.node(prefix+"__coords", kwargs.get("coords"))
            self.pairs(prefix+"__edges", args[2])
            return None
        return hook

    def post(self, index):
        def hook(module, args, kwargs, output):
            if not isinstance(output, (tuple, list)) or len(output) != 5:
                raise ValueError("Unexpected SemlaLayer output contract.")
            self.layer_visits[index] = self.layer_visits.get(index, 0)+1
            prefix = f"layer_{index:02d}__out"
            self.node(prefix+"__equis", output[0])
            self.node(prefix+"__invs", output[1])
            self.node(prefix+"__coords", output[2])
            self.pairs(prefix+"__edges", output[3] if output[3] is not None else args[2])
            return None
        return hook

    def bond_refine(self, module, args, kwargs, output):
        self.pairs("bond_refine__out", output)
        return None

    def bond_proj(self, module, args, kwargs, output):
        self.pairs("bond_proj__in", args[0])
        self.pairs("bond_proj__out", output)
        return None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def forward_once(model, call, pocket, encoding, device, torch, np):
    # Fresh copies matter: earlier computations may mutate a conditional tensor.
    batch = move_copy(call["state_before_forward"], device, torch)
    condition = move_copy(call["self_condition"], device, torch)
    times = move_copy(call["times"], device, torch)
    restore_rng(call["rng_before_forward"], device, torch, np)
    with torch.no_grad():
        return model(batch, pocket, times, cond_batch=condition,
                     pocket_equis=encoding["pocket_equis"], pocket_invs=encoding["pocket_invs"],
                     training=False)


def tracked_indices(task, final, sites):
    fixed = set(task["fixed_sampler_indices"])
    valid = final["mask"][0].bool().tolist()
    bonds = final["bonds"][0].argmax(-1).tolist()
    partners = {j for i in sites for j, ok in enumerate(valid)
                if ok and j not in fixed and bonds[max(i, j)][min(i, j)] != 0}
    return sorted(fixed | partners)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="ptp1b")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 47, 49])
    parser.add_argument("--calls", type=int, nargs="+", default=DEFAULT_CALLS)
    parser.add_argument("--sites", type=int, nargs="+", default=[0, 5])
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/sh_hidden_ptp1b"))
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)):
        parser.error("Tolerances must be finite and nonnegative.")
    root = Path.cwd()
    if not (root / "flowr/scriptutil.py").is_file():
        parser.error("Run from /home/ljl/projects/flowr_root with the existing .venv active.")
    output = args.save_dir.resolve()
    if output.exists():
        parser.error(f"Output already exists; keep it and choose a new --save-dir: {output}")
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from flowr.scriptutil import load_model
    from flowr.util.device import resolve_device
    seeds, calls, sites = list(dict.fromkeys(args.seeds)), sorted(set(args.calls)), sorted(set(args.sites))
    if any(i < 0 for i in calls+sites):
        parser.error("Call and atom indices must be nonnegative.")
    runs = args.output_root.resolve()
    manifests = {s: read_json(runs / f"sh_record_{args.case}_seed{s}/research_record/manifest.json") for s in seeds}
    base = manifests[seeds[0]]
    for seed, manifest in manifests.items():
        if manifest["args"]["seed"] != seed or condition_key(manifest) != condition_key(base):
            raise ValueError("Recorded conditions differ beyond seed/output; do not pool these replays.")
    for name, expected in {**base["source_sha256"], **HOOK_SOURCE_HASHES}.items():
        if sha256(root / name) != expected:
            raise ValueError(f"Source differs from the inspected recording version: {name}")
    checkpoint = Path(base["args"]["ckpt_path"])
    if sha256(checkpoint) != base["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the recorded run.")
    if str(torch.__version__) != base["torch_version"]:
        raise ValueError("PyTorch version changed; use the same environment as recording.")
    output.mkdir(parents=True)
    summary = {"mode": "verified_single_forward_replay_with_observation_hooks",
               "replay_version": "2026-10-15.2_native_parameter_flags",
               "seeds": seeds, "calls": calls, "focus_fixed_sampler_indices": sites,
               "call_results": [], "error": None, "all_requested_calls_verified": False,
               "scope": "Hidden observations only. No intervention, causal attribution, or full-trajectory replay has been performed.",
               "design_note": "Calls bracket observed changes in this pilot. These cases are exploratory; use independent cases for later confirmation.",
               "source_sha256": {p: sha256(root/p) for p in HOOK_SOURCE_HASHES},
               "checkpoint_sha256": base["checkpoint_sha256"], "script_sha256": sha256(Path(__file__))}
    accepted_files = []
    load = lambda p: torch.load(p, map_location="cpu", weights_only=True)
    observer = None
    try:
        torch.set_float32_matmul_precision("high")
        model_args = argparse.Namespace(**base["args"])
        model_args.save_dir = str(output / "model_context")
        device = resolve_device(model_args)
        print(f"Loading the recorded model for replay on {device}...", flush=True)
        model = load_model(model_args)[0].to(device).eval()
        # Preserve the flags used by the original inference pipeline. The user's
        # first-forward diagnostic matched exactly with native flags, whereas
        # requires_grad_(False) reproduced the earlier numerical mismatch.
        # forward_once already uses torch.no_grad(); no optimizer or backward
        # call exists here, so retaining these flags does not train the model.
        summary["parameter_tensors_requiring_grad"] = sum(p.requires_grad for p in model.parameters())
        summary["device"] = str(device)
        summary["n_ligand_layers"] = len(model.gen.ligand_dec.layers)
        for seed in seeds:
            folder = runs / f"sh_record_{args.case}_seed{seed}/research_record/attempt_000"
            task = read_json(folder / "task_definition.json")
            if not task.get("mapping_verified") or not set(sites).issubset(task["fixed_sampler_indices"]):
                raise ValueError(f"Focus atoms are not verified fixed atoms for seed {seed}.")
            inputs = load(folder / "inputs.pt")
            if bool(model.self_condition) != inputs["self_condition"] or int(model.sc_recycle_steps) != inputs["sc_recycle_steps"]:
                raise ValueError("Loaded model conditioning settings differ from the saved run.")
            pocket = move_copy(inputs["pocket_data"], device, torch)
            encoding = move_copy(load(folder / "pocket_encoding.pt"), device, torch)
            if encoding["pocket_equis"] is None or encoding["pocket_invs"] is None:
                raise ValueError("Missing recorded pocket encoding; replay needs the original conditioning.")
            tracked = tracked_indices(task, load(folder / "raw_final_prediction.pt"), sites)
            seed_output = output / f"seed_{seed}"
            seed_output.mkdir()
            write_json(seed_output / "metadata.json", {
                "seed": seed, "tracked_atom_indices": tracked, "focus_fixed_sampler_indices": sites,
                "task_definition": task, "recorded_manifest": manifests[seed], "coord_scale": inputs["coord_scale"],
                "array_note": "Node arrays select tracked_atom_indices. Pair arrays have axes (focus-site-list index, all sampler slots, feature); from/to retain both directions. Equis are internal equivariant features, not physical atom positions. Internal coordinate arrays use model units.",
                "matching_note": "Fixed atom identities are mapped by the task. Generated slots must not be matched by equal index across seeds.",
                "omitted": "Pocket hidden states, full ligand-pocket pair tensors and untracked node features are omitted from the compact observer output."})
            accepted_files.append(seed_output / "metadata.json")
            for index in calls:
                print(f"Replay seed {seed}, call {index}...", flush=True)
                call = load(folder / "forward_calls" / f"call_{index:04d}.pt")
                if call["call_index"] != index:
                    raise ValueError("Saved call index does not match the requested filename.")
                original = call["network_output_before_postprocessing"]
                baseline = forward_once(model, call, pocket, encoding, device, torch, np)
                first = compare_predictions(baseline, original, torch, np, args.atol, args.rtol)
                result = {"seed": seed, "call_index": index, "time": call["times_scalar"],
                          "replay_vs_original": first, "observation_vs_replay": None, "accepted": False}
                summary["call_results"].append(result)
                if not first["passed"]:
                    raise RuntimeError(f"Replay did not match original at seed {seed}, call {index}; hidden analysis is stopped.")
                observer = HiddenObserver(model, tracked, sites, torch)
                try:
                    observed = forward_once(model, call, pocket, encoding, device, torch, np)
                finally:
                    observer.close()
                second = compare_predictions(observed, baseline, torch, np, args.atol, args.rtol)
                third = compare_predictions(observed, original, torch, np, args.atol, args.rtol)
                result["observation_vs_replay"], result["observation_vs_original"] = second, third
                if not second["passed"] or not third["passed"] or observer.layer_visits != {k: 1 for k in range(observer.n_layers)}:
                    raise RuntimeError("Observed replay or layer-call counts did not match; hidden analysis is stopped.")
                result["layer_visits"] = observer.layer_visits
                result["accepted"] = True
                arrays = observer.arrays
                arrays["tracked_atom_indices"] = np.asarray(tracked)
                arrays["focus_fixed_sampler_indices"] = np.asarray(sites)
                arrays["input_coords"] = call["state_before_forward"]["coords"][0].cpu().float().numpy().copy()
                arrays["input_atom_classes"] = call["state_before_forward"]["atomics"][0].argmax(-1).cpu().numpy().copy()
                arrays["input_bond_classes"] = call["state_before_forward"]["bonds"][0].argmax(-1).cpu().numpy().copy()
                arrays["output_bond_logits"] = observed["bonds"][0].detach().cpu().float().numpy().copy()
                arrays["output_atom_logits"] = observed["atomics"][0].detach().cpu().float().numpy().copy()
                path = seed_output / f"call_{index:04d}_hidden.npz"
                np.savez_compressed(path, **arrays)
                accepted_files.append(path)
                observer = None
                del observed, baseline, call, arrays
                print("  MATCH: replay and observation verified; hidden features saved.", flush=True)
        summary["all_requested_calls_verified"] = len(summary["call_results"]) == len(seeds)*len(calls)
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        (output / "replay_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"CHECK: {summary['error']}", flush=True)
    finally:
        if observer is not None:
            observer.close()
        write_json(output / "replay_summary.json", summary)
        bundle = output / "hidden_review.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for path in [output / "replay_summary.json", *accepted_files]:
                z.write(path, path.relative_to(output))
            if (output / "replay_error.txt").is_file():
                z.write(output / "replay_error.txt", "replay_error.txt")
        print(f"\nReview file: {bundle}", flush=True)
        print(f"All requested calls verified: {summary['all_requested_calls_verified']}", flush=True)
    return 0 if summary["all_requested_calls_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
