#!/usr/bin/env python3
"""Diagnose the first FLOWR scaffold-hopping forward replay, without training.

Run alongside replay_connection_hidden.py in the existing FLOWR repository.
Executes the native pipeline only up to its first network prediction, then a
few controlled forward-only comparisons. Does not collect research activations,
change weights, relax the original comparison threshold, or overwrite records.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
import zipfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path


class FirstForwardCaptured(BaseException):
    """Intentional exit after the native pipeline's first forward pass."""


def snapshot(value, torch):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {str(k): snapshot(v, torch) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(v, torch) for v in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {"omitted_python_object_type": type(value).__qualname__}


def layouts(value, torch, helpers):
    return {k: {"shape": list(v.shape), "stride": list(v.stride()),
                "dtype": str(v.dtype), "device": str(v.device),
                "storage_offset": v.storage_offset(), "contiguous": v.is_contiguous(),
                "requires_grad": v.requires_grad}
            for k, v in helpers.tensor_leaves(value, torch).items()}


def compare_values(actual, expected, torch, np, helpers):
    a, b = helpers.tensor_leaves(actual, torch), helpers.tensor_leaves(expected, torch)
    fields = {}
    passed = set(a) == set(b)
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b or a[name].shape != b[name].shape:
            fields[name] = {"exact": False, "reason": "missing tensor or shape mismatch"}
            passed = False
            continue
        x, y = a[name].detach().cpu(), b[name].detach().cpu()
        equal = x.dtype == y.dtype and bool(torch.equal(x, y))
        xa, ya = x.double().numpy(), y.double().numpy()
        finite = bool(np.isfinite(xa).all() and np.isfinite(ya).all())
        delta = float(np.max(np.abs(xa-ya))) if xa.size and finite else (0.0 if not xa.size else None)
        fields[name] = {"exact": equal, "max_abs_difference": delta,
                        "actual_dtype": str(x.dtype), "recorded_dtype": str(y.dtype)}
        passed = passed and equal
    return {"all_tensors_exact": bool(passed), "fields": fields}


def copy_native_layout(value, template, torch):
    """Copy saved values into the native tensor strides, including broadcast axes.

    Diagnostic only: value equality is checked before execution. Layout templates
    come from the newly observed native first call, not a guessed shape/layout.
    """
    if torch.is_tensor(value):
        if not torch.is_tensor(template) or value.shape != template.shape or value.dtype != template.dtype:
            raise ValueError("Native layout template differs in shape or dtype.")
        shape, stride = list(template.shape), list(template.stride())
        base_shape = [1 if s == 0 else n for n, s in zip(shape, stride)]
        index = tuple(slice(0, 1) if s == 0 else slice(None) for s in stride)
        source = value.to(template.device)
        offset = template.storage_offset()
        storage_size = offset + 1 + sum((n-1)*s for n, s in zip(base_shape, stride))
        storage = torch.empty(max(0, storage_size), dtype=template.dtype, device=template.device)
        base = storage.as_strided(base_shape, stride, storage_offset=offset)
        base.copy_(source[index])
        restored = base.expand(shape)
        if not torch.equal(restored, source):
            raise ValueError("Native broadcast layout would alter saved values.")
        return restored
    if isinstance(value, Mapping):
        if not isinstance(template, Mapping):
            if set(value) == {"omitted_python_object_type"}:
                return value  # Non-tensor complex objects are not used by forward.
            raise ValueError("Native layout template differs in mapping type.")
        return {k: copy_native_layout(v, template[k], torch) if k in template else v for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        if len(value) != len(template):
            raise ValueError("Native layout template differs in sequence length.")
        return type(value)(copy_native_layout(v, t, torch) for v, t in zip(value, template))
    return value


def runtime(torch, model):
    device = next(model.parameters()).device
    return {"torch_version": str(torch.__version__), "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_version": torch.version.cuda,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "grad_enabled": torch.is_grad_enabled(), "model_training": model.training,
            "parameter_tensors_requiring_grad": sum(p.requires_grad for p in model.parameters()),
            "duplicate_pocket_equi": bool(model.gen.duplicate_pocket_equi),
            "model_flags": {k: getattr(model, k, None) for k in [
                "self_condition", "self_condition_mode", "sc_recycle_steps", "sc_charges",
                "inpainting_mode", "graph_inpainting", "use_is_fixed_embed", "add_feats"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="ptp1b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    if not (root / "flowr/gen/generate_from_pdb.py").is_file():
        parser.error("Run from /home/ljl/projects/flowr_root with the existing .venv active.")
    if not (root / "replay_connection_hidden.py").is_file():
        parser.error("Keep the previously downloaded replay_connection_hidden.py in the same project folder.")
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    import replay_connection_hidden as helpers
    import flowr.gen.generate_from_pdb as entry

    out = (args.save_dir or Path("outputs") / (
        f"sh_replay_diagnostic_{args.case}_seed{args.seed}_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if out.exists():
        parser.error(f"Output exists; choose a new --save-dir: {out}")
    out.mkdir(parents=True)
    report = {"purpose": "First-forward replay diagnostics only; no hidden activations accepted for research.",
              "seed": args.seed, "case": args.case, "error": None,
              "diagnostic_completed": False, "variants": {},
              "script_sha256": helpers.sha256(Path(__file__)),
              "replay_helper_sha256": helpers.sha256(root / "replay_connection_hidden.py")}
    handles, captured, predictions = [], {}, {}
    old_load = entry.load_model
    try:
        record = root / f"outputs/sh_record_{args.case}_seed{args.seed}/research_record"
        manifest = helpers.read_json(record / "manifest.json")
        for name, expected in {**manifest["source_sha256"], **helpers.HOOK_SOURCE_HASHES}.items():
            if helpers.sha256(root / name) != expected:
                raise ValueError(f"Source changed since recording: {name}")
        if str(torch.__version__) != manifest["torch_version"]:
            raise ValueError("PyTorch version changed since recording.")
        if helpers.sha256(Path(manifest["args"]["ckpt_path"])) != manifest["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed since recording.")
        for path, digest in manifest["input_sha256"].items():
            if helpers.sha256(Path(path)) != digest:
                raise ValueError(f"Native input file changed since recording: {path}")
        load = lambda p: torch.load(p, map_location="cpu", weights_only=True)
        folder = record / "attempt_000"
        call = load(folder / "forward_calls/call_0000.pt")
        inputs = load(folder / "inputs.pt")
        encoding = load(folder / "pocket_encoding.pt")
        recorded_inputs = {"batch": call["state_before_forward"], "times": call["times"],
                           "condition": call["self_condition"], "pocket": inputs["pocket_data"],
                           "encoding": encoding}
        original = call["network_output_before_postprocessing"]
        report["recorded_input_layouts"] = layouts(recorded_inputs, torch, helpers)
        report["checkpoint_sha256"] = manifest["checkpoint_sha256"]

        def before(module, positional, keyword):
            values = {"batch": positional[0], "pocket": positional[1], "times": positional[2],
                      "condition": keyword.get("cond_batch"),
                      "encoding": {k: keyword.get(k) for k in ["pocket_equis", "pocket_invs"]}}
            captured["live_inputs"] = values
            captured["cpu_inputs"] = snapshot(values, torch)
            report["native_input_layouts"] = layouts(values, torch, helpers)
            report["native_runtime"] = runtime(torch, module)
            report["native_rng_matches_recorded"] = {
                "torch_cpu": bool(torch.equal(torch.get_rng_state(), call["rng_before_forward"]["torch_cpu"])),
                "torch_cuda": bool(torch.equal(torch.cuda.get_rng_state().cpu(), call["rng_before_forward"]["torch_cuda"]))
                    if "torch_cuda" in call["rng_before_forward"] else None}
            return None

        def after(module, positional, keyword, result):
            captured["native_output"] = snapshot(result, torch)
            raise FirstForwardCaptured()

        def loading(*a, **kw):
            values = old_load(*a, **kw)
            captured["model"] = values[0]
            handles.append(values[0].register_forward_pre_hook(before, with_kwargs=True))
            handles.append(values[0].register_forward_hook(after, with_kwargs=True))
            return values

        native_args = argparse.Namespace(**manifest["args"])
        native_args.save_dir = str(out / "native_context")
        Path(native_args.save_dir).mkdir()
        entry.load_model = loading
        print("DIAGNOSTIC: run the native pipeline only to its first network prediction...", flush=True)
        try:
            entry.evaluate(native_args)
        except FirstForwardCaptured:
            pass
        finally:
            entry.load_model = old_load
            for handle in handles:
                handle.remove()
            handles.clear()
        if "native_output" not in captured:
            raise RuntimeError("Native pipeline did not reach its first forward pass.")
        model = captured["model"]
        device = next(model.parameters()).device
        native = captured["native_output"]
        report["native_inputs_vs_recorded"] = compare_values(captured["cpu_inputs"], recorded_inputs, torch, np, helpers)
        report["native_forward_vs_recorded"] = helpers.compare_predictions(native, original, torch, np, 1e-5, 1e-5)
        print(f"Native first prediction matches recording: {report['native_forward_vs_recorded']['passed']}", flush=True)

        def probe(name, values):
            print(f"DIAGNOSTIC: {name}", flush=True)
            try:
                helpers.restore_rng(call["rng_before_forward"], device, torch, np)
                with torch.no_grad():
                    output = model(values["batch"], values["pocket"], values["times"],
                                   cond_batch=values["condition"], training=False, **values["encoding"])
                copied = snapshot(output, torch)
                predictions[name] = copied
                report["variants"][name] = {
                    "runtime": runtime(torch, model),
                    "input_values_vs_recorded": compare_values(values, recorded_inputs, torch, np, helpers),
                    "input_layouts": layouts(values, torch, helpers),
                    "vs_recorded": helpers.compare_predictions(copied, original, torch, np, 1e-5, 1e-5),
                    "vs_native": helpers.compare_predictions(copied, native, torch, np, 1e-5, 1e-5)}
                print(f"  matches recording: {report['variants'][name]['vs_recorded']['passed']}", flush=True)
            except Exception as exc:
                report["variants"][name] = {"error": f"{type(exc).__name__}: {exc}",
                                          "traceback": traceback.format_exc()}

        clone = lambda: helpers.move_copy(recorded_inputs, device, torch)
        probe("saved_values_native_grad_flags", clone())
        probe("saved_values_repeat", clone())
        if all(k in predictions for k in ["saved_values_native_grad_flags", "saved_values_repeat"]):
            report["repeatability"] = helpers.compare_predictions(
                predictions["saved_values_repeat"], predictions["saved_values_native_grad_flags"], torch, np, 1e-5, 1e-5)
        # Reproduce the extra parameter flag change made by the previous replay script.
        # no_grad is used in every run; neither flag state performs training.
        grad_flags = [(p, p.requires_grad) for p in model.parameters()]
        try:
            model.requires_grad_(False)
            probe("saved_values_previous_script_grad_flags", clone())
        finally:
            for parameter, flag in grad_flags:
                parameter.requires_grad_(flag)
        # Source constructs some tensors as views. Preserve saved values while
        # borrowing the actually observed native strides; do not guess strides.
        try:
            restored = copy_native_layout(recorded_inputs, captured["live_inputs"], torch)
            before_compare = compare_values(restored, recorded_inputs, torch, np, helpers)
            if not before_compare["all_tensors_exact"]:
                raise ValueError("Layout diagnostic altered input values.")
            probe("saved_values_native_layout", restored)
            del restored
        except Exception as exc:
            report["variants"]["saved_values_native_layout"] = {"error": f"{type(exc).__name__}: {exc}",
                                                                    "traceback": traceback.format_exc()}
        report["diagnostic_completed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        (out / "diagnostic_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"DIAGNOSTIC ERROR: {report['error']}", flush=True)
    finally:
        entry.load_model = old_load
        for handle in handles:
            handle.remove()
        helpers.write_json(out / "diagnostic_summary.json", report)
        arrays = {}
        for name, value in {"native": captured.get("native_output"), **predictions}.items():
            for key, tensor in helpers.tensor_leaves(value, torch).items():
                arrays[name+"__"+key] = tensor.detach().cpu().float().numpy()
        if arrays:
            np.savez_compressed(out / "diagnostic_predictions.npz", **arrays)
        bundle = out / "replay_diagnostic.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name in ["diagnostic_summary.json", "diagnostic_error.txt", "diagnostic_predictions.npz"]:
                if (out / name).is_file():
                    z.write(out / name, name)
        print(f"\nDiagnostic completed: {report['diagnostic_completed']}", flush=True)
        print(f"Upload this file: {bundle}", flush=True)
        print("This is a diagnostic report, not accepted hidden-layer research data.", flush=True)
    return 0 if report["diagnostic_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
