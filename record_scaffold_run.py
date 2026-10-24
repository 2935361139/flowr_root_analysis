#!/usr/bin/env python3
"""FLOWR.ROOT single-attempt research recorder, for the source uploaded 2026-10-15.

Run from the FLOWR.ROOT repository with its existing Python environment.
This is an observer: no weights, sampler rules, repair rules or source files are edited.
It records actual forward calls and sampler states, including unsuccessful outputs.
It does NOT yet record hidden layers or provide a complete scaffold-hopping success label.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import random
import subprocess
import sys
import traceback
import zipfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import MethodType


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def snapshot(value):
    """Clone now: inpainting later mutates some tensors in place."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(snapshot(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"omitted_python_object_type": type(value).__qualname__}


def rng_snapshot():
    result = {"python": random.getstate(), "numpy": snapshot(np.random.get_state()),
              "torch_cpu": torch.get_rng_state().clone()}
    if torch.cuda.is_initialized():
        result["cuda_device"] = torch.cuda.current_device()
        result["torch_cuda"] = torch.cuda.get_rng_state().cpu().clone()
    return result


def scalar_times(times):
    return [float(v.detach().reshape(-1)[0].cpu()) for v in times]


def mol_status(mol):
    from rdkit import Chem
    if mol is None:
        return {"built": False, "sanitizable": False, "connected": False,
                "n_atoms": None, "n_components": None}
    copy = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(copy)
        sanitized = True
    except Exception:
        sanitized = False
    count = len(Chem.GetMolFrags(copy))
    return {"built": True, "sanitizable": sanitized,
            "connected": bool(copy.GetNumAtoms() > 0 and count == 1),
            "n_atoms": copy.GetNumAtoms(), "n_components": count}


def save_molecules(path, mols):
    from rdkit import Chem
    with Chem.SDWriter(str(path)) as writer:
        for index, mol in enumerate(mols):
            if mol is not None:
                copy = Chem.Mol(mol)
                copy.SetIntProp("recorded_sample_index", index)
                writer.write(copy)
    # Missing molecules remain represented in decoded_status.json, not by a fake SDF entry.


def indexed_graph(mol):
    return (
        [(a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsAromatic(),
          int(a.GetChiralTag())) for a in mol.GetAtoms()],
        sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                str(b.GetBondType()), int(b.GetStereo())) for b in mol.GetBonds()),
    )


def compare_sdf(old_path, new_path):
    from rdkit import Chem
    if not Path(old_path).is_file() or not Path(new_path).is_file():
        return {"checked": False, "reason": "baseline or new SDF does not exist"}
    old = list(Chem.SDMolSupplier(str(old_path), removeHs=False))
    new = list(Chem.SDMolSupplier(str(new_path), removeHs=False))
    result = {"checked": True, "baseline_count": len(old), "new_count": len(new)}
    equal = len(old) == len(new) and len(old) > 0 and all(
        a is not None and b is not None and indexed_graph(a) == indexed_graph(b)
        for a, b in zip(old, new)
    )
    result["same_indexed_graphs"] = equal
    if equal:
        delta = max(float(np.max(np.abs(a.GetConformer().GetPositions() -
                                       b.GetConformer().GetPositions())))
                    for a, b in zip(old, new))
        result["max_coordinate_difference_A"] = delta
        result["matches_at_sdf_precision"] = delta <= 1.1e-4
    else:
        result["matches_at_sdf_precision"] = False
    result["scope"] = "Final SDF comparison only; not proof of identical hidden trajectories."
    return result


class Recorder:
    def __init__(self, output):
        self.root = Path(output) / "research_record"
        self.root.mkdir(parents=True, exist_ok=False)
        self.model = None
        self.active = None
        self.attempts = []
        self.pending = None
        self.restores = []
        self.hooks = []

    def replace_method(self, obj, name, replacement):
        original = getattr(obj, name)
        had_local = name in vars(obj)
        local_value = vars(obj).get(name)
        setattr(obj, name, MethodType(replacement, obj))
        self.restores.append((obj, name, had_local, local_value))
        return original

    def attach(self, model):
        self.model = model
        json_write(self.root / "modules.json", [
            {"name": name, "class": type(module).__qualname__}
            for name, module in model.gen.named_modules()
        ])
        old_generate = model._generate
        old_build = model._build_trajectory
        signature = inspect.signature(old_generate)

        def generate(_model, *args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            folder = self.root / f"attempt_{len(self.attempts):03d}"
            folder.mkdir()
            (folder / "forward_calls").mkdir()
            (folder / "sampler_steps").mkdir()
            record = {"folder": folder, "n_forward_calls": 0, "n_frames": 0,
                      "steps": int(arguments["steps"]), "completed": False,
                      "prior": snapshot(arguments["prior"]),
                      "pocket": arguments["pocket_data"], "output": None,
                      "decoded": None}
            self.attempts.append(record)
            self.active = record
            pocket = arguments["pocket_data"]
            torch.save({
                "prior": record["prior"], "pocket_data": snapshot(pocket),
                "pocket_com": snapshot([s.com for s in pocket.get("complex", [])]),
                "times": snapshot(arguments["times"]), "rng": rng_snapshot(),
                "coord_scale": float(model.coord_scale),
                "coord_frame": "Internal centered/scaled model frame. Multiply by coord_scale, then add pocket_com to obtain original coordinates.",
                "self_condition": bool(model.self_condition),
                "sc_recycle_steps": int(model.sc_recycle_steps),
                "final_inpaint": bool(arguments["final_inpaint"]),
                "solver": arguments["solver"], "strategy": arguments["strategy"],
                "note": "Snapshots are analysis data; this version does not implement a mid-trajectory replay driver.",
            }, folder / "inputs.pt")
            try:
                output = old_generate(*args, **kwargs)
                record["output"] = output
                record["completed"] = True
                torch.save(snapshot(output), folder / "raw_final_prediction.pt")
                return output
            except BaseException:
                (folder / "generation_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                if self.pending is not None:
                    torch.save(self.pending, folder / "unfinished_forward_input.pt")
                raise
            finally:
                self.active = None
                self.pending = None
                self.write_attempt_summary(record)

        def build(_model, curr, predicted, pocket_data, iter=0, step=0):
            record = self.active
            if record is not None:
                kind = ("integrated_state_after_inpainting" if step < record["steps"]
                        else "final_corrector_prediction")
                torch.save({"step_index": int(step), "stage": kind,
                            "current": snapshot(curr),
                            "prediction_argument": snapshot(predicted),
                            "coord_frame": "internal centered/scaled model frame",
                            "note": "For integration steps, current is after the update and inpainting; prediction_argument came from the preceding network prediction and may have been touched by self-conditioning inpainting. Use forward_calls for unmodified network outputs."},
                           record["folder"] / "sampler_steps" / f"step_{int(step):04d}.pt")
                record["n_frames"] += 1
            return old_build(curr, predicted, pocket_data, iter=iter, step=step)

        self.replace_method(model, "_generate", generate)
        self.replace_method(model, "_build_trajectory", build)
        self.hooks.append(model.register_forward_pre_hook(self.before_forward, with_kwargs=True))
        self.hooks.append(model.register_forward_hook(self.after_forward, with_kwargs=True))

    def before_forward(self, module, args, kwargs):
        if self.active is None:
            return None
        batch = args[0] if args else kwargs["batch"]
        times = args[2] if len(args) > 2 else kwargs["t"]
        self.pending = {"call_index": self.active["n_forward_calls"],
                        "times": snapshot(times), "times_scalar": scalar_times(times),
                        "state_before_forward": snapshot(batch),
                        "self_condition": snapshot(kwargs.get("cond_batch")),
                        "rng_before_forward": rng_snapshot()}
        if self.active["n_forward_calls"] == 0:
            torch.save({"pocket_equis": snapshot(kwargs.get("pocket_equis")),
                        "pocket_invs": snapshot(kwargs.get("pocket_invs"))},
                       self.active["folder"] / "pocket_encoding.pt")
        return None

    def after_forward(self, module, args, kwargs, output):
        if self.active is None or self.pending is None:
            return None
        self.pending["network_output_before_postprocessing"] = snapshot(output)
        self.pending["note"] = "Network output includes logits and coordinates, not hidden layer activations. Call index is not necessarily a time-step index because self-conditioning can repeat calls."
        index = self.active["n_forward_calls"]
        torch.save(self.pending, self.active["folder"] / "forward_calls" / f"call_{index:04d}.pt")
        self.active["n_forward_calls"] += 1
        self.pending = None
        return None  # A hook must not replace the model's output.

    def capture_target_result(self, posterior, mols):
        from rdkit import Chem
        if not self.attempts:
            raise RuntimeError("No generation attempt was recorded.")
        record = self.attempts[-1]
        record["posterior"] = posterior
        record["decoded"] = [Chem.Mol(m) if m is not None else None for m in mols]
        torch.save(snapshot(posterior), record["folder"] / "reference_batch.pt")
        json_write(record["folder"] / "decoded_status.json", [mol_status(m) for m in mols])
        save_molecules(record["folder"] / "decoded_before_filter.sdf", mols)

    def write_attempt_summary(self, record):
        json_write(record["folder"] / "attempt.json", {
            "generation_completed": record["completed"],
            "forward_calls": record["n_forward_calls"],
            "saved_sampler_frames": record["n_frames"],
            "integration_steps_requested": record["steps"],
        })

    def restore(self):
        for hook in self.hooks:
            hook.remove()
        for obj, name, had_local, value in reversed(self.restores):
            if had_local:
                setattr(obj, name, value)
            else:
                delattr(obj, name)

    def audit(self, record):
        """Run diagnostics only after inference, so extra decodes cannot influence it."""
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
        from flowr.data.interpolate import extract_scaffolds

        model, folder = self.model, record["folder"]
        result = {"generation_completed": record["completed"],
                  "forward_calls": record["n_forward_calls"],
                  "saved_sampler_frames": record["n_frames"],
                  "full_scaffold_hopping_success": "NOT_YET_ASSESSED"}
        if record["output"] is None:
            return result
        # Raw decode is a separate, post-inference diagnostic. Default decode above is preserved.
        before_stats = snapshot(getattr(model.builder, "repair_stats", {}))
        raw_mols = model._generate_mols(record["output"], valence_repair=False)
        result["decode_without_valence_repair"] = [mol_status(m) for m in raw_mols]
        result["decode_with_run_setting"] = [mol_status(m) for m in (record["decoded"] or [])]
        result["repair_stats_before_diagnostic_decode"] = before_stats
        save_molecules(folder / "decoded_without_repair.sdf", raw_mols)
        if "posterior" not in record:
            result["task_definition"] = "Unavailable: target-level call did not return."
            return result
        posterior = record["posterior"]
        reference = model._generate_ligs(posterior, lig_mask=posterior["lig_mask"].bool(),
                                         scale=model.coord_scale, sanitise=True)[0]
        if reference is None:
            raise RuntimeError("Could not reconstruct the preprocessed reference ligand for mapping.")
        save_molecules(folder / "reference_preprocessed.sdf", [reference])
        fixed_ref = extract_scaffolds([reference], invert_mask=True)[0].bool()
        ref_ids = fixed_ref.nonzero(as_tuple=True)[0].tolist()
        prior = record["prior"]
        fixed_sampler = prior["fragment_mask"][0].bool() & prior["mask"][0].bool()
        sampler_ids = fixed_sampler.nonzero(as_tuple=True)[0].tolist()
        coords = model.builder.undo_zero_com_batch(
            prior["coords"] * model.coord_scale, prior["mask"],
            [s.com for s in record["pocket"]["complex"]])
        same_count = len(ref_ids) == len(sampler_ids) and len(ref_ids) > 0
        delta = None
        atoms_match = False
        if same_count:
            delta = float(np.max(np.abs(coords[0, sampler_ids].cpu().numpy() -
                                        reference.GetConformer().GetPositions()[ref_ids])))
            tokens = model.builder.vocab.tokens_from_indices(
                prior["atomics"][0, sampler_ids].argmax(-1).tolist())
            atoms_match = list(tokens) == [reference.GetAtomWithIdx(i).GetSymbol() for i in ref_ids]
        mapping_verified = bool(same_count and atoms_match and delta is not None and delta < 1e-3)
        mapping = dict(zip(ref_ids, sampler_ids)) if mapping_verified else {}
        anchors = []
        for bond in reference.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if bool(fixed_ref[i]) != bool(fixed_ref[j]):
                f, v = (i, j) if bool(fixed_ref[i]) else (j, i)
                anchors.append({"fixed_reference_index": f,
                                "fixed_sampler_index": mapping.get(f),
                                "removed_scaffold_reference_index": v,
                                "reference_bond_type": str(bond.GetBondType())})
        task = {
            "mode": "scaffold_hopping", "fixed_definition": "Complement of RDKit Murcko scaffold, as used by this FLOWR version.",
            "reference_index_space": "Preprocessed reference, not necessarily original input SDF indices.",
            "fixed_reference_indices": ref_ids, "fixed_sampler_indices": sampler_ids,
            "n_reference_atoms": reference.GetNumAtoms(),
            "n_fixed_atoms": len(sampler_ids),
            "n_generated_slots": int(prior["mask"][0].sum().item()) - len(sampler_ids),
            "mapping_verified": mapping_verified,
            "fixed_coordinate_mapping_max_abs_error_A": delta,
            "fixed_element_mapping_matches": atoms_match,
            "reference_attachment_sites": anchors,
            "attachment_note": "Reference cut-bond endpoints; these are not explicitly hard-constrained attachment labels in ROOT. Generated core atoms have no assumed one-to-one identity with the removed core.",
        }
        json_write(folder / "task_definition.json", task)
        with (folder / "atom_map.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["reference_index", "element", "role", "sampler_index_if_verified"])
            for atom in reference.GetAtoms():
                i = atom.GetIdx()
                writer.writerow([i, atom.GetSymbol(), "fixed" if bool(fixed_ref[i]) else "replace_scaffold", mapping.get(i, "")])
        drawing = Chem.Mol(reference)
        rdDepictor.Compute2DCoords(drawing)
        drawer = rdMolDraw2D.MolDraw2DCairo(1100, 760)
        drawer.drawOptions().addAtomIndices = True
        colors = {i: ((0.35, 0.7, 1.0) if bool(fixed_ref[i]) else (1.0, 0.65, 0.4))
                  for i in range(reference.GetNumAtoms())}
        drawer.DrawMolecule(drawing, legend="Blue: fixed fragments | Orange: scaffold to replace | Numbers: preprocessed reference indices",
                            highlightAtoms=list(colors), highlightAtomColors=colors)
        drawer.FinishDrawing()
        (folder / "fixed_fragments.png").write_bytes(drawer.GetDrawingText())
        result["task_definition"] = task
        if sampler_ids:
            final = record["output"]
            displacement = final["coords"][0, sampler_ids].cpu() - coords[0, sampler_ids].cpu()
            result["raw_final_fixed_displacement_rms_A"] = float(
                displacement.square().sum(-1).mean().sqrt())
            result["raw_final_fixed_displacement_max_A"] = float(
                displacement.square().sum(-1).sqrt().max())
            result["raw_final_fixed_element_changes"] = int((
                final["atomics"][0, sampler_ids].argmax(-1).cpu() !=
                prior["atomics"][0, sampler_ids].argmax(-1).cpu()).sum())
        return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["ptp1b", "bace", "cdk2", "tyk2"], default="ptp1b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ligand_idx", type=int, default=0)
    parser.add_argument("--ckpt_path", default="checkpoints/flowr_root_v2.2.ckpt")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--baseline", help="Existing unoptimized SDF from the same case, seed and settings.")
    options = parser.parse_args()
    root = Path.cwd()
    if not (root / "flowr/gen/generate_from_pdb.py").is_file():
        parser.error("Run from ~/projects/flowr_root with the existing .venv activated.")
    sys.path.insert(0, str(root))
    pdb = root / f"examples/{options.case}_protein.pdb"
    ligand = root / f"examples/{options.case}_ligands.sdf"
    checkpoint = Path(options.ckpt_path).resolve()
    for path in [pdb, ligand, checkpoint]:
        if not path.is_file():
            parser.error(f"File does not exist: {path}")
    output = Path(options.save_dir or (
        f"outputs/sh_record_{options.case}_seed{options.seed}_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if output.exists():
        parser.error(f"Output already exists; choose a new --save_dir: {output}")

    global torch, np
    import torch
    import numpy as np
    from rdkit import rdBase
    import flowr.gen.generate_from_pdb as entry

    argv = ["generate_from_pdb", "--pdb_file", str(pdb), "--ligand_file", str(ligand),
            "--ligand_idx", str(options.ligand_idx), "--arch", "pocket", "--pocket_type", "holo",
            "--cut_pocket", "--pocket_cutoff", "7", "--ckpt_path", str(checkpoint),
            "--save_dir", str(output), "--gpus", "1", "--num_workers", "1", "--batch_cost", "1",
            "--sample_n_molecules_per_target", "1", "--max_sample_iter", "0",
            "--integration_steps", "100", "--seed", str(options.seed), "--scaffold_hopping", "--save_traj"]
    original_argv = sys.argv
    try:
        sys.argv = argv
        args = entry.get_args()
    finally:
        sys.argv = original_argv
    recorder = Recorder(output)
    source_paths = ["flowr/gen/generate_from_pdb.py", "flowr/gen/generate.py",
                    "flowr/models/fm_pocket.py", "flowr/models/mol_builder.py",
                    "flowr/data/interpolate.py", "flowr/scriptutil.py"]
    manifest = {"recorder_version": "2026-10-15.1", "args": vars(args),
                "torch_version": str(torch.__version__), "rdkit_version": rdBase.rdkitVersion,
                "source_sha256": {p: file_hash(root / p) for p in source_paths},
                "input_sha256": {str(p): file_hash(p) for p in [pdb, ligand]},
                "checkpoint_sha256": file_hash(checkpoint),
                "recorder_sha256": file_hash(__file__),
                "hidden_layer_activations_recorded": False,
                "scope": "One attempted generation, default repair and native trajectory behavior preserved; additional observer files retain failures."}
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True)
        manifest["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
    except FileNotFoundError:
        manifest["git_commit"] = None
    json_write(recorder.root / "manifest.json", manifest)
    old_load, old_target = entry.load_model, entry.generate_ligands_per_target

    def load(*a, **kw):
        values = old_load(*a, **kw)
        recorder.attach(values[0])
        return values

    def target(*a, **kw):
        bound = inspect.signature(old_target).bind(*a, **kw)
        mols = old_target(*a, **kw)
        recorder.capture_target_result(bound.arguments["posterior"], mols)
        return mols

    entry.load_model, entry.generate_ligands_per_target = load, target
    error = None
    print(f"[Recorder] output: {output}", flush=True)
    try:
        entry.evaluate(args)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (recorder.root / "pipeline_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[Recorder] Original pipeline reported: {error}", file=sys.stderr)
    finally:
        entry.load_model, entry.generate_ligands_per_target = old_load, old_target
        recorder.restore()

    summary = {"pipeline_error": error, "n_attempts": len(recorder.attempts), "attempts": [],
               "interpretation": "Connectivity and fixed-atom checks are diagnostics. They do not constitute a complete scaffold-hopping success label or a causal explanation."}
    audit_failed = False
    for record in recorder.attempts:
        try:
            summary["attempts"].append(recorder.audit(record))
        except Exception:
            audit_failed = True
            (record["folder"] / "audit_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            summary["attempts"].append({"audit_error": "See audit_error.txt; raw records were retained."})
    if options.baseline:
        summary["baseline_comparison"] = compare_sdf(
            Path(options.baseline), output / f"samples_{pdb.stem}.sdf")
    json_write(recorder.root / "summary.json", summary)
    # Compact review bundle excludes large tensor snapshots and model weights.
    bundle = output / "review_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(recorder.root.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".csv", ".png", ".sdf", ".txt"}:
                z.write(path, path.relative_to(output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[Recorder] Review bundle: {bundle}")
    print("[Recorder] Tensor snapshots remain in research_record/; no source files or weights were changed.")
    return 1 if error or audit_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
