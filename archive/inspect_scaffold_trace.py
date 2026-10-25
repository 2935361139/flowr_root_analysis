#!/usr/bin/env python3
"""Inspect existing record_scaffold_run.py output; no generation or training.

Usage: python inspect_scaffold_trace.py outputs/sh_record_ptp1b_seed42
Requires the existing environment's PyTorch. Matplotlib plots are optional.
Indices are taken from the saved sampler mask, not guessed from an input SDF.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def displacement_metrics(coords, reference, indices, scale):
    """RMS over per-atom Euclidean displacements; no rigid-body realignment."""
    if not indices:
        return {"fixed_rms_A": None, "fixed_max_A": None, "finite": True}
    squared = [sum(((coords[i][k] - reference[i][k]) * scale) ** 2
                   for k in range(3)) for i in indices]
    if not all(math.isfinite(x) for x in squared):
        return {"fixed_rms_A": None, "fixed_max_A": None, "finite": False}
    return {"fixed_rms_A": math.sqrt(sum(squared) / len(squared)),
            "fixed_max_A": math.sqrt(max(squared)), "finite": True}


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_stages(rows, tolerance):
    integration = [r for r in rows if r["stage"] == "integrated_state_after_inpainting"]
    final = [r for r in rows if r["stage"] == "final_corrector_prediction"]
    first = next((r for r in rows if r["fixed_max_A"] is not None
                  and r["fixed_max_A"] > tolerance), None)
    return {
        "first_saved_frame_exceeding_numerical_tolerance": first,
        "last_integrated_state_after_inpainting": integration[-1] if integration else None,
        "final_corrector_prediction": final[-1] if final else None,
        "all_saved_integrated_states_within_numerical_tolerance": (
            all(r["finite"] and r["fixed_max_A"] is not None
                and r["fixed_max_A"] <= tolerance for r in integration)
            if integration else None
        ),
    }


def optional_plot(output, calls, frames):
    # Avoid including an older plot if plotting becomes unavailable on a rerun.
    (output / "fixed_drift.png").unlink(missing_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "Matplotlib unavailable; JSON and CSV analyses are complete. No installation is required."
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for ax, rows, xkey, title in [
        (axes[0], calls, "call_index", "Raw network predictions: fixed-atom displacement"),
        (axes[1], frames, "step_index", "Saved sampler states: fixed-atom displacement"),
    ]:
        points = [(r[xkey], r["fixed_rms_A"]) for r in rows if r["fixed_rms_A"] is not None]
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points], lw=1.6)
        ax.set_title(title)
        ax.set_ylabel("RMS displacement (angstrom)")
        ax.grid(alpha=0.25)
    axes[0].set_xlabel("Network call index (recycling may add calls)")
    axes[1].set_xlabel("Saved frame index (last point may be an extra prediction)")
    finals = [r for r in frames if r["stage"] == "final_corrector_prediction"
              and r["fixed_rms_A"] is not None]
    if finals:
        axes[1].scatter([r["step_index"] for r in finals], [r["fixed_rms_A"] for r in finals],
                        color="darkorange", zorder=4, label="Final corrector prediction")
        axes[1].legend()
    fig.savefig(output / "fixed_drift.png", dpi=160)
    plt.close(fig)
    return "Saved fixed_drift.png."


def analyze_attempt(folder, torch, numerical_tolerance):
    def load(path):
        return torch.load(path, map_location="cpu", weights_only=True)

    inputs = load(folder / "inputs.pt")
    prior = inputs["prior"]
    if prior["coords"].shape[0] != 1:
        raise ValueError("This diagnostic expects one molecule per recorded attempt.")
    valid = prior["mask"][0].bool()
    fixed = prior["fragment_mask"][0].bool() & valid
    ids = fixed.nonzero(as_tuple=True)[0].tolist()
    variable_ids = (valid & ~fixed).nonzero(as_tuple=True)[0].tolist()
    if not ids or not variable_ids:
        raise ValueError("This attempt does not have both fixed and generated atoms; check the task definition.")
    reference = prior["coords"][0].tolist()
    reference_types = prior["atomics"][0].argmax(-1).tolist()
    scale = float(inputs["coord_scale"])
    task_path = folder / "task_definition.json"
    task = json.loads(task_path.read_text()) if task_path.exists() else {}
    anchors = sorted({s["fixed_sampler_index"] for s in task.get("reference_attachment_sites", [])
                      if s.get("fixed_sampler_index") in ids}) if task.get("mapping_verified") else []

    def measure(data):
        result = displacement_metrics(data["coords"][0].tolist(), reference, ids, scale)
        elements = data["atomics"][0].argmax(-1).tolist()
        result["fixed_element_changes"] = sum(elements[i] != reference_types[i] for i in ids)
        return result

    frames = []
    for path in sorted((folder / "sampler_steps").glob("step_*.pt")):
        item = load(path)
        frames.append({"step_index": int(item["step_index"]), "stage": item["stage"],
                       **measure(item["current"])})
    calls = []
    for path in sorted((folder / "forward_calls").glob("call_*.pt")):
        item = load(path)
        prediction = item["network_output_before_postprocessing"]
        row = {"call_index": int(item["call_index"]),
               "continuous_time": float(item["times_scalar"][0]), **measure(prediction)}
        # FLOWR uses bond class 0 for no bond. These are diagnostic predictions,
        # not chemical-validity labels or a calibrated probability of success.
        if anchors and "bonds" in prediction:
            probs = torch.softmax(prediction["bonds"][0], dim=-1)
            for anchor in anchors:
                value = float((1.0 - probs[anchor, variable_ids, 0]).max())
                row[f"anchor_{anchor}_max_predicted_bond_probability"] = value if math.isfinite(value) else None
        calls.append(row)

    if not frames or not calls:
        raise ValueError("Missing sampler_steps or forward_calls snapshots; use the full local run directory, not the compact review bundle.")
    output = folder / "trace_check"
    output.mkdir(exist_ok=True)
    write_csv(output / "sampler_fixed_drift.csv", frames)
    write_csv(output / "network_fixed_drift.csv", calls)
    expected_path = folder / "attempt.json"
    expected = json.loads(expected_path.read_text()) if expected_path.exists() else {}
    complete = (len(frames) == expected.get("saved_sampler_frames") and
                len(calls) == expected.get("forward_calls"))
    report = {
        "attempt": folder.name,
        "n_fixed_atoms": len(ids), "n_generated_slots": len(variable_ids),
        "reference_attachment_sampler_indices": anchors,
        "sampler_frames_read": len(frames), "network_calls_read": len(calls),
        "snapshot_counts_match_recorder": complete,
        "numerical_detection_tolerance_A": numerical_tolerance,
        "tolerance_note": "Used to locate departures from numerical fixation; NOT a scaffold-hopping success/failure cutoff.",
        "coordinate_note": "Compare against the saved fixed coordinates in the same model frame, converted to angstroms. No alignment or fitting.",
        "bond_probability_note": "Maximum predicted nonzero bond-class probability from an attachment atom to any generated slot; NOT a success label.",
        **summarize_stages(frames, numerical_tolerance),
        "scope": "Temporal observation of states and network outputs; no hidden-layer causal attribution or intervention has been performed.",
    }
    try:
        report["plot"] = optional_plot(output, calls, frames)
    except Exception as exc:
        report["plot"] = f"Plot unavailable: {type(exc).__name__}: {exc}; CSV/JSON were retained."
    write_json(output / "trace_report.json", report)
    return report, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Full local run directory, containing research_record/")
    parser.add_argument("--numerical-tolerance", type=float, default=1e-4,
                        help="Numerical detection tolerance in angstroms, not a molecular success criterion.")
    args = parser.parse_args()
    if not math.isfinite(args.numerical_tolerance) or args.numerical_tolerance <= 0:
        parser.error("--numerical-tolerance must be finite and positive")
    root = args.run_dir.resolve() / "research_record"
    folders = sorted(p for p in root.glob("attempt_*") if p.is_dir())
    if not folders:
        parser.error(f"No recorded attempts found in {root}")
    import torch
    outputs = []
    failed = False
    for folder in folders:
        try:
            report, output = analyze_attempt(folder, torch, args.numerical_tolerance)
            outputs.append(output)
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            if not report["snapshot_counts_match_recorder"]:
                failed = True
                print("CHECK: snapshot counts do not match the recorder summary.")
        except Exception as exc:
            failed = True
            print(f"Could not analyze {folder.name}: {type(exc).__name__}: {exc}")
    if outputs:
        bundle = args.run_dir.resolve() / "trace_review.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for output in outputs:
                for name in ("trace_report.json", "sampler_fixed_drift.csv", "network_fixed_drift.csv", "fixed_drift.png"):
                    path = output / name
                    if path.exists():
                        z.write(path, path.relative_to(args.run_dir.resolve()))
            for name in ("manifest.json", "summary.json"):
                path = root / name
                if path.exists():
                    z.write(path, path.relative_to(args.run_dir.resolve()))
        print(f"\nReview file: {bundle}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
