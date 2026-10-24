#!/usr/bin/env python3
"""Collect a small fixed-input scaffold-hopping pilot using the existing recorder.

Run in the FLOWR.ROOT repository and its working Python environment.
Default: ptp1b ligand 0, seeds 42..51, one attempt per seed, unchanged sampler.
Existing run directories are retained. Failed attempts are retained too.
This script does not assign a full scaffold-hopping success label.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def condition_key(manifest):
    """Compare recorded conditions; only seed and output location may differ."""
    return {
        "args": {k: v for k, v in manifest.get("args", {}).items() if k not in {"seed", "save_dir"}},
        "input_sha256": manifest.get("input_sha256"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "source_sha256": manifest.get("source_sha256"),
        "recorder_sha256": manifest.get("recorder_sha256"),
        "torch_version": manifest.get("torch_version"),
        "rdkit_version": manifest.get("rdkit_version"),
    }


def stream_process(command, log_path):
    print("Running:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace", bufsize=1) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                return process.wait()
            except KeyboardInterrupt:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise


def one_run_row(run, seed):
    record = run / "research_record"
    summary = read_json(record / "summary.json")
    manifest = read_json(record / "manifest.json")
    attempts = summary.get("attempts", [])
    attempt = attempts[0] if len(attempts) == 1 else {}
    task = attempt.get("task_definition", {})
    if not isinstance(task, dict):
        task = {}
    trace = read_json(record / "attempt_000/trace_check/trace_report.json")
    raw_list = attempt.get("decode_without_valence_repair", [])
    run_list = attempt.get("decode_with_run_setting", [])
    raw = raw_list[0] if len(raw_list) == 1 else {}
    decoded = run_list[0] if len(run_list) == 1 else {}
    row = {
        "seed": seed,
        "run_directory": str(run),
        "summary_available": bool(summary),
        "recorded_seed_matches": manifest.get("args", {}).get("seed") == seed,
        "n_attempts": summary.get("n_attempts"),
        "generation_completed": attempt.get("generation_completed"),
        "pipeline_error": summary.get("pipeline_error"),
        "audit_error": attempt.get("audit_error"),
        "mapping_verified": task.get("mapping_verified"),
        "n_fixed_atoms": task.get("n_fixed_atoms"),
        "n_generated_slots": task.get("n_generated_slots"),
        "raw_decode_built": raw.get("built"),
        "raw_decode_sanitizable": raw.get("sanitizable"),
        "raw_decode_connected": raw.get("connected"),
        "run_decode_built": decoded.get("built"),
        "run_decode_sanitizable": decoded.get("sanitizable"),
        "run_decode_connected": decoded.get("connected"),
        "fixed_rms_A": attempt.get("raw_final_fixed_displacement_rms_A"),
        "fixed_max_A": attempt.get("raw_final_fixed_displacement_max_A"),
        "fixed_element_changes": attempt.get("raw_final_fixed_element_changes"),
        "trace_counts_match": trace.get("snapshot_counts_match_recorder"),
        "integrated_fixed_states_within_numerical_tolerance": trace.get(
            "all_saved_integrated_states_within_numerical_tolerance"),
        "first_displaced_saved_stage": (
            trace.get("first_saved_frame_exceeding_numerical_tolerance") or {}).get("stage"),
        "full_scaffold_hopping_success": "NOT_YET_ASSESSED",
    }
    return row, manifest


def collect(output_root, case, seeds, folder, events):
    rows, runs, reference_key = [], [], None
    for seed in seeds:
        run = output_root / f"sh_record_{case}_seed{seed}"
        runs.append((seed, run))
        try:
            row, manifest = one_run_row(run, seed)
            if manifest:
                key = condition_key(manifest)
                if reference_key is None:
                    reference_key = key
                row["same_recorded_settings_except_seed"] = key == reference_key
            else:
                row["same_recorded_settings_except_seed"] = None
        except Exception as exc:
            row = {"seed": seed, "run_directory": str(run),
                   "collection_error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (folder / "pilot_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "case": case, "requested_seeds": seeds, "events": events,
        "completed_generation_attempts": sum(r.get("generation_completed") is True for r in rows),
        "raw_decoded_sanitizable_and_connected": sum(
            r.get("raw_decode_sanitizable") is True and r.get("raw_decode_connected") is True for r in rows),
        "records": rows,
        "scope": "Single-complex pilot for collecting contrasting outcomes, not a held-out benchmark or a mechanistic result.",
        "label_note": "Decode validity, connectivity and fixed-coordinate displacement are separate diagnostics. Attachment-site geometry, scaffold change and full task success still require evaluation. A failed decode alone does not establish the cause of failure.",
        "comparison_note": "Check task definitions and saved coordinates as well as settings before matching trajectories. Different random seeds are observational comparisons, not causal interventions.",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(folder / "pilot_summary.json", report)
    bundle = folder / "pilot_review.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ("pilot_summary.csv", "pilot_summary.json"):
            z.write(folder / name, name)
        for seed, run in runs:
            record = run / "research_record"
            if record.is_dir():
                for p in sorted(record.rglob("*")):
                    if p.is_file() and not p.is_symlink() and p.suffix in {".json", ".csv", ".sdf", ".png", ".txt"}:
                        z.write(p, Path(f"seed_{seed}") / p.relative_to(run))
        for p in sorted((folder / "logs").glob("*.log")):
            z.write(p, Path("logs") / p.name)
    print(f"\nGenerated attempts completed: {report['completed_generation_attempts']}/{len(seeds)}", flush=True)
    print(f"Review file: {bundle}", flush=True)
    print("Large tensor snapshots remain in the individual run directories.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["ptp1b", "bace", "cdk2", "tyk2"], default="ptp1b")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--collect-only", action="store_true", help="Rebuild the small review bundle from existing files; do not launch inference or analysis.")
    args = parser.parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    if any(s < 0 or s > 4294967295 for s in seeds):
        parser.error("Seeds must be integers between 0 and 4294967295.")
    root = Path.cwd()
    if not args.collect_only:
        for name in ("record_scaffold_run.py", "inspect_scaffold_trace.py", "flowr/gen/generate_from_pdb.py"):
            if not (root / name).is_file():
                parser.error(f"Missing {name}; run from the FLOWR.ROOT repository with both earlier scripts present.")
    output_root = args.output_root.resolve()
    folder = output_root / f"sh_pilot_{args.case}"
    (folder / "logs").mkdir(parents=True, exist_ok=True)
    events = []
    try:
        if not args.collect_only:
            for index, seed in enumerate(seeds, 1):
                run = output_root / f"sh_record_{args.case}_seed{seed}"
                print(f"\n[{index}/{len(seeds)}] {args.case}, seed {seed}", flush=True)
                if run.exists():
                    print("Keeping existing run; generation will not be repeated.", flush=True)
                    events.append({"seed": seed, "action": "existing_run_retained"})
                else:
                    code = stream_process([
                        sys.executable, "-u", str(root / "record_scaffold_run.py"),
                        "--case", args.case, "--seed", str(seed), "--save_dir", str(run)],
                        folder / "logs" / f"seed_{seed}_record.log")
                    events.append({"seed": seed, "action": "record", "returncode": code})
                    if code:
                        print("Recorder reported a nonzero exit; retaining the output and continuing. The report distinguishes incomplete runs from completed generations.", flush=True)
                record = run / "research_record"
                manifest = read_json(record / "manifest.json")
                attempt = read_json(record / "attempt_000/attempt.json")
                if manifest.get("args", {}).get("seed") != seed:
                    events.append({"seed": seed, "action": "trace_skipped", "reason": "missing or mismatched recorded seed"})
                    continue
                if attempt.get("forward_calls", 0) and attempt.get("saved_sampler_frames", 0):
                    code = stream_process([
                        sys.executable, "-u", str(root / "inspect_scaffold_trace.py"), str(run)],
                        folder / "logs" / f"seed_{seed}_trace.log")
                    events.append({"seed": seed, "action": "inspect_trace", "returncode": code})
                else:
                    events.append({"seed": seed, "action": "trace_skipped", "reason": "no complete saved calls/frames"})
    except KeyboardInterrupt:
        events.append({"action": "interrupted"})
        print("\nInterrupted; packaging the records collected so far.", flush=True)
    finally:
        report = collect(output_root, args.case, seeds, folder, events)
    incomplete = any(r.get("generation_completed") is not True or
                     r.get("same_recorded_settings_except_seed") is not True or
                     r.get("recorded_seed_matches") is not True or
                     r.get("trace_counts_match") is not True or r.get("audit_error")
                     for r in report["records"])
    if incomplete:
        print("CHECK: some runs are incomplete or need review; their records are included.", flush=True)
    return 2 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
