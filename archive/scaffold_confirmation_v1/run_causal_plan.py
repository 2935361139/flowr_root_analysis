#!/usr/bin/env python3
"""Run a new, explicit causal-scan plan using the verified v1 runner and JSON fix.

Model computation, intervention construction, evaluation and replay checks are
delegated to the existing code. This launcher also supports --resume.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

FIX_SHA = "a706d1a21a52093a29bb9e3a4715a1081c3a152108a62f3f2c649a2159f9b28c"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--runner-dir", type=Path, default=Path("scaffold_causal_scan_v1"))
    ap.add_argument("--json-fix", type=Path, default=Path("resume_scaffold_scan.py"))
    ap.add_argument("--head-dir", type=Path, default=Path("outputs/sh_head_bace_lig16_v3"))
    ap.add_argument("--output-root", type=Path,
                    default=Path("outputs/sh_boundary_screen_v1/bace_lig16"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    helper = args.json_fix.resolve()
    if not helper.is_file() or hashlib.sha256(helper.read_bytes()).hexdigest() != FIX_SHA:
        raise ValueError("Missing/changed resume_scaffold_scan.py. Use the previously verified JSON-fix file.")
    spec = importlib.util.spec_from_file_location("verified_causal_json_fix", helper)
    fix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fix)
    directory, out = args.runner_dir.resolve(), args.output_dir.resolve()
    plan_path = args.plan.resolve()
    forwarded = ["--runner-dir", str(directory), "--plan", str(plan_path),
                 "--head-dir", str(args.head_dir.resolve()),
                 "--output-root", str(args.output_root.resolve()), "--output-dir", str(out)]

    if args.resume:
        # Reuse the existing identity and completed-file checks without alteration.
        status = fix.main(forwarded + ["--resume"])
    else:
        fix.require(not out.exists(), "Output already exists. Use --resume; do not overwrite it.")
        fix.verify_sources(directory, {"code_sha256": fix.SUPPORTED_SOURCE})
        runner = fix.load_runner(directory)
        plan = runner.read_plan(plan_path)
        trials = runner.plan_trials(plan)
        print(f"New plan: {len(trials)} trials. Model computation and evaluation use the original runner.",
              flush=True)
        old_writer, old_argv = runner.native.write_json, sys.argv
        runner.native.write_json = fix.write_json
        sys.argv = [str(directory / "scan_causal_features.py"), *forwarded[2:]]
        try:
            status = runner.main()
        finally:
            runner.native.write_json, sys.argv = old_writer, old_argv
            if out.is_dir():
                fix.write_json(out / "serialization_fix.json", {
                    "fix_version": "2026-10-22.numpy-json-new-plan.1",
                    "helper_sha256": FIX_SHA,
                    "wrapper_sha256": fix.digest(__file__),
                    "plan_sha256": fix.digest(plan_path),
                    "scope": "Only NumPy scalar JSON serialization; scientific computation unchanged.",
                    "nonfinite_values_rejected": True,
                    "original_identity_and_replay_checks_enabled": True,
                    "planned_trial_count": len(trials),
                })
                (out / "resume_scaffold_scan.py").write_bytes(helper.read_bytes())

    if out.is_dir():
        # Include the exact launcher/plan so the review ZIP is auditable.
        (out / "run_causal_plan.py").write_bytes(Path(__file__).read_bytes())
        (out / "selected_plan.json").write_bytes(plan_path.read_bytes())
        import zipfile
        with zipfile.ZipFile(out / "causal_scan_review.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
            for p in sorted(out.rglob("*")):
                rel = p.relative_to(out)
                if (p.is_file() and p.name != "causal_scan_review.zip"
                        and p.suffix not in {".pt", ".tmp"} and "model_context" not in rel.parts):
                    bundle.write(p, rel)
    return status


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, TypeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(1)
