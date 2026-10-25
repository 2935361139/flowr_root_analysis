#!/usr/bin/env python3
"""Resume the existing causal scan after its NumPy scalar JSON failure.

Only the result writer is replaced, at runtime. The original runner, model,
experiment plan, comparisons, random controls and identity checks are unchanged.
No training or new experimental conditions are introduced.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

FIX_VERSION = "2026-10-22.numpy-json-resume.1"
SUPPORTED_SOURCE = {
    "scan_causal_features.py": "1d692c91982b11595dc621bda971f9f1becf485023d6bdf242b682cd4814cf78",
    "native_adapter.py": "99b694cc265d831c26ffbadd6991b2dd40cc4a1692dddc46c7becf5eafe7e117",
    "evaluate_scaffold_outputs.py": "97eda26eca03e1f8dcb3d59b8b33430cbb855e87508eeee20f17905a3fe66f75",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def numpy_scalar(value):
    """Convert ordinary NumPy scalars without rounding or changing diagnostics."""
    if isinstance(value, np.generic):
        converted = value.item()
        if type(converted) in (bool, int, float, str):
            return converted
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path, value):
    # NaN/Infinity still fail. Arrays and tensors are not silently converted.
    # Serialize before writing so an invalid value cannot damage an old result.
    content = json.dumps(value, ensure_ascii=False, indent=2,
                         allow_nan=False, default=numpy_scalar)
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def load_runner(directory):
    require("native_adapter" not in sys.modules,
            "Run this file as a fresh Python command, not inside a notebook.")
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(
        "original_scaffold_scan_for_resume", directory / "scan_causal_features.py")
    require(spec is not None and spec.loader is not None, "Cannot load original runner.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(Path(module.native.__file__).resolve() == directory / "native_adapter.py",
            "Wrong native_adapter imported; keep the original runner folder intact.")
    return module


def verify_sources(directory, identity):
    require(identity.get("code_sha256") == SUPPORTED_SOURCE,
            "This fix supports the uploaded causal-scan v1 only. Preserve your files.")
    for name, expected in SUPPORTED_SOURCE.items():
        require(digest(directory / name) == expected,
                f"Original source changed: {name}. Do not edit the run identity.")


def verify_completed(out, trials):
    completed = []
    expected = {t["id"] for t in trials}
    for path in (out / "trials").glob("*/result.json"):
        require(path.parent.name in expected, f"Unexpected completed trial: {path.parent.name}")
    for trial in trials:
        folder = out / "trials" / trial["id"]
        result_path = folder / "result.json"
        if not result_path.is_file():
            continue
        saved = read_json(result_path)
        require(saved.get("completed") is True and saved.get("trial") == trial,
                f"Completed-trial metadata differs: {trial['id']}")
        hashes = saved.get("file_sha256")
        require(isinstance(hashes, dict) and hashes,
                f"Completed-trial hashes missing: {trial['id']}")
        for name, expected_sha in hashes.items():
            require(Path(name).name == name, "Unexpected result-file path.")
            path = folder / name
            require(path.is_file() and digest(path) == expected_sha,
                    f"Missing or changed completed output: {path}. Use the original output folder.")
        completed.append(trial["id"])
    return completed


def preserve_previous_errors(out):
    history = out / "serialization_fix_history"
    history.mkdir(exist_ok=True)
    archived = []
    for name in ("error.txt", "report_error.txt"):
        path = out / name
        if not path.is_file():
            continue
        target = history / f"{path.stem}_{digest(path)}.txt"
        require(not target.exists() or target.read_bytes() == path.read_bytes(),
                "An error-history file conflicts; preserve files for inspection.")
        path.replace(target)
        archived.append(str(target.relative_to(out)))
    return archived


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--runner-dir", type=Path, default=Path("scaffold_causal_scan_v1"))
    ap.add_argument("--plan", type=Path)
    ap.add_argument("--head-dir", type=Path, default=Path("outputs/sh_head_bace_lig16_v3"))
    ap.add_argument("--output-root", type=Path,
                    default=Path("outputs/sh_boundary_screen_v1/bace_lig16"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("outputs/sh_causal_scan_bace_lig16_v1"))
    ap.add_argument("--resume", action="store_true", help="Always enabled by this launcher.")
    args = ap.parse_args(argv)
    directory, out = args.runner_dir.resolve(), args.output_dir.resolve()
    identity_path = out / "run_identity.json"
    require(identity_path.is_file(), "Use the existing local scan output, not the review ZIP.")
    identity_hash = digest(identity_path)
    identity = read_json(identity_path)
    verify_sources(directory, identity)
    runner = load_runner(directory)
    plan_path = (args.plan or directory / "bace_scan_plan.json").resolve()
    plan = runner.read_plan(plan_path)
    require(plan == identity["plan"], "Plan differs from the existing run.")
    trials = runner.plan_trials(plan)
    require(read_json(out / "planned_trials.json") == trials,
            "Planned trials differ from the existing run.")
    completed = verify_completed(out, trials)

    # Preserve the old summary and traceback; keep the original identity untouched.
    history = out / "serialization_fix_history"
    history.mkdir(exist_ok=True)
    summary_path = out / "scan_summary.json"
    if summary_path.is_file():
        (history / f"summary_{digest(summary_path)}.json").write_bytes(summary_path.read_bytes())
    archived = preserve_previous_errors(out)
    receipt_path = out / "serialization_fix.json"
    receipt = {
        "fix_version": FIX_VERSION,
        "wrapper_sha256": digest(__file__),
        "original_run_identity_sha256": identity_hash,
        "original_code_sha256": SUPPORTED_SOURCE,
        "numpy_version": np.__version__,
        "scope": "Only native_adapter.write_json; convert NumPy scalars to Python scalars.",
        "nonfinite_values_rejected": True,
        "original_identity_and_replay_checks_enabled": True,
        "verified_completed_trials": completed,
        "planned_trial_count": len(trials),
        "archived_previous_errors": archived,
    }
    if receipt_path.is_file():
        (history / f"receipt_{digest(receipt_path)}.json").write_bytes(receipt_path.read_bytes())
    write_json(receipt_path, receipt)
    (out / "resume_scaffold_scan.py").write_bytes(Path(__file__).read_bytes())
    require(digest(identity_path) == identity_hash, "Run identity unexpectedly changed.")

    print(f"JSON fix active. Verified {len(completed)}/{len(trials)} completed trials; "
          "continuing the original plan.", flush=True)
    old_writer, old_argv = runner.native.write_json, sys.argv
    runner.native.write_json = write_json
    sys.argv = [str(directory / "scan_causal_features.py"),
                "--plan", str(plan_path), "--head-dir", str(args.head_dir.resolve()),
                "--output-root", str(args.output_root.resolve()),
                "--output-dir", str(out), "--resume"]
    try:
        return runner.main()
    finally:
        runner.native.write_json = old_writer
        sys.argv = old_argv


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, TypeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(1)
