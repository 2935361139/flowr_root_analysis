"""Meaningful software checks; run from the ROOT repository using NumPy.

python /path/to/scaffold_feedback_v1/tests/check_feedback.py --output /path/result.json
Uses the actual native sampler function bodies with isolated synthetic tensors.
This does not load a checkpoint and does not establish scientific efficacy.
"""
from __future__ import annotations
import argparse
import ast
import copy
import importlib.util
import json
import random
import sys
import tempfile
import types
import contextlib
import io
from pathlib import Path
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import native_adapter as native
import feedback_core as core
from feedback_hook import FeedbackPatch
import run_feedback_path as runner
import _numpy_native as backend


def run():
    checks = []
    # Vary scales, native precisions, centres, and geometry rather than test one pair.
    random.seed(104)
    np.random.seed(218)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    rng = np.random.default_rng(12)
    count = 0
    for dtype in [np.float32, np.float64]:
        for scale in [.01, 1., 100.]:
            for k in range(25):
                b = (rng.normal(size=(2, 3)) * scale).astype(dtype)
                x = (rng.normal(size=(2, 3)) * scale).astype(dtype)
                target, info = core.build_change(b, x, "targeted", f"{k}")
                assert np.linalg.norm((target[1] - target[0]) - (x[1] - x[0])) < 2 * info["rounding_bound_model_units"]
                assert np.allclose(target.mean(0), b.mean(0), atol=info["rounding_bound_model_units"], rtol=0)
                for c in range(3):
                    a, receipt = core.build_change(b, x, "random_direction", f"{k}:{c}")
                    repeated, _ = core.build_change(b, x, "random_direction", f"{k}:{c}")
                    assert np.array_equal(a, repeated)
                    assert abs(np.linalg.norm(a.astype(float)-b) - np.linalg.norm(target.astype(float)-b)) <= receipt["rounding_bound_model_units"]
                    delta = np.asarray(receipt["designed_delta"])
                    intended = np.asarray(info["designed_delta"])
                    assert abs(np.sum(delta * intended)) < 1e-10 * max(1, np.linalg.norm(intended)**2)
                identity, receipt = core.build_change(b, x, "sham", "unused")
                assert np.array_equal(identity, b) and receipt["delta_norm_model_units"] == 0
                q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                shift = rng.normal(size=(1, 3)) * scale
                rotated, _ = core.build_change((b @ q + shift).astype(dtype), (x @ q + shift).astype(dtype), "targeted", "unused")
                assert np.allclose(rotated, target @ q + shift, atol=info["rounding_bound_model_units"] * 2, rtol=0)
                count += 1
    assert python_rng == random.getstate()
    now = np.random.get_state()
    assert numpy_rng[0] == now[0] and np.array_equal(numpy_rng[1], now[1]) and numpy_rng[2:] == now[2:]
    for kind in ["sham", "targeted", "random_direction"]:
        zero = np.array([[0., 0., 0.], [1., 0., 0.]], np.float32)
        a, receipt = core.build_change(zero, zero, kind, "zero")
        assert np.array_equal(a, zero) and not receipt["informative_magnitude"]
    checks.append(dict(name="geometry_controls", cases=count, passed=True, scope="centroid, relative vector, scale, dtype, rotation/translation of target, orthogonal norm-matched random directions, RNG isolation, zero delta"))
    plan = core.validate_plan(native.read_json(PACKAGE / "bace_feedback_plan.json"))
    trials = core.plan_trials(plan)
    assert len(trials) == 50 and all(t["kind"] == "sham" for t in trials[:10])
    for task, expected in [({"fixed_reference_indices": [8, 13, 20], "fixed_sampler_indices": [0, 4, 6], "mapping_verified": True}, {"site": 4, "partner": 6}),
                           ({"fixed_reference_indices": [20, 13, 8], "fixed_sampler_indices": [1, 2, 7], "mapping_verified": True}, {"site": 2, "partner": 1})]:
        assert core.resolve_focus(task, {"site_reference_index": 13, "partner_reference_index": 20}) == expected
    try:
        core.resolve_focus(task, {"site_reference_index": 99, "partner_reference_index": 20})
    except ValueError:
        pass
    else:
        raise AssertionError("Generated/unmapped atom accepted")
    checks.append(dict(name="plan_and_atom_mapping", passed=True, planned_rollouts=50))
    T, Torch, Mod, OldModel = backend.T, backend.Torch, backend.Mod, backend.Model
    fixed_indices = [0, 2]
    class Model(OldModel):
        def __init__(self, d=5):
            super().__init__(d)
            self._inpaint_self_condition = False
            def inpaint(data, prediction, **kw):
                for key in ["coords", "atomics", "charges"]:
                    if key in prediction:
                        prediction[key].x[:, fixed_indices] = data[key].x[:, fixed_indices]
                return prediction
            self.builder.inpaint_molecule = inpaint
        def forward(self, curr, pocket, t, cond_batch=None, **kwargs):
            out = super().forward(curr, pocket, t, cond_batch=cond_batch, **kwargs)
            # A synthetic global interaction allows fixed-atom feedback to affect
            # generated atoms. The older independent-node fixture loses every
            # fixed-only perturbation under repeated hard inpainting.
            coupling = np.float32(.03) * np.mean(backend.u(cond_batch["coords"]) ** 2, dtype=np.float32)
            out["coords"] = out["coords"] + T(coupling)
            return out
    backend.Model = Model
    for channels, site, partner in [(5, 0, 2), (11, 1, 4)]:
        fixed_indices[:] = [site, partner]
        model, data, _ = backend.fixture(channels)
        original_data = backend.snap(data)
        trial = dict(id="test_sham", kind="sham", seed=103, site=site, partner=partner, call=1, target_bond_class=3, reference_bond_class=0)
        held = {}
        def factory(m, t, clock, torch):
            patch = FeedbackPatch(m, t, clock, torch, plan, data["inputs"]["coord_scale"])
            held["patch"] = patch
            return patch
        final, replay, logs, sampler = native.run_rollout(model, data, trial, factory, backend.helper, model.device, Torch)
        assert replay["baseline_or_sham_full_match"] and replay["final_match"]["passed"] and not replay["patch"]["applied"]
        assert len(held["patch"].rows) == len(logs) == len(sampler) == 6
        assert all(e["max_difference"] == 0 for e in replay["checked_calls"])
        reference = held["patch"].capture
        sham_rows = copy.deepcopy(held["patch"].rows)
        for kind in ["targeted", "random_direction"]:
            t = {**trial, "kind": kind, "control_index": 0}
            def factory(m, t, clock, torch):
                patch = FeedbackPatch(m, t, clock, torch, plan, data["inputs"]["coord_scale"], reference)
                held["patch"] = patch
                return patch
            final, replay, logs, sampler = native.run_rollout(model, data, t, factory, backend.helper, model.device, Torch)
            assert replay["random_streams_matched_all_calls"] and replay["patch"]["visits"] == 1 and replay["patch"]["applied"]
            assert not np.array_equal(native.array(final["coords"]), native.array(data["final"]["coords"]))
            assert held["patch"].rows[0] == sham_rows[0]
            assert held["patch"].rows[2]["feedback_pair_before_A"] != sham_rows[2]["feedback_pair_before_A"]
            assert len(held["patch"].rows) == 6
            # Later feedback is the altered prediction, not a later saved baseline state.
            assert np.allclose(held["patch"].rows[2]["feedback_pair_before_A"], held["patch"].rows[1]["network_pair_A"], atol=0, rtol=0)
            assert not model.pre and not model.post
            assert "_build_trajectory" not in vars(model) and "get_pocket_encoding" not in vars(model.gen)
        # Snapshot all numerical source leaves; the pocket's SimpleNamespace
        # intentionally has no value-based equality operator in this test backend.
        leaves = backend.helper.tensor_leaves(data, Torch)
        original_leaves = backend.helper.tensor_leaves(original_data, Torch)
        assert set(leaves) == set(original_leaves)
        assert all(np.array_equal(backend.u(v), backend.u(original_leaves[k])) for k, v in leaves.items())
        assert np.array_equal(backend.u(data["inputs"]["pocket_data"]["complex"][0].com),
                              backend.u(original_data["inputs"]["pocket_data"]["complex"][0].com))
        wrong = {k: v.copy() for k, v in reference.items()}
        wrong["feedback_pair"][0, 0] += .1
        bad = lambda m, t, clock, torch: FeedbackPatch(m, t, clock, torch, plan, data["inputs"]["coord_scale"], wrong)
        try:
            native.run_rollout(model, data, t, bad, backend.helper, model.device, Torch)
        except ValueError as exc:
            assert "Pre-intervention state" in str(exc)
        else:
            raise AssertionError("Wrong sham reference accepted")
        assert not model.pre and not model.post
        damaged = backend.snap(data)
        damaged["calls"][1]["state_before_forward"]["coords"].x[0, 2, 0] += .1
        try:
            native.run_rollout(model, damaged, trial, factory, backend.helper, model.device, Torch)
        except ValueError as exc:
            assert "Native replay diverges" in str(exc)
        else:
            raise AssertionError("Wrong original trajectory accepted")
        assert not model.pre and not model.post
        checks.append(dict(name="actual_native_sampler_with_synthetic_tensors", channels=channels, site=site, partner=partner,
                           passed=True, identity_replay_exact=True, both_patch_types_change_continuation=True,
                           original_records_unchanged=True, later_states_not_overwritten=True,
                           full_random_stream_matched=True, wrong_reference_and_state_drift_rejected=True, cleanup_on_error=True))
    # JSON error that previously interrupted the old scan must not recur.
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "check.json"
        native.write_json(path, {"x": np.float32(.25), "ok": np.bool_(True), "a": np.arange(3)})
        assert native.read_json(path) == {"x": .25, "ok": True, "a": [0, 1, 2]}
        try:
            native.write_json(path, {"x": np.float32(np.nan)})
        except ValueError:
            pass
        else:
            raise AssertionError("NaN accepted in result JSON")
    checks.append(dict(name="numpy_json_and_nonfinite_rejection", passed=True))
    # Exercise the actual CLI bookkeeping/report/resume path with a synthetic
    # runtime. Model loading, preflight data and chemistry are explicitly mocked;
    # no mock chemistry outcome is delivered as a scientific result.
    test_plan = copy.deepcopy(plan)
    test_plan["seeds"] = [103]
    model, data, _ = backend.fixture(7)
    data["focus"] = {"site": fixed_indices[0], "partner": fixed_indices[1]}
    data["inputs"].update(self_condition=True, sc_recycle_steps=1)
    model.to = lambda device: model
    model.eval = lambda: model
    model.parameters = lambda: []
    Torch.set_float32_matmul_precision = staticmethod(lambda _: None)
    Torch.save = staticmethod(lambda obj, path: Path(path).write_bytes(b"SYNTHETIC SOFTWARE TEST ONLY"))
    fake_flowr = types.ModuleType("flowr")
    fake_flowr.__path__ = []
    fake_util = types.ModuleType("flowr.util")
    fake_util.__path__ = []
    fake_script = types.ModuleType("flowr.scriptutil")
    fake_script.load_model = lambda args: (model,)
    fake_device = types.ModuleType("flowr.util.device")
    fake_device.resolve_device = lambda args: model.device
    injected = {"torch": Torch, "flowr": fake_flowr, "flowr.util": fake_util,
                "flowr.scriptutil": fake_script, "flowr.util.device": fake_device}
    old_modules = {k: sys.modules.get(k) for k in injected}
    old_prepare, old_evaluate, old_rollout = runner.prepare, native.evaluate_final, native.run_rollout
    count = {"rollouts": 0}
    def counted(*args, **kwargs):
        count["rollouts"] += 1
        return old_rollout(*args, **kwargs)
    def fake_evaluate(*args, **kwargs):
        raw = dict(stage="raw_without_repair", root_fixed_substructure_match=np.bool_(False),
                   fixed_chemistry_preserved=False, fc_valid=True, posebusters_valid=False,
                   murcko_scaffold_changed=True, errors=[])
        return dict(stages=[raw], raw_valid_connected_root_fragment=False,
                    final_graph={"bond_class": 1, "site_explicit_bond_order_sum": 4.})
    try:
        sys.modules.update(injected)
        runner.prepare = lambda *args: ({103: data}, {"parameter_tensors_requiring_grad": 0}, {"native_precision": "high"})
        native.evaluate_final, native.run_rollout = fake_evaluate, counted
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            config = folder / "plan.json"
            native.write_json(config, test_plan)
            out = folder / "synthetic_output"
            command = ["--plan", str(config), "--output-dir", str(out)]
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                status = runner.main(command)
            assert status == 0, captured.getvalue()
            assert count["rollouts"] == 5
            assert native.read_json(out / "feedback_summary.json")["completed"]
            assert (out / "REPORT.html").is_file() and (out / "feedback_trajectories.png").is_file()
            import zipfile
            with zipfile.ZipFile(out / "feedback_review.zip") as z:
                assert z.testzip() is None and "feedback_results.csv" in z.namelist()
                assert not any(n.endswith(".pt") for n in z.namelist())
            with contextlib.redirect_stdout(captured):
                status = runner.main(command + ["--resume"])
            assert status == 0 and count["rollouts"] == 5, captured.getvalue()
            damaged = out / "trials/sham_s103/feedback_timeline.json"
            damaged.write_text(damaged.read_text() + " ")
            with contextlib.redirect_stdout(captured):
                status = runner.main(command + ["--resume"])
            assert status == 1 and "Resume output changed" in native.read_json(out / "feedback_summary.json")["error"]
            assert count["rollouts"] == 5
    finally:
        runner.prepare, native.evaluate_final, native.run_rollout = old_prepare, old_evaluate, old_rollout
        for k, original in old_modules.items():
            if original is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = original
    checks.append(dict(name="CLI_reports_resume_and_tamper_detection_with_mock_runtime", passed=True,
                       note="Preflight and chemistry mocked; real native-sampler synthetic rollouts and real output/resume code exercised."))
    for p in PACKAGE.rglob("*.py"):
        ast.parse(p.read_text())
    return {"software_checks_passed": True, "checks": checks,
            "pretrained_model_executed_here": False, "GPU_experiment_executed_here": False,
            "limitation": "Software and actual-sampler control-flow checks with synthetic tensors. Scientific effects and GPU equivalence remain to be measured by the mandatory runtime identity replays.",
            "package_source_sha256": {str(p.relative_to(PACKAGE)): native.digest(p) for p in PACKAGE.rglob("*.py")}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = run()
    if args.output:
        native.write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
