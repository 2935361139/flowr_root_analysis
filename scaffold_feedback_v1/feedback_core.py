"""Pure NumPy construction of a predeclared, single-call feedback intervention.

No model RNG is used. Coordinates are in the model's native units. Random
controls live in the plane orthogonal to the intended relative-vector change;
both atoms move oppositely, preserving the observed feedback pair centroid.
"""
from __future__ import annotations
import hashlib
import re
import numpy as np


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def validate_plan(p):
    require(isinstance(p.get("case"), str) and re.fullmatch(r"[A-Za-z0-9_-]+", p["case"]), "Invalid case name.")
    seeds = p.get("seeds")
    require(isinstance(seeds, list) and seeds and all(type(s) is int and s >= 0 for s in seeds)
            and len(seeds) == len(set(seeds)), "Seeds must be distinct nonnegative integers.")
    focus = p.get("focus", {})
    for k in ["site_reference_index", "partner_reference_index"]:
        require(type(focus.get(k)) is int and focus[k] >= 0, f"Invalid focus.{k}.")
    require(focus["site_reference_index"] != focus["partner_reference_index"], "Self-pair unsupported.")
    require(type(focus.get("expected_bond_class")) is int and focus["expected_bond_class"] in [1, 2, 3, 4], "Expected bond must exist in the fixed reference.")
    require(type(p.get("intervention_call")) is int and p["intervention_call"] >= 1, "A feedback intervention requires a prior network call.")
    calls = p.get("early_calls")
    require(isinstance(calls, list) and calls and all(type(c) is int and c >= p["intervention_call"] for c in calls)
            and calls == sorted(set(calls)), "Invalid early observation calls.")
    require(p.get("random_controls") == 3, "This preregistered pilot uses exactly three random-direction controls.")
    require(type(p.get("control_seed")) is int and p["control_seed"] >= 0, "Invalid control seed.")
    require(p.get("expected_inpaint_self_condition") is False, "This adapter tests the observed unclamped self-conditioning configuration.")
    require(type(p.get("expected_final_inpaint")) is bool, "Specify the native final-inpaint setting.")
    for key in ["record_root", "head_dir", "output_dir"]:
        require(isinstance(p.get(key), str) and p[key], f"Missing {key}.")
    return p


def plan_trials(p):
    base = dict(call=p["intervention_call"], target_bond_class=p["focus"]["expected_bond_class"], reference_bond_class=0)
    # Every identity replay must pass before any active intervention.
    trials = [dict(id=f"sham_s{s}", kind="sham", seed=s, **base) for s in p["seeds"]]
    for s in p["seeds"]:
        trials.append(dict(id=f"targeted_s{s}", kind="targeted", seed=s, **base))
        trials.extend(dict(id=f"random{s}_r{r}", kind="random_direction", seed=s, control_index=r, **base)
                      for r in range(p["random_controls"]))
    require(len(trials) == 5 * len(p["seeds"]) and len({t["id"] for t in trials}) == len(trials), "Invalid trial enumeration.")
    return trials


def resolve_focus(task, focus):
    a, b = task["fixed_reference_indices"], task["fixed_sampler_indices"]
    require(task.get("mapping_verified") and len(a) == len(b) and len(set(a)) == len(a) and len(set(b)) == len(b), "Invalid fixed-atom mapping.")
    mapping = dict(zip(a, b))
    require(all(focus[k] in mapping for k in ["site_reference_index", "partner_reference_index"]), "Both focus atoms must belong to the fixed fragment.")
    return {"site": int(mapping[focus["site_reference_index"]]), "partner": int(mapping[focus["partner_reference_index"]])}


def stable_probabilities(logits):
    a = np.asarray(logits, dtype=np.float64)
    require(a.shape == (5,) and np.isfinite(a).all(), "Invalid bond-logit vector.")
    x = np.exp(a - a.max())
    return x / x.sum()


def build_change(before, fixed_input, kind, control_key):
    before, fixed_input = np.asarray(before), np.asarray(fixed_input)
    require(before.shape == fixed_input.shape == (2, 3) and before.dtype.kind == "f", "Expected two 3D floating-point coordinates.")
    require(np.isfinite(before).all() and np.isfinite(fixed_input).all(), "Nonfinite intervention coordinates.")
    require(kind in {"sham", "targeted", "random_direction"}, "Unknown intervention type.")
    b, x = before.astype(np.float64), fixed_input.astype(np.float64)
    desired_vector = x[1] - x[0]
    relative_delta = desired_vector - (b[1] - b[0])
    target_delta = np.stack([-relative_delta / 2, relative_delta / 2])
    target = (b + target_delta).astype(before.dtype)
    intended = float(np.linalg.norm(target_delta))
    actual_target = float(np.linalg.norm(target.astype(np.float64) - b))
    key_seed = int.from_bytes(hashlib.sha256(str(control_key).encode()).digest()[:8], "little")
    if kind == "sham":
        actual = before.copy()
        designed_delta = np.zeros((2, 3), dtype=np.float64)
    elif kind == "targeted":
        actual, designed_delta = target, target_delta
    elif intended == 0.0:
        actual, designed_delta = before.copy(), np.zeros((2, 3), dtype=np.float64)
    else:
        direction = relative_delta / np.linalg.norm(relative_delta)
        rng = np.random.default_rng(key_seed)
        # Bounded rejection only for numerical degeneracy, never for outcomes.
        for _ in range(16):
            v = rng.normal(size=3)
            v -= np.dot(v, direction) * direction
            length = np.linalg.norm(v)
            if length > 1e-10:
                break
        else:
            raise ValueError("Degenerate random control direction.")
        offset = v / length * np.linalg.norm(relative_delta)
        designed_delta = np.stack([-offset / 2, offset / 2])
        actual = (b + designed_delta).astype(before.dtype)
        require(abs(float(np.sum(designed_delta * target_delta))) <= 1e-10 * max(1.0, intended**2), "Control direction is not orthogonal.")
    delta = actual.astype(np.float64) - b
    tolerance = 16 * np.finfo(before.dtype).eps * (float(np.linalg.norm(b)) + intended + 1)
    require(np.isfinite(actual).all(), "Nonfinite replacement.")
    require(float(np.linalg.norm(delta.sum(axis=0))) <= tolerance, "Pair centroid changed beyond rounding.")
    if kind != "sham":
        require(abs(float(np.linalg.norm(delta)) - actual_target) <= tolerance, "Actual control magnitude differs from target beyond rounding.")
    if kind == "targeted":
        require(np.linalg.norm((actual[1].astype(float) - actual[0]) - desired_vector) <= tolerance, "Fixed relative vector was not restored.")
    info = dict(control=kind, control_seed=key_seed if kind == "random_direction" else None,
                delta_norm_model_units=float(np.linalg.norm(delta)), target_norm_model_units=actual_target,
                intended_norm_model_units=intended, rounding_bound_model_units=float(tolerance),
                centroid_error_model_units=float(np.linalg.norm(delta.mean(axis=0))),
                informative_magnitude=bool(actual_target > tolerance),
                actual_delta=delta.tolist(), designed_delta=designed_delta.tolist(),
                before=before.tolist(), after=actual.tolist(), fixed_input=fixed_input.tolist())
    return actual, info
