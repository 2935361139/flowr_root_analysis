"""One feedback-coordinate write and passive observations of the real rollout."""
from __future__ import annotations
import numpy as np
import native_adapter as native
from feedback_core import build_change, stable_probabilities


class FeedbackPatch:
    def __init__(self, model, trial, clock, torch, plan, scale, reference=None):
        self.trial, self.clock, self.torch = trial, clock, torch
        self.plan, self.scale, self.reference = plan, float(scale), reference
        self.visits = 0
        self.receipt = None
        self.rows = []
        self.pending = None
        self.capture = None
        self.handles = []
        try:
            # Registered after the adapter's pre-hook: unchanged input is verified first.
            self.handles.append(model.register_forward_pre_hook(self.pre, with_kwargs=True))
            self.handles.append(model.register_forward_hook(self.post, with_kwargs=True))
        except BaseException:
            self.close()
            raise

    def pre(self, module, args, kwargs):
        t, k = self.trial, self.clock["call"]
        batch = args[0] if args else kwargs["batch"]
        cond = kwargs.get("cond_batch")
        native.require(cond is not None and "coords" in cond, "No coordinate feedback at this forward.")
        s, j = t["site"], t["partner"]
        native.require(cond["coords"].shape == batch["coords"].shape and cond["coords"].shape[0] == 1, "Unexpected feedback shape.")
        x = native.array(batch["coords"][0])[[s, j]]
        b = native.array(cond["coords"][0])[[s, j]]
        actual = b.copy()
        changed_kwargs = None
        if k == t["call"]:
            self.visits += 1
            native.require(self.visits == 1, "Intervention was visited more than once.")
            if self.reference is not None:
                native.require(np.array_equal(b, self.reference["feedback_pair"]) and np.array_equal(x, self.reference["input_pair"]), "Pre-intervention state differs from the same-seed sham.")
            self.capture = {"feedback_pair": b.copy(), "input_pair": x.copy()}
            key = f"{self.plan['control_seed']}:{self.plan['case']}:{t['seed']}:{k}:{t.get('control_index', -1)}"
            actual, info = build_change(b, x, t["kind"], key)
            # Clone only the field being changed; no mutation of prior, main state,
            # other feedback fields, recorded source states, or future states.
            new_coords = cond["coords"].clone()
            for n, idx in enumerate([s, j]):
                new_coords[0, idx] = self.torch.as_tensor(actual[n], dtype=new_coords.dtype, device=new_coords.device)
            before_all, after_all = native.array(cond["coords"]), native.array(new_coords)
            allowed = np.zeros(before_all.shape[:2], dtype=bool)
            allowed[0, [s, j]] = True
            native.require(np.array_equal(before_all[~allowed], after_all[~allowed]), "Unselected feedback coordinates changed.")
            native.require(np.array_equal(after_all[0, [s, j]], actual), "Native tensor replacement differs from planned values.")
            new_cond = dict(cond)
            new_cond["coords"] = new_coords
            native.require(all(new_cond[key] is value for key, value in cond.items() if key != "coords"), "Another feedback field was replaced.")
            changed_kwargs = dict(kwargs)
            changed_kwargs["cond_batch"] = new_cond
            self.receipt = dict(visits=1, applied=bool(np.any(actual != b)), kind=t["kind"], call=k,
                                selected_slots=[s, j], prewrite_matches_sham=self.reference is not None,
                                only_selected_feedback_coordinates_changed=True,
                                delta_norm_A=info["delta_norm_model_units"] * self.scale, **info)
        self.pending = dict(call=k, input_pair_A=(x.astype(float) * self.scale).tolist(),
                            feedback_pair_before_A=(b.astype(float) * self.scale).tolist(),
                            feedback_pair_used_A=(actual.astype(float) * self.scale).tolist(),
                            input_distance_A=float(np.linalg.norm(x[1].astype(float) - x[0]) * self.scale),
                            feedback_distance_before_A=float(np.linalg.norm(b[1].astype(float) - b[0]) * self.scale),
                            feedback_distance_used_A=float(np.linalg.norm(actual[1].astype(float) - actual[0]) * self.scale))
        return (args, changed_kwargs) if changed_kwargs is not None else None

    def post(self, module, args, kwargs, out):
        t = self.trial
        native.require(self.pending is not None and self.pending["call"] == self.clock["call"], "Observation call mismatch.")
        s, j = t["site"], t["partner"]
        coords = native.array(out["coords"][0])[[s, j]].astype(float) * self.scale
        logits = native.array(out["bonds"][0, max(s, j), min(s, j)])
        probs = stable_probabilities(logits)
        fixed = np.asarray(self.pending["input_pair_A"])
        row = {**self.pending, "network_pair_A": coords.tolist(),
               "network_distance_A": float(np.linalg.norm(coords[1] - coords[0])),
               "relative_vector_error_A": float(np.linalg.norm((coords[1] - coords[0]) - (fixed[1] - fixed[0]))),
               "bond_class": int(np.argmax(logits)), "expected_bond_probability": float(probs[t["target_bond_class"]]),
               "no_bond_probability": float(probs[0]), "single_bond_probability": float(probs[1]),
               "expected_vs_no_bond_score": float(logits[t["target_bond_class"]] - logits[0]),
               "site_atom_class": int(native.array(out["atomics"][0, s]).argmax()),
               "partner_atom_class": int(native.array(out["atomics"][0, j]).argmax())}
        self.rows.append(row)
        self.pending = None
        return None

    def validate_complete(self):
        native.require(self.visits == 1 and self.receipt is not None and self.capture is not None, "Intervention did not execute exactly once.")
        native.require([r["call"] for r in self.rows] == list(range(self.clock["call"] + 1)), "Incomplete observations.")
        if self.trial["kind"] == "sham":
            native.require(not self.receipt["applied"] and self.receipt["delta_norm_model_units"] == 0, "Sham is not an identity operation.")

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
