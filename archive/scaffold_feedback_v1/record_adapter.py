"""Validate recorded ROOT inputs and preserve their native generation identity."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import native_adapter as native
from feedback_core import resolve_focus

INTEGRATOR_SHA = "e9abd83167f708da4712db2ac49ab24ac44a39c59c46a54c20789c7873c58000"
require = native.require


def prepare(args, plan, helpers, quality, torch):
    from rdkit import Chem, rdBase
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from flowr.data.interpolate import extract_scaffolds
    head = native.read_json(args.head_dir / "summary.json")
    require(head["capture_completed"] and head["analysis_completed"] and head["error"] is None, "Incomplete head reference; use the previously successful sh_head_bace_lig16_v3 directory.")
    require(head["plan"]["case"] == plan["case"], "Head reference belongs to another case.")
    require(native.digest(args.head_dir / "observer_source.py") == head["script_sha256"], "Head reference source changed.")
    for e in head["states"]:
        require(e["accepted"] and e["numerics"]["passed"] and native.digest(args.head_dir / e["file"]) == e["sha256"], "Head reference capture is missing, changed, or unverified.")
    sources = {**head["source_sha256"], **quality.SOURCE_HASHES,
               **helpers.HOOK_SOURCE_HASHES, "flowr/models/integrator.py": INTEGRATOR_SHA}
    for name, sha in sources.items():
        require(native.digest(Path.cwd() / name) == sha, f"Native source changed: {name}. Do not relax this check.")
    data = {}
    load = lambda p: torch.load(p, map_location="cpu", weights_only=True)
    required_calls = {plan["intervention_call"], *plan["early_calls"]}
    for seed in plan["seeds"]:
        folder = args.record_root / f"sh_record_{plan['case']}_seed{seed}/research_record/attempt_000"
        manifest = native.read_json(folder.parent / "manifest.json")
        task = native.read_json(folder / "task_definition.json")
        require(str(seed) in head["metadata"] and manifest == head["metadata"][str(seed)]["manifest"], f"Manifest differs for seed {seed}.")
        require(task == head["metadata"][str(seed)]["task"] and task["mapping_verified"], "Fixed-fragment mapping changed.")
        focus = resolve_focus(task, plan["focus"])
        require(focus["site"] == head["plan"]["site"], "Head reference and intervention attachment atom differ.")
        require(manifest["torch_version"] == str(torch.__version__) and manifest["rdkit_version"] == rdBase.rdkitVersion, "Use the original Torch/RDKit environment.")
        for name, sha in manifest["input_sha256"].items():
            require(native.digest(Path(name)) == sha, f"Input changed or moved: {name}")
        cfg = manifest["args"]
        inputs = load(folder / "inputs.pt")
        attempt = native.read_json(folder / "attempt.json")
        require(cfg["seed"] == seed and cfg["scaffold_hopping"] and attempt["generation_completed"], "Incomplete/wrong scaffold-hopping record.")
        require(cfg["guidance_config"] is None and not cfg.get("optimize_gen_ligs") and not cfg.get("optimize_gen_ligs_hs"), "Guidance/optimization is outside this recorded protocol.")
        require(inputs["solver"] == cfg["solver"] and inputs["strategy"] == cfg["ode_sampling_strategy"]
                and inputs["final_inpaint"] == cfg["final_inpaint"] == plan["expected_final_inpaint"], "Sampler settings changed.")
        require(inputs["self_condition"] and int(inputs["sc_recycle_steps"]) == 1, "This protocol requires the recorded one-forward-per-step self-conditioning mode.")
        require(inputs["prior"]["mask"].shape[0] == 1, "Expected batch-one records.")
        valid = native.array(inputs["prior"]["mask"][0]).astype(bool)
        require(all(0 <= i < len(valid) and valid[i] for i in focus.values()), "Mapped focus atom is inactive.")
        reference = quality.read_one_sdf(folder / "reference_preprocessed.sdf", Chem)
        require(reference is not None, "Missing reference molecule.")
        Chem.SanitizeMol(reference)
        require(extract_scaffolds([reference], invert_mask=True)[0].nonzero(as_tuple=True)[0].tolist() == task["fixed_reference_indices"], "Native fixed-fragment extraction changed.")
        bond = reference.GetBondBetweenAtoms(plan["focus"]["site_reference_index"], plan["focus"]["partner_reference_index"])
        cls = None if bond is None else {1.0: 1, 2.0: 2, 3.0: 3, 1.5: 4}.get(bond.GetBondTypeAsDouble())
        require(cls == plan["focus"]["expected_bond_class"], "Expected bond class does not match the fixed reference.")
        attachment_refs = {a["fixed_reference_index"] for a in task["reference_attachment_sites"]}
        require(plan["focus"]["site_reference_index"] in attachment_refs, "Focus site must be a reference scaffold attachment atom.")
        paths = sorted((folder / "forward_calls").glob("call_*.pt"))
        require(len(paths) == attempt["forward_calls"], "Missing recorded forward calls.")
        calls = [load(p) for p in paths]
        require([c["call_index"] for c in calls] == list(range(len(calls))) and max(required_calls) < len(calls), "Requested call is absent.")
        s, j = focus["site"], focus["partner"]
        fixed_coords = native.array(inputs["prior"]["coords"][0])[[s, j]]
        for k in required_calls | {0}:
            c = calls[k]
            require(c["self_condition"] is not None, "Missing self-conditioning record.")
            x = native.array(c["state_before_forward"]["coords"][0])[[s, j]]
            require(np.allclose(x, fixed_coords, atol=1e-6, rtol=1e-6), "Focus input is not fixed at the requested time.")
        # The causal question is about a real, existing previous-output feedback edge.
        c, prev = calls[plan["intervention_call"]], calls[plan["intervention_call"] - 1]
        require(np.array_equal(native.array(c["self_condition"]["coords"]), native.array(prev["network_output_before_postprocessing"]["coords"])), "Recorded feedback is not the preceding raw coordinate prediction.")
        identity_files = [folder / "inputs.pt", folder / "pocket_encoding.pt", folder / "raw_final_prediction.pt",
                          folder / "task_definition.json", folder / "reference_preprocessed.sdf", *paths]
        data[seed] = dict(folder=folder, manifest=manifest, task=task, focus=focus, inputs=inputs, calls=calls,
                          encoding=load(folder / "pocket_encoding.pt"), final=load(folder / "raw_final_prediction.pt"),
                          reference=reference, reference_scaffold=Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(reference), isomericSmiles=False),
                          input_hashes={str(p.relative_to(folder)): native.digest(p) for p in identity_files})
    base = data[plan["seeds"][0]]
    for d in data.values():
        require(helpers.condition_key(d["manifest"]) == helpers.condition_key(base["manifest"]), "Seeds have different conditions beyond random seed/output path.")
        require(d["task"]["fixed_reference_indices"] == base["task"]["fixed_reference_indices"]
                and d["task"]["fixed_sampler_indices"] == base["task"]["fixed_sampler_indices"], "Fixed identities differ across seeds.")
    require(native.digest(Path(base["manifest"]["args"]["ckpt_path"])) == head["checkpoint_sha256"], "Checkpoint changed.")
    precision = {e["numerics"]["original_precision"] for e in head["states"]}
    require(len(precision) == 1, "Mixed native precision settings.")
    identity = dict(plan=plan, head_summary_sha256=native.digest(args.head_dir / "summary.json"),
                    source_sha256=sources, checkpoint_sha256=head["checkpoint_sha256"],
                    record_root=str(args.record_root.resolve()), head_dir=str(args.head_dir.resolve()),
                    native_precision=next(iter(precision)),
                    input_sha256={str(s): d["input_hashes"] for s, d in data.items()},
                    mapped_focus={str(s): d["focus"] for s, d in data.items()})
    return data, head, identity
