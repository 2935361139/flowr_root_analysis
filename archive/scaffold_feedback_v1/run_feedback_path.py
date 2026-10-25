#!/usr/bin/env python3
"""One preregistered feedback-path test on saved native ROOT trajectories.

Run from the existing FLOWR.ROOT project. Identity replays come first; subsequent
rollouts change one feedback coordinate pair at one call and run the native
sampler to completion. No model training, later-state substitution, or bond edit.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import sys
import time
import traceback
import zipfile
from pathlib import Path
import numpy as np
import native_adapter as native
from feedback_core import validate_plan, plan_trials
from feedback_hook import FeedbackPatch
from record_adapter import prepare

VERSION = "2026-10-23.feedback-path.1"
ROOT = Path(__file__).resolve().parent
HELPER_SHA = "1d5b6cfef030f85d1b0df939d96b6402d64f68e9b2a8ef0b746c12236a84ff82"
QUALITY_SHA = "97eda26eca03e1f8dcb3d59b8b33430cbb855e87508eeee20f17905a3fe66f75"
require = native.require


def primary_metrics(rows, plan):
    chosen = [r for r in rows if r["call"] in plan["early_calls"]]
    require([r["call"] for r in chosen] == plan["early_calls"], "Missing primary observation window.")
    return dict(early_mean_relative_vector_error_A=float(np.mean([r["relative_vector_error_A"] for r in chosen])),
                early_mean_expected_bond_probability=float(np.mean([r["expected_bond_probability"] for r in chosen])),
                early_mean_absolute_length_error_A=float(np.mean([abs(r["network_distance_A"] - r["input_distance_A"]) for r in chosen])))


def result_row(r, sham):
    t, e, m = r["trial"], r["evaluation"], r["primary_metrics"]
    raw = e["stages"][0]
    info = r["replay"]["patch"]
    return dict(trial=t["id"], seed=t["seed"], kind=t["kind"], control_index=t.get("control_index"),
                site=t["site"], partner=t["partner"], call=t["call"],
                **m,
                delta_error_vs_sham_A=m["early_mean_relative_vector_error_A"] - sham["primary_metrics"]["early_mean_relative_vector_error_A"],
                delta_expected_probability_vs_sham=m["early_mean_expected_bond_probability"] - sham["primary_metrics"]["early_mean_expected_bond_probability"],
                raw_chemical_endpoint=e["raw_valid_connected_root_fragment"],
                root_fragment_match=raw["root_fixed_substructure_match"],
                slot_fixed_chemistry_preserved=raw["fixed_chemistry_preserved"],
                raw_valid_connected=raw["fc_valid"], raw_posebusters_valid=raw["posebusters_valid"],
                raw_scaffold_changed=raw.get("murcko_scaffold_changed"),
                final_focus_bond_class=e["final_graph"]["bond_class"],
                final_site_bond_order_sum=e["final_graph"]["site_explicit_bond_order_sum"],
                intervention_norm_A=info["delta_norm_A"], informative_magnitude=info["informative_magnitude"],
                evaluation_errors=[s["errors"] for s in e["stages"]], elapsed_seconds=r["elapsed_seconds"])


def summaries(out, trials, results, summary, plot=False):
    rows = [result_row(results[t["id"]], results[f"sham_s{t['seed']}"])
            for t in trials if t["id"] in results and f"sham_s{t['seed']}" in results]
    if rows:
        native.write_csv(out / "feedback_results.csv", rows)
    completed_seeds = []
    for seed in summary["plan"]["seeds"]:
        own = [r for r in rows if r["seed"] == seed]
        if len(own) != 5:
            continue
        base = next(r for r in own if r["kind"] == "sham")
        target = next(r for r in own if r["kind"] == "targeted")
        controls = [r for r in own if r["kind"] == "random_direction"]
        completed_seeds.append(dict(seed=seed,
            baseline_chemical_endpoint=base["raw_chemical_endpoint"], target_chemical_endpoint=target["raw_chemical_endpoint"],
            baseline_slot_preserved=base["slot_fixed_chemistry_preserved"], target_slot_preserved=target["slot_fixed_chemistry_preserved"],
            target_delta_error_A=target["delta_error_vs_sham_A"],
            random_delta_error_A=[r["delta_error_vs_sham_A"] for r in controls],
            target_delta_expected_probability=target["delta_expected_probability_vs_sham"],
            random_delta_expected_probability=[r["delta_expected_probability_vs_sham"] for r in controls],
            random_chemical_endpoints=[r["raw_chemical_endpoint"] for r in controls],
            target_raw_posebusters_valid=target["raw_posebusters_valid"],
            interpretation="Descriptive paired effects in one complex, not a root-cause label or independent-sample success rate."))
    native.write_json(out / "paired_effects.json", completed_seeds)
    if plot and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        available = [s for s in summary["plan"]["seeds"] if f"sham_s{s}" in results]
        fig, axes = plt.subplots(len(available), 2, figsize=(11, max(3, 2.1 * len(available))), squeeze=False, layout="constrained")
        for i, seed in enumerate(available):
            for r in rows:
                if r["seed"] != seed:
                    continue
                timeline = native.read_json(out / "trials" / r["trial"] / "feedback_timeline.json")
                color = {"sham": "#444444", "targeted": "#007A6E", "random_direction": "#CA8122"}[r["kind"]]
                label = r["kind"] if r["kind"] != "random_direction" else f"random {r['control_index']}"
                early = [x for x in timeline if x["call"] <= max(summary["plan"]["early_calls"])]
                axes[i, 0].plot([x["call"] for x in early], [x["relative_vector_error_A"] for x in early], color=color, alpha=.65 if r["kind"] == "random_direction" else 1, label=label)
                axes[i, 1].plot([x["call"] for x in timeline], [x["expected_bond_probability"] for x in timeline], color=color, alpha=.65 if r["kind"] == "random_direction" else 1)
            axes[i, 0].set(ylabel=f"seed {seed}: error (A)", xlabel="Network call (zero-based)")
            axes[i, 1].set(ylabel="P(expected bond)", xlabel="Network call (zero-based)", ylim=(-.02, 1.02))
        axes[0, 0].set_title("Early relative-vector error; lower is closer to fixed input")
        axes[0, 1].set_title("Full trajectory; terminal validity evaluated separately")
        axes[0, 0].legend(fontsize=7)
        fig.savefig(out / "feedback_trajectories.png", dpi=140)
        plt.close(fig)
    note = f"""# 固定片段反馈路径实验\n\n执行完成：{summary.get('completed', False)}；{len(results)}/{len(trials)} 条续跑。\n错误：{summary.get('error')}\n\n这次只在一个调用中改变反馈中的一个固定原子对的相对位置。主生成状态、其它反馈字段和模型权重保持原样，之后由原生采样器继续。原样回放必须复现全部网络输出和终态；每条续跑检查随机流一致性。\n\nfeedback_results.csv 中：delta_error_vs_sham_A < 0 表示早期相对位置误差减小；delta_expected_probability_vs_sham > 0 表示早期预期键概率提高。它们是连续观察量，不是终态成败阈值。informative_magnitude=False 表示本次修改小到不足以支持幅度比较。\n\n最终分别报告：原始解码化学有效且连通并通过 ROOT 固定片段匹配；原槽位固定化学身份；PoseBusters；骨架是否变化。严格外连键模式仅作诊断，不要求新骨架恢复参考结构。\n\n先看同种子的定向修改与原样回放，再与三个等幅随机方向对照比较。早期改善但终态未修复，只能支持有限的过程作用；定向修改无优势，不支持本次特定干预，不能因此排除所有反馈作用。即使定向组修复，也不能写成唯一根因。\n\n所有种子来自一个复合物；种子、随机对照和逐步记录不是独立复合物。后续需要冻结方案并在未用于选择方案的复合物上检验。同一程序可通过配置和固定原子映射复用，不意味着已经证明机制泛化。\n\n执行结束后保留结果，不自动扫描其它调用、换原子对或加大修改幅度。\n"""
    (out / "RESULTS_CN.md").write_text(note, encoding="utf-8")
    fields = ["seed", "kind", "control_index", "delta_error_vs_sham_A", "delta_expected_probability_vs_sham",
              "raw_chemical_endpoint", "slot_fixed_chemistry_preserved", "raw_posebusters_valid", "final_focus_bond_class"]
    def show(v):
        return f"{v:.6g}" if isinstance(v, float) else str(v)
    table = "<tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in fields) + "</tr>"
    table += "".join("<tr>" + "".join(f"<td>{html.escape(show(r.get(k)))}</td>" for k in fields) + "</tr>" for r in rows)
    (out / "REPORT.html").write_text("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>固定片段反馈路径实验</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 18px;line-height:1.7;color:#18252d}pre{white-space:pre-wrap;font:inherit}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #d4dce1}th{background:#e6f0ef}img{width:100%}</style><h1>固定片段反馈路径实验</h1><pre>" + html.escape(note) + "</pre><div style='overflow:auto'><table>" + table + "</table></div>" + ("<img src='feedback_trajectories.png' alt='All paired trajectories'>" if (out / "feedback_trajectories.png").is_file() else "") + "</html>", encoding="utf-8")


def bundle(out):
    # Preserve large tensors locally; the review ZIP contains auditable compact data.
    temp = out / "feedback_review.tmp"
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            rel = p.relative_to(out)
            if p.is_file() and p.name != "feedback_review.zip" and p.suffix not in {".pt", ".tmp", ".pyc"} and "model_context" not in rel.parts and "__pycache__" not in rel.parts:
                z.write(p, rel)
    temp.replace(out / "feedback_review.zip")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--plan", type=Path, default=ROOT / "bace_feedback_plan.json")
    ap.add_argument("--record-root", type=Path)
    ap.add_argument("--head-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--list-plan", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    plan = validate_plan(native.read_json(args.plan))
    trials = plan_trials(plan)
    if args.list_plan:
        print(json.dumps(dict(total=len(trials), sham=len(plan["seeds"]), targeted=len(plan["seeds"]), random_controls=3 * len(plan["seeds"]), intervention_call=plan["intervention_call"], training=False), indent=2))
        return 0
    for attr, key in [("record_root", "record_root"), ("head_dir", "head_dir"), ("output_dir", "output_dir")]:
        setattr(args, attr, (getattr(args, attr) or Path(plan[key])).resolve())
    out = args.output_dir
    if out.exists() and not args.resume:
        ap.error("Output exists. Preserve it and use --resume for the same unchanged run, or choose a new --output-dir.")
    if args.resume and not (out / "run_identity.json").is_file():
        ap.error("No complete run identity for resume. Preserve this directory and choose a new --output-dir.")
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    summary = dict(version=VERSION, plan=plan, planned_trials=len(trials), completed=False, completed_trials=0,
                   error=None, scope="One-complex feedback intervention; no unique-root-cause or generalization claim.")
    return_status = 1
    try:
        require((Path.cwd() / "flowr/scriptutil.py").is_file(), "Run from /home/ljl/projects/flowr_root with the original environment active.")
        sys.path.insert(0, str(Path.cwd()))
        import torch
        from flowr.scriptutil import load_model
        from flowr.util.device import resolve_device
        helpers = native.import_file(ROOT / "adapters/replay.py", "feedback_replay_helpers", HELPER_SHA)
        quality = native.import_file(ROOT / "adapters/quality.py", "feedback_quality_helpers", QUALITY_SHA)
        print("Checking original inputs, atom mapping, native sources and recorded feedback ...", flush=True)
        data, head, identity = prepare(args, plan, helpers, quality, torch)
        for t in trials:
            t.update(data[t["seed"]]["focus"])
        identity.update(version=VERSION, code_sha256={str(p.relative_to(ROOT)): native.digest(p) for p in ROOT.rglob("*.py")})
        if args.resume:
            require(native.read_json(out / "run_identity.json") == identity, "Resume identity changed. Use a new output directory; do not reuse incompatible results.")
        else:
            native.write_json(out / "run_identity.json", identity)
        for p in ROOT.rglob("*.py"):
            dest = out / "program" / p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
        native.write_json(out / "selected_plan.json", plan)
        native.write_json(out / "planned_trials.json", trials)
        native.write_json(out / "feedback_summary.json", summary)
        torch.set_float32_matmul_precision(identity["native_precision"])
        ma = argparse.Namespace(**data[plan["seeds"][0]]["manifest"]["args"])
        ma.save_dir = str(out / "model_context")
        device = resolve_device(ma)
        print(f"Loading recorded ROOT on {device}; {len(trials)} full native rollouts.", flush=True)
        model = load_model(ma)[0].to(device).eval()
        require(sum(p.requires_grad for p in model.parameters()) == head["parameter_tensors_requiring_grad"], "Native parameter flags changed.")
        require(bool(model._inpaint_self_condition) == plan["expected_inpaint_self_condition"], "Self-conditioning inpainting differs from the observed configuration.")
        for d in data.values():
            require(bool(model.self_condition) == d["inputs"]["self_condition"] and int(model.sc_recycle_steps) == d["inputs"]["sc_recycle_steps"], "Self-conditioning settings changed.")
            require(float(model.coord_scale) == d["inputs"]["coord_scale"], "Coordinate units changed.")
        for n, t in enumerate(trials):
            summary["active_trial"] = t["id"]
            native.write_json(out / "feedback_summary.json", summary)
            folder = out / "trials" / t["id"]
            folder.mkdir(parents=True, exist_ok=True)
            done = folder / "result.json"
            if args.resume and done.is_file():
                saved = native.read_json(done)
                require(saved["completed"] and saved["trial"] == t, "Resume trial changed.")
                for name, sha in saved["file_sha256"].items():
                    require(native.digest(folder / name) == sha, f"Resume output changed: {name}")
                results[t["id"]] = saved
                print(f"[{n + 1}/{len(trials)}] Reuse verified {t['id']}", flush=True)
                continue
            reference = None
            if t["kind"] != "sham":
                require(all(f"sham_s{s}" in results and results[f"sham_s{s}"]["replay"]["baseline_or_sham_full_match"] for s in plan["seeds"]), "All same-seed identity replays must pass first.")
                with np.load(out / "trials" / f"sham_s{t['seed']}" / "intervention_source.npz", allow_pickle=False) as a:
                    reference = {key: a[key].copy() for key in a.files}
            print(f"[{n + 1}/{len(trials)}] {t['id']}", flush=True)
            started = time.monotonic()
            native.write_json(folder / "trial.json", t)
            captured = {}
            def factory(m, trial, clock, torch_module):
                observer = FeedbackPatch(m, trial, clock, torch_module, plan, data[t["seed"]]["inputs"]["coord_scale"], reference)
                captured["observer"] = observer
                return observer
            final, replay, logs, sampler = native.run_rollout(model, data[t["seed"]], t, factory, helpers, device, torch, debug_dir=folder)
            observer = captured["observer"]
            np.savez_compressed(folder / "intervention_source.npz", **observer.capture)
            native.write_json(folder / "feedback_timeline.json", observer.rows)
            native.write_json(folder / "network_timeline.json", logs)
            native.write_json(folder / "sampler_timeline.json", sampler)
            native.write_json(folder / "replay_checks.json", replay)
            torch.save(helpers.move_copy(final, "cpu", torch), folder / "final_prediction.pt")
            evaluation = native.evaluate_final(final, model, data[t["seed"]], t, folder, quality, torch)
            saved = dict(completed=True, trial=t, replay=replay, evaluation=evaluation,
                         primary_metrics=primary_metrics(observer.rows, plan), elapsed_seconds=round(time.monotonic() - started, 3),
                         file_sha256={p.name: native.digest(p) for p in folder.iterdir() if p.is_file() and p.name != "result.json" and p.suffix != ".tmp"})
            native.write_json(done, saved)
            results[t["id"]] = saved
            summary["completed_trials"] = len(results)
            native.write_json(out / "feedback_summary.json", summary)
            summaries(out, trials, results, summary)
            print(f"  {saved['elapsed_seconds']:.1f}s; chemical endpoint: {evaluation['raw_valid_connected_root_fragment']}; early relative-vector error: {saved['primary_metrics']['early_mean_relative_vector_error_A']:.5f} A", flush=True)
            del final
        summary.update(completed=len(results) == len(trials), active_trial=None,
                       evaluation_errors=[dict(trial=k, stage=s["stage"], errors=s["errors"]) for k, r in results.items() for s in r["evaluation"]["stages"] if s["errors"]])
        summary["evaluation_complete"] = not summary["evaluation_errors"]
        return_status = 0 if summary["completed"] else 1
    except KeyboardInterrupt:
        summary["error"] = "Interrupted. Completed trials are retained; use the same command with --resume."
        return_status = 130
        print("STOP: " + summary["error"], flush=True)
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("STOP: " + summary["error"], flush=True)
    finally:
        summary["completed_trials"] = len(results)
        native.write_json(out / "feedback_summary.json", summary)
        try:
            summaries(out, trials, results, summary, plot=bool(results))
        except Exception:
            (out / "report_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        bundle(out)
        print(f"Completed: {summary['completed']}; {len(results)}/{len(trials)}", flush=True)
        print(f"Review file: {out / 'feedback_review.zip'}", flush=True)
    return return_status


if __name__ == "__main__":
    raise SystemExit(main())
