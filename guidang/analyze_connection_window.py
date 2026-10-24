#!/usr/bin/env python3
"""Observational analysis of the verified PTP1B pilot; never loads a generator.

Requires numpy and matplotlib. Reads the existing connection arrays and
verified hidden replays. This script is deliberately specific to this pilot,
not a universal carbon-valence detector or a causal localization method.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = (43, 47, 50)
CALLS = tuple(range(30, 41))
PARTNERS = {43: (22,), 47: (15, 17), 50: (25,)}
ORDER = np.array([0.0, 1.0, 2.0, 3.0, 1.5])
STAGES = [("L3", "layer_02__out__edges"),
          ("L6", "layer_05__out__edges"),
          ("L9", "layer_08__out__edges"),
          ("L12", "layer_11__out__edges"),
          ("Readout input", "bond_proj__in")]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    ex = np.exp(values - values.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        result = {k: source[k] for k in source.files}
    require(all(np.isfinite(v).all() for v in result.values()),
            f"Nonfinite array in {path}")
    return result


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def intervals(numbers):
    result = []
    for n in numbers:
        n = int(n)
        if result and n == result[-1][1] + 1:
            result[-1][1] = n
        else:
            result.append([n, n])
    return result


def distance(coords, i, j):
    return float(np.linalg.norm(coords[i].astype(float) - coords[j].astype(float)))


def load_and_verify(hidden_dir, connection_dir):
    replay = read_json(hidden_dir / "replay_summary.json")
    require(replay["all_requested_calls_verified"] and replay["error"] is None,
            "Replay did not complete successfully.")
    require(replay["seeds"] == list(SEEDS) and replay["calls"] == list(CALLS),
            "This analysis expects the specific three-seed, call-30-to-40 pilot.")
    require(replay["focus_fixed_sampler_indices"] == [0], "Expected fixed site 0.")
    require(replay["n_ligand_layers"] == 12, "Unexpected layer count.")
    records = replay["call_results"]
    require(len(records) == 33 and len({(r["seed"], r["call_index"]) for r in records}) == 33,
            "Missing or duplicated replay records.")
    comparisons = [r[k] for r in records for k in
                   ("replay_vs_original", "observation_vs_replay", "observation_vs_original")]
    require(all(r["accepted"] for r in records) and all(c["passed"] for c in comparisons),
            "At least one replay comparison failed.")
    require(all(r["layer_visits"] == {str(i): 1 for i in range(12)} for r in records),
            "Unexpected layer visits.")
    max_replay_error = max(f["max_abs_difference"] for c in comparisons for f in c["fields"].values())
    times = {(r["seed"], r["call_index"]): r["time"] for r in records}
    all_data = {}
    hashes = {str(hidden_dir / "replay_summary.json"): digest(hidden_dir / "replay_summary.json")}
    max_probability_error = 0.0
    max_projection_error = 0.0
    max_symmetrization_error = 0.0
    baseline_condition = None
    for seed in SEEDS:
        hd, cd = hidden_dir / f"seed_{seed}", connection_dir / f"seed_{seed}"
        m = read_json(hd / "metadata.json")
        report = read_json(cd / "connection_report.json")
        trace = load_npz(cd / "connection_arrays.npz")
        for path in [hd / "metadata.json", cd / "connection_report.json", cd / "connection_arrays.npz"]:
            hashes[str(path)] = digest(path)
        require(m["seed"] == seed and report["seed"] == seed, "Seed mismatch.")
        require(m["task_definition"] == report["task_definition"], "Task mapping mismatch.")
        task = m["task_definition"]
        require(task["mapping_verified"] and task["fixed_sampler_indices"] == list(range(9)),
                "Unverified or unexpected fixed atoms.")
        require(task["fixed_reference_indices"][0] == 5, "Unexpected reference atom mapping.")
        require(report["atom_vocabulary"][3] == "C", "Unexpected carbon token.")
        manifest = m["recorded_manifest"]
        for key in ("checkpoint_sha256", "input_sha256", "source_sha256", "args"):
            require(manifest[key] == report["manifest"][key], f"Manifest mismatch: {key}.")
        condition = {key: manifest[key] for key in ("checkpoint_sha256", "input_sha256", "source_sha256")}
        condition["args"] = {k: v for k, v in manifest["args"].items() if k not in ("seed", "save_dir")}
        if baseline_condition is None:
            baseline_condition = condition
        require(condition == baseline_condition, "Generation conditions differ beyond seed/output.")
        indices = trace["network_prediction_indices"]
        require(np.array_equal(indices, np.arange(101)), "Unexpected full-trajectory call indices.")
        require(np.array_equal(trace["fixed_sampler_indices"], np.arange(9)), "Fixed row indices differ.")
        require((trace["network_prediction_atom_classes"][:, 0] == 3).all(),
                "Site 0 is not carbon throughout the predictions.")
        require((trace["network_prediction_charge_classes"][:, 0] == 1).all(),
                "Site 0 is not neutral throughout this pilot (charge class 1).")
        require(trace["network_prediction_valid"].all(), "Unexpected masked atoms; adapt the analysis.")
        fixed_bonds = trace["network_prediction_bond_classes"][:, 0, :9]
        require((ORDER[fixed_bonds].sum(-1) == 3).all(), "Fixed internal bond orders changed.")
        final_external = np.flatnonzero(trace["raw_final_bond_classes"][0, 9:]) + 9
        require(tuple(final_external) == PARTNERS[seed], "Expected retrospective final partners differ.")
        hidden = {}
        for call in CALLS:
            path = hd / f"call_{call:04d}_hidden.npz"
            a = load_npz(path)
            hashes[str(path)] = digest(path)
            require(np.array_equal(a["tracked_atom_indices"], m["tracked_atom_indices"]), "Node order mismatch.")
            require(np.array_equal(a["focus_fixed_sampler_indices"], [0]), "Pair rows do not refer to site 0.")
            p = softmax(a["output_bond_logits"][:, 0, :])
            old_p = trace["network_prediction_bond_probabilities"][call, 0]
            max_probability_error = max(max_probability_error, float(np.max(abs(p - old_p))))
            require(np.allclose(p, old_p, atol=2e-6, rtol=2e-6), "Replay/trajectory probabilities differ.")
            require(np.array_equal(a["output_bond_logits"][:, 0].argmax(-1),
                                   trace["network_prediction_bond_classes"][call, 0]), "Bond class mismatch.")
            require(np.array_equal(a["output_atom_logits"].argmax(-1),
                                   trace["network_prediction_atom_classes"][call]), "Atom class mismatch.")
            # The diagonal is masked by the model after the projection; exclude self pairs.
            proj_error = np.max(abs(a["bond_proj__out__to_sites"][0, 1:] - a["output_bond_logits"][1:, 0]))
            max_projection_error = max(max_projection_error, float(proj_error))
            require(proj_error < 1e-5, "Projection observation does not match the network output.")
            sym = .5 * (a["bond_refine__out__from_sites"] + a["bond_refine__out__to_sites"])
            sym_error = float(np.max(abs(sym - a["bond_proj__in__from_sites"])))
            max_symmetrization_error = max(max_symmetrization_error, sym_error)
            require(sym_error < 1e-5, "Unexpected bond-refine to readout pathway.")
            hidden[call] = a
        all_data[seed] = {"metadata": m, "trace": trace, "report": report, "hidden": hidden}
    verification = {"replay_calls": 33, "replay_max_abs_error": max_replay_error,
                    "trajectory_probability_max_abs_error": max_probability_error,
                    "readout_output_max_abs_error": max_projection_error,
                    "symmetrization_max_abs_error": max_symmetrization_error,
                    "fixed_sampler_site": 0, "fixed_reference_site": 5,
                    "node_mapping_and_provenance": "passed",
                    "array_meanings_and_output_consistency": "passed"}
    return all_data, verification, hashes, times


def collect_tables(data, times):
    timeline, pairs, geometry, features = [], [], [], []
    for seed, d in data.items():
        trace, hidden, metadata = d["trace"], d["hidden"], d["metadata"]
        for call in range(101):
            classes = trace["network_prediction_bond_classes"][call, 0]
            ext = classes[9:]
            valence = float(ORDER[classes].sum())
            timeline.append({"seed": seed, "call_index_zero_based": call, "forward_number_one_based": call+1,
                             "site": 0, "predicted_bond_order_sum": valence,
                             "neutral_carbon_overvalence": valence > 4,
                             "n_generated_neighbors": int(np.count_nonzero(ext)),
                             "generated_neighbors": json.dumps((np.flatnonzero(ext)+9).tolist())})
        ids = metadata["tracked_atom_indices"]
        scale = float(metadata["coord_scale"])
        for call, a in hidden.items():
            probs = softmax(a["output_bond_logits"][:, 0])
            top = sorted(range(9, 27), key=lambda j: 1-probs[j, 0], reverse=True)
            for j in PARTNERS[seed]:
                margin = float(a["output_bond_logits"][j, 0, 1] - a["output_bond_logits"][j, 0, 0])
                row = {"seed": seed, "call_index_zero_based": call, "recorded_time": times[(seed, call)][0],
                       "site": 0, "generated_slot": j,
                       "model_probability_any_bond": float(1-probs[j, 0]),
                       "single_minus_no_bond_logit": margin,
                       "predicted_bond_class": int(probs[j].argmax()),
                       "input_noisy_distance_A": distance(a["input_coords"]*scale, 0, j),
                       "prediction_distance_A": distance(trace["network_prediction_coords"][call], 0, j),
                       "second_strongest_external_probability": float(1-probs[top[1], 0])}
                i0, ij = ids.index(0), ids.index(j)
                for layer in range(12):
                    before = a[f"layer_{layer:02d}__in__coords"] * scale
                    after = a[f"layer_{layer:02d}__out__coords"] * scale
                    geometry.append({"seed": seed, "call_index_zero_based": call, "site": 0,
                                     "generated_slot": j, "layer_number_one_based": layer+1,
                                     "distance_in_A": distance(before, i0, ij),
                                     "distance_out_A": distance(after, i0, ij),
                                     "distance_change_A": distance(after, i0, ij)-distance(before, i0, ij)})
                final_internal_distance = distance(a["layer_11__out__coords"]*scale, i0, ij)
                row["last_internal_coordinate_distance_A"] = final_internal_distance
                row["final_coordinate_output_distance_change_A"] = row["prediction_distance_A"]-final_internal_distance
                pairs.append(row)
            if call == CALLS[0]:
                continue
            prev = hidden[call-1]
            # Include all generated slots, not only the retrospectively selected final partners.
            for stage_name, key in STAGES:
                for j in range(9, 27):
                    x = np.concatenate([a[key+"__from_sites"][0, j], a[key+"__to_sites"][0, j]]).astype(float)
                    y = np.concatenate([prev[key+"__from_sites"][0, j], prev[key+"__to_sites"][0, j]]).astype(float)
                    denom = np.linalg.norm(x)+np.linalg.norm(y)
                    relative = float(2*np.linalg.norm(x-y)/denom) if denom else 0.0
                    features.append({"seed": seed, "call_from": call-1, "call_to": call,
                                     "site": 0, "generated_slot": j, "stage": stage_name,
                                     "symmetric_relative_l2_change": relative,
                                     "retrospective_final_partner": j in PARTNERS[seed]})
    return timeline, pairs, geometry, features


def make_figures(output, data, timeline, pairs, geometry, features):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 130, "savefig.dpi": 170})
    colors = {43: "#237a8b", 47: "#bd453c", 50: "#6172bd"}
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4), layout="constrained")
    for seed in SEEDS:
        rows = [r for r in timeline if r["seed"] == seed]
        axes[0, 0].plot([r["call_index_zero_based"] for r in rows],
                        [r["predicted_bond_order_sum"] for r in rows], label=f"seed {seed}",
                        color=colors[seed], lw=2, ls="--" if seed==50 else "-")
    axes[0, 0].axhline(4, color="#444", ls=":", lw=1)
    axes[0, 0].axvspan(30, 40, color="#dcc48a", alpha=.3)
    axes[0, 0].set(title="A. Fixed carbon 0: predicted total bond order", xlabel="Network call (zero-based)", ylabel="Bond-order sum", yticks=[3,4,5,6])
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=9)
    for j, color in [(15,"#bd453c"),(17,"#237a8b")]:
        r = [r for r in pairs if r["seed"]==47 and r["generated_slot"]==j]
        axes[0,1].plot(CALLS,[x["model_probability_any_bond"] for x in r], "o-",color=color,label=f"0-{j}")
        axes[1,0].plot(CALLS,[x["prediction_distance_A"] for x in r], "o-",color=color,label=f"seed 47: 0-{j}")
    for seed in (43,50):
        r=[r for r in pairs if r["seed"]==seed]
        axes[1,0].plot(CALLS,[x["prediction_distance_A"] for x in r],ls="--",color=colors[seed],label=f"seed {seed}: its own partner")
    axes[0,1].set(title="B. Two competing connections in seed 47",xlabel="Network call (zero-based)",ylabel="Model probability of any bond",ylim=(-.03,1.03))
    axes[0,1].legend(frameon=False)
    axes[1,0].set(title="C. Distances in each network prediction",xlabel="Network call (zero-based)",ylabel="Distance (angstrom)")
    axes[1,0].legend(frameon=False,fontsize=8)
    for seed in SEEDS:
        vals=[]
        for c in CALLS:
            p=softmax(data[seed]["hidden"][c]["output_bond_logits"][:,0])
            vals.append(sorted((1-p[j,0] for j in range(9,27)),reverse=True)[1])
        axes[1,1].plot(CALLS,vals,"o-",color=colors[seed],label=f"seed {seed}")
    axes[1,1].set(title="D. Second-strongest external connection",xlabel="Network call (zero-based)",ylabel="Model probability (log scale)",yscale="log",ylim=(1e-6,1.1))
    axes[1,1].legend(frameon=False)
    for ax in [axes[0,1],axes[1,0],axes[1,1]]:
        ax.axvline(34,color="#888",ls=":",lw=1)
        ax.axvline(38,color="#888",ls=":",lw=1)
        ax.set_xticks([30,32,34,36,38,40])
    fig.suptitle("Observed connection conflict | PTP1B pilot | no intervention",fontsize=14)
    fig.savefig(output/"01_connection_timeline.png")
    plt.close(fig)

    fig, axes=plt.subplots(1,3,figsize=(12,4.4),layout="constrained",sharey=True)
    positions=[0,1,2,3,4]
    for ax,c in zip(axes,[34,35,38]):
        for seed,j,color,label in [(47,15,"#bd453c","seed 47: 0-15"),(47,17,"#d89433","seed 47: 0-17"),
                                   (43,22,"#237a8b","seed 43: own partner"),(50,25,"#6172bd","seed 50: own partner")]:
            pp=next(r for r in pairs if r["seed"]==seed and r["generated_slot"]==j and r["call_index_zero_based"]==c)
            vals=[next(r["distance_out_A"] for r in geometry if r["seed"]==seed and r["generated_slot"]==j and r["call_index_zero_based"]==c and r["layer_number_one_based"]==l) for l in [3,6,9,12]]
            ax.plot(positions,vals+[pp["prediction_distance_A"]],"o-",color=color,label=label)
        ax.set(title=f"call {c}",xticks=positions,xticklabels=["L3","L6","L9","L12","Output"],xlabel="Coordinate observation stage")
    axes[0].set_ylabel("Pair distance (angstrom)")
    axes[0].legend(fontsize=8,frameon=False)
    fig.suptitle("Internal coordinate updates and final coordinate output\nInternal coordinates are not separate generated molecules",fontsize=12)
    fig.savefig(output/"02_coordinate_path.png")
    plt.close(fig)

    targets=[(47,15),(47,17),(43,22),(50,25)]
    matrices=[]
    for seed,j in targets:
        matrices.append(np.array([[next(r["symmetric_relative_l2_change"] for r in features
                                        if r["seed"]==seed and r["generated_slot"]==j and r["call_to"]==c and r["stage"]==stage)
                                   for c in range(31,41)] for stage,_ in STAGES]))
    vmax=max(float(x.max()) for x in matrices)
    fig,axes=plt.subplots(2,2,figsize=(11.5,6.5),layout="constrained")
    for ax,(seed,j),matrix in zip(axes.flat,targets,matrices):
        im=ax.imshow(matrix,aspect="auto",vmin=0,vmax=vmax,cmap="viridis")
        ax.set(title=f"seed {seed}: connection 0-{j}",xticks=range(10),xticklabels=range(31,41),
               yticks=range(5),yticklabels=[s for s,_ in STAGES],xlabel="Current call (compared with previous call)")
    fig.colorbar(im,ax=list(axes.flat),label="Relative feature change (not causal importance)",shrink=.8)
    fig.suptitle("Same-stage temporal changes | descriptive, not a key-layer ranking",fontsize=13)
    fig.savefig(output/"03_feature_changes.png")
    plt.close(fig)


def html_table(headers, rows):
    cells=lambda row,tag: "<tr>"+"".join(f"<{tag}>{html.escape(str(v))}</{tag}>" for v in row)+"</tr>"
    return "<table><thead>"+cells(headers,"th")+"</thead><tbody>"+"".join(cells(row,"td") for row in rows)+"</tbody></table>"


def make_report(output, verification, timeline, pairs, summary):
    table=[]
    for c in [33,34,35,36,37,38,39,40]:
        a=next(r for r in pairs if r["seed"]==47 and r["generated_slot"]==15 and r["call_index_zero_based"]==c)
        b=next(r for r in pairs if r["seed"]==47 and r["generated_slot"]==17 and r["call_index_zero_based"]==c)
        v=next(r["predicted_bond_order_sum"] for r in timeline if r["seed"]==47 and r["call_index_zero_based"]==c)
        table.append([c,f"{a['model_probability_any_bond']:.4f}",f"{b['model_probability_any_bond']:.4f}",
                      f"{a['prediction_distance_A']:.3f}",f"{b['prediction_distance_A']:.3f}",f"{v:.0f}"])
    def img(name):
        encoded=base64.b64encode((output/name).read_bytes()).decode("ascii")
        return f'<img src="data:image/png;base64,{encoded}" alt="{name}">'
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Scaffold hopping：连接异常的首轮过程分析</title>
<style>body{{font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;color:#20313a;max-width:1080px;margin:36px auto;padding:0 24px;background:#fbfcfd}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #d4dde2;text-align:left}}th{{background:#eaf0f3}}img{{width:100%;height:auto;margin:16px 0}}.note{{padding:16px 20px;background:#edf4f6;border-left:4px solid #237a8b}}.limit{{background:#fff5e6;padding:16px 20px}}code{{background:#eaf0f3;padding:2px 5px}}</style>
<h1>固定片段—新骨架连接：首轮观察分析</h1>
<p>FLOWR.ROOT / PTP1B / seed 47 异常案例，seed 43、50 对照。当前阶段：过程观察与候选机制形成。未训练模型，未实施内部干预，未开展 Shapley 输入归因。</p>
<div class="note"><strong>已证实：</strong>固定原子 0（预处理参考原子 5）的中性碳，在 call 34 首次出现输出超价；call 35–37 暂时恢复；call 38–100 持续超价。<br><strong>还未证实：</strong>哪一层、哪类内部信息是原因。不能把发生异常的时间直接叫作因果起点。</div>
<h2>1. 三项核对</h2>
<ol><li><strong>记录来源、编号与原子映射：</strong>同一条件和 checkpoint 的三条轨迹；固定位点映射一致；33 组回放记录完整。生成原子编号仅在各自轨迹内使用。</li>
<li><strong>数值与计算路径：</strong>回放与原记录最大差值 {verification['replay_max_abs_error']:.1g}；由新记录重新计算的概率与旧轨迹最大差值 {verification['trajectory_probability_max_abs_error']:.2g}；成键输出及对称化计算均已核对。</li>
<li><strong>解释边界：</strong>不把水分子距离未通过叫作连接失败；不把中间含噪状态当成最终合法分子；不把表征变化量当作因果重要性。33 个时刻记录不是 33 个独立样本。</li></ol>
<h2>2. 异常如何出现</h2>
<p>固定碳与固定片段内部的键级总和一直为 3。在这里，同时预测两条外连单键会使总键级达到 5，超过该中性碳允许的价态。判定不要求新骨架复原成参考结构。</p>
{html_table(['call（从0编号）','P(0–15成键)','P(0–17成键)','预测距离0–15 / Å','预测距离0–17 / Å','碳总键级'],table)}
<p>概率来自模型 logits 的 softmax；不代表已校准的物理成键概率。是否成键使用模型实际的最大类别决策，不另外用一个人为的 0.5 阈值筛选。</p>
{img('01_connection_timeline.png')}
<p>图 A：固定碳的输出总键级；图 B：seed 47 的两个候选连接；图 C：预测距离；图 D：每个时刻第二强的外连候选。对照中生成原子 22、25 只代表各自轨迹的连接对象，不与 seed 47 的 15、17 逐原子配对。</p>
<h2>3. 新线索：末端坐标输出与成键变化同步</h2>
<p>seed 47 的 0–15 距离在最后内部坐标与最终预测坐标之间：call 34 从约 2.425 Å 变为 1.840 Å；call 35 从约 2.175 Å 变为 2.609 Å；call 38 从约 2.301 Å 变为 1.893 Å。对应成键概率为 0.515、0.122、0.542。</p>
<p>这说明末端几何输出值得继续检查，但<strong>不能由同步变化推断因果</strong>。call 36 的距离也缩短到约 1.911 Å，成键概率为 0.4986，仍由“无键”类别略占优势，提示必须看连续分数和多个输入分支，不能只看二元标签。</p>
{img('02_coordinate_path.png')}
<p>源码中的末端路径为：坐标输出与节点表征、连接表征一起进入 <code>bond_refine</code>，再经对称化和 <code>bond_proj</code> 得到键类别分数。这个结构事实提供了后续区分“几何分支”和“表征分支”的位置；它本身不是故障定位结果。</p>
<h2>4. 逐层表征：有差异，但尚不能选定故障层</h2>
<p>下图计算同一条轨迹、同一阶段的相邻时刻特征变化，公式为 2‖h(t)−h(t−1)‖/(‖h(t)‖+‖h(t−1)‖)，双向连接特征拼接后计算。层编号从 1 开始。结果描述变化幅度，不衡量因果重要性。</p>
{img('03_feature_changes.png')}
<p>seed 47 的 0–15 在 L6 观察点最大变化出现在 call 35→36；call 36 尚未被输出为成键。因此“哪个变化最大，哪个就是故障来源”在这个案例上没有得到支持。只有 L3、L6、L9、L12 更新连接表征是模型结构安排，对照也一样。</p>
<h2>5. 当前候选解释与可检验预测</h2>
<p><strong>候选 H1：</strong>末端坐标输出对 0–15 的靠近，与既有节点/连接表征共同使该连接的成键分数超过无键分数，同时 0–17 继续保持较强成键支持。两条候选未得到足够协调，最终形成固定碳超价。</p>
<p><strong>竞争解释：</strong>变化主要来自节点或连接表征；几何变化只是伴随现象，或者只是接近决策边界的暂态波动。当前结果无法区分这些解释。</p>
<p>后续验证需要先写明位置（末端坐标输出至 bond_refine）、时间（包含异常、恢复、近边界的对照时刻）、对象（同一轨迹内的两条候选）、预期效果，再设计受控干预。不能直接把不同 seed 中相同编号的生成原子互换。单次前向分数变化也不能代替最终连接改善；后续需继续原采样流程、保持条件/随机性可比并重新评价终态。</p>
<h2>6. 生成与重建分开记账</h2>
<p>原始最终预测中的固定碳超价，关闭价态修复时不能得到合法分子；原流程修复后，该位点从 C 变成 S。生成阶段的连接冲突与修复阶段的固定片段破坏分别记录。seed 47 的 UFF 能量计算曾报参数错误，不能把 energy=false 解释为“已经测得能量过高”。本分析不使用它作为失败依据。</p>
<div class="limit"><strong>结论适用范围：</strong>同一个复合物上的探索性案例。对照按既有终态评价挑选，候选连接和窗口按已观察到的异常选择。未检验独立复合物、其他模型、全局输入原子贡献或干预效果；也未证明单一关键层。当前成果是可复查的过程证据与候选路径。</div>
<h2>7. 文件说明</h2>
<ul><li><code>trajectory_site0.csv</code>：三条完整轨迹的固定碳输出状态。</li><li><code>connection_window.csv</code>：关键窗口的模型概率、距离、最后坐标输出变化。</li><li><code>layer_geometry.csv</code>：已记录候选连接在 12 层前后的内部坐标距离。</li><li><code>temporal_feature_changes.csv</code>：全部 18 个生成槽位的连接特征变化，包含未在图中展示的候选。</li><li><code>observation_summary.json</code>：核对结果、输入 SHA256、范围限制与版本。</li></ul></html>'''
    (output/"REPORT.html").write_text(report,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-dir",type=Path,default=Path("outputs/sh_hidden_valence_window"))
    parser.add_argument("--connection-dir",type=Path,default=Path("outputs/sh_connections_ptp1b"))
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/sh_observation_ptp1b"))
    args=parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists; keep it and choose a new --output-dir.")
    data,verification,hashes,times=load_and_verify(args.hidden_dir,args.connection_dir)
    timeline,pairs,geometry,features=collect_tables(data,times)
    summary={"analysis_version":"2026-10-17.1", "mode":"observational_only",
             "verification":verification,"input_sha256":hashes,
             "script_sha256":digest(Path(__file__)),
             "versions":{"numpy":np.__version__,"matplotlib":matplotlib.__version__},
             "overvalence_intervals":{str(s):intervals(r["call_index_zero_based"] for r in timeline if r["seed"]==s and r["neutral_carbon_overvalence"]) for s in SEEDS},
             "new_findings":"Late coordinate-output changes accompany competing-bond scores; no unique causal layer established.",
             "scope_limits":["One complex; three outcome-selected seeds, not 33 independent examples.",
                             "Final partners and the time window were selected retrospectively.",
                             "Feature-change magnitude is not causal importance; no intervention has been run.",
                             "Internal coordinates and noisy sampler states are not final molecular structures.",
                             "Generated slot identities must not be equated between seeds.",
                             "Input-atom Shapley attribution and independent-case validation are not yet done."]}
    require(summary["overvalence_intervals"]=={"43":[],"47":[[34,34],[38,100]],"50":[]},
            "Pilot findings differ; inspect before using the case-specific narrative.")
    args.output_dir.mkdir(parents=True)
    for name,rows in [("trajectory_site0.csv",timeline),("connection_window.csv",pairs),
                      ("layer_geometry.csv",geometry),("temporal_feature_changes.csv",features)]:
        write_csv(args.output_dir/name,rows)
    make_figures(args.output_dir,data,timeline,pairs,geometry,features)
    write_json(args.output_dir/"observation_summary.json",summary)
    make_report(args.output_dir,verification,timeline,pairs,summary)
    print("Observation analysis completed. No model loaded, no intervention performed.")
    print("Report:", (args.output_dir/"REPORT.html").resolve())
    print("Overvalence intervals:",json.dumps(summary["overvalence_intervals"]))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
