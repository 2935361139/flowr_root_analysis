#!/usr/bin/env python3
"""Offline coordinate-source analysis of verified ROOT observations.

Reads the case/site/calls from a completed head-v2 summary. Offline mode does
not load a generator. Optional --capture-readout replays recorded inputs with
read-only hooks and captures all atom slots, without changing features.
Earlier missing coordinate coverage is explicit.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import importlib.util
import json
import re
import sys
import traceback
import zipfile
from pathlib import Path

import numpy as np


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stage_coordinates(hidden, observed, metadata, site, partner, n_layers):
    """Return (stage, fixed/generated atom, xyz), in angstroms, or uncovered."""
    tracked = [int(i) for i in hidden["tracked_atom_indices"]]
    require(tracked == metadata["tracked_atom_indices"] and len(set(tracked)) == len(tracked),
            "Hidden tracked-atom mapping differs from metadata.")
    require(site in metadata["task_definition"]["fixed_sampler_indices"], "Requested site is not fixed.")
    require(partner not in metadata["task_definition"]["fixed_sampler_indices"], "Partner is not generated.")
    if site not in tracked or partner not in tracked:
        return None
    indices = [tracked.index(site), tracked.index(partner)]
    scale = float(metadata["coord_scale"])
    require(np.isfinite(scale) and scale > 0, "Invalid coordinate scale.")
    initial = hidden["input_coords"][[site, partner]].astype(np.float64)
    previous = initial
    stages = [initial]
    for layer in range(n_layers):
        entry = hidden[f"layer_{layer:02d}__in__coords"][indices].astype(np.float64)
        output = hidden[f"layer_{layer:02d}__out__coords"][indices].astype(np.float64)
        require(np.array_equal(entry, previous), "Coordinate continuity fails between recorded layers.")
        require(entry.shape == (2, 3) and output.shape == (2, 3), "Unexpected coordinate shape.")
        stages.append(output)
        previous = output
    stages.append(observed["head_coords"][[site, partner]].astype(np.float64))
    result = np.stack(stages) * scale
    require(np.isfinite(result).all(), "Nonfinite recorded coordinates.")
    return result


def distance_ledger(before, after):
    """Exact accounting of recorded squared-distance changes, NOT causality.

    r = sum(component vectors); |r_b|^2-|r_a|^2 =
    (r_b+r_a) dot sum(component_b-component_a).
    Components are initial relative coordinates and successive recorded updates.
    """
    a, b = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    require(a.shape == b.shape and a.ndim == 3 and a.shape[1:] == (2, 3) and len(a) >= 3,
            "Expected matching coordinate paths with input, layers and final output.")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite coordinate path.")
    ra, rb = a[:, 1]-a[:, 0], b[:, 1]-b[:, 0]
    ca = np.concatenate([ra[:1], np.diff(ra, axis=0)])
    cb = np.concatenate([rb[:1], np.diff(rb, axis=0)])
    direction = rb[-1]+ra[-1]
    parts = ((cb-ca)*direction).sum(-1)
    actual = float(rb[-1]@rb[-1]-ra[-1]@ra[-1])
    residual = actual-float(parts.sum())
    require(abs(residual) <= 1e-9*max(1.0, abs(actual)), "Squared-distance accounting does not close.")
    qa, qb = a[-1]-a[-2], b[-1]-b[-2]
    fixed = -float(direction@(qb[0]-qa[0]))
    generated = float(direction@(qb[1]-qa[1]))
    require(abs(fixed+generated-parts[-1]) <= 1e-9*max(1.0, abs(parts[-1])),
            "Fixed/generated readout accounting does not close.")
    summary = {
        "internal_distance_before_A": float(np.linalg.norm(ra[-2])),
        "internal_distance_after_A": float(np.linalg.norm(rb[-2])),
        "output_distance_before_A": float(np.linalg.norm(ra[-1])),
        "output_distance_after_A": float(np.linalg.norm(rb[-1])),
        "output_squared_distance_change_A2": actual,
        "before_readout_vector_term_A2": float(parts[:-1].sum()),
        "readout_correction_vector_term_A2": float(parts[-1]),
        "fixed_readout_vector_term_A2": fixed,
        "generated_readout_vector_term_A2": generated,
        "distance_accounting_residual_A2": residual,
    }
    return summary, parts


def analyze(head_dir, hidden_dir, out, ref):
    report = read_json(head_dir/"summary.json")
    prior = read_json(hidden_dir/"replay_summary.json")
    plan = report["plan"]
    require(report["capture_completed"] and report["analysis_completed"] and report["error"] is None,
            "Use a completed head-v2 analysis.")
    require(digest(Path(ref.__file__)) == report["script_sha256"], "Head reference script differs from captured version.")
    require(prior["all_requested_calls_verified"] and prior["error"] is None, "Hidden replay was not verified.")
    require(digest(hidden_dir/"replay_summary.json") == report["previous_summary_sha256"], "Hidden replay summary changed.")
    require(prior["checkpoint_sha256"] == report["checkpoint_sha256"], "Checkpoint identities differ.")
    calls, seeds, site = plan["calls"], plan["seeds"], plan["site"]
    require(len(calls) >= 2 and all(b-a == 1 for a,b in zip(calls,calls[1:])), "Use consecutive calls.")
    target, n_layers = plan["target_bond_class"], prior["n_ligand_layers"]
    require(n_layers >= 1 and len(set(seeds)) == len(seeds), "Invalid layer count or seeds.")
    requested = {(s,c) for s in seeds for c in calls}
    require(len(report["states"]) == len(requested) and {(r["seed"],r["call"]) for r in report["states"]} == requested,
            "Missing/duplicated head states.")
    prior_records = {(r["seed"],r["call_index"]):r for r in prior["call_results"]}
    metadata, provenance, cache = {}, {}, {}
    for seed in seeds:
        path = hidden_dir/f"seed_{seed}/metadata.json"
        metadata[seed] = read_json(path)
        provenance[str(path)] = digest(path)
        require(metadata[seed]["recorded_manifest"] == report["metadata"][str(seed)]["manifest"], "Manifests differ.")
        require(metadata[seed]["task_definition"] == report["metadata"][str(seed)]["task"], "Task mappings differ.")
        require(metadata[seed]["coord_scale"] == report["metadata"][str(seed)]["coord_scale"], "Coordinate scales differ.")
    require(digest(head_dir/"head_weights.npz") == report["weights_sha256"], "Head weights changed.")
    weights = {k:v.astype(np.float64) for k,v in ref.load_npz(head_dir/"head_weights.npz").items()}
    for entry in report["states"]:
        seed, call = entry["seed"], entry["call"]
        key = (seed,call)
        require(key in prior_records, "State absent from hidden replay summary.")
        previous = prior_records[key]
        require(entry["accepted"] and entry["numerics"]["passed"] and previous["accepted"], "Unverified state.")
        for check in ["replay_vs_original", "observation_vs_original", "observation_vs_replay"]:
            require(entry[check]["passed"] and previous[check]["passed"], "Replay consistency check failed.")
        require(entry["times"] == previous["time"], "Recorded times differ.")
        hp, xp = head_dir/entry["file"], hidden_dir/f"seed_{seed}/call_{call:04d}_hidden.npz"
        require(digest(hp) == entry["sha256"], "Head capture file changed.")
        observed, hidden = ref.load_npz(hp), ref.load_npz(xp)
        provenance[str(hp)], provenance[str(xp)] = digest(hp), digest(xp)
        ref.validate_layout(observed,report["d_message"],report["d_edge"])
        require(int(observed["site"]) == site, "Captured fixed site differs from plan.")
        partners = observed["partners"].astype(int)
        fixed = metadata[seed]["task_definition"]["fixed_sampler_indices"]
        require(observed["fixed_indices"].tolist() == fixed, "Head fixed mapping differs from task.")
        require(np.array_equal(partners, ref.generated_partners(metadata[seed]["task_definition"],observed["output_mask"],site)),
                "Generated partner coverage differs from the saved mask.")
        expected = hidden["output_bond_logits"][np.maximum(site,partners),np.minimum(site,partners)]
        require(np.array_equal(expected,observed["bond_logits"]), "Head and hidden output logits differ.")
        independent = ref.predict_head(observed["mlp_input"].astype(np.float64),weights)
        require(ref.compare_arrays(independent,observed["ieee_head_logits"],atol=2e-4,rtol=1e-5)["passed"],
                "Head reference verification failed.")
        found = sorted(int(re.fullmatch(r"layer_(\d+)__out__coords",k).group(1)) for k in hidden if re.fullmatch(r"layer_(\d+)__out__coords",k))
        require(found == list(range(n_layers)), "Hidden coordinate layers are missing or duplicated.")
        paths = {int(p):stage_coordinates(hidden,observed,metadata[seed],site,int(p),n_layers) for p in partners}
        cache[key] = (observed,paths)
    changes, stages, states = [], [], []
    stage_names = ["input"]+[f"layer_{i+1}" for i in range(n_layers)]+["final_coordinate_output"]
    for (seed,call),(observed,paths) in cache.items():
        for partner,path in paths.items():
            if path is None:continue
            for name,coords in zip(stage_names,path):
                states.append({"seed":seed,"call":call,"site":site,"partner":partner,"stage":name,
                               "distance_A":float(np.linalg.norm(coords[1]-coords[0]))})
    for seed in seeds:
        for c0,c1 in zip(calls,calls[1:]):
            a,pa = cache[(seed,c0)]
            b,pb = cache[(seed,c1)]
            require(np.array_equal(a["partners"],b["partners"]), "Partner mapping changed between calls.")
            for head_row in ref.pair_transition(a,b,weights,report["d_message"],report["d_edge"],target):
                partner = head_row["partner"]
                row = {"seed":seed,"call_from":c0,"call_to":c1,"site":site,**head_row}
                row["bond_class_changed"] = row["class_before"] != row["class_after"]
                row["target_bond_emerged"] = row["class_before"] == 0 and row["class_after"] == target
                # Availability is separate from normal/abnormal labels.
                row["geometry_source_covered"] = pa[partner] is not None and pb[partner] is not None
                row["coverage_note"] = "recorded" if row["geometry_source_covered"] else "atom_not_in_previous_hidden_capture"
                if row["geometry_source_covered"]:
                    ledger,parts = distance_ledger(pa[partner],pb[partner])
                    row.update(ledger)
                    for name,value in zip(stage_names,parts):
                        stages.append({"seed":seed,"call_from":c0,"call_to":c1,"site":site,"partner":partner,
                                       "stage":name,"squared_distance_vector_term_A2":float(value)})
                changes.append(row)
    events = [r for r in changes if r["bond_class_changed"]]
    covered = [r for r in changes if r["geometry_source_covered"]]
    summary = {
        "version":"2026-10-20.2", "completed":True, "case":plan["case"], "plan":plan,
        "n_replays":len(cache), "n_layers":n_layers, "head_pair_transitions":len(changes),
        "geometry_source_transitions":len(covered), "uncovered_transitions":len(changes)-len(covered),
        "bond_class_change_events":len(events), "uncovered_events":sum(not r["geometry_source_covered"] for r in events),
        "max_distance_accounting_residual_A2":max((abs(r["distance_accounting_residual_A2"]) for r in covered),default=None),
        "events":events, "input_sha256":provenance,
        "script_sha256":digest(Path(__file__)), "head_summary_sha256":digest(head_dir/"summary.json"),
        "coverage_note":"Only atoms tracked in the earlier hidden replay have a coordinate-source ledger. Uncovered pairs are not labeled normal. The earlier tracked set was selected using final partners.",
        "interpretation":"Geometric vector accounting on two observed calls, not causal layer importance. Head contributions retain their reference/path dependence. No input-atom SHAP, intervention, training, or mechanism generalization has been established.",
    }
    write_csv(out/"source_changes.csv",changes)
    write_csv(out/"stage_vector_terms.csv",stages)
    write_csv(out/"stage_distances.csv",states)
    make_report(out,summary,changes,states)
    write_json(out/"source_summary.json",summary)
    return summary


def make_report(out,summary,changes,states):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plan = summary["plan"]
    available = sorted({(r["seed"],r["partner"]) for r in changes if r["geometry_source_covered"]})
    # Display specified pairs if present, otherwise a deterministic small subset.
    display = [(s,p) for s,p in available if p in plan.get("focus_partners",{}).get(str(s),[])]
    if not display:display = available[:6]
    pictures = []
    if display:
        fig,axes = plt.subplots(len(display),2,figsize=(13,3.4*len(display)),squeeze=False,constrained_layout=True)
        for index,(seed,partner) in enumerate(display):
            left,right = axes[index]
            for stage,label,color in [(f"layer_{summary['n_layers']}","Before final coordinate output","#718096"),
                                      ("final_coordinate_output","Actual output distance","#137c91")]:
                data=sorted([r for r in states if r["seed"]==seed and r["partner"]==partner and r["stage"]==stage],key=lambda r:r["call"])
                left.plot([r["call"] for r in data],[r["distance_A"] for r in data],"o-",ms=3,color=color,label=label)
            data=[r for r in changes if r["seed"]==seed and r["partner"]==partner and r["geometry_source_covered"]]
            x=np.arange(len(data))
            right.bar(x-.18,[r["before_readout_vector_term_A2"] for r in data],width=.35,color="#718096",label="Before-readout term")
            right.bar(x+.18,[r["readout_correction_vector_term_A2"] for r in data],width=.35,color="#d68525",label="Final-correction term")
            right.plot(x,[r["output_squared_distance_change_A2"] for r in data],"ko-",ms=3,label="Actual total")
            right.set_xticks(x,[f"{r['call_from']}>{r['call_to']}" for r in data],rotation=45)
            left.set_title(f"Seed {seed}: fixed {plan['site']} - generated {partner}")
            left.set_xlabel("Recorded network call (zero-based)")
            left.set_ylabel("Distance (angstrom)")
            right.set_title("Observed vector accounting; not causal importance")
            right.set_ylabel("Δ distance² (Å²)")
            for ax in [left,right]:ax.grid(axis="y",alpha=.2);ax.legend(fontsize=8)
            right.axhline(0,color="#444",lw=.6)
        fig.suptitle(f"{summary['case']}: coordinate-source observations")
        path=out/"coordinate_sources.png"
        fig.savefig(path,dpi=140);plt.close(fig)
        pictures.append(path)
    def val(value):
        if isinstance(value,float):return f"{value:.4f}"
        return html.escape(str(value))
    event_rows="".join("<tr>"+"".join(f"<td>{val(v)}</td>" for v in [
        r["seed"],f"{r['call_from']}→{r['call_to']}",f"{r['site']}–{r['partner']}",
        f"{r['class_before']}→{r['class_after']}",r["geometry"],r["nodes"],r["pairs"],
        r.get("internal_distance_before_A","缺记录"),r.get("internal_distance_after_A","缺记录"),
        r.get("output_distance_before_A","缺记录"),r.get("output_distance_after_A","缺记录")])+"</tr>" for r in summary["events"])
    if not event_rows:event_rows='<tr><td colspan="11">本窗口没有检测到键类别变化；不代表不存在其他失败。</td></tr>'
    images="".join('<img alt="坐标来源观察" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">' for p in pictures)
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>连接机制分析：坐标来源</title>
<style>body{{font:16px/1.7 sans-serif;max-width:1280px;margin:30px auto;padding:0 20px;color:#253644}}.notice{{background:#fff1da;padding:14px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #ddd;padding:7px;text-align:right}}th{{background:#edf4f6}}.scroll{{overflow-x:auto}}</style>
<h1>连接机制分析：几何变化从哪里进入成键预测</h1>
<p>案例：{html.escape(summary['case'])}；已核对 {summary['n_replays']} 个真实回放状态。本页结果由已有记录离线计算。可选的新增只读回放另存 readout_capture_summary.json；两种模式均不做内部干预。</p>
<p class="notice"><strong>这已经是内部计算分析，但仍是候选机制证据。</strong>成键头贡献说明局部得分变化如何分配到几何、节点和原子对输入；下面继续核对几何变化在坐标计算路径中的来源。不能把读出阶段承载变化理解成该模块本身就是错误根因，上游隐藏表示也可能决定这种变化。</p>
<h2>自动列出的键类别变化</h2><p>0=无键；1=单键；2=双键；3=三键；4=芳香键。下表列出窗口内所有类别变化，不预先写入种子或异常调用。键类别变化本身不是分子失败标签，必须结合固定片段、化学有效性和最终结构检查。</p>
<div class="scroll"><table><tr><th>种子</th><th>调用</th><th>原子对</th><th>类别</th><th>几何贡献</th><th>节点贡献</th><th>原子对贡献</th><th>内部距离：前</th><th>内部距离：后</th><th>输出距离：前</th><th>输出距离：后</th></tr>{event_rows}</table></div>
<p>三路贡献的单位是目标键类相对无键的 logit 变化；距离单位为 Å。数值差异及精度标志保留在 source_changes.csv 中。贡献不是百分比，也不是输入原子的独占贡献。</p>
{images}
<h2>本次覆盖范围</h2><p>成键头共分析 {summary['head_pair_transitions']} 条原子对转换；其中 {summary['geometry_source_transitions']} 条具备逐层坐标记录，另外 {summary['uncovered_transitions']} 条因早期记录未跟踪相关原子，不能继续追溯坐标来源。检测到 {summary['bond_class_change_events']} 次键类别变化，其中 {summary['uncovered_events']} 次缺少坐标路径。没有坐标记录不等于没有异常。</p>
<p>显示对象取自配置或已记录对象，完整可用结果保留在 CSV。旧隐藏记录使用终态连接对象选取跟踪原子，存在事后选择；后续泛化验证应预先记录全部候选原子并固定选择规则。</p>
<h2>如何理解右侧图</h2><p>把输出相对位置写成“最后一层内部相对位置＋最终坐标修正”，可以精确核算相邻调用的距离平方变化。灰柱和橙柱相加等于黑线。详细逐层分项见 stage_vector_terms.csv。这个数学等式只用于定位候选计算路径，不能给层做因果重要性排名。</p>
<p>核算最大残差：{summary['max_distance_accounting_residual_A2']} Å²。它不是解释准确率。</p>
<h2>下一项需要检验的假设</h2><p>若异常连接得分的变化主要分配到几何输入，且距离变化主要由最终坐标修正承载，进一步记录该通路的最后隐藏向量、坐标归一化前后数值和投影输出，区分上游隐藏变化、归一化缩放及投影方向。对照同一轨迹的恢复阶段和成功轨迹，形成具体预测后再做选择性干预。</p>
<p>本分析只检验当前样本。不同复合物需使用自己的映射和记录；参数化不等于泛化已经成立。节点或原子对通路占主导的其他失败，应沿相应通路追踪，不强行归为几何原因。</p></html>'''
    (out/"REPORT.html").write_text(doc,encoding="utf-8")


class ReadoutObserver:
    """Copies full-node coordinates/features; hooks never replace outputs."""
    def __init__(self,decoder,to_array):
        self.arrays,self.handles,self.visits={},[],{}
        self.to_array=to_array
        for i,layer in enumerate(decoder.layers):
            self.add(layer,f"layer_{i:02d}_in",self.layer_pre(i),pre=True,kwargs=True)
            self.add(layer,f"layer_{i:02d}_out",self.layer_post(i),kwargs=True)
        self.add(decoder.final_coord_norm,"final_coordinate_norm",self.norm)
        self.add(decoder.coord_out_proj,"coordinate_projection",self.projection)

    def save(self,name,tensor):
        require(tensor is not None and tensor.shape[0]==1,"Readout observer expects batch-one tensors.")
        self.arrays[name]=self.to_array(tensor[0])

    def add(self,module,name,fn,pre=False,kwargs=False):
        self.visits[name]=0
        def hook(*args):
            self.visits[name]+=1
            require(self.visits[name]==1,f"Unexpected repeated invocation: {name}")
            fn(*args)
            return None
        register=module.register_forward_pre_hook if pre else module.register_forward_hook
        self.handles.append(register(hook,with_kwargs=True) if kwargs else register(hook))

    def layer_pre(self,i):
        def hook(module,args,kwargs):
            self.save(f"layer_{i:02d}__in__equis",args[0])
            self.save(f"layer_{i:02d}__in__coords",kwargs["coords"])
        return hook

    def layer_post(self,i):
        def hook(module,args,kwargs,output):
            require(len(output)==5,"Unexpected ROOT layer output contract.")
            self.save(f"layer_{i:02d}__out__equis",output[0])
            self.save(f"layer_{i:02d}__out__coords",output[2])
        return hook

    def norm(self,module,inputs,output):
        self.save("final_norm_input",inputs[0])
        self.save("final_norm_mask",inputs[1])
        self.save("final_norm_output",output)

    def projection(self,module,inputs,output):
        self.save("coordinate_projection_input",inputs[0])
        self.save("coordinate_projection_output",output)

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles.clear()


def check_readout_arrays(a,n_layers,tracked,old,head):
    """Check observed identities and shapes without cross-precision recomputation."""
    n=len(a["output_mask"])
    shape=a["final_norm_input"].shape
    require(len(shape)==3 and shape[:2]==(n,3),"Unexpected equivariant feature shape.")
    require(a["coordinate_projection_output"].shape==(n,3,1),"Unexpected projection output shape.")
    require(a["output_coords"].shape==(n,3),"Unexpected coordinate output shape.")
    require(all(np.isfinite(v).all() for v in a.values()),"Nonfinite readout observations.")
    require(np.array_equal(a["final_norm_mask"],a["output_mask"]),"Coordinate masks differ.")
    require(np.array_equal(a["final_norm_output"],a["coordinate_projection_input"]),"Norm-to-projection continuity fails.")
    require(np.array_equal(a["final_norm_input"],a[f"layer_{n_layers-1:02d}__out__equis"]),"Final hidden input differs from last layer.")
    require(np.array_equal(a["output_coords"],head["head_coords"]),"New output differs from verified head coordinates.")
    previous_coords=a["input_coords"]
    previous_equis=None
    for i in range(n_layers):
        for side in ["in","out"]:
            prefix=f"layer_{i:02d}__{side}"
            require(a[prefix+"__equis"].shape==shape and a[prefix+"__coords"].shape==(n,3),"Layer tensor shape changed.")
            for kind in ["coords","equis"]:
                require(np.array_equal(a[prefix+"__"+kind][tracked],old[prefix+"__"+kind]),"New capture differs from previously verified hidden values.")
        require(np.array_equal(previous_coords,a[f"layer_{i:02d}__in__coords"]),"Coordinate layer continuity fails.")
        if previous_equis is not None:
            require(np.array_equal(previous_equis,a[f"layer_{i:02d}__in__equis"]),"Equivariant layer continuity fails.")
        previous_coords=a[f"layer_{i:02d}__out__coords"]
        previous_equis=a[f"layer_{i:02d}__out__equis"]
    return True


def capture_readout(args,out,ref):
    """Original-precision replay only; preserves native parameter grad flags."""
    root=Path.cwd()
    require((root/"flowr/scriptutil.py").is_file(),"Readout capture must run in the existing FLOWR.ROOT repo.")
    helper=root/"replay_connection_hidden.py"
    require(helper.is_file() and digest(helper)==ref.HELPER_SHA,"Keep the verified replay helper unchanged.")
    sys.path.insert(0,str(root))
    import torch
    import replay_connection_hidden as helpers
    from flowr.scriptutil import load_model
    from flowr.util.device import resolve_device
    head_dir,hidden_dir=args.head_dir.resolve(),args.hidden_dir.resolve()
    source=read_json(head_dir/"summary.json")
    plan=source["plan"]
    require(source["capture_completed"] and source["analysis_completed"],"Head capture is incomplete.")
    base=source["metadata"][str(plan["seeds"][0])]["manifest"]
    for name,expected in source["source_sha256"].items():
        require(digest(root/name)==expected,f"Model source changed: {name}")
    require(digest(Path(base["args"]["ckpt_path"]))==base["checkpoint_sha256"],"Checkpoint changed.")
    require(str(torch.__version__)==base["torch_version"],"PyTorch environment changed.")
    result={"version":"2026-10-20.2", "completed":False,"error":None,"plan":plan,"states":[],
            "checkpoint_sha256":base["checkpoint_sha256"],"source_sha256":source["source_sha256"],
            "head_summary_sha256":digest(head_dir/"summary.json"),"script_sha256":digest(Path(__file__)),
            "scope":"Read-only full-node coordinate and equivariant feature capture. No model changes, intervention, gradient attribution, or rollout.",
            "metadata":{}}
    write_json(out/"readout_capture_summary.json",result)
    observer=None
    try:
        torch.set_float32_matmul_precision("high")
        model_args=argparse.Namespace(**base["args"])
        model_args.save_dir=str(out/"model_context")
        device=resolve_device(model_args)
        print(f"Loading original model on {device}; read-only full-node capture.",flush=True)
        model=load_model(model_args)[0].to(device).eval()
        decoder=model.gen.ligand_dec
        n_layers=len(decoder.layers)
        require(n_layers==read_json(hidden_dir/"replay_summary.json")["n_ligand_layers"],"Layer count changed.")
        result.update(n_layers=n_layers,precision=torch.get_float32_matmul_precision(),device=str(device),
                      torch_version=str(torch.__version__),parameter_tensors_requiring_grad=sum(p.requires_grad for p in model.parameters()),
                      coord_skip_connect=bool(decoder.coord_skip_connect),pocket_conditioned=decoder.d_pocket_inv is not None,
                      final_norm_zero_com=bool(decoder.final_coord_norm.zero_com),final_norm_eps=float(decoder.final_coord_norm.eps),
                      intermediate_coord_updates=[bool(layer.intermediate_coord_updates) for layer in decoder.layers])
        weights={"coordinate_projection_weight":ref.to_array(decoder.coord_out_proj.weight),
                 "final_norm_set_weights":ref.to_array(decoder.final_coord_norm.set_weights)}
        if decoder.coord_out_proj.bias is not None:weights["coordinate_projection_bias"]=ref.to_array(decoder.coord_out_proj.bias)
        np.savez_compressed(out/"coordinate_readout_weights.npz",**weights)
        result["weights_sha256"]=digest(out/"coordinate_readout_weights.npz")
        load=lambda path:torch.load(path,map_location="cpu",weights_only=True)
        for seed in plan["seeds"]:
            recorded=args.output_root.resolve()/f"sh_record_{plan['case']}_seed{seed}/research_record"
            manifest=read_json(recorded/"manifest.json")
            require(manifest==source["metadata"][str(seed)]["manifest"],"Recorded manifest changed.")
            folder=recorded/"attempt_000"
            task=read_json(folder/"task_definition.json")
            require(task==source["metadata"][str(seed)]["task"],"Task definition changed.")
            inputs=load(folder/"inputs.pt")
            require(bool(model.self_condition)==inputs["self_condition"] and int(model.sc_recycle_steps)==inputs["sc_recycle_steps"],"Self-conditioning settings changed.")
            pocket=helpers.move_copy(inputs["pocket_data"],device,torch)
            encoding=helpers.move_copy(load(folder/"pocket_encoding.pt"),device,torch)
            require(encoding["pocket_equis"] is not None and encoding["pocket_invs"] is not None,"Missing original pocket encoding.")
            result["metadata"][str(seed)]={"task":task,"coord_scale":inputs["coord_scale"],"manifest":manifest}
            destination=out/f"seed_{seed}"
            destination.mkdir()
            meta=read_json(hidden_dir/f"seed_{seed}/metadata.json")
            for call_index in plan["calls"]:
                print(f"Read-only coordinate capture: seed {seed}, call {call_index}",flush=True)
                entry={"seed":seed,"call":call_index,"accepted":False,"file":f"seed_{seed}/call_{call_index:04d}_readout.npz"}
                result["states"].append(entry)
                call=load(folder/"forward_calls"/f"call_{call_index:04d}.pt")
                require(call["call_index"]==call_index,"Call index mismatch.")
                baseline=helpers.forward_once(model,call,pocket,encoding,device,torch,np)
                entry["replay_vs_original"]=helpers.compare_predictions(baseline,call["network_output_before_postprocessing"],torch,np,1e-5,1e-5)
                require(entry["replay_vs_original"]["passed"],"Original replay differs; stop capture.")
                observer=ReadoutObserver(decoder,ref.to_array)
                try:
                    observed=helpers.forward_once(model,call,pocket,encoding,device,torch,np)
                finally:
                    observer.close()
                a=observer.arrays
                for name,key in [("output_coords","coords"),("output_mask","mask"),("output_bond_logits","bonds"),("output_atom_logits","atomics"),("output_charge_logits","charges")]:
                    a[name]=ref.to_array(observed[key][0])
                a["input_coords"]=ref.to_array(call["state_before_forward"]["coords"][0])
                a["all_sampler_indices"]=np.arange(len(a["output_mask"]))
                a["fixed_sampler_indices"]=np.array(task["fixed_sampler_indices"],dtype=np.int64)
                np.savez_compressed(out/entry["file"],**a)
                entry["sha256"]=digest(out/entry["file"])
                entry["times"]=call["times_scalar"]
                entry["observation_vs_original"]=helpers.compare_predictions(observed,call["network_output_before_postprocessing"],torch,np,1e-5,1e-5)
                entry["observation_vs_replay"]=helpers.compare_predictions(observed,baseline,torch,np,1e-5,1e-5)
                require(entry["observation_vs_original"]["passed"] and entry["observation_vs_replay"]["passed"],"Read-only capture changed predictions.")
                require(all(v==1 for v in observer.visits.values()),"Unexpected hook invocation count.")
                old=ref.load_npz(hidden_dir/f"seed_{seed}/call_{call_index:04d}_hidden.npz")
                previous=ref.load_npz(head_dir/f"seed_{seed}/call_{call_index:04d}_head.npz")
                check_readout_arrays(a,n_layers,meta["tracked_atom_indices"],old,previous)
                entry["accepted"]=True
                entry["all_atom_slots_recorded"]=len(a["output_mask"])
                write_json(out/"readout_capture_summary.json",result)
                print("  MATCH: native replay, all-node hooks and earlier observations agree.",flush=True)
                observer=None
                del a,observed,baseline,call,old,previous
        result["completed"]=len(result["states"])==len(plan["seeds"])*len(plan["calls"]) and all(r["accepted"] for r in result["states"])
    except Exception as exc:
        result["error"]=f"{type(exc).__name__}: {exc}"
        if observer is not None:
            np.savez_compressed(out/"failed_readout_observation.npz",**observer.arrays)
        raise
    finally:
        if observer is not None:observer.close()
        write_json(out/"readout_capture_summary.json",result)
    return result


def readout_reference(a, weights, config):
    """Literal FP64 reference for the captured pocket-conditioned ROOT readout.

    This is a numerical reference, not a modified model run. Unsupported
    architectures fail explicitly rather than inheriting an incorrect formula.
    """
    require(config['coord_skip_connect'] and config['pocket_conditioned']
            and not config['final_norm_zero_com'], 'Unsupported coordinate-readout architecture.')
    e = a['final_norm_input'].astype(np.float64)
    mask = a['final_norm_mask'].astype(np.float64)
    require(e.ndim == 3 and e.shape[:2] == (len(mask), 3), 'Unexpected hidden layout.')
    require(np.isin(mask, [0, 1]).all() and mask.sum() > 0, 'Invalid node mask.')
    gamma = weights['final_norm_set_weights'].astype(np.float64).reshape(-1)
    w = weights['coordinate_projection_weight'].astype(np.float64)
    require(w.shape == (1, e.shape[-1]) and gamma.shape == (e.shape[-1],), 'Readout weight shape differs.')
    bias = weights.get('coordinate_projection_bias', np.zeros(1)).astype(np.float64)
    require(bias.shape == (1,), 'Unexpected readout bias.')
    eps = float(config['final_norm_eps'])
    require(eps > 0 and np.isfinite(eps), 'Invalid norm epsilon.')
    # Mirror _CoordNorm literally: lengths sum over every slot; mask supplies
    # the denominator and is applied after normalisation.
    lengths = np.linalg.norm(e, axis=1).sum(axis=0) / mask.sum()
    normalized = e * gamma / (lengths + eps) * mask[:, None, None]
    scale = gamma * w[0] / (lengths + eps)
    q = ((e * scale).sum(-1) + bias[0]) * mask[:, None]
    actual_q = a['coordinate_projection_output'][..., 0].astype(np.float64) * mask[:, None]
    base = a[f"layer_{config['n_layers']-1:02d}__out__coords"].astype(np.float64)
    output = a['output_coords'].astype(np.float64)
    require(np.isfinite(normalized).all() and np.isfinite(q).all(), 'Nonfinite reference output.')
    require(np.allclose(normalized, a['final_norm_output'], atol=2e-6, rtol=1e-5),
            'Literal normalization reference disagrees with recorded normalization.')
    return dict(e=e, mask=mask, scale=scale, q=q, actual_q=actual_q, base=base, output=output,
                norm_max_abs_difference=float(np.max(abs(normalized-a['final_norm_output']))),
                projection_max_abs_difference=float(np.max(abs(q-actual_q))),
                addition_max_abs_difference=float(np.max(abs(output-base-actual_q))))


def readout_ledger(a, b, ra, rb, site, partner, coord_scale, n_layers):
    """Symmetric two-factor accounting of E * scale, followed by exact d² ledger.

    Layer terms allocate recorded net feature updates; they are not independent
    interventions or causal importance. Reset layers remain net updates.
    """
    require(np.array_equal(ra['mask'], rb['mask']), 'Mask changed within comparison.')
    require(ra['mask'][site] == rb['mask'][partner] == 1, 'Pair includes an inactive node.')
    require(coord_scale > 0 and np.isfinite(coord_scale), 'Invalid coordinate units.')
    rel = lambda v: (v[partner]-v[site]) * coord_scale
    direction = rel(ra['output']) + rel(rb['output'])
    average_scale = (ra['scale']+rb['scale'])/2
    hidden = ((rb['e']-ra['e'])*average_scale).sum(-1)*ra['mask'][:, None]
    norm = ((ra['e']+rb['e'])/2*(rb['scale']-ra['scale'])).sum(-1)*ra['mask'][:, None]
    require(np.allclose(hidden+norm, rb['q']-ra['q'], atol=1e-10, rtol=1e-9),
            'Two-factor identity failed.')
    vectors = dict(
        before_readout=rb['base']-ra['base'], hidden_features=hidden, norm_scale=norm,
        projection_numerics=(rb['actual_q']-rb['q'])-(ra['actual_q']-ra['q']),
        output_rounding=(rb['output']-rb['base']-rb['actual_q'])-(ra['output']-ra['base']-ra['actual_q']))
    terms = {k+'_term_A2': float(direction@rel(v)) for k,v in vectors.items()}
    d0, d1 = np.linalg.norm(rel(ra['output'])), np.linalg.norm(rel(rb['output']))
    total = float(d1*d1-d0*d0)
    residual = total-sum(terms.values())
    require(abs(residual) <= 1e-9*max(1., abs(total)), 'Full readout ledger does not close.')
    layers = []
    for layer in range(-1, n_layers):
        def component(state):
            if layer == -1:return state['layer_00__in__equis'].astype(np.float64)
            return (state[f'layer_{layer:02d}__out__equis'].astype(np.float64)
                    -state[f'layer_{layer:02d}__in__equis'].astype(np.float64))
        q_part = ((component(b)-component(a))*average_scale).sum(-1)*ra['mask'][:, None]
        layers.append({'stage': 'initial_hidden' if layer == -1 else f'layer_{layer+1}',
                       'hidden_feature_term_A2': float(direction@rel(q_part))})
    layer_residual = terms['hidden_features_term_A2']-sum(r['hidden_feature_term_A2'] for r in layers)
    require(abs(layer_residual) <= 1e-9*max(1., abs(terms['hidden_features_term_A2'])),
            'Layer net-feature accounting does not close.')
    numerical = terms['projection_numerics_term_A2']+terms['output_rounding_term_A2']
    return dict(**terms, readout_accounting_residual_A2=residual,
                layer_hidden_accounting_residual_A2=layer_residual,
                readout_needs_precision_review=abs(numerical) >= abs(total) and abs(numerical)>1e-8), layers


def analyze_readout(head_dir, hidden_dir, readout_dir, out, ref):
    """Analyze every valid generated partner of the configured fixed site."""
    h = read_json(head_dir/'summary.json')
    s = read_json(readout_dir/'readout_capture_summary.json')
    old = read_json(hidden_dir/'replay_summary.json')
    require(s['completed'] and s['error'] is None, 'Incomplete full-node capture.')
    require(h['capture_completed'] and h['analysis_completed'] and h['error'] is None,
            'Incomplete head analysis.')
    require(old['all_requested_calls_verified'] and old['error'] is None, 'Incomplete earlier hidden capture.')
    require(digest(head_dir/'summary.json') == s['head_summary_sha256'], 'Head summary changed.')
    require(digest(hidden_dir/'replay_summary.json') == h['previous_summary_sha256'], 'Earlier hidden summary changed.')
    require(digest(Path(ref.__file__)) == h['script_sha256'], 'Head reference version changed.')
    require(s['checkpoint_sha256'] == h['checkpoint_sha256'] == old['checkpoint_sha256'], 'Checkpoint mismatch.')
    require(s['source_sha256'] == h['source_sha256'], 'Model source identity mismatch.')
    require(s['plan'] == h['plan'] and s['metadata'] == h['metadata'], 'Task or metadata mismatch.')
    require(digest(readout_dir/'coordinate_readout_weights.npz') == s['weights_sha256'], 'Readout weights changed.')
    require(digest(head_dir/'head_weights.npz') == h['weights_sha256'], 'Head weights changed.')
    plan, n_layers = s['plan'], s['n_layers']
    require(n_layers == old['n_ligand_layers'] and n_layers > 0, 'Layer counts disagree.')
    seeds, calls, site, target = plan['seeds'], plan['calls'], plan['site'], plan['target_bond_class']
    require(len(calls)>1 and all(y-x==1 for x,y in zip(calls,calls[1:])), 'Calls must be consecutive.')
    require(len(seeds)==len(set(seeds)), 'Duplicate seeds.')
    expected = {(seed,call) for seed in seeds for call in calls}
    for report in [s,h]:
        require(len(report['states'])==len(expected) and {(v['seed'],v['call']) for v in report['states']}==expected,
                'Missing or duplicated captured states.')
    head_entries = {(v['seed'],v['call']):v for v in h['states']}
    weights = ref.load_npz(readout_dir/'coordinate_readout_weights.npz')
    head_weights = {k:v.astype(np.float64) for k,v in ref.load_npz(head_dir/'head_weights.npz').items()}
    cache, provenance, state_rows, verification = {}, {}, [], []
    for entry in s['states']:
        seed, call = entry['seed'], entry['call']
        he = head_entries[(seed,call)]
        require(entry['accepted'] and he['accepted'] and he['numerics']['passed'], 'Unaccepted capture.')
        require(entry['times'] == he['times'], 'Recorded times disagree.')
        for e in [entry,he]:
            for name in ['replay_vs_original','observation_vs_original','observation_vs_replay']:
                require(e[name]['passed'], 'Native replay check failed.')
        path, hp = readout_dir/entry['file'], head_dir/he['file']
        require(digest(path)==entry['sha256'] and digest(hp)==he['sha256'], 'Captured file changed.')
        provenance[entry['file']] = entry['sha256']
        a, observed = ref.load_npz(path), ref.load_npz(hp)
        meta = read_json(hidden_dir/f'seed_{seed}/metadata.json')
        require(meta['task_definition']==s['metadata'][str(seed)]['task'] and
                meta['recorded_manifest']==s['metadata'][str(seed)]['manifest'] and
                meta['coord_scale']==s['metadata'][str(seed)]['coord_scale'], 'Earlier metadata changed.')
        previous = ref.load_npz(hidden_dir/f'seed_{seed}/call_{call:04d}_hidden.npz')
        check_readout_arrays(a,n_layers,meta['tracked_atom_indices'],previous,observed)
        require(np.array_equal(a['all_sampler_indices'],np.arange(len(a['output_mask']))), 'Full-node mapping mismatch.')
        require(a['fixed_sampler_indices'].tolist()==meta['task_definition']['fixed_sampler_indices'], 'Fixed mapping mismatch.')
        ref.validate_layout(observed,h['d_message'],h['d_edge'])
        require(int(observed['site'])==site, 'Head site mismatch.')
        partners = np.asarray(ref.generated_partners(meta['task_definition'],a['output_mask'],site), dtype=int)
        require(np.array_equal(observed['partners'],partners), 'Generated coverage differs from actual mask.')
        require(np.array_equal(a['output_bond_logits'][np.maximum(site,partners),np.minimum(site,partners)],observed['bond_logits']),
                'Head and readout logits disagree.')
        require(ref.compare_arrays(ref.predict_head(observed['mlp_input'].astype(np.float64),head_weights),
                                  observed['ieee_head_logits'],atol=2e-4,rtol=1e-5)['passed'], 'Head reference mismatch.')
        r = readout_reference(a,weights,s)
        cache[(seed,call)] = (a,observed,r)
        verification.append({'seed':seed,'call':call,**{k:r[k] for k in
                             ['norm_max_abs_difference','projection_max_abs_difference','addition_max_abs_difference']}})
        scale = float(meta['coord_scale'])
        for index,p in enumerate(partners.tolist()):
            logits = observed['bond_logits'][index].astype(np.float64)
            state_rows.append(dict(seed=seed,call=call,site=site,partner=p,
                output_distance_A=float(np.linalg.norm(r['output'][p]-r['output'][site])*scale),
                internal_distance_A=float(np.linalg.norm(r['base'][p]-r['base'][site])*scale),
                bond_class=int(logits.argmax()), margin=float(logits[target]-logits[0])))
    changes, layers, coordinate_stages = [], [], []
    for seed in seeds:
        scale = float(s['metadata'][str(seed)]['coord_scale'])
        for c0,c1 in zip(calls,calls[1:]):
            a,ha,ra = cache[(seed,c0)]; b,hb,rb = cache[(seed,c1)]
            require(np.array_equal(ha['partners'],hb['partners']), 'Partners changed within trajectory.')
            for head_row in ref.pair_transition(ha,hb,head_weights,h['d_message'],h['d_edge'],target):
                p = head_row['partner']
                ident = dict(seed=seed,call_from=c0,call_to=c1,site=site,partner=p)
                terms, layer_terms = readout_ledger(a,b,ra,rb,site,p,scale,n_layers)
                def coords(v):
                    keys = ['input_coords']+[f'layer_{i:02d}__out__coords' for i in range(n_layers)]+['output_coords']
                    return np.stack([v[k][[site,p]].astype(np.float64) for k in keys])*scale
                distances, coord_terms = distance_ledger(coords(a),coords(b))
                row = {**ident,**head_row,**distances,**terms,
                       'bond_class_changed':head_row['class_before']!=head_row['class_after']}
                changes.append(row)
                layers.extend({**ident,**t} for t in layer_terms)
                names = ['input']+[f'layer_{i+1}' for i in range(n_layers)]+['final_readout']
                coordinate_stages.extend({**ident,'stage':name,'coordinate_term_A2':float(v)} for name,v in zip(names,coord_terms))
    events = [r for r in changes if r['bond_class_changed']]
    summary = dict(version='2026-10-20.3',completed=True,case=plan['case'],plan=plan,n_layers=n_layers,
        n_replays=len(cache),head_pair_transitions=len(changes),geometry_source_transitions=len(changes),uncovered_transitions=0,
        bond_class_change_events=len(events),events=events,
        head_precision_flagged_transitions=sum(r['needs_precision_review'] for r in changes),
        readout_precision_flagged_transitions=sum(r['readout_needs_precision_review'] for r in changes),
        max_native_replay_difference=max(f['max_abs_difference'] for e in s['states'] for k in
             ['replay_vs_original','observation_vs_original','observation_vs_replay'] for f in e[k]['fields'].values()),
        maximum_reference_errors={k:max(v[k] for v in verification) for k in verification[0] if k not in ['seed','call']},
        max_readout_accounting_residual_A2=max(abs(r['readout_accounting_residual_A2']) for r in changes),
        max_layer_hidden_accounting_residual_A2=max(abs(r['layer_hidden_accounting_residual_A2']) for r in changes),
        input_sha256=provenance,readout_summary_sha256=digest(readout_dir/'readout_capture_summary.json'),
        head_summary_sha256=digest(head_dir/'summary.json'),weights_sha256=s['weights_sha256'],
        checkpoint_sha256=s['checkpoint_sha256'],script_sha256=digest(Path(__file__)),
        interpretation='Observational candidate mechanism and reference-dependent algebraic allocation. Not causal validation, input-atom SHAP, or evidence of cross-complex generalization.',
        coverage_note='All active generated partners of the single configured fixed site; not all fixed sites. Display focus is selected, computation covers all pairs in this plan.')
    write_csv(out/'readout_changes.csv',changes)
    write_csv(out/'hidden_layer_terms.csv',layers)
    write_csv(out/'coordinate_stage_terms.csv',coordinate_stages)
    write_csv(out/'readout_states.csv',state_rows)
    write_csv(out/'readout_numeric_checks.csv',verification)
    write_json(out/'readout_analysis_summary.json',summary)
    make_readout_report(out,summary,changes,layers,state_rows)
    return summary


def make_readout_report(out,summary,changes,layers,states):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan = summary['plan']
    focus = [(s,p) for s in plan['seeds'] for p in plan.get('focus_partners',{}).get(str(s),[])]
    valid = {(r['seed'],r['partner']) for r in states}
    focus = [x for x in focus if x in valid] or sorted(valid)[:6]
    pictures = []
    if focus:
        fig,axes=plt.subplots(len(focus),2,figsize=(12.4,3.0*len(focus)),squeeze=False,layout='constrained')
        for (seed,p),(left,right) in zip(focus,axes):
            rows=sorted([r for r in states if r['seed']==seed and r['partner']==p],key=lambda r:r['call'])
            left.plot([r['call'] for r in rows],[r['internal_distance_A'] for r in rows],'o-',color='#8b95a5',ms=3,label='Before final readout')
            left.plot([r['call'] for r in rows],[r['output_distance_A'] for r in rows],'o-',color='#146c94',ms=3,label='Output prediction')
            left.set(title=f"Seed {seed}: fixed {plan['site']} / generated {p}",xlabel='Network call (zero-based)',ylabel='Distance (angstrom)')
            left.legend(fontsize=8);left.grid(alpha=.15)
            rs=[r for r in changes if r['seed']==seed and r['partner']==p];x=np.arange(len(rs))
            for shift,key,label,color in [(-.25,'before_readout_term_A2','Before readout','#8b95a5'),
                        (0,'hidden_features_term_A2','Hidden-vector term','#146c94'),(.25,'norm_scale_term_A2','Norm-scale term','#dd9d32')]:
                right.bar(x+shift,[r[key] for r in rs],width=.24,label=label,color=color)
            right.plot(x,[r['output_squared_distance_change_A2'] for r in rs],'k.-',label='Observed total')
            right.set_xticks(x,[f"{r['call_from']}>{r['call_to']}" for r in rs],rotation=45)
            right.axhline(0,color='black',lw=.5);right.set(ylabel='Squared-distance allocation (A²)',xlabel='Adjacent calls')
            right.legend(fontsize=7,ncols=2)
        path=out/'readout_sources.png';fig.savefig(path,dpi=160);plt.close(fig);pictures.append(path)
    events=summary['events']
    if events:
        fig,axes=plt.subplots(len(events),1,figsize=(11,2.9*len(events)),squeeze=False,layout='constrained')
        for e,(ax,) in zip(events,axes):
            rs=[r for r in layers if all(r[k]==e[k] for k in ['seed','call_from','call_to','site','partner'])]
            names=[r['stage'].replace('initial_hidden','Initial').replace('layer_','L') for r in rs]
            values=[r['hidden_feature_term_A2'] for r in rs]
            ax.bar(names,values,color=['#ba4a49' if v<0 else '#146c94' for v in values])
            ax.axhline(0,color='black',lw=.5);ax.set(ylabel='Allocation (A²)',
               title=f"Seed {e['seed']} / pair {e['site']}-{e['partner']} / calls {e['call_from']}->{e['call_to']}: net hidden updates")
        path=out/'hidden_layer_allocation.png';fig.savefig(path,dpi=160);plt.close(fig);pictures.append(path)
    def table(headers,rows):
        def cell(v):return html.escape(f'{v:.6f}' if isinstance(v,float) else str(v))
        return '<div class="scroll"><table><tr>'+''.join('<th>'+cell(v)+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+cell(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
    event_table=table(['种子','调用','原子对','键类别','输出距离变化 Å','读出前项 Å²','隐藏向量项 Å²','归一化尺度项 Å²'],[
        [e['seed'],f"{e['call_from']}→{e['call_to']}",f"{e['site']}–{e['partner']}",f"{e['class_before']}→{e['class_after']}",
         f"{e['output_distance_before_A']:.3f}→{e['output_distance_after_A']:.3f}",e['before_readout_term_A2'],
         e['hidden_features_term_A2'],e['norm_scale_term_A2']] for e in events])
    head_table=table(['种子','调用','原子对','几何项','节点项','原子对项','数值差异','实际得分变化'],[
        [e['seed'],f"{e['call_from']}→{e['call_to']}",f"{e['site']}–{e['partner']}",e['geometry'],e['nodes'],e['pairs'],e['numerical_difference'],e['observed_score_change']] for e in events])
    imgs=''.join('<img alt="内部记录分析图" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'">' for p in pictures)
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Scaffold hopping：初步内部机制分析</title>
<style>body{{font:16px/1.75 sans-serif;max-width:1200px;margin:32px auto;padding:0 22px;color:#213547}}h1,h2{{line-height:1.4}}.note{{padding:15px;background:#eef5f8;border-left:4px solid #146c94}}.limit{{background:#fff4e1;padding:15px}}table{{border-collapse:collapse;white-space:nowrap;font-size:13px;width:100%}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}th{{background:#f1f5f8}}.scroll{{overflow:auto}}img{{width:100%;height:auto}}code{{background:#f1f5f8;padding:2px 5px}}</style>
<h1>固定片段条件下的 scaffold hopping：内部观察分析</h1>
<p>FLOWR.ROOT · 案例 {html.escape(summary['case'])} · {len(plan['seeds'])} 条选定轨迹 · {summary['n_replays']} 个真实网络调用状态 · 固定位点 {plan['site']}</p>
<p class="note"><strong>这一步已完成内部记录的离线来源分析，可以作为阶段性汇报材料。</strong>这里提供异常连接的时间、位置、局部贡献和候选计算来源。尚未证明最终失败的因果根源，也没有完成输入片段原子的 SHAP 贡献分析。</p>
<h2>1. 本次实际分析了什么</h2><p>新记录完整保存所有原子槽位的逐层三维隐藏向量、内部坐标、最终归一化及坐标投影。全部状态与原始生成结果一致，最大记录差异为 {summary['max_native_replay_difference']}。计算覆盖当前位点对所有有效生成原子的 {summary['head_pair_transitions']} 条相邻调用转换；缺失 {summary['uncovered_transitions']} 条。它不是对所有固定位点或所有失败类型的完整扫描。</p>
<h2>2. 窗口内自动检出的键类别变化</h2><p>调用编号从 0 开始；网络层编号从 1 开始。0=无键，1=单键。变化本身不是失败标准，应结合片段条件、价态与最终结构评价。距离缩短不自动判错，新骨架也无需等于参考骨架。</p>{event_table}
<p>正项表示为距离平方增加分配的量，负项表示为距离平方减小分配的量。数值差异和浮点加法舍入单独保留在 CSV，不合并到隐藏向量或归一化项。</p>
<h2>3. 怎样进入成键得分</h2><p>下表沿用已验证的头部参考分解，指标为“目标键类相对无键的 logit”。它与上表的距离平方单位不同，不能直接相加。这些是参考路径下的局部分配值，不是百分比。</p>{head_table}
<h2>4. 成功对照、恢复阶段与逐层记录</h2><p>图中仅显示配置指定的原子对，CSV 包含全部已覆盖原子对。不同随机种子的生成原子编号不代表同一个化学原子，各自与自己的固定位点比较。</p>{imgs}
<p class="limit"><strong>逐层柱状图不能读成“某层造成了错误”。</strong>它把最终隐藏向量变化按实际记录的每层净更新进行代数分配；后层更新依赖前层输入，内部坐标更新还可能重置隐藏向量。最大项仅用于提出候选位置。分配闭合也不等于因果解释正确。</p>
<h2>5. 方法与复核</h2><p>令 E 为归一化前隐藏向量，s 为固定归一化权重、固定投影权重和当前全局长度尺度合成的系数。参考读出 q=Σ(E·s)。两个调用之间用对称分配 Δq=Σ[平均(s)·ΔE]+Σ[平均(E)·Δs]。再用 Δ|r|²=(r后+r前)·Δr 核算距离平方变化。层内净更新先相加再投影，保持精确代数关系。这里不把固定投影权重称为发生了变化。</p>
<p>参考归一化、投影、最终加法的最大数值差异分别为 {summary['maximum_reference_errors']['norm_max_abs_difference']:.3g}、{summary['maximum_reference_errors']['projection_max_abs_difference']:.3g}、{summary['maximum_reference_errors']['addition_max_abs_difference']:.3g}（模型单位）。距离平方分解最大残差 {summary['max_readout_accounting_residual_A2']:.3g} Å²。头部 {summary['head_precision_flagged_transitions']} 条转换保留精度提示；读出 {summary['readout_precision_flagged_transitions']} 条有提示，均见完整 CSV。</p>
<h2>6. 当前可以暂停的位置</h2><p>可汇报“小样本初步内部机制分析”：已追踪异常时间和位置，并量化相关内部计算路径的变化。关于改哪一层、哪个时间段会改变后续连接的预测，尚需将来受控干预验证。输入原子归因和独立复合物验证也是后续任务，不必为本次阶段汇报立即开展。</p>
<p>程序从记录读取种子、时间、原子映射、维度和层数；这说明接口可复用，不证明机制泛化。当前仅一个复合物，且观察窗口与展示对象是事后选定，不能用这些结果估计总体检出率或因果效果。</p></html>'''
    (out/'REPORT.html').write_text(doc,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head-dir",type=Path,required=True)
    p.add_argument("--hidden-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--head-script",type=Path,default=Path("trace_connection_head_v2.py"))
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--capture-readout",action="store_true",help="Also replay original states and capture all-node coordinate readout data, without interventions.")
    mode.add_argument("--readout-dir",type=Path,help="Analyze a completed full-node capture offline; no generator or GPU required.")
    p.add_argument("--output-root",type=Path,default=Path("outputs"))
    args=p.parse_args()
    out=args.output_dir.resolve()
    if out.exists():p.error("Keep existing results; choose a new --output-dir.")
    if not args.head_script.is_file():p.error("Run beside the verified trace_connection_head_v2.py, or provide --head-script.")
    spec=importlib.util.spec_from_file_location("head_reference",args.head_script.resolve())
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    out.mkdir(parents=True)
    success=False
    try:
        if args.readout_dir:
            summary=analyze_readout(args.head_dir.resolve(),args.hidden_dir.resolve(),args.readout_dir.resolve(),out,ref)
        else:
            summary=analyze(args.head_dir.resolve(),args.hidden_dir.resolve(),out,ref)
        print(f"Verified states: {summary['n_replays']}")
        print(f"Coordinate coverage: {summary['geometry_source_transitions']} / {summary['head_pair_transitions']} transitions")
        print(f"Bond-class changes: {summary['bond_class_change_events']}")
        print(f"Report: {out/'REPORT.html'}")
        if args.capture_readout:
            captured=capture_readout(args,out,ref)
            require(captured["completed"],"Coordinate capture did not complete.")
            print(f"Readout capture verified: {len(captured['states'])} states",flush=True)
        success=True
    except Exception as exc:
        (out/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        print(f"STOP: {type(exc).__name__}: {exc}")
    finally:
        bundle=out/"source_review.zip"
        with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
            for path in sorted(out.rglob("*")):
                if path.is_file() and path!=bundle and "model_context" not in path.relative_to(out).parts:
                    z.write(path,path.relative_to(out))
        print(f"Review file: {bundle}")
        print(f"Completed: {success}; observational analysis, not causal validation")
    return 0 if success else 1


if __name__=="__main__":
    raise SystemExit(main())
