#!/usr/bin/env python3
"""Analyze saved evidence without model inference or dependency changes."""
import argparse
import sys
import zipfile
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scaffold_study_v1 import __version__
from scaffold_study_v1.io_utils import read_json, write_json, write_csv, sha
from scaffold_study_v1.core import analyze
from scaffold_study_v1.flowr_adapter import load
from scaffold_study_v1.protocol import final_constraint_plan


def render(report):
    lines = ['# 统一诊断结果', '', f"处理完成：{report['n_runs']} 条轨迹，{report['n_input_groups']} 个输入组。",
             '原始解码和修复解码各评价一次，不能算成双倍生成样本。', '',
             '| 原始输出检查 | 通过 | 未通过 | 未评价 |', '|---|---:|---:|---:|']
    names = {'chemical_valid': '化学有效', 'connected': '连通', 'fixed_chemistry_preserved': '固定化学结构保持',
             'fixed_substructure_match': '原生固定子结构匹配', 'geometry_valid': '三维检查', 'scaffold_changed': 'Murcko骨架改变'}
    for metric, c in report['counts']['raw'].items():
        lines.append(f"| {names[metric]} | {c['pass']} | {c['fail']} | {c['unknown']} |")
    lines += ['', '读取 findings.csv 查看逐项问题；diagnosis.json 保留原子对、原生时间、前一帧是否可判定及所有细节。',
              '终态异常的持续区间只是观察，不是根因时刻。原始中间状态带噪，不按最终合法分子的标准判失败。',
              '固定原子的末两帧位移保留为数值；没有自行增加通过阈值。',
              '没有执行神经网络干预，没有证明任何内部模块是根因。',
              '完整 scaffold-hopping 成功率暂不合成：还需预先固定完整任务要求和坐标容差。', '',
              '## 本轮停止点', '', '到本报告为止。不会自动生成分子、追加种子、选择网络层或修改旧结果。',
              '后续按照 EXPERIMENT_RULES_CN.md 固定一个失败现象和有限实验预算。']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--quality-report', type=Path)
    source.add_argument('--normalized-input', type=Path, help='Model-neutral scaffold-evidence/1 JSON')
    p.add_argument('--diagnostic-dir', type=Path)
    p.add_argument('--geometry-dir', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--plan-final-constraint', nargs='+', metavar='INPUT_ID')
    p.add_argument('--max-continuations', type=int, default=40)
    a = p.parse_args(argv)
    if a.output_dir.exists():
        p.error('Output exists. Use a new directory; old results will not be overwritten.')
    if a.quality_report and not a.diagnostic_dir:
        p.error('--quality-report requires --diagnostic-dir')
    if a.normalized_input and (a.diagnostic_dir or a.geometry_dir):
        p.error('Normalized input cannot be mixed with FLOWR report paths')
    try:
        dataset = read_json(a.normalized_input) if a.normalized_input else load(a.quality_report, a.diagnostic_dir, a.geometry_dir)
        report = analyze(dataset)
        plan = final_constraint_plan(dataset, a.plan_final_constraint, a.max_continuations) if a.plan_final_constraint else None
        # Validate completely before creating any output. No partial result may look complete.
        out = a.output_dir.resolve()
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/'diagnosis.json', report)
        write_json(out/'normalized_evidence.json', dataset)
        rows = []
        for r in report['runs']:
            for i in r['issues']:
                rows.append({'run_id': r['run_id'], 'input_id': r['input_id'], 'seed': r['seed'],
                             'endpoint_stage': i['endpoint_stage'], 'problem': i['problem'],
                             'evidence_level': i['evidence_level'], 'locations': i['locations'],
                             'temporal_evidence': i['temporal_evidence'], 'internal_module': None,
                             'causal_status': 'not_tested'})
        write_csv(out/'findings.csv', rows)
        write_csv(out/'sample_inventory.csv', [dict(run_id=r['run_id'], input_id=r['input_id'], seed=r['seed'],
                  input_group=r['input_group'], raw=r['endpoints']['raw'], decoded=r['endpoints']['decoded']) for r in report['runs']])
        (out/'REPORT_CN.md').write_text(render(report), encoding='utf-8')
        if plan:
            write_json(out/'final_constraint_plan.json', plan)
        write_json(out/'provenance.json', {'package_version': __version__,
            'inputs': dataset.get('provenance', {}),
            'normalized_input_sha256': sha(a.normalized_input) if a.normalized_input else None,
            'code': {f.name: sha(f) for f in Path(__file__).parent.glob('*.py')},
            'outputs': {f.name: sha(f) for f in out.iterdir() if f.is_file()},
            'geometry_supplied': bool(a.geometry_dir) if not a.normalized_input else 'see_normalized_evidence',
            'verification_scope': 'Report identities and saved timelines checked; original tensors and GPU execution not replayed.'})
        with zipfile.ZipFile(out/'study_review.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.iterdir()):
                if f.suffix in ('.json', '.csv', '.md'):
                    z.write(f, f.name)
        print(f"Completed: True; {report['n_runs']} runs; {report['n_input_groups']} input groups")
        print('Causal interventions: 0; root causes: not established')
        if a.quality_report and not a.geometry_dir:
            print('Geometry pair table not supplied; geometry failures remain spatially unresolved.')
        if plan:
            print(f"Plan saved: {plan['planned_continuations']} continuations; NOT executed")
        print(f'Review: {out / "study_review.zip"}')
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
