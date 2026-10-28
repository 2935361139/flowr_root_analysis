#!/usr/bin/env python3
"""Read-only checkpoint/configuration audit. No model construction or generation.

Standalone: Python standard library and the project's existing PyTorch only.
Writes exclusively to a newly created audit directory; never edits source/input.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import traceback
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

VERSION = '1.0.0'
EXPECTED_SHA256 = 'b818f41dc12ffb6bc558bb0ad997055581e07cd9e49dcac1b794ed9993c46e4c'
CASE = 'ptp1b_lig3_seed_47'
RECORD = Path('outputs/sh_discovery_cohort_v2/runs/ptp1b_lig3/seed_47/research_record')
EXPECTED_SOURCE = {
    'flowr/gen/generate_from_pdb.py': 'c562ea9183fb31dbe0424ed6688f7a258af38927cd2227ae8a43cfb07358dad8',
    'flowr/gen/generate.py': 'cb397a0d903e447cec873826300b562b9e7e2666daaad1e5f1cd42178e4e61f9',
    'flowr/models/fm_pocket.py': 'beaace3ab92a8377bc8eab95333c8f40ce370ca9224f7a78927c5a424bb38426',
    'flowr/models/mol_builder.py': '30df0c63e75d4350b9d38d97a13f67ba14cf8a45b847faf257a34ddfb3d035a3',
    'flowr/data/interpolate.py': '171af7162fe2b7fc37e8ffdcbf90f6ea5c99bebed0b17a3d825ef24dee7cdd96',
    'flowr/scriptutil.py': '94422ce28c701966cd54f1d46b4bbb22438204a38789a3e897301f4fb71da6ff',
}
SOURCE_FILES = list(EXPECTED_SOURCE) + [
    'flowr/models/losses.py', 'flowr/models/pocket.py', 'flowr/models/pocket_util.py',
    'flowr/models/integrator.py', 'flowr/util/functional.py', 'flowr/train.py',
    'record_scaffold_run.py', 'scaffold_evaluation_v2/evaluate_scaffold_outputs.py',
    'pyproject.toml', 'uv.lock',
]
FIELDS = [
    'mask_fixed_discrete_loss', 'type_loss_weight', 'bond_loss_weight',
    'charge_loss_weight', 'coord_loss_weight', 'hybridization_loss_weight',
    'interaction_loss_weight', 'bond_length_loss_weight', 'bond_angle_loss_weight',
    'dihedral_loss_weight', 'energy_loss_weight', 'use_t_loss_weights',
    'use_velocity_loss', 'type_strategy', 'bond_strategy',
    'self_condition', 'self_cond', 'self_condition_mode', 'inpaint_self_condition',
    'self_condition_prob', 'sc_charges', 'sc_distance_edges', 'sc_recycle_steps',
    'sc_schedule', 'sc_warmup_steps', 'use_is_fixed_embed', 'use_inpaint_mode_embed',
    'graph_inpainting', 'scaffold_hopping', 'fragment_inpainting',
    'mixed_uncond_inpaint', 'coord_scale', 'pocket_noise', 'add_feats',
    'use_ema', 'architecture', 'd_inv', 'd_equi', 'd_model', 'd_edge',
    'n_layers', 'n_attn_heads', 'emb_size', 'categorical_strategy',
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def plain(value, depth=0):
    """Never turn an omitted/opaque value into a guessed bool or default."""
    if depth > 12:
        return {'unread_value': 'nesting_limit'}
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {'nonfinite_float': str(value)}
    if isinstance(value, argparse.Namespace):
        return plain(vars(value), depth + 1)
    if isinstance(value, Mapping):
        return {str(k): plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 500:
            return {'unread_value': 'sequence_too_large', 'length': len(value)}
        return [plain(v, depth + 1) for v in value[:500]]
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        return {'unread_value': 'tensor_or_array', 'shape': list(value.shape),
                'dtype': str(value.dtype)}
    return {'unread_value': 'unsupported_type',
            'type': type(value).__module__ + '.' + type(value).__qualname__}


def command(repo, args):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    try:
        p = subprocess.run(args, cwd=repo, env=env, capture_output=True,
                           text=True, errors='replace', timeout=30)
        return {'returncode': p.returncode, 'stdout': p.stdout[:150000],
                'stderr': p.stderr[:12000], 'truncated': len(p.stdout) > 150000}
    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'}


class Files:
    def __init__(self, repo):
        self.repo = repo
        r = command(repo, ['git', 'rev-parse', '--git-path', 'lfs/objects'])
        p = Path(r.get('stdout', '').strip() or '.git/lfs/objects')
        self.lfs = p if p.is_absolute() else repo / p

    def inspect(self, name, checksum=False):
        p = self.repo / name
        row = {'path': str(name), 'status': 'missing', 'content_verified': False}
        if not p.is_file():
            return row, None
        row['size_bytes'] = p.stat().st_size
        with p.open('rb') as f:
            head = f.read(1024)
        actual = p
        if head.startswith(b'version https://git-lfs.github.com/spec/v1'):
            try:
                lines = dict(line.split(' ', 1) for line in head.decode().splitlines())
                oid = lines['oid'].removeprefix('sha256:')
                if len(oid) != 64 or any(c not in '0123456789abcdef' for c in oid):
                    raise ValueError('Malformed LFS object ID')
                actual = self.lfs / oid[:2] / oid[2:4] / oid
                row.update(lfs_sha256=oid, expected_bytes=int(lines['size']))
                if not actual.is_file():
                    row['status'] = 'lfs_pointer_only'
                    return row, None
                if actual.stat().st_size != int(lines['size']):
                    row['status'] = 'lfs_size_mismatch'
                    return row, None
                row['status'] = 'available_in_local_lfs'
            except (ValueError, KeyError, UnicodeError) as e:
                row.update(status='invalid_lfs_pointer', error=str(e))
                return row, None
        else:
            row['status'] = 'available' if row['size_bytes'] else 'empty'
        if checksum:
            row['sha256'] = sha256(actual)
            if 'lfs_sha256' in row and row['sha256'] != row['lfs_sha256']:
                row['status'] = 'lfs_hash_mismatch'
                return row, None
            row['content_verified'] = 'lfs_sha256' in row
        return row, actual if row['status'] != 'empty' else None


def metadata_load(torch, path):
    # Never construct the model, load tensor storage into GPU, import project
    # classes, or fall back to unrestricted pickle execution.
    options = {'map_location': 'meta', 'weights_only': True}
    if zipfile.is_zipfile(path):
        options['mmap'] = True
    with torch.serialization.safe_globals([argparse.Namespace, torch.torch_version.TorchVersion]):
        return torch.load(path, **options)


def extract_config(payload):
    if not isinstance(payload, Mapping):
        raise ValueError('Checkpoint root is not a mapping')
    roots = {}
    for name in ('hyper_parameters', 'hparams'):
        if name in payload:
            value = payload[name]
            if isinstance(value, argparse.Namespace):
                value = vars(value)
            if not isinstance(value, Mapping):
                raise ValueError(f'{name} is not a readable mapping/Namespace')
            roots[name] = value
    occurrences = {}

    def walk(value, path, depth=0):
        if isinstance(value, argparse.Namespace):
            value = vars(value)
        if not isinstance(value, Mapping) or depth > 8:
            return
        for k, v in value.items():
            if not isinstance(k, str):
                continue
            location = path + [k]
            if k in FIELDS or k.endswith('_loss_weight') or k.startswith('distance_loss_weight_'):
                occurrences.setdefault(k, []).append({'path': location, 'value': plain(v)})
            if k not in ('state_dict', 'optimizer_states'):
                walk(v, location, depth + 1)

    for name, value in roots.items():
        walk(value, [name])
    fields = {}
    for k in list(dict.fromkeys(FIELDS + sorted(occurrences))):
        values = occurrences.get(k, [])
        fields[k] = {'status': 'recorded' if values else 'not_recorded',
                     'label_cn': '已记录' if values else '未记录', 'occurrences': values}
        if len(values) > 1:
            fields[k]['multiple_locations'] = True
            fields[k]['conflicting_values'] = any(v['value'] != values[0]['value'] for v in values[1:])
    state = payload.get('state_dict', {})
    shapes = []
    if isinstance(state, Mapping):
        for name, value in state.items():
            if isinstance(name, str) and any(s in name for s in (
                '.inv_emb.atom_emb.', '.is_fixed_node_embed.', '.is_fixed_position_embed.',
                '.atom_type_proj.', '.inv_emb.bond_proj.')):
                shapes.append({'name': name, 'metadata': plain(value)})
    return {'status': 'read' if roots else 'no_hyperparameter_mapping',
            'roots_present': list(roots), 'fields': fields,
            'selected_parameter_shapes': shapes[:100],
            'interpretation': 'Only saved fields. No defaults or inferred training settings. Parameter shapes are architecture evidence, not training-supervision evidence.'}


def inspect_checkpoint(torch, path, expected_sha=EXPECTED_SHA256):
    result = {'expected_sha256': expected_sha, 'actual_sha256': sha256(path)}
    result['hash_matches'] = result['actual_sha256'] == expected_sha
    if not result['hash_matches']:
        result.update(status='hash_mismatch_not_loaded', configuration={'status': 'not_read'})
        return result
    try:
        payload = metadata_load(torch, path)
        result.update(status='inspected', configuration=extract_config(payload))
        del payload
    except Exception as e:
        result.update(status='metadata_read_failed', error=f'{type(e).__name__}: {e}',
                      traceback=traceback.format_exc(), configuration={'status': 'not_read'})
        try:
            result['unsupported_pickle_globals'] = torch.serialization.get_unsafe_globals_in_checkpoint(path)
        except Exception as scan_error:
            result['global_scan_error'] = str(scan_error)
    return result


def collect_sources(files, out):
    rows = []
    for name in SOURCE_FILES:
        row, path = files.inspect(name, checksum=True)
        if name in EXPECTED_SOURCE:
            row['original_generation_manifest_sha256'] = EXPECTED_SOURCE[name]
            row['matches_original_generation_source'] = row.get('sha256') == EXPECTED_SOURCE[name] if path else None
        if path and Path(name).suffix in ('.py', '.toml') and path.stat().st_size <= 2 * 1024 * 1024:
            target = out / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            row['snapshot'] = str(target.relative_to(out))
        rows.append(row)
    return rows


def record_inventory(files, torch=None):
    base = RECORD / 'attempt_000'
    names = [RECORD/'manifest.json', RECORD/'modules.json']
    names += [base/n for n in ('inputs.pt', 'pocket_encoding.pt', 'raw_final_prediction.pt',
                              'reference_batch.pt', 'task_definition.json', 'reference_preprocessed.sdf')]
    names += [Path('examples/ptp1b_protein.pdb'), Path('examples/ptp1b_ligands.sdf')]
    names += [base/'forward_calls'/f'call_{i:04d}.pt' for i in range(101)]
    names += [base/'sampler_steps'/f'step_{i:04d}.pt' for i in range(101)]
    selected = {'inputs.pt', 'pocket_encoding.pt', 'raw_final_prediction.pt',
                'call_0000.pt', 'call_0008.pt', 'call_0010.pt', 'call_0100.pt',
                'step_0099.pt', 'step_0100.pt'}
    rows, selected_meta = [], {}
    for name in names:
        important = name.name in selected or name.suffix != '.pt'
        row, path = files.inspect(name, checksum=important)
        rows.append(row)
        if path and torch is not None and name.name in selected:
            try:
                obj = metadata_load(torch, path)
                meta = {'status': 'read', 'keys': list(obj) if isinstance(obj, Mapping) else None}
                if name.name == 'inputs.pt' and isinstance(obj, Mapping):
                    meta['saved_generation_settings'] = {k: plain(obj[k]) for k in (
                        'self_condition', 'sc_recycle_steps', 'final_inpaint', 'solver', 'strategy', 'coord_scale') if k in obj}
                    meta['rng_keys'] = list(obj.get('rng', {}))
                    meta['pocket_data_keys'] = list(obj.get('pocket_data', {}))
                    meta['pocket_complex'] = plain(obj.get('pocket_data', {}).get('complex'))
                    meta['note'] = 'Saved inference settings are NOT checkpoint training settings. Object placeholders are not directly replayable.'
                selected_meta[str(name)] = meta
                del obj
            except Exception as e:
                selected_meta[str(name)] = {'status': 'read_failed', 'error': f'{type(e).__name__}: {e}'}
    return {'files': rows, 'selected_metadata': selected_meta,
            'available_files': sum(r['status'] in ('available', 'available_in_local_lfs') for r in rows),
            'expected_files': len(rows),
            'scope': 'Presence/LFS check plus selected tensor metadata only; not full replay validation. No download performed.'}


def render_report(report):
    ck = report.get('checkpoint', {})
    lines = ['# 原检查点检查结果', '', f"案例：{CASE}",
             f"工具状态：{report['status']}",
             '本次生成0次，重放0次，干预0次；未验证任何内部机制。', '',
             f"权重SHA256匹配：{ck.get('hash_matches', '未完成')}",
             f"实际SHA256：{ck.get('actual_sha256', '未完成')}",
             f"读取状态：{ck.get('status', '未完成')}", '',
             '## 检查点实际记录的配置', '',
             'False、0、null均保留原值；未记录不等于False。重复位置不自动合并。', '',
             '| 字段 | 状态 | 原始记录 |', '|---|---|---|']
    fields = ck.get('configuration', {}).get('fields', {})
    for k in FIELDS:
        field = fields.get(k)
        if field is None:
            lines.append(f'| {k} | 无法读取（不判断是否记录） | — |')
        else:
            value = json.dumps(field['occurrences'], ensure_ascii=False).replace('|', '\\|') if field['occurrences'] else '—'
            lines.append(f"| {k} | {field['label_cn']} | {value} |")
    inv = report.get('records', {})
    lines += ['', '## 后续所需记录', '',
              f"实际可见文件：{inv.get('available_files', '未完成')}/{inv.get('expected_files', '未完成')}。",
              '文件可见不代表已验证内容正确或可重放。Git LFS指针与本地真实对象分开记录。',
              '', '## 说明', '',
              '本报告不判断该案例是否适合干预；待审阅原始字段及源码后再决定。',
              '源码快照属于你当前工作区；未被原生成manifest覆盖的训练源码不能冒充原训练版本。',
              '检查点中没有保存的字段不能通过当前默认值补全。',
              '如读取失败或发现不匹配，不要改配置绕过检查；把整个结果ZIP上传回来。',
              'Git状态在创建本次结果目录之前采集；未提交修改不会被清理或覆盖。',
              '本包仅含报告、日志、文件清单和相关源码，不含模型权重、轨迹张量或分子数据。', '',
              '## 需要关注的问题', '']
    lines += [f'- {x}' for x in report.get('issues', [])] or ['- 未发现工具错误；这不代表模型监督或重放已验证。']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/flowr_root_v2.2.ckpt'))
    args = parser.parse_args(argv)
    repo = args.repo.expanduser().resolve()
    if not (repo/'flowr').is_dir():
        parser.error('请在FLOWR项目根目录执行，或用 --repo 指定目录；没有写入文件。')
    git = {name: command(repo, cmd) for name, cmd in {
        'head': ['git', 'rev-parse', 'HEAD'], 'branch': ['git', 'branch', '--show-current'],
        'status': ['git', 'status', '--short', '--untracked-files=normal'],
        'unstaged_numstat': ['git', 'diff', '--no-ext-diff', '--numstat'],
        'staged_numstat': ['git', 'diff', '--no-ext-diff', '--cached', '--numstat'],
    }.items()}
    parent = repo/'outputs'
    parent.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='checkpoint_audit_' + datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=parent))
    report = {'tool_version': VERSION, 'tool_sha256': sha256(Path(__file__)),
              'created_utc': datetime.now(timezone.utc).isoformat(), 'repo': str(repo),
              'case': CASE, 'status': 'running', 'issues': [], 'git': git,
              'generation_count': 0, 'replay_count': 0, 'intervention_count': 0,
              'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform()}}
    log = []

    def note(message):
        print(message, flush=True)
        log.append(message)

    def section(name, fn):
        try:
            report[name] = fn()
        except Exception as e:
            report['issues'].append(f'{name}: {type(e).__name__}: {e}')
            log.append(traceback.format_exc())

    note('只读检查开始，不加载生成模型，不运行实验。')
    note('输出目录：' + str(out))
    try:
        files = Files(repo)
        section('sources', lambda: collect_sources(files, out))
        if any(r['path'].endswith('.py') and r['status'] not in ('available', 'available_in_local_lfs')
               for r in report.get('sources', [])):
            report['issues'].append('部分相关源码不存在或未取得真实内容，详见sources。')
        torch = None
        try:
            import torch as torch_module
            torch = torch_module
            report['environment'].update(torch=str(torch.__version__), torch_cuda_build=torch.version.cuda,
                                         cuda_available=torch.cuda.is_available(), cuda_device_count=torch.cuda.device_count())
            report['environment']['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except Exception as e:
            report['issues'].append(f'PyTorch环境检查失败：{type(e).__name__}: {e}')
            log.append(traceback.format_exc())
        checkpoint = args.checkpoint.expanduser()
        if not checkpoint.is_absolute():
            checkpoint = repo/checkpoint
        row, actual = files.inspect(checkpoint)
        report['checkpoint_file'] = row
        if actual is None:
            report['checkpoint'] = {'status': row['status'], 'expected_sha256': EXPECTED_SHA256}
        elif torch is None:
            actual_sha = sha256(actual)
            report['checkpoint'] = {'status': 'torch_unavailable_not_loaded', 'expected_sha256': EXPECTED_SHA256,
                                    'actual_sha256': actual_sha, 'hash_matches': actual_sha == EXPECTED_SHA256}
        else:
            note('正在核对约1GB权重的SHA256并读取元数据，请等待。')
            section('checkpoint', lambda: inspect_checkpoint(torch, actual))
        ck = report.get('checkpoint', {})
        if ck.get('status') != 'inspected':
            report['issues'].append('检查点没有完成读取：' + ck.get('status', 'unknown'))
            if ck.get('error'):
                log.append(ck['error'] + '\n' + ck.get('traceback', ''))
        elif ck['configuration']['status'] != 'read':
            report['issues'].append('检查点没有可读的hyper_parameters/hparams；无法确认训练设置。')
        note('正在检查指定案例记录的存在性；不重放轨迹。')
        section('records', lambda: record_inventory(files, torch if ck.get('hash_matches') else None))
        inv = report.get('records', {})
        if inv.get('available_files') != inv.get('expected_files'):
            report['issues'].append('部分原始记录缺失或只有LFS指针，详见records.files。')
        if any(v.get('status') == 'read_failed' for v in inv.get('selected_metadata', {}).values()):
            report['issues'].append('部分保存状态的元数据读取失败，详见records.selected_metadata。')
        if any(r.get('matches_original_generation_source') is False for r in report.get('sources', [])):
            report['issues'].append('部分当前源码与原生成manifest不一致，已保留版本信息；不自动修改。')
        if any(r.get('returncode') != 0 for r in git.values()):
            report['issues'].append('部分Git版本检查未完成，详见git字段。')
    except (Exception, KeyboardInterrupt) as e:
        report['issues'].append(f'检查中断：{type(e).__name__}: {e}')
        log.append(traceback.format_exc())
    report['status'] = 'completed_with_issues' if report['issues'] else 'completed'
    write_json(out/'audit.json', report)
    (out/'REPORT_CN.md').write_text(render_report(report), encoding='utf-8')
    (out/'run.log').write_text('\n'.join(log) + '\n', encoding='utf-8')
    fields = report.get('checkpoint', {}).get('configuration', {}).get('fields', {})
    with (out/'configuration.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['field', 'status', 'saved_occurrences'])
        for key, value in fields.items():
            w.writerow([key, value['status'], json.dumps(value['occurrences'], ensure_ascii=False)])
    contents = sorted(p for p in out.rglob('*') if p.is_file())
    manifest = out/'SHA256SUMS'
    manifest.write_text(''.join(f'{sha256(p)}  {p.relative_to(out).as_posix()}\n' for p in contents), encoding='utf-8')
    bundle = out/'checkpoint_audit_result.zip'
    with zipfile.ZipFile(bundle, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in contents + [manifest]:
            z.write(p, p.relative_to(out))
    print('检查完成：' + report['status'], flush=True)
    print('请上传这个文件：' + str(bundle), flush=True)
    print('生成/重放/干预次数：0/0/0。发现问题时保留ZIP和终端输出，不修改配置绕过。', flush=True)
    return 2 if report['issues'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
