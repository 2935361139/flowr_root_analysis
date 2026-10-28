"""FLOWR report adapter. No module names, bond heads or layer numbers enter core.py."""
from collections import defaultdict
from pathlib import Path
import math
import json
from .io_utils import boolean, read_json, read_csv, require, sha
from .core import SCHEMA

STAGES = {'raw_without_repair': 'raw', 'run_decode': 'decoded'}


def endpoint(row):
    changed = set(row.get('fixed_missing_sites', []))
    for key in ('fixed_element_changed_sites', 'fixed_charge_changed_sites'):
        changed.update(row.get(key, []))
    for pair in row.get('fixed_internal_bond_changed_pairs', []):
        changed.update(pair)
    return {'chemical_valid': row.get('rdkit_valid'),
            'connected': row.get('raw_graph_connected') if row['stage'] == 'raw_without_repair' else
                         row.get('fc_valid') if row.get('rdkit_valid') is True else None,
            'fixed_chemistry_preserved': row.get('fixed_chemistry_preserved'),
            'fixed_substructure_match': row.get('root_fixed_substructure_match'),
            'geometry_valid': row.get('posebusters_valid'),
            'geometry_failed_checks': row.get('posebusters_failed_checks', []),
            'scaffold_changed': row.get('murcko_scaffold_changed'),
            'fixed_changed_atoms': sorted(changed),
            'neutral_carbon_overvalent_atoms': row.get('fixed_neutral_carbon_overvalent_sites', []),
            'fixed_max_displacement_A': row.get('fixed_max_displacement_A'),
            'native_chemical_connection_endpoint': row.get('chemical_connection_endpoint')}


def finite(value):
    x = float(value)
    require(math.isfinite(x), 'Nonfinite numeric observation')
    return x


def observations(folder, report):
    site_rows, graph_rows = read_csv(folder/'connection_timeline.csv'), read_csv(folder/'graph_timeline.csv')
    fixed = report['task_definition']['fixed_sampler_indices']
    require(len(set(fixed)) == len(fixed) and fixed, 'Invalid fixed-atom map')
    series, displacement = [], []
    require(set(r['stream'] for r in site_rows) == set(report['streams']), 'Site streams missing/unrecognized')
    require(set(r['stream'] for r in graph_rows) == set(report['streams']), 'Graph streams missing/unrecognized')
    for stream, metadata in report['streams'].items():
        require(stream in ('network_prediction', 'sampler_state'), 'Unknown FLOWR stream')
        indices = metadata['indices']
        require(indices == list(range(len(indices))) and indices, 'Invalid saved index inventory')
        require(len(metadata['stages']) == len(indices) == len(metadata['times']), 'Invalid stage/time inventory')
        sr = [r for r in site_rows if r['stream'] == stream]
        gr = [r for r in graph_rows if r['stream'] == stream]
        require(len(sr) == len(indices)*len(fixed), 'Incomplete or duplicate site timeline')
        require({int(r['fixed_sampler_index']) for r in sr} == set(fixed), 'Site IDs differ from verified task map')

        def points(rows, field, invert=False):
            require([int(r['index']) for r in rows] == indices, 'Timeline truncated, duplicated or out of order')
            out = []
            for j, r in enumerate(rows):
                require(r['stage'] == metadata['stages'][j], 'Timeline stage differs from recorded stream')
                time = finite(r['continuous_time']) if r['continuous_time'] else None
                expected_time = metadata['times'][j]
                require(time == expected_time, 'Timeline native time differs from stream inventory')
                v = boolean(r[field])
                out.append({'index': j, 'native_time': time, 'stage': r['stage'],
                            'value': None if v is None else not v if invert else v})
            return out

        series.append({'stream': stream, 'entity': 'whole_molecule', 'metric': 'disconnected',
                       'points': points(gr, 'connected', True)})
        for atom in fixed:
            rows = [r for r in sr if int(r['fixed_sampler_index']) == atom]
            for field, name in [('fixed_chemistry_changed', 'fixed_chemistry_changed'),
                                ('neutral_carbon_overvalent', 'neutral_carbon_overvalence')]:
                series.append({'stream': stream, 'entity': f'fixed_atom:{atom}', 'metric': name,
                               'points': points(rows, field)})
            # A measured displacement is not a failure without a declared tolerance.
            displacement.append({'stream': stream, 'atom': atom,
                                 'penultimate': {'index': int(rows[-2]['index']), 'stage': rows[-2]['stage'],
                                                 'distance_A': finite(rows[-2]['fixed_displacement_A'])} if len(rows)>1 else None,
                                 'terminal': {'index': int(rows[-1]['index']), 'stage': rows[-1]['stage'],
                                              'distance_A': finite(rows[-1]['fixed_displacement_A'])},
                                 'failure_threshold_A': None, 'causal_status': 'not_tested'})
    return series, displacement


def load(quality_path, diagnostic_dir, geometry_dir=None):
    quality_path, diagnostic_dir = Path(quality_path).resolve(), Path(diagnostic_dir).resolve()
    evidence = {}

    def track(path):
        path = Path(path)
        evidence[str(path)] = sha(path)
        return path

    q = read_json(track(quality_path))
    s = read_json(track(diagnostic_dir/'summary.json'))
    require(q.get('completed') is True and not q.get('errors'), 'Quality evaluation is incomplete or contains errors')
    require(s.get('completed') is True and all(r.get('analysis_completed') is True for r in s['records']), 'Diagnostic evaluation is incomplete')
    require(s['quality_report_sha256'] == sha(quality_path), 'Quality/diagnostic hashes disagree: do not mix batches')
    require(len(q['runs']) == q['requested_run_count'] == len(s['records']), 'Run inventories disagree')
    info_by_run = {r['run_directory']: r for r in q['runs']}
    require(len(info_by_run) == len(q['runs']), 'Duplicate quality run paths')
    by_run = defaultdict(dict)
    for row in q['rows']:
        require(row['run_directory'] in info_by_run and row['stage'] in STAGES, 'Unrecognized quality row')
        require(not row.get('errors'), 'Endpoint row contains errors')
        require(row['stage'] not in by_run[row['run_directory']], 'Duplicate endpoint stage')
        by_run[row['run_directory']][row['stage']] = row
    normalized, seen = [], set()
    for summary_record in s['records']:
        folder = (diagnostic_dir/summary_record['report_directory']).resolve()
        require(folder.is_relative_to(diagnostic_dir), 'Report path escapes diagnostic directory')
        report = read_json(track(folder/'connection_report.json'))
        run = report['run']
        require(run in info_by_run and run not in seen, 'Unmatched/duplicate diagnostic run path')
        seen.add(run)
        info, rows = info_by_run[run], by_run[run]
        require(set(rows) == set(STAGES), 'Missing raw/decoded endpoint')
        require(report['endpoint_evaluation'] == [rows[x['stage']] for x in report['endpoint_evaluation']]
                and len(report['endpoint_evaluation']) == 2
                and {x['stage'] for x in report['endpoint_evaluation']} == set(STAGES), 'Endpoint report changed since diagnosis')
        m = report['manifest']
        require(report['task_definition'].get('mapping_verified') is True, 'Unverified atom correspondence')
        require(report['seed'] == info['seed'] == m['args']['seed'] == summary_record['seed'], 'Seed mismatch')
        require(m['checkpoint_sha256'] == info['checkpoint_sha256'] and m['source_sha256'] == info['recorded_source_sha256'], 'Generator identity mismatch')
        require(m['input_sha256'] == info['input_sha256'], 'Input identity mismatch')
        for name in ('connection_timeline.csv', 'graph_timeline.csv'):
            track(folder/name)
        series, displacement = observations(folder, report)
        ligand = info.get('ligand_idx')
        require(type(ligand) is int, 'Missing ligand identity')
        input_id = f"{info['case']}_lig{ligand}"
        group = info['reference_sdf_sha256'] + ':' + m['input_sha256'][m['args']['pdb_file']]
        normalized.append({'run_id': f"{input_id}_seed_{info['seed']}", 'input_id': input_id,
                           'input_group': group, 'seed': info['seed'],
                           'source_record': {'original_run_directory': run, 'diagnostic_report': str(folder/'connection_report.json')},
                           'endpoints': {STAGES[k]: endpoint(v) for k, v in rows.items()},
                           'observations': series, 'fixed_displacement_observations': displacement,
                           'runtime_settings': {'final_inpaint': m['args'].get('final_inpaint')},
                           'generator_identity': {'checkpoint': m['checkpoint_sha256'], 'sources': m['source_sha256']},
                           'geometry': {}})
    require(seen == set(info_by_run), 'Not every evaluated run was imported')
    if geometry_dir is not None:
        g = Path(geometry_dir).resolve()
        gs, gp = read_json(track(g/'summary.json')), read_json(track(g/'protocol.json'))
        require(gs.get('completed') is True and gs.get('error_count') == 0, 'Geometry audit is incomplete')
        require(gp['quality_report_sha256'] == sha(quality_path) and gp['diagnostic_summary_sha256'] == sha(diagnostic_dir/'summary.json'), 'Geometry hashes do not match this batch')
        all_samples = read_csv(track(g/'sample_summary.csv'))
        all_pairs = read_csv(track(g/'clash_pairs.csv'))
        indexed = {r['run_id']: r for r in normalized}
        pairs = defaultdict(list)
        for p in all_pairs:
            require(p['id'] in indexed and p['stage'] in STAGES, 'Unmatched geometry atom pair')
            require(p['input_id'] == indexed[p['id']]['input_id'] and int(p['seed']) == indexed[p['id']]['seed'], 'Geometry pair identity mismatch')
            require(p['check'] in indexed[p['id']]['endpoints'][STAGES[p['stage']]]['geometry_failed_checks'], 'Clash pair refers to a check that did not fail')
            for key in ('sampler_atom_index', 'ligand_sdf_atom_index', 'environment_atom_index'):
                p[key] = int(p[key])
                require(p[key] >= 0, 'Negative geometry atom index')
            for key in ('distance_A', 'relative_distance', 'native_clash_cutoff'):
                p[key] = finite(p[key])
                require(p[key] >= 0, 'Negative geometry distance/cutoff')
            for key in ('reference_check_pass', 'is_reference_attachment_atom'):
                p[key] = boolean(p[key])
            pairs[p['id'], p['stage']].append(p)
        used = set()
        for sample in all_samples:
            key = sample['id'], sample['stage']
            require(key not in used and sample['id'] in indexed and sample['stage'] in STAGES, 'Unmatched/duplicate geometry endpoint')
            used.add(key)
            r = indexed[sample['id']]
            stage = STAGES[sample['stage']]
            require(sample['input_id'] == r['input_id'] and int(sample['seed']) == r['seed'], 'Geometry sample identity mismatch')
            require(boolean(sample['posebusters_valid']) == r['endpoints'][stage]['geometry_valid'], 'Geometry endpoint differs from quality report')
            require(json.loads(sample['failed_checks']) == r['endpoints'][stage]['geometry_failed_checks'], 'Geometry failed checks differ from quality report')
            require(boolean(sample['fixed_chemistry_preserved']) == r['endpoints'][stage]['fixed_chemistry_preserved'], 'Geometry fixed-chemistry label differs')
            require(len(pairs[key]) == int(sample['n_native_clash_pairs']), 'Clash pair table incomplete')
            r['geometry'][stage] = {'clash_pairs': pairs[key], 'reference_context': sample}
        expected = {(r['run_id'], native) for r in normalized for native, stage in STAGES.items() if r['endpoints'][stage]['chemical_valid'] is True}
        require(used == expected and set(pairs).issubset(used), 'Geometry sample inventory incomplete')
        require(gs['requested_runs'] == len(normalized), 'Geometry batch size differs')
    for path, digest in evidence.items():
        require(sha(path) == digest, f'Input changed during analysis: {path}')
    return {'schema': SCHEMA, 'model': 'FLOWR.ROOT', 'runs': normalized,
            'provenance': evidence, 'input_scope': 'Previously evaluated records; original tensors and model weights are not replayed here.'}
