"""Model-independent analysis of normalized, already evaluated evidence.

No neural root cause is inferred from a final molecule or a time-series suffix.
Each adapter must supply native stages, units and verified within-run identities.
"""
from collections import Counter
import math
from .io_utils import require

SCHEMA = "scaffold-evidence/1"
METRICS = {
    "chemical_valid": "chemical_invalid",
    "connected": "disconnected",
    "fixed_chemistry_preserved": "fixed_chemistry_changed",
    "fixed_substructure_match": "fixed_substructure_mismatch",
    "geometry_valid": "geometry_failed",
    "scaffold_changed": "scaffold_unchanged",
}


def validate(dataset):
    require(dataset.get("schema") == SCHEMA, "Unsupported normalized schema")
    require(bool(dataset.get("model")), "Model identity is required")
    runs = dataset.get("runs", [])
    require(bool(runs), "No runs: an empty analysis is not validation")
    require(len({r["run_id"] for r in runs}) == len(runs), "Duplicate run IDs")
    for r in runs:
        require(bool(r.get("input_id")) and bool(r.get("input_group")), "Input identity/group missing")
        require(set(r["endpoints"]) == {"raw", "decoded"}, "Need both endpoint stages; missing values must be null")
        for e in r["endpoints"].values():
            for metric in METRICS:
                require(metric in e and (e[metric] is None or type(e[metric]) is bool), f"Invalid/missing {metric}")
        require(len({(o['stream'], o['entity'], o['metric']) for o in r['observations']}) == len(r['observations']), "Duplicate observation series")
        for o in r['observations']:
            require(o['metric'] in {'fixed_chemistry_changed', 'neutral_carbon_overvalence', 'disconnected'}, "Unsupported temporal metric")
            points = o['points']
            require(bool(points), "Empty observation series")
            require([p['index'] for p in points] == list(range(len(points))), "Missing/duplicate/out-of-order observations")
            for p in points:
                require(p['value'] is None or type(p['value']) is bool, "Observation is not bool/null")
                require(bool(p.get('stage')), "Native stage is required")
                t = p.get('native_time')
                require(t is None or (type(t) in (int, float) and math.isfinite(t)), "Invalid native time")


def terminal_window(points):
    """Only the observed suffix; unknown is a boundary, never evidence of normality."""
    if points[-1]['value'] is not True:
        return {'status': 'terminal_signature_absent_or_unknown', 'first_persistent_observation': None}
    i = len(points) - 1
    while i > 0 and points[i - 1]['value'] is True:
        i -= 1
    return {'status': 'observed', 'first_persistent_observation': points[i],
            'last_observation': points[-1], 'preceding_observation': points[i-1] if i else None,
            'boundary_is_confirmed_change': i > 0 and points[i-1]['value'] is False,
            'causal_onset': None}


def analyze(dataset):
    validate(dataset)
    result = []
    for r in dataset['runs']:
        issues = []
        for stage, e in r['endpoints'].items():
            for metric, problem in METRICS.items():
                if e[metric] is not False:
                    continue
                # scaffold change is a task outcome, not a chemical/connection defect.
                issue = {'problem': problem, 'endpoint_stage': stage,
                         'locations': [], 'temporal_evidence': [], 'causal_status': 'not_tested',
                         'internal_module': None, 'evidence_level': 'endpoint_observation'}
                if stage == 'raw':
                    for o in r['observations']:
                        # Link a temporal signature only to an atom implicated at the endpoint.
                        # A different atom's anomalous prediction cannot explain this final defect.
                        applies = problem == 'disconnected' and o['metric'] == 'disconnected'
                        if problem == 'fixed_chemistry_changed' and o['metric'] == problem:
                            applies = o['entity'] in {f'fixed_atom:{a}' for a in e.get('fixed_changed_atoms', [])}
                        if problem == 'chemical_invalid' and o['metric'] == 'neutral_carbon_overvalence':
                            applies = o['entity'] in {f'fixed_atom:{a}' for a in e.get('neutral_carbon_overvalent_atoms', [])}
                        if applies:
                            window = terminal_window(o['points'])
                            if window['status'] == 'observed':
                                issue['temporal_evidence'].append({'stream': o['stream'], 'entity': o['entity'],
                                                                 'signature': o['metric'], **window})
                    if issue['temporal_evidence']:
                        issue['evidence_level'] = 'trajectory_observation'
                if problem == 'geometry_failed':
                    issue['failed_native_checks'] = e.get('geometry_failed_checks', [])
                    issue['locations'] = r.get('geometry', {}).get(stage, {}).get('clash_pairs', [])
                    issue['reference_context'] = r.get('geometry', {}).get(stage, {}).get('reference_context')
                    if issue['locations']:
                        issue['evidence_level'] = 'localized_endpoint_observation'
                    issue['unlocalized_checks'] = [x for x in issue['failed_native_checks'] if x not in {p['check'] for p in issue['locations']}]
                if problem == 'fixed_chemistry_changed':
                    issue['locations'] = e.get('fixed_changed_atoms', [])
                issue['localization_status'] = ('observed_window_available' if issue['temporal_evidence'] else 'time_and_module_unresolved')
                issues.append(issue)
        changed = [{'metric': k, 'raw': r['endpoints']['raw'][k], 'decoded': r['endpoints']['decoded'][k]}
                   for k in METRICS if r['endpoints']['raw'][k] is not None and r['endpoints']['decoded'][k] is not None
                   and r['endpoints']['raw'][k] != r['endpoints']['decoded'][k]]
        result.append({k: r[k] for k in ('run_id', 'input_id', 'input_group', 'seed')} | {
            'endpoints': r['endpoints'], 'issues': issues, 'decode_stage_changes': changed,
            'fixed_displacement_observations': r.get('fixed_displacement_observations', []),
            'runtime_settings': r.get('runtime_settings', {}),
            'causal_status': 'not_tested', 'full_scaffold_hopping_success': None,
            'full_success_note': 'No complete predeclared task contract, including fixed geometry tolerance; component outcomes remain separate.'})
    counts = {}
    for stage in ('raw', 'decoded'):
        counts[stage] = {}
        for metric in METRICS:
            c = Counter('unknown' if r['endpoints'][stage][metric] is None else 'pass' if r['endpoints'][stage][metric] else 'fail' for r in result)
            counts[stage][metric] = {k: c[k] for k in ('pass', 'fail', 'unknown')} | {'total': len(result)}
    families = {}
    for problem in METRICS.values():
        affected = [r for r in result if any(i['problem'] == problem and i['endpoint_stage'] == 'raw' for i in r['issues'])]
        families[problem] = {'runs': len(affected), 'distinct_input_groups': len({r['input_group'] for r in affected})}
    return {'schema': 'scaffold-diagnosis/1', 'completed': True, 'model': dataset['model'],
            'n_runs': len(result), 'n_input_groups': len({r['input_group'] for r in result}),
            'counts': counts, 'failure_families': families, 'runs': result,
            'causal_interventions_performed': 0, 'causal_status': 'not_tested',
            'limitations': ['Observation time is not causal onset.', 'Call/frame indices and native times are not interchangeable.',
                            'Same failed reference check does not establish that all generated defects were inherited.',
                            'No second-model or held-out mechanism validation has been performed.']}
