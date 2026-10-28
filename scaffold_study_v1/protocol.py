"""Finite experiment manifests. These files are plans, never execution results."""
from .io_utils import require


def final_constraint_plan(dataset, input_ids, max_continuations):
    require(dataset['model'] == 'FLOWR.ROOT', 'final_inpaint is a FLOWR-specific operation')
    require(input_ids and len(set(input_ids)) == len(input_ids), 'Provide unique input IDs')
    known = {r['input_id'] for r in dataset['runs']}
    require(set(input_ids).issubset(known), f'Unknown input IDs: {set(input_ids)-known}')
    selected = [r for r in dataset['runs'] if r['input_id'] in input_ids]
    require(type(max_continuations) is int and max_continuations > 0, 'Positive budget required')
    require(2*len(selected) <= max_continuations, 'Budget too small: never drop seeds or controls to fit it')
    jobs = []
    for r in selected:
        setting = r['runtime_settings'].get('final_inpaint')
        require(setting is False, 'This contrast requires an explicitly recorded final_inpaint=False baseline')
        for arm, value in [('baseline_replay', False), ('final_constraint_on', True)]:
            jobs.append({'run_id': r['run_id'], 'input_id': r['input_id'], 'seed': r['seed'],
                         'source_record': r.get('source_record'),
                         'arm': arm, 'final_inpaint': value, 'generator_identity': r['generator_identity'],
                         'status': 'not_run'})
    return {'schema': 'scaffold-experiment-plan/1', 'kind': 'final_constraint_control',
            'status': 'planned_not_executed', 'runtime_executor_included': False,
            'input_ids': input_ids, 'selection': 'Every existing seed of the explicitly selected inputs, regardless of outcome',
            'n_baselines': len(selected), 'n_interventions': len(selected), 'planned_continuations': len(jobs),
            'maximum_continuations': max_continuations, 'jobs': jobs,
            'preconditions': ['Original full records and matching checkpoint are available.',
                              'All baseline replays match original inputs, RNG state, outputs and source identity before any intervention.',
                              'Use the same initial sample, preprocessing, sampler and downstream random draws in each pair.',
                              'Only final_inpaint changes. Record all intermediate and final results; do not filter failures.'],
            'outcomes': ['fixed chemistry', 'fixed displacement in angstroms', 'chemical validity', 'connectivity',
                         'native PoseBusters checks (protein and water separately)', 'scaffold change',
                         'new failures among baseline-successful samples'],
            'stop_rules': ['Stop on a baseline mismatch; it is an execution problem, not a molecular failure.',
                           'Finish this manifest once; no added seeds, network modules or timepoints.',
                           'Restoring a hard constraint is pipeline evidence, not a hidden-network mechanism.',
                           'A remaining defect is unresolved; any new hypothesis needs a separate predeclared experiment.'],
            'provenance': dataset['provenance']}
