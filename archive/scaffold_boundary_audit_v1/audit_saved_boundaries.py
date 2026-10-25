#!/usr/bin/env python3
"""Audit ALL saved baseline trajectories on CPU; no model/weights/generation.

Uses this FLOWR version's own molecule builder with repair OFF, and RDKit's
chemistry-problem detector. All fixed sites are examined, not a named atom/case.
Intermediate argmax violations are observations, NOT proof of generation failure
or a causal commitment point. Existing terminal labels remain unchanged.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
import traceback
import zipfile
from pathlib import Path
import numpy as np

VERSION = '2026-10-23.saved-boundary-audit.1'
ORDERS = np.array([0., 1., 2., 3., 1.5])


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def graph(data, atom_tokens, charge_tokens):
    """Match native prefix masks, virtual-atom removal and LOWER-triangle argmax."""
    mask = array(data['mask'])[0].astype(bool)
    n = len(mask)
    require(np.array_equal(mask, np.arange(n) < int(mask.sum())), 'Native builder expects a prefix mask; unsupported record')
    atomic = array(data['atomics'])[0]
    charges = array(data['charges'])[0]
    bonds = array(data['bonds'])[0]
    coords = array(data['coords'])[0]
    require(bonds.shape == (n, n, 5) and coords.shape == (n, 3), 'Unsupported tensor dimensions')
    require(all(np.isfinite(x).all() for x in [atomic, charges, bonds, coords]), 'Nonfinite saved computational tensor')
    require(atomic.shape[-1] == len(atom_tokens) and charges.shape[-1] == len(charge_tokens), 'Vocabulary mismatch')
    elements = [atom_tokens[int(i)] for i in atomic.argmax(-1)]
    q = [charge_tokens[int(i)] for i in charges.argmax(-1)]
    keep = mask & np.array([a != '<NOATOM>' for a in elements])
    lower = np.tril(bonds.argmax(-1), -1)
    classes = lower + lower.T
    classes = classes * keep[:, None] * keep[None, :]
    return {'elements': elements, 'charges': q, 'keep': keep, 'slots': np.flatnonzero(keep).tolist(),
            'classes': classes, 'coords': coords, 'bond_values': bonds}


def event_timing(rows):
    """First appearance versus start of the FINAL contiguous run; not causal onset."""
    require([r['call'] for r in rows] == list(range(len(rows))), 'Missing/reordered calls')
    names = sorted({e for row in rows for e in row['events']})
    answer = []
    for name in names:
        present = [name in row['events'] for row in rows]
        first = present.index(True)
        suffix = None
        if present[-1]:
            suffix = len(rows) - 1
            while suffix > 0 and present[suffix - 1]:
                suffix -= 1
        answer.append({'event': name, 'first_observed_call': first,
                       'present_at_last_network_call': present[-1],
                       'terminal_contiguous_start_call': suffix,
                       'first_observed_time': rows[first]['time'],
                       'terminal_contiguous_start_time': rows[suffix]['time'] if suffix is not None else None})
    return answer


def native_unsanitized(data, builder):
    parts = list(builder._extract_mols(data['coords'], data['atomics'], data['mask'],
                 bond_dists=data['bonds'], charge_dists=data['charges'],
                 hybridization_dists=data.get('hybridization')))
    require(len(parts) == 1, 'Expected one molecule')
    return builder._mol_from_tensors(*parts[0], sanitise=False, add_hs=False, valence_repair=False)


def chemistry(data, builder, g, Chem, rdBase, mol_is_valid):
    # Block normal RDKit error messages from flooding the console; retain messages below.
    with rdBase.BlockLogs():
        mol = native_unsanitized(data, builder)
        if mol is None:
            return {'native_buildable': False, 'native_rdkit_valid': False, 'problems': [],
                    'status': 'native_unsanitized_builder_returned_none'}, None
        require(mol.GetNumAtoms() == len(g['slots']), 'Native molecule-to-sampler atom mapping differs')
        for i, slot in enumerate(g['slots']):
            require(mol.GetAtomWithIdx(i).GetSymbol() == g['elements'][slot], 'Native atom order differs')
        problems = []
        for problem in Chem.DetectChemistryProblems(Chem.Mol(mol)):
            if hasattr(problem, 'GetAtomIdx'):
                ids = [int(problem.GetAtomIdx())]
            elif hasattr(problem, 'GetAtomIndices'):
                ids = [int(i) for i in problem.GetAtomIndices()]
            else:
                ids = []
            require(all(0 <= i < len(g['slots']) for i in ids), 'Unmapped RDKit problem atom')
            problems.append({'type': str(problem.GetType()), 'message': str(problem.Message()),
                             'sampler_slots': [g['slots'][i] for i in ids]})
        return {'native_buildable': True, 'native_rdkit_valid': bool(mol_is_valid(mol, connected=False)),
                'problems': problems, 'status': 'assessed'}, mol


def reference_profile(task, ref):
    require(task.get('mapping_verified') is True, 'Unverified reference/sampler mapping')
    fixed = list(task['fixed_sampler_indices'])
    refs = list(task['fixed_reference_indices'])
    require(len(fixed) == len(refs) and len(set(fixed)) == len(fixed), 'Invalid fixed mapping')
    mapping = dict(zip(fixed, refs))
    anchors = {int(x['fixed_sampler_index']) for x in task['reference_attachment_sites']}
    expected = {s: {'element': ref.GetAtomWithIdx(r).GetSymbol(), 'charge': ref.GetAtomWithIdx(r).GetFormalCharge(),
                    'reference_index': r} for s, r in mapping.items()}
    expected_bonds, pairs = {}, []
    types = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'AROMATIC': 4}
    for i, s in enumerate(fixed):
        for j in fixed[:i]:
            b = ref.GetBondBetweenAtoms(mapping[s], mapping[j])
            expected_bonds[(min(s,j), max(s,j))] = types[str(b.GetBondType())] if b else 0
            if b and (s in anchors or j in anchors):
                site, partner = (s,j) if s in anchors else (j,s)
                if s in anchors and j in anchors:
                    site, partner = min(s,j), max(s,j)
                pairs.append((site, partner, types[str(b.GetBondType())]))
    return {'fixed': fixed, 'anchors': anchors, 'expected': expected,
            'expected_bonds': expected_bonds, 'pairs': sorted(pairs)}


def site_observations(g, profile, chem):
    fixed = set(profile['fixed'])
    events, sites = [], []
    for s in profile['fixed']:
        exp = profile['expected'][s]
        if not g['keep'][s]:
            events.append(f'fixed_missing:{s}')
        if g['keep'][s] and g['elements'][s] != exp['element']:
            events.append(f'fixed_element_changed:{s}')
        if g['keep'][s] and g['charges'][s] != exp['charge']:
            events.append(f'fixed_charge_changed:{s}')
        neighbors = np.flatnonzero(g['classes'][s]).tolist()
        sites.append({'slot': s, 'reference_index': exp['reference_index'], 'attachment': s in profile['anchors'],
                      'present': bool(g['keep'][s]), 'element': g['elements'][s], 'charge': g['charges'][s],
                      'explicit_bond_order_sum': float(ORDERS[g['classes'][s]].sum()),
                      'neighbors': [{'slot': j, 'class': int(g['classes'][s,j]), 'fixed': j in fixed} for j in neighbors]})
    for (s,j), expected in profile['expected_bonds'].items():
        if g['keep'][s] and g['keep'][j] and g['classes'][s,j] != expected:
            events.append(f'fixed_internal_bond_changed:{s}:{j}')
    for problem in chem['problems']:
        for s in problem['sampler_slots']:
            if s in fixed:
                events.append(f"fixed_chemistry_problem:{problem['type']}:{s}")
    return sorted(set(events)), sites


def pair_metrics(g, frame, prior_coords, previous_coords, profile, scale):
    feedback = frame.get('self_condition')
    feedback_coords = array(feedback['coords'])[0] if feedback is not None and 'coords' in feedback else None
    out = []
    for s,j,expected_class in profile['pairs']:
        ref = (prior_coords[j] - prior_coords[s]) * scale
        predicted = (g['coords'][j] - g['coords'][s]) * scale
        values = g['bond_values'][max(s,j), min(s,j)].astype(float)
        probs = np.exp(values - values.max()); probs /= probs.sum()
        row = {'site': s, 'partner': j, 'expected_bond_class': expected_class,
               'predicted_bond_class': int(g['classes'][s,j]),
               'expected_bond_probability': float(probs[expected_class]),
               'relative_vector_error_A': float(np.linalg.norm(predicted - ref)),
               'reference_length_A': float(np.linalg.norm(ref)),
               'feedback_relative_vector_error_A': None, 'feedback_vs_previous_prediction_max_A': None}
        if feedback_coords is not None:
            row['feedback_relative_vector_error_A'] = float(np.linalg.norm((feedback_coords[j]-feedback_coords[s])*scale-ref))
            if previous_coords is not None:
                row['feedback_vs_previous_prediction_max_A'] = float(np.max(np.abs(feedback_coords[[s,j]]-previous_coords[[s,j]]))*scale)
        out.append(row)
    return out


def compare_decoded(raw_graph, decoded, profile):
    """Use saved output coordinates to verify atom order before attributing edits."""
    if decoded is None:
        return {'mapping_verified': False, 'reason': 'No decoded molecule', 'fixed_atom_edits': []}
    slots = raw_graph['slots']
    require(decoded.GetNumAtoms() == len(slots), 'Decoded atom count changed; requires a separate mapping adapter')
    delta = float(np.max(np.abs(decoded.GetConformer().GetPositions() - raw_graph['coords'][slots])))
    require(delta <= 0.002, 'Decoded coordinates/order differ from raw-final frame')
    inverse = {slot: i for i,slot in enumerate(slots)}
    changes = []
    for s in profile['fixed']:
        if s not in inverse:
            continue
        atom = decoded.GetAtomWithIdx(inverse[s])
        old = {'element': raw_graph['elements'][s], 'charge': raw_graph['charges'][s]}
        new = {'element': atom.GetSymbol(), 'charge': atom.GetFormalCharge()}
        if old != new:
            changes.append({'slot': s, 'reference_index': profile['expected'][s]['reference_index'],
                            'before_decode': old, 'after_decode': new})
    return {'mapping_verified': True, 'sdf_rounding_max_A': delta, 'fixed_atom_edits': changes}


def chemistry_selftest(Chem, rdBase):
    # Checks actual RDKit in the USER environment before processing any trajectories.
    examples = [('CO(C)C', True), ('CN(C)(C)C', True), ('CC(C)(C)(C)C', True),
                ('C[N+](C)(C)C', False), ('C[O+](C)C', False), ('CS(=O)(=O)C', False)]
    with rdBase.BlockLogs():
        for smi, bad in examples:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            require(mol is not None, 'RDKit self-test could not construct a test graph')
            found = any(p.GetType() == 'AtomValenceException' for p in Chem.DetectChemistryProblems(mol))
            require(found == bad, f'RDKit chemistry self-test failed: {smi}')
    return {'passed': True, 'examples': len(examples), 'scope': 'Charge-aware C/N/O/S checks, not a hardcoded carbon cutoff'}


def audit_run(run, info, qrows, expected_environment, load, factories, output):
    torch, Chem, rdBase, mol_is_valid, MolBuilder, vocab_builders, core_atoms = factories
    folder = run / 'research_record/attempt_000'
    manifest = read_json(run / 'research_record/manifest.json')
    require(manifest['torch_version'] == expected_environment['torch_version'] and
            manifest['rdkit_version'] == expected_environment['rdkit_version'], 'Recorded environment differs')
    require(manifest['checkpoint_sha256'] == expected_environment['checkpoint_sha256'], 'Recorded checkpoint differs')
    for name,key in [('raw_final_prediction.pt','raw_final_sha256'), ('task_definition.json','task_sha256'),
                     ('reference_preprocessed.sdf','reference_sdf_sha256')]:
        require(sha(folder/name) == info[key], f'Recorded file changed: {run.name}/{name}')
    inputs, final = load(folder/'inputs.pt'), load(folder/'raw_final_prediction.pt')
    task = read_json(folder/'task_definition.json')
    require(int(manifest['args']['seed']) == info['seed'] and manifest['args']['ligand_idx'] == info['ligand_idx'], 'Run identity differs')
    ref = Chem.SDMolSupplier(str(folder/'reference_preprocessed.sdf'), sanitize=True, removeHs=False)[0]
    require(ref is not None, 'Unreadable reference')
    profile = reference_profile(task, ref)
    prior = inputs['prior']; prior_coords = array(prior['coords'])[0]
    recorded_fixed = np.flatnonzero(array(prior['mask'])[0].astype(bool) & array(prior['fragment_mask'])[0].astype(bool)).tolist()
    require(recorded_fixed == profile['fixed'], 'Prior fixed-mask mapping differs')
    scale = float(inputs['coord_scale']); require(math.isfinite(scale) and scale > 0, 'Invalid coordinate scale')
    width = final['atomics'].shape[-1]
    require(width in [len(core_atoms)+1, len(core_atoms)+2], 'Unknown atom vocabulary')
    atom_vocab = vocab_builders[0](virtual_nodes=width == len(core_atoms)+2)
    charge_vocab = vocab_builders[1]()
    builder = MolBuilder(atom_vocab, charge_vocab, vocab_hybridization=vocab_builders[2](),
                         n_workers=1, ligand_valence_repair=False)
    atom_tokens = atom_vocab.tokens_from_indices(list(range(width)))
    charge_tokens = charge_vocab.tokens_from_indices(list(range(final['charges'].shape[-1])))
    charge_tokens = [int(v) if str(v).lstrip('-').isdigit() else str(v) for v in charge_tokens]
    fg = graph(final, atom_tokens, charge_tokens)
    final_chem, _ = chemistry(final, builder, fg, Chem, rdBase, mol_is_valid)
    raw_label = next(r for r in qrows if r['stage']=='raw_without_repair')
    require(final_chem['native_rdkit_valid'] is raw_label['rdkit_valid'], 'Offline raw-graph validity differs from the recorded native evaluator')
    require(fg['slots'] and len(fg['slots']) == info['raw_graph']['active_atoms'], 'Active graph size differs')
    for site in info['raw_graph']['fixed_sites']:
        s=site['sampler_index']
        require(given_equal(site['explicit_bond_order_sum'], float(ORDERS[fg['classes'][s]].sum())), 'Final fixed-site bond sum differs')
        require(np.flatnonzero(fg['classes'][s]).tolist() == site['neighbors'], 'Final fixed neighbors differ')
    fixed_events, fixed_sites = site_observations(fg, profile, final_chem)
    sdf = folder/'decoded_before_filter.sdf'
    expected_decoded = next(r for r in qrows if r['stage']=='run_decode')
    require(sha(sdf) == expected_decoded['input_sdf_sha256'], 'Recorded decoded SDF changed')
    decoded = None
    if sdf.read_text().strip():
        supplier=Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False)
        require(len(supplier)==1 and supplier[0] is not None, 'Unreadable recorded decode')
        decoded=supplier[0]
    decode = compare_decoded(fg, decoded, profile)
    if decode['mapping_verified']:
        # Comparison to the reference may include earlier network changes; keep those distinct.
        decoded_ref_changes = [s for s in profile['fixed'] if s in fg['slots'] and
            decoded.GetAtomWithIdx(fg['slots'].index(s)).GetSymbol()!=profile['expected'][s]['element']]
        require(sorted(decoded_ref_changes)==sorted(expected_decoded.get('fixed_element_changed_sites',[])), 'Decoded/reference element audit differs')
    with (folder/'trace_check/network_fixed_drift.csv').open() as f:
        old_drift = {int(r['call_index']):float(r['fixed_rms_A']) for r in csv.DictReader(f)}
    paths = sorted((folder/'forward_calls').glob('call_*.pt'))
    attempt = read_json(folder/'attempt.json')
    require(len(paths)==attempt['forward_calls']==101, 'Incomplete native call trajectory')
    network_rows, input_rows, previous = [], [], None
    file_hashes = [{'name':str(p.relative_to(run)), 'sha256':sha(p)} for p in
                   [run/'research_record/manifest.json', folder/'inputs.pt', folder/'raw_final_prediction.pt',
                    folder/'task_definition.json', folder/'reference_preprocessed.sdf',
                    folder/'decoded_before_filter.sdf', folder/'trace_check/network_fixed_drift.csv']]
    for k,path in enumerate(paths):
        frame=load(path); require(frame['call_index']==k, 'Call index mismatch')
        prediction=frame['network_output_before_postprocessing']
        ng=graph(prediction,atom_tokens,charge_tokens)
        nc,_=chemistry(prediction,builder,ng,Chem,rdBase,mol_is_valid)
        events, sites=site_observations(ng,profile,nc)
        fixed_rms=float(np.sqrt(np.square((ng['coords'][profile['fixed']]-prior_coords[profile['fixed']])*scale).sum(axis=1).mean()))
        require(abs(fixed_rms-old_drift[k])<=1e-5, 'Offline coordinate/frame comparison differs from saved observation')
        network_rows.append({'call':k,'time':float(frame['times_scalar'][0]),'events':events,'fixed_sites':sites,
                             'chemistry':nc,'fixed_rms_A':fixed_rms,
                             'fixed_pair_feedback':pair_metrics(ng,frame,prior_coords,previous,profile,scale)})
        state=frame['state_before_forward']; sg=graph(state,atom_tokens,charge_tokens)
        sc,_=chemistry(state,builder,sg,Chem,rdBase,mol_is_valid)
        se,ss=site_observations(sg,profile,sc)
        input_rows.append({'call':k,'time':float(frame['times_scalar'][0]),'events':se,'fixed_sites':ss,'chemistry':sc})
        previous=ng['coords'];file_hashes.append({'name':str(path.relative_to(run)),'sha256':sha(path)})
    timings=event_timing(network_rows)
    report={'case':info['case'],'ligand_idx':info['ligand_idx'],'seed':info['seed'],'run_directory':str(run),
            'calls':len(network_rows),'fixed_mapping':profile['expected'],
            'terminal_labels_unchanged':qrows,'final_raw_chemistry':final_chem,'final_raw_fixed_events':fixed_events,
            'final_raw_fixed_sites':fixed_sites,'decode_changes':decode,'network_event_timing':timings,
            'input_state_event_timing':event_timing(input_rows),'terminal_fixed_events': [t for t in timings if t['present_at_last_network_call']],
            'causal_status':'No interventions or hidden-module inference in this offline audit.',
            'time_note':'First appearance and final contiguous diagnostic interval in recorded raw predictions; not proof of irreversible failure or a causal commitment point.',
            'trajectory_identity':file_hashes}
    write_json(output/'report.json',report)
    write_json(output/'network_timeline.json',network_rows)
    write_json(output/'input_timeline.json',input_rows)
    return report


def given_equal(a,b):
    return abs(float(a)-float(b)) < 1e-7


def package(out):
    with zipfile.ZipFile(out/'boundary_audit_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*.json')):
            z.write(p,p.relative_to(out))
        z.write(Path(__file__),'program/audit_saved_boundaries.py')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-root',type=Path,default=Path('outputs/sh_validation_baselines_v1'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();repo=Path.cwd().resolve();batch=args.batch_root.resolve()
    out=(args.output or batch/'boundary_audit_v1').resolve()
    require(out!=batch and batch in out.parents,'Audit output must be a new subdirectory inside the batch')
    plan=read_json(batch/'frozen_plan.json');q=read_json(batch/'quality/quality_report.json')
    require(q.get('completed') is True and not q['errors'],'Baseline evaluation is incomplete')
    require(sha(batch/'frozen_plan.json')==(batch/'frozen_plan.sha256').read_text().strip(),'Frozen plan changed')
    for name,expected in plan['config']['source_sha256'].items():
        require(sha(repo/name)==expected,f'Native source changed: {name}')
    planned={(x['case'],x['ligand_idx'],s) for x in plan['selected_inputs'] for s in plan['seeds']}
    actual={(x['case'],x['ligand_idx'],x['seed']) for x in q['runs']}
    require(actual==planned and len(q['runs'])==len(planned),'Batch membership differs from frozen plan')
    sys.path.insert(0,str(repo))
    import torch
    from rdkit import Chem,rdBase
    from flowr.constants import CORE_ATOMS
    from flowr.models.mol_builder import MolBuilder
    from flowr.scriptutil import _build_vocab,_build_vocab_charges,_build_vocab_hybridization
    from flowr.util.rdkit import mol_is_valid
    require(str(torch.__version__)==plan['torch_version'] and rdBase.rdkitVersion==plan['rdkit_version'],'Use the original recorded environment')
    identity={'version':VERSION,'program_sha256':sha(__file__),'plan_sha256':sha(batch/'frozen_plan.json'),
              'quality_sha256':sha(batch/'quality/quality_report.json')}
    if out.exists():
        require(args.resume,'Output exists. Use --resume to retain completed audits.')
        require(read_json(out/'identity.json')==identity,'Audit code or baseline identity changed')
    else:
        out.mkdir(parents=True);write_json(out/'identity.json',identity)
    summary={'version':VERSION,'completed':False,'planned':len(planned),'reports':[],'errors':[],
             'native_generations':0,'interventions':0,'model_weights_loaded':False,
             'scope':'Same offline rules across all planned inputs/fixed sites; diagnostic coverage, not proven mechanistic generalization.'}
    factories=(torch,Chem,rdBase,mol_is_valid,MolBuilder,(_build_vocab,_build_vocab_charges,_build_vocab_hybridization),CORE_ATOMS)
    load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
    try:
        summary['chemistry_selftest']=chemistry_selftest(Chem,rdBase)
        for number,info in enumerate(q['runs'],1):
            key=f"{info['case']}_lig{info['ligand_idx']}_seed{info['seed']}"
            dest=out/'runs'/key;dest.mkdir(parents=True,exist_ok=True)
            run=batch/'runs'/f"{info['case']}_lig{info['ligand_idx']}"/f"seed_{info['seed']}"
            print(f'[{number}/{len(planned)}] CPU audit {key}',flush=True)
            qrows=[r for r in q['rows'] if r['run_directory']==info['run_directory']]
            require(len(qrows)==2 and all(not x['errors'] for x in qrows),'Missing or erroneous terminal labels')
            if (dest/'complete.json').exists():
                saved=read_json(dest/'complete.json')
                require(all(sha(dest/n)==h for n,h in saved['output_sha256'].items()),'Completed audit output changed')
                report=read_json(dest/'report.json')
                require(all(sha(run/x['name'])==x['sha256'] for x in report['trajectory_identity']),'Saved trajectory changed on resume')
            else:
                report=audit_run(run,info,qrows,plan,load,factories,dest)
                write_json(dest/'complete.json',{'output_sha256':{n:sha(dest/n) for n in ['report.json','network_timeline.json','input_timeline.json']}})
            summary['reports'].append({'id':key,'final_raw_fixed_events':report['final_raw_fixed_events'],
                  'terminal_fixed_events':report['terminal_fixed_events'],'decode_changes':report['decode_changes']})
            write_json(out/'summary.json',summary)
        summary['completed']=True
    except BaseException as exc:
        summary['errors'].append({'message':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()})
        raise
    finally:
        write_json(out/'summary.json',summary);package(out)
        print(f"REVIEW: {out/'boundary_audit_review.zip'}",flush=True)
    print('COMPLETE: offline stage/time/site observations only. No new generation or intervention.',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'STOP: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(2)
