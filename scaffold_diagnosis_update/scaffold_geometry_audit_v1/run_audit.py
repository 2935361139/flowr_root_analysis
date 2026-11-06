#!/usr/bin/env python3
"""One bounded endpoint audit. No sampling, hidden-layer scans or interventions."""
import argparse
import csv
import json
from pathlib import Path
import sys
import traceback
import zipfile
from collections import Counter
from temporal import track_pairs
from core import atom_regions, fixed_pair_comparison
from flowr_adapter import inventory, load_molecule, endpoint_molecule, require, sha


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'item'):
        return clean(value.item())
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else None
    return value


def write_json(path, data):
    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fields or ['status'])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(clean(v), ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def check_name(module, output):
    return module['rename_outputs'].get(output, output).lower().replace(' ', '_')


def environment_atom(atom):
    info = atom.GetPDBResidueInfo()
    return dict(environment_atom_index=atom.GetIdx(), environment_element=atom.GetSymbol(),
                pdb_serial=info.GetSerialNumber() if info else None,
                atom_name=info.GetName().strip() if info else None,
                residue=info.GetResidueName().strip() if info else None,
                residue_number=info.GetResidueNumber() if info else None,
                chain=info.GetChainId().strip() if info else None)


def regions(mol, ids, fixed, anchors):
    edges = [(ids[b.GetBeginAtomIdx()], ids[b.GetEndAtomIdx()]) for b in mol.GetBonds()]
    return atom_regions(ids, edges, fixed, anchors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort-dir', type=Path, default=Path('outputs/sh_discovery_cohort_v2'))
    p.add_argument('--diagnostic-dir', type=Path, default=Path('outputs/sh_discovery_cohort_v2_diagnostics'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/sh_geometry_audit_v1'))
    p.add_argument('--repo', type=Path, default=Path.cwd())
    args = p.parse_args()
    repo, out = args.repo.resolve(), args.output_dir.resolve()
    if out.exists(): p.error('Output exists; preserve it and select a new --output-dir')
    sys.path.insert(0, str(repo))
    import numpy as np
    import rdkit
    from rdkit import Chem
    import posebusters
    from posebusters import PoseBusters
    from posebusters.tools.loading import safe_load_mol
    from posebusters.modules.intermolecular_distance import check_intermolecular_distance
    from posebusters.modules.volume_overlap import check_volume_overlap
    q, items = inventory(args.cohort_dir.resolve(), args.diagnostic_dir.resolve(), repo)
    require(rdkit.__version__ == q['versions']['rdkit'], 'Use the RDKit version from the completed evaluation')
    require(Path(posebusters.__file__).resolve().parent == repo / 'posebusters', 'Expected the repository PoseBusters fork')
    config = PoseBusters('dock').config
    modules = [m for m in config['modules'] if m['function'] in ('intermolecular_distance', 'volume_overlap')]
    out.mkdir(parents=True)
    write_json(out / 'protocol.json', {
        'scope': 'All existing inputs and seeds, both endpoint decodes; no new generation, alignment, optimization or interventions.',
        'primary_localization': 'Native PoseBusters intermolecular clash pairs. No new distance thresholds.',
        'regions': 'Fixed/generated, current cross-boundary bond endpoints, exact graph hops and original fixed attachment labels. Descriptive, not causal.',
        'comparison': 'Reference checked in original protein frame. Same failed check in reference does not prove the generated clash is inherited. Only fixed atom correspondences permit pairwise comparison.',
        'limits': 'Whole-molecule energy and volume overlap are not assigned to a causal atom. Water validity and conditioning visibility remain unverified. Reference SDF has rounded coordinates. Final clash-pair observations are tracked in saved frames; no hidden-module or causal localization.',
        'stop_rule': 'Produce one table and stop. Do not choose mechanisms or run rescue experiments automatically.',
        'rdkit': rdkit.__version__, 'posebusters_config': config,
        'posebusters_source_sha256': {str(f.relative_to(repo)): sha(f) for f in sorted((repo / 'posebusters').rglob('*')) if f.suffix in ('.py', '.yml')},
        'quality_report_sha256': sha(args.cohort_dir / 'quality/quality_report.json'),
        'diagnostic_summary_sha256': sha(args.diagnostic_dir / 'summary.json'),
        'script_sha256': {f.name: sha(f) for f in Path(__file__).parent.glob('*.py')},
    })
    pairs, reference_rows, samples, unlocalized, errors, check_rows = [], [], [], [], [], []
    references, environments = {}, {}
    temporal_rows, temporal_sources = [], {}
    for number, item in enumerate(items, 1):
        print(f'[{number}/{len(items)}] {item["id"]}', flush=True)
        base = {k:item[k] for k in ('id', 'input_id', 'seed')}
        try:
            protein_hash = sha(item['protein'])
            if protein_hash not in environments:
                env = safe_load_mol(item['protein'], **config.get('loading', {}).get('mol_cond', {}))
                require(env is not None, 'Native protein loader failed')
                environments[protein_hash] = env
            env = environments[protein_hash]
            task = item['task']
            refkey = (protein_hash, item['info']['reference_sdf_sha256'])
            if refkey not in references:
                ref = load_molecule(item['folder'] / 'reference_preprocessed.sdf', Chem)
                ref_h = Chem.AddHs(Chem.Mol(ref), addCoords=True)
                table = PoseBusters('dock').bust([ref_h], None, str(item['protein']))
                require(len(table) == 1 and not table.iloc[0].isna().any(), 'Reference PB evaluation is incomplete')
                refchecks = {str(k): bool(v) for k,v in table.iloc[0].items()}
                details = {}
                for module in modules:
                    if module['function'] == 'intermolecular_distance':
                        label = check_name(module, 'no_clashes')
                        result = check_intermolecular_distance(ref_h, env, **module.get('parameters', {}))
                        require(bool(result['results']['no_clashes']) == refchecks[label], 'Reference check/detail mismatch')
                        details[label] = result['details']
                references[refkey] = (refchecks, details)
                reference_rows.append({'input_id':item['input_id'], 'protein_sha256':protein_hash,
                    'reference_sha256':refkey[1], 'posebusters_valid':all(refchecks.values()),
                    'failed_checks':[k for k,v in refchecks.items() if not v], 'checks':refchecks})
            refchecks, refdetails = references[refkey]
            local_all_pairs = []
            for stage in ('raw_without_repair', 'run_decode'):
                baseline = item['rows'][stage]
                identity = {**base, 'stage':stage}
                if baseline.get('rdkit_valid') is not True:
                    unlocalized.append({**identity, 'problem':'invalid_or_unassessed_chemistry', 'status':'geometry_not_assessed'})
                    continue
                mol, slots = endpoint_molecule(item, stage, Chem, np)
                mapping = dict(zip(task['fixed_sampler_indices'], task['fixed_reference_indices']))
                anchor = {a['fixed_sampler_index'] for a in task['reference_attachment_sites']}
                roles = regions(mol, slots, set(mapping), anchor)
                mol_h = Chem.AddHs(Chem.Mol(mol), addCoords=True)
                local_pairs, tested = [], set()
                for module in modules:
                    is_distance = module['function'] == 'intermolecular_distance'
                    label = check_name(module, 'no_clashes' if is_distance else 'no_volume_clash')
                    fn = check_intermolecular_distance if is_distance else check_volume_overlap
                    result = fn(mol_h, env, **module.get('parameters', {}))
                    ok = bool(result['results']['no_clashes' if is_distance else 'no_volume_clash'])
                    require(ok == baseline['posebusters_checks'][label], f'Native check no longer reproduces baseline: {label}')
                    tested.add(label)
                    check_rows.append({**identity, 'check':label, 'generated_pass':ok,
                        'reference_pass':refchecks[label], 'baseline_reproduced':True})
                    if not is_distance:
                        if not ok:
                            unlocalized.append({**identity, 'problem':label, 'status':'global_volume_metric_not_atom_attributed',
                                'reference_pass':refchecks[label], 'value':result['results'].get('volume_overlap')})
                        continue
                    ref_pairs = {(int(x['ligand_atom_id']),int(x['protein_atom_id']))
                                 for x in refdetails[label].to_dict('records') if bool(x['clash'])}
                    for x in result['details'].to_dict('records'):
                        if not bool(x['clash']): continue
                        idx, other = int(x['ligand_atom_id']), int(x['protein_atom_id'])
                        require(idx < len(slots), 'Native heavy-atom clash unexpectedly points to added H')
                        slot = slots[idx]
                        entry = {**identity, 'check':label, 'ligand_sdf_atom_index':idx,
                            'sampler_atom_index':slot, 'fixed_reference_atom_index':mapping.get(slot),
                            'element':mol.GetAtomWithIdx(idx).GetSymbol(), **roles[slot],
                            **environment_atom(env.GetAtomWithIdx(other)),
                            'distance_A':float(x['distance']), 'relative_distance':float(x['relative_distance']),
                            'native_clash_cutoff':module['parameters']['clash_cutoff'],
                            'reference_check_pass':refchecks[label],
                            'reference_pair_comparison':fixed_pair_comparison(slot,other,mapping,ref_pairs),
                            'environment_visibility_to_model':'not_assessed', 'causal_status':'not_tested'}
                        local_pairs.append(entry)
                for problem in baseline['posebusters_failed_checks']:
                    if problem not in tested:
                        unlocalized.append({**identity,'problem':problem,'reference_pass':refchecks.get(problem),
                            'status':'endpoint_failure_retained_not_atom_attributed'})
                pairs.extend(local_pairs)
                local_all_pairs.extend(local_pairs)
                samples.append({**identity,'posebusters_valid':baseline['posebusters_valid'],
                    'fixed_chemistry_preserved':baseline['fixed_chemistry_preserved'],
                    'failed_checks':baseline['posebusters_failed_checks'],
                    'reference_failed_checks':[k for k,v in refchecks.items() if not v],
                    'n_native_clash_pairs':len(local_pairs),
                    'regions_with_clashes':sorted({x['region'] for x in local_pairs}),
                    'n_boundary_clash_pairs':sum(x['graph_hops_to_current_boundary']==0 for x in local_pairs),
                    'n_original_attachment_clash_pairs':sum(x['is_reference_attachment_atom'] for x in local_pairs),
                    'n_fixed_pairs_already_clashing_in_reference':sum(x['reference_pair_comparison'].startswith('same_fixed') for x in local_pairs),
                    'same_failed_check_in_reference':[k for k in baseline['posebusters_failed_checks'] if refchecks.get(k) is False],
                    'native_checks_reproduced':len(tested), 'causal_status':'not_tested'})
            observed, hashes = track_pairs(item, local_all_pairs, env, modules)
            temporal_rows.extend(observed); temporal_sources.update(hashes)
        except Exception as exc:
            errors.append({**base,'error':f'{type(exc).__name__}: {exc}', 'traceback':traceback.format_exc()})
            print('CHECK:', errors[-1]['error'], flush=True)
    for path, digest in temporal_sources.items():
        if sha(Path(path)) != digest:
            errors.append({'error':f'Input changed during audit: {path}'})
    write_json(out/'temporal_evidence.json', temporal_rows)
    write_json(out/'temporal_sources.json', temporal_sources)
    write_csv(out/'clash_pairs.csv', pairs)
    write_csv(out/'sample_summary.csv', samples)
    write_csv(out/'reference_checks.csv', reference_rows)
    write_csv(out/'check_comparison.csv', check_rows)
    write_json(out/'unlocalized.json', unlocalized)
    write_json(out/'errors.json', errors)
    raw = [r for r in samples if r['stage']=='raw_without_repair']
    summary = {'completed':not errors, 'requested_runs':len(items), 'endpoint_rows':len(samples),
        'reference_inputs':len(reference_rows), 'raw_pb_failures':sum(not r['posebusters_valid'] for r in raw),
        'raw_runs_with_native_clash_pairs':sum(r['n_native_clash_pairs']>0 for r in raw),
        'raw_runs_with_current_boundary_clash':sum(r['n_boundary_clash_pairs']>0 for r in raw),
        'raw_runs_with_original_attachment_clash':sum(r['n_original_attachment_clash_pairs']>0 for r in raw),
        'raw_runs_with_same_failed_check_in_reference':sum(bool(r['same_failed_check_in_reference']) for r in raw),
        'raw_clash_pair_regions':dict(Counter(x['region'] for x in pairs if x['stage']=='raw_without_repair')),
        'error_count':len(errors), 'causal_status':'not_tested',
        'warning':'Current boundary overlap is descriptive. It does not prove a connection mechanism or an unsuitable input. Global metrics and water assumptions remain explicit limitations.'}
    write_json(out/'summary.json',summary)
    with zipfile.ZipFile(out/'geometry_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.iterdir()):
            if path.suffix in ('.csv','.json'): z.write(path,path.name)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Review:',out/'geometry_review.zip')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
