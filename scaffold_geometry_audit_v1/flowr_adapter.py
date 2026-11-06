"""Read existing ROOT records. No generator, checkpoint or torch import."""
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def inventory(cohort, diagnostics, repo):
    quality_path = cohort / 'quality/quality_report.json'
    q = read_json(quality_path)
    d = read_json(diagnostics / 'summary.json')
    require(q.get('completed') and not q.get('errors'), 'Incomplete quality report')
    require(d.get('completed'), 'Incomplete trajectory diagnosis')
    require(d['quality_report_sha256'] == sha(quality_path), 'Diagnosis belongs to a different quality report')
    reports = {}
    for item in d['records']:
        report_dir = diagnostics / item['report_directory']
        r = read_json(report_dir / 'connection_report.json')
        require(r['run'] not in reports, 'Duplicate diagnostic run')
        reports[r['run']] = (r, report_dir)
    require(len(q['runs']) == q['requested_run_count'] == len(reports), 'Run inventory mismatch')
    output = []
    for info in q['runs']:
        old = Path(info['run_directory'])
        candidates = [old, cohort / 'runs' / old.name,
                      cohort / 'runs' / old.parent.name / old.name]
        matches = {p.resolve() for p in candidates if (p / 'research_record/manifest.json').is_file()}
        require(len(matches) == 1, f'Missing or ambiguous run directory: {old}')
        run = matches.pop()
        folder = run / 'research_record/attempt_000'
        r, diag_dir = reports[str(old)]
        task = read_json(folder / 'task_definition.json')
        manifest = read_json(run / 'research_record/manifest.json')
        require(task == r['task_definition'] and task['mapping_verified'], 'Task mapping changed')
        require(manifest == r['manifest'], 'Manifest changed since diagnosis')
        for name, key in [('task_definition.json', 'task_sha256'),
                          ('reference_preprocessed.sdf', 'reference_sdf_sha256')]:
            require(sha(folder / name) == info[key], f'Changed record: {name}')
        protein_old = Path(manifest['args']['pdb_file'])
        # A moved checkout is allowed only when the exact protein bytes match.
        protein = protein_old if protein_old.is_file() else repo / 'examples' / protein_old.name
        require(sha(protein) == manifest['input_sha256'][str(protein_old)], 'Protein input bytes changed')
        rows = {row['stage']: row for row in q['rows'] if row['run_directory'] == str(old)}
        require(set(rows) == {'raw_without_repair', 'run_decode'}, 'Missing or repeated endpoint stages')
        input_id = f"{info['case']}_lig{info['ligand_idx']}"
        require(len([row for row in q['rows'] if row['run_directory'] == str(old)]) == 2,
                'Duplicate endpoint rows')
        require(all(not row.get('errors') for row in rows.values()), 'Endpoint evaluation error')
        output.append(dict(id=f"{input_id}_seed_{info['seed']}", input_id=input_id,
                           seed=info['seed'], folder=folder, diag_dir=diag_dir,
                           task=task, report=r, manifest=manifest, info=info,
                           protein=protein, rows=rows))
    return q, output


def load_molecule(path, Chem):
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    require(len(supplier) == 1 and supplier[0] is not None, f'Unreadable single molecule: {path}')
    mol = supplier[0]
    Chem.SanitizeMol(mol)
    require(all(a.GetAtomicNum() != 1 for a in mol.GetAtoms()), 'Expected recorded heavy-atom SDF')
    return mol


def endpoint_molecule(item, stage, Chem, np):
    name = 'decoded_without_repair.sdf' if stage == 'raw_without_repair' else 'decoded_before_filter.sdf'
    mol = load_molecule(item['folder'] / name, Chem)
    with np.load(item['diag_dir'] / 'connection_arrays.npz', allow_pickle=False) as data:
        vocab = item['report']['atom_vocabulary']
        slots = [int(i) for i in range(len(data['raw_final_valid']))
                 if data['raw_final_valid'][i] and vocab[int(data['raw_final_atom_classes'][i])] != '<NOATOM>']
        xyz = data['raw_final_coords'][slots].copy()
        fixed = data['fixed_sampler_indices'].tolist()
    require(fixed == item['task']['fixed_sampler_indices'], 'NPZ fixed-atom mapping changed')
    require(mol.GetNumAtoms() == len(slots), 'SDF atom count differs from recorded final state')
    require(np.isfinite(xyz).all(), 'Nonfinite final coordinates')
    require(float(np.max(np.abs(mol.GetConformer().GetPositions() - xyz))) < .002,
            'SDF atom order or coordinate frame differs from final tensor')
    for i, position in enumerate(xyz):
        mol.GetConformer().SetAtomPosition(i, tuple(float(v) for v in position))
    # Reconcile identity checks; never silently accept a different SDF under the old labels.
    canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
    require(canonical == item['rows'][stage]['molecule_smiles'], 'Endpoint graph differs from evaluated SDF')
    return mol, slots
