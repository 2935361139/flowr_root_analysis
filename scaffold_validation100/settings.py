"""Read literal path/quantity settings without executing configuration code."""
import ast
from pathlib import Path

REQUIRED={'SDF_PATH','PDB_PATH','LIGAND_INDICES','SAMPLES_PER_INPUT','SEED_START',
          'GENERATION_DIR','INTERVENTION_INPUT_DIR','INTERVENTION_OUTPUT_DIR','INPUTS_PER_GROUP'}


def load_config(path, repo):
    values={}
    for node in ast.parse(Path(path).read_text(encoding='utf-8')).body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):continue
        if not (isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)):
            raise ValueError('config.py only accepts literal parameter assignments')
        name=node.targets[0].id
        if name not in REQUIRED or name in values:raise ValueError(f'Unknown/repeated setting: {name}')
        try:values[name]=ast.literal_eval(node.value)
        except (ValueError,TypeError):raise ValueError(f'{name} must be a literal value') from None
    if set(values)!=REQUIRED:raise ValueError(f'Missing settings: {sorted(REQUIRED-set(values))}')
    for name in ['SAMPLES_PER_INPUT','INPUTS_PER_GROUP']:
        if type(values[name]) is not int or values[name]<1:raise ValueError(f'{name} must be a positive integer')
    start,count=values['SEED_START'],values['SAMPLES_PER_INPUT']
    if type(start) is not int or start<0 or start+count>2**32:raise ValueError('Seeds must fit 0..4294967295')
    ids=values['LIGAND_INDICES']
    if ids!='all':
        if not isinstance(ids,(list,tuple)) or not ids or any(type(x) is not int or x<0 for x in ids):
            raise ValueError('LIGAND_INDICES must be "all" or a nonempty list of 0-based integers')
        if len(ids)!=len(set(ids)):raise ValueError('Repeated ligand indices; use SAMPLES_PER_INPUT instead')
        ids=list(ids)
    result=dict(ligand_indices=ids,samples_per_input=count,seed_start=start,seeds=list(range(start,start+count)),
                inputs_per_group=values['INPUTS_PER_GROUP'])
    for key in ['SDF_PATH','PDB_PATH','GENERATION_DIR','INTERVENTION_INPUT_DIR','INTERVENTION_OUTPUT_DIR']:
        value=values[key]
        if type(value) is not str or not value.strip():raise ValueError(f'{key} must be a nonempty path')
        result[key.lower()]=str((Path(repo)/Path(value).expanduser()).resolve())
    return result
