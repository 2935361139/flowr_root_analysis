"""Load literal settings, so editing the parameter page cannot execute code."""
import ast
from pathlib import Path


ALLOWED_CASES = {'cdk2', 'tyk2', 'ptp1b', 'bace'}
REQUIRED = {'INPUTS', 'SAMPLES_PER_INPUT', 'SEED_START', 'INPUTS_PER_GROUP', 'OUTPUT_PREFIX'}


def load_config(path):
    path = Path(path)
    values = {}
    for node in ast.parse(path.read_text(encoding='utf-8'), filename=str(path)).body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            raise ValueError('config.py only supports literal parameter assignments, not imports/functions/expressions')
        name = node.targets[0].id
        if name not in REQUIRED or name in values:
            raise ValueError(f'Unknown or repeated configuration key: {name}')
        try:
            values[name] = ast.literal_eval(node.value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{name}: enter a literal value') from exc
    if set(values) != REQUIRED:
        raise ValueError(f'Missing configuration keys: {sorted(REQUIRED - set(values))}')
    for name in ('SAMPLES_PER_INPUT', 'INPUTS_PER_GROUP'):
        if type(values[name]) is not int or values[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    start, count = values['SEED_START'], values['SAMPLES_PER_INPUT']
    if type(start) is not int or start < 0 or start + count > 2**32:
        raise ValueError('Seeds must be integers in 0..4294967295')
    items = values['INPUTS']
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError('INPUTS must contain at least one (case, ligand_idx) pair')
    normalized, seen = [], set()
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f'Invalid input pair: {item!r}')
        case, idx = item
        if type(case) is not str or case not in ALLOWED_CASES:
            raise ValueError(f'Unsupported example {case!r}; recorder currently supports {sorted(ALLOWED_CASES)}')
        if type(idx) is not int or idx < 0:
            raise ValueError('ligand_idx must be a non-negative integer (0-based)')
        if (case, idx) in seen:
            raise ValueError(f'Duplicate input: {(case, idx)}; use SAMPLES_PER_INPUT for repeated sampling')
        seen.add((case, idx))
        normalized.append({'case': case, 'ligand_idx': idx})
    prefix = values['OUTPUT_PREFIX']
    if type(prefix) is not str or not prefix or len(prefix) > 80 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in prefix):
        raise ValueError('OUTPUT_PREFIX must use 1..80 letters, digits, underscores or hyphens')
    return {'inputs': normalized, 'samples_per_input': count, 'seed_start': start,
            'seeds': list(range(start, start + count)), 'inputs_per_group': values['INPUTS_PER_GROUP'],
            'output_prefix': prefix}
