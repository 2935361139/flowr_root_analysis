"""Install path-based input/output handling; back up all replaced files."""
import argparse
import ast
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
TARGETS=['scaffold_validation100','record_scaffold_run.py','scaffold_evaluation_v2/evaluate_scaffold_outputs.py','scaffold_study_v1/terminal_control.py']
CONFIG='scaffold_validation100/config.py'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def path_config(path):
    try:
        nodes=ast.parse(path.read_text(encoding='utf-8')).body
        return {n.targets[0].id:ast.literal_eval(n.value) for n in nodes if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    except (OSError,SyntaxError,ValueError):return {}

def check_idle(repo):
    locks=list((repo/'outputs').rglob('RUNNING.lock')) if (repo/'outputs').is_dir() else []
    values=path_config(repo/CONFIG)
    for key in ['GENERATION_DIR','INTERVENTION_INPUT_DIR','INTERVENTION_OUTPUT_DIR']:
        value=values.get(key)
        if isinstance(value,str):
            base=(repo/Path(value).expanduser()).resolve()
            locks.extend(p for p in [base/'RUNNING.lock',base.parent/'RUNNING.lock'] if p.exists())
    if locks:raise RuntimeError(f'Experiment is active: {locks[0]}. Finish it before updating.')

def copy(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if source.is_dir():shutil.copytree(source,dest,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    else:shutil.copy2(source,dest)

def remove(path):
    if path.is_dir() and not path.is_symlink():shutil.rmtree(path)
    elif path.exists() or path.is_symlink():path.unlink()

def install(repo):
    repo=Path(repo).resolve();check_idle(repo)
    manifest=read(HERE/'package_manifest.json');payload=HERE/'payload'
    for name,digest in manifest['payload_sha256'].items():
        if not (payload/name).is_file() or sha(payload/name)!=digest:raise RuntimeError(f'Incomplete package: {name}')
    for name,allowed in manifest['allowed_source_sha256'].items():
        if not (repo/name).is_file() or sha(repo/name) not in allowed:raise RuntimeError(f'Existing method differs or is missing: {name}. Nothing changed.')
    if all((repo/n).is_file() and sha(repo/n)==h for n,h in manifest['payload_sha256'].items() if n!=CONFIG) and (repo/CONFIG).is_file():
        print('Already installed; current config.py preserved.');return None
    for name in TARGETS:
        if (repo/name).is_symlink():raise RuntimeError(f'Refusing to replace a symlink: {name}')
    archive=repo/'archive';archive.mkdir(exist_ok=True)
    backup=Path(tempfile.mkdtemp(prefix='validation_paths_backup_',dir=archive))
    existed=[]
    for name in TARGETS:
        if (repo/name).exists():copy(repo/name,backup/name);existed.append(name)
    hashes={str(p.relative_to(backup)):sha(p) for p in backup.rglob('*') if p.is_file() and not p.is_symlink()}
    (backup/'backup_manifest.json').write_text(json.dumps(dict(repo=str(repo),targets=existed,sha256=hashes),indent=2))
    stage=Path(tempfile.mkdtemp(prefix='validation_paths_stage_',dir=repo));moved=[];placed=[]
    preserve='SDF_PATH' in path_config(repo/CONFIG) and 'INTERVENTION_OUTPUT_DIR' in path_config(repo/CONFIG)
    try:
        for name in TARGETS:copy(payload/name,stage/'new'/name)
        if preserve:shutil.copy2(repo/CONFIG,stage/'new'/CONFIG)
        for name in TARGETS:
            target=repo/name
            if target.exists():
                (stage/'old'/name).parent.mkdir(parents=True,exist_ok=True);target.rename(stage/'old'/name);moved.append(name)
            (stage/'new'/name).rename(target);placed.append(name)
        for name,digest in manifest['payload_sha256'].items():
            if name==CONFIG and preserve:continue
            if sha(repo/name)!=digest:raise RuntimeError(f'Installed bytes differ: {name}')
    except BaseException:
        for name in reversed(placed):remove(repo/name)
        for name in reversed(moved):(stage/'old'/name).rename(repo/name)
        raise
    finally:shutil.rmtree(stage)
    print('Installed. Backup:',backup)
    print('Edit:',repo/CONFIG)
    print('Default settings select one CDK2 ligand and generate 5 samples. Check paths before running.')
    print('Run: python scaffold_validation100/run.py --repo .')
    return backup

def restore(repo,backup):
    repo=Path(repo).resolve();backup=Path(backup).resolve();check_idle(repo)
    if repo/'archive' not in backup.parents:raise RuntimeError('Select a backup inside this repository archive/')
    data=read(backup/'backup_manifest.json')
    if data['repo']!=str(repo) or any(not (backup/n).is_file() or sha(backup/n)!=h for n,h in data['sha256'].items()):
        raise RuntimeError('Backup identity/content differs')
    saved=Path(tempfile.mkdtemp(prefix='validation_paths_before_restore_',dir=repo/'archive'));moved=[];placed=[]
    try:
        for name in TARGETS:
            if (repo/name).exists():
                (saved/name).parent.mkdir(parents=True,exist_ok=True);(repo/name).rename(saved/name);moved.append(name)
            if name in data['targets']:copy(backup/name,repo/name);placed.append(name)
    except BaseException:
        for name in reversed(placed):remove(repo/name)
        for name in reversed(moved):(saved/name).rename(repo/name)
        raise
    print('Restored. Replaced code retained at:',saved)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',type=Path,default=Path.cwd());parser.add_argument('--restore',type=Path)
    args=parser.parse_args()
    if args.restore:restore(args.repo,args.restore)
    else:install(args.repo)
