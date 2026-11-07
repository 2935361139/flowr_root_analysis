"""Install the configurable runner with a complete backup of the replaced files."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCH = Path('scaffold_validation100')
TERMINAL = Path('scaffold_study_v1/terminal_control.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_idle(repo):
    locks = list((repo/'outputs').rglob('RUNNING.lock')) if (repo/'outputs').is_dir() else []
    if locks:
        raise RuntimeError(f'An experiment lock exists: {locks[0]}. Finish the running experiment before installation.')


def install(repo):
    repo = Path(repo).resolve()
    check_idle(repo)
    manifest = read(HERE/'package_manifest.json')
    payload = HERE/'payload'
    for name, digest in manifest['payload_sha256'].items():
        if not (payload/name).is_file() or sha(payload/name) != digest:
            raise RuntimeError(f'Incomplete update package: {name}')
    for name, digest in manifest['previous_source_sha256'].items():
        allowed = {digest}
        if name == str(TERMINAL):
            allowed.add(manifest['payload_sha256'][name])
        if not (repo/name).is_file() or sha(repo/name) not in allowed:
            raise RuntimeError(f'Existing method differs or is missing: {name}. Nothing changed; keep the current files.')
    configurable = str(BATCH/'config.py')
    config_digest = sha(repo/configurable) if (repo/configurable).is_file() else manifest['payload_sha256'][configurable]
    installed = all((repo/name).is_file() and sha(repo/name) == digest
                    for name, digest in manifest['payload_sha256'].items() if name != configurable)
    if installed and (repo/configurable).is_file():
        print('Already installed; your config.py is preserved.')
        return None
    for target in (repo/BATCH, repo/TERMINAL):
        if target.is_symlink():
            raise RuntimeError(f'Symlink target is not supported: {target}')
    archive = repo/'archive'
    archive.mkdir(exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix='validation_config_backup_', dir=archive))
    previous = {}
    if (repo/BATCH).is_dir():
        shutil.copytree(repo/BATCH, backup/BATCH, symlinks=True)
        previous.update({str(p.relative_to(backup)): sha(p)
                         for p in (backup/BATCH).rglob('*') if p.is_file() and not p.is_symlink()})
    (backup/TERMINAL).parent.mkdir(parents=True)
    shutil.copy2(repo/TERMINAL, backup/TERMINAL)
    previous[str(TERMINAL)] = sha(backup/TERMINAL)
    (backup/'backup_manifest.json').write_text(json.dumps({'repo': str(repo), 'sha256': previous}, indent=2)+'\n')
    stage = Path(tempfile.mkdtemp(prefix='validation_config_stage_', dir=repo))
    old_exists = (repo/BATCH).is_dir()
    swapped = False
    try:
        shutil.copytree(payload/BATCH, stage/'new_batch')
        if (repo/configurable).is_file():
            shutil.copy2(repo/configurable, stage/'new_batch/config.py')
        shutil.copy2(payload/TERMINAL, stage/'new_terminal.py')
        if old_exists:
            (repo/BATCH).rename(stage/'old_batch')
        (stage/'new_batch').rename(repo/BATCH)
        swapped = True
        (stage/'new_terminal.py').replace(repo/TERMINAL)
        for name, digest in manifest['payload_sha256'].items():
            if name == configurable:
                digest = config_digest
            if sha(repo/name) != digest:
                raise RuntimeError(f'Installed file check failed: {name}')
    except BaseException:
        if swapped and (repo/BATCH).exists():
            shutil.rmtree(repo/BATCH)
        if (stage/'old_batch').exists():
            (stage/'old_batch').rename(repo/BATCH)
        shutil.copy2(backup/TERMINAL, repo/TERMINAL)
        raise
    finally:
        shutil.rmtree(stage)
    print('Installed. Original files backed up to:', backup)
    print('Edit only:', repo/BATCH/'config.py')
    print('Run: python scaffold_validation100/run.py --repo .')
    return backup


def restore(repo, backup):
    repo, backup = Path(repo).resolve(), Path(backup).resolve()
    check_idle(repo)
    if repo/'archive' not in backup.parents:
        raise RuntimeError('Choose an installer backup inside this repository archive/')
    data = read(backup/'backup_manifest.json')
    if data['repo'] != str(repo):
        raise RuntimeError('Backup belongs to another repository')
    for name, digest in data['sha256'].items():
        if not (backup/name).is_file() or sha(backup/name) != digest:
            raise RuntimeError(f'Backup differs: {name}')
    saved = Path(tempfile.mkdtemp(prefix='validation_before_restore_', dir=repo/'archive'))
    if (repo/BATCH).exists():
        (repo/BATCH).rename(saved/BATCH)
    (saved/TERMINAL).parent.mkdir(parents=True)
    shutil.copy2(repo/TERMINAL, saved/TERMINAL)
    try:
        if (backup/BATCH).is_dir():
            shutil.copytree(backup/BATCH, repo/BATCH, symlinks=True)
        shutil.copy2(backup/TERMINAL, repo/TERMINAL)
    except BaseException:
        if (repo/BATCH).exists():
            shutil.rmtree(repo/BATCH)
        if (saved/BATCH).exists():
            (saved/BATCH).rename(repo/BATCH)
        shutil.copy2(saved/TERMINAL, repo/TERMINAL)
        raise
    print('Restored original code. Replaced code/config preserved in:', saved)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--restore', type=Path)
    args = parser.parse_args()
    if args.restore:
        restore(args.repo, args.restore)
    else:
        install(args.repo)
