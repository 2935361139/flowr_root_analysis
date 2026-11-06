"""Install one label-bookkeeping correction, with exact version guards and backup."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def install(repo):
    change = json.loads((HERE/'compatibility_change.json').read_text())
    target = repo/change['target']; source = HERE/'terminal_control.py'
    if sha(source) != change['after_sha256']:
        raise RuntimeError('Package file changed; do not install')
    if sha(target) == change['after_sha256']:
        print('Compatibility fix already installed; no change.')
        return
    if sha(target) != change['before_sha256']:
        raise RuntimeError('Installed terminal_control.py differs from the exercised v2. Preserve it and send the error.')
    archive = repo/'archive'; archive.mkdir(exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix='validation100_compat_backup_', dir=archive))
    shutil.copy2(target, backup/'terminal_control.py')
    shutil.copy2(HERE/'compatibility_change.json', backup/'compatibility_change.json')
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='terminal_compat_', delete=False) as f:
            tmp = Path(f.name); f.write(source.read_bytes())
        shutil.copymode(target, tmp)
        if sha(tmp) != change['after_sha256']:
            raise RuntimeError('Write verification failed; original file preserved')
        tmp.replace(target)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    print('Installed one label correction. Backup:', backup)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    a = p.parse_args(); install(a.repo.resolve())
