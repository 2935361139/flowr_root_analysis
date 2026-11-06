"""Install only the packaged analysis files, backing up any existing copies."""
import argparse, hashlib, json, shutil
from pathlib import Path
from datetime import datetime
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
root=a.repo.resolve();src=Path(__file__).resolve().parent
if not (root/'record_scaffold_run.py').is_file():raise SystemExit('STOP: not the FLOWR project directory')
for required in ('scaffold_study_v1/io_utils.py','scaffold_geometry_audit_v1/flowr_adapter.py',
                 'scaffold_evaluation_v2/evaluate_scaffold_outputs.py'):
 if not (root/required).is_file():raise SystemExit('STOP: required existing file missing: '+required)
manifest=json.loads((src/'package_manifest.json').read_text())
files=[]
for name,digest in manifest.items():
 path=Path(name)
 if path.is_absolute() or '..' in path.parts:raise SystemExit('Invalid package path')
 if hashlib.sha256((src/path).read_bytes()).hexdigest()!=digest:raise SystemExit('Package integrity error: '+name)
 if path.parts[0] in ('scaffold_geometry_audit_v1','scaffold_study_v1'):
  target=root/path
  if target.is_symlink() or not target.resolve().is_relative_to(root):raise SystemExit('Unsafe installation path')
  files.append(path)
backup=root/'archive'/('rescue_v1_backup_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
backup.mkdir(parents=True,exist_ok=False)
for path in files:
 target=root/path
 if target.exists():
  (backup/path).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,backup/path)
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/path,target)
print('Installed analysis update. Previous files saved in:',backup)
