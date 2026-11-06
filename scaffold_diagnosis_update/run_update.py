"""Run geometry localization and existing study aggregation once, then package reports."""
import argparse,json,subprocess,sys,zipfile,traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
p.add_argument('--cohort',default='outputs/sh_restart_v1');a=p.parse_args()
repo=a.repo.resolve();batch=(repo/a.cohort).resolve();out=batch/'localization_v2'
if out.exists():raise SystemExit('STOP: localization_v2 exists; preserve it and inspect its report. No automatic retry.')
out.mkdir(parents=True);status={'completed':False,'stage':'geometry','errors':[],'generations':0,'interventions':0}
try:
 commands=[('geometry',[sys.executable,str(repo/'scaffold_geometry_audit_v1/run_audit.py'),
 '--cohort-dir',str(batch),'--diagnostic-dir',str(batch/'diagnostics'),
 '--output-dir',str(out/'geometry'),'--repo',str(repo)]),
 ('study',[sys.executable,str(repo/'scaffold_study_v1/run.py'),
 '--quality-report',str(batch/'quality/quality_report.json'),'--diagnostic-dir',str(batch/'diagnostics'),
 '--geometry-dir',str(out/'geometry'),'--output-dir',str(out/'study')])]
 for name,cmd in commands:
  status['stage']=name;print('Running',name,flush=True)
  with (out/(name+'.log')).open('w') as log:
   result=subprocess.run(cmd,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(name+' stopped; see '+name+'.log')
 status.update(completed=True,stage='finished')
except Exception as exc:
 status['errors'].append(str(exc));(out/'error.txt').write_text(traceback.format_exc())
finally:
 (out/'status.json').write_text(json.dumps(status,indent=2))
 with zipfile.ZipFile(out/'localization_review.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(out.rglob('*')):
   if f.is_file() and f.suffix in ('.json','.csv','.md','.log','.txt'):z.write(f,f.relative_to(out))
 print('Completed:',status['completed']);print('Upload:',out/'localization_review.zip')
sys.exit(0 if status['completed'] else 1)
