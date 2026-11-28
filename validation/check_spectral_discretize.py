"""Actual Source rejection parity for spectral discretization."""
from pathlib import Path
import argparse,contextlib,io,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from sklearn.cluster._spectral import discretize

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 for case in range(11):
  x=np.arange(1,19,dtype=float).reshape((6,3));seed=21;restarts=30
  if case==0:x[:,1]=0
  if case==1:x[1,:]=0
  if case==2:x[:]=0
  if case==3:x[0,0]=np.nan
  if case==4:x[0,0]=np.inf
  if case==5:seed=-1
  if case==6:restarts=0
  if case==7:restarts=-1
  if case==8:x=x[:0]
  if case==9:x=x[:,:0]
  if case==10:x[:]=.5e308
  source_error=None;capture=io.StringIO()
  try:
   with warnings.catch_warnings(),contextlib.redirect_stdout(capture):
    warnings.simplefilter('ignore');discretize(x,random_state=seed,max_svd_restarts=restarts)
  except Exception as exc:source_error=f'{type(exc).__name__}: {exc}'
  text=f'{x.shape[0]} {x.shape[1]} {seed} 20 {restarts}\n'+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in x)+'\n';inp=a.report.with_suffix('.in');out=a.report.with_suffix('.out');inp.write_text(text);p=subprocess.run([str(a.executable),str(inp),str(out)],capture_output=True,text=True)
  records.append(dict(case=case,input=text,source_error=source_error,source_stdout=capture.getvalue(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,passed=source_error is not None and p.returncode==1 and 'fscikit-learn:' in p.stdout))
 a.report.write_text(json.dumps(records,indent=2)+'\n');print('Spectral discretize guards',sum(x['passed'] for x in records),'/',len(records));raise SystemExit(not all(x['passed'] for x in records))
if __name__=='__main__':main()
