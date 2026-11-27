"""Compare actual sklearn and native rejection of invalid GLM inputs."""
from pathlib import Path
import argparse,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
from sklearn.linear_model import TweedieRegressor

def source(which):
    x=np.array([[.1,.4],[.2,.1],[.3,.2],[.4,.3]])
    y=np.arange(1.,5.);w=np.ones(4);m=TweedieRegressor(solver='newton-cholesky')
    if which==0:m.alpha=-1
    if which==1:m.tol=0
    if which==2:m.max_iter=-1
    if which==3:m.link='invalid'
    if which==4:m.power=np.nan
    if which==5:m.alpha=np.inf
    if which==6:m.tol=np.nan
    if which==7:w[:]=0
    if which==8:w[0]=np.nan
    if which==9:w[0]=np.inf
    if which==10:y[0]=np.nan
    if which==11:x[0,0]=np.inf
    if which in (12,13):m.power=(.5 if which==12 else 1.5);y[0]=-1
    if which in (14,15):m.power=(2 if which==14 else 3);y[0]=0
    if which==16:w=w[:3]
    if which==17:y=np.column_stack([y,y])
    if which==18:y=y[:3]
    if which==19:return m.score(x,y)
    if which in (20,21,22,23):
        if which==23:m.power=3
        m.fit(x,y)
        if which==20:m.warm_start=True;return m.fit(x[:,:1],y)
        if which==21:return m.score(x,y,sample_weight=w[:3])
        if which==22:return m.score(x,y,sample_weight=w*0)
        y[0]=0;return m.score(x,y)
    return m.fit(x,y,sample_weight=w)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
    for which in range(24):
        error=None
        try:
            with warnings.catch_warnings():warnings.simplefilter('ignore');source(which)
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        p=subprocess.run([str(a.executable),str(which)],capture_output=True,text=True)
        passed=error is not None and p.returncode==1 and 'fscikit-learn:' in p.stdout
        records.append(dict(case=which,source_error=error,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,passed=passed))
    a.report.write_text(json.dumps(records,indent=2)+'\n');print('GLM guards:',sum(r['passed'] for r in records),'/',len(records));raise SystemExit(not all(r['passed'] for r in records))
if __name__=='__main__':main()
