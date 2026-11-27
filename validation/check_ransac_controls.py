"""RANSAC parameter/weight acceptance against actual installed sklearn fits."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import numpy as np
from compare_ransac_controls import config,arrays,wire,oracle

def cases():
    for key,values in [('trials',[-1,0,1]),('skips',[-1,0,1]),('stop_n',[-1,0,2]),
                       ('probability',[-.1,0.,1.,1.1,np.nan,np.inf]),('stop_score',[-np.inf,np.inf,np.nan]),
                       ('threshold',[-2.,-1.,0.,np.inf,np.nan,.15])]:
        for value in values:yield key,str(value),{key:value},None
    for count in [-1,0,1,32]:yield 'min_integer',str(count),dict(min_kind=1,min_int=count),None
    for value in [-.1,0.,1e-300,.2,1.,1.1,np.nan,np.inf]:yield 'min_real',str(value),dict(min_kind=2,min_real=value),None
    for name in ['zero','negative_used','negative_unused','nan','inf']:yield 'weight',name,dict(weighted=1,trials=1),name
    for data,model in [(2,0),(0,2)]:yield 'callbacks',f'{data}_{model}',dict(data=data,model=model,skips=0),None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
    for style in [0,1]:
        for name,token,changes,weight in cases():
            c=config(dict(style=style)|changes);data=list(arrays(c))
            if weight:
                w=data[2]
                if weight=='zero':w.fill(0.)
                if weight=='negative_used':w[2]=-1.
                if weight=='negative_unused':w[0]=-1.
                if weight=='nan':w[0]=np.nan
                if weight=='inf':w[0]=np.inf
            source_error=None
            try:oracle(c,data)
            except (ValueError,ZeroDivisionError,UnboundLocalError,OverflowError) as e:source_error=dict(type=type(e).__name__,message=str(e))
            text='1\n'+wire(c,data);p=subprocess.run([str(a.executable)],input=text,text=True,capture_output=True,timeout=60);output=p.stdout+p.stderr
            controlled=p.returncode!=0 and 'fscikit-learn:' in output and not any(v in output for v in ['SIGFPE','SIGSEGV','Fortran runtime error'])
            passed=controlled if source_error else p.returncode==0
            records.append(dict(style=style,name=name,token=token,case=c,input=text,source_error=source_error,native_returncode=p.returncode,native_output=output,passed=passed))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),Path('validation/compare_ransac_controls.py'),a.executable,Path('src/sklearn_linear_model.f90')]
    d=dict(configurations=len(records),passed=all(r['passed'] for r in records),records=records,sha256={str(p):sha(p) for p in files})
    a.report.write_text(json.dumps(d,indent=2)+'\n');print('cases',len(records),'failed',[(r['style'],r['name'],r['token']) for r in records if not r['passed']]);assert d['passed']
if __name__=='__main__':main()
