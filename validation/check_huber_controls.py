"""Actual sklearn Huber parameter, weight, shape and query acceptance."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings,sys,os
from concurrent.futures import ThreadPoolExecutor
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.linear_model import HuberRegressor

def oracle(representation,name,value):
    x=np.array([[-2.,1.],[-1.,0.],[0.,1.],[1.,2.],[2.,-1.]])
    y=np.array([2.,-1.,3.,1.,0.]);w=np.ones(5);m=HuberRegressor()
    if name in ['epsilon','alpha','tol','max_iter']:m.set_params(**{name:int(value) if name=='max_iter' else value})
    if name=='weight_nan':w[0]=np.nan
    if name=='weight_inf':w[0]=np.inf
    if name=='weight_negative':w[0]=-1.
    if name=='weight_all_negative':w.fill(-1.)
    if name=='weight_zero':w.fill(0.)
    if name=='weight_length':w=w[:2]
    if name=='x_nan':x[0,0]=np.nan
    if name=='x_inf':x[0,0]=np.inf
    if name=='y_nan':y[0]=np.nan
    if name=='y_inf':y[0]=np.inf
    if name=='empty_rows':x=x[:0];y=y[:0];w=w[:0]
    if name=='empty_columns':x=x[:,:0]
    if name=='target_column':y=y[:,None]
    if name=='target_matrix':y=np.column_stack([y,y])
    if name.startswith('warm_'):
        m.fit(x,y);m.warm_start=True
        x=np.column_stack([x,x[:,0]]) if name=='warm_wider' else x[:,:1]
    convert=lambda x:x if representation=='dense' else sp.csr_matrix(x)
    m.fit(convert(x),y,sample_weight=w)
    if name=='query_width':x=x[:,:1]
    if name=='query_empty':x=x[:0]
    if name=='query_nan':x[0,0]=np.nan
    if name=='query_inf':x[0,0]=np.inf
    if name.startswith('query_'):m.predict(convert(x))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
    cases=[(name,value) for name,values in [('epsilon',[.5,1.,np.nan,np.inf]),('alpha',[-1.,0.,np.nan,np.inf]),('tol',[-1.,0.,np.nan,np.inf]),('max_iter',[-1,0,1])] for value in values]
    cases += [(name,0.) for name in ['weight_nan','weight_inf','weight_negative','weight_all_negative','weight_zero','weight_length','x_nan','x_inf','y_nan','y_inf','empty_rows','empty_columns','target_column','target_matrix','warm_wider','warm_narrower','query_width','query_empty','query_nan','query_inf']]
    def check(case):
        representation,name,value=case
        source=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--source-worker',representation,name,str(value)],text=True,capture_output=True,timeout=60)
        error=None;source_result=None
        if source.stdout.strip():
            try:source_result=json.loads(source.stdout)
            except json.JSONDecodeError:pass
        if source_result is not None:error=source_result['error']
        abnormal=source.returncode!=0
        if not abnormal:assert source_result is not None,(representation,name,source.stdout,source.stderr)
        p=subprocess.run([str(a.executable),representation,name,str(value)],text=True,capture_output=True,timeout=30);output=p.stdout+p.stderr
        controlled=p.returncode!=0 and 'fscikit-learn:' in output and not any(x in output for x in ['SIGFPE','SIGSEGV','Fortran runtime error'])
        passed=False if abnormal else controlled if error is not None else p.returncode==0 and 'FIT ACCEPTED' in output
        return dict(representation=representation,name=name,token=str(value),source_error=error,
            source_returncode=source.returncode,source_stdout=source.stdout,source_stderr=source.stderr,
            source_abnormal_termination=abnormal,adaptation='controlled native diagnostic after abnormal Source process termination' if abnormal else None,
            native_returncode=p.returncode,native_output=output,passed=passed)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(check,[(representation,name,value) for representation in ['dense','csr'] for name,value in cases]))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),a.executable,Path('src/sklearn_huber_solver.f90'),Path('src/sklearn_linear_model.f90'),Path('app/reference_huber_invalid_driver.f90')]
    d=dict(configurations=len(records),passed=all(r['passed'] for r in records),records=records,sklearn=sklearn.__version__,sha256={str(p):sha(p) for p in files})
    a.report.write_text(json.dumps(d,indent=2)+'\n');print('cases',len(records),'failures',[(r['representation'],r['name'],r['token']) for r in records if not r['passed']]);assert d['passed']
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--source-worker':
        error=None
        try:
            with warnings.catch_warnings(),np.errstate(all='ignore'):
                warnings.simplefilter('ignore');oracle(sys.argv[2],sys.argv[3],float(sys.argv[4]))
        except Exception as e:error=dict(type=type(e).__name__,message=str(e))
        print(json.dumps(dict(error=error)),flush=True)
    else:main()
