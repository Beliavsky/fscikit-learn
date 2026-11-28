"""CSR passive-aggressive validation decisions and a bounded-memory fit/query."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import PassiveAggressiveClassifier,PassiveAggressiveRegressor
from check_minibatch_sparse_memory import peak_memory


def source_reject(mode):
    dense=np.array([[0.,0.],[-1.,2.],[2.,1.],[.5,-.5],[1.,1.],[-2.,1.]])
    x=sp.csr_matrix(dense);y=2*dense[:,0]-.5*dense[:,1]+.3;labels=np.array([0,1,2,0,1,2])
    r=PassiveAggressiveRegressor();c=PassiveAggressiveClassifier()
    if mode=='reg_unfitted':r.predict(x);return
    if mode=='class_unfitted':c.decision_function(x);return
    if mode=='empty':x=sp.csr_matrix((0,2));y=np.array([])
    if mode=='zero_width':x=sp.csr_matrix((6,0))
    if mode in ['nan','inf']:x.data[0]=np.nan if mode=='nan' else np.inf
    if mode=='pointer':x.indptr[0]=-1;x.check_format(full_check=True);return
    if mode=='column':x.indices[0]=2;x.check_format(full_check=True);return
    if mode=='target_count':r.fit(x,y[:5]);return
    if mode=='target_nan':y[0]=np.nan
    if mode=='class_count':c.fit(x,labels[:2]);return
    if mode=='reg_width':r.fit(x,y).predict(sp.csr_matrix((6,3)));return
    if mode=='class_width':c.fit(x,labels).decision_function(sp.csr_matrix((6,3)));return
    if mode=='query_nan':
        r.fit(x,y);x.data[0]=np.nan;r.predict(x);return
    if mode=='fit_weight':r.fit(x,y,sample_weight=np.ones(6));return
    r.fit(x,y)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();before=sha(args.executable);guards=[]
    for mode in ['reg_unfitted','class_unfitted','unallocated','empty','zero_width','nan','inf','pointer','column',
                 'target_count','target_nan','class_count','reg_width','class_width','query_nan','fit_weight']:
        oracle='native_typed_adapter' if mode=='unallocated' else 'source_csr_format' if mode in ['pointer','column'] else 'source'
        if mode=='unallocated':source='CSR buffers must be allocated'
        else:
            try:
                with warnings.catch_warnings():warnings.simplefilter('ignore');source_reject(mode)
            except (ValueError,TypeError) as error:source=str(error)
            else:raise AssertionError(('Source accepted',mode))
        run=subprocess.run([str(args.executable.resolve()),mode],capture_output=True,text=True);message=run.stdout+run.stderr
        assert run.returncode and 'fscikit-learn:' in message,(mode,message)
        assert 'SIGFPE' not in message and 'Fortran runtime error' not in message,(mode,message)
        guards.append(dict(mode=mode,oracle=oracle,source=source,native=message,passed=True))
    process=subprocess.Popen([str(args.executable.resolve()),'memory'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    stdout,stderr=process.communicate();peak=peak_memory(process)
    assert process.returncode==0 and 'Large CSR passive-aggressive memory case: PASS' in stdout,(stdout,stderr)
    assert 0<peak<128*1024**2,peak
    assert before==sha(args.executable)
    args.report.write_text(json.dumps(dict(passed=True,guards=guards,peak_working_set_bytes=peak,limit_bytes=128*1024**2,
        dense_design_bytes=512*500000*8,executable_sha256=before,validator_sha256=sha(__file__)),indent=2)+'\n')
    print(len(guards),'CSR PA guards and',peak,'byte memory peak PASS')


if __name__=='__main__':main()
