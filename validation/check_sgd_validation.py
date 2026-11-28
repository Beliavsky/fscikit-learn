"""Compare actual sklearn/native early-stopping rejection decisions."""
from pathlib import Path
import argparse, hashlib, json, subprocess, warnings
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import SGDRegressor, SGDClassifier


def source_reject(mode):
    i=np.arange(1,25);x=np.column_stack([i/24,np.sin(i)]);y=.4+x[:,0]-x[:,1];labels=i%3;weights=np.ones(24)
    model=(SGDClassifier if mode.startswith('cpartial') or mode in ['singleton','small_validation','small_training','zero_validation_csr'] else SGDRegressor)(
        early_stopping=True,validation_fraction=.25,random_state=7,max_iter=40,n_iter_no_change=2,tol=.1)
    if mode.startswith('rpartial'):model.partial_fit(sp.csr_matrix(x) if mode.endswith('csr') else x,y);return
    if mode.startswith('cpartial'):model.partial_fit(sp.csr_matrix(x) if mode.endswith('csr') else x,labels,classes=[0,1,2]);return
    if mode in ['fraction_zero','fraction_one','fraction_nan','fraction_disabled']:
        model.validation_fraction={'fraction_zero':0.,'fraction_one':1.,'fraction_nan':np.nan,'fraction_disabled':0.}[mode]
        if mode=='fraction_disabled':model.early_stopping=False
    if mode=='empty_training':model.validation_fraction=.99
    if mode=='small_validation':model.validation_fraction=.01
    if mode=='small_training':model.validation_fraction=.95
    if mode=='singleton':labels=np.zeros(24,dtype=int);labels[0]=1
    if mode in ['zero_validation','zero_validation_csr']:
        mask=model._make_validation_split(labels if mode.endswith('csr') else y,weights>0);weights[mask]=0
    model.fit(sp.csr_matrix(x) if mode.endswith('csr') else x,labels if isinstance(model,SGDClassifier) else y,sample_weight=weights)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();before=sha(args.executable);guards=[]
    for mode in ['rpartial','rpartial_csr','cpartial','cpartial_csr','fraction_zero','fraction_one','fraction_nan',
                 'fraction_disabled','empty_training','singleton','small_validation','small_training','zero_validation','zero_validation_csr']:
        try:
            with warnings.catch_warnings():warnings.simplefilter('ignore');source_reject(mode)
        except ValueError as error:source=str(error)
        else:raise AssertionError(('Source accepted',mode))
        run=subprocess.run([str(args.executable.resolve()),mode],capture_output=True,text=True)
        message=run.stdout+run.stderr
        assert run.returncode and 'fscikit-learn:' in message,(mode,message)
        assert 'SIGFPE' not in message and 'Fortran runtime error' not in message,(mode,message)
        guards.append(dict(mode=mode,source=source,native=message,passed=True))
    assert before==sha(args.executable)
    args.report.write_text(json.dumps(dict(passed=True,guards=guards,executable_sha256=before,validator_sha256=sha(__file__)),indent=2)+'\n')
    print(len(guards),'early-stopping Source/native guards PASS')


if __name__=='__main__':main()
