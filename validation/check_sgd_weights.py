"""Signed-weight validation errors and wide CSR SGD/Perceptron memory."""
from pathlib import Path
import argparse, hashlib, json, subprocess, warnings
import numpy as np
import scipy.sparse as sp
from check_minibatch_sparse_memory import peak_memory
from sklearn.linear_model import SGDRegressor, SGDClassifier, Perceptron


MODES=['reg_nonpositive_validation','clf_nonpositive_validation','pc_nonpositive_validation',
       'reg_zero_validation','clf_zero_validation','pc_zero_validation','class_weight_nan']


def source_reject(mode):
    i=np.arange(1,41);x=np.column_stack([i/40,np.sin(i)]);y=x[:,0]-x[:,1];labels=i%2;weights=np.ones(40)
    cls=SGDRegressor if mode.startswith('reg_') else Perceptron if mode.startswith('pc_') else SGDClassifier
    model=cls(early_stopping=True,validation_fraction=.3,random_state=0,max_iter=3,tol=None)
    if 'nonpositive' in mode:weights[:]=-1
    elif mode=='class_weight_nan':model.class_weight={0:np.nan,1:1.}
    else:
        mask=model._make_validation_split(y if mode.startswith('reg_') else labels,np.ones(40,dtype=bool))
        weights[mask]=np.tile([1.,-1.],sum(mask)//2)
    model.fit(x,y if mode.startswith('reg_') else labels,sample_weight=weights)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();before=sha(args.executable);guards=[]
    for mode in MODES:
        try:
            with warnings.catch_warnings():warnings.simplefilter('ignore');source_reject(mode)
        except (ValueError,ZeroDivisionError) as error:source=str(error)
        else:raise AssertionError(('Source accepted',mode))
        run=subprocess.run([str(args.executable.resolve()),mode],capture_output=True,text=True)
        message=run.stdout+run.stderr
        assert run.returncode and 'fscikit-learn:' in message,(mode,message)
        assert 'SIGFPE' not in message and 'Fortran runtime error' not in message,(mode,message)
        guards.append(dict(mode=mode,source=source,native=message,passed=True))
    memory={}
    for mode in ['memory_reg','memory_clf','memory_pc']:
        process=subprocess.Popen([str(args.executable.resolve()),mode],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        stdout,stderr=process.communicate();peak=peak_memory(process)
        assert process.returncode==0 and 'Large signed-weight CSR memory case: PASS' in stdout,(mode,stdout,stderr)
        assert 0<peak<128*1024**2,(mode,peak)
        memory[mode]=dict(peak_working_set_bytes=peak,limit_bytes=128*1024**2,dense_design_bytes=256*500000*8)
    assert before==sha(args.executable)
    args.report.write_text(json.dumps(dict(passed=True,guards=guards,memory=memory,executable_sha256=before,validator_sha256=sha(__file__)),indent=2)+'\n')
    print(len(guards),'signed-weight Source/native guards and memory peaks',memory,'PASS')


if __name__=='__main__':main()
