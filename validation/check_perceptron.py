"""Perceptron Source rejection decisions, explicit engine restrictions and wide CSR memory."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Perceptron
from check_minibatch_sparse_memory import peak_memory


MODES=['missing_classes','one_class','changed_classes','unknown_class','early_partial','balanced_partial',
       'bad_penalty','bad_alpha','bad_mix','bad_eta','bad_tol','bad_seed','bad_fraction','bad_weight_size',
       'nan_weight','bad_init','wrong_width']
RESTRICTIONS=['negative_weight','nonpositive_class_weight']


def source_reject(mode):
    x=np.column_stack([[-2.,-1.,0.,1.,2.,3.],np.ones(6)]);labels=[-5,-5,-5,7,7,7];w=np.ones(6)
    c=Perceptron(max_iter=3,tol=None)
    if mode=='missing_classes':c.partial_fit(x,labels)
    elif mode=='one_class':c.fit(x,[7]*6)
    elif mode=='changed_classes':c.partial_fit(x,labels,classes=[-5,7]);c.partial_fit(x,labels,classes=[-5,9])
    elif mode=='unknown_class':c.partial_fit(x,labels,classes=[-5,9])
    elif mode=='early_partial':c.early_stopping=True;c.partial_fit(x,labels,classes=[-5,7])
    elif mode=='balanced_partial':c.class_weight='balanced';c.partial_fit(x,labels,classes=[-5,7])
    elif mode=='bad_penalty':c.penalty='bad';c.fit(x,labels)
    elif mode=='bad_alpha':c.alpha=-1.;c.fit(x,labels)
    elif mode=='bad_mix':c.l1_ratio=2.;c.fit(x,labels)
    elif mode=='bad_eta':c.eta0=0.;c.fit(x,labels)
    elif mode=='bad_tol':c.tol=-.5;c.fit(x,labels)
    elif mode=='bad_seed':c.random_state=-1;c.fit(x,labels)
    elif mode=='bad_fraction':c.validation_fraction=0.;c.fit(x,labels)
    elif mode=='bad_weight_size':c.fit(x,labels,sample_weight=[1.])
    elif mode=='nan_weight':w[0]=np.nan;c.fit(x,labels,sample_weight=w)
    elif mode=='bad_init':c.fit(x,labels,coef_init=[1.])
    elif mode=='wrong_width':c.partial_fit(x,labels,classes=[-5,7]);c.partial_fit(x[:,:1],labels)
    elif mode=='negative_weight':w[0]=-1.;c.fit(x,labels,sample_weight=w)
    elif mode=='nonpositive_class_weight':c.class_weight={-5:0.,7:1.};c.fit(x,labels)
    else:raise AssertionError(mode)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();before=sha(args.executable);guards=[]
    for mode in MODES:
        try:
            with warnings.catch_warnings():warnings.simplefilter('ignore');source_reject(mode)
        except (ValueError,TypeError,AttributeError) as error:source=str(error)
        else:raise AssertionError(('Source accepted',mode))
        run=subprocess.run([str(args.executable.resolve()),mode],capture_output=True,text=True);message=run.stdout+run.stderr
        assert run.returncode and 'fscikit-learn:' in message,(mode,message)
        assert 'SIGFPE' not in message and 'Fortran runtime error' not in message,(mode,message)
        guards.append(dict(mode=mode,oracle='source',source=source,native=message,passed=True))
    for mode in RESTRICTIONS:
        with warnings.catch_warnings():warnings.simplefilter('ignore');source_reject(mode)
        run=subprocess.run([str(args.executable.resolve()),mode],capture_output=True,text=True)
        message=run.stdout+run.stderr
        assert run.returncode and 'fscikit-learn:' in message and 'SIGFPE' not in message,(mode,message)
        guards.append(dict(mode=mode,oracle='native_adapter',source='Source accepts; shared SGD engine restriction',native=message,passed=True))
    process=subprocess.Popen([str(args.executable.resolve()),'memory'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    stdout,stderr=process.communicate();peak=peak_memory(process)
    assert process.returncode==0 and 'Large CSR Perceptron memory case: PASS' in stdout,(stdout,stderr)
    assert 0<peak<128*1024**2,peak;assert before==sha(args.executable)
    args.report.write_text(json.dumps(dict(passed=True,guards=guards,peak_working_set_bytes=peak,limit_bytes=128*1024**2,
        dense_design_bytes=256*500000*8,executable_sha256=before,validator_sha256=sha(__file__)),indent=2)+'\n')
    print(len(guards),'Perceptron Source/adapter guards and',peak,'byte memory peak PASS')


if __name__=='__main__':main()
