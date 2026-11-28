"""Actual Source/native PA streaming rejection decisions and wide CSR memory."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import PassiveAggressiveClassifier,PassiveAggressiveRegressor
from check_minibatch_sparse_memory import peak_memory


def source_reject(mode):
    one=np.ones((1,1));two=np.ones((1,2));r=PassiveAggressiveRegressor();c=PassiveAggressiveClassifier()
    if mode=='missing_classes':c.partial_fit(one,[-5])
    elif mode=='one_class':c.partial_fit(one,[-5],classes=[-5])
    elif mode=='changed_classes':c.partial_fit(one,[-5],classes=[-5,7]);c.partial_fit(one,[7],classes=[-5,9])
    elif mode=='unknown_class':c.partial_fit(one,[9],classes=[-5,7])
    elif mode.startswith('reg_width'):
        if mode.endswith('csr'):one=sp.csr_matrix(one);two=sp.csr_matrix(two)
        r.partial_fit(one,[1.]);r.partial_fit(two,[1.])
    elif mode.startswith('class_width'):
        if mode.endswith('csr'):one=sp.csr_matrix(one);two=sp.csr_matrix(two)
        c.partial_fit(one,[-5],classes=[-5,7]);c.partial_fit(two,[7])
    elif mode=='balanced_first':c.class_weight='balanced';c.partial_fit(one,[-5],classes=[-5,7])
    elif mode=='balanced_missing':c.class_weight='balanced';c.fit(np.array([[1.],[2.]]),[-5,7]);c.partial_fit(one,[-5])
    elif mode=='reg_enable_average':r.partial_fit(one,[1.]);r.average=1;r.partial_fit(one,[1.])
    elif mode=='class_enable_average':c.partial_fit(one,[-5],classes=[-5,7]);c.average=1;c.partial_fit(one,[7])
    else:raise AssertionError(mode)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();before=sha(args.executable);guards=[]
    for mode in ['missing_classes','one_class','changed_classes','unknown_class','reg_width','reg_width_csr',
                 'class_width','class_width_csr','balanced_first','balanced_missing','reg_enable_average','class_enable_average']:
        try:
            with warnings.catch_warnings():warnings.simplefilter('ignore');source_reject(mode)
        except (ValueError,TypeError,AttributeError) as error:source=str(error)
        else:raise AssertionError(('Source accepted',mode))
        run=subprocess.run([str(args.executable.resolve()),mode],capture_output=True,text=True);message=run.stdout+run.stderr
        assert run.returncode and 'fscikit-learn:' in message,(mode,message)
        assert 'SIGFPE' not in message and 'Fortran runtime error' not in message,(mode,message)
        guards.append(dict(mode=mode,oracle='source',source=source,native=message,passed=True))
    process=subprocess.Popen([str(args.executable.resolve()),'memory'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    stdout,stderr=process.communicate();peak=peak_memory(process)
    assert process.returncode==0 and 'Large streaming CSR passive-aggressive memory case: PASS' in stdout,(stdout,stderr)
    assert 0<peak<128*1024**2,peak;assert before==sha(args.executable)
    args.report.write_text(json.dumps(dict(passed=True,guards=guards,peak_working_set_bytes=peak,limit_bytes=128*1024**2,
        dense_design_bytes=256*500000*8,executable_sha256=before,validator_sha256=sha(__file__)),indent=2)+'\n')
    print(len(guards),'streaming Source/native guards and',peak,'byte memory peak PASS')


if __name__=='__main__':main()
