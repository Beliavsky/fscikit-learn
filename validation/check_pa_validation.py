"""Actual Source/native PA validation rejection decisions and wide CSR memory."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import PassiveAggressiveClassifier,PassiveAggressiveRegressor
from check_minibatch_sparse_memory import peak_memory


MODES=['reg_partial','class_partial','reg_partial_csr','class_partial_csr','reg_fraction_zero',
       'class_fraction_one','reg_fraction_nan','class_fraction_disabled','reg_empty_train',
       'class_rare','class_small_validation','class_small_train']


def source_reject(mode):
    x=np.column_stack([np.sin(np.arange(1,13)),np.cos(.3*np.arange(1,13))]);y=2*x[:,0]-x[:,1];labels=np.arange(1,13)%3
    r=PassiveAggressiveRegressor(early_stopping=True,validation_fraction=.25,random_state=42)
    c=PassiveAggressiveClassifier(early_stopping=True,validation_fraction=.25,random_state=42)
    if mode.endswith('csr'):x=sp.csr_matrix(x)
    if mode.startswith('reg_partial'):r.partial_fit(x,y)
    elif mode.startswith('class_partial'):c.partial_fit(x,labels,classes=[0,1,2])
    elif mode=='reg_fraction_zero':r.validation_fraction=0.;r.fit(x,y)
    elif mode=='class_fraction_one':c.validation_fraction=1.;c.fit(x,labels)
    elif mode=='reg_fraction_nan':r.validation_fraction=np.nan;r.fit(x,y)
    elif mode=='class_fraction_disabled':c.early_stopping=False;c.validation_fraction=-.1;c.fit(x,labels)
    elif mode=='reg_empty_train':r.fit(x[:1],y[:1])
    elif mode=='class_rare':c.fit(x,[0]*8+[1]*3+[2])
    elif mode=='class_small_validation':c.validation_fraction=.1;c.fit(x,labels)
    elif mode=='class_small_train':c.validation_fraction=.9;c.fit(x,labels)
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
    process=subprocess.Popen([str(args.executable.resolve()),'memory'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    stdout,stderr=process.communicate();peak=peak_memory(process)
    assert process.returncode==0 and 'Large validation CSR passive-aggressive memory case: PASS' in stdout,(stdout,stderr)
    assert 0<peak<128*1024**2,peak;assert before==sha(args.executable)
    args.report.write_text(json.dumps(dict(passed=True,guards=guards,peak_working_set_bytes=peak,limit_bytes=128*1024**2,
        dense_design_bytes=256*500000*8,executable_sha256=before,validator_sha256=sha(__file__)),indent=2)+'\n')
    print(len(guards),'validation Source/native guards and',peak,'byte memory peak PASS')


if __name__=='__main__':main()
