"""Guard and peak-memory checks for CSR primal linear SVMs."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np
import scipy.sparse as sp
from sklearn.svm import LinearSVC, LinearSVR
from check_minibatch_sparse_memory import peak_memory


def reject_source(name):
    x=sp.csr_matrix([[-1.],[1.]]);y=np.array([-3,9])
    if name=='unfitted': LinearSVC().predict(x)
    elif name in ['nan','inf']:
        x.data[0]=np.nan if name=='nan' else np.inf
        LinearSVC().fit(x,y)
    elif name=='shape': LinearSVC().fit(x,y).predict(sp.csr_matrix((2,2)))
    elif name=='bad_index':
        x.indices[0]=-1;x.check_format(full_check=True)
    elif name=='bad_ptr':
        x.indptr[0]=-1;x.check_format(full_check=True)
    elif name=='bad_y': LinearSVC().fit(x,[-3])
    elif name=='one_class': LinearSVC().fit(x,[-3,-3])
    elif name=='bad_c': LinearSVC(C=0).fit(x,y)
    elif name=='bad_epsilon': LinearSVR(epsilon=-1).fit(x,[-2.,2.])
    else: raise AssertionError(name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    before=sha(args.executable);guards=[]
    native={'negative_weight':'existing_nonnegative_weight_policy',
            'bad_class_weight':'native_class_weight_array_shape',
            'conflicting_weights':'native_mutually_exclusive_weight_controls'}
    for name in ['unfitted','nan','inf','shape','bad_index','bad_ptr','bad_y','one_class',
                 'negative_weight','bad_c','bad_epsilon','bad_class_weight','conflicting_weights']:
        if name in native: source=native[name]
        else:
            try: reject_source(name)
            except (ValueError,TypeError) as error: source=type(error).__name__
            else: raise AssertionError(('source accepts',name))
        run=subprocess.run([str(args.executable),name],capture_output=True,text=True)
        message=run.stdout+run.stderr
        assert run.returncode and 'fscikit-learn:' in message,(name,message)
        assert not any(s in message for s in ['SIGFPE','Fortran runtime error']),(name,message)
        guards.append(dict(name=name,source=source,passed=True))
    process=subprocess.Popen([str(args.executable),'memory'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    stdout,stderr=process.communicate();peak=peak_memory(process)
    assert process.returncode==0 and 'Large CSR linear SVM memory case: PASS' in stdout,(stdout,stderr)
    assert 0<peak<128*1024**2,peak
    assert before==sha(args.executable)
    result=dict(passed=True,guards=guards,peak_working_set_bytes=peak,limit_bytes=128*1024**2,
        n_samples=128,n_features=200000,dense_design_bytes=128*200000*8,
        dense_augmented_hessian_bytes=200001**2*8,executable_sha256=before)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(len(guards),'guards; peak memory',peak,'bytes PASS')


if __name__=='__main__':main()
