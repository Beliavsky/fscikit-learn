"""Actual Source/native PA warm-fit rejection decisions and accepted public replacement."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import PassiveAggressiveClassifier,PassiveAggressiveRegressor
from check_minibatch_sparse_memory import peak_memory


MODES=['reg_width','class_width','class_heads','reg_nan','reg_inf','reg_bias_nan',
       'class_nan','class_inf','class_bias_nan','class_bias_shape','class_bias_absent','reg_enable_average']


def source_reject(mode):
    x=np.array([[1.],[2.],[-1.],[-2.]]);y=[1.,0.,-1.,0.];labels=[0,1,0,1]
    opts=dict(max_iter=1,tol=None,shuffle=False,warm_start=True)
    r=PassiveAggressiveRegressor(fit_intercept=False,**opts);c=PassiveAggressiveClassifier(**opts)
    if mode=='reg_average_nan':r.average=1
    r.fit(x,y);c.fit(x,labels)
    bad=np.inf if 'inf' in mode else np.nan
    if mode=='reg_width':r.fit(np.hstack([x,x]),y)
    elif mode=='class_width':c.fit(np.hstack([x,x]),labels)
    elif mode=='class_heads':c.fit(x,[0,1,2,0])
    elif mode in ['reg_nan','reg_inf']:r.coef_[:]=bad;r.fit(x,y)
    elif mode=='reg_bias_nan':r.intercept_[:]=bad;r.fit(x,y)
    elif mode in ['class_nan','class_inf']:c.coef_[:]=bad;c.fit(x,labels)
    elif mode=='class_bias_nan':c.intercept_[:]=bad;c.fit(x,labels)
    elif mode=='class_bias_shape':c.intercept_=np.zeros(2);c.fit(x,labels)
    elif mode=='class_bias_absent':del c.intercept_;c.fit(x,labels)
    elif mode=='reg_enable_average':r.average=1;r.fit(x,y)
    elif mode=='reg_average_nan':
        r.coef_=np.full_like(r.coef_,bad);r.intercept_=np.full_like(r.intercept_,bad);r.fit(x,y)
        assert np.isfinite(r.coef_).all()
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
    with warnings.catch_warnings():warnings.simplefilter('ignore');source_reject('reg_average_nan')
    run=subprocess.run([str(args.executable.resolve()),'reg_average_nan'],capture_output=True,text=True)
    assert run.returncode==0 and 'PASS' in run.stdout,(run.stdout,run.stderr)
    assert before==sha(args.executable)
    args.report.write_text(json.dumps(dict(passed=True,guards=guards,accepted=['reg_average_nan'],
        executable_sha256=before,validator_sha256=sha(__file__)),indent=2)+'\n')
    print(len(guards),'warm Source/native guards and averaged replacement acceptance PASS')



if __name__=='__main__':main()
