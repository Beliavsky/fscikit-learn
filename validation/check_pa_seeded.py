"""Compare actual Source/native PA rejection decisions and adapter guards."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier,PassiveAggressiveRegressor


def source_reject(mode):
    x=np.array([[0.,0.],[-1.,2.],[2.,1.],[.5,-.5],[1.,1.],[-2.,1.]])
    labels=np.array([-4,7,21,-4,7,21]);y=2*x[:,0]-.5*x[:,1]+.3
    classifier=mode.startswith('class') or mode.startswith('weight')
    options={}
    if mode.endswith('seed'):options.update(random_state=-1,shuffle=False)
    if mode.endswith('_c'):options['C']=0
    if mode=='epsilon':options['epsilon']=-1
    if mode.endswith('loss'):options['loss']='log_loss' if classifier else 'squared_error'
    if mode.startswith('weight'):options['class_weight']={-4:np.nan if mode=='weight_nan' else np.inf,7:2.,21:1.3}
    (PassiveAggressiveClassifier if classifier else PassiveAggressiveRegressor)(**options).fit(x,labels if classifier else y)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();before=sha(args.executable);guards=[]
    for mode in ['reg_seed','class_seed','reg_c','class_c','epsilon','reg_loss','class_loss','weight_nan','weight_inf','weight_count','weight_conflict']:
        if mode in ['weight_count','weight_conflict']:
            oracle='native_typed_adapter';source='class weights must have one entry per sorted class and select one weighting mode'
        else:
            oracle='source'
            try:
                with warnings.catch_warnings():warnings.simplefilter('ignore');source_reject(mode)
            except ValueError as error:source=str(error)
            else:raise AssertionError(('Source accepted',mode))
        run=subprocess.run([str(args.executable.resolve()),mode],capture_output=True,text=True);message=run.stdout+run.stderr
        assert run.returncode and 'fscikit-learn:' in message,(mode,message)
        assert 'SIGFPE' not in message and 'Fortran runtime error' not in message,(mode,message)
        guards.append(dict(mode=mode,oracle=oracle,source=source,native=message,passed=True))
    assert before==sha(args.executable)
    args.report.write_text(json.dumps(dict(passed=True,guards=guards,executable_sha256=before,validator_sha256=sha(__file__)),indent=2)+'\n')
    print(len(guards),'PA guards PASS (nine Source decisions, two typed-adapter checks)')


if __name__=='__main__':main()
