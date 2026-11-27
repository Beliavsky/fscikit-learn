"""Signed-weight validation stopping, including Source-accepted NaN losses."""
from pathlib import Path
import argparse, hashlib, itertools, json, os, subprocess, warnings
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from compare_gb_warm_start import snapshot
from sklearn_tree_runtime import tree_runtime_provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kind', choices=['regression','classification'], required=True)
    ap.add_argument('--executable', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    a = ap.parse_args()
    regression = a.kind == 'regression'
    records, payload = [], []
    rng = np.random.RandomState(19)
    x = rng.normal(size=(30,3))
    y = rng.normal(size=30) if regression else np.arange(30) % 3
    for seed, pattern, sub in itertools.product(range(8), range(3), [.65,1.]):
        w = np.resize([1.,-1.],30) if pattern == 0 else np.resize([1.,-.9,.01],30) if pattern == 1 else -np.ones(30)
        cls = GradientBoostingRegressor if regression else GradientBoostingClassifier
        params = dict(n_estimators=6, n_iter_no_change=2, validation_fraction=.3,
            random_state=seed, subsample=sub, init='zero')
        model = cls(**params)
        error, score_error, expected, prior = None, None, None, None
        delta = np.zeros(6)
        def monitor(i, est, local):
            delta[i] = 0.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(x,y,sample_weight=w,monitor=monitor if regression else None)
                if not regression:
                    train,_ = train_test_split(np.arange(30), test_size=.3, random_state=seed, stratify=y)
                    prior = DummyClassifier(strategy='prior').fit(x[train],y[train],sample_weight=w[train]).class_prior_
                expected, score_error = snapshot(model,x,y,w,a.kind,prior,delta[:len(model.estimators_)])
        except (ValueError,ZeroDivisionError) as e:
            error = dict(type=type(e).__name__,message=str(e))
        c = dict(seed=seed,pattern=pattern,subsample=sub,score_ok=int(score_error is None))
        loss = 'squared_error' if regression else 'log_loss'
        fields = [30,3,loss,'friedman_mse',seed,6,3,'none',1,2]
        fields += [0,.1,.9,0.,0.,0.,1,1] if regression else [.1,0.,1,1,0,0.,0.]
        fields += [sub,0,1,c['score_ok'],2,.3,1.e-4]
        text = ' '.join(map(str,fields))+'\n'+'\n'.join(' '.join(map(str,row)) for row in x)+'\n'
        text += ' '.join(map(str,y))+'\n'+' '.join(map(str,w))+'\n'
        record = dict(case=c,source=expected,source_error=error,score_error=score_error,input=text,differences=[])
        records.append(record)
        if error is None:
            payload.append(text)
    text = ''.join(payload)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.with_suffix('.in').write_text(text)
    p = subprocess.run([str(a.executable)],input=text,text=True,capture_output=True,timeout=180)
    a.report.with_suffix('.stdout').write_text(p.stdout)
    a.report.with_suffix('.stderr').write_text(p.stderr)
    lines=p.stdout.splitlines()
    valid=[v for v in records if v['source'] is not None]
    width=23 if regression else 26
    integer={'metadata','classes','prediction','leaves','tree_metadata','count','left','right','feature'}
    for i,record in enumerate(valid):
        chunk=lines[i*width:(i+1)*width];record['native']={}
        if len(chunk)!=width:
            record['differences'].append('missing native output');continue
        for (key,want),line in zip(record['source'].items(),chunk):
            got=np.fromstring(line,sep=' ');want=np.asarray(want);record['native'][key]=got.tolist()
            ok=got.shape==want.shape and (np.array_equal(got,want) if key in integer else
                np.allclose(got,want,rtol=2e-10,atol=2e-12,equal_nan=True))
            if not ok:record['differences'].append(key)
    for record in records:
        if record['source_error'] is None:continue
        rejected=subprocess.run([str(a.executable)],input=record['input'],text=True,capture_output=True,timeout=30)
        output=rejected.stdout+rejected.stderr;record['native_error']=dict(returncode=rejected.returncode,output=output)
        if not(rejected.returncode!=0 and 'fscikit-learn:' in output and not any(v in output for v in [
                'SIGFPE','SIGSEGV','Fortran runtime error'])):record['differences'].append('expected controlled Source rejection')
    driver='gbr' if regression else 'gbc'
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=[Path(__file__),Path('validation/compare_gb_warm_start.py'),a.executable,
        Path(f'app/reference_{driver}_early_warm_driver.f90'),Path('src/sklearn_classic_gradient_boosting.f90')]
    report=dict(kind=a.kind,configurations=len(records),successful_fits=len(valid),source_rejections=len(records)-len(valid),
        nonfinite_training_losses=sum(not np.all(np.isfinite(v['source']['loss'])) for v in valid),records=records,
        failed_configurations=sum(bool(v['differences']) for v in records),native_returncode=p.returncode,
        input_sha256=hashlib.sha256(text.encode()).hexdigest(),sklearn=sklearn.__version__,runtime=tree_runtime_provenance(),
        sha256={str(p):sha(p) for p in files})
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k not in ['records','runtime','sha256']})
    assert p.returncode==0 and len(lines)==len(valid)*width and report['failed_configurations']==0


if __name__=='__main__':
    main()
