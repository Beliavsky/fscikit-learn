"""LogisticRegressionCV as a base in the existing native classifier wrappers."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn import config_context
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier,BaggingClassifier,StackingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'validation/results'
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile):
    options=['--profile',profile] if profile=='release' else ['--flag',FLAGS]
    p=subprocess.run(['fpm','run',*options,'--target','reference_logistic_cv_composites_driver','--runner'],cwd=ROOT,
                     text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    paths=[ROOT/s.strip().strip('"') for s in p.stdout.splitlines()
           if s.strip().strip('"').endswith(('reference_logistic_cv_composites_driver','reference_logistic_cv_composites_driver.exe'))]
    paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,p.stdout
    return paths[0]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path)
    parser.add_argument('--profile',choices=['checked','release'],default='checked');args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True);exe=args.executable.resolve() if args.executable else resolve(args.profile)
    checks=[];source_warnings=[]
    def base():return LogisticRegressionCV(Cs=[.1,1.],cv=2,tol=1e-12,max_iter=2000,solver='newton-cg')
    for seed in [0,7,42,2026,314159]:
        rng=np.random.RandomState(seed);n=96;p=3;m=13
        x=rng.normal(size=(n,p));query=rng.normal(size=(m,p));w=rng.uniform(.3,2,n)
        y=np.where(x[:,0]+.4*x[:,1]+rng.normal(size=n)*1.5>0,7,-3)
        y[:18]=np.resize([-3,7],18)
        many=np.array([-3,7,19])[np.argmax(x@rng.normal(size=(p,3))+rng.normal(size=(n,3))*1.5,axis=1)]
        many[:18]=np.resize([-3,7,19],18);targets=np.column_stack([y,many]);unlabeled=y.copy();unlabeled[::4]=-1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');expected={'cross_val_score':cross_val_score(base(),x,y,cv=3)}
            def save(name,model):
                expected[name+'_proba']=model.predict_proba(query)
                expected[name+'_predict']=model.predict(query)
            vote=VotingClassifier([('lrcv',base()),('nb',GaussianNB())],voting='soft').fit(x,y,sample_weight=w);save('voting',vote)
            bag=BaggingClassifier(base(),n_estimators=2,random_state=7).fit(x,y,sample_weight=w);save('bagging',bag)
            stack=StackingClassifier([('lrcv',base()),('nb',GaussianNB())],final_estimator=base(),cv=3).fit(x,y,sample_weight=w)
            save('stacking',stack)
            with config_context(enable_metadata_routing=True):
                scoring=get_scorer('accuracy').set_score_request(sample_weight=True)
                estimator=base().set_params(scoring=scoring).set_fit_request(sample_weight=True)
                ovr=OneVsRestClassifier(estimator).fit(x,many,sample_weight=w)
            save('ovr',ovr)
            multi=MultiOutputClassifier(base()).fit(x,targets,sample_weight=w)
            expected['multi_predict']=multi.predict(query)
            expected['multi_proba1'],expected['multi_proba2']=multi.predict_proba(query)
            calibrated=CalibratedClassifierCV(base(),cv=2,method='sigmoid',ensemble=True).fit(x,y,sample_weight=w)
            save('calibrated',calibrated)
            semi=SelfTrainingClassifier(base(),criterion='k_best',k_best=4,max_iter=1).fit(x,unlabeled)
            save('self',semi);expected['transduction']=semi.transduction_;expected['labeled_iter']=semi.labeled_iter_
        source_warnings.extend(dict(seed=seed,category=q.category.__name__,message=str(q.message)) for q in caught)
        lines=[f'{n} {p} {m}']+[' '.join(map(str,row)) for row in x]
        lines+=[' '.join(map(str,y)),' '.join(map(str,many)),' '.join(map(str,w))]
        lines+=[' '.join(map(str,row)) for row in query]
        inp=RESULTS/f'logistic_cv_composites_{args.profile}_input.txt'
        out=RESULTS/f'logistic_cv_composites_{args.profile}_output.txt'
        inp.write_text('\n'.join(lines)+'\n')
        proc=subprocess.run([str(exe),str(inp),str(out)],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert proc.returncode==0,(seed,proc.stdout)
        actual={s.split()[0]:np.asarray(s.split()[1:],float) for s in out.read_text().splitlines()}
        assert actual.keys()==expected.keys(),(actual.keys(),expected.keys())
        for name,want in expected.items():
            want=np.asarray(want).ravel(order='F')
            if name.endswith('_predict') or name in ['transduction','labeled_iter']:
                np.testing.assert_array_equal(actual[name],want,err_msg=f'{seed} {name}')
            else:np.testing.assert_allclose(actual[name],want,rtol=3e-7,atol=3e-8,err_msg=f'{seed} {name}')
            checks.append(dict(seed=seed,name=name,passed=True,oracle='sklearn'))
        print(f'seed {seed}: logistic-CV composite dispatch passed',flush=True)
    classes=[LogisticRegressionCV,VotingClassifier,BaggingClassifier,StackingClassifier,OneVsRestClassifier,
             MultiOutputClassifier,CalibratedClassifierCV,SelfTrainingClassifier]
    paths=sorted(set(Path(inspect.getfile(c)) for c in classes));root=Path(inspect.getfile(LogisticRegressionCV)).parent.parent
    report=dict(sklearn=sklearn.__version__,profile=args.profile,fixtures=5,checks=checks,source_warnings=source_warnings,
                source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (RESULTS/f'logistic_cv_composites_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} logistic-CV composite groups passed')


if __name__=='__main__':main()
