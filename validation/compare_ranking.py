"""Exact binary curve structure and weighted scores against installed sklearn."""
from pathlib import Path
import argparse,json,subprocess,warnings
import numpy as np
import sklearn
from sklearn.metrics import (roc_curve,precision_recall_curve,det_curve,auc,roc_auc_score,
                             average_precision_score,brier_score_loss)
from sklearn.calibration import calibration_curve
ROOT=Path(__file__).resolve().parents[1];RESULTS=ROOT/'validation/results'
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve():
    p=subprocess.run(['fpm','run','--flag',FLAGS,'--target','reference_ranking_driver','--runner'],cwd=ROOT,
                     text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    paths=[ROOT/line.strip().strip('"') for line in p.stdout.splitlines()
           if line.strip().strip('"').endswith(('reference_ranking_driver','reference_ranking_driver.exe'))]
    paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,p.stdout
    return paths[0]


def fixtures(seed):
    rng=np.random.default_rng(seed)
    for n in [1,2,3,4,7,16,31,64,127,513]:
        for kind in range(6):
            y=rng.integers(0,2,n);s=rng.normal(size=n);prob=rng.random(n);w=np.ones(n)
            if kind==0:s=np.round(s,1);prob=np.round(prob,1)
            if kind==1:s[:]=.25;prob[:]=.5
            if kind==2:w=rng.uniform(.05,3,n);w[::4]=0
            if kind==3:w=10.**rng.uniform(-8,8,n)
            if kind==4:y[:]=0
            if kind==5:y[:]=1
            if not np.any(w):w[0]=1
            pos=1;explicit=kind%2==0
            if kind==3:y=np.where(y==1,17,-9);pos=17;explicit=True
            if kind==2:y=np.where(y==1,1,-1)
            yield y,s,prob,w,dict(pos=pos,explicit=explicit,weighted=kind in [2,3],drop_roc=kind%2==0,
                drop_pr=kind%3==0,det=len(np.unique(y))==2,scores=True,prob=True,
                bins=[1,2,3,5,8,11][kind],strategy=kind%2,limit=[.01,.1,.3,.5,.9,1.][kind])
    # Explicit positive label may be the lower label; curves can also binarize multiclass targets.
    for n in [7,19,47]:
        y=rng.integers(0,3,n);y[:3]=[0,1,2];s=np.round(rng.normal(size=n),1)
        yield y,s,rng.random(n),np.ones(n),dict(pos=0,explicit=True,weighted=False,drop_roc=True,
            drop_pr=True,det=False,scores=False,prob=False,bins=5,strategy=0,limit=.5)
        y=np.where(y==0,-7,5)
        yield y,s,rng.random(n),rng.uniform(.2,3,n),dict(pos=-7,explicit=True,weighted=True,
            drop_roc=False,drop_pr=False,det=True,scores=True,prob=True,bins=4,strategy=1,limit=.7)
    # Zero weights remove a class; signed weights are accepted by sklearn's binary curve primitive.
    for y,s,w in [(np.array([0,0,1,1]),np.array([.1,.4,.35,.8]),np.array([1.,2,0,0])),
                  (np.array([0,0,1,1]),np.array([.1,.4,.35,.8]),np.array([0.,0,1,2])),
                  (np.array([0,1,0,1]),np.array([.1,.2,.3,.4]),np.array([-1.,2,3,4])),
                  (np.array([0,1,0,1]),np.array([.1,.2,.3,.4]),np.array([1.,2,3,-1])),
                  (np.array([0,1,0,1]),np.array([.1,.2,.3,.4]),np.array([1.,-1,1,1])),
                  (np.array([0,1,0,1]),np.array([.1,.2,.3,.4]),np.array([-2.,1,-2,1]))]:
        yield y,s,np.array([0.,.25,.75,1.]),w,dict(pos=1,explicit=False,weighted=True,drop_roc=False,
            drop_pr=True,det=True,scores=False,prob=True,bins=4,strategy=0,limit=.5)
    # Single arbitrary class: calibration needs an explicit label, while scorer defaults differ.
    yield np.full(5,7),np.array([0.,.5,.5,.7,1.]),np.array([0.,.25,.5,.75,1.]),np.ones(5),dict(
        pos=7,explicit=True,weighted=False,drop_roc=True,drop_pr=True,det=False,scores=True,prob=True,
        bins=4,strategy=1,limit=.5)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path);args=parser.parse_args()
    executable=args.executable.resolve() if args.executable else resolve();RESULTS.mkdir(exist_ok=True);checks=[];rejections=[]
    for seed in [0,7,42,2026,314159]:
        rng=np.random.default_rng(seed+101)
        for case,(y,s,prob,w,cfg) in enumerate(fixtures(seed)):
            n=len(y);ax=np.sort(rng.random(9));ay=rng.normal(size=9)
            if case%3==0:ax=ax[::-1].copy();ay=ay[::-1].copy()
            if case%5==0:ax[:]=1
            data=RESULTS/'ranking_input.txt';out=RESULTS/'ranking_actual.txt'
            positive=cfg['pos'] if cfg['explicit'] else y.max()
            do_partial=(not cfg['weighted'] or cfg['limit']==1 or len(np.unique(y))==1 or np.sum(w[y!=positive])>0)
            flags=[cfg[k] for k in ['explicit','weighted','drop_roc','drop_pr','det','scores','prob']]+[do_partial]
            data.write_text(f"{n} 9 {cfg['pos']} {cfg['bins']} {cfg['strategy']} {cfg['limit']}\n"+
                ' '.join(str(b)[0] for b in flags)+'\n'+'\n'.join(' '.join(map(str,a)) for a in [y,s,prob,w,ax,ay])+'\n')
            proc=subprocess.run([str(executable),str(data),str(out)],cwd=ROOT,text=True,
                                stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            assert proc.returncode==0,(seed,case,cfg,proc.stdout)
            actual={}
            for line in out.read_text().splitlines():
                name,*values=line.split();actual[name]=np.array(values,float)
            pos=cfg['pos'] if cfg['explicit'] else None;sw=w if cfg['weighted'] else None
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fpr,tpr,thresholds=roc_curve(y,s,pos_label=pos,sample_weight=sw,drop_intermediate=cfg['drop_roc'])
                p,r,t=precision_recall_curve(y,s,pos_label=pos,sample_weight=sw,drop_intermediate=cfg['drop_pr'])
                expected=dict(roc_fpr=fpr,roc_tpr=tpr,roc_thresholds=thresholds,
                    pr_precision=p,pr_recall=r,pr_thresholds=t)
                if cfg['det']:
                    f,nr,t=det_curve(y,s,pos_label=pos,sample_weight=sw)
                    expected.update(det_fpr=f,det_fnr=nr,det_thresholds=t)
                if cfg['scores']:
                    positive=cfg['pos'] if cfg['explicit'] else y.max()
                    binary=y==positive
                    expected.update(roc_auc=[roc_auc_score(binary,s,sample_weight=sw)])
                    if do_partial:expected['partial_auc']=[roc_auc_score(binary,s,sample_weight=sw,max_fpr=cfg['limit'])]
                    else:
                        try:roc_auc_score(binary,s,sample_weight=sw,max_fpr=cfg['limit'])
                        except ValueError as error:
                            assert 'At least 2 points' in str(error)
                            rejected=subprocess.run([str(executable),str(data),str(out),'partial'],cwd=ROOT,text=True,
                                stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                            assert rejected.returncode!=0 and 'partial ROC AUC requires negative-class support' in rejected.stdout
                            rejections.append(dict(seed=seed,case=case,source_error='ValueError',native_rejected=True))
                            checks.append(dict(seed=seed,case=case,name='partial_auc_rejected',oracle='sklearn_rejection',passed=True))
                        else:raise AssertionError('source unexpectedly accepted unsupported partial AUC')
                if cfg['scores'] or cfg['prob']:
                    expected['average_precision']=[average_precision_score(y,s,pos_label=1 if pos is None else pos,sample_weight=sw)]
                if cfg['prob']:
                    a,b=calibration_curve(y,prob,n_bins=cfg['bins'],strategy='uniform' if cfg['strategy']==0 else 'quantile',pos_label=pos)
                    expected.update(calibration_true=a,calibration_pred=b,brier=[brier_score_loss(y,prob,pos_label=pos,sample_weight=sw)])
                expected.update(generic_auc=[auc(ax,ay)],preserved=np.concatenate([y,s,prob,w,ax,ay]))
            assert actual.keys()==expected.keys(),(seed,case,actual.keys(),expected.keys())
            for name,want in expected.items():
                if name=='preserved' or name.endswith('thresholds'):np.testing.assert_array_equal(actual[name],want)
                else:np.testing.assert_allclose(actual[name],want,atol=3.e-12,rtol=3.e-12,equal_nan=True,
                                               err_msg=f'{seed} {case} {name} {cfg}')
                checks.append(dict(seed=seed,case=case,name=name,oracle='input_identity' if name=='preserved' else 'sklearn',passed=True))
        print(f'seed {seed}: ranking and calibration passed',flush=True)
    (RESULTS/'ranking.json').write_text(json.dumps(dict(sklearn=sklearn.__version__,checks=checks,source_rejections=rejections),indent=2)+'\n')
    print(f'{len(checks)} ranking and calibration result groups passed')


if __name__=='__main__':main()
