"""Histogram bins and numeric Newton trees vs sklearn's internal implementation."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import warnings
from collections import Counter
import numpy as np
import sklearn
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower

ROOT=Path(__file__).resolve().parents[1]
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'

def fixtures():
    for seed in [0,7,42,2026,314159]:
        rng=np.random.RandomState(seed)
        for case in range(48):
            n=[1,7,47,101,300,255][case%6];p=1+case%5
            x=rng.normal(size=(n,p));q=rng.normal(size=(13,p))
            if case%4==0:x=np.round(x);q=np.round(q)
            if case%3==0:x[rng.uniform(size=x.shape)<.2]=np.nan;q[:2]=np.nan
            if case%11==0:x[:,0]=np.nan
            if case%13==0:x[:]=1;q[0]=np.nan
            if case%17==0:x[0]=-np.inf;x[-1]=np.inf;q[0]=-np.inf;q[-1]=np.inf
            g=rng.normal(size=n).astype('float32');h=(.1+rng.uniform(size=n)).astype('float32')
            if case%2==0:h[:]=1
            monotonic=rng.randint(-1,2,size=p) if case%7==0 else np.zeros(p,int)
            yield dict(seed=seed,case=case,max_bins=[2,3,7,31,255][case%5],
                       min_leaf=1+case%5,max_leaves=[2,3,7,31][case%4],max_depth=[None,1,3][case%3],
                       l2=0. if case%2 else .3,rate=.1),x,q,g,h,monotonic
    rng=np.random.RandomState(2718);n=200017
    x=rng.normal(size=(n,1));q=np.array([[-np.inf],[-1.],[0.],[1.],[np.inf],[np.nan]])
    g=rng.normal(size=n).astype('float32');h=np.ones(n,'float32')
    for seed in [2147483648,4294967295]:
        yield dict(seed=seed,case=5000,max_bins=31,min_leaf=20,max_leaves=3,max_depth=None,
                   l2=.3,rate=.1),x,q,g,h,np.zeros(1,int)

def resolve(profile):
    flags=['--profile','release'] if profile=='release' else ['--flag',FLAGS]
    r=subprocess.run(['fpm','run',*flags,'--target','reference_histogram_tree_driver','--runner'],
                     cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    paths=[ROOT/s.strip().strip('"') for s in r.stdout.splitlines()
           if s.strip().strip('"').endswith(('reference_histogram_tree_driver.exe','reference_histogram_tree_driver'))]
    paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,r.stdout
    return paths[0]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path)
    ap.add_argument('--profile',choices=['checked','release'],default='checked');args=ap.parse_args()
    exe=args.executable.resolve() if args.executable else resolve(args.profile)
    checks=[];caught_warnings=Counter();cases=0;errors={}
    for c,x,q,g,h,monotonic in fixtures():
        with warnings.catch_warnings(record=True) as caught:
            mapper=_BinMapper(n_bins=c['max_bins']+1,random_state=c['seed'],n_threads=1)
            bins=mapper.fit_transform(x)
            grower=TreeGrower(bins,g,h,n_bins=c['max_bins']+1,n_bins_non_missing=mapper.n_bins_non_missing_,
                             has_missing_values=np.any(bins==mapper.missing_values_bin_idx_,axis=0),
                             max_leaf_nodes=c['max_leaves'],max_depth=c['max_depth'],min_samples_leaf=c['min_leaf'],
                             l2_regularization=c['l2'],shrinkage=c['rate'],monotonic_cst=monotonic,n_threads=1)
            grower.grow();predictor=grower.make_predictor(mapper.bin_thresholds_)
            bitsets,feature_map=mapper.make_known_categories_bitsets()
            expected_prediction=predictor.predict(q,bitsets,feature_map,n_threads=1)
        caught_warnings.update(str(w.message) for w in caught)
        def line(a):return ' '.join(str(v) for v in a)
        n,p=x.shape
        lines=[f"{n} {p} {len(q)} {c['max_bins']} {c['min_leaf']} {c['max_leaves']} {c['max_depth'] or 0} "
               f"{c['l2']} {c['rate']} {c['seed']}"]
        lines.extend(line(row) for a in [x,q] for row in a)
        lines.extend([line(map(float,g)),line(map(float,h)),line(monotonic)])
        run=subprocess.run([str(exe)],input='\n'.join(lines)+'\n',cwd=ROOT,text=True,
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert run.returncode==0,(c,run.stdout)
        out=run.stdout.splitlines();count=int(out[0]);assert count==len(predictor.nodes),(c,count,len(predictor.nodes))
        np.testing.assert_array_equal(np.fromstring(out[1],sep=' ',dtype=int),mapper.n_bins_non_missing_,err_msg=str(c))
        np.testing.assert_array_equal(np.fromstring(out[2],sep=' ',dtype=int).reshape(x.shape,order='F')-1,bins,err_msg=str(c))
        for j in range(p):np.testing.assert_array_equal(np.fromstring(out[3+j],sep=' '),mapper.bin_thresholds_[j],err_msg=str(c))
        actual_prediction=np.fromstring(out[3+p],sep=' ')
        np.testing.assert_allclose(actual_prediction,expected_prediction,rtol=3e-11,atol=3e-11,err_msg=str(c))
        errors['prediction']=max(errors.get('prediction',0.),float(np.max(abs(actual_prediction-expected_prediction))))
        for j,want in enumerate(predictor.nodes):
            integers=np.fromstring(out[4+p+2*j],sep=' ',dtype=int)
            values=np.fromstring(out[5+p+2*j],sep=' ')
            leaf=bool(want['is_leaf'])
            expected=np.array([0 if leaf else want['feature_idx']+1,0 if leaf else int(want['bin_threshold'])+1,
                               0 if leaf else want['left']+1,0 if leaf else want['right']+1,want['count'],want['depth'],
                               want['missing_go_to_left']])
            np.testing.assert_array_equal(integers,expected,err_msg=f'{c} node {j}')
            np.testing.assert_allclose(values,[want['value'],want['gain'],want['num_threshold']],
                                       rtol=3e-11,atol=3e-11,err_msg=f'{c} node {j}')
        for name in ['bin_counts','binned_training','thresholds','prediction','node_structure','node_values']:
            checks.append(dict(**c,name=name,passed=True))
        cases+=1
        if cases%48==0:print(f'{cases} histogram tree fixtures passed',flush=True)
    root=Path(sklearn.__file__).parent
    source=['ensemble/_hist_gradient_boosting/'+p for p in ['binning.py','grower.py','splitting.pyx','histogram.pyx']]
    report=dict(sklearn=sklearn.__version__,numpy=np.__version__,profile=args.profile,fixtures=cases,
                checks=checks,errors=errors,source_warnings=dict(caught_warnings),
                native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source})
    (ROOT/'validation/results'/f'histogram_tree_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {cases} histogram tree fixtures passed')

if __name__=='__main__':main()
