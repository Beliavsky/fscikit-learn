"""CSR graph transformers vs installed sklearn; retain tie certificates explicitly."""
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
import scipy
from scipy.sparse import csr_matrix
import sklearn
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsTransformer, RadiusNeighborsTransformer

ROOT=Path(__file__).resolve().parents[1]
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'

def resolve(profile):
    flags=['--profile','release'] if profile=='release' else ['--flag',FLAGS]
    r=subprocess.run(['fpm','run',*flags,'--target','reference_neighbor_graph_driver','--runner'],
                     cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    paths=[ROOT/line.strip().strip('"') for line in r.stdout.splitlines()
           if line.strip().strip('"').endswith(('reference_neighbor_graph_driver.exe','reference_neighbor_graph_driver'))]
    paths=[p.resolve() for p in paths if p.is_file()]
    assert len(paths)==1,r.stdout
    return paths[0]

def fixtures():
    for seed in [0,7,42,2026,314159]:
        rng=np.random.RandomState(seed)
        for metric in ['euclidean','sqeuclidean','manhattan','cityblock','chebyshev','minkowski','cosine','precomputed']:
            for tied in [False,True]:
                x=rng.normal(size=(13,4));q=rng.normal(size=(6,4))
                if tied:
                    x=rng.randint(-2,3,size=x.shape).astype(float);q=x[:6].copy();x[1]=x[0];x[-1]=0;q[-1]=0
                if metric=='precomputed':
                    x=pairwise_distances(x,metric='manhattan');q=np.abs(rng.normal(size=(6,13)))
                    if tied:q[:]=0
                for kind in [0,1]:
                    for mode in ['distance','connectivity']:
                        for op in [0,1,2]:
                            yield dict(seed=seed,metric=metric,tied=tied,kind=kind,mode=mode,op=op,storage=0),x,q
        # Explicit zero edges and missing entries. Each row has enough stored
        # neighbors for k+1 plus implicit-self removal. Unsorted input is retained.
        n=9
        d=np.abs(rng.normal(size=(n,n)));np.fill_diagonal(d,0)
        z=np.abs(rng.normal(size=(5,n)));z[0,0]=0
        def stored(a,duplicates):
            ptr=[0];cols=[];values=[]
            for row in range(a.shape[0]):
                anchor=row if a.shape[0]==n else 0
                others=np.delete(np.arange(n),anchor)
                chosen=np.r_[anchor,rng.choice(others,6,replace=duplicates)]
                rng.shuffle(chosen)
                cols.extend(chosen);values.extend(a[row,chosen]);ptr.append(len(cols))
            return csr_matrix((values,cols,ptr),shape=a.shape)
        for duplicates in [False,True]:
            train=stored(d,duplicates);query=stored(z,duplicates)
            for kind in [0,1]:
                for mode in ['distance','connectivity']:
                    for op in [0,1,2]:
                        yield dict(seed=seed,metric='precomputed',tied=False,kind=kind,mode=mode,
                                   op=op,storage=1,duplicates=duplicates),train,query
        # Exact radius boundaries, all-empty held-out rows, infinite radius,
        # and counts close to the fitted-sample limit.
        if seed!=0:continue
        x=np.array([[0.],[1.],[1.],[2.],[4.],[8.]])
        q=np.array([[-1.],[.5],[9.]])
        for mode in ['distance','connectivity']:
            for op in [0,1,2]:
                for k in [1,4]:
                    yield dict(seed=seed,metric='euclidean',tied=True,kind=0,mode=mode,
                               op=op,storage=0,k=k,boundary=True),x,q
                for radius in [0.,1.,float('inf')]:
                    yield dict(seed=seed,metric='euclidean',tied=True,kind=1,mode=mode,
                               op=op,storage=0,radius=str(radius),boundary=True),x,q

def encode(c,x,q):
    n,p=x.shape;m=q.shape[0]
    radius=c.get('radius','1.5');radius='Infinity' if radius=='inf' else radius
    lines=[f"{c['kind']} {c['storage']} {c['op']} {c.get('k',3)} {radius} 3 {n} {p} {m}",f"{c['metric']} {c['mode']}"]
    def line(a):return ' '.join(str(v) for v in a)
    if c['storage']:
        for a in [x,q]:lines.extend([str(a.nnz),line(a.indptr+1),line(a.indices+1),line(a.data)])
    else:
        lines.extend(line(row) for a in [x,q] for row in a)
    return '\n'.join(lines)+'\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
    ap.add_argument('--executable',type=Path);args=ap.parse_args()
    exe=args.executable.resolve() if args.executable else resolve(args.profile)
    checks=[];certificates=[];caught_warnings=Counter();count=0
    for c,x,q in fixtures():
        original_x=x.copy();original_q=q.copy()
        cls=KNeighborsTransformer if c['kind']==0 else RadiusNeighborsTransformer
        kw={'n_neighbors':c.get('k',3)} if c['kind']==0 else {'radius':float(c.get('radius',1.5))}
        model=cls(**kw,mode=c['mode'],metric=c['metric'],p=3,algorithm='brute',n_jobs=1)
        with warnings.catch_warnings(record=True) as caught:
            if c['op']==1:want=model.fit_transform(x)
            else:
                model.fit(x);want=model.transform(q if c['op']==0 else None)
        caught_warnings.update(str(w.message) for w in caught)
        run=subprocess.run([str(exe)],input=encode(c,x,q),cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert run.returncode==0,(c,run.stdout)
        lines=run.stdout.splitlines();nr,nc,nn=map(int,lines[0].split())
        ptr=np.fromstring(lines[1],sep=' ',dtype=int)-1
        idx=np.fromstring(lines[2],sep=' ',dtype=int)-1
        data=np.fromstring(lines[3],sep=' ') if len(lines)>3 else np.array([])
        actual=csr_matrix((data,idx,ptr),shape=(nr,nc))
        assert actual.shape==want.shape and actual.nnz==nn==want.nnz,c
        np.testing.assert_array_equal(actual.indptr,want.indptr,err_msg=str(c))
        if c['metric']=='precomputed':
            distances=x if c['op'] else q
            distances=distances.toarray() if c['storage'] else distances
        else:
            target=x if c['op'] else q
            distances=pairwise_distances(target,x,metric=c['metric'],**({'p':3} if c['metric']=='minkowski' else {}))
            if c['op']:np.fill_diagonal(distances,0)
        differs=[]
        for row in range(nr):
            sl=slice(ptr[row],ptr[row+1]);ai=idx[sl];wi=want.indices[sl]
            ad=data[sl];wd=want.data[sl]
            if c['storage']:
                # Sparse precomputed rows are stably sorted by both libraries.
                # Compare stored slots directly, preserving duplicate columns
                # rather than using toarray(), which would sum their distances.
                np.testing.assert_array_equal(ai,wi,err_msg=str(c))
                np.testing.assert_allclose(ad,wd,rtol=0,atol=0,err_msg=str(c))
                original=x if c['op'] else q
                available=original.indices[original.indptr[row]:original.indptr[row+1]]
                assert not Counter(ai)-Counter(available),c
                if c['op']==2:assert row not in ai,c
                continue
            if c['mode']=='distance':
                # Check edge distances independently, then check the selected
                # distance multiset against sklearn (including cut-off ties).
                np.testing.assert_allclose(ad,distances[row,ai],rtol=2e-12,atol=2e-12,err_msg=str(c))
            else:np.testing.assert_array_equal(ad,np.ones(len(ai)),err_msg=str(c))
            if c['kind']==1 and c['mode']=='connectivity':
                np.testing.assert_array_equal(ai,wi,err_msg=str(c))
            np.testing.assert_allclose(distances[row,ai],distances[row,wi],
                                       rtol=2e-12,atol=2e-12,err_msg=str(c))
            assert len(set(ai))==len(ai),(c,'unexpected duplicate edge')
            if c['op']==2:assert row not in ai,c
            if not np.array_equal(ai,wi):
                # Alternative memberships are permitted ONLY at exact equal
                # input distances. Numerical comparison tolerance cannot admit
                # a different neighbor support.
                missing=Counter(wi)-Counter(ai);extra=Counter(ai)-Counter(wi)
                md=sorted(distances[row,j] for j,v in missing.items() for _ in range(v))
                ed=sorted(distances[row,j] for j,v in extra.items() for _ in range(v))
                assert md==ed,(c,row,md,ed)
                # Equal support with a different permutation is also allowed
                # only inside an exact distance tie, never across ranks.
                np.testing.assert_array_equal(distances[row,ai],distances[row,wi],err_msg=str(c))
                differs.append(dict(row=row,source=wi.tolist(),native=ai.tolist(),tie_distances=md))
            if c['mode']=='distance':np.testing.assert_allclose(ad,wd,rtol=2e-12,atol=2e-12)
        checks.extend([dict(**c,name=name,passed=True) for name in ['shape_nnz','row_pointers','edge_values','neighbor_distances']])
        for before,after in [(original_x,x),(original_q,q)]:
            if c['storage']:
                for key in ['indptr','indices','data']:np.testing.assert_array_equal(getattr(before,key),getattr(after,key))
            else:np.testing.assert_array_equal(before,after)
        checks.append(dict(**c,name='input_identity',passed=True))
        if differs:certificates.append(dict(**c,rows=differs))
        count+=1
        if count%200==0:print(f'{count} graph fixtures passed',flush=True)
    root=Path(sklearn.__file__).parent
    report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,
                fixtures=count,checks=checks,tie_certificates=certificates,source_warnings=dict(caught_warnings),
                native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                source_sha256={str(p):hashlib.sha256((root/p).read_bytes()).hexdigest()
                               for p in ['neighbors/_graph.py','neighbors/_base.py','metrics/pairwise.py']})
    dest=ROOT/'validation/results'/f'neighbor_graph_{args.profile}.json'
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(checks)} groups / {count} fixtures passed; {len(certificates)} fixtures retain exact-distance tie certificates')

if __name__=='__main__':main()
