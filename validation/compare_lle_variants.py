"""Source local alignments, fitted subspaces and barycentric query transforms."""
from pathlib import Path
import argparse,json,os,subprocess
from unittest.mock import patch
os.environ.setdefault('LOKY_MAX_CPU_COUNT','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import sklearn
from scipy.linalg import eigh,null_space as orthogonal_complement
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import _locally_linear as source
from sklearn.neighbors import NearestNeighbors
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'validation/results'
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve():
    proc=subprocess.run(['fpm','run','--flag',FLAGS,'--target','reference_lle_variants_driver','--runner'],
        cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    paths=[ROOT/line.strip().strip('"') for line in proc.stdout.splitlines()
        if line.strip().strip('"').endswith(('reference_lle_variants_driver','reference_lle_variants_driver.exe'))]
    paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,proc.stdout
    return paths[0]


def fixtures(seed):
    rng=np.random.default_rng(seed)
    for n,p,k,d in [(18,3,8,1),(24,4,10,2),(30,5,12,2),(26,12,9,2),
                    (28,10,6,1),(25,3,9,2),(31,5,12,2),(32,3,10,3)]:
        x=rng.normal(size=(n,p))*np.linspace(.7,1.6,p);query=rng.normal(size=(5,p))
        for method in ['hessian','modified']:
            if method=='hessian' and k<=d*(d+3)//2:continue
            yield x,query,dict(k=k,d=d,method=method,hessian_tol=1.e-4,modified_tol=1.e-12,reg=.001)
    # Non-default thresholds and query regularization; fit's modified regularizer is fixed in sklearn.
    x=rng.normal(size=(24,4));query=rng.normal(size=(5,4))
    for method,tol in [('hessian',1.e-2),('modified',0.),('modified',1.e-8)]:
        yield x,query,dict(k=11,d=2,method=method,hessian_tol=tol if method=='hessian' else 1.e-4,
                          modified_tol=tol if method=='modified' else 1.e-12,reg=.05)


def source_alignment(x,cfg):
    captured={};bases=[];spectra=[];original_svd=source.svd;original_eigh=source.eigh
    def collect(matrix,*args,**kwargs):
        captured['matrix']=np.asarray(matrix);return np.zeros((len(x),cfg['d'])),0.
    def svd_record(*args,**kwargs):
        result=original_svd(*args,**kwargs);bases.append(result[0]);spectra.append(result[1]**2);return result
    def eigh_record(*args,**kwargs):
        result=original_eigh(*args,**kwargs);bases.append(result[1][:,::-1]);spectra.append(result[0][::-1]);return result
    with patch.object(source,'null_space',collect),patch.object(source,'svd',svd_record),patch.object(source,'eigh',eigh_record):
        source._locally_linear_embedding(x,n_neighbors=cfg['k'],n_components=cfg['d'],method=cfg['method'],
            hessian_tol=cfg['hessian_tol'],modified_tol=cfg['modified_tol'],reg=cfg['reg'],eigen_solver='dense')
    return captured['matrix'],np.array(bases),np.array(spectra)


def nullity_parameters(values,d,k):
    nev=values.shape[1];rho=values[:,d:].sum(axis=1)/values[:,:d].sum(axis=1);eta=np.median(rho)
    cumulative=np.cumsum(values,axis=1,dtype=np.float64)
    ratios=(cumulative[:,-1:]/cumulative[:,:-1]-1)[:,::-1]
    sizes=np.array([np.searchsorted(row,eta) for row in ratios])+k-nev
    return sizes,eta,ratios


def assemble_modified(v,values,sizes,tolerance,neighbors):
    n,k,_=v.shape;nev=values.shape[1];matrix=np.zeros((n,n));regularized=np.empty((n,k))
    reg=.001*values.sum(axis=1)
    for i in range(n):
        projection=v[i].T@np.ones(k);projection[:nev]/=values[i]+reg[i];projection[nev:]/=reg[i]
        regularized[i]=v[i]@projection;regularized[i]/=regularized[i].sum()
        s=sizes[i];vi=v[i,:,k-s:];alpha=np.linalg.norm(vi.sum(axis=0))/np.sqrt(s)
        h=np.full(s,alpha)-vi.T@np.ones(k);length=np.linalg.norm(h)
        if length<tolerance:h*=0
        else:h/=length
        w=vi-2*np.outer(vi@h,h)+(1-alpha)*regularized[i,:,None]
        idx=neighbors[i];matrix[np.ix_(idx,idx)]+=w@w.T
        sums=w.sum(axis=1);matrix[i,idx]-=sums;matrix[idx,i]-=sums;matrix[i,i]+=s
    return matrix,regularized


def cutoff_certificate(native_values,source_values,native_sizes,source_sizes,native_eta,source_eta,d,k):
    _,_,rn=nullity_parameters(native_values,d,k);_,_,rs=nullity_parameters(source_values,d,k)
    assert np.allclose(native_values,source_values,atol=3.e-10,rtol=3.e-10)
    budget=float(np.max(abs(rn-rs))+abs(native_eta-source_eta)+
        128*np.finfo(float).eps*max(1,abs(native_eta),abs(source_eta),np.max(abs(rn)),np.max(abs(rs))))
    changes=[];base=k-native_values.shape[1]
    for i in np.flatnonzero(native_sizes!=source_sizes):
        for j in range(min(native_sizes[i],source_sizes[i])-base,max(native_sizes[i],source_sizes[i])-base):
            assert max(abs(rn[i,j]-native_eta),abs(rs[i,j]-source_eta))<=budget,(i,j,rn[i,j],rs[i,j],native_eta,source_eta,budget)
        changes.append(dict(row=int(i),native_nullity=int(native_sizes[i]),source_nullity=int(source_sizes[i])))
    assert changes
    return dict(changes=changes,bound=budget,native_eta=float(native_eta),source_eta=float(source_eta))


def spectral_certificate(matrix,z,query,x,cfg,actual_error):
    n,d=z.shape;scale=max(1,np.linalg.norm(matrix,2));values=eigh(matrix,eigvals_only=True)
    margin=3.e-8*scale
    if values[1]-values[0]>margin and values[d+1]-values[d]>margin:return None
    np.testing.assert_allclose(z.T@z,np.eye(d),atol=3.e-9,rtol=3.e-9)
    np.testing.assert_allclose(z.sum(axis=0),0,atol=3.e-8,rtol=0)
    complement=orthogonal_complement(np.ones((1,n)))
    expected=eigh(complement.T@matrix@complement,eigvals_only=True)[:d]
    energy=np.trace(z.T@matrix@z);residual=matrix@z-z@(z.T@matrix@z)
    np.testing.assert_allclose(energy,expected.sum(),atol=3.e-8*scale,rtol=3.e-8)
    assert np.max(abs(residual))<3.e-8*scale
    np.testing.assert_allclose(actual_error,max(0,expected.sum()),atol=3.e-8*scale,rtol=3.e-8)
    indices=NearestNeighbors(n_neighbors=cfg['k'],algorithm='brute').fit(x).kneighbors(query,return_distance=False)
    weights=source.barycenter_weights(query,x,indices,reg=cfg['reg']);projected=np.einsum("ij,ijk->ik",weights,z[indices])
    return dict(low_spectrum=values[:d+2].tolist(),scale=float(scale),energy=float(energy),
                lowest_restricted_energy=float(expected.sum()),max_residual=float(np.max(abs(residual)))),projected


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path);args=parser.parse_args()
    executable=args.executable.resolve() if args.executable else resolve();RESULTS.mkdir(exist_ok=True);checks=[];spectral=[];cutoffs=[]
    for seed in [0,7,42,2026,314159]:
        for case,(x,query,cfg) in enumerate(fixtures(seed)):
            n,p=x.shape;k=cfg['k'];d=cfg['d'];method=1 if cfg['method']=='hessian' else 2
            data=RESULTS/'lle_variants_input.txt';out=RESULTS/'lle_variants_actual.txt'
            data.write_text(f"{n} {p} {k} {d} {method} {cfg['hessian_tol']} {cfg['modified_tol']} {cfg['reg']}\n"+
                '\n'.join(' '.join(map(str,row)) for row in np.vstack([x,query]))+'\n')
            proc=subprocess.run([str(executable),str(data),str(out)],cwd=ROOT,text=True,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            assert proc.returncode==0,(seed,case,cfg,proc.stdout)
            actual={}
            for line in out.read_text().splitlines():
                name,*values=line.split();actual[name]=np.array(values,float)
            neighbors=NearestNeighbors(n_neighbors=k+1,algorithm='brute').fit(x).kneighbors(x,return_distance=False)[:,1:]
            matrix,source_v,source_values=source_alignment(x,cfg)
            model=LocallyLinearEmbedding(n_neighbors=k,n_components=d,method=cfg['method'],
                hessian_tol=cfg['hessian_tol'],modified_tol=cfg['modified_tol'],reg=cfg['reg'],
                eigen_solver='dense',neighbors_algorithm='brute').fit(x)
            z=actual['embedding'].reshape(n,d);q=actual['query'].reshape(5,d)
            expected=dict(neighbors=(neighbors+1).ravel(),alignment=matrix.ravel(),
                gram=(model.embedding_@model.embedding_.T).ravel(),
                query_gram=(model.transform(query)@model.transform(query).T).ravel(),
                cross_gram=(model.transform(query)@model.embedding_.T).ravel(),
                error=[model.reconstruction_error_],preserved=np.vstack([x,query]).ravel())
            actual['gram']=(z@z.T).ravel();actual['query_gram']=(q@q.T).ravel();actual['cross_gram']=(q@z.T).ravel()
            cutoff=None;native_matrix=actual['alignment'].reshape(n,n)
            if method==2:
                native_values=actual['spectrum'].reshape(n,min(k,p))
                native_v=np.array([actual[f'basis_{i+1}'].reshape(k,k) for i in range(n)])
                native_sizes=actual['nullity'].astype(int)
                source_sizes,source_eta,_=nullity_parameters(source_values,d,k)
                native_sizes_from_values,native_eta,_=nullity_parameters(native_values,d,k)
                expected.update(spectrum=source_values.ravel(),eta=[source_eta],nullity=source_sizes,
                    regularized=source.barycenter_weights(x,x,neighbors,reg=.001).ravel())
                assembled,regularized=assemble_modified(native_v,native_values,native_sizes,cfg['modified_tol'],neighbors)
                np.testing.assert_allclose(native_matrix,assembled,atol=3.e-8,rtol=3.e-8)
                np.testing.assert_allclose(actual['regularized'].reshape(n,k),regularized,atol=3.e-8,rtol=3.e-8)
                checks.append(dict(seed=seed,case=case,name='native_frame_alignment',oracle='sklearn_from_native_eigensystem',passed=True))
                for i in range(n):
                    local=x[neighbors[i]]-x[i];cov=local@local.T
                    basis=native_v[i];ev=np.r_[native_values[i],np.zeros(k-native_values.shape[1])]
                    np.testing.assert_allclose(basis.T@basis,np.eye(k),atol=3.e-8,rtol=3.e-8)
                    residual=np.max(abs(cov@basis-basis*ev))
                    assert residual<3.e-8*max(1,np.max(abs(cov)))
                    for name in ['orthogonality','eigen_residual']:
                        checks.append(dict(seed=seed,case=case,name=f'basis_{i+1}_{name}',oracle='spectral_identity',passed=True))
                if not np.array_equal(native_sizes,source_sizes):
                    cutoff=cutoff_certificate(native_values,source_values,native_sizes,source_sizes,
                        actual['eta'][0],source_eta,d,k)
                    counterfactual,_=assemble_modified(native_v,native_values,source_sizes,cfg['modified_tol'],neighbors)
                    np.testing.assert_allclose(counterfactual,matrix,atol=3.e-7,rtol=3.e-7)
                    cutoff['source_size_alignment_error']=float(np.max(abs(counterfactual-matrix)))
                    cutoffs.append(dict(seed=seed,case=case,**cutoff))
                else:np.testing.assert_array_equal(native_sizes,native_sizes_from_values)
            checked_matrix=native_matrix if cutoff is not None else matrix
            certificate=spectral_certificate(checked_matrix,z,query,x,cfg,actual['error'][0])
            indices=NearestNeighbors(n_neighbors=k,algorithm='brute').fit(x).kneighbors(query,return_distance=False)
            projected=np.einsum('ij,ijk->ik',source.barycenter_weights(query,x,indices,reg=cfg['reg']),z[indices])
            np.testing.assert_allclose(q,projected,atol=3.e-8,rtol=3.e-8)
            if certificate is not None:
                metadata,_=certificate;spectral.append(dict(seed=seed,case=case,**metadata))
            if cutoff is not None:
                complement=orthogonal_complement(np.ones((1,n)))
                smallest=eigh(complement.T@native_matrix@complement,eigvals_only=True)[:d]
                np.testing.assert_allclose(z.T@z,np.eye(d),atol=3.e-9,rtol=3.e-9)
                np.testing.assert_allclose(z.sum(axis=0),0,atol=3.e-8,rtol=0)
                assert np.max(abs(native_matrix@z-z@(z.T@native_matrix@z)))<3.e-8*max(1,np.max(abs(native_matrix)))
                np.testing.assert_allclose(actual['error'],[max(0,smallest.sum())],atol=3.e-8,rtol=3.e-8)
            for name,want in expected.items():
                oracle='input_identity' if name=='preserved' else 'sklearn'
                if cutoff is not None and name in ['alignment','gram','query_gram','cross_gram','error','nullity'] and not np.allclose(actual[name],want,atol=3.e-7,rtol=3.e-7):
                    oracle='sklearn_with_cutoff_certificate'
                elif name in ['gram','query_gram','cross_gram'] and certificate is not None and not np.allclose(actual[name],want,atol=3.e-7,rtol=3.e-7):
                    oracle='sklearn_with_nonunique_eigenspace_certificate'
                elif name in ['neighbors','preserved','nullity']:np.testing.assert_array_equal(actual[name],want)
                else:np.testing.assert_allclose(actual[name],want,atol=3.e-7,rtol=3.e-7,err_msg=f'{seed} {case} {name} {cfg}')
                checks.append(dict(seed=seed,case=case,name=name,oracle=oracle,passed=True))
        print(f'seed {seed}: Hessian and modified LLE passed',flush=True)
    (RESULTS/'lle_variants.json').write_text(json.dumps(dict(sklearn=sklearn.__version__,checks=checks,spectral_certificates=spectral,cutoff_certificates=cutoffs),indent=2)+'\n')
    print(f'{len(checks)} LLE variants result groups passed')


if __name__=='__main__':main()
