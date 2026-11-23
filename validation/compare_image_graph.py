"""Image/grid graphs against sklearn 1.6.1, with explicit CSR/IEEE checks."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import warnings

import numpy as np
import scipy
from scipy.sparse import csr_matrix
import sklearn
from sklearn.feature_extraction.image import grid_to_graph,img_to_graph

REJECTIONS={
    'zero_x':'grid graph dimensions must be positive',
    'negative_y':'grid graph dimensions must be positive',
    'zero_z':'grid graph dimensions must be positive',
    'volume_overflow':'volume exceeds int64 range',
    'index_range':'volume exceeds native index range',
    'stored_range':'stored count exceeds native index range',
    **{f'grid_mask_{r}':'grid graph mask size mismatch' for r in ['scalar','1','2','3']},
    **{f'image_mask_{r}':'image graph mask shape mismatch' for r in ['1','2','3']},
}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fields(output):
    tokens=iter(output.split());result=[]
    while True:
        kind=next(tokens,None)
        if kind is None:return result
        assert kind in ['I','F'],kind
        n,p=int(next(tokens)),int(next(tokens));convert=int if kind=='I' else float
        result.append(np.array([convert(next(tokens)) for _ in range(n*p)],
                              dtype=np.int64 if kind=='I' else float).reshape((n,p),order='F'))


def floating(actual,expected,fixture,field):
    assert actual.shape==expected.shape,(fixture,field,actual.shape,expected.shape)
    for label,test in [('nan',np.isnan),('positive_inf',np.isposinf),('negative_inf',np.isneginf)]:
        np.testing.assert_array_equal(test(actual),test(expected),err_msg=f'{fixture} {field} {label}')
    finite=np.isfinite(expected)
    np.testing.assert_allclose(actual[finite],expected[finite],rtol=3.e-14,atol=0,
                               err_msg=f'{fixture} {field}')
    return float(np.max(np.abs(actual[finite]-expected[finite]),initial=0.))


def selection(shape,pattern,rng):
    n=int(np.prod(shape,dtype=np.int64)) if shape else 1
    if pattern=='none' or pattern=='all':result=np.ones(n,dtype=bool)
    elif pattern=='empty':result=np.zeros(n,dtype=bool)
    elif pattern=='checker':result=np.indices(shape).sum(axis=0).ravel()%2==0 if shape else np.array([True])
    elif pattern=='random':result=rng.uniform(size=n)>.4
    elif pattern=='singleton':
        result=np.zeros(n,dtype=bool)
        if n:result[n//2]=True
    elif pattern=='corners':
        result=np.zeros(n,dtype=bool)
        if n:result[0]=result[-1]=True
    else:raise AssertionError(pattern)
    return result.reshape(shape)


def images(shape,regime,rng):
    n=int(np.prod(shape,dtype=np.int64)) if shape else 1
    if regime=='random':a=rng.normal(size=n)
    elif regime=='integers':a=rng.randint(-3,4,size=n).astype(float)
    elif regime=='constant':a=np.full(n,2.)
    elif regime=='signed_zeros':a=np.zeros(n);a[::2]=-0.
    elif regime=='nonfinite':a=np.resize(np.array([np.nan,np.inf,-np.inf,0.,1.]),n)
    elif regime=='overflow':a=np.resize(np.array([np.finfo(float).max,-np.finfo(float).max,1.e200,-1.e200]),n)
    elif regime=='tiny':a=np.resize(np.array([0.,np.nextafter(0.,1.),-np.nextafter(0.,1.),1.e-200,-1.e-200]),n)
    else:raise AssertionError(regime)
    return a.reshape(shape)


def fixtures():
    grids=[(1,1,1),(1,5,1),(5,1,1),(2,3,1),(3,4,1),(2,2,2),(3,4,2),(1,3,4),(3,1,4)]
    patterns=['none','all','empty','checker','random','singleton','corners']
    for dims,pattern in itertools.product(grids,patterns):
        for seed in ([0,42,2026] if pattern=='random' else [0]):
            for mask_rank in ([3] if pattern=='none' else [1,2,3]+([0] if dims==(1,1,1) else [])):
                yield dict(mode=0,rank=3,dims=dims,seed=seed,regime='random',pattern=pattern,
                           mask_rank=mask_rank,summary=False)
    shapes=[(),(0,),(1,),(7,),(0,3),(3,0),(1,1),(1,5),(5,1),(2,3),(3,4),
            (0,3,2),(2,0,3),(2,3,0),(1,1,1),(1,3,4),(3,1,4),(3,4,1),(2,3,4)]
    for shape,seed,regime,pattern in itertools.product(shapes,[0,42],
            ['random','integers','constant','signed_zeros','nonfinite','overflow','tiny'],
            ['none','all','empty','checker','random','corners']):
        dims=shape if len(shape)==3 else (*shape,1) if len(shape)==2 else (1,shape[0],1) if shape else (1,1,1)
        yield dict(mode=1,rank=len(shape),dims=dims,seed=seed,regime=regime,pattern=pattern,
                   mask_rank=len(shape),summary=False)
    for mode,dims,pattern in [(0,(200,300,1),'none'),(0,(80,90,3),'random'),(1,(60,70,2),'random')]:
        yield dict(mode=mode,rank=3,dims=dims,seed=42,regime='random',pattern=pattern,mask_rank=3,summary=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True,type=Path)
    parser.add_argument('--profile',choices=['development','checked','release'],default='development')
    parser.add_argument('--report',required=True,type=Path);args=parser.parse_args();exe=args.executable.resolve()
    cases=list(fixtures());checks=[];inventories=[];source_errors=[];nonfinite=[]
    for index,cfg in enumerate(cases):
        mode,rank=cfg['mode'],cfg['rank'];nx,ny,nz=cfg['dims'];rng=np.random.RandomState(cfg['seed'])
        shape=(nx,ny,nz) if rank==3 else (nx,ny) if rank==2 else (ny,) if rank==1 else ()
        image=images(shape,cfg['regime'],rng);keep=selection(shape,cfg['pattern'],rng)
        masked=cfg['pattern']!='none'
        with warnings.catch_warnings(),np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            if mode==0:
                mr=cfg['mask_rank']
                mask=keep.reshape((nx,ny,nz) if mr==3 else (nx,ny*nz) if mr==2 else (-1,) if mr==1 else ())
                reference=grid_to_graph(nx,ny,nz,mask=mask if masked else None,return_as=csr_matrix,dtype=float)
            else:
                if masked:
                    try:raw=img_to_graph(image,mask=keep,return_as=csr_matrix)
                    except (IndexError,ValueError) as error:
                        source_errors.append(dict(fixture=index,error=type(error).__name__,message=str(error),
                                                  image_shape=list(shape),mask_shape=list(keep.shape)))
                        raw=None
                    # sklearn squeezes only the image when selecting diagonal
                    # values. Squeezing both arrays gives the intended voxel mask.
                    reference=img_to_graph(image,mask=keep.squeeze(),return_as=csr_matrix)
                    if raw is not None:
                        np.testing.assert_array_equal(raw.indptr,reference.indptr)
                        np.testing.assert_array_equal(raw.indices,reference.indices)
                        np.testing.assert_array_equal(raw.data,reference.data)
                else:reference=img_to_graph(image,return_as=csr_matrix)
            reference.sort_indices()
            signs=np.where(np.isnan(reference.data),0,np.signbit(reference.data)).astype(np.int64)
            expected=[('metadata',np.array([*reference.shape,reference.nnz],dtype=np.int64)[:,None]),
                      ('indptr',reference.indptr[:,None]+1),('indices',reference.indices[:,None]+1),
                      ('data',reference.data[:,None]),('signs',signs[:,None])]
            if cfg['summary']:
                right=np.column_stack([np.ones(reference.shape[1]),(np.arange(reference.shape[1])+1)%17-8])
                expected.append(('matmul',reference@right))
            else:
                stored=reference.copy();stored.data=np.ones(reference.nnz,dtype=np.int64)
                expected.extend([('dense',reference.toarray()),('stored_mask',stored.toarray().astype(np.int64))])
        # Driver arrays use Fortran element order; voxel numbering remains C-order.
        image3=image.reshape((nx,ny,nz));mask3=keep.reshape((nx,ny,nz))
        header=[mode,rank,nx,ny,nz,int(masked),cfg['mask_rank'],int(cfg['summary'])]
        payload=' '.join(map(str,header))+'\n'+' '.join(format(v,'.17g') for v in image3.ravel(order='F'))+'\n'
        payload+=' '.join(str(int(v)) for v in mask3.ravel(order='F'))+'\n'
        proc=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
        assert proc.returncode==0,(index,cfg,proc.stderr)
        actual=fields(proc.stdout);assert len(actual)==len(expected),(index,cfg,len(actual),len(expected))
        names=[]
        for value,(field,reference_value) in zip(actual,expected):
            if np.issubdtype(reference_value.dtype,np.integer):
                np.testing.assert_array_equal(value,reference_value,err_msg=f'{index} {field}');error=0.
            else:
                error=floating(value,reference_value,index,field)
                if np.any(~np.isfinite(reference_value)):
                    nonfinite.append(dict(fixture=index,field=field,nan=int(np.isnan(reference_value).sum()),
                                          positive_inf=int(np.isposinf(reference_value).sum()),
                                          negative_inf=int(np.isneginf(reference_value).sum())))
            checks.append(dict(fixture=index,field=field,max_abs_error=error));names.append(field)
        checks.append(dict(fixture=index,field='input_preservation',max_abs_error=0.));names.append('input_preservation')
        inventories.append(dict(fixture=index,**cfg,fields=names))
        if index%200==0:print(f'{index}/{len(cases)} fixtures passed',flush=True)
    for name,message in REJECTIONS.items():
        proc=subprocess.run([str(exe),name],text=True,capture_output=True)
        assert proc.returncode!=0 and message in proc.stdout+proc.stderr,(name,proc.stdout,proc.stderr)
        assert 'SIGFPE' not in proc.stderr and 'Fortran runtime error' not in proc.stderr,(name,proc.stderr)
    report=dict(profile=args.profile,sklearn_version=sklearn.__version__,scipy_version=scipy.__version__,
        numpy_version=np.__version__,fixtures=len(cases),checks=checks,inventories=inventories,
        source_mask_errors=source_errors,nonfinite_reference_results=nonfinite,rejections=list(REJECTIONS),
        executable=str(exe),executable_sha256=digest(exe),validator_sha256=digest(__file__),
        source_sha256={'feature_extraction/image.py':digest(Path(sklearn.__file__).parent/'feature_extraction/image.py')})
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'PASS: {len(cases)} fixtures, {len(checks)} groups, {len(REJECTIONS)} rejections')


if __name__=='__main__':main()
