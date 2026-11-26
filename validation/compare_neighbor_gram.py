"""Bitwise neighbor norm/middle-term comparison with actual sklearn BLAS calls."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,tempfile
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np,scipy,sklearn
from sklearn.utils import _cython_blas as blas
from threadpoolctl import threadpool_info


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--record-differences',action='store_true');a=ap.parse_args();cases=[]
    shapes=[(1,1),(1,9),(9,1),(9,9),(33,33),(65,65),(256,1),(1,256),(256,256)]
    for seed in [7,42]:
        rng=np.random.RandomState(seed)
        for m,n in shapes:
            for p in [1,3,5,16,32,65,129,257,513,769,1025]:
                for same in ([False,True]if m==n else[False]):
                    x=rng.normal(size=(m,p));y=x.copy()if same else rng.normal(size=(n,p))
                    nx=np.array([blas._dot_memview(v,v)for v in x]);ny=np.array([blas._dot_memview(v,v)for v in y])
                    # The Source test wrapper selects its memory order from A;
                    # match C's storage when a one-feature A is also F-contiguous.
                    middle=np.zeros((m,n),order='F'if x.strides[0]==x.itemsize else'C')
                    blas._gemm_memview(blas.BLAS_Trans.NoTrans,blas.BLAS_Trans.Trans,-2.,x,y,0.,middle)
                    text=f'{m} {n} {p}\n'+'\n'.join(' '.join(map(repr,row.tolist()))for row in np.vstack([x,y]))+'\n'
                    cases.append((dict(seed=seed,m=m,n=n,p=p,same=same),[nx,ny,middle.ravel()],text))
    differences=[];digest=hashlib.sha256();worst=0
    for start in range(0,len(cases),8):
        batch=cases[start:start+8];text=''.join(c[2]for c in batch);digest.update(text.encode())
        with tempfile.TemporaryFile(mode='w+',encoding='utf-8')as output:
            proc=subprocess.run([str(a.executable)],input=text,text=True,stdout=output,stderr=subprocess.PIPE);output.seek(0);lines=output.read().splitlines()
        if proc.returncode or len(lines)!=3*len(batch):
            prefix=a.report.with_name(a.report.stem+f'_batch_{start}')
            Path(str(prefix)+'.in').write_text(text);Path(str(prefix)+'.stdout').write_text('\n'.join(lines)+'\n');Path(str(prefix)+'.stderr').write_text(proc.stderr)
        assert not proc.returncode,(start,proc.stderr)
        assert len(lines)==3*len(batch),(start,len(lines))
        for j,(label,expected,_)in enumerate(batch):
            for k,field in enumerate(['x_norm','y_norm','middle']):
                actual=np.fromstring(lines[3*j+k],sep=' ');want=expected[k]
                if not np.array_equal(actual,want):differences.append(dict(case=label,field=field,actual=actual.tolist(),expected=want.tolist()))
                if actual.shape==want.shape:worst=max(worst,float(np.max(abs(actual-want)/(1+abs(want)))))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();root=Path(__file__).resolve().parents[1]
    pools=threadpool_info()
    for pool in pools:pool['binary_sha256']=sha(pool['filepath'])
    report=dict(configurations=len(cases),groups=3*len(cases),failed_comparison_groups=len(differences),strict_parity=not differences,max_scaled_error=worst,
                provenance=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,blas_wrapper_sha256=sha(blas.__file__),threadpools=pools),
                source_sha256=sha(root/'src/sklearn_neighbor_gram.f90'),driver_sha256=sha(root/'app/reference_neighbor_gram_driver.f90'),validator_sha256=sha(__file__),
                executable_sha256=sha(a.executable),input_sha256=digest.hexdigest(),differences=differences)
    a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='differences'},indent=2))
    assert not differences or a.record_differences,differences[:1]


if __name__=='__main__':main()
