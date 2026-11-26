"""Direct serial ArgKmin and NumPy fallback selection, including cutoff ties."""
from pathlib import Path
import argparse,hashlib,importlib,json,os,subprocess,sys
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np,sklearn
from sklearn.metrics._pairwise_distances_reduction import ArgKmin
import sklearn.metrics._pairwise_distances_reduction._dispatcher as source


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--executable',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--record-differences',action='store_true')
    a=ap.parse_args();cases=[];requests=[]
    for seed in [7,42]:
        rng=np.random.RandomState(seed)
        for n in [1,2,3,9,33,65,257,513,1025]:
            for kind in ['continuous','duplicate','constant','near_ties','overflow_all','overflow_mixed']:
                values=rng.uniform(0,8,n)
                if kind=='duplicate':values=rng.randint(0,5,n).astype(float)
                if kind=='constant':values=np.ones(n)
                if kind=='near_ties':values=1+rng.randint(-3,4,n)*np.spacing(1.)
                if kind.startswith('overflow'):values=np.full(n,np.inf)
                if kind=='overflow_mixed':values[0]=0
                for k in sorted(set([1,min(3,n),max(1,n//2),n])):
                    for queries in sorted(set([1,n,n+1,1024,1025])):
                        x=np.zeros((queries,1));y=np.ascontiguousarray(values[:,None])
                        distance,indices=ArgKmin.compute(x,y,k=k,metric='manhattan',strategy='auto',return_distance=True)
                        expected=(indices[0]+1,distance[0])
                        label=dict(seed=seed,n=n,k=k,queries=queries,kind=kind,backend='argkmin')
                        cases.append((label,expected));requests.append(f'{n} {k} {queries} 0\n'+' '.join(map(repr,values.tolist()))+'\n')
                    order=np.argpartition(values,k-1)[:k];order=order[np.argsort(values[order])];expected=(order+1,values[order])
                    label=dict(seed=seed,n=n,k=k,queries=1,kind=kind,backend='pairwise')
                    cases.append((label,expected));requests.append(f'{n} {k} 1 1\n'+' '.join(map(repr,values.tolist()))+'\n')
    text=''.join(requests);proc=subprocess.run([str(a.executable)],input=text,text=True,capture_output=True)
    lines=proc.stdout.splitlines()
    if proc.returncode or len(lines)!=2*len(cases):
        prefix=a.report.with_suffix('')
        Path(str(prefix)+'.in').write_text(text);Path(str(prefix)+'.stdout').write_text(proc.stdout)
        Path(str(prefix)+'.stderr').write_text(proc.stderr)
        Path(str(prefix)+'_failure.json').write_text(json.dumps(dict(returncode=proc.returncode,lines=len(lines),requested=len(cases)),indent=2)+'\n')
    assert proc.returncode==0,proc.stderr
    assert len(lines)==2*len(cases),(len(lines),len(cases))
    differences=[]
    for j,(label,expected) in enumerate(cases):
        for field,line,want,dtype in [('indices',lines[2*j],expected[0],int),('values',lines[2*j+1],expected[1],float)]:
            actual=np.fromstring(line,sep=' ',dtype=dtype)
            if not np.array_equal(actual,want):differences.append(dict(case=label,field=field,actual=actual.tolist(),expected=want.tolist()))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source_root=Path(sklearn.__file__).parent
    oracle_files=['utils/_heap.pyx','utils/_sorting.pyx','metrics/_dist_metrics.pyx.tp',
                  'metrics/_pairwise_distances_reduction/_argkmin.pyx.tp',
                  'metrics/_pairwise_distances_reduction/_base.pyx.tp']
    oracle_hashes={name:sha(source_root/name)for name in oracle_files}
    for name in ['sklearn.metrics._pairwise_distances_reduction._argkmin','numpy._core._multiarray_umath']:
        oracle_hashes[name]=sha(importlib.import_module(name).__file__)
    report=dict(configurations=len(cases),groups=2*len(cases),failed_comparison_groups=len(differences),strict_parity=not differences,
                provenance=dict(python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__,dispatcher_sha256=sha(source.__file__),oracle_sha256=oracle_hashes,threads=1,chunk_size=sklearn.get_config()['pairwise_dist_chunk_size']),
                source_sha256=sha(Path(__file__).resolve().parents[1]/'src/sklearn_neighbor_selection.f90'),
                validator_sha256=sha(__file__),executable_sha256=sha(a.executable),input_sha256=hashlib.sha256(text.encode()).hexdigest(),differences=differences)
    a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='differences'},indent=2))
    assert not differences or a.record_differences,differences[:1]


if __name__=='__main__':main()
