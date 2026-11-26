"""Compare complete native KD partitions/bounds with actual sklearn KDTree."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,tempfile
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np,sklearn
from sklearn.neighbors import KDTree
import sklearn.neighbors._kd_tree as source


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True);ap.add_argument('--record-differences',action='store_true')
    a=ap.parse_args();cases=[]
    for seed in [7,42]:
        rng=np.random.RandomState(seed)
        for n in [1,2,9,32,33,40,41,60,61,65,129,257,513]:
            for p in [1,5,16]:
                original=rng.normal(size=(n,p))
                for kind in ['regular','duplicate','constant','equal_spread']:
                    x=original.copy()
                    if kind=='duplicate':x[:min(5,n)]=x[0]
                    if kind=='constant':x[:]=0
                    if kind=='equal_spread':x=np.repeat(np.arange(n,dtype=float)[:,None],p,axis=1)
                    for leaf in [1,2,7,30,40]:
                        tree=KDTree(x,leaf_size=leaf);_,ids,nodes,bounds=tree.get_arrays()
                        spans=np.column_stack([nodes['idx_start']+1,nodes['idx_end'],nodes['is_leaf']])
                        expected=[ids+1,spans.ravel(),bounds[0].ravel(),bounds[1].ravel()]
                        request=f'{n} {p} {leaf}\n'+'\n'.join(' '.join(map(repr,row.tolist()))for row in x)+'\n'
                        cases.append((dict(seed=seed,n=n,p=p,kind=kind,leaf_size=leaf),expected,request))
    differences=[];digest=hashlib.sha256()
    for start in range(0,len(cases),32):
        batch=cases[start:start+32];text=''.join(c[2]for c in batch);digest.update(text.encode())
        with tempfile.TemporaryFile(mode='w+',encoding='utf-8')as output:
            proc=subprocess.run([str(a.executable)],input=text,text=True,stdout=output,stderr=subprocess.PIPE)
            output.seek(0);lines=output.read().splitlines()
        if proc.returncode or len(lines)!=4*len(batch):
            prefix=a.report.with_name(a.report.stem+f'_batch_{start}')
            Path(str(prefix)+'.in').write_text(text);Path(str(prefix)+'.stdout').write_text('\n'.join(lines)+'\n')
            Path(str(prefix)+'.stderr').write_text(proc.stderr)
        assert proc.returncode==0,(start,proc.stderr)
        assert len(lines)==4*len(batch),(start,len(lines),len(batch))
        for j,(label,expected,_)in enumerate(batch):
            for k,field in enumerate(['indices','spans','low','high']):
                actual=np.fromstring(lines[4*j+k],sep=' ')
                if not np.array_equal(actual,expected[k]):
                    differences.append(dict(case=label,field=field,actual=actual.tolist(),expected=expected[k].tolist()))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();root=Path(__file__).resolve().parents[1]
    report=dict(configurations=len(cases),groups=4*len(cases),failed_comparison_groups=len(differences),strict_parity=not differences,
                provenance=dict(python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__,oracle_sha256=sha(source.__file__)),
                source_sha256=sha(root/'src/sklearn_tree_partition.f90'),driver_sha256=sha(root/'app/reference_tree_partition_driver.f90'),
                validator_sha256=sha(__file__),executable_sha256=sha(a.executable),input_sha256=digest.hexdigest(),differences=differences)
    a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='differences'},indent=2))
    assert not differences or a.record_differences,differences[:1]


if __name__=='__main__':main()
