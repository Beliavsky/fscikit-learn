"""Actual BallTree depth-first KNN/radius queries, storage order and cutoffs."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,tempfile
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np,sklearn
from sklearn.neighbors import BallTree
import sklearn.neighbors._ball_tree as source


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True);ap.add_argument('--record-differences',action='store_true')
    a=ap.parse_args();cases=[]
    metrics=[('euclidean',2.),('l2',2.),('manhattan',1.),('cityblock',1.),('l1',1.),('chebyshev',np.inf),
             ('minkowski',1.),('minkowski',2.),('minkowski',.5),('minkowski',.75),('minkowski',1.5),('minkowski',3.),('minkowski',np.inf),
             ('haversine',2.),('braycurtis',2.),('canberra',2.),('hamming',2.),('dice',2.),
             ('jaccard',2.),('rogerstanimoto',2.),('russellrao',2.),('sokalmichener',2.),('sokalsneath',2.)]
    for seed in [7,42]:
        rng=np.random.RandomState(seed)
        for n in [1,9,33,65,129]:
            for p in [1,2,5]:
                original=rng.normal(size=(n,p))
                for kind in ['regular','duplicate','constant','boolean','integer']:
                    x=original.copy()
                    if kind=='duplicate':x[:min(5,n)]=x[0]
                    if kind=='constant':x[:]=0
                    if kind=='boolean':x=(original>0).astype(float)
                    if kind=='integer':x=np.round(original)
                    q=np.vstack([x[0],np.zeros(p),rng.normal(size=p)])
                    for leaf in [1,30]:
                        for metric,power in metrics:
                            if metric=='haversine'and p!=2:continue
                            kw=dict(p=power)if metric=='minkowski'else{}
                            tree=BallTree(x,leaf_size=leaf,metric=metric,**kw)
                            _,order,nodes,bounds=tree.get_arrays()
                            spans=np.column_stack([nodes['idx_start']+1,nodes['idx_end'],nodes['is_leaf']])
                            fields=[('tree_indices',order+1),('spans',spans.ravel()),('centers',bounds[0].ravel()),('radii',nodes['radius'])]
                            cutoff=float(tree.query(q[:1],k=min(3,n))[0][0,-1])
                            radii=[-1.,np.nan,0.,.8,np.inf,cutoff,np.nextafter(cutoff,0),np.nextafter(cutoff,np.inf)]
                            specs=[(0,k,0.)for k in sorted(set([1,min(3,n),n]))]+[(1,1,r)for r in radii]
                            for op,k,radius in specs:
                                if op==0:
                                    distances,indices=tree.query(q,k=k,dualtree=False,breadth_first=False,sort_results=True)
                                else:
                                    indices,distances=tree.query_radius(q,r=radius,return_distance=True,sort_results=False)
                                for j in range(len(q)):
                                    name=f'{op}/{k}/{radius}/{j}'
                                    fields.extend([(name+'/count',np.array([len(indices[j])])),(name+'/indices',indices[j]+1),(name+'/distance',distances[j])])
                            rows=[f'{n} {p} {len(q)} {leaf} {len(specs)} {metric} {power}']
                            rows += [f'{op} {k} {r}'for op,k,r in specs]
                            rows += [' '.join(map(repr,row.tolist()))for row in x]
                            rows += [' '.join(map(repr,row.tolist()))for row in q]
                            cases.append((dict(seed=seed,n=n,p=p,kind=kind,leaf_size=leaf,metric=metric,power=power),fields,'\n'.join(rows)+'\n'))
    differences=[];groups=0;worst=0;digest=hashlib.sha256()
    for start in range(0,len(cases),8):
        batch=cases[start:start+8];text=''.join(c[2]for c in batch);digest.update(text.encode())
        with tempfile.TemporaryFile(mode='w+',encoding='utf-8')as output:
            proc=subprocess.run([str(a.executable)],input=text,text=True,stdout=output,stderr=subprocess.PIPE)
            output.seek(0);lines=output.read().splitlines()
        if proc.returncode or len(lines)!=sum(len(c[1])for c in batch):
            prefix=a.report.with_name(a.report.stem+f'_batch_{start}')
            Path(str(prefix)+'.in').write_text(text);Path(str(prefix)+'.stdout').write_text('\n'.join(lines)+'\n')
            Path(str(prefix)+'.stderr').write_text(proc.stderr)
        assert proc.returncode==0,(start,proc.stderr)
        assert len(lines)==sum(len(c[1])for c in batch),(start,len(lines))
        if start%400==0:print(start,'/',len(cases),'configurations passed',flush=True)
        offset=0
        for label,fields,_ in batch:
            for name,want in fields:
                actual=np.fromstring(lines[offset],sep=' ');offset+=1;groups+=1
                try:
                    if name.endswith('/distance'):np.testing.assert_allclose(actual,want,rtol=3.e-12,atol=3.e-12,equal_nan=True)
                    else:np.testing.assert_array_equal(actual,want)
                except AssertionError:differences.append(dict(case=label,field=name,actual=actual.tolist(),expected=want.tolist()))
                if actual.shape==want.shape:
                    mask=np.isfinite(actual)&np.isfinite(want)
                    if mask.any():worst=max(worst,float(np.max(abs(actual[mask]-want[mask])/(1+abs(want[mask])))))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();root=Path(__file__).resolve().parents[1]
    report=dict(configurations=len(cases),groups=groups,failed_comparison_groups=len(differences),strict_parity=not differences,max_scaled_error=worst,
                provenance=dict(python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__,oracle_sha256=sha(source.__file__)),
                sources={name:sha(root/'src'/name)for name in ['sklearn_tree_partition.f90','sklearn_neighbor_ball_tree.f90','sklearn_neighbor_selection.f90']},
                validator_sha256=sha(__file__),driver_sha256=sha(root/'app/reference_neighbor_ball_tree_driver.f90'),executable_sha256=sha(a.executable),input_sha256=digest.hexdigest(),differences=differences)
    a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='differences'},indent=2))
    assert not differences or a.record_differences,differences[:1]


if __name__=='__main__':main()
