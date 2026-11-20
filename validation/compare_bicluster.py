"""Dense spectral co-/biclustering, intermediate stages and bicluster getters."""
from pathlib import Path
import argparse
import json
import os
import subprocess
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
import sklearn
from scipy.linalg import norm
from sklearn.cluster import SpectralCoclustering,SpectralBiclustering
from sklearn.cluster._bicluster import _scale_normalize
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"validation/results"
FLAGS="-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow"


class Trace:
    def _svd(self,array,n_components,n_discard):
        left,right=super()._svd(array,n_components,n_discard)
        self.normalized=array.copy(); self.left=left.copy(); self.right=right.copy()
        self.requested=n_components; self.discard=n_discard
        return left,right

    def _k_means(self,data,n_clusters):
        centers,labels=super()._k_means(data,n_clusters)
        if getattr(self,"active_piece",None) is not None:
            self.active_piece.append(norm(data.ravel()-centers[labels].ravel()))
        self.last_cost=np.sum((data-centers[labels])**2)
        return centers,labels

    def _fit_best_piecewise(self,vectors,n_best,n_clusters):
        self.active_piece=[]
        chosen=super()._fit_best_piecewise(vectors,n_best,n_clusters)
        if not hasattr(self,"piece_errors"): self.piece_errors=[]; self.chosen=[]
        self.piece_errors.append(np.array(self.active_piece))
        self.chosen.append(np.array([np.flatnonzero(np.all(vectors==v,axis=1))[0] for v in chosen]))
        self.active_piece=None
        return chosen


class SourceCo(Trace,SpectralCoclustering):
    pass


class SourceBi(Trace,SpectralBiclustering):
    pass


def order(labels,k):
    ids=list(dict.fromkeys(map(int,labels))); ids += [i for i in range(k) if i not in ids]
    inverse=np.empty(k,dtype=int); inverse[ids]=np.arange(k)
    return inverse[np.asarray(labels,dtype=int)],np.array(ids)


def canonical(data,kind,kr,kc,n,p):
    data={key:np.asarray(value).copy() for key,value in data.items()}
    rl,cl=data['row_labels'].astype(int),data['column_labels'].astype(int)
    if kind==0:
        joined,blocks=order(np.r_[rl,cl],kr); data['row_labels']=joined[:n]; data['column_labels']=joined[n:]
    else:
        data['row_labels'],ro=order(rl,kr); data['column_labels'],co=order(cl,kc)
        blocks=(ro[:,None]*kc+co).ravel()
    for key,width in [('rows',n),('columns',p),('copied_rows',n),('copied_columns',p)]:
        data[key]=data[key].reshape((len(blocks),width),order='F')[blocks]
    original=data.copy()
    for new,old in enumerate(blocks):
        for suffix in ('rows','columns','shape','data'):
            data[f'block{new}_{suffix}']=original[f'block{old}_{suffix}']
    return data


def normalize_info(x,method):
    if method!='bistochastic': return [1,1]
    current=x+max(0,-x.min())
    for it in range(1,1001):
        updated,_,_=_scale_normalize(current)
        if norm(current-updated)<1e-5: return [it,1]
        current=updated
    return [1000,0]


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--executable',type=Path); args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable: executable=args.executable.resolve()
    else:
        proc=subprocess.run(['fpm','run','--flag',FLAGS,'--target','reference_bicluster_driver','--runner'],
                            cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if proc.returncode: raise RuntimeError(proc.stdout)
        paths=[ROOT/s.strip().strip(chr(34)) for s in proc.stdout.splitlines()
               if s.strip().strip(chr(34)).endswith(('reference_bicluster_driver','reference_bicluster_driver.exe'))]
        paths=[p for p in paths if p.is_file()]; assert len(paths)==1,proc.stdout
        executable=paths[0].resolve()
    checks=[]
    for seed,n,p in ((7,13,11),(42,17,14),(0,15,19),(2026,23,16)):
        rng=np.random.default_rng(seed)
        rows=rng.permutation(np.arange(n)%3); cols=rng.permutation(np.arange(p)%3)
        levels=rng.uniform(2,8,(3,3))+15*np.eye(3)
        original=levels[rows[:,None],cols]+rng.uniform(.1,.8,(n,p))
        cases=[]
        for clusters,mini,init,negative in ((3,False,'k-means++',False),(2,False,'k-means++',False),
            (4,False,'k-means++',False),(3,False,'random',False),(3,True,'k-means++',False),
            (3,True,'random',False),(5,False,'k-means++',False),(3,False,'k-means++',True)):
            cases.append((0,clusters,clusters,'scale',6,3,mini,init,negative))
        for method in ('scale','bistochastic','log'):
            for kr,kc,mini,init,components,best,negative in ((3,3,False,'k-means++',6,3,False),
                (2,3,False,'random',6,3,False),(3,2,True,'k-means++',6,3,False),(2,2,False,'k-means++',3,2,True)):
                cases.append((1,kr,kc,method,components,best,mini,init,negative))
        for case,(kind,kr,kc,method,components,best,mini,init,negative) in enumerate(cases):
            x=original.copy()
            if negative: x-=x.min()+3
            kwargs=dict(n_init=10,random_state=seed,mini_batch=mini,init=init,n_svd_vecs=max(n,p))
            if kind==0: model=SourceCo(n_clusters=kr,**kwargs)
            else: model=SourceBi(n_clusters=(kr,kc),method=method,n_components=components,n_best=best,**kwargs)
            data,output=RESULTS/'bicluster_input.txt',RESULTS/'bicluster_actual.txt'
            with data.open('w') as stream:
                stream.write(f'{kind} {n} {p} {kr} {kc} {components} {best} 10 {seed} {mini}\n{method} {init}\n')
                np.savetxt(stream,x,fmt='%.17g')
            subprocess.run([str(executable),str(data),str(output)],cwd=ROOT,check=True)
            actual={}
            for line in output.read_text().splitlines():
                name,*values=line.split(); actual[name]=np.array(values,dtype=float)
            model.fit(x)
            signs=np.sign(model.right[np.argmax(abs(model.right),axis=0),np.arange(model.right.shape[1])])
            left,right=model.left*signs,model.right*signs
            blocks=kr if kind==0 else kr*kc
            expected=dict(row_labels=model.row_labels_,column_labels=model.column_labels_,rows=model.rows_,
                columns=model.columns_,copied_rows=model.biclusters_[0],copied_columns=model.biclusters_[1],
                normalized=model.normalized,left=left,right=right,
                singular=np.linalg.svd(model.normalized,compute_uv=False)[:min(model.requested,n,p)],
                normalization=normalize_info(x,method),meta=[n,p,blocks])
            if kind==0:
                _,rs,cs=_scale_normalize(x)
                expected.update(row_scale=rs,column_scale=cs,embedding=np.vstack([left*rs[:,None],right*cs[:,None]]),
                                cost=[model.last_cost])
            else:
                li,ri=model.chosen
                expected.update(left_errors=model.piece_errors[0],right_errors=model.piece_errors[1],
                    best_left=li,best_right=ri,row_projection=x@right[:,ri],column_projection=x.T@left[:,li])
            for i in range(blocks):
                ri,ci=model.get_indices(i)
                expected.update({f'block{i}_rows':ri+1,f'block{i}_columns':ci+1,
                                 f'block{i}_shape':model.get_shape(i),f'block{i}_data':model.get_submatrix(i,x)})
            assert actual.keys()==expected.keys(),(seed,case,actual.keys()^expected.keys())
            actual=canonical(actual,kind,kr,kc,n,p); expected=canonical(expected,kind,kr,kc,n,p)
            for name,value in expected.items():
                discrete=(name in ('row_labels','column_labels','rows','columns','copied_rows','copied_columns',
                                  'meta','normalization','best_left','best_right') or name.endswith(('_rows','_columns','_shape')))
                tol=0 if discrete else 3e-8
                np.testing.assert_allclose(actual[name].ravel(order='F'),np.asarray(value).ravel(order='F'),
                    atol=tol,rtol=tol,err_msg=f'seed={seed} case={case} {name}')
                checks.append(dict(seed=seed,case=case,name=name,passed=True))
    (RESULTS/'bicluster.json').write_text(json.dumps(dict(sklearn=sklearn.__version__,checks=checks),indent=2)+'\n')
    print(f'{len(checks)} spectral bicluster result groups passed')


if __name__=='__main__': main()
