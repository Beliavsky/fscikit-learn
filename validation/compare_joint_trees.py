"""Joint multioutput CART/ExtraTree regressions versus actual sklearn trees."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn_tree_runtime import tree_runtime_provenance

def fixtures():
 for mode,criterion,seed,q,(n,p),weighted,budget in itertools.product(
   [0,1],['squared_error','friedman_mse','absolute_error','poisson'],[0,7],[1,2,3],
   [(9,2),(21,4)],[False,True],[0,5]):
  rng=np.random.RandomState(140+n+p+q)
  x=rng.normal(size=(n,p));y=rng.normal(size=(n,q))
  if criterion=='poisson':y=np.abs(y)+.1
  weights=None if not weighted else np.resize([1.,0.,.5,2.,3.],n)
  yield dict(mode=mode,criterion=criterion,seed=seed,q=q,n=n,p=p,weighted=weighted,
   budget=budget,depth=4,nf=0,leaf=1,alpha=.01,decrease=0.),x,y,weights
 for mode,criterion,seed,q,pattern,control in itertools.product(
   [0,1],['squared_error','friedman_mse','absolute_error','poisson'],[0,42],[2,5],range(4),range(2)):
  n=12;p=3;t=np.arange(n,dtype=float)-5
  x=np.column_stack([t,np.sin(t),np.repeat([0.,1.,2.],4)])
  y=np.column_stack([(-1.)**k*t+np.cos(t*(k+1))for k in range(q)])
  if pattern==0:y[:,:2]=np.column_stack([t,-t])
  if pattern==1:y[:]=3.14
  if pattern==2:y[:,0]=0
  if pattern==3:y[:]=t[:,None]
  if criterion=='poisson':
   y=np.abs(y)+.1
   if pattern==2:y[:,0]=0
  weight_sets=[None]if control==0 else[np.resize([1.,0.,.5,2.,3.],n)]
  if control==1 and criterion!='absolute_error':weight_sets.append(np.r_[-.1,np.full(n-1,2.)])
  for iw,w in enumerate(weight_sets):
   yield dict(mode=mode,criterion=criterion,seed=seed,q=q,n=n,p=p,weighted=w is not None,
    budget=0 if control==0 else 4,depth=3,nf=0 if control==0 else 2,leaf=2,
    alpha=0. if control==0 else .02,decrease=.01 if control==0 else .05,
    pattern=pattern,control=control,weight_pattern=iw),x,y,w

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
 a=ap.parse_args();records=[];payloads=[]
 for c,x,y,w in fixtures():
  cls=DecisionTreeRegressor if c['mode']==0 else ExtraTreeRegressor
  model=cls(criterion=c['criterion'],random_state=c['seed'],max_depth=c['depth'],
   max_leaf_nodes=c['budget']or None,ccp_alpha=c['alpha'],max_features=c['nf']or None,
   min_samples_leaf=c['leaf'],min_impurity_decrease=c['decrease'])
  with warnings.catch_warnings():
   warnings.simplefilter('ignore');model.fit(x,y,sample_weight=w);path=model.cost_complexity_pruning_path(x,y,sample_weight=w)
   score=model.score(x,y,sample_weight=w)
  t=model.tree_;expected=dict(metadata=list(map(int,[t.node_count,model.n_outputs_,model.n_features_in_,model.get_depth(),model.get_n_leaves()])),
   prediction=model.predict(x).ravel().tolist(),leaves=(model.apply(x)+1).tolist(),values=t.value[:,:,0].ravel().tolist(),
   impurity=t.impurity.tolist(),mass=t.weighted_n_node_samples.tolist(),count=t.n_node_samples.tolist(),
   left=(t.children_left+1).tolist(),right=(t.children_right+1).tolist(),
   feature=np.where(t.children_left<0,0,t.feature+1).tolist(),threshold=np.where(t.children_left<0,0,t.threshold).tolist(),
   importance=model.feature_importances_.tolist(),alphas=path.ccp_alphas.tolist(),path_impurities=path.impurities.tolist(),score=[score])
  records.append(dict(case=c,source=expected,differences=[]))
  text=' '.join(str(c[k])for k in ['n','p','q','mode','criterion','seed','depth','budget','nf','leaf','alpha','decrease'])+f' {int(w is not None)}\n'
  for arr in [x,y]:text+='\n'.join(' '.join(map(str,row))for row in arr)+'\n'
  text+=' '.join(map(str,np.ones(len(x))if w is None else w))+'\n';payloads.append(text)
 a.report.parent.mkdir(parents=True,exist_ok=True);payload=''.join(payloads);a.report.with_suffix('.in').write_text(payload)
 result=subprocess.run([str(a.executable)],input=payload,text=True,capture_output=True)
 a.report.with_suffix('.stdout').write_text(result.stdout);a.report.with_suffix('.stderr').write_text(result.stderr)
 lines=result.stdout.splitlines();fields=list(records[0]['source'])
 for i,r in enumerate(records):
  chunk=lines[15*i:15*i+15];r['native']={}
  if len(chunk)!=15:r['differences'].append('missing native output');continue
  for key,line in zip(fields,chunk):
   got=np.fromstring(line,sep=' ');r['native'][key]=got.tolist();want=r['source'][key]
   try:
    if key in ['metadata','leaves','left','right','feature','count']:np.testing.assert_array_equal(got,want)
    else:np.testing.assert_allclose(got,want,rtol=2e-10,atol=2e-12,equal_nan=True)
   except AssertionError:r['differences'].append(key)
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 report=dict(configurations=len(records),records=records,failed_configurations=sum(bool(r['differences'])for r in records),
  native_returncode=result.returncode,input_sha256=hashlib.sha256(payload.encode()).hexdigest(),
  sklearn=sklearn.__version__,runtime=tree_runtime_provenance(),
  sha256={str(p):sha(p)for p in [Path(__file__),a.executable,Path('src/sklearn_tree.f90')]})
 a.report.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items()if k not in ['records','sha256','runtime']})
 assert not result.returncode and not report['failed_configurations']

if __name__=='__main__':main()
