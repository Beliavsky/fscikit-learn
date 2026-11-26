"""Shared multioutput classification trees, ragged classes and probabilities."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn_tree_runtime import tree_runtime_provenance

def fixtures():
 for mode,criterion,seed,q,pattern,weighted,budget in itertools.product(
   [0,1],['gini','entropy','log_loss'],[0,7],[1,2,3],range(3),[False,True],[0,5]):
  n=19;p=4;rng=np.random.RandomState(531+q)
  x=rng.normal(size=(n,p))
  if pattern==0:y=np.column_stack([rng.randint(k%3+1,size=n)*17-9+k for k in range(q)])
  elif pattern==1:y=rng.randint(2,size=(n,q))
  else:y=rng.randint(2,size=(n,q))*22-5
  w=None if not weighted else np.resize([1.,0.,.5,2.,3.],n)
  yield dict(mode=mode,criterion=criterion,seed=seed,q=q,n=n,p=p,weighted=weighted,
   budget=budget,depth=4,nf=0,leaf=1,alpha=.01,decrease=0.,pattern=pattern),x,y,w
 for mode,criterion,seed,q,pattern,control in itertools.product(
   [0,1],['gini','entropy','log_loss'],[0,42],[2,5],range(4),range(2)):
  n=15;p=3;rng=np.random.RandomState(719+q)
  x=np.column_stack([np.arange(n),rng.normal(size=n),np.arange(n)%3]).astype(float)
  y=np.column_stack([rng.randint(k%4+1,size=n)*19-11+k for k in range(q)])
  if pattern==1:y[:]=np.arange(n)[:,None]%2
  if pattern==2:y[:]=np.arange(q)[None,:]*17-9
  if pattern==3:
   y=rng.randint(2,size=(n,q));y[-1,0]=99
  w=np.resize([1.,0.,.5,2.,3.],n)if control==0 else np.r_[-.1,np.full(n-1,2.)]
  if pattern==3:w[-1]=0
  yield dict(mode=mode,criterion=criterion,seed=seed,q=q,n=n,p=p,weighted=True,budget=0 if control==0 else 4,
   depth=3,nf=0 if control==0 else 2,leaf=2,alpha=0. if control==0 else .02,
   decrease=.01 if control==0 else .05,pattern=pattern+3),x,y,w
 for mode,criterion,q in itertools.product([0,1],['gini','entropy','log_loss'],[1,2,5]):
  n=7;p=2;x=np.column_stack([np.arange(n),np.arange(n)%2]).astype(float)
  y=np.column_stack([np.arange(n)%(k+2)for k in range(q)])
  yield dict(mode=mode,criterion=criterion,seed=7,q=q,n=n,p=p,weighted=True,budget=0,depth=3,nf=0,
   leaf=1,alpha=0.,decrease=0.,pattern=9),x,y,np.zeros(n)

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
 a=ap.parse_args();records=[];payloads=[]
 for c,x,y,w in fixtures():
  cls=DecisionTreeClassifier if c['mode']==0 else ExtraTreeClassifier
  # ExtraTree's source default sqrt is represented by native max_features=0.
  model=cls(criterion=c['criterion'],random_state=c['seed'],max_depth=c['depth'],
   max_leaf_nodes=c['budget']or None,ccp_alpha=c['alpha'],min_samples_leaf=c['leaf'],
   min_impurity_decrease=c['decrease'],max_features=c['nf']or(None if c['mode']==0 else 'sqrt'))
  with warnings.catch_warnings():
   warnings.simplefilter('ignore');model.fit(x,y,sample_weight=w);path=model.cost_complexity_pruning_path(x,y,sample_weight=w)
   try:score=[model.score(x,y,sample_weight=w)];score_error=None
   except (ValueError,ZeroDivisionError) as error:score=[];score_error=str(error)
  t=model.tree_;classes=[model.classes_]if c['q']==1 else model.classes_
  probs=model.predict_proba(x);probs=[probs]if c['q']==1 else probs
  counts=[len(v)for v in classes]
  expected=dict(metadata=list(map(int,[t.node_count,model.n_outputs_,model.n_features_in_,model.get_depth(),model.get_n_leaves()])),
   class_counts=counts,classes=np.concatenate(classes).tolist(),prediction=model.predict(x).ravel().tolist(),
   probabilities=np.concatenate([v.ravel()for v in probs]).tolist(),leaves=(model.apply(x)+1).tolist(),
   values=np.concatenate([t.value[:,k,:counts[k]]for k in range(c['q'])],axis=1).ravel().tolist(),
   impurity=t.impurity.tolist(),mass=t.weighted_n_node_samples.tolist(),count=t.n_node_samples.tolist(),
   left=(t.children_left+1).tolist(),right=(t.children_right+1).tolist(),
   feature=np.where(t.children_left<0,0,t.feature+1).tolist(),threshold=np.where(t.children_left<0,0,t.threshold).tolist(),
   importance=model.feature_importances_.tolist(),alphas=path.ccp_alphas.tolist(),path_impurities=path.impurities.tolist(),score=score)
  records.append(dict(case=c,source=expected,score_error=score_error,differences=[]))
  text=' '.join(str(c[k])for k in ['n','p','q','mode','criterion','seed','depth','budget','nf','leaf','alpha','decrease'])+f' {int(w is not None)} {int(score_error is None)}\n'
  for arr in [x,y]:text+='\n'.join(' '.join(map(str,row))for row in arr)+'\n'
  text+=' '.join(map(str,np.ones(len(x))if w is None else w))+'\n';payloads.append(text)
 a.report.parent.mkdir(parents=True,exist_ok=True);payload=''.join(payloads);a.report.with_suffix('.in').write_text(payload)
 native=subprocess.run([str(a.executable)],input=payload,text=True,capture_output=True)
 a.report.with_suffix('.stdout').write_text(native.stdout);a.report.with_suffix('.stderr').write_text(native.stderr)
 lines=native.stdout.splitlines();fields=list(records[0]['source'])
 for i,r in enumerate(records):
  chunk=lines[18*i:18*i+18];r['native']={}
  if len(chunk)!=18:r['differences'].append('missing native output');continue
  for key,line in zip(fields,chunk):
   got=np.fromstring(line,sep=' ');r['native'][key]=got.tolist()
   try:
    if key in ['metadata','class_counts','classes','prediction','leaves','left','right','feature','count']:np.testing.assert_array_equal(got,r['source'][key])
    else:np.testing.assert_allclose(got,r['source'][key],rtol=2e-10,atol=2e-12,equal_nan=True)
   except AssertionError:r['differences'].append(key)
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 report=dict(configurations=len(records),records=records,failed_configurations=sum(bool(r['differences'])for r in records),
  native_returncode=native.returncode,input_sha256=hashlib.sha256(payload.encode()).hexdigest(),sklearn=sklearn.__version__,runtime=tree_runtime_provenance(),
  sha256={str(p):sha(p)for p in [Path(__file__),a.executable,Path('src/sklearn_tree.f90')]})
 a.report.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items()if k not in ['records','sha256','runtime']})
 assert not native.returncode and not report['failed_configurations']

if __name__=='__main__':main()
