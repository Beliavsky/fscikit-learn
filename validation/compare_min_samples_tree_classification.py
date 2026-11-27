"""Shared multioutput classification trees, ragged classes and probabilities."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn_tree_runtime import tree_runtime_provenance

def fixtures(smoke=False):
 controls=[(1,2),(.01,.01),(.15,.4),(2,.8),(.3,3),(np.nextafter(1.,0.),1.),(.20000000001,.40000000001)]
 for mode,criterion,seed,q,n,control,weighted in itertools.product([0,1],['gini', 'entropy', 'log_loss'],[0,7],[1,3],[10,25],range(len(controls)),[0,1]):
  if smoke and (mode or seed or q!=1 or n!=10 or control!=2 or not weighted):continue
  p=3;rng=np.random.RandomState(871+n+q);x=rng.normal(size=(n,p))
  y=np.column_stack([rng.randint(k%3+2,size=n)*17-9+k for k in range(q)])
  w=np.resize([1.,0.,.5,2.,3.],n) if weighted else None;leaf,split=controls[control]
  yield dict(mode=mode,criterion=criterion,seed=seed,q=q,n=n,p=p,weighted=weighted,
   budget=5 if seed else 0,depth=4,nf=0,leaf=leaf,split=split,alpha=.01 if seed else 0.,decrease=0.,control=control),x,y,w

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
 ap.add_argument('--smoke',action='store_true')
 a=ap.parse_args();records=[];payloads=[]
 a.features=False
 for c,x,y,w in fixtures(a.smoke):
  cls=DecisionTreeClassifier if c['mode']==0 else ExtraTreeClassifier
  # ExtraTree's source default sqrt is represented by native max_features=0.
  model=cls(criterion=c['criterion'],random_state=c['seed'],max_depth=c['depth'],
   max_leaf_nodes=c['budget']or None,ccp_alpha=c['alpha'],min_samples_leaf=c['leaf'],min_samples_split=c['split'],
   min_impurity_decrease=c['decrease'],max_features=None if c['nf']=='none' else c['nf']or(None if c['mode']==0 else 'sqrt'))
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
  if a.features:expected['resolved_features']=[model.max_features_]
  records.append(dict(case=c,source=expected,score_error=score_error,differences=[]))
  text=' '.join(str(c[k])for k in ['n','p','q','mode','criterion','seed','depth','budget','nf','leaf','alpha','decrease'])+f' {int(w is not None)} {int(score_error is None)} {c["split"]}\n'
  for arr in [x,y]:text+='\n'.join(' '.join(map(str,row))for row in arr)+'\n'
  text+=' '.join(map(str,np.ones(len(x))if w is None else w))+'\n';payloads.append(text)
 a.report.parent.mkdir(parents=True,exist_ok=True);payload=''.join(payloads);a.report.with_suffix('.in').write_text(payload)
 native=subprocess.run([str(a.executable)]+(['--features'] if a.features else []),input=payload,text=True,capture_output=True,timeout=180)
 a.report.with_suffix('.stdout').write_text(native.stdout);a.report.with_suffix('.stderr').write_text(native.stderr)
 lines=native.stdout.splitlines();fields=list(records[0]['source'])
 assert len(lines)==len(records)*len(fields),'unexpected native output length'
 width=18+int(a.features)
 for i,r in enumerate(records):
  chunk=lines[width*i:width*(i+1)];r['native']={}
  if len(chunk)!=width:r['differences'].append('missing native output');continue
  for key,line in zip(fields,chunk):
   got=np.fromstring(line,sep=' ');r['native'][key]=got.tolist()
   try:
    if key in ['resolved_features','metadata','class_counts','classes','prediction','leaves','left','right','feature','count']:np.testing.assert_array_equal(got,r['source'][key])
    else:np.testing.assert_allclose(got,r['source'][key],rtol=2e-10,atol=2e-12,equal_nan=True)
   except AssertionError:r['differences'].append(key)
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 report=dict(features=a.features,configurations=len(records),records=records,failed_configurations=sum(bool(r['differences'])for r in records),
  native_returncode=native.returncode,input_sha256=hashlib.sha256(payload.encode()).hexdigest(),sklearn=sklearn.__version__,runtime=tree_runtime_provenance(),
  sha256={str(p):sha(p)for p in [Path(__file__),Path('app/reference_min_samples_tree_classification_driver.f90'),a.executable,Path('src/sklearn_tree.f90'),Path('src/sklearn_tree_features.f90'),Path('validation/sklearn_feature_fixtures.py')]})
 a.report.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items()if k not in ['records','sha256','runtime']})
 assert not native.returncode and not report['failed_configurations']

if __name__=='__main__':main()
