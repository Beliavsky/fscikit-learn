"""Direct sklearn 1.6.1 SMACOF embeddings, stress, seeded restarts and stopping."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np,sklearn,scipy
from sklearn.manifold import smacof,MDS
import sklearn.manifold._mds as source
from sklearn.metrics import euclidean_distances

def fixtures(smoke=False):
 for seed in ([7] if smoke else [0,7,42]):
  rng=np.random.RandomState(seed)
  for n in ([7] if smoke else [1,2,7,17,33]):
   x=rng.normal(size=(n,3));d=euclidean_distances(x)
   for k in ([1,2] if smoke else [1,2,9]):
    for metric,normalized in [(True,'false'),(False,'auto'),(False,'false'),(False,'true')]:
     if not metric and n==1:continue
     for initial in [0,k,k+1]:
      init=rng.normal(size=(n,initial)) if initial else None
      for limit in [1,2,25]:
       for runs in [1,3]:
        yield dict(d=d,init=init,k=k,metric=metric,normalized=normalized,limit=limit,
                   runs=runs,seed=seed,eps=[0.,1.e-4,.01][limit%3])
  for kind in ['zeros','missing','ties','negative','diagonal','near_symmetric','collapsed']:
   n=7;x=rng.normal(size=(n,3));d=euclidean_distances(x)
   if kind=='zeros':d[:]=0
   if kind=='missing':d[0,1]=d[1,0]=d[2,4]=d[4,2]=0
   if kind=='ties':d=np.round(d)
   if kind=='negative':d=-d
   if kind=='diagonal':np.fill_diagonal(d,.25)
   if kind=='near_symmetric':d[0,1]+=1.e-7
   for metric,normalized in [(True,'auto'),(False,'auto'),(False,'false')]:
    for limit in [1,3,30]:
     yield dict(d=d,init=np.zeros((n,2)) if kind=='collapsed' else None,k=2,metric=metric,
                normalized=normalized,limit=limit,runs=2,seed=seed,eps=0.,kind=kind)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True)
 ap.add_argument('--report',type=Path,required=True);ap.add_argument('--reference-cache',type=Path)
 ap.add_argument('--smoke',action='store_true');ap.add_argument('--estimator',action='store_true');args=ap.parse_args()
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 provenance=dict(python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,source_sha256=sha(source.__file__),api='MDS'if args.estimator else 'smacof')
 cache={}
 if args.reference_cache and args.reference_cache.exists():
  saved=json.loads(args.reference_cache.read_text());assert saved['provenance']==provenance;cache=saved['cases']
 cases=[];calls=hits=0
 for index,c in enumerate(fixtures(args.smoke)):
  d,init=c['d'],c['init'];n=len(d);initial=0 if init is None else init.shape[1]
  # The existing native MDS class retains stricter precomputed-input controls
  # and its documented zero-matrix shortcut. The functional grid above is complete.
  if args.estimator and (n<2 or np.any(d<0) or np.any(np.diag(d)!=0) or not np.all(abs(d-d.T)<=1.e-10*np.maximum(1.,np.maximum(abs(d),abs(d.T)))) or not np.any(d)):continue
  lines=[f'{n} {c["k"]} {initial} {int(c["metric"])} {c["seed"]} {c["runs"]} {c["limit"]} {c["eps"]} {c["normalized"]}']
  lines+=[' '.join(map(str,row))for row in d]
  if init is not None:lines+=[' '.join(map(str,row))for row in init]
  request='\n'.join(lines)+'\n';key=hashlib.sha256(request.encode()).hexdigest()
  if key in cache:expected=cache[key];hits+=1
  else:
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    try:
     options=dict(metric=c['metric'],n_components=c['k'],n_init=c['runs'],max_iter=c['limit'],eps=c['eps'],random_state=c['seed'],normalized_stress=c['normalized'] if c['normalized']=='auto' else c['normalized']=='true',n_jobs=1)
     if args.estimator:
      model=MDS(**options,dissimilarity='precomputed').fit(d,init=init)
      z,stress,iterations=model.embedding_,model.stress_,model.n_iter_
     else:z,stress,iterations=smacof(d,init=init,return_n_iter=True,**options)
     norm=not c['metric'] if c['normalized']=='auto' else c['normalized']=='true'
     expected=dict(meta=[*z.shape,iterations,1 if init is not None else c['runs'],int(norm)],embedding=z.ravel(order='F').tolist(),stress=[float(stress)])
    except (ValueError,ArithmeticError)as error:expected=dict(error=str(error))
   expected['warnings']=len(caught);cache[key]=expected;calls+=1
  label={key:value for key,value in c.items()if key not in ['d','init']};label.update(index=index,n=n,initial=initial)
  cases.append((request,expected,label))
 if args.reference_cache:args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 good=[c for c in cases if 'error'not in c[1]];bad=[c for c in cases if 'error'in c[1]]
 command=[args.executable]+(['estimator']if args.estimator else [])
 request=''.join(c[0]for c in good);proc=subprocess.run(command,input=request,text=True,capture_output=True)
 assert proc.returncode==0,proc.stdout[-1000:]+'\n'+proc.stderr
 rows=iter(proc.stdout.splitlines());worst=0;groups=0
 for index,(_,expected,label)in enumerate(good,1):
  assert next(rows)==f'CASE {index}'
  for field in ['meta','embedding','stress']:
   actual=np.fromstring(next(rows),sep=' ');want=np.asarray(expected[field]);groups+=1
   if field=='meta':np.testing.assert_array_equal(actual,want,err_msg=str(label))
   else:
    np.testing.assert_allclose(actual,want,rtol=3.e-8,atol=3.e-8,equal_nan=True,err_msg=str((label,field)))
    finite=np.isfinite(actual)&np.isfinite(want)
    if finite.any():worst=max(worst,float(np.max(abs(actual[finite]-want[finite])/(1+abs(want[finite])))))
 assert next(rows,None)is None
 for request,expected,label in bad:
  proc=subprocess.run(command,input=request,text=True,capture_output=True);output=proc.stdout+proc.stderr
  assert proc.returncode and 'fscikit-learn:'in output,(label,expected,output)
  assert 'SIGFPE'not in output and 'Fortran runtime error'not in output,(label,output)
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),successful_fits=len(good),source_rejections=len(bad),groups=groups,
  max_scaled_error=worst,source_calls=calls,reference_cache_hits=hits,source_warnings=sum(c[1]['warnings']for c in cases),
  provenance=provenance,input_sha256=hashlib.sha256(''.join(c[0]for c in cases).encode()).hexdigest(),
  executable_sha256=sha(args.executable),validator_sha256=sha(__file__),
  source_sha256={n:sha(root/'src'/n)for n in ['sklearn_smacof.f90','sklearn_mds.f90']},reference_cache_sha256=sha(args.reference_cache)if args.reference_cache else None,
  inventory=[c[2]for c in cases])
 args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='inventory'},indent=2))
if __name__=='__main__':main()
