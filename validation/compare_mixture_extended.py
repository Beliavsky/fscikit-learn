"""Actual sklearn GaussianMixture initialization, warm EM and seeded samples."""
from pathlib import Path
import argparse,hashlib,json,os,sys,subprocess,warnings,concurrent.futures,tempfile
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np,scipy,sklearn
from sklearn.mixture import GaussianMixture
import sklearn.mixture._gaussian_mixture as source,sklearn.mixture._base as base
COVS=['full','tied','diag','spherical'];INITS=['kmeans','random','random_from_data','k-means++']
FIELDS=['meta','weights','means','covariance','precision_cholesky','precision','lower_bound','scores','probability','labels','criteria','fit_labels','samples','sample_labels']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fixtures(smoke):
 for seed in ([7]if smoke else[0,7,42]):
  rng=np.random.RandomState(seed)
  for n,p,k in ([(17,2,3)]if smoke else[(8,1,1),(17,2,3),(31,4,2),(7,9,3)]):
   x=rng.normal(size=(n,p))+.4;q=rng.normal(size=(5,p));a=rng.normal(size=(k,p,p))
   for cov in COVS:
    prec=np.array([v.T@v+.5*np.eye(p)for v in a])
    if cov=='tied':prec=prec[:1]
    if cov in ['diag','spherical']:
     prec=np.array([np.diag(np.linspace(.5,2,p))for _ in range(k)])
     if cov=='spherical':prec=np.array([np.eye(p)*1.25 for _ in range(k)])
    for init in INITS:
     for limit in ([0,1]if smoke else[0,1,8]):
      for mask in ([0,7]if smoke else[0,1,2,4,7]):
       w=np.arange(1,k+1,dtype=float);w/=w.sum()
       yield dict(seed=seed,x=x,q=q,k=k,cov=cov,init=init,limit=limit,mask=mask,w=w,means=x[:k]+.25,prec=prec,reg=1.e-4,tol=1.e-4,kind='regular')
 if not smoke:
  rng=np.random.RandomState(19)
  for kind in ['duplicate','zero','thin']:
   n,p,k=(7,9,3)if kind=='thin'else(13,3,3)
   x=rng.normal(size=(n,p));x[:,-1]=x[:,0]
   if kind=='zero':x[:]=0
   for cov in COVS:
    for init in INITS:
     for reg in [0.,1.e-6]:
      yield dict(seed=19,x=x,q=x[:5]+.1,k=k,cov=cov,init=init,limit=3,mask=0,w=np.ones(k)/k,means=x[:k],prec=None,reg=reg,tol=0.,kind=kind)
def expand(d,cov,k):
 d=np.asarray(d)
 if cov=='full':return d.transpose(1,2,0)
 if cov=='tied':return np.repeat(d[:,:,None],k,axis=2)
 if cov=='diag':return np.stack([np.diag(v)for v in d],axis=2)
 p=expand.p;return np.stack([np.eye(p)*v for v in d],axis=2)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
 ap.add_argument('--reference-cache',type=Path);ap.add_argument('--smoke',action='store_true');ap.add_argument('--record-differences',action='store_true');args=ap.parse_args()
 provenance=dict(oracle_schema=3,python=sys.version,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,source_sha256=sha(source.__file__),base_sha256=sha(base.__file__))
 cache={}
 if args.reference_cache and args.reference_cache.exists():
  old=json.loads(args.reference_cache.read_text());assert old['provenance']==provenance;cache=old['cases']
 cases=[];hits=calls=0
 for index,c in enumerate(fixtures(args.smoke)):
  x,q,k=c['x'],c['q'],c['k'];n,p=x.shape;expand.p=p;cov=c['cov'];mask=c['mask'];warm=int(c['limit']>0 and index%4==0);runs=1 if index%3 else 3;draws=[1,7,61][index%3]
  rows=[f"{n} {p} {k} {len(q)} {COVS.index(cov)+1} {INITS.index(c['init'])+1} {c['seed']} {c['limit']} {runs} {warm} {mask} {draws} {c['tol']} {c['reg']}"]
  rows+=[' '.join(map(str,row))for row in x];rows+=[' '.join(map(str,row))for row in q]
  if mask&1:rows+=[' '.join(map(str,c['w']))]
  if mask&2:rows+=[' '.join(map(str,row))for row in c['means']]
  if mask&4:rows+=[' '.join(map(str,row))for v in c['prec']for row in v]
  request='\n'.join(rows)+'\n';key=hashlib.sha256(request.encode()).hexdigest()
  if key in cache:expected=cache[key];hits+=1
  else:
   model=GaussianMixture(n_components=k,covariance_type=cov,init_params=c['init'],random_state=c['seed'],max_iter=c['limit'],n_init=runs,warm_start=bool(warm),tol=c['tol'],reg_covar=c['reg'])
   if mask&1:model.weights_init=c['w']
   if mask&2:model.means_init=c['means']
   if mask&4:
    model.precisions_init=c['prec']if cov=='full'else(c['prec'][0]if cov=='tied'else(np.diagonal(c['prec'],axis1=1,axis2=2)if cov=='diag'else c['prec'][:,0,0]))
   with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always')
    stage='fit'
    try:
     for _ in range(warm+1):fitlabels=model.fit_predict(x)
     sample=np.empty((0,p));sl=np.empty(0,dtype=int);stage='evaluation'
     expected=dict(meta=[model.n_features_in_,model.n_iter_,int(model.converged_)],weights=model.weights_.tolist(),means=model.means_.ravel(order='F').tolist(),covariance=expand(model.covariances_,cov,k).ravel(order='F').tolist(),precision_cholesky=expand(model.precisions_cholesky_,cov,k).ravel(order='F').tolist(),precision=expand(model.precisions_,cov,k).ravel(order='F').tolist(),lower_bound=[model.lower_bound_],scores=model.score_samples(q).tolist(),probability=model.predict_proba(q).ravel(order='F').tolist(),labels=model.predict(q).tolist(),criteria=[model.score(q),model.aic(q),model.bic(q)],fit_labels=fitlabels.tolist(),samples=[],sample_labels=[])
     try:
      sample,sl=model.sample(draws);expected['samples']=sample.ravel(order='F').tolist();expected['sample_labels']=sl.tolist()
     except (ValueError,TypeError,ArithmeticError)as error:expected['sample_error']=str(error)
    except (ValueError,TypeError,ArithmeticError,np.linalg.LinAlgError,RuntimeError,AttributeError)as error:expected=dict(error=str(error),error_stage=stage)
   expected['warnings']=len(caught);cache[key]=expected;calls+=1
  label={key:c[key]for key in ['seed','cov','init','limit','mask','reg','tol','kind']};label.update(index=index,n=n,p=p,k=k,warm=warm,runs=runs,draws=draws)
  cases.append((request,expected,label))
 if args.reference_cache:args.reference_cache.write_text(json.dumps(dict(provenance=provenance,cases=cache),separators=(',',':'))+'\n')
 good=[c for c in cases if 'error'not in c[1]];bad=[c for c in cases if 'error'in c[1]];differences=[];by_request={};offset=0;width=len(FIELDS)+1
 def fit_request(c):
  if 'sample_error'not in c[1]:return c[0]
  lines=c[0].splitlines();header=lines[0].split();header[11]='0';lines[0]=' '.join(header);return '\n'.join(lines)+'\n'
 while offset<len(good):
  chunk=good[offset:offset+64]
  # Bound each result stream and read file-backed output after process exit.
  # This avoids intermittent truncation of the large Windows pipe capture.
  with tempfile.TemporaryFile(mode='w+',encoding='utf-8')as output:
   proc=subprocess.run([args.executable],input=''.join(fit_request(c)for c in chunk),text=True,stdout=output,stderr=subprocess.PIPE)
   output.seek(0);proc.stdout=output.read()
  lines=proc.stdout.splitlines();completed=sum(row.startswith('CASE ')for row in lines)
  if not proc.returncode:assert len(lines)==len(chunk)*width,(offset,len(chunk),len(lines))
  for j in range(completed):
   assert lines[j*width]==f'CASE {j+1}'
   by_request[good[offset+j][0]]=subprocess.CompletedProcess([],0,'\n'.join(lines[j*width:j*width+width])+'\n','')
  offset+=completed
  if proc.returncode:
   if not args.record_differences:raise RuntimeError(str(good[offset][2])+'\n'+proc.stdout[-500:]+'\n'+proc.stderr)
   by_request[good[offset][0]]=proc;offset+=1
  else:assert completed==len(chunk)
 def native(c):return subprocess.run([args.executable],input=c[0],text=True,capture_output=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
  for c,proc in zip(bad,pool.map(native,bad)):by_request[c[0]]=proc
 sample_bad=[c for c in good if 'sample_error'in c[1]];sample_probes={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
  for c,proc in zip(sample_bad,pool.map(native,sample_bad)):sample_probes[c[0]]=proc
 groups=0;worst=0;fits=0
 for request,expected,label in cases:
  proc=by_request[request];out=proc.stdout+proc.stderr
  if proc.returncode:assert 'fscikit-learn:'in out and 'SIGFPE'not in out and 'Fortran runtime error'not in out,(label,out)
  if 'error'in expected:
   if not proc.returncode:differences.append(dict(case=label,field='Source rejection accepted by native',source=expected['error']))
   continue
  if proc.returncode:
   differences.append(dict(case=label,field='native rejected Source fit',diagnostic=next(row.strip()for row in out.splitlines()if 'fscikit-learn:'in row)));continue
  assert len(proc.stdout.splitlines())==width,(label,len(proc.stdout.splitlines()),repr(proc.stdout))
  fits+=1;rows=iter(proc.stdout.splitlines());next(rows)
  for field in FIELDS:
   actual=np.fromstring(next(rows),sep=' ');want=np.asarray(expected[field])
   if field in ['samples','sample_labels']and 'sample_error'in expected:
    assert actual.size==0;continue
   groups+=1
   try:
    if field in ['meta','labels','fit_labels','sample_labels']:np.testing.assert_array_equal(actual,want,err_msg=str((label,field)))
    else:np.testing.assert_allclose(actual,want,atol=3.e-8,rtol=3.e-8,equal_nan=True,err_msg=str((label,field)))
   except AssertionError:
    if not args.record_differences:raise
    differences.append(dict(case=label,field=field,actual=actual.tolist(),expected=want.tolist()))
   if actual.shape==want.shape:
    finite=np.isfinite(actual)&np.isfinite(want)
    if finite.any():worst=max(worst,float(np.max(abs(actual[finite]-want[finite])/(1+abs(want[finite])))))
  assert next(rows,None)is None
  if 'sample_error'in expected:
   sample_proc=sample_probes[request];text=sample_proc.stdout+sample_proc.stderr
   if sample_proc.returncode:assert 'fscikit-learn:'in text and 'SIGFPE'not in text and 'Fortran runtime error'not in text,(label,text)
   else:differences.append(dict(case=label,field='Source sampling rejection accepted by native',source=expected['sample_error']))
 if differences and not args.record_differences:raise AssertionError(differences)
 root=Path(__file__).resolve().parents[1]
 report=dict(configurations=len(cases),source_successful_fits=sum(c[1].get('error_stage')!='fit'for c in cases),source_successful_configurations=len(good)-len(sample_bad),successful_fits=fits,source_rejections=len(bad)+len(sample_bad),source_fit_rejections=sum(c[1].get('error_stage')=='fit'for c in cases),source_sample_rejections=len(sample_bad),groups=groups,failed_comparison_groups=len(differences),strict_parity=not differences,max_scaled_error=worst,source_calls=calls,reference_cache_hits=hits,source_warnings=sum(c[1]['warnings']for c in cases),provenance=provenance,source_sha256={n:sha(root/'src'/n)for n in ['sklearn_mixture.f90','sklearn_kmeans_core.f90','sklearn_kmeans_seeding.f90']},executable_sha256=sha(args.executable),validator_sha256=sha(__file__),reference_cache_sha256=sha(args.reference_cache)if args.reference_cache else None,input_sha256=hashlib.sha256(''.join(c[0]for c in cases).encode()).hexdigest(),inventory=[c[2]for c in cases],differences=differences)
 args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['inventory','differences']},indent=2))
if __name__=='__main__':main()
