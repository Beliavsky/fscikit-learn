"""Dense/CSR AdditiveChi2Sampler, including raw storage and defined IEEE outcomes."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.kernel_approximation import AdditiveChi2Sampler
from compare_csr_projections import csr_payload
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def config(**kw):return dict(n=7,p=3,m=4,fit_kind=1,query_kind=1,steps=2,interval=-1.,query_steps=2,query_interval=-1.,action=0,mode=0,style=0,scenario='random',qwidth=0)|kw
def fixtures():
 for (n,p),fit_kind,query_kind,style,(steps,interval),action in itertools.product([(7,1),(9,3),(4,7)],[0,1],[0,1],range(4),[(1,-1.),(2,-1.),(3,-1.),(4,.3),(2,0.),(3,1.2),(2,300.),(2,1000.)],[0,1]):
  yield config(n=n,p=p,fit_kind=fit_kind,query_kind=query_kind,style=style,steps=steps,interval=interval,query_steps=steps,query_interval=interval,action=action)
 for scenario,interval,fit_kind,query_kind in itertools.product(['tiny','huge','zero','one'],[0.,1e-300,.5,300.,1e308],[0,1],[0,1]):
  yield config(scenario=scenario,interval=interval,query_interval=interval,fit_kind=fit_kind,query_kind=query_kind,style=1)
 for fit_kind,query_kind,action in itertools.product([0,1],[0,1],[0,1]):
  yield config(fit_kind=fit_kind,query_kind=query_kind,action=action,steps=1,query_steps=3)
  if action:yield config(fit_kind=fit_kind,query_kind=query_kind,action=action,qwidth=5)
def matrix(n,p,style,scenario):
 rng=np.random.RandomState(n*100+p);a=rng.uniform(.1,4,(n,p));a[rng.uniform(size=(n,p))<.4]=0;a[::4]=0
 if scenario!='random':a.fill(dict(tiny=1e-310,huge=1e308,zero=0.,one=1.)[scenario])
 if style==0:return sp.csr_matrix(a)
 data=[];indices=[];ptr=[0]
 for row in a:
  for j in reversed(range(p)):
   if style==2:data.extend([.25*row[j],.75*row[j]]);indices.extend([j,j])
   else:data.append(row[j]);indices.append(j)
  ptr.append(len(data))
 if style==3:return sp.csr_matrix(a.shape)
 return sp.csr_matrix((data,indices,ptr),shape=a.shape)
def inputs(c):return matrix(c['n'],c['p'],c['style'],c['scenario']),matrix(c['m'],c['qwidth'] or c['p'],c['style'],c['scenario'])
def payload(c,arrays):return ' '.join(str(c[k]) for k in ['fit_kind','query_kind','steps','interval','query_steps','query_interval','action','mode'])+'\n'+''.join(csr_payload(a) for a in arrays)
def oracle(c,arrays):
 x,q=arrays;x=x if c['fit_kind'] else x.toarray();q=q if c['query_kind'] else q.toarray();original=[a.copy() for a in [x,q]];state=np.random.get_state()
 model=AdditiveChi2Sampler(sample_steps=c['steps'],sample_interval=None if c['interval']==-1 else c['interval'])
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore');z=model.transform(x) if c['action'] else model.fit_transform(x)
  model.sample_steps=c['query_steps'];model.sample_interval=None if c['query_interval']==-1 else c['query_interval'];query=model.transform(q)
 fields={}
 for prefix,a in [('train',z),('query',query)]:
  fields[prefix+'_shape']=a.shape;fields[prefix+'_ptr']=a.indptr+1 if sp.issparse(a) else [];fields[prefix+'_idx']=a.indices+1 if sp.issparse(a) else [];fields[prefix+'_data']=a.data if sp.issparse(a) else a
 same=lambda a,b:all(np.array_equal(getattr(a,k),getattr(b,k)) for k in ['indptr','indices','data']) if sp.issparse(a) else np.array_equal(a,b)
 fields['metadata']=[getattr(model,'n_features_in_',0),model.sample_steps];fields['controls']=[c['query_interval']]
 fields['preserved']=[int(same(a,b)) for a,b in zip([x,q],original)]+[int(all(np.array_equal(a,b) for a,b in zip(state,np.random.get_state())))]
 assert all(fields['preserved'])
 return fields
def main():
 np.seterr(all='ignore')
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);args=ap.parse_args()
 paths=[Path(__file__),ROOT/'validation/compare_csr_projections.py',args.executable,ROOT/'src/sklearn_kernel_approximation.f90',ROOT/'src/sklearn_sparse.f90',ROOT/'src/sklearn_ieee_math.f90',ROOT/'app/reference_additive_csr_driver.f90',Path(sklearn.__file__).parent/'kernel_approximation.py'];hashes={str(p):sha(p) for p in paths}
 cases=list(fixtures());assert len({json.dumps(c,sort_keys=True) for c in cases})==len(cases);arrays=[inputs(c) for c in cases];expected=[oracle(c,a) for c,a in zip(cases,arrays)]
 r=subprocess.run([str(args.executable)],input=str(len(cases))+'\n'+''.join(payload(c,a) for c,a in zip(cases,arrays)),text=True,capture_output=True)
 if r.returncode:raise AssertionError((r.stdout[-800:],r.stderr[-1400:]))
 lines=iter(r.stdout.splitlines());index=-1;checks=[];inventories=[];nonfinite=0
 for line in lines:
  p=line.split()
  if not p:continue
  if p[0]=='BEGIN':index+=1;fields=set();assert int(p[1])==index+1
  elif p[0]=='END':assert fields==expected[index].keys();inventories.append(sorted(fields))
  else:
   field=p[0];assert field not in fields;fields.add(field);values=np.fromstring(next(lines),sep=' ');assert len(values)==int(p[1]);want=np.asarray(expected[index][field]).ravel(order='F');assert values.shape==want.shape
   if field.endswith('_data'):np.testing.assert_allclose(values,want,rtol=3e-11,atol=3e-12,equal_nan=True,err_msg=str((index,cases[index],field)))
   else:np.testing.assert_array_equal(values,want,err_msg=str((index,cases[index],field)))
   finite=np.isfinite(want);nonfinite+=int(np.count_nonzero(~finite));checks.append(dict(fixture=index,field=field,passed=True,max_abs_error=float(np.max(np.abs(values[finite]-want[finite]),initial=0))))
 assert index+1==len(cases)
 rejections=[]
 for kind in [0,1]:
  base=config(fit_kind=kind,query_kind=kind,style=1)
  variants=[('zero_steps',dict(steps=0)),('negative_steps',dict(steps=-1)),('negative_interval',dict(interval=-2.)),('nan_interval',dict(interval=float('nan'))),('inf_interval',dict(interval=float('inf'))),('missing_interval',dict(steps=4)),('query_missing_interval',dict(query_steps=4)),('nan_fit',dict(mode=1)),('inf_fit',dict(mode=2)),('nan_query',dict(mode=3)),('inf_query',dict(mode=4)),('empty_rows',dict(n=0)),('empty_features',dict(p=0)),('empty_query',dict(m=0)),('query_width',dict(qwidth=5)),('negative_fit',{}),('negative_query',{})]
  if kind:variants += [('bad_index',dict(mode=5)),('bad_start',dict(mode=6)),('bad_pointer',dict(mode=7))]
  for name,change in variants:
   c=base|change;x,q=inputs(c)
   if name=='negative_fit':x.data[0]=-1
   if name=='negative_query':q.data[0]=-1
   wire=payload(c,(x,q))
   try:
    if c['mode'] in [5,6,7]:
     if c['mode']==5:x.indices[0]=-1
     if c['mode']==6:x.indptr[0]=-1
     if c['mode']==7:x.indptr[1]=len(x.data)+1
     x.check_format(full_check=True)
    else:
     if c['mode']==1:x.data[0]=np.nan
     if c['mode']==2:x.data[0]=np.inf
     if c['mode']==3:q.data[0]=np.nan
     if c['mode']==4:q.data[0]=np.inf
     x=x if kind else x.toarray();q=q if kind else q.toarray()
     model=AdditiveChi2Sampler(sample_steps=c['steps'],sample_interval=None if c['interval']==-1 else c['interval']);model.fit_transform(x)
     model.sample_steps=c['query_steps'];model.sample_interval=None if c['query_interval']==-1 else c['query_interval'];model.transform(q)
   except ValueError as error:source_error=type(error).__name__
   else:raise AssertionError(('source accepts invalid',kind,name))
   r=subprocess.run([str(args.executable)],input='1\n'+wire,text=True,capture_output=True);out=r.stdout+r.stderr
   assert r.returncode and 'fscikit-learn:' in out,(kind,name,out)
   assert not any(t in out for t in ['SIGFPE','Fortran runtime error']),(kind,name,out)
   rejections.append(dict(name=str(kind)+'_'+name,passed=True,oracle='scipy_csr_check_format' if c['mode'] in [5,6,7] else 'sklearn_estimator',source_error=source_error))
 for name,c,arrays in [('width',config(p=2,steps=2147483647,interval=1.),None),('stored_count',config(p=1,n=16,steps=100000000,interval=1.,style=1,scenario='one'),None)]:
  x,q=inputs(c)
  # Use a fully stored CSR input so the second case crosses the native index bound.
  r=subprocess.run([str(args.executable)],input='1\n'+payload(c,(x,q)),text=True,capture_output=True);out=r.stdout+r.stderr
  assert r.returncode and 'fscikit-learn:' in out and 'overflow' in out,(name,out)
  assert not any(t in out for t in ['SIGFPE','Fortran runtime error']),(name,out)
  rejections.append(dict(name='native_'+name,passed=True,oracle='native_csr_index_limit'))
 assert hashes=={str(p):sha(p) for p in paths}
 report=dict(profile=args.profile,sklearn=sklearn.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=cases,rejections=rejections,source_nonfinite_values=nonfinite,hashes=hashes)
 (args.report or ROOT/'validation/results'/f'additive_csr_{args.profile}.json').write_text(json.dumps(report,indent=2)+'\n');print(len(cases),len(checks),len(rejections),nonfinite,'PASS')
if __name__=='__main__':main()
