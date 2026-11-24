"""StandardScaler weighted dense/CSR streaming contracts against local sklearn."""
from pathlib import Path
import argparse,hashlib,itertools,json,subprocess,warnings,os
os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import numpy as np
import scipy.sparse as sp
import sklearn
from sklearn.preprocessing import StandardScaler
from compare_robust import matrix,put
from compare_csr_projections import csr_payload
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fixtures():
 for seed,fit,batch,query,center,scaling,wm,vm in itertools.product([0,42],[0,1],[-1,0,1],[0,1],[0,1],[0,1],[0,1,2],[0,1,2]):
  if center and (fit or batch==1 or query):continue
  if batch<0 and vm:continue
  yield dict(seed=seed,fit=fit,batch=batch,query=query,center=center,scaling=scaling,wm=wm,vm=vm,qcenter=center,qscaling=scaling,scenario='random',style=1,p=5)
 for fit,batch,query,scaling,scenario,style in itertools.product([0,1],[0,1],[0,1],[0,1],['missing','allnan','zero','tiny','large','negative','zero_weight','fractional'],[1,2]):
  yield dict(seed=7,fit=fit,batch=batch,query=query,center=0,scaling=scaling,wm=1,vm=0,qcenter=0,qscaling=scaling,scenario=scenario,style=style,p=3)
 for p,scenario,fit,batch in itertools.product([1,8],['missing','allnan','tiny','large','negative','zero_weight','fractional'],[0,1],[0,1]):
  yield dict(seed=123,fit=fit,batch=batch,query=0,center=0,scaling=1,wm=2,vm=0,qcenter=0,qscaling=1,scenario=scenario,style=1,p=p)
 for query in [0,1]:
  yield dict(seed=13,fit=0,batch=0,query=query,center=0,scaling=1,wm=1,vm=1,qcenter=0,qscaling=0,scenario='random',style=1,p=4)

 for scenario,fit,batch,scaling in itertools.product(['scalar_nan','scalar_inf','scalar_ninf','overflow'],[0,1],[0,1],[0,1]):
  if scenario.startswith('scalar_') and fit and not scaling:continue
  yield dict(seed=97,fit=fit,batch=batch,query=0,center=0,scaling=scaling,wm=2,vm=0,qcenter=0,qscaling=scaling,scenario=scenario,style=1,p=3)

def inputs(c):
 arrays=[matrix(n,c['p'],c['style'],c['seed']+i) for i,n in enumerate([9,7,5])]
 for a in arrays[:2]:
  if c['scenario']=='missing':a.data[::3]=np.nan
  if c['scenario']=='allnan':a.data[:]=np.nan
  if c['scenario']=='zero':a.data[:]=0
  if c['scenario']=='tiny':a.data*=1e-160
  if c['scenario']=='large':a.data=a.data*1e140+1e150
  if c['scenario']=='overflow':a.data[:]=1e308
 w=[np.random.default_rng(c['seed']+i+20).uniform(.1,2,a.shape[0]) for i,a in enumerate(arrays[:2])]
 if c['scenario']=='negative':w[0][::2]*=-1
 if c['scenario']=='zero_weight':w[0][:]=0
 if c['scenario']=='fractional':w[0][:]=.15
 if c['scenario'].startswith('scalar_'):w[0][:]=dict(scalar_nan=np.nan,scalar_inf=np.inf,scalar_ninf=-np.inf)[c['scenario']]
 return arrays,w

def oracle(c,arrays,weights):
 x,b,q=[a.copy() if kind==1 else a.toarray() for a,kind in zip(arrays,[c['fit'],c['batch'],c['query']])]
 model=StandardScaler(with_mean=bool(c['center']),with_std=bool(c['scaling']));f={}
 def snapshot(prefix):
  for k in ['mean','var','scale']:f[prefix+k]=[] if getattr(model,k+'_') is None else getattr(model,k+'_').copy()
  n=np.asarray(model.n_samples_seen_);f[prefix+'counts']=np.broadcast_to(n,(c['p'],)).copy()
  f[prefix+'metadata']=[model.n_features_in_,int(n.ndim==0),int(n.dtype.kind=='f')]
 def weight(mode,w):return None if mode==0 else w if mode==1 else w[0]
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore');model.fit(x,sample_weight=weight(c['wm'],weights[0]));snapshot('fit_')
  if c['batch']>=0:model.partial_fit(b,sample_weight=weight(c['vm'],weights[1]))
  snapshot('batch_');model.with_mean=bool(c['qcenter']);model.with_std=bool(c['qscaling'])
  put(f,'query',model.transform(q));put(f,'inverse',model.inverse_transform(q));f['preserved']=[1]*4
  model.with_mean=bool(c['center']);model.with_std=bool(c['scaling']);model.fit(arrays[2].toarray());snapshot('refit_')
 return f

def payload(c,arrays,w):
 return ' '.join(str(c[k]) for k in ['fit','batch','query','center','scaling','wm','vm','qcenter','qscaling'])+'\n'+''.join(csr_payload(a) for a in arrays)+''.join(' '.join(map(str,a))+'\n' for a in w)

def reject_cases(executable):
 results=[]
 base=next(fixtures())
 for name in ['infinite_fit','infinite_batch','infinite_query','bad_weight','sparse_center','batch_width','query_width','empty_fit','sparse_no_std_nan_scalar']:
  c=base|dict(batch=0);arrays,w=inputs(c)
  if name=='sparse_no_std_nan_scalar':c.update(fit=1,center=0,scaling=0,wm=2);w[0][:]=np.nan
  if name=='infinite_fit':arrays[0].data[0]=np.inf
  if name=='infinite_batch':arrays[1].data[0]=np.inf
  if name=='infinite_query':arrays[2].data[0]=np.inf
  if name=='bad_weight':c['wm']=1;w[0][0]=np.nan
  if name=='sparse_center':c['fit']=1;c['center']=1
  if name=='batch_width':arrays[1]=arrays[1][:,:-1].tocsr()
  if name=='query_width':arrays[2]=arrays[2][:,:-1].tocsr()
  if name=='empty_fit':arrays[0]=sp.csr_matrix((0,c['p']));w[0]=np.empty(0)
  try:oracle(c,arrays,w)
  except (ValueError,TypeError):pass
  else:raise AssertionError(('source accepted',name))
  r=subprocess.run([str(executable)],input='1\n'+payload(c,arrays,w),capture_output=True,text=True)
  out=r.stdout+r.stderr
  assert r.returncode and 'fscikit-learn:' in out and 'SIGFPE' not in out and 'Fortran runtime error' not in out,(name,out)
  results.append(dict(name=name,passed=True,oracle='sklearn'))
 return results

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);args=ap.parse_args()
 paths=[Path(__file__),args.executable,ROOT/'app/reference_standard_driver.f90',ROOT/'src/sklearn_standard_scaling.f90',ROOT/'validation/compare_robust.py',ROOT/'validation/compare_csr_projections.py',Path(sklearn.__file__).parent/'preprocessing/_data.py',Path(sklearn.__file__).parent/'utils/extmath.py',Path(sklearn.__file__).parent/'utils/sparsefuncs_fast.pyx'];hashes={str(p):sha(p) for p in paths}
 cases=list(fixtures());assert len({json.dumps(c,sort_keys=True) for c in cases})==len(cases)
 data=[inputs(c) for c in cases];expected=[oracle(c,*a) for c,a in zip(cases,data)]
 r=subprocess.run([str(args.executable)],input=str(len(cases))+'\n'+''.join(payload(c,*a) for c,a in zip(cases,data)),capture_output=True,text=True);assert r.returncode==0,(r.stdout[-800:],r.stderr[-1500:])
 lines=iter(r.stdout.splitlines());index=-1;checks=[];inventories=[];nonfinite=0
 for line in lines:
  p=line.split()
  if not p:continue
  if p[0]=='BEGIN':index+=1;fields=set();assert int(p[1])==index+1
  elif p[0]=='END':assert fields==expected[index].keys();inventories.append(sorted(fields))
  else:
   field=p[0];assert field not in fields;fields.add(field);values=np.fromstring(next(lines),sep=' ');want=np.asarray(expected[index][field]).ravel(order='F');assert len(values)==int(p[1]) and values.shape==want.shape,(index,field)
   rtol=0.
   exact=field.endswith(('_shape','_ptr','_idx','metadata')) or field=='preserved'
   if exact:np.testing.assert_array_equal(values,want,err_msg=str((index,cases[index],field)))
   else:
    # Incremental variance subtracts nearly equal weighted sums. Permit the
    # forward error implied by mean/std conditioning on the large-offset case.
    rtol=3e-8
    if cases[index]['scenario']=='large' and field not in ['fit_mean','batch_mean','refit_mean','fit_counts','batch_counts','refit_counts']:
     mean=np.asarray(expected[index]['batch_mean']);var=np.asarray(expected[index]['batch_var'])
     std=np.sqrt(np.abs(var));condition=np.max(np.divide(np.abs(mean),std,out=np.ones_like(mean),where=std>0),initial=1)
     rtol=max(rtol,float(64*np.finfo(float).eps*condition))
    np.testing.assert_allclose(values,want,rtol=rtol,atol=3e-8,equal_nan=True,err_msg=str((index,cases[index],field)))
   finite=np.isfinite(want);nonfinite+=int(np.count_nonzero(~finite));checks.append(dict(fixture=index,field=field,passed=True,rtol=rtol,max_abs_error=float(np.max(np.abs(values[finite]-want[finite]),initial=0))))
 assert index+1==len(cases) and hashes=={str(p):sha(p) for p in paths}
 d=dict(rejections=reject_cases(args.executable),profile=args.profile,sklearn=sklearn.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=cases,source_nonfinite_values=nonfinite,hashes=hashes)
 (args.report or ROOT/'validation/results'/f'standard_{args.profile}.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(len(cases),len(checks),nonfinite,'PASS')
if __name__=='__main__':main()
