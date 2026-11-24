"""Seeded dictionary signals and sparse SPD samples against installed sklearn/SciPy."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sklearn,scipy
from sklearn.datasets import make_sparse_coded_signal,make_sparse_spd_matrix
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fixtures():
 for seed,n,(k,p) in itertools.product([0,7,42,2026,2147483647],[1,2,7,17],[(1,1),(1,17),(7,1),(5,7),(13,129)]):
  for s in sorted({1,min(3,k),k}):yield dict(kind=0,n=n,k=k,p=p,s=s,seed=seed,normalized=0,alpha=.95,low=.1,high=.9)
 for seed,n,a,normalized,(low,high) in itertools.product([0,7,42,2026,2147483647],[1,2,3,7,16,31,64],[0.,.05,.5,.95,1.],[0,1],[(.1,.9),(.8,.2),(0.,0.),(1.,1.)]):
  yield dict(kind=1,n=n,k=1,p=1,s=1,seed=seed,normalized=normalized,alpha=a,low=low,high=high)
 # scipy/NumPy round-half-to-even population sizes (including zero selections).
 for n,a in [(2,.875),(2,.625),(3,1-2.5/9),(3,1-3.5/9),(7,1-20.5/49)]:
  yield dict(kind=1,n=n,k=1,p=1,s=1,seed=42,normalized=0,alpha=a,low=.1,high=.9)
def expected(c):
 if c['kind']==0:
  x,d,z=map(np.asarray,make_sparse_coded_signal(c['n'],n_components=c['k'],n_features=c['p'],n_nonzero_coefs=c['s'],random_state=c['seed']))
  # Python squeezes singleton axes; the Fortran API consistently retains rank two.
  x=x.reshape(c['n'],c['p']);d=d.reshape(c['k'],c['p']);z=z.reshape(c['n'],c['k'])
  np.testing.assert_allclose(x,z@d,rtol=2e-14,atol=2e-14)
  return dict(data=x.ravel(),dictionary=d.ravel(),code=z.ravel(),metadata=np.array([c['n'],c['p'],c['k'],c['p'],c['n'],c['k']]),support=np.count_nonzero(z,axis=1),repeat_identity=np.ones(1))
 options=dict(n_dim=c['n'],alpha=c['alpha'],norm_diag=bool(c['normalized']),smallest_coef=c['low'],largest_coef=c['high'],random_state=c['seed'])
 x=make_sparse_spd_matrix(**options);csr=make_sparse_spd_matrix(**options,sparse_format='csr')
 np.testing.assert_array_equal(x,csr.toarray())
 return dict(matrix=x.ravel(),csr_dense=x.ravel(),metadata=np.array([c['n'],c['n']]),csr_consistency=np.ones(1),csr_nonzeros=np.array([np.count_nonzero(x)]))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--report',required=True,type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--case',type=int);args=ap.parse_args()
 cases=list(fixtures())
 if args.case is not None:cases=[cases[args.case]]
 refs=[expected(c) for c in cases];inputs=[str(len(cases))]+[' '.join(str(c[k]) for k in ['kind','n','k','p','s','seed','normalized','alpha','low','high']) for c in cases]
 print('Prepared',len(cases),'source fixtures',flush=True)
 exe=args.executable.resolve();r=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True);assert r.returncode==0,(r.stdout[-2000:],r.stderr[-2000:])
 checks=[];inventories=[];index=-1
 for line in r.stdout.splitlines():
  parts=line.split()
  if parts[0]=='BEGIN':index+=1;fields={}
  elif parts[0]=='END':
   assert fields.keys()==refs[index].keys(),(index,fields.keys(),refs[index].keys());inventories.append(sorted(fields))
   for name,value in fields.items():
    want=refs[index][name];tol=3e-13 if name in ['data','dictionary','matrix','csr_dense'] else 0
    np.testing.assert_allclose(value,want,rtol=tol,atol=tol,err_msg=str((index,cases[index],name)))
    checks.append(dict(fixture=index,field=name,passed=True,max_abs_error=float(np.max(np.abs(value-want),initial=0))))
  else:
   assert parts[0] not in fields;fields[parts[0]]=np.array(list(map(float,parts[2:])));assert len(fields[parts[0]])==int(parts[1])
 assert index+1==len(cases)
 invalid=dict(samples='n_samples',components='n_components',features='n_features',support='n_nonzero_coefs',oversupport='n_nonzero_coefs',coded_seed='nonnegative',dimension='n_dim',population='index range',alpha_low='[0,1]',alpha_high='[0,1]',alpha_nan='finite',alpha_inf='finite',coef_low='[0,1]',coef_high='[0,1]',coef_nan='finite',coef_inf='finite',spd_seed='nonnegative')
 rejections=[]
 for mode,message in invalid.items():
  r=subprocess.run([str(exe),mode],text=True,capture_output=True);output=r.stdout+r.stderr;primary=next((s for s in output.splitlines() if s.startswith('fscikit-learn:')),'')
  assert r.returncode!=0 and message in primary,(mode,output);assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(mode,output)
  rejections.append(dict(case=mode,passed=True))
 import sklearn.datasets._samples_generator as source
 import scipy.sparse._construct as sparse_source
 d=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,fixtures=len(cases),checks=checks,configurations=cases,inventories=inventories,rejections=rejections,executable_sha256=sha(exe),validator_sha256=sha(__file__),source_sha256=sha(source.__file__),scipy_source_sha256=sha(sparse_source.__file__),native_source_sha256={n:sha(ROOT/'src'/n) for n in ['sklearn_structured_datasets.f90','sklearn_datasets.f90','sklearn_legacy_rng.f90','sklearn_sparse.f90']})
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(len(checks),'groups /',len(cases),'fixtures /',len(rejections),'diagnostics PASS',flush=True)
if __name__=='__main__':main()
