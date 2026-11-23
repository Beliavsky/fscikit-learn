"""Seeded blobs, moons and circles against sklearn 1.6.1."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,subprocess
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import sklearn
from sklearn.datasets import make_blobs,make_moons,make_circles
ROOT=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SEEDS=[0,7,42,2026,2147483647]
def fixtures():
 for seed,p,k,form,center_mode,scenario,shuf in itertools.product(SEEDS,[1,2,5],[1,3,5],[0,1],range(4),range(6),[False,True]):
  if center_mode==3 and form==0 and k!=3:continue
  n=[1,7,23,1,19,13][scenario]
  counts=[[0]*k,[int(i==k-1) for i in range(k)],list(range(1,k+1)),[i%2 for i in range(k)],[2 if i%2==0 else 0 for i in range(k)],[3]*k][scenario]
  stdmode,std=[(0,1.),(1,0.),(2,1.),(2,1.),(1,1.e308),(1,.5)][scenario]
  boxmode=0 if scenario==0 else 2 if scenario==3 and center_mode in [1,2] else 1
  bounds=[2.,2.] if scenario==5 else [-2.,3.]
  yield dict(kind=0,form=form,n=n,p=p,k=k,seed=seed,shuf=shuf,noise_kind=0,noise=0.,factor=.8,center_mode=center_mode,stdmode=stdmode,std=std,boxmode=boxmode,bounds=bounds,counts=counts,scenario=scenario)
 for seed,kind,form,size_index,noise_index,shuf in itertools.product(SEEDS,[1,2],[0,1],range(7),range(5),[False,True]):
  n=[1,2,3,4,7,23,100][size_index];counts=[[0,0],[0,1],[1,0],[1,1],[2,3],[7,2],[0,11]][size_index]
  noise=[None,0.,.2,3.,1.e308][noise_index]
  for factor in ([.8] if kind==1 else [0.,.1,.8,.99]):
   yield dict(kind=kind,form=form,n=n,p=2,k=2,seed=seed,shuf=shuf,noise_kind=int(noise is not None),noise=noise or 0.,factor=factor,center_mode=0,stdmode=0,std=1.,boxmode=0,bounds=[-10.,10.],counts=counts,scenario=noise_index)
def arrays(c):
 scales=np.resize([0.,.2,1.7],c['k']) if c['scenario']!=3 else np.resize([np.nan,np.inf,0.],c['k'])
 centers=(np.arange(c['k']*c['p'],dtype=float).reshape(c['k'],c['p'])*.7-2.3) if c['center_mode'] in [1,2] else None
 return scales,centers
def expected(c):
 sizes=c['n'] if c['form']==0 else tuple(c['counts']);scales,centers=arrays(c)
 with np.errstate(over='ignore',invalid='ignore'):
  if c['kind']==0:
   if centers is not None:loc=centers
   elif c['form']==1 or c['center_mode']==3:loc=None
   else:loc=c['k']
   options=dict(n_samples=sizes,n_features=c['p']+5 if c['center_mode']==2 else c['p'],centers=loc,random_state=c['seed'],shuffle=c['shuf'])
   if c['stdmode']!=0:options['cluster_std']=c['std'] if c['stdmode']==1 else scales
   if c['boxmode']!=0:options['center_box']=(np.nan,np.inf) if c['boxmode']==2 else tuple(c['bounds'])
   snapshot=None if centers is None else centers.copy()
   x,y,means=make_blobs(**options,return_centers=True);other,again=make_blobs(**options)
   if snapshot is not None:np.testing.assert_array_equal(centers,snapshot)
   result=dict(centers=means,metadata=[len(x),x.shape[1],len(means)])
  else:
   options=dict(n_samples=sizes,random_state=c['seed'],shuffle=c['shuf'])
   if c['noise_kind']:options['noise']=c['noise']
   function=make_moons if c['kind']==1 else make_circles
   if c['kind']==2:options['factor']=c['factor']
   x,y=function(**options);other,again=function(**options);result=dict(metadata=[len(x),2,2])
 np.testing.assert_array_equal(x,other);np.testing.assert_array_equal(y,again)
 result.update(x=x,labels=y,repeated_x=other,repeated_labels=again)
 return {k:np.asarray(v,dtype=float).ravel() for k,v in result.items()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True,type=Path);ap.add_argument('--report',required=True,type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--case',type=int)
 args=ap.parse_args();cases=list(fixtures())
 if args.case is not None:cases=[cases[args.case]]
 inputs=[str(len(cases))];references=[]
 for index,c in enumerate(cases):
  inputs.append(' '.join(map(str,[c['kind'],c['form'],c['n'],c['p'],c['k'],c['seed'],int(c['shuf']),c['noise_kind'],c['center_mode'],c['stdmode'],c['boxmode']])))
  inputs.append(' '.join(format(v,'.17g') for v in [c['noise'],c['factor'],c['std'],*c['bounds']]))
  scales,centers=arrays(c);inputs.append(' '.join(map(str,c['counts'])));inputs.append(' '.join(format(v,'.17g') for v in scales))
  if centers is not None:inputs.extend(' '.join(format(v,'.17g') for v in row) for row in centers)
  references.append(expected(c))
  if (index+1)%1000==0:print('Prepared',index+1,'source configurations',flush=True)
 exe=args.executable.resolve();run=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,cwd=ROOT)
 assert run.returncode==0,(run.stdout[-2500:],run.stderr[-2500:])
 checks=[];inventories=[];index=-1;fields={};nonfinite=0
 for line in run.stdout.splitlines():
  words=line.split()
  if words[0]=='BEGIN':index+=1;fields={}
  elif words[0]=='END':
   oracle=references[index];assert fields.keys()==oracle.keys(),(index,fields.keys()^oracle.keys());inventories.append(sorted(fields))
   for name,value in fields.items():
    want=oracle[name];tolerance=0 if name in ['labels','repeated_labels','metadata','centers'] else 3e-12
    np.testing.assert_allclose(value,want,atol=tolerance,rtol=tolerance,equal_nan=True,err_msg=str((index,cases[index],name)))
    finite=np.isfinite(value)&np.isfinite(want);nonfinite+=int(np.count_nonzero(~np.isfinite(want)))
    checks.append(dict(fixture=index,field=name,passed=True,max_abs_error=float(np.max(np.abs(value[finite]-want[finite]),initial=0))))
   np.testing.assert_array_equal(fields['x'],fields['repeated_x']);np.testing.assert_array_equal(fields['labels'],fields['repeated_labels'])
   checks.append(dict(fixture=index,field='repeat_identity',passed=True,max_abs_error=0.))
  else:
   name=words[0];assert name not in fields;fields[name]=np.array(list(map(float,words[2:])));assert int(words[1])==len(fields[name])
 assert index+1==len(cases)
 invalid=dict(blob_samples='n_samples',blob_features='n_features',blob_centers='n_centers',empty_counts='nonempty',negative_counts='nonnegative',count_overflow='too large',count_centers='lengths differ',center_rows='count mismatch',center_features='have features',nan_centers='finite',negative_std='nonnegative',nan_std='finite',inf_std='finite',vector_std='length',negative_vector_std='nonnegative',matrix_std='scalar or a vector',box_size='two bounds',box_order='upper bound',box_nan='finite',box_overflow='range must be finite',blob_seed='nonnegative',moon_samples='n_samples',circle_samples='n_samples',moon_counts='two entries',circle_counts='two entries',curve_negative='nonnegative',curve_noise='nonnegative',curve_nan='finite',curve_inf='finite',factor_one='[0,1)',factor_negative='[0,1)',factor_nan='finite',curve_seed='nonnegative')
 rejections=[]
 for mode,message in invalid.items():
  result=subprocess.run([str(exe),mode],text=True,capture_output=True,cwd=ROOT);output=result.stdout+result.stderr
  diagnostic=next((line for line in output.splitlines() if line.startswith('fscikit-learn:')), '')
  assert result.returncode!=0 and message.lower() in diagnostic.lower(),(mode,output)
  assert not any(s in output for s in ['SIGFPE','Fortran runtime error']),(mode,output)
  rejections.append(dict(case=mode,passed=True))
 import sklearn.datasets._samples_generator as source
 report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,fixtures=len(cases),checks=checks,inventories=inventories,configurations=cases,rejections=rejections,source_nonfinite_values=nonfinite,source_sha256=sha(source.__file__),validator_sha256=sha(__file__),executable_sha256=sha(exe),native_source_sha256={name:sha(ROOT/'src'/name) for name in ['sklearn_datasets.f90','sklearn_legacy_rng.f90','sklearn_ieee_math.f90']})
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(len(checks),'groups /',len(cases),'fixtures /',len(rejections),'diagnostics passed;',nonfinite,'nonfinite source values',flush=True)
if __name__=='__main__':main()
