"""Actual sklearn LinearModelLoss gradients/Hessians at dense/raw CSR points."""
from pathlib import Path
import argparse,json,os,subprocess,itertools
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np
import sklearn
from scipy.sparse import csr_matrix
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn._loss.loss import HalfTweedieLoss,HalfTweedieLossIdentity

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 cases=[];records=[];rng=np.random.default_rng(4102)
 for power,link,intercept,representation,signed in itertools.product((-2.,-1.,0.,.5,1.,1.5,2.,3.),('identity','log'),(0,1),(0,1,2),(False,True)):
  n,p=13,4;x=rng.normal(size=(n,p))*.2;y=np.exp(rng.normal(size=n)*.3);w=.2+rng.random(n)
  if signed:w[::4]=-.1;w[1]=0
  if power<=0:y[::5]*=-.1
  if 0<power<2:y[::6]=0
  point=rng.normal(size=p+intercept)*.1
  if intercept:point[-1]=1.2 if link=='identity' else .2
  # Identity raw points with no intercept also exercise Source's nonfinite
  # trial arithmetic; this is an objective probe, not a fit acceptance claim.
  sparse=csr_matrix(x)
  if representation==2:
   data=[];columns=[];pointers=[0]
   for row in x:
    for column in reversed(range(p)):
     data.extend([.25*row[column],.75*row[column]]);columns.extend([column,column])
    data.append(0.);columns.append(0);pointers.append(len(data))
   sparse=csr_matrix((data,columns,pointers),shape=x.shape)
  design=x if representation==0 else sparse
  base=(HalfTweedieLossIdentity if link=='identity' else HalfTweedieLoss)(power=power)
  model=LinearModelLoss(base_loss=base,fit_intercept=bool(intercept))
  with np.errstate(all='ignore'):g,h,warning=model.gradient_hessian(point,design,y,sample_weight=w,l2_reg_strength=.2,n_threads=1,hessian_out=np.full((len(point),len(point)),np.nan))
  text=f'{n} {p} {power} {link} .2 {intercept} {representation}\n'+' '.join(format(z,'.17g') for z in point)+'\n'
  text+='\n'.join(' '.join(format(z,'.17g') for z in row) for row in np.column_stack([x,y,w]))+'\n'
  cases.append(text);records.append(dict(power=power,link=link,intercept=intercept,representation=representation,signed=signed,input=text,source_gradient=g.tolist(),source_hessian=h.ravel(order="F").tolist(),source_warning=bool(warning)))
 input_path=a.report.with_suffix('.in');output_path=a.report.with_suffix('.out');input_path.write_text(str(len(cases))+'\n'+''.join(cases))
 result=subprocess.run([str(a.executable),str(input_path),str(output_path)],capture_output=True,text=True)
 if result.returncode:raise RuntimeError((result.returncode,result.stdout,result.stderr))
 lines=output_path.read_text().splitlines();assert len(lines)==3*len(records)
 for i,record in enumerate(records):
  g=np.fromstring(lines[3*i],sep=' ');h=np.fromstring(lines[3*i+1],sep=' ');meta=np.fromstring(lines[3*i+2],sep=' ',dtype=int)
  record.update(native_gradient=g.tolist(),native_hessian=h.tolist(),native_warning=bool(meta[0]),ieee_preserved=bool(meta[1]))
  record['passed']=bool(np.allclose(g,record['source_gradient'],atol=3e-12,rtol=3e-12,equal_nan=True) and record['ieee_preserved'] and record['native_warning']==record['source_warning'])
  if not record['source_warning']:record['passed'] &= bool(np.allclose(h,record['source_hessian'],atol=3e-12,rtol=3e-12,equal_nan=True))
 report=dict(sklearn_version=sklearn.__version__,configurations=len(records),failed=[i for i,r in enumerate(records) if not r['passed']],records=records,stdout=result.stdout,stderr=result.stderr)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print('GLM Hessians',len(records),'failures',report['failed']);raise SystemExit(bool(report['failed']))
if __name__=='__main__':main()
