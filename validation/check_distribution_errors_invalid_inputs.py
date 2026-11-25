"""Controlled log-domain/deviance guards, including RMSLE legacy adaptation."""
from pathlib import Path
import argparse,json,subprocess,warnings
import numpy as np
import compare_distribution_errors as ref
BASE={'empty_rows':'nonempty targets','empty_outputs':'nonempty targets','row_shape':'target shapes differ','column_shape':'target shapes differ','mixed_shape':'target shapes differ','target_nan':'targets must be finite','prediction_inf':'targets must be finite','weight_length':'sample weight length','empty_weights':'sample weight length','zero_weight':'sample weights sum to zero','cancelled_weight':'sample weights sum to zero'}
LOG={'output_length':'output weight length','empty_output_weights':'output weight length','single_output_custom':'require multiple outputs','output_nan':'output weights must be finite','output_inf':'output weights must be finite','zero_output_weight':'output weights sum to zero','raw_scalar':'use *_raw','variance_mode':'invalid logarithmic multioutput mode','bad_mode':'invalid logarithmic multioutput mode','target_minus_one':'greater than -1','target_below_minus_one':'greater than -1','prediction_minus_one':'greater than -1'}
DEV={'multioutput':'deviance rejects multioutput','prediction_zero':'predictions must be strictly positive','prediction_negative':'predictions must be strictly positive','target_negative':None}

def fixture(case,metric):
 q=2 if metric<2 else 1;y=np.arange(1,2*q+1,dtype=float).reshape(2,q)
 c=dict(y=y,p=y.copy(),w=None,ow=None,mode=0,ky=2,kp=2)
 if case=='empty_rows':c['y']=y[:0];c['p']=y[:0]
 if case=='empty_outputs':c['y']=y[:,:0];c['p']=y[:,:0]
 if case=='row_shape':c['p']=y[:1]
 if case=='column_shape':c['p']=np.column_stack([y,y]) if q==1 else y[:,:1]
 if case=='mixed_shape':
  if q==1:c['p']=np.column_stack([y,y]);c['ky']=1
  else:c['y']=y[:,:1];c['ky']=1
 if case=='target_nan':c['y'][0,0]=np.nan
 if case=='prediction_inf':c['p'][0,0]=np.inf
 if case=='weight_length':c['w']=np.ones(1)
 if case=='empty_weights':c['w']=np.ones(0)
 if case=='zero_weight':c['w']=np.zeros(2)
 if case=='cancelled_weight':c['w']=np.array([1.,-1.])
 if case in ['output_length','empty_output_weights','single_output_custom','output_nan','output_inf','zero_output_weight','legacy_single_output','legacy_mixed_options']:
  c['mode']=2;c['ow']=np.ones(q)
  if case=='output_length':c['ow']=np.ones(1)
  if case=='empty_output_weights':c['ow']=np.ones(0)
  if case in ['single_output_custom','legacy_single_output']:c.update(y=y[:,:1],p=y[:,:1],ky=1,kp=1,ow=np.ones(1))
  if case=='output_nan':c['ow'][0]=np.nan
  if case=='output_inf':c['ow'][0]=np.inf
  if case=='zero_output_weight':c['ow']=np.array([1.,-1.])
  if case=='legacy_single_output':c['mode']=6
  if case=='legacy_mixed_options':c['mode']=7
 if case in ['raw_scalar','variance_mode','bad_mode']:c['mode']={'raw_scalar':3,'variance_mode':4,'bad_mode':5}[case]
 if case=='target_minus_one':c['y'][0,0]=-1
 if case=='target_below_minus_one':c['y'][0,0]=np.nextafter(-1.,-np.inf)
 if case=='prediction_minus_one':c['p'][0,0]=-1
 if case=='multioutput':c['y']=np.column_stack([y,y]);c['p']=c['y'].copy()
 if case=='prediction_zero':c['p'][0,0]=0
 if case=='prediction_negative':c['p'][0,0]=-1
 if case=='target_negative':c['y'][0,0]=-1
 if case=='gamma_zero':c['y'][0,0]=0
 return c

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 paths=[a.executable,Path(__file__),Path(ref.__file__),ref.ROOT/'src/sklearn_distribution_errors.f90',ref.ROOT/'src/sklearn_metrics.f90',ref.ROOT/'src/sklearn_regression_metrics.f90',ref.ROOT/'app/reference_distribution_errors_driver.f90'];before={str(p):ref.sha(p) for p in paths};checks=[]
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore')
  for metric in range(4):
   cases=dict(BASE);cases.update(LOG if metric<2 else DEV)
   if metric==1:cases.update(legacy_single_output='require multiple outputs',legacy_mixed_options='cannot combine multioutput and output_weights')
   if metric>=2:cases['target_negative']='Poisson targets must be nonnegative' if metric==2 else 'Gamma targets must be strictly positive'
   if metric==3:cases['gamma_zero']='Gamma targets must be strictly positive'
   for case,message in cases.items():
    c=fixture(case,metric);source_error=None;adapter=case in ['raw_scalar','legacy_mixed_options']
    try:ref.source(c,metric)
    except (ValueError,ZeroDivisionError,TypeError) as exc:source_error=type(exc).__name__
    assert source_error is not None or adapter,(metric,case)
    r=subprocess.run([str(a.executable)],input='1\n'+ref.encode(c,metric,0),text=True,capture_output=True)
    ref.controlled(r);assert message in r.stdout+r.stderr,(metric,case,r.stderr)
    checks.append(dict(metric=metric,case=case,source_type=source_error,native_adapter=adapter,passed=True))
 assert before=={str(p):ref.sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,hashes=before,checks=checks),indent=2)+'\n');print(len(checks),'controlled distribution error diagnostics PASS',flush=True)

if __name__=='__main__':main()
