"""Source-compatible quantile-loss guards and native keyword/rank adapters."""
from pathlib import Path
import argparse,json,subprocess,warnings
import numpy as np
import compare_quantile_losses as ref
BASE={'empty_rows':'nonempty','empty_outputs':'nonempty','row_shape':'shapes differ','column_shape':'shapes differ','mixed_shape':'shapes differ','target_nan':'must be finite','prediction_inf':'must be finite','weight_length':'length mismatch','empty_weights':'length mismatch','weight_zero':'nonzero total','weight_cancel':'nonzero total','output_length':'length mismatch','empty_output_weights':'length mismatch','single_output_custom':'multiple outputs','output_nan':'must be finite','output_inf':'must be finite','output_zero':'nonzero total','raw_scalar':'use *_raw','variance_mode':'invalid quantile multioutput','bad_mode':'invalid quantile multioutput','mixed_options':'cannot combine'}
def fixture(case):
 y=np.array([[1.,2.],[3.,4.]]);c=dict(y=y,p=y.copy(),w=None,ow=None,alpha=.5,mode=0,ky=2,kp=2)
 if case=='empty_rows':c['y']=y[:0];c['p']=y[:0]
 if case=='empty_outputs':c['y']=y[:,:0];c['p']=y[:,:0]
 if case=='row_shape':c['p']=y[:1]
 if case=='column_shape':c['p']=y[:,:1]
 if case=='mixed_shape':c['y']=y[:,:1];c['ky']=1
 if case=='target_nan':c['y'][0,0]=np.nan
 if case=='prediction_inf':c['p'][0,0]=np.inf
 if case=='weight_length':c['w']=np.ones(1)
 if case=='empty_weights':c['w']=np.ones(0)
 if case=='weight_zero':c['w']=np.zeros(2)
 if case=='weight_cancel':c['w']=np.array([1.,-1.])
 if case in ['weight_nan','weight_inf']:c['w']=np.array([np.nan if case=='weight_nan' else np.inf,1.])
 if case in ['output_length','empty_output_weights','single_output_custom','output_nan','output_inf','output_zero','mixed_options']:
  c['mode']=2;c['ow']=np.ones(2)
  if case=='output_length':c['ow']=np.ones(1)
  if case=='empty_output_weights':c['ow']=np.ones(0)
  if case=='single_output_custom':c.update(y=y[:,:1],p=y[:,:1],ky=1,kp=1,ow=np.ones(1))
  if case=='output_nan':c['ow'][0]=np.nan
  if case=='output_inf':c['ow'][0]=np.inf
  if case=='output_zero':c['ow']=np.array([1.,-1.])
  if case=='mixed_options':c['mode']=7
 if case in ['raw_scalar','variance_mode','bad_mode']:c['mode']={'raw_scalar':3,'variance_mode':4,'bad_mode':5}[case]
 if case.startswith('alpha_'):c['alpha']={'alpha_low':-.1,'alpha_high':1.1,'alpha_nan':np.nan,'alpha_inf':np.inf}[case]
 if case=='baseline_nonfinite':
  c.update(y=np.array([[-np.finfo(float).max],[np.finfo(float).max]]),p=np.zeros((2,1)))
 return c
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 paths=[a.executable,Path(__file__),Path(ref.__file__),ref.ROOT/'src/sklearn_quantile_losses.f90',ref.ROOT/'src/sklearn_numpy_ordering.f90',ref.ROOT/'app/reference_quantile_losses_driver.f90'];before={str(p):ref.sha(p) for p in paths};checks=[]
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore')
  for metric in range(3):
   cases=dict(BASE)
   if metric<2:cases.update(alpha_low='[0,1]',alpha_high='[0,1]',alpha_nan='alpha must be finite',alpha_inf='alpha must be finite')
   if metric>0:cases.update(weight_nan='sample weights must be finite',weight_inf='sample weights must be finite',baseline_nonfinite='quantile baseline must be finite')
   for case,message in cases.items():
    c=fixture(case);error=None;adapter=case in ['raw_scalar','mixed_options']
    try:ref.source(c,metric)
    except (ValueError,ZeroDivisionError,TypeError) as exc:error=type(exc).__name__
    assert error is not None or adapter,(metric,case)
    r=subprocess.run([str(a.executable)],input='1\n'+ref.encode(c,metric,0),text=True,capture_output=True)
    ref.controlled(r);assert message in r.stdout+r.stderr,(metric,case,r.stderr)
    checks.append(dict(metric=metric,case=case,source_type=error,native_adapter=adapter,passed=True))
 assert before=={str(p):ref.sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,hashes=before,checks=checks),indent=2)+'\n');print(len(checks),'controlled quantile diagnostics PASS',flush=True)
if __name__=='__main__':main()
