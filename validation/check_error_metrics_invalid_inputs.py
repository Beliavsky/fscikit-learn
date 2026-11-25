"""Isolated public guards for weighted MSE/RMSE/MAE/MAPE."""
from pathlib import Path
import argparse,hashlib,json,subprocess,warnings
import numpy as np
import compare_error_metrics as ref
CASES={'empty_rows':'nonempty targets','empty_outputs':'nonempty targets','row_shape':'target shapes differ','column_shape':'target shapes differ','mixed_shape':'target shapes differ','target_nan':'targets must be finite','prediction_inf':'targets must be finite','weight_length':'sample weight length','empty_weights':'sample weight length','zero_weight':'sample weights sum to zero','cancelled_weight':'sample weights sum to zero','output_length':'output weight length','empty_output_weights':'output weight length','single_output_custom':'require multiple outputs','output_nan':'output weights must be finite','output_inf':'output weights must be finite','zero_output_weight':'output weights sum to zero','raw_scalar':'use *_raw','variance_mode':'invalid error multioutput mode','bad_mode':'invalid error multioutput mode'}

def fixture(case):
 y=np.array([[1.,2.],[3.,4.]]);c=dict(y=y,p=y.copy(),w=None,ow=None,mode=0,ky=2,kp=2)
 if case=='empty_rows':c['y']=y[:0];c['p']=y[:0]
 if case=='empty_outputs':c['y']=y[:,:0];c['p']=y[:,:0]
 if case=='row_shape':c['p']=y[:1]
 if case=='column_shape':c['p']=y[:,:1]
 if case=='mixed_shape':c['y']=y[:,:1];c['ky']=1
 if case=='target_nan':c['y'][0,0]=np.nan
 if case=='prediction_inf':c['p'][0,0]=np.inf
 if case=='weight_length':c['w']=np.ones(1)
 if case=='empty_weights':c['w']=np.ones(0)
 if case=='zero_weight':c['w']=np.zeros(2)
 if case=='cancelled_weight':c['w']=np.array([1.,-1.])
 if case in ['output_length','empty_output_weights','single_output_custom','output_nan','output_inf','zero_output_weight']:
  c['mode']=2;c['ow']=np.ones(2)
  if case=='output_length':c['ow']=np.ones(1)
  if case=='empty_output_weights':c['ow']=np.ones(0)
  if case=='single_output_custom':c.update(y=y[:,:1],p=y[:,:1],ky=1,kp=1,ow=np.ones(1))
  if case=='output_nan':c['ow'][0]=np.nan
  if case=='output_inf':c['ow'][0]=np.inf
  if case=='zero_output_weight':c['ow']=np.array([1.,-1.])
 if case in ['raw_scalar','variance_mode','bad_mode']:c['mode']={'raw_scalar':3,'variance_mode':4,'bad_mode':5}[case]
 return c

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 paths=[a.executable,Path(__file__),Path(ref.__file__),ref.ROOT/'src/sklearn_error_metrics.f90',ref.ROOT/'app/reference_error_metrics_driver.f90'];before={str(p):ref.sha(p) for p in paths};checks=[]
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore')
  for metric in range(4):
   for case,message in CASES.items():
    if metric==1 and case in ['single_output_custom','output_nan','output_inf']:continue
    c=fixture(case);source_error=None
    try:ref.source(c,metric)
    except (ValueError,ZeroDivisionError,TypeError) as exc:source_error=type(exc).__name__
    assert source_error is not None or case=='raw_scalar',(metric,case)
    r=subprocess.run([str(a.executable)],input='1\n'+ref.encode(c,metric,0),text=True,capture_output=True)
    ref.controlled(r);assert message in r.stdout+r.stderr,(metric,case,r.stderr)
    checks.append(dict(metric=metric,case=case,source_type=source_error,native_static_adapter=case=='raw_scalar',passed=True))
 assert before=={str(p):ref.sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,hashes=before,checks=checks),indent=2)+'\n');print(len(checks),'controlled error metric diagnostics PASS',flush=True)

if __name__=='__main__':main()
