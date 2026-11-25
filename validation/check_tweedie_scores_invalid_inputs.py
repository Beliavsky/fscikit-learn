"""Controlled Tweedie power/domain, shape, weight and null-model guards."""
from pathlib import Path
import argparse,json,subprocess,warnings
import numpy as np
import compare_tweedie_scores as ref
CASES={'empty_rows':'nonempty','empty_outputs':'nonempty','row_shape':'shapes differ','column_shape':'shapes differ','mixed_shape':'shapes differ','multioutput':'reject multioutput','target_nan':'must be finite','prediction_inf':'must be finite','weight_length':'length mismatch','empty_weights':'length mismatch','weight_zero':'nonzero total','weight_cancel':'nonzero total','power_gap':'at most zero','power_nan':'power must be finite','power_inf':'power must be finite','power_negative_inf':'power must be finite','prediction_zero':'strictly positive','prediction_negative':'strictly positive','nonnegative_target':'nonnegative','positive_target':'strictly positive','constant_null':'null deviance is zero'}

def fixture(case):
 y=np.array([[1.],[2.],[3.]]);c=dict(y=y,p=y.copy(),w=None,power=0.,ky=2,kp=2)
 if case=='empty_rows':c['y']=y[:0];c['p']=y[:0]
 if case=='empty_outputs':c['y']=y[:,:0];c['p']=y[:,:0]
 if case=='row_shape':c['p']=y[:1]
 if case=='column_shape':c['p']=np.column_stack([y,y])
 if case=='mixed_shape':c['p']=np.column_stack([y,y]);c['ky']=1
 if case=='multioutput':c['y']=np.column_stack([y,y]);c['p']=c['y'].copy()
 if case=='target_nan':c['y'][0,0]=np.nan
 if case=='prediction_inf':c['p'][0,0]=np.inf
 if case=='weight_length':c['w']=np.ones(1)
 if case=='empty_weights':c['w']=np.ones(0)
 if case=='weight_zero':c['w']=np.zeros(3)
 if case=='weight_cancel':c['w']=np.array([1.,-1.,0.])
 if case=='power_gap':c['power']=.5
 if case=='power_nan':c['power']=np.nan
 if case=='power_inf':c['power']=np.inf
 if case=='power_negative_inf':c['power']=-np.inf
 if case in ['prediction_zero','prediction_negative']:c['power']=1.;c['p'][0,0]=0 if case=='prediction_zero' else -1
 if case=='nonnegative_target':c['power']=1.;c['y'][0,0]=-1
 if case=='positive_target':c['power']=2.;c['y'][0,0]=0
 if case=='constant_null':c['y'][:]=1;c['p'][:]=2
 return c

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--profile',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 paths=[a.executable,Path(__file__),Path(ref.__file__),ref.ROOT/'src/sklearn_tweedie_scores.f90',ref.ROOT/'src/sklearn_regression_metrics.f90',ref.ROOT/'app/reference_tweedie_scores_driver.f90'];before={str(p):ref.sha(p) for p in paths};checks=[]
 with warnings.catch_warnings(),np.errstate(all='ignore'):
  warnings.simplefilter('ignore')
  for metric in range(2):
   for case,message in CASES.items():
    if metric==0 and case=='constant_null':continue
    c=fixture(case);error=None
    try:ref.source(c,metric)
    except (ValueError,ZeroDivisionError,TypeError) as exc:error=type(exc).__name__
    assert error is not None,(metric,case)
    r=subprocess.run([str(a.executable)],input='1\n'+ref.encode(c,metric,0),text=True,capture_output=True)
    ref.controlled(r);assert message in r.stdout+r.stderr,(metric,case,r.stderr)
    checks.append(dict(metric=metric,case=case,source_type=error,passed=True))
 assert before=={str(p):ref.sha(p) for p in paths}
 a.report.write_text(json.dumps(dict(profile=a.profile,hashes=before,checks=checks),indent=2)+'\n');print(len(checks),'controlled Tweedie diagnostics PASS',flush=True)

if __name__=='__main__':main()
