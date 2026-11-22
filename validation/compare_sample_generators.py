"""Direct sklearn sample-generator arrays, seeded draws and isolated diagnostics."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import numpy as np
import sklearn
from sklearn import datasets
from compare_function_transformer import ROOT,resolve

FUNCTIONS=[datasets.make_friedman1,datasets.make_friedman2,datasets.make_friedman3,
           datasets.make_sparse_uncorrelated,datasets.make_s_curve,datasets.make_swiss_roll,
           datasets.make_gaussian_quantiles]
REJECTIONS={
 'friedman1_samples':'n_samples must be positive','friedman2_samples':'n_samples must be positive',
 'friedman3_samples':'n_samples must be positive','sparse_samples':'n_samples must be positive',
 's_curve_samples':'n_samples must be positive','swiss_samples':'n_samples must be positive',
 'quantile_samples':'n_samples must be positive','friedman1_features':'at least five features',
 'sparse_features':'at least four features','negative_noise':'noise must be nonnegative',
 'nan_noise':'noise must be finite','infinite_noise':'noise must be finite','seed':'seed must be nonnegative',
 'quantile_features':'n_features must be positive','quantile_classes':'n_classes must be positive',
 'quantile_too_many_classes':'n_samples must be at least n_classes','quantile_cov':'cov must be nonnegative',
 'quantile_nan_cov':'cov must be finite','quantile_infinite_cov':'cov must be finite',
 'quantile_mean_length':'mean length mismatch','quantile_mean_nan':'mean must be finite',
}


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['checked','release'],default='checked')
 parser.add_argument('--executable',type=Path);args=parser.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_sample_generators_driver')
 checks=[];errors={'X':0.,'target':0.};fixtures=0
 for family,function in enumerate(FUNCTIONS):
  for seed in [0,1,42,2026]:
   for case in range(12):
    n=[1,2,7,17,40,64,101,113,9,31,16,100][case]
    p=[5,9,17,6,10,51][case%6] if family==0 else [4,7,11,51][case%4] if family==3 else [1,2,3,5][case%4]
    noise=[0.,.1,1.,3.][case%4];config=case
    kwargs=dict(n_samples=n,random_state=seed)
    if family in [0,3]:kwargs['n_features']=p
    if family in [0,1,2,4,5]:kwargs['noise']=noise
    if family==5:kwargs['hole']=bool(case%2)
    center=np.linspace(-1.5,2.5,p)*(case%3)
    if family==6:
     noise=[0.,.01,1.,3.][case%4];config=min(n,[1,2,3,4,7,9][case%6])
     kwargs.update(n_features=p,n_classes=config,cov=noise,mean=center,shuffle=seed%2==0)
    want_x,want_y=function(**kwargs);want_y=np.atleast_1d(want_y)
    payload=f'{family} {n} {p} {seed} {noise:.17g} {config}\n'
    if family==6:payload+=' '.join(f'{v:.17g}' for v in center)+'\n'
    run=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
    assert run.returncode==0,(family,seed,case,run.stdout,run.stderr)
    lines=iter(run.stdout.splitlines());shape=tuple(map(int,next(lines).split()))
    assert shape==want_x.shape,(family,seed,case,shape,want_x.shape)
    got_x=np.array([np.fromstring(next(lines),sep=' ') for j in range(shape[1])]).T
    got_y=np.fromstring(next(lines),sep=' ');assert next(lines,None) is None
    for name,got,want in [('X',got_x,want_x),('target',got_y,want_y)]:
     if family==6 and name=='target':np.testing.assert_array_equal(got,want)
     else:np.testing.assert_allclose(got,want,rtol=3e-14,atol=3e-13,err_msg=f'{function.__name__} seed={seed} case={case}')
     errors[name]=max(errors[name],float(np.max(abs(got-want),initial=0)))
     checks.append(dict(family=function.__name__,seed=seed,case=case,n=n,p=p,name=name,passed=True))
    fixtures+=1
  print(function.__name__+' passed',flush=True)
 report=dict(profile=args.profile,fixtures=fixtures,checks=checks,sklearn=sklearn.__version__,numpy=np.__version__,
             max_absolute_errors=errors,native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
             validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             source_sha256=hashlib.sha256((Path(sklearn.__file__).parent/'datasets/_samples_generator.py').read_bytes()).hexdigest())
 results=ROOT/'validation/results';results.mkdir(exist_ok=True)
 (results/f'sample_generators_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejected=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True)
  assert run.returncode!=0 and 'fscikit-learn:' in run.stdout and message in run.stdout,(case,run.stdout,run.stderr)
  rejected.append(dict(name=case,expected=message,passed=True))
 invalid=dict(profile=args.profile,checks=rejected,native_sha256=report['native_sha256'],validator_sha256=report['validator_sha256'])
 (results/f'sample_generators_invalid_{args.profile}.json').write_text(json.dumps(invalid,indent=2,allow_nan=False)+'\n')
 print(f'{len(checks)} sample-generator groups / {fixtures} fixtures and {len(rejected)} isolated rejection checks passed')


if __name__=='__main__':main()
