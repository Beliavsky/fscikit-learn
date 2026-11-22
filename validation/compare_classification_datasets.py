"""Full seeded make_classification and Hastie arrays against local sklearn."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import numpy as np
import sklearn
from sklearn.datasets import make_classification,make_hastie_10_2
from compare_function_transformer import ROOT,resolve

REJECTIONS={
 'samples':'sample and feature counts must be positive','features':'sample and feature counts must be positive',
 'informative':'informative count must be positive','redundant':'feature counts must be nonnegative',
 'repeated':'feature counts must be nonnegative','blocks':'feature blocks exceed n_features',
 'classes':'class and cluster counts must be positive','clusters':'class and cluster counts must be positive',
 'cluster_overflow':'cluster count exceeds native integer range','vertices':'needs more informative dimensions',
 'negative_flip':'flip_y must be in [0,1]','large_flip':'flip_y must be in [0,1]','nan_flip':'flip_y must be finite',
 'zero_separation':'class_sep must be positive','nan_separation':'class_sep must be finite','seed':'seed must be nonnegative',
 'weight_length':'weight count mismatch','negative_weight':'weights must be nonnegative',
 'negative_inferred':'weights must be nonnegative','nan_weight':'weights must be finite',
 'large_weight':'weighted cluster size exceeds native integer range','shift_length':'feature_shift length mismatch',
 'scale_length':'feature_scale length mismatch','nan_shift':'shift must be finite','nan_scale':'scale must be finite',
 'zero_scale':'scalar scale must be positive','negative_scale':'scalar scale must be positive',
 'vector_shift_nan':'feature shifts must be finite','vector_scale_infinite':'feature scales must be finite',
 'random_shift_conflict':'random and explicit feature shifts conflict',
 'random_scale_conflict':'random and explicit feature scales conflict',
 'hastie_samples':'sample count must be positive','hastie_seed':'seed must be nonnegative',
}


def configuration(case,seed):
 c=dict(n_samples=101,n_features=20,n_informative=2,n_redundant=2,n_repeated=0,n_classes=2,
        n_clusters_per_class=2,random_state=seed,hypercube=True,shuffle=True,flip_y=.01,class_sep=1.,shift=0.,scale=1.)
 if case==1:c.update(n_samples=1,n_features=4,shuffle=False,flip_y=0.)
 if case==2:c.update(n_samples=2,n_features=1,n_informative=1,n_redundant=0,n_classes=1,n_clusters_per_class=1)
 if case==3:c.update(n_features=6,n_redundant=0,n_repeated=2,shuffle=False,flip_y=0.)
 if case==4:c.update(n_samples=17,n_features=11,n_informative=3,n_repeated=3,n_classes=4)
 if case==5:c.update(n_features=13,n_informative=6,n_repeated=2,n_classes=3)
 if case==6:c.update(n_samples=31,n_features=15,n_informative=10)
 if case==7:c.update(n_samples=9,n_features=33,n_informative=30,n_redundant=1,n_repeated=1)
 if case==8:c.update(n_samples=7,n_features=36,n_informative=31,n_redundant=1,n_repeated=2,n_classes=3)
 if case==9:c.update(n_samples=17,n_features=47,n_informative=40,n_repeated=2,n_classes=4,n_clusters_per_class=1)
 if case==10:c.update(n_samples=31,n_features=9,n_informative=3,n_redundant=1,n_repeated=1,n_classes=3,weights=[.1,.2],shuffle=False)
 if case==11:c.update(n_samples=17,n_features=9,n_informative=4,n_redundant=0,n_classes=3,n_clusters_per_class=3,weights=[0.,0.,0.])
 if case==12:c.update(n_samples=9,n_features=10,n_informative=3,n_repeated=3,weights=[.8,.7])
 if case==13:c.update(n_samples=17,n_features=7,n_informative=3,n_classes=3,n_clusters_per_class=1,weights=[.1,.2,.3])
 if case==14:c.update(n_samples=7,n_features=5,n_informative=1,n_redundant=0,n_classes=1,n_clusters_per_class=1,weights=[],flip_y=1.)
 if case==15:c.update(n_samples=31,n_features=2,n_redundant=0,n_clusters_per_class=1,class_sep=.001,hypercube=False)
 if case==16:c.update(n_features=21,shift=1.5,scale=3.2,hypercube=False,flip_y=1.)
 if case==17:c.update(n_samples=113,class_sep=10.,shift=-2.5,scale=.1,shuffle=False,flip_y=0.)
 if case==18:c.update(shift=None,hypercube=False)
 if case==19:c.update(scale=None)
 if case==20:c.update(shift=None,scale=None,hypercube=False)
 if case==21:c.update(n_samples=17,n_features=17,n_informative=7,n_redundant=3,n_repeated=4,n_classes=4,
                      shift=np.linspace(-2,2,17),scale=np.linspace(-3,3,17),hypercube=False)
 if case==22:c.update(n_samples=16,n_features=5,scale=np.array([0.,-1.,2.,0.,3.]),shift=np.arange(5.),shuffle=False)
 if case==23:c.update(n_samples=3,n_features=5,n_informative=3,n_redundant=0,n_classes=4,flip_y=1.)
 if case==24:c.update(n_samples=17,n_features=8,n_informative=8,n_redundant=0,n_classes=1,n_clusters_per_class=1)
 if case==25:c.update(n_samples=9,n_features=2,n_redundant=0,n_classes=1,n_clusters_per_class=3)
 if case==26:c.update(n_samples=9,n_features=2,n_redundant=0,n_classes=1,n_clusters_per_class=4,hypercube=False)
 if case in [27,28,29]:c.update(n_samples=9,n_features=8,n_informative=8,n_redundant=0,n_classes=1,n_clusters_per_class=282-case,shift=None,scale=None)
 if case==30:c.update(n_samples=17,n_features=8,n_informative=7,n_redundant=0,n_classes=1,n_clusters_per_class=1,hypercube=False,shift=None)
 if case==31:c.update(n_samples=7,n_features=7,n_informative=3,n_redundant=1,n_classes=3,weights=[1.,1.,1.],shuffle=False)
 return c


def payload(family,c):
 fs=isinstance(c['shift'],np.ndarray);fc=isinstance(c['scale'],np.ndarray)
 keys=['n_samples','n_features','n_informative','n_redundant','n_repeated','n_classes','n_clusters_per_class','random_state']
 header=[family]+[c[k] for k in keys]+[int(c['hypercube']),int(c['shuffle']),int(c['shift'] is None),int(c['scale'] is None)]
 text=' '.join(map(str,header))+'\n'
 text+=' '.join(f'{v:.17g}' for v in [c['flip_y'],c['class_sep'],0. if fs or c['shift'] is None else c['shift'],1. if fc or c['scale'] is None else c['scale']])+'\n'
 weights=c.get('weights');text+=f'{-1 if weights is None else len(weights)} {int(fs)} {int(fc)}\n'
 for a in [weights,c['shift'] if fs else None,c['scale'] if fc else None]:
  if a is not None and len(a):text+=' '.join(f'{v:.17g}' for v in a)+'\n'
 return text


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['checked','release'],default='checked')
 parser.add_argument('--executable',type=Path);args=parser.parse_args()
 exe=args.executable.resolve() if args.executable else resolve(args.profile,'reference_classification_datasets_driver')
 checks=[];fixtures=0;max_error=0.
 seeds=[0,1,7,42,137,991,2026,2147483647]
 for family in [0,1]:
  for seed in seeds if family==0 else seeds[:4]:
   for case in range(32 if family==0 else 12):
    c=configuration(case if family==0 else 0,seed)
    if family==0:want_x,want_y=make_classification(**c)
    else:
     c['n_samples']=[1,2,7,17,31,100,257,1001,4097,16,113,12000][case]
     want_x,want_y=make_hastie_10_2(n_samples=c['n_samples'],random_state=seed)
    run=subprocess.run([str(exe)],input=payload(family,c),capture_output=True,text=True)
    assert run.returncode==0,(family,seed,case,run.stdout,run.stderr)
    lines=iter(run.stdout.splitlines());shape=tuple(map(int,next(lines).split()));assert shape==want_x.shape
    got_x=np.array([np.fromstring(next(lines),sep=' ') for j in range(shape[1])]).T
    got_y=np.fromstring(next(lines),sep=' ',dtype=int);assert next(lines,None) is None
    np.testing.assert_allclose(got_x,want_x,rtol=5e-13,atol=3e-12,err_msg=f'{family} {seed} {case}')
    np.testing.assert_array_equal(got_y,want_y,err_msg=f'{family} {seed} {case}')
    max_error=max(max_error,float(np.max(abs(got_x-want_x),initial=0)))
    for name in ['X','labels']:checks.append(dict(family=family,seed=seed,case=case,n=c['n_samples'],p=shape[1],name=name,passed=True))
    fixtures+=1
   print(f'family {family} seed {seed} passed',flush=True)
 results=ROOT/'validation/results';results.mkdir(exist_ok=True);source=Path(sklearn.__file__).parent
 report=dict(profile=args.profile,fixtures=fixtures,checks=checks,max_absolute_error=max_error,sklearn=sklearn.__version__,
             numpy=np.__version__,native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
             validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             source_sha256={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in ['datasets/_samples_generator.py','utils/_random.pyx']})
 (results/f'classification_datasets_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 rejected=[]
 for case,message in REJECTIONS.items():
  run=subprocess.run([str(exe),case],text=True,capture_output=True)
  assert run.returncode!=0 and 'fscikit-learn:' in run.stdout and message in run.stdout,(case,run.stdout,run.stderr)
  rejected.append(dict(name=case,expected=message,passed=True))
 invalid=dict(profile=args.profile,checks=rejected,native_sha256=report['native_sha256'],validator_sha256=report['validator_sha256'])
 (results/f'classification_datasets_invalid_{args.profile}.json').write_text(json.dumps(invalid,indent=2,allow_nan=False)+'\n')
 print(f'{len(checks)} classification dataset groups / {fixtures} fixtures and {len(rejected)} isolated rejections passed')


if __name__=='__main__':main()
