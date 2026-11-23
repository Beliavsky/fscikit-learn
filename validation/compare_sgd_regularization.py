"""SGD penalties, schedules, source shuffling and streaming with sklearn 1.6.1."""
import argparse,hashlib,itertools,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import numpy as np
import sklearn
from sklearn.linear_model import SGDClassifier,SGDRegressor
from sklearn.exceptions import ConvergenceWarning
from compare_image_graph import fields
ROOT=Path(__file__).resolve().parents[1]
REJECTIONS={'ratio_negative':'SGD l1_ratio must be in [0,1]','ratio_large':'SGD l1_ratio must be in [0,1]',
 'ratio_nan':'SGD l1_ratio must be finite','optimal_alpha':'optimal SGD requires positive alpha',
 'seed_negative':'SGD random_state must be nonnegative','classifier_ratio':'SGD l1_ratio must be in [0,1]',
 'classifier_optimal':'optimal SGD requires positive alpha'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def vector(a):return ' '.join(format(v,'.17g') for v in np.asarray(a).ravel(order='F'))+'\n'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path,required=True)
 parser.add_argument('--profile',default='checked');parser.add_argument('--report',type=Path,default=ROOT/'validation/results/sgd_regularization.json')
 args=parser.parse_args();checks=[];fixtures=0
 process=subprocess.Popen([str(args.executable.resolve()),'batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
   stderr=subprocess.PIPE,text=True,bufsize=1)
 import atexit
 atexit.register(lambda:process.terminate() if process.poll() is None else None)
 def run(payload):
  process.stdin.write(payload);process.stdin.flush();output=[]
  while True:
   line=process.stdout.readline()
   assert line,process.stderr.read()
   if line.strip()=='END':return fields(''.join(output))
   output.append(line)
 warnings.simplefilter('ignore',ConvergenceWarning)
 for seed in [0,42,2026]:
  rng=np.random.RandomState(seed);x=rng.normal(scale=.2,size=(23,5));x[:,0]=0;x[::3,1:]=0
  weights=rng.uniform(.2,2,len(x));weights[0]=0
  for kind in [0,2,3]:
   y=x@rng.normal(size=5)+.3 if not kind else np.array([3+7*(i%kind) for i in range(len(x))])
   losses=['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive'] if not kind else ['hinge','squared_hinge','log_loss','modified_huber','perceptron']
   for loss,penalty,rate,average,shuffle in itertools.product(losses,['none','l1','l2','elasticnet'],['constant','invscaling','optimal','adaptive'],[0,1,10],[False,True]):
    # Main Cartesian grid covers all settings; stream a deterministic subset.
    streaming=average==10 or (average==0 and penalty=='elasticnet' and rate in ['constant','optimal'])
    intercept=not streaming or average==0;chunks=[7,1,15] if streaming else []
    alpha=.08;eta=.02;ratio=.37;tol=-1.;epochs=7
    opts=dict(loss=loss,penalty=None if penalty=='none' else penalty,learning_rate=rate,average=average if average else False,
      shuffle=shuffle,fit_intercept=intercept,random_state=seed,max_iter=epochs,n_iter_no_change=2,
      alpha=alpha,eta0=eta,power_t=.4,l1_ratio=ratio,tol=None)
    if not kind:opts['epsilon']=.15
    source=(SGDClassifier if kind else SGDRegressor)(**opts)
    header=[len(x),x.shape[1],kind,epochs,average,int(shuffle),int(intercept),seed,2,len(chunks)]
    payload=vector(header)+f'{loss} {penalty} {rate}\n'+vector([alpha,eta,.4,ratio,tol,.15])+vector(x)+vector(y)+vector(weights)
    if chunks:payload+=vector(chunks)
    actual=run(payload);trace=[];lo=0
    for length in chunks or [len(x)]:
     if chunks:
      kw=dict(classes=np.unique(y)) if kind else {}
      source.partial_fit(x[lo:lo+length],y[lo:lo+length],sample_weight=weights[lo:lo+length],**kw)
     else:source.fit(x,y,sample_weight=weights)
     coef=np.atleast_2d(source.coef_);bias=np.atleast_1d(source.intercept_)
     trace.append(np.r_[coef.ravel(order='F'),bias,source.n_iter_,source.t_]);lo+=length
    expected=[coef,bias[:,None],source.decision_function(x).reshape(len(x),-1) if kind else source.predict(x)[:,None],
      source.predict_proba(x) if kind and loss in ['log_loss','modified_huber'] else np.empty((0,0)),
      source.predict(x)[:,None] if kind else np.empty((0,1)),np.array(trace)]
    label=f'{seed}/{kind}/{loss}/{penalty}/{rate}/{average}/{shuffle}'
    for name,got,want in zip(['coefficients','intercepts','decision_or_prediction','probabilities','labels','stream_trace'],actual,expected):
     np.testing.assert_allclose(got,want,rtol=3e-9,atol=3e-10,err_msg=f'{label}/{name}')
     checks.append(dict(fixture=label,field=name,max_abs_error=float(np.max(np.abs(got-want),initial=0)),passed=True))
    fixtures+=1
   print(f'seed {seed} kind {kind}: Cartesian penalty/schedule/average/shuffle grid passed',flush=True)
 # Elastic-net endpoints, no regularization, repeated lazy-scale resets,
 # and all-zero weights, including averaged coefficient materialization.
 rng=np.random.RandomState(91);x=rng.normal(scale=.1,size=(13,4))
 extra=[('endpoints',k,p,r,a,z,'constant',.5,1,True,4,-1.) for k,p,r,a,z in
   itertools.product([0,2,3],['l1','l2','elasticnet'],[0.,1.],[0.,8.],[False,True])]
 extra += [('adaptive_updates',k,p,.37,.0001,False,'adaptive',.03,av,sh,100,.05) for k,p,av,sh in
   itertools.product([0,2,3],['none','l1','l2','elasticnet'],[0,1,10],[False,True])]
 extra += [('optimal_zero_eta',k,'elasticnet',.37,a,False,'optimal',0.,1,True,7,-1.) for k,a in
   itertools.product([0,2,3],[.02,1.,20.])]
 for group,kind,penalty,ratio,alpha,zero_weight,rate,eta,average,shuffle,epochs,tol in extra:
  y=x[:,0]-.2 if not kind else np.array([3+7*(i%kind) for i in range(len(x))])
  weights=np.zeros(len(x)) if zero_weight else np.linspace(.1,2,len(x))
  loss='squared_error' if not kind else 'log_loss'
  source=(SGDClassifier if kind else SGDRegressor)(loss=loss,penalty=None if penalty=='none' else penalty,l1_ratio=ratio,alpha=alpha,
    learning_rate=rate,eta0=eta,average=average if average else False,max_iter=epochs,tol=tol if tol>=0 else None,
    n_iter_no_change=2,shuffle=shuffle,random_state=91)
  source.fit(x,y,sample_weight=weights)
  payload=vector([13,4,kind,epochs,average,int(shuffle),1,91,2,0])+f'{loss} {penalty} {rate}\n'
  payload+=vector([alpha,eta,.25,ratio,tol,.1])+vector(x)+vector(y)+vector(weights)
  actual=run(payload);coef=np.atleast_2d(source.coef_);bias=np.atleast_1d(source.intercept_)
  expected=[coef,bias[:,None],source.decision_function(x).reshape(len(x),-1) if kind else source.predict(x)[:,None],
    source.predict_proba(x) if kind else np.empty((0,0)),source.predict(x)[:,None] if kind else np.empty((0,1)),
    np.r_[coef.ravel(order='F'),bias,source.n_iter_,source.t_][None,:]]
  label=f'{group}/{kind}/{penalty}/{ratio}/{alpha}/{zero_weight}/{average}/{shuffle}'
  if group=='adaptive_updates':assert np.any(coef!=0),label
  for name,got,want in zip(['coefficients','intercepts','decision_or_prediction','probabilities','labels','stream_trace'],actual,expected):
   np.testing.assert_allclose(got,want,rtol=3e-9,atol=3e-10,err_msg=f'{label}/{name}')
   checks.append(dict(fixture=label,field=name,max_abs_error=float(np.max(np.abs(got-want),initial=0)),passed=True))
  fixtures+=1
 # Adaptive reductions and stopping on an exactly stagnant training loss.
 for kind in [0,2,3]:
  for eta in [.1,1e-6,1e-7]:
   x=np.zeros((12,3));y=np.zeros(12) if not kind else np.array([3+7*(i%kind) for i in range(12)])
   source=(SGDClassifier if kind else SGDRegressor)(loss='squared_error' if not kind else 'hinge',
     learning_rate='adaptive',eta0=eta,alpha=.1,penalty='elasticnet',l1_ratio=.8,max_iter=100,
     n_iter_no_change=2,tol=.001,shuffle=False,fit_intercept=False)
   source.fit(x,y)
   payload=vector([12,3,kind,100,0,0,0,0,2,0])+('squared_error' if not kind else 'hinge')+' elasticnet adaptive\n'
   payload+=vector([.1,eta,.25,.8,.001,.1])+vector(x)+vector(y)+vector(np.ones(12))
   actual=run(payload);expected=np.r_[source.n_iter_,source.t_]
   np.testing.assert_array_equal(actual[-1][0,-2:],expected)
   checks.append(dict(fixture=f'adaptive_stop/{kind}/{eta}',field='clock',max_abs_error=0.,passed=True));fixtures+=1
 process.stdin.close();assert process.wait(timeout=30)==0,process.stderr.read()
 for name,message in REJECTIONS.items():
  proc=subprocess.run([str(args.executable.resolve()),name],text=True,capture_output=True)
  output=proc.stdout+proc.stderr
  assert proc.returncode!=0 and message in output and 'SIGFPE' not in output and 'Fortran runtime error' not in output,(name,output)
 src=Path(sklearn.__file__).parent
 report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,fixtures=fixtures,checks=checks,
  rejections=list(REJECTIONS),executable_sha256=sha(args.executable),validator_sha256=sha(__file__),
  native_source_sha256=sha(ROOT/'src/sklearn_sgd.f90'),
  parser_sha256=sha(ROOT/'validation/compare_image_graph.py'),
  source_sha256={f:sha(src/f) for f in ['linear_model/_stochastic_gradient.py','linear_model/_sgd_fast.pyx.tp','utils/_weight_vector.pyx.tp','utils/_seq_dataset.pyx.tp','utils/_random.pxd']})
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(f'{len(checks)} result groups across {fixtures} fixtures and {len(REJECTIONS)} diagnostics passed',flush=True)
if __name__=='__main__':main()
