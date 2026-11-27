"""Bit-exact objective/gradient checks at actual sklearn optimization points."""
from pathlib import Path
import sys,tempfile,json,subprocess,warnings
sys.path.insert(0,str(Path(__file__).resolve().parent))
import compare_huber_controls as v
import numpy as np
import argparse,hashlib,sklearn
ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
a.report.parent.mkdir(parents=True,exist_ok=True);records=[];wires=[]
original=v.source._huber_loss_and_gradient
for rep in range(4):
 for intercept in [0,1]:
  c=dict(seed=7,representation=rep,intercept=intercept,weights=1,profile=(1.35,.0001,1e-5,100),kind='cold',snapshots=1,scenario='wide')
  x,y,w,q=v.arrays(c,0);xx=np.asfortranarray(x) if rep==3 else v.matrix(x,rep)
  def fg(point,*args,**kwargs):
   result=original(point,*args,**kwargs)
   if rep in [0,3]:features=' '.join(format(z,'.17g') for z in x.ravel(order='F'))+'\n';stored=-2 if rep==3 else -1
   else:
    stored=xx.nnz;features=''.join(' '.join(map(str,a))+'\n' for a in [xx.indptr+1,xx.indices+1,[format(z,'.17g') for z in xx.data]])
   wire=f'{len(y)} {x.shape[1]} {len(point)} {stored} 1.35 .0001\n'+features+''.join(' '.join(format(z,'.17g') for z in a)+'\n' for a in [y,w,point])
   wires.append(wire);records.append(dict(rep=rep,intercept=intercept,point=point.tolist(),source_loss=float(result[0]),source_g=result[1].tolist()))
   return result
  v.source._huber_loss_and_gradient=fg
  with warnings.catch_warnings():
   warnings.simplefilter('ignore');v.HuberRegressor(fit_intercept=bool(intercept),max_iter=100).fit(xx,y,sample_weight=w)
v.source._huber_loss_and_gradient=original
# Separate direct checks exercise signed and zero weight objectives, including
# a near-zero scale, without substituting their values for fitted-path oracles.
for rep in range(4):
 for intercept in [0,1]:
  rng=np.random.RandomState(19);x=rng.normal(size=(13,4));y=rng.normal(size=13);xx=np.asfortranarray(x) if rep==3 else v.matrix(x,rep)
  for style in ['signed','negative','zero']:
   weight=rng.normal(size=13) if style=='signed' else -np.ones(13) if style=='negative' else np.zeros(13)
   for sigma in [.01,1.,20.]:
    point=np.r_[np.linspace(-.4,.8,4),.3,sigma] if intercept else np.r_[np.linspace(-.4,.8,4),sigma]
    loss,g=original(point,xx,y,1.35,.0001,weight)
    if rep in [0,3]:features=' '.join(format(z,'.17g') for z in x.ravel(order='F'))+'\n';stored=-2 if rep==3 else -1
    else:
     stored=xx.nnz;features=''.join(' '.join(map(str,a))+'\n' for a in [xx.indptr+1,xx.indices+1,[format(z,'.17g') for z in xx.data]])
    wire=f'{len(y)} {x.shape[1]} {len(point)} {stored} 1.35 .0001\n'+features+''.join(' '.join(format(z,'.17g') for z in a)+'\n' for a in [y,weight,point])
    wires.append(wire);records.append(dict(rep=rep,intercept=intercept,origin='direct_'+style,point=point.tolist(),source_loss=float(loss),source_g=g.tolist()))
exe=a.executable
text=str(len(wires))+'\n'+''.join(wires);a.report.with_suffix('.in').write_text(text)
p=subprocess.run([str(exe)],input=text,text=True,capture_output=True);assert p.returncode==0,p.stderr
a.report.with_suffix('.stdout').write_text(p.stdout);lines=p.stdout.splitlines();assert len(lines)==2*len(records)
for i,z in enumerate(records):
 z['native_loss']=float(lines[2*i]);z['native_g']=np.fromstring(lines[2*i+1],sep=' ').tolist();z['loss_error']=abs(z['native_loss']-z['source_loss']);z['gradient_error']=float(np.max(np.abs(np.array(z['native_g'])-z['source_g'])));z['exact']=z['native_loss']==z['source_loss'] and z['native_g']==z['source_g']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=[Path(__file__),Path('validation/compare_huber_controls.py'),exe,Path('src/sklearn_huber_solver.f90'),Path('src/sklearn_kernel_dispatch.f90'),Path('app/reference_huber_objective_driver.f90'),Path(v.source.__file__)]
d=dict(configurations=len(records),failed=sum(not x['exact'] for x in records),records=records,sklearn=sklearn.__version__,sha256={str(p):sha(p) for p in files})
a.report.write_text(json.dumps(d,indent=2)+'\n')
for rep in range(4):
 for intercept in [0,1]:
  rows=[(i,z) for i,z in enumerate(records) if z['rep']==rep and z['intercept']==intercept];bad=[(i,z) for i,z in rows if not z['exact']]
  print(rep,intercept,'points',len(rows),'inexact',len(bad),'first',[(i,z['loss_error'],z['gradient_error']) for i,z in bad[:3]],flush=True)

assert d['failed']==0,(d['failed'],d['configurations'])
