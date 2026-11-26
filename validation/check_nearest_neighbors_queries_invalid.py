"""Actual constructor/query rejections for dense NearestNeighbors."""
from pathlib import Path
import argparse,hashlib,json,subprocess,os,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np,sklearn
from sklearn.neighbors import NearestNeighbors

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
 x=np.arange(8,dtype=float).reshape(4,2,order='F');q=np.zeros((1,2));cases=[]
 options={'neighbors_zero':dict(n_neighbors=0),'neighbors_negative':dict(n_neighbors=-1),'leaf_zero':dict(leaf_size=0),'leaf_negative':dict(leaf_size=-1),'p_zero':dict(p=0),'p_negative':dict(p=-1),'p_nan':dict(p=np.nan),'algorithm':dict(algorithm='invalid'),'metric':dict(metric='invalid'),'brute_p_alias':dict(algorithm='brute',metric='p'),'brute_infinity_alias':dict(algorithm='brute',metric='infinity'),'kd_cosine':dict(algorithm='kd_tree',metric='cosine'),'ball_cosine':dict(algorithm='ball_tree',metric='cosine'),'kd_fractional':dict(algorithm='kd_tree',p=.5),'ball_fractional':dict(algorithm='ball_tree',p=.5),'radius_negative':dict(radius=-1),'radius_nan':dict(radius=np.nan),'fit_precomputed_width':dict(metric='precomputed')}
 for name,kw in options.items():cases.append((name,lambda kw=kw:NearestNeighbors(**kw).fit(x)))
 for name,data in [('fit_empty',x[:0]),('fit_zero_width',x[:,:0]),('fit_nan',np.full_like(x,np.nan)),('fit_infinity',np.full_like(x,np.inf))]:cases.append((name,lambda data=data:NearestNeighbors().fit(data)))
 bad=np.zeros((4,4));bad[0,1]=-1;cases.append(('fit_precomputed_negative',lambda:NearestNeighbors(metric='precomputed').fit(bad)))
 for name,call in [('unfitted_knn',lambda m:m.kneighbors(q)),('unfitted_radius',lambda m:m.radius_neighbors_graph(q)),('unfitted_graph',lambda m:m.kneighbors_graph(q))]:cases.append((name,lambda call=call:call(NearestNeighbors())))
 for name,k in [('query_zero',0),('query_negative',-1),('query_limit',5)]:cases.append((name,lambda k=k:NearestNeighbors().fit(x).kneighbors(q,n_neighbors=k)))
 cases.append(('query_self_limit',lambda:NearestNeighbors().fit(x).kneighbors(n_neighbors=4)))
 for name,data in [('query_width',np.zeros((1,3))),('query_empty',q[:0]),('query_nan',np.full_like(q,np.nan)),('query_infinity',np.full_like(q,np.inf))]:cases.append((name,lambda data=data:NearestNeighbors().fit(x).kneighbors(data,n_neighbors=1)))
 cases.extend([('knn_mode',lambda:NearestNeighbors(n_neighbors=1).fit(x).kneighbors_graph(q,mode='invalid')),('radius_mode',lambda:NearestNeighbors().fit(x).radius_neighbors_graph(q,mode='invalid')),('brute_negative_radius',lambda:NearestNeighbors(algorithm='brute').fit(x).radius_neighbors_graph(q,radius=-1))])
 checks=[]
 for name,source in cases:
  with warnings.catch_warnings():
   warnings.simplefilter('ignore')
   try:source()
   except (ValueError,TypeError,AttributeError)as error:message=type(error).__name__+': '+str(error)
   else:raise AssertionError((name,'Source accepted invalid case'))
  p=subprocess.run([str(a.executable),name],text=True,capture_output=True);text=p.stdout+p.stderr
  assert p.returncode!=0 and 'fscikit-learn:'in text and not any(s in text for s in ['SIGFPE','SIGSEGV','Fortran runtime error','invalid accepted']),(name,p.returncode,text)
  checks.append(dict(case=name,source_rejection=message,passed=True))
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();report=dict(count=len(checks),checks=checks,sklearn=sklearn.__version__,validator_sha256=sha(__file__),executable_sha256=sha(a.executable))
 a.report.write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(checks),'Source-matched rejections')
if __name__=='__main__':main()
