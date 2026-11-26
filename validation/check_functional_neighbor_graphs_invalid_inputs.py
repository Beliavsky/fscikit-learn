"""Live sklearn functional graph/sort rejections and native CSR safety contracts."""
from pathlib import Path
import argparse,json,os,subprocess,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors,kneighbors_graph,radius_neighbors_graph,sort_graph_by_row_values
from compare_functional_neighbor_graphs import csr_text

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--executable',required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();cases=[]
 x=np.array([[0.],[1.],[3.]])
 def graph_case(name,family=0,k=1,radius=1.,metric='minkowski',power=2.,mode='connectivity',include=0,data=None,sparse=False,fitted=False,fit_metric=None,fit_power=None,unfitted=False):
  data=x.copy()if data is None else data.copy();n,p=data.shape;storage=int(sparse)+(2 if metric=='precomputed'else 0)
  if sparse and not hasattr(data,'indptr'):data=csr_matrix(data)
  fm=metric if fit_metric is None else fit_metric;fp=power if fit_power is None else fit_power;fit_kind=-1 if unfitted else int(fitted)
  text=['0',f'{n} {p} {storage} {family} {k} {radius} {include} {fit_kind} 0',f'{mode} {metric} {power} {fm} {fp}']
  text+=csr_text(data)if sparse else[' '.join(map(str,row))for row in data]
  def source():
   obj=data
   if fitted or unfitted:
    obj=NearestNeighbors(metric=fm,p=fp,algorithm='brute')
    if not unfitted:obj.fit(data)
   kw=dict(mode=mode,metric=metric,p=power,include_self=[False,True,'auto'][include])
   return kneighbors_graph(obj,k,**kw)if family==0 else radius_neighbors_graph(obj,radius,**kw)
  cases.append((name,'\n'.join(text)+'\n',source))
 for family in [0,1]:
  for metric in ['invalid']:graph_case(f'{family}_metric',family,metric=metric)
  graph_case(f'{family}_mode',family,mode='invalid')
  graph_case(f'{family}_p_zero',family,power=0)
  graph_case(f'{family}_p_nan',family,power=np.nan)
  bad=x.copy();bad[0,0]=np.nan;graph_case(f'{family}_nan_input',family,data=bad)
  bad=x.copy();bad[0,0]=np.inf;graph_case(f'{family}_infinite_input',family,data=bad)
  graph_case(f'{family}_empty',family,data=x[:0])
  graph_case(f'{family}_zero_features',family,data=np.empty((3,0)))
  graph_case(f'{family}_nonsquare_precomputed',family,metric='precomputed')
  bad=np.abs(x-x.T);bad[0,1]=-1;graph_case(f'{family}_negative_precomputed',family,metric='precomputed',data=bad)
  graph_case(f'{family}_negative_sparse_precomputed',family,metric='precomputed',data=bad,sparse=True)
  graph_case(f'{family}_sparse_metric',family,metric='chebyshev',sparse=True)
  graph_case(f'{family}_sparse_fractional_p',family,power=.5,sparse=True)
  graph_case(f'{family}_fitted_metric',family,fitted=True,fit_metric='euclidean')
  graph_case(f'{family}_fitted_p',family,fitted=True,fit_power=1.)
  graph_case(f'{family}_unfitted',family,unfitted=True)
 for k in [0,-1,2147483647]:graph_case(f'k_{k}',k=k)
 graph_case('k_self_limit',k=3)
 graph_case('k_explicit_limit',k=4,include=1)
 graph_case('sparse_stored_count',k=2,include=1,metric='precomputed',data=csr_matrix(np.eye(3)),sparse=True)
 for radius in[-1.,np.nan]:graph_case(f'radius_{radius}',family=1,radius=radius)
 for nnz in [0,1]:
  graph=csr_matrix(([1.]*nnz,[0]*nnz,[0,0,nnz]),shape=(2,2))
  cases.append((f'sort_source_empty_boundary_{nnz}','\n'.join(['1','2 2 0 0',*csr_text(graph)])+'\n',lambda graph=graph:sort_graph_by_row_values(graph)))
 # Do not pass malformed CSR column/pointer buffers into SciPy compiled kernels.
 for name,ptr,cols,data in [('pointer',[0,1,2],[1],[0.]),('column',[1,2,2],[3],[0.])]:
  request='\n'.join(['1','2 2 0 0','1',' '.join(map(str,ptr)),' '.join(map(str,cols)),' '.join(map(str,data))])+'\n'
  cases.append(('native_csr_'+name,request,None))
 checks=[]
 for name,request,source in cases:
  diagnostic=None
  if source is not None:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    try:source()
    except (ValueError,TypeError,IndexError,AttributeError)as error:diagnostic=type(error).__name__+': '+str(error)
    else:raise AssertionError(('Source accepted',name))
  proc=subprocess.run([a.executable],input=request,text=True,capture_output=True);text=proc.stdout+proc.stderr
  assert proc.returncode and 'fscikit-learn:'in text and not any(s in text for s in['SIGFPE','SIGSEGV','Fortran runtime error']),(name,text)
  checks.append(dict(name=name,source_rejection=diagnostic,native_diagnostic=next(s.strip()for s in text.splitlines()if 'fscikit-learn:'in s)))
 report=dict(count=len(checks),source_rejections=sum(c['source_rejection']is not None for c in checks),native_contracts=sum(c['source_rejection']is None for c in checks),checks=checks);a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
