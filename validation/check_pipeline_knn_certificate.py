"""Regression for exact Source distances under squared-norm cancellation."""
from pathlib import Path
import argparse,hashlib,json,os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from pipeline_knn_certificate import certify

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();records=[]
 x=np.array([[1.,1.],[1.0001,1.],[1.,1.0001],[0.,0.]])
 q=np.array([[1.00001,1.00002]])
 for classifier in [False,True]:
  y=np.array([0,1,1,0]) if classifier else np.array([0.,1.,2.,3.])
  model=(KNeighborsClassifier if classifier else KNeighborsRegressor)(n_neighbors=3,weights='distance',algorithm='brute').fit(x,y)
  distance,index=model.kneighbors(q)
  _,_,w=certify(model,x,y,[q],[q],[(distance,index+1)],y,classifier)
  assert w[0]['exact_source_native_distances']
  assert not w[0]['legacy_native_relative_bound_passed'],'fixture must expose the invalid relative bound'
  corrupted=distance.copy();corrupted[0,0]=np.nextafter(corrupted[0,0],np.inf)
  rejected=False
  try:certify(model,x,y,[q],[q],[(corrupted,index+1)],y,classifier)
  except AssertionError:rejected=True
  assert rejected,'exact Source check must reject even a one-ULP mutation'
  records.append(dict(classifier=classifier,passed=True,cancellation_accepted=True,one_ulp_rejected=True,witness=w[0]))
 report=dict(sklearn=sklearn.__version__,certificate_sha256=hashlib.sha256(Path(__file__).with_name('pipeline_knn_certificate.py').read_bytes()).hexdigest(),records=records)
 a.report.write_text(json.dumps(report,indent=2)+'\n');print('Pipeline KNN certificate: Source cancellation accepted, one-ULP mutations rejected')
if __name__=='__main__':main()
