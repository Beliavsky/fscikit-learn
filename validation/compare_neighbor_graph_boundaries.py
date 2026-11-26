"""Actual free/fitted radius graphs at and adjacent to computed cutoffs."""
from pathlib import Path
from unittest.mock import patch
import argparse,hashlib,json,sys,warnings
import compare_functional_neighbor_graphs as reference
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def fixtures():
    rng=np.random.RandomState(89)
    metrics=[('euclidean',2.),('l2',2.),('sqeuclidean',2.),('manhattan',2.),('cityblock',2.),
             ('l1',2.),('chebyshev',2.),('cosine',2.),('minkowski',.5),('minkowski',1.5),
             ('minkowski',3.),('minkowski',np.inf),('precomputed',2.)]
    for n in [9,33,65]:
        for p in [3,16]:
            original=rng.normal(size=(n,p))
            for kind in ['regular','duplicate']:
                data=original.copy()
                if kind=='duplicate':data[:3]=data[0]
                for metric,power in metrics:
                    for sparse in [False,True]:
                        if sparse and metric not in ['euclidean','l2','manhattan','cityblock','l1','cosine','precomputed']:continue
                        x=cdist(data,data)if metric=='precomputed'else data.copy()
                        if sparse:x=csr_matrix(x)
                        for fitted in ([0]if sparse else[0,1]):
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                model=NearestNeighbors(metric=metric,p=power,algorithm='brute'if fitted else'auto').fit(x)
                                cutoff=float(model.kneighbors(x[:1],n_neighbors=3)[0][0,-1])
                            for radius in [np.nextafter(cutoff,0),cutoff,np.nextafter(cutoff,np.inf)]:
                                yield dict(operation=0,seed=89,kind=kind,x=x.copy(),storage=int(sparse)+(2 if metric=='precomputed'else 0),
                                           family=1,k=1,radius=float(radius),include=0,metric=metric,power=power,fit_kind=fitted,mode='distance')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--reference-cache',type=Path);ap.add_argument('--record-differences',action='store_true');a=ap.parse_args()
    argv=[reference.__name__,'--executable',str(a.executable),'--report',str(a.report)]
    if a.reference_cache:argv+=['--reference-cache',str(a.reference_cache)]
    if a.record_differences:argv+=['--record-differences']
    with patch.object(reference,'graph_cases',side_effect=lambda smoke:fixtures()),patch.object(reference,'sort_cases',return_value=iter([])),patch.object(sys,'argv',argv):reference.main()
    report=json.loads(a.report.read_text());report['fixture_validator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
