"""Actual automatic BallTree graphs and brute fallback metrics/cutoffs."""
from pathlib import Path
from unittest.mock import patch
import argparse,hashlib,json,sys,warnings
import compare_functional_neighbor_graphs as reference
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def fixtures():
    rng=np.random.RandomState(174)
    metrics=['haversine','braycurtis','canberra','hamming','dice','jaccard',
             'rogerstanimoto','russellrao','sokalmichener','sokalsneath']
    for n in [9,12,65]:
        for p in [2,16]:
            original=rng.normal(size=(n,p))
            for kind in ['regular','boolean','constant']:
                data=original.copy()
                if kind=='boolean':data=(data>0).astype(float)
                if kind=='constant':data[:]=0
                for metric in metrics:
                    if metric=='haversine'and p!=2:continue
                    for fitted in [0,1]:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            model=NearestNeighbors(metric=metric,algorithm='brute'if fitted else'auto').fit(data)
                            cutoff=float(model.kneighbors(data[:1],n_neighbors=3)[0][0,-1])
                        specs=[(0,1,0.),(0,n//2,0.)]+[(1,1,r)for r in [0.,np.nextafter(cutoff,0),cutoff,np.nextafter(cutoff,np.inf),np.inf]]
                        for family,k,radius in specs:
                            for include in [0,1,2]:
                                yield dict(operation=0,seed=174,kind=kind,x=data.copy(),storage=0,family=family,k=k,
                                    radius=float(radius),include=include,metric=metric,power=2.,fit_kind=fitted,
                                    mode='connectivity'if include==2 else'distance')
    # Actual rejection probes for the newly accepted metric names.
    for n in [9,65]:
        for family in [0,1]:
            for fitted in [0,1]:
                yield dict(operation=0,seed=174,kind='invalid_haversine_width',x=np.ones((n,3)),storage=0,
                    family=family,k=1,radius=1.,include=0,metric='haversine',power=2.,fit_kind=fitted,mode='distance')
    for metric in metrics:
        for family in [0,1]:
            yield dict(operation=0,seed=174,kind='unsupported_sparse_metric',x=csr_matrix(np.ones((9,2))),storage=1,
                family=family,k=1,radius=1.,include=0,metric=metric,power=2.,fit_kind=0,mode='distance')


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
