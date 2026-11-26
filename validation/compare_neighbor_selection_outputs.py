"""Additional KNN metrics, surrogate rounding and finite-input overflow probes.

Reuse the full reference readers with additional fixtures; every expected value
is still computed by the installed sklearn estimator. Original grids remain
separate and are not filtered or replaced by these probes.
"""
from pathlib import Path
import argparse,hashlib,json,sys
from unittest.mock import patch
import compare_neighbor_classification_outputs as classification
import compare_neighbor_outputs as regression
import numpy as np
from scipy.spatial.distance import cdist


def fixtures(family):
    rng=np.random.RandomState(71)
    metrics=[('euclidean',2.),('l2',2.),('sqeuclidean',2.),('manhattan',2.),
             ('cityblock',2.),('l1',2.),('chebyshev',2.),('cosine',2.),
             ('minkowski',.5),('minkowski',1.5),('minkowski',3.),
             ('minkowski',np.inf),('precomputed',2.)]
    for n in [9,65]:
        original=rng.normal(size=(n,5));targets=rng.normal(size=(n,3))
        labels=rng.randint(0,4,size=(n,3))*3-4
        for kind in ['regular','duplicate']:
            data=original.copy()
            if kind=='duplicate':data[:4]=data[0]
            query=np.vstack([data[:2],rng.normal(size=(3,5))])
            for metric,power in metrics:
                x=cdist(data,data) if metric=='precomputed' else data
                q=cdist(query,data) if metric=='precomputed' else query
                for weights in ['uniform','distance']:
                    for t in [1,3]:
                        for self_query in [False,True]:
                            for k in [1,3]:
                                c=dict(seed=71,kind=kind,x=x,q=x.copy()if self_query else q,
                                       y=(labels if family=='classification' else targets)[:,:t],
                                       metric=metric,power=power,weights=weights,t=t,self_query=self_query)
                                if family=='classification':c.update(label_kind='ragged',k=k)
                                else:c.update(family=1,selector=k)
                                yield c
    # All entries are finite: only the distance calculation overflows. Source
    # ArgKmin retains DBL_MAX and its initial index when no candidate enters.
    for metric,power in [('euclidean',2.),('l2',2.),('sqeuclidean',2.),
                         ('manhattan',2.),('chebyshev',2.),('minkowski',3.)]:
        for mixed in [False,True]:
            x=np.full((4,2),np.finfo(float).max);q=-x[:2].copy()
            if mixed:x[0]=q[0]
            for weights in ['uniform','distance']:
                c=dict(seed=71,kind='overflow_mixed'if mixed else'overflow_all',x=x,q=q,
                       y=np.array([[-4],[2],[5],[8]])if family=='classification'else np.array([[1.],[2.],[4.],[8.]]),
                       metric=metric,power=power,weights=weights,t=1,self_query=False)
                if family=='classification':c.update(label_kind='ragged',k=3)
                else:c.update(family=1,selector=3)
                yield c


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['classification','regression'],required=True)
    ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--reference-cache',type=Path);ap.add_argument('--record-differences',action='store_true')
    a=ap.parse_args();module=classification if a.family=='classification' else regression
    argv=[module.__name__,'--executable',str(a.executable),'--report',str(a.report)]
    if a.reference_cache:argv+=['--reference-cache',str(a.reference_cache)]
    if a.record_differences:argv+=['--record-differences']
    with patch.object(module,'fixtures',side_effect=lambda smoke:fixtures(a.family)),patch.object(sys,'argv',argv):module.main()
    report=json.loads(a.report.read_text());report['fixture_validator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
