"""Dense HDBSCAN brute fits, hierarchy/selection stages and finite-row remapping."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import warnings
import tempfile
import hashlib
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import sklearn
from sklearn.cluster import HDBSCAN
from sklearn.cluster._hdbscan import _tree
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import squareform
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'validation/results'
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def canonical(labels):
    labels=np.asarray(labels,dtype=int).copy(); mapping={}
    for i,value in enumerate(labels):
        if value<0: continue
        if value not in mapping: mapping[value]=len(mapping)
        labels[i]=mapping[value]
    return labels


def hierarchy(children,distances,sizes):
    result=np.zeros(len(distances),dtype=_tree.HIERARCHY_dtype)
    result['left_node']=children[:,0]; result['right_node']=children[:,1]
    result['value']=distances; result['cluster_size']=sizes
    return result


def finite_hierarchy(raw,finite,n_raw):
    nf=len(finite); inverse={raw:i for i,raw in enumerate(finite)}
    result=raw[:nf-1].copy()
    for name in ('left_node','right_node'):
        result[name]=[inverse[v] if v<n_raw else v-n_raw+nf for v in result[name]]
    return result


def linkage_array(tree):
    return np.column_stack([tree[name] for name in ('left_node','right_node','value','cluster_size')])


def stability(tree):
    root=tree['parent'].min(); last=tree['parent'].max(); birth={root:0.}
    birth.update(zip(tree['child'],tree['value'])); result=np.zeros(last-root+1)
    with np.errstate(invalid='ignore'):
        for row in tree:
            result[row['parent']-root]+=(row['value']-birth[row['parent']])*row['cluster_size']
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--executable',type=Path); parser.add_argument('--report',type=Path); args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable: executable=args.executable.resolve()
    else:
        proc=subprocess.run(['fpm','run','--flag',FLAGS,'--target','reference_hdbscan_driver','--runner'],
                            cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if proc.returncode: raise RuntimeError(proc.stdout)
        paths=[ROOT/s.strip().strip(chr(34)) for s in proc.stdout.splitlines()
               if s.strip().strip(chr(34)).endswith(('reference_hdbscan_driver','reference_hdbscan_driver.exe'))]
        paths=[p for p in paths if p.is_file()]; assert len(paths)==1,proc.stdout
        executable=paths[0].resolve()
    work=tempfile.TemporaryDirectory(prefix='fsklearn-hdbscan-dense-reference-')
    before=hashlib.sha256(executable.read_bytes()).hexdigest()
    checks=[]; direct_matches=[]
    for seed,n,p in ((7,72,2),(42,89,3),(0,96,4),(2026,65,2)):
        rng=np.random.default_rng(seed); groups=np.arange(n)%3
        original=rng.normal(size=(n,p))*np.array([.12,.4,.9])[groups,None]
        original+=np.array([-5.,0.,6.])[groups,None]*np.linspace(.5,1.2,p)
        original[:2]+=12
        for case in range(30):
            x=original.copy(); options=dict(min_cluster_size=5,algorithm='brute',copy=True)
            store='both'; operation=2 if case==0 else 1; cut_min=3
            if case==1: options['cluster_selection_method']='leaf'
            if case==2: options.update(min_cluster_size=8,min_samples=3)
            if case==3: options['min_samples']=1
            if case==4: x=rng.normal(size=(n,p)); options['allow_single_cluster']=True
            if case==5: x=rng.normal(size=(n,p)); options.update(allow_single_cluster=True,cluster_selection_method='leaf')
            if case==6: options['cluster_selection_epsilon']=.5
            if case==7: options.update(cluster_selection_method='leaf',cluster_selection_epsilon=.7)
            if case==8: options['max_cluster_size']=12
            if case==9: options.update(max_cluster_size=12,cluster_selection_epsilon=3.)
            if case==10: options['alpha']=2.1
            if case==11: options.update(alpha=.7,cluster_selection_epsilon=.8)
            if case==12: options.update(metric='minkowski',metric_params=dict(p=2.5))
            if case==13: options['metric']='manhattan'
            if case==14: options['metric']='chebyshev'
            if case==15: options['metric']='cosine'; x[0]=0
            if case==16: options['metric']='sqeuclidean'
            if case in (17,18,19,26):
                x=pairwise_distances(x); options['metric']='precomputed'; store='none'
            if case==17: options['alpha']=1.7
            if case==18: np.fill_diagonal(x,.05); options['min_samples']=1
            if case==19: options.update(allow_single_cluster=True,cluster_selection_method='leaf',cluster_selection_epsilon=.4)
            if case==20: options.update(min_samples=n,allow_single_cluster=True)
            if case==21: options.update(min_cluster_size=n+10,min_samples=2,allow_single_cluster=True)
            if case in (22,23):
                x=np.repeat(np.array([-4.,0.,5.])[groups,None],p,axis=1)
                if case==23: options['cluster_selection_method']='leaf'
            if case in (24,25):
                x[0,0]=np.inf; x[1,0]=np.nan; x[2,0]=np.inf; x[2,1]=np.nan
                x[3,0]=np.inf; x[3,1]=-np.inf
                if case==24: store='none'
            if case==26: x[groups[:,None]!=groups]=np.inf
            if case==27: cut_min=1; options['min_samples']=1
            if case==28: options.update(min_cluster_size=n-1,min_samples=1)
            if case==29: options.update(cluster_selection_method='leaf',allow_single_cluster=True,cluster_selection_epsilon=50.)
            metric=options.get('metric','euclidean'); power=options.get('metric_params',{}).get('p',2.)
            finite=np.arange(n) if metric=='precomputed' else np.flatnonzero(np.isfinite(x).all(axis=1))
            marker=np.zeros(n,dtype=int)
            if metric!='precomputed':
                with np.errstate(invalid='ignore'):
                    reduced=x.sum(axis=1)
                marker[np.isinf(reduced)]=-2; marker[np.isnan(reduced)]=-3
            nf=len(finite); cuts=np.array([0.,.2,.7,3.,10.,np.inf])
            data,output=Path(work.name)/'input.txt',Path(work.name)/'output.txt'
            with data.open('w') as stream:
                stream.write(f'{n} {x.shape[1]} {len(cuts)} {cut_min} {operation}\n')
                stream.write(f"{options['min_cluster_size']} {options.get('min_samples',0)} "
                    f"{options.get('max_cluster_size',0)} {options.get('alpha',1.)} "
                    f"{options.get('cluster_selection_epsilon',0.)} {power} {options.get('allow_single_cluster',False)}\n")
                stream.write(f"{metric} {options.get('cluster_selection_method','eom')} {store}\n")
                np.savetxt(stream,x,fmt='%.17g'); np.savetxt(stream,cuts[None,:],fmt='%.17g')
            subprocess.run([str(executable),str(data),str(output)],cwd=ROOT,check=True)
            actual={}
            for line in output.read_text().splitlines():
                name,*values=line.split(); actual[name]=np.array(values,dtype=float)
            with warnings.catch_warnings(),np.errstate(invalid='ignore',divide='ignore'):
                warnings.simplefilter('ignore',UserWarning)
                source=HDBSCAN(**options).fit(x)
            raw_children=actual['children'].reshape((n-1,2),order='F').astype(int)-1
            node_sizes=np.ones(2*n-1,dtype=int)
            for i,(left,right) in enumerate(raw_children):
                assert left<n+i and right<n+i and left!=right,(seed,case,'invalid linkage children')
                node_sizes[n+i]=node_sizes[left]+node_sizes[right]
            np.testing.assert_array_equal(actual['sizes'],node_sizes[n:])
            native_raw=hierarchy(raw_children,actual['distances'],actual['sizes'].astype(int))
            native=finite_hierarchy(native_raw,finite,n)
            reference=finite_hierarchy(source._single_linkage_tree_,finite,n)
            condensed=_tree._condense_tree(native,options['min_cluster_size'])
            selection=dict(min_cluster_size=options['min_cluster_size'],
                cluster_selection_method=options.get('cluster_selection_method','eom'),
                allow_single_cluster=options.get('allow_single_cluster',False),
                cluster_selection_epsilon=options.get('cluster_selection_epsilon',0.),max_cluster_size=options.get('max_cluster_size'))
            with warnings.catch_warnings(),np.errstate(invalid='ignore',divide='ignore'):
                warnings.simplefilter('ignore',DeprecationWarning)
                labels,probabilities=_tree.tree_to_labels(native,**selection)
            expected_labels=marker.copy(); expected_labels[finite]=labels
            expected_probabilities=np.zeros(n); expected_probabilities[finite]=probabilities; expected_probabilities[marker==-3]=np.nan
            equal=np.array_equal(canonical(actual['labels']),canonical(source.labels_))
            same_prob=np.allclose(actual['probabilities'],source.probabilities_,atol=3e-8,rtol=3e-8,equal_nan=True)
            if not(equal and same_prob):
                # Equal-height edge orders can choose different zero-lifetime branches.
                assert len(np.unique(reference['value']))<len(reference),(seed,case,'no tie to explain partition difference')
            direct_matches.append(dict(seed=seed,case=case,matched=equal and same_prob,labels_matched=equal,probabilities_matched=same_prob))
            kw=options.get('metric_params',{})
            d=x.copy() if metric=='precomputed' else pairwise_distances(x[finite],metric=metric,**kw)
            d=d/options.get('alpha',1.); k=options.get('min_samples',options['min_cluster_size'])
            core=np.partition(d,k-1,axis=1)[:,k-1]; mutual=np.maximum(np.maximum(d,core[:,None]),core[None,:])
            edges=actual['mst_edges'].reshape((nf-1,2),order='F').astype(int)-1
            inverse={raw:i for i,raw in enumerate(finite)}; internal=np.array([[inverse[v] for v in edge] for edge in edges])
            assert len(set(np.r_[internal[0,0],internal[:,1]]))==nf
            assert np.array_equal(internal[1:,0],internal[:-1,1])
            keys=np.full(nf,np.inf); seen=np.zeros(nf,dtype=bool); frontier=[]
            for left,right in internal:
                seen[left]=True; keys=np.minimum(keys,mutual[left]); minimum=keys[~seen].min()
                assert keys[right]<=minimum+3e-8*(1+abs(minimum))
                frontier.append(minimum)
            expected=dict(labels=canonical(expected_labels),probabilities=expected_probabilities,finite=finite+1,core=core,
                mst_distances=frontier,condensed_parents=condensed['parent']+1,condensed_children=condensed['child']+1,
                condensed_sizes=condensed['cluster_size'],condensed_lambdas=condensed['value'],stabilities=stability(condensed),
                meta=[x.shape[1],len(set(labels)-{-1}),int(np.isinf(reference['value']).any())])
            center_order=list(dict.fromkeys(int(v) for v in actual['labels'] if v>=0))
            for center_name in ('centroids','medoids'):
                if center_name in actual:
                    actual[center_name]=actual[center_name].reshape((len(center_order),x.shape[1]),order='F')[center_order].ravel(order='F')
            actual['labels']=canonical(actual['labels'])
            expected['hierarchy_cophenetic']=cophenet(linkage_array(reference))
            actual['hierarchy_cophenetic']=cophenet(linkage_array(native))
            expected['hierarchy_heights']=reference['value']; actual['hierarchy_heights']=native['value']
            chosen=actual['selected'].astype(int)-1; mapping={c:i for i,c in enumerate(chosen)}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',DeprecationWarning)
                via_chosen=_tree._do_labelling(condensed,set(chosen),mapping,selection['allow_single_cluster'],selection['cluster_selection_epsilon'])
            expected['selected_labelling']=canonical(labels); actual['selected_labelling']=canonical(via_chosen)
            if operation==2: expected['fit_predict']=canonical(expected_labels); actual['fit_predict']=canonical(actual['fit_predict'])
            if store!='none':
                labs=canonical(labels); weights=probabilities; centers=[]; medoids=[]
                for c in range(len(set(labs)-{-1})):
                    mask=labs==c; points=x[finite][mask]; w=weights[mask]
                    centers.append(np.average(points,axis=0,weights=w))
                    costs=pairwise_distances(points,metric=metric,**kw)@w
                    medoids.append(points[np.argmin(costs)])
                expected['centroids']=np.array(centers).reshape((-1,x.shape[1]))
                expected['medoids']=np.array(medoids).reshape((-1,x.shape[1]))
            for i,cut in enumerate(cuts,1):
                lab=_tree.labelling_at_cut(native_raw,float(cut),cut_min); lab[marker<0]=marker[marker<0]
                expected[f'cut{i}']=canonical(lab); actual[f'cut{i}']=canonical(actual[f'cut{i}'])
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    direct=source.dbscan_clustering(float(cut),min_cluster_size=cut_min)
                assert np.array_equal(actual[f'cut{i}'],canonical(direct)),(seed,case,cut,'direct DBSCAN cut')
            for name,value in expected.items():
                tol=0 if name in ('labels','finite','meta','selected_labelling','fit_predict','condensed_parents',
                                 'condensed_children','condensed_sizes') or name.startswith('cut') else 3e-8
                np.testing.assert_allclose(actual[name],np.asarray(value).ravel(order='F'),atol=tol,rtol=tol,
                                           err_msg=f'seed={seed} case={case} {name}')
                checks.append(dict(seed=seed,case=case,name=name,passed=True))
        print(f'seed {seed}: 30 configurations passed',flush=True)
    assert before==hashlib.sha256(executable.read_bytes()).hexdigest()
    work.cleanup()
    (args.report or RESULTS/'hdbscan.json').write_text(json.dumps(dict(sklearn=sklearn.__version__,numpy=np.__version__,executable_sha256=before,checks=checks,direct_matches=direct_matches),indent=2)+'\n')
    print(f'{len(checks)} HDBSCAN result groups passed; {sum(c["matched"] for c in direct_matches)}/{len(direct_matches)} direct fits')


if __name__=='__main__': main()
