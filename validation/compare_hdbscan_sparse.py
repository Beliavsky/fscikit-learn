"""CSR precomputed HDBSCAN: direct MST, hierarchy, condensation and extraction."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,tempfile,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import scipy
import sklearn
from scipy.sparse import csr_matrix
from sklearn.cluster import HDBSCAN
from sklearn.cluster._hdbscan.hdbscan import _brute_mst
from sklearn.cluster._hdbscan._reachability import mutual_reachability_graph
from sklearn.cluster._hdbscan import _tree
from sklearn.metrics import pairwise_distances
from compare_hdbscan import canonical,stability

def fixtures():
    for seed,n in [(0,19),(7,47),(42,71),(2026,129),(314159,257),(123,513)]:
        rng=np.random.RandomState(seed)
        points=rng.normal(size=(n,3));points[n//2:,0]+=5
        distances=pairwise_distances(points);np.fill_diagonal(distances,0)
        for case in range(38):
            if n>129 and case not in [0,1,9,14,17,19,28,31,34,35,36,37]:continue
            d=distances.copy();options=dict(min_cluster_size=3,min_samples=3,algorithm='brute',metric='precomputed',copy=True)
            if case in [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29]:
                neighbors=np.argsort(d,axis=1)[:,:min(8,n)];mask=np.zeros_like(d,dtype=bool);mask[np.arange(n)[:,None],neighbors]=True;mask|=mask.T
                # Keep the two clouds connected without inventing missing distances.
                for i in range(n-1):mask[i,i+1]=mask[i+1,i]=True
            else:mask=np.ones_like(d,dtype=bool)
            np.fill_diagonal(mask,case%3==0)
            if case==2:options.update(min_samples=1)
            if case==3:options.update(min_samples=2)
            if case==4:options.update(min_samples=n-1)
            if case==5:options.update(min_samples=5)
            if case==6:options.update(min_samples=None,min_cluster_size=4)
            if case==7:options.update(min_samples=2,min_cluster_size=7)
            if case==8:options.update(alpha=.3)
            if case==9:options.update(alpha=3.7)
            if case==10:options.update(cluster_selection_method='leaf')
            if case==11:options.update(cluster_selection_method='leaf',allow_single_cluster=True)
            if case==12:options.update(allow_single_cluster=True)
            if case==13:options.update(cluster_selection_epsilon=1.2)
            if case==14:options.update(cluster_selection_epsilon=10.,allow_single_cluster=True)
            if case==15:options.update(max_cluster_size=6)
            if case==16:options.update(max_cluster_size=n,allow_single_cluster=True)
            if case==17:options.update(cluster_selection_method='leaf',cluster_selection_epsilon=.7)
            if case==18:d=np.round(d,0);d[(d==0)&~np.eye(n,dtype=bool)]=.25
            if case==19:d=np.where(np.eye(n,dtype=bool),0.,1.)
            if case==20:options.update(alpha=.7,metric_params=dict(max_distance=12.))
            if case==21:options.update(metric_params=dict(max_distance=-2.))
            if case==22:options.update(metric_params=dict(max_distance=float('nan')))
            if case==23:options.update(min_cluster_size=n+1,allow_single_cluster=True)
            if case==24:d*=1.e-150
            if case==25:d*=1.e150
            if case==26:options.update(cluster_selection_epsilon=1.e-150)
            if case==27:options.update(min_samples=1,allow_single_cluster=True)
            if case==28:options.update(alpha=1.e-310,metric_params=dict(max_distance=10.))
            if case==29:options.update(alpha=1.e-310,metric_params=dict(max_distance=0.))
            if case==30:
                options.update(alpha=1.e-310,metric_params=dict(max_distance=10.));np.fill_diagonal(mask,True)
            if case==31:
                options.update(alpha=1.e-310,min_samples=n,metric_params=dict(max_distance=10.));mask[:]=True
            if case==32:options.update(alpha=1.e-310,metric_params=dict(max_distance=float('inf')))
            if case==33:options.update(alpha=3.7,metric_params=dict(max_distance=1.));np.fill_diagonal(mask,True)
            if case==35:d[0,1]*=1+5.e-8
            if case==36:d*=1.e-310
            if case==37:options.update(min_samples=n);mask[:]=True
            pointers=[0];indices=[];values=[]
            for row in range(n):
                entries=[]
                for col in np.flatnonzero(mask[row]):
                    value=d[row,col]
                    entries.extend([(col,-value),(col,2*value)] if case==34 else [(col,value*.25),(col,value*.75)] if case in [16,17,18,19] else [(col,value)])
                if case in [14,15,16,17]:entries.reverse()
                indices.extend(col for col,value in entries);values.extend(value for col,value in entries);pointers.append(len(indices))
            x=csr_matrix((np.array(values),np.array(indices,dtype=np.int32),np.array(pointers,dtype=np.int32)),shape=(n,n))
            yield seed,case,x,options

def write_input(path,x,options,cuts,operation):
    n=x.shape[0];max_distance=options.get('metric_params',{}).get('max_distance',0.)
    lines=[f'{n} {n} {len(cuts)} 2 {operation}',f"{options['min_cluster_size']} {options['min_samples'] or 0} {options.get('max_cluster_size',0)} {options.get('alpha',1.)} {options.get('cluster_selection_epsilon',0.)} 2 {options.get('allow_single_cluster',False)}",f"precomputed {options.get('cluster_selection_method','eom')} none",str(max_distance),str(x.nnz),' '.join(str(v+1) for v in x.indptr),' '.join(str(v+1) for v in x.indices),' '.join(format(v,'.17g') for v in x.data),' '.join(format(v,'.17g') for v in cuts)]
    path.write_text('\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();exe=a.executable.resolve();digest=hashlib.sha256(exe.read_bytes()).hexdigest();checks=[];records=[]
    with tempfile.TemporaryDirectory(prefix='fsklearn-hdbscan-csr-reference-') as work:
        work=Path(work)
        for seed,case,x,options in fixtures():
            graph=x.copy();graph.sum_duplicates();k=options['min_samples'] or options['min_cluster_size']
            with warnings.catch_warnings(),np.errstate(all='ignore'):
                warnings.simplefilter('ignore')
                source=HDBSCAN(**options).fit(x.copy())
                graph/=options.get('alpha',1.)
                core=np.array([np.partition(graph.data[graph.indptr[i]:graph.indptr[i+1]],k-1)[k-1] for i in range(x.shape[0])])
                mutual_reachability_graph(graph,min_samples=k,max_distance=options.get('metric_params',{}).get('max_distance',0.))
                mst=_brute_mst(graph,k);tree=source._single_linkage_tree_;condensed=_tree._condense_tree(tree,options['min_cluster_size'])
                cuts=np.r_[0.,np.quantile(tree['value'],[.2,.5,.8]) if np.isfinite(tree['value']).all() else [1.,10.,100.],np.inf]
                operation=2 if case%2 else 1;inp=work/'input.txt';out=work/'output.txt';write_input(inp,x,options,cuts,operation)
                result=subprocess.run([str(exe),str(inp),str(out),'csr'],capture_output=True,text=True);assert result.returncode==0,(seed,case,result.stdout,result.stderr)
                actual={name:np.array(values,float) for name,*values in map(str.split,out.read_text().splitlines())}
                expected=dict(labels=canonical(source.labels_),probabilities=source.probabilities_,children=np.column_stack([tree['left_node'],tree['right_node']])+1,distances=tree['value'],sizes=tree['cluster_size'],finite=np.arange(x.shape[0])+1,core=core,mst_edges=np.column_stack([mst['current_node'],mst['next_node']])+1,mst_distances=mst['distance'],condensed_parents=condensed['parent']+1,condensed_children=condensed['child']+1,condensed_sizes=condensed['cluster_size'],condensed_lambdas=condensed['value'],stabilities=stability(condensed),meta=[x.shape[1],len(set(source.labels_)-{-1}),int(np.isinf(mst['distance']).any())])
                actual['labels']=canonical(actual['labels'])
                if operation==2:expected['fit_predict']=canonical(source.labels_);actual['fit_predict']=canonical(actual['fit_predict'])
                for i,cut in enumerate(cuts,1):expected[f'cut{i}']=canonical(source.dbscan_clustering(float(cut),min_cluster_size=2));actual[f'cut{i}']=canonical(actual[f'cut{i}'])
                chosen=actual['selected'].astype(int)-1
                mapping={c:i for i,c in enumerate(chosen)}
                via_chosen=_tree._do_labelling(condensed,set(chosen),mapping,options.get('allow_single_cluster',False),options.get('cluster_selection_epsilon',0.))
                np.testing.assert_array_equal(canonical(via_chosen),canonical(source.labels_))
                checks.append(dict(seed=seed,case=case,name='selected_labelling',passed=True))
                assert actual.keys()==expected.keys()|{'selected'}
                for name,value in expected.items():
                    exact=name in ['labels','children','sizes','finite','core','distances','mst_distances','mst_edges','condensed_parents','condensed_children','condensed_sizes','meta','fit_predict'] or name.startswith('cut')
                    np.testing.assert_allclose(actual[name],np.asarray(value).ravel(order='F'),atol=0 if exact else 3.e-8,rtol=0 if exact else 3.e-8,equal_nan=True,err_msg=str((seed,case,name)))
                    checks.append(dict(seed=seed,case=case,name=name,passed=True))
                records.append(dict(seed=seed,case=case,stored=x.nnz,passed=True))
            if case==37:print(f'CSR HDBSCAN seed {seed}: configurations passed',flush=True)
    assert digest==hashlib.sha256(exe.read_bytes()).hexdigest()
    source_root=Path(sklearn.__file__).parent
    source_files=['cluster/_hdbscan/hdbscan.py','cluster/_hdbscan/_reachability.pyx','cluster/_hdbscan/_linkage.pyx','cluster/_hdbscan/_tree.pyx']
    a.report.write_text(json.dumps(dict(sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,fixtures=len(records),checks=checks,records=records,executable_sha256=digest,source_sha256={name:hashlib.sha256((source_root/name).read_bytes()).hexdigest() for name in source_files}),indent=2)+'\n')
    print(len(checks),'CSR HDBSCAN groups passed')
if __name__=='__main__':main()
