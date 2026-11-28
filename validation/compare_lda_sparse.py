"""Raw CSR LDA storage and streaming contracts against scikit-learn 1.6.1."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,tempfile,warnings
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation

def storage(dense,layout):
    pointers=[0];indices=[];values=[]
    for row in dense:
        entries=[]
        for j,v in enumerate(row):
            if v:
                entries.extend([(j,v*.25),(j,v*.75)] if 'duplicates' in layout else [(j,v)])
            elif 'zeros' in layout:entries.append((j,0.))
        if 'reverse' in layout:entries.reverse()
        indices.extend(j for j,v in entries);values.extend(v for j,v in entries);pointers.append(len(values))
    return csr_matrix((np.array(values,float),np.array(indices,dtype=np.int32),np.array(pointers,dtype=np.int32)),shape=dense.shape)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
    executable=a.executable.resolve();binary=hashlib.sha256(executable.read_bytes()).hexdigest();checks=[];sequences=0
    with tempfile.TemporaryDirectory(prefix='fsklearn-lda-csr-input-') as directory:
        directory=Path(directory)
        for seed in [0,7,42]:
            rng=np.random.RandomState(seed);base=rng.poisson(1.2,(11,7)).astype(float);base[0]=0;query=rng.poisson(.5,(4,7)).astype(float);query[0]=0
            for layout in ['canonical','reverse','duplicates','duplicates_reverse','zeros_reverse','duplicates_zeros_reverse']:
                for method in ['batch','online']:
                    for initial in [2,3]:
                        config=dict(n_components=3,max_iter=3,batch_size=4,evaluate_every=1,max_doc_update_iter=50,random_state=seed,doc_topic_prior=.3,topic_word_prior=.7,learning_decay=.7,learning_offset=10.,total_samples=123.5,perp_tol=.0001,mean_change_tol=.001,learning_method=method)
                        x=storage(base,layout);q=storage(query,layout)
                        ops=[(initial,x),(4,q),(5,q),(6,q),(7,q),(8,q),(2,x[:3]),(6,x),(9,x),(4,q)]
                        model=LatentDirichletAllocation(n_jobs=1,**config);expected={};lines=[str(len(ops))]
                        with warnings.catch_warnings(),np.errstate(all='ignore'):
                            warnings.simplefilter('ignore',RuntimeWarning)
                            for i,(op,data) in enumerate(ops,1):
                                prefix=f'{i}_';lines.extend([f'{op} {data.shape[0]} {data.shape[1]}',' '.join(str(config[k]) for k in ['n_components','max_iter','batch_size','evaluate_every','max_doc_update_iter','random_state']),' '.join(str(config[k]) for k in ['doc_topic_prior','topic_word_prior','learning_decay','learning_offset','total_samples','perp_tol','mean_change_tol']),method,str(data.nnz),' '.join(str(v+1) for v in data.indptr),' '.join(str(v+1) for v in data.indices),' '.join(format(v,'.17g') for v in data.data)])
                                result=None
                                if op==2:model.partial_fit(data)
                                elif op==3:result=model.fit_transform(data)
                                elif op==4:result=model.transform(data)
                                elif op==5:result=model.transform(data,normalize=False)
                                elif op==6:result=[model.score(data)]
                                elif op==7:result=[model.perplexity(data)]
                                elif op==8:result=[model.perplexity(data,sub_sampling=True)]
                                elif op==9:result=model.fit_transform(data,normalize=False)
                                if result is not None:expected[prefix+'result']=result
                                expected[prefix+'components']=model.components_.copy();expected[prefix+'exp_components']=model.exp_dirichlet_component_.copy();expected[prefix+'meta']=[model.n_features_in_,model.n_iter_,model.n_batch_iter_];expected[prefix+'priors']=[model.doc_topic_prior_,model.topic_word_prior_];expected[prefix+'bound']=[getattr(model,'bound_',np.nan)]
                        inp=directory/'input.txt';out=directory/'output.txt';inp.write_text('\n'.join(lines)+'\n')
                        proc=subprocess.run([str(executable),str(inp),str(out),'raw'],capture_output=True,text=True);assert proc.returncode==0,(seed,layout,method,initial,proc.stdout,proc.stderr)
                        actual={name:np.array(values,float) for name,*values in map(str.split,out.read_text().splitlines())};assert actual.keys()==expected.keys()
                        for name,values in expected.items():
                            np.testing.assert_allclose(actual[name],np.asarray(values).ravel(order='F'),rtol=0 if name.endswith('meta') else 3.e-8,atol=0 if name.endswith('meta') else 3.e-8,equal_nan=True,err_msg=str((seed,layout,method,initial,name)))
                        sequences+=1;checks.append(dict(seed=seed,layout=layout,method=method,initial=initial,groups=len(expected),passed=True))
            print(f'raw CSR LDA seed {seed}: passed',flush=True)
    assert binary==hashlib.sha256(executable.read_bytes()).hexdigest()
    a.report.write_text(json.dumps(dict(sklearn=sklearn.__version__,sequences=sequences,groups=sum(c['groups'] for c in checks),checks=checks,executable_sha256=binary),indent=2)+'\n')
    print(sequences,'raw CSR LDA sequences passed')
if __name__=='__main__':main()
