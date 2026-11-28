"""Dense single-worker batch/online variational LDA against installed sklearn."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import warnings
import tempfile
import hashlib
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import sklearn
from scipy.sparse import csr_matrix
import mpmath as mp
from scipy.special import gammaln
from sklearn.decomposition import LatentDirichletAllocation
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'validation/results'
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def high_precision_bound(model,x,sub_sampling=False):
    """Independent 60-digit variational bound with exact digamma/log-Gamma."""
    docs=model.transform(x,normalize=False)
    with mp.workdps(60):
        def expectation(row):
            values=[mp.mpf(float(v)) for v in row]; total=sum(values)
            return [mp.digamma(v)-mp.digamma(total) for v in values]
        log_docs=[expectation(row) for row in docs]
        log_topics=[expectation(row) for row in model.components_]
        value=mp.mpf(0)
        for i,row in enumerate(x):
            for j,count in enumerate(row):
                if count:value+=mp.mpf(float(count))*mp.log(sum(mp.exp(log_docs[i][k]+log_topics[k][j]) for k in range(model.n_components)))
        def prior_part(prior,values,logs):
            prior=mp.mpf(float(prior));result=mp.mpf(0)
            for row,log_mean in zip(values,logs):
                row=[mp.mpf(float(v)) for v in row]
                result+=sum((prior-v)*e+mp.loggamma(v)-mp.loggamma(prior) for v,e in zip(row,log_mean))
                result+=mp.loggamma(prior*len(row))-mp.loggamma(sum(row))
            return result
        value+=prior_part(model.doc_topic_prior_,docs,log_docs)
        if sub_sampling:value*=mp.mpf(float(model.total_samples))/len(x)
        value+=prior_part(model.topic_word_prior_,model.components_,log_topics)
        return float(value)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path);parser.add_argument('--sparse-input',action='store_true');parser.add_argument('--report',type=Path);args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable:executable=args.executable.resolve()
    else:
        proc=subprocess.run(['fpm','run','--flag',FLAGS,'--target','reference_lda_driver','--runner'],cwd=ROOT,
                            text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/line.strip().strip('"') for line in proc.stdout.splitlines()
               if line.strip().strip('"').endswith(('reference_lda_driver','reference_lda_driver.exe'))]
        paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,proc.stdout
        executable=paths[0]
    binary=hashlib.sha256(executable.read_bytes()).hexdigest()
    work=tempfile.TemporaryDirectory(prefix="fsklearn-lda-reference-")
    checks=[];corrections=[];roundoff=[]
    for seed,n,p in [(0,23,7),(7,31,5),(42,36,11),(2026,19,1)]:
        rng=np.random.RandomState(seed)
        means=.3+3*(np.arange(n)[:,None]%3==np.arange(p)[None,:]%3)
        counts=rng.poisson(means).astype(float);counts[0]=0
        query=rng.poisson(1.3,size=(8,p)).astype(float);query[0]=0
        for case in range(30):
            x=counts.copy();q=query.copy();ops=[]
            controls=dict(n_components=3,max_iter=5,batch_size=5,evaluate_every=-1,max_doc_update_iter=100,
                          random_state=seed,doc_topic_prior=None,topic_word_prior=None,learning_decay=.7,
                          learning_offset=10.,total_samples=1.e6,perp_tol=.1,mean_change_tol=.001,learning_method='batch')
            def add(op,data,**changes):
                controls.update(changes);ops.append((op,np.array(data,copy=True),controls.copy()))
            if case==1:controls.update(learning_method='online')
            if case==2:controls.update(max_iter=0,evaluate_every=1)
            if case==4:controls.update(n_components=1,max_iter=4)
            if case==5:controls.update(doc_topic_prior=.2,topic_word_prior=.7)
            if case==6:controls.update(n_components=7,mean_change_tol=1.e-6,max_doc_update_iter=200)
            if case==7:controls.update(learning_method='online',learning_decay=0.,learning_offset=1.)
            if case==8:controls.update(learning_method='online',learning_decay=1.,learning_offset=100.)
            if case==9:controls.update(max_doc_update_iter=1,mean_change_tol=0.)
            if case==10:controls.update(max_doc_update_iter=0)
            if case==11:x=rng.uniform(size=(n,p))*3;q=rng.uniform(size=(8,p))*2
            if case==12:x[:]=0;q[:]=0;controls.update(evaluate_every=1)
            if case==13:x[:]=1
            if case==14:x[:,::2]=0;q[:,::2]=0
            if case==15:controls.update(evaluate_every=1,perp_tol=1.e9,max_iter=20)
            if case==16:controls.update(evaluate_every=2,perp_tol=1.e9,max_iter=20)
            if case==17:controls.update(learning_method='online',evaluate_every=2,perp_tol=1.e9,max_iter=20)
            if case==18:controls.update(learning_method='online',batch_size=n+5)
            if case==19:controls.update(learning_method='online',batch_size=1,n_components=2,max_iter=3)
            if case==24:controls.update(doc_topic_prior=1.e-7,topic_word_prior=1.e-7,max_iter=2,max_doc_update_iter=20)
            if case==25:controls.update(doc_topic_prior=1.,topic_word_prior=1.,max_iter=4)
            if case==28:controls.update(n_components=2,max_doc_update_iter=3,mean_change_tol=.1)
            if case==29:controls.update(n_components=n+2,max_iter=1)
            if case==3:
                controls.update(max_iter=0)
                for a in np.array_split(x,3):add(2,a)
            elif case==20:
                add(2,x[:5],total_samples=100.5,batch_size=3)
                add(4,q);add(7,q)
                add(2,x[5:],learning_method='online',batch_size=4,doc_topic_prior=.6,topic_word_prior=.4,random_state=seed+3)
            elif case==21:
                add(1,x);add(2,x[:1],total_samples=123.4,random_state=seed+17)
                add(2,x[-8:],learning_decay=.5,learning_offset=1.,batch_size=3)
            elif case==22:
                add(2,x);add(4,q)
                x=np.column_stack([x,np.ones(n)]);q=np.column_stack([q,np.zeros(len(q))])
                add(1,x,n_components=4,random_state=seed+1);add(2,x[:3])
            elif case==23:add(9,x)
            elif case==26:
                add(2,x[:7],learning_offset=1.5,total_samples=1.e9)
                add(2,x[7:],batch_size=2,learning_decay=1.)
            elif case==27:
                add(2,x[:4]);add(4,q);add(5,q);add(6,q);add(7,q);add(8,q);add(2,x[4:])
            else:add(3,x)
            for op,data in [(4,q),(5,q),(6,q),(7,q),(8,q),(4,q),(2,x[:7]),(6,q)]:add(op,data)
            source=LatentDirichletAllocation(n_jobs=1);expected={};oracles={};allowances={};zero_fields=set();bound_zero=False;lines=[str(len(ops))]
            integer_keys=['n_components','max_iter','batch_size','evaluate_every','max_doc_update_iter','random_state']
            real_keys=['doc_topic_prior','topic_word_prior','learning_decay','learning_offset','total_samples','perp_tol','mean_change_tol']
            with warnings.catch_warnings(),np.errstate(divide='ignore',invalid='ignore',over='ignore'):
                warnings.simplefilter('ignore',RuntimeWarning)
                for i,(op,data,config) in enumerate(ops,1):
                    source.set_params(**config);prefix=f'{i}_'
                    lines.extend([f'{op} {data.shape[0]} {data.shape[1]}',
                                  ' '.join(str(config[k]) for k in integer_keys),
                                  ' '.join(f'{(-1 if config[k] is None else config[k]):.17g}' for k in real_keys),config['learning_method']])
                    lines.extend(' '.join(f'{v:.17g}' for v in row) for row in data)
                    source_data=csr_matrix(data) if args.sparse_input else data
                    result=None
                    if op==1:source.fit(source_data)
                    elif op==2:source.partial_fit(source_data)
                    elif op==3:result=source.fit_transform(source_data)
                    elif op==4:result=source.transform(source_data)
                    elif op==5:result=source.transform(source_data,normalize=False)
                    elif op==6:result=[source.score(source_data)]
                    elif op==7:result=[source.perplexity(source_data)]
                    elif op==8:result=[source.perplexity(source_data,sub_sampling=True)]
                    elif op==9:result=source.fit_transform(source_data,normalize=False)
                    if result is not None:
                        name=prefix+'result';expected[name]=result
                        if op in (7,8) and data.sum()==0 and np.allclose(source.components_,source.topic_word_prior_,atol=0,rtol=8*np.finfo(float).eps):
                            corrections.append(dict(seed=seed,case=case,name=name,source=float(result[0]) if np.isfinite(result[0]) else str(float(result[0])),reason='zero words and zero evidence: undefined 0/0'))
                            expected[name]=[np.nan];zero_fields.add(name)
                        if op in (6,7,8) and source.components_.max()>1.e6 and data.sum()>0:
                            budget=32*np.finfo(float).eps*(np.abs(gammaln(source.components_)).sum()+np.abs(gammaln(source.components_.sum(axis=1))).sum())
                            oracle_bound=high_precision_bound(source,data,op==8)
                            words=data.sum()*(source.total_samples/len(data) if op==8 else 1.)
                            oracle=oracle_bound if op==6 else np.exp(-oracle_bound/words)
                            oracles[name]=oracle;allowances[name]=(budget,words,op)

                    expected[prefix+'components']=source.components_.copy()
                    expected[prefix+'exp_components']=source.exp_dirichlet_component_.copy()
                    expected[prefix+'meta']=[source.n_features_in_,source.n_iter_,source.n_batch_iter_]
                    expected[prefix+'priors']=[source.doc_topic_prior_,source.topic_word_prior_]
                    expected[prefix+'bound']=[getattr(source,'bound_',np.nan)]
                    if op in (1,3,9):bound_zero=data.sum()==0
                    if bound_zero and np.allclose(source.components_,source.topic_word_prior_,atol=0,rtol=8*np.finfo(float).eps):
                        corrections.append(dict(seed=seed,case=case,name=prefix+'bound',source=float(expected[prefix+'bound'][0]) if np.isfinite(expected[prefix+'bound'][0]) else str(float(expected[prefix+'bound'][0])),reason='zero training words: undefined fitted perplexity'))
                        expected[prefix+'bound']=[np.nan];zero_fields.add(prefix+'bound')
            inp=Path(work.name)/'input.txt';out=Path(work.name)/'output.txt';inp.write_text('\n'.join(lines)+'\n')
            proc=subprocess.run([str(executable),str(inp),str(out), 'csr' if args.sparse_input else 'dense'],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            assert proc.returncode==0,(seed,case,proc.stdout)
            actual={}
            for line in out.read_text().splitlines():
                name,*values=line.split();actual[name]=np.array(values,dtype=float)
            assert actual.keys()==expected.keys()
            for name,value in expected.items():
                tol=0 if name.endswith('meta') else 3.e-8
                if name in allowances:
                    budget,words,op=allowances[name];a=actual[name][0];b=float(np.asarray(value).ravel()[0])
                    difference=abs(a-b) if op==6 else abs(np.log(a)-np.log(b))*words
                    assert difference<=budget,(seed,case,name,difference,budget)
                    roundoff.append(dict(seed=seed,case=case,name=name,bound_difference=float(difference),budget=float(budget)))
                    np.testing.assert_allclose(a,oracles[name],atol=3.e-8,rtol=3.e-8,err_msg=f'high precision {seed} {case} {name}')
                    checks.append(dict(seed=seed,case=case,name=name+'_high_precision',passed=True,oracle='60-digit mpmath'))
                else:
                    np.testing.assert_allclose(actual[name],np.asarray(value).ravel(order='F'),atol=tol,rtol=tol,equal_nan=True,
                                               err_msg=f'seed={seed} case={case} {name}')
                checks.append(dict(seed=seed,case=case,name=name,passed=True,
                                   oracle='zero-word definition' if name in zero_fields else 'source roundoff budget' if name in allowances else 'sklearn'))
        print(f'seed {seed}: 30 configurations passed',flush=True)
    assert binary==hashlib.sha256(executable.read_bytes()).hexdigest()
    work.cleanup()
    (args.report or RESULTS/('lda_csr.json' if args.sparse_input else 'lda.json')).write_text(json.dumps(dict(sklearn=sklearn.__version__,sparse_input=args.sparse_input,executable_sha256=binary,checks=checks,zero_word_corrections=corrections,source_roundoff=roundoff),indent=2)+'\n')
    print(f'{len(checks)} LatentDirichletAllocation result groups passed')


if __name__=='__main__':main()
