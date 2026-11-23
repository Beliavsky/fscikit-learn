"""PolynomialFeatures dense/CSR values, powers, names, counts and source defects."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import warnings
from types import SimpleNamespace

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import _polynomial
from compare_stateless_preprocessing import sparse_variant, payload_matrix, assert_close

REJECTIONS = {
    'negative_degree':'invalid polynomial degree range', 'negative_min':'invalid polynomial degree range',
    'degree_order':'invalid polynomial degree range', 'zero_no_bias':'zero polynomial degree without bias',
    'empty_rows':'X must be nonempty', 'empty_features':'X must be nonempty', 'nan':'NaN or infinity',
    'inf':'NaN or infinity', 'sparse_nan':'NaN or infinity', 'csr_index':'CSR column index',
    'csr_pointer':'CSR pointer endpoints', 'unfitted':'PolynomialFeatures is not fitted',
    'unfitted_powers':'PolynomialFeatures is not fitted', 'unfitted_names':'PolynomialFeatures is not fitted',
    'query_width':'feature count mismatch', 'sparse_width':'feature count mismatch',
    'changed_degree':'parameters changed', 'changed_bias':'parameters changed', 'changed_interaction':'parameters changed',
    'fit_name_count':'fit name count mismatch', 'fit_name_unallocated':'feature name is unallocated',
    'name_count':'input name count mismatch', 'name_text':'input feature names differ',
    'name_length':'input feature names differ', 'name_unallocated':'feature name is unallocated',
    'count_overflow':'feature count exceeds int64', 'csr_output_range':'CSR feature count exceeds native index range',
    'dense_address_range':'dense allocation exceeds addressable', 'powers_address_range':'powers allocation exceeds addressable',
    'names_address_range':'feature names allocation exceeds addressable', 'canonical_overflow':'canonicalized input exceeds finite',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fields(output):
    tokens = iter(output.split());result = []
    while True:
        kind = next(tokens, None)
        if kind is None:
            return result
        assert kind in ['I', 'F'], (kind, output)
        n, p = int(next(tokens)), int(next(tokens))
        convert = int if kind == 'I' else float
        result.append(np.array([convert(next(tokens)) for _ in range(n*p)],
                               dtype=np.int64 if kind == 'I' else float).reshape((n, p), order='F'))


def names_matrix(names):
    encoded = [str(s).encode('utf8') for s in names]
    result = np.full((max(map(len, encoded), default=0), len(encoded)), -1, dtype=np.int64)
    for j, name in enumerate(encoded):
        result[:len(name), j] = list(name)
    return result


def append_result(expected, prefix, x, sparse):
    if not sparse:
        expected.append((prefix, np.asarray(x)))
    else:
        x = x.tocsr()
        expected.extend([(prefix, x.toarray()), (prefix+'_shape', np.array(x.shape)[:,None]),
                         (prefix+'_indptr', x.indptr[:,None]+1), (prefix+'_indices', x.indices[:,None]+1),
                         (prefix+'_data', x.data[:,None])])


def serial_sparse(x):
    encode = lambda v: float(v) if np.isfinite(v) else ('nan' if np.isnan(v) else '+inf' if v>0 else '-inf')
    return dict(shape=list(x.shape), indptr=x.indptr.tolist(), indices=x.indices.tolist(), data=[encode(v) for v in x.data])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--profile', choices=['development','checked','release'], default='development')
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--start-fixture', type=int, default=0)
    args = parser.parse_args();exe = args.executable.resolve()
    fixtures = []
    for seed, shape, degrees, bias, interaction, named, storage in itertools.product(
            [0,42], [(5,3),(3,5),(4,1)], [(0,0),(0,1),(0,2),(0,3),(0,4),(2,3),(3,5)],
            [False,True], [False,True], range(3), range(4)):
        if degrees[1]==0 and not bias:
            continue
        fixtures.append((seed,shape,degrees,bias,interaction,named,storage,0,'ordinary'))
    for degrees, interaction, storage, special in itertools.product(
            [(0,3),(0,4),(2,5)], [False,True], range(4), ['large','tiny','association','zeros']):
        fixtures.append((42,(3,3),degrees,True,interaction,0,storage,0,special))
    for degree, storage in itertools.product([13,21,32], [0,1]):
        fixtures.append((0,(2,1),(degree,degree),False,False,0,storage,0,'high_degree'))
    # Fit-only metadata must not allocate the enormous expanded array/powers.
    for p,lo,hi,interaction,bias in [
            (100000,0,2,False,True),(100000,0,3,False,True),(1000000,0,2,False,False),
            (2147483647,0,1,False,False),(2,2147483647,2147483647,False,False),
            (3,2147483647,2147483647,False,False),(63,0,63,True,True),
            (100000,99999,100000,True,False)]:
        # 2**63 outputs exceed signed int64; omit the bias in this exact limit case.
        if p==63:bias=False
        fixtures.append((0,(1,p),(lo,hi),bias,interaction,0,1,1,'count_only'))
    for degree in [2147483647,2**40+1,2**60+1]:
        fixtures.append((0,(1,1),(degree,degree),False,False,0,1,2,'symbolic_high_degree'))
    checks, inventories, corrections, nonfinite, empty_metadata, symbolic, empty_transform, count_shortcuts = [], [], [], [], [], [], [], []
    name_refit_stress=[]
    assert 0<=args.start_fixture<len(fixtures)
    for index,(seed,shape,degrees,bias,interaction,named,storage,mode,special) in enumerate(fixtures):
        if index<args.start_fixture:continue
        n,p=shape;lo,hi=degrees;rng=np.random.RandomState(seed)
        if mode==1:
            x=csr_matrix(shape);q=csr_matrix((1,p))
        else:
            x=rng.uniform(-.8,.8,size=shape);q=rng.uniform(-.8,.8,size=(2,p))
            x[rng.uniform(size=shape)<.4]=0;q[rng.uniform(size=q.shape)<.4]=0
            if special in ['high_degree','symbolic_high_degree']:
                x[:]=.5;q[:]=-.5
            elif special=='large':x*=1.e200;q*=1.e200
            elif special=='tiny':x*=1.e-200;q*=1.e-200
            elif special=='association':
                x[:]=[1.e200,1.e200,0.];q[:]=[1.e200,1.e-200,1.e-200]
            elif special=='zeros':x[:]=0;q[:]=0
            if storage:
                x=sparse_variant(x,storage,rng);q=sparse_variant(q,storage,rng)
        canonical=x.copy();canonical_q=q.copy()
        if storage:
            canonical.sum_duplicates();canonical.sort_indices()
            canonical_q.sum_duplicates();canonical_q.sort_indices()
        names=[f'axis {j}' for j in range(p)] if mode!=1 else []
        if named==2:
            names[0]=''
            if p>=2:names[1]='heat flux'
            if p>=3:names[2]='β'
        model=PolynomialFeatures(degree=(lo,hi) if lo else hi,include_bias=bias,interaction_only=interaction,order='F')
        expected=[]
        with warnings.catch_warnings(),np.errstate(all='ignore'):
            warnings.simplefilter('ignore')
            fit_data=pd.DataFrame(canonical.toarray() if storage else canonical,columns=names) if named else canonical
            if mode==1 and interaction and p>1000 and hi>1000:
                count=PolynomialFeatures._num_combinations(p,lo,hi,interaction,bias)
                count_shortcuts.append(dict(fixture=index,p=p,lo=lo,hi=hi,count=count,
                    oracle='sklearn exact _num_combinations; avoids unrelated full-degree auxiliary count'))
                model=SimpleNamespace(n_output_features_=count,_min_degree=lo,_max_degree=hi)
            else:
                model.fit(fit_data)
            expected.append(('metadata',np.array([p,model.n_output_features_,model._min_degree,model._max_degree],dtype=np.int64)[:,None]))
            if mode==0:
                if model.n_output_features_==0:
                    results=[]
                    for field,data in [('train',canonical),('query',canonical_q)]:
                        try:result=model.transform(data)
                        except (ValueError,IndexError) as error:
                            empty_transform.append(dict(fixture=index,field=field,source_error=type(error).__name__,message=str(error)))
                            result=csr_matrix((data.shape[0],0)) if storage else np.empty((len(data),0))
                        results.append(result)
                    train,query=results
                else:
                    train=model.transform(canonical);query=model.transform(canonical_q)
                append_result(expected,'train',train,storage);append_result(expected,'query',query,storage)
                append_result(expected,'fit_transform',train,storage)
                if not storage:append_result(expected,'legacy_array',train,False)
                if storage in [2,3]:
                    if special=='ordinary':
                        dense_model=PolynomialFeatures(degree=(lo,hi) if lo else hi,include_bias=bias,interaction_only=interaction)
                        dense_oracle=dense_model.fit_transform(canonical.toarray())
                        assert_close(train.toarray(),dense_oracle,index,'canonical_sparse_vs_dense_identity')
                    try:
                        raw=model.transform(x);raw.check_format(full_check=True)
                        corrections.append(dict(fixture=index,kind='canonicalized_csr',raw=serial_sparse(raw),
                                                canonical=serial_sparse(train)))
                    except (ValueError,OverflowError,IndexError) as error:
                        corrections.append(dict(fixture=index,kind='canonicalized_csr',raw_error=type(error).__name__,
                                                message=str(error),canonical=serial_sparse(train)))
            if mode!=1:
                if special=='symbolic_high_degree':
                    powers=np.array([[hi]],dtype=np.int64)
                    default_names=[f'x0^{hi}'];supplied_names=[f'axis 0^{hi}']
                    symbolic.append(dict(fixture=index,degree=hi,formula='one feature at one degree has exponent degree'))
                elif model.n_output_features_==0:
                    try:model.powers_
                    except ValueError as error:
                        empty_metadata.append(dict(fixture=index,source_error=type(error).__name__,message=str(error)))
                    else:raise AssertionError('expected sklearn 1.6.1 empty powers defect')
                    powers=np.zeros((0,p),dtype=np.int64);default_names=[];supplied_names=[]
                else:
                    powers=model.powers_
                    default_names=model.get_feature_names_out();supplied_names=model.get_feature_names_out(names)
                expected.extend([('powers',powers),('powers_alias',powers),('names',names_matrix(default_names)),
                                 ('supplied_names',names_matrix(supplied_names))])
                if special=='symbolic_high_degree':refit_names=default_names
                elif model.n_output_features_==0:refit_names=[]
                else:
                    model.fit(canonical);refit_names=model.get_feature_names_out()
                expected.append(('refit_names',names_matrix(refit_names)))
        header=[mode,int(storage!=0),n,p,q.shape[0],lo,hi,int(bias),int(interaction),named]
        payload=' '.join(map(str,header))+'\n'+payload_matrix(x)+payload_matrix(q)
        proc=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
        assert proc.returncode==0,(index,header,special,proc.stdout,proc.stderr)
        actual=fields(proc.stdout);assert len(actual)==len(expected),(index,header,len(actual),len(expected))
        group_names=[]
        for value,(field,reference) in zip(actual,expected):
            reference=np.asarray(reference)
            if np.issubdtype(reference.dtype,np.integer):
                np.testing.assert_array_equal(value,reference,err_msg=f'{index} {field}');error=0.
            else:error=assert_close(value,reference,index,field)
            checks.append(dict(fixture=index,field=field,max_abs_error=error));group_names.append(field)
            if not np.issubdtype(reference.dtype,np.integer) and np.any(~np.isfinite(reference)):
                nonfinite.append(dict(fixture=index,field=field,nan=int(np.isnan(reference).sum()),
                    positive_inf=int(np.isposinf(reference).sum()),negative_inf=int(np.isneginf(reference).sum())))
        checks.append(dict(fixture=index,field='input_preservation',max_abs_error=0.));group_names.append('input_preservation')
        if index==1520:
            # Repeated fresh processes exposed an intermittent formatted-I/O
            # failure in the original exponent-name conversion, in both profiles.
            assert (shape,degrees,named,storage)==((3,5),(3,5),2,0)
            for repeat in range(1,128):
                repeated=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
                assert repeated.returncode==0,('name/refit stress',repeat,repeated.stderr)
                repeated_fields=fields(repeated.stdout)
                assert len(repeated_fields)==len(actual)
                for value,reference in zip(repeated_fields,actual):
                    np.testing.assert_array_equal(value,reference)
            name_refit_stress.append(dict(fixture=index,processes=128,fields=len(actual)))
        inventories.append(dict(fixture=index,mode=mode,storage=storage,p=p,lo=lo,hi=hi,bias=bias,
                                interaction=interaction,named=named,special=special,fields=group_names))
        if index%200==0:print(f'{index}/{len(fixtures)} fixtures passed',flush=True)
    for name,message in REJECTIONS.items():
        proc=subprocess.run([str(exe),name],text=True,capture_output=True)
        assert proc.returncode!=0 and message in proc.stdout+proc.stderr,(name,proc.stdout,proc.stderr)
        assert 'SIGFPE' not in proc.stderr and 'Fortran runtime error' not in proc.stderr,(name,proc.stderr)
    report=dict(profile=args.profile,sklearn_version=sklearn.__version__,scipy_version=scipy.__version__,
        fixtures=len(fixtures),start_fixture=args.start_fixture,checks=checks,inventories=inventories,rejections=list(REJECTIONS),
        canonicalization_records=corrections,empty_metadata_source_errors=empty_metadata,
        empty_transform_source_errors=empty_transform,
        fit_count_oracles=count_shortcuts,
        symbolic_metadata=symbolic,nonfinite_reference_results=nonfinite,name_refit_stress=name_refit_stress,
        executable=str(exe),executable_sha256=digest(exe),validator_sha256=digest(__file__),
        source_sha256={name:digest(Path(sklearn.__file__).parent/name) for name in
                      ['preprocessing/_polynomial.py','preprocessing/_csr_polynomial_expansion.pyx']},
        scipy_source_sha256={name:digest(Path(scipy.__file__).parent/name) for name in ['sparse/_compressed.py','sparse/_data.py']})
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'PASS: {len(fixtures)} fixtures, {len(checks)} groups, {len(REJECTIONS)} rejections')


if __name__=='__main__':main()
