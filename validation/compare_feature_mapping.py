"""DictVectorizer, FeatureHasher and MurmurHash3 against installed sklearn."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse
import hashlib
import json
import math
import subprocess
import warnings
import numpy as np
import scipy
from scipy.sparse import csr_matrix
import sklearn
from sklearn.feature_extraction import DictVectorizer,FeatureHasher
from sklearn.utils import murmurhash3_32

ROOT=Path(__file__).resolve().parents[1]
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


class Items:
    """An ordered .items() stream, including deliberate duplicate keys."""
    def __init__(self,pairs):self.pairs=pairs
    def items(self):return self.pairs


def minimum_hash_key():
    mask=2**32-1
    def unxor(x,s):
        y=x
        for shift in range(s,32,s):y^=x>>shift
        return y
    def rotr(x,s):return ((x>>s)|(x<<(32-s)))&mask
    x=unxor(2**31,16);x=x*pow(0xc2b2ae35,-1,2**32)&mask
    x=unxor(x,13);x=x*pow(0x85ebca6b,-1,2**32)&mask;x=unxor(x,16)
    x^=4;x=(x-0xe6546b64)*pow(5,-1,2**32)&mask;x=rotr(x,13)
    x=x*pow(0x1b873593,-1,2**32)&mask;x=rotr(x,15);x=x*pow(0xcc9e2d51,-1,2**32)&mask
    key=x.to_bytes(4,'little');assert murmurhash3_32(key)==-2**31
    return key


def text(value):
    data=value.encode('utf8') if isinstance(value,str) else value
    return [str(len(data)),data.hex()]


def rows_payload(rows,mode):
    result=[]
    for row in rows:
        if mode=='string':
            result.append(f'-1 {len(row)}')
            for token in row:result.extend(text(token))
        else:
            result.append(f'{len(row)} -1')
            for key,value in row:
                result.extend(text(key))
                if isinstance(value,str):result.append('1');result.extend(text(value))
                elif isinstance(value,list):
                    result.extend(['2',str(len(value))])
                    for item in value:result.extend(text(item))
                elif value is None:result.append('3')
                else:result.extend(['0',format(value,'.17g')])
    return result


def encode(c,train,query,support):
    header=[c['kind'],len(train),len(query),c['sort'],c['operation'],c['n_features'],
            c['alternate_sign'],c['restriction'],c['seed']]
    lines=[' '.join(map(str,header)),*text(c['separator']),c['mode'],*rows_payload(train,c['mode']),
           *rows_payload(query,c['mode'])]
    if c['restriction']:lines.extend([str(len(support)),' '.join(map(str,support))])
    return '\n'.join(lines)+'\n'


def fixtures():
    for seed in [0,42,2026,2147483647]:
        rng=np.random.RandomState(seed)
        for sorting in [0,1]:
            for case in range(16):
                separator=['=','','::'][case%3];n=[1,5,17,64][case%4]
                if case==14:n=100
                train=[];query=[]
                for group,count in [(train,n),(query,4)]:
                    for i in range(count):
                        row=[('number'+str(j),float(rng.randint(-8,9))/4) for j in range(4)]
                        row+=[('city',['Paris','Rome','Caf?','??'][i%4])]
                        if case==1 and group is query:row+=[('unseen',10.),('city','new')]
                        if case==2:row+=[('tags',['a','a','?'] if i%2 else [])]
                        if case==3:row=[('k'+separator+'x',-1.),('k','x')]
                        if case==4:row+=[('missing',None),('inf',np.inf),('nan',np.nan)]
                        if case==5:row=[] if i%2 else [('zero',0.),('negative_zero',-0.)]
                        if case==6:row=[('',1.),('a',2.),('a ',3.),('nul\0x',4.),('emoji??','? '),('empty',''),('nullvalue','\0')]
                        if case==11:row=[('empty',[])]
                        if case==12:row=[('ill',v) for v in [1e16,1.,-1e16,.5]*70]
                        if case==13:row=[('x'*3000+'a',1.),('x'*3000+'b',2.),('?'*511,'tail ')]
                        if case==14:row=[(f'row{i}_feature{j}',float(j)) for j in range(12)]
                        if case==15 and group is query:row=[('unknown_only',5.)]
                        group.append(row)
                if case==10:train=[]
                c=dict(kind=0,seed=seed,sort=sorting,case=case,separator=separator,mode='dict',
                       n_features=0,alternate_sign=0,restriction=1 if case==7 else (2 if case==8 else 0),
                       operation=2 if case==9 else (1 if case%2 or case==10 else 0))
                yield c,train,query
        for mode in ['dict','pair','string']:
            for case in range(16):
                n=[1,5,17,64][case%4];nf=[1,2,7,31,1024,1048576,2147483647][case%7]
                if case==5:nf={0:7,42:31,2026:1048576,2147483647:2147483647}[seed]
                if case==11:nf=2
                if case==15:n=3;nf=257
                train=[];query=[]
                for group,count in [(train,n),(query,3)]:
                    for i in range(count):
                        keys=['a','b','a ','','Caf?','??','??']
                        if case==5:keys=[minimum_hash_key()]
                        if case==6:keys=[b'\x00',b'\xff',b'a\x00b',bytes(range(256))]
                        if case==7:keys=['x'*1000+'a','x'*1000+'b','?'*511]
                        if case==15:keys=[f'large_feature_{j}' for j in range(2000)]
                        if mode=='string':row=keys+keys[:i%4]
                        else:
                            row=[(key,float(rng.randint(-8,9))/4) for key in keys]
                            if case==2:row+=[('city','Paris'),('city','Caf?'),('empty','')]
                            if case==3:row=[(key,0.) for key in keys]
                            if case==4:row=[('a',np.nan),('b',np.inf)]
                            if case==8:row=[('a',np.inf),('a',-np.inf)]
                            if case==9:row=[('a',1e308)]*8
                            if case==10:row=[('ill',v) for v in [1e16,1.,-1e16,.5]*70]
                            if case==12:row=[('cancel',1.),('cancel',-1.)]
                            if case==13:row=[('x','v'),('x=v',-1.)]
                        if case==14 and i%2:row=[]
                        group.append(row)
                c=dict(kind=1,seed=seed,sort=0,case=case,separator='=',mode=mode,n_features=nf,
                       alternate_sign=0 if case==9 else case%2,restriction=0,operation=case%2)
                yield c,train,query


def expanded(rows,c,vocabulary=None):
    result=[]
    for row in rows:
        buckets=defaultdict(list)
        entries=[(key,1.) for key in row] if c['mode']=='string' else row
        for key,value in entries:
            if isinstance(value,list):values=[(key+c['separator']+v,1.) for v in value]
            elif isinstance(value,str):values=[(key+('=' if c['kind'] else c['separator'])+value,1.)]
            else:values=[(key,float(value) if value is not None else np.nan)]
            for name,v in values:
                if c['kind']:
                    if v==0:continue
                    h=murmurhash3_32(name);index=abs(h)%c['n_features']
                    if c['alternate_sign'] and h<0:v=-v
                else:
                    if name not in vocabulary:continue
                    index=vocabulary[name]
                buckets[index].append(v)
        result.append(buckets)
    return result


def resolve(profile):
    options=['--profile','release'] if profile=='release' else ['--flag',FLAGS]
    r=subprocess.run(['fpm','run',*options,'--target','reference_feature_mapping_driver','--runner'],cwd=ROOT,
                     text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    paths=[ROOT/s.strip().strip('"') for s in r.stdout.splitlines()
           if s.strip().strip('"').endswith(('reference_feature_mapping_driver','reference_feature_mapping_driver.exe'))]
    paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,r.stdout
    return paths[0]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
    ap.add_argument('--executable',type=Path);ap.add_argument('--cases',type=int,nargs='*');args=ap.parse_args()
    exe=args.executable.resolve() if args.executable else resolve(args.profile)
    checks=[];certificates=[];source_warnings=Counter();count=0;max_error=0.
    def record(c,name):checks.append(dict(**c,name=name,passed=True))
    for c,train,query in fixtures():
        if args.cases is not None and c['case'] not in args.cases:continue
        def convert(rows):
            if c['mode']=='string':return rows
            return [Items(row) for row in rows] if c['mode']=='dict' else rows
        support=[]
        with warnings.catch_warnings(record=True) as caught,np.errstate(over='ignore',invalid='ignore'):
            warnings.simplefilter('always')
            if c['kind']==0:
                model=DictVectorizer(sort=bool(c['sort']),separator=c['separator'])
                if c['operation']==0:a=model.fit_transform(convert(train))
                else:
                    model.fit(convert(train));a=model.transform(convert(train)) if train else csr_matrix((0,0))
                initial_names=list(model.get_feature_names_out());vocab=model.vocabulary_.copy()
                inverse_sparse=model.inverse_transform(a) if a.shape[0] and a.shape[1] and np.isfinite(a.data).all() else None
                dense=a.toarray()
                inverse_dense=model.inverse_transform(dense) if inverse_sparse is not None and np.isfinite(dense).all() else None
                if c['restriction']==1:
                    support=[int(i%2==0) for i in range(a.shape[1])];model.restrict(np.array(support,dtype=bool))
                elif c['restriction']==2:
                    indices=list(range(a.shape[1]-1,-1,-2));support=[i+1 for i in indices];model.restrict(indices,indices=True)
                if c['operation']==2:model.separator='::'
                final_names=list(model.get_feature_names_out());b=model.transform(convert(query))
                qc=dict(c,separator=model.separator)
                sources=[expanded(train,c,vocab),expanded(query,qc,model.vocabulary_)]
            else:
                model=FeatureHasher(n_features=c['n_features'],input_type=c['mode'],alternate_sign=bool(c['alternate_sign']))
                a=model.fit_transform(convert(train)) if c['operation']==0 else model.fit().transform(convert(train))
                b=model.transform(convert(query));initial_names=[];final_names=[];inverse_sparse=inverse_dense=None
                sources=[expanded(train,c),expanded(query,c)]
        source_warnings.update(str(w.message) for w in caught)
        run=subprocess.run([str(exe)],input=encode(c,train,query,support),cwd=ROOT,text=True,
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert run.returncode==0,(c,run.stdout)
        lines=iter(run.stdout.splitlines())
        def read_text():
            n=int(next(lines));value=bytes.fromhex(next(lines));assert len(value)==n
            return value.decode('utf8')
        def read_names():return [read_text() for _ in range(int(next(lines)))]
        def numeric(name,actual,want,buckets,columns):
            nonlocal max_error
            actual=np.asarray(actual);want=np.asarray(want)
            assert actual.shape==want.shape,(c,name,actual.shape,want.shape)
            finite=np.isfinite(actual)&np.isfinite(want)
            max_error=max(max_error,float(np.max(abs(actual[finite]-want[finite]),initial=0)))
            close=np.isclose(actual,want,rtol=3e-13,atol=3e-13,equal_nan=True)
            for index in np.flatnonzero(~close.ravel()):
                values=buckets[columns[index]]
                assert np.isfinite(values).all() and np.isfinite(actual.ravel()[index]) and np.isfinite(want.ravel()[index])
                exact=math.fsum(values);l1=math.fsum(abs(v) for v in values);eps=len(values)*np.finfo(float).eps
                bound=eps/(1-eps)*l1+np.finfo(float).tiny*len(values)
                av=float(actual.ravel()[index]);wv=float(want.ravel()[index])
                assert abs(av-exact)<=bound and abs(wv-exact)<=bound,(c,name,av,wv,exact,bound)
                certificates.append(dict(**c,group=name,column=int(columns[index]),terms=len(values),
                                         native=av,source=wv,correctly_rounded_sum=exact,error_bound=bound))
        def read_csr(want,buckets,label):
            nr,nc,nnz,has_dense=map(int,next(lines).split())
            assert (nr,nc,nnz)==(*want.shape,want.nnz),(c,label,'shape')
            ptr=np.fromstring(next(lines),sep=' ',dtype=int)-1
            col=np.fromstring(next(lines),sep=' ',dtype=int)-1;data=np.fromstring(next(lines),sep=' ')
            np.testing.assert_array_equal(ptr,want.indptr,err_msg=str(c));np.testing.assert_array_equal(col,want.indices,err_msg=str(c))
            record(c,label+'_csr_structure')
            if c['kind']==0:
                for row in range(nr):
                    for column in set(col[ptr[row]:ptr[row+1]]):
                        av=data[ptr[row]:ptr[row+1]][col[ptr[row]:ptr[row+1]]==column]
                        wv=want.data[want.indptr[row]:want.indptr[row+1]][want.indices[want.indptr[row]:want.indptr[row+1]]==column]
                        np.testing.assert_array_equal(np.sort(av),np.sort(wv),err_msg=str(c))
                record(c,label+'_stored_values')
            else:
                for row in range(nr):
                    lo,hi=ptr[row:row+2]
                    numeric(label+f'_row{row}_values',data[lo:hi],want.data[lo:hi],buckets[row],col[lo:hi])
                record(c,label+'_summed_values')
            dense_line=next(lines)
            if has_dense:
                actual=np.fromstring(dense_line,sep=' ').reshape((nr,nc),order='F');expected=want.toarray()
                for row in range(nr):numeric(label+f'_row{row}_dense',actual[row],expected[row],buckets[row],list(range(nc)))
                record(c,label+'_dense')
            return nr,nc
        def read_inverse(want,buckets,label):
            n=int(next(lines));assert n==(-1 if want is None else len(want)),(c,label,n)
            if want is None:return
            for row in range(n):
                actual={}
                for _ in range(int(next(lines))):
                    key=read_text();actual[key]=float(next(lines))
                assert actual.keys()==want[row].keys(),(c,label,actual,want[row])
                keys=list(actual);columns=[vocab[key] for key in keys]
                numeric(label+f'_row{row}',list(actual.values()),[want[row][key] for key in keys],buckets[row],columns)
            record(c,label)
        assert read_names()==initial_names,(c,'initial names')
        if c['kind']==0:record(c,'initial_names')
        read_csr(a,sources[0],'train')
        read_inverse(inverse_sparse,sources[0],'inverse_sparse');read_inverse(inverse_dense,sources[0],'inverse_dense')
        assert read_names()==final_names,(c,'final names')
        if c['kind']==0:record(c,'final_names')
        read_csr(b,sources[1],'query')
        actual=np.fromstring(next(lines),sep=' ',dtype=int)
        np.testing.assert_array_equal(actual,np.arange(1,len(final_names)+1))
        if c['kind']==0:record(c,'vocabulary_indices')
        assert next(lines,None) is None,(c,'extra output')
        count+=1
        if count%64==0:print(f'{count} structured feature fixtures passed',flush=True)
    lengths=list(range(34))+[63,64,65,127,128,129,255,256,257,1024,4097]
    for seed in [0,42,2147483648,4294967295]:
        rng=np.random.RandomState(seed)
        for length in lengths:
            for pattern in range(1 if length==0 else 3):
                key=rng.bytes(length) if pattern==0 else (b'\0'*length if pattern==1 else (b'Abcde'*((length+4)//5))[:length])
                c=dict(kind=2,seed=seed,length=length,pattern=pattern)
                payload=f'2 0 0 0 0 0 0 0 {seed}\n'+'\n'.join(text(key))+'\n'
                run=subprocess.run([str(exe)],input=payload,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                assert run.returncode==0,(c,run.stdout)
                assert int(run.stdout)==murmurhash3_32(key,seed=seed),(c,run.stdout)
                record(c,'signed_hash');count+=1
    key=minimum_hash_key();payload='2 0 0 0 0 0 0 0 0\n'+'\n'.join(text(key))+'\n'
    run=subprocess.run([str(exe)],input=payload,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert run.returncode==0 and int(run.stdout)==-2**31,run.stdout
    record(dict(kind=2,seed=0,length=4,pattern='INT32_MIN'),'signed_hash');count+=1
    assert not certificates,('Outstanding source numerical mismatches',certificates[:3])
    root=Path(sklearn.__file__).parent
    report=dict(profile=args.profile,sklearn=sklearn.__version__,scipy=scipy.__version__,numpy=np.__version__,
                fixtures=count,cases_filter=args.cases,checks=checks,roundoff_certificates=certificates,
                max_absolute_difference=max_error,source_warnings=dict(source_warnings),
                native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                numpy_pairwise_source=dict(version='2.2.6',url='https://raw.githubusercontent.com/numpy/numpy/v2.2.6/numpy/_core/src/umath/loops_utils.h.src',
                    sha256='ed85d5b99f0b457cb7a316bf8c4acb884034bbbb81a89c5cedb62be80bf7c4f8'),
                scipy_source_sha256={'sparse/_compressed.py':hashlib.sha256(
                    (Path(scipy.__file__).parent/'sparse/_compressed.py').read_bytes()).hexdigest()},
                source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                               ['feature_extraction/_dict_vectorizer.py','feature_extraction/_hash.py',
                                'feature_extraction/_hashing_fast.pyx','utils/murmurhash.pyx']})
    (ROOT/'validation/results'/f'feature_mapping_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {count} feature mapping fixtures passed; {len(certificates)} roundoff certificates')


if __name__=='__main__':main()
