"""PolynomialFeatures via dense typed pipelines, unions and column selections."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import warnings

os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import PolynomialFeatures,Normalizer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from compare_polynomial import fields,names_matrix
from compare_stateless_preprocessing import assert_close


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True,type=Path)
    parser.add_argument('--profile',choices=['development','checked','release'],default='development')
    parser.add_argument('--report',required=True,type=Path);args=parser.parse_args();exe=args.executable.resolve()
    fixtures=[]
    for seed,shape,degrees,bias,interaction,named,mode in itertools.product(
            [0,42],[(17,3),(23,5)],[(0,2),(2,3),(0,4)],[False,True],[False,True],[False,True],range(4)):
        for weighted in ([False,True] if mode==1 else [False]):
            fixtures.append((seed,shape,degrees,bias,interaction,named,mode,weighted))
    checks,inventories=[],[]
    for index,(seed,shape,degrees,bias,interaction,named,mode,weighted) in enumerate(fixtures):
        rng=np.random.RandomState(seed);n,p=shape;lo,hi=degrees
        x=rng.uniform(-.8,.8,size=shape);q=rng.uniform(-.8,.8,size=(5,p));x[0]=0;q[0]=0
        y=1+2*x[:,0]**2-3*x[:,1]+.2*rng.normal(size=n)
        weights=rng.uniform(.1,2.,size=n);weights[1]=0
        names=[f'axis {j}' for j in range(p)]
        train=pd.DataFrame(x,columns=names) if named else x;query=pd.DataFrame(q,columns=names) if named else q
        poly=PolynomialFeatures(degree=(lo,hi) if lo else hi,include_bias=bias,interaction_only=interaction,order='F')
        expected=[]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if mode in [0,1]:
                model=Pipeline([('poly',poly),('unit',Normalizer())] if mode==0 else [('poly',poly),('ridge',Ridge(alpha=.75,solver='svd'))])
                if mode==0:
                    z=model.fit_transform(train)
                    expected.extend([('fit_transform',z),('query',model.transform(query)),('names',names_matrix(model.get_feature_names_out())),
                                     ('metadata',np.array([p,z.shape[1]],dtype=np.int64)[:,None])])
                else:
                    model.fit(train,y,**({'ridge__sample_weight':weights} if weighted else {}))
                    expected.extend([('train_predict',model.predict(train)[:,None]),('query_predict',model.predict(query)[:,None]),
                                     ('coef',model[-1].coef_[:,None]),('intercept',np.array([[model[-1].intercept_]])),
                                     ('names',names_matrix(model[:-1].get_feature_names_out()))])
                expected.append(('can_inverse',np.array([[int(hasattr(model,'inverse_transform'))]],dtype=np.int64)))
            elif mode==2:
                model=FeatureUnion([('poly',poly),('raw','passthrough')],transformer_weights={'poly':.7,'raw':1.2})
                z=model.fit_transform(train)
                expected.extend([('fit_transform',z),('query',model.transform(query)),('names',names_matrix(model.get_feature_names_out())),
                                 ('metadata',np.array([p,z.shape[1]],dtype=np.int64)[:,None])])
            else:
                model=ColumnTransformer([('poly',poly,[2,0])],remainder='passthrough')
                z=model.fit_transform(train)
                expected.extend([('fit_transform',z),('query',model.transform(query)),('names',names_matrix(model.get_feature_names_out())),
                                 ('metadata',np.array([p,z.shape[1]],dtype=np.int64)[:,None])])
        header=[mode,n,p,len(q),lo,hi,int(bias),int(interaction),int(named),int(weighted)]
        payload=' '.join(map(str,header))+'\n'+'\n'.join(' '.join(format(v,'.17g') for v in a.ravel(order='F')) for a in [x,q,y,weights])+'\n'
        proc=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
        assert proc.returncode==0,(index,header,proc.stdout,proc.stderr)
        actual=fields(proc.stdout);assert len(actual)==len(expected),(index,header,len(actual),len(expected))
        group_names=[]
        for value,(field,reference) in zip(actual,expected):
            reference=np.asarray(reference)
            if np.issubdtype(reference.dtype,np.integer):np.testing.assert_array_equal(value,reference);error=0.
            elif mode==1:
                # Independent SVD implementations differ at roundoff level.
                # Keep transform-only cases at the stricter preprocessing tolerance.
                np.testing.assert_allclose(value,reference,rtol=3.e-12,atol=3.e-13,
                                           err_msg=f'{index} {field}')
                error=float(np.max(np.abs(value-reference),initial=0.))
            else:error=assert_close(value,reference,index,field)
            checks.append(dict(fixture=index,field=field,max_abs_error=error));group_names.append(field)
        checks.append(dict(fixture=index,field='prototype_preserved',max_abs_error=0.));group_names.append('prototype_preserved')
        inventories.append(dict(fixture=index,mode=mode,weighted=weighted,named=named,fields=group_names))
        if index%100==0:print(f'{index}/{len(fixtures)} fixtures passed',flush=True)
    report=dict(profile=args.profile,sklearn_version=sklearn.__version__,fixtures=len(fixtures),checks=checks,inventories=inventories,
        executable=str(exe),executable_sha256=digest(exe),validator_sha256=digest(__file__),
        core_validator_sha256=digest(Path(__file__).with_name('compare_polynomial.py')))
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'PASS: {len(fixtures)} fixtures, {len(checks)} composition groups')


if __name__=='__main__':main()
