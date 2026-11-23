"""Dense beta-divergence MU NMF against sklearn 1.6.1 public and update APIs."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import warnings

os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['LOKY_MAX_CPU_COUNT']='2'
import numpy as np
import sklearn
from sklearn.decomposition import NMF, _nmf
from compare_nmf_cd import REJECTIONS as CD_REJECTIONS, fields

ROOT=Path(__file__).resolve().parents[1]
METHODS=[None,'random','nndsvd','nndsvda','nndsvdar','custom']
BETAS=[-1.,0.,.5,1.,1.5,2.,2.5,3.,4.]
REJECTIONS=dict(CD_REJECTIONS)
REJECTIONS['factor_range']='working range'
REJECTIONS.update(beta_nan='NMF controls must be finite',beta_inf='NMF controls must be finite',
    beta_zero_input='NMF nonpositive beta requires strictly positive X',
    beta_negative_input='NMF nonpositive beta requires strictly positive X',
    beta_query_zero='NMF nonpositive beta requires strictly positive X',
    fitted_h_nan='NMF transform components must be finite',
    fitted_h_negative='NMF transform components must be nonnegative',
    power_range='working range',negative_power_range='working range',product_range='working range')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(x,w,h,beta,steps,tol,aw,ah,ratio):
    w=w.copy();h=h.copy();initial=_nmf._beta_divergence(x,w,h,beta,square_root=True)
    previous=initial;converged=False
    gamma=1/(2-beta) if beta<1 else (1/(beta-1) if beta>2 else 1.)
    lw=x.shape[1]*aw;lh=len(x)*(aw if ah==-1 else ah)
    with np.errstate(invalid='ignore',divide='ignore'):
        for iteration in range(1,steps+1):
            w,*_=_nmf._multiplicative_update_w(x,w,h,beta,lw*ratio,lw*(1-ratio),gamma)
            if beta<1:w[w<np.finfo(float).eps]=0
            h=_nmf._multiplicative_update_h(x,w,h,beta,lh*ratio,lh*(1-ratio),gamma)
            if beta<=1:h[h<np.finfo(float).eps]=0
            if tol>0 and iteration%10==0:
                last=_nmf._beta_divergence(x,w,h,beta,square_root=True)
                if np.divide(previous-last,initial)<tol:
                    converged=True;break
                previous=last
        last=_nmf._beta_divergence(x,w,h,beta,square_root=True)
    return w,h,iteration,np.array([initial,last,previous])[:,None],converged


def independent_objective(x,w,h,beta,aw,ah,ratio):
    # Mathematical beta divergence, with all X cells retained (the public
    # sklearn reporting function separately applies its float32-epsilon mask).
    prediction=w@h
    safe=np.maximum(prediction,np.finfo(float).tiny)
    if beta==2:div=.5*np.sum((x-prediction)**2)
    elif beta==1:
        positive=x>0
        div=np.sum(x[positive]*np.log(x[positive]/safe[positive]))-np.sum(x)+np.sum(prediction)
    elif beta==0:div=np.sum(x/safe-np.log(x/safe)-1)
    else:
        div=np.sum(x**beta-beta*x*safe**(beta-1)+(beta-1)*prediction**beta)/(beta*(beta-1))
    lw=x.shape[1]*aw;lh=len(x)*(aw if ah==-1 else ah)
    return float(div+lw*ratio*w.sum()+.5*lw*(1-ratio)*np.sum(w*w)
                 +lh*ratio*h.sum()+.5*lh*(1-ratio)*np.sum(h*h))


def error_roundoff_bound(x,iw,ih,sw,sh,w,h,beta):
    assert np.all(x>_nmf.EPSILON), 'masked/tiny data require their ordinary source comparisons'
    def scale(a,b):
        p=a@b;safe=np.maximum(p,_nmf.EPSILON)
        if beta==2:return float(np.sum(x*x)+np.sum(p*p))
        if beta==1:return float(np.sum(abs(x*np.log(x/safe)))+np.sum(p)+np.sum(x))
        if beta==0:return float(np.sum(x/safe)+x.size+np.sum(abs(np.log(x/safe))))
        return float((np.sum(x**beta)+abs(beta)*np.sum(x*safe**(beta-1))
                      +abs(beta-1)*np.sum(p**beta))/abs(beta*(beta-1)))
    arithmetic_scale=max(scale(iw,ih),scale(sw,sh),scale(w,h))
    bound=256*np.finfo(float).eps*max(*x.shape,w.shape[1])*arithmetic_scale
    # Decimal is used explicitly: NumPy longdouble is only float64 on this
    # Windows build. Certify actual final divergence from exact float inputs.
    def decimal_divergence(a,b):
        with localcontext() as context:
            context.prec=60;value=Decimal(0);bb=Decimal.from_float(beta)
            for i,j in np.ndindex(x.shape):
                xx=Decimal.from_float(float(x[i,j]))
                pp=sum((Decimal.from_float(float(a[i,r]))*Decimal.from_float(float(b[r,j]))
                        for r in range(a.shape[1])),Decimal(0))
                assert pp>0
                if beta==2:term=(xx-pp)**2/2
                elif beta==1:term=xx*(xx/pp).ln()-xx+pp
                elif beta==0:term=xx/pp-(xx/pp).ln()-1
                else:term=(xx**bb-bb*xx*pp**(bb-1)+(bb-1)*pp**bb)/(bb*(bb-1))
                value+=term
            return float(max(value,0))
    actual_divergence=max(decimal_divergence(w,h),decimal_divergence(sw,sh))
    assert actual_divergence<=bound,(actual_divergence,bound)
    return dict(arithmetic_scale=arithmetic_scale,divergence_bound=bound,
                decimal_divergence=actual_divergence,decimal_precision=60)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--executable',required=True,type=Path)
    parser.add_argument('--profile',choices=['development','checked','release'],default='development')
    parser.add_argument('--report',type=Path,default=ROOT/'validation/results/nmf_mu.json')
    parser.add_argument('--start-fixture',type=int,default=0,help='development only; reports record the selected range')
    args=parser.parse_args();exe=args.executable.resolve()
    fixtures=[];checks=[];certificates=[]
    for seed,shape,method,beta,reg,stopping in itertools.product(
        [0,42],[(8,5),(5,8)],range(6),BETAS,[(0.,-1.,0.),(.02,-1.,.5),(.01,.03,1.)],
        [(1,0.),(13,0.),(300,1.e-5)]):
        fixtures.append((seed,shape,2 if shape[0]>shape[1] else 3,method,beta,reg,stopping,'ordinary'))
    for seed,beta,special in itertools.product([0,42],BETAS,
                                               ['auto','overcomplete','zero_entries','zero_factor','exact']):
        if special=='zero_entries' and beta<=0:continue
        fixtures.append((seed,(6,4),2,5 if special in ['zero_factor','exact'] else 0,
                         beta,(0.,-1.,0.),(31,1.e-5),special))
    for seed,beta,special in itertools.product([0,42],[.5,1.,1.5,2.,2.5,3.],['zero','tiny','below_epsilon']):
        fixtures.append((seed,(6,4),2,1,beta,(.01,.02,.5),(21,0.),special))
    for beta,fraction in itertools.product([2.5,3.,4.],[.1,.49,.51,.75,1.,1.25,1.6,2.5]):
        fixtures.append((0,(1,1),1,5,beta,(0.,-1.,0.),(1,0.),('subnormal',fraction)))
    for index,(seed,shape,k,method,beta,reg,stopping,special) in enumerate(fixtures):
        if index<args.start_fixture:continue
        rng=np.random.RandomState(seed);n,p=shape
        x=.2+abs(rng.normal(size=shape));q=.2+abs(rng.normal(size=(3,p)));requested=k
        if special=='auto':requested=0;k=p
        if special=='overcomplete':requested=k=p+2
        if special=='zero_entries':x[rng.random_sample(shape)<.3]=0;q[:,0]=0
        if special=='exact':
            requested=k=1;x=np.arange(1.,n+1)[:,None]*np.arange(1.,p+1)[None,:];q=x[:3].copy()
        if special=='zero':x[:]=0;q[:]=0
        if special=='tiny':x*=1.e-4;q*=1.e-4
        if special=='below_epsilon':x*=1.e-9;q*=1.e-9
        w0=.1+rng.random_sample((n,k));h0=.1+rng.random_sample((k,p))
        if special=='zero_factor':w0[:,0]=0;h0[0,:]=0
        if special=='exact':w0=np.arange(1.,n+1)[:,None];h0=np.arange(1.,p+1)[None,:]
        subnormal=isinstance(special,tuple) and special[0]=='subnormal'
        if subnormal:
            small=np.exp((np.log(np.nextafter(0.,1.))+np.log(special[1]))/(beta-1))
            x[:]=5*small;q[:]=5*small;w0[:]=small;h0[:]=1
        steps,tol=stopping;aw,ah,l1=reg;shuffle=bool(index%2)
        source=NMF(solver='mu',n_components='auto' if requested==0 else requested,init=METHODS[method],
                   beta_loss=beta,max_iter=steps,tol=tol,random_state=seed,shuffle=shuffle,
                   alpha_W=aw,alpha_H='same' if ah==-1 else ah,l1_ratio=l1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sw=source.fit_transform(x,W=w0.copy() if method==5 else None,H=h0.copy() if method==5 else None)
        sh=source.components_.copy()
        if method==5:iw,ih=w0.copy(),h0.copy()
        else:iw,ih=_nmf._initialize_nmf(x,k,init=METHODS[method],random_state=seed)
        rw,rh,iterations,diag,converged=replay(x,iw,ih,beta,steps,tol,aw,ah,l1)
        np.testing.assert_allclose(rw,sw,atol=1.e-12,rtol=1.e-12)
        np.testing.assert_allclose(rh,sh,atol=1.e-12,rtol=1.e-12)
        assert iterations==source.n_iter_
        query=bool(np.max(sh)>0)
        expected=[sw,sh,[[source.reconstruction_err_]],np.array([source.n_iter_,k,p])[:,None],
                  diag,[[int(converged)]],sw@sh]
        if query:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');sq=source.transform(q)
            expected.extend([sq,sq@sh])
        else:
            try:source.transform(q)
            except ValueError as e:assert 'zero' in str(e)
            else:raise AssertionError('source accepted zero H transform')
        header=[n,p,len(q),requested,k,steps,seed,int(shuffle),method,int(query),beta,aw,ah,l1,tol]
        def payload(header):
            arrays=[x,q]+([w0,h0] if method==5 else [])
            return ' '.join(map(str,header))+'\n'+'\n'.join(' '.join(format(v,'.17g') for v in a.ravel(order='F'))
                                                          for a in arrays)+'\n'
        proc=subprocess.run([str(exe)],input=payload(header),text=True,capture_output=True)
        assert proc.returncode==0,(index,header,special,proc.stdout,proc.stderr)
        actual=fields(proc.stdout);assert len(actual)==len(expected),(index,proc.stdout)
        native_iterations=int(actual[3][0,0]);native_converged=bool(actual[5][0,0])
        assert 1<=native_iterations<=steps
        if native_converged:
            ni,nf,npast=actual[4].ravel()
            assert tol>0 and native_iterations%10==0
            assert ((npast-nf)/ni<tol if ni>0 else npast-nf<0),(index,'native MU stopping identity')
        else:assert native_iterations==steps
        if native_iterations!=source.n_iter_ or native_converged!=converged:
            proof=error_roundoff_bound(x,iw,ih,sw,sh,actual[0],actual[1],beta)
            bound=proof['divergence_bound']
            assert max(float(diag[0,0])**2,float(actual[4][0,0])**2,
                       float(diag[1,0])**2,float(actual[4][1,0])**2)<=2*bound,(index,'stopping',proof)
            certificates.append(dict(fixture=index,field='stopping',source_iterations=int(source.n_iter_),
                native_iterations=native_iterations,source_converged=bool(converged),
                native_converged=native_converged,source_initial=float(diag[0,0]),
                native_initial=float(actual[4][0,0]),source_final=float(diag[1,0]),
                native_final=float(actual[4][1,0]),**proof))
            expected[3][0,0]=native_iterations;expected[5]=[[int(native_converged)]]
        names=['W','H','error','metadata','mu_errors','converged','reconstruction','query_W','query_reconstruction']
        for name,value,reference in zip(names,actual,expected):
            reference=np.asarray(reference)
            absolute=0. if subnormal and name in ['W','H'] else 3.e-8
            relative=3.e-12 if subnormal and name in ['W','H'] else 3.e-7
            mismatch=~np.isclose(value,reference,atol=absolute,rtol=relative)
            if np.any(mismatch) and name in ['error','mu_errors']:
                proof=error_roundoff_bound(x,iw,ih,sw,sh,actual[0],actual[1],beta)
                assert max(float(np.max(value[mismatch]**2)),float(np.max(reference[mismatch]**2)))<= \
                    2*proof['divergence_bound'],(index,name,proof,value,reference)
                certificates.append(dict(fixture=index,field=name,source_values=reference[mismatch].tolist(),
                                         native_values=value[mismatch].tolist(),**proof))
            else:
                try:np.testing.assert_allclose(value,reference,atol=absolute,rtol=relative)
                except AssertionError as e:raise AssertionError((index,header,special,name,str(e))) from e
            checks.append(dict(fixture=index,field=name,max_abs_error=float(np.max(abs(value-reference)))))
        w,h=actual[:2];assert np.min(w)>=0 and np.min(h)>=0
        np.testing.assert_allclose(actual[6],w@h,atol=1.e-10,rtol=1.e-10)
        with np.errstate(invalid='raise',divide='raise',over='raise'):
            objective=independent_objective(x,w,h,beta,aw,ah,l1)
            reference_objective=independent_objective(x,sw,sh,beta,aw,ah,l1)
        np.testing.assert_allclose(objective,reference_objective,atol=3.e-8,rtol=3.e-7)
        checks.append(dict(fixture=index,field='objective_nonnegativity_reconstruction',
                           max_abs_error=float(abs(objective-reference_objective))))
        if not query:
            header[9]=1;fail=subprocess.run([str(exe)],input=payload(header),text=True,capture_output=True)
            assert fail.returncode!=0 and 'NMF transform components must not be all zero' in fail.stdout+fail.stderr
            checks.append(dict(fixture=index,field='source_zero_component_transform_rejection',max_abs_error=0.))
        if index%200==0:print(f'{index}/{len(fixtures)} fixtures passed',flush=True)
    rejections=[]
    for name,message in REJECTIONS.items():
        proc=subprocess.run([str(exe),name],text=True,capture_output=True)
        assert proc.returncode!=0 and message.lower() in (proc.stdout+proc.stderr).lower(),(name,proc.stdout,proc.stderr)
        assert 'SIGFPE' not in proc.stderr and 'Fortran runtime error' not in proc.stderr,(name,proc.stderr)
        rejections.append(name)
    report=dict(profile=args.profile,sklearn_version=sklearn.__version__,fixtures=len(fixtures),
                start_fixture=args.start_fixture,checks=checks,rejections=rejections,certificates=certificates,
                executable=str(exe),executable_sha256=digest(exe),validator_sha256=digest(__file__),
                cd_validator_sha256=digest(ROOT/'validation/compare_nmf_cd.py'),
                source_sha256={'decomposition/_nmf.py':digest(_nmf.__file__)})
    args.report.parent.mkdir(exist_ok=True,parents=True)
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'PASS: {len(fixtures)} fixtures from {args.start_fixture}, {len(checks)} groups, {len(rejections)} rejections')


if __name__=='__main__':main()
