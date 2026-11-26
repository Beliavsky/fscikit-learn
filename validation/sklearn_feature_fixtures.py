"""Shared actual-sklearn feature-count cases for CART, ExtraTree and forests."""
import itertools
import numpy as np

def feature_values(p):
    return [1,1.,.001,np.nextafter(.5,0),.5,np.nextafter(.5,1),'sqrt','log2','none',0,p+3]

def tree_feature_fixtures(kind):
    criteria=['gini','entropy','log_loss'] if kind=='classification' else ['squared_error','friedman_mse','absolute_error','poisson']
    for mode,criterion,q,p,seed in itertools.product([0,1],criteria,[1,2],[3,8],[0,7]):
        rng=np.random.RandomState(1917+p+q);n=19;x=rng.normal(size=(n,p))
        if seed==7:x[:,:p//2]=1
        y=rng.randint(2,size=(n,q)) if kind=='classification' else np.abs(rng.normal(size=(n,q)))+.1
        w=np.resize([1.,0.,.5,2.,3.],n) if seed==7 else None
        for value in feature_values(p):
            yield dict(mode=mode,criterion=criterion,seed=seed,q=q,n=n,p=p,weighted=w is not None,
                       budget=5 if seed==7 else 0,depth=4,nf=value,leaf=1,alpha=.01,decrease=0.),x,y,w

def forest_feature_fixtures(kind):
    criteria=['gini','entropy','log_loss'] if kind=='classification' else ['squared_error','friedman_mse','absolute_error','poisson']
    for sequence,(mode,criterion,q,p,seed,boot,path) in enumerate(itertools.product([0,1],criteria,[1,2],[3,8],[0,7],[0,1],[0,1])):
        values=feature_values(p)
        values=values[:6] if path==0 else values[6:]+[.5]
        for step,value in enumerate(values):
            n=17;rng=np.random.RandomState(1131+q+step);x=rng.normal(size=(n,p))
            if seed==7:x[:,:p//2]=1
            y=np.abs(rng.normal(size=(n,q)))+.1 if kind=='regression' else rng.randint(2,size=(n,q))
            if kind=='classification':y[0,:]=0;y[1,:]=1
            w=np.resize([1.,.5,2.,3.],n)
            yield dict(sequence=sequence,step=step,reset=int(step==0),warm=1,mode=mode,criterion=criterion,
                       seed=seed,q=q,n=n,p=p,nt=[2,2,3,4,5,6][step],boot=boot,ns=.7 if boot else 0,oob=boot,
                       weighted=seed==7,depth=3,budget=0,nf=value,leaf=1,alpha=0.,decrease=0.,
                       cw=2 if boot else 0,path=path),x,y,w
