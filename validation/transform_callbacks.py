"""Deterministic reference callbacks matching the validation-only Fortran module."""
import numpy as np


def callbacks(mode):
    def forward(x):
        x=np.asarray(x)
        if mode in [1,6]:return 2*x+3
        if mode==2:return np.log(x)
        if mode==3:return x*x
        if mode==4:return np.column_stack((x,x))
        if mode==5:
            y=x.copy();y[:,1:]+=.25*x[:,:-1];return y
        if mode==8:return np.full(x.shape,np.nan)
        return x.copy()
    def inverse(x):
        x=np.asarray(x)
        if mode==1:return (x-3)/2
        if mode==2:return np.exp(x)
        if mode==3:return np.sqrt(x)
        if mode==4:return x[:,:x.shape[1]//2].copy()
        if mode==5:
            y=x.copy()
            for j in range(1,x.shape[1]):y[:,j]=x[:,j]-.25*y[:,j-1]
            return y
        if mode==6:return 2*x+3
        if mode==7:return x+5e-9
        if mode==10:return np.where(x==7.7,x+.01,x)
        return x.copy()
    return forward,inverse


def prefixed_names(transformer,names):
    return ['z_'+str(name) for name in names] if names is not None and len(names) else ['z_unknown']
