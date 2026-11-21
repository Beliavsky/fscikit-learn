"""Image patch extraction/sampling/reconstruction against sklearn 1.6.1."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import warnings
from collections import Counter
import numpy as np
import sklearn
from sklearn.feature_extraction.image import PatchExtractor,extract_patches_2d,reconstruct_from_patches_2d

ROOT=Path(__file__).resolve().parents[1]
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def resolve(profile):
    options=['--profile','release'] if profile=='release' else ['--flag',FLAGS]
    r=subprocess.run(['fpm','run',*options,'--target','reference_image_driver','--runner'],cwd=ROOT,
                     text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    paths=[ROOT/s.strip().strip('"') for s in r.stdout.splitlines()
           if s.strip().strip('"').endswith(('reference_image_driver','reference_image_driver.exe'))]
    paths=[p.resolve() for p in paths if p.is_file()]
    assert len(paths)==1,r.stdout
    return paths[0]


def fixtures():
    for seed in [0,42,2026,2147483647]:
        rng=np.random.RandomState(seed)
        for h,w in [(1,1),(3,5),(4,4),(7,9),(12,15),(17,21)]:
            for kind in range(4):
                n=3 if kind<2 else 1
                c=1 if kind in [0,2] or h%2 else 3
                x=rng.normal(size=(n,h,w,c))
                for mode in range(9):
                    if mode==6 and (h<10 or kind>=2):
                        continue
                    ph,pw=(h,w) if mode==1 else (max(1,h//3),max(1,w//3))
                    limit=3 if mode==2 else (h*w+4 if mode==3 else 0)
                    fraction=.37 if mode==4 else (.001 if mode==5 else 0.)
                    if mode==6:ph=pw=0
                    used=[-1,-1,2,-1,0,2,-1,-1,-1][mode]
                    values=x if mode<7 else np.full_like(x,1e308)
                    if mode==8:values[:,::2]=-1e308
                    yield dict(seed=seed,h=h,w=w,c=c,n=n,kind=kind,mode=mode,ph=ph,pw=pw,
                               limit=limit,fraction=fraction,used=used),values


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['checked','release'],default='checked')
    ap.add_argument('--executable',type=Path);args=ap.parse_args()
    exe=args.executable.resolve() if args.executable else resolve(args.profile)
    checks=[];count=0;max_error=0.;source_warnings=Counter()
    for cfg,x in fixtures():
        seed,h,w,c,n,kind,mode,ph,pw,limit,fraction,used=(cfg[k] for k in
            ['seed','h','w','c','n','kind','mode','ph','pw','limit','fraction','used'])
        patch_size=None if ph==0 else (ph,pw)
        kwargs=dict(max_patches=limit or fraction or None,random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if kind<2:
                model=PatchExtractor(patch_size=patch_size,**kwargs)
                data=x[:,:,:,0] if kind==0 else x
                want=model.fit_transform(data)
                repeated=model.transform(data)
            else:
                data=x[0,:,:,0] if kind==2 else x[0]
                want=extract_patches_2d(data,patch_size,**kwargs);repeated=want
            if want.ndim==3:want=want[:,:,:,None]
            if repeated.ndim==3:repeated=repeated[:,:,:,None]
            take=len(want) if used<0 else min(used,len(want))
            with np.errstate(over='ignore',invalid='ignore'):
                rec=reconstruct_from_patches_2d(want[:take],(h,w,c))
        source_warnings.update(str(w.message) for w in caught)
        header=[n,h,w,c,ph,pw,limit,fraction,seed,kind,used]
        payload=' '.join(map(str,header))+'\n'+' '.join(format(v,'.17g') for v in x.ravel(order='F'))+'\n'
        r=subprocess.run([str(exe)],input=payload,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert r.returncode==0,(cfg,r.stdout)
        lines=r.stdout.splitlines();shape=tuple(map(int,lines[0].split()))
        np.testing.assert_array_equal(shape,want.shape,err_msg=str(cfg))
        checks.append(dict(**cfg,name='shape',passed=True))
        for key,line,expected in [('patches',1,want),('reconstruction',2,rec),('repeat',3,repeated)]:
            actual=np.fromstring(lines[line],sep=' ').reshape(expected.shape,order='F')
            if key=='reconstruction':
                np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=2e-14,err_msg=str(cfg))
                finite=np.isfinite(actual)&np.isfinite(expected)
                max_error=max(max_error,float(np.max(abs(actual[finite]-expected[finite]),initial=0)))
            else:np.testing.assert_array_equal(actual,expected,err_msg=str(cfg))
            checks.append(dict(**cfg,name=key,passed=True))
        count+=1
        if count%100==0:print(f'{count} image fixtures passed',flush=True)
    source=Path(sklearn.__file__).parent/'feature_extraction/image.py'
    report=dict(profile=args.profile,sklearn=sklearn.__version__,numpy=np.__version__,fixtures=count,
                checks=checks,max_reconstruction_error=max_error,source_warnings=dict(source_warnings),
                native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                source_sha256={'feature_extraction/image.py':hashlib.sha256(source.read_bytes()).hexdigest()})
    (ROOT/'validation/results'/f'image_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'{len(checks)} groups / {count} image fixtures passed')


if __name__=='__main__':main()
