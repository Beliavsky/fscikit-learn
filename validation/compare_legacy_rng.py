"""Separate NumPy RandomState stream checks, including the full uint32 seed range."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FLAGS='-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path)
    ap.add_argument('--profile',choices=['checked','release'],default='checked');args=ap.parse_args()
    if args.executable:exe=args.executable.resolve()
    else:
        flags=['--profile','release'] if args.profile=='release' else ['--flag',FLAGS]
        run=subprocess.run(['fpm','run',*flags,'--target','reference_legacy_rng_driver','--runner'],cwd=ROOT,
                           text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/s.strip().strip('"') for s in run.stdout.splitlines()
               if s.strip().strip('"').endswith(('reference_legacy_rng_driver.exe','reference_legacy_rng_driver'))]
        paths=[p.resolve() for p in paths if p.is_file()];assert len(paths)==1,run.stdout;exe=paths[0]
    checks=[]
    for seed in [0,42,2026,2147483647,2147483648,4294967295]:
        dest=ROOT/'validation/results'/f'legacy_rng_{args.profile}_{seed}.txt'
        subprocess.run([str(exe),str(seed),str(dest)],check=True,cwd=ROOT)
        actual={}
        for line in dest.read_text().splitlines():
            key,*values=line.split();actual[key]=np.array(values,float)
        expected=dict(raw=np.random.RandomState(seed).randint(0,2**32,size=2048,dtype=np.uint32),
                      uniform=np.random.RandomState(seed).random_sample(2048),
                      normal=np.random.RandomState(seed).normal(size=2048),
                      index=np.random.RandomState(seed).choice(17,size=2048)+1)
        rng=np.random.RandomState(seed);indices=[];normal=[];uniform=[]
        for j in range(1,1001):
            indices.append(rng.choice(j%57+1)+1);normal.append(rng.normal());uniform.append(rng.random_sample())
        expected.update(mixed_index=indices,mixed_normal=normal,mixed_uniform=uniform)
        for key,want in expected.items():
            if 'normal' in key:np.testing.assert_allclose(actual[key],want,rtol=2e-14,atol=2e-14)
            else:np.testing.assert_array_equal(actual[key],want)
            checks.append(dict(seed=seed,name=key,passed=True))
    report=dict(numpy=np.__version__,profile=args.profile,checks=checks,
                native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    (ROOT/'validation/results'/f'legacy_rng_{args.profile}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(checks)} separate NumPy RandomState stream groups passed')

if __name__=='__main__':main()
