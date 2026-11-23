"""Bit-exact quiet subtraction boundary checks against NumPy."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
from compare_image_graph import fields


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixtures():
    h=np.finfo(float).max
    anchors=[0.,-0.,np.nextafter(0.,1.),np.finfo(float).tiny,np.finfo(float).eps,1.,1.e-200,1.e200,
             h,np.nextafter(h,0.),h/2,np.nextafter(h/2,0.),np.ldexp(1.,969),np.ldexp(1.,970),
             np.ldexp(1.,971),np.inf,np.nan]
    anchors=anchors+[-v for v in anchors[2:]]
    for a,b in itertools.product(anchors,repeat=2):yield 'anchors',a,b
    rng=np.random.RandomState(42)
    bits=rng.randint(0,2**32,size=(8192,2),dtype=np.uint64)
    numbers=((bits[:,0]<<np.uint64(32))|bits[:,1]).view(np.float64)
    numbers=numbers[np.isfinite(numbers)]
    for a,b in numbers[:len(numbers)//2*2].reshape((-1,2)):yield 'random_bits',a,b
    step=np.ldexp(1.,971)
    for a in np.ldexp(rng.uniform(1.,2.,size=128),1023):
        gap=h-a
        partners=[gap,np.nextafter(gap,0.),np.nextafter(gap,np.inf),
                  gap+step/4,gap+step/2,gap+3*step/4,gap+step]
        for b in partners:
            for left,right in [(a,-b),(b,-a),(-a,b),(-b,a)]:yield 'overflow_rounding',left,right


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',required=True,type=Path)
    parser.add_argument('--profile',choices=['development','checked','release'],default='development')
    parser.add_argument('--report',required=True,type=Path);args=parser.parse_args();exe=args.executable.resolve()
    cases=list(fixtures());pairs=np.array([(a,b) for _,a,b in cases])
    payload=str(len(cases))+'\n'+' '.join(format(v,'.17g') for v in pairs.ravel(order='F'))+'\n'
    proc=subprocess.run([str(exe)],input=payload,text=True,capture_output=True)
    assert proc.returncode==0,proc.stderr
    actual=fields(proc.stdout);assert len(actual)==2
    with np.errstate(all='ignore'):
        expected=np.subtract(pairs[:,0],pairs[:,1])[:,None]
        # Retain evidence of the old premature-overflow guard, independently of
        # the new implementation. NumPy subtraction is the numerical oracle.
        old=expected[:,0].copy();a,b=pairs.T;finite=np.isfinite(a)&np.isfinite(b)
        old[finite&(b<0)&(a>np.finfo(float).max+b)]=np.inf
        old[finite&(b>0)&(a<-np.finfo(float).max+b)]=-np.inf
    for classification in [np.isnan,np.isposinf,np.isneginf]:
        np.testing.assert_array_equal(classification(actual[0]),classification(expected))
    finite=np.isfinite(expected)
    np.testing.assert_array_equal(actual[0][finite].view(np.uint64),expected[finite].view(np.uint64))
    signs=np.where(np.isnan(expected),0,np.signbit(expected)).astype(np.int64)
    np.testing.assert_array_equal(actual[1],signs)
    previous_errors=[]
    for i,((regime,a,b),result,prior) in enumerate(zip(cases,expected[:,0],old)):
        if not (np.isnan(result) and np.isnan(prior)) and result!=prior:
            previous_errors.append(dict(fixture=i,a=float(a).hex(),b=float(b).hex(),
                                        expected=float(result).hex(),previous=float(prior).hex()))
    assert previous_errors,'the suite must expose the original rounding defect'
    checks=[dict(fixture=i,field=field,passed=True,max_abs_error=0.)
            for i in range(len(cases)) for field in ['value_bits','sign']]
    binary=Path(np._core._multiarray_umath.__file__)
    report=dict(profile=args.profile,numpy_version=np.__version__,fixtures=len(cases),checks=checks,
        inventories=[dict(fixture=i,regime=regime,fields=['value_bits','sign']) for i,(regime,_,_) in enumerate(cases)],
        previous_guard_errors=previous_errors,nonfinite_results=dict(nan=int(np.isnan(expected).sum()),
            positive_inf=int(np.isposinf(expected).sum()),negative_inf=int(np.isneginf(expected).sum())),
        executable=str(exe),executable_sha256=digest(exe),validator_sha256=digest(__file__),
        parser_sha256=digest(Path(__file__).with_name('compare_image_graph.py')),
        numpy_binary=str(binary),numpy_binary_sha256=digest(binary))
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'PASS: {len(cases)} pairs, {len(checks)} groups; {len(previous_errors)} old-guard errors exposed')


if __name__=='__main__':main()
