"""Isolated diagnostics for graph parameters, dense distances and CSR structure."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from compare_neighbor_graph import ROOT,resolve

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path)
    ap.add_argument('--profile',choices=['checked','release'],default='checked');args=ap.parse_args()
    exe=args.executable.resolve() if args.executable else resolve(args.profile)
    cases=[]
    for kind in [0,1]:
        def dense(name,expect,header=None,metric='euclidean',mode='distance',train='0\n1\n3',query='2'):
            text=(header or f'{kind} 0 0 1 1 2 3 1 1')+f'\n{metric} {mode}\n'+train+'\n'+query+'\n'
            cases.append((f'{kind}_{name}',text,expect))
        dense('mode','invalid neighbor graph mode',mode='bad')
        dense('metric','unsupported neighbor graph metric',metric='bad')
        dense('power','p must be at least one',header=f'{kind} 0 0 1 1 .5 3 1 1')
        dense('nan_power','p must be finite',header=f'{kind} 0 0 1 1 NaN 3 1 1')
        dense('infinite_power','p must be finite',header=f'{kind} 0 0 1 1 Infinity 3 1 1')
        dense('nan_fit','X contains NaN',train='0\nNaN\n3')
        dense('infinite_query','X contains NaN or infinity',query='Infinity')
        dense('empty_fit','X must be nonempty',header=f'{kind} 0 0 1 1 2 0 1 1',train='')
        dense('empty_query','X must be nonempty',header=f'{kind} 0 0 1 1 2 3 1 0',query='')
        dense('nonsquare_precomputed','fit matrix must be square',metric='precomputed')
        square='0 1 2\n1 0 3\n2 3 0'
        dense('negative_fit','distances must be nonnegative',header=f'{kind} 0 0 1 1 2 3 3 1',
              metric='precomputed',train='0 -1 2\n1 0 3\n2 3 0',query='1 2 3')
        dense('negative_query','query distances must be nonnegative',header=f'{kind} 0 0 1 1 2 3 3 1',
              metric='precomputed',train=square,query='1 -2 3')
        def sparse(name,expect,ptr='1 3 5',idx='1 2 1 2',data='0 1 1 0',metric='precomputed'):
            text=f'{kind} 1 0 1 1 2 2 2 1\n{metric} distance\n4\n{ptr}\n{idx}\n{data}\n2\n1 3\n1 2\n0 1\n'
            cases.append((f'{kind}_csr_{name}',text,expect))
        sparse('endpoint','CSR pointer endpoints',ptr='0 3 5')
        sparse('pointer_range','CSR pointer is out of bounds',ptr='1 6 5')
        sparse('column','CSR column index',idx='1 3 1 2')
        sparse('negative','CSR distances must be nonnegative',data='0 -1 1 0')
        sparse('nan','CSR contains nonfinite',data='0 NaN 1 0')
        sparse('features','currently requires metric=precomputed',metric='euclidean')
    # Selection limits are intentionally checked at transform, as in sklearn.
    for k in [0,-1,2147483647]:
        cases.append((f'bad_k_{k}',f'0 0 0 {k} 1 2 3 1 1\neuclidean distance\n0\n1\n3\n2\n','invalid graph n_neighbors'))
    cases.append(('too_many_neighbors','0 0 0 3 1 2 3 1 1\neuclidean distance\n0\n1\n3\n2\n',
                  'not enough stored neighbors'))
    for radius,expect in [('-1','radius must be nonnegative'),('NaN','radius is NaN')]:
        cases.append((f'bad_radius_{radius}',f'1 0 0 1 {radius} 2 3 1 1\neuclidean distance\n0\n1\n3\n2\n',expect))
    checks=[]
    for name,inp,expect in cases:
        run=subprocess.run([str(exe)],input=inp,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert run.returncode!=0 and 'fscikit-learn:' in run.stdout and expect in run.stdout,(name,run.stdout)
        checks.append(dict(name=name,expected=expect,passed=True))
    dest=ROOT/'validation/results'/f'neighbor_graph_invalid_{args.profile}.json'
    dest.write_text(json.dumps(dict(checks=checks,profile=args.profile,
                    native_sha256=hashlib.sha256(exe.read_bytes()).hexdigest()),indent=2)+'\n')
    print(f'{len(checks)} isolated neighbor graph invalid-input checks passed')

if __name__=='__main__':main()
