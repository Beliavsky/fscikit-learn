"""Exact ordered indices against local sklearn; native preservation/repeatability checks."""
from pathlib import Path
import argparse
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.model_selection import (ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit,
    RepeatedKFold, RepeatedStratifiedKFold, LeavePOut, LeavePGroupsOut)
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"
FLAGS = "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow"


def resolve():
    proc = subprocess.run(["fpm", "run", "--flag", FLAGS, "--target", "reference_resampling_driver", "--runner"],
                          cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    paths = [ROOT / line.strip().strip('"') for line in proc.stdout.splitlines()
             if line.strip().strip('"').endswith(("reference_resampling_driver", "reference_resampling_driver.exe"))]
    paths = [p.resolve() for p in paths if p.is_file()]
    assert len(paths) == 1, proc.stdout
    return paths[0]


def fixtures(seed):
    rng = np.random.default_rng(seed)
    for n in [2, 3, 7, 16, 31, 64, 127, 513]:
        settings = [(None, None), (.25, None), (None, .6), (.25, .5), (1, None), (None, 1), (1, 1)]
        for te, tr in settings:
            yield 1, np.zeros(n, int), np.zeros(n, int), dict(test_size=te, train_size=tr, n_splits=4)
    for counts in [[2, 2], [4, 4, 4], [2, 3, 7, 11], [2, 2, 2, 2, 2, 2], [19, 31, 47], [8], [2, 2, 99]]:
        y = rng.permutation(np.repeat(np.arange(len(counts))*17-32, counts))
        for te, tr in [(None, None), (.35, None), (None, .6), (.3, .5), (len(counts), len(counts))]:
            # sklearn rejects default/rounded sizes smaller than the number of classes.
            k = len(counts); n = len(y)
            nt = int(np.ceil((.1 if te is None and tr is None else te)*n)) if isinstance(te, float) or (te is None and tr is None) else te
            nr = int(np.floor(tr*n)) if isinstance(tr, float) else tr
            if nt is None: nt = n-nr
            if nr is None: nr = n-nt
            if min(nt, nr) < k: continue
            yield 2, y, np.zeros(len(y), int), dict(test_size=te, train_size=tr, n_splits=7)
    for counts in [[1, 7], [1, 2, 3, 7, 11], [1]*9, [3, 3, 3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]:
        groups = rng.permutation(np.repeat(np.arange(len(counts))*19-51, counts))
        for te, tr in [(None, None), (.35, None), (None, .6), (.3, .5), (1, 1)]:
            yield 3, np.zeros(len(groups), int), groups, dict(test_size=te, train_size=tr, n_splits=5)
    for n in [2, 3, 7, 16, 31, 64, 127]:
        for k in sorted({2, min(n, 5), n}):
            yield 4, np.zeros(n, int), np.zeros(n, int), dict(n_splits=k, n_repeats=3)
    for counts in [[2, 2], [5, 5, 5], [1, 2, 9], [2, 7, 11], [8], [17, 31, 43]]:
        y = rng.permutation(np.repeat(np.arange(len(counts))*-17+19, counts))
        for k in sorted({2, min(max(counts), 5), max(counts)}):
            yield 5, y, np.zeros(len(y), int), dict(n_splits=k, n_repeats=3)
    for n in [2, 3, 5, 7, 9]:
        for p in sorted({1, 2, n-1}):
            if p < n:
                yield 6, np.zeros(n, int), np.zeros(n, int), dict(p=p)
    for counts in [[1, 7], [1, 2, 3, 7], [1]*7, [3, 3, 3, 3, 3]]:
        groups = rng.permutation(np.repeat(np.arange(len(counts))*19-51, counts))
        for p in sorted({1, 2, len(counts)-1}):
            if p < len(counts):
                yield 7, np.zeros(len(groups), int), groups, dict(n_groups=p)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--executable", type=Path)
    args = parser.parse_args(); executable = args.executable.resolve() if args.executable else resolve()
    RESULTS.mkdir(exist_ok=True); checks = []; fixture_count = 0
    classes = [None, ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit, RepeatedKFold,
               RepeatedStratifiedKFold, LeavePOut, LeavePGroupsOut]
    for seed in [0, 7, 42, 2026, 314159]:
        for case, (method, y, groups, options) in enumerate(fixtures(seed)):
            fixture_count += 1; n = len(y); params = dict(options)
            if method <= 5: params['random_state'] = seed
            cv = classes[method](**params)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                folds = list(cv.split(np.zeros(n), y, groups=groups if method in [3, 7] else None))
            expected = dict(count=[len(folds)], declared=[cv.get_n_splits(np.zeros(n), y, groups)],
                            labels=y, groups=groups, repeatable=[1])
            for i, (train, test) in enumerate(folds, 1):
                expected[f'fold_{i}_train'] = train+1; expected[f'fold_{i}_test'] = test+1
                assert not set(train) & set(test)
                if method in [3, 7]: assert not set(groups[train]) & set(groups[test])
            te=options.get('test_size'); tr=options.get('train_size')
            tc=te if isinstance(te, int) else 0; trc=tr if isinstance(tr, int) else 0
            te=te if isinstance(te, float) else -1; tr=tr if isinstance(tr, float) else -1
            p=options.get('p', options.get('n_groups', 2))
            inp=RESULTS/'resampling_input.txt'; out=RESULTS/'resampling_actual.txt'
            inp.write_text(f"{method} {n} {options.get('n_splits', 4)} {options.get('n_repeats', 3)} {seed} {te} {tr} {tc} {trc} {p}\n"
                           +' '.join(map(str,y))+'\n'+' '.join(map(str,groups))+'\n')
            subprocess.run([str(executable), str(inp), str(out)], cwd=ROOT, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            actual={}
            for line in out.read_text().splitlines():
                name,*values=line.split(); actual[name]=np.array(values,dtype=np.int64)
            assert actual.keys()==expected.keys(), (seed,case,method,actual.keys(),expected.keys())
            for name,want in expected.items():
                np.testing.assert_array_equal(actual[name],want,err_msg=f'{seed} {case} {method} {options} {name}')
                oracle='input_identity' if name in ['labels','groups','repeatable'] else 'sklearn'
                checks.append(dict(seed=seed,case=case,method=method,name=name,oracle=oracle,passed=True))
        print(f'seed {seed}: resampling splitters passed',flush=True)
    report=dict(sklearn=sklearn.__version__,fixtures=fixture_count,checks=checks)
    (RESULTS/'resampling.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(checks)} resampling result groups passed across {fixture_count} fixtures')


if __name__ == '__main__':
    main()
