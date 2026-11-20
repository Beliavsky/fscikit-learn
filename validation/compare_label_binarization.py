"""Exact dense LabelBinarizer and ragged MultiLabelBinarizer source checks."""
from pathlib import Path
import argparse
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'validation/results'
FLAGS = '-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable:
        executable = args.executable.resolve()
    else:
        proc = subprocess.run(['fpm', 'run', '--flag', FLAGS, '--target',
                               'reference_label_binarization_driver', '--runner'],
                              cwd=ROOT, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=True)
        paths = [ROOT / line.strip().strip('"') for line in proc.stdout.splitlines()
                 if line.strip().strip('"').endswith(('reference_label_binarization_driver',
                                                      'reference_label_binarization_driver.exe'))]
        paths = [p.resolve() for p in paths if p.is_file()]
        assert len(paths) == 1, proc.stdout
        executable = paths[0]
    checks = []
    scenarios = 0

    def run(kind, seed, case, lines, expected):
        nonlocal scenarios
        scenarios += 1
        inp = RESULTS / 'label_binarization_input.txt'
        out = RESULTS / 'label_binarization_output.txt'
        inp.write_text('\n'.join(lines) + '\n')
        proc = subprocess.run([str(executable), str(inp), str(out)], cwd=ROOT,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.returncode == 0, (kind, seed, case, proc.stdout)
        actual = {}
        for line in out.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=np.int64)
        assert actual.keys() == expected.keys(), (kind, seed, case, actual.keys(), expected.keys())
        for name, want in expected.items():
            np.testing.assert_array_equal(actual[name], np.asarray(want, dtype=np.int64).ravel(order='F'),
                                          err_msg=f'{kind} {seed} {case} {name}')
            checks.append(dict(kind=kind, seed=seed, case=case, name=name, passed=True, oracle='sklearn'))

    def sets_expected(expected, name, rows):
        expected[name + '_counts'] = [len(row) for row in rows]
        expected[name + '_values'] = [value for row in rows for value in row]

    for seed in [0, 7, 42, 2026, 314159]:
        rng = np.random.RandomState(seed)
        case = 0
        for c in [1, 2, 3, 7, 19]:
            classes = np.arange(c) * 17 - 83
            train = rng.permutation(np.concatenate([classes, classes, rng.choice(classes, size=11)]))
            for neg, pos in [(0, 1), (-3, 4), (-9, 0), (2, 9), (-2147483648, 2147483647)]:
                for variant in range(3):
                    case += 1
                    query = (rng.choice(classes, size=13) if variant == 0 else
                             np.resize([classes[0], 9001, 9002], 13) if variant == 1 else np.full(13, 9001))
                    width = 1 if c == 1 else 1 + variant % 2 if c == 2 else c + [-1, 0, 2][variant]
                    scores = rng.randint(-3, 4, size=(13, width)).astype(float) * .5
                    threshold = None if variant == 0 else [.5, 0, -1][variant]
                    model = LabelBinarizer(neg_label=neg, pos_label=pos).fit(train)
                    fitted = model.transform(train)
                    encoded = model.fit_transform(train)
                    q = model.transform(query)
                    expected = dict(fit=fitted, classes=model.classes_, train=encoded, query=q,
                                    shape=q.shape, type=[int(model.y_type_ == 'multiclass')],
                                    inverse=model.inverse_transform(q),
                                    scores=model.inverse_transform(scores, threshold=threshold))
                    model.neg_label = -5
                    model.pos_label = 3
                    expected['live'] = model.transform(query)
                    expected['refit'] = model.fit_transform([99, 98, 99])
                    expected['refit_classes'] = model.classes_
                    lines = [f'1 {len(train)} {len(query)}',
                             f'{neg} {pos} {width} {int(threshold is not None)} {threshold or 0}',
                             ' '.join(map(str, train)), ' '.join(map(str, query))]
                    lines.extend(' '.join(map(str, row)) for row in scores)
                    run('label', seed, case, lines, expected)
        for case in range(36):
            universe = np.arange(1 + case % 9) * 13 - 41
            n, m = (0, 0) if case == 0 else (1 + case % 11, 2 + case % 7)
            train = [rng.choice(universe, size=rng.randint(0, 8)).tolist() for _ in range(n)]
            query = [rng.choice(np.r_[universe, 999], size=rng.randint(0, 9)).tolist() for _ in range(m)]
            if case in (1, 2): train = [[] for _ in train]
            if case % 4 == 0:
                configured = rng.permutation(universe).tolist()
            elif case % 4 == 1:
                configured = []
            elif case % 4 == 2:
                configured = [999, int(universe[-1]), -1234]
            else:
                configured = None
            model = MultiLabelBinarizer(classes=configured)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                model.fit(train)
                fitted = model.transform(train)
                encoded = model.fit_transform(train)
                q = model.transform(query)
                expected = dict(fit=fitted, classes=model.classes_, train=encoded, query=q, shape=q.shape)
                sets_expected(expected, 'inverse', model.inverse_transform(q))
                sets_expected(expected, 'complement', model.inverse_transform((1-q).astype(float)))
                if configured is not None: model.classes = []
                expected['snapshot'] = model.transform(query)
                model.classes = None
                expected['refit'] = model.fit_transform(query)
                expected['refit_classes'] = model.classes_
            lines = [f'2 {n} {m}', str(-1 if configured is None else len(configured))]
            if configured: lines.append(' '.join(map(str, configured)))
            for row in train + query:
                lines.append(str(len(row)))
                if row: lines.append(' '.join(map(str, row)))
            run('multi', seed, case, lines, expected)
        print(f'seed {seed}: 75 label and 36 multilabel scenarios passed', flush=True)
    (RESULTS / 'label_binarization.json').write_text(json.dumps(
        dict(sklearn=sklearn.__version__, scenarios=scenarios, checks=checks), indent=2) + '\n')
    print(f'{len(checks)} label binarization result groups passed across {scenarios} scenarios')


if __name__ == '__main__':
    main()
