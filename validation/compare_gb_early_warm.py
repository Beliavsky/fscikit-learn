"""Classic boosting validation stopping across repeated warm/cold fits."""
from pathlib import Path
import argparse, hashlib, itertools, json, os, subprocess, warnings
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from compare_gb_warm_start import snapshot
from sklearn_tree_runtime import tree_runtime_provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kind', choices=['regression', 'classification'], required=True)
    ap.add_argument('--executable', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    regression = a.kind == 'regression'
    pairs = [(loss, 1) for loss in ['squared_error', 'absolute_error', 'quantile', 'huber']] if regression else [
        ('log_loss', 2), ('log_loss', 3), ('log_loss', 5), ('exponential', 2)]
    cls = GradientBoostingRegressor if regression else GradientBoostingClassifier
    records, payload = [], []
    for group, ((loss, classes), seed, weighted, sub, profile) in enumerate(itertools.product(
            pairs, [0, 7], [0, 1], [.65, 1.], range(4))):
        if a.smoke and (seed or not weighted or sub != .65 or profile != 2):
            continue
        model = cls()
        delta = np.empty(0)
        prior = None
        for step in range(4):
            n = 43 + (6 if profile == 2 and step > 1 else 0)
            rng = np.random.RandomState(383 + n)
            x = rng.normal(size=(n, 3))
            y = 2*x[:, 0] + x[:, 1]*x[:, 2] + rng.normal(size=n) if regression else np.arange(n) % classes*17-9
            if not regression:
                rng.shuffle(y)
            w = np.resize([1., .5, 2., 3.], n)
            weights = w if weighted else None
            patience = [3, 3, 3, 3] if profile == 0 else [0, 3, 1, 0] if profile == 1 else [3, 3, 5, 2] if profile == 2 else [1]*4
            patience = patience[step]
            fraction = [.2, .35, .25, .4][step] if profile == 2 else .25
            current_seed = seed + step if profile == 2 else seed
            warm = step > 0 and not (profile == 3 and step == 3)
            nt = 8 if step == 0 else len(model.estimators_) + [0, 0, 5, 7][step]
            tol = 1.e6 if profile == 3 else .01
            zero = profile % 2
            rate = 0. if profile == 3 else .15
            c = dict(n=n, p=3, loss=loss, criterion='friedman_mse', seed=current_seed, nt=nt,
                depth=3, feature='sqrt', leaf=1, split=2, budget=0, rate=rate, alpha=.9,
                decrease=0., ccp=0., fraction=0., weighted=weighted, zero=zero, subsample=sub,
                warm=int(warm), fresh=int(step == 0), score_ok=1, patience=patience,
                validation_fraction=fraction, tol=tol, group=group, step=step, profile=profile)
            train = np.arange(n)
            if patience:
                train, _ = train_test_split(train, random_state=current_seed, test_size=fraction,
                    stratify=None if regression else y)
            if not warm and not regression:
                prior = DummyClassifier(strategy='prior').fit(x[train], y[train],
                    sample_weight=None if weights is None else weights[train]).class_prior_
            delta = np.resize(delta if warm else np.empty(0), nt)
            params = dict(n_estimators=nt, warm_start=warm, loss=loss, random_state=current_seed,
                max_depth=3, max_features='sqrt', learning_rate=rate, init='zero' if zero else None,
                subsample=sub, n_iter_no_change=patience or None, validation_fraction=fraction, tol=tol)
            model.set_params(**params)
            def monitor(i, est, local):
                delta[i] = est._loss.closs.delta if loss == 'huber' else 0.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(x, y, sample_weight=weights, monitor=monitor if regression else None)
                delta = delta[:len(model.estimators_)]
                expected, error = snapshot(model, x, y, weights, a.kind, prior, delta)
            assert error is None
            records.append(dict(case=c, source=expected, differences=[]))
            keys = ['n','p','loss','criterion','seed','nt','depth','feature','leaf','split']
            keys += ['budget','rate','alpha','decrease','ccp','fraction','weighted','zero'] if regression else [
                'rate','decrease','zero','weighted','budget','ccp','fraction']
            keys += ['subsample','warm','fresh','score_ok','patience','validation_fraction','tol']
            text = ' '.join(str(c[k]) for k in keys) + '\n'
            text += '\n'.join(' '.join(map(str, row)) for row in x) + '\n'
            text += ' '.join(map(str, y)) + '\n' + ' '.join(map(str, w)) + '\n'
            payload.append(text)
    text = ''.join(payload)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.with_suffix('.in').write_text(text)
    p = subprocess.run([str(a.executable)], input=text, text=True, capture_output=True, timeout=180)
    a.report.with_suffix('.stdout').write_text(p.stdout)
    a.report.with_suffix('.stderr').write_text(p.stderr)
    lines = p.stdout.splitlines()
    width = 23 if regression else 26
    integer = {'metadata','classes','prediction','leaves','tree_metadata','count','left','right','feature'}
    for i, record in enumerate(records):
        chunk = lines[i*width:(i+1)*width]
        record['native'] = {}
        if len(chunk) != width:
            record['differences'].append('missing native output')
            continue
        for (key, want), line in zip(record['source'].items(), chunk):
            got = np.fromstring(line, sep=' ')
            want = np.asarray(want)
            record['native'][key] = got.tolist()
            ok = got.shape == want.shape and (np.array_equal(got, want) if key in integer else
                np.allclose(got, want, rtol=2e-10, atol=2e-12, equal_nan=True))
            if not ok:
                record['differences'].append(key)
    driver = 'gbr' if regression else 'gbc'
    files = [Path(__file__), Path('validation/compare_gb_warm_start.py'),
        Path(f'app/reference_{driver}_early_warm_driver.f90'), a.executable,
        Path('src/sklearn_classic_gradient_boosting.f90'), Path('src/sklearn_resampling.f90')]
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report = dict(kind=a.kind, configurations=len(records), records=records,
        failed_configurations=sum(bool(r['differences']) for r in records), native_returncode=p.returncode,
        input_sha256=hashlib.sha256(text.encode()).hexdigest(), sklearn=sklearn.__version__,
        runtime=tree_runtime_provenance(), sha256={str(p): sha(p) for p in files})
    a.report.write_text(json.dumps(report, indent=2)+'\n')
    print({k: v for k, v in report.items() if k not in ['records','runtime','sha256']})
    assert p.returncode == 0 and len(lines) == len(records)*width and report['failed_configurations'] == 0


if __name__ == '__main__':
    main()
