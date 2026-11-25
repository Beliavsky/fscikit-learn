"""Gram LARS knots, dropouts, positive constraints and read-only inputs against sklearn."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.linear_model import lars_path_gram

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'validation/results'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--profile', default='checked')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    checks, fixtures, dropouts, nonunique = [], 0, 0, []
    for seed, n, p in [(7, 47, 8), (42, 21, 11), (2026, 9, 14), (0, 12, 10), (91, 20, 10)]:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p)); x[:, 1:] += .8*x[:, :1]
        y = rng.normal(size=n)
        for variant in ['regular', 'duplicate', 'zero_response', 'scaled']:
            train = x.copy(); response = y.copy()
            if variant == 'duplicate':
                train[:, -1] = train[:, 0]
            if variant == 'zero_response':
                response[:] = 0
            if variant == 'scaled':
                train *= .01
            gram = train.T @ train; xy = train.T @ response
            for method, positive in [('lar', False), ('lasso', False), ('lasso', True)]:
                for limit, alpha in [(500, 0), (500, .03), (0, 0), (1, .001), (3, 0), (500, 1000)]:
                    fixtures += 1
                    eps = .1 if p > n and limit == 500 else np.finfo(float).eps
                    options = dict(eps=eps, method=method, positive=positive, max_iter=limit,
                                   alpha_min=alpha, return_n_iter=True)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        alphas, active, coefs, iterations = lars_path_gram(xy.copy(), gram.copy(), n_samples=n, **options)
                    if method == 'lasso':
                        dropouts += int(np.count_nonzero((abs(coefs[:, :-1]) > 1e-8) & (abs(coefs[:, 1:]) < 1e-10)))
                    inp = RESULTS / 'lars_gram_input.txt'; out = RESULTS / 'lars_gram_output.txt'
                    with inp.open('w') as stream:
                        stream.write(f'{n} {p} {limit} {alpha} {eps} {int(positive)} {method}\n')
                        np.savetxt(stream, xy[None, :], fmt='%.17g')
                        np.savetxt(stream, gram, fmt='%.17g')
                    proc = subprocess.run([str(args.executable.resolve()), str(inp), str(out)],
                                          capture_output=True, text=True)
                    assert proc.returncode == 0, (seed, variant, options, proc.stdout, proc.stderr)
                    actual = {line.split()[0]: np.array(line.split()[1:], dtype=float)
                              for line in out.read_text().splitlines()}
                    expected = dict(alphas=alphas, path=coefs.ravel(order='F'), active=np.array(active)+1,
                                    iterations=[iterations])
                    if variant == 'duplicate':
                        # Identical columns admit interchangeable active representatives.
                        # Certify the entire prediction path and merged coefficients.
                        native = actual['path'].reshape(p, -1, order='F')
                        np.testing.assert_allclose(train @ native, train @ coefs, rtol=3e-8, atol=3e-8)
                        merged_native = native[:-1].copy(); merged_native[0] += native[-1]
                        merged_source = coefs[:-1].copy(); merged_source[0] += coefs[-1]
                        actual['path'] = merged_native.ravel(order='F')
                        expected['path'] = merged_source.ravel(order='F')
                        actual['active'][actual['active'] == p] = 1
                        expected['active'][expected['active'] == p] = 1
                        nonunique.append(dict(seed=seed, method=method, positive=positive,
                                              max_iter=limit, alpha_min=alpha))
                    for key, want in expected.items():
                        np.testing.assert_allclose(actual[key], want, rtol=3e-8, atol=3e-8,
                                                   err_msg=str((seed, variant, options, key)))
                        checks.append(dict(seed=seed, variant=variant, method=method, positive=positive,
                                           max_iter=limit, alpha_min=alpha, field=key))
    assert dropouts > 0
    source = Path(inspect.getfile(inspect.unwrap(lars_path_gram)))
    report = dict(sklearn=sklearn.__version__, profile=args.profile, fixtures=fixtures,
                  groups=len(checks), dropout_events=dropouts, checks=checks,
                  duplicate_column_prediction_certificates=nonunique,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                  native_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [ROOT/"src/sklearn_least_angle.f90",
                                           ROOT/"app/reference_lars_gram_driver.f90", Path(__file__)]})
    (RESULTS / f'lars_gram_{args.profile}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'{len(checks)} Gram LARS groups across {fixtures} fixtures passed; {dropouts} dropouts')


if __name__ == '__main__':
    main()
