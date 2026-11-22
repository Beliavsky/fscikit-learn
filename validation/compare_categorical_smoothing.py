"""CategoricalNB smoothing, priors and streaming edge cases against installed sklearn."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.naive_bayes import CategoricalNB
from compare_function_transformer import ROOT, resolve
from compare_gp_kernels import block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['checked', 'release'], default='checked')
    parser.add_argument('--executable', type=Path)
    args = parser.parse_args()
    exe = args.executable.resolve() if args.executable else resolve(args.profile, 'reference_categorical_smoothing_driver')
    checks, counts, undefined, ties, equivalence = [], [], [], [], []
    fixtures = snapshots = 0
    for seed in [0, 42, 2026]:
        rng = np.random.RandomState(seed)
        for n, p in [(12, 1), (18, 3), (9, 7)]:
            for classes_count in [1, 2, 3]:
                x = rng.randint(0, 6, (n, p)); x[0] = 0
                # Later smaller ranges test metadata independently from retained table widths.
                x[n//2:] %= 2; x[1] = 5
                y = np.arange(n) % classes_count * 5 - 2
                q = rng.randint(0, 12, (5, p)); q[0] = 0; q[1] = 5
                weights = rng.uniform(.1, 3, n); weights[::4] = 0
                for alpha, force in [(0., True), (0., False), (1e-15, True), (1e-15, False), (.1, True), (3., True)]:
                    for scenario in range(8):
                        stream = scenario in [1, 2, 3, 6, 7]
                        weighted = scenario != 0
                        fitprior = scenario not in [2, 3, 7]
                        fixed = scenario in [3, 5]
                        declared = np.unique(y)
                        if scenario == 1:
                            declared = np.r_[declared, declared[-1]+5]
                        w = np.zeros(n) if scenario in [4, 6, 7] else weights
                        prior = np.arange(1, len(declared)+1)*.4
                        if scenario == 3 and len(prior) > 1: prior[0] = 0
                        if scenario == 5: prior[:] = 0
                        mode = fixtures % 3
                        minimum = None if mode == 0 else (4 if mode == 1 else np.arange(p)+2)
                        minima = np.zeros(p, dtype=int) if mode != 2 else minimum
                        ends = [1, n//2, n] if stream else [n]
                        model = CategoricalNB(alpha=alpha, force_alpha=force, fit_prior=fitprior,
                                              class_prior=prior if fixed else None, min_categories=minimum)
                        payload = f'{n} {p} {len(q)} {len(declared)} {len(ends)} {int(stream)} {int(weighted)} {int(fixed)} {int(fitprior)} {int(force)} {alpha:.17g} {4 if mode == 1 else 0}\n'
                        for values in [x.ravel(order='F'), y, q.ravel(order='F'), w, declared[::-1], ends, minima]:
                            payload += ' '.join(f'{v:.17g}' for v in values)+'\n'
                        if fixed: payload += ' '.join(f'{v:.17g}' for v in prior)+'\n'
                        run = subprocess.run([str(exe)], input=payload, text=True, capture_output=True, cwd=ROOT, check=True)
                        lines = iter(run.stdout.splitlines()); start = 0
                        for stage, end in enumerate(ends):
                            with warnings.catch_warnings(record=True), np.errstate(divide='ignore', invalid='ignore'):
                                if stream:
                                    model.partial_fit(x[start:end], y[start:end],
                                                      classes=declared[::-1] if stage < 2 else None,
                                                      sample_weight=w[start:end] if weighted else None)
                                else:
                                    model.fit(x, y, sample_weight=w if weighted else None)
                                widths = np.array([table.shape[1] for table in model.category_count_])
                                query = q % widths
                                joint = model._joint_log_likelihood(query)
                                train_joint = model._joint_log_likelihood(x[:end])
                                expected = dict(classes=model.classes_, class_count=model.class_count_,
                                                prior=model.class_log_prior_, categories=model.n_categories_,
                                                features=[model.n_features_in_])
                                for j in range(p):
                                    expected[f'count{j}'] = model.category_count_[j]
                                    expected[f'feature{j}'] = model.feature_log_prob_[j]
                                expected.update(joint=joint, logp=model.predict_log_proba(query),
                                                proba=model.predict_proba(query), predict=model.predict(query),
                                                train_predict=model.predict(x[:end]), score=[model.score(x[:end], y[:end])])
                            actual = {}
                            for name, value in expected.items():
                                got = block(lines); want = np.asarray(value)
                                if want.ndim == 1: want = want[:, None]
                                assert got.shape == want.shape, (fixtures, stage, name, got.shape, want.shape)
                                actual[name] = got
                                if name in ['predict', 'train_predict']:
                                    scores = joint if name == 'predict' else train_joint
                                    for row, (a, b) in enumerate(zip(got[:, 0], want[:, 0])):
                                        assert a in model.classes_
                                        if a == b: continue
                                        assert not np.isnan(scores[row]).any(), (fixtures, stage, name, row)
                                        maximum = np.max(scores[row]); selected = scores[row, np.searchsorted(model.classes_, a)]
                                        bound = 64*np.finfo(float).eps*max(1., np.max(np.abs(scores[row, np.isfinite(scores[row])])))
                                        gap = float(maximum-selected)
                                        assert np.isfinite(gap) and 0 <= gap <= bound, (fixtures, stage, name, row, gap, bound)
                                        ties.append(dict(fixture=fixtures, stage=stage, name=name, row=row, gap=gap, bound=float(bound), passed=True))
                                elif name == 'score':
                                    np.testing.assert_allclose(got[0, 0], np.mean(actual['train_predict'][:, 0] == y[:end]), atol=0, rtol=0)
                                else:
                                    np.testing.assert_allclose(got, want, atol=3e-10, rtol=3e-10, equal_nan=True,
                                                               err_msg=f'{fixtures} {stage} {name}')
                                if np.isnan(want).any():
                                    assert np.array_equal(np.isnan(got), np.isnan(want))
                                    undefined.append(dict(fixture=fixtures, stage=stage, name=name, passed=True))
                                checks.append(dict(fixture=fixtures, stage=stage, name=name, passed=True))
                            mass = w[:end] if weighted else np.ones(end)
                            for j in range(p):
                                per_class = actual[f'count{j}'].sum(axis=1)
                                want = np.array([mass[y[:end] == label].sum() for label in model.classes_])
                                np.testing.assert_allclose(per_class, want, atol=2e-11, rtol=2e-11)
                            counts.append(dict(fixture=fixtures, stage=stage, passed=True))
                            start = end; snapshots += 1
                        assert next(lines, None) is None
                        if stream and np.array_equal(declared, np.unique(y)):
                            with warnings.catch_warnings(record=True), np.errstate(divide='ignore', invalid='ignore'):
                                batch = CategoricalNB(**model.get_params()).fit(x, y, sample_weight=w if weighted else None)
                            for j in range(p):
                                np.testing.assert_allclose(model.category_count_[j], batch.category_count_[j], atol=3e-11, rtol=3e-11)
                                np.testing.assert_allclose(model.feature_log_prob_[j], batch.feature_log_prob_[j], atol=3e-10, rtol=3e-10)
                            np.testing.assert_allclose(model.class_log_prior_, batch.class_log_prior_, atol=3e-10, rtol=3e-10)
                            equivalence.append(dict(fixture=fixtures, passed=True))
                        fixtures += 1
                        if fixtures % 72 == 0: print(f'{fixtures} categorical smoothing fixtures passed', flush=True)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(profile=args.profile, sklearn_version=sklearn.__version__, fixtures=fixtures, snapshots=snapshots,
                  checks=checks, count_certificates=counts, undefined_certificates=undefined,
                  label_certificates=ties, batch_equivalence=equivalence, native_sha256=sha(exe),
                  validator_sha256=sha(Path(__file__)),
                  source_sha256={'naive_bayes.py': sha(Path(sklearn.__file__).parent/'naive_bayes.py')})
    destination = ROOT/'validation/results'; destination.mkdir(exist_ok=True)
    (destination/f'categorical_smoothing_{args.profile}.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'{len(checks)} groups, {fixtures} fixtures, {snapshots} snapshots, {len(undefined)} undefined fields, '
          f'{len(ties)} tied labels and {len(equivalence)} batch equivalence cases PASS')


if __name__ == '__main__':
    main()
