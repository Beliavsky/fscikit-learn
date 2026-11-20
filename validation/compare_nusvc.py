"""NuSVC decisions, LIBSVM layout and two-group dual constraints."""
from pathlib import Path
import argparse
import json
import subprocess
import numpy as np
import sklearn
from sklearn.svm import NuSVC
from sklearn.metrics.pairwise import rbf_kernel, pairwise_kernels
from sklearn.utils.multiclass import _ovr_decision_function
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"
FLAGS = "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow"


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--executable", type=Path); args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable:
        executable = args.executable.resolve()
    else:
        result = subprocess.run(["fpm", "run", "--flag", FLAGS, "--target", "reference_nusvc_driver", "--runner"],
                                cwd=ROOT, text=True, stdout=subprocess.PIPE, check=True)
        candidates = [ROOT / line.strip().strip('"') for line in result.stdout.splitlines()
                      if line.strip().strip('"').endswith(("reference_nusvc_driver", "reference_nusvc_driver.exe"))]
        candidates = [p for p in candidates if p.is_file()]
        assert len(candidates) == 1, result.stdout
        executable = candidates[0].resolve()
    checks = []
    for seed, n, p in ((7, 48, 3), (42, 60, 4), (2026, 36, 5)):
        rng = np.random.default_rng(seed)
        noise = rng.normal(scale=.3, size=(n, p)); order = rng.permutation(n)
        weights = rng.uniform(.5, 2, n); weights[0] = 0
        for case in range(22):
            k = 2 if case in (0, 1, 2, 6, 9, 11, 13, 15, 18) else 3
            if case == 14: k = 4
            groups = np.arange(n) % k
            if case == 15: groups[:] = 1; groups[:5] = 0
            groups = groups[order]
            labels = np.array([-5, 7, 21, 40])[:k]; y = labels[groups]
            x = noise.copy(); angle = 2*np.pi*groups/k
            x[:, 0] += 2.5*np.cos(angle); x[:, 1] += 2.5*np.sin(angle)
            w = weights.copy() if case in (2, 11, 12, 16) else np.ones(n)
            options = dict(kernel="rbf", gamma="scale", degree=2, coef0=.4, nu=.35,
                           tol=1e-8, max_iter=200000, shrinking=False, decision_function_shape="ovr", break_ties=False)
            if case == 1: options.update(gamma="auto", nu=.6)
            if case == 2: options.update(gamma=.4)
            if case == 3: options["decision_function_shape"] = "ovo"
            if case == 5: options["break_ties"] = True
            if case in (6, 7): options["kernel"] = "linear"
            if case == 8: options.update(kernel="poly", gamma=.2)
            if case == 9: options.update(kernel="sigmoid", gamma=.01, coef0=.05)
            if case == 10: options.update(kernel="precomputed", gamma=.4)
            if case == 11: options.update(gamma=.7, nu=.6)
            if case == 13: options.update(kernel="poly", degree=3, gamma=.1)
            if case == 14: options["decision_function_shape"] = "ovo"
            if case == 15: options["nu"] = .1
            if case == 16: options.update(kernel="precomputed", gamma=.4)
            if case == 17: options.update(kernel="sigmoid", gamma=.01, coef0=.05)
            if case == 18: x[:, -1] = 1
            if case == 19: options["nu"] = .85
            if case == 20: options["nu"] = .5
            if case == 21: options.update(kernel="linear", nu=.37)
            query = np.vstack([x+.137, np.zeros((1, p)), np.full((1, p), 3.)])
            train = x
            if case in (10, 16):
                train = rbf_kernel(x, gamma=.4); query = rbf_kernel(query, x, gamma=.4)
            gamma = options["gamma"]; rule = gamma if isinstance(gamma, str) else "scale"
            data, output = RESULTS / "nusvc_input.txt", RESULTS / "nusvc_actual.txt"
            with data.open("w") as stream:
                stream.write(f"{n} {train.shape[1]} {len(query)} {int(options['break_ties'])}\n")
                stream.write(f"{options['kernel']} {rule} {options['decision_function_shape']}\n")
                stream.write(f"{options['nu']} {gamma if not isinstance(gamma, str) else -1} "
                             f"{options['coef0']} {options['degree']} 1e-9 200000\n")
                np.savetxt(stream, np.column_stack([train, y, w]), fmt="%.17g")
                np.savetxt(stream, query, fmt="%.17g")
            subprocess.run([str(executable), str(data), str(output)], cwd=ROOT, check=True)
            actual = {}
            for line in output.read_text().splitlines():
                name, *values = line.split(); actual[name] = np.array(values, dtype=float)
            fit_train, fit_y, fit_w = train, y, w
            evaluate_train, evaluate_query = train, query
            if case == 16:
                # sklearn 1.6.1's zero-weight precomputed wrapper retains compacted
                # support indices but indexes unreduced query columns. Supply the
                # equivalent explicitly reduced positive-weight Gram to its backend.
                positive = np.flatnonzero(w > 0)
                fit_train = train[np.ix_(positive, positive)]
                fit_y, fit_w = y[positive], w[positive]
                evaluate_train, evaluate_query = train[:, positive], query[:, positive]
            model = NuSVC(**options).fit(fit_train, fit_y, sample_weight=fit_w)
            alpha = actual["alpha"].reshape(n, -1, order="F")
            gram = train if options["kernel"] == "precomputed" else pairwise_kernels(
                train, metric=options["kernel"], filter_params=True, gamma=model._gamma,
                degree=options["degree"], coef0=options["coef0"])
            original_support = np.flatnonzero(w > 0)[model.support_]
            native_support = actual["support"].astype(int)-1
            native_dual = actual["dual"].reshape(k-1, len(native_support), order="F")
            native_full = np.zeros((n, k*(k-1)//2)); source_full = np.zeros_like(native_full); channel = 0
            for a in range(k-1):
                for b in range(a+1, k):
                    for c, row in ((a, b-1), (b, a)):
                        mask = y[native_support] == labels[c]
                        native_full[native_support[mask], channel] = native_dual[row, mask]
                        mask = y[original_support] == labels[c]
                        source_full[original_support[mask], channel] = model.dual_coef_[row, mask]
                    channel += 1
            actual["dual_full"] = native_full.ravel(order="F")
            unscaled = np.zeros_like(alpha); source_unscaled = np.zeros_like(alpha)
            nonunique_normalization = False; pair = 0
            for a in range(k-1):
                for b in range(a+1, k):
                    internal = np.zeros(n)
                    for c, row in ((a, b-1), (b, a)):
                        selected = y[original_support] == labels[c]
                        internal[original_support[selected]] = model.dual_coef_[row, selected]
                    if k == 2: internal *= -1
                    involved = (y == labels[a]) | (y == labels[b])
                    target = options["nu"]*w[involved].sum()/2
                    source_scale = target/internal[y == labels[a]].sum()
                    source_unscaled[:, pair] = internal*source_scale
                    signs = np.where(y == labels[a], 1., -1.)
                    unscaled[:, pair] = alpha[:, pair]*signs
                    gradient = signs*(gram @ unscaled[:, pair])
                    native_bias = actual["intercept"][pair]*(-1 if k == 2 else 1)
                    source_bias = model.intercept_[pair]*(-1 if k == 2 else 1)
                    for c, direction in ((a, -1), (b, 1)):
                        members = (y == labels[c]) & (w > 0)
                        bound_tol = 32*np.finfo(float).eps*w
                        above = members & (alpha[:, pair] > bound_tol)
                        below = members & (alpha[:, pair] < w-bound_tol)
                        lo, hi = gradient[below].min(), gradient[above].max()
                        free = above & below
                        if not np.any(free) and lo-hi > 1e-7: nonunique_normalization = True
                        native_level = actual["scales"][pair]*(1+direction*native_bias)
                        source_level = source_scale*(1+direction*source_bias)
                        assert hi-2e-7 <= native_level <= lo+2e-7, (seed, case, "native margin", pair)
                        assert hi-3e-4 <= source_level <= lo+3e-4, (seed, case, "source margin", pair)
                    pair += 1
            np.testing.assert_allclose(gram @ unscaled, gram @ source_unscaled, atol=3e-4, rtol=3e-4,
                                       err_msg=f"{seed} {case} unscaled primal agreement")
            expected = dict(classes=model.classes_, intercept=model.intercept_,
                            train=model.decision_function(evaluate_train).reshape(n, -1),
                            query=model.decision_function(evaluate_query).reshape(len(query), -1),
                            predict=model.predict(evaluate_query), train_predict=model.predict(evaluate_train),
                            score=[model.score(evaluate_train, y)],
                            meta=[model._gamma, train.shape[1], model.fit_status_])
            # Strictly positive definite RBF fits have identifiable individual duals.
            # Linear/polynomial fits can have equivalent, nonunique support layouts.
            if options["kernel"] in ("rbf", "precomputed"):
                # Floating-point mass accumulation can leave a nominal support
                # coefficient around 1e-15 in LIBSVM. Compare full dual vectors and
                # the meaningful (>1e-10) support metadata instead of that artifact.
                smask = np.any(np.abs(model.dual_coef_) > 1e-10, axis=0)
                nmask = np.any(np.abs(native_dual) > 1e-10, axis=0)
                actual["support_meaningful"] = native_support[nmask]+1
                actual["n_support_meaningful"] = np.array([np.sum(y[native_support[nmask]] == c) for c in labels])
                actual["vectors_meaningful"] = (actual["vectors"].reshape(len(native_support), p, order="F")[nmask].ravel(order="F")
                                                 if options["kernel"] != "precomputed" else actual["vectors"])
                expected.update(support_meaningful=original_support[smask]+1,
                                n_support_meaningful=[np.sum(y[original_support[smask]] == c) for c in labels],
                                dual_full=source_full,
                                vectors_meaningful=model.support_vectors_[smask] if options["kernel"] != "precomputed" else [])
            if options["kernel"] == "linear": expected["coef"] = model.coef_
            if nonunique_normalization:
                # Fully bound solutions admit an interval of rho/r choices. Compare
                # their unscaled primal solutions and both normalizations' KKT ranges.
                for name in ("intercept", "train", "query", "predict", "train_predict", "score", "coef", "dual_full"):
                    expected.pop(name, None)
            for name, value in expected.items():
                want = np.asarray(value).ravel(order="F")
                tolerance = 3e-4 if name in ("dual_full", "coef") else 4e-5
                if name in ("support_meaningful", "n_support_meaningful", "classes", "predict", "train_predict"): tolerance = 0
                np.testing.assert_allclose(actual[name], want, atol=tolerance, rtol=tolerance,
                                           err_msg=f"seed={seed} case={case} {name}")
                checks.append(dict(seed=seed, case=case, name=name, kind="sklearn", passed=True))
            alpha = actual["alpha"].reshape(n, -1, order="F"); pair = 0
            for a in range(k-1):
                for b in range(a+1, k):
                    involved = (y == labels[a]) | (y == labels[b]); target = options["nu"]*w[involved].sum()/2
                    np.testing.assert_allclose(alpha[y == labels[a], pair].sum(), target, atol=1e-9, rtol=1e-9)
                    np.testing.assert_allclose(alpha[y == labels[b], pair].sum(), target, atol=1e-9, rtol=1e-9)
                    assert np.all(alpha[:, pair] >= -1e-12) and np.all(alpha[:, pair] <= w+1e-12)
                    assert np.all(alpha[~involved, pair] == 0)
                    pair += 1
            assert np.all(actual["kkt"] <= 1.001e-9) and np.all(np.isfinite(actual["scales"]))
            public_expected = unscaled/actual["scales"]*(-1 if k == 2 else 1)
            np.testing.assert_allclose(native_full, public_expected, atol=1e-12, rtol=1e-12)
            np.testing.assert_array_equal(actual["n_support"], [np.sum(y[native_support] == c) for c in labels])
            if options["kernel"] == "linear":
                np.testing.assert_allclose(actual["coef"], (native_full.T @ train).ravel(order="F"), atol=1e-12)
            # Independently reconstruct query scores from unscaled duals and the
            # native valid normalization, including sklearn's OVR tie transform.
            gquery = query if options["kernel"] == "precomputed" else pairwise_kernels(
                query, train, metric=options["kernel"], filter_params=True, gamma=model._gamma,
                degree=options["degree"], coef0=options["coef0"])
            raw = gquery @ (unscaled/actual["scales"])
            raw += actual["intercept"]*(-1 if k == 2 else 1)
            if k == 2: mapped = -raw
            elif options["decision_function_shape"] == "ovr": mapped = _ovr_decision_function(raw < 0, -raw, k)
            else: mapped = raw
            np.testing.assert_allclose(actual["query"], mapped.ravel(order="F"), atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose(actual["score"], np.mean(actual["train_predict"] == y), atol=1e-14)
            for name in ("class_masses", "box_bounds", "excluded_rows", "kkt", "native_margin", "source_margin",
                         "unscaled_primal", "public_dual_layout", "public_support_counts", "query_identity", "score_identity"):
                checks.append(dict(seed=seed, case=case, name=name, kind="certificate", passed=True))
    (RESULTS / "nusvc.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} NuSVC result groups passed")


if __name__ == "__main__":
    main()
