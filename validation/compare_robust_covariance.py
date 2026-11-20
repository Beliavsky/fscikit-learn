"""FastMCD, MinCovDet and EllipticEnvelope seeded sklearn comparisons."""
from pathlib import Path
import argparse
import json
import subprocess
import warnings
import numpy as np
import sklearn
from scipy.linalg import pinvh
from scipy.stats import chi2
from sklearn.covariance import MinCovDet, EllipticEnvelope, fast_mcd, empirical_covariance
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"
FLAGS = "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow"


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--executable", type=Path); args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable:
        executable = args.executable.resolve()
    else:
        result = subprocess.run(["fpm", "run", "--flag", FLAGS, "--target", "reference_robust_covariance_driver", "--runner"],
                                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(result.stdout)
        paths = [ROOT / s.strip().strip('"') for s in result.stdout.splitlines()
                 if s.strip().strip('"').endswith(("reference_robust_covariance_driver", "reference_robust_covariance_driver.exe"))]
        paths = [p for p in paths if p.is_file()]; assert len(paths) == 1, result.stdout
        executable = paths[0].resolve()
    checks = []
    for seed, n, p in ((7, 71, 3), (42, 93, 4), (2026, 67, 1), (0, 601, 2), (17, 1601, 2)):
        rng = np.random.default_rng(seed)
        original = rng.normal(size=(n, p)) @ (np.eye(p)+.15*np.tril(np.ones((p,p)), -1))
        original[:n//8] = original[:n//8]*2+8
        for case in range(4 if n > 500 else (8 if p == 1 else 9)):
            x = original.copy()+np.arange(p)+1
            fraction = None; centered = case % 2 == 1; store = case % 3 != 1
            contamination = .1
            if case in (2, 3): fraction = .7
            if case == 4: fraction = 1.
            if case == 5: x = original.copy()
            if case == 6: contamination = .25
            if case == 7: x *= 10; fraction = .8
            if case == 8: x[:, -1] = 0
            query = np.vstack([x+.137, np.zeros((1,p)), np.full((1,p), 15.)])
            labels = np.ones(len(query), dtype=int); labels[:n//8] = -1; labels[-1] = -1
            weights = rng.uniform(.2, 2, len(query)); weights[0] = 0
            data, output = RESULTS / "robust_covariance_input.txt", RESULTS / "robust_covariance_actual.txt"
            with data.open("w") as stream:
                stream.write(f"{n} {p} {len(query)} {int(centered)} {int(store)}\n")
                stream.write(f"{fraction or 0} {seed} {contamination}\n")
                np.savetxt(stream, x, fmt="%.17g")
                np.savetxt(stream, np.column_stack([query, labels, weights]), fmt="%.17g")
            subprocess.run([str(executable), str(data), str(output)], cwd=ROOT, check=True)
            actual = {}
            for line in output.read_text().splitlines():
                name, *values = line.split(); actual[name] = np.array(values, dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", (RuntimeWarning, UserWarning))
                raw = fast_mcd(x, support_fraction=fraction, random_state=seed)
                model = MinCovDet(support_fraction=fraction, assume_centered=centered,
                                  store_precision=store, random_state=seed).fit(x)
                envelope = EllipticEnvelope(support_fraction=fraction, assume_centered=centered,
                                            store_precision=store, contamination=contamination, random_state=seed).fit(x)
            raw_dist = raw[3]
            if centered:
                raw_cov = empirical_covariance(x[raw[2]], assume_centered=True)
                raw_dist = np.sum((x @ pinvh(raw_cov))*x, axis=1)
            correction = np.median(raw_dist)/chi2.isf(.5, p)
            comp = model.covariance_.copy(); comp[0,0] += .1
            expected = dict(fast_location=raw[0], fast_covariance=raw[1], fast_support=raw[2].astype(int), fast_dist=raw[3],
                            raw_location=model.raw_location_, raw_covariance=model.raw_covariance_,
                            raw_support=model.raw_support_.astype(int), location=model.location_, covariance=model.covariance_,
                            precision=model.get_precision(), support=model.support_.astype(int), dist=model.dist_,
                            mahalanobis=model.mahalanobis(query), likelihood=[model.score(query)],
                            diagnostics=[correction, chi2.isf(.025, p), np.linalg.matrix_rank(model.covariance_), p, int(store)],
                            norms=[model.error_norm(comp), model.error_norm(comp, norm="spectral"),
                                   model.error_norm(comp, scaling=False, squared=False)],
                            envelope_location=envelope.location_, envelope_covariance=envelope.covariance_,
                            samples=envelope.score_samples(query), decision=envelope.decision_function(query),
                            predict=envelope.predict(query), offset=[envelope.offset_],
                            accuracy=[envelope.score(query, labels), envelope.score(query, labels, sample_weight=weights)])
            tie_fields = set()
            if p == 1 and not np.array_equal(actual["fast_support"], raw[2]):
                chosen = actual["fast_support"].astype(bool)
                h = int(fraction*n) if fraction is not None else int(np.ceil((n+p+1)/2))
                assert chosen.sum() == h
                absolute = np.abs(x[:, 0]-raw[0][0]); boundary = np.sort(absolute)[h-1]
                changed = chosen != raw[2]
                np.testing.assert_allclose(absolute[changed], boundary, atol=8*np.finfo(float).eps*max(1., boundary), rtol=0)
                assert np.max(absolute[chosen]) <= np.min(absolute[~chosen])+8*np.finfo(float).eps*max(1., boundary)
                native_fast_cov = empirical_covariance(x[chosen])
                native_fast_dist = np.sum(((x-raw[0]) @ pinvh(native_fast_cov))*(x-raw[0]), axis=1)
                native_raw_cov = empirical_covariance(x[chosen], assume_centered=centered)
                native_raw_location = np.zeros(p) if centered else raw[0]
                centered_x = x-native_raw_location
                native_raw_dist = np.sum((centered_x @ pinvh(native_raw_cov))*centered_x, axis=1)
                expected.update(fast_covariance=native_fast_cov, fast_support=chosen.astype(int), fast_dist=native_fast_dist,
                                raw_covariance=native_raw_cov, raw_support=chosen.astype(int))
                expected["diagnostics"][0] = np.median(native_raw_dist)/chi2.isf(.5, p)
                tie_fields = {"fast_covariance", "fast_support", "fast_dist", "raw_covariance", "raw_support", "diagnostics"}
            assert actual.keys() == expected.keys(), (seed, case, actual.keys() ^ expected.keys())
            for name, value in expected.items():
                tolerance = 0 if name in ("fast_support", "raw_support", "support", "predict") else 3e-7
                np.testing.assert_allclose(actual[name], np.asarray(value).ravel(order="F"), atol=tolerance, rtol=tolerance,
                                           err_msg=f"seed={seed} case={case} {name}")
                checks.append(dict(seed=seed, case=case, name=name, kind="tie_certificate" if name in tie_fields else "sklearn", passed=True))
    (RESULTS / "robust_covariance.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} robust covariance result groups passed")


if __name__ == "__main__":
    main()
