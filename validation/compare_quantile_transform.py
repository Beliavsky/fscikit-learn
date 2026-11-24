"""QuantileTransformer percentiles, duplicate interpolation and tail conventions."""
from pathlib import Path
import argparse,hashlib
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);args=ap.parse_args()
    paths=[Path(__file__),ROOT/'src/sklearn_quantile_transform.f90',ROOT/'app/reference_quantile_transform_driver.f90']
    if args.executable:paths.append(args.executable)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();hashes={str(p):sha(p) for p in paths}
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 45, 2), (42, 54, 3), (2026, 63, 4)):
        rng = np.random.default_rng(seed)
        base = rng.lognormal(size=(n, p))-1
        cases = []
        for j in range(1, 13):
            x = base.copy()
            if j == 8: x[:, -1] = 2
            if j in (9, 10):
                x[::5, :] = np.nan
                x[:, -1] = np.nan
            if j in (11, 12): x = np.round(x, 0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lower, upper = np.nanmin(x, axis=0), np.nanmax(x, axis=0)
            query = x+.137
            query[0] = lower-.5; query[-1] = upper+.5
            query[2] = lower+5e-8; query[3] = upper-5e-8
            if j in (9, 10): query[:, -1] = 0  # finite queries into an all-missing table
            normal = j % 2 == 0
            probe = np.broadcast_to(np.linspace(-8, 8, n)[:, None] if normal else
                                    np.linspace(-.2, 1.2, n)[:, None], (n, p)).copy()
            if j in (9, 10): probe[4, 0] = np.nan
            nq = n+10 if j == 3 else n if j == 4 else 1 if j in (5, 6) else 3 if j == 7 else 9
            cases.append((x, query, probe, nq, "normal" if normal else "uniform"))
        data, output = RESULTS/f"quantile_{args.profile}_{seed}.txt", RESULTS/f"quantile_actual_{args.profile}_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            for x, query, probe, _, _ in cases:
                for matrix in (x, query, probe): np.savetxt(stream, matrix, fmt="%.17g")
        command=([str(args.executable)] if args.executable else ["fpm", "run", "--flag",
            "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow", "--target", "reference_quantile_transform_driver", "--"])
        subprocess.run(command+[str(data),str(output)],cwd=ROOT,check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        expected = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j, (x, query, probe, nq, distribution) in enumerate(cases, 1):
                model = QuantileTransformer(n_quantiles=nq, subsample=None, output_distribution=distribution)
                train = model.fit_transform(x)
                values = dict(train=train, roundtrip=model.inverse_transform(train), references=model.references_,
                              quantiles=model.quantiles_, metadata=[model.n_quantiles_, p],
                              query=model.transform(query), inverse=model.inverse_transform(probe))
                expected.update({f"quantile{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            got = actual[name]
            assert got.shape == want.shape, (seed, name, got.shape, want.shape)
            np.testing.assert_allclose(got, want, atol=2e-10, rtol=2e-11, equal_nan=True, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    assert hashes=={str(p):sha(p) for p in paths}
    (args.report or RESULTS/"quantile_transform.json").write_text(json.dumps(dict(profile=args.profile,hashes=hashes,sklearn=sklearn.__version__, checks=checks),
                                                           indent=2)+"\n")
    print(f"{len(checks)} QuantileTransformer result groups passed")

if __name__ == "__main__":
    main()
