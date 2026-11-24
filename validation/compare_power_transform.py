"""PowerTransformer reference checks against the installed sklearn baseline."""
from pathlib import Path
import argparse,hashlib
import json
import subprocess
import numpy as np
import sklearn
from sklearn.preprocessing import PowerTransformer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);args=ap.parse_args()
    paths=[Path(__file__),ROOT/'src/sklearn_power_transform.f90',ROOT/'app/reference_power_transform_driver.f90']
    if args.executable:paths.append(args.executable)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();hashes={str(p):sha(p) for p in paths}
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 45, 3), (42, 60, 4), (2026, 80, 5), (91, 70, 3), (92, 70, 3)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p)) * np.linspace(.3, 1.3, p) + np.linspace(-2, 2, p)
        if seed in (91, 92):
            x = rng.lognormal(mean=-.3, sigma=.9, size=(n, p))
            if seed == 92:
                x = -x
        data, output = RESULTS/f"power_{args.profile}_{seed}.txt", RESULTS/f"power_actual_{args.profile}_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, x, fmt="%.17g")
        command=([str(args.executable)] if args.executable else ["fpm","run","--target","reference_power_transform_driver","--"])
        subprocess.run(command+[str(data),str(output)],cwd=ROOT,check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        expected = {}
        for j in range(1, 9):
            train, query = x.copy(), x + .17
            method = "yeo-johnson"
            if j in (3, 4):
                train, query, method = np.exp(x), np.exp(x+.17), "box-cox"
            if j in (5, 6):
                train[1, 0], query[2, 0] = np.nan, np.nan
            if j in (7, 8):
                train[:, -1], query[:, -1] = 3, 3.17
            model = PowerTransformer(method=method, standardize=j % 2 == 1)
            z = model.fit_transform(train)
            expected.update({f"power{j}_lambda": model.lambdas_, f"power{j}_train": z,
                             f"power{j}_query": model.transform(query),
                             f"power{j}_inverse": model.inverse_transform(z)})
        assert actual.keys() == expected.keys()
        for name, want in expected.items():
            np.testing.assert_allclose(actual[name], np.asarray(want).ravel(order="F"),
                                       atol=3e-6, rtol=3e-6, equal_nan=True, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, api=name, passed=True))
        print(f"PASS PowerTransformer seed={seed}", flush=True)
    assert hashes=={str(p):sha(p) for p in paths}
    (args.report or RESULTS/"power_transform_report.json").write_text(json.dumps(
        dict(profile=args.profile,hashes=hashes,sklearn_version=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} power-transform comparisons passed against sklearn {sklearn.__version__}")


if __name__ == "__main__":
    main()
