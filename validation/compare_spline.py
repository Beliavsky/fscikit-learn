"""Native B-spline basis and extrapolation comparisons."""
from pathlib import Path
import json
import argparse,hashlib
import subprocess
import numpy as np
import sklearn
from sklearn.preprocessing import SplineTransformer
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path);ap.add_argument('--profile',default='development');ap.add_argument('--report',type=Path);args=ap.parse_args()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();paths=[Path(__file__),ROOT/'src/sklearn_spline.f90',ROOT/'app/reference_spline_driver.f90']
    if args.executable:paths.append(args.executable)
    hashes={str(p):sha(p) for p in paths}
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 45, 2), (42, 60, 3), (2026, 24, 4)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))
        data, output = RESULTS/f"spline_{args.profile}_{seed}.txt", RESULTS/f"spline_actual_{args.profile}_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, x, fmt="%.17g")
        command=[str(args.executable)] if args.executable else ['fpm','run','--target','reference_spline_driver','--']
        subprocess.run(command+[str(data),str(output)],cwd=ROOT,check=True)
        actual, expected = {}, {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        low, high = x.min(axis=0), x.max(axis=0)
        custom = low+(high-low)*np.array([0., .1, .35, .8, 1.])[:, None]
        for j in range(1, 13):
            options = {}
            query = low+(high-low)*(-.5+np.arange(11)[:, None]/5)
            if j == 2: options["degree"] = 1
            if j == 3: options.update(degree=2, include_bias=False)
            if j == 4: options["knots"] = "quantile"
            if j == 5: options["extrapolation"] = "linear"
            if j == 6: options["extrapolation"] = "continue"
            if j == 7: options["extrapolation"] = "periodic"
            if j == 8: options.update(extrapolation="periodic", include_bias=False)
            if j == 9: options["knots"] = custom
            if j == 10: options.update(knots=custom, extrapolation="periodic", degree=2)
            if j == 11: options["extrapolation"] = "error"; query = np.clip(query, low, high)
            if j == 12: options.update(degree=10, extrapolation="continue")
            model = SplineTransformer(**options)
            fitted = model.fit_transform(x); result = model.transform(query)
            extended = np.column_stack([s.t for s in model.bsplines_])
            base = extended[model.degree:-model.degree]
            values = dict(train=fitted, query=result, knots=extended, base=base,
                          shape=[model.n_features_in_, model.n_features_out_])
            expected.update({f"case{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            assert actual[name].shape == want.shape, (seed, name, actual[name].shape, want.shape)
            np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    assert hashes=={str(p):sha(p) for p in paths}
    (args.report or RESULTS/"spline.json").write_text(json.dumps(dict(profile=args.profile,hashes=hashes,sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} spline result groups passed")

if __name__ == "__main__":
    main()
