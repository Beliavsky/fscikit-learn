"""Dense KBinsDiscretizer reference checks against sklearn 1.6.1."""
from pathlib import Path
import json
import os
import subprocess
import warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import sklearn
from sklearn.preprocessing import KBinsDiscretizer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"


def main():
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 70, 3), (42, 80, 4), (2026, 90, 5)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))*np.linspace(.8, 2.2, p)+np.linspace(-2, 2, p)
        w = rng.uniform(.2, 2, size=n)
        w[::11] = 0
        data, output = RESULTS/f"bins_{seed}.txt", RESULTS/f"bins_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, np.column_stack([x, w]), fmt="%.17g")
        subprocess.run(["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                        "--target", "reference_discretization_driver", "--", str(data), str(output)],
                       cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        expected = {}
        for j in range(1, 13):
            train, query = x.copy(), x+.137
            encode = "onehot-dense" if j in (2, 4, 6, 8, 11, 12) else "ordinal"
            strategy = "uniform"
            if j in (3, 4, 7, 10, 11):
                strategy = "quantile"
            if j in (5, 6, 8, 12):
                strategy = "kmeans"
            if j == 10:
                train = np.floor(x)
                query = train+.137
            if j == 11:
                train[:, -1], query[:, -1] = 3, 3.137
            counts = 2+np.arange(1, p+1) % 4 if j in (9, 12) else 4
            model = KBinsDiscretizer(n_bins=counts, strategy=strategy, encode=encode, subsample=None)
            with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                warnings.simplefilter("ignore")
                model.fit(train, sample_weight=w if j in (7, 8) else None)
                z = model.transform(train)
                expected.update({f"bins{j}_counts": model.n_bins_, f"bins{j}_width": [z.shape[1]],
                                 f"bins{j}_train": z, f"bins{j}_query": model.transform(query),
                                 f"bins{j}_inverse": model.inverse_transform(z)})
                for h, edges in enumerate(model.bin_edges_, 1):
                    expected[f"bins{j}_edges{h}"] = edges
        assert actual.keys() == expected.keys()
        for name, want in expected.items():
            np.testing.assert_allclose(actual[name], np.asarray(want).ravel(order="F"),
                                       atol=2e-10, rtol=2e-10, equal_nan=True, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, api=name, passed=True))
        print(f"PASS KBinsDiscretizer seed={seed}", flush=True)
    (RESULTS/"discretization_report.json").write_text(json.dumps(
        dict(sklearn_version=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} discretization comparisons passed against sklearn {sklearn.__version__}")


if __name__ == "__main__":
    main()
