"""Feature clustering, pooling and reconstruction against scikit-learn."""
from pathlib import Path
import json
import argparse
import subprocess
import numpy as np
import sklearn
from sklearn.cluster import FeatureAgglomeration
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--executable")
    parser.add_argument("--report",type=Path,default=RESULTS/"feature_agglomeration.json")
    args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 37, 8), (42, 19, 6), (2026, 5, 11)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))*np.arange(1, p+1)+np.arange(p)*.3
        data, output = RESULTS/f"feature_agg_{seed}.txt", RESULTS/f"feature_agg_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, x, fmt="%.17g")
        command=([args.executable] if args.executable else ["fpm", "run", "--flag",
                 "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow", "--target",
                 "reference_feature_agglomeration_driver", "--"])
        subprocess.run(command+[str(data),str(output)],cwd=ROOT,check=True)
        actual, expected = {}, {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        for j in range(1, 15):
            options = dict(n_clusters=3, compute_distances=True)
            if j <= 12:
                options.update(linkage=("ward", "single", "complete", "average")[(j-1)//3],
                               pooling_func=(np.mean, np.median, np.max)[(j-1)%3])
            else: options["n_clusters"] = 1 if j == 13 else p
            model = FeatureAgglomeration(**options).fit(x)
            # Native cluster IDs follow first-feature order; sklearn's IDs are arbitrary.
            order = sorted(range(model.n_clusters_), key=lambda k: np.flatnonzero(model.labels_ == k)[0])
            labels = np.empty_like(model.labels_)
            for canonical, original in enumerate(order): labels[model.labels_ == original] = canonical
            z = model.transform(x)
            values = dict(labels=labels, children=np.sort(model.children_, axis=1), distances=model.distances_,
                          train=z[:, order], query=model.transform(x+.137)[:, order],
                          inverse=model.inverse_transform(z),
                          shape=[model.n_features_in_, model.n_clusters_, model.n_leaves_, model.n_connected_components_])
            expected.update({f"case{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            assert actual[name].shape == want.shape, (seed, name)
            np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    args.report.write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} FeatureAgglomeration result groups passed")

if __name__ == "__main__":
    main()
