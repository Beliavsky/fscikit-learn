"""Dense streaming PCA comparisons against installed scikit-learn."""
from pathlib import Path
import json
import argparse
import subprocess
import numpy as np
import sklearn
from sklearn.decomposition import IncrementalPCA
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--report', type=Path, default=RESULTS/'incremental_pca.json')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 37, 5), (42, 43, 4), (2026, 29, 8)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))*np.arange(1, p+1)+np.arange(p)*3
        data = args.report.with_name(args.report.stem+f'_{seed}.in')
        output = args.report.with_name(args.report.stem+f'_{seed}.out')
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, x, fmt="%.17g")
        command = ([str(args.executable), str(data), str(output)] if args.executable else
                   ["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow",
                    "--target", "reference_incremental_pca_driver", "--", str(data), str(output)])
        subprocess.run(command, cwd=ROOT, check=True)
        actual, expected = {}, {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        for j in range(1, 9):
            options = dict(n_components=2)
            if j == 2: options["batch_size"] = 7
            if j == 3: options.update(batch_size=9, whiten=True)
            if j == 4: options.update(n_components=p, batch_size=p+2)
            if j == 5: options["batch_size"] = 8
            if j == 6: options.update(n_components=None, batch_size=p+2)
            if j == 7: options.update(n_components=1, batch_size=1)
            if j == 8: options["batch_size"] = 11
            model = IncrementalPCA(**options)
            if j == 5:
                model.partial_fit(x[:9]); model.partial_fit(x[9:10]); model.partial_fit(x[10:])
            elif j == 7:
                model.partial_fit(x[:3])
                for row in x[3:]: model.partial_fit(row[None, :])
            else: model.fit(x)
            z = model.transform(x)
            values = dict(components=model.components_, mean=model.mean_, var=model.var_,
                          singular=model.singular_values_, variance=model.explained_variance_,
                          ratio=model.explained_variance_ratio_, noise=[model.noise_variance_],
                          train=z, inverse=model.inverse_transform(z), query=model.transform(x+.137),
                          shape=[model.n_samples_seen_, model.n_features_in_, model.n_components_])
            expected.update({f"case{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            assert actual[name].shape == want.shape, (seed, name)
            np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    args.report.write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} IncrementalPCA result groups passed")

if __name__ == "__main__":
    main()
