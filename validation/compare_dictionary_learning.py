"""SparsePCA, DictionaryLearning, sparse coding and legacy resampling streams."""
from pathlib import Path
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.decomposition import SparsePCA, DictionaryLearning, SparseCoder, sparse_encode
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"
FLAGS = "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow"


def run_driver(target, *args):
    subprocess.run(["fpm", "run", "--flag", FLAGS, "--target", target, "--", *map(str, args)],
                   cwd=ROOT, check=True)


def read_output(path):
    result = {}
    for line in path.read_text().splitlines():
        name, *values = line.split(); result[name] = np.array(values, dtype=float)
    return result


def compare_rng():
    checks = []
    for seed in (0, 42, 2026, 2147483647):
        output = RESULTS / f"legacy_rng_{seed}.txt"
        run_driver("reference_legacy_rng_driver", seed, output)
        actual = read_output(output)
        expected = dict(raw=np.random.RandomState(seed).randint(0, 2**32, size=2048, dtype=np.uint32),
                        uniform=np.random.RandomState(seed).random_sample(2048),
                        normal=np.random.RandomState(seed).normal(size=2048),
                        index=np.random.RandomState(seed).choice(17, size=2048)+1)
        rng = np.random.RandomState(seed); indices, normals, uniforms = [], [], []
        for i in range(1, 1001):
            indices.append(rng.choice(i % 57+1)+1); normals.append(rng.normal()); uniforms.append(rng.random_sample())
        expected.update(mixed_index=indices, mixed_normal=normals, mixed_uniform=uniforms)
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            if "normal" in name:
                np.testing.assert_allclose(actual[name], value, atol=2e-14, rtol=2e-14, err_msg=f"{seed} {name}")
            else:
                np.testing.assert_array_equal(actual[name], value, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    (RESULTS / "legacy_rng.json").write_text(json.dumps(dict(numpy=np.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} separate NumPy RandomState stream groups passed", flush=True)


def main():
    RESULTS.mkdir(exist_ok=True); compare_rng(); checks = []
    for seed, n, p in ((7, 31, 7), (42, 18, 9), (2026, 7, 11), (91, 31, 21)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))*np.linspace(.7, 1.4, p)+np.linspace(-.5, .3, p)
        x[:, 1:] += .3*x[:, :1]
        atoms = np.sin(.37*np.arange(1, 5)[:, None]*np.arange(1, p+1))+.1*np.cos(
            np.arange(1, 5)[:, None]+np.arange(1, p+1))
        atoms /= np.linalg.norm(atoms, axis=1)[:, None]
        warm = .1*np.sin(.21*np.arange(1, n+1)[:, None]*np.arange(1, 5))
        data = RESULTS / f"dictionary_{seed}.txt"
        output = RESULTS / f"dictionary_actual_{seed}.txt"
        np.savetxt(data, x, header=f"{n} {p} {seed}", comments="", fmt="%.17g")
        run_driver("reference_sparse_dictionary_driver", data, output)
        actual, expected = read_output(output), {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for j in range(1, 35):
                if j <= 12:
                    options = dict(n_components=3, alpha=.3, max_iter=20, random_state=seed)
                    if j in (2, 5, 8, 11): options["method"] = "cd"
                    if j == 3: options.update(n_components=None, alpha=.1, max_iter=12)
                    elif j == 4: options.update(n_components=p+2, alpha=.2, max_iter=12)
                    elif j == 5: options.update(alpha=.8, max_iter=35, tol=1e-4, ridge_alpha=.2)
                    elif j == 6: options.update(alpha=1e5, max_iter=8)
                    elif j == 7: options["max_iter"] = 0
                    elif j == 8: options.update(n_components=2, alpha=.2, max_iter=12, ridge_alpha=0)
                    elif j == 9: options.update(alpha=.4, tol=.1)
                    elif j in (10, 11): options.update(max_iter=15, U_init=x[:, :3].copy(), V_init=atoms[:3].copy())
                    elif j == 12: options.update(alpha=0, max_iter=3, ridge_alpha=0)
                    model = SparsePCA(**options)
                    z = model.fit_transform(x)
                    values = dict(components=model.components_, mean=model.mean_, errors=model.error_,
                                  shape=[model.n_features_in_, model.n_components_, model.n_iter_],
                                  train=z, query=model.transform(x+.137), inverse=model.inverse_transform(z))
                elif j <= 24:
                    options = dict(n_components=3, alpha=.3, max_iter=20, random_state=seed)
                    if j in (14, 16, 17, 20, 21, 23): options["fit_algorithm"] = "cd"
                    if j == 14: options["transform_algorithm"] = "lasso_cd"
                    elif j == 15:
                        options.update(n_components=p+2, alpha=.2, max_iter=12,
                                       transform_algorithm="threshold", transform_alpha=.1, split_sign=True)
                    elif j == 16: options.update(alpha=.2, positive_dict=True, transform_algorithm="lasso_lars")
                    elif j == 17:
                        options.update(alpha=.2, positive_code=True, positive_dict=True, transform_algorithm="lasso_cd")
                    elif j == 18: options.update(alpha=1e5, max_iter=8, transform_algorithm="lasso_lars")
                    elif j == 19:
                        options.update(max_iter=0, transform_algorithm="lars", transform_n_nonzero_coefs=2)
                    elif j == 20: options.update(max_iter=15, transform_n_nonzero_coefs=2)
                    elif j in (21, 22):
                        options.update(max_iter=15, dict_init=atoms[:3].copy(), code_init=warm[:, :3].copy(),
                                       transform_algorithm="lasso_lars", transform_alpha=.15)
                    elif j == 23:
                        options.update(max_iter=15, transform_algorithm="lars", transform_n_nonzero_coefs=2, split_sign=True)
                    elif j == 24: options.update(tol=.1, transform_algorithm="threshold")
                    model = DictionaryLearning(**options)
                    train = np.abs(x) if j == 17 else x
                    code = model.fit_transform(train)
                    values = dict(components=model.components_, errors=model.error_,
                                  shape=[model.n_features_in_, model.n_iter_], fit_code=code,
                                  train=model.transform(train), query=model.transform(train+.137))
                else:
                    algorithm = {25: "lasso_lars", 26: "lasso_cd", 27: "lasso_lars", 28: "lasso_cd",
                                 29: "lars", 30: "omp", 31: "threshold", 32: "threshold", 33: "omp", 34: "lasso_lars"}[j]
                    positive = j in (27, 28, 32)
                    sparse_options = dict(algorithm=algorithm, alpha=.2, positive=positive,
                                          n_nonzero_coefs=2 if j in (29, 30) else None,
                                          max_iter=0 if j == 34 else 1000)
                    code = sparse_encode(x, atoms, init=warm if j == 26 else None, **sparse_options)
                    model = SparseCoder(atoms, transform_algorithm=algorithm, transform_alpha=.2,
                                        positive_code=positive, transform_n_nonzero_coefs=sparse_options["n_nonzero_coefs"],
                                        transform_max_iter=sparse_options["max_iter"], split_sign=j == 31)
                    model.fit(x[:0])  # sklearn's SparseCoder.fit deliberately ignores its input.
                    z = model.fit_transform(x)
                    values = dict(direct=code, coder=z, shape=[model.n_components_, model.n_features_in_, z.shape[1]])
                expected.update({f"case{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            assert actual[name].shape == want.shape, (seed, name, actual[name].shape, want.shape)
            np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    (RESULTS / "dictionary_learning.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} sparse-dictionary result groups passed")


if __name__ == "__main__":
    main()
