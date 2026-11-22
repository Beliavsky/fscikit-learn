"""Fixed-kernel binary and one-vs-rest Laplace GPC reference comparisons."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',choices=['checked','release'],default='checked')
    parser.add_argument('--executable',type=Path)
    args=parser.parse_args()
    from compare_function_transformer import resolve
    executable=args.executable.resolve() if args.executable else resolve(args.profile,'reference_gp_classification_driver')
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 30, 2), (42, 36, 3), (2026, 24, 4)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))
        binary = np.where(x[:, 0]+.3*x[:, 1] > 0, 11, -5)
        scores = np.column_stack([x[:, 0], -.6*x[:, 0]+x[:, 1], -x[:, 1]])
        multi = np.array([-4, 7, 21])[scores.argmax(axis=1)]
        assert len(np.unique(multi)) == 3 and len(np.unique(binary)) == 2
        data, output = RESULTS/f"gpc_{seed}.txt", RESULTS/f"gpc_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, np.column_stack([x, binary, multi]), fmt="%.17g")
        subprocess.run([str(executable),str(data),str(output)],cwd=ROOT,check=True)
        actual, expected = {}, {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        for j in range(1, 9):
            train, target = x.copy(), binary
            amplitude, length, iterations = 1., 1., 100
            if j == 2: target = multi
            if j == 3: amplitude, length = 2.5, .4
            if j == 4: amplitude, length = .3, 3.
            if j == 5: iterations, target = 1, multi
            if j == 6: iterations = 2
            if j == 7: train[:, -1] = train[:, 0]; target = multi
            if j == 8: amplitude, length, target = 10., 1.7, multi
            kernel = ConstantKernel(amplitude, "fixed") * RBF(length, "fixed")
            model = GaussianProcessClassifier(kernel=kernel, optimizer=None, max_iter_predict=iterations).fit(train, target)
            estimators = ([model.base_estimator_] if len(model.classes_) == 2
                          else model.base_estimator_.estimators_)
            values = dict(classes=model.classes_, pi=np.column_stack([e.pi_ for e in estimators]),
                          w=np.column_stack([e.W_sr_ for e in estimators]),
                          mode=np.column_stack([e.f_cached for e in estimators]),
                          alpha=np.column_stack([e.y_train_-e.pi_ for e in estimators]),
                          l=np.stack([e.L_ for e in estimators], axis=2),
                          lmls=[e.log_marginal_likelihood_value_ for e in estimators],
                          lml=[model.log_marginal_likelihood()], train=model.predict_proba(train),
                          query=model.predict_proba(train+.137), train_predict=model.predict(train),
                          predict=model.predict(train+.137), score=[model.score(train, target)])
            expected.update({f"case{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            assert actual[name].shape == want.shape, (seed, name, actual[name].shape, want.shape)
            np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    report=dict(sklearn=sklearn.__version__,profile=args.profile,checks=checks,
                native_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (RESULTS/f'gp_classification_{args.profile}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if args.profile=='checked':
        (RESULTS/'gp_classification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(checks)} Gaussian process classification result groups passed")

if __name__ == "__main__":
    main()
