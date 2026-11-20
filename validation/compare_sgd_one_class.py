"""SGDOneClassSVM fit, shuffle, averaging, warm starts and streaming paths."""
from pathlib import Path
import argparse
import json
import subprocess
import warnings
import numpy as np
import sklearn
from sklearn.linear_model import SGDOneClassSVM
from sklearn.exceptions import ConvergenceWarning
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"
FLAGS = "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow"


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--executable", type=Path); args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.executable:
        executable = args.executable.resolve()
    else:
        result = subprocess.run(["fpm", "run", "--flag", FLAGS, "--target", "reference_sgd_one_class_driver", "--runner"],
                                cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE)
        paths = [ROOT / s.strip().strip('"') for s in result.stdout.splitlines()
                 if s.strip().strip('"').endswith(("reference_sgd_one_class_driver", "reference_sgd_one_class_driver.exe"))]
        paths = [p for p in paths if p.is_file()]; assert len(paths) == 1, result.stdout
        executable = paths[0].resolve()
    checks = []
    for seed, n, p in ((7, 57, 3), (42, 72, 4), (2026, 45, 5), (0, 12, 2)):
        rng = np.random.default_rng(seed)
        original = rng.normal(scale=.5, size=(n, p))+1
        weights = rng.uniform(.2, 2, n); weights[0] = 0
        for case in range(30):
            x, w = original.copy(), np.ones(n)
            options = dict(nu=.35, learning_rate="optimal", eta0=0., power_t=.5, max_iter=12,
                           tol=None, shuffle=True, fit_intercept=True, average=False, random_state=seed)
            operation = 1; no_change = 5
            if case == 1: options.update(learning_rate="constant", eta0=.02, shuffle=False)
            if case == 2: options.update(learning_rate="invscaling", eta0=.1, power_t=.3)
            if case == 3: options.update(learning_rate="adaptive", eta0=.1, tol=1e-3, max_iter=500)
            if case == 4: options["fit_intercept"] = False
            if case == 5: w = weights.copy()
            if case == 6: options["average"] = True
            if case == 7: options["average"] = n+3
            if case == 8: operation = 2
            if case == 9: operation = 2; options["average"] = True
            if case == 10: operation = 2; options.update(average=n//2, learning_rate="constant", eta0=.02)
            if case == 11: operation = 3
            if case == 12: operation = 3; options["average"] = True
            if case == 13: operation = 4
            if case == 14: operation = 4; options.update(average=2, fit_intercept=False)
            if case == 15: operation = 5; options["average"] = True
            if case == 16: options.update(tol=1e-3, max_iter=200)
            if case == 17: options.update(nu=1., average=n*20)
            if case == 18: w[:] = 0
            if case == 19: x[:] = 0; options.update(learning_rate="constant", eta0=.1, shuffle=False)
            if case == 20: options.update(learning_rate="constant", eta0=10., average=True)
            if case == 21: options.update(learning_rate="invscaling", eta0=.001, power_t=-.2)
            if case == 22: operation = 2; w = weights.copy(); options.update(average=True, fit_intercept=False)
            if case == 23: options.update(learning_rate="adaptive", eta0=.1, tol=1e-2, max_iter=300); no_change = 2
            if case == 24: options.update(average=True, learning_rate="adaptive", eta0=.02, tol=1e-3, max_iter=500)
            if case == 25: options.update(average=True, shuffle=False, learning_rate="constant", eta0=.01)
            if case == 26: x[:, -1] = 0; w = weights.copy(); options["nu"] = .1
            if case == 27: options.update(average=n+2); operation = 5
            if case == 28: options.update(nu=.5, tol=1e-3, max_iter=1000)
            if case == 29: operation = 6; w = weights.copy()
            data, output = RESULTS / "sgd_one_class_input.txt", RESULTS / "sgd_one_class_actual.txt"
            average = int(options["average"])
            with data.open("w") as stream:
                stream.write(f"{n} {p} {operation} {int(options['shuffle'])} {int(options['fit_intercept'])}\n")
                stream.write(f"{options['learning_rate']} {options['nu']} {options['eta0']} {options['power_t']} "
                             f"{options['tol'] if options['tol'] is not None else -1}\n")
                stream.write(f"{options['max_iter']} {no_change} {average} {seed}\n")
                np.savetxt(stream, np.column_stack([x, w]), fmt="%.17g")
            subprocess.run([str(executable), str(data), str(output)], cwd=ROOT, check=True)
            actual = {}
            for line in output.read_text().splitlines():
                name, *values = line.split(); actual[name] = np.array(values, dtype=float)
            expected = {}
            model = SGDOneClassSVM(**options); model.n_iter_no_change = no_change
            def emit(prefix):
                values = dict(coef=model.coef_, offset=model.offset_, decision=model.decision_function(x+.137),
                              samples=model.score_samples(x+.137), predict=model.predict(x+.137),
                              state=[model.t_, model.n_iter_, model.n_features_in_])
                expected.update({prefix+"_"+name: np.asarray(value).copy() for name, value in values.items()})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                if operation == 2:
                    for part in range(3):
                        first, last = part*n//3, (part+1)*n//3
                        model.partial_fit(x[first:last], sample_weight=w[first:last]); emit(f"partial{part+1}")
                else:
                    initial = dict(coef_init=np.full(p, .1), offset_init=.2) if operation == 4 else {}
                    if operation == 6: model.fit_predict(x, sample_weight=w)
                    else: model.fit(x, sample_weight=w, **initial)
                    emit("fit")
                    if operation == 3:
                        model.warm_start = True; model.fit(x+.02, sample_weight=w); emit("warm")
                    if operation == 5: model.partial_fit(x, sample_weight=w); emit("after")
            assert actual.keys() == expected.keys(), (seed, case)
            for name, value in expected.items():
                tolerance = 0 if name.endswith(("_state", "_predict")) else 3e-8
                np.testing.assert_allclose(actual[name], np.asarray(value).ravel(), atol=tolerance, rtol=tolerance,
                                           err_msg=f"seed={seed} case={case} {name}")
                checks.append(dict(seed=seed, case=case, name=name, passed=True))
    (RESULTS / "sgd_one_class.json").write_text(json.dumps(dict(sklearn=sklearn.__version__, checks=checks), indent=2)+"\n")
    print(f"{len(checks)} SGDOneClassSVM result groups passed")


if __name__ == "__main__":
    main()
