"""Weighted decision-stump SAMME comparisons against scikit-learn 1.6.1."""
from pathlib import Path
import json
import argparse
import hashlib
import subprocess
import numpy as np
import sklearn
from sklearn.ensemble import AdaBoostClassifier
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path)
    parser.add_argument('--profile',choices=['checked','release'],default='checked')
    parser.add_argument('--report',type=Path);args=parser.parse_args()
    if args.executable:executable=args.executable.resolve()
    else:
        options=['--profile','release'] if args.profile=='release' else ['--flag','-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow']
        resolved=subprocess.run(['fpm','run',*options,'--target','reference_adaboost_driver','--runner'],
                                cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        candidates=[(ROOT/line.strip().strip('"')).resolve() for line in resolved.stdout.splitlines()
                    if line.strip().strip('"').endswith(('reference_adaboost_driver','reference_adaboost_driver.exe'))]
        candidates=[p for p in candidates if p.is_file()];assert len(candidates)==1,resolved.stdout
        executable=candidates[0]
    RESULTS.mkdir(exist_ok=True)
    checks = []
    for seed, n, p in ((7, 57, 3), (42, 71, 4), (2026, 43, 5)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p)); w = rng.uniform(.2, 2, n); w[0] = 0
        binary = np.where(x[:, 0]+.5*x[:, 1]+.6*rng.normal(size=n)>0, 11, -5)
        multi = np.array([-4, 7, 21])[np.argmax(np.column_stack([x[:, 0], x[:, 1], -x[:, 0]-x[:, 1]])+.5*rng.normal(size=(n, 3)), axis=1)]
        data, output = RESULTS/f"adaboost_{args.profile}_{seed}.txt", RESULTS/f"adaboost_actual_{args.profile}_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, np.column_stack([x, w, binary, multi]), fmt="%.17g")
        subprocess.run([str(executable), str(data), str(output)], cwd=ROOT, check=True)
        actual, expected = {}, {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        for j in range(1, 7):
            options = dict(n_estimators=8, random_state=7)
            train, weights, y = x.copy(), w.copy(), binary
            if j == 2: y = multi
            if j == 3: train[:, 1:] = 0; weights = np.ones(n)
            if j == 4: options.update(learning_rate=.3, n_estimators=12); y = multi
            if j == 5: options["n_estimators"] = 1
            if j == 6: train[:, 1:] = 0; y = np.where(train[:, 0]>0, 7, -4)
            model = AdaBoostClassifier(**options).fit(train, y, sample_weight=weights)
            query = train+.137
            decisions = model.decision_function(train).reshape(n, -1)
            staged = np.stack([value.reshape(n, -1) for value in model.staged_decision_function(query)], axis=2)
            values = dict(weights=model.estimator_weights_, errors=model.estimator_errors_, classes=model.classes_,
                          importance=model.feature_importances_, train=decisions,
                          query=model.decision_function(query).reshape(n, -1), proba=model.predict_proba(query),
                          predict=model.predict(query), staged=staged,
                          staged_proba=np.stack(list(model.staged_predict_proba(query)), axis=2),
                          staged_predict=np.stack(list(model.staged_predict(query)), axis=1),
                          shape=[len(model.estimators_), model.n_classes_, model.n_features_in_])
            expected.update({f"case{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            assert actual[name].shape == want.shape, (seed, name)
            np.testing.assert_allclose(actual[name], want, atol=3e-8, rtol=3e-8, err_msg=f"{seed} {name}")
            checks.append(dict(seed=seed, name=name, passed=True))
    report=dict(profile=args.profile,sklearn=sklearn.__version__,checks=checks,
                executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output=args.report or RESULTS/'adaboost.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(checks)} AdaBoost result groups passed")

if __name__ == "__main__":
    main()
