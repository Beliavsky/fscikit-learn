"""Primal L2 squared-loss linear SVM reference comparisons."""
from pathlib import Path
import json
import subprocess
import argparse
import hashlib
import tempfile
import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
from sklearn.svm import LinearSVC, LinearSVR
import sklearn.svm._liblinear as backend
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation/results"

def sparse_input(a, layout):
    if layout == 'dense': return a
    if layout == 'csr': return sp.csr_matrix(a)
    data, indices, ptr = [], [], [0]
    for row in a:
        for j in reversed(range(len(row))):
            if layout != 'zeros' and row[j] == 0: continue
            if layout == 'duplicates':
                data.extend([.25*row[j], .75*row[j], 0.]); indices.extend([j, j, j])
            else:
                data.append(row[j]); indices.append(j)
        ptr.append(len(data))
    return sp.csr_matrix((data, indices, ptr), shape=a.shape)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--layout', choices=['dense','csr','reverse','duplicates','zeros'], default='dense')
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    hashes = {str(p):sha(p) for p in [Path(__file__),ROOT/'src/sklearn_linear_svm.f90',
        ROOT/'app/reference_linear_svm_driver.f90',Path(backend.__file__),
        Path(sklearn.__file__).parent/'svm/_base.py',Path(sklearn.__file__).parent/'svm/_classes.py',
        Path(sklearn.__file__).parent/'svm/src/liblinear/linear.cpp',
        Path(sklearn.__file__).parent/'svm/src/liblinear/tron.cpp',
        *([args.executable] if args.executable else [])]}
    temporary = tempfile.TemporaryDirectory(prefix='fsklearn-linearsvm-'); work = Path(temporary.name)
    wire = lambda a:sparse_input(a,args.layout)
    RESULTS.mkdir(exist_ok=True)
    checks = []
    retained=[];gradient_certificates=[]
    configurations=[(7,60,3),(42,45,4),(2026,9,14)]
    if args.extended: configurations += [(11,31,17),(99,17,43),(2025,12,257)]
    for seed, n, p in configurations:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, p))
        if args.layout != 'dense': x[rng.uniform(size=x.shape)<.55] = 0
        binary = np.where(x[:, 0]+.3*x[:, 1] > .2, 11, -5)
        scores = np.column_stack([x[:, 0], -.6*x[:, 0]+x[:, 1], -x[:, 1]])
        multi = np.array([-4, 7, 21])[scores.argmax(axis=1)]
        if n < 20:
            binary = np.where(np.arange(n)<n//2,-5,11); multi = np.resize([-4,7,21],n)
        assert len(np.unique(multi)) == 3 and len(np.unique(binary)) == 2
        target = x[:, 0]*1.3-x[:, 1]*.7+.8+rng.normal(scale=.3, size=n)
        w = rng.uniform(.2, 2, n); w[0] = 0
        data, output = work/f"linearsvm_{seed}.txt", work/f"linearsvm_actual_{seed}.txt"
        with data.open("w") as stream:
            stream.write(f"{n} {p}\n")
            np.savetxt(stream, np.column_stack([x, w, target, binary, multi]), fmt="%.17g")
        command = [str(args.executable)] if args.executable else ["fpm", "run", "--flag", "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow", "--target", "reference_linear_svm_driver", "--"]
        subprocess.run([*command,str(data),str(output),*([args.layout] if args.layout!='dense' else [])],cwd=ROOT,check=True)
        actual, expected = {}, {}
        for line in output.read_text().splitlines():
            name, *values = line.split(); actual[name] = np.array(values, dtype=float)
        for j in range(1, 11):
            train, labels, weight = x.copy(), binary, np.ones(n)
            options = dict(dual=False, tol=1e-12, max_iter=10000)
            if j == 2: labels = multi
            if j == 3: options["fit_intercept"] = False
            if j == 4: options.update(C=.05, intercept_scaling=2)
            if j in (5, 6, 7): labels, weight = multi, w
            if j == 6: options["class_weight"] = "balanced"
            if j == 7: options["class_weight"] = {-4: .5, 7: 2., 21: 1.5}
            if j == 8: weight = w; options["class_weight"] = {-5: .5, 11: 2.}
            if j == 9: train[:, -1] = train[:, 0]; labels = multi
            if j == 10: options.update(C=10, intercept_scaling=4)
            model = LinearSVC(**options).fit(wire(train), labels, sample_weight=weight)
            decisions = model.decision_function(wire(train)).reshape(n, -1)
            regularizer = .5*np.sum(model.coef_**2, axis=1)
            if model.fit_intercept: regularizer += .5*(model.intercept_/model.intercept_scaling)**2
            class_weights = np.ones(len(model.classes_))
            if options.get("class_weight") == "balanced":
                class_weights = np.array([n/(len(model.classes_)*np.sum(labels == c)) for c in model.classes_])
            elif isinstance(options.get("class_weight"), dict):
                class_weights = np.array([options["class_weight"][c] for c in model.classes_])
            objectives = [];native_norms=[]
            native_coefs = actual[f"svc{j}_coef"].reshape(model.coef_.shape, order="F")
            augmented = (np.column_stack([train, np.full(n, model.intercept_scaling)])
                         if model.fit_intercept else train)
            for k in range(decisions.shape[1]):
                positive = model.classes_[1] if len(model.classes_) == 2 else model.classes_[k]
                signed = np.where(labels == positive, 1., -1.)
                costs = model.C*weight
                if len(model.classes_) == 2:
                    costs = costs*class_weights[np.searchsorted(model.classes_, labels)]
                else:
                    costs = costs*np.where(labels == positive, class_weights[k], 1.)
                objectives.append(regularizer[k]+np.sum(costs*np.maximum(0, 1-signed*decisions[:, k])**2))
                native_w = native_coefs[k]
                if model.fit_intercept:
                    native_w = np.r_[native_w, actual[f"svc{j}_intercept"][k]/model.intercept_scaling]
                native_loss = np.maximum(0, 1-signed*(augmented@native_w))
                gradient = native_w-2*augmented.T@(costs*signed*native_loss)
                native_norms.append(float(np.linalg.norm(gradient)))
                np.testing.assert_allclose(actual[f"svc{j}_gradient_norm"][k],np.linalg.norm(gradient),atol=1e-10,rtol=1e-8)
                source_w=model.coef_[k]
                if model.fit_intercept: source_w=np.r_[source_w,model.intercept_[k]/model.intercept_scaling]
                source_loss=np.maximum(0,1-signed*(augmented@source_w))
                source_gradient=source_w-2*augmented.T@(costs*signed*source_loss)
                gradient_certificates.append(dict(seed=seed,family='svc',configuration=j,channel=k,
                    gradient_norm=float(np.linalg.norm(gradient)),source_gradient_norm=float(np.linalg.norm(source_gradient)),
                    strict_optimality=bool(np.linalg.norm(gradient)<=1e-8*max(1,np.linalg.norm(native_w))),
                    source_strict_optimality=bool(np.linalg.norm(source_gradient)<=1e-8*max(1,np.linalg.norm(source_w))),
                    bound=float(1e-8*max(1,np.linalg.norm(native_w))),passed=True))
                native_objective = .5*native_w@native_w+np.sum(costs*native_loss**2)
                np.testing.assert_allclose(actual[f"svc{j}_objective"][k], native_objective, atol=1e-10, rtol=1e-10)
            values = dict(coef=model.coef_, intercept=np.broadcast_to(model.intercept_, (model.coef_.shape[0],)),
                          classes=model.classes_, train=decisions,
                          query=model.decision_function(wire(train+.137)).reshape(n, -1),
                          predict=model.predict(wire(train+.137)), score=[model.score(wire(train), labels)], objective=objectives,
                          gradient_norm=native_norms)
            expected.update({f"svc{j}_{name}": value for name, value in values.items()})
        for j in range(1, 9):
            train, weight = x.copy(), np.ones(n)
            options = dict(dual=False, loss="squared_epsilon_insensitive", tol=1e-12, max_iter=10000)
            if j == 2: options["epsilon"] = .2
            if j == 3: options.update(fit_intercept=False, epsilon=.1)
            if j == 4: options.update(C=.05, intercept_scaling=2, epsilon=.3)
            if j == 5: weight = w; options["epsilon"] = .2
            if j == 6: weight = w; options.update(C=10, intercept_scaling=4, epsilon=.5)
            if j == 7: train[:, -1] = train[:, 0]; options["epsilon"] = .15
            if j == 8: options["epsilon"] = 100
            model = LinearSVR(**options).fit(wire(train), target, sample_weight=weight)
            prediction = model.predict(wire(train))
            objective = .5*np.sum(model.coef_**2)
            if model.fit_intercept: objective += .5*(model.intercept_[0]/model.intercept_scaling)**2
            objective += model.C*np.sum(weight*np.maximum(0, np.abs(target-prediction)-model.epsilon)**2)
            augmented = (np.column_stack([train, np.full(n, model.intercept_scaling)])
                         if model.fit_intercept else train)
            native_w = actual[f"svr{j}_coef"]
            if model.fit_intercept:
                native_w = np.r_[native_w, actual[f"svr{j}_intercept"][0]/model.intercept_scaling]
            residual = augmented@native_w-target
            excess = np.sign(residual)*np.maximum(0, np.abs(residual)-model.epsilon)
            gradient = native_w+2*model.C*augmented.T@(weight*excess)
            # TRON also stops on tiny predicted/actual objective reductions.
            # Check the reported derivative independently, and retain whether
            # this stronger historical optimality bound is met.
            np.testing.assert_allclose(actual[f"svr{j}_gradient_norm"][0],np.linalg.norm(gradient),atol=1e-10,rtol=1e-8)
            source_w=model.coef_
            if model.fit_intercept: source_w=np.r_[source_w,model.intercept_[0]/model.intercept_scaling]
            source_residual=augmented@source_w-target
            source_excess=np.sign(source_residual)*np.maximum(0,np.abs(source_residual)-model.epsilon)
            source_gradient=source_w+2*model.C*augmented.T@(weight*source_excess)
            gradient_certificates.append(dict(seed=seed,family='svr',configuration=j,
                gradient_norm=float(np.linalg.norm(gradient)),source_gradient_norm=float(np.linalg.norm(source_gradient)),
                strict_optimality=bool(np.linalg.norm(gradient)<=1e-8*max(1,np.linalg.norm(native_w))),
                source_strict_optimality=bool(np.linalg.norm(source_gradient)<=1e-8*max(1,np.linalg.norm(source_w))),
                bound=float(1e-8*max(1,np.linalg.norm(native_w))),passed=True))
            native_objective = .5*native_w@native_w+model.C*np.sum(weight*excess**2)
            np.testing.assert_allclose(actual[f"svr{j}_objective"][0], native_objective, atol=1e-10, rtol=1e-10)
            values = dict(coef=model.coef_, intercept=np.atleast_1d(model.intercept_), train=prediction,
                          query=model.predict(wire(train+.137)), score=[model.score(wire(train), target)], objective=[objective],
                          gradient_norm=[np.linalg.norm(gradient)])
            expected.update({f"svr{j}_{name}": value for name, value in values.items()})
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            want = np.asarray(value).ravel(order="F")
            assert actual[name].shape == want.shape, (seed, name, actual[name].shape, want.shape)
            # TRON can stop on tiny objective reductions before its requested gradient tolerance.
            coefficient_output = name.startswith("svc") and name.rsplit("_", 1)[1] in ("coef", "intercept", "train", "query")
            tolerance = 1e-6 if coefficient_output else 3e-8
            passed=bool(np.allclose(actual[name],want,atol=tolerance,rtol=tolerance,equal_nan=True))
            checks.append(dict(seed=seed,name=name,passed=passed,tolerance=tolerance,
                oracle='native_derivative_replay' if name.endswith('_gradient_norm') else 'source',
                max_abs_error=float(np.max(np.abs(actual[name]-want),initial=0))))
            retained.append(dict(seed=seed,name=name,actual=actual[name].tolist(),expected=want.tolist()))
    assert hashes=={p:sha(p) for p in hashes}
    gaps=[c for c in checks if not c['passed']]
    (args.report or RESULTS/"linear_svm.json").write_text(json.dumps(dict(sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,layout=args.layout,configurations=configurations,checks=checks,gaps=gaps,retained=retained,gradient_certificates=gradient_certificates,hashes=hashes,strict_parity=not gaps), indent=2)+"\n")
    print(f"{len(checks)} linear SVM groups; {len(gaps)} gaps; {len(gradient_certificates)} gradient certificates")
    if gaps: print(gaps[:8])
    temporary.cleanup()
    raise SystemExit(bool(gaps) and not args.diagnostic)

if __name__ == "__main__":
    main()
