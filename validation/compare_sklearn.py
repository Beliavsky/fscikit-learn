"""Numerical integration checks against installed scikit-learn; no network needed.

Run: python validation/compare_sklearn.py
Python dependencies are validation-only: numpy, scipy, scikit-learn.
"""
from pathlib import Path
import json
import os
import subprocess
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import sklearn
from sklearn import preprocessing as pp, metrics as mt, linear_model as lm
from sklearn import naive_bayes as nb, neighbors as nn, decomposition as dc, cluster as cl
from sklearn import tree, ensemble, feature_selection as fs, discriminant_analysis as da
from sklearn import kernel_ridge as kr, svm, mixture
from sklearn import covariance as cov, gaussian_process as gp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"


def reference(x, y, labels):
    expected = {}
    for key, model in [
        ("standard", pp.StandardScaler()), ("minmax", pp.MinMaxScaler()),
        ("maxabs", pp.MaxAbsScaler()), ("robust", pp.RobustScaler()),
        ("polynomial", pp.PolynomialFeatures(3)),
    ]:
        expected[key] = model.fit_transform(x)
    expected["normalize"] = pp.normalize(x)
    ols = lm.LinearRegression().fit(x, y)
    expected["ols_coef"] = ols.coef_
    expected["ols_predict"] = ols.predict(x)
    for key, model in [
        ("ridge", lm.Ridge(alpha=.7)),
        ("lasso", lm.Lasso(alpha=.07, tol=1e-12, max_iter=10000)),
        ("elastic", lm.ElasticNet(alpha=.07, tol=1e-12, max_iter=10000)),
    ]:
        expected[key] = model.fit(x, y).predict(x)
    logistic = lm.LogisticRegression(C=1.3, tol=1e-12, max_iter=2000).fit(x, labels)
    expected["logistic"] = logistic.predict_proba(x)
    expected["gaussian"] = nb.GaussianNB().fit(x, labels).predict_proba(x)
    positive = np.abs(x)
    for key, model in [("multinomial", nb.MultinomialNB()),
                       ("bernoulli", nb.BernoulliNB(binarize=.5)),
                       ("complement", nb.ComplementNB())]:
        expected[key] = model.fit(positive, labels).predict_proba(positive)
    expected["knn_classifier"] = nn.KNeighborsClassifier(weights="distance").fit(x, labels).predict_proba(x+.013)
    expected["knn_regressor"] = nn.KNeighborsRegressor(weights="distance").fit(x, y).predict(x+.013)
    pca = dc.PCA(n_components=min(3, x.shape[1]), svd_solver="full").fit(x)
    expected["pca_components"] = pca.components_
    expected["pca_variance"] = pca.explained_variance_
    expected["pca_transform"] = pca.transform(x)
    expected["svd_values"] = np.linalg.svd(x, compute_uv=False)[:min(3, x.shape[1])]
    expected["euclidean"] = mt.pairwise.euclidean_distances(x[:7], x[:9])
    # sklearn's norm expansion can leave ~1e-7 cancellation on identical rows;
    # the Fortran direct-difference implementation returns their exact distance.
    np.fill_diagonal(expected["euclidean"], 0)
    expected["cosine"] = mt.pairwise.cosine_similarity(x[:7], x[:9])
    expected["rbf"] = mt.pairwise.rbf_kernel(x[:7], x[:9])
    km = cl.KMeans(n_clusters=3, init=x[:3].copy(), n_init=1, tol=0, algorithm="lloyd").fit(x)
    expected["kmeans_labels"] = km.labels_
    expected["kmeans_inertia"] = [km.inertia_]
    expected["dbscan"] = cl.DBSCAN(eps=1.4, min_samples=3).fit_predict(x)
    expected["ward"] = cl.AgglomerativeClustering(n_clusters=3).fit_predict(x)
    pred = logistic.predict(x)
    expected["classification_metrics"] = [
        mt.accuracy_score(labels, pred), mt.f1_score(labels, pred, average="macro"),
        mt.precision_score(labels, pred, average="weighted", zero_division=0),
        mt.recall_score(labels, pred, average="micro"), mt.matthews_corrcoef(labels, pred),
        mt.cohen_kappa_score(labels, pred), mt.balanced_accuracy_score(labels, pred),
    ]
    pred = ols.predict(x)
    expected["regression_metrics"] = [mt.mean_squared_error(y, pred), mt.mean_absolute_error(y, pred),
        mt.r2_score(y, pred), mt.explained_variance_score(y, pred), mt.median_absolute_error(y, pred)]
    expected["cluster_metrics"] = [mt.silhouette_score(x, km.labels_),
        mt.calinski_harabasz_score(x, km.labels_), mt.davies_bouldin_score(x, km.labels_),
        mt.adjusted_rand_score(labels, km.labels_), mt.rand_score(labels, km.labels_),
        mt.homogeneity_score(labels, km.labels_), mt.completeness_score(labels, km.labels_),
        mt.v_measure_score(labels, km.labels_)]
    # One feature removes sklearn's random feature-permutation tie breaker.
    # Multifeature tree behavior is covered by the native invariant tests.
    tc = tree.DecisionTreeClassifier(max_depth=4, random_state=0).fit(x[:, :1], labels)
    expected["tree_classifier"] = tc.predict_proba(x[:, :1]+.013)
    expected["tree_class_importance_total"] = [tc.feature_importances_.sum()]
    tr = tree.DecisionTreeRegressor(max_depth=4, random_state=0).fit(x, y)
    expected["tree_regressor"] = tr.predict(x+.013)
    # Identical partitions can be attributed to different features under tied
    # split gains. Their normalized total is invariant; a separate unit test
    # checks individual importance on a unique informative-feature fixture.
    expected["tree_reg_importance_total"] = [tr.feature_importances_.sum()]
    expected["forest_classifier"] = ensemble.RandomForestClassifier(
        n_estimators=3, max_depth=4, bootstrap=False, max_features=1, random_state=0,
    ).fit(x[:, :1], labels).predict_proba(x[:, :1]+.013)
    expected["forest_regressor"] = ensemble.RandomForestRegressor(
        n_estimators=3, max_depth=4, bootstrap=False, random_state=0,
    ).fit(x, y).predict(x+.013)
    expected["boosting"] = ensemble.GradientBoostingRegressor(
        n_estimators=15, random_state=0).fit(x, y).predict(x)
    expected["f_classif"] = fs.f_classif(x, labels)[0]
    expected["f_regression"] = fs.f_regression(x, y)[0]
    expected["chi2"] = fs.chi2(positive, labels)[0]
    expected["lda"] = da.LinearDiscriminantAnalysis(solver="lsqr").fit(x, labels).predict_proba(x+.013)
    expected["qda"] = da.QuadraticDiscriminantAnalysis(reg_param=.1).fit(x, labels).predict_proba(x+.013)
    expected["kernel_ridge"] = kr.KernelRidge(alpha=.3, kernel="rbf").fit(x, y).predict(x+.013)
    rc = lm.RidgeCV(alphas=[.1, 1, 10], cv=5, scoring="neg_mean_squared_error").fit(x, y)
    expected["ridge_cv"] = [rc.alpha_, rc.best_score_]
    model = svm.SVC(tol=1e-7).fit(x, labels)
    expected["svc"] = model.predict(x+.013)
    expected["svc_converged"] = [1]
    for kind in ("full", "diag"):
        model = mixture.GaussianMixture(n_components=1, covariance_type=kind, random_state=0).fit(x)
        expected[f"mixture_{kind}"] = model.score_samples(x)
    ec = cov.EmpiricalCovariance().fit(x)
    expected["empirical_covariance"] = ec.covariance_
    expected["mahalanobis"] = ec.mahalanobis(x+.013)
    expected["covariance_score"] = [ec.score(x+.013)]
    expected["shrunk_covariance"] = cov.ShrunkCovariance().fit(x).covariance_
    lw = cov.LedoitWolf().fit(x)
    expected["ledoit_wolf"] = lw.covariance_
    expected["ledoit_shrinkage"] = [lw.shrinkage_]
    oa = cov.OAS().fit(x)
    expected["oas"] = oa.covariance_
    expected["oas_shrinkage"] = [oa.shrinkage_]
    model = gp.GaussianProcessRegressor(
        kernel=gp.kernels.RBF(1)+gp.kernels.WhiteKernel(.02),
        alpha=.01, optimizer=None, normalize_y=True).fit(x, y)
    mean, std = model.predict(x+.013, return_std=True)
    expected["gp_prediction"] = mean
    expected["gp_std"] = std
    expected["gp_cov"] = model.predict(x[:7]+.013, return_cov=True)[1]
    expected["gp_likelihood"] = [model.log_marginal_likelihood_value_]
    documents = ["This is the first document.", "This document is the second document.",
                 "And this is the third one.", "Is this the first document?"]
    expected["text_counts"] = CountVectorizer().fit_transform(documents).toarray()
    expected["text_tfidf"] = TfidfVectorizer().fit_transform(documents).toarray()
    return expected


def main():
    RESULTS.mkdir(exist_ok=True)
    report = {"sklearn_version": sklearn.__version__, "checks": []}
    for seed, classes in [(7, 2), (42, 3), (2026, 4)]:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(80, 5)) @ rng.normal(size=(5, 5))
        y = x @ rng.normal(size=5) + 2 + rng.normal(size=80)*.3
        logits = x @ rng.normal(size=(5, classes)) + rng.normal(size=(80, classes))
        labels = np.argmax(logits, axis=1)*3-2
        data = RESULTS / f"case_{seed}.txt"
        output = RESULTS / f"actual_{seed}.txt"
        with data.open("w") as f:
            f.write(f"{len(x)} {x.shape[1]}\n")
            np.savetxt(f, np.column_stack([x, y, labels]), fmt="%.17g")
        subprocess.run(["fpm", "run", "--target", "reference_driver", "--",
                        str(data), str(output)], cwd=ROOT, check=True)
        actual = {}
        for line in output.read_text().splitlines():
            name, *values = line.split()
            actual[name] = np.array(values, dtype=float)
        expected = reference(x, y, labels)
        assert actual.keys() == expected.keys(), (actual.keys() ^ expected.keys())
        for name, values in expected.items():
            want = np.asarray(values).ravel(order="F")
            got = actual[name]
            if name in ("ward", "dbscan", "kmeans_labels"):
                # Cluster identifiers are arbitrary; noise is separately checked.
                assert mt.adjusted_rand_score(want, got) == 1, (seed, name)
                if name == "dbscan":
                    np.testing.assert_array_equal(got == -1, want == -1)
            else:
                atol = 3e-7 if name == "logistic" else 3e-8
                np.testing.assert_allclose(got, want, atol=atol, rtol=atol, err_msg=f"seed={seed} {name}")
            report["checks"].append({"seed": seed, "api": name, "passed": True})
            print(f"PASS seed={seed} {name}")
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"{len(report['checks'])} reference comparisons passed against scikit-learn {sklearn.__version__}")


if __name__ == "__main__":
    main()
