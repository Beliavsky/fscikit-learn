# Validation

Run from the package root:

```
fpm test --flag "-fcheck=all -fbacktrace"
fpm test --profile release
python validation/compare_sklearn.py
python validation/check_invalid_inputs.py
```

The reference validator requires NumPy, SciPy and scikit-learn in Python. The
Fortran package itself does not. It generates three seeded 80-by-5 datasets,
with binary, three-class and four-class integer targets and a noisy regression
target, then compares 64 result groups per case. Outputs include model
coefficients, predictions, probabilities, transforms, metrics, likelihoods,
covariances and posterior uncertainties. Baseline: scikit-learn **1.6.1**.

The default numerical tolerance is `atol=rtol=3e-8`, or `3e-7` for logistic
probabilities (different iterative optimizers). Cluster partitions are compared
using adjusted Rand agreement, with a separate DBSCAN noise-mask check. Exact
cluster identifier equality is not meaningful. Identical-row Euclidean
distances are set to zero in the reference because sklearn's norm-expansion
formula can leave approximately `1e-7` cancellation residue on the diagonal.

The reference forest fixture disables bootstrap and uses every feature to remove
random-stream dependence. The classification-tree/forest reference uses one
feature to remove random feature-permutation ties. Multifeature tree controls,
feature importance on a uniquely informative feature, bootstrap/Extra Trees,
loss reduction, and stochastic fitting are tested separately in native tests.
Boosting is compared on training predictions because different tied splits can
give equivalent training partitions but different off-sample boundaries.

Mixture reference likelihoods use one Gaussian component (full/diagonal); native
tests cover multi-component partitions and all four covariance modes. Neural
networks are validated by learning nonlinear XOR/regression, loss decrease and
normalized probabilities, rather than equating different random initializations.

`check_invalid_inputs.py` checks 16 error paths in separate bounds-checked
processes, requiring both an error exit and a specific input diagnostic.
These cover unfitted use, nonfinite data, feature/target shape mismatches,
invalid hyperparameters, empty text and unusable/all-missing data.

Generated datasets, outputs, JSON reports and logs are saved in `validation/results/`
(ignored by Git). The reference report records the actual Python package version.
Passing checks establish the documented dense subset on these fixtures; they do
not establish full API equivalence or production-scale performance.
