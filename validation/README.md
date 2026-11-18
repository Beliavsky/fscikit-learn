# Validation

Run from the package root:

```
fpm test --flag "-fcheck=all -fbacktrace"
fpm test --profile release
python validation/compare_sklearn.py
python validation/compare_graph_algorithms.py
python validation/compare_regression_projection.py
python validation/compare_robust_glm.py
python validation/compare_modes_density.py
python validation/check_invalid_inputs.py
```

The reference validator requires NumPy, SciPy and scikit-learn in Python. The
Fortran package itself does not. It generates three seeded 80-by-5 datasets,
with binary, three-class and four-class integer targets and a noisy regression
target, then compares 64 result groups per case. Outputs include model
coefficients, predictions, probabilities, transforms, metrics, likelihoods,
covariances and posterior uncertainties. Baseline: scikit-learn **1.6.1**.

`compare_graph_algorithms.py` adds 40 result groups on each of three seeded
60-by-4 datasets: radius neighbors, mean/median/shrunk centroids, LOF training
and novelty scores, Kernel PCA, Isomap and semi-supervised label diffusion.
It checks both RBF and KNN diffusion graphs, training distributions, inductive
probabilities, iteration counts and convergence status. The two suites together
provide **312** result-group comparisons. Graph comparisons use the same
`atol=rtol=3e-8` threshold, including deterministically oriented coordinates.

The Isomap reference explicitly exercises disconnected-graph completion, and
captures sklearn's expected component/sparse-efficiency warnings for that
fixture. The semi-supervised fixtures can exhaust their iteration cap; the
comparison checks that Fortran reports the same cap and convergence status.

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

`compare_regression_projection.py` adds 68 result groups on each of three
datasets (80-by-6, 60-by-5 and wide 8-by-12), bringing the three-suite total to
**516** comparisons. It checks Bayesian/ARD coefficients, precisions, posterior
covariance and predictive uncertainty, pruning, intercept-free and capped fits,
OMP sparsity/tolerance modes, factor covariance/precision/latent Gram matrices,
log likelihood and iteration counts, and projection transforms/pseudoinverses.
The tolerance is `atol=rtol=2e-7`, except wide ARD uses `atol=rtol=1e-5` to allow
inverse-roundoff differences. Wide ARD fits use `tol=1e-4` to avoid comparing
roundoff-dominated stopping decisions at `1e-8`: sklearn uses a Woodbury/SVD
inverse while Fortran uses a Cholesky inverse in active feature space.

Wide Bayesian covariance checks use full-row-rank data without centering;
centered wide matrices have nonunique retained null-space directions in sklearn's
thin-SVD covariance convention. Factor latent Grams remove arbitrary SVD signs.
Projection reference models receive the native drawn matrix, so these checks
validate algebra and reconstruction, not equality of random streams. Matrix
inputs themselves are excluded from the comparison count. Native tests
separately check distribution moments, density, seed repeatability, distance
preservation, zero-rank inverse behavior, sparse support recovery and all-pruned
ARD posteriors.

`compare_robust_glm.py` adds 71 result groups on each of three datasets
(90-by-3, 84-by-4 and 96-by-5), for **213** added comparisons and **729** total
across all four suites. It checks Huber coefficients, predictions, joint scale,
outlier masks and penalized objective; RANSAC consensus masks and final refits;
Poisson, Gamma and Tweedie powers 0/1/1.5/1.9/2, positive-mean predictions,
intercept settings, L2 penalties, D2 scores and capped/converged fits.
GLM/RANSAC tolerance is `atol=rtol=3e-8`; Huber uses `1e-4` because sklearn's
L-BFGS-B relative-loss stopping and the native projected-gradient BFGS stopping
produce different final parameter accuracy. RANSAC data have a known, widely
separated clean consensus, checked independently; trial counts are not equated
across different random streams. GLM references use `solver='newton-cholesky'`.
Native checks additionally exercise singular Hessians, near-boundary Tweedie
powers, unpenalized target-scale equivariance, the epsilon-one Huber scale
boundary, dynamic trial caps and GLM cross-validation.

`compare_modes_density.py` adds 77 result groups on each of three datasets
(60-by-2, 72-by-3 and 84-by-4), for **231** additions and **960** comparisons
across five suites. It checks fixed/estimated-bandwidth MeanShift, seed/orphan
behavior and update caps; affinity propagation in feature/precomputed modes,
working similarities, exemplar indices, predictions and convergence; spectral
RBF/k-neighbor/precomputed graphs, embedding Grams and QR/k-means partitions;
five normalized density kernels, sample weighting, total log likelihood and
Scott/Silverman bandwidth rules. Tolerance is `atol=rtol=3e-8`.

Spectral Grams remove arbitrary eigenvector signs/rotations; clustering labels
are compared with adjusted Rand agreement and a separately known partition.
Continuous data are used for connected embedding graphs; cluster fixtures also
exercise disconnected graphs with the entire null eigenspace retained. Seeded
AP perturbations can differ across runtimes; these nondegenerate fixtures also
agree on exemplar indices and iteration counts. Weighted compact-kernel queries
have positive support to avoid sklearn tree cancellation residue at exact zero
density. Native tests separately verify zero-density negative infinities,
normalized kernel integrals, Gaussian/uniform-ball sampling moments and seed
repeatability, zero-weight exclusion, spectral weighted orthonormality and
invariance to large self loops (with entry-wise symmetry validation),
automatic zero bandwidth and equal-similarity/no-exemplar special cases.

`check_invalid_inputs.py` checks 62 error paths in separate bounds-checked
processes, requiring both an error exit and a specific input diagnostic.
These cover unfitted use, nonfinite data, feature/target shape mismatches,
invalid hyperparameters, empty text and unusable/all-missing data.
New cases cover empty radius neighborhoods, LOF novelty misuse, nonsymmetric
and indefinite kernels, Isomap neighbor counts, fully unlabeled training data
and invalid soft-clamping parameters.
Bayesian hyperpriors, ARD thresholds, OMP sparsity, factor components/noise,
projection density/automatic dimensions and unfitted uncertainty/inversion are
also rejected with explicit diagnostics.
Robust-regression controls, GLM target domains, unsupported Tweedie powers/links,
regularization, unfitted prediction and constant-target D2 are checked as well.
Further cases check empty MeanShift seed neighborhoods, invalid clustering
controls, precomputed prediction misuse, malformed spectral graphs/components,
and density bandwidths, kernels, weights, unfitted use and unsupported sampling.

Generated datasets, outputs, JSON reports and logs are saved in `validation/results/`
(ignored by Git). The reference report records the actual Python package version.
Passing checks establish the documented dense subset on these fixtures; they do
not establish full API equivalence or production-scale performance.

## FastICA addition (2026-11-18)

Added the core sklearn FastICA counterpart, bringing coverage to 81 types
in 32 source modules. Dense SVD whitening supports unit/arbitrary variance
and already-whitened data, parallel/deflation solvers, three contrasts,
fixed or seeded initialization, transformations and pseudoinverse reconstruction.
Rank-deficient whitening and singular iterations are explicitly diagnosed.
No extensions or third-party runtime dependencies were added. Original work
remains MIT, with sklearn BSD-3-Clause attribution retained.

Validation: 23 native programs in checked and release builds; 648 new sklearn
1.6.1 result groups (36 configurations x six outputs x three seeded fixtures),
bringing the six-suite inventory to 1,608 comparisons. Fixed starts permit
elementwise checks of components, mixing, sources, shifted transforms, inverse
transforms and iteration counts, including single-step capped fits and reduced
components. Unwhitened fixtures are externally whitened before both fits.
Tolerance is atol=rtol=2e-6. Native checks independently cover source recovery,
centering, variance, full reconstruction, reduced-space projection and seeded
repeatability. The isolated invalid-input suite now covers 72 cases.
