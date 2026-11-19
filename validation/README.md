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

## Isotonic and kernel approximation addition (2026-11-18)

Added `isotonic_regression`, `rbf_sampler`, and `nystroem`, bringing the core
inventory to 84 types in 34 modules. Isotonic regression supports weighted
monotonic least squares, Spearman direction selection, bounds, input grouping
and interpolation. Kernel approximations support random Fourier RBF features
and PSD landmark maps for RBF/linear/polynomial/cosine kernels. No extensions
or runtime dependencies were added. MIT project licensing and retained
sklearn BSD-3-Clause attribution continue to apply.

`compare_isotonic_kernel.py` adds 189 sklearn 1.6.1 comparisons across three
seeded datasets, bringing the seven-suite inventory to 1,797 result groups.
Checks cover six isotonic configurations, standalone PAVA variants, gamma
scaling, Fourier transforms, landmark components, normalization and transforms.
NumPy receives the actual Fourier draws; only the landmark permutation is
controlled during sklearn's full Nystroem fit. Tolerances are atol=rtol=2e-8,
with explicit NaN agreement for outside-domain isotonic queries.

The 25 native programs pass in checked/FPE-trapping and release builds. New
independent properties include PAVA primal/dual optimality, weight-scale
invariance, ties and zero weights, nearly equal inputs, seeded repeatability,
Fourier distribution moments and approximation error, exact full-landmark
reconstruction, and rank-deficient/duplicate/zero-vector kernel bases.
The isolated diagnostic suite passes 92 cases. The runnable
`monotone_kernel_features` example also passes. Limits are in `COVERAGE.md`.

## Cross-decomposition addition (2026-11-18)

Added `pls_regression`, `pls_canonical`, `cca`, and `pls_svd`, bringing coverage
to 88 core types in 35 modules. These accept paired dense predictor and
response matrices with multiple response columns. NIPALS regression/canonical
deflation, CCA pseudoinverse updates, canonical direct SVD, and PLSSVD are
supported, together with transforms and the corresponding prediction and
reconstruction contracts. No public extensions or runtime dependencies were
added; MIT original-work licensing and sklearn BSD-3-Clause attribution remain.

`compare_cross_decomposition.py` adds 760 sklearn 1.6.1 comparisons across
four seeded datasets (190 result groups each), bringing the eight-suite
inventory to 2,557. Fixtures include single/multiple responses, more responses
than predictors, duplicate and constant columns, scaling on/off, single-step
capped iterations and constant responses. We compare weights, loadings,
rotations, coefficients, intercepts, training scores, transforms, inverse
transforms, predictions, R2 and iteration counts with atol=rtol=2e-7.

The 26 native programs pass with bounds checks/FPE traps and in release mode.
Independent properties check full linear-model recovery, coefficient scaling,
full predictor/response reconstruction, score orthogonality, perfectly shared
canonical variates, diagonalized PLSSVD cross-covariance and constant-response
early stopping. The isolated invalid-input suite now passes 109 cases. It
resolves the checked executable via fpm's documented `--runner` path listing
and runs each rejection directly in its own process, avoiding repeated fpm
startup while retaining a fresh checked build. The `paired_latent_components`
example passes. API limitations are recorded in `COVERAGE.md`.

## MDS and LLE addition (2026-11-18)

Added `mds` and `locally_linear_embedding`, bringing core coverage to 90 types
in 37 modules. MDS supports metric and nonmetric SMACOF, explicit or seeded
initialization, multi-start selection and precomputed dissimilarities. LLE
supports standard reconstruction and LTSA alignment with barycentric query
transforms. The implementation retains the sklearn 1.6.1 stress/update
semantics; supported limits and degenerate cases are detailed in `COVERAGE.md`.
No extensions or dependencies were added; MIT original-work licensing and
sklearn BSD-3-Clause attribution remain.

`compare_manifold.py` adds 138 reference comparisons on three seeded datasets,
bringing the nine-suite inventory to 2,695 groups. Eight MDS configurations
cover raw/normalized stress, metric/nonmetric fits, missing dissimilarities,
explicit dimensionality, iteration caps, and multi-start selection using
actual Fortran random initial matrices in sklearn. Four LLE configurations
cover standard/LTSA methods, component counts, regularization, neighbors,
weights and out-of-sample geometry. Embedding/query/cross Gram matrices remove
eigenvector signs and rotations. Larger LTSA reference neighborhoods keep all
observations referenced and leave a unique skipped constant mode; truncated
geometry is otherwise nonunique. Tolerances are atol=rtol=2e-7.

The 28 native programs pass checked/FPE-trapping and release builds. New
independent checks cover the exact Guttman update, legacy pre-update stress,
Euclidean distance recovery, random-start selection/reproducibility, ordinal
invariance, all-zero metric data, affine barycentric weights, eigenvector
orthonormality, the LLE reconstruction objective, exclusion of self-neighbors,
and translation/scale invariance of fits and queries. The diagnostic suite
passes 132 isolated cases; `manifold_coordinates` also runs successfully.

A nearly one-dimensional native fixture exposed constant-mode mixing in the
full alignment eigensolve. LLE now solves in an explicit orthonormal basis
perpendicular to constants, preserving centering when the first nonconstant
eigenvalue is small; the sklearn reference fixtures retain the same geometry.

## SVR, NuSVR and OneClassSVM addition (2026-11-18)

Added `svr`, `nu_svr`, and `one_class_svm`, bringing coverage to 93 core types
in 38 source modules. The native pairwise dual solvers support sample weights,
PSD linear/RBF/polynomial/precomputed kernels, gamma rules, regression and
novelty scores. Fitted KKT diagnostics expose convergence. No new algorithm
extensions or runtime dependencies were added. Original contributions remain
MIT; sklearn and bundled LIBSVM BSD-3-Clause notices are retained, including
`LICENSES/libsvm-COPYRIGHT`.

`compare_kernel_svm.py` adds 318 sklearn 1.6.1 comparisons on three seeded
datasets, bringing the ten-suite inventory to 3,013 result groups. Fixtures
cover gamma scale/auto/numeric, weights including zero, four kernels, constants
and empty-support regression. Predictions, scores, intercepts, offsets, gamma
and linear coefficients use atol=rtol=3e-5; unique RBF duals use 2e-4 because
LIBSVM caches kernel values in single precision. Nonunique linear/polynomial
duals are checked through their primal decision functions instead. Original
training-row indices are recovered from sklearn's zero-weight-filtered support
indices before comparison.

An origin-containing linear one-class fixture reached LIBSVM's iteration
limit due to its float kernel cache. The numeric reference instead uses a
nonzero linear boundary at the same tight solver tolerance; a native analytic
check independently covers the zero-boundary case.

All 30 native programs pass checked/FPE-trapping and release builds. New
properties cover dual feasibility/mass, KKT complementarity, epsilon tubes,
NuSVR/learned-epsilon SVR equivalence, precomputed kernels, removal of zero-weight
observations, constant data, empty support, capped convergence status, known
novelty classifications and the one-class offset convention. The diagnostic
suite passes 156 isolated cases. `kernel_regression_outliers` also passes.
Supported restrictions and numeric limits are recorded in `COVERAGE.md`.

A fully bounded linear one-class dual can admit a threshold interval. The
native solver chooses its midpoint, while LIBSVM can choose an endpoint when
roundoff leaves a tiny free coefficient. Numeric linear reference fits use a
fractional mass to determine a unique threshold; a native analytic saturated
dual checks the midpoint convention. Threshold recovery uses relative bound
tolerance, with a fallback for genuinely tiny free masses.

A tight-tolerance symmetric-ring check motivated gain-based selection of the
second dual coordinate and a feasible weighted interior start for OneClassSVM.
The interior start preserves exact symmetry without relaxing the KKT stopping
criterion. The maximum feasible gradient gap is checked independently and
refreshed before reporting convergence.

## PowerTransformer reference suite

Run `python validation/compare_power_transform.py` for 160 result-group checks
against sklearn 1.6.1 on five seeded matrices. Together the fixtures cover
Yeo-Johnson and Box-Cox, with/without standardization, partially missing
observations, strongly skewed positive/negative features, and constant
Yeo-Johnson columns. Learned lambdas, training/query transforms and
inverse transforms use atol=rtol=3e-6; NaN positions must agree. Native search
uses golden-section minimization and a log-scaled likelihood rather than SciPy's
Brent routine, so fitted parameters need not agree in the last digits.

`test_power_transform` independently checks mean/variance, reconstruction,
limiting powers at zero and two, missing values, refitting and fitted settings.
The input harness adds 11 rejection cases (167 total). `power_transforms` is a
small runnable example. The eleven-suite inventory is 3,173 result groups.

## Discretization reference suite

Run `python validation/compare_discretization.py` for 324 result-group checks
against sklearn 1.6.1 (`subsample=None`). Three seeded datasets each exercise
12 configurations: uniform/quantile/k-means, ordinal/dense one-hot, weighted
quantile/k-means, heterogeneous requested bin counts, duplicate quantiles and a
constant column. Counts, output widths, edges, training/query transforms and
midpoint inverses use atol=rtol=2e-10, with NaN positions and infinities compared.
The twelve-suite inventory is 3,497 result groups.

Native properties additionally check exact interior boundaries, out-of-range
clipping, one active bin per feature, transform-inverse bin consistency, known
weighted centers/quantiles, constant inverse NaNs, empty-cluster handling and
fitted encoding retention. The input harness adds 20 rejection cases (187 total).
The `discretization` example demonstrates query codes and approximate inverses.

## GraphicalLasso reference suite

Run `python validation/compare_graphical_lasso.py` for 144 result-group checks
against sklearn 1.6.1. Three seeded datasets each exercise eight configurations:
regularization strengths including zero/fully diagonal, empirical/precomputed
covariance, assume-centered fits, rank-deficient sample covariance, and a
capped two-sweep solver. Locations, covariance/precision matrices, final
objective/gap, query Mahalanobis distances and query likelihoods use
atol=rtol=3e-8. The reference driver runs with bounds checks and FPE traps.
The thirteen-suite inventory is 3,641 result groups.

Native properties check an analytic 2D solution, active/inactive KKT conditions,
unpenalized covariance diagonals, covariance/precision inversion, convergence
history, zero-alpha empirical equivalence, precomputed equivalence and iteration
caps. The input harness adds 20 rejection cases (207 total). The
`sparse_precision` example demonstrates a precomputed sparse precision fit.

## Theil-Sen reference suite

Run `python validation/compare_theil_sen.py` for 120 result-group comparisons
against sklearn 1.6.1 on three seeded contaminated regression fixtures. Seven
configurations use the public estimator with exhaustive subsets: intercept and
no intercept, larger subset size, all-row least squares, rank-deficient all-row
fits, a capped spatial median, and the scalar-median shortcut. An eighth uses
sklearn's own `_lstsq` and `_spatial_median` on the exact Fortran-selected subsets;
this checks the sampled algorithm without assuming identical NumPy random draws.
The input subset indices must be in bounds and distinct within every subset.
Coefficients, intercept, breakdown, population/subset counts, iteration index,
training/query predictions and R2 use atol=rtol=2e-7. The driver runs checked with
FPE traps. The fourteen-suite inventory is 3,761 result groups.

Native properties cover contaminated-line recovery, improvement over OLS,
lexicographic enumeration, full-subset OLS equivalence, sample reproducibility,
valid sampling without replacement, capped convergence, scalar medians,
rank-deficient subset solutions and an underdetermined minimum-norm fit.
The input harness adds 14 rejection cases (221 total). `robust_subsets` is a
complete executable comparing contaminated Theil-Sen and OLS slopes/intercepts.

## Passive-aggressive reference suite

Run `python validation/compare_passive_aggressive.py` for 336 result-group checks
against sklearn 1.6.1 on three seeded fixtures. Eight classifier configurations
cover both PA updates, intercept control, balanced weights, immediate/delayed
averaging, patience stopping, binary/multiclass labels and zero-feature rows.
Eight regressor configurations additionally cover epsilon widths, unreachable
averaging thresholds, zero-row PA-I/II behavior and an all-inside-epsilon tube.
All reference fits disable shuffling, providing exactly the same update order.
Coefficients, intercepts, decisions/predictions, classes, epoch/sample counts,
accuracy and R2 use atol=rtol=3e-8. The driver runs checked with FPE traps.
The fifteen-suite inventory is 4,097 result groups.

Native properties independently check analytic PA-I/II steps, intercept norms,
averaging, zero-row differences, patience and fixed epochs, regression and
multiclass recovery, binary ties and native shuffle reproducibility. The input
harness adds 20 rejection cases (241 total). `passive_aggressive` is a complete
example for both models. Native shuffle results are not equated to NumPy draws.


## OPTICS graph and cluster extraction

`python validation/compare_optics.py` passes 432 result groups against sklearn
1.6.1 across three seeded, variable-density datasets. Twelve configurations
per fixture cover precomputed, Manhattan and Minkowski distances, finite and
zero search radii, duplicate points, fractional neighborhood/cluster sizes,
Xi thresholds, predecessor correction on/off, default cluster size, and
DBSCAN fits. Ordering, predecessors, labels, hierarchy endpoints and resolved
sizes match exactly. Core and reachability distances use atol=rtol=3e-14,
including identical infinity masks. Both public extraction helpers are
compared at additional radii/Xi settings without rebuilding the graph.
The sixteen-suite inventory is now 4,529 result groups; unchanged older Python
suites were not rerun for this batch.

Native OPTICS properties check a known nested hierarchy, precomputed/feature
agreement, DBSCAN radius extraction, hierarchy clearing on refit, disconnected
roots, duplicate plateaus, stable row-order ties, fractional sizes and the
sklearn convention for infinite DBSCAN eps. The input harness adds 26 isolated
rejection cases for estimator controls, precomputed matrices and extraction
graph validation (267 total). `optics_demo` demonstrates both extraction methods.


## BIRCH clustering-feature tree

`python validation/compare_birch.py` passes 237 result groups against sklearn
1.6.1 across three seeded datasets and ten configurations each. Fixtures cover
small and large branching factors, deep splits, several radius thresholds,
duplicates, no global clustering, Ward clustering, insufficient subclusters,
compute_labels=false, streamed insertion and global-only reclustering.
Checks include every centroid, its global label, batch labels, training/query
predictions, training/query distance matrices and sample/feature/fallback
metadata. Farthest-pair floating-point ties can reverse split orientation;
centroid-indexed outputs are aligned by a one-to-one minimum-distance
assignment, then every centroid must agree within atol=rtol=2e-12. Cluster
partitions use a shared label mapping after alignment. Distance outputs use
atol=2e-7, rtol=2e-12 to accommodate sklearn's cancellation near zero distances.
Global-only updates retain previous batch labels; that retained partition is
checked separately from current predictions. The seventeen-suite inventory
is now 4,766 result groups; unchanged older reference suites were not rerun.

Native BIRCH properties independently check streamed/batch equivalence under
multi-level tree splitting, sample counters, nearest-centroid label assignment,
Ward grouping, centroid statistics, refit reset, copy ownership, single-point
fits, skipped label computation and distance transforms. The rejection harness
adds 15 BIRCH control/state/shape cases (282 total). `birch_streaming` demonstrates
insertion in batches and global-only Ward clustering.


## QuantileTransformer distribution mapping

`python validation/compare_quantile_transform.py` passes 252 result groups
against sklearn 1.6.1 across three seeded datasets and twelve configurations
each. The suite checks uniform and normal outputs, small/full/capped/single
quantile tables, duplicates, constant and all-missing columns, scattered NaNs,
finite queries into missing tables, near-boundary tolerance, extrapolation,
and inverse uniform/normal probes beyond fitted bounds. Compared outputs are
quantile/reference tables, fitted dimensions, training/query transforms,
inverse probes and forward/inverse composition. Every numeric output uses
atol=2e-10, rtol=2e-11, with matching NaN masks. Reference fits use every row
(subsample=0 in Fortran, None in Python). The eighteen-suite inventory is
5,018 result groups; unchanged older Python suites were not rerun.

Native properties independently verify duplicate-rank averaging, analytic
uniform scores, normal median/tail behavior, endpoint inverse consistency,
NaN preservation, all-missing tables, constant columns, quantile-count capping,
single-quantile tables, monotonicity and seeded native subsampling. Native
random selections are not equated to NumPy selections. The rejection harness
adds 14 control/state/shape/infinity cases (296 total). `quantile_transform`
demonstrates uniform and normal scores and recovery of the original data.


## Additive and skewed chi2 samplers

`python validation/compare_chi2_samplers.py` passes 204 result groups against
sklearn 1.6.1 on three seeded datasets. Eight additive configurations cover
default intervals for steps 1..3, explicit intervals and larger step counts,
zero interval, large interval and zero-valued entries. Six skewed configurations
cover positive/zero/negative skew, one and many components, and negative input
values within the shifted domain. Compared outputs include training/query
features, feature Gram matrices, dimensions and the skewed weights/offsets.
All outputs use atol=rtol=3e-12. Raw native uniform draws are supplied to
sklearn's full skewed fit via a controlled random-state object, independently
checking inverse-CDF sampling as well as transform algebra. The raw draws are
range-checked and are not counted as comparison groups. NumPy and Fortran RNG
streams are intentionally not equated. The nineteen-suite inventory is now
5,222 result groups; unchanged older reference suites were not rerun.

Native properties independently check the analytic first additive block,
stateless transformation, zeros, dimensions, zero/large intervals, positive
additive-kernel approximation, multiplicative skewed-kernel approximation,
feature scaling bounds, known weights/offsets, seeded reproducibility, and
skewed fit's dependence on dimensions rather than training values. The input
harness adds 22 control/state/shape/domain checks (318 total), including an
output-width overflow guard. `chi2_kernel_features` demonstrates both maps.


## GradientBoostingClassifier

`python validation/compare_gradient_boosting.py` passes 510 result groups
against sklearn 1.6.1 across three seeded datasets and ten configurations each.
Checks cover binary/multiclass log loss, binary exponential loss, prior/zero
initialization, learning rates, shallow/deeper trees, minimum leaf/split sizes,
nonzero Friedman impurity thresholds, and a noisy-label fixture. All numeric
comparisons use atol=rtol=3e-8. Outputs include classes, priors, initial raw
scores, complete loss histories, importances, fitted dimensions, train/query
decisions, probabilities/log probabilities, labels, accuracy, all staged
outputs and individual tree predictions. Native 1-based leaf IDs are checked
for positive integer values and each native leaf must map to a constant reference tree
prediction; redundant constant-valued sklearn subdivisions are allowed.
Reference inputs have one varying feature to avoid differences in feature-tie
selection. The native suite separately checks two-feature multiclass learning.
The twenty-suite inventory is now 5,732 result groups; unchanged older Python
reference suites were not rerun.

Native properties independently verify analytic binary/exponential/multiclass
Newton steps, frozen multiclass gradients and the (K-1)/K correction, deviance
scaling, probability links/normalization, split-based importance normalization,
staged/final consistency, leaf tensor mapping, deterministic refitting, tie
labels and finite log probabilities under saturated probabilities. The input
harness adds 24 control/state/shape/domain cases (342 total).
`boosted_classification` demonstrates training and class probabilities.

The native tree's normalized squared-error threshold is divided by the
training count to implement sklearn's Friedman SSE-reduction threshold.
This conversion was checked with nonzero-threshold reference configurations.
Native tree tie handling and float64 thresholds can still differ from
sklearn's tree backend. Supported limits are recorded in COVERAGE.md.


## RidgeClassifier and RidgeClassifierCV

`python validation/compare_ridge_classification.py` passes 420 result groups
against sklearn 1.6.1 on three seeded fixtures, including a wide design.
Ten classifier and seven CV configurations cover binary/multiclass labels,
weighted/unweighted centering, absent intercepts, balanced/explicit class
weights, zero-weight rows, zero penalty and duplicated features. Comparisons
include coefficients, intercepts, sorted labels, train/query decisions,
predictions, accuracy, selected penalties, candidate CV scores and stored
leave-one-out errors, using atol=rtol=3e-8. Scalar zero intercepts from sklearn
are expanded to the native per-output vector for comparison. The twenty-one
reference suites now contain 6,152 groups; older suites were not rerun.

Native properties check analytic weighted slopes/intercepts, equivalence of
row and class weights, explicit leave-one-out refits, weighted negative-MSE
averaging, selected-penalty refitting, rank-deficient minimum-norm solutions,
finite wide-design CV results, clearing optional stored results and binary
tie labels. The isolated input harness adds 28 cases (370 total), covering
penalties, weights, labels, state, shapes and finite input requirements.
`ridge_classification` demonstrates default leave-one-out penalty selection.

The implementation shares one weighted dense SVD across candidate penalties.
Its CV mode implements sklearn's default cv=None, scoring=None: selection by
negative squared error on binarized targets. Explicit folds and custom scorers
are outside this batch. Supported limits and weighting/shape conventions are
recorded in COVERAGE.md. Both types are exported through sklearn_linear_model
and the sklearn facade. Original code remains MIT with sklearn BSD attribution;
no dependency or public extension was added. Inventory: 107 core types in
43 source modules, with 41 native test programs.


All 41 native test programs passed checked builds with bounds/backtrace and
invalid/zero/overflow traps, and optimized release builds. All 370 isolated
input rejection cases passed. The checked ridge_classification example passed,
selecting alpha=1 with training accuracy 0.75 on its small fixture. The 420 new
sklearn comparison groups passed. Python syntax, Fortran formatting, inventory
and diff checks passed. MIT licensing and sklearn BSD notices are retained.


## MultiTaskElasticNet and MultiTaskLasso

`python validation/compare_multitask.py` passes 192 result groups against
sklearn 1.6.1 across three seeded tall/wide fixtures and eight configurations.
Cases cover Lasso, mixed elastic penalties, ratio=1 equivalence, absent
intercepts, complete group shrinkage, one-iteration fits, warm starts and a
single response column. Fixtures include constant and duplicated features.
Checks compare coefficients, intercepts, train/query predictions, uniform R2,
duality gaps, scaled stopping tolerances and iteration counts, using
atol=rtol=3e-8. The twenty-two reference suites now contain 6,344 groups;
older Python reference suites were not rerun.

Independent native checks verify analytic group soft-threshold solutions,
mixed L1/L2 and ridge shrinkage, whole-feature removal, zero columns, warm/cold
solution agreement, cold refitting, response means for constant designs and
zero-target convergence/scoring. The input harness adds 24 shape/state/domain
cases (394 total), including warm-start feature/target mismatches. The
multitask_regression example demonstrates shared feature coefficients.

Both estimators fit rank-two target arrays using native cyclic block coordinate
descent and primal-dual stopping. MultiTaskLasso always uses mixing ratio=1.
Positive alpha and mixing ratio are required in this port. Sparse data,
weights, random coordinate selection and CV/path variants are outside this
batch; see COVERAGE.md. Original contributions remain MIT with sklearn BSD
notices retained. No runtime dependency or public extension was added.
Inventory: 109 core types in 44 source modules and 42 native test programs.


All 42 native test programs passed checked builds with bounds/backtrace and
invalid/zero/overflow traps, and optimized release builds. All 394 isolated
input rejection cases passed. The checked multitask_regression example passed
with uniform training R2 about 0.99994. All 192 new sklearn comparison groups
passed, including explicit output-shape checks. Python syntax, Fortran
formatting, inventory and diff checks passed. MIT licensing and sklearn BSD
notices are retained.


## CategoricalNB

`python validation/compare_categorical_nb.py` passes 480 result groups against
sklearn 1.6.1 across three seeded fixtures and ten configurations each.
Cases cover multiple feature cardinalities, scalar smoothing, empirical and
uniform priors, weighted counts, explicit/zero priors, scalar/vector reserved
category ranges, incremental fitting with initially absent classes and
single-class targets. Checks cover classes, class/category counts, retained
widths, feature log probabilities, class log priors, query likelihoods and
probabilities, training/query labels and accuracy. Numerical outputs use
atol=rtol=3e-8. Predicted classes must maximize the reference log likelihood
within 1e-12, allowing roundoff between mathematically tied classes; clear
winners must match. Accuracy is independently recomputed from the validated
training labels. The twenty-three reference suites now contain 6,824 result
groups; unchanged older Python suites were not rerun.

Native properties check a hand-derived weighted posterior and smoothed counts,
batch/incremental agreement, category expansion, absent-class zero priors,
retention of widths after a smaller batch, reserved categories, explicit zero
priors, single-class normalization, zero-weight class counts and exact-tie
labels. The input harness adds 30 state/shape/domain cases (424 total).
The categorical_classification example demonstrates two incremental batches.

The native implementation uses ragged per-feature dense tables. Fractional
codes are rejected, and n_categories_ consistently records retained table
widths across partial_fit calls (unlike sklearn 1.6.1 latest-batch metadata).
The public categorical_table storage helper is excluded from estimator counts.
Supported limits and prior/weight/category conventions are in COVERAGE.md.
The package now has 110 core types in 45 source modules, with 43 native test
programs. Original contributions remain MIT with sklearn BSD attribution;
no runtime dependency or public algorithm extension was added.


All 43 native test programs passed checked builds with bounds/backtrace and
invalid/zero/overflow traps, and optimized release builds. All 424 isolated
input rejection cases passed. All 480 new sklearn comparison groups passed.
The checked categorical_classification example passed, predicting class 9
with probabilities [0.4,0.6]. Python syntax, Fortran formatting, module/test
inventory and staged diff checks passed. MIT licensing and sklearn BSD notices
are retained.


## KNNImputer and MissingIndicator

`python validation/compare_knn_impute.py` passes 156 result groups against
sklearn 1.6.1 across three seeded datasets. Eight imputer configurations cover
one/three/default/excess neighbors, uniform/distance weights, all-missing
training columns, retained empty features and appended indicators. Queries
include newly missing features and an entirely missing row. Three indicator
configurations cover missing-only/all modes and strict/permissive transforms.
Checks compare fitted/query transformations, output shapes, valid-column masks
and 1-based indicator indices at atol=rtol=3e-8. Continuous complete anchor
columns avoid ambiguous neighbor cutoff ties. The twenty-four reference suites
now contain 6,980 result groups; older Python suites were not rerun.

Independent native checks verify hand-derived uniform and inverse-distance
averages, zero-distance weighting, mean fallback without shared observations,
empty-feature dropping/zero retention, original missingness indicators, input
preservation, all-empty outputs, stable repeated transforms and indicator modes.
The input harness adds 21 shape/state/control/infinity cases (445 total).
The neighbor_imputation example fills a missing value from two nearest donors.

Distances use original observed receiver values throughout the transform.
Undefined donor distances are ignored; neighbor ties retain training order,
which can differ from sklearn's partition tie handling. Only NaN sentinels,
nan-Euclidean distances and dense numeric arrays are supported. MissingIndicator
returns dense logical arrays; the imputer appends numeric 0/1 indicators.
Supported controls, shape rules and limits are recorded in COVERAGE.md.
Inventory: 112 core types in 46 source modules, with 44 native test programs.
Original contributions remain MIT with sklearn BSD attribution. No runtime
dependency or public algorithm extension was added.


All 44 native test programs passed checked builds with bounds/backtrace and
invalid/zero/overflow traps, and optimized release builds. All 445 isolated
input rejection cases passed. All 156 new sklearn comparison groups passed.
The checked neighbor_imputation example passed, completing the row as [1,15].
Python syntax, Fortran formatting, module/test inventory and staged diff checks
passed. MIT licensing and sklearn BSD notices are retained.


## GaussianProcessClassifier

`python validation/compare_gp_classification.py` passes 312 result groups
against sklearn 1.6.1 across three seeded datasets and eight configurations.
Cases cover binary/three-class labels, several kernel amplitudes and length
scales, one/two-iteration caps, duplicated features and one-vs-rest fitting.
The reference explicitly fixes ConstantKernel*RBF and disables optimization.
Checks compare sorted classes, full posterior pi/W/latent-mode/alpha state,
Cholesky factors, per-channel and aggregate Laplace evidence, train/query
probabilities, labels and accuracy, using atol=rtol=3e-8. The twenty-five
reference suites now contain 7,292 result groups; older Python suites were
not rerun.

Independent native checks verify analytic one-step evidence for duplicate
observations, a symmetric posterior, iteration-cap/convergence diagnostics,
OVR normalization and evidence averaging, exact binary/multiclass tie rules,
separable labels, bounded/normalized probabilities, fitted-kernel retention,
cold-refit determinism and far-field prior probabilities. The input harness
adds 21 control/state/shape/finite-data cases (466 total). The
Gaussian-process classification example reports accuracy and probabilities.

The native implementation uses dense Newton/Cholesky Laplace updates and the
reference's five-erf logistic-normal approximation. The probability integral
is algebraically simplified to avoid division by near-zero variance. Kernel
optimization, alternate kernels, weights, one-vs-one modes and warm starts are
outside this batch; see COVERAGE.md for state, stopping and shape conventions.
Original contributions remain MIT with sklearn BSD attribution retained. No
runtime dependency or public algorithm extension was added. Inventory: 113
core types in 47 source modules, with 45 native test programs.


All 45 native test programs passed checked builds with bounds/backtrace and
invalid/zero/overflow traps, and optimized release builds. All 466 isolated
input rejection cases passed. All 312 new sklearn comparison groups passed.
The checked gaussian_process_classification example passed with training
accuracy 1.0 and finite class probabilities. Python syntax, Fortran formatting,
module/test inventory and staged diff checks passed. MIT licensing and sklearn
BSD notices are retained.


## SplineTransformer

`python validation/compare_spline.py` passes 180 result groups against sklearn
1.6.1 across three seeded datasets and twelve configurations. Cases cover
degrees 1/2/3/10, uniform/quantile/explicit nonuniform knots, all five
extrapolation modes, periodic folding and per-feature bias removal. Checks
compare complete train/query bases, base and extended knots, and input/output
dimensions at atol=rtol=3e-8. Queries include endpoints and points beyond both
sides of the interval. The twenty-six reference suites now contain 7,472 result
groups; older Python suites were not rerun.

Independent native properties verify an analytic degree-one basis, right
endpoints, constant/linear/continued extrapolation, periodic wrapping, a zero-
column periodic result, cubic partition of unity, nonnegativity in range,
per-feature bias removal, fitted-setting retention and explicit knot behavior.
The reference check exposed a one-ulp uniform upper-endpoint error; placement
now preserves extrema directly, with a regression test ensuring training rows
are accepted in error mode. The input harness adds 23 control/state/shape/domain
cases (489 total), including output-dimension overflow. The spline_features
example demonstrates cubic feature counts and basis sums.

Native Cox-de Boor and derivative recurrences replace runtime SciPy BSpline
objects. Positive degrees are supported; degree zero, sample-weighted knot
placement, repeated knots, sparse output and feature-name APIs are outside
this batch. Constant or duplicate-quantile knot columns are rejected. Full
controls, endpoint and shape conventions are recorded in COVERAGE.md.
Original contributions remain MIT with sklearn BSD attribution retained.
No runtime dependency or public algorithm extension was added. Inventory:
114 core types in 48 source modules, with 46 native test programs.


All 46 native test programs passed checked builds with bounds/backtrace and
invalid/zero/overflow traps, and optimized release builds. All 489 isolated
input rejection cases passed. All 180 new sklearn comparison groups passed.
The checked spline_features example passed with five cubic basis columns and
row sums equal to one within roundoff. Python syntax, Fortran formatting,
module/test inventory and staged diff checks passed. MIT licensing and sklearn
BSD notices are retained.


## LinearSVC and LinearSVR

`python validation/compare_linear_svm.py` passes 384 result groups against
sklearn 1.6.1 across three seeded fixtures, including a wide design. Ten
classifier and eight regressor configurations cover binary/multiclass labels,
absent/scaled intercepts, small/large C, weighted samples, balanced/explicit
class weights, duplicated features, several epsilon widths and a fully inactive
regression loss. The reference uses dual=False and the supported squared losses.
Checks compare coefficients, intercepts, classes, train/query decisions or
predictions, labels, scores and per-channel primal objectives. The twenty-seven
reference suites now contain 7,856 result groups; older Python suites were not
rerun.

Classifier coefficients/intercepts/decisions use atol=rtol=1e-6; all other
outputs, including objectives and labels, use 3e-8. LIBLINEAR TRON can stop when
actual and predicted objective reductions fall below 1e-12 times the objective,
even with tighter requested tolerances. Observed classifier decision differences
were below 5e-7. For an inspected case, native and reference objectives agreed
within roundoff while gradient norms were about 8e-17 and 1e-7 respectively.
Every fitted native channel is independently checked for primal stationarity
(norm(gradient)<=1e-8*max(1,norm(augmented coefficients))) and its reported
objective is recomputed at atol=rtol=1e-10. This verifies optimizer accuracy
independently of LIBLINEAR's stopping floor.

Native tests verify analytic squared-hinge and squared-epsilon optima, zero-
epsilon ridge equivalence, synthetic-intercept regularization, binary class/
sample weight equivalence, repeated-observation equivalence, iteration caps,
zero-decision ties and stationary zero-step fits. The input harness adds 33
control/state/shape/finite-data cases (522 total). linear_support_vectors
demonstrates classification and regression.

The native dense primal solver uses active-set Newton steps and Armijo search.
It implements L2 squared losses only; other penalties/losses, dual solvers and
Crammer-Singer classification are outside this batch. Multiclass OVR class
weights affect positive examples only, matching LIBLINEAR. Full controls,
objective, weighting and stopping conventions are recorded in COVERAGE.md.
The bundled LIBLINEAR BSD-3-Clause notice is retained alongside sklearn notices.
Original contributions remain MIT. No runtime dependency or public algorithm
extension was added. Inventory: 116 core types in 49 source modules, with
47 native test programs.


Final validation: all 47 native test programs passed in checked mode
(-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow) and release mode.
All 384 linear SVM reference groups and 522 isolated invalid-input cases passed.
The checked linear_support_vectors example returned classification accuracy 1
and regression predictions approximately [-3.809524,-1.904762,1.904762,3.809524].
Python syntax, Fortran line-length/whitespace, git diff and byte-identical
LIBLINEAR notice checks passed.


## IncrementalPCA

Added incremental_pca with batched fit, streaming partial_fit, transform,
inverse_transform and fit_transform. Native tests cover full-rank agreement
with PCA, mean/population variance, whitening, reconstruction, single-row
updates, fit reset and constant-data ratios. The Python reference suite checks
264 result groups (eight configurations on three seeded datasets), including
components, moments, singular values, variance/ratios, noise estimates,
transforms, reconstruction and fitted shape. Tolerances are atol=rtol=3e-8.
Configurations include default/explicit batches, rank truncation, full rank,
automatic rank, whitening and externally supplied batches with single-row updates.

Fifteen new invalid-input cases cover control values, first-batch size, changing
state dimensions, nonfinite/empty inputs, query shapes and unfitted methods
(537 total cases). See COVERAGE.md for dense-SVD scope and intentional finite
constant-data handling. Attribution points to sklearn 1.6.1's incremental PCA
implementation; original contributions remain MIT. Inventory: 117 core types,
50 source modules, 48 native test programs and 28 reference suites (8,120 groups).


Final validation: all 48 native test programs passed with runtime checks and
floating-point traps (-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow)
and in release mode. All 264 reference result groups and 537 isolated
invalid-input cases passed. The checked streaming_pca example processed six
samples and returned a leading explained-variance ratio of 1 within roundoff.
Python syntax, Fortran formatting and git diff checks passed.
