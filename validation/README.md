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

The reference validator requires NumPy, SciPy and scikit-learn in Python.
compare_lda.py additionally requires mpmath for 60-digit scoring certificates
(`python -m pip install mpmath`). The
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


## FeatureAgglomeration

Added feature_agglomeration with complete Euclidean Ward/single/complete/average
feature trees, mean/median/maximum pooling, fit_transform and inverse_transform.
The reference suite checks 294 result groups: fourteen configurations across
three seeded matrices, including a wide matrix. It compares feature partitions,
full merge trees (ignoring sibling ordering), merge distances, pooled training/
query values, reconstructed inputs and fitted shape attributes. Cluster IDs
are canonicalized by first feature, since sklearn's numbering is arbitrary.
All numeric comparisons use atol=rtol=3e-8; no geometric ties are manufactured.

Native tests check known feature pairs, pooled values, inverse replication,
projection idempotence, median/max pooling, tree size and the identity limit.
Fifteen new invalid cases cover cluster counts, linkage/pooling controls,
empty/nonfinite inputs, feature count, pooled width and unfitted methods
(552 total cases). See COVERAGE.md for tie handling and dense-tree limits.
The feature_pooling example demonstrates compression of correlated columns.
Inventory: 118 core types, 50 modules, 49 native tests, 29 reference suites and
8,414 result groups. Original contributions remain MIT with upstream sklearn
BSD attribution; no dependency or public extension was added.


Final validation: all 49 native test programs passed with runtime checks and
floating-point traps (-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow)
and in release mode. All 294 FeatureAgglomeration reference groups and 552
isolated invalid-input cases passed. The checked feature_pooling example
returned labels [0,0,1,1] and first-row pooled values [1.05,20.1]. Python syntax,
Fortran line-length/whitespace, inventory and git diff checks passed.


## ElasticNetCV and LassoCV

Added single-target elastic_net_cv and lasso_cv with deterministic K-fold alpha/
l1-ratio selection and full-data refitting. Explicit descending alpha grids,
automatic geometric grids, multiple elastic-net ratios and full fold MSE paths
are supported. Native tests cover sorting, path shapes, exclusion of validation
rows from fitting, selected-model refitting, mean-fold error selection, exact
ties, constant targets and the Lasso ratio override.

The Python reference suite checks 216 result groups across three seeded
matrices and nine configurations: explicit/automatic grids, multiple ratios,
no intercept, singleton alpha, duplicate features and Lasso variants. It checks
coefficients, intercepts, selected parameters/errors, grids, every fold MSE,
training/query predictions and R2, with atol=rtol=3e-8. The native solver uses
independent cold starts and its existing coordinate-change stopping rule;
iteration parity with sklearn is not claimed.

Twenty-three new invalid-input cases cover folds, iteration controls, ratio/
alpha grids, automatic grid controls, input shapes/finite values and prediction
state/width (575 total cases). regularization_cv demonstrates model selection.
See COVERAGE.md for array layout, tie handling, defaults and scope. Original
contributions remain MIT with sklearn BSD attribution. Inventory: 120 core
types, 50 source modules, 50 native tests, 30 reference suites and 8,630 groups.
No runtime dependency or public extension was added.


Final validation: all 50 native test programs passed with runtime checks and
floating-point traps (-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow)
and in release mode. All 216 coordinate CV reference groups and 575 isolated
invalid-input cases passed. The checked regularization_cv example selected
alpha=.01 and l1_ratio=1 with training R2 approximately .9998865. Python syntax,
Fortran formatting, reference report, inventory and git diff checks passed.


## AdaBoostClassifier (SAMME stumps)

Added adaboost_classifier with weighted Gini decision stumps, binary/multiclass
SAMME, learning-rate control, sample weights, early stopping, importance,
probabilities and staged decisions/probabilities/predictions. Native tests cover
analytic stump error and log weight, sklearn 1.6 binary decision scaling,
probability normalization/conversion, perfect-fit stopping and agreement of
final/staged outputs. Weak learners use float32 comparisons as in sklearn.

The reference suite checks 216 result groups over three seeded datasets and
six configurations: weighted binary/multiclass, unweighted one-feature data,
learning rate, one learner and perfect-fit stopping. Checks include padded
learner errors/weights, class labels, importance, decisions, query probabilities/
labels, all staged outputs and fitted dimensions, with atol=rtol=3e-8.

Twenty new isolated invalid-input cases cover estimator/rate controls, weights,
labels, random-guessing failure, finite data, float32 range, prediction shapes
and unfitted methods (595 total). adaboost_stumps demonstrates classification.
See COVERAGE.md for fixed weak-learner scope and deterministic split ties.
Inventory: 121 core types, 51 modules, 51 native tests, 31 reference suites and
8,846 result groups. Original contributions remain MIT with sklearn BSD notice
and attribution retained. No public extension or runtime dependency was added.


Final validation: all 51 native test programs passed with runtime checks and
floating-point traps (-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow)
and in release mode. All 216 AdaBoost reference groups and 595 isolated
invalid-input cases passed. The checked adaboost_stumps example fitted ten
stumps and returned training accuracy 1. The sklearn reference emits its
expected log(0) warning for original zero sample weights; the native log update
avoids that operation and passed floating-point traps. Python syntax, Fortran
formatting, reference report, inventory and git diff checks passed.


## SGDRegressor

Added sgd_regressor with ordered/native-shuffled dense SGD, squared-error/
Huber/epsilon-insensitive/squared-epsilon losses, L2/no penalty, constant and
inverse-scaling rates, sample weights, iterate averaging and streaming
partial_fit. Native tests verify analytic updates, sample clock, averaging,
batch continuation, zero-weight L2 decay, stopping and seeded native shuffle.

The reference suite checks 216 result groups across three seeded datasets and
twelve configurations: all losses, both rates, delayed/immediate averaging,
partial_fit batches including one row, training-loss stopping, no intercept,
clipped L2 decay and all-zero sample weights. It compares coefficients,
intercepts, train/query predictions, R2 and exact epoch/sample counters at
atol=rtol=3e-8. Exact comparisons use shuffle=False; native RNG permutation
identity with sklearn is not claimed.

Twenty-four isolated invalid-input cases cover loss/penalty/rate and numerical
controls, sample weights, shapes/finite data, prediction state and fixed
partial_fit width/averaging configuration (619 total cases). The
streaming_regression example demonstrates repeated streaming updates.
Inventory: 122 core types, 52 modules, 52 native tests, 32 reference suites and
9,062 result groups. See COVERAGE.md for ordered-default behavior and scope.
Original contributions remain MIT with sklearn BSD attribution retained.
No runtime dependency or public extension was added.


Final validation: all 52 native test programs passed with runtime checks and
floating-point traps (-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow)
and in release mode. All 216 SGD reference groups and 619 isolated invalid-input
cases passed. Reference stopping tests matched early stopping at 78/68 epochs
and a 300-epoch capped fit; sklearn's convergence warning for that capped case
is expected. The checked streaming_regression example processed 600 samples
and returned R2 approximately .999999983. Python syntax, Fortran formatting,
reference report, inventory and git diff checks passed.


## SGDClassifier

Added sgd_classifier using the shared native SGD engine with hinge, squared-
hinge, perceptron, logistic and modified-Huber classification losses. Binary and
multiclass OVR fits, declared-class streaming partial_fit, averaging, sample/
class weights and logistic/modified-Huber probabilities are supported. Native
tests cover analytic hinge steps, class discovery/ties, one-class batches,
reset/streaming behavior, averaging, probability normalization and seeded native
shuffle reproducibility.

The reference suite checks 306 result groups across three seeded matrices and
twelve configurations. It covers all losses, binary/multiclass fits, explicit/
balanced class weights, sample weights, no intercept, ordered constant/inverse
rates, immediate/delayed averaging, a one-class initial batch, one-row update,
fit-then-partial_fit, training-loss stopping, decisions/labels/accuracy,
probabilities and exact epoch/sample clocks. Tolerances are atol=rtol=3e-8.
The existing 216 SGDRegressor groups are rerun after the shared-engine change.

Thirty-one new isolated input cases cover class metadata, changing streaming
state, class/sample weights, loss selection, probabilities, finite data, shapes
and unfitted calls (650 total). streaming_classification demonstrates logistic
streaming. See COVERAGE.md for default-rate differences and fixed subset scope.
Inventory: 123 core types, 52 modules, 53 native tests, 33 reference suites and
9,368 result groups. Original contributions remain MIT with retained sklearn
BSD attribution. No dependency or public extension was added.


Scope adjustment from reference inspection: sklearn 1.6.1 does not consistently
retain its standard classifier intercept after publishing averages, in both
binary and multiclass continuation. Averaged classifier partial_fit therefore
requires fit_intercept=false in this port. Nonaveraged streaming and averaged
fit retain intercept support. The reference checks binary and multiclass
averaged no-intercept streaming, including a one-class initial batch, one-row
continuation and fit-then-partial_fit.


Final validation: all 53 native test programs passed with runtime checks and
floating-point traps (-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow)
and in release mode. All 306 SGDClassifier groups and the rerun 216
SGDRegressor groups passed against sklearn 1.6.1. All 650 isolated invalid-input
cases passed. The checked streaming_classification example processed 600
samples and returned accuracy 1. Python syntax, Fortran formatting, both
reference reports, inventory and git diff checks passed. The supported
averaged-streaming boundary (fit_intercept=false) is documented and validated.


## 2026-11-19: Group and explicit-fold splitters

Added GroupKFold, LeaveOneOut, LeaveOneGroupOut and PredefinedSplit as native
Fortran types. GroupKFold implements unshuffled largest-group-first balancing;
ties use descending numeric group labels and may differ from NumPy quicksort.
The leave-out splitters use ascending sample/group order. PredefinedSplit
accepts per-call assignments, preserves -1 training-only rows, supports other
negative labels, and matches zero-fold/empty-training edge cases.
All outputs are materialized, ascending 1-based index arrays. LeaveOneOut has
quadratic index storage. No new cross_val_score estimator dispatch was added.

The new comparison suite checks 362 result groups against sklearn 1.6.1 across
three seeded group datasets with two split counts each. It compares exact
fold counts and every train/test index, including uneven/tied groups,
noncontiguous and negative labels, training-only rows, empty assignments and
empty training folds. Seven isolated invalid-input cases bring the total to
657. The group_validation example fits and scores four held-out groups.
Original contributions remain MIT with sklearn BSD attribution retained.
Inventory: 127 core types, 52 source modules, 54 native test programs,
34 Python reference suites and 9,730 reference result groups.


Final validation: all 54 native test programs passed with runtime checks and
floating-point traps (-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow)
and in release mode. All 362 new splitter result groups and all 192 rerun
baseline result groups passed against sklearn 1.6.1. All 657 isolated input-error
cases passed. The checked group_validation example returned R2=1 on each of
four held-out groups. Python syntax, Fortran line lengths, inventory, reference
report and staged diff checks passed. No dependency or extension was added.


## 2026-11-19: SequentialFeatureSelector for regression

Added a native greedy forward/backward selector for the linear-regression
family, with fixed counts, auto half-selection, tolerance-based auto stopping,
unshuffled K-fold mean R2 scoring, logical support masks, fit_transform,
transform and zero-filled inverse_transform. Candidate fits use independent
estimator copies and leave the caller's estimator unchanged. The initial
subset does not support classification, fractional counts or custom scorers/CV.
R2 folds must each contain at least two test samples. See COVERAGE.md for
sklearn 1.6.1's odd-feature backward auto behavior and tolerance rules.

The reference suite passed 225 result groups across three seeded matrices and
15 configurations: OLS/ridge/lasso/elastic net, both directions, fixed and auto
counts, positive/negative tolerance, first-step stopping, ignored stopping for
fixed counts, and one/p-1 selected features. Every support mask, fitted count,
training/query transform and inverse projection matched sklearn 1.6.1.
Twenty-one new isolated input-error cases bring the total to 678.
The sequential_selection example selects informative columns before fitting.
Inventory: 128 core types, 53 source modules, 55 native test programs,
35 reference suites and 9,955 result groups. Original contributions remain MIT
with retained sklearn BSD attribution. No dependency or public extension added.


Final validation: all 55 native test programs passed with runtime checks and
floating-point traps (-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow)
and in release mode. All 225 new reference result groups passed against
sklearn 1.6.1, and all 678 isolated input-error cases passed. The checked
sequential_selection example selected columns 2 and 4 and returned R2=1.
Python syntax, Fortran line lengths, inventory, the reference report and staged
diff checks passed. The initial estimator scope is regression only; no new
dependency or public extension was added.


## 2026-11-20: Recursive Feature Elimination and RFECV

Added native RFE and RFECV with linear-regression-family and logistic-regression
estimators, squared coefficient importances, iterative elimination/ranking,
integer counts, fractional/integer step widths, fitted estimators, transformation,
prediction, probabilities and type-dispatched scoring. RFECV evaluates complete
per-fold paths, selects the feature count (fewer on exact score ties), and refits
on all data. R2 uses unshuffled KFold; accuracy uses first-appearance-encoded
StratifiedKFold, including imbalanced classes smaller than cv. Score/count grids,
per-fold scores and population standard deviations are retained in cv_results_.
Caller estimators remain unchanged and refits clear the other estimator kind.

The new suite passed 1,132 result groups across four datasets and 20 settings:
OLS/ridge/lasso/elastic net, binary/multiclass logistic models, default/fixed/all
feature counts, nondivisible/fractional/oversized steps, varying CV/minimums,
no intercept, strongly imbalanced classes, all masks/ranks/projections, final
coefficients/intercepts/predictions, probabilities/decisions and every CV score.
Of these, 924 groups use default sklearn coefficient ordering; 208 regularized
regression groups use a sklearn importance_getter that standardizes exact-zero
coefficient ties to stable input order. Multinomial intercepts/decisions are
centered over classes to remove their unidentifiable common shift; probabilities
and labels are checked directly. Numerical tolerances are atol=rtol=3e-8.

Forty-two new input cases bring the total to 720. Native tests include an exact
CV-score/coefficient tie, estimator independence, fitted mode changes, ranking,
score-grid ordering and count/step boundaries. The recursive_selection example
shows ridge RFECV selecting columns before prediction. See COVERAGE.md for the
supported estimators, deterministic tie rule, R2 fold sizes and classification
minimum counts. Sparse arrays, custom scorers/CV, tree importance getters,
fractional feature counts and multiple targets remain outside this subset.
Inventory: 130 core types, 54 modules, 56 native test programs, 36 reference
suites and 11,087 reference result groups. Original contributions remain MIT
with sklearn BSD attribution retained. No dependency or extension was added.


Final validation: all 56 native test programs passed with runtime checks and
floating-point traps (-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow)
and in release mode. All 1,132 RFE/RFECV reference groups passed against
sklearn 1.6.1 under the documented coefficient-tie and multinomial-gauge
conventions. All 720 isolated input-error cases passed, including 42 new ones.
The checked recursive_selection example selected columns 2 and 4, with
full-data R2=0.9999805392 and mean validation R2=0.9999624316 at two features.
Python syntax, Fortran line lengths, reference report, inventory and staged
diff checks passed. Original contributions remain MIT, with upstream BSD
notices retained. No dependency or extension was added.


## 2026-11-20: Lars, LassoLars and least-angle coefficient paths

Added dense native LAR and Lasso homotopy solvers, Lars/LassoLars estimators and
the uncentered lars_path function. Active-set Cholesky solves generate
equiangular directions; the Lasso path handles coefficient dropouts and re-entry,
positive constraints and alpha interpolation. Estimators support centering,
truncated paths, endpoint prediction/R2 and optional published path storage.
The direct result records alpha knots, coefficient matrices, 1-based final
active indices, iterations and degenerate candidate counts. See COVERAGE.md
for numerical thresholds, tie/degeneracy behavior and unsupported options.

The comparison suite passed 730 result groups against sklearn 1.6.1 across
five datasets and 20 configurations. It checks every coefficient/alpha knot,
active order, iteration count, final coefficients/intercepts, predictions,
scores, centered/uncentered fits, full/truncated/zero-iteration paths,
positive Lasso, zero/large/interpolated alpha, disabled path storage, wide
matrices and correlated matrices. The fifth dataset explicitly contains four
coefficient dropout events, asserted by the suite. Tolerances are atol=rtol=3e-8.
Native tests additionally compare LassoLars with coordinate-descent lasso and
duplicate-feature predictions with SVD least squares, plus zero-response paths.

Twenty-eight new isolated input-error cases bring the total to 748. They cover
counts, iteration limits, finite controls/data, method/constraint validation,
unfitted calls and query/target shapes. The least_angle_regression example
shows a sparse fitted path on synthetic regression data.
Inventory: 132 core types, 55 source modules, 57 native test programs,
37 reference suites and 11,817 reference result groups. Original contributions
remain MIT with sklearn BSD attribution retained. No dependency or extension
was added; the lars_path_result record is an API adapter, not a counted estimator.


Final validation: all 57 native test programs passed with runtime checks and
floating-point traps (-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow)
and in release mode. All 730 new least-angle reference groups passed against
sklearn 1.6.1, including four asserted coefficient dropout events. All 748
isolated input-error cases passed, including 28 new cases. The checked
least_angle_regression example produced two nonzero coefficients at alpha=.05
and R2=0.9984471569. Python syntax, Fortran line lengths, reference report,
dropout coverage, inventory and staged diff checks passed. Original
contributions remain MIT with upstream BSD notices retained. No dependency
or public extension was added.


## 2026-11-20: LarsCV, LassoLarsCV and LassoLarsIC

Added least-angle model selection by K-fold MSE, AIC and BIC. CV centers each
training fold independently, merges exact alpha knots, applies sklearn's stride,
interpolates held-out residuals before averaging squares, and refits at the
selected alpha. Stable sorting handles nonmonotonic LAR knots; boundary insertion
preserves sklearn 1.6.1's subsampled-grid semantics. IC uses path RSS and
coefficient-count degrees of freedom, with supplied variance or full-feature
SVD/active-set NNLS noise estimation. Selected fits expose prediction/R2,
alpha_, CV grids/error matrices or information-criterion/noise-variance arrays.

The new reference suite passed 1,000 groups across five datasets and 18 settings,
including full-rank/correlated/wide/dropout data, both CV methods, grid sampling,
positive constraints, no intercept, truncated/zero-iteration paths, AIC/BIC,
supplied/estimated variance and all model parameters/predictions/path diagnostics.
940 groups compare directly with sklearn classes. For max_n_alphas=1, sklearn
1.6.1 creates duplicate interpolation endpoints and can fail with an empty
finite-error grid; 60 groups instead compare sklearn fold paths, manually scored
endpoint errors and the final sklearn LassoLars refit. Complete rank-deficient
LAR comparison paths use eps=.1 on both sides; full-rank/truncated LAR and wide
Lasso CV retain default epsilon. All tolerances remain atol=rtol=3e-8.

Thirty-four isolated input cases bring the total to 782. Native tests cover
CV minima/shape/order, single-point grids, positive fits, both variance modes,
zero iterations and a separate default-epsilon rank-deficient LarsCV fit.
The lars_model_selection example compares CV and BIC fits on noisy regression.
See COVERAGE.md for the -1 automatic-variance sentinel, inherited fields,
interpolation behavior, singular paths, numerical controls and supported scope.
Inventory: 135 core types, 55 source modules, 58 native test programs,
38 reference suites and 12,817 reference result groups. Original contributions
remain MIT with sklearn BSD attribution retained. No dependency or public
extension was added.


Final validation: all 58 native test programs passed with runtime checks and
floating-point traps (-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow)
and in release mode, including default-epsilon rank-deficient LarsCV. All 1,000
new reference groups passed at atol=rtol=3e-8 under the documented comparison
settings: 940 sklearn-class groups and 60 independently scored single-alpha
CV groups. All 782 isolated input-error cases passed, including 34 new ones.
The checked lars_model_selection example returned CV alpha=0.0067046369,
BIC alpha=0.0098408485, estimated variance=0.0219782468, and R2 values
0.9971311517 (CV) and 0.9970881893 (BIC). Python syntax, Fortran line lengths,
reference reports, inventory and staged diff checks passed. Rank-deficient
LAR numerical differences and the single-alpha upstream issue are documented.
Original contributions remain MIT with retained BSD notices. No dependency
or public extension was added.


## 2026-11-20: SparsePCA, DictionaryLearning and SparseCoder

Added a shared dense dictionary-learning engine with SVD or paired-factor
initialization, Lasso-LARS/cyclic Gram coordinate descent code updates,
sequential atom updates, norm/positive constraints, unused-atom resampling and
objective stopping/history. SparsePCA centers inputs, normalizes/orients sparse
loadings, projects with ridge/pseudoinverse least squares and reconstructs with
the mean. DictionaryLearning retains learned training codes and exposes fresh
coding transforms. SparseCoder supports supplied dictionaries without fitting.
sparse_encode provides lasso_lars, lasso_cd, lars, omp and threshold, with
positive constraints where supported, CD initialization and split-sign wrappers.

A scalar-seeded MT19937 helper reproduces NumPy RandomState raw/uniform streams,
masked bounded indices and cached polar Gaussians for resampling. It uses int64
bit arithmetic to avoid signed 32-bit overflow and is internal to this feature,
not an additional counted estimator or public extension. NumPy 2.2.6 root,
MT/RandomKit, random-module and distributions notices are retained under LICENSES;
original contributions remain MIT and no NumPy/SciPy runtime is linked.

The main suite passed 744 sklearn 1.6.1 result groups over four seeded matrices
and 34 configurations: LARS/CD learning, automatic/overcomplete component counts,
wide data, supplied factor pairs, no iterations, high/zero penalties, loose/tight
stopping, positive dictionary/codes, all coding algorithms, warm-start codes,
split signs, ridge-zero transforms and inverse projections. Components, codes,
error histories, iteration/shape metadata, training/query transforms and inverses
match at atol=rtol=3e-8. Separate NumPy 2.2.6 compatibility tests passed 28 stream
groups over four seeds and multiple MT state cycles; raw/uniform/index streams
compare exactly, Gaussian streams at atol=rtol=2e-14. Those 28 groups are excluded
from sklearn coverage totals. Native tests also check known MT/Gaussian values,
atom constraints, coding identities and mean reconstruction from zero components.

Seventy-four new isolated diagnostics bring the total to 856. See COVERAGE.md
and API.md for initializer pairing, transform-vs-training codes, random seed
scope, degenerate SVD bases and supported options. The sparse_components example
shows sparse projection/reconstruction and dictionary training codes.
Inventory: 138 core types, 57 source modules, 60 native test programs,
39 sklearn reference suites and 13,561 sklearn result groups, plus 28 separately
tracked NumPy compatibility groups. No dependency or public extension was added.


Final validation: all 60 native test programs passed with runtime checks and
floating-point traps (-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow)
and in release mode. All 744 new sklearn result groups passed at atol=rtol=3e-8.
All 28 separate NumPy RandomState stream groups passed (exact raw/uniform/index
streams, Gaussian atol=rtol=2e-14). All 856 isolated diagnostics passed, including
74 new cases. The checked sparse_components example completed 40 iterations
for each learner, returning SparsePCA reconstruction RMSE=0.4294850842,
dictionary objective=17.6185554505 and 39 nonzero learned code entries.
Python syntax, Fortran line lengths, both reference reports, inventory and
staged diff checks passed. Original contributions retain MIT licensing;
complete sklearn and NumPy upstream notices are retained. No NumPy/SciPy
runtime dependency or public extension was added.


## AdaBoostRegressor / AdaBoost.R2 (2026-11-20)

Added native adaboost_regressor with depth-three squared-error CART defaults,
configurable tree controls, weighted bootstrap sampling, all three AdaBoost.R2
losses, learning-rate scaling, weighted-median prediction, staged predictions
and R2 scores, early stopping and weighted feature importances. The private
CART path follows sklearn feature rounding, feature RNG, constant caching,
introsort tie ordering, split evaluation and partitioning. Existing public
tree/forest implementations were not changed. No dependency or public
extension was added. Original contributions retain MIT licensing with the
existing sklearn BSD notice and NumPy RNG notices; provenance was expanded.

compare_adaboost_regression.py passed 1,649 sklearn 1.6.1 result groups at
atol=rtol=3e-8 across 36 configurations on three seeded datasets. Checks include
full requested weight/error arrays, fitted counts, train/query predictions,
every prediction/score stage, weighted importances and every retained tree's
seed, depth, leaf count, split features/thresholds, leaf/node values and query
predictions. Fixtures cover all losses, learning rates, depth/leaf/split/feature
controls, impurity thresholds, constant features/targets, rounded input ties,
initial zero/tiny weights and a zero-weight extreme target. First-tree
rejection is checked with the explicitly documented native zero-importance
convention in place of sklearn's NaNs; prediction behavior agrees.

Inventory: 139 core types, 58 source modules, 61 native test programs,
40 sklearn reference suites and 15,210 sklearn result groups, plus the existing
28 separately tracked NumPy compatibility groups. Eighteen new isolated
input-error fixtures bring the diagnostic inventory to 874. The
boosted_regression example demonstrates fitting and query predictions.

Final validation: all 61 native programs passed checked builds with
-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow and release builds.
All 1,649 new sklearn groups and all 874 isolated diagnostics passed. The
checked boosted_regression example retained eight trees and achieved training
R2=0.9471185508 on its illustrative fixture. Python syntax, Fortran formatting,
reference-report counts and staged diff checks passed. Older Python reference
suites were not rerun; existing tree/forest implementations were unchanged.


## QuantileRegressor / native quantile LP (2026-11-20)

Added quantile_regressor with the sklearn weighted pinball-loss plus L1
objective, optional unpenalized intercept, zero-weight exclusion, original-unit
coefficients, prediction and R2 score. An independent native dense primal
simplex uses a feasible residual basis, scaling and Bland pivot rules. Native
convergence, iteration, primal objective, dual coefficients, dual-gap and
reduced-cost diagnostics are exposed; arbitrary SciPy/HiGHS solver selection
is outside this subset. No dependency or public extension was added. MIT
project licensing and retained sklearn BSD terms are unchanged; provenance
now identifies the LP formulation and independent solver.

compare_quantile_regression.py adds 2,089 result groups across 120
configurations on five seeded datasets. Of these, 444 directly compare sklearn
HiGHS objectives and, on unique-solution fixtures, coefficients, intercepts and
train/query predictions; 1,645 check convergence, feature counts, independent
objectives, dual bounds, feature/intercept constraints, zero-weight dual entries,
primal-dual agreement, reduced costs, prediction identities and R2 identities.
Fixtures include quantiles .01 through .99, alpha zero through 1000, optional
intercepts, heterogeneous/zero/tiny/scaled weights, scaled features/targets,
duplicate/constant/zero columns, rounded data, constant and exact-linear
targets, zero-weight outliers and underdetermined systems. Nonunique optima
are compared by objective and certificates. The tiny-weight reference fixture
uses equivalent unit-sum weights for sklearn to avoid HiGHS absolute-cost
tolerances; the native call uses the original weights. General comparison
atol=rtol=2e-7; objective and certificate checks use the explicit target/feature
scales and tighter tolerances recorded in the script.

Inventory: 140 core types, 59 source modules, 62 native test programs,
41 reference suites and 17,299 result groups, plus 28 separately tracked NumPy
stream groups. Twenty-three new isolated diagnostics bring the diagnostic
inventory to 897. conditional_quantiles demonstrates three fitted quantiles.
The stale remaining-areas list was also corrected for already implemented
Gaussian-process classification and dictionary learning.

Final validation: all 62 native test programs passed with runtime checks and
floating-point traps (-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow)
and in release mode. All 2,089 quantile result groups and all 897 isolated
diagnostics passed. The checked conditional_quantiles example converged for
quantiles .1, .5 and .9, with slopes 1.7914809230, 1.9602359719 and 2.1831911834
and reported zero primal-dual gaps. These are illustrative fixture results.
The AdaBoostRegressor reference runner now filters actual executable files
from cold-build progress output; its 1,649 existing groups were rerun and passed.
Other older Python reference suites were not rerun. Python syntax, Fortran
formatting, inventory/report counts and staged diff checks passed.


## NuSVC binary and multiclass classification (2026-11-20)

Added nu_svc with weighted two-group nu dual optimization, feasibility checks,
LIBSVM margin scaling, binary and multiclass OVO fitting, linear/RBF/polynomial/
sigmoid/precomputed kernels, gamma rules, OVR decision transformation, optional
OVR tie breaking, class-grouped public support/dual layout, linear coefficients,
accuracy score and native convergence/dual diagnostics. Original-row support
indices and precomputed query columns are preserved with zero sample weights.
Existing SVC and other kernel SVM implementations were not changed. MIT project
licensing, retained sklearn/LIBSVM terms and no-runtime-dependency status remain
unchanged. Provenance and API/scope documentation were expanded.

compare_nusvc.py adds 1,396 result groups across 66 configurations on three
seeded datasets: 670 direct sklearn estimator-output comparisons and 726 dual,
normalization, layout and prediction/score checks. It covers binary, three- and
four-class labels, all five kernels, gamma scale/auto/numeric, polynomial degrees,
nonzero coef0, weighted/zero-weight fits, feasible class imbalance, nu .1 through
.85, both multiclass score formats, break_ties, constant features and exact
multiclass coefficient packing. General estimator atol=rtol=4e-5; normalized
duals and linear coefficients use 3e-4, discrete meaningful support metadata and
labels are exact, and native KKT violations must be <=1.001e-9.

Fully bound solutions are checked against source unscaled kernel projections
and both normalizations' KKT intervals instead of forcing an arbitrary rho/r
choice. Raw and expanded public dual-layout identities are checked separately;
only numerical support artifacts below 1e-10 are excluded from exact support
metadata comparisons. A verified extra LIBSVM support coefficient was
7.13e-16. The zero-weight precomputed reference explicitly reduces its positive
rows/columns to avoid sklearn 1.6.1's compacted-index prediction mismatch.
The native call retains its full original input/query layout. See COVERAGE.md
for solver precision, nonconvex-kernel stationarity and degenerate margin scope.

Inventory: 141 core types, 60 source modules, 63 native test programs,
42 reference suites and 18,695 result groups, plus 28 separate NumPy stream
groups. Thirty-three new isolated diagnostics bring the inventory to 930.
nu_classification demonstrates nu-constrained model fitting and query labels.

Final validation: all 63 native test programs passed checked builds with
-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow and release builds.
All 1,396 new result groups and all 930 isolated diagnostic cases passed.
The checked nu_classification example converged with ten support vectors per
class, training accuracy 1.0 and query labels -5/7 on its illustrative fixture.
Python syntax, Fortran formatting, inventory/report classifications and staged
diff checks passed. Older Python reference suites were not rerun; existing
SVC/kernel-SVM implementations were unchanged. The original MIT license and
complete upstream sklearn/LIBSVM notices remain in place.


## SGDOneClassSVM streaming outlier detection (2026-11-20)

Added sgd_one_class_svm with sklearn's one-class hinge/L2/offset updates,
default optimal schedule and shuffled training, all four public schedules,
unweighted-loss stopping and adaptive reductions, finite nonnegative sample
weights (including zero/all-zero), optional intercepts, explicit initializers,
warm starts, averaging and one-epoch partial_fit. Lazy scaling, sparse-trick
average accumulation and deterministic forward shuffle follow the pinned
Cython contracts; the arrays themselves are dense native Fortran. Native
state/stopping diagnostics are exposed. No dependency or public extension was
added. Original contributions retain MIT; sklearn and existing NumPy notices
remain retained, with expanded provenance.

compare_sgd_one_class.py adds 1,008 sklearn 1.6.1 result groups across 120
configurations on four seeded datasets, including seed zero and actual default
training settings. Every recorded fit, warm fit, three-batch stream and
fit-then-partial state compares coefficients, offset, decision and raw sample
scores, labels, n_features_in_, exact n_iter_ and exact cumulative t_. Parameter
checks use atol=rtol=3e-8; discrete labels and state counts match exactly.
Coverage includes delayed and immediate averaging, averaged warm starts,
constant/optimal/invscaling/adaptive schedules, negative powers, adaptive
stopping, all-zero/zero weights, constant/zero features, eta-induced scale
resets, optional intercepts and coefficient/offset initialization. Reference
state snapshots copy arrays to preserve earlier coefficients across sklearn's
in-place later updates. Native tests additionally verify two analytic hinge,
offset and streaming-average steps, repeatable shuffled refits, fit_predict
label mapping, score identities and clock resets/persistence.

Inventory: 142 core types, 61 source modules, 64 native test programs,
43 reference suites and 19,703 result groups, plus 28 separately tracked NumPy
stream groups. Twenty-nine new isolated diagnostics bring the inventory to
959. online_outliers demonstrates incremental updates and query labels.

Final validation: all 64 native test programs passed checked builds with
-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow and release builds.
All 1,008 new sklearn result groups and all 959 isolated diagnostic cases
passed. The checked online_outliers example processed 1,800 samples and
returned query labels +1,+1,-1, with decision scores approximately .08950,
.08993 and -.75404 on its illustrative fixture. Python syntax, Fortran
formatting, inventories, reports and staged diff checks passed. Older Python
reference suites were not rerun; existing SGD implementations and the legacy
RNG helper were unchanged. MIT project licensing and full upstream notices
remain in place.


## IsolationForest anomaly detection (2026-11-20)

Added isolation_forest with dense random isolation trees, exact seeded row/
feature sampling, bootstrap sample counts, weighted node statistics, automatic
and empirical contamination thresholds, path-length scores, integer labels,
fit_predict and warm-start tree growth. Native per-tree metadata stores the
actual draws for all warm generations; tree storage exposes native node values,
counts, splits and importances. Empty weighted roots use finite zero diagnostic
statistics while matching sklearn's scoring. Shared native tree and forest
builders were unchanged. MIT project licensing and existing sklearn/NumPy
notices remain, with expanded provenance. No dependency or public extension
was added.

compare_isolation_forest.py adds 9,060 result groups across 84 configurations
on four seeded datasets, including seed zero, more than 256 training rows and
full 100-tree default fits. It checks train/query path scores, decisions, exact
labels, offsets and fitted counts. Every retained tree compares selected
features/rows (including bootstrap duplicates), random state, depth, leaf count,
node thresholds/features/counts/values/impurities and normalized importances.
Fixtures cover automatic/count/fraction sampling, all automatic sampling
regimes, feature subsets, bootstrap, heterogeneous/zero/all-zero weights,
empty weighted roots, warm growth, constant features/data, oversized row counts,
singletons, empty bootstrap draws and empirical contamination. Nonempty-tree
statistics and scores use atol=rtol=3e-10; sample/feature/node indices, counts,
labels and per-tree seeds/depths/leaf counts match exactly. Empty-root NaN
statistics are compared with the documented finite native zero convention.

Native tests additionally check warm vs direct growth, empirical contamination,
distant-point scores, fit_predict, automatic metadata, empty weighted roots and
singleton/empty-bootstrap score normalization. Native accumulation preserves
sklearn's path-length operation order, including exact label checks at
constant-data decision boundaries.
A zero-row fraction without bootstrap is rejected explicitly, matching the
upstream failing path; the bootstrap variant remains supported.

Inventory: 143 core types, 62 source modules, 65 native test programs,
44 reference suites and 28,763 result groups, plus 28 separate NumPy stream
groups. Twenty-eight new isolated diagnostics bring the inventory to 987.
isolation_outliers demonstrates anomaly scores and query labels. The shared
input-error runner now checks for duplicate literal registrations before
building, protecting the case inventory from silent Python dictionary overrides.

Final validation: all 65 native test programs passed checked builds with
-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow and release builds.
All 9,060 new sklearn result groups and all 987 isolated diagnostic cases
passed. The checked isolation_outliers example uses a filled-grid fixture,
fits 100 trees and returns query labels +1,+1,-1, with scores approximately
-.48430, -.50903 and -.62710. These are illustrative fixture results.
Python syntax, Fortran formatting, reference reports, inventory counts and
staged diff checks passed. All registered diagnostic keys and native case
labels are unique. Older Python reference suites were not rerun; shared tree
builders and the legacy RNG helper were unchanged. MIT licensing and full
upstream sklearn/NumPy notices remain in place.


## FastMCD robust covariance and EllipticEnvelope (2026-11-20)

Added min_cov_det, elliptic_envelope and standalone fast_mcd with the full
pinned FastMCD search: one-dimensional shortest halves, small-data candidate
screening/refinement, and larger-data subset, merged-set and full-data stages.
Raw attributes, normal consistency correction, chi-square reweighting,
assumed-centering behavior, optional cached precision, singular pseudoinverse,
Mahalanobis distances, covariance errors and Gaussian likelihood are native.
EllipticEnvelope adds percentile offsets, negative-distance scores, decisions,
labels, fit_predict and optional weighted accuracy. Existing covariance fits
were unchanged. Original contributions remain MIT with expanded sklearn/NumPy
provenance; no dependency or public extension was added.

compare_robust_covariance.py adds 782 result groups over 34 seeded
configurations, spanning 1-4 features and 67-1601 rows. Of these, 770 compare
sklearn estimator/fast_mcd outputs directly; 12 additionally validate exact
univariate raw-boundary ties by nearest-set eligibility, selected covariance,
distances and consistency. Both affected fixtures still compare final
reweighted supports/covariances, distances and labels directly against sklearn.
Tolerance is atol=rtol=3e-7; support masks and labels match exactly except the
explicitly checked equivalent raw tied choices. Fixtures include automatic/
.7/.8/full support fractions, centering, precision caching on/off, contamination
.1/.25, translated/scaled data, substantial outliers, exact zero-column singular
covariance, subset/merge search at 601 rows and full refinement at 1601 rows.

Native tests also check an analytic univariate shortest-half/reweighted fit,
contamination resistance, support size, inverse consistency, Frobenius/spectral
errors, uncached precision, envelope fit_predict/accuracy/labels and assumed
centering. Native chi-square thresholds are checked against scipy.stats through
the reference diagnostics; SciPy is only used by the validation scripts.

Inventory: 145 core types, 63 source modules, 66 native test programs,
45 reference suites and 29,545 result groups, plus 28 separate NumPy stream
groups. Thirty unique new isolated cases bring the diagnostic inventory to
1,017. robust_ellipses demonstrates robust location/covariance and outlier labels.

Final validation: all 66 native test programs passed checked builds with
-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow and release builds.
All 782 reference/tie-validation groups and all 1,017 isolated diagnostic cases
passed. The checked robust_ellipses example retained 106 of 120 observations,
estimated location approximately (-.11677,.05262), and returned query distances
.02994/172.14508 with labels +1/-1 on its illustrative fixture. Python syntax,
Fortran formatting, inventories, reports and staged diff checks passed. All
registered keys/native case labels are unique. Older Python reference suites
were not rerun; existing covariance estimator bodies and the legacy RNG helper
were unchanged. MIT project licensing and full upstream notices remain in place.


## BisectingKMeans weighted hierarchical clustering (2026-11-20)

Added bisecting_kmeans with both split strategies, exact seeded weighted random
and greedy k-means++ initialization, repeated split restarts, native Lloyd and
Elkan bounds/pruning, empty-center handling, hierarchical prediction, final
center distance transforms and global nearest-center score. Fitted labels and
centers follow depth-first hierarchy order; native node and per-split diagnostics
are exposed. The source one-cluster unweighted-root convention and raw bisection
tolerance are preserved. Existing KMeans was unchanged. Original contributions
retain MIT licensing with expanded sklearn/NumPy provenance; no dependency or
public extension was added.

compare_bisecting_kmeans.py adds 1,152 direct sklearn 1.6.1 result groups across
88 configurations on four seeded datasets, including seed zero and more than
one source chunk. It compares final centers, exact labels, training inertia,
train/query hierarchical predictions, transforms, unweighted/weighted scores,
fit_predict/fit_transform outputs and every tree's centers, child links,
leaf labels, sample counts and selection scores. Nodes are reconstructed in
creation order from the source tree, preserving score-based split choices.
The source OpenMP thread count is fixed to one for reproducible reductions.
Numerical comparisons use atol=rtol=3e-8; labels, children, counts and metadata
match exactly.

Fixtures cover both strategies/solvers/initializers, restarts, heterogeneous/
zero/all-zero weights, weighted k-means++, one cluster, one cluster per sample,
max_iter=1, large/zero tolerances, constant features, duplicate data with empty
leaves, translated data, and a single positive-weight k-means++ initialization.
Native tests additionally verify analytic weighted centers/inertia, repeatable
fits, the one-cluster root, empty leaves, topology, and a concrete difference
between hierarchical routing and global nearest-center assignment.

Inventory: 146 core types, 64 source modules, 67 native test programs,
46 reference suites and 30,697 result groups, plus 28 separate NumPy stream
groups. Twenty-six unique new isolated cases bring the inventory to 1,043.
hierarchical_centroids demonstrates fitting and query cluster labels.

Validation completed: all 1,152 new sklearn reference result groups passed
using the package-built driver; all 67 native test programs passed with
-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow and in release mode.
All 1,043 isolated invalid-input cases passed, including 26 new BisectingKMeans
cases. The hierarchical_centroids example passed, producing query labels
0/2/1 and training inertia 180.29171226379120. Python syntax, Fortran formatting,
inventory/report checks and git diff checks passed. Older Python reference
suites were not rerun; existing KMeans bodies and the legacy RNG helper were
unchanged. No runtime dependency or public extension was added.

## MiniBatchKMeans batch and streaming clustering (2026-11-20)

Added mini_batch_kmeans with full-data sampled-batch fitting and persistent
partial_fit. Weighted running centers/counts, NumPy-compatible seeded random
and greedy k-means++ initialization, explicit initial centers, validation-set
restart selection, automatic initialization sizing, low-count reassignment and
batch/epoch counters are implemented. fit monitors source variance-scaled center
motion and exponentially smoothed batch inertia. As in the source, tol=0 skips
the full-data variance pass. partial_fit applies one update
to every row supplied. predict/transform/score and fit_predict/fit_transform are
available. Existing KMeans, BisectingKMeans and the RNG helper bodies are unchanged.

compare_minibatch_kmeans.py checks 2,440 direct sklearn 1.6.1 result groups in
136 configurations across four seeded datasets, including seed zero and more
than one source chunk. It compares centers, accumulated cluster weights,
labels/inertia, batch/epoch counters, predictions, distance transforms, weighted
scores and fit method outputs. Streaming fixtures compare every batch's state,
including fit followed by partial_fit and partial_fit followed by fit. The
source OpenMP thread count is fixed to one for reproducible reductions.
Numerical tolerances are atol=rtol=3e-8; labels and metadata match exactly.

Fixtures cover random/greedy/explicit initialization, automatic/explicit restart
counts, initialization subsampling/fallback, both stopping rules and their
disabled forms, one epoch/one cluster, heterogeneous/zero/all-zero weights,
compute_labels=false, a short final streaming batch, translated data, constant
features and duplicate zero centers. A distinct-count fixture checks the
reassignment cap and reset counts. A billion-iteration budget exits after two
steps and verifies that sample/iteration products use wide integer arithmetic.
Equal-count ties at the cap use native first-
index order; NumPy's unstable argsort may choose different tied centers. Near-
ties between duplicate floating-point centers may also differ. These limits
and native sentinel values are documented in COVERAGE.md and API.md.

Native tests independently verify analytic weighted running means/counts and
scores, zero-weight state preservation, wide iteration budgets with early
stopping, small subsequent batches, deterministic
fit resets, copied streaming state, counters and unavailable one-step smoothed
inertia. The native has_inertia_ flag handles the source corner where
compute_labels=false with only one update leaves no smoothed estimate.

Inventory: 147 core types, 65 source modules, 68 native test programs,
47 reference suites and 33,137 result groups, plus 28 separate NumPy stream
groups. Thirty-seven new isolated cases bring the inventory to 1,080.
streaming_centroids demonstrates two successive batch updates and prediction.
MIT project licensing and complete upstream notices remain in place; no
runtime dependency or public extension was added.

Validation completed on the final implementation: all 2,440 sklearn result
groups passed through the package-built reference driver. All 68 native test
programs passed with -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow
and in release mode, including the wide-budget early-stopping regression.
All 1,080 isolated invalid-input cases passed. The runtime-checked
streaming_centroids example passed with centers -2/5, accumulated weights 3/4
and query labels 0/1. Python syntax, Fortran formatting, inventories, report
contents, unique diagnostic registrations and git diff checks passed. Earlier
Python reference suites were not rerun; existing estimator and RNG helper
bodies were unchanged. MIT licensing and full upstream notices are retained.

## Spectral co-/biclustering (2026-11-20)

Added spectral_coclustering and spectral_biclustering. Paired Dhillon embeddings
and Kluger checkerboard partitions use native dense thin SVD, transpose factoring
for wide matrices, orthonormal null-space completion, all three source dense
normalizations, piecewise vector ranking, original-data projections and joint
or independent centroid fits. A private centroid solver follows source seeded
greedy/random initialization, Lloyd updates, stopping and same-partition restart
selection. mini_batch=true reuses native MiniBatchKMeans. Existing estimator and
RNG helper bodies were unchanged. Source counterpart membership arrays and all
bicluster getters are available, with one-based returned Fortran indices and
zero-based block IDs. Empty blocks and repeated fits are supported.

compare_bicluster.py checks 2,992 result groups across 80 configurations on four
seeded positive/shifted matrices, including wide data and seed zero. Direct
installed sklearn 1.6.1 fits use randomized SVD oversampling equal to the larger
matrix dimension to span the complete smaller dimension. Trace subclasses
observe source intermediate arrays without replacing source computation. The
comparison covers normalized matrices, sign-aligned singular vectors, singular
values against NumPy SVD, row/column scalings, paired embeddings, centroid SSE,
piecewise approximation errors and selected vectors, original-data projections,
row/column partitions, masks, copied biclusters and every block's indices, shape
and extracted data. Normalization iteration/convergence diagnostics are checked
against the source dense scale loop. Numerical tolerance is atol=rtol=3e-8.
Discrete membership, selection, shape and metadata comparisons are exact after
consistent cluster relabeling (joint for paired clusters; independent row/column
relabeling and Cartesian block permutation for checkerboards).

Fixtures exercise both centroid solvers, both initializers, several cluster
counts, rectangular grids, all normalization methods, negative inputs,
component selection and matrix orientations. Native analytic tests additionally
verify checkerboard recovery, Cartesian coverage, original-value extraction,
seeded repeated fits, constructors, log centering identities, orthogonal
null-space completion (including 10,000-row constant data), duplicate projections,
empty blocks, component truncation
and mini-batch shifted input. Repeated singular values/null spaces and exact
centroid or ranking ties can differ from source choices, as documented. The
native backend does not expose approximate randomized/ARPACK solver controls.

Inventory: 149 core types, 66 source modules, 69 native test programs,
48 reference suites and 36,129 result groups, plus 28 separate NumPy stream
groups. Forty-seven new isolated input cases bring the inventory to 1,127.
checkerboard demonstrates row/column groups and block extraction. MIT licensing
and upstream attribution are retained; no runtime dependency or public extension
was added.

Validation completed on the final implementation: all 2,992 reference result
groups passed through the package-built driver. All 69 native test programs
passed with -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow and in
release mode, including the optimized 10,000-row null-space regression. All
1,127 isolated invalid-input cases passed, including 47 new spectral bicluster
cases. The checked checkerboard example passed with row labels 1/1/1/0/0/0,
column labels 1/1/0/0, four grid blocks and a first-block shape of 3-by-2.
Python syntax, Fortran formatting, inventories, report contents, unique
registrations and git diff checks passed. Earlier Python reference suites
were not rerun; existing estimator bodies and the RNG helper were unchanged.
MIT project licensing and full upstream notices are retained.

## Variational Bayesian Gaussian mixtures (2026-11-20)

Added bayesian_gaussian_mixture with both source weight prior families, four
covariance forms, four seeded initializers, prior validation/defaults, variational
E/M updates, expected-log scoring, source lower-bound stopping and best-restart
selection. Warm starts, zero-iteration initialization, fresh dimension/covariance
refits, fit_predict and native Gaussian-mixture sampling are supported. All
covariance/precision outputs use the native expanded (p,p,K) layout; tied degrees
are repeated, and Dirichlet/Beta concentration arrays have one/two columns.
Diagonal/spherical factorization avoids cubic work; tied factors are reused.
Bound history grows with actual iterations, and int64 iteration counters allow
large budgets without allocating their full histories.

The tested private biclustering centroid helper was extracted into
sklearn_centroid_core and is reused without changing its update math. It now
accepts an optional continued legacy RNG state for mixture restarts. Existing
public estimator and RNG helper bodies were unchanged apart from biclustering's
delegation to this core and the new mixture re-export. Its 2,992 reference groups
were rerun to verify this refactor.

compare_bayesian_mixture.py checks 5,524 result groups across 280 configurations
on four seeded datasets. Direct installed sklearn 1.6.1 comparisons cover all
posterior parameters, resolved priors, precision factors, best-run bound history,
convergence/iteration state, responsibilities, predictions and variational scores.
Fixtures include both weight families, four covariance forms and initializers,
restarts, custom priors, one-component models, constant/univariate/translated
inputs, max_iter=0/1, tol=0, a billion-iteration budget with early stopping,
same/shifted-data warm starts and fresh fits with changed dimension/covariance.
Numerical tolerance is atol=rtol=3e-8; labels and metadata are exact. Source bound
histories are observed without replacing source computation.

Of these groups, 192 check native sampling frequency/moment certificates,
grouped labels and repeated-call determinism on 20,000 draws per sampled model.
Analytic mixture first/fourth moments supply seven-standard-error frequency,
mean and covariance bounds. Sampling follows the source distribution and grouped
labels, but categorical counts/Cholesky roots intentionally use a different
seeded stream from NumPy multinomial/SVD sampling; posterior comparisons remain
direct. Native tests independently verify analytic Normal-Wishart posterior
means/covariances, mean precision and degrees of freedom, expected-log scores
under both priors, covariance inverse identities, nondecreasing bounds,
initialization-only state, warm continuation, huge-budget history allocation,
refit shapes and Gaussian sampling moments.

Inventory: 150 core types, 68 source modules, 70 native test programs,
49 reference suites and 41,653 result groups, plus 28 separate NumPy stream
groups. Fifty-four new isolated cases bring the inventory to 1,181.
variational_mixture demonstrates posterior weights, effective components and
query probabilities. MIT licensing and upstream notices remain in place;
no runtime dependency or public extension was added.

Validation completed on the final implementation: all 5,524 Bayesian mixture
result groups and all 2,992 biclustering regression groups passed through their
package-built drivers. All 70 native test programs passed with
-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow and in release mode.
All 1,181 isolated invalid-input cases passed, including 54 new Bayesian mixture
cases. The checked variational_mixture example converged in 15 iterations,
found two components above 0.05 weight, returned query labels 5/1/1 and unit
probability row sums. Python syntax, Fortran formatting, inventories, report
contents, unique registrations and git diff checks passed. Other Python
reference suites were not rerun. MIT project licensing and complete upstream
notices are retained, with no runtime dependency added.


## HDBSCAN hierarchical density clustering (2026-11-20)

Added dense hdbscan with sklearn 1.6.1 brute mutual reachability, self-included
core distances, Prim linkage, condensed tree, raw stability, EOM/leaf selection,
epsilon merging, size caps, optional root selection, membership strengths,
weighted centroids/medoids and strict-height DBSCAN cuts. Supports six feature
metric names plus the cityblock alias, and precomputed distances including
positive infinity. Nonfinite feature rows retain source -2/-3 labels. Native
centers and raw-tree remapping correct source edge cases with excluded rows.
The dense workspace is updated in place and freed before center computation.
There are no sparse/tree/parallel backends or prediction method; alpha follows
source brute scaling, which differs from auto/tree semantics for alpha!=1.

compare_hdbscan.py checks 2,604 groups across 120 configurations on four seeded
datasets. All 120 compare source linkage ultrametrics and sorted heights, source
tree selection on the native tied hierarchy, condensed arrays, stability,
strengths, core distances, Prim frontier keys, raw tree sizes, weighted centers
and DBSCAN cuts. Of these, 114 complete direct fits match source labels and
strengths; six single-cloud/cosine plateau or roundoff tie cases differ in
selection under equal-height ordering. Those differences are recorded, not
counted as direct-fit identity. Discrete comparisons are exact after label
renumbering and numeric comparisons use atol=rtol=3e-8 with explicit NaN handling.
Native analytic tests additionally cover self-included neighbor ranks, known
strengths/centers, strict cuts, alpha scaling, disconnected infinite distances,
subnormal/infinite lambdas, NaN stability, identical rows and excluded-row root
remapping. The hierarchical_density example demonstrates fitting and cuts.

Inventory: 151 core types, 69 source modules, 71 native test programs,
50 reference suites and 44,257 result groups, plus 28 separate NumPy stream
groups. Thirty-five new diagnostics bring the isolated invalid-input inventory
to 1,216. MIT licensing is retained alongside the full HDBSCAN component BSD
notice in LICENSES/hdbscan-BSD.md and existing sklearn notices. No runtime
dependency or public extension was added. Existing estimator bodies are
unchanged apart from the clustering re-export.

Validation completed on the final implementation: all 71 native test programs
passed with -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow and in release
mode. The package-built HDBSCAN driver passed all 2,604 result groups (114/120
direct fits; all 120 hierarchy-conditioned comparisons passed). All 1,216
isolated invalid-input cases passed, including 35 new HDBSCAN cases. The checked
hierarchical_density example found three clusters and no noise, with membership
strengths from 0.3769913 to 1 and no noise at cut 0.7/minimum size five.
Python syntax, Fortran formatting, report contents, inventory, unique diagnostic
registrations and git diff checks passed. Earlier Python reference suites were
not rerun; existing estimator bodies were unchanged apart from the new export.
MIT project licensing and complete upstream notices are retained.


## Bernoulli restricted Boltzmann machine (2026-11-20)

Added bernoulli_rbm with stochastic maximum likelihood/persistent contrastive
divergence, conditional hidden transforms, fit_transform, incremental training,
Gibbs transitions and pseudo-likelihood scoring. Native components use source
(n_components,n_features) shape. The existing legacy RNG reproduces NumPy's
logical row-major draws; no RNG helper or existing MLP computation was changed.
Training follows source unshuffled batch scheduling and actual-positive-batch
normalization, retaining the full negative chain for short/oversized batches.
Integer-seeded fit has a local RNG, partial_fit/Gibbs share a persistent stream,
and scoring recreates its own RNG per call. Fresh fit resets model arrays while
preserving any existing sampling stream. Stable sigmoid/compensated softplus
handle extreme logits. Dense finite inputs are accepted with source range-check
semantics; sparse, float32, object RNGs and verbose logs are outside this scope.

compare_rbm.py directly checks 4,552 result groups across 80 configurations and
four seeded datasets against installed sklearn 1.6.1. Every operation checks
components, both biases, persistent hidden chains and feature counts; transforms,
Gibbs outputs and scores are checked when called. Fixtures cover zero iterations,
unit/full/short/oversized batches, multiple hidden widths, fractional/signed data,
constant zeros/ones, small/large learning rates, mixed fit/partial/Gibbs sequences,
changed seed/rate controls, changed-dimensional fresh refits, preserved chain size
when batch_size changes, repeated scoring and extreme query logits. Numeric
comparisons use atol=rtol=2e-10; binary chains, Gibbs samples and feature counts
are exact. Analytic native tests cover conditional means, uniform visible scores,
a one-step PCD gradient, saturated logits, copying streams, refit shapes and
input preservation. binary_features demonstrates training, transformation,
pseudo-likelihood, Gibbs sampling and an incremental update.

Inventory: 152 core types, 70 source modules, 72 native test programs,
51 reference suites and 48,809 result groups, plus 28 separate NumPy stream
groups. Thirty new diagnostics bring the isolated invalid-input inventory to
1,246. MIT project licensing and full upstream notices are retained. No runtime
dependency or public extension was added. Existing estimator bodies are unchanged
apart from the neural-network re-export.

Validation completed on the final implementation: all 72 native test programs
passed with -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow and in release
mode. The package-built RBM driver passed all 4,552 direct sklearn result groups
across 80 configurations, including exact binary chain and Gibbs outputs.
All 1,246 isolated invalid-input cases passed, including 30 new RBM cases.
The checked binary_features example returned finite hidden probabilities,
mean pseudo-likelihood -2.7178464, 16 active visible units in its Gibbs draw and
an updated weight norm of 0.1287291 after partial_fit. Python syntax, Fortran
formatting, report contents, inventories, unique diagnostic registrations and
git diff checks passed. Earlier Python reference suites were not rerun;
existing estimator bodies were unchanged apart from the new export. MIT
licensing and full upstream notices are retained, with no runtime dependency.


## Variational Latent Dirichlet Allocation (2026-11-20)

Added latent_dirichlet_allocation with batch/online variational EM, streaming
updates, normalized/unnormalized topic transforms, evidence-bound score and
perplexity with optional population compensation. Source single-worker Gamma
initialization/draw order, AS 103 digamma approximation, E-step stopping,
minibatch scaling, evaluation frequency and early-stop counters are retained.
Native controls use -1 for automatic priors; explicit priors must be in (0,1],
excluding source's improper zero-prior boundary. Dense fractional counts, zero
rows/columns, zero iteration budgets and fresh dimension/component refits are
supported. partial_fit always performs online chunks, preserving resolved
priors/RNG, while fit resets both. Inference does not consume training draws.
Bound metadata follows source lifetimes; partial-only bound_ is native NaN.

Native Dirichlet scoring removes cancellation through stable log-Gamma/digamma
remainders, exact zero-KL identities and a Stirling expansion. Exponential
overflow returns infinity explicitly; zero-word/zero-evidence perplexity returns
NaN. Fitted feature counts are validated for perplexity, correcting the source
metadata-reset/truncated-input behavior. Existing estimator bodies and the
legacy RNG helper were unchanged apart from the decomposition re-export; the
shape-100 Gamma sampler is private to LDA. Native analytic tests verify known
NumPy initialization, one-topic posterior counts/conjugate evidence, score/
perplexity identity, iteration counters, copied streams, empty documents,
zero-word definitions, overflow handling and input preservation.

compare_lda.py checks 6,734 groups across 120 configurations and four seeded
datasets: 6,627 strict source comparisons, 26 source scoring comparisons with a
roundoff budget, 26 additional independent 60-digit mpmath certificates and 55
explicit zero-word definition checks. Ordinary tolerance is atol=rtol=3e-8;
metadata is exact. The roundoff budget is 32*eps times summed absolute log-Gamma
component and row-total terms, propagated through log perplexity; observed
bound-scale discrepancies reach 4.19e-5 for large populations. The high-precision
certificates validate native scoring at the ordinary tolerance. Undefined source
perplexity is recorded separately, without claiming direct identity. The suite
covers both learning modes, streaming, altered controls, single words/topics,
topics exceeding documents, tiny/default/unit priors, zero/fractional counts,
zero inner/outer budgets, early stopping, large populations and mixed calls.
mpmath is required only by this validator. document_topics demonstrates fitting,
per-document proportions, scoring, perplexity and streaming updates.

Inventory: 153 core types, 71 source modules, 73 native test programs,
52 reference suites and 55,543 result groups, plus 28 separate NumPy stream
groups. Forty-nine new diagnostics bring the isolated invalid-input inventory
to 1,295. MIT project licensing, complete sklearn and NumPy BSD notices and
research attribution are retained. No runtime dependency or public extension
was added.

Validation completed on the final implementation: all 73 native test programs
passed with -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow and in release
mode. The package-built LDA driver passed all 6,734 checks with the documented
6,627/26/26/55 oracle breakdown. All 1,295 isolated invalid-input cases passed,
including 49 new LDA cases. The checked document_topics example separated its
two document groups, returned evidence bound -274.0400696 and perplexity
5.0128289, and counted 31 EM updates after an incremental update. Python syntax,
Fortran formatting, strict JSON/report contents, inventories, unique diagnostic
registrations and git diff checks passed. Earlier Python reference suites were
not rerun; existing estimator bodies and the RNG helper were unchanged apart
from the new decomposition export. MIT licensing and complete upstream notices
are retained, with no runtime dependency added.


## Mini-batch dictionary learning and SparsePCA (2026-11-20)

Added mini_batch_dictionary_learning and mini_batch_sparse_pca. The dictionary
estimator implements source default randomized-SVD initialization, supplied
initializer padding/truncation, one-time row shuffle, fixed batch cycling,
actual-batch forgetting/normalization, constrained atom updates, seeded unused
atom resampling, EWA/motion/patience stopping, streaming statistics and all five
existing sparse transform methods with optional sign splitting. Partial calls
use the full supplied matrix once and retain the initialized solver/RNG; fresh
fit resets state. MiniBatchSparsePCA trains on centered X.T, recodes it into
normalized sparse components, and provides ridge projection and mean-restoring
inversion. Training controls retain source defaults; native scalar seeds default to zero.
Native int64 step counters
and loop budgets permit huge iteration limits with early stopping and no
history-sized allocation. Source None sentinels map to native zero/-1 controls
and NaN convergence-state diagnostics as documented in COVERAGE.md.

A private randomized SVD core implements default oversampling, four/seven LU
power iterations, wide-matrix transposition, Householder QR, projected Jacobi
SVD and original-left-vector sign selection. It preserves the existing legacy
RNG's logical draw order/state. Existing batch dictionary atom updates and
SparsePCA ridge projection were extracted into shared private helpers without
changing arithmetic, requiring regression checks for that existing functionality.

compare_minibatch_dictionary.py checks 3,468 groups over 200 configurations:
100 dictionary fits/streams, 68 SparsePCA fits and 32 randomized-SVD backends
across four seeds. It compares parameters, accumulated statistics, convergence
state, counters, transforms and reconstructions directly with sklearn 1.6.1,
including default training/solver controls with fixed seeds. Backend checks compare singular values/nonzero
directions/reconstruction and certify orthogonality, with exact subsequent RNG
values. Tolerance is atol=rtol=3e-8 except exact metadata/RNG. Null bases are not
claimed identical. Fixtures include all sparse coding methods, both fitting
solvers, positivity, supplied/default initialization, dead atoms, short/unit/
oversized batches, initialization-only and huge-budget early stopping, cached
partial-fit solver changes, dimension refits, constants, wide/tall matrices,
zero ridge and overcomplete models. Native analytic tests independently verify
statistic recurrences, soft thresholding, atom projection, copied streams,
patience/warmup counters, sampled-range SVD identities and constant-data SparsePCA.

Inventory: 155 core types, 72 source modules, 74 native test programs,
53 reference suites and 59,011 result groups, plus 28 separate NumPy stream
groups. Fifty-one new diagnostics bring the isolated invalid-input inventory
to 1,346. streaming_sparse_features demonstrates incremental dictionary updates
and mini-batch sparse projections. MIT licensing and full sklearn/NumPy notices
are retained. No runtime dependency or public algorithm extension was added.

Validation completed on the final implementation: all 74 native test programs
passed with -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow and in release
mode. The package-built new driver passed all 3,468 groups (2,444 dictionary,
816 SparsePCA and 208 randomized-SVD checks) across 200 configurations. The
existing 744 sparse-dictionary groups and 28 separate NumPy RNG groups were
rerun and passed, verifying both shared-helper extractions. All 1,346 isolated
invalid-input cases passed, including 51 new cases. The checked
streaming_sparse_features example performed two incremental dictionary updates,
returned 31 nonzero code entries, and completed 20 SparsePCA epochs/40 updates
with reconstruction RMSE 0.3132014. Python syntax, Fortran formatting, report
contents, inventories, unique registrations and git diff checks passed. Other
Python reference suites were not rerun. MIT licensing and complete upstream
notices are retained, with no runtime dependency or public extension added.
