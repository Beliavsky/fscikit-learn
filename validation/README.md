# Validation

Run from the package root:

```
fpm test --flag "-fcheck=all -fbacktrace"
fpm test --profile release
python validation/compare_cluster_information.py
python validation/compare_resampling.py
python validation/compare_tsne.py
python validation/compare_lle_variants.py
python validation/compare_classifier_chains.py
python validation/compare_chains.py
python validation/compare_multioutput.py
python validation/compare_weighted_classifiers.py
python validation/compare_calibrated.py
python validation/compare_self_training.py
python validation/compare_ranking.py
python validation/compare_mutual_info.py
python validation/compare_label_binarization.py
python validation/compare_polynomial_count_sketch.py
python validation/compare_sklearn.py
python validation/compare_graph_algorithms.py
python validation/compare_regression_projection.py
python validation/compare_robust_glm.py
python validation/compare_modes_density.py
python validation/check_invalid_inputs.py
```

The reference validator requires NumPy, SciPy and scikit-learn in Python.
compare_lda.py and compare_discriminant_auto.py additionally require mpmath for high-precision certificates
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


## Mini-batch nonnegative matrix factorization (2026-11-20)

Added mini_batch_nmf with source beta-divergence/MM updates, L1/L2 scaling,
random/NNDSVD/custom initialization, online accumulators, fixed batching, fresh
restarts, fitted transforms/inversion and EWA/motion/patience stopping. Native
real beta_loss maps source named losses to 2/1/0 and supports other finite real
values. Integer-seeded draw order and NNDSVDAR's scalar-seed restart match source.
Native init='auto'/n_components=0 and alpha_h=-1/transform_max_iter=0/patience=-1
sentinels map source None/auto/same controls. Partial calls always solve fresh
coefficients, reuse cached beta/rank/rho/transform limit and read current scaled
penalties. Fresh fit resets caches and supports dimension/rank changes.

Explicit null singular directions and zero-norm convergence are handled without
invalid arithmetic. Nonpositive beta's strictly-positive-input requirement is
checked consistently for fit, streaming and transformation. Custom inputs are
copied, and partial_fit discards unused initial W storage before solving its
coefficient batch. Native int64 counters allow huge budgets with early stopping
and no history allocation. Existing NMF, randomized-SVD and RNG helper bodies
are unchanged; only a decomposition re-export integrates the new type.

compare_minibatch_nmf.py checks 6,500 direct sklearn 1.6.1 groups across 164
configurations and four seeds at atol=rtol=3e-8, with exact metadata. Comparisons
cover training factors/components, both accumulators, resolved caches, counters,
EWA/error state, transforms and inverses. Cases cover beta=-0.5/0/0.5/1/1.5/2/
2.5/3 and named source aliases; all initializers and auto/custom ranks; L1/L2/mixed
penalties; forget endpoints; fresh restarts; unit/short/oversized batches;
huge-budget early stopping; cached-control changes; partial-only/mixed fits;
new dimensions; tiny/sparse counts; all-zero random/automatic initialization;
and default training controls. Native analytic tests verify exact rank-one
Frobenius/KL/IS fixed points, accumulated-statistic recurrences, copied objects,
input preservation, zero-norm patience and null NNDSVD handling.
nonnegative_stream demonstrates fitting, KL reconstruction, transformation,
linear inversion and incremental updates.

Inventory: 156 core types, 73 source modules, 75 native test programs,
54 reference suites and 65,511 result groups, plus 28 separate NumPy stream
groups. Fifty-six new diagnostics bring the isolated invalid-input inventory
to 1,402. MIT licensing and complete sklearn/NumPy notices and research
attribution are retained. No runtime dependency or public extension was added.

Validation completed on the final implementation: all 75 native test programs
passed with -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow and in release
mode. The package-built driver passed all 6,500 direct sklearn groups across
164 configurations. All 1,402 isolated invalid-input cases passed, including
56 new MiniBatchNMF cases. The checked nonnegative_stream example stopped at
four epochs/12 updates, reported KL reconstruction error 1.054803 and Frobenius
RMSE 0.1220615, then reached 13 updates after streaming. Minimum component and
factor values remained positive (0.002413391 and 4.86832e-7). Python syntax,
Fortran formatting, strict JSON/report contents, inventories, unique diagnostic
registrations and git diff checks passed. Earlier Python reference suites were
not rerun; existing estimator and shared-backend bodies were unchanged apart
from the new export. MIT licensing and complete upstream notices are retained,
with no runtime dependency or public algorithm extension added.


## Multi-task regularization-path cross-validation (2026-11-20)

Added multi_task_elastic_net_cv and multi_task_lasso_cv with automatic/explicit
alpha grids, scalar/vector mixing ratios, warm coefficient paths per fold,
equal-fold mean multi-target MSE selection and cold full-data refits. Dense
rank-two targets include a single column. Native CV results retain fixed
(alpha,fold,ratio) axes and expose path iterations/gaps and partition metadata.
Integer KFold and allocated test_fold PredefinedSplit maps support ordinary,
unbalanced, train-only and single-holdout partitions. LassoCV fixes mixing to
one. General splitter objects/pair iterables, weighting, parallelism and
metadata routing remain outside the current wrapper scope.

The shared base solver now supports selection=random with sklearn uint32
XorShift/feature modulo seeded by the existing NumPy-compatible bounded draw.
Path RNGs reset per fold/ratio and advance per alpha, and the refit starts from
a fresh scalar seed. Cyclic arithmetic is unchanged. Base alpha=0 and mixing
ratio=0 are now supported, matching source boundaries, with analytic least-
squares/Ridge checks. The obsolete zero-alpha rejection was retired, and the
old zero-ratio rejection now tests a negative ratio. With zero L1 strength,
source dual-gap stopping can remain unsatisfied despite accurate coefficients;
source iterations/diagnostics are retained. Inputs are preserved.

compare_multitask_cv.py checks 7,960 groups across 272 cases and four seeds:
192 CV scenarios and 80 base-model scenarios. All path computations remain the
actual installed sklearn 1.6.1 functions, observed for iterations/gaps. There
are 7,957 direct comparisons and three records for one zero-alpha selection
with an equivalent inactive ratio (.5 native, 0 source). Both effective
penalties are exactly zero and the native ratio is a supplied candidate;
coefficients, predictions and all path diagnostics still compare directly.
No active-penalty selection difference is permitted. Numeric tolerance is
atol=rtol=3e-8; counters/partition metadata are exact. Fixtures include default
controls, random warm/single-sweep updates, duplicate/zero/custom penalties,
pure L2, predefined/leave-one-out/unequal folds, constant/tiny/zero targets,
constant/duplicate features, single targets, wide matrices, epsilon ordering
and repeated dimension changes. Native analytic tests verify known fold MSEs,
selected coefficients, exact ties, predefined holdouts, input preservation and
huge-budget zero-target stopping. multitask_penalty_selection demonstrates
random-coordinate multi-ratio selection and joint prediction.

Inventory: 158 core types, 73 source modules, 76 native test programs,
55 reference suites and 73,471 result groups, plus 28 separate NumPy stream
groups. Forty-six new diagnostic cases and one retired obsolete case bring the
isolated invalid-input inventory to 1,447. MIT licensing and full sklearn/NumPy
attribution remain in place. No runtime dependency or public algorithm
extension was added.

Validation completed: all 76 native test programs passed with bounds checks,
backtraces and invalid/zero/overflow traps, and again with the release profile.
The new reference suite passed 7,960 groups (7,957 direct source comparisons
and three documented inactive-ratio equivalences); all 192 existing multi-task
reference groups passed. All 1,447 isolated invalid-input checks passed.
The checked example selected alpha=0.0017144762758537057 and l1_ratio=1,
with validation MSE=0.0026628639982208802 and training mean R2=0.999729179755852.
Python syntax, Fortran line limits, diagnostic case-set equality, inventory,
license notices and git whitespace checks passed. Other reference suites were
not rerun for this batch. Removed a stale numeric count from EXTENSIONS.md;
its inventory still contains no implemented public extensions.


## OrthogonalMatchingPursuitCV (2026-11-20)

Added orthogonal_matching_pursuit_cv in a new sklearn_omp_cv module, re-exported
through sklearn_linear_model and the sklearn facade. Native incremental
Cholesky paths reproduce sklearn 1.6.1 linear_model/_omp.py, including active
column permutations, original-correlation stopping, absolute machine-epsilon
dependence tests, fold-local centering, shortest-path truncation, equal-fold
MSE selection and cold refitting. Direct X is used in CV and source-style
auto-precomputed Gram matrices for the N>P refit. Existing base OMP code is
unchanged. The default max_iter=0 maps to min(max(P/10,5),P), including the
source explicit-zero convention. Targets are rank-one and X requires P>=2.

Partitions support contiguous KFold, PredefinedSplit maps with -1 training-only
rows, and arbitrary paired Boolean train/test masks. Masks cover grouped,
shuffled, gap and rolling-window splits without a new splitter abstraction.
Python splitter objects, repeated row copies in a fold, weights, multi-output,
parallel execution and metadata routing remain outside this wrapper. Inputs
are preserved. Native diagnostic arrays expose retained fold MSEs, per-fold
path lengths and training/test sizes. Empty fold paths are diagnosed explicitly
rather than passing an empty array into minimum selection.

The new reference suite observes actual source fold paths without replacing
source computation. Its 88 cases across four seeds each perform two fits on
the same native instance, checking predictions, coefficients, intercepts,
selected sparsity/refit iterations, source fold errors, best MSE, path lengths,
shapes and split sizes. Default and bounded paths, P>50 default scaling, two
features, wide/rank-deficient/near-dependent designs, duplicate/constant columns,
noiseless targets, scaled inputs, leave-one-out, predefined single holdouts,
train-only rows, grouped/time-series/shuffled masks and changed dimensions are
covered. The standalone checked comparison passed 1,936 groups: 1,921 direct
source checks, one rank-deficient coefficient equivalence and 14 unused path-
tail counter differences. The equivalence requires rank deficiency and zero
training/query action of the coefficient delta. Tail differences require the
same shortest path, with differing folds saturated to their training rank or
from the verified exact sparse-signal fixture. All retained errors, selected
sparsities, final refit iterations and predictions compare directly. The query
fixture respects the feature dependencies; coefficients outside the training
affine span are not identifiable. Tolerances are atol=rtol=3e-8; ordinary integer
metadata is exact. Numerical differences are listed separately in the report.

The independent native test checks an orthogonal sparse signal with known
first-step/selected errors, coefficients, intercept and score, common early
stopping, repeated fits, input preservation and custom partitions. Added
omp_sparsity_selection example and 30 isolated invalid-input cases.

Inventory: 159 core types, 74 source modules, 77 native test programs,
56 reference suites and 75,407 result groups, plus 28 separate NumPy stream
groups. Isolated invalid-input inventory is 1,477. Original contributions
remain MIT licensed, with the full sklearn BSD notice and detailed provenance
retained. No runtime dependency or public algorithm extension is added.

Validation completed: all 77 native test programs passed with bounds checking,
backtraces and invalid/zero/overflow traps, and again in the release profile.
The 1,936-group OMP CV comparison passed against both the fpm checked and
optimized release executables, including two separately recorded source
empty-path failures. Numerical-difference groups are reported explicitly rather
than counted as direct source equality. All 1,477 isolated invalid-input cases
passed. The checked example selected three features, with validation MSE
0.0014229439397628596 and training R2=0.9997607529259565. Python syntax, Fortran
line limits, diagnostic inventories, coverage/type counts, license retention and
git whitespace checks passed. Other Python reference suites were not rerun;
existing numerical estimator bodies are unchanged apart from the new export.

The final checked report has 1,921 direct comparisons, one coefficient
equivalence and 14 unused tail-counter differences. The release report has
1,922 direct comparisons and 14 unused tail-counter differences; no coefficient
exception is needed in that profile.


## Statistical univariate selectors (2026-11-20)

Added SelectPercentile, SelectFpr, SelectFdr, SelectFwe and GenericUnivariateSelect
in sklearn_univariate_selection, with exports through sklearn_feature_selection
and the facade. Controls reproduce source linear percentile interpolation,
NaN cleaning, input-order percentile ties, stable top-k ties, strict FPR/FWE
and inclusive Benjamini-Hochberg FDR thresholds. Generic mode supports
percentile/k_best/fpr/fdr/fwe, including zero/all/above-P top-k. Projection
controls remain live after fitting. fit_scores accepts external scoring results;
probability-free scores work for percentile/top-k. Public fitted arrays retain
scores, p-values and feature counts. Refit replaces/clears state correctly.

New classification/regression score helpers provide ANOVA, chi-square and
force-finite regression F scores with p-values, without modifying the older
score-only routines or SelectKBest. Typed fit defaults follow the existing
Fortran convention (integer classification, real regression). Classification
can choose chi2; regression can disable centering. Explicit IEEE handling
preserves constant/degenerate NaNs, exact-correlation finite maxima and the
negative-F rounding endpoint distinction between source stats.f.sf and fdtrc.
Native private survival functions use DLMF beta fractions/symmetry, modified
Lentz evaluation and integer/half-integer upper-gamma recurrences with erfc and
positive log-domain terms. Mathematical provenance links and sklearn/NumPy
BSD notices are retained; no SciFort source is copied and no dependency added.

The standalone checked suite passed 12,964 groups across 1,320 cases: four
seeds, five types and 66 successive fits per type/seed. There are 11,060 direct
source comparisons, 1,884 zero-column inverse identities and 20 separately
reported near-perfect squared-correlation checks. The native zero-column
inverse returns N-by-P zeros, consistent with existing native selectors; the
source dense inverse's minimum-feature rejection is explicitly verified.
Regression F scores above 1.e10 that fail ordinary comparison must still agree
in 1-df/(F+df) within 64 machine epsilons; all other score elements and all
probabilities, masks and data projections remain source checked. Maximum
observed squared-correlation difference was 5.78e-15. Positive normal-range
p-values have 1,092 additional log-tail groups, avoiding misleading absolute
near-zero agreement. Numeric tolerance is atol=rtol=3e-8 for ordinary scores/
probabilities and atol=1e-7, rtol=3e-8 for log tails; masks, indices, dimensions
and copied/restored input values are exact.

Fixtures cover balanced/unbalanced/many/single/one-sample-per-class targets,
constant/duplicate/zero features, real target constants and perfect correlation,
centering, feature scaling, single/wide/changing feature counts, large N,
all generic modes, threshold equalities, zero/full selections, NaN/infinite/
extreme/tied custom scores and tiny probabilities. Analytic native tests check
masks, tie rules, strict/inclusive boundaries, post-fit control changes, refits,
input preservation, null restoration and independent F(1,2), F(2,2) and
chi-square(2) tail identities. Forty-two new isolated diagnostics cover invalid
controls, mode/score mismatches, missing probabilities, residual degrees of
freedom, data/query dimensions and finiteness, nonnegative chi2 and unfitted
methods. A statistical_feature_selection example fits a regression after FDR
screening. Dense real64 arrays and integer classification labels remain the
supported data representation; sparse/object inputs, Python callback objects,
feature-name integration, metadata routing and other score families are outside
this wrapper.

Inventory: 164 core types, 75 source modules, 78 native test programs,
57 reference suites and 88,371 result groups, plus 28 separate NumPy stream
groups. Isolated invalid-input inventory is 1,519. Original contributions
remain MIT licensed with complete upstream notices and specific attribution.
No runtime dependency or public algorithm extension was added.

Validation completed: all 78 native test programs passed in both checked and
release profiles. The full checked suite was repeated successfully on the final
sources after the regression rounding-edge correction. The 12,964-group suite
passed against both fpm checked and optimized release executables, with the
same breakdown in each: 11,060 direct source comparisons, 1,884 explicit
zero-column inverse identities and 20 near-perfect squared-correlation checks.
All 1,092 log-tail groups passed. All 1,519 isolated invalid-input cases passed.
The checked example selected columns 1 and 3, with p-values 2.347592312456038e-11
and 1.377539268473622e-20 and post-selection training R2=0.9985943264061256.
Python syntax, Fortran line limits, diagnostic case-set equality/uniqueness,
coverage/type/source/test/reference counts, license retention and git whitespace
checks passed. Other Python reference suites were not rerun; their existing
numerical estimator bodies were unchanged by this batch.

## PolynomialCountSketch (2026-11-20)

Added the core sklearn.kernel_approximation.PolynomialCountSketch counterpart,
exported as polynomial_count_sketch by sklearn, sklearn_kernel_approximation
and its focused module. Dense real64 fit/transform/fit_transform implement
augmented polynomial TensorSketch: sqrt(gamma) feature scaling, an optional
sqrt(coef0) constant, independent degree hash/sign tables, per-degree count
sketches and the inverse FFT of their spectral product. Scalar seeds reproduce
NumPy legacy RandomState integer draws, including C-order traversal and the
one-bucket case. Public index_hash_ values remain zero-based, with table shape
(degree,augmented_features); bit_hash_ stores -1/+1 signs. Refit replaces all
hash tables and fitted dimensions. Inputs are preserved.

Positive arbitrary component counts use radix-2 FFTs or Bluestein convolution,
with a plan reused across rows/degrees within each transform. The MIT SciFort
FFT helpers are adapted from local src/signal_spectral.f90, identified by SHA-256
77283c0c5c14214e9a9c95c04ac9d737ee5153444a1027d38acd2f2d14f082a6.
The SciFort copyright/header and complete MIT license are retained, and the
modifications and Bluestein citation are recorded in THIRD_PARTY_NOTICES.md.
The plan is implementation support, excluded from the sklearn facade/core type
count. No SciFort runtime dependency or public algorithm extension is added.

Controls default to gamma=1, degree=2, coef0=0, n_components=100 and native
random_state=0. Fitted hash tables and n_features_in_/n_features_out_ are public.
Source live controls are retained: gamma/coef0 magnitudes, a prefix degree and
component counts that contain all active buckets can change after fitting.
Actual output width follows current n_components; n_features_out_ stays at the
fitted value, matching source. Changing zero/nonzero coef0 status requires a
refit because it changes augmented width. Negative finite coef0 is accepted
but produces source-compatible all-NaN features, implemented explicitly to
avoid invalid floating traps. Native transform checks invalid changed controls
and inconsistent fitted hashes with explicit diagnostics. Supported seeds are
nonnegative default integers. Sparse input, Python RNG objects/global streams,
feature-name methods and metadata routing are outside this wrapper.

compare_polynomial_count_sketch.py checks 2,950 direct sklearn 1.6.1 result
groups across five seeds, 140 scenarios and 590 fit/transform operations.
Seeds include zero and the largest supported default integer. Each operation
checks exact bucket/sign tables and fitted/output metadata, plus features and
Gram matrices at atol=rtol=3e-8 with equal NaNs. Fixtures cover one/power-of-two/
composite/prime component counts through 2,048, degrees 1 through 7, gamma=0,
zero/positive/negative offsets, zero rows, one/wide/changing feature counts,
constant inputs, live controls, repeated fitting and independent query data.
All groups use the source oracle without identity-based substitutions.

Native tests independently enumerate an exact degree-two tensor sketch and
compare complex FFTs to direct Fourier sums for short lengths. Eighteen FFT
lengths through 1,009 check forward/inverse round trips, including radix-2,
composite and prime sizes. Additional checks cover input preservation, signed
constant maps, deterministic seed/refit behavior and negative-offset NaNs.
Thirty-six new isolated diagnostics cover parameter domains, empty/nonfinite
data, unfitted transforms, query/augmented dimensions, invalid changed
controls, hash shapes/buckets, integer work-size overflow and FFT plan misuse.
The polynomial_kernel_features example uses the public facade to compare a
mapped dot product with the corresponding exact polynomial kernel.

Inventory: 165 core types, 77 source modules, 79 native test programs,
58 reference suites and 91,321 result groups, plus 28 separate NumPy stream
groups. Isolated invalid-input inventory is 1,555. Original contributions
remain MIT licensed with complete upstream notices and specific attribution.

Validation completed: all 79 native test programs passed with bounds checks,
backtraces and invalid/zero/overflow traps, and all 79 passed in release mode.
The 2,950-group source suite passed against both fpm checked and optimized
release executables, with exact hashes/signs/metadata and no substituted
oracles. All 1,555 isolated invalid-input cases passed. The checked public
example mapped three samples to 1,009 components; its first-pair approximate
kernel was 10.890624999999352 versus the exact polynomial kernel 11.390625.
These illustrate a randomized approximation, not an exact kernel guarantee.
Python syntax, all Fortran line limits/trailing whitespace, diagnostic case
uniqueness/equality, type/module/test/reference inventories, retained licenses,
source hash and git whitespace checks passed. Other Python estimator reference
suites were not rerun; their existing numerical bodies were unchanged. Their
saved passing results form the historical aggregate of 91,321 core groups,
with the 28 NumPy stream groups reported separately.

## LabelBinarizer and MultiLabelBinarizer (2026-11-20)

Added label_binarizer and multi_label_binarizer as core sklearn preprocessing
counterparts, exported by sklearn, sklearn_preprocessing and the focused
sklearn_label_binarization module. Both implement fit, transform, fit_transform
and integer/real inverse_transform overloads, with allocated integer outputs.
The public label_set container represents a ragged row of integer labels;
it is data representation, excluded from the algorithm count like index_split.
No public algorithm extension or runtime dependency is added.

LabelBinarizer discovers sorted classes from a nonempty rank-one integer target,
preserves source one/two-class versus multiclass encoding, configurable integer
negative/positive values and unknown-label negative entries. Default neg_label=0
and pos_label=1 can change after fitting. One-class binary queries are entirely
negative. Source query-dependent target typing is retained: three distinct
query labels produce all fitted class columns even after binary fitting, and
can mark a known single class positively. Refit replaces classes_/y_type_.
Binary inverse uses a strict midpoint/explicit threshold, using the second
column for two-column scores. Multiclass inverse picks the first maximal score
and clips an extra-column winning index, as in source. Native midpoint
arithmetic avoids default-integer overflow for extreme encoding values.
Empty inverse row counts are supported. Native inverse requires finite scores,
positive width and a finite binary threshold; source's absent single-class
positive index becomes an explicit diagnostic. Rank-two multilabel-indicator
LabelBinarizer fitting, sparse output and string/object labels are unsupported.

MultiLabelBinarizer discovers a sorted union or preserves a unique constructor
classes ordering. Constructor changes wait for the next fit; classes_ is a
fitted snapshot. Repeated labels are idempotent. Unknown labels are ignored
with source indicator values, without Python warnings. Unallocated/empty row
values, empty datasets, empty fitted/configured vocabularies and zero-column
indicator matrices are supported. Inverse validates exact 0/1 integer or real
indicators and returns allocated label sets in fitted class order. Inputs are
preserved. Python iterable/object/string APIs, sparse formats and feature-name
integration are outside this wrapper.

compare_label_binarization.py checks 6,285 exact sklearn 1.6.1 result groups
across 555 scenarios and five seeds. The 375 LabelBinarizer scenarios check
one/two/many classes, known/unknown/multiclass queries, zero/negative/positive/
extreme integer encodings, fit and fit_transform parity, query dimensions,
target type, both inverse overload paths, threshold equalities, tied maxima,
changed score widths, live encoding controls and dimension-changing refits.
The 180 MultiLabelBinarizer scenarios check automatically discovered, reversed,
subset and empty class orders, duplicates, unknown labels, empty rows/datasets,
integer and real inverse indicators, complemented indicators, fitted snapshots
and rediscovery on refit. All groups use the source oracle and exact integer
array equality, including ragged counts and flattened inverse values.

Native tests independently assert binary class ordering, strict thresholds,
query-dependent widths, zero positive encoding, first-max ties and index
clipping, single-class negatives, empty inverse rows, duplicate/empty label
sets, explicit class ordering, fitted snapshots, zero-column inverse and input
preservation. Twenty-six new isolated diagnostics check invalid encoding
controls, empty/unfitted label methods, inverse widths and nonfinite values,
missing single-class positive labels, duplicate constructor classes and invalid
integer/real indicators. example/label_encoding.f90 exercises both types through
the public facade. sklearn source attribution and complete BSD notices remain
retained; original contributions use the project MIT license.

Inventory: 167 core types, 78 source modules, 80 native test programs,
59 reference suites and 97,606 result groups, plus 28 separate NumPy stream
groups. Isolated invalid-input inventory is 1,581.

Validation completed: all 80 native test programs passed with bounds checks,
backtraces and invalid/zero/overflow traps, and all 80 passed in release mode.
All 6,285 exact source comparison groups passed against both fpm checked and
optimized release executables, covering 555 scenarios with no substitute
oracles. All 1,581 isolated invalid-input cases passed, including the 26 new
diagnostics. The checked public example restored binary targets [-3,7,7,-3]
and produced multilabel columns in requested order [30,10,20], preserving
an empty sample and removing repeated labels on inverse restoration.
Python syntax, Fortran line limits/trailing whitespace, diagnostic case
uniqueness/equality, type/module/test/reference inventories, MIT/upstream
notice retention and git whitespace checks passed. Other Python estimator
reference suites were not rerun; their numerical bodies were unchanged.
Their saved passing results form the historical aggregate of 97,606 core
groups, with 28 NumPy stream groups reported separately.

## Mutual-information and Pearson scoring (2026-11-20)

Added four core sklearn functions: mutual_info_classif, mutual_info_regression,
r_regression and mutual_info_score. Feature scorers are exported through sklearn
and sklearn_feature_selection; the clustering metric through sklearn_metrics;
all are available in the focused sklearn_mutual_info module. They add algorithms
without increasing the estimator/transformer/splitter type count. No public
extension, SciFort dependency or external runtime is introduced.

MI implements Kraskov continuous/continuous Chebyshev neighbor radii, Ross
continuous/discrete class-wise radii and singleton removal, and categorical
contingency MI. Integer digamma differences use harmonic tables. Negative
estimates are clipped to zero. Dense all-pairs distance/count passes use O(N)
scratch per feature, quickselect for kth radii and O(N**2) distance work. The
categorical metric stores observed joint pairs rather than a dense K-by-L table.
Source preprocessing scales without centering, distinguishes feature-array and
scalar-target near-zero scales, and adds 1.e-10 Gaussian perturbations with
NumPy-compatible scalar-seeded C-order draw traversal. The private pairwise sum
adapts NumPy 2.2.6's eight-accumulator/128-element/aligned-recursion reduction.
For class-wise Euclidean radii, native arithmetic follows source's auto backend
choice: ordered Gram terms for brute force and squared differences otherwise.
Radius counts use reduced squared distances, retaining source boundary rules.

Dense real64 X and regression targets, integer classification/cluster labels,
positive n_neighbors and nonnegative default-integer seeds are supported.
discrete_features accepts scalar logical, logical mask or nonempty native
one-based column indices. Fractional real category codes are allowed. Continuous
neighbors must be fewer than N; class-wise counts adapt to available repeats,
ignoring singleton labels. Copy=False, sparse data, Python RNG/callback objects,
joblib parallelism and string/object labels are outside the native interface.
Inputs are preserved. MI vectors can feed score-only univariate selectors.

r_regression retains signed correlations, centered/uncentered moment formulas,
division order and force_finite semantics: NaNs become zero by default, while
signed infinities remain. Explicit IEEE branches handle undefined/underflowed
denominators with invalid/zero/overflow traps enabled. Exact constant columns
have undefined mathematical correlation; BLAS summation can change source/native
outputs among NaN, zero, signed infinity or tiny moment-rounding artifacts.
Those cases are reported separately rather than asserted to be source-equal.
mutual_info_score accepts nonempty integer label pairs and preserves zero-entropy
and label-permutation invariance; precomputed contingency input is not exposed.

compare_mutual_info.py checks 2,196 result groups over six seeded datasets with
N=7,16,31,64,127,257, including seed 0 and the largest supported scalar seed.
There are 1,158 score groups and 1,038 exact input-preservation identities.
The pre-build checked run has 1,150 direct source score groups and eight groups
with an explicit exact-constant Pearson certificate. Such groups still compare
every nonconstant column directly. Constant columns must be exactly equal down
the input column; both outputs must be IEEE undefined/zero or finite artifacts
bounded by 32*sqrt(N*epsilon). force_finite=True cannot return NaN. Other source
comparisons use atol=rtol=3e-8, with equal NaNs and signed-infinity agreement.
The report records affected columns, source/native values and bounds.

Fixtures cover all continuous/discrete combinations, mixed features, every
native mask/index/scalar overload (with repeated/reordered indices),
k=1/2/3/N-1 and adaptive counts above N,
single/wide features, singleton categories, repeated/constant/rounded data,
small scales, offsets, constant targets, one-to-four-sample MI edges, recursive
reductions, both Pearson options, tied/duplicate features, underflowed norms,
and empty/single/permuted clustering labels. Empty metric inputs are explicitly
confirmed as source rejections rather than source score comparisons. Four source
rejections cover that case, singleton-only continuous/discrete samples, unique
discrete regression features and invalid continuous neighbor counts.

Native tests assert categorical entropy/independence, label permutation,
single-sample categorical input, mask/index equivalence, deterministic seeds,
input preservation, signed Pearson values, uncentered cosine, undefined NaNs,
finite fallbacks, one-sample centering and preserved underflow infinities.
Thirty-four new isolated diagnostics cover invalid controls, masks/indices,
data sizes/finiteness, neighbor counts, absent repeated labels and metric sizes.
example/nonlinear_feature_scoring.f90 combines MI ranking with score-only
selection, Pearson scores, classification MI and the discrete metric via sklearn.

Original contributions remain MIT; complete sklearn/NumPy BSD notices and the
NumPy summation source attribution are retained. Research references follow
sklearn: Kraskov, Stogbauer and Grassberger (2004), and Ross (2014).
Inventory: 167 core types, 79 source modules, 81 native test programs,
60 reference suites and 99,802 result groups, plus 28 separate NumPy stream
groups. Isolated invalid-input inventory is 1,615.

Validation completed: all 81 native test programs passed with bounds checks,
backtraces and invalid/zero/overflow traps, and all 81 passed in release mode.
The full checked suite was repeated successfully on the final sources after
the repeated-index definition correction. Verbose execution confirms 81 unique
test executables; the new test and module artifacts postdate the final sources.
Both fpm checked and optimized source suites passed all 2,196 groups: 1,150
direct source score comparisons, 1,038 exact input identities and eight explicitly
reported constant-Pearson certificate groups. All 846 MI feature-score groups
and 120 categorical metric groups compare directly to source; nonconstant
Pearson columns remain directly checked even in certificate groups. All four
source rejection checks and all 1,615 isolated native invalid-input cases passed.

The checked example gave regression MI=1.2226076540426423 on the nonlinear
first feature, versus Pearson r=-1.1615170503735291e-4. The fixed two-feature
selector returned columns 1 and 4; column 4 resolves a zero-score tie, rather
than establishing additional informative structure. The selected shape is 80x2.
Classification MI for the first feature was .64314486656419945 nats and target
self-information .68185460873078285. Python syntax, Fortran line/whitespace
checks, diagnostic uniqueness/equality, inventories, license retention and git
whitespace checks passed. Other Python estimator reference suites were not
rerun; their numerical bodies were unchanged. Their saved passing results
contribute to the historical aggregate of 99,802 core groups, with 28 NumPy
stream groups reported separately. The project MIT license and complete
sklearn/NumPy upstream notices remain retained.

## Clustering information and pair agreement (2026-11-20)

Added normalized_mutual_info_score, adjusted_mutual_info_score,
fowlkes_mallows_score, contingency_matrix, pair_confusion_matrix, entropy,
expected_mutual_information and homogeneity_completeness_v_measure. All eight
are sklearn.metrics.cluster counterparts, exported through sklearn_metrics and
sklearn, with focused access through sklearn_cluster_information. Existing
homogeneity_score/completeness_score/v_measure_score now delegate to the same
backend, adding source empty-input conventions and optional V-measure beta.
These add metrics/helpers rather than estimator types. No public extension or
runtime dependency is introduced.

Sorted label bins and observed joint cells use heap sorting and O(N) histogram
storage. Observed information follows source log/epsilon conventions with
NumPy-compatible pairwise reduction. Pair confusion stores ordered distinct
sample counts in int64, with rows TN/FP and FN/TP; Fowlkes-Mallows uses the
geometric mean of pair precision/recall, returning zero with no true positives.
The explicit contingency function allocates a dense sorted-class count matrix;
its two-argument overload returns int64, while real64 eps selects an allocated
real64 matrix with finite nonnegative smoothing added to every cell. Metrics
themselves do not materialize a dense K-by-L matrix.

Normalized and adjusted MI implement min/geometric/arithmetic/max entropy
averages, defaulting to arithmetic. Expected MI follows sklearn's hypergeometric
sum over feasible joint counts with cached intrinsic log factorials. AMI
retains negative values and source signed-epsilon denominator flooring. The
public expected-MI helper accepts default/int64 nonnegative count matrices,
requires their sum to match nonnegative n_samples, supports zero margins and
nonempty all-zero tables, and guards invalid extents/work sizes and sum overflow.

Dense integer label arrays must have equal lengths. Empty NMI/AMI and H/C/V
return one, entropy(empty) returns one, contingency(empty) is 0-by-0, pair
confusion(empty) is zero and empty/singleton Fowlkes-Mallows is zero, following
source. Beta is finite/nonnegative. Source's beta=0, zero-completeness division
error becomes an explicit native diagnostic; independent partitions with H=C=0
still give V=0. Sparse matrices, dtype overrides, object/string labels and the
deprecated ignored FMI sparse flag are not exposed. Inputs are preserved.

compare_cluster_information.py checks 2,720 groups over five seeds, 200 paired
partition fixtures and 125 manual count matrices. The initial checked run has
2,510 direct source groups, 200 exact input identities and ten AMI groups with
explicit zero-chance-gap certificates. Every ordinary AMI value and unaffected
normalization remains directly source compared at atol=rtol=3e-8. Expected MI
is directly compared at atol=rtol=5e-11; all integer tables, pair counts, shapes
and input copies are exact. Source beta-zero division and empty expected-table
rejections are explicitly verified, not reported as numerical equalities.

The adjustment denominator is mathematically zero for min-normalized AMI when
one partition is entirely singleton clusters (and for every average when both
are singleton partitions). Intrinsic/SciPy/libc log-gamma rounding can then
substantially change the final source/native score. Certificates are limited to
those exact partition conditions: source/native MI and expected MI must agree
with the other entropy within 64*N*epsilon*max(1,H), and each reported adjusted
score must lie in the source epsilon-guarded ratio interval obtained from its
observed NMI/entropy/expected MI, allowing four epsilon units for reconstructing
the observed MI. All nondegenerate components remain source checked. Reports
record conditions, source/native values, component witnesses and ratio bounds.

Fixtures cover N=0,1,2,3,4,5,7,16,31,64,127,257,513, independent/identical/
renamed/refined/unbalanced/singleton partitions, all four averages, zero through
1.e300 beta, eps 0/1.e-12/.25/10, empty and zero-entropy limits, zero matrix
margins, count overloads and expected MI up to N=1,200. Native tests independently
check sorted contingency orientation, smoothing, analytic ordered pairs/FMI,
binary entropy, the exact balanced hypergeometric EMI log(2)/3, AMI=-.5 on
balanced independent partitions, perfect-label scores, empty conventions,
weighted V-measure wrappers and 70,000-sample counts above 32-bit range.
Thirty-four new diagnostics cover lengths, methods, eps/beta controls,
zero denominators, negative/inconsistent counts, empty matrices and overflow.
The clustering_agreement example exercises all eight functions through sklearn.

Source paths/hashes, sklearn and NumPy BSD notices, and Vinh/Epps/Bailey (2010)
and Fowlkes/Mallows (1983) research references are recorded in
THIRD_PARTY_NOTICES.md. Original contributions retain the project MIT license.
Inventory: 167 core types, 80 source modules, 82 native test programs,
61 reference suites and 102,522 result groups, plus 28 separate NumPy stream
groups. Isolated invalid-input inventory is 1,649. The original integration
comparison is rerun because the existing H/C/V numerical wrappers changed;
compare_sklearn.py now also accepts a prebuilt executable for profile reuse.

Completed validation: all 82 native test programs passed both the checked
build (-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow) and release
build. Each profile passed 2,720 new result groups: 2,510 direct sklearn
comparisons, 200 input-preservation checks and 10 explicitly recorded
zero-chance-gap AMI certificates. The original 192-group integration suite
also passed against the checked executable. All 1,649 isolated invalid-input
cases passed, and the checked clustering_agreement example ran successfully
(NMI 0.18872187554086706, AMI 0.08346281722824393, FMI 0.5).
A diagnostic fixture initially exceeded the driver's 32-character argument
buffer; its name was shortened and a registration-length guard added before
the complete diagnostic rerun passed. No numerical implementation changed
after the checked/release tests. Final audits passed for Python syntax,
Fortran line lengths/whitespace, unique and complete diagnostic registration,
report/inventory totals, source hashes, retained licenses and git diff checks.
Other reference suites contribute their previously recorded results to the
aggregate inventory; they were not all rerun for this batch.

## Randomized, repeated and exhaustive resampling batch

Added seven core splitter types: shuffle_split, stratified_shuffle_split,
group_shuffle_split, repeated_kfold, repeated_stratified_kfold, leave_p_out,
and leave_p_groups_out. Both sklearn_model_selection and sklearn export them;
the new sklearn_resampling backend also owns index_split, whose existing
model_selection import remains source compatible. All seven expose split and
get_n_splits. Randomized splits reproduce scalar-seeded NumPy RandomState
indices via the existing native legacy RNG. Repeated folds keep a single
stream across repeats and return sorted sample indices. Stratified shuffling
uses source quota allocation and randomized ties, while repeated stratified
folds encode classes by order of first appearance. Group holdouts operate on
distinct sorted labels and prevent group leakage. Exhaustive holdouts follow
lexicographic combinations with checked binomial counts before allocation.

Shuffle controls support independent integer counts or real64 fractions,
source ceil/floor/complement rules, and partially unused samples/groups.
GroupShuffleSplit defaults to five iterations and 20% of groups; other shuffle
types default to ten iterations and 10% of samples. All random seeds default
to zero; Python None/object seeds are outside the native API. Repeated
stratification allows rare classes when another class meets n_splits, returning
the source indices without Python's warning. Returned folds are eager arrays,
so exhaustive splitting retains combinatorial memory/time costs. Core API and
limitations are documented in API.md; no new extension or dependency is added.

The new reference suite checks 23,280 ordered-index/count/identity groups over
820 fixtures and five seeds (0, 7, 42, 2026, 314159): 20,820 sklearn groups and
2,460 input-preservation/repeatability groups. Cases include populations
2--513, unequal class/group sizes, a single class, rare classes, randomly
ordered negative labels, quota ties, rounded/fractional/count/mixed controls,
unused samples/groups, uneven repeated folds, up to 127 folds per repeat,
and exhaustive p=1/2/n-1 combinations. Every native index is compared exactly
with sklearn's one-based translated output; no tolerance or alternative RNG
oracle is used. New native tests independently check defaults, source counts,
no sample/group leakage, balanced class allocations, sorted repeated indices,
once-per-repeat sample coverage and lexicographic exhaustive combinations.

Forty-five additional isolated diagnostic cases cover invalid samples,
counts/fractions/conflicts, nonfinite fractions, empty training sets, class
constraints, group constraints, repeat controls, invalid seeds and split-count
overflow. A resampling example demonstrates random/repeated/group holdouts.
Baseline source paths/hashes and retained BSD notices are recorded in
THIRD_PARTY_NOTICES.md. Original contributions remain MIT.
Inventory: 174 public core types, 81 source modules, 83 native test programs,
62 reference suites and 125,802 result groups, plus 28 separate NumPy RNG
stream groups. Isolated invalid-input inventory is 1,694. Existing numerical
splitter implementations are unchanged; the original integration comparison
and group-splitter reference suite are rerun to check the index_split move.

Completed validation: all 83 native test programs passed both checked
(-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow) and release builds.
Each profile passed all 23,280 new result groups across 820 fixtures:
20,820 exact sklearn index/count comparisons and 2,460 input-preservation or
repeatability checks. The original 192-group integration suite and all 362
existing group-splitter result groups also passed. All 1,694 isolated
invalid-input cases passed. Two empty-group rejection expectations initially
named the outer splitter diagnostic, while the shared label helper correctly
reported 'labels must be nonempty'; the expectations were corrected before
the full diagnostic rerun passed. Numerical sources did not change after the
checked/release tests. The checked resampling example passed, reporting six
repeated stratified folds and six exhaustive two-group holdouts, with first
held-out sample indices [4,5,6,7]. Final audits passed for Python syntax,
Fortran formatting, unique/complete diagnostic registration, exact oracle and
inventory totals, upstream source hashes, retained licenses and git diff
checks. Other reference suites retain previously recorded results in the
aggregate inventory and were not all rerun for this batch.

## Dense exact t-SNE and trustworthiness batch

Added tsne and trustworthiness to sklearn_manifold and the sklearn facade,
with sklearn_tsne and internal sklearn_tsne_core implementations. Dense exact
t-SNE supports PCA, random and explicit float64 initialization; the local
NumPy-compatible Gaussian stream; randomized-SVD loading sign conventions;
source float32 probability search and single/mixed/double optimizer paths;
Student degrees of freedom above three embedding dimensions; adaptive gains;
two exaggeration/momentum phases; 50-step gradient/progress checkpoints;
automatic/explicit learning rates and fitted embedding, KL, iteration,
feature count and resolved rate attributes. The largest-finite-KL/iteration
convention for a full max_iter=250 run is preserved. Stable scaled gradient
norms prevent underflow-based premature stopping. Source precomputed squaring
is reproduced on a copy, preserving caller input. Trustworthiness computes
neighborhood rank penalties with native stable index tie choices.

This is the exact dense algorithm, not Barnes-Hut. Quadratic memory/pair work,
no out-of-sample transform, nonnegative default-integer seeds, float32 working
ranges, native metric support and tie differences are explicit in API.md.
No new dependency or public extension is added. Internal diagnostic helpers
and trace outputs are not exported by the sklearn facade.

The new reference suite checks 1,620 result groups over 95 fixtures and five
seeds (0,7,42,2026,314159). It covers 1/2/4 dimensions, sample counts 7--31,
Euclidean/Manhattan/cosine/precomputed input, explicit/random/PCA starts,
normal and exaggerated probabilities, fixed/automatic rates, float32 and
float64 gradients/short trajectories, full two-phase fits, checkpoint stops,
max_iter=250 and unchanged inputs. Affinities, source objective/gradient at
initial/final native positions, short optimizer steps, stable full trajectories
and unique-neighborhood trustworthiness are compared directly to sklearn.
All explicit-initialization paths additionally replay the public fit exactly
and verify each native step against the source objective/gradient and NumPy
update formulas, including gains, stopping and the pre-update reported KL.
Small reduction errors can change near-zero gain signs or amplify over a
nonconvex trajectory; such fields require recorded step certificates rather
than broader coordinate tolerances. Tied embedded-neighborhood differences
require the independent native stable-rank identity and valid rank bounds
for both source and native scores, with unique original-space ranks.

The initial checked prototype recorded 1,306 direct sklearn groups,
195 replay-identity groups, 95 input-identity groups, 15 sklearn/stable-tie
certificates and nine sklearn/trajectory certificates. Its 65 trajectories
contained 18,300 source-checked steps and 73 cancellation-bounded gain choices;
maximum scaled-gradient/update/position/objective differences were about
1.1e-15/5.1e-16/1.8e-15/3.6e-14. Final checked/release reports retain their own
profile-specific oracle/cancellation counts. No broad geometric alignment or
relaxed final-coordinate assertion substitutes for step-level verification.

Native tests independently check uniform joint probabilities, zero objective
and gradient at uniform collapse, finite-difference gradients, fixed points,
very-small-gradient stopping without norm underflow, deterministic refits,
input preservation, both float32 initialization paths, source max_iter=250
and checkpoint behavior, analytic trustworthiness and collapsed ties.
Forty-six new diagnostic cases cover invalid controls, seeds, initialization,
metric shape/negativity, probability/objective domains and working ranges.
The nonlinear_embedding example demonstrates PCA-initialized exact t-SNE and
trustworthiness. Upstream source hashes, research attribution and retained BSD
notices are recorded in THIRD_PARTY_NOTICES.md; original work remains MIT.
Inventory: 175 public core types, 83 source modules, 84 native test programs,
63 reference suites and 127,422 result groups, plus 28 separate NumPy RNG
stream groups. Isolated invalid-input inventory is 1,740. Existing numerical
implementations are unchanged; manifold re-exports the new implementation.

Completed validation: all 84 native test programs passed checked
(-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow) and release builds.
Both profiles passed all 1,620 result groups: 1,306 direct sklearn comparisons,
195 replay identities, 95 input identities, 15 stable-tie certificates and
nine trajectory-certified source comparisons. Each profile verified 18,300
steps across 65 explicit-initialization paths, with 73 recorded near-zero
cancellation gain choices. The original 192-group integration comparison and
all 1,740 isolated invalid-input cases passed. The checked nonlinear_embedding
example returned a 48-by-2 embedding, learning rate 50, KL
0.13799521601073550, zero-based iteration 399 and four-neighbor
trustworthiness 0.98820281124497988. Final audits passed Python syntax,
Fortran line lengths/whitespace, unique and complete diagnostic registration,
source/certificate oracle accounting, inventory totals, upstream source
hashes, retained licenses and git diff checks. Numerical sources did not
change after checked/release testing. Other reference suites retain their
previously recorded results in the aggregate inventory and were not all
rerun for this batch.

## Hessian eigenmapping and modified LLE batch

Extended the existing locally_linear_embedding type with method='hessian'
and method='modified', completing all four sklearn LLE methods. New finite,
nonnegative hessian_tol=1.e-4 and modified_tol=1.e-12 controls follow source
validation. Hessian fitting builds centered tangent/monomial neighborhoods,
full Householder QR complements and tolerance-normalized alignment blocks.
Modified fitting builds full local eigensystems, fixed trace-regularized
affine weights, median spectral ratios, cumulative-energy/searchsorted
almost-null-space sizes and Householder multiple-weight blocks. The native
backend reuses Jacobi SVD/eigen routines and completes local orthogonal bases.
The shared fit/fit_transform/transform and analytic constant-mode removal
remain intact; public reg affects fitting only for standard LLE and affects
query interpolation for every method. No public type or extension is added.

The new reference suite checks 3,535 groups over 95 fixtures and five seeds
(0,7,42,2026,314159), covering 18--32 samples, 3--12 input features,
6--12 neighbors, 1/2/3 embedding dimensions, both SVD and local covariance
branches, odd/even median selection, default/modified tolerances and nondefault
query regularization. Local source alignments are captured from the installed
sklearn implementation before global eigenanalysis. Source local spectral
values, regularized weights, thresholds, nullities, ordinary fitted/query
geometry and reconstruction error are compared; native local bases also
satisfy independent orthogonality and covariance eigen-residual identities.
Every modified alignment is independently rebuilt using the source formulas
on its validated native local eigensystem.

Nonunique global eigenspaces use source-matrix residual/lowest restricted
energy, orthogonality, zero-mean and source barycentric-query certificates.
Spectral-cutoff differences require measured near-equality bounds and a
counterfactual native-frame assembly using source nullities that matches the
source alignment. The initial checked prototype recorded 695 direct sklearn
groups, 2,620 spectral identities, 95 input identities, 50 source-formula/native
frame assemblies, 51 nonunique-eigenspace comparisons and 24 cutoff-certified
comparisons. There were 17 nonunique spectral fixtures and four single-row
cutoff fixtures. Final profiles retain their own complete oracle/certificate
accounting; no broad coordinate tolerance replaces these checks.

Native tests independently check alignment symmetry, positive semidefiniteness,
constant-mode annihilation, translation/scale invariance, affine constrained
regularized weights, embedding orthonormality and energy, deterministic refit,
input preservation, query transforms and the public reg control's distinct
fit/query behavior. Twenty-five new diagnostics cover tolerance domains,
method-specific neighborhood limits, zero spectra/empty nullspaces, zero
Householder denominator, malformed neighbor arrays and invalid feature data.
All 25 also passed a standalone checked probe before full fpm validation.
An lle_variants example demonstrates both methods and query coordinates.

The existing compare_manifold.py additionally accepts --executable, preserving
its default fpm invocation while enabling checked-profile regression testing
without a redundant build. Standard LLE, LTSA and MDS source comparisons are
rerun because the LLE type/interface was extended. Existing standard/LTSA
numerical bodies are unchanged. Dense spectral/rank/tie/working-range
limitations and source paths/hashes/research attribution are documented.
Original work remains MIT with retained sklearn BSD notices.
Inventory: 175 public core types, 84 source modules, 85 native test programs,
64 reference suites and 130,957 result groups, plus 28 separate NumPy RNG
stream groups. Isolated invalid-input inventory is 1,765. No dependency is added.

Completed validation: all 85 native test programs passed both checked
(-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow) and release builds.
Each profile passed all 3,535 result groups: 695 direct sklearn comparisons,
2,620 local spectral identities, 95 input identities, 50 source-formula/native
frame assemblies, 51 nonunique-eigenspace comparisons and 24 cutoff-certified
comparisons. Both recorded 17 nonunique spectral fixtures and four single-row
near-threshold cutoff fixtures. The original 192-group integration comparison
and all 138 existing MDS/standard-LLE/LTSA reference groups passed as well.
All 1,765 isolated invalid-input cases passed. The checked lle_variants example
ran both methods, returning 24-by-2 embeddings and query coordinates, with
alignment energies 0.99104370270070774 (Hessian) and 0.77574065232598721
(modified). Final audits passed Python syntax, Fortran formatting,
unique/complete diagnostic registration, report and oracle accounting,
certificate witnesses, inventory totals, upstream source hashes, retained
licenses and git diff checks. Numerical sources did not change after the
checked/release tests. Other reference suites retain previously recorded
results in the aggregate inventory and were not all rerun for this batch.

## Ranking curves, calibration and weighted-score batch

Added six public sklearn functions: roc_curve, precision_recall_curve,
det_curve, auc, calibration_curve and brier_score_loss. Added sklearn_ranking
and sklearn_calibration modules, metric re-exports and the calibration facade
import. Existing roc_auc_score and average_precision_score now use the shared
stable O(N log N) threshold backend, retain their first three native arguments,
and add sample_weight; ROC AUC also adds standardized max_fpr partial area.
The backend follows source stable-ascending/reversed tie order, zero-weight
filtering, independent weighted cumulative counts, curve thinning, endpoint
and DET trimming conventions. Explicit positive-label and single-class rules,
quiet IEEE undefined outputs and signed-weight arithmetic follow source
semantics. Partial AUC specifically preserves the source rejection when
zero weights remove all negative support. Calibration uses uniform/quantile
bins, source linear interpolation and left-edge assignment, and drops empty
bins. Brier uses source numeric-label inference and weighted squared error.
Native score/probability/AUC-coordinate inputs are finite dense real64 values;
working-range and invalid-domain diagnostics are explicit.

The new reference suite checks 5,584 groups over 365 fixtures and five seeds
(0,7,42,2026,314159): 5,215 direct sklearn groups, 365 unchanged-input checks
and four source/native partial-AUC rejections. Cases cover 1--513 samples,
unique/tied/constant scores, uniform/random/log-scale/zero/signed weights,
removed classes, singleton and custom labels, explicit one-vs-rest curves,
all thinning choices, increasing/decreasing/constant AUC coordinates,
partial limits .01--1, 1--11 uniform/quantile bins, repeated probabilities,
empty calibration bins and IEEE NaN/infinite outputs. Curve thresholds and
array shapes are compared exactly; other values use tight 3.e-12 tolerances
with matching NaNs. No alternative numerical oracle replaces source checks.

Independent native tests verify analytic ROC/PR/DET outputs, trapezoidal and
partial ROC areas, AP step integration, calibration edge/duplicate behavior,
weighted and singleton Brier rules, singleton/zero-support NaNs, signed AP
and a 100,000-sample closed-form rank identity. Fifty-three new diagnostic
cases cover every public path's sizes, finite data, labels, weights, partial
limits, coordinate monotonicity and calibration controls. All passed a
standalone checked probe before full fpm testing; two fixture literals were
corrected to real64 to match the max_fpr interface. The ranking_curves example
demonstrates threshold curves, scoring and calibration. Existing numerical bodies outside
ROC AUC/AP remain unchanged. Provenance and complete upstream notices are
retained; original work stays MIT. No new public type, extension or dependency
is added; the existing native positive-label AUC convenience is documented.
Inventory: 175 core types, 86 source modules, 86 native test programs,
65 reference suites and 136,541 result groups, plus 28 separate NumPy RNG
stream groups. Isolated invalid-input inventory is 1,818.

Completed validation: all 86 native test programs passed checked
(-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow) and release builds.
Each profile passed 5,584 result groups: 5,215 direct sklearn comparisons,
365 exact input identities and four paired source/native partial-AUC
rejections. The original 192-group integration comparison and all 1,818
isolated invalid-input cases passed. The checked example returned ROC AUC
.75, standardized partial AUC 2/3, average precision 5/6, Brier loss .158125
and calibration fractions [0,.5,1] at mean probabilities [.1,.375,.8].
The 100,000-sample native rank identity passed both profiles. Final audits
passed Python syntax, Fortran formatting, exact diagnostic registration,
report/oracle/rejection accounting, inventory totals, sklearn/NumPy source
hashes, retained licenses and git diff checks. Numerical sources did not
change after checked/release testing. Other reference suites retain their
previously recorded results in the aggregate inventory and were not all
rerun for this batch.


## Self-training classifier validation

`python validation/compare_self_training.py` compares threshold/k-best
self-training with Gaussian NB, uniform/distance-weighted KNN and logistic
bases against sklearn 1.6.1. It accepts --executable for an already-built
driver and --profile release for an optimized fpm driver. There are 495
fixtures over five seeds, one to four classes, zero/unlimited/capped iterations,
fully labeled data and inductive probability/decision outputs. Floating
outputs use atol=rtol=2.e-7; class/iteration arrays and input preservation are
exact. Multiclass logits are centered per row to remove their common gauge.

The native stable tied-cutoff rule can differ from source argpartition.
Every native iteration is independently checked using a source base fit:
original labels, new class assignments, threshold membership, k-best counts,
confidence ordering and valid stopping/refit behavior. Gaussian NB/KNN
probability saturation allows only 32 double precision epsilons at a cutoff;
logistic confidence ordering is exact. A history that differs from the
ordinary source estimator requires a certified tied cutoff. The final source
base is refitted on the certified transduction before query comparison.
Both profiles recorded 30 such fixtures, with 221 result groups. These are
separate from the 3,694 direct sklearn groups and 990 input identities.
See docs/API.md, docs/COVERAGE.md and example/self_training.f90 for scope.

Completed validation: all 87 native test programs passed checked
(-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow) and release builds.
Each profile passed all 4,905 groups across 495 fixtures, with the same
3,694 direct sklearn groups, 990 input identities and 221 source-certified
tied-cutoff groups. The original 192-group integration comparison and all
1,837 isolated invalid-input cases passed. The checked example ran two
iterations, returned all_labeled and predicted query classes [0,1].
The first orchestration invocation used a mistyped example target; fpm
returned success without executing it. The corrected target was run directly
and its output explicitly verified. This required no numerical source change.
Final audits passed Python syntax, Fortran formatting, exact diagnostic
registration, result/oracle accounting, inventory totals, upstream source
hashes, retained licenses and git diff checks. Numerical sources did not
change after checked/release testing. Existing reference suites retain their
previously recorded aggregate results; they were not all rerun for this batch.


## CalibratedClassifierCV validation

`python validation/compare_calibrated.py` exercises sigmoid/isotonic maps,
ensemble and pooled out-of-fold CV, prefit calibration, weighted KNN CV and
weighted prefit calibration with Gaussian NB, KNN and logistic bases. It
accepts --executable or --profile release. All 690 fixtures use installed
sklearn 1.6.1 with five seeds, 2/3/4 classes, source first-appearance fold
encoding, 3/5 folds, uniform/distance KNN, missing/unknown calibration
labels, zero calibration weights and large decision scores.

Raw base calibration/query responses and public probabilities compare
directly to source at atol=rtol=2e-6 (3e-6 for logistic). Every sigmoid map
also passes an independent source-objective gradient bound of 2.1e-9 and
cannot exceed a source optimizer loss by more than 2e-10, using native raw
responses. Each isotonic map is compared to a fresh sklearn fit on native
responses at the union of both knot sets and query responses, certifying
its complete piecewise linear function at 3e-12. Composition, original
estimator and input data have separate identities. Prediction differences
are allowed only when both probability gaps are within 64 machine epsilons;
all four observed source gaps are exactly zero and individually recorded.

Native analytic tests cover constant-score Platt smoothing versus empirical
isotonic means, weighted/zero priors, complements, deep copies, CV encoding,
mandatory refitting, state replacement and multiclass uniform fallback.
The shared logistic fit now centers multiclass intercepts to remove a
softmax null mode from pooled decision scores. This preserves its model
probabilities and improves calibration fidelity; binary fitting is unchanged.
Scope and current base/weight limitations are explicit in docs/API.md and
docs/COVERAGE.md. The package remains MIT with retained upstream BSD notices.

Completed validation: all 88 native test programs passed checked
(-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow) and release builds.
Each profile passed all 11,680 calibration groups over 690 fixtures with the
same oracle accounting: 6,086 direct sklearn groups, 3,520 fitted-map
certificates, 1,380 input identities, 690 composition identities and four
machine-precision argmax-tie groups. The source probability gap in every
recorded argmax exception is exactly zero, and both native/source gaps are
checked against 64 machine epsilons. Self-training additionally passed its
4,905-group source/identity/certificate suite in both profiles after the
shared logistic gauge correction. The original 192-group integration
comparison and all 1,860 isolated invalid-input cases passed.

The checked example returned sigmoid ensemble positive probabilities
[.23250143941352105,.38829563824685470,.76859123230451598] at inputs
[-2,0,2], and out-of-fold isotonic query labels [0,1,1]. Its output marker
was explicitly verified, preventing an unmatched fpm target from counting
as a successful example run. Final audits passed Python syntax, Fortran
formatting, exact diagnostic registration, report/oracle/exception
accounting, inventory totals, upstream source hashes, retained licenses
and git diff checks. Numerical sources did not change after checked/release
testing. Other existing reference suites retain their previously recorded
aggregate results; they were not all rerun for this batch.


## Weighted and streaming classifier validation

`python validation/compare_weighted_classifiers.py` accepts --executable
or --profile release. It validates every Gaussian batch/partial/refit state,
weighted logistic parameters and outputs, exact labels and unchanged inputs.
There are 681 fixtures with five seeds, 1--4 observed classes plus a declared
unseen class in selected cases, 1/4 features, uniform/integer/random/log-scale/
tiny/zero weights, fixed/zero/approximately normalized priors, changing batch
scales, constant data and tiny-C/large-weight fits. The 1,441 Gaussian
snapshots include source NaNs and infinite log posteriors under checked FP
traps. Counts/epsilon have zero absolute tolerance and 4e-11 relative tolerance;
other statistics use 4e-11, Gaussian outputs 3e-8, logistic outputs 3e-6.

All 481 logistic fits independently satisfy a normalized gradient bound of
3e-8. Thirty zero-weight-class cases have no finite intercept optimum;
probabilities, coefficients, predictions and gradients remain checked, with
60 finite intercept/decision groups recorded as limiting representations.
A synthetic saved fixture reproduces source Newton-CG line-search stalling.
Recovery requires a measured source gradient above 1e-8 and at least 100
times the native gradient (with a 1e-12 floor), a source Newton-Cholesky
solution whose gradient is at most 3e-8, non-worse native loss and recovered
loss agreement within 1e-10. Five resulting public-output groups are labeled
separately. The saved regression is retained in
validation/fixtures/logistic_tiny_weight_line_search.json.

The calibration suite now covers weighted CV for all three native bases,
including full-data refitting. Native tests additionally verify moment-skip
boundaries, source NaN argmax behavior, integer-weight replication, tiny-mass
intercept fitting, combined C/weight scaling and raw-score classification
when probabilities round to .5/uniform values. Self-training prediction
delegation and finite-confidence diagnostics are covered. See docs/API.md
for explicit API, weight and numerical-working-range limits.

Completed validation (2026-11-21): all 89 native test programs passed checked
(-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow) and release builds.
Each profile passed all 16,536 classifier groups over 681 fixtures, including
1,441 Gaussian snapshots and independent gradient checks for 481 logistic
fits. Both profiles retained identical oracle accounting: 15,309 ordinary
source groups, 681 input identities, 481 optimality checks, 60 limiting
intercept/decision groups and five source Newton-Cholesky groups from the
single saved Newton-CG stall. No softmax gauge exception was needed.

Both profiles additionally passed the expanded 13,760-group calibration
suite over 810 fixtures and the 4,905-group self-training regression over
495 fixtures. Their oracle counts and four calibration argmax-tie cases
remained identical between profiles. The original 192-group integration
comparison and all 1,902 isolated input diagnostics passed. The checked
example accumulated class masses [6,0] then [6,6], predicted Gaussian query
labels [0,1,1], and returned weighted logistic positive probabilities
[.08027985401471212,.54245967513063542,.94153307049888624] at [-2,0,2].
The example's actual output marker was verified.

Final audits passed Python syntax, Fortran formatting, unique diagnostic
registration/driver coverage, report/oracle/recovery accounting, inventory
counts, upstream source hashes, retained licenses and git diff checks.
Numerical sources stayed unchanged after checked/release validation.
Other reference suites retain their previously recorded aggregate results;
they were not all rerun for this batch. Original work remains MIT with
retained sklearn BSD attribution and no added dependency or public extension.


### Independent multioutput wrappers (2026-11-21)

Added core counterparts of MultiOutputRegressor and MultiOutputClassifier,
with independent fitted copies per target, refit/deep-copy behavior, ragged
per-output class probabilities, exact subset classification accuracy and
uniform-average regression R2. Streaming supports native SGD regression,
Gaussian NB and SGD classification; training weights follow each native
base's actual capabilities. The supported base families, first-call class
lists, source partial_fit metadata/shrink quirks and inherited limits are
specified in ../docs/API.md. Chains and a general estimator protocol remain outside
this implementation. See example/multioutput.f90.

The 3,030-group suite spans 255 fixtures across five seeds, six regression
base configurations and eight classification base configurations. It checks
single/multiple outputs, constant targets, nonconsecutive labels, weighted
fits and scores, pretrained template preservation, streaming first batches
with unseen classes, shrink/refit, averaged SGD, hinge prediction, ragged
probabilities, fitted coefficients/counters and input preservation.

Ordinary floating comparisons retain 3e-8 prediction/state/score and 3e-7
probability tolerances. Exceptions have explicit certificates: source-stump
and exhaustive SSE checks establish equal-gain CART splits at every native
node and reconstruct predictions; Complement NB differing class selections
must lie within 64 machine epsilons (scaled by source score magnitude) of
the source maximum. Weighted constant-target source R2 roundoff is bounded
and checked against source R2 on identically translated targets/predictions.
No difficult fixtures are removed or ordinary tolerances widened.


### RegressorChain and source lazy SGD scaling (2026-11-21)

RegressorChain joins the independent multioutput wrappers. Native fitting
supports observed preceding targets or cross-validated predictions, explicit
one-based ordering, seeded NumPy-compatible random ordering, integer KFold
and explicit index partitions. Query outputs are restored to original target
order. Native base families and training-weight limits are inherited from
multi_output_regressor. Full fits receive weights; source-style internal
cross_val_predict fits are unweighted when metadata routing is disabled.
Single-output chains do not evaluate unused CV folds. Explicit iterable
training indices preserve source behavior, including repeats or overlap.
See ../docs/API.md for configuration, metadata and scoring contracts.

The new reference suite covers 430 fixtures over five seeds and six native
base families. It checks ordering, teacher-forced versus out-of-fold training,
weighted fits/scores, singleton queries, fitted coefficients and counters,
refits and preserved pretrained templates/inputs. Tree validation certifies
every full and CV-fold tree against source stumps and exhaustive SSE scans,
then reconstructs both training features and sequential query predictions.
One duplicated-training-row fixture selects a different equal-gain split;
its two prediction/score groups carry a separate source tie certificate.
Ordinary prediction/state/score tolerance remains 3e-8; no difficult fixture
was removed or tolerance widened.

A weighted cross-validated SGD chain exposed roundoff accumulation in the
former eager L2 updates. The shared SGD implementation now uses source
WeightVector64 lazy scaling, ordered dot products, scale reset below 1e-9
and final materialization. The saved synthetic fixture
validation/fixtures/regressor_chain_lazy_l2.json is checked against generated
inputs so later fixture edits cannot silently drop it. On that input, the
former code's maximum scaled component error was 4.3424e-8, failing the
original 3e-8 tolerance; lazy scaling reduced it to 7.9580e-9. Existing dense
averaging behavior is retained. This remains original MIT work with retained
sklearn BSD attribution; no dependency or public extension is added.


For SGD chains the suite independently reconstructs native out-of-fold
feature matrices, checks every generated column against source fold fits
and predictions, and compares full fitted coefficients/counters to source
fits on those identical matrices. Full-chain query predictions and scores
still compare against the ordinary sklearn chain. All comparisons retain
3e-8 tolerance. This separates amplification of tiny feature-rounding
changes from a fitting or routing error: the saved fixture's release
coefficient discrepancy against the unconditioned source chain was about
4.0191e-8, while identical-design source fits reproduced native stages
exactly in the diagnostic probe. Rounding witnesses are recorded separately
from the consistently checked conditioned-stage groups; their presence can
differ by optimization profile. The full checked/release native test runs
were completed before adding these reference-driver witnesses; numerical
production sources were unchanged afterward.


### ClassifierChain (2026-11-21)

Added ordered and seeded-random classifier chains with native label,
probability, log-probability and decision-score feature responses, ordered
method preferences, observed-target or out-of-fold training features,
weighted full fits and restored target-column query outputs. Integer CV
uses source-compatible first-appearance StratifiedKFold, including minority
classes smaller than the fold count. Probability CV pads absent fold
classes before selecting the second global class column. Explicit iterable
folds preserve repeated/overlapping training rows and require test partitions.
Single-output chains ignore unused CV specifications. ../docs/API.md describes
base-family capabilities, binary query-response restrictions and scoring.

The 8,665-group reference suite uses 785 fixtures over five seeds and eight
base families. It covers all four response methods and capability fallback,
log-loss/hinge/modified-Huber SGD, averaging, weighted fits and signed query
weights, rare and constant classes, multiclass label chains, shared nonzero
indicator labels, single outputs/queries, refits and input/template copies.
Source CV plans are supplied independently to the reference driver's feature
witnesses. Fitted means/variances, priors, log-feature probabilities,
coefficients, counters, KNN stored arrays and class/order metadata are
checked against source fits on those actual feature matrices.

Tree witnesses check every full and CV-fold node using source stumps and
exhaustive Gini scans, including genuinely distinct equal-optimum candidates,
and reconstruct all native CV and sequential query paths. Complement NB
alternatives require source log-score gaps within a scaled 64-epsilon bound;
this also checks any near-tied labels propagated into later chain features.
Ordinary probabilities/decisions/linear state and CV feature comparisons
retain 3e-7 tolerances, NB moments/log parameters use 3e-8, and score/tree
identity checks use 3e-12. These exceptions are explicitly reported by oracle
and witness rather than widening tolerances or removing difficult inputs.


## Multiclass reductions

Run `python validation/compare_multiclass.py` (optionally `--profile release`
or `--executable PATH`). Five seeds exercise 1,930 fixtures and 27,490 groups
for OVR, OVO and output-code classification across 19 native base
configurations. Cases include binary/multiclass/constant targets, multilabel
indicators, one-column matrices, shared nonzero indicator labels, signed
scoring weights, weighted fitting, singleton queries, constant code columns,
streaming with unused declared classes, skipped OVO pairs, averaged SGD,
modified Huber/hinge/log losses, zero-weight Gaussian NB NaNs and refits.
Template parameters, fitted state, counters, predictions and input arrays
are checked before and after use as well as across refits.

Ordinary fitted responses and model parameters are compared against sklearn
1.6.1. For reduction identities, a copy of the actual source wrapper receives
the observed native head responses while retaining source estimator tags and
response capabilities. Thus source thresholding and combination routines
check native reduction behavior independently of small fitted-head differences.
Tree alternatives require exhaustive optimal split certificates and source
stump cost agreement; binary regression-tree SSE is checked through its
exact Gini equivalence. SVC heads require dual feasibility, KKT residuals and
source objective agreement as well as response comparisons. Label and code
nearest-neighbor boundary alternatives require explicit scaled 64-epsilon
witnesses; reports identify them separately. Ordinary tolerances remain fixed.
Native decoding uses explicit distances, so the identity checks also certify
rare equal-cost choices of source's optimized distance implementation.


## Voting ensembles

Run `python validation/compare_voting.py` (optionally `--profile release`,
`--executable PATH`, or repeatable `--seed N`). Five seeds check 16,965 groups
across 1,200 fixtures with 15 classifier and 10 regressor configurations.
Cases cover single and heterogeneous lists, dropped/unallocated slots,
noncontiguous labels, constant and binary targets, signed voting and score
weights, all-zero hard-voting weights, per-base transforms, flattened/tensor
probabilities, weighted fitting, zero-weight Gaussian NB NaNs, singleton
queries, refits and fitted-template/input preservation.

Native base responses and learned linear/Gaussian/count-NB/KNN/dummy/kernel
ridge states are compared with source. CART predictions additionally require
exhaustive split-cost optimality and agreement with source stump costs; all
equal-cost alternative partitions are counted. Source VotingClassifier and
VotingRegressor copies combine the actual native base responses, independently
checking the native reductions, transformed outputs and weighted scores.
The source proxy retains real estimator types and class metadata. Genuine
label boundaries require recorded scaled 64-epsilon witnesses. Certificates
are distinct from ordinary fixed-tolerance source comparisons. Every template
has its full emitted parameter state and predictions checked before and
after ensemble fitting/refitting.


## Permutation importance

Run `python validation/compare_inspection.py` (optionally `--profile release`,
`--executable PATH` or repeatable `--seed N`). Five seeds check 17,820 groups
across 600 fixtures using an analytic multiple-output scorer and fitted OLS,
Ridge, logistic, Gaussian NB and nearest-neighbor models. Source sklearn is
called with the same fitted model/scoring contract and one-based row identity
labels. Both the scalar callable and dictionary-of-scores source paths are
compared with their native callback counterparts.

Every scorer call is traced: exact feature matrices, original row indices,
scoring weights, metric values and call count. This independently checks all
three sampling branches, cumulative shuffles, shared per-feature seeds,
full-data baselines and callback caching. Cases include integer/fraction
limits, single rows/features/repeats, weighted and signed scoring, constant
columns, NaN features, fractions rounded to empty subsets, NaN/infinite scores
and finite-score subtraction/summary overflow. Raw importances and population
mean/deviation are compared with source; emitted model state, responses and
input arrays are preserved. No fit is performed by the inspection algorithm.


## Partial dependence and ICE

Run `python validation/compare_partial_dependence.py` (optionally
`--profile release`, `--executable PATH` or repeatable `--seed N`). Five seeds
check 13,395 groups over 835 fixtures: analytical multiple-output regression,
OLS, Ridge, logistic probabilities/decisions, Gaussian NB, nearest-neighbor
probabilities and LinearSVC decisions. Source runs use method='brute' and
real estimator tags; traced methods preserve their response-method names.
Every substituted feature matrix and numerical response is checked, with
unchanged background columns compared exactly. Axes, Cartesian order, means,
ICE arrays, fitted responses/parameters and original inputs are also checked.

Cases include all result kinds, reordered and duplicate feature axes,
three-way grids, Cunnane interpolation, the strict unique-count cutoff,
categorical logical/default-int64 selectors, unused categorical indices,
single/two-row backgrounds, signed/negative-total weights, NaN unique and
percentile grids, and NaN/infinite prediction reductions.

Source sklearn 1.6.1's binary response reshape errors are retained as explicit
fixtures. A failure is accepted only for binary responses with two grid points
or two background rows, the known cannot-reshape message, and a complete
source prediction trace. Source-trace weighted averages and individual
responses independently verify the native mathematical curve. The report
records each source error and labels these result groups separately as
source_binary_shape_identity. Ordinary source tolerances remain fixed.


### Weighted CART and standalone ExtraTree estimators

Run `python validation/compare_weighted_trees.py` for the checked profile or
add `--profile release`. `--exe PATH` uses an already built reference driver.
The suite covers 552 seeded fixtures: weighted best/random splitting, default
and subset feature counts, constant features, zero/negative/signed weights,
bootstrap multiplicities, three classification criteria, impurity controls,
leaf cardinality and NaN queries after finite training. It writes profile
reports with every comparison group, installed-source hashes and split witnesses.

Ordinary comparisons retain fixed tolerances (2e-10 relative, 2e-11 absolute;
integer choices and labels are exact). Source incremental criterion updates
and native recomputation can choose different partitions at floating-point
ties. Five checked fixtures use source-conditioned certificates: Python
replays the documented RandomState/xorshift feature and threshold draws,
fits source stumps on each native node's rows, verifies every chosen split
against all admissible visited candidates within an explicit accumulation
bound, and reconstructs predictions from source leaf responses. Feature
importance is independently reconstructed from verified node statistics.
These groups are labeled `source_conditioned_tree_certificate`.

Pure subtrees are collapsed only after proving all descendant responses equal
exactly and impurity is within 1e-13 of zero; the remaining source/native
structure, statistics and responses are compared. Those groups are labeled
`source_pure_subtree_identity`. This handles response-identical splits caused
by impurity roundoff without removing fixtures or widening ordinary comparison
tolerances. The suite does not promise identical topology for every floating
point dataset. Training NaNs and unsupported single-tree options remain out
of scope as documented in the public coverage/API records.


### Bagging ensembles

`python validation/compare_bagging.py` checks BaggingClassifier/Regressor
against sklearn 1.6.1; add `--profile release`, or supply `--executable PATH`
for an already compiled reference driver. Successful fixtures cover default
and custom decision/Extra trees, KNN, Gaussian NB, logistic/ridge
classification, OLS/Ridge/ElasticNet, weighted and indexed fitting, both
sampling axes with/without replacement, tracking/permutation/reservoir
boundaries, nonuniform/zero/signed fitting weights, hard-vote fallback,
warm starts, missing local classes and OOB reductions. An incompatible OOB
class-width fixture separately checks the observed source broadcast error.

Every member's sample/feature arrays are compared with source-generated draws;
tree seeds and local class lists are checked separately. Member training and
query responses, ensemble reductions, weighted scores and OOB results are
compared. Native tree responses that differ because of weighted CART roundoff
ties require the source-conditioned stump/RNG certificate in
`compare_weighted_trees.py`; aggregation is then checked using those verified
members, with distinct oracle labels. Ordinary comparisons keep fixed 3e-8
relative/3e-9 absolute tolerance; integer choices and labels are exact. Source
warm-start sample-property discrepancies are recorded rather than substituting
that incomplete source property for the actual stored member draws. The
reports include group-level oracles, split witnesses and primary-source hashes.

A fresh build is required to audit binary compatibility after public Fortran
type layouts change. An observed fpm 0.12 alpha incremental build reused a test
object predating the tree layout while relinking a newer library; that binary
crashed, whereas freshly compiling the same test passed. This batch preserves
the old cache under ignored validation results and runs full checked/release
builds from freshly generated objects. Source comparisons run against those
fresh libraries/executables.


Bagging OLS and Ridge use a native weighted SVD adapter on the original row
count, matching the source weighted fit path. The observed OLS rank failure
is retained exactly in `fixtures/bagging_ols_rank_deficient.json`; every source
run checks its predictions, coefficients, intercept and numerical rank. The
suite also exercises explicit nonuniform weights for these adapters. Original
ordinary comparison tolerances and the failing fixture are preserved.


### Stacking and leave-one-out RidgeCV

`python validation/compare_stacking.py` validates stacking and the new
RidgeCV cv=0 path; add `--profile release` or use `--executable PATH` with a
profile-compatible reference driver. Per-profile input/output files permit
independent checked and release source runs. The suite exercises response
selection, binary probability-column removal, multiclass decision columns,
full and OOF transforms, weights, dropped bases, explicit split order, prefit
original labels, missing fold probability classes, final estimators, weighted
scores, and singleton/wide/weighted/no-intercept LOO calculations.

The exact weighted multinomial false-convergence design is retained in
fixtures/stacking_logistic_roundoff.json. It failed the requested gradient
tolerance before adding roundoff slack to the native Armijo acceptance test.
Every run compares that design with source and independently checks its
unnormalized weighted gradient. Ordinary tolerances remain fixed at 3e-8
relative and 3e-9 absolute; integer choices/labels and split arrays are exact.

Fold and full tree states use the existing source-conditioned CART/RNG oracle.
LinearSVC optimizer differences require an independent squared-hinge objective
and gradient check, plus a score-error bound from L2 strong convexity. Logistic
optimizer differences require unprojected stationarity/objective checks and a
positive segment-Hessian lower bound; multinomial parameter distances are
measured in the zero-mean class gauge, with the original gauge difference
retained in score bounds. Source optimizer warnings are recorded in reports.
The certificates verify the native solution rather than loosening ordinary
comparisons to match a less accurate source stopping point.

The final source estimator is fitted independently on the verified emitted
native OOF design. Coefficients, intercepts, LOO scores and selected alpha are
labeled source_on_verified_meta; source default Newton counterparts use the
native algorithm's strict optimization settings. Final predictions, transforms
and scores are also compared with the complete source stacking model whenever
no certified member/optimizer alternative is involved. Reports retain all
group-level oracles, split/optimizer witnesses and primary-source hashes.


## TargetEncoder reference checks

`python validation/compare_target_encoding.py` resolves the checked fpm
reference driver; add `--profile release` for the optimized build or supply
`--executable` for an existing matching binary. Reports and per-profile
input/output files are written under ignored `validation/results`.

The suite compares full-data statistics, cross-fitted training features,
query encodings, refit results, parameter snapshots, feature widths, sorted
categories/classes and exact fold assignments against installed sklearn.
Seeds 0, 7, 42, 2026 and 314159 cover continuous, binary and multiclass targets,
automatic and fixed smoothing (including zero and very large values), rare and
single classes, NaNs, singleton/constant categories, explicit unused/omitted
categories, and shuffled/unshuffled cross-fitting. Exact category and index
checks are separate from numeric comparisons (rtol 2e-12, atol 2e-13).
No alternative numerical optimizer certificates are used. The source hashes
include `_target_encoder.py`, `_target_encoder_fast.pyx`, `_encoders.py` and
`model_selection/_split.py`; source warnings remain in the report.

The explicit omitted-category fixtures exercise sklearn 1.6.1's first-category
fitting and final-fold unknown-mask behavior; the implementation matches those
observable source results, as documented in API.md. Large integral-looking
floating targets check NumPy int64-range inference, and forced binary fits
check fractional numeric label handling separately from cross-fitting.


## OrdinalEncoder source validation

Run `python validation/compare_ordinal_encoding.py` for the checked fpm driver,
or add `--profile release`. `--executable` selects an already built matching
reference driver, allowing source comparisons to overlap a different profile's
build without concurrent fpm processes. Per-profile reports and input/output
files are retained under ignored validation/results.

The five-seed suite compares training/query codes, sorted categories, category
counts, infrequent categories and their sizes, learned width, fitted grouping
snapshots, inverse numeric values and None/infrequent tags, input preservation,
and refitting to a new width after grouping. All comparisons use exact array
equality, treating NaNs at matching positions as equal; no optimizer or wider
tolerance certificates are needed. Recipes cover explicit unused and omitted
categories, automatic NaNs, entirely missing features, count/fraction thresholds,
max-category limits and count ties, error/encoded-value unknown handling,
sentinel collisions, negative/fractional inverse indices and invalid indices.

sklearn 1.6.1 grouped/missing inverse errors are retained as source_errors in
the report. Such fixtures first compare all normal learned/forward fields,
then require the native driver to fail with an explicit inverse-index diagnostic
instead of a runtime bounds error. Successful inverses compare the exact numeric
result and the source None/infrequent tags. Source inverse receives a copy of
its input; the native driver additionally verifies that its own input is
preserved. Source hashes cover preprocessing/_encoders.py and utils/_encode.py.


## GraphicalLassoCV reference checks

`python validation/compare_graphical_cv.py` resolves the checked fpm driver;
add `--profile release` or supply `--executable` for a built matching driver.
Per-profile input/output files and reports are retained in validation/results.
Five seeds cover refined and explicit grids, duplicates and unsorted alphas,
centered/uncentered final fits, capped and zero iterations, custom folds with
ordering/repeats/overlap, a single fold, all-candidate failures, n_samples=2,
exactly diagonal automatic grids, scaling and more features than training rows.
The driver emits complete CV arrays, exact fold indices, endpoint matrices,
final covariance/precision/location/history and prediction scores/distances.

Ordinary numeric tolerance is rtol=3e-8 and atol=3e-8, matching the existing
standalone GraphicalLasso comparison. Fold indices, feature/iteration counts
and convergence flags are exact. Source warnings are retained; NaN/negative
infinity results are compared explicitly. Reports record hashes for
_graph_lasso.py, _empirical_covariance.py, _split.py and _cd_fast.pyx.

Endpoint conditioning checks apply only when the appended empirical score or
precision fails ordinary comparison. Covariance/test moments must agree with
source at rtol=3e-13 and atol=3e-14. For a full-rank empirical covariance, both
inverses must be backward stable, and their difference must satisfy the inverse
perturbation identity including the two residual terms. The endpoint likelihood
then uses sklearn on the verified native precision and test moment. For rank
deficiency, all four Moore-Penrose equations and the source precision comparison
must pass and numerical rank must agree. A partial-pivot growth/Weyl singular
value bound limits the log determinant when roundoff changes its sign near zero.
The finite/-infinity endpoint score and its recomputed aggregates are retained
with a source_conditioned_empirical_endpoint oracle and detailed certificate.
No positive-alpha score, selected alpha or final model is replaced by such a
certificate. Nonfinite certificate scalars use strings in strict JSON reports.

The existing compare_graphical_lasso.py remains a regression gate for the shared
CD kernel. The sole change to that kernel is source's zero-Gram-diagonal skip,
needed for constant/one-row CV folds. Native tests independently cover inverse
consistency, final standalone equivalence, refined-grid shape, exact ties,
refitting/copying, zero-iteration conventions and restoration of IEEE halting
modes and flags after failed/NaN-grid paths.

## LogisticRegressionCV increment (2026-11-21)

Added `logistic_regression_cv`, a dense L2 binary/multinomial counterpart with
warm-started candidate paths, stratified or explicit folds, global class weights,
weighted scoring, first-best selection, and refit/averaging modes. The native
full Newton solver retains the original plain logistic fit behavior. Existing
logistic-capable wrappers and nested cross-validation preserve the new dynamic
type. The API documents supported scorers, tensor layout, finite-intercept
training requirements and native iteration-count differences.

`compare_logistic_cv.py` uses five seeds and 255 fixtures, retaining 7,560 groups
per profile. It compares installed sklearn 1.6.1 newton-cg (tol 1e-12,
max_iter 2000) against native full Newton (tol 1e-10, max_iter 1000), with zero
iterations explicitly tested in both. Ordinary floating comparisons use
rtol=3e-7, atol=3e-8; labels and fold indices are exact. Source multinomial
intercepts and decisions are canonicalized by removing their common intercept
shift, matching the native solver's existing centered representation. Slopes
and all binary results are unchanged. The source shift is retained in reports.
A seed-0 unsorted path exposed a common source shift of about 6.12e-6 at C=1e4;
both source and native gradients were already small. No tolerance was widened.

Independent checks verify raw weighted L2 stationarity for every converged
fold/candidate and final refit (maximum gradient 1e-7), exact zero-iteration
parameters, iteration bounds and selection consistency. Source C selections,
paths, predictions, weighted/unweighted scores, final model parameters,
class weights and refit initialization are compared independently. Fixtures
include repeated/unsorted grids, automatic grids, globally balanced and explicit
class weights, zero weights, overlap/repeated explicit indices, single folds,
refit=false, binary one-class training without an intercept, and NaN ROC scores.

`compare_logistic_cv_composites.py` adds five seeds through nested CV, voting,
bagging, stacking, OVR, independent multioutput, sigmoid calibration, and
self-training. It uses the same ordinary tolerances, exact predictions and
label-state checks, and records hashes of the consulted source implementations.
Reference reports and raw I/O are profile-specific under validation/results.
The project remains MIT with adapted sklearn BSD-3-Clause attribution retained;
no external runtime dependency or public extension is introduced.

Checked gates passed all 104 native programs with `-fcheck=all -fbacktrace
-ffpe-trap=invalid,zero,overflow`. The full isolated rejection run passed 2,447
unique cases, including 43 LogisticRegressionCV controls and three optional
plain-logistic score-weight cases. Its main program was linked without FPE
traps so nonfinite-input diagnostics could be checked. Existing 192-group basic
and 16,536-group weighted/streaming classifier source suites also passed.

Current inventory: 196 core estimator/transformer/splitter types, 102 source
modules, 104 native programs, and 83 core reference suites retaining 347,125
result groups, plus the separately counted 28 RNG checks. Older source suites
outside the two regression gates were not rerun for this batch. The new public
example agrees with sklearn on C=10 and training accuracy 1.0 for its synthetic
fixture; these values are not a general performance guarantee.

Both profiles passed all 7,650 new groups: the core suite contains 6,795 sklearn
comparisons, 255 iteration-bound checks, 245 stationarity checks, ten exact
zero-iteration identities and 255 selection identities; the composite suite
adds 90 sklearn comparisons. Core reports retain 124 source warnings, while
composite reports have none. Checked/optimized fixture/check-oracle records,
source hashes and warnings agree exactly, as do the recorded source intercept
canonicalizations. All ordinary comparisons retained their original tolerances.

Final optimized gates passed all 104 native programs and the public
`fpm run --profile release --example logistic_cv` example. Both examples
selected C=10 and reported training accuracy 1.0. Checked and optimized builds
used GNU Fortran 17 on Windows. Final source/diagnostic reports, source hashes,
Python syntax, Fortran line lengths/whitespace and staged-file scope were
verified. Production Fortran stayed fixed throughout the final gates.

## Binary threshold wrappers and unified CART (2026-11-21)

Added the FixedThresholdClassifier and TunedThresholdClassifierCV counterparts,
with probability/margin orientation, equality-at-cutoff semantics, weighted
built-in objectives and custom callbacks, independent fitting/scoring weights,
stratified CV/holdout, explicit folds, prefit validation and refit modes.
Learned fold curves and mean interpolation/selection are optionally retained.
The public example uses training-fold tuning and a separate held-out slice.

The first expanded sweep retained an unweighted tree discrepancy at seed 0,
case 75: the former real64/native RNG backend chose a different tree. The
existing sklearn-derived weighted kernel with unit weights matched the failing
fold's source probabilities exactly. Omitted-weight CART now uses that kernel,
and the unused old backend was removed. This also changes the underlying trees
of unweighted ensembles. Source float32 geometry, feature permutation and NaN
query routing now apply consistently. Native tests verify omitted/unit-weight
identity and NaN routing; XOR remains supported.

`compare_threshold.py` checks five seeds, seven base families and 455 fixtures:
logistic regression, Gaussian NB, ridge classification, KNN, CART, Dummy and
nested LogisticRegressionCV. It exercises probability/margin/auto responses,
positive class reversal, grids of 2/11/31/100 or repeated explicit candidates,
2/3/5-fold stratification, shuffled holdout, repeated/overlapping explicit folds,
single-fold refit=false, prefit, independent/zero score weights, distinct simultaneous training/validation weights, callbacks,
constant responses on explicit grids, infinite fixed cutoffs, copies/refits and
optional curve storage. Normal floating comparisons use rtol=3e-7, atol=3e-8;
labels and row indices are exact. Raw fold responses are replayed independently.

The checked run passed 10,535 groups: 7,838 direct sklearn comparisons, 2,275
input/metadata identities, 420 metric replay identities and two conditioned
curve groups. Source warnings (87) and source hashes are retained in strict
JSON. A single retained weighted CART fixture (seed 2026, case 73, prefit fold)
places 25 responses at a cutoff of 0.8: sklearn emits 0.8 while native emits
0.7999999999999999, despite identical training/query leaf partitions.

Only affected fold/mean curve groups may use the source_conditioned_tree_boundary
oracle. The certificate requires equal leaf membership for every crossing,
100-digit Decimal sums of original binary64 training weights, both probabilities
within a standard positive-summation/division forward-error bound, and tightly
bounded response/grid differences. Crossing points must lie within those bounds
of the cutoff. sklearn metrics replay the native response classification; other
curve entries remain source values, and NumPy interpolation reconstructs the
mean. Native curves independently match metric replay to 1e-12. Selected grid
index, best score and final predictions retain ordinary source comparisons;
a certificate cannot change the selected index or its score. Exact decimal
moments, training rows, crossings and bounds are retained in reports. Ordinary
tolerances were not widened and no fixture was dropped.

The expanded existing tree suite passed 9,308 groups across 720 fixtures, an
increase of 2,184 groups and 168 omitted-weight fixtures without changing the
prior RNG recipes. It retains 9,156 direct groups, 140 pure-subtree identities
and 12 conditioned groups across five existing tree certificates. The original
192-group source suite and 510-group GradientBoostingClassifier suite passed.
All 105 native programs passed with runtime checks and invalid/zero/overflow
traps. A separate threshold driver passed 58 isolated diagnostic cases; the
existing 2,447-case rejection suite also passed, for 2,505 total cases. Python validation can
use an explicit executable to avoid concurrent fpm builds.

Current inventory: 198 core types, 103 source modules, 105 native programs and
84 core reference suites with 359,844 retained groups, plus separate RNG checks.
Original contributions remain MIT with retained sklearn BSD attribution. No
public algorithm beyond sklearn or external runtime dependency was introduced.

Run `python validation/compare_threshold.py` for checked reference validation,
or add `--profile release`. `--executable` uses a matching existing driver.
Run `python validation/check_threshold_invalid_inputs.py` for the 58 threshold
rejections in addition to `check_invalid_inputs.py` for the 2,447 existing cases.
The rejection driver is linked without floating-point traps to inspect
nonfinite-input diagnostics; native valid-input tests keep those traps enabled.

Final checked/optimized gates passed all 105 native programs, both fpm examples,
10,535 threshold groups and the 9,308-group expanded tree suite per profile.
Threshold/tree fixture, oracle, warning, certificate and source-hash records
match across profiles. Source hashes, strict JSON, 2,505 unique diagnostics,
Python syntax and Fortran formatting were audited. The checked object for the
late-added CART invariant assertions was initially stale; a focused fpm rebuild
and rerun passed, and object timestamps now confirm both profiles include those
assertions. Production Fortran stayed fixed through the final gates. Earlier
source suites other than the stated regression gates were not all rerun.

The public example matches an independent sklearn fit: threshold
0.252489573748904, CV balanced accuracy 0.8793217576434813, and held-out accuracy
0.933333 for both default and tuned prediction. These synthetic-fixture values
are not a general performance guarantee. Validation used GNU Fortran 17 on
Windows and installed sklearn 1.6.1.

## Random tree embeddings and kernel centering (2026-11-21)

Added RandomTreesEmbedding and KernelCenterer counterparts, with native tests,
source comparison drivers, a combined classifier/kernel example and retained
BSD attribution in the MIT project. Embeddings support dense leaf encodings,
weighted randomized trees, exact scalar-seeded NumPy random targets/tree seeds,
warm growth and no-growth encoder refits, feature names, sample metadata,
apply and dense decision paths. Centering supports arbitrary square kernels,
rectangular query kernels, copies/refits and source-order arithmetic, including
quiet nonfinite outputs from finite inputs that overflow intermediate operations.

A near-constant float32 fixture exposed a source/binary discrepancy. The local
and release-tag Cython sources declare FEATURE_THRESHOLD=1e-7, but the installed
1.6.1 extension exports 0.0. Read-only Cython capsule inspection in fresh Python
processes and two-point DecisionTree/ExtraTree fits both confirmed zero runtime
tolerance. Native CART/ExtraTree grouping now keeps distinct float32 values
distinct, matching this installed reference. `sklearn_tree_runtime.py` records
the capsule ABI/value and SHA-256 of four loaded tree binaries; it does not
modify the installation. Reports retain source hashes as well. This is an
observation about the tested installation, not a claim about every 1.6.1 build.

The same source sweep found sklearn 1.6.1 decision_path sends NaNs right while
apply uses learned missing routing. The new path method reproduces that
distinction rather than reconstructing paths from apply's leaf IDs. No source
fixture was removed and structural comparisons remain exact.

Checked new reference gates passed 2,175 embedding groups across 115 fixtures
and 145 fits, and 6,240 kernel-centering groups across 520 fixtures. Five seeds
cover staged growth, changed seeds/data/controls, no-growth encoder changes,
non-warm width resets, signed/zero/tiny/large weights, all-constant features,
near-constant float32 data, unbounded depth, a single sample, 100 trees and
130 input features. Embedding matrices, leaf IDs, paths, seeds, names and all
structural metadata are exact; importances use rtol=3e-10/atol=3e-12.
Uniform random targets exactly match the NumPy protocol.

Kernel fixtures include linear/RBF/nonsymmetric kernels, large offsets,
cancellation, constants, signed zero and overflow. Sizes around 8 and 128
exercise NumPy's eight-accumulator/recursive pairwise boundaries. Reference
arrays use C layout; ordinary numeric tolerance is rtol=3e-13/atol=3e-13,
with explicit NaN/Inf comparison and exact finite-zero sign bits. The pinned
NumPy 2.2.6 summation source URL/hash is retained; validation needs no network.
All new groups use direct sklearn/NumPy or input identities, with no conditional
replacement oracle or widened tolerance.

The expanded tree regression suite passed 9,932 groups across 768 fixtures,
including 48 new below-1e-7 geometry fixtures (624 added groups). Existing
threshold, 192-group basic and 510-group GradientBoostingClassifier gates are
also run against the corrected kernel. The new isolated rejection driver
passed 44 cases. Full native and diagnostic profile gates follow below.

Current inventory: 200 core types, 105 source modules, 107 native programs and
86 core reference suites retaining 368,883 groups, plus separate RNG checks.
The three rejection registries contain 2,549 cases. Earlier source suites
outside the stated regression gates are retained but are not all rerun here.
No public algorithm outside sklearn or external runtime dependency was added.

Final checked and optimized gates passed all 107 native programs and the public
`random_feature_maps` fpm example. New suites passed 2,175 embedding and 6,240
centering groups per profile; expanded tree and existing threshold suites
passed 9,932 and 10,535 groups respectively. Checked/optimized fixture, oracle,
warning, source hash, binary provenance and certificate records agree.
The new suites have no conditional groups: embedding has 2,030 sklearn groups
and 145 exact NumPy RNG groups; centering has 5,720 sklearn groups and 520 input
identities. Ten source no-growth warnings and 470 expected centering arithmetic
warnings are retained. The older tree/threshold conditional checks keep their
existing recorded scope.

All 2,549 unique isolated invalid-input cases passed against the updated
checked library. Final audits verified source/binary hashes, strict JSON,
Python syntax, Fortran formatting, module/test inventory and fresh test objects
for both profiles. Production Fortran remained fixed through these gates.
The example matches independent sklearn output: 39 leaf features, held-out
accuracy 0.95, centered-kernel column residual 9.325873406851315e-15 and a
40-by-80 query kernel. These synthetic-fixture values are not a general
performance guarantee. Validation used GNU Fortran 17 on Windows and the
installed sklearn 1.6.1 / NumPy 2.2.6 baseline.

Run `python validation/compare_tree_embedding.py` and
`python validation/compare_kernel_centering.py`; add `--profile release` for
optimized checks or `--executable` for an existing matching driver. Source
reports/inputs/outputs are profile-specific. `check_feature_maps_invalid_inputs.py`
adds 44 rejections to the existing 2,447 and threshold 58 cases. Rejection mains
omit floating-point traps to inspect nonfinite-input diagnostics; valid-input
native tests use invalid/divide-by-zero/overflow traps. Source and binary
provenance is read-only, and future validation does not fetch network data.


## Neighbor graph transformer increment (2026-11-21)

Added KNeighborsTransformer and RadiusNeighborsTransformer with original MIT CSR
storage helpers. This raises the core inventory to 202 types in 107 source
modules, with 108 native test programs. There are 87 core reference suites with
374,433 retained result groups, plus the separate existing NumPy RNG checks.
Four invalid-input registries now contain 2,591 unique cases; this increment
adds and reruns 42 cases rather than rerunning the older three registries.

`validation/compare_neighbor_graph.py` retains 5,550 groups across 1,110 fixtures
and five seeds: 4,440 source-based shape/nnz, pointer, edge-value and neighbor-
distance groups plus 1,110 caller-input identity groups. It exercises eight
metrics, both modes, explicit/implicit training queries, held-out queries,
integer/duplicate/zero-vector inputs, unsorted sparse precomputed rows, missing
edges, repeated columns and stored zero distances. Thirty additional boundary
fixtures cover zero/infinite radius, empty held-out graphs and near-limit k. Both matrix structure and
stored slot values are checked for CSR inputs, without coalescing duplicates.
Expected unsorted-sparse warnings are captured (120 per profile).

Dense stable tie selection differs from sklearn's optimized heap/partition
selection in 245 retained fixtures. Each changed support must have exactly the
same source-computed distance multiset, and each changed position must be an
exact distance tie. Certificates record source/native indices by row. Membership
and ordering certificates use exact equality: the 2e-12 numerical edge-distance
tolerance cannot admit a different-distance neighbor. Radius connectivity
order and sparse precomputed indices/data are checked directly. These are
specified subset differences, not claims of exact backend tie parity.

Run `python validation/compare_neighbor_graph.py --profile checked` or
`--profile release`; `--executable PATH` bypasses fpm for an already-built driver.
`validation/check_neighbor_graph_invalid_inputs.py` accepts the same options.
Its 42 isolated subprocess checks cover bad modes/metrics/powers/counts/radii,
NaN/Inf inputs, empty inputs, nonsquare/negative precomputed distances, too few
neighbors, malformed CSR pointers/columns and unsupported sparse features.
JSON reports in ignored `validation/results` record profile, source/native
hashes, versions, check results, warnings and exact-tie certificates.

`test_neighbor_graph` independently checks k versus k+1 counts, stored zeros,
self inclusion/exclusion, inclusive and zero-radius boundaries, copy/refit
state, feature names, duplicate-preserving precomputed graphs, input identity,
empty radius rows, dense conversion and CSR multiplication. The public example
`neighbor_graphs` constructs both graphs and aggregates values through CSR.
Only dense features and dense/CSR precomputed neighbor distances are supported;
this increment does not add general sparse inputs to existing estimators.


Final gates: all 108 native programs passed both checked runtime/FPE-trap and
optimized fpm builds. Both profiles passed all 5,550 new reference groups and
42 isolated rejections, with identical 245-fixture exact-tie certificates.
The existing 192-group sklearn integration comparison also passed against the
updated checked executable. The public example passed both profiles and matched
independent sklearn: 15 stored distance edges and radius-one neighbor sums
[3, 6, 5, 4, 5]. Production Fortran stayed fixed through these final gates.

The optimized reference executable was linked from the current fpm release
object and archive into a temporary directory while the remaining fpm targets
built. Its .text section SHA256 is identical to the final fpm driver
(b2b1827c8a72681de72041dce5bfab15148ccd1b5c072bb710cff7ae33dae974).
Source hashes, report binary hashes, strict JSON, both profiles' 108 fresh native
test objects, Python syntax, Fortran line lengths/whitespace and staged Git
whitespace checks passed. No older reference-suite rerun is implied beyond the
192-group integration gate. Original MIT licensing and sklearn BSD attribution
remain in place; no dependency or public extension algorithm was added.


## Iterative multivariate imputation increment (2026-11-21)

Added the experimental sklearn IterativeImputer counterpart using native
Bayesian ridge chained equations, including posterior sampling. The inventory
is 203 core types, 108 source modules, 109 native programs, and 88 core reference
suites retaining 377,025 groups, plus the separate existing RNG checks.
The five rejection registries retain 2,625 unique cases; this increment adds
and runs 34 cases without claiming a rerun of the older registries.

`validation/compare_iterative_impute.py` retains 2,592 groups across 324 fixtures:
2,240 direct sklearn groups, 324 exact input/observed-value identities and
28 explicitly conditional source-replay groups across four fixtures. The 300
seeded fixtures use five seeds, NaN and finite markers, all five update orders,
all four initial strategies, posterior/deterministic fits, optional indicators,
empty columns kept/dropped, newly missing complete-at-fit features, scalar and
vector bounds, finite/infinite limits, correlation-weighted neighborhoods,
zero-variance matrices, early exits and repeated stochastic transforms.
Twenty-four additional fixtures have only one to three rows with seven to
12 features, covering wide/underdetermined fits as well as rank deficiency.

Each output/query/repeated-query/statistics group is compared at rtol=atol=3e-11;
metadata, names, feature indices and every stored update/neighbor sequence are
exact. Native drivers return their unchanged input arrays, so input identity
is checked after actual Fortran calls, including NaN masks. Python reference
calls receive copies because sklearn 1.6.1 can mutate an F-contiguous input
when retaining empty features (one-row arrays are both C/F contiguous).

The four posterior conditional fixtures retain sklearn's direct-default
output deltas and 126 independent covariance certificates. The native
Bayesian ridge baseline already documents nonunique thin-SVD covariance in
wide rank-deficient problems. Certificates require covariance symmetry,
correct eigenvalues and unique action on the observed centered row space;
these establish a legitimate retained null-space completion. A validation-only
BayesianRidge subclass runs sklearn fitting on the replay's CURRENT predictors
and targets, checks each native fitted coefficient/precision against that fit,
and substitutes the certified covariance only in that mathematically nonunique
case. All subsequent source imputation, clipping, truncnorm draws, RNG advances,
ordering and transforms run normally. No installed reference code is patched.
The conditional groups are explicitly distinguished from direct-default parity;
original discrepancies can exceed one and are not accepted by the 3e-11 test.

Source warnings (including nonconvergence and dropped empty columns), source
hashes, native binary hashes, profile/version metadata, maximum numerical
errors and covariance certificates are stored in ignored validation/results.
`python validation/compare_iterative_impute.py --profile checked` or `--profile
release` builds the driver; `--executable PATH` bypasses fpm. The isolated
rejection script has the same options. Its 34 cases exercise unfitted use,
empty/mismatched arrays and names, bad controls/sentinels, infinities, scalar
and vector bounds, integer sequence capacity and invalid estimator settings.

The native test independently checks correlated-feature recovery, unchanged
observed values, indicator masks, stored sequence replay, deep-copy/refit
isolation, the source constant-fill naming quirk, empty-column retention,
finite markers, bounded posterior draws, persistent RNG state and newly
missing complete-at-fit features with skip_complete. The public
`chained_imputation` example demonstrates recovery and indicators. The code
retains MIT/BSD attribution and adds no dependency or public extension algorithm.


Final gates passed all 109 native programs under checked runtime/FPE-trap flags
and optimized fpm builds. Both profiles passed all 2,592 groups with the same
oracle classification (2,240 direct, 324 identity, 28 certified replay), the
same four conditional cases, and 126 covariance certificates per profile.
The 34 isolated rejections passed both profiles. The existing 192-group sklearn
integration gate passed against the updated checked executable; no broader
older-suite rerun is implied. Largest output errors against the stated oracle
were 5.01e-14 checked and 4.90e-14 optimized. Certificates use 3e-9 relative/
absolute tolerance for spectral/row-space and fitted-parameter checks, and
3e-11 for covariance symmetry; output comparisons remain 3e-11.

The public example passed both profiles and matched independent sklearn: three
rounds, six output columns including indicators, and recovered values
1.9999994241590575, 8.000000361057754, 17.9999999536776. These are synthetic
fixture results, not general recovery guarantees. Production Fortran and the
native test stayed fixed through the final gates. Validation driver diagnostic
output was expanded afterward, and the final checked build refreshed both
application drivers without changing the tested library or native tests.

Reference drivers were linked against current fpm libraries in temporary
directories during the builds. Their executable .text hashes match the final
fpm executables exactly: checked
c4c93bbaa4c216e44abbdc44b539f1c29e60809acf6862e22a57e42836687820,
optimized e536ee221fc7a288d96c35ee86280ce5f331d2a3ecf39401788c02e4654c6cb1.
Final audits verified upstream source hashes, tested binary hashes, strict JSON,
source/Python formatting, fresh production and 109 native test objects in each
profile, and staged Git whitespace. The active broader translation goal remains
open; this increment is one additional core sklearn algorithm.


## Numeric histogram gradient boosting (2026-11-21)

Added HistGradientBoostingRegressor and HistGradientBoostingClassifier, exposed
through sklearn_ensemble and the sklearn facade. Numeric binning and best-first
Newton trees support missing values, sample weights, monotonic constraints,
warm starts and staged outputs. Regression losses are squared_error,
absolute_error, quantile, poisson and gamma; classification uses log_loss.
Early stopping optionally monitors training loss. Categorical features,
validation holdouts, feature subsampling and composite dispatch remain outside
this increment; see API.md and COVERAGE.md for the precise subset.

Both checked runtime/FPE-trap and optimized fpm builds passed all 110 native
programs. The new direct sklearn comparisons pass 1,452 groups across 242 tree
fixtures and 5,234 groups across 487 complete boosting fixtures per profile.
These include exact bins, thresholds, node metadata, staged outputs, weights,
missing routing, warm starts, monotonicity, unsigned binning seeds and two
200,017-row subsampling fixtures. Additional fixtures cover valid maximum
integer min_samples_leaf, constant 1e308 median/quantile baselines and stable
150-round binary loss tails with zero stopping tolerance. The latter checks
loss and probability tails with zero absolute tolerance. No conditional replay
or substitute oracle is used for these new comparisons.

All 69 new isolated invalid-input checks and 42 separate NumPy RandomState
stream groups pass in both profiles. RNG streams include full unsigned 32-bit
seeds; the existing default-integer seed binding is preserved. The existing
192-group sklearn integration gate passed against the updated checked build;
this does not claim a rerun of every historical reference suite.

The public histogram_boosting example passes both profiles and matches an
independent sklearn run: synthetic training R2 0.9941744173693174 and accuracy
1.0. These are fixture results, not general performance claims.

Inventory is now 205 core public types, 111 source modules, 110 native test
programs and 90 reference suites retaining 383,711 core result groups. The
separate RNG stream groups are excluded from that core count. Six invalid-input
registries retain 2,694 cases. Original code remains MIT; adapted sklearn and
NumPy contracts retain their BSD attribution in THIRD_PARTY_NOTICES.md.

Reproduce this increment with compare_histogram_tree.py,
compare_hist_boosting.py, check_hist_boosting_invalid_inputs.py and
compare_legacy_rng.py. Each accepts --profile checked or --profile release;
--executable supplies a prebuilt matching driver and bypasses fpm. Full boosting
reports have cases_filter=null; --cases is only for targeted debugging.
Reports in validation/results include binary hashes and, for source comparison
suites, upstream source hashes. Numerical tolerances are 3e-11 for tree results
and 3e-10 for boosting outputs, with discrete structure compared exactly.


## Image patch algorithms (2026-11-21)

Added PatchExtractor, extract_patches_2d and reconstruct_from_patches_2d in
sklearn_image and the sklearn facade. Stateless dense real64 extraction covers
grayscale and multichannel batches, default or explicit sizes, exhaustive
width-first ordering, integer caps and fractional counts, and NumPy-compatible
sampling with replacement. Each transform resets the configured integer seed;
images within a batch share the stream. Rank-specific Fortran overloads retain
a singleton channel dimension for color inputs. Native controls and other API
limits are documented in API.md and COVERAGE.md.

Reference validation uses the installed sklearn 1.6.1 image.py directly. There
are 784 fixtures and 3,136 groups per profile: exact shapes, exact patch values,
repeated calls, and reconstructed images. Four seeds include 2147483647; cases
cover one-pixel images, full-image patches, three-channel data, zero fractional
counts, integer limits above all possible positions, auto sizes, partial/empty
reconstruction inputs and overflow/cancellation at 1e308. Reconstruction keeps
source's denominator even when patch dimensions exceed half the image; it is
not replaced with an identity-preserving overlap count. Quiet IEEE primitives
preserve source infinities/NaNs with FPE traps enabled. Reconstruction comparisons
use 2e-14 relative/absolute tolerance; all extraction comparisons are exact.

Inventory after this increment is 206 core public types, 112 source modules,
111 native test programs and 91 reference suites retaining 386,847 core groups.
The separate 42 NumPy RNG groups remain excluded. The new 24 isolated rejection
cases bring seven invalid-input registries to 2,718 retained cases. No additional
runtime dependency or public extension algorithm was introduced. Original code
is MIT, with sklearn's BSD attribution retained in THIRD_PARTY_NOTICES.md.

Run `python validation/compare_image.py --profile checked` and
`python validation/check_image_invalid_inputs.py --profile checked`, and repeat
with `--profile release`. Both scripts accept `--executable PATH` to bypass fpm
and use an already built matching driver. The resulting image_PROFILE.json and
image_invalid_PROFILE.json record tested executable hashes; reference reports
also retain the upstream image.py hash. Avoid concurrent fpm invocations.


Final gates passed all 111 native programs in both the checked runtime/FPE-trap
and optimized fpm builds. Both profiles passed all 3,136 image result groups
across the same 784 fixtures and all 24 isolated rejections. Extraction was
exact and maximum measured reconstruction error was zero against sklearn in
both profiles. Expected NumPy finite-check reduction warnings on large finite
pixels are captured in the reports. The existing 192-group sklearn integration
gate also passed against the updated checked executable; no rerun of all
historical reference suites is claimed.

The image_patches example passes both profiles and independently matches
sklearn: 25 patches and maximum reconstruction error 1.1102230246251565e-16 on
its small synthetic image. Final audits verified upstream source and native
executable hashes, strict JSON, identical checked/release fixture groups, and
fresh objects for all 112 source modules and 111 native test programs in each
profile. Fortran formatting, Python compilation and staged Git whitespace
checks passed. Production code and native tests remained fixed during these
final gates. The broader translation goal remains active.


## Neighborhood components analysis (2026-11-21)

Added NeighborhoodComponentsAnalysis with the source leave-one-out stochastic
neighbor objective, analytic gradient and L-BFGS-B fitting. Supported controls
include identity/random/PCA/LDA/auto initialization, explicit matrices, output
dimension, stopping tolerances, max iterations, integer seeds, warm starts and
read-only iteration callbacks. Methods include fit, transform and fit_transform.
The source callback counter, fitted objective/gradient, evaluation counts and
optimizer/line-search statuses are exposed. Dense working-range and rank
limitations are documented in API.md and COVERAGE.md.

The numerical kernel is an unmodified vendored local fpm dependency from
Jacob Williams lbfgsb 1.0.0 (commit bcb0ddc71ae1afdf5715ededaf9453aa704c2384),
found in SciFort's dependency cache and independently compared with upstream.
The four Fortran sources, license and upstream manifest match after LF
normalization. Full provenance and BSD-3-Clause terms are retained in
vendor/lbfgsb; original adapter code remains MIT. The dependency uses its own
Fortran BLAS/LINPACK helpers, with no external BLAS binary or network build
requirement. No public extension ML algorithm is introduced.

The reference suite has 276 fixtures and 2,100 groups: 180 complete fits with
all six initialization choices (including matrices), plus 96 independent
source objective/gradient fixtures. It checks source initialization metrics,
exact seeded random matrices, transforms, callback metadata and 18 warm-start
prefixes. Every fitted native final objective/gradient is checked directly
against sklearn's formula. Every accepted step is checked for descent and
Armijo decrease plus strong Wolfe curvature, or the optimizer's explicitly
recorded bracket-warning criterion. Converged fits require the source gradient
or relative-function stopping test; capped fits must reach max_iter. SVD basis
freedom is checked through initialization A.T*A, then source fitting is replayed
from that checked native matrix. Direct fitted matrices/outputs are retained
where they agree within 3e-7 relative/absolute tolerances; every divergent fit is listed separately and its
source transforms at native parameters and optimizer trajectory are certified.
Divergent matrices are not counted as direct sklearn fitted-output matches.
The reports retain upstream sklearn/SciPy source hashes, tested binary hashes,
optimizer provenance, fit comparisons, stopping messages and per-step evidence.

There are now 207 core public types, 113 core source modules, 112 native test
programs and 92 reference suites retaining 388,947 core result groups. Four
vendored optimizer modules are excluded from core module/type counts; the
separate 42 NumPy RNG checks remain excluded from core reference groups.
The 40 new isolated rejection cases bring eight registries to 2,758 retained
cases. The public neighborhood_components example uses only the sklearn
facade to learn a two-dimensional embedding of a synthetic 24-row dataset.

Run `python validation/compare_nca.py --profile checked` and
`python validation/check_nca_invalid_inputs.py --profile checked`, then repeat
with `--profile release`. Both accept --executable PATH to bypass fpm and use
a prebuilt matching driver. The reference script's --cases filter is for
investigation only; full reports have cases_filter=null. Vendor digests are
verified before reference comparisons. Reproduction records both direct fit
agreement and trajectory certificates instead of hiding divergent matrices.


Final gates passed all 112 native programs in clean checked runtime/FPE-trap
and optimized fpm builds. Both profiles passed the full unfiltered 276-fixture,
2,100-group reference suite and all 40 isolated rejection checks against their
final fpm executables. The profile-specific optimization evidence is:

checked: 165 direct fitted-output fixtures, 15 trajectory-certified fixtures, 3148 accepted-step certificates, 6 certified bracket-warning steps; maximum source gradient error at native fitted parameters 1.11e-10.

release: 166 direct fitted-output fixtures, 14 trajectory-certified fixtures, 3152 accepted-step certificates, 7 certified bracket-warning steps; maximum source gradient error at native fitted parameters 3.02e-10.

The 96 standalone objective/gradient fixtures and 18 warm-start prefix checks
passed in each profile. Both reports cover the same fixture configurations;
the direct-versus-certified fitted-output classification differs by profile
and is retained explicitly. No divergent fitted matrix is recorded as a direct
match. The existing 192-group sklearn integration gate passed on the updated
checked executable; this does not claim a rerun of all older reference suites.

The public example matched independent sklearn: a (24,2) embedding, five
optimizer steps and objective 23.980587277129224 (checked) or
23.980587277129221 (optimized). The source value is 23.980587277129224.
These are synthetic fixture results, not general performance guarantees.

Final audits verified strict JSON, unique result identifiers, complete fixture
coverage, upstream sklearn/SciPy hashes, tested canonical executable hashes,
all six pinned vendor-file digests and fresh objects for 113 core source
modules, four vendor modules and 112 native test programs in each profile.
New Fortran formatting and Python compilation passed. Staged Git whitespace
passed except for intentionally pristine upstream License.txt whitespace.
Production Fortran and native tests remained fixed through the final gates.
Native logs are retained in validation/results/nca_native_checked.log and
nca_native_release.log. The broader translation goal remains active.


## Structured feature mappings (2026-11-21)

Added DictVectorizer and FeatureHasher in sklearn_feature_extraction, exported
through the sklearn facade. Native ordered feature_row/item/string records
support numeric values, string categories, iterable string categories,
explicit None values, UTF-8 names, embedded NULs and trailing blanks. Forward
outputs use the existing 1-based real64 CSR helper, with explicit dense methods.
DictVectorizer uses a growing hash-table vocabulary, optional lexicographic
ordering, unknown-name omission, feature restriction and dense/CSR inverse
mapping. FeatureHasher supports dict/pair/string inputs, optional hash-dependent sign flips,
 canonical collision aggregation and int32-range feature widths.
MurmurHash3 x86_32 uses explicit byte assembly and bounded int64 arithmetic for
portable uint32 products, including the signed INT32_MIN result. Helpers are
not counted as additional estimator algorithms.

The reference suite retains 2,945 groups across 853 fixtures: 128 dictionary
fixtures, 192 hashing fixtures and 533 exact signed-hash fixtures. Four data
seeds, three hasher input modes, sorted/first-seen vocabularies, category lists,
zero-width results, restriction and live separators are exercised. Cases
include empty strings, raw bytes, long UTF-8 keys, 1,200 learned columns,
2,000-token rows, explicit/cancelled zeros, NaN/Inf forward values, overflow,
all hash-tail lengths, uint32 seeds and an inverted four-byte preimage of
INT32_MIN across multiple feature widths. The hash, feature names and CSR
shape/pointer/index comparisons are exact. Dictionary stored-value multisets
are compared exactly, allowing equivalent ordering among duplicate columns.
Numeric outputs use 3e-13 relative/absolute tolerance. Protocol placeholders
for absent hasher vocabularies are not counted as algorithm result groups.

An initial reduction-order investigation found that SciPy sparse scalar lookup
uses get_csr_submatrix(...).data.sum(dtype=...), hence NumPy's pairwise sum.
The native CSR inverse originally used sequential addition. After identifying
the actual operation, it was changed to the source pairwise order using the
existing attributed quiet-IEEE KernelCenterer reduction. The obsolete release
build was deliberately stopped for this correction. A native regression checks
16 for sparse inverse versus 0.5 for dense materialization on 48 ill-conditioned
contributions; the reference suite also exercises 280-term recursive blocks.
All observed discrepancies were removed. Diagnostic forward-error calculations
remain available on a mismatch, but the suite accepts no roundoff exceptions.
The NumPy 2.2.6 loops_utils.h.src SHA256 was reverified against upstream.

Inventory is now 209 core public types, 115 core source modules, 113 native
test programs and 93 retained reference suites totaling 391,892 core groups.
Four existing vendor modules and the separate 42 NumPy RNG checks are excluded
from those core counts. The 40 new isolated rejections bring nine registries
to 2,798 cases. Original code remains MIT, with sklearn, SciPy and NumPy BSD
attribution and Austin Appleby's public-domain MurmurHash provenance retained.
No dependency or public extension algorithm is added by this increment.

Reproduce with compare_feature_mapping.py and
check_feature_mapping_invalid_inputs.py, using --profile checked or
--profile release. Both accept --executable PATH to use prebuilt matching
drivers without invoking fpm. --cases filters structured fixture cases for
debugging; the independent hash checks still run. Full reports have
cases_filter=null. Reports retain source/binary hashes, exact structure
checks, maximum numerical differences and any diagnostic sum bounds; a
nonempty roundoff diagnostic list fails the reference gate.


Final gates passed all 113 native programs in checked runtime/FPE-trap and
optimized fpm builds after the pairwise-reduction correction. Both final fpm
reference executables passed all 2,945 groups across the same 853 fixtures,
with zero measured numerical difference and no roundoff exceptions. Each
profile also passed all 40 isolated rejections. The existing 192-group sklearn
integration gate passed on the updated checked executable; no rerun of all
older reference suites is claimed.

The structured_features example passes both profiles and matches independent
sklearn: dictionary shape (2,3), temperature column 3 in native indexing,
hashed shape (2,32), and four stored hashed entries. These are illustrative
fixture results, not general performance claims.

Final audits verified strict JSON, unique result identifiers, identical
checked/release check configurations, complete unfiltered reports, upstream
sklearn/SciPy hashes and tested canonical executable hashes. Fresh objects were
verified for all 115 core source modules and 113 native test programs in each
profile. Fortran formatting, Python compilation and staged Git whitespace
checks passed. Production code and native tests remained fixed through these
final gates. Native logs are retained in validation/results as
feature_mapping_native_checked.log and feature_mapping_native_release.log.
The broader translation goal remains active.


## Callable transforms and target wrappers (2026-11-21)

Added the sklearn FunctionTransformer and TransformedTargetRegressor
counterparts. Core inventory is now 211 public estimator/transformer/splitter
types in 117 source modules, plus the existing four vendored optimizer modules.
There are 115 native test programs and 95 retained core reference suites.
The two new suites contain 6,114 groups across 972 fixtures: 1,410 groups /
300 FunctionTransformer fixtures and 4,704 groups / 672 transformed-target
fixtures. Total retained core groups are 398,006; the separate 42 legacy RNG
groups remain excluded. The new isolated registry has 33 cases, bringing the
ten registries to 2,831 cases. These counts measure API subsets, not complete
sklearn class ports or a percentage of sklearn functionality.

FunctionTransformer covers identity and numeric callbacks, optional dense
validation, feature metadata, generated/explicit/callback feature names,
fit-free transforms, changed-width inverse inputs and inverse diagnostics.
The target wrapper covers cloned native transformer prototypes, paired target
functions, default or supported native regressors, weighted SVD linear fits,
scalar/matrix targets, original-scale scoring and independent multiple outputs.
Unsupported joint multi-target bases are diagnosed rather than replaced with
independent models. Full callback/rank/weighting limits are in docs/API.md.
No dependency or public algorithm extension is added; original code remains
MIT with sklearn-derived BSD attribution retained.

Reference gates use installed sklearn 1.6.1, NumPy 2.2.6 and existing native
transformers. Function results use 3e-12 relative/absolute tolerance; target
results use 3e-9 except inherited PowerTransformer fits at 3e-6. Transformer
state can have large absolute differences for very large power-fit parameters;
reports retain maximum absolute differences, fixture contexts and fractions
of the allowed relative/absolute tolerance. Every transformed training target
is also checked directly. Metadata, names and output shapes are exact checks.

The target validator retains all fixtures. If a native CART split differs,
it uses the existing source-conditioned certificate: source sklearn stumps
verify every node's response, impurity and admissible split within a measured
floating-point accumulation bound. Predictions and inverse mapping are then
checked against these certified source leaf responses. Constant-target R2
is discontinuous at exact prediction equality; numerical discrepancies are
recorded explicitly and the native score is checked against sklearn r2_score
on native predictions. This is separate from direct prediction comparisons.
Strict JSON reports record source/binary hashes, tree runtime provenance,
warning counts, certificates and constant-target score differences.

Reproduce after building the desired profile:

```
python validation/compare_function_transformer.py --profile checked
python validation/compare_transformed_target.py --profile checked
python validation/check_target_wrappers_invalid_inputs.py --profile checked
```

Use --profile release for the optimized gate. --executable accepts a completed
native executable, avoiding another fpm invocation. Do not run simultaneous fpm
builds in this checkout. The transformed_targets example shows log/exp target
regression and a standalone callable transform through the sklearn facade.


Final gates passed all 115 native test programs in both runtime/FPE-checked
and optimized fpm profiles. Both canonical fpm reference executables passed
all 6,114 new groups across the same 972 fixtures; each profile also passed
all 33 isolated rejections. Each target report retained five source-conditioned
CART certificates and 15 constant-target score discrepancies, with every
fixture checked. The largest fraction of an allowed numerical tolerance was
0.942568, in an inherited PowerTransformer fitted-state comparison. The
FunctionTransformer suite had zero measured numerical difference.

The existing 192-comparison sklearn integration gate passed against the updated
checked executable. The transformed_targets example passed both profiles and
matched an independent sklearn calculation: original-scale R2=1 and first
log target 1.1666666667, with native maximum prediction error 1.0658e-14.
No rerun of every historical reference suite is claimed.

Final audits verified strict JSON, unique check identifiers, complete fixture
counts, matching profile configurations, installed source hashes, canonical
executable hashes and fresh objects for all 117 core modules, 115 native
programs and four vendored optimizer modules per profile. Python compilation,
Fortran line lengths and Git whitespace checks passed. Production and native
test sources stayed fixed through these final gates. Native logs are retained
as validation/results/target_wrappers_native_checked.log and
validation/results/target_wrappers_native_release.log. The broader translation
goal remains active.


## Dense column and parallel feature composition (2026-11-21)

Added sklearn ColumnTransformer and FeatureUnion counterparts with 17 existing
native child transformer types and recursive composition of both new graph
types. Selection supports ordered/repeated 1-based positions, logical masks,
explicit feature names and fit-time callbacks. Column transforms clone models,
retain empty selections without fitting, exclude all explicitly selected columns
from remainder, and append a dropped/passthrough/transformed remainder. Fitted
union lists follow sklearn's update convention. Both support branch weighting,
feature names and empty results; ColumnTransformer additionally exposes output
indices, custom name formatting and named query reordering/missing-drop columns.
Native tests cover independent graph copies, prototype cloning, changed-row
single branches and current-weight transforms. Production/native sources were
frozen before the final checked/release fpm gates.

Core inventory is now 213 public estimator/transformer/splitter types in 119
source modules, plus the existing four vendor optimizer modules. There are 117
native test programs and 97 retained core reference suites. The new suites
contain 2,160 groups / 360 ColumnTransformer fixtures and 1,200 groups / 240
FeatureUnion fixtures, covering all 17 supported existing child types, both
nested graph types, callback expansion, selector variants, remainder modes,
fit versus fit_transform, prefit prototypes and generated/explicit feature
names. Total retained core groups are 401,366; the separate 42 legacy RNG
groups remain excluded. The new isolated registry contains 43 cases, bringing
the eleven registries to 2,874 cases. Counts describe API subsets, not full
sklearn class ports or a percentage of sklearn's functionality.

The reference engine uses installed sklearn 1.6.1, NumPy 2.2.6 and pandas for
named-column reference semantics. Numeric comparisons use 3e-9 relative/absolute
tolerance, except inherited PowerTransformer fits at 3e-6. Metadata, output
shapes, names, selector call counts and native/source output slices are exact.
No fixtures are filtered. Reports retain source hashes, canonical native binary
hashes, warning counts, numeric differences and full per-fixture checks.
Numeric category names include subnormal and maximum finite values, exponent
layout boundaries, negative zero, full-precision random values and ordinary
decimals. KNNImputer naming preserves sklearn 1.6.1's all-missing-column omission
even when the numeric output retains the columns. Active zero-width blocks
still participate in concatenation row checks; a single active branch may
change row count, as in source.

These are dense, serial, unsupervised composition APIs. Existing child limits
continue to apply. No generic external estimator objects, supervised y routing,
sparse stacking, pandas output containers, metadata routing or general
sequential Pipeline is added. Python slicing/scalar column selection is
represented with explicit native position vectors. Explicit name vectors
provide named matrix indexing without importing a Python container library.
The API and example document native storage/indexing conventions. Original
contributions remain MIT, with sklearn BSD attribution retained. No new
dependency or public extension algorithm is added.

Reproduce the new gates with one fpm build at a time:

```
python validation/compare_column_transformer.py --profile checked
python validation/compare_feature_union.py --profile checked
python validation/check_composition_invalid_inputs.py --profile checked
```

Use --profile release for optimized gates, or --executable to inspect completed
canonical executables without another fpm invocation. Example:
`fpm run --example feature_composition`.


The first optimized rejection gate exposed an unsafe assignment from an
unallocated allocatable selector-function result. The native selector interface
now uses a subroutine with an allocatable integer intent(out) argument; its
allocation is validated before copying or indexing. Focused checked and
optimized builds both passed all 43 rejection cases, and the optimized normal
selector probe passed 360 groups across 60 fixtures. Complete native/reference
gates were restarted after this correction. The final results below refer to
the corrected subroutine interface, not the initial function-result build.


Final gates after the selector correction passed all 117 native test programs
in both runtime/FPE-checked and optimized fpm profiles. Both canonical fpm
reference drivers passed all 3,360 groups across the same 600 unfiltered
fixtures, and each profile passed all 43 isolated rejection cases, including
the previously failing unallocated-selector case. No roundoff exception or
conditional certificate was needed for these new comparisons; the existing
PowerTransformer numerical tolerance remains explicit.

The updated checked reference executable also passed the existing 192-group
sklearn integration gate. The feature_composition example passed both profiles
and matched an independent sklearn calculation: processed shape (6,4), first
output name numeric__x0, and union shape (6,7). No rerun of every historical
reference suite is claimed.

Final audits verified strict JSON, unique identifiers, full fixture counts,
identical checked/release configurations, validator/installed-source hashes and
canonical executable hashes. Current objects were verified for all 119 core
modules, 117 native test programs, four vendored optimizer modules, and the
new helper/driver/example sources in each profile. Python compilation,
Fortran line lengths and Git whitespace checks passed. Production/native
sources remained fixed through these final reruns. Corrected native logs are
retained as validation/results/composition_native_checked.log and
validation/results/composition_native_release.log; the original selector
failure log is retained separately for diagnosis. The broader translation
goal remains active.


## Sequential Pipeline and recursive dense transformer bridge (2026-11-21)

Added the sklearn Pipeline counterpart with typed classifier/regressor
prediction, sequential fit/fit_transform, supervised selector/TargetEncoder
steps, supported native estimator tails, original final-estimator scoring,
inverse transforms, feature names and independent contiguous slices. A new
internal abstract dense_transformer protocol lets pipelines appear inside
ColumnTransformer and FeatureUnion without a module cycle. Pipelines can also
contain either parallel graph type or another pipeline. No additional algorithm
is represented by this protocol or pipeline_step storage.

The implementation preserves intermediate fit_transform versus final fit
ordering. TargetEncoder training encodings are cross-fitted; subsequent
transform/predict uses the fitted full-data mappings. Final estimator type,
not target dtype, controls classification/regression. Real integral labels
and integer regression targets have native adapter tests. Weights go to the
final estimator. Scoring transforms input once and delegates the correct score
family, including GLM D2. Existing component numerical/weighting restrictions
remain explicit. The dense API is fit-first and excludes caching, metadata
routing, arbitrary Python estimators and sparse paths; detailed method/rank,
name-container, slicing and matrix-target limits are in docs/API.md.

Inventory is now 214 core public types, 120 core source modules, 119 native
test programs and 98 retained core reference suites totaling 409,922 result
groups. The separate 42 legacy RNG groups remain excluded. The new pipeline
suite has 8,556 groups across 756 unfiltered fixtures: three seeds, 13
preprocessing families, transformer/regression/classification/matrix/GLM tasks,
four configurations, twelve additional final-handoff fixtures, weighted tails,
binary/multiclass and integer/real labels,
fit versus fit_transform, names, inverse outputs, nested graphs and callback
call counts. Matrix-target cases omit supervised single-target transformations
that are not part of that API. New isolated checks add 58 cases, bringing twelve
registries to 2,932 cases. These counts measure API subsets, not full sklearn
class ports or a percentage of sklearn functionality.

Reference tests use installed sklearn 1.6.1 with 3e-8 relative/absolute numeric
tolerance, except native/source logistic and Poisson optimization at 3e-6.
Classifier labels, names, classes, dimensions and callback counts are exact.
Binary decision scores are compared in native (samples,1) form. Inverse groups
cover available native component inverses; components whose native inverse is
absent produce an explicit empty placeholder, not an asserted Python inverse.
The source imputer's inverse also requires add_indicator, absent in the native
simple imputer. Source KNN references explicitly select brute search to match
the existing native backend.

KNN cases additionally compare actual fitted training features and target order.
Rare default-source differences from ties and Euclidean distance cancellation
are not excluded or accepted under wider prediction tolerances. A recorded
source-conditioned certificate validates native neighbor indices/distances
against direct Euclidean geometry and relative accumulation bounds, verifies
source-space admissibility under measured input perturbations, and bounds the
source optimized distance kernel's squared error. Sklearn precomputed-distance
KNN then independently reconstructs weighted predictions/probabilities and
scores for the certified selected neighbors. The report marks affected groups
with this oracle and retains changed-neighbor counts, perturbations and kernel
error/bound witnesses. All other groups keep direct sklearn comparisons.

The dispatcher change also requires rerunning both existing parallel-composition
reference suites against updated binaries. The existing 192-group integration
gate is rerun on the checked executable. Reports retain strict JSON, full
fixture/check lists, validator/certificate/source hashes and canonical native
executable hashes. Original contributions remain MIT with sklearn BSD notices;
there is no new external dependency or public extension algorithm.

Reproduce with one fpm build at a time:

```
python validation/compare_pipeline.py --profile checked
python validation/check_pipeline_invalid_inputs.py --profile checked
python validation/compare_column_transformer.py --profile checked
python validation/compare_feature_union.py --profile checked
```

Use --profile release for optimized gates or --executable for a completed
canonical binary. The public example is `fpm run --example sequential_pipeline`.


The final handoff audit identified two nested-final cases requiring correction:
explicit feature names after a passthrough prefix, and integer regression
targets entering a nested pipeline's supervised selector. Names and original
target representation are now forwarded to the nested pipeline; conversion to
a base regressor's real target occurs only at that estimator boundary. Twelve
focused source fixtures passed 147 groups, including direct comparison of
integer-target selector scores with sklearn. These fixtures and final nested
input-name checks are retained in the complete 756-fixture suite. Production
and native sources were frozen again, and both full fpm/native/reference gates
were restarted. Final results below refer to this corrected handoff build.


Final verification of the corrected build passed all 119 native test programs
in both checked and optimized profiles. Each profile passed 8,556 Pipeline
groups across 756 fixtures, 58 Pipeline rejection checks, 3,360 existing
composition reference groups and 43 composition rejection checks. Four KNN
fixtures per profile required the retained geometric certificates. The checked
integration gate passed all 192 comparisons. Both canonical example binaries
reported training R2 0.9999792, feature numeric__x0 and 20 predictions, matching
the independent sklearn example (R2 0.9999792424116486).

The final audit verified complete strict reports, matching profile fixture
configurations, executable/validator/source hashes and fpm object content
digests for 243 source/test/vendor files plus four new app/example files in
each profile. Two byte-identical sources per profile had newer timestamps
than their cached objects; their content digests matched. Retained evidence
is in validation/results/pipeline_final_audit.json and the Pipeline reports
and native/integration logs. Older unrelated reference suites were retained,
not all rerun for this increment. The broad sklearn translation remains ongoing.


## Model-based feature selection (SelectFromModel)

This increment adds select_from_model, bringing the inventory to 215 core
public types, 121 core source modules and 120 native test programs. Ninety-nine
retained core reference suites total 413,810 result groups; the separate
42-group NumPy legacy RNG suite remains outside that count. The new registry
adds 47 isolated rejection checks, bringing 13 registries to 2,979 cases.
These are implemented API subsets rather than complete sklearn class ports.

The dense selector clones supported coefficient/tree-importance estimators,
accepts prefit models, aggregates matrix coefficients with nonzero integral or
infinite norms, interprets mean/median/scaled/numeric thresholds, applies stable
descending feature caps, and provides zero-filling inverse transforms, feature
names and support indices. Native callbacks provide custom importances and
feature caps. SGD partial_fit retains successive fitted state while first
cloning only estimator parameters. Sequential Pipeline dispatch supports this
supervised transformer, including integer regression targets, names and inverse
transforms. The existing native estimator fit/weighting restrictions still apply.

The new source suite retains 432 fixtures and 3,888 result groups over three
seeds, twelve estimator/importance families and twelve control configurations.
It checks support, importance norms, thresholds, fitted metadata, names, callback
counts and full fit/query/inverse matrices, including empty selection, exact
ties, positive/negative infinite thresholds and direct prefit transforms.
Families cover explicit vector/matrix prefit coefficients, weighted Ridge,
Lasso, binary/multiclass logistic regression, regression/classification CART,
MultiTaskLasso, three-batch regression/classification SGD, and vector/matrix
importance callbacks. The existing SGD solver supports L2/no penalty; no new
SGD loss or penalty algorithm is added by this selector.

All 72 fitted tree fixtures additionally receive independent certificates of
node membership, weighted impurity, globally optimal split loss and normalized
feature contributions. Six fixtures in the development checked run differed
from sklearn in feature attribution on existing tied split paths. Those retain
all selection checks against sklearn using the certified native fitted
importance vector. Reports identify the selection oracle for every fixture,
retain the source importance difference and numerical certificate witnesses,
and hash the helper. They do not loosen feature-mask or transform comparisons.
The remaining fixtures compare directly with the fitted source estimator.

New implementation/adapters remain MIT with retained sklearn BSD attribution;
no external repository or dependency is imported. Sparse arrays, arbitrary
Python attribute paths, metadata routing and general estimator reflection are
not implemented. Importances and numeric dense inputs must be finite; norm
overflow raises a native diagnostic. Empty transform results are supported,
while inverse_transform requires a nonempty selected matrix, as in sklearn.
Naming requires fit even though direct prefit get_support/transform works.
See docs/API.md and example/model_based_selection.f90 for native signatures.

Reproduce after building one profile at a time:

```
python validation/compare_model_selector.py --profile checked
python validation/check_model_selector_invalid_inputs.py --profile checked
```

Use --profile release for optimized validation, or --executable to select an
already-built canonical executable without invoking fpm again. Full verification
results for the frozen implementation are recorded below when complete.


Final verification of the frozen implementation passed all 120 native test
programs in checked and optimized fpm builds. Each canonical profile passed
3,888 selector groups across 432 fixtures and all 47 selector rejection checks,
8,556 existing Pipeline groups plus 58 rejection checks, and 3,360 existing
composition groups plus 43 rejection checks. All 72 tree certificates passed
per profile; six fixtures per profile used the certified native-tree selection
oracle. The checked integration gate passed all 192 comparisons.

Both canonical public example binaries retained columns 2 and 4 and printed
training R2 0.9999949, matching the independent sklearn value
0.9999949201575922. The final audit verified complete strict reports, identical
profile fixture configurations, executable/validator/certificate/source hashes,
and all 248 compiled content digests per profile (245 core/test/vendor files
plus the two new drivers and example). Evidence is retained under
validation/results/model_selector_final_audit.json, model_selector_*.json,
and the native/integration logs. Existing unrelated reference-suite reports
were retained rather than all rerun for this increment. The broad sklearn
translation remains ongoing.


## General parameter search (GridSearchCV and RandomizedSearchCV)

This increment adds grid_search_cv and randomized_search_cv through sklearn_search
and the sklearn facade. To avoid a cycle through the existing model-selection
and estimator-dispatch modules, sklearn_model_selection does not re-export these
new types. There are now 217 core public estimator/transformer/splitter types,
124 core source modules and 122 native test programs. Parameter storage/grid/
result/scorer helpers are excluded from estimator counts. Two new reference
suites bring 101 retained core suites to 419,854 result groups. The separate
42-group legacy RNG suite remains excluded. The new 61-case registry brings
14 rejection registries to 3,040 cases.

Typed scalar parameter spaces support real, integer, logical and text values;
axes sort by name and rightmost values vary fastest. Lists of grids preserve
candidate order, including empty dictionaries. Finite random sampling matches
sklearn's permutation, tracking-selection and reservoir branches without
replacement. If any axis has an rvs callback, grid choices and sorted axes are
sampled with replacement. Large finite spaces are indexed lazily: the test
samples a million-point grid without creating a million candidate objects.
Integer sizes and seeds use native default-integer limits; random_state defaults
to deterministic zero. None/array/estimator-valued parameters require an explicit
native representation/setter callback rather than Python object reflection.

Searches clone native models for each candidate/fold, slice fit weights, use
unshuffled KFold or first-appearance-encoded StratifiedKFold, accept explicit
1-based folds, and retain complete split/mean/std/rank tables and measured
fit/score times. Scoring can be a built-in name, a scalar callback, or named
multiple scorers. Refit can use the best metric, be disabled, or use a result
callback; ties choose the first minimum-rank candidate. Optional train scores
and refitted prediction/probability/decision/score delegation are available.
Nested Pipeline, SelectFromModel estimator and transformed-target regressor
parameter paths are supported. Existing native child fit/weighting/solver
restrictions still apply; Pipeline weights go to its final estimator.

The supported scope is serial, dense finite X and vector regression/class labels.
Original project adapters remain MIT and sklearn BSD notices are retained. No
external repository/dependency or extension algorithm is imported. The native
error_score default is 'raise' and only that policy is supported: existing
child estimators use terminating require diagnostics, so failed fits cannot yet
be recovered as numeric replacement scores. Parallel jobs, general metadata
routing, sparse data, unsupervised/matrix targets and arbitrary Python clone/
parameter reflection remain outside this increment. Timing measurements use
native wall-clock resolution and are not expected to match Python timings.

The parameter-space suite passes 1,976 candidate checks across 112 fixtures:
four seeds, heterogeneous values, empty/list grids, distribution callbacks,
sampling ratio boundaries, oversubscribed sampling and lazy large-grid indexing.
The search suite retains 4,068 groups across 270 fixtures over three seeds,
ten model/scorer families and nine control configurations. It checks full
candidate/fold results, ranks, refit metadata and score availability, candidate
parameters, split indices, callback fit/set counts, timing shape/nonnegativity
and refitted predictions/responses. Families include Ridge, ElasticNet, binary/
multiclass logistic regression, Gaussian NB, KNN, scaling pipelines, nested
model-based selection, nonfinite scalar scorers and custom fit/set callbacks.

All numerical score tables retain direct sklearn comparisons. Ranking is also
checked exactly with scipy.stats.rankdata applied to the emitted native means.
Twenty-eight metric/fixture records in the development checked run needed
roundoff certificates: any changed pair ordering must lie within
128*n_splits*machine_epsilon times the source score scale, including the source/
native score differences. Those records retain the changed pairs and bound.
If such an ordering changes the selected candidate, the source refit/prediction
oracle uses that certified selection. NaN/infinity ranking and one-row R2
folds remain covered. No rank check is dropped and score tolerances are not
relaxed to conceal a changed ordering.

Reproduce after building one fpm profile at a time:

```
python validation/compare_parameter_space.py --profile checked
python validation/compare_search.py --profile checked
python validation/check_search_invalid_inputs.py --profile checked
```

Use --profile release for optimized gates or --executable with a completed
canonical binary. See docs/API.md and example/parameter_search.f90 for usage.
Final verification of the frozen implementation is recorded below.


Final verification of the frozen implementation passed all 122 native programs
in checked and optimized fpm builds. Each canonical profile passed 1,976
parameter-space checks across 112 fixtures, 4,068 search checks across 270
fixtures, and all 61 isolated rejection checks. Twenty-eight metric/fixture
roundoff certificates passed in each profile. The checked integration gate
passed all 192 comparisons.

Both canonical public example binaries selected alpha 0.01 from three candidates
and printed mean CV score -1.104558e-6, matching the independent sklearn value
-1.1045576422766414e-6. The final audit verified complete strict reports, identical
profile check configurations, canonical executable/validator/certificate and
installed sklearn/SciPy source hashes, plus all 254 compiled content digests
per profile (250 core/test/vendor files plus the three new drivers and example).
Evidence is retained under validation/results/search_final_audit.json,
search_*.json, parameter_space_*.json and the native/integration logs. Existing
unrelated reference suites were retained rather than all rerun for this
increment. The broader sklearn translation remains ongoing.


## General cross-validation and out-of-fold responses

This increment adds cross_validate and cross_val_predict, plus general scalar
cross_val_score and a typed class-label prediction adapter, in sklearn_validation.
The facade exports cross_validate, cross_val_predict and cross_val_predict_class;
it retains the older typed cross_val_score from sklearn_model_selection. Import
the broader scalar helper directly from sklearn_validation (rename it on import
when also importing the facade). No new estimator types are counted: the core
inventory remains 217 estimator/transformer/splitter types, with 125 source
modules and 124 native test programs. Two new reference suites bring 103 core
suites to 424,408 retained result groups. The separate 42-group legacy NumPy RNG
suite remains excluded. Fifty new rejection cases bring 15 registries to 3,090.

Cross-validation supports native estimator dispatch, real/integer vector targets,
unshuffled KFold or first-appearance-encoded StratifiedKFold, explicit 1-based
folds, sliced fit weights, scalar/multiple scoring, train scores, returned fitted
models/indices, measured fit/score times and custom fit/score callbacks. No final
full-data fit occurs. The scalar helper disables train scoring and returned
artifacts, as sklearn cross_val_score does. No-target fitting supports the
existing composition transformer adapters with explicit scoring callbacks;
other native no-target fits can use a fitter callback.

Out-of-fold predictions require the held-out indices to partition all samples
exactly once, while ordinary cross_validate permits repeated/nonpartition tests.
The output always restores original sample order. Supported classifier response
methods are predict, predict_proba, predict_log_proba and decision_function.
Response methods encode global labels before fitting each fold and place missing
class columns in global sorted-label order, using zero probabilities and minimum
finite real64 log/margin placeholders. Incompatible missing-class decision shapes
are rejected. Stable native NB/boosting log probabilities are retained rather
than reconstructed from underflowed probabilities. Known estimators without a
log-probability API, including KNN, raise a native diagnostic. Scalar/binary
responses are stored as n-by-1 matrices; the class-label adapter returns integers.

Integer regression targets retain their original representation in native
Pipeline fitting, including supervised selector dispatch. Native fit/scoring
callbacks receive normalized real regression or integer class targets; encoded
class targets are visible to OOF response fit callbacks. Custom response and
class-metadata hooks support native wrappers. Returned models are independent
value copies; procedure pointer lifetimes remain the caller's responsibility.

The supported scope remains serial dense finite X, vector targets and existing
native child fit/weight/solver restrictions. Only error_score='raise' is supported
because child diagnostics terminate execution. Sparse/precomputed pairwise slicing,
metadata routing, parallel jobs, recoverable fit/scoring errors, multioutput/
multilabel targets, and no-target cross_val_predict are not implemented here.
No-target default scoring requires a callback. Native callback response matrices
must be allocated and correctly sized. None of these additions is a public
extension algorithm; original contributions remain MIT with sklearn BSD notices.
No external repository or dependency is imported.

The cross_validate source suite retains 3,888 groups across 288 fixtures:
three seeds, twelve model/scorer families and eight configurations. It compares
complete test/train scores, scalar wrapper scores, callback counts, returned
model predictions/probabilities/classes, explicit indices and timing shapes.
Families include Ridge, ElasticNet, logistic regression, Gaussian NB, KNN,
scaling/selection pipelines, no-target scaling and nonfinite scalar scores.
Custom CV includes repeated test rows and one-row R2 folds. The OOF suite retains
666 groups across 222 fixtures covering all supported response capabilities,
nonmonotone partitions, missing classes, weighted fitting, encoded fit targets,
custom response callbacks and original row order. Unsupported capability cases
are retained in the isolated rejection registry. All numerical/label comparisons
use direct sklearn 1.6.1 oracles; no roundoff certificate is needed for these
suites.

Reproduce with one fpm build at a time:

```
python validation/compare_cross_validate.py --profile checked
python validation/compare_cross_val_predict.py --profile checked
python validation/check_validation_invalid_inputs.py --profile checked
```

Use --profile release for optimized runs or --executable for completed canonical
binaries. See docs/API.md and example/out_of_fold_predictions.f90 for usage.
Final verification of the frozen implementation is recorded below.


Final verification of the frozen implementation passed all 124 native test
programs in checked and optimized fpm builds. Each canonical profile passed
3,888 cross_validate groups across 288 fixtures, 666 cross_val_predict groups
across 222 fixtures, and all 50 isolated rejection checks. The checked
integration gate passed all 192 comparisons. No roundoff certificate was used
for either new reference suite.

Both canonical public example binaries reported four folds, 40 out-of-fold
predictions and R2 0.9915055, matching the independent sklearn value
0.9915054641718475. The final audit verified complete strict reports, identical
profile check configurations, canonical executable/validator/entry and installed
source hashes, and all 256 compiled content digests per profile (253 core/test/
vendor files plus the two new drivers and example). Evidence is retained under
validation/results/general_cv_final_audit.json, cross_validate_*.json,
cross_val_predict_*.json, validation_invalid_*.json and native/integration logs.
Existing unrelated reference suites were retained rather than all rerun for
this increment. The broad sklearn translation remains ongoing.


## Learning curves and permutation significance tests

`sklearn_validation` and the `sklearn` facade now expose `learning_curve`,
`validation_curve` and `permutation_test_score`, adapted from sklearn 1.6.1
`model_selection/_validation.py` and `_split.py`. These functions add no estimator
classes: the inventory remains 217 core types and 125 source modules, with 126
native test programs. There are now 106 core reference suites retaining 429,685
result groups, plus the separate 42-group legacy NumPy RNG suite. The new
34-case rejection registry brings 16 registries to 3,124 cases.

Learning curves accept real or integer targets, or no targets with a scoring
callback. `learning_curve_options` extends `cross_validation_options`; supply
fractional `train_sizes` in (0,1] or integer `train_counts`, never both. Defaults
are five evenly spaced fractions from 0.1 to 1. Sizes are sorted and deduplicated
against the first fold's training length. Later shorter folds clip their prefixes;
`actual_train_sizes` records the resulting counts. Seeded optional training-index
shuffles use the existing NumPy-compatible legacy RNG. Results contain folds,
sizes, and train/test score matrices indexed by size then fold. Optional timing
matrices report individual fits, with test scoring time for batch fits and both
train/test scoring time for incremental fits, matching the pinned source.

Batch fits start afresh for each size/fold. Incremental learning is available for
SGDRegressor, SGDClassifier and GaussianNB through
`exploit_incremental_learning`; fitted prototype state is cleared, each fold gets
one model, and successive chunks update it with partial_fit. Classifiers receive
the full target class set. Custom models can supply `partial_fit_func` and
`incremental_init_func`; the latter must clear any custom fitted state. Pinned
sklearn 1.6.1 forwards fit parameters unsliced on its incremental path, so full
vector sample weights normally fail the child learner's chunk-length check here
as well. Empty chunks on uneven folds also retain the source learner's rejection.

Validation curves take a parameter name and typed `parameter_value` array made
with `param(...)`. Parameter order and duplicates are retained; each value/fold
gets a fresh fit. Native nested Pipeline parameters and an optional
`set_params_func(model, parameter_set)` callback use the existing search adapters.
The result stores the parameter range, folds, and train/test score matrices.

Permutation tests return the observed mean CV score, all permuted scores, and
`pvalue=(1+count(permuted_score >= observed_score))/(1+n_permutations)`. A positive
permutation count is required (default 100); the seed defaults to zero. Optional
integer groups constrain target shuffling within each group. Default classifier
folds are recomputed for each permuted target; explicit folds remain fixed.
One cloned model is reused across folds within each observed/permuted run,
matching the pinned source, including custom fit callbacks. Fit weights retain
their original row association and are sliced per fold. NaN comparisons do not
count as exceedances.

Scope remains serial dense finite X, vector targets, scalar scoring and
`error_score='raise'`, with the existing child estimator restrictions. Sparse
inputs, metadata routing, parallel execution, recoverable errors and multioutput
targets are not provided. Native callbacks receive normalized targets; ordinary
Pipeline fits retain original integer regression targets. Procedure pointer
lifetimes remain the caller's responsibility. No extension algorithms or new
dependencies were introduced. Original contributions remain MIT, with retained
sklearn BSD attribution.

The direct source suites retain 2,733 learning-curve groups over 360 fixtures,
1,584 validation-curve groups over 264 fixtures and 960 permutation groups over
240 fixtures. Learning fixtures include 21 matching source rejections for
one-class prefixes or empty incremental chunks. Comparisons cover complete
score arrays, prefix sizes, fold indices, fit traces, callback counts, parameter
order, incremental resets, seeded shuffles, groups, nonfinite scores and exact
p-values. Timing checks compare shapes and nonnegative values, not wall times.
See `example/model_learning_curves.f90` for a complete public API example.


Reproduce the new source and rejection suites after building one profile at a time:

```
python validation/compare_learning_curve.py --profile checked
python validation/compare_validation_curve.py --profile checked
python validation/compare_permutation_test_score.py --profile checked
python validation/check_curve_invalid_inputs.py --profile checked
```

Use `--profile release` for optimized binaries or pass `--executable` to use an
already built executable without invoking fpm. Python dependencies are validation
only; the Fortran package has no new runtime dependency.


Final verification passed all 126 native test programs in both checked and
optimized fpm builds. Each canonical profile passed all three new source suites
(2,733 / 1,584 / 960 groups), 34 new rejection checks, the existing 3,888-group
cross_validate and 666-group out-of-fold suites, and 50 existing CV rejection
checks. The checked integration suite passed all 192 comparisons. Both public
example binaries matched independent sklearn results: sizes [7,15,30], final
mean test R2 0.9992057722522532 and permutation p-value 0.05.

The final audit verified identical checked/release check lists, strict JSON,
canonical executable/validator/entry and installed source hashes, and 258 compiled
content digests in each profile (255 core/test/vendor files plus the two new
drivers and example). Evidence is retained in validation/results/curves_final_audit.json,
learning_curve_*.json, validation_curve_*.json, permutation_test_score_*.json,
curve_invalid_*.json and curves_*_tests.log / curves_*_gates.log. Historical unrelated
reference suites were retained rather than all rerun. The broader sklearn
translation remains ongoing.


## Seeded regression, manifold and quantile sample generators

Seven additional sklearn 1.6.1 datasets are available from `sklearn_datasets`
and the `sklearn` facade. Implementation is in `sklearn_sample_generators`,
adapted from `datasets/_samples_generator.py` with retained BSD-3-Clause
attribution. There are now 126 core source modules and 127 native test programs;
the estimator/transformer/splitter inventory remains 217. The new reference suite
brings 107 core suites to 430,357 retained result groups. The separate 42-group
NumPy RNG suite remains excluded. Twenty-one rejection cases bring 17 isolated
registries to 3,145 cases. No public extension or dependency was added.

All outputs are allocated by the subroutine: X has shape (samples,features),
regression/latent targets are real vectors and classification labels are integer
vectors starting at zero. Samples must be positive. The interfaces are:

```
call make_friedman1(n_samples,n_features,x,y,noise,random_state)
call make_friedman2(n_samples,x,y,noise,random_state)
call make_friedman3(n_samples,x,y,noise,random_state)
call make_sparse_uncorrelated(n_samples,n_features,x,y,random_state)
call make_s_curve(n_samples,x,t,noise,random_state)
call make_swiss_roll(n_samples,x,t,noise,random_state,hole)
call make_gaussian_quantiles(n_samples,n_features,x,labels,n_classes,mean,cov,shuffle,random_state)
```

Trailing parameters are optional. Noise defaults to zero and must be finite and
nonnegative; an omitted seed deterministically selects zero. These seven routines
use independent local NumPy-compatible legacy RandomState streams and do not
change the Fortran global RNG. Required sample/feature sizes follow the existing
native dataset API rather than Python's optional size defaults. The four earlier
native dataset routines retain their existing RNG behavior.

Friedman1 requires at least five features; only the first five enter its nonlinear
response. Friedman2 and Friedman3 share the four-feature ranges of the electrical
circuit benchmark, using square-root and arctangent responses respectively.
Sparse uncorrelated regression returns a dense Gaussian design, four informative
linear terms and unit Gaussian response noise. At least four features are required;
the pinned Python implementation fails by indexing for smaller positive counts,
while native code provides an explicit diagnostic.

S-curve and Swiss roll return noisy three-dimensional coordinates plus the clean
latent coordinate t. Swiss roll supports the source `hole` option, default false,
including its discrete corner draws and parameter order. Even a single sample
returns a length-one native t vector, whereas Python squeezes t to a scalar.
Gaussian quantiles support a finite mean vector (default zero), scalar isotropic
covariance (default one, including zero), class count (default three), and optional
sample shuffling (default true). Samples must be at least the class count. Radial
sorting precedes label allocation; any remainder belongs to the last class.

Draw order follows the source, including row-major feature draws, cached Gaussian
values, manifold noise generated in (3,samples) order, Gaussian quantile draws
before sorting, and legacy Fisher-Yates shuffling. Source comparisons check full
feature and target arrays, not only moments. The suite retains 672 groups over
336 fixtures: seven generators, four seeds and twelve configurations, including
one-row outputs, odd lengths, high feature counts, zero/nonzero noise, hole/no-hole,
zero covariance, shifted means, singleton classes and uneven class allocation.
It also runs 21 isolated invalid-input cases.

Run `python validation/compare_sample_generators.py --profile checked`, or use
`--profile release` / `--executable` for an existing binary. The public example is
`example/synthetic_sample_generators.f90`. Python is used only for validation.
Original native contributions remain MIT; sklearn-derived portions retain the
existing upstream BSD notice and the existing RNG notices remain applicable.


Final verification passed all 127 native test programs in both checked and
optimized fpm builds. Each canonical profile passed all 672 generator result
groups over 336 fixtures and all 21 isolated rejection checks. Full generated
feature and target arrays were identical to the installed sklearn outputs for
all retained fixtures in both profiles. The checked integration gate passed
192 comparisons. Both compiled public examples matched independent sklearn
values: Friedman1 mean target 14.392904265139222, Swiss-roll mean latent coordinate
9.539033758128216, and four Gaussian quantile classes of 25 samples each.

The final audit checked strict reports, identical profile fixture/check lists,
canonical executable/validator and installed source hashes, and 259 current
compiled content digests per profile (257 core/test/vendor files plus the new
driver and example). Reports and logs are retained under validation/results as
sample_generators_final_audit.json, sample_generators_{checked,release}.json,
sample_generators_invalid_*.json, sample_generators_example_*.json and
sample_generators_*_tests.log / sample_generators_*_gates.log. Other historical
reference suites were retained rather than all rerun for this increment. The
broader sklearn translation remains ongoing.


## Clustered classification and Hastie benchmark generators

`make_classification` and `make_hastie_10_2` are now exported by
`sklearn_datasets` and the facade. They adapt sklearn 1.6.1
`datasets/_samples_generator.py` in `sklearn_classification_datasets`. The
`classification_dataset_options` type stores generation parameters; it is not
counted as an estimator. The inventory remains 217 core estimator/transformer/
splitter types, with 127 source modules and 128 native test programs. There are
108 core reference suites retaining 430,965 groups, plus the separate 42-group
legacy RNG suite. Thirty-three new isolated rejection cases bring 18 registries
to 3,178 cases. No extension algorithm or new dependency is introduced.

```
type(classification_dataset_options) :: options
options%n_informative=4
options%n_redundant=2
options%n_repeated=1
options%n_classes=3
options%weights=[.2_dp,.3_dp]
options%random_state=42
call make_classification(120,9,x,labels,options)
call make_hastie_10_2(120,x,labels,random_state=42)
```

Both routines allocate real X matrices and integer label vectors. Native sample
and feature counts are required, following the existing dataset API; an omitted
seed deterministically uses zero rather than Python's global random state.
`make_classification` accepts optional options with Python's remaining defaults:
2 informative, 2 redundant, 0 repeated features; 2 classes and 2 clusters per
class; flip_y=0.01; class_sep=1; hypercube=true; shift=0; scale=1; shuffle=true.

The generator covers distinct binary hypercube vertices, scaled random polytopes,
per-cluster random covariance, redundant linear combinations, repeated columns,
uninformative Gaussian columns, weighted cluster allocation, randomly replaced
labels, affine transforms, sample shuffling and independent feature shuffling.
Informative dimensions above 30 follow the source's leading random bits plus
30-bit distinct vertices. Empty clusters still consume their covariance draws.
Full and partial class weights retain source truncation and cyclic remainder
allocation. Weights are not normalized: totals below one receive the remaining
samples, and overallocated clusters are clipped to the requested row count, as
in the source. A missing last class weight is inferred from one minus the sum.

Options `feature_shift(:)` and `feature_scale(:)` represent Python vector
transforms and override the corresponding scalar fields. `random_shift` and
`random_scale` represent Python shift=None and scale=None. Random flags cannot
be combined with explicit vectors. Scalars and vectors must be finite; scalar
scale must be positive, while vector scales may include zero and negative
entries, matching the source array parameter path. Native weights must be finite
and nonnegative, including an inferred last class; negative-weight slice behavior
accepted incidentally by Python is outside the native scope. Cluster products and
weighted cluster sizes must fit native default integers. Noise fractions must be
in [0,1], class separation positive, feature blocks must fit, and the requested
vertices must be representable by the informative dimensions.

Hastie 10.2 generates ten standard Gaussian features with labels +1 where squared
radius exceeds 9.34, and -1 otherwise. Labels are native integers rather than
Python's float array, with identical class values. Both generators use independent
local legacy RandomState-compatible RNG streams and leave caller options intact.

The shared bagging sampling helper now allocates its population pool only for
its permutation branch. Tracking and reservoir selection retain their previous
random draws without a needless population-sized allocation; this makes sampling
small vertex sets from a 2**30 population practical. Native tracking membership
uses a linear search, so very large tracked subsets can still be expensive.

The new source suite retains 608 result groups over 304 fixtures: 256 clustered
classification fixtures (32 configurations, eight seeds including the largest
native positive default integer) and 48 Hastie fixtures, including 12,000-row
benchmark draws. Complete features and labels are compared against sklearn.
Cases cover each sampling branch, empty and overallocated clusters, inferred/
zero/unnormalized weights, all feature blocks, one class, all/no label flips,
high informative dimensions, random polytopes and scalar/vector/random affine
transforms. There are also 33 isolated invalid-input checks.

Run `python validation/compare_classification_datasets.py --profile checked`;
use `--profile release` or `--executable` for an existing canonical driver.
See `example/classification_sample_generators.f90` for public usage. Original
native contributions remain MIT with retained sklearn BSD and existing RNG
notices. No external repository was imported.


Final verification passed all 128 native test programs in both checked and
optimized fpm builds. Each canonical profile passed the new 608-group / 304-fixture
classification dataset suite and 33 isolated rejections; the existing 672-group /
336-fixture sample-generator suite and 21 rejections; and all 32,100 bagging groups
over 527 fixtures, including 61 tree certificates. The checked integration gate
passed 192 comparisons. No unrelated historical reference suites were all rerun.

Both public example binaries matched independent sklearn values: classification
class counts [26,38,56], mean feature value 0.3055021930578388, and Hastie negative/
positive counts [63,57]. The audit verified strict reports, identical checked/
release check lists, canonical executable/validator and installed source hashes,
and 261 compiled content digests per profile (259 core/test/vendor files plus
the new driver and example). Evidence is retained under validation/results as
classification_datasets_final_audit.json, classification_datasets_*.json,
classification_datasets_invalid_*.json, classification_datasets_example_*.json,
bagging_*.json, sample_generators_*.json and classification_datasets_*_tests.log /
classification_datasets_*_gates.log. The broader sklearn translation remains ongoing.


## Gaussian-process covariance kernels and fitted kernel snapshots

`sklearn_gp_kernels`, also exported by `sklearn`, provides RBF, Matern,
RationalQuadratic, ExpSineSquared, DotProduct, ConstantKernel and WhiteKernel
counterparts through a value-type `gp_kernel`. Constructors are
`rbf_covariance(length_scale,fixed)`, `matern_covariance(length_scale,nu,fixed)`,
`rational_quadratic_covariance(length_scale,alpha,fixed_length,fixed_alpha)`,
`periodic_covariance(length_scale,periodicity,fixed_length,fixed_periodicity)`,
`dot_product_covariance(sigma_0,fixed)`, `constant_covariance(constant_value,fixed)`
and `white_covariance(noise_level,fixed)`. Parameters have sklearn's defaults.
RBF and Matern accept scalar or per-feature length scales. Matern supports all
positive finite nu values and positive infinity, with closed forms for 0.5, 1.5,
2.5 and infinity and a log-scaled Bessel K evaluation otherwise.

Kernel expressions support sum, product and integer/real powers, including
nonnegative real64 constants on either side of + or *. Flat internal expression
storage gives independent value copies without recursive allocation ownership.
The public methods are `%matrix(x[,y])`, `%diag(x)`, `%gradient(x)`, `%theta()`,
`%with_theta(theta)`, `%n_dims()` and `%is_set()`. Gradients are with respect to
log hyperparameters, in source order: left before right in compositions, alpha
before length scale for RQ, length scale before periodicity for periodic kernels.
Fixed flags omit whole parameter groups from theta and gradient width, including
all coordinates of a fixed anisotropic length scale. Power exponents and Matern
nu are not included in theta, matching sklearn. `%with_theta` returns a copy.

White noise contributes only on the diagonal of auto covariance. A supplied Y
always produces zero white cross covariance, even for identical arrays. Kernel
diagonals use a direct O(samples * expression_nodes) path. Gradients accept only
auto covariance, represented by their x-only signature. Length scales, RQ alpha
and periodicity must be finite and positive; constants, white noise and dot-product
sigma_0 finite and nonnegative. Zero active scalar values use negative-infinite
log theta, as in the source. Invalid real powers raise a diagnostic; gradients
at zero bases use zero for nonnegative powers. Inputs remain nonempty finite
dense real64 matrices. Hyperparameter bounds and optimizer integration,
PairwiseKernel/CompoundKernel adapters and sparse/object inputs are not provided.

Both `gaussian_process_regressor` and `gaussian_process_classifier` accept an
optional `%kernel` expression. This replaces their legacy amplitude/length-scale
RBF configuration (and GPR white-noise field). When no expression is set, existing
scalar controls retain their meaning. Fit clones the configured kernel into an
internal snapshot, so later input-kernel or scalar edits do not alter an existing
fit; a subsequent fit uses the changed configuration. This also fixes the older
GPR behavior of consulting live scalar kernel controls during prediction.
GPR now supports unfitted prior predictions: zero means, kernel covariance and
kernel standard deviations. Posterior covariance is returned without diagonal
clipping, matching sklearn; standard deviations clip negative roundoff variances.
The existing scalar alpha, vector-target regression, binary/OVR classification,
normalization and fixed-parameter training scope remains. Automatic hyperparameter
optimization, GPR multiple targets, vector alpha and posterior sampling remain
outside this increment. Set `optimizer=None` for the Python reference equivalent.

General Matern values use an adapted MIT SciFort Bessel helper from
`c:/fortran/scifort/src/special/bessel_k.f90`, source SHA256
`b03ec89063b912c46e2d0899d79f10eb0375b81cc978ffb2808fc82798d0a53d`.
Its copyright header is retained and the full license is in LICENSES/scifort-MIT.txt.
Only the implementation support module is imported; SciFort is not linked and
its additional Bessel APIs are not exported by the facade. General Matern length
gradients use an analytic Bessel-ratio identity. The pinned source approximates
these with a 1e-10 forward difference; comparisons therefore also use independent
1e-5 central differences of sklearn kernel values. This numerical choice adds no
extension algorithm. All kernel algorithms have sklearn counterparts; original
native interfaces remain MIT with retained sklearn BSD notices.

The inventory remains 217 core estimator/transformer/splitter types, with 129
source modules and 130 native test programs. Kernel expression/evaluation storage
is counted separately from that estimator inventory. There are 109 core reference
suites retaining 440,097 groups, plus the separate 42-group legacy RNG suite.
Twenty-seven new rejection cases bring 19 isolated registries to 3,205 cases.

The new suite retains 9,132 groups over 876 fixtures: 300 standalone kernel,
288 GPR and 288 GPC fixtures. It compares auto/cross matrices, diagonals, theta,
fixed groups, gradients, cloning, shifted parameters, prior/posterior predictions,
uncertainty, Cholesky/dual state, binary/multiclass probabilities and likelihoods,
fitted snapshots and refits. There are also 45 independent general-Matern central
gradient groups and 48 SciPy log-Bessel support points within the stated group
count. Duplicate rows, zero scalar kernels, anisotropy and nested expressions are
retained. The development central-gradient maximum error was 3.82e-10; source
forward-difference gradients differed by up to 1.22e-4 due to their small step.
The source comparison uses 3e-4 tolerance for those noisy gradients and stricter
central-difference checks, rather than claiming bitwise equivalence.

Run `python validation/compare_gp_kernels.py --profile checked`; use
`--profile release` or `--executable` for completed binaries. The existing
`compare_gp_classification.py` now accepts the same profile/executable arguments,
so regression gates never need concurrent fpm invocations. See
`example/gaussian_process_kernels.f90` for a complete public example.


The GP kernel suite additionally retains three constant-kernel multiclass tie
certificates per profile. Each has three equally populated classes and a constant
Gram/cross matrix: the OVR heads are permutation-equivalent, so exact normalized
probabilities are 1/3 for every class. Source and native probabilities must both
satisfy this symmetry within 64*epsilon*sum(abs(Laplace_erf_coefficients)), and
reported labels must belong to the analytically tied class set. Observed source
and native labels, class counts and errors are retained in the reports. Optimized
reductions select a different tied class in these fixtures; all other label
comparisons remain exact. These three certificate groups are included in the
9,132-group total; no fixture is dropped or relabeled.


Final verification passed all 130 native test programs in checked and optimized
fpm builds. Both canonical profiles passed all 9,132 new kernel/GP result groups
over 876 fixtures, 27 new rejection checks, 312 existing GP classification groups
and 22 existing GP diagnostic checks. The checked integration gate passed 192
comparisons. Three retained constant-kernel multiclass fixtures per profile have
analytic tie certificates; optimized labels differ from the source while both
probability arrays satisfy the exact 1/3 symmetry to within 6e-14. Every other
label comparison is exact. Both public examples matched independent sklearn
means, standard deviations and class probabilities at their printed precision.

The final audit verified strict reports, identical profile check lists, the
three explicit certificates, canonical executable/validator and installed source
hashes, the retained SciFort license/import provenance, and 265 compiled content
digests per profile (263 core/test/vendor files plus the new driver and example).
Evidence is retained under validation/results as gp_kernels_final_audit.json,
gp_kernels_{checked,release}.json, gp_kernels_invalid_*.json,
gp_classification_{checked,release}.json, gp_legacy_invalid_*.json,
gp_kernel_example_*.json, gp_kernels_*_tests.log, gp_kernels_*_gates.log and
gp_kernels_checked_reference_final.log. Other historical reference suites were
retained rather than all rerun for this increment. The broader sklearn
translation remains ongoing.


## Multi-target GP regression and marginal-likelihood kernel fitting

`gaussian_process_regressor` now supports matrix targets, observation-specific
noise, log-marginal likelihood evaluation/gradients and opt-in bounded L-BFGS-B
kernel fitting with seeded restarts. Algorithms adapt sklearn 1.6.1
`gaussian_process/_gpr.py`, with existing kernel derivatives and the already
vendored L-BFGS-B 3.0 implementation. Native defaults retain optimizer='none';
set `%optimizer='fmin_l_bfgs_b'` to learn free parameters in an explicit kernel.
Legacy scalar RBF controls remain fixed, matching the no-free-parameter default
kernel convention. GPC optimization is not added in this increment.

Use `%fit(x,y)` with a real vector or (samples,targets) matrix. Scalar `%alpha`
is retained; an allocated `%alpha_values(:)` overrides it and accepts one value
(broadcast) or one per training row, all finite and nonnegative. Noise is added
in normalized target units without target-scale adjustment, as in sklearn. The
same covariance hyperparameters are shared across independent output columns,
and joint likelihood is the sum of their likelihoods. `%n_targets=0` means
automatic; a positive count configures prior output width and validates fitted
width. `%n_targets_` and `%n_features_in_` describe the fitted data.

`%predict_multi` and `%predict_std_multi` always return (samples,targets)
matrices; `%predict_cov_multi` returns (samples,samples,targets). This explicit
rank remains for a one-column matrix, whereas Python squeezes one-target
predictions. Existing vector prediction/std/cov methods work with one target and
raise a clear diagnostic for multiple targets. Matrix `%score` averages target
R2 scores uniformly, with optional finite nonnegative sample weights; one-row
R2 returns NaN. Covariance and standard deviation scale independently with each
target's normalization. Matrix normalization clips near-zero scales; vector
normalization replaces only exact zeros, following the source distinction.
Scalar/one-column fast-axis reductions use the existing NumPy-compatible pairwise
sum contract, including the non-binary constant-target regression case.

New fitted storage includes normalized `%y_train_`, `%alpha_multi_`, target mean/
scale vectors, and public `%kernel_`. `%alpha_`, `%y_train_mean_` and
`%y_train_std_` remain authoritative for single-target prediction compatibility;
matching matrix fields retain copies for diagnostics. Input `%kernel` remains
unchanged by fitting; learned parameters are in `%kernel_`.

`value=model%log_marginal_likelihood(theta,gradient,clone_kernel)` evaluates the
joint likelihood and optional derivative with respect to log kernel parameters.
With no theta it returns the cached fit likelihood and cannot return a gradient.
The default clone_kernel=true leaves fitted state intact; false updates fitted
kernel parameters, matching the source's side effect, without rebuilding cached
prediction factors. A non-positive-definite candidate returns negative infinity
and a zero gradient, while an invalid final fit raises a diagnostic.

`gp_kernel%bounds()` returns log bounds in (n_dims,2) form, including (0,2) for an
all-fixed expression. `%with_bounds(log_bounds)` returns a copy. Initial free
bounds default to log([1e-5,1e5]) for each parameter; composed kernels preserve
child bounds and theta clones retain them. This typed log-matrix API represents
Python's per-hyperparameter positive-domain constructor bounds. Ordered finite
or correctly signed infinite endpoints are accepted; random restarts require
finite bounds. Bounds remain in the same order as theta/gradient.

The optimizer uses ten L-BFGS correction vectors, default projected-gradient
tolerance 1e-5, ftol=2.220446049250313e-9, and 15,000 iteration/function limits,
matching SciPy defaults. `%n_restarts_optimizer` adds log-uniform initial points
from the local legacy RNG, seeded by `%random_state` (default zero). The first
run starts from the configured theta; the lowest objective is selected with
first-best tie behavior. Controls `%tol`, `%ftol`, `%max_iter` and `%max_fun`
allow explicit solver settings. Diagnostics include run points/objectives,
iteration/evaluation totals, convergence and the chosen termination message.
Optimizer paths and iteration counts need not be bit-identical to SciPy's C
translation. General Matern derivatives retain the prior analytic implementation
rather than sklearn's noisy 1e-10 numerical difference.

The inventory remains 217 core estimator/transformer/splitter types and 129 core
source modules, with 132 native test programs. A new suite brings 110 core
reference suites to 448,167 retained groups, plus the separate 42-group legacy
RNG suite. Twenty-eight new rejection cases bring 20 registries to 3,233 cases.
The suite retains 8,070 groups over 384 fixtures, including 96 optimization
comparisons across nine kernel configurations, scalar/vector noise, vector/
matrix/one-column targets, normalization, priors, weighted scores, bounds,
likelihood clone semantics and non-PD candidates. Posterior state is compared
strictly at the native learned theta; learned objectives and predictions are
also compared with independent SciPy optimizations, including every restart.
Likelihood gradients are checked with independent sklearn central differences.
No fixture is dropped for an optimizer-path difference.

Reproduce with `python validation/compare_gpr_advanced.py --profile checked`,
or use `--profile release` / `--executable` for an existing driver. See
`example/gaussian_process_regression.f90` for multi-target fitting with learned
kernel parameters and heteroscedastic noise. Posterior sampling, sparse data,
metadata routing and general GP classifier kernel optimization remain outside
this increment. No new dependency or public extension algorithm is introduced.
Original contributions remain MIT with the existing sklearn, NumPy and optimizer
notices retained.


Final verification passed all 132 native test programs in both checked and
optimized fpm builds. Each canonical profile passed 8,070 new GPR result groups
over 384 fixtures, including 96 independent optimization comparisons and 28
isolated rejection checks. The maximum observed optimized-likelihood difference
from SciPy was 1.16e-9 in each profile. Both profiles also passed all 9,132 earlier
kernel/GP groups (retaining the three documented constant-kernel tie certificates),
27 kernel rejections, 312 earlier GP classification groups and 22 existing GP
diagnostic checks. The checked integration gate passed 192 comparisons.

Both compiled examples matched independent sklearn predictions, standard
deviations and joint likelihood (15.567164098309043). The audit verified strict
reports, identical profile check lists, optimizer comparison records, retained
tie certificates, canonical executable/validator and installed source hashes,
and 267 compiled content digests per profile (265 core/test/vendor files plus
the new driver and example). Evidence is retained under validation/results as
gpr_advanced_final_audit.json, gpr_advanced_{checked,release}.json,
gpr_advanced_invalid_*.json, gpr_advanced_example_*.json, the affected GP reports,
and gpr_advanced_*_tests.log / gpr_advanced_*_gates.log. Other historical reference
suites were retained rather than all rerun for this increment. The broader
sklearn translation remains ongoing.


## GP classifier likelihood gradients, kernel fitting and warm starts

`gaussian_process_classifier` now evaluates binary Laplace marginal-likelihood
gradients and optionally learns kernel parameters using bounded L-BFGS-B with
seeded restarts. Multiclass one-vs-rest heads are optimized independently and
retain separate fitted `%kernels_(:)` in sorted class order; binary fits have
one kernel. Prediction uses each head's own covariance and diagonal. Original
scalar RBF controls remain supported and fixed. Native optimizer='none' remains
the default; set `%optimizer='fmin_l_bfgs_b'` for free parameters of an explicit
kernel expression. No classifier algorithm outside sklearn is added.

Controls `%n_restarts_optimizer`, `%random_state`, `%max_iter`, `%max_fun`, `%tol`
and `%ftol` follow the GP regressor's native interface, using the existing
L-BFGS-B 3.0 implementation with ten correction vectors, gtol=1e-5,
ftol=2.220446049250313e-9 and default 15,000 iteration/evaluation limits. Each
head receives the same scalar RNG seed, matching sklearn's independently cloned
binary estimators. There is a pinned-source distinction: sklearn 1.6.1 GPC
exponentiates its uniform draw in log bounds before passing it as optimizer
theta. This differs from GPR, which passes the log-uniform draw directly. Native
GPC deliberately preserves that exponentiation and subsequent bound clipping;
it does not silently substitute the GPR convention. Raw initial points are
retained in `%optimization_initial_points_` and compared directly with sklearn.

`%warm_start` reuses the last posterior mode while evaluating candidate kernels.
It defaults false. A new fit resets caches, as sklearn's outer classifier
recreates its binary estimators. Fit performs the source's likelihood evaluation
and separate predictive posterior-mode calculation. Fitted mode-control snapshots
preserve the settings used at fit even if parent controls are edited later.
`%posterior_mode_` retains fitted prediction state; `%mode_cache_` tracks the
separate likelihood-evaluation cache, including updates on cloned evaluations.
`%fit_initial_cache_` records the cache entering the final predictive mode solve.

`value=model%log_marginal_likelihood()` remains callable from an intent(in) model
and returns the cached fit value. With a theta vector, optional allocated output
`gradient` returns the binary derivative, and optional `clone_kernel` defaults
true. Cloned evaluation preserves fitted kernel/prediction factors while updating
the mode cache, as in the source. False updates fitted kernel parameters without
rebuilding prediction factors. Multiclass evaluation accepts either one shared
theta or class-major concatenated head parameters and returns mean likelihood;
multiclass gradients raise a diagnostic, matching sklearn. Requesting a cached
gradient without theta is rejected via `eval_gradient=.true.`.

The likelihood derivative implements GPML Algorithm 5.1, including the third
logistic derivative and posterior-mode correction. Posterior iteration follows
Algorithm 3.1 and retains the pre-update pi/W/L and previous likelihood at the
source stopping point. NumPy-compatible pairwise sums, source multiplication
grouping and a standard C-runtime log1p call reduce numerical differences without
adding a package dependency. General Matern gradients retain the existing analytic
Bessel implementation; independent central differences additionally validate them.

Diagnostics distinguish posterior convergence from optimizer convergence and
retain per-head iteration/evaluation totals, per-run points/objectives, per-run
termination messages and convergence flags, and the selected run's status.
`%optimization_eval_cache_` retains the input cache associated with each returned
run objective, including a solver-restored accepted point after line-search stops.
Warm objectives depend on cache history and truncated Newton convergence, so
validation replays sklearn's mode solve from the same recorded input state for
endpoint and final-prediction transitions. Independent full sklearn fits still
check learned likelihoods and probabilities, every restart result and initial
point. No fixture is removed for a path difference. Explicit likelihood evaluations leave the cached fit likelihood unchanged.

The inventory remains 217 core estimator/transformer/splitter types and 129 core
source modules, with 134 native test programs. A new suite brings 111 core
reference suites to 457,011 retained groups, plus the separate 42-group legacy
RNG suite. Thirteen new rejection cases bring 21 registries to 3,246 cases.
The new suite retains 8,844 groups over 324 fixtures: two/three classes, nine
kernel configurations, three seeds and six cold/warm/optimized/truncated-mode
configurations. It includes 96 independent optimization comparisons, 288 restart
endpoint checks, analytic/central gradients, posterior state, shared/packed theta,
clone/mutation semantics, refits and exact labels. Development learned-probability
differences were below 5e-6 and selected likelihood differences below 8e-7;
conditional posterior and cache transitions agreed to near machine precision.

Run `python validation/compare_gpc_advanced.py --profile checked`; use
`--profile release` or `--executable` for completed binaries. See
`example/gaussian_process_classification.f90` for learned multiclass kernels.
One-vs-one GPC, multiclass likelihood gradients, object/sparse kernels and metadata
routing remain outside this increment. Original native interfaces and diagnostic
storage remain MIT, with sklearn, NumPy and the existing optimizer notices
retained. No external repository or dependency is imported.


Final verification for this increment passed in checked and release profiles:
all 134 native test programs; 8,844 new GPC groups over 324 fixtures, including
96 independent optimization comparisons and 288 restart checks; the existing
8,070 advanced GPR, 9,132 kernel and 312 GPC groups; and 90 affected invalid-input
cases (13 new, 28 GPR, 27 kernel and 22 legacy GP cases). The checked integration
suite also passed all 192 groups. The three constant-kernel floating-point tie
certificates per profile remain explicit; every other classification fixture
retains exact-label comparison. The updated multiclass example matched sklearn
in both profiles after explicitly rebuilding its stale checked executable.

The final ignored `validation/results/gpc_advanced_final_audit.json` validates
269 compiled-content digests per profile (129 core modules, four vendor modules,
134 native test programs, the new driver and the updated example), executable
and validator hashes, installed sklearn source hashes, strict JSON reports and
identical checked/release check inventories. The 457,011 retained core
reference groups are the cumulative inventory; only the affected suites
listed above and the checked integration suite were rerun for this increment.


## Gaussian process one-vs-one classification

Added sklearn's `GaussianProcessClassifier(multi_class="one_vs_one")` path to the
existing native classifier. Each sorted class pair trains on its own row subset
and retains its own fitted kernel, mode state and optimizer diagnostics. Binary
behavior is unchanged. Prediction uses pair votes plus sklearn's bounded
positive-probability confidence correction and first-class final tie behavior.
Multiclass one-vs-one probability requests are rejected, as in sklearn. Refit
transitions between OVO, OVR and binary models reset subset maps and caches.

`validation/compare_gpc_ovo.py` retains 15,600 groups over 324 fixtures: two,
three and four classes, imbalanced class sizes, shuffled training rows, nine
kernel configurations, three seeds, fixed/learned and cold/warm fits, and
truncated posterior solves. One seed also shuffles labels to exercise voting
cycles and confidence tie breaking. Eleven queries select a different class
with confidence correction than with first-best raw votes; native labels match
sklearn exactly. The suite includes 144 independent sklearn optimization
comparisons, 720 independently replayed restart objectives, input row maps,
per-pair posterior factors, shared/packed likelihoods, clone/mutation behavior
and six isolated invalid-input cases. No classification fixture is excluded.
The prior 8,844-group GPC suite also passed during development.

sklearn 1.6.1 checks packed theta against class count even in OVO mode; this fails
for complete packed kernels when there are four or more classes. The native API
accepts pair count instead. Packed evaluations are checked against each source
binary estimator directly; shared evaluations use the full source outer model.
Stateful warm optimizer endpoints and final mode transitions reuse the previously
documented independent sklearn cache replay at recorded native input states.
Full independent learned-model likelihoods and labels are checked as well.

The inventory remains 217 core estimator/transformer/splitter types and 129 core
source modules, with 135 native test programs. There are now 112 core reference
suites with 472,611 retained groups, plus the separate 42-group legacy RNG suite.
Six new rejection cases bring 22 registries to 3,252 cases. These are cumulative
coverage counts, not a claim that every historical suite was rerun in this batch.

Run `python validation/compare_gpc_ovo.py --profile checked`; use `--profile
release` or `--executable` for completed binaries. See
`example/gaussian_process_ovo.f90`. The existing sklearn BSD notices are retained;
native interfaces, storage and tests remain MIT. No new dependency, public type
or extension algorithm is added. Sparse inputs, parallel head fitting and
multiclass likelihood gradients remain outside this implementation.


Final verification passed in checked and release profiles: all 135 native test
programs; 15,600 OVO GPC, 8,844 prior advanced GPC, 8,070 advanced GPR, 9,132
kernel and 312 legacy GPC groups (41,958 affected reference groups per profile);
and 96 affected invalid-input cases (6 OVO, 13 advanced GPC, 28 advanced GPR,
27 kernel and 22 legacy GP). The checked integration suite passed 192 groups.
The new example matched sklearn labels, pair training sizes and mean likelihood
in both profiles. All new OVO label comparisons are exact, including all eleven
confidence tie-breaking queries. The three existing constant-kernel roundoff
tie certificates per profile in the earlier kernel suite remain explicit.

The ignored `validation/results/gpc_ovo_final_audit.json` verifies strict JSON
reports, executable/validator and installed sklearn source hashes, identical
checked/release check inventories and 270 compiled-content digests per profile:
129 core modules, four vendor modules, 135 native test programs, the new driver
and the new example. It also verifies 144 optimization comparisons and 720
restart checks per profile. Only the affected reference suites and checked
integration suite listed here were rerun; 472,611 groups is the cumulative core
coverage inventory. Test, gate, example-build and audit logs are retained beside
the ignored reports.


## Gaussian process sampling

Added `sample_y` and `sample_y_multi` to the existing GP regressor, covering
unfitted priors and fitted posteriors, scalar/multiple targets, normalized
responses, heteroscedastic alpha, fitted kernel snapshots, zero requested draws,
singular/repeated-query and zero covariance. Each call owns a legacy scalar-seeded
RNG (default zero), consumes target/sample/query-major normals and leaves the
intrinsic RNG and fitted model unchanged. The covariance uses a scaled native
Jacobi SVD; tiny/huge scalar covariance fixtures reach 1e-240 and 1e240 without
squaring those values directly.

The native SVD basis can differ in sign or orientation from NumPy LAPACK, so
seeded sample paths are reproducible natively but need not match sklearn paths.
`validation/compare_gp_sampling.py` checks the intended Gaussian law and exact
normal-stream scheduling without assuming a basis: whitening produces draw Gram
matrices matching the source-seeded standard normals, and nullspace projections
stay within numerical covariance-error bounds. This works for full rank, repeated
singular values and rank-deficient kernels. Independent 20,000-draw fixtures check
means, full covariance and cross-target independence against analytic sklearn
moments with eight-standard-error bounds; sklearn's own draws are checked against
the same bounds. Scalar native paths are compared directly with NumPy draws.

The suite retains 6,450 groups over 588 fixtures, including 108 moment fixtures:
13 ordinary kernel configurations, three seeds, one/three targets, six prior/
posterior/normalization/snapshot/noise configurations, zero/one/seven draws and
additional tiny/huge scalar priors. Twelve singleton-query prior fixtures record
sklearn 1.6.1's sample_y shape failures and validate the intended draws directly
with NumPy multivariate_normal; no fixture is silently omitted. Eight isolated
rejections cover invalid sample counts, seeds, target count, query values/features,
output capacity and nonfinite predictive moments.

The inventory remains 217 core estimator/transformer/splitter types and 129 core
modules, with 136 native test programs. There are 113 core reference suites with
479,061 retained groups, plus the separate 42-group legacy RNG suite. Eight new
rejection cases bring 23 registries to 3,260 cases. These are cumulative counts;
final verification below lists the suites actually rerun for this increment.

Run `python validation/compare_gp_sampling.py --profile checked`, or use
`--profile release` / `--executable` for completed binaries. See
`example/gaussian_process_sampling.f90`. Original contributions remain MIT;
existing sklearn/NumPy notices apply. No dependency, public type or extension
algorithm is added. Sparse/object inputs and RandomState objects remain outside
the native interface.


Final verification passed in checked and release profiles: all 136 native test
programs; 6,450 sampling, 15,600 OVO GPC, 8,844 advanced GPC, 8,070 advanced GPR,
9,132 kernel and 312 legacy GPC groups (48,408 affected reference groups per
profile); and 104 affected invalid-input cases (8 sampling, 6 OVO, 13 advanced
GPC, 28 advanced GPR, 27 kernel and 22 legacy GP). The checked integration suite
passed 192 groups. The sampling example matched sklearn's posterior mean and
source-seeded whitened draw Gram matrix in both profiles. The three existing
constant-kernel roundoff tie certificates per profile remain explicit in the
earlier kernel suite.

The ignored `validation/results/gp_sampling_final_audit.json` verifies strict
JSON reports, executable/validator and installed sklearn source hashes, identical
checked/release check inventories and 271 compiled-content digests per profile:
129 core modules, four vendor modules, 136 native test programs, the new driver
and the new example. It verifies all 588 sampling fixtures, including 108 moment
fixtures, ranks zero/one/four/five and twelve recorded source singleton-prior
shape failures. The largest development seeded Gaussian Gram discrepancy was
below 1.7e-12. Native SVD trajectories remain distinct from a promise of NumPy
LAPACK trajectory identity. Only the affected suites and checked integration
suite listed here were rerun; 479,061 groups is the cumulative core inventory.
Test, gate, example-build and audit logs are retained beside the ignored reports.


## FactorAnalysis rotations and randomized SVD

Added varimax/quartimax orthogonal loading rotation and randomized-SVD fitting to
the existing FactorAnalysis estimator. Native defaults retain full SVD and no
rotation. Explicit `iterated_power` (default 3) and scalar seed (default 0) reuse
the native randomized SVD core, including no normalization for powers 0--2 and
pivoted LU for higher powers. Existing consumers retain their previous automatic
4/7-power path. The fit seeds a local RNG once and advances it across iterations.
Full SVD is the existing native Jacobi algorithm, not a new LAPACK dependency.

Rotation follows sklearn's post-fit orthomax iteration: varimax's column-energy
penalty or quartimax's unpenalized fourth moment, orthogonal polar updates,
relative 1e-6 stopping and at most 100 steps. Native zero singular columns are
completed orthogonally. Fitted noise, covariance, precision and likelihood history
are preserved; factor coordinates change. Native matrix/iteration/convergence
rotation diagnostics reset on refit, including zero-factor no-ops. Nonfinite
tolerances/noise initialization now raise explicit diagnostics before arithmetic.

`validation/compare_factor_options.py` retains 6,564 groups over 396 fixtures:
three seeds, seven tall/wide/full/zero/singleton dimension configurations, full
SVD and randomized powers 0/1/2/3/7, all three rotation choices, normal/custom
noise initialization, converged/capped fits and additional all-zero loadings.
There are 228 independent rotation comparisons, with source-helper SVD calls
instrumented to check diagnostic iteration counts. Checks include fitted
loadings and factors aligned only for signed permutations, raw loadings,
noise, means, history, covariance, precision, scores, orthogonality, covariance
preservation, criterion improvement and an independent rotation replay from the
native unrotated state. All fixtures remain in the suite. Fifteen isolated
rejections cover solver/rotation/iteration/seed controls, nonfinite inputs,
noise shape/range, component count and fitted/query contracts.

Explicit certificates retain capped tol=0 convergence-flag differences only
when both final likelihood improvements lie within 128*eps times the larger of
n*p and the final source likelihood magnitudes, with the same iteration count.
One-feature/one-factor varimax has identically zero criterion; a source power-
rounding residual can stop its helper in two steps while the native exact-zero
gradient reaches 100. Those diagnostic differences retain an analytic certificate
and strict factor/covariance/orthogonality checks. Development retained twelve
fit-stop and one trivial-rotation certificates; final profile reports record the
actual certificates. No model comparison is silently omitted or relaxed.

The inventory remains 217 core estimator/transformer/splitter types and 129 core
modules, with 137 native test programs. There are 114 core reference suites and
485,625 retained groups, plus the separate 42-group legacy RNG suite. Fifteen
new rejection cases bring 24 registries to 3,275 cases. These are cumulative
counts; the final verification record lists the suites actually rerun.

Run `python validation/compare_factor_options.py --profile checked`, or use
`--profile release` / `--executable` for completed binaries. See
`example/factor_analysis_rotation.f90`. Original interfaces and diagnostic
storage remain MIT with existing sklearn and NumPy notices retained. No source
package, dependency or extension algorithm is added. Dense real64 arrays,
scalar integer seeds, the existing min(samples,features) component limit and
zero-singular-value likelihood floor remain the documented native scope.


Final verification passed in checked and release profiles: all 137 native test
programs; 6,564 new FactorAnalysis groups, 204 existing regression/factor/
projection groups, 744 dictionary-learning groups, 6,500 MiniBatchNMF groups and
1,620 t-SNE groups (15,632 affected reference groups per profile); and 17 affected
invalid-input cases (15 new and two legacy FactorAnalysis cases). The checked
integration suite passed 192 groups. The rotation example matched sklearn's
latent shape, fitted iteration count and mean likelihood in both profiles.
The separate legacy RNG reference suite was not rerun; dictionary validation
retained all 744 algorithm groups using completed canonical binaries.

Both profiles retained twelve bounded fit-stop certificates. One trivial
one-feature varimax iteration certificate was retained in checked mode and four
in release mode; these are the same identically zero criterion geometry, with
all loading, model, criterion and orthogonality comparisons retained. All 228
independent rotation comparisons passed in each profile. Existing t-SNE
trajectory/tie certificates remain in the copied profile reports.

The ignored `validation/results/factor_options_final_audit.json` validates strict
JSON reports, executable/validator and installed sklearn source hashes, matching
checked/release fixture inventories, every retained new certificate and 276
compiled-content digests per profile: 129 core modules, four vendor modules,
137 native test programs, five affected reference drivers and the new example.
The copied t-SNE inventories match by seed/case/name; their numeric error and
trajectory details are retained separately per profile. Only the affected suites
and checked integration suite listed here were rerun; 485,625 groups is the
cumulative core reference inventory. Test, gate, example-build and audit logs
are retained beside the ignored reports.


## PCA dimension selection and probabilistic models

Added variance-fraction and Minka MLE dimension selection, explicit zero-factor
models, covariance, precision and per-sample/mean Gaussian likelihoods to the
existing full-SVD PCA. Default count-mode behavior remains unchanged, including
legacy zero count selecting all available components. Active dimension modes
map sklearn's integer/float/string n_components union to typed Fortran controls.
Fitted n_components_ and MLE candidate diagnostics support inspection/refits.
Zero-dimensional inverses now return the mean. The native scoring path uses a
Cholesky factor and requires a proper positive-definite covariance; covariance
and transforms remain available for singular models. The pinned sklearn 1.6.1
whitened generative covariance scaling is preserved deliberately.

The new `validation/compare_pca_probability.py` retains 5,270 groups over 252
fixtures: five tall/wide/single-feature shapes, three seeds, count/all/variance/
MLE/zero choices, whitening on/off, constant matrices and exact right-sided
fraction thresholds. It compares fitted dimensions, means, spectral statistics,
noise, covariance, transforms/inverses, precision, scores and Minka candidate
likelihoods with independent installed sklearn fits. There are 230 proper
Gaussian fixtures. Twenty-two singular covariance models retain covariance/
transform checks and a verified singular-domain record; new precision/score
rejections separately verify the method preconditions.

Sixteen nonunique-basis certificates verify orthonormality, eigen equations and
membership in the independent source eigenspaces before source operators are
evaluated in the native basis. For zero-variance whitening, training projections
are compared before the epsilon division so floating-point null coordinates
are not treated as meaningful. Eight undefined-ratio certificates retain the
existing native zero convention for source NaN 0/0 ratios. Selected dimensions,
spectra, covariance and proper Gaussian scores remain independently checked;
no fixture is silently dropped. Exact Minka spectral ties raise an explicit
native diagnostic, matching the source math-domain failure.

Nineteen isolated rejection cases cover active modes/fractions, MLE shape/ties,
component/sample/feature constraints, nonfinite inputs, unfitted calls, singular
precision/scoring and zero-dimensional inverse shape. The inventory remains
217 core estimator/transformer/splitter types and 129 core modules, with 138 native
test programs. There are 115 core reference suites with 490,895 retained groups,
plus the separate 42-group legacy RNG suite. Nineteen new rejection cases bring
25 registries to 3,294 cases. These counts are cumulative; the final record below
lists the suites actually rerun for this increment.

Run `python validation/compare_pca_probability.py --profile checked`, or use
`--profile release` / `--executable` for completed binaries. See
`example/probabilistic_pca.f90`. Original interfaces/diagnostic storage remain MIT
with existing sklearn BSD notices retained. No dependency, source package,
public estimator type or extension algorithm is added. Randomized/ARPACK/
covariance-eigh PCA solvers, sparse inputs and singular Gaussian pseudodensities
remain outside this increment.


Final verification passed in checked and release profiles: all 138 native test
programs; 5,270 new PCA groups over 252 fixtures, 264 existing IncrementalPCA
groups and 192 existing integration groups (5,726 reference groups per profile);
and all 19 new invalid-input cases. The new example matched sklearn's Minka
rank (six), noise variance, mean likelihood and 90-percent variance rank (two)
in both profiles. Legacy integer-count PCA integration behavior remains checked.

Both profiles retain sixteen independently verified nonunique-basis certificates,
eight undefined-ratio certificates and twenty-two singular-domain model records,
with 230 proper Gaussian fixtures checking precision and likelihood. Every
source eigenspace membership and eigen residual is checked within its recorded
bound; no null-space coordinate identity is claimed. No input fixture is omitted.

The ignored `validation/results/pca_probability_final_audit.json` verifies strict
JSON reports, executable/validator and installed sklearn source hashes, matching
checked/release check inventories and 275 compiled-content digests per profile:
129 core modules, four vendor modules, 138 native test programs, three affected
reference drivers and the new example. All new certificates are validated by
the audit. Only the suites listed here were rerun; 490,895 retained core groups
is the cumulative reference inventory. Test, gate, example-build and audit logs
are retained beside the ignored reports.


## PCA full, randomized, covariance and automatic solvers

Added sklearn's randomized, covariance_eigh and automatic solver paths to the
existing full-SVD PCA, retaining the native full default. Randomized controls
cover automatic/explicit powers, oversampling, auto/none/LU/QR normalization and
scalar seeds. Fitted solver diagnostics and source-specific randomized left
training coordinates are exposed. Covariance PCA forms the uncentered Gram
correction, uses the existing symmetric eigensolver and clips negative roundoff
values; residual noise uses the full feature spectrum. Automatic routing follows
the pinned source's tall/shape/rank rules. MLE, fraction and zero modes are
supported by full/covariance paths, with unsupported randomized modes diagnosed.
Orthogonal null completion is restricted to the new PCA caller; older core
consumers preserve their previous defaults and handling.

`validation/compare_pca_solvers.py` retains 11,578 groups over 686 PCA and 380
shared-core fixtures. PCA coverage includes tall/wide and large routing cases,
all four solver settings, three seeds, integer/all/fraction/MLE/zero selection,
whitening, low-power projected training coordinates, covariance/noise statistics,
inverses and zero covariance. Direct core checks cover all normalizers, zero/
positive oversampling, auto/explicit powers, both sign settings, requests above
matrix dimension, orthogonality and exact normal/uniform RNG continuity.
Fifty-six null-basis certificates check source null-space membership and native
orthogonality while retaining nonzero-coordinate comparisons. Eight undefined
ratio certificates preserve the existing native zero convention for source NaNs.
Thirteen isolated rejections cover new PCA/core controls and integer-capacity
limits. No fixture is dropped for a solver or basis difference.

The inventory remains 217 core estimator/transformer/splitter types and 129 core
modules, with 139 native test programs. There are 116 core reference suites with
502,473 retained groups, plus the separate 42-group legacy RNG suite. Thirteen
new rejections bring 26 registries to 3,307 cases. These are cumulative counts;
the final verification record below lists the suites actually rerun.

Run `python validation/compare_pca_solvers.py --profile checked`, or use
`--profile release` / `--executable` for completed binaries. See
`example/pca_solver_selection.f90`. Existing PCA dimension/probability behavior
and consumers of the randomized SVD core are regression-checked. Original
interfaces/null-completion storage remain MIT with existing sklearn and NumPy
notices retained. No dependency, public estimator type or extension algorithm is
added. ARPACK, sparse inputs and alternate dtypes remain outside this increment.


Final verification passed in checked and release profiles: all 139 native test
programs; 11,578 new PCA/core groups, 5,270 prior PCA probability/dimension groups,
6,564 factor-options groups, 204 regression/factor/projection groups, 744 dictionary
learning groups, 6,500 MiniBatchNMF groups, 1,620 t-SNE groups, 264 IncrementalPCA
groups and 192 integration groups (32,936 reference groups per profile). All 49
affected rejections passed: 13 new, 19 prior PCA, 15 factor-options and two legacy
factor cases. The new example matched sklearn's randomized automatic choice,
explained variances and 520-by-three training shape in both profiles.

The new reports retain 56 null-basis and eight undefined-ratio certificates per
profile. Prior PCA reports retain their 16 basis/eight ratio certificates and
22 singular-domain records. Factor reports retain twelve bounded fit-stop
certificates in each profile, plus one trivial-varimax diagnostic certificate
in checked mode and four in release mode. Existing t-SNE trajectory/tie records
remain in the copied profile reports. All new fixtures remain checked; the
separate legacy RNG reference suite was not rerun, while the direct core fixtures
verify post-call normal/uniform stream continuity independently.

The ignored `validation/results/pca_solvers_final_audit.json` validates strict
JSON reports, executable/validator and installed sklearn source hashes, matching
checked/release check inventories and 282 compiled-content digests per profile:
129 core modules, four vendor modules, 139 native test programs, nine affected
reference drivers and the new example. It validates all new certificates and
retained PCA/factor evidence; t-SNE inventories match by seed/case/name with
numeric details kept per profile. Only the suites listed above were rerun;
502,473 retained core groups is the cumulative inventory. Test, gate, example-
build and audit logs are retained beside the ignored reports.


## Dense and sparse randomized TruncatedSVD

Added randomized TruncatedSVD with seeded power/oversampling/normalization
controls, dense/CSR generic fitting and inference, actual fitted shape metadata,
and dense inverse transforms. The prior exact dense backend remains the native
default. The shared randomized SVD core now uses one dense/CSR factorization
body, with sparse forward/transpose products and no dense input conversion.
TruncatedSVD does not center input, and fitted variance uses actual right-space
training projections. Sparse variance includes implicit zeros and aggregates
unsorted duplicate cells before centered square accumulation. Inputs are
preserved; only reduced working matrices and fitted components are dense.

`validation/compare_truncated_sparse.py` retains 8,160 groups over 680 fixtures,
including 218 raw duplicate/unsorted/stored-zero cases, tall/wide/single-row
shapes, three seeds, all supported normalizers, low/high powers, requests above
sample count, the legacy dense backend and zero/constant matrices. A CSR case
with shape 5,000 by 100,000 and 10,000 stored entries succeeds without input
densification; a dense input representation would require 4,000,000,000 bytes.
There are 43 nonunique-null-basis certificates and 174 undefined-ratio
certificates. Source zero-total-variance ratios can be NaN or infinity from
projection roundoff; the native zero convention is retained explicitly.

The suite records 216 sklearn 1.6.1 QR validation failures caused by its 'OR'
typo and validates the same source algorithm with that erroneous validation
bypassed. All otherwise valid Source parameters remain unchanged. Duplicate
references canonicalize a separate scipy CSR copy; the encoded native matrix
and its storage remain unchanged. No input fixture is removed. Nineteen
isolated rejection cases cover controls/ranks/capacity, dense and CSR finiteness,
CSR structure/shape, unsupported sparse full fitting, fitted/query contracts and
inverse shape.

The inventory remains 217 core estimator/transformer/splitter types and 129 core
modules, with 140 native test programs. There are 117 core reference suites with
510,633 retained groups, plus the separate 42-group legacy RNG suite. Nineteen
new rejection cases bring 27 registries to 3,326 cases. These are cumulative
counts; final verification below lists the affected suites actually rerun.

Run `python validation/compare_truncated_sparse.py --profile checked`, or use
`--profile release` / `--executable` for completed binaries. See
`example/sparse_latent_semantics.f90`. Original interfaces and sparse products
remain MIT with existing sklearn and NumPy notices retained. No dependency,
public estimator type or extension algorithm is added. Legacy single-feature
exact dense fitting remains; new randomized fitting requires two features as
in sklearn. ARPACK and CSC input remain outside this increment.


Final verification: all 140 native test programs passed in both checked mode
(`-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow`) and release mode.
Each profile passed 41,096 affected reference groups and 68 rejection cases,
including all 8,160 new TruncatedSVD groups and 19 new rejections. The rerun
also covered PCA solvers and probability, FactorAnalysis options, regression
and projections, dictionary learning, MiniBatchNMF, t-SNE, IncrementalPCA and
192 integration comparisons. The separate 42-group legacy RNG suite was not
rerun; the cumulative inventory above does not describe a complete rerun.

The final audit matched reports to validator, upstream-source and executable
hashes and verified 284 compiled source-content digests per profile. It retained
43 null-basis certificates, 174 undefined-ratio certificates, 218 duplicate
storage cases and 216 recorded Source QR-validation defects. A 5,000 by 100,000
CSR fixture with 10,000 stored entries passed; its equivalent dense input would
occupy 4,000,000,000 bytes. Both explicit fpm example runs passed, and the example
validator checked the tied leading singular subspace using its invariant total
variance ratio. Reports, audit and terminal logs are retained under the ignored
`validation/results` directory.


## KernelPCA learned pre-images (2026-11-22)

Added sklearn KernelPCA's optional kernel-ridge inverse fitting and latent-query
reconstruction, with alpha regularization and fitted latent/dual diagnostics.
The uncentered latent kernel reuses the forward model's fitted gamma, and the
linear inverse preserves sklearn's omission of the original feature mean.
Refitting without inverse fitting clears stored inverse state. Precomputed
inverse fitting and singular/indefinite ridge systems are rejected explicitly.

`validation/compare_kernel_preimage.py` retains 4,520 groups across 452 fixtures,
using installed sklearn 1.6.1 and 452 independent ridge-equation residual
certificates. Cases cover linear/RBF/polynomial/cosine/sigmoid kernels, three
seeds, tall/wide matrices, regularization including zero, explicit/default
gamma, automatic/explicit component counts, zero/constant input, one-sample
fits, fit versus fit_transform, training/query/arbitrary-latent reconstruction
and clearing inverse state on refit. Fourteen rejection cases cover finite
controls, alpha bounds, disabled/unfitted/stale inverse state, inverse query
shape/finiteness/emptiness, precomputed fitting, zero-dimensional latent space
and singular/indefinite systems. The indefinite sigmoid rejection is confirmed
with the installed source as well as the native implementation.

Inventory: 217 public core types, 129 core source modules, 141 native test
programs and 118 core reference suites with 515,153 retained groups. The
separate 42-group legacy RNG suite remains excluded. There are 28 invalid-input
registries with 3,340 cases. These counts are cumulative; final verification
below identifies the checks actually rerun for this change.

Run `python validation/compare_kernel_preimage.py --profile checked`, or select
`--profile release` / an explicit `--executable`. The original Fortran interfaces
and Cholesky adaptation remain MIT, with the existing sklearn BSD notice
retained. No additional source package, dependency or extension algorithm is
imported. Dense inverse outputs and the existing eigensolver remain the scope.


Final verification: all 141 native test programs passed in checked mode
(`-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow`) and release mode.
Each profile passed 4,832 reference groups: 4,520 new pre-image groups, all
120 retained graph/neighbor/KernelPCA/Isomap groups and 192 integration groups.
All 14 new rejection cases and four retained KernelPCA/Isomap rejection cases
passed in each profile. The final audit matched validator, upstream-source and
executable hashes, confirmed all 452 ridge certificates per profile and checked
279 compiled source-content digests per profile (129 core modules, 141 tests,
four vendor sources, four affected drivers and the example).

Both explicit fpm example runs passed. The independent example validator
matched sklearn's reconstruction RMSE of 0.00724053 on the 20-row illustrative
fixture; this is not a general reconstruction guarantee. Reports, the final
audit and seven terminal logs are retained in ignored `validation/results`.
The cumulative inventory above is not a full rerun count. Other reference
suites and the separate 42-group legacy RNG reference suite were not rerun.


## SVD linear discriminant analysis (2026-11-22)

Added sklearn's within/between thin-SVD discriminant solver, projection methods,
component selection, rank tolerance and optional covariance storage. The existing
native LSQR default remains; SVD is selected explicitly. Empirical priors and
all retained discriminant axes drive classification, independently of the
requested transform width. Rank-deficient/wide matrices, constant or duplicated
features and singleton classes are supported when within-class rank is nonzero.
The default SVD path allocates no feature-by-feature covariance matrix.

`validation/compare_discriminant_svd.py` retains 11,084 groups over 652 fixtures
and 652 independent within-covariance whitening / between-scatter diagonalization
certificates. Three seeds cover binary and multiclass labels, tall/wide and
single-feature inputs, singleton classes, unequal feature scales, duplicated or
rank-one features, two tolerances, automatic/explicit component counts,
fit/fit_transform, optional covariance, probabilities, predictions, scores,
coefficients and training/query projections. Signs are aligned only for
nonunique SVD axes; binary class-row coefficients/scores are compared in the
source's margin convention. Three coincident-mean cases explicitly preserve
source's undefined (NaN) variance ratios and zero transform rank.

A 12 by 20,000 fixture validates the thin-SVD path with covariance absent in
both implementations; one dense feature covariance would occupy 3.2 GB.
Twenty isolated rejection cases cover controls, shrinkage, dimensions, finite
inputs, fit state, class/sample counts, empty retained rank and stale diagnostics
after LSQR refitting. Two recorded source empty-rank IndexErrors explain the
native diagnostic for zero within variance or an excessively high threshold.

Inventory: 217 core public types, 129 core modules, 142 native test programs,
119 core reference suites and 526,237 cumulative retained groups. The separate
42-group legacy RNG reference suite remains excluded. There are 29 invalid-input
registries with 3,360 cases. Cumulative counts are not a complete rerun claim;
final verification below identifies the checks run for this increment.

Run `python validation/compare_discriminant_svd.py --profile checked`, or use
`--profile release` / `--executable`. Native interfaces and thin-SVD adaptations
remain MIT with existing sklearn BSD notices retained. No new source dependency,
public estimator type or extension algorithm is added. Custom priors and eigen
solver support remain outside this implementation.


Final verification: all 142 native test programs passed in checked mode
(`-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow`) and release mode.
Each profile passed 11,276 reference groups: 11,084 new SVD discriminant groups
and all 192 integration groups, including the retained LSQR LDA and QDA cases.
All 20 new rejection cases passed in each profile. Multiclass coefficients,
intercepts and decision scores were compared directly; only binary outputs
were converted to sklearn's single-margin convention. Axis sign alignment
applied only to discriminant scalings and projected coordinates.

The final audit matched validator, upstream-source and executable hashes,
retained all 652 whitening/diagonalization certificates, three undefined-ratio
certificates and two source empty-rank errors per profile, and checked 278
compiled source-content digests per profile (129 core modules, 142 tests,
four vendor sources, two affected drivers and the example). The 12 by 20,000
wide case passed in both profiles with covariance storage absent; a single
dense feature covariance would require 3,200,000,000 bytes.

Both explicit fpm example runs passed, and their coordinates' dimensions,
variance ratios and illustrative training accuracy matched sklearn. Reports,
the final audit and seven terminal logs are retained in ignored
`validation/results`. The cumulative inventory above is not a complete rerun
claim; unrelated reference suites and the separate 42-group legacy RNG
reference suite were not rerun.


## Eigen discriminant projection with shrinkage (2026-11-22)

Added sklearn LDA's generalized eigen solver with scalar covariance shrinkage,
uncentered discriminant projection and full-basis classification. Cholesky
whitening reduces the covariance pair to the existing symmetric eigensolver.
The native default remains LSQR; SVD keeps its thin, covariance-free path.
Eigen fitting retains the within covariance regardless of store_covariance,
with normalized scalings and native generalized-eigenvalue diagnostics.

`validation/compare_discriminant_eigen.py` retains 13,452 groups across 708
fixtures with 708 independent covariance-normalization / generalized-eigen
residual certificates. Three seeds cover binary/multiclass labels, tall/wide and
single-feature input, singleton classes, zero-variance features, shrinkage
0/.15/.7/1, automatic/explicit components, both fit entry points and covariance
storage flags. Coefficients, scores, probabilities and predictions compare
directly with installed sklearn 1.6.1 (binary native class rows use source's
single-margin convention). Transform caps do not reduce the classification basis.

There are 225 source eigenspace certificates for repeated eigenvalues, including
165 crossing the component cutoff. Alignment uses only rotations within source
eigenvalue clusters, with separate metric normalization and eigen-equation
checks; no fixture is discarded. Six zero-between-energy cases retain NaN
variance ratios and full requested eigen transform width. Twenty rejection
cases cover controls, sample/classes, non-positive-definite within covariance,
finite/shape contracts and fitted/refit state. Two singular covariance failures
are confirmed with source's generalized eigensolver as well as native Cholesky.

Inventory: 217 public core types, 129 core modules, 143 native test programs,
120 core reference suites and 539,689 cumulative retained groups. The separate
42-group legacy RNG reference suite remains excluded. Thirty invalid-input
registries retain 3,380 cases. These are cumulative counts, not a full rerun
claim; final verification below states the suites rerun for this change.

Run `python validation/compare_discriminant_eigen.py --profile checked`, or use
`--profile release` / `--executable`. Native interfaces and Cholesky/eigen calls
remain MIT with existing sklearn BSD notices retained. No public estimator type,
extension algorithm or dependency is added. Custom priors and automatic
shrinkage remain outside this increment.


Final verification: all 143 native test programs passed in checked mode
(`-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow`) and release mode.
Each profile passed 24,728 reference groups: 13,452 new eigen groups, all
11,084 retained SVD discriminant groups and 192 integration groups, including
LSQR LDA and QDA. All 20 new and 20 retained SVD rejection cases passed per
profile. The retained 12 by 20,000 SVD fixture still allocated no covariance.

The final audit matched validator, upstream-source and executable hashes and
verified 280 compiled source-content digests per profile (129 core modules,
143 tests, four vendor sources, three affected drivers and the new example).
It retained 708 eigen residual/normalization certificates, 225 source basis
certificates (165 crossing the component cutoff), six undefined-ratio cases
and two source singular failures per profile, plus all 652 SVD rank certificates,
three SVD undefined-ratio cases and two source empty-rank errors.

Both explicit fpm example runs passed; dimensions, explained variance ratios
and illustrative training accuracy matched installed sklearn. Reports, the
audit and seven terminal logs are retained in ignored `validation/results`.
The cumulative inventory above is not a full rerun claim. Unrelated reference
suites and the separate 42-group legacy RNG reference suite were not rerun.


## Automatic discriminant covariance shrinkage (2026-11-22)

Added sklearn LDA's classwise standardized Ledoit-Wolf shrinkage for LSQR/eigen,
with independent total covariance estimation for eigen. The existing native
Ledoit-Wolf estimator is reused; corrected two-pass variance and source's
mean-dependent constant-feature test provide standardization. Ordered class and
eigen-total shrinkage diagnostics are stored and cleared on fixed-mode refitting.
Automatic LSQR uses numerical least squares for singular covariance; eigen
retains its positive-definite requirement. Fixed solver defaults remain unchanged.

Automatic decision scores evaluate centered affine margins relative to the first
class. This preserves the classifier while avoiding cancellation of large common
intercepts. The stored coefficient/intercept rows keep their native interfaces.

`validation/compare_discriminant_auto.py` retains 6,650 groups across 357 fixtures
and 357 covariance/pseudoinverse certificates. Three seeds cover binary and
multiclass labels, tall/wide/single-feature input, singleton classes, unequal
feature units, constant and near-constant features, both solvers and component
caps. Per-class and total estimates compare with separately standardized source
Ledoit-Wolf fits. Six numerical-rank-deficient LSQR cases include zero covariance
and rank-one fits; 134 cases retain negative source generalized eigenvalues.

All fixtures remain. Six extreme near-constant cases use 80-digit classifier
certificates, source cancellation bounds and scaled covariance comparisons.
Six spectral certificates bound small-root uncertainty relative to the whitened
matrix norm, check basis error in covariance units and verify projection identities
against the retained source-certified axes. Seven repeated-basis certificates
and one undefined-ratio certificate are retained. Six classwise isotropy
certificates explain differing shrinkage coefficients with equivalent covariance;
the coefficient has no identifiable effect at an isotropic covariance.
Twenty rejection cases cover modes/controls, unsupported SVD automatic shrinkage,
class/sample counts, singular eigen fitting, finite/shape and fitted contracts.

Inventory: 217 core public types, 129 core modules, 144 native test programs,
121 core reference suites and 546,339 cumulative retained groups. The separate
42-group legacy RNG reference suite remains excluded. Thirty-one invalid-input
registries retain 3,400 cases. These are cumulative counts, not a complete rerun
claim; final verification below identifies the suites actually rerun.

Run `python validation/compare_discriminant_auto.py --profile checked`, or use
`--profile release` / `--executable`. The validator needs mpmath for its 80-digit
certificates. Fortran interfaces and centered evaluation remain MIT, with the
existing sklearn BSD notice retained. No public estimator type, extension
algorithm or external Fortran dependency is added. Custom priors remain outside
this increment.


Final verification: all 144 native test programs passed in checked mode
(`-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow`) and release mode.
Each profile passed 31,378 reference groups: 6,650 automatic-shrinkage groups,
13,452 retained eigen groups, 11,084 retained SVD groups and all 192 integration
groups, including fixed LSQR LDA and QDA. All 20 new, 20 eigen and 20 SVD
rejection checks passed per profile. The 20,000-feature SVD case still passed
without allocating feature covariance.

The final audit matched executable, validator and upstream-source hashes and
checked 282 compiled source-content digests per profile (129 core modules,
144 native tests, four vendor sources, four affected drivers and the example).
It retained all 357 new covariance certificates, seven basis certificates,
six 80-digit classifier certificates, six spectral roundoff certificates,
six isotropic-coefficient certificates, six numerical-rank-deficient LSQR cases,
134 cases with negative source generalized eigenvalues and one undefined-ratio
case, plus the retained eigen/SVD certificates. No fixture or query row was
removed. Both profiles retained identical case inventories and certificate counts.

Both explicit fpm example runs passed; shape, class/total shrinkage and
illustrative training accuracy matched sklearn. Reports, the final audit and
seven terminal logs are retained in ignored `validation/results`. These checks
are the affected suites rerun for this increment; the cumulative inventory
above is not a complete rerun claim. Unrelated reference suites and the
separate 42-group legacy RNG reference suite were not rerun.


## Custom discriminant class priors and thin QDA (2026-11-22)

Added `%priors` input for LDA/QDA, preserved caller weights, fitted normalization
status and zero-prior handling. LDA follows source's 1.e-5 renormalization
threshold; QDA retains unnormalized supplied weights. All LDA solvers weight
covariance and geometry by fitted priors, including optional SVD covariance.
Explicit-prior LSQR accepts numerical singularity through least squares.
Automatic decision scores select the first positive-prior reference class.
Negative infinite log priors exclude zero-weight classes safely under traps.

The new reference cases exposed and fixed a prior QDA mismatch for classes with
fewer samples than features. Source regularizes the thin retained SVD subspace,
rather than adding variance in every ambient direction. Native QDA now uses that
thin spectrum and an orthogonal completion of its retained null direction.
Positive regularization is required on this wide path; empirical/fixed LDA
solver defaults remain unchanged.

`validation/compare_discriminant_priors.py` retains 13,594 groups over 841
fixtures and 841 normalization/input-preservation certificates. Three seeds
cover all three LDA solvers, fixed/automatic shrinkage and QDA; uniform, skewed,
nonunit, near-unit, zero, one-positive, tiny and large priors; tall, wide and
single-feature data; both fit entry points and automatic/explicit components.
The QDA extra case accepts finite prior weights whose sum overflows, without
summing them. There are 210 zero-prior posterior/prediction certificates and
27 one-positive-prior SVD undefined-energy certificates.

Forty-eight thin-QDA certificates verify the source regularized spectrum,
projector idempotence/rank and inclusion of the centered training span. The
retained null direction is arbitrary: source query/training scoring uses the
same independently certified native direction. No fixture is removed.
Seventeen rejection cases cover malformed/negative/nonfinite/zero prior vectors,
LDA sum overflow, sample/class limits, singular eigen fitting, empty SVD rank
and unregularized wide QDA.

Inventory: 217 core public types, 129 core modules, 145 native test programs,
122 core reference suites and 559,933 cumulative retained groups. The separate
42-group legacy RNG reference suite remains excluded. Thirty-two invalid-input
registries retain 3,417 cases. Cumulative counts are not a complete rerun claim;
final verification below lists the checks actually rerun.

Run `python validation/compare_discriminant_priors.py --profile checked`, or use
`--profile release` / `--executable`. Original Fortran prior interfaces, safe
logarithms, reference selection and null completion remain MIT, with existing
sklearn BSD notices retained. No estimator type, extension algorithm or external
Fortran dependency is added. Sparse inputs and custom covariance objects remain
outside this interface.


Final verification: all 145 native test programs passed in checked mode
(`-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow`) and release mode.
Each profile passed 44,972 reference groups: 13,594 new prior groups, all
6,650 automatic-shrinkage groups, 13,452 eigen groups, 11,084 SVD groups and
192 integration groups. All 17 new and 60 retained rejection checks passed.
The existing high-precision and spectral/roundoff certificates still passed,
as did the 20,000-feature covariance-free SVD case.

The final audit matched validator, upstream-source and executable hashes and
verified 284 compiled source-content digests per profile (129 core modules,
145 tests, four vendor sources, five affected drivers and the new example).
It confirmed all 841 prior-normalization/input-preservation certificates,
210 zero-prior certificates, 27 one-positive-prior SVD undefined-ratio cases
and 48 thin-QDA spectrum/projector/training-span certificates per profile.
No fixture was removed. Checked and release case inventories matched.

Both explicit fpm example runs passed; fitted LDA priors, zero-class posterior
and retained QDA prior weights matched sklearn. Reports, the final audit and
seven terminal logs are retained in ignored `validation/results`. The cumulative
inventory above is not a complete rerun claim; unrelated reference suites and
the separate 42-group legacy RNG reference suite were not rerun.


## Weighted discrete NB streaming (2026-11-22)

Added weighted batch and partial fitting for MultinomialNB, BernoulliNB and
ComplementNB, persistent raw feature/class counts, fitted feature width and
Complement total-feature counts. First-call declared classes, missing classes,
class-set/width continuity, fit resets and independent copied state follow the
source streaming contracts. Optional class priors override empirical/uniform
prior selection without renormalization. Source-positive scalar alpha remains
the native scope. Bernoulli thresholds now reject source-invalid negative values.

`validation/compare_discrete_stream.py` retains 22,150 groups across 824 fixtures
and 1,796 fitted snapshots. Three seeds cover all three estimators, one/two/three
observed classes, tall/wide/single-feature arrays, weighted/unweighted fitting,
multiple thresholds, both Complement normalizations, missing declared classes,
zero/fractional weights, empirical/uniform/fixed priors and nonunit/zero prior
entries. There are 1,796 independent weighted count-conservation certificates
and 324 final streaming/batch equivalence cases. All query and training labels
and scores are checked at every snapshot.

The suite retains 909 field-level NaN certificates for zero-mass empirical
priors, one-feature normalized Complement weights and saturated Bernoulli
likelihoods; no fixture is discarded.
Complement label ties use the existing classifier-chain rounding certificate,
with training accuracy checked against the certified training predictions.
Finite score/probability fields still compare directly with installed sklearn
1.6.1. Twenty-five rejection cases cover first/changed class contracts, labels,
feature widths, finite/nonnegative data and weights, priors, thresholds, alpha,
query contracts and weighted-product overflow.

Inventory: 217 public core types, 129 core modules, 146 native test programs,
123 core reference suites and 582,083 cumulative retained groups. The separate
42-group legacy RNG reference suite remains excluded. Thirty-three invalid-input
registries retain 3,442 cases. These are cumulative counts, not a full rerun
claim; final verification below identifies the suites rerun for this increment.

Run `python validation/compare_discrete_stream.py --profile checked`, or use
`--profile release` / `--executable`. Native state/weight interfaces remain MIT,
with existing sklearn BSD notices retained. No estimator type, public extension
algorithm or external Fortran dependency is added. Sparse inputs, vector/zero
alpha and signed weights remain outside this native interface.

Final verification: all 146 native test programs passed with
`-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow` and with
`--profile release`. Each profile passed 75,033 affected reference groups:
22,150 new discrete-stream groups, 16,536 weighted-classifier groups, 27,490
multiclass groups, 8,665 classifier-chain groups and 192 integration groups.
All 25 new rejection cases passed in both profiles. The new suite retained
824 fixtures and 1,796 snapshots, with 1,796 count-conservation certificates,
909 undefined-field certificates, 324 streaming/batch equivalence cases and
three tied-label certificates per profile. Saturated Bernoulli fixtures retain
source NaNs safely under the checked floating-point traps; no fixture was removed.

The final audit matched all 285 compiled source-content digests per profile,
validator/source/executable hashes and cross-profile check inventories.
`fpm run --example --target streaming_document_nb` passed explicitly in both
profiles, and example output also matched sklearn. Reports, terminal logs and
`discrete_stream_final_audit.json` are retained in ignored `validation/results`.
Only the affected reference suites listed here were rerun; the cumulative
582,083-group inventory is not a full-rerun claim. The separate 42-group legacy
RNG reference suite remains excluded and was not rerun.


## CategoricalNB zero smoothing and prior boundaries

CategoricalNB now implements scalar alpha=0 and `force_alpha` (default true),
including the source 1e-10 floor when false. Fixed class priors are retained
without renormalization and override fit_prior; all-zero fixed priors and
zero total sample weight are accepted. Undefined feature rows, empirical
priors and posterior normalization produce explicit NaNs safely under floating-
point traps. Predictions follow source first-NaN/first-maximum behavior.

Streaming `n_categories_` now matches sklearn 1.6.1 latest-batch metadata,
while ragged tables retain their full accumulated widths and remain queryable.
Previously rejected zero alpha, zero weight mass and nonunit priors moved to
success-path checks. The invalid registry instead checks smoothing/weight
range and nonfinite priors, and adds first/later empty-class and cumulative
weight-range cases. Finite nonnegative sample weights, compact integer-valued
categories and scalar alpha remain the native scope; source Python warnings
are not emitted. Sparse arrays and vector alpha are outside this interface.

The new `compare_categorical_smoothing.py` suite compares all fitted tables,
latest-batch metadata, priors, query/training labels, joint likelihoods,
probabilities and scores after every update. Three seeds cover one/two/three
classes, one/three/seven features, six smoothing/floor configurations, batch
and streaming fits, category growth followed by shrinking batch ranges,
scalar/per-feature minima, missing declared classes, zero/fractional weights,
uniform/empirical/nonunit/zero priors. Weighted per-feature class-mass identities
and final streaming/batch equivalence supplement installed sklearn comparisons.
No fixture is removed when source results are undefined. See
`example/unsmoothed_categories.f90`.

Final verification passed in checked and release builds. All 146 native test
programs passed; checked flags were `-fcheck=all -fbacktrace
-ffpe-trap=invalid,zero,overflow`. Each profile passed 76,282 reference groups:
53,460 new categorical-smoothing groups, 480 original CategoricalNB groups,
22,150 discrete-stream groups and 192 integration groups. The new suite retained
1,296 fixtures and 2,916 snapshots, including 2,916 count certificates, 5,563
fields containing source NaNs and 648 streaming/batch equivalence cases. No
label tie certificates were needed. All 25 discrete-stream rejection cases
and all 33 categorical rejection cases passed in both canonical profiles.
The example also passed explicitly through fpm in both profiles.

The full 2,450-case general invalid-input registry passed in each profile.
Its standard checked entry point uses `-fcheck=all -fbacktrace`, without
floating-point traps, linked against the checked library. An initial attempt
with traps stopped at the existing spectral-biclustering
`bcsp_co_margin_overflow` case before its expected diagnostic; the complete
registry was then rerun in its standard configuration. All categorical cases
were additionally rerun against the canonical executable with traps enabled.
No legacy case was filtered out. This increment does not change the spectral
biclustering overflow diagnostic.

The final audit matched cross-profile inventories, executable/validator/source
hashes and all 285 compiled source-content digests per profile. Reports, terminal
logs, the standard checked registry executable and orchestration/audit scripts
are retained in ignored `validation/results`, including
`categorical_smoothing_final_audit.json`.

Cumulative inventory: 217 public core types, 129 core modules, 146 native test
programs, 124 core reference suites and 635,543 retained groups. Thirty-three
invalid-input registries retain 3,445 cases. Only the affected reference suites
listed above were rerun; these cumulative counts do not imply a full rerun.
The separate 42-group legacy RNG reference suite remains excluded and was not
rerun. MIT licensing and sklearn BSD attribution remain; no external package,
estimator type or public algorithm extension is added.


## NearestCentroid class priors and discriminants

NearestCentroid now supports empirical and explicit class priors, their source
normalization tolerance, zero-prior classes and all-zero prior normalization.
It exposes n_features_in_ and class_prior_, Euclidean decision_function and
log/probability methods, and prior-aware Manhattan prediction. Uniform-prior
prediction retains raw-distance routing, which may differ from standardized
probability argmax. Optional priors override uniform/empirical mode. Existing
mean/median and shrunken-centroid fitting remains available.

Zero pooled scales and singleton classes now retain source NaN/Inf diagnostics;
queries reject undefined standardized scales or nonfinite shrunken centroids.
Uniform singleton prediction remains defined. All-global-constant inputs are
rejected, matching sklearn. Explicit safe divisions and range checks avoid
floating-point traps. Source warning objects, sparse inputs and weighted score
remain outside this native interface. See docs/API.md and
`example/prior_weighted_centroids.f90`.

`compare_centroid_priors.py` uses installed sklearn 1.6.1 across seeded binary/
multiclass fits, tall/wide arrays, anisotropic/constant/classwise-constant data,
singleton classes, both metrics, shrinkage and seven prior configurations.
It compares all fitted statistics, query/training labels, accuracy, decisions,
probabilities and their identities. Source NaN/Inf fields are retained, source
method errors are certified against native errors, and roundoff label ties
require source-score bounds. No fixture is discarded for undefined behavior.

Final verification: all 147 native test programs passed with checked flags
`-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow` and in release mode.
Each profile passed 22,398 affected reference groups: 22,086 new NearestCentroid
groups, all 120 original graph/neighbor groups and 192 integration groups.
All 30 new rejection cases passed in each profile. The new suite retains 2,520
fixtures, 3,798 nonfinite diagnostic fields, 1,458 source/native method-error
certificates and 756 probability identities. Fourteen bounded label-rounding
certificates were needed per profile. Their scale includes point/centroid
magnitudes to account for cancellation in source Euclidean norm identities;
direct numerical-output tolerances remain unchanged. No fixture was dropped.

The final audit independently reconstructed the required fitted-statistic,
available-method and expected-error inventory for every fixture. It matched
cross-profile inventories, executable/validator/source hashes and all 284
compiled source-content digests per profile. The example passed explicitly
through fpm in both profiles and its output also matched installed sklearn.
Reports, terminal logs and orchestration/audit scripts are archived in ignored
`validation/results`, including `centroid_priors_final_audit.json`.

Cumulative inventory: 217 public core types, 129 core modules, 147 native test
programs, 125 core reference suites and 657,629 retained groups. Thirty-four
invalid-input registries retain 3,475 cases. Only the affected reference suites
listed here were rerun; the cumulative inventory is not a full-rerun claim.
The separate 42-group legacy RNG reference suite remains excluded and was not
rerun. All non-centroid neighbor procedure bodies were verified unchanged.
MIT licensing and sklearn BSD attribution remain; no estimator type, external
Fortran dependency or public extension algorithm is added.


## Nonnegative discrete NB smoothing

`multinomial_nb`, `bernoulli_nb` and `complement_nb` now accept finite scalar
`alpha>=0`. Their `%force_alpha` defaults to true, retaining zero/tiny values;
false floors effective smoothing to 1e-10 without changing input controls.
Multinomial/Complement additionally accept allocated `%alpha_per_feature(:)`,
which overrides scalar alpha and must have n_features finite nonnegative
entries. Flooring applies per feature. Bernoulli retains scalar smoothing;
feature-vector controls are explicitly rejected. Installed sklearn 1.6.1's
Bernoulli vector denominator has class/feature broadcasting artifacts; a new
mathematical variant is not substituted for that source implementation.

Zero priors are accepted even when every fixed prior is zero, matching source;
fixed priors remain unnormalized. `partial_fit` recomputes smoothed probabilities
from raw accumulated counts with the current alpha controls. Changing smoothing
then making a zero-weight update therefore recomputes probabilities without
changing counts. `fit` resets fitted state, preserving input controls.

Dense likelihood arithmetic preserves source NaN/Inf behavior. In particular,
zero times an infinite feature weight is NaN; query rows and classes are handled
individually. Complement normalization retains Inf/Inf and 0/0 NaNs. Positive-
infinite joint scores normalize to NaN at infinite entries and negative infinity
elsewhere, matching source logsumexp behavior. Undefined empirical priors,
zero-total smoothed rows and unit Bernoulli probabilities remain explicit NaNs,
without invalid/zero floating-point traps. Source first-NaN argmax is retained.
Complement counts now follow source addition-before-subtraction order, including
positive smoothing lost at large totals. Python warning objects are not emitted.

See `example/feature_smoothed_nb.f90`. Sparse estimator data, vector Bernoulli
alpha, unthresholded Bernoulli `binarize=None`, signed weights and weighted score
remain outside this interface. Count/smoothing/query intermediates must fit the
native real64 working range; existing explicit count range checks are retained.

`validation/compare_nb_smoothing.py` retains seeded batch/streaming comparisons
for scalar zero/tiny alpha, feature vectors containing zeros, force_alpha on/off,
empirical/uniform/all-zero fixed priors, missing declared classes, zero/fractional
weights and both Complement normalizations. Tall, wide and single-feature data
cover one/two/three observed classes. Large-count cases retain source lost
positive smoothing. All fitted state, query/training labels, scores and
probabilities are checked at each snapshot; count-conservation and final
streaming/single-update identities supplement source comparisons. No NaN/Inf
fixture is dropped. Eleven new rejection cases cover vector controls and
smoothing range checks. The two previously rejected zero-alpha/zero-prior cases
in the streaming registry moved to supported checks, with negative-alpha and
empty-prior rejection cases replacing them.

Final verification: all 148 native test programs passed with checked flags
`-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow` and in release mode.
Each profile passed 134,675 affected reference groups: 59,642 new smoothing
groups, 22,150 discrete-stream groups, 16,536 weighted-classifier groups,
27,490 multiclass groups, 8,665 classifier-chain groups and 192 integration
groups. All 11 new and 25 retained rejection cases passed per profile.

The new suite retains 2,378 fixtures and 4,754 fitted snapshots, with 4,754
independent count certificates, 11,359 fields containing source NaN/Inf,
1,188 streaming/single-update equivalence cases and 69 bounded label-rounding
certificates per profile. The final audit independently reconstructed every
snapshot's required fitted-state and prediction fields; none was filtered out.
It matched cross-profile inventories, executable/validator/source hashes and
all 288 compiled source-content digests per profile. The example passed
explicitly through fpm in both profiles and its output matched installed sklearn.
Gaussian NB procedure bodies were independently verified unchanged.

Reports, terminal logs and orchestration/audit scripts are archived in ignored
`validation/results`, including `nb_smoothing_final_audit.json`.
Cumulative inventory: 217 public core types, 129 core modules, 148 native test
programs, 126 core reference suites and 717,271 retained groups. Thirty-five
invalid-input registries retain 3,486 cases. Only the affected reference suites
listed above were rerun; the cumulative inventory is not a full-rerun claim.
The separate 42-group legacy RNG reference suite remains excluded and was not
rerun. MIT licensing and sklearn BSD attribution remain; no estimator type,
external Fortran dependency or public algorithm extension is added.


## CSR discrete Naive Bayes

`multinomial_nb`, `bernoulli_nb` and `complement_nb` overload `fit`, `partial_fit`,
`joint_log_likelihood`, `predict_log_proba`, `predict_proba`, `predict` and `score`
for `type(csr_matrix)`, alongside their existing dense interfaces. Import CSR
storage from `sklearn_sparse` or the `sklearn` facade. Its pointer and column
arrays are 1-based. Optional sample weights, first-call class declarations,
fixed/uniform/empirical priors, smoothing controls and fitted diagnostics use
the same contracts as dense input. Dense and CSR updates can be interleaved in
one fitted model. A new `fit` resets state regardless of representation.

Training traverses stored entries and keeps dense (n_classes,n_features) fitted
statistics. Sparse prediction uses CSR multiplication and returns dense
(n_samples,n_classes) scores/probabilities. Sample matrices are never converted
to dense in the estimator. Inputs are validated and preserved. Empty rows and
zero-stored-entry matrices with positive dimensions are supported. CSR indices
may be unsorted and duplicate columns are retained. Matrix structure and finite
data are checked; Multinomial/Complement stored values must be nonnegative.

Sparse source likelihoods differ from dense likelihoods when weights include
NaN/Inf: omitted entries contribute nothing, while stored zeros participate in
IEEE multiplication. Bernoulli thresholds stored entries independently and
eliminates resulting zeros before counting or prediction; duplicate entries are
not first summed. Signed finite Bernoulli entries below the nonnegative threshold
are therefore removed. These rules match installed sklearn 1.6.1, including
representation-dependent zero-smoothing predictions. The shared source
normalization and first-NaN argmax rules remain unchanged.

Use `example/sparse_document_nb.f90` for a 10,000-feature weighted document fit.
Gaussian/Categorical NB, pipeline/meta-estimator sparse inputs, CSC/COO storage,
vector Bernoulli smoothing, unthresholded Bernoulli `binarize=None`, signed
sample weights and weighted score remain outside this interface. Fitted
statistics still require O(n_classes*n_features) storage, even for sparse data.

`validation/compare_nb_sparse.py` checks canonical and noncanonical CSR input,
explicit/implicit zeros, duplicates, empty rows and all-zero sample matrices.
Seeded one/two/three-class fits span single-feature, tall and wide inputs;
positive/zero/vector/floored smoothing, weights, missing declared classes,
fixed-zero priors, both Complement normalizations and dense/CSR mixed updates
are covered. Signed Bernoulli duplicate cases verify threshold-before-sum
behavior. Every snapshot checks fitted state, sparse and dense query outputs,
training labels and accuracy. Independent event-count conservation and pure-CSR
streaming/single-update identities supplement source comparisons. Thirty-one
new rejection cases cover CSR structure, data and estimator contracts. A native
10,000-by-10,000 training case contains two stored values; production NB was
also inspected to verify it contains no CSR-to-dense conversion.

Final verification: all 149 native test programs passed with checked flags
`-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow` and in release mode.
Each profile passed 234,499 affected reference groups: 99,824 new CSR NB groups,
59,642 smoothing groups, 22,150 discrete-stream groups, 16,536 weighted-classifier
groups, 27,490 multiclass groups, 8,665 classifier-chain groups and 192 integration
groups. All 31 new CSR, 11 smoothing and 25 streaming rejection cases passed per
profile. Release primary NB checks used separate stdin/report paths and ran
alongside the checked meta-estimator checks; shared-file release meta-estimator
checks started only after the checked run completed.

The CSR suite retains 2,594 fixtures and 6,050 snapshots, with 6,050 independent
event-count certificates, 35,696 fields containing source NaN/Inf, 864 pure-CSR
streaming/single-update equivalence cases and 82 bounded label-rounding
certificates per profile. Both sparse and dense query representations are
checked after every fit/update. No nonfinite or noncanonical fixture was dropped.
The final audit independently reconstructed every snapshot's required fields,
matched cross-profile inventories and executable/validator/source hashes, and
verified all 290 compiled source-content digests per profile. It also confirmed
that production NB contains no CSR-to-dense conversion. Existing Gaussian,
count-update and dense likelihood procedure bodies were verified unchanged;
the dense log-probability normalizer was shared without changing its arithmetic.

The example passed explicitly through fpm in both profiles and its output
matched installed sklearn. Reports, terminal logs and orchestration/audit scripts
are archived in ignored `validation/results`, including
`nb_sparse_final_audit.json`. Release primary and legacy phase logs are retained
separately. Native large-shape coverage uses a 10,000-by-10,000 sample matrix
with two stored values and dense class-by-feature fitted statistics.

Cumulative inventory: 217 public core types, 129 core modules, 149 native test
programs, 127 core reference suites and 817,095 retained groups. Thirty-six
invalid-input registries retain 3,517 cases. Only the affected reference suites
listed above were rerun; cumulative counts do not imply a full rerun. The separate
42-group legacy RNG reference suite remains excluded and was not rerun.
MIT licensing and sklearn BSD attribution remain; no estimator type, external
Fortran dependency or public algorithm extension is added.

## Coordinate-descent NMF

After building the checked or release reference executable, run:

```text
python validation/compare_nmf_cd.py --executable build/gfortran_81A15A8739108A9E/app/reference_nmf_cd_driver.exe --profile checked --report validation/results/nmf_cd_checked.json
python validation/compare_nmf_cd.py --executable build/gfortran_C983BD113DCA31CD/app/reference_nmf_cd_driver.exe --profile release --report validation/results/nmf_cd_release.json
```

The validator retains 2,022 fixtures: six initialization choices, tall/wide
matrices, ranks one/two, fixed and converged iterations, seeded shuffling,
dimension-scaled L1/L2 penalties, auto/overcomplete component counts, zero data,
zero columns and zero-Hessian coordinates. It compares the public sklearn NMF
fit/transform results and replays sklearn's actual compiled coordinate kernel
to check projected-gradient diagnostics. Nonnegativity, reconstruction and the
regularized objective are checked independently. All-zero fitted H receives the
same transform rejection as sklearn. Thirty-four isolated native rejections
cover unsupported controls, shapes, nonfinite/negative inputs and numerical
range limits.

Reduction-order rounding can change an exact-zero stopping decision at zero
tolerance, or when initialization is already stationary to machine precision.
Those cases retain all factor/objective comparisons and receive explicit
certificates bounding both sweep residuals and independently computed projected
gradients using the magnitude of the gradient arithmetic. Other iteration
differences fail. Reports retain executable, validator and upstream-source
SHA256 hashes and reject nonfinite JSON values. Existing MiniBatchNMF and
integration suites check the shared initializer and decomposition regressions.

Final verification passed all 150 native tests in checked and release builds,
20,214 CD groups, 6,500 MiniBatchNMF groups, 192 integration groups and all 34
rejections per profile. Both explicit fpm example runs matched sklearn. Sixteen
roundoff stopping certificates per profile were retained. The final audit
verified all field inventories, report hashes and 287 compiled-content digests
per profile. Logs, reports and audit scripts are archived in ignored
`validation/results`, including `nmf_cd_final_audit.json`. Cumulative counts are
128 core reference suites and 837,309 groups; the separate 42-group legacy RNG
suite remains excluded and was not rerun.

## NMF multiplicative updates and beta divergences

Build the reference driver first, then run the appropriate profile explicitly:

```text
python validation/compare_nmf_mu.py --executable build/gfortran_81A15A8739108A9E/app/reference_nmf_mu_driver.exe --profile checked --report validation/results/nmf_mu_checked.json
python validation/compare_nmf_mu.py --executable build/gfortran_C983BD113DCA31CD/app/reference_nmf_mu_driver.exe --profile release --report validation/results/nmf_mu_release.json
```

The canonical sweep retains 2,090 fixtures across six initialization modes,
nine primary beta values, tall/wide data, three regularization settings,
one/thirteen/converged iteration limits, auto/overcomplete counts, custom zero
factors, exact rank-one fits, zero entries and data below source epsilon.
Twenty-four powered-underflow boundary fixtures retain zero absolute tolerance
for factor comparisons and check rounding on both sides of the subnormal boundary.
It checks public sklearn fit/transform results, replays the upstream MU updates
and stopping rule, and independently checks reconstruction, nonnegativity and
the regularized mathematical objective. Fixed-H KL/L1 caching is included.
Forty-four isolated rejections cover controls, shapes, data/fitted-state domains
and numerical working ranges without trapped arithmetic.

Near an exact fit, source beta-divergence subtraction can cancel at floating-point
precision before taking its square root. Any affected diagnostic comparison
retains an explicit arithmetic-scale error bound and a separate 60-digit Decimal
calculation from the exact input floats. All factors, objectives and fixture
fields remain checked. Decimal is used because NumPy longdouble on this Windows
build has float64 precision. Full reports record `start_fixture=0`; the optional
`--start-fixture` switch is only for development debugging, and partial runs are
not accepted by the final audit. Reports retain executable, validator, imported
CD-validator and installed upstream SHA256 hashes.

Final corrected checked and release builds each passed all 151 native tests,
20,844 MU groups, 20,214 CD groups, 6,500 MiniBatchNMF groups, 192 integration
groups and 78 rejection checks. Both explicit KL examples matched sklearn.
The final audit verified 290 compiled-content digests per profile, complete
field inventories, six MU Decimal certificates and sixteen CD stopping
certificates per profile. Final artifacts are archived in ignored
`validation/results/nmf_mu_final_audit.json` and companion files. Pre-subnormal-fix
reports are retained separately and do not satisfy the final audit.

## Positive linear coefficients

Build the reference drivers, then pass explicit executables:

```text
python validation/compare_positive_linear.py --executable build/gfortran_81A15A8739108A9E/app/reference_positive_linear_driver.exe --profile checked --report validation/results/positive_linear_checked.json
python validation/compare_positive_composition.py --executable build/gfortran_81A15A8739108A9E/app/reference_positive_composition_driver.exe --profile checked --report validation/results/positive_composition_checked.json
```

Use the release executable directory and `--profile release` for optimized
validation. Core checks retain 1,044 fit/CV fixtures and 8,388 groups. Composition
checks retain 192 fixtures and 1,608 groups across pipelines, transformed targets,
bagging, independent multioutput and regressor chains. Thirty isolated rejection
cases check unsupported flags, domains, fitted state, weights and numerical
ranges. Partial `--start-fixture` runs are development evidence only.

Primary comparisons use installed sklearn 1.6.1 and SciPy 1.15.3, with independent
objective, nonnegativity and projected KKT checks. Three explicit certificate
categories retain reference differences: the SciPy stale-QR-tail defect (checked
against BVLS), nonunique NNLS representatives (null-image and KKT bounds, with
query differences recorded), and public Ridge finite-optimizer stopping
(strong-convexity coefficient bounds against augmented NNLS). Composition Ridge
children receive the same constrained-solution certificates; public and exact
composition predictions are both retained. No fixture or objective is discarded.
The ordinary SVD and coordinate solvers remain regression controls.

Final checked and release profiles each pass all 153 native tests, 55,764 affected
reference groups and 30 new rejections. This includes 9,996 new positive groups
plus 45,768 legacy coordinate-CV, bagging, pipeline, transformed-target and
integration groups. The audit verifies all field inventories and 296 compiled
source-content digests per profile. Each retains one QR-tail correction, two
nonunique NNLS cases, thirteen direct Ridge certificates and 216 composition
Ridge child certificates with 96 parent prediction records. Final artifacts are
archived in ignored `validation/results`, including `positive_final_audit.json`;
the initial composition comparison log is retained separately.

## Free nonnegative factorization

`compare_non_negative_factorization.py` requires `--executable`, `--profile`
and `--report`, following the CD/MU stdin-driver workflow. For example:

```powershell
fpm build --flag "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow"
python validation/compare_non_negative_factorization.py --executable build/gfortran_81A15A8739108A9E/app/reference_non_negative_factorization_driver.exe --profile checked --report validation/results/nonnegative_free_checked.json
```

Use the executable for the intended compiler/profile. The suite compares 916
fixtures / 3,664 groups with sklearn's public `non_negative_factorization`,
plus 12 diagnostic cases. Fixed-H cases include ignored nonfinite W,
overcomplete dictionaries, named initialization modes, zero component columns,
regularized fits and general MU beta losses. Outputs are checked for matching
factors, iterations, independent objective values, nonnegativity and exact
fixed-H preservation. Source/native CD stopping differences at `tol=0` require
projected-gradient residuals below an explicit floating-point reduction bound;
the report retains both iteration counts and the certificate.

Final checked and release profiles each pass 154 native tests, 44,914 affected
reference groups and all 90 new/prior diagnostic cases. The new suite retains
14 checked and 16 release stationarity-level stopping certificates. The final
audit checks 294 compiled-content digests per profile, complete new field
inventories, executable/validator/source hashes and unchanged existing
decomposition arithmetic. Logs, reports and audit scripts are archived in
ignored `validation/results`, including `nonnegative_final_audit.json`.

## HashingVectorizer

Build the intended profile and pass its `reference_hashing_vectorizer_driver`
executable to `compare_hashing_vectorizer.py --executable PATH --profile checked
--report validation/results/hashing_vectorizer_checked.json` (or use `release`).
The reference suite uses sklearn 1.6.1's public HashingVectorizer and built
analyzers. It retains 1,779 fixtures / 14,232 groups and 13 native diagnostic
cases. Tests compare all analyzer tokens in order, exact CSR metadata and index
arrays, normalized values, independently reconstructed signed hash counts,
row norms, stateless transforms and explicit dense adapters. Token fingerprints
and pinned source/validator/executable hashes are retained in the JSON report.

The grid covers all three analyzers, four n-gram ranges, three normalization
modes, signs, binary/nonbinary counts, lowercasing, English/custom/no stop words,
one/two-character token minima, exact/padded document adapters and sparse widths
up to the default million features. Boundary fixtures cover ASCII controls,
2,048-character tokens, every English stop word, cancellation zeros, short
padded words and maximum-integer n-gram ranges. Unicode is explicitly outside
the native analyzer domain and receives a diagnostic; it is not included in
claims of source parity. Numeric comparisons use `rtol=2e-14`, `atol=1e-15`;
CSR metadata and analyzer strings must match exactly. No numerical exceptions
are accepted by this suite.

Final checked and release runs each pass 155 native tests, 17,369 affected
reference groups and 53 diagnostics. The affected group total includes 14,232
new text-hashing groups, 2,945 prior feature-mapping/hash groups and 192
integration groups. All 288,369 analyzer tokens and stored values match sklearn
exactly in each profile; row-norm identity errors are below `4.5e-16`. Both sparse
NB examples match source. The final audit checks 296 compiled-content digests
per profile, complete analyzer/field inventories, pinned stop words, hashes and
unchanged existing text/hash/CSR arithmetic. Artifacts and pre-fix logs are
archived in ignored `validation/results`, including `hashing_text_final_audit.json`.

## Normalizer/Binarizer and composition

`compare_stateless_preprocessing.py --executable PATH --profile checked --report
validation/results/stateless_preprocessing_checked.json` compares 1,801 dense/CSR
fixtures / 19,386 groups and 29 diagnostics against sklearn 1.6.1. Use the
`reference_stateless_preprocessing_driver` executable for the desired profile.
The suite checks free functions, both estimator APIs, both axes, returned norms,
prefit transforms, fitted widths/names, unnamed refits, zero/tiny/subnormal/large
values, threshold boundaries, explicit zeros, unsorted duplicates, cancellation
and duplicate overflow. NaN/Inf result masks must match exactly; numeric arrays
use `rtol=3e-13`, `atol=3e-14`. CSR metadata, names and discrete outputs are exact.
Native drivers verify input preservation. Source hashes pin sklearn `_data.py`,
`sparsefuncs.py`, `sparsefuncs_fast.pyx`, `extmath.py` and SciPy sparse sources.

`compare_stateless_composition.py` uses the same arguments with
`reference_stateless_composition_driver`. It retains 864 fixtures / 5,328 groups
for transformer pipelines, weighted Ridge/NB pipelines, weighted feature unions,
column selections/remainders, bagging and voting. All three discrete NB classes
exercise both weighted and unweighted paths. It checks predictions/probabilities,
coefficients, output names/widths, inverse availability and untouched prototypes.

The shared dispatch change is accompanied by the existing pipeline, column/union,
voting, stacking and integration regression suites. Those file-based legacy
suites run sequentially within and between profiles to preserve their shared
validation files. The new stdin suites can run concurrently with the build.

Final checked and release profiles each pass all 157 native tests, 57,356 affected
reference groups and 130 diagnostics. The final audit checks 304 compiled-content
digests per profile, complete new-suite field inventories, all finite/nonfinite
outcome masks, pinned hashes and unchanged surrounding preprocessing/dispatch
procedure bodies. Both examples match sklearn. Artifacts are archived in ignored
`validation/results`, including `stateless_final_audit.json`.

## PolynomialFeatures reference suites

`compare_polynomial.py --executable PATH --profile checked --report
validation/results/polynomial_checked.json` uses `reference_polynomial_driver`
and retains 1,985 fixtures / 38,020 groups plus 31 diagnostics. It compares dense
and canonical CSR output, legacy array calls, degree ranges, interactions, bias,
powers, feature counts, UTF-8/empty names and refits. Input preservation is checked
natively. CSR shape/pointers/indices, names and int64 counts/exponents compare
exactly, without float conversion of large integers. Floating arrays use
`rtol=3e-13`, `atol=3e-14` with exact NaN and signed-infinity masks. Reports retain
243 nonfinite reference fields, including higher-degree sparse overflow times
implicit zero. They pin installed sklearn/SciPy source hashes.

The suite retains 984 raw noncanonical source outcomes/errors while comparing
native canonicalized output against public canonical-CSR results. Ordinary raw
cases also check the equivalent dense result. Source empty-interaction defects
are recorded as 48 metadata and 36 sparse-transform exceptions; native outputs
are correctly empty. Three huge one-feature exponents use symbolic powers/names
rather than allocating source tuples. One huge interaction fit-only case uses
source `_num_combinations` on the requested range to avoid an enormous unused
auxiliary count. Reports identify these oracle limits explicitly. Development
`--start-fixture` runs are partial; final reports require start zero.

`compare_polynomial_composition.py` uses the same executable/profile/report
arguments with `reference_polynomial_composition_driver`. Its 480 fixtures /
2,880 groups check polynomial/normalization pipelines, weighted/unweighted Ridge,
weighted feature unions, column selection/remainders, names, inverse availability
and untouched prototypes. Transform-only outputs retain the core tolerance;
independent Ridge SVD implementations use `rtol=3e-12`, `atol=3e-13`.

Affected legacy regressions include stateless preprocessing/composition,
pipeline, column/union and general integration suites. File-based legacy suites
run sequentially within and between profiles to avoid shared-file races.

The original name/refit fixture also runs in 128 fresh native processes, comparing
all emitted fields exactly. These repetitions retain a regression for an
intermittent formatted-I/O failure found during stress verification and are
reported separately from distinct fixture/group counts. The implementation now
converts integer exponents/indices to text without internal formatted writes.

Final checked and release verification each passes 159 native tests, 77,722
affected reference groups and 161 diagnostics. Both explicit examples match source.
The audit verifies 307 compiled-content digests per profile, complete field and
fixture inventories (including 128 name/refit processes), pinned hashes and
unchanged surrounding code. Artifacts and pre-fix logs remain in ignored
`validation/results`, including `polynomial_final_audit.json`.

## Image/grid graph and subtraction reference suites

`compare_image_graph.py --executable PATH --profile checked --report
validation/results/image_graph_checked.json` uses `reference_image_graph_driver`.
The suite retains 1,832 fixtures / 14,653 groups and 13 diagnostics, spanning
scalar/vector/matrix/volume images, 2D/3D grids, scalar and rank-one/two/three
masks, disconnected/empty selections, zeros, negative values, tiny values,
near-overflow differences and NaN/Inf inputs. It checks CSR metadata, pointers,
indices, data signs, dense views, stored-zero masks and untouched inputs. Three
large graph cases check sparse multiplication without dense adjacency. Integer
outputs/signs compare exactly, and floats use `rtol=3e-14`, `atol=0` with exact
NaN and signed-infinity masks. Reports retain 352 nonfinite source fields.

The reports preserve 560 raw sklearn 1.6.1 exceptions for masks with singleton
axes: source squeezes the image alone when selecting diagonal values. Native
same-shaped masks are valid. The numerical oracle is the public source function
with a mask matched to its internally squeezed image. Ordinary successful raw
source outputs are also checked against that oracle. Grid masks retain source
C-order flattening. Image/mask rank and shape rules are documented in API.md.

`compare_ieee_difference.py` uses the same arguments with
`reference_ieee_difference_driver`. It checks 8,703 scalar pairs / 17,406 groups
against NumPy float64 subtraction: a Cartesian anchor grid, random finite bit
patterns and explicit overflow-rounding boundary cases. Every finite result bit,
including signed zero, must match exactly; NaN/signed-infinity masks and non-NaN
signs must also match. Reports retain the 258 original guard errors. The source
NumPy binary, validator, parser and executable hashes are pinned. This checks
the shared internal helper, not a new public extension.

Final checked and release profiles each pass 161 native tests, 276,126 affected
reference groups and 308 diagnostics. The shared-helper regressions include image,
structured-feature, histogram-boosting, sparse/smoothed NB, positive-regression,
preprocessing, weighted-classifier/tree and pipeline/column/union/integration
suites. File-based legacy suites run sequentially within/between profiles.
The final audit checks 321 compiled-content digests per profile, complete field
inventories, graph-mask outcomes, exact subtraction bits, pinned hashes, distinct
optimized executables and unchanged surrounding procedure bodies. Both examples
match source. Reports/pre-fix logs remain in ignored `validation/results`,
including `image_graph_final_audit.json`.


Nystroem CSR coverage can be run with
`python validation/compare_nystroem_csr.py --executable <reference_nystroem_csr_driver.exe>`.
It checks 870 configurations / 15,018 field groups and 29 diagnostics. Regular
basis coordinates compare directly; singular normalization/feature Grams and
complete SVD certificates compare separately. The normalization Gram uses a
matrix-norm-scaled roundoff bound to handle null-space cancellation. Active CSR
storage is exact; source trailing unused capacity is excluded. Native immutable
inputs and refit identity checks are labeled separately; source Laplacian input
mutation is recorded. See docs/VALIDATION.md for the precise scope and profiles.


Run `compare_polynomial_csr.py --executable <reference_polynomial_csr_driver.exe>`
for the dense/CSR TensorSketch suite. It requires 26,496 groups / 2,944 fixtures
and 98 diagnostics, including 63,736 nonfinite source values. Hashes, shape,
metadata, input/RNG identity and refit hashes are exact; numerical features use
3e-8 tolerances with exact NaN and signed-infinity masks. Six native exception
flags/halting checks form a separately labeled environment group per fixture.
The original dense PolynomialCountSketch reference suite remains in place.


`compare_maxabs.py --executable <reference_maxabs_driver.exe>` checks dense/CSR
MaxAbsScaler fits, mixed streaming batches, transformations/inverses and both
axes of maxabs_scale, including rank-one arrays. It requires 55,644 groups /
2,932 fixtures and 101 diagnostics, comparing 14,272 nonfinite source values.
Numeric fields use 3e-8 tolerances and exact NaN/signed-infinity masks; CSR
layouts and metadata are exact. Native caller/RNG identities are separately
labeled; source free-row-scale input mutation is recorded. Sparse inverse
requires matching width; dense inverse retains source length-one broadcasting.


`compare_minmax.py --executable <reference_minmax_driver.exe>` checks 596
MinMaxScaler/minmax_scale configurations / 9,532 required groups and 16
diagnostics. It compares 8,544 nonfinite source values with exact NaN and
signed-infinity masks. Numerical fields use 3e-8 tolerances; metadata and native
caller/RNG identities compare exactly. Streaming all-NaN propagation, refit
reset, source nonfinite range endpoints, live reversed clip bounds, inverse
broadcasting and vector/matrix free functions are explicitly represented.


`compare_robust.py --executable <reference_robust_driver.exe>` checks 1,332
RobustScaler/robust_scale configurations / 31,944 groups and 69 diagnostics.
All 32,575 nonfinite source values are compared. Numeric fields use 3e-8
absolute/relative tolerances and exact NaN/signed-infinity masks; layouts and
metadata are exact. Native caller/RNG identities are separately classified.
Fixtures include nullable statistics, raw CSR duplicates/count constraints,
normal-unit adjustment, subnormal quantile probabilities and NumPy median
599/600/601-row dispatch. Sparse fit quantiles intentionally preserve the
source raw stored-entry contract without densifying input or padded columns.


`compare_standard.py --executable <reference_standard_driver.exe>` checks
724 configurations / 17,376 groups and nine isolated diagnostics. It exercises
weighted dense/CSR fitting, mixed incremental batches, missing values, negative
and zero weights, scalar NaN/Inf source behavior, refit resets, live query flags,
raw sparse storage, absent statistics and scalar/vector count dtype metadata.
All 2,188 nonfinite source outcomes are compared. CSR layout, fitted metadata
and native caller/RNG identities are exact. Numerical fields use 3e-8 absolute
and relative tolerances, except large-offset variance/scale/map fixtures whose
relative allowance is max(3e-8,64*epsilon*max(abs(mean)/sqrt(abs(var)))). Means
and counts retain ordinary tolerances; NaN and signed-infinity masks remain
required. Each report records field inventories, configurations, tolerances,
source/executable/validator hashes, errors and rejection outcomes.

The final StandardScaler package audit passed in checked and optimized release
profiles: 21 selected native tests, four scaler examples, 54,420 groups across
seven affected comparison runs, 21 diagnostics, and 202 compiled units per
profile. Full scope and the separate existing logistic endpoint differences
are recorded in [the validation record](../docs/VALIDATION.md#standardscaler-final-package-verification-2026-11-24).


`compare_scale.py --executable <reference_scale_driver.exe>` checks the free
preprocessing.scale algorithm on dense matrices, vectors and raw CSR input.
It retains 420 generated configurations: 412 accepted fixtures / 3,296 groups
and eight expected rejections after duplicate-heavy dense input conversion
overflows. Sixteen further input/format cases bring diagnostics to 24. All
6,074 nonfinite source outcomes remain compared. Scalar-vector zero handling,
matrix standard-deviation thresholding, sparse variance thresholding, global
precision corrections, both dense axes, pairwise-sum size boundaries and
caller/RNG identities are represented. Source CSC output is permuted back to
original CSR slots without coalescing. Corrections, layouts and identities are
exact. Numeric relative tolerance is 3e-8; absolute tolerance is
min(3e-8,max(64*smallest_subnormal,3e-8*max(abs(finite_expected)))). This protects
tiny and subnormal outputs against replacement with zero. NaN and signed
infinity outcomes remain required. Reports pin source/driver/validator/executable hashes,
including installed NumPy nan-statistic sources.

The final package audit passed in checked and optimized release profiles:
23 selected native tests, five examples, 20,864 groups across three comparison
runs, 33 diagnostics and 206 compiled source units per profile. See the
[verification record](../docs/VALIDATION.md#free-scale-final-package-verification-2026-11-24)
for scope and limitations. Shared-file core comparisons must run sequentially
across profiles; the two modern scale validators have profile-specific reports.


`compare_quantile_seeded.py --executable <reference_quantile_seeded_driver.exe>`
checks seeded QuantileTransformer dense/CSR fits and both-axis free quantile maps.
Its 1,732 generated configurations retain 1,620 accepted fixtures / 37,260 groups
and 112 expected source boundaries (64 input/conversion overflows, 48 negative
sparse inputs). Twenty-one explicit fit/query/format checks bring diagnostics
to 133. Reference grids, metadata, layouts and 1,620 caller/RNG identities are
exact; 35,640 other groups use sklearn. All 27,904 nonfinite source outcomes
remain compared. Numeric relative tolerance is 3e-8, with absolute tolerance
min(3e-8,max(64*smallest_subnormal,3e-8*max(abs(finite_expected)))). Fixtures cover
source RNG consumption, zero-length sparse samples, CSR row-order-sensitive
sampling, query-only live parameters, duplicate/all-missing quantile tables,
normal clipping, percentile overflow and stateful interpolation search guesses.

`compare_quantile_transform.py` now accepts --executable/--profile/--report to
run its existing 252 groups against an already built driver. Its no-argument
fpm workflow is retained. Data/output paths are profile-specific and reports pin
the source, driver, validator and executable hashes. New canonical verification
also runs the 14 original quantile_* diagnostic cases and directly affected
composition slices without changing the original fixture-generator streams.

The final quantile package audit passed in checked and optimized release
profiles: 25 selected native programs, two examples, 39,138 required groups
across seven comparison runs, 147 diagnostics and 212 compiled source units per
profile. Full scope and the corrected native rounding fixture are recorded in
[the verification record](../docs/VALIDATION.md#source-seeded-quantile-final-verification-2026-11-24).


`compare_simple_imputer.py --executable <reference_simple_imputer_driver.exe>`
checks numeric SimpleImputer dense/CSR fits, mixed queries, missing markers,
empty-feature rules, indicators, callbacks, feature names, live controls,
inverse mapping and refit reset. It retains 2,183 configurations: 1,843 accepted
fixtures / 42,389 groups and 340 expected source boundaries (240 zero-marker
sparse inputs/queries, 80 input/conversion overflows and 20 source inverse
IndexErrors). Sixteen explicit cases bring diagnostics to 356. There are 40,546
source/adapted-storage groups and 1,843 native caller/RNG identities. All 4,237
nonfinite source values are compared. Metadata, names, indicator indices and
raw CSR layout are exact. Numeric tolerance is 3e-8 relative, with absolute
min(3e-8,max(64*smallest_subnormal,3e-8*max(abs(finite_expected)))). Source sparse
results are rearranged to native row-owned slots without coalescing duplicates.

The validator checks NumPy masked-mean safe-division domains, odd matrix-median
overflow, source sparse zero multiplicities, callable empty-column shortcuts,
constant NaN/Inf fills, and the source inverse routine's indicator-based mapping.
Source failures are retained as required diagnostics rather than repaired or
silently discarded. Reports include full accepted/rejected inventories, fields,
versions and source/validator/driver/executable hashes. Named dense cases use the
installed pandas DataFrame interface; native feature names use feature_string.

The final imputer package audit passed in checked and optimized release
profiles: 26 selected native programs, two examples, 69,377 required groups
across six comparison runs, 427 diagnostics and 217 compiled source units per
profile. Scope, refreshed legacy assertions and separate existing logistic
endpoint differences are recorded in the
[verification record](../docs/VALIDATION.md#simpleimputer-final-package-verification-2026-11-24).


## PowerTransformer source maps and free power_transform

`compare_power_edges.py --executable PATH --profile checked --report REPORT`
compares dense fits, free maps, direct scalar forward/inverse queries, missing
observations, constant columns and extreme finite Box-Cox constraints against
installed sklearn 1.6.1 / SciPy 1.15.3. Seed 4701 fixes the distribution stream.
Scalar queries include zero/two limiting powers, NaN/infinite powers, negative
inverse bases, underflow and overflow. NaN and signed infinity masks must match
exactly; finite fitted maps use 3e-6 tolerance, overridden scalar maps 3e-12.
Source-rejected configurations remain in the report with native diagnostics.
Reports retain every configuration, input/query fingerprints, field checks,
nonfinite counts and stable source/driver/executable/oracle hashes.

The existing `compare_power_transform.py` now supports `--executable`,
`--profile` and `--report`, keeping profile-specific data/output files and its
160 established checks. `test_power_edges` additionally checks free-map input
ownership, retained scaler state, live controls, near-zero constant shortcuts,
inverse boundaries and preservation of invalid/divide-by-zero/overflow modes
and flags. The example
`power_transforms` compares the free and fitted maps.


## Weighted, degree-zero and sparse SplineTransformer

`compare_spline_extended.py --executable PATH --profile checked --report REPORT`
uses the unmodified installed sklearn 1.6.1 / SciPy 1.15.3 reference. It retains
all 1,068 configurations: 918 required fits/queries, and 150 endpoint cases for
mixed negative weights with ties. Required fields include base/extended knots,
fitted metadata, dense maps, raw CSR pointers/indices/data and name encodings.
Finite data tolerances are 3e-8; masks, metadata, indices, shapes and names are
exact. Seed 7231 fixes the ordinary stream, seed 931 the signed tied stream.
Every configuration and data fingerprint is retained with source/driver/
validator/executable/oracle-source hashes. Source rejections remain in reports.

Historical note: the former `--stable-ties` mode ran a controlled-oracle certificate:
NumPy argsort uses kind=stable only during sklearn weighted-percentile fitting.
It checks the same 150 signed-tie configurations and labels the changed oracle
policy explicitly. Its 1,125 groups and 45 rejections validated the previous native stable-tie
behavior. The ordinary report separately retains 275 differences among 1,128
endpoint diagnostics from the installed CPU backend; they are not counted as
required passes. See COVERAGE.md for the compatibility gap.

`compare_spline.py` retains its existing 180-group suite, adding --executable,
--profile and --report with profile-specific files. Native test_spline_extended
covers weighted extrema/quantiles, degree-zero bases, repeated knots, sparse
zero-slot retention, input/output ownership, RNG preservation, names/refitting
and live controls. example/spline_sparse.f90 demonstrates weighted named CSR
feature output. Both build profiles use fresh binaries.


## Seeded KBinsDiscretizer and native NumPy ordering

`compare_kbins.py --executable PATH --profile checked --report REPORT` retains
597 configurations / 5,976 required groups, 175 source rejections and 588
nonfinite output values. It tests all strategies/encodings, per-feature bins,
integer and mutable-object RNG sampling, full uint32 seeds, named fits,
post-sampling weight shapes, signed/zero weights, refits/live encoding, soft and
signed one-hot inverse data, fractional/negative ordinal indexing, raw inverse
CSR duplicates/unsorted columns/zeros, and 200,003-row data crossing the default
sampling threshold. Seed 5201 fixes ordinary fixtures; seed 931 fixes signed ties.
Discrete maps, metadata, shapes, indices, names and RNG draws compare exactly;
finite edge/midpoint fields use 2e-10 tolerance. Nonfinite masks match exactly.

`compare_numpy_ordering.py` takes the same executable/profile/report options.
It compares 11,024 configurations / 22,048 exact argsort and argpartition arrays
against the installed NumPy 2.2.6 AVX512/MSVC backend. Cases cover lengths 1..269
and larger boundaries through 10,000, random continuous/tied data, constants,
ascending/descending/peaked arrays, signed zeros/infinities, and first/middle/
last-three partition positions. CPU feature metadata is retained in the report.
The native routines use scalar Fortran buffers and require non-NaN keys.

`compare_spline_extended.py` now checks all 1,068 configurations against
unmodified sklearn: 10,206 required groups and 198 source rejections, with
1,105 nonfinite values. The prior 275 differences are resolved. Endpoint
diagnostics are empty; the controlled stable-tie mode is retired. Its 1,125
previous certificate groups are replaced by unmodified-source checks of the
same cases, with no duplicate coverage-count inflation.

The legacy compare_discretization.py supports explicit --executable, --profile
and --report and profile-specific input/output files, preserving its 324 groups.
Native programs test_kbins_extended and test_numpy_ordering complement the
existing tests; example/kbins_sparse.f90 shows named seeded CSR bins and inverse
midpoints. Attribution and full Intel BSD/Microsoft Apache-LLVM notices are
retained under THIRD_PARTY_NOTICES.md and LICENSES/.


Weighted multioutput R2 and explained variance:
```
python validation/compare_variance_scores.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_variance_scores_driver.exe --profile checked --report validation/results/variance_checked.json
python validation/check_variance_scores_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/variance_scores_invalid_driver.exe --profile checked --report validation/results/variance_invalid_checked.json
```
Repeat with the release executables and profile. The oracle pins sklearn 1.6.1,
NumPy 2.2.6, C-contiguous float64 matrices and buffered vector reductions.
Each successful fixture checks score values/nonfinite masks, input identity
and IEEE status/mask preservation. Rejected source calls require controlled
Fortran diagnostics. `test_variance_scores` also checks RNG preservation,
owned raw arrays, mixed ranks, bias, constants and overflow under checked traps.


Weighted multioutput regression errors:
```
python validation/compare_error_metrics.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_error_metrics_driver.exe --profile checked --report validation/results/error_new_checked.json
python validation/check_error_metrics_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_error_metrics_driver.exe --profile checked --report validation/results/error_invalid_checked.json
```
Repeat with release executables and profile. Successful configurations run in
one batched native process and check values, exact nonfinite masks, input
identity and IEEE-state preservation. Source rejections and targeted public
guards each run in isolated processes. The oracle pins sklearn 1.6.1, NumPy
2.2.6, real64 C-contiguous target matrices and the default reduction buffer.
The native test also checks RNG preservation, owned raw results and RMSE's
per-output roots/custom-weight differences. No historical suite is implied by
these commands; affected native/core regressions are recorded separately.


Weighted median and column maximum errors:
```
python validation/compare_order_errors.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_order_errors_driver.exe --profile checked --report validation/results/order_new_checked.json
python validation/check_order_errors_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_order_errors_driver.exe --profile checked --report validation/results/order_invalid_checked.json
```
Repeat for release executables/profile. Successful calls run in a batch and
check values, nonfinite masks, input identity and IEEE state. Expected source
rejections and targeted guards run in isolation. The source fixes sklearn
1.6.1, NumPy 2.2.6 and the existing Intel index-order reference backend. Signed
and zero-total CDFs, ties, odd/even huge errors, long arrays and static raw-rank
adaptation are explicit fixtures. Selected native/core regressions and RNG
preservation are recorded separately in docs/VALIDATION.md.


Weighted logarithmic and direct deviance errors:
```
python validation/compare_distribution_errors.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_distribution_errors_driver.exe --profile checked --report validation/results/distribution_new_checked.json
python validation/check_distribution_errors_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_distribution_errors_driver.exe --profile checked --report validation/results/distribution_invalid_checked.json
```
Repeat for release/profile. Successful calls run in one batch and check values,
nonfinite masks, input identity and IEEE state. Expected source rejections and
targeted guards remain isolated. The source fixes sklearn 1.6.1, NumPy 2.2.6,
C-contiguous float64 matrices and the default reduction buffer. The earlier
regression-metric reference suite is also rerun for RMSLE's existing module,
positional and output_weights callers. Native tests check legacy keyword
adapters, valid negative log targets, tiny log1p losses, roots per output, RNG
and owned results. Affected/core validation is recorded in docs/VALIDATION.md.


General Tweedie deviance/D2:
```
python validation/compare_tweedie_scores.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_tweedie_scores_driver.exe --profile checked --report validation/results/tweedie_new_checked.json
python validation/check_tweedie_scores_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_tweedie_scores_driver.exe --profile checked --report validation/results/tweedie_invalid_checked.json
```
Repeat for release/profile. The batched accepted cases check values/nonfinite
masks, input identity and IEEE state, alternating the incoming inexact flag
between clear/set. Source rejections and targeted guards run separately.
Coverage includes all rank pairs, powers near boundaries, wide powers, extreme
ratios, signed/nonfinite weights, D2 early/null-model behavior and buffered
sums. Earlier regression-metric and selected native/core suites are recorded
separately; these commands do not imply all historical suites were rerun.


Pinball loss and D2 quantile/absolute scores:
```
python validation/compare_quantile_losses.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_quantile_losses_driver.exe --profile checked --report validation/results/quantile_loss_new_checked.json
python validation/check_quantile_losses_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_quantile_losses_driver.exe --profile checked --report validation/results/quantile_loss_invalid_checked.json
```
Repeat for release/profile. Accepted calls run in one batch; each source
rejection and targeted guard is isolated. Checks include values, nonfinite
masks, input identity and IEEE state with alternating incoming inexact flags.
Fixtures cover rank pairs, modern/legacy weighting, alpha endpoints, CDF ties,
signed/nonfinite weights, interpolation overflow and conditional early results.
Native/earlier regression/core checks are recorded separately. The old facade's
dead kernels are removed, retaining its public imports and legacy arguments.


Streaming nearest-pair reductions:
```
python validation/compare_pairwise_reductions.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_pairwise_reductions_driver.exe --profile checked --report validation/results/pairwise_reductions_new_checked.json
python validation/check_pairwise_reductions_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_pairwise_reductions_driver.exe --profile checked --report validation/results/pairwise_reductions_invalid_checked.json
```
Repeat with the optimized executable/profile. Comparisons check both public
APIs, exact indices, distance/nonfinite results, input identity and IEEE state.
Fixtures include all supported names, all storage combinations, both axes,
Minkowski powers, squared distances, sparse storage fallbacks, missing overlap,
chunk boundaries, cancellation and overflow. Source expected rejections run
separately. The invalid suite identifies its NaN-power native adapter explicitly.
Older pairwise/core/native checks are recorded separately.


Additional numeric/Boolean nearest-pair dissimilarities:
```
python validation/compare_extra_dissimilarities.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_pairwise_reductions_driver.exe --profile checked --report validation/results/extra_dissimilarities_new_checked.json
python validation/check_extra_dissimilarities_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_pairwise_reductions_driver.exe --profile checked --report validation/results/extra_dissimilarities_invalid_checked.json
```
Repeat for release. The suite separately counts strict sklearn configurations
and the explicitly identified dense-contract adapter for incomplete CSR
Bray-Curtis rows, avoiding the pinned source's exhausted-buffer accesses.
Both methods compare exact nearest indices, distance arrays/nonfinite masks,
API agreement, input identity and IEEE state. Controlled source rejections
run separately. Sixty targeted diagnostics cover source sparse-storage and
finite-input restrictions. Existing nearest-pair/pairwise/core/native suites
are recorded separately.


Parameterized and aliased-input nearest-pair distances:
```
python validation/compare_parameter_distances.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_parameter_distances_driver.exe --profile checked --report validation/results/parameter_distances_new_checked.json
python validation/check_parameter_distances_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_parameter_distances_driver.exe --profile checked --report validation/results/parameter_distances_invalid_checked.json
```
Repeat for release. Cases compare both APIs, exact indices, distance/nonfinite
arrays, target/parameter identity and IEEE state. They include all storage/axis
forms, general variance/precision, weighted powers, self-object fallbacks,
automatic self covariance and the default auto-strategy threshold. Source
rejection records identify representative native probes. Targeted diagnostics
label six source-unsafe precision shapes (reference not called) and the native
NaN-power guard separately. Other fixtures are strict sklearn comparisons.


Pairwise kernel dispatch:
```
python validation/compare_kernel_dispatch.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_kernel_dispatch_driver.exe --profile checked --report validation/results/kernel_dispatch_new_checked.json
python validation/check_kernel_dispatch_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_kernel_dispatch_driver.exe --profile checked --report validation/results/kernel_dispatch_invalid_checked.json
```
Repeat for release. Accepted calls compare kernel matrices, nonfinite masks,
input identity and IEEE state. Cases cover each named route/storage combination,
filtering and permissive applicable parameters, integer/real powers, callbacks,
GP objects, precomputed storage, aliasing, tiny norms and pinned BLAS overflow
paths. Reports identify source sparse results that become owned dense arrays.
Controlled source rejections and 34 targeted diagnostics run separately.


Gram least-angle paths and graphical LARS:
```
python validation/compare_lars_gram.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_lars_gram_driver.exe --profile checked
python validation/check_lars_gram_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_lars_gram_driver.exe --profile checked
python validation/compare_graphical_lasso.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_graphical_lasso_driver.exe --mode lars
python validation/compare_graphical_cv.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_graphical_cv_driver.exe --profile checked --mode lars
```
Repeat for release; use --mode cd to retain the original graphical regressions.
The least-angle and least-angle-selection validators also accept --executable
to validate a completed build without invoking fpm again. Gram fixtures include
LAR, Lasso, positive Lasso, interpolation, iteration caps, zero responses,
duplicate columns and wide matrices. Duplicate columns use merged-coefficient
and prediction-path certificates. Wide full paths use eps=0.1; early wide
knots retain machine epsilon. The existing backend's machine-epsilon
rank-deficient full-path limitations are documented in the API and coverage
records. A separate guard script checks 11 controls/nonfinite-input errors;
the central input-error registry adds empty/shape Gram guards and retains
unknown graphical-mode rejection after replacing its former LARS fixtures.

Complete each Windows build before running its executables, so the linker
can replace the output files. Checked and release validators should run
sequentially when they share reference input/output paths.

The optimized least-angle selection regression has one existing
rank-deficient LarsCV MSE tolerance miss (seed 2026, case1_mse). The default
validator keeps its original strict tolerance. For an explicit nonregression
certificate, --baseline-executable may name a separately compiled pre-change
reference driver: the entire 200-group fixture must match that binary exactly.
The report retains the source-comparison failure and baseline provenance
separately; this option does not widen numerical tolerances.


Direct and Gram OMP paths:
```
python validation/compare_omp_paths.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_omp_paths_driver.exe --profile checked
python validation/check_omp_paths_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_omp_invalid_driver.exe --profile checked
```
Repeat for release. The reference driver batches 2,310 configurations, testing
owned coefficients/paths, per-target iterations, backend selection, warning
flags and input retention. Column-wise source solves supply ragged paths and
precomputed iteration counts; source wrapper failures are recorded separately.
Duplicate atoms use merged-coefficient/prediction certificates. Numerical
full-span certificates retain rank-deficient extra-step cases and verify
common prefixes, every extra reconstruction and backward error. Exact-zero
tolerance warning-state differences are explicitly retained. Twenty-seven
diagnostics include eight identified native finite/shape/pivot contracts.


Dense elastic-net/Lasso paths:
```
python validation/compare_coordinate_paths.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_coordinate_paths_driver.exe --profile checked
python validation/check_coordinate_paths_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_coordinate_invalid_driver.exe --profile checked
```
Repeat for release. The batched driver covers scalar/column/multi-task targets,
Gram/direct/auto/supplied precomputation, supplied/discarded correlations,
alpha sorting/duplicates/empty arrays, automatic grids, warm initial values,
cyclic/random coordinate sequences, positivity, zero sweeps and degenerate
data. All alpha and coefficient arrays remain direct sklearn comparisons.
When stopping metadata differs, retained certificates verify zero-model KKT
conditions or independent zero-L1 ridge/least-squares optimality within the
requested tolerance. Reports keep source and native gaps/counts/status plus
certificate bounds. This does not widen coefficient tolerances or claim exact
stopping equivalence at a floating-point zero boundary. Thirty-nine guards
include thirteen native finite/shape/convex-domain constraints.


Sparse elastic-net/Lasso paths:
```
python validation/compare_sparse_coordinate_paths.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_sparse_coordinate_driver.exe --profile checked
python validation/check_sparse_coordinate_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_sparse_coordinate_invalid_driver.exe --profile checked
```
Repeat for release. Cases retain canonical/unsorted/zero/duplicate storage,
unweighted/fractional/zero/signed weights, implicit means/scales, positive
coefficients, random and cyclic paths, warm starts, ignored Gram/Xy/scale
inputs, automatic and explicit grids, and zero sweeps. A 32768-by-8192
design with one stored entry exercises storage whose dense equivalent is
2 GiB. An explicit-grid case avoids unused overflowing raw correlations.
All nine result groups per configuration compare directly to source; the
raw duplicate uint32 implicit-zero wrap is retained explicitly. Thirty-three
diagnostics identify twenty-one native buffer/finite/mean/domain guards.


Public ridge regression solvers:
```
python validation/compare_ridge_solver.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_ridge_solver_driver.exe --report validation/results/ridge_checked_report.json
python validation/check_ridge_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_ridge_invalid_driver.exe
```
Repeat with the release executable directory. The 336 configurations retain
672 direct comparison groups: coefficients at 3e-8 absolute/relative tolerance
and exact source LSQR iteration counts (other solvers use -1 for None).
Tall, wide and square designs cover one/multiple targets, per-target penalties,
dense/CSR storage, sample weighting, all supported solver choices, positivity
and iteration caps. Degenerate cases include zero design/targets/weights,
rank deficiency and zero penalties, including primal SVD and dual least-squares
Cholesky fallbacks. No numerical stopping exemptions are used.
The report records dependency versions and source/validator/executable hashes.
Thirteen isolated guards include finite/domain/shape constraints and rejected
unsupported solver combinations. The native program checks all eight typed
response/penalty/storage overloads, weighted analytic solutions, zero LSQR
iterations, preserved inputs and a 100000-feature matrix-free CSR problem.

Checked and optimized fpm 0.14 builds passed on 2026-11-25. Both profiles
passed all 672 ridge groups (maximum scaled error 3.2798103331884335e-9),
13 rejection processes, eight native programs and the example. Selected
regressions passed 192 core and 27,684 earlier coordinate-path groups per
profile; the earlier suite keeps its documented stopping certificates.
The final audit retained stable source manifests, 196 compiled-unit
fingerprints per profile and distinct checked/release executable hashes.
Artifacts are ignored under validation/results/ridge_*; this record does not
claim a rerun of every historical suite. LSQR's separate Stanford BSD notice is
retained in LICENSES/lsqr-BSD.txt alongside the project and upstream licenses.


Squared-loss SAG/SAGA ridge:
```
python validation/compare_ridge_stochastic.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_ridge_stochastic_driver.exe --report validation/results/ridge_stochastic_checked_new.json
python validation/check_ridge_stochastic_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_ridge_stochastic_invalid_driver.exe
```
Repeat with the release executable directory. The batched suite retains 1,533
configurations / 4,599 direct groups: coefficients, intercepts and exact epoch
counts. It covers seeded dense/CSR tall, wide and one-sample problems, multiple
targets, per-target penalties, omitted/unit/fractional/zero/signed weights,
zero/short/full epoch caps, direct/auto SAG intercepts and SAGA. Additional
fixtures preserve unsorted/duplicate/explicit-zero CSR entries, force internal
scale resets with large penalties, and exercise zero designs, targets and penalties.
No numerical exemptions or relaxed iteration comparisons are used. The report
retains configuration labels, input/native/reference/validator/executable
hashes and upstream warning counts. Fifteen isolated diagnostics cover bad
seeds, solver/intercept/positive combinations, finite/shape/domain constraints,
zero Lipschitz constants and SAG's unit step-times-penalty rejection.
The native program additionally verifies analytic weighted solutions,
coefficient-only stopping, the source zero-epoch count and a 100000-feature CSR
problem. Python warning emission is represented by native status fields.

Checked and optimized fpm 0.14 builds passed on 2026-11-25. Both profiles
passed all 4,599 stochastic ridge groups, with exact epoch counts and maximum
scaled coefficient/intercept error 1.0359420622090646e-15. Source warning counts
were 1,423 per profile. Each profile also passed 15 new guards, 672 previous
ridge groups / 13 guards, nine selected native programs, two examples, 192 core
comparisons and 27,684 earlier coordinate-path groups (including their existing
stopping certificates). Source fingerprints for 200 compiled units per profile,
stable manifests, identical inputs and distinct profile executables passed the
final audit. Deterministic ridge kernel bodies remain byte-identical to
00b398d. Other historical suites were not rerun. The reports and audit are
ignored under validation/results/ridge_stochastic_*.


Streamed OPTICS graphs:
```
python validation/compare_optics_graph.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_optics_graph_driver.exe --allow-rounding-certificates --report validation/results/optics_graph_checked_new.json
python validation/check_optics_graph_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_optics_graph_invalid_driver.exe
python validation/compare_optics.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_optics_driver.exe --report validation/results/optics_graph_checked_old.json
```
Repeat with the release executable directory. The new suite retains 1,280
configurations / 6,400 groups. It covers three seeds, integer/fractional sizes,
finite/zero/infinite radii, Minkowski powers including subunit and infinite p,
Euclidean/squared Euclidean, Manhattan aliases, Chebyshev, cosine including
zero/tiny norms, row callbacks, asymmetric/nonzero-diagonal precomputed input,
plateaus and overflow in decimal rounding.

By default the validator compares directly with source and rejects numerical
ordering differences. The explicit certificate flag permits four Minkowski
p=1.5 configurations to replay sklearn's radius queries and _set_reach_dist
along the native permutation. Every selected frontier value, near-minimum
choice, predecessor edge and final state must satisfy the retained roundoff
bounds. Core distances remain direct source comparisons. Reachability in
these configurations compares to replay, with its difference from the
original source graph retained. One of these graphs has a later reachability
value that differs materially after its changed tie, so it is not a direct
source result match. Precomputed graphs never use this exception. The remaining
1,276 configurations retain direct ordering/predecessor comparisons and tight
3e-14 distance comparisons. This is not a general widening of distance or
index tolerances.

Sixteen input rejections cover size/fraction, radius, power, metric, shape,
finite/nonnegative data and callback contracts. A native 8,192-point analytic
case exercises streamed storage whose full distance matrix would use 512 MiB.
The estimator is checked against all 432 earlier OPTICS groups, preserving
its prior supported metric conventions. The existing extraction and kernel
algorithm bodies remain unchanged; native scalar BLAS helpers gain internal
aliases for reuse.

Checked and optimized fpm builds passed on 2026-11-25. Both profiles retained
1,276 direct graph configurations and four replay certificates. Maximum
compared source/replay scaled error was 9.022582387150723e-16. One certificate
retains a material later reachability difference from the original source
ordering (maximum scaled error 0.0024488865025990426); it is not treated as a
direct source output match. Each profile recorded 578 source warnings and
passed 16 guards, 432 previous OPTICS groups, seven native programs, the
example, 192 core comparisons and 6,015 kernel groups / 276 source-rejection
probes. Source/dependency fingerprints for 196 compiled units per profile,
stable manifests, identical inputs and distinct executables passed the final
audit. Other historical suites were not rerun. Artifacts remain ignored under
validation/results/optics_graph_*.


Blocked Ledoit-Wolf shrinkage:
```
python validation/compare_shrinkage.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_shrinkage_driver.exe --report validation/results/shrinkage_checked_new.json
python validation/check_shrinkage_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_shrinkage_invalid_driver.exe
```
Repeat with the release executable directory. The suite retains 1,080
configurations / 5,112 direct comparison groups: public coefficients and
estimator shrinkage, locations, covariance and precision matrices. Three seeds,
tall/wide/one-sample/one-feature data, centered/raw inputs, zero/constant/rank-
deficient/isotropic/shifted cases and several block/remainder sizes are covered.
Extreme finite scalar inputs exercise underflow, overflow, quiet NaN behavior
and the one-feature shortcut. The native test independently checks analytic
rank-one shrinkage, estimator covariance, all IEEE flags/modes and an
8,192-feature case avoiding a 512 MiB cross-moment matrix. Seven isolated
rejections cover block sizes, empty/nonfinite input and estimator validation.
One-sample estimator fixtures also cover tiny negative computed coefficients,
which source applies without an additional range clamp. No numerical exceptions
or certificate substitutions are used in this suite; finite values use the
3e-8 source comparison tolerance and NaN patterns must match.

Checked and optimized fpm builds passed on 2026-11-25. Both profiles passed
all 5,112 direct groups, seven rejection checks, five selected native programs,
the example and 192 core comparisons. Maximum scaled error was
1.095989315518997e-8 checked and 1.1367939906712503e-8 optimized, with 774
source warnings per profile. No tolerance exceptions or certificate substitutions
were used. All 193 compiled-unit fingerprints per profile, stable manifests,
identical inputs and distinct executables passed the final audit. Other
covariance algorithm bodies are unchanged. Other historical suites were not
rerun. Artifacts remain ignored under validation/results/shrinkage_*.


Public Ward trees:
```
python validation/compare_ward_tree.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_ward_tree_driver.exe --reference-cache validation/results/ward_reference_cache.json --report validation/results/ward_checked_new.json
python validation/check_ward_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_ward_invalid_driver.exe
```
Repeat with the release executable directory. Omitting the reference-cache
option recomputes every source tree. The optional ignored cache records source
versions, relevant Python/extension hashes, input-specific keys and source
outputs/warnings; provenance mismatches are rejected. Repeated profiles reuse
captured sklearn outputs and retain cache/input/validator/executable hashes.
The initial capture made 5,820 source calls and reused 2,772 identical cases.
No expected trees are inferred from native output.

The 8,592 configurations / 34,368 groups cover three seeds, random/grid/
duplicate/zero data, one-feature and structured single-sample cases, full and
partial trees, omitted/requested distances, disconnected completion, directed/
random/chain/empty/diagonal/cancelling connectivity, dense real/logical matrices
and canonical/unsorted-duplicate/explicit-zero CSR. Children and parents compare
directly; distances use the standard 3e-8 bound, without exceptions or numerical
certificates. Native tests additionally cover all eight typed wrappers, golden
heights/parent forests, IEEE state and a 32,768-node sparse partial forest that
avoids an 8 GiB dense connectivity matrix. Fifteen isolated guards cover shape,
finite input, cluster counts and arithmetic working-range errors.


Checked and optimized fpm builds passed on 2026-11-25. Both profiles passed
all 8,592 configurations / 34,368 direct groups with zero observed error,
15 input guards, five selected native programs, the example and 192 core
comparisons. Each retains 3,968 source warnings from the captured outputs.
Both final profiles reused all 8,592 cached cases. No numerical certificates
or tolerance exceptions were used. All 194 compiled-unit fingerprints per
profile, stable manifests, identical inputs and distinct executables passed
the final audit. Existing cluster algorithm bodies and license/dependency
manifests are unchanged. Other historical suites were not rerun. Artifacts
remain ignored under validation/results/ward_*.


Fitted structured Ward clustering and feature agglomeration:
```
python validation/compare_structured_ward.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_structured_ward_driver.exe --reference-cache validation/results/structured_ward_reference_cache.json --report validation/results/structured_ward_checked_new.json
python validation/check_structured_ward_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_structured_ward_invalid_driver.exe
```
Repeat with the release executable directory. The optional ignored cache stores
actual installed-sklearn outputs, source versions/module hashes, input-specific
keys and warnings; changed provenance is rejected. Omitting it recomputes the
source fits. No expected values are inferred from native output. Labels and
feature-pooling columns are canonicalized by first sample/feature to match the
established native convention; tree children and metadata compare directly.
Distances, pooled values and inverse maps use the unchanged 3e-8 bound.

The suite exercises dense real/logical and canonical/duplicate/explicit-zero
CSR connectivity, full/partial/auto builds, requested/omitted distances,
cluster-count and threshold cuts, repeated fitting, all three pooling modes,
and the 100-cluster auto boundary. Three seeds, random/grid/duplicate/zero data,
one-feature/two-sample cases, threshold equality and disconnected/cancelling
connectivity are included. Native tests add analytic cuts and pool/inverse
values, IEEE preservation and a 32,768-node sparse auto forest avoiding an
8 GiB dense distance/connectivity matrix. Twenty-three isolated rejections
cover optional-control combinations, domain/shape errors and single-linkage precomputed-shape and threshold/full-tree
requirements. Existing FeatureAgglomeration and BIRCH
validators accept --executable and --report for serialized profile checks.


Full checked and optimized fpm builds passed on 2026-11-25. Each profile passed
11,826 configurations / 64,392 source comparison groups with zero observed
error, 23 isolated input guards, six selected native programs, the example,
192 core comparisons, 294 previous FeatureAgglomeration groups and 237 BIRCH
groups. There are 3,888 threshold cases and 8,544 feature-agglomeration cases.
No tolerance exceptions or numerical certificates are used. Each profile
retains 6,561 source warnings; initial capture made 10,107 source estimator fits and
reused 1,719 identical cases. Both final profiles reused all 11,826 captured
cases. The final audit verifies 197 compiled-unit fingerprints per profile,
stable manifests, identical inputs, distinct executables and unchanged license/
dependency declarations. Other historical suites were not rerun. Artifacts
remain ignored under validation/results/structured_ward_*.

The source_calls field counts fitted-estimator reference calls. Each run also
makes 36 source Ward-tree calls to select threshold fixtures, including when
all fitted-estimator cases are cached.


Complete/average linkage trees and fitted estimators:
```
python validation/compare_linkage_tree.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_linkage_tree_driver.exe --reference-cache validation/results/linkage_reference_cache.json --report validation/results/linkage_checked_tree.json
python validation/compare_linkage_fit.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_linkage_fit_driver.exe --reference-cache validation/results/linkage_fit_reference_cache.json --report validation/results/linkage_checked_fit.json
python validation/check_linkage_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_linkage_invalid_driver.exe
```
Repeat with release executables. Optional ignored caches retain actual source
outputs, warning counts, input keys and source/runtime provenance; changes are
rejected. Tree children and parents compare exactly. Fitted labels and pooled
columns use the established first-sample/feature order; children, partitions
and metadata otherwise compare directly. Distances, pooled values and inverse
maps use the unchanged 3e-8 bound without numerical certificates.

The tree suite covers 9,904 configurations / 39,616 groups, dense/logical/
canonical/duplicate/explicit-zero connectivity, all supported named metrics,
precomputed asymmetric/nonzero-diagonal/signed matrices, partial forests,
disconnected completion, aliases, wide cosine rows and large tied heaps. The
fitted suite covers 11,826 configurations / 64,392 groups, repeated fitting,
full/partial/auto modes, threshold/cluster-count cuts, distance retention, all
three feature-pooling modes and the 100-cluster auto boundary. Its source_calls
counter counts estimator fits; 36 additional source complete-linkage trees per
run select threshold fixtures. Neither suite infers expected values from native
output. The native test checks all eight tree wrappers, missing-neighbor copy
semantics, analytic trees/thresholds/pooling, an exact NumPy-rounding threshold
regression, IEEE state, and an 8,192-node CSR partial tree avoiding a 512 MiB
dense distance/connectivity matrix. Twenty-five isolated guards include the
source's unsupported empty structured tree and connected metric restrictions.


Full checked and optimized fpm builds passed on 2026-11-25. Each profile passed
all 104,008 new groups, with exact children/parents/canonicalized partitions
and maximum scaled numerical error 2.2938989500484793e-16. No tolerance
exceptions or numerical certificates are used. Each also passed 48 guards
(25 new plus 23 fitted-Ward), seven selected native programs, the example,
64,392 prior Ward groups, 192 core groups, 294 previous FeatureAgglomeration
groups and 237 BIRCH groups. Other historical suites were not rerun.

The final audit checks 201 compiled-unit fingerprints per profile, stable
source/validator/executable/attribution manifests, identical inputs, distinct
executables and unchanged license/dependency declarations. Twenty original
Ward procedures and 35 unrelated cluster procedures retain their bodies.
Both final profiles reuse all captured tree/estimator cases. Initial captures
made 8,481 tree calls / 1,423 duplicate reuses and 9,895 estimator fits / 1,931
reuses, in addition to the fitted suite's 36 source threshold-fixture selectors
per run. Artifacts remain ignored under validation/results/linkage_*.


Single-linkage spanning trees and fitted estimators:
```
python validation/compare_single_tree.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_single_tree_driver.exe --reference-cache validation/results/single_reference_cache.json --report validation/results/single_checked_tree.json
python validation/compare_single_fit.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_single_fit_driver.exe --reference-cache validation/results/single_fit_reference_cache.json --report validation/results/single_checked_fit.json
python validation/check_single_invalid_inputs.py --executable build/gfortran_E1FE13DB8881E18E/app/reference_single_invalid_driver.exe
```
Repeat with release executables. Optional ignored caches contain actual source
outputs, warning counts, input keys and source/runtime hashes, including the
SciPy sparse-MST and sklearn distance-metric extensions. Omitting caches
recomputes references. Tree children and parents compare exactly; fitted labels
and pooled columns use the native first-sample/feature order. Other values use
the unchanged 3e-8 bound, without numerical certificates.

The tree suite covers 5,022 configurations / 20,088 groups, fast/fallback
orientation, full-child/partial-parent forests, all supported named metrics,
dense/logical/canonical/duplicate/explicit-zero connectivity, disconnected
completion, asymmetric/signed precomputed matrices, tiny/zero/genuine-epsilon
edges, aliases and the connected precomputed singleton. The fitted suite
covers 11,886 configurations / 64,722 groups, refits, full/partial/auto modes,
thresholds, all feature pooling modes, one-cluster-per-sample forests and the
100-cluster auto boundary. Sixty cases target epsilon threshold behavior.
Its source_calls counter counts estimator fits; 36 additional source single
Euclidean trees per run choose threshold fixtures. Expected outputs are never
inferred from native results.

Native checks cover all eight typed tree forms, oriented versus sorted child
roots, epsilon collision behavior, partial parents with full children/heights,
feature pooling, threshold equality, IEEE state, a streamed 4,096-sample tree
and a 16,384-node sparse forest avoiding a 2 GiB dense graph distance matrix.
Twenty isolated guards cover shape/domain errors and private-tree versus
estimator metric-alias rules. Prior guard registries now test valid current
restrictions instead of rejecting newly supported single-linkage options.


Full checked and optimized fpm builds passed on 2026-11-25. Each profile passed
all 84,810 new groups, with exact children/parents/canonicalized partitions.
Maximum scaled numerical errors were 6.992240957504238e-17 for trees and
2.453147316499086e-16 for fitted outputs. No tolerance exceptions or numerical
certificates were used. Both profiles also passed 68 guards, eight selected
native programs, the example, 104,008 previous complete/average groups,
64,392 Ward groups, 192 core groups, 294 prior FeatureAgglomeration groups and
237 BIRCH groups. Other historical suites were not rerun.

The final audit verifies 205 compiled-unit fingerprints per profile, stable
source/validator/executable/attribution manifests, identical inputs, distinct
executables and unchanged license/dependency declarations. Thirty-five existing
hierarchy, 37 cluster and 18 ranking procedures retain their bodies. Both final
profiles reused every captured tree/estimator case. Initial capture made 4,326
tree calls / 696 duplicate reuses and 9,955 estimator fits / 1,931 reuses, plus
the fitted suite's 36 source threshold-fixture trees per run. Tree reports
retain 2,885 source warnings and fitted reports retain 6,561 per profile.
Artifacts remain ignored under validation/results/single_*.

## Joint neighbor regression

`compare_neighbor_outputs.py` compares actual installed sklearn 1.6.1
KNeighborsRegressor and RadiusNeighborsRegressor (`algorithm='brute'`) with the
Fortran vector/matrix overloads. Pass `--executable`, `--report` and optionally
`--reference-cache`. The full 5,280-configuration inventory covers three seeds,
four training shapes, regular/duplicate points, five metrics, uniform/distance
weights, one/three targets, explicit/self queries, neighbor counts and radius
zero/.8/infinity. Metadata and retained targets are compared even when Source
rejects prediction; rejected score calls retain prediction comparisons and
receive separate native rejection probes. Native stdout uses bounded file-backed
batches. Reference caches preserve Source versions, source hashes and requests.

Default execution remains strict at 3e-8 absolute/relative tolerances, with exact
metadata and matching NaNs. `--record-differences` retains every failed field;
`--regular` explicitly selects regular-point fixtures. Fixed-count cutoff ties
can select different target rows: native stable training order differs from
Source brute selection. No conditioned oracle or broadened tolerance hides
those differences. `check_neighbor_regression_invalid_inputs.py` checks 28
actual Source rejections and the native vector-prediction rank contract.

`test_neighbor_outputs` checks analytic joint predictions, weighted scores,
IEEE state, self exclusion, zero-distance weighting, precomputed geometry,
radius limits and vector refits. `test_neighbor_frozen_outputs` checks borrowed
fitted-state detection, joint prediction/scoring and vector/matrix refits.
`neighbor_outputs` demonstrates the public facade.

The checked and optimized package grids agree exactly. Each regular-point run
passes 11,540 groups across 2,640 fits (maximum scaled error
9.765330951305197e-16), including 240 prediction and 940 score rejections.
Each full run checks 23,088 groups across 5,280 fits, with 480 prediction and
1,872 score rejections. It retains 492 failed fields across 164 duplicate-point
KNN configurations: prediction, matrix-helper prediction and R2 each fail 164
times. All radius cases pass. Maximum scaled prediction error is
.6930737981278454 and maximum scaled R2 error is .6547419723635348. These
material tie-selection differences remain failures in default strict mode.
The two profiles use identical input inventories and Source cache content;
their reference executables have distinct hashes. Artifacts remain ignored
under `validation/results/neighbor_package_*`.

Both complete fpm builds succeed. Each profile passes 15 selected native tests,
the example, all 29 input checks, 120 retained graph/neighbor comparisons and
192 core comparisons. Its 22 verification jobs have expected outcomes,
including the deliberate full strict failure. The final audit verifies 212
dependency-aware compiled fingerprints per profile (192 library units and 20
programs), current source/validator/API/attribution manifests, Source cache
provenance and distinct executable hashes. No additional dependency was added;
the package retains its MIT license and existing third-party BSD notices.

## Joint KNN classification

`compare_neighbor_classification_outputs.py` compares actual installed sklearn
1.6.1 KNeighborsClassifier (`algorithm='brute'`) with shared-search joint labels,
ragged probability lists, vector/column targets and weighted accuracy. Pass
`--executable`, `--report` and optionally `--reference-cache`. Its full inventory
contains 2,240 configurations across two seeds, three shapes, regular/duplicate
points, five metrics, uniform/distance weights, binary/ragged classes, one/three
targets, explicit/self queries and one/three/all neighbors. Default execution
compares labels and metadata exactly and probabilities/scores at unchanged 3e-8
absolute/relative tolerances. `--regular` selects regular-point fixtures;
`--record-differences` preserves every full-grid failure. Cached Source results
include version/source hashes, requests and warnings. Native evaluation uses
bounded file-backed output batches and separate rejection probes.

`check_neighbor_classification_invalid_inputs.py` checks 19 actual Source
rejections plus two native matrix/list rank guards. Analytic native tests check
shared outputs, differing class counts, self exclusion, zero-distance weights,
precomputed geometry, IEEE state, scalar refits and weighted subset accuracy.
The frozen test checks borrowed fitted-state detection and scalar refits; its
scalar prediction APIs do not gain joint classification adapters.

Both checked and optimized builds pass all 8,660 regular-point groups across
1,120 fits, including 240 prediction and 220 score rejections. Maximum scaled
probability error is 6.427874999854559e-16; labels and metadata match exactly.
Each full report retains 17,320 groups across 2,240 fits, with 480 prediction
and 440 score rejections. There are 383 failed fields across 162 duplicate-point
configurations: matrix labels 91, vector labels 37, joint probabilities 162,
scalar probabilities 70 and accuracy 23. The 16,937 remaining groups pass.
Maximum scaled label/probability/accuracy errors are respectively
3.3333333333333335, .3333333333333333 and .16967509025270755. The failures are
material tie-selection differences and remain failures in default strict mode.

Both full fpm builds succeed. Each profile passes 21 selected native tests,
the example, all 21 classification and 29 retained regression input checks,
120 legacy graph/neighbor comparisons and 192 core comparisons. The complete
5,280-fit regression grid retains the preceding increment's identical 492
failed fields across 23,088 groups. The checked profile completes 30 jobs with
expected outcomes, including a deliberately failing full strict comparison;
optimized verification completes 29 jobs. The audit verifies 220 compiled-unit
fingerprints per profile, current source/validator/API/attribution manifests,
matching Source caches and inputs, identical classification failure records,
and distinct executables. Artifacts remain ignored under
`validation/results/knc_package_*`. MIT licensing and dependency declarations
are unchanged; the existing third-party notices are retained.

## Joint radius-neighbor classification

`compare_radius_classification_outputs.py` compares installed sklearn 1.6.1
RadiusNeighborsClassifier with the typed matrix/list helpers. Pass
`--executable`, `--report` and optionally `--reference-cache`. The 2,882-case
inventory covers two seeds, three shapes, regular/duplicate points, five
metrics, both weights, binary/ragged classes, one/three targets, explicit/self
queries and radius zero/.8/infinity. Error/scalar/list/most-frequent outlier
policies rotate across the grid. Two supplemental precomputed fixtures probe
finite-distance reciprocal overflow, exact-zero domination and NaN argmax.
Labels, classes, metadata and fitted outlier labels compare exactly;
probabilities/scores use unchanged 3e-8 absolute/relative tolerances and matching
NaNs. Default full execution is strict; diagnostic mode retains every mismatch.
Cached Source calls preserve versions, source hashes and the integer adapter.
Native output uses bounded file-backed batches and separate rejection probes.

Source labels are normalized to int64 with Python integer/list outlier
parameters. The Fortran API compares fixed-kind label values and does not model
Python dtype-promotion checks. `check_radius_classification_invalid_inputs.py`
pairs 23 actual Source rejections with native diagnostics and checks three
native rank/enum contracts. It additionally retains two Source-only int32 and
NumPy-scalar rejection observations, explicitly marked unrepresented in the
native API. They are not counted as paired native rejections. Native analytic
tests cover joint votes, known/unknown/per-output outliers, most-frequent ties,
zero/infinite radii, precomputed geometry, IEEE state, overflow NaNs, weighted
subset accuracy and vector/column refits. Earlier live legacy-label tests remain.

Both checked and optimized full fpm builds succeed. Their strict full radius
runs pass all 25,851 groups across 2,882 fits, with 456 matching prediction
rejections and 689 matching score rejections. Maximum scaled probability error
is 6.427874999854559e-16; labels, class lists, fitted outlier labels, raw targets,
metadata and accuracy match exactly, including the overflow fixtures' NaN
patterns. Each report retains 3,880 Source warnings.

Each profile passes 22 selected native tests, the example, all 76 input/rank/enum
checks across radius classification, KNN classification and neighbor regression,
120 legacy graph/neighbor groups and 192 core groups. The two Source-only dtype
observations remain separate. Each completes 31 verification jobs with expected
outcomes. The complete KNN grid retains its identical 383 failed fields across
17,320 groups/2,240 fits, and the complete neighbor-regression grid retains its
identical 492 failed fields across 23,088 groups/5,280 fits. Known cutoff-tie
differences remain documented rather than being reclassified as passes.

The audit verifies 223 compiled-unit fingerprints per profile, current
source/validator/API/attribution manifests, matching inputs and Source caches,
unchanged old failure records and distinct executables. Artifacts remain
ignored under `validation/results/rnc_package_*`. No dependency or licensing
change was introduced.

## Functional neighbor graphs and CSR row-value sorting

`compare_functional_neighbor_graphs.py` calls the actual installed sklearn 1.6.1
free graph builders and sorter. Pass `--executable`, `--report` and optionally
`--reference-cache`. The full 3,972-case inventory covers two seeds, four shapes
through 65 rows, regular/duplicated geometry, dense/CSR features and precomputed
graphs, multiple neighbor counts/radii, boolean/auto self inclusion, mode controls
and fitted neighbor models. Additional cases cross the automatic KD-tree leaf
boundary, mutate fitted public parameters, omit stored diagonals, and retain
noncanonical CSR coordinate buffers. Sort cases include rectangular/empty shapes,
copy/warning flags, duplicate columns, explicit zeros, NaNs and infinities.

Rows/columns/counts/pointers and initial sortedness compare exactly; data and
dense graph values use unchanged 3e-8 absolute/relative tolerances and matching
NaNs. Raw CSR indices/data are never canonicalized in the comparison. Dense
graphs are an additional field, not a replacement oracle. Default full execution
is strict; `--record-differences` retains every failed field. `--sort-only`
independently checks the complete sorting inventory. Source cache provenance
records graph/base source hashes and owned-return, warning and brute-backend
adapters. Native output uses bounded file-backed batches; Source rejection cases
receive isolated native calls.

`check_functional_neighbor_graphs_invalid_inputs.py` pairs 42 live Source
rejections with native diagnostics and adds two native-only malformed-CSR safety
contracts. It avoids passing unsafe CSR buffers into SciPy compiled kernels.
Native tests check graph semantics, all supported fitted families, parameter
snapshots, exact-k/self controls, raw CSR arithmetic, stable mutation/copy and
NaN/leading-empty sortedness behavior with preserved IEEE flags/modes.

GNU Fortran 17 checked/optimized full functional reports agree: 3,972 cases,
2,942 successful calls, 1,030 matching rejections and 26,478 comparison groups.
They retain 625 failed fields across 435 configurations (435 index fields,
184 dense graph fields, six raw data fields). Thus 25,853 groups pass. Six
explicit automatic-tree cases differ in storage order; changed edge placement
remains a material failure. Maximum scaled index/data/dense errors are
30, 11.227754979715137 and 10.940319379974252. All noncanonical CSR probes and
fitted-parameter mutation probes pass. The strict sort selection passes 1,332
groups exactly, with 148 successful calls and 32 matched Source rejections.

Both full fpm builds succeed. Each profile passes 24 selected native tests,
the example, 162 input/contract checks, 25,851 radius groups, 5,550 transformer
groups (1,110 fixtures and 245 retained tie certificates), 120 graph/neighbor
groups and 192 core groups. The checked primary run completes 39 jobs with
expected outcomes; optimized primary verification completes 38 jobs with one
unexpected incomplete-frame failure in the legacy KNN grid.

That native SIGSEGV reproduced once in an exact batch replay, with partial
output matching a later successful output prefix. The root cause remains
unresolved. Forty stress batch runs, two ordinary replays, 13 debugger launches,
eight attach-debugger runs, six isolated-case runs, six prefix runs and six
output-only runs then completed normally. Original failure evidence remains
under `fng_knc_*` and the initial primary manifests/outcomes are preserved.
The KNN reader now records only complete frames and saves failed-batch input,
stdout, stderr and return-code evidence. Both complete KNN reruns pass with the
same 383 tied fields; the regression grids retain the same 492 failed fields.
These reruns do not erase the original crash. The audit checks 228 compiled
fingerprints per profile, primary/current manifests, Source caches and distinct
executables.

The independent full optimized GNU Fortran 15.2 build succeeds using
`C:/Programs/Octave-11.1.0/mingw64/bin/gfortran.exe`, fpm 0.14.0 and flags
`-O3 -funroll-loops -fcoarray=single`, in a separate temporary build directory.
Only the verification processes prepend that compiler's runtime directory to
PATH; the default compiler and system PATH are unchanged. All 32 selected jobs
complete normally: 24 native tests, the example, strict sorting, the full
functional/KNN/radius/regression grids, legacy transformers and core checks.
The audit validates 224 compiled fingerprints (193 library units and 31 programs)
and exact equality with GNU Fortran 17 for inventories, input/cache hashes,
retained differences and maximum scaled errors. This includes 625 functional
graph differences, 383 KNN differences and 492 regression differences; strict
sorting, radius classification, transformer and core checks pass.

Forty further exact replays of the previously crashing 64-case batch complete
normally under GNU Fortran 15.2, with all 64 frames per replay. The original
input hash is `b08c2a504eaf212f4f0627151e5bba8fbd982f30a738bafcc651da36fef9644c`.
Successful independent runs do not establish a compiler defect, root cause or
repair for the GNU Fortran 17 failure. The independent evidence is retained in
`fng_gcc15_*`, including build logs, manifests, reports, audit and stress outputs.
Artifacts remain ignored under `validation/results/fng_*`. Other historical
suites were not rerun.
