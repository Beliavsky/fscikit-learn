# Translation work log

## Initial package milestone (2026-11-17/18)

The requested target directory was empty. The reference source was recovered
from the locally installed scikit-learn 1.6.1 Python package. A standalone native
Fortran fpm package was built in this directory, with no external runtime or
network dependencies. SciFort remains available for subsequent backend work.

Implemented 56 public estimator/transformer/splitter types across preprocessing,
linear models, Naive Bayes, neighbors, clustering, decomposition, CART trees,
ensembles, model selection, feature selection, text features, dummy estimators,
kernel ridge, discriminant analysis, SVC, Gaussian mixtures, MLPs, covariance and
fixed-kernel Gaussian-process regression, plus metrics, kernels and generators.

Validation includes 14 native test programs, a 192-comparison Python reference
suite, 16 isolated invalid-input tests and four runnable examples. See
`validation/README.md` for the comparison scope and exceptions. Direct Jacobi SVD
was added after initial normal-equation work to preserve nearly collinear
least-squares solutions; tests include rank-deficient, nearly collinear, wide,
and zero matrices.

## Git checkpoint status

`git init` succeeded. The first `git add .` and `git commit` failed with:

```
fatal: Unable to create 'C:/fortran/fscikit-learn/.git/index.lock': Permission denied
```

The session's filesystem policy explicitly grants read-only access to `.git`,
despite permitting writes to the source workspace. A request to enable write
access was sent to the user while implementation continued. No existing user
repository or source work was overwritten. The restriction was not bypassed or
worked around with an alternate repository location.

On 2026-11-18 the user enabled unrestricted filesystem access and requested a
commit. The tested initial package is being recorded as one initial commit,
including the implementation, examples, validation scripts and documentation.

## Graph, neighbor and outlier algorithms (2026-11-18)

Added eight further scikit-learn counterpart types: radius-neighbor
classification/regression, nearest centroids, Local Outlier Factor, Kernel PCA,
Isomap, LabelPropagation and LabelSpreading. Kernel PCA is re-exported from
`sklearn_decomposition` and LOF from `sklearn_neighbors`, with focused modules
also available. All are exported by the core facade; no extension was added.

Added two native test programs, a semi-supervised example, 120 numerical/status
comparisons against sklearn 1.6.1 and ten additional input-error cases. The
existing 192 reference comparisons continue to pass. Scope, API, provenance and
coverage documentation now distinguish the supported dense subsets and their
remaining limitations.

## Bayesian/sparse regression, factors and projections (2026-11-18)

Added six core sklearn counterparts: BayesianRidge, ARDRegression,
OrthogonalMatchingPursuit, FactorAnalysis, GaussianRandomProjection and
SparseRandomProjection. Bayesian and sparse regressors inherit the existing
linear-regression prediction, scoring and cross-validation APIs. FactorAnalysis
is re-exported from `sklearn_decomposition`; projections have a focused
`sklearn_random_projection` module. All six are exported by `use sklearn`.

The increment adds two native test programs, 204 reference comparisons across
tall and wide datasets, nine diagnostic checks and a Bayesian regression
example. Reference checks distinguish random-matrix inputs from independently
checked outputs. Documentation records wide Bayesian covariance null-space
ambiguity and the ARD inverse/convergence differences, alongside the supported
dense, unrotated and single-target subsets. No extension or runtime dependency
was added. See `VALIDATION.md` for the completed verification record.

## Robust regression and generalized linear models (2026-11-18)

Added five core sklearn counterparts: HuberRegressor, RANSACRegressor,
PoissonRegressor, GammaRegressor and TweedieRegressor. They use the existing
linear-model inheritance for prediction/scoring/CV, with GLM inverse-link
prediction and D2 scoring overrides. Huber jointly fits scale and coefficients
using bounded BFGS; RANSAC selects a consensus then refits ordinary least
squares. GLMs use damped Newton updates and Hessian pseudoinverses, supporting
singular unpenalized designs. Tweedie formulas avoid cancellation near powers
one and two; roundoff-aware line searches still require gradient improvement.

Added two native test programs, a robust regression example, 213 numerical
reference comparisons and twelve input-error cases. The documented scope is
dense single-target regression, built-in linear RANSAC, and Tweedie powers zero
and [1,2] with their standard links. No public extension or dependency was added.
The total is now 75 core types. See `VALIDATION.md` for verification results.

## Mode/spectral clustering and kernel density (2026-11-18)

Added MeanShift, AffinityPropagation, SpectralClustering, SpectralEmbedding and
KernelDensity, plus the sklearn `estimate_bandwidth` function. Mode seeking
deduplicates endpoints and suppresses nearby centers by intensity; affinity
propagation performs damped responsibility/availability updates with exemplar
refinement. Spectral methods use normalized-Laplacian eigenvectors and k-means
or pivoted QR assignment. Kernel density evaluates weighted, normalized mixtures
in log space and samples Gaussian/uniform-ball kernels.
Spectral degrees exclude diagonals before summation to avoid cancellation from
large self loops; symmetry checks use entry-wise tolerances. Native regression
tests cover both cases.

New implementations are re-exported from the corresponding cluster, manifold
and neighbor modules and the core facade. Added two native test programs, 231
reference comparisons, fifteen input diagnostics and a clustering/density
example. The package now has 80 core types in 31 source modules. No public
extension, vendored repository or runtime dependency was added. See
`VALIDATION.md` and `COVERAGE.md` for observed checks and supported subsets.

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

## PowerTransformer addition (2026-11-18)

Added native `power_transformer` with Yeo-Johnson and Box-Cox likelihood fits,
optional population standardization, inverse transforms and NaN preservation.
The type is re-exported through `sklearn_preprocessing` and `sklearn`. Its
log-scaled likelihood avoids overflow in trial powers; the native optimizer
uses bracketing and golden-section minimization. Constant Yeo-Johnson features
use the identity power. Box-Cox requires positive, nonconstant observations.
The package now has 94 core types in 39 source modules; no extensions were added.

All 31 native programs passed checked builds with bounds/backtrace and
invalid/zero/overflow traps, and release builds. The isolated diagnostic suite
passed 167 cases, including 11 new power-transform cases. Native properties
cover standardization, reconstruction, limiting powers, NaN preservation,
refitting and fitted-state retention. `power_transforms` reconstructed its
example to 5.33e-15 maximum absolute error.

The dedicated sklearn 1.6.1 reference suite passed 160 result-group comparisons across
five seeded datasets: signed/positive/skewed inputs, missing values and constants,
with both standardization settings. Learned lambdas and training/query/inverse
transforms use atol=rtol=3e-6 because native golden-section search differs from
SciPy Brent iteration. The eleven-suite inventory is 3,173 comparisons. Existing
unaffected Python suites were not rerun in this batch; all native tests were.

Original contributions remain MIT with sklearn BSD-3-Clause attribution retained
in the source and third-party notices. No additional runtime dependency or
external Fortran repository was introduced. See COVERAGE.md for limits.

## KBinsDiscretizer addition (2026-11-18)

Added native `kbins_discretizer`, re-exported through preprocessing and the
facade. It provides uniform, quantile and weighted 1D k-means strategies;
ordinal and dense one-hot outputs; scalar/per-feature requested counts;
width pruning; fit/transform/fit_transform; and midpoint inversion. Full-data
fits correspond to Python `subsample=None`. Quantile and k-means accept sample
weights. Constant columns use one infinite-ended bin and return NaN on inverse,
as in sklearn 1.6.1. The package now contains 95 core types and 40 source modules.

The new reference suite passed 324 result-group comparisons against sklearn
1.6.1 on three seeded datasets with 12 configurations each. It covers weights
including zeros, heterogeneous bin counts, duplicates, constants and both dense
encodings, at atol=rtol=2e-10. The twelve-suite inventory is 3,497 result groups;
unchanged older Python suites were not rerun for this batch.

Native tests check exact edge assignment, extreme bins, inverse/code consistency,
one-hot block sums, weighted quantiles and known weighted k-means centers,
constant inverse NaNs, duplicate/empty clusters and fitted encoding retention.
All 32 programs passed checked builds with bounds/backtrace and
invalid/zero/overflow traps, plus release builds. All 187 isolated invalid-input
cases passed, including 20 new discretization cases. The `discretization`
example passed. Formatting and Python syntax audits passed.

Sklearn provenance and retained BSD-3-Clause terms are recorded in the module
and third-party notices; original contributions remain MIT. No third-party
runtime dependency or public extension was added. COVERAGE.md records dense-only
encoding, full-data fitting, stricter inverse validation and numerical limits.

## GraphicalLasso addition (2026-11-18)

Added `graphical_lasso` to the covariance module, including empirical and
precomputed covariance input, centering, off-diagonal L1 regularization, and
inherited distances, likelihood and error norms. It uses native Cholesky
initialization plus sklearn's cyclic block/Gram coordinate updates and inner
dual-gap test. Zero alpha directly inverts an SPD covariance. Fitted state
includes covariance, precision, location, objective/gap history, outer iteration
count and convergence status. The package now has 96 core types in 40 modules.

The new reference suite passed 144 result-group checks against sklearn 1.6.1
on three seeded datasets with eight configurations each, using atol=rtol=3e-8.
It checks multiple penalties (including zero and fully diagonal), centering,
precomputed input, rank-deficient sample covariance and a capped two-sweep fit.
Compared outputs are locations, covariance/precision matrices, final objective
and gap, query Mahalanobis distances and query likelihood. The thirteen-suite
inventory is 3,641 result groups; older unchanged Python suites were not rerun.

All 33 native programs passed bounds/backtrace/FPE-trapping builds and release
builds. New native properties cover an analytic 2D solution, covariance/precision
inversion, active/inactive KKT conditions, fixed covariance diagonals, fitted
history, precomputed equivalence, zero-alpha empirical equivalence, iteration
caps and regularized singular sample covariance. All 207 isolated invalid-input
cases passed, including 20 new covariance controls/domain cases. The
`sparse_precision` example passed, as did source formatting and Python syntax
audits.

Original contributions remain MIT; sklearn BSD-3-Clause provenance and notices
are retained. No new runtime dependency or extension was added. Only mode cd is
supported; numerical SPD checks and API restrictions are documented in COVERAGE.md.

## TheilSenRegressor addition (2026-11-18)

Added `theil_sen_regressor` to the linear-model family: exhaustive or capped
random row subsets, minimum-norm SVD least squares, modified-Weiszfeld spatial
median, scalar-median shortcut, optional intercept, prediction and R2 scoring.
Fitted diagnostics include the approximate breakdown, resolved subset/population
sizes, retained one-based subset indices, exhaustive/sampled flag and spatial
median convergence status. Native sampling is repeatable with this compiler
but does not reproduce NumPy draws. The package now has 97 core types in 40
source modules.

The reference suite passed 120 result groups against sklearn 1.6.1 across three
seeded contaminated datasets. Seven configurations use the public exhaustive
estimator and one checks sklearn's least-squares/spatial-median routines on the
same Fortran-selected subsets. Intercept/no-intercept, larger/all-row subsets,
rank-deficient all-row fits, capped iterations, scalar medians and sampled fits
are covered. Coefficients, parameters, predictions and R2 use atol=rtol=2e-7.
The fourteen-suite inventory is 3,761 result groups; unchanged earlier Python
suites were not rerun in this batch.

All 34 native programs passed bounds/backtrace/FPE-trapping builds and release
builds. Native properties cover robust line recovery, improvement over OLS,
combination enumeration/counts, full-subset OLS equivalence, repeatable sampling
without replacement, capped status, scalar median and rank-deficient/wide
minimum-norm fits. All 221 isolated invalid-input cases passed, including 14
new Theil-Sen cases. The `robust_subsets` example passed, along with source
formatting and Python syntax audits. The breakdown test checks sklearn's
finite-sample approximation, which is slightly positive even for all-row subsets.

Original contributions remain MIT; sklearn BSD-3-Clause attribution is retained.
No runtime dependency or public extension was added. COVERAGE.md documents the
SVD rank threshold, random-stream difference and sklearn's zero-based Weiszfeld
iteration reporting convention.

## Passive-aggressive models addition (2026-11-18)

Added `passive_aggressive_classifier` and `passive_aggressive_regressor` to the
linear-model family. Both implement PA-I/PA-II, intercept control, optional
immediate/delayed coefficient averaging, deterministic native shuffling and
training-loss patience stopping. The classifier handles binary and multiclass
one-versus-rest fits with optional balanced weights. The regression type
inherits linear prediction/R2; classification adds decision, prediction and
accuracy methods. The package now has 99 core types in 40 source modules.

The new reference suite passed 336 result groups against sklearn 1.6.1 across
three seeded datasets, using atol=rtol=3e-8. Eight classifier and eight regressor
configurations per fixture cover the two updates, intercepts, weighting,
averaging thresholds, patience, epsilon tubes, binary/multiclass labels and
zero-feature rows. Shuffling is disabled in reference fits to align update
order. Compared outputs include coefficients/intercepts, scores/decisions,
predictions, labels and epoch/sample counters. The fifteen-suite inventory is
4,097 result groups; unchanged older Python suites were not rerun in this batch.

All 35 native programs passed checked builds with bounds/backtrace and
invalid/zero/overflow traps, plus release builds. New properties independently
check analytic PA-I/II steps, intercept norm convention, averaging, zero-row
behavior, patience and fixed epochs, model recovery, binary ties and shuffle
reproducibility. All 241 isolated invalid-input cases passed, including 20 new
control/domain checks. The `passive_aggressive` example passed, as did source
formatting, Python syntax and inventory audits.

The native implementation preserves sklearn 1.6.1's PA-I skipped-zero-row
behavior, online pre-update loss criterion, PA-II loss-name convention and
public epoch counters. Binary decisions are rank two in this API; native
shuffle streams differ from NumPy. Partial-fit, warm starts, sparse input,
validation early stopping and arbitrary class-weight dictionaries are outside
this initial port. COVERAGE.md and API.md record these supported limits.
Original contributions remain MIT; sklearn BSD-3-Clause attribution is retained.
No runtime dependency or public extension was added.


## OPTICS addition

Added dense OPTICS graph construction, Xi hierarchy extraction with optional
predecessor correction, and DBSCAN extraction. The public extraction helpers
reuse fitted reachability graphs. Exposed infinite core/reachability values,
resolved size metadata and consistent 1-based row/hierarchy indexing. Included
an example, native properties and input diagnostics. Updated sklearn BSD
attribution; original contributions remain MIT. No runtime dependency or
public extension was added. Core inventory: 100 types, 41 source modules.


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


All 36 native test programs passed checked builds with bounds/backtrace and
invalid/zero/overflow traps, and optimized release builds. All 267 isolated
input rejection cases passed. The checked `optics_demo` example passed and
printed two clusters under both extraction methods. Source line-length and
whitespace checks, Python syntax checks, inventory counts and staged diff
checks passed. The 432 new reference groups passed; prior suites were not
rerun. Original contributions remain MIT with sklearn BSD notices retained.


## BIRCH incremental clustering

Added native BIRCH to sklearn_cluster: owned CF summary/node storage,
recursive radius merges and farthest-pair splits, ordered leaf collection,
fit/partial_fit, optional Ward grouping, prediction and distance transforms.
Included fit_predict/fit_transform, latest-batch labels, fallback metadata,
example, native properties and diagnostics. Original contributions remain
MIT with sklearn BSD attribution retained. No dependencies or public
extensions were added. Core inventory: 101 types in 41 source modules.


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


All 37 native test programs passed checked builds with bounds/backtrace and
invalid/zero/overflow traps, and optimized release builds. All 282 isolated
input rejection cases passed. The checked `birch_streaming` example passed,
showing two clusters before and after global-only Ward clustering. Python
syntax, Fortran line-length/whitespace, inventory and staged diff checks passed.
The 237 new sklearn comparison groups passed; older reference suites were not
rerun. Original contributions remain MIT with sklearn BSD notices retained.


## QuantileTransformer addition

Added sklearn_quantile_transform, re-exported by sklearn_preprocessing and
the sklearn facade. Implemented percentile/reference fitting, two-direction
duplicate interpolation, uniform/normal mapping, inverse transforms, missing
columns, finite tail clipping and seeded row subsampling. The normal inverse
uses native erfc inversion without external approximation coefficients.
Included native properties, Python comparisons, diagnostics, example and API
limits. Original contributions remain MIT with sklearn BSD attribution
retained. No runtime dependencies or public extensions were added. Core
inventory: 102 types in 42 source modules.


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


All 38 native test programs passed checked builds with bounds/backtrace and
invalid/zero/overflow traps, and optimized release builds. All 296 isolated
input rejection cases passed. The checked `quantile_transform` example passed,
producing expected uniform ranks, finite normal scores and exact recovery of
its input data. Python syntax, Fortran line-length/whitespace, inventory and
staged diff checks passed. The 252 new reference groups passed; prior reference
suites were not rerun. MIT licensing and sklearn BSD notices are retained.


## Chi-squared kernel feature maps

Added additive_chi2_sampler and skewed_chi2_sampler to
sklearn_kernel_approximation and the sklearn facade. Implemented regular
harmonic features with stable log amplitudes, zero handling and stateless
transformation, plus hyperbolic-secant random weights, phase offsets and
stable shifted-log features. Included native accuracy properties, controlled
sklearn sampling comparisons, diagnostics, example and supported-limit docs.
Original contributions remain MIT with sklearn BSD attribution retained.
No runtime dependency or public extension was added. Core inventory:
104 types in 42 source modules.


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


All 39 native test programs passed checked builds with bounds/backtrace and
invalid/zero/overflow traps, and optimized release builds. All 318 isolated
input rejection cases passed. The checked `chi2_kernel_features` example
passed, showing additive feature dimensions and a skewed-kernel Gram matrix.
Python syntax, Fortran formatting, inventory and staged diff checks passed.
The 204 new sklearn comparison groups passed; older reference suites were not
rerun. MIT licensing and sklearn BSD notices are retained.


## GradientBoostingClassifier addition

Added native gradient_boosting_classifier to sklearn_ensemble and the facade:
log-loss binary/multiclass and exponential binary gradients, Newton leaf
updates, class-prior/zero initialization, Friedman threshold conversion,
feature importances, probabilities/log probabilities, staged outputs and leaf
assignments. Included analytic and learning properties, sklearn comparisons,
input diagnostics, a complete example and documented API limits. Existing
estimators were not changed. Original contributions remain MIT with sklearn
BSD attribution retained. No dependency or public extension was added. Core
inventory: 105 types in 42 source modules.


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


All 40 native test programs passed checked builds with bounds/backtrace and
invalid/zero/overflow traps, and optimized release builds. All 342 isolated
input rejection cases passed. The checked `boosted_classification` example
passed with training accuracy 1.0 on its small fixture. Python syntax, Fortran
formatting, inventory and staged diff checks passed. The 510 new sklearn
comparison groups passed; prior reference suites were not rerun. MIT licensing
and sklearn BSD notices are retained.


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
