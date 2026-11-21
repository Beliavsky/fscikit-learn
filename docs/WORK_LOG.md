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


## Self-training classifier batch

Added self_training_classifier, exported by sklearn_semi_supervised and the
sklearn facade. The sklearn 1.6.1 loop supports native logistic_regression,
gaussian_nb and kneighbors_classifier bases, copied without mutating the
input estimator. It implements threshold and k_best selection, strict
threshold comparison, original-label preservation, per-sample iteration
history, transduction, zero/finite/unlimited iteration controls, source
stopping-rule precedence and mandatory final refitting. Inductive class
predictions, probabilities, accuracy and available base log-probabilities
and decision functions use the final fitted estimator. One-class fits are
supported where the base supports them. Other estimator types, sparse data,
metadata/sample-weight routing and verbose output are outside this port.

Confidence ties use stable input order; source argpartition leaves the tied
cutoff choice unspecified. Rounding of saturated base probabilities can also
change tied selections. The reference suite checks all native iterations
against source base fits: correct labels, exact threshold choices, k-best
counts and confidence ordering. Tied cutoff comparisons allow only 32 double
precision epsilons for Gaussian NB/KNN arithmetic; logistic ordering is exact.
Histories that differ from ordinary sklearn require an independently verified
tied cutoff. Final source-base fits on the certified transduction validate
query predictions and probabilities. Direct source histories remain required
for ordinary cases. Multiclass logistic decisions are centered per row to
remove their unidentifiable common intercept; probabilities are compared
directly. The stable tie policy itself has an analytic native test.

The suite has 4,905 groups across 495 fixtures and five seeds
(0,7,42,2026,314159), with one through four classes and all three supported
bases: 3,694 direct sklearn groups, 990 input identities and 221 groups from
30 tied-cutoff cases certified using source fits. Gaussian NB and KNN include
single-class cases; logistic requires at least two initial classes. Uniform
and distance-weighted KNN, both selection rules, zero/unlimited/capped fits,
no-change stopping, fully labeled inputs and query decision/probability
outputs are covered. Native tests additionally verify analytic transduction,
strict threshold equality, all stopping-rule precedence, state replacement,
deep copies, unchanged fitted/unfitted input estimators and mandatory final
refits. Nineteen new isolated diagnostics cover invalid controls, input
sizes/data, missing initial labels, unsupported bases, unfitted prediction,
feature mismatches and unavailable delegated methods.

Source provenance and retained BSD attribution are in THIRD_PARTY_NOTICES.md;
original contributions remain MIT. No extension or dependency is added.
Inventory: 176 core types, 87 source modules, 87 native test programs,
66 reference suites and 141,446 result groups, plus 28 separate NumPy RNG
stream groups. Isolated invalid-input inventory is 1,837.

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


## Cross-validated probability calibration batch

Added calibrated_classifier_cv, exported by sklearn_calibration and sklearn.
It implements sigmoid and isotonic calibration using native logistic,
Gaussian NB or KNN bases. Integer CV uses source first-appearance class
encoding and unshuffled stratified allocation. Ensemble fitting averages
individually normalized fold classifier/map pairs; nonensemble fitting
collects out-of-fold responses and refits the final base on all rows.
Prefit calibration copies an already fitted base without refitting it and
retains its output classes, including missing/unknown calibration labels.
Binary response selection/complement, one-vs-rest multiclass targets,
zero-mass uniform fallback and final probability handling follow source.

Weighted CV is supported for KNN; weighted prefit calibration supports all
three bases. Weighted logistic/Gaussian NB CV fitting is explicitly rejected
until those native base fit APIs support weights. Weights are finite,
nonnegative and bounded by the documented native working range. All-zero
sigmoid weights retain source .5 map outputs; isotonic fitting requires
positive weight support. General estimator/default LinearSVC cloning,
FrozenEstimator, custom CV splitters, sparse arrays, metadata routing,
parallelism and arbitrary fit parameters remain outside this interface.

Sigmoid maps minimize the source Platt objective with weighted Bayesian
smoothing and response scaling at maximum absolute score >=30. A native
damped two-variable Newton solver uses a rank-one pseudoinverse direction
for constant designs and a gradient-decrease certificate when summed loss
roundoff obstructs a near-optimal step. Isotonic maps reuse weighted PAV and
clipped interpolation, explicitly bounding binary-target fits to [0,1] to
contain endpoint roundoff. The shared logistic solver now centers fitted
multiclass intercepts: this removes an arbitrary common softmax offset that
otherwise varies across fold fits and contaminates pooled raw scores.
Its probabilities retain the same model; binary fitting is unchanged.

The new suite checks 11,680 groups across 690 fixtures and five seeds
(0,7,42,2026,314159), with 2/3/4 classes, both map methods, all three bases,
3/5-fold CV, ensemble/nonensemble/prefit paths, uniform/distance KNN,
weighted/zero-weight fits, missing/unknown calibration labels and very large
logistic decision scores. There are 6,086 direct sklearn groups, 3,520
source-calibrator certificates, 1,380 input identities, 690 composition
identities and four machine-precision argmax-tie groups. Each sigmoid map's
independent normalized gradient must be at most 2.1e-9, and its objective
cannot exceed a source optimizer fit by more than 2e-10. Each isotonic map
must match a fresh source fit on native response values to 3e-12 over the
union of both knot sets and query responses, certifying the full piecewise
linear function. Raw base responses and final probabilities compare directly
to source; the latter use 2e-6 tolerances (3e-6 for logistic). Argmax exceptions
require both native and source gaps within 64 machine epsilons and are
recorded individually; all four observed source gaps are exactly zero.

Native analytic tests cover smoothed versus empirical constant-score targets,
weighted priors and zero support, complements, deep copies, source fold
encoding, mandatory full-data refitting, input estimator preservation,
canonical softmax gauge, refit state replacement and uniform multiclass
fallback. Twenty-three isolated diagnostics cover invalid controls, data,
weight domains, unsupported bases/weighted CV paths, prefit/fitted state,
class counts, feature mismatches and zero-weight isotonic fits. A checked
standalone probe passed the new native tests, all 23 diagnostics and the
complete reference suite before full package validation.

Original contributions remain MIT with retained sklearn BSD attribution.
No dependency or public extension is added. Inventory: 177 core types,
88 source modules, 88 native test programs, 67 reference suites and 153,126
result groups, plus 28 separate NumPy RNG groups. Isolated diagnostic
inventory is 1,860.

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


## Weighted and streaming classifier batch

Extended existing Gaussian NB and logistic fit APIs with optional real64
sample weights. Gaussian NB adds partial_fit, optional fixed priors and
n_features_in_. Initial streaming calls declare the full class set; repeated
class declarations normalize order/duplicates and later calls reject changed
classes, unknown targets and feature counts. fit resets accumulated state.
Weighted population moments use Chan/Golub/LeVeque merging. The pinned source's
unweighted current-batch smoothing, current-epsilon subtraction and <=1e-8
weighted-mass moment skip are retained, including count accumulation for
skipped updates. Priors use source length/nonnegative/np.isclose validation
and are retained without normalization. Zero empirical mass and zero
variances preserve source IEEE undefined posteriors; zero priors and unseen
classes are handled explicitly. Gaussian prediction uses joint likelihoods,
including the source's first-NaN argmax behavior. This replaces the former
native zero-variance fit rejection.

Weighted logistic uses mean weighted loss plus 1/(C*sum(weights)) L2 strength,
diagonal equilibration of its Newton system and safe combined-penalty
formation. Tiny C can be supported with large weights even if 1/C alone
would overflow. Binary/multiclass prediction now uses source raw decision
scores, preserving labels when sigmoid/softmax probabilities round to .5
or a uniform vector. Self-training prediction delegates to its base; its
confidence/probability paths require finite base probabilities. Existing
unweighted fit calls remain supported. Inputs/weights are finite dense
real64 arrays; weights are nonnegative and bounded by documented numerical
working ranges. Signed/scalar weights and logistic class_weight are outside
this interface.

CalibratedClassifierCV now forwards training-fold weights to Gaussian NB
and logistic and forwards all weights for the nonensemble full-data refit.
All three supported bases therefore allow weighted CV; KNN retains source
calibration-only weighting. Two obsolete unsupported-weighted-CV diagnostic
cases now check weight-length errors. Forty-two additional diagnostics cover
streaming class/state/feature rules, priors, weights, finite data, moment and
penalty working ranges, zero-mass logistic rejection and undefined base
responses in calibration/self-training. All 42 passed isolated checked probes.

The new reference suite has 16,536 groups across 681 fixtures and five seeds
(0,7,42,2026,314159), including 1,441 Gaussian fitted/streaming/refit states.
Cases span one through four observed classes (plus an additional declared
unseen class in selected streaming fixtures), 1/4 features, uniform/integer/random/
log-scale/tiny/zero weights, missing or declared unseen classes, fixed/zero/
approximately normalized priors, changing batch scales and smoothing,
constant data and tiny-C/large-weight logistic fits. Counts and epsilon use
zero absolute tolerance with 4e-11 relative tolerance; other statistics use
4e-11 absolute/relative, Gaussian posterior outputs 3e-8 and logistic outputs
3e-6. Prediction labels are exact. All 481 logistic fits additionally satisfy
an independent normalized gradient bound of 3e-8.

Oracle accounting: 15,309 ordinary direct sklearn groups, 681 input
identities, 481 independent optimality groups, 60 limiting-intercept groups
from 30 zero-weight-class cases, and five direct sklearn Newton-Cholesky
groups from one certified Newton-CG recovery. Zero-weight classes can have
no finite unregularized intercept optimum; their coefficients, probabilities,
predictions and gradient remain checked, while finite intercept/decision
representations are recorded separately. Conditional softmax common-offset
comparison is available for nonunique intercept gauges; none was needed in
the final checked prototype corpus.

The saved synthetic line-search regression has source Newton-CG normalized
gradient 5.58358e-6 despite requested tolerance 1e-12. Native gradient is
7.37199e-12; source Newton-Cholesky reaches 1.11022e-16. Native and recovered
source losses both equal .6924822701505227, below the stalled source loss
.6924822702129585. Recovery requires an independently measured source stall,
a much smaller native gradient and source-recovered loss agreement; it does
not increase public-output tolerances or remove the difficult case.

The expanded calibration suite has 13,760 groups over 810 fixtures: 7,166
direct source groups, 4,160 source-map certificates, 1,620 input identities,
810 composition identities and four machine-precision argmax exceptions.
It adds weighted ensemble/nonensemble CV for Gaussian NB and logistic.
Native analytic tests cover weighted moments, pooled/streaming equivalence
at zero smoothing, zero-weight batches, source mass-cutoff boundaries,
fixed priors, quiet undefined outputs, integer-weight replication, tiny-mass
intercept fitting, safe C rescaling and raw-score prediction/delegation.
A runnable weighted_streaming_classifiers example demonstrates both models.

Original contributions remain MIT with retained sklearn BSD attribution;
no dependency or public extension is added. Core type count stays 177.
Inventory: 88 source modules, 89 native test programs, 68 reference suites
and 171,742 core result groups, plus 28 separate NumPy RNG groups.
Isolated invalid-input inventory is 1,902.

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


## Independent multioutput estimators ? 2026-11-21

Added MultiOutputRegressor and MultiOutputClassifier counterparts, supporting
independently fitted native base estimators, weighted fitting where the base
supports it, streaming SGD/Gaussian NB, ragged class probabilities, refit and
deep-copy behavior. Pretrained templates remain unchanged; first partial_fit
clones configuration without retaining fitted statistics. Source 1.6.1 output
shrink and partial-only classes_ metadata behavior are retained. API.md lists
supported base families and inherited limits. No public extension or dependency
was added; original work remains MIT with retained sklearn BSD attribution.

Completed validation on the same Windows/fpm/GNU Fortran toolchain:

- All 90 native test programs passed with bounds checks, backtraces and
  invalid/zero/overflow floating-point traps, and under the release profile.
- The new reference suite passed 3,030 groups across 255 fixtures in each
  profile; archived checked/release reports match exactly. Its oracle counts
  are 2,478 ordinary sklearn groups, 510 input/template identity groups,
  29 source constant-R2 identity groups, nine CART tie certificate groups,
  and four Complement NB tie certificate groups. There are 29 constant-R2,
  five CART and two class-tie witnesses. Source stumps and exhaustive node
  SSE scans certify tree choices; source scores certify near-tied labels.
  Ordinary tolerances were not widened and difficult fixtures were retained.
- The original 192-group integration comparison and all 1,934 isolated
  invalid-input cases passed, including 32 new wrapper diagnostics.
- The checked example ran and printed regression outputs [-3,5] and [5,1],
  classification outputs [0,4] and [1,11], and class lists [0,1] and [-7,4,11].
  Both actual prediction output markers were verified.

Python syntax, Fortran formatting, unique registry/driver case correspondence,
report/oracle/certificate accounting, source hashes, inventory counts, retained
licenses and git diff checks passed. Production sources stayed unchanged
through both profile runs. Other reference suites retain their previously
recorded results; they were not all rerun for this batch.

Current inventory: 179 core public types, 89 source modules, 90 native test
programs and 69 core reference suites with 174,772 retained result groups,
plus 28 separately counted NumPy RNG groups.


## RegressorChain and lazy SGD scaling ? 2026-11-21

Added the core RegressorChain counterpart, supporting explicit and seeded
random order, observed-target or out-of-fold chain features, integer KFold
and explicit index partitions, native weighted fitting, restored target
order, fitted state inspection and uniform-average R2. Existing native base
capabilities are preserved. Explicit training-index repeats/overlap and
unused single-output CV behavior match the pinned source; full fits receive
weights while source-style internal CV fits are unweighted. API.md specifies
the Fortran interface and supported base families. Classifier chains remain
unimplemented. Original work remains MIT with retained sklearn BSD notices;
there is no new runtime dependency or public extension.

The shared SGD core now uses source WeightVector64 lazy L2 scaling,
ordered inner products, a 1e-9 scale-reset threshold and final
materialization. A saved synthetic weighted-chain fixture proves the
previous eager implementation fails the original 3e-8 coefficient tolerance
(maximum scaled component error 4.3424e-8), while lazy scaling passes in the
checked probe (7.9580e-9). Under release optimization, tiny out-of-fold
prediction differences are amplified in one downstream coefficient vector
(4.0191e-8 against the unconditioned source chain). Source fits on identical
native feature matrices reproduce those native stages exactly in the
release diagnostic. The suite therefore validates each native CV feature
column against source fold fits, then checks full coefficients/counters
against source fits on those same matrices. Full-chain predictions and
scores still compare against the ordinary sklearn chain. All numerical
comparisons keep their original 3e-8 tolerances.

Completed checks on the existing Windows/fpm/GNU Fortran toolchain:

- All 91 native test programs passed in checked and release builds.
- RegressorChain passed 3,211 groups over 430 fixtures in each profile:
  1,963 ordinary sklearn groups, 310 source-conditioned SGD groups,
  860 input/template identity groups, 75 source CART certificates, two
  CART tie-certificate groups and one saved-fixture identity group.
  Tree certificates validate all full/fold nodes against source stumps
  and exhaustive SSE scans and reconstruct training/query features.
  One duplicated-training-row fixture has a certified alternate equal-gain
  split. Check signatures and CART witnesses match between profiles;
  only release records the single SGD rounding witness described above.
- Both profiles passed the affected 3,030-group multioutput suite,
  216-group SGD regression suite and 306-group SGD classification suite.
  Their archived checked/release reports match exactly.
- The original 192-group integration comparison and all 1,963 isolated
  invalid-input cases passed, including 29 new chain diagnostics.
- The checked regressor_chains example ran and emitted the actual order,
  prediction and R2 markers; it used reversed order with three CV folds.

Full native runs preceded the final reference-driver witness addition;
numerical production sources stayed unchanged afterward. Other reference
suites retain their previously recorded results; they were not all rerun.
Final audits verify syntax, formatting, diagnostic registration/branches,
report/oracle accounting, source hashes, fixture preservation, retained
licenses and git diff checks.

Inventory: 180 core public types, 89 source modules, 91 native test programs,
70 core reference suites and 177,983 retained core result groups, plus
28 separately counted NumPy RNG groups.


## ClassifierChain ? 2026-11-21

Added the core ClassifierChain counterpart with observed-label and
cross-validated chain features; explicit/seeded random ordering; labels,
probabilities, log probabilities and decision scores; response preference
lists; weighted full fitting; class/order metadata; refits/deep copies;
and restored target-column predictions/probabilities/scores. Its eight
native base families retain their own fitting and response capabilities.
Integer CV follows source first-appearance stratification, including small
minority classes and absent-fold-class probability padding. Explicit CV
plans preserve training repeats/overlap. Single-output chains ignore unused
CV specifications. Source-compatible scoring supports shared nonzero binary
indicator labels and finite signed query weights. API.md records the
interface, response-shape restrictions and base working-range limits.

Completed gates on the existing Windows/fpm/GNU Fortran toolchain:

- All 92 native test programs passed in checked and release profiles.
- Both profiles passed 8,665 classifier-chain groups over 785 fixtures and
  five seeds. They cover all supported response choices and fallback,
  log-loss/hinge/modified-Huber SGD, averaging, weighted fitting and signed
  scoring weights, rare/constant/multiclass outputs, nonzero indicator
  labels, single outputs/queries, custom folds, refits and preserved inputs.
- Checked oracles: 3,307 ordinary sklearn groups, 2,265 input/storage identity
  groups, 1,725 conditioned source fit groups, 934 conditioned source CV
  groups, 350 source CART certificate groups, 56 propagated CART alternative
  groups and 28 propagated Complement NB tie groups. Release has 3,305
  ordinary groups and 30 Complement tie groups; other counts are identical.
  Check signatures match across profiles. There are 53 tree witness records
  in each report, plus 26 checked/28 release Complement witness records.
  Every selected CART split is independently optimal under exhaustive Gini
  scans and agrees in cost with a source stump. Distinct equal-gain choices
  are explicitly witnessed. Complement alternatives require source log-score
  gaps within a scaled 64-epsilon bound; the reports retain 39/41 such row
  witnesses. These small profile differences are certified, not suppressed.
  Ordinary tolerances were not widened and difficult fixtures were retained.
- Both profiles passed the existing 3,211-group regressor-chain suite and
  3,030-group multioutput suite. The original 192-group integration comparison
  and all 2,015 isolated input diagnostics passed, including 52 new cases.
- The checked classifier_chains example ran with order [2,1] and emitted
  prediction rows [0,0] and [1,0], with positive probabilities approximately
  [.0483867193,.5] and [.8566033011,.5]. Actual output markers were verified.

Reference-driver state witnesses were strengthened after the checked
native run; numerical production sources stayed unchanged through both
full native profiles and final reference comparisons. The other retained
reference suites were not all rerun for this batch. Python syntax, Fortran
formatting, unique diagnostic registry/driver correspondence, report/oracle
and certificate accounting, source hashes, retained licenses and git diff
checks passed.

Original work remains MIT with retained sklearn BSD attribution. There is
no new runtime dependency or public extension. Inventory: 181 core public
types, 89 source modules, 92 native test programs, 71 core reference suites
and 186,648 retained core result groups, plus 28 separate NumPy RNG groups.


## Multiclass reductions - 2026-11-21

Added OneVsRestClassifier, OneVsOneClassifier and OutputCodeClassifier
counterparts with 19 tested native base configurations, fresh fitted heads,
constant heads, source response thresholds/ties, multilabel OVR, weighted
fitting for supported bases, Gaussian NB/SGD classification streaming,
sorted class pairs and legacy NumPy-compatible seeded output codebooks.
The binary Perceptron coefficient/decision shape now uses one positive-class
head; its label behavior and multiclass layout are preserved. API.md details
rank adaptations, response availability, streaming restrictions and limits.

Completed validation on the existing Windows/fpm/GNU Fortran toolchain:

- All 93 native test programs passed with bounds/runtime checks and floating
  point traps, and in the optimized release profile.
- Both profiles passed 27,490 new groups over 1,930 fixtures and five seeds.
  Cases include all supported base configurations, binary/multiclass/constant
  and indicator targets, nonzero indicators, singleton queries, weighted
  fits/signed scoring, zero-weight Gaussian NB NaNs, streamed unused classes,
  skipped OVO pairs, averaged SGD, constant code columns, refits, deep copies,
  and full template state/input preservation.
- Both profiles also passed 8,665 existing classifier-chain and 3,030
  multioutput groups. The 192-group integration comparison passed.
- All 2,112 isolated invalid-input cases passed, including 97 new diagnostics.
- The checked multiclass_reductions example ran and its actual output markers
  were verified. All three reductions predicted [-7,4,19]; the output-code book had shape
  [3,6]. Actual output is retained in multiclass_example.log.

New-suite checked oracle counts: sklearn=20,753, source_native_head_identity=2,169, input_identity=3,860, source_cart_certificate=390, source_svm_optimality=225, source_binary_cart_certificate=41, source_certified_head_reduction=39, source_native_decoder_boundary=1, source_decoder_boundary=6, source_label_boundary=6.
Release oracle counts: sklearn=20,753, source_native_head_identity=2,169, input_identity=3,860, source_cart_certificate=390, source_svm_optimality=225, source_binary_cart_certificate=41, source_certified_head_reduction=39, source_native_decoder_boundary=1, source_decoder_boundary=6, source_label_boundary=6.
Certificate record counts, checked/release: tree=41, native_decoder_boundary=1, decoder_boundary=3, head_boundary=6 / tree=41, native_decoder_boundary=1, decoder_boundary=3, head_boundary=6.
Ordinary fitted-head comparisons retain their fixed tolerances. Source
wrappers also combine observed native head responses while retaining their
original estimator tags and method capabilities. Tree differences require
exhaustive Gini/SSE-equivalent split optimality and source stump agreement.
SVC fits require dual feasibility, KKT and objective checks. Label/code
boundary alternatives require recorded scaled 64-epsilon witnesses; no
fixture was removed or ordinary tolerance widened.

Reference-driver template-state witnesses were strengthened after the
checked native run; numerical production sources were unchanged through
both native profiles and final comparisons. Other retained reference suites
were not all rerun. Final audits check Python syntax, Fortran formatting,
unique diagnostic registration/branches, report/oracle/certificate accounting,
installed source hashes, retained licenses and git diff consistency.

Original work remains MIT with retained sklearn BSD notices. No runtime
dependency or public extension was added. Inventory: 184 core public types,
90 source modules, 93 native programs, 72 core reference suites and 214,138
retained core groups, plus 28 separately counted NumPy RNG groups.


## Voting ensembles - 2026-11-21

Added the core VotingClassifier and VotingRegressor counterparts, with named
heterogeneous native bases, dropped slots, fresh fitted clones, class-label
encoding/restoration, hard/soft voting, signed averaging weights, per-base
transforms and unflattened probability tensors. Metadata includes fitted
names, one-based active input indices, original classifier labels and feature
count. Fitted templates and inputs are preserved; refits replace the fitted
ensemble, and Fortran assignment deep-copies models. Base capabilities and
weighting limits remain explicit native dispatch. No numerical production
code in existing estimators was changed; sklearn_ensemble re-exports the new
module. API.md documents controls, rank adaptations and supported bases.

Completed gates on the existing Windows/fpm/GNU Fortran toolchain:

- All 94 native test programs passed in checked/runtime-trap and optimized
  release profiles.
- Both profiles passed 16,965 new reference groups across 1,200 fixtures and
  five seeds, covering 15 classifier and 10 regressor configurations.
- Cases include single/mixed lists, dropped unallocated slots, noncontiguous
  and constant labels/targets, signed voting and scoring weights, all-zero
  hard-voting weights, weighted base fits, zero-weight Gaussian NB NaNs,
  singleton queries, source transform layouts, refits, deep copies and
  complete emitted template-state/input preservation.
- The existing 192-group integration comparison passed. All 2,150 isolated
  invalid-input cases passed, including 38 new diagnostics. The checked
  voting_ensembles example ran with labels [-7,-7,19]; its actual output
  markers were verified.

Checked oracle counts: sklearn=10,482, input_identity=4,900, source_native_base_identity=1,370, source_cart_certificate=180, source_template_cart_certificate=3, source_certified_base=18, source_certified_reduction=10, source_template_label_boundary=2.
Release oracle counts: sklearn=10,482, input_identity=4,900, source_native_base_identity=1,370, source_cart_certificate=180, source_template_cart_certificate=3, source_certified_base=18, source_certified_reduction=10, source_template_label_boundary=2.
Certificate record counts, checked/release: template_tree=3, tree=16, template_label_boundary=2, label_boundary=2 / template_tree=3, tree=16, template_label_boundary=2, label_boundary=2.

Ordinary base responses and supported learned parameter states retain fixed
source tolerances. Source ensemble copies independently combine actual
native base responses, checking transformations, final predictions and
weighted scores. CART alternatives require exhaustive Gini/SSE-optimal split
certificates and source stump cost agreement. All equal-cost alternative
partitions are counted, including when the source stump and full source tree
choose different optimal partitions. Label boundary differences require
recorded scaled 64-epsilon witnesses. No fixture was removed and no ordinary
tolerance widened.

Production numerics were unchanged after the checked native run. Reference
certificates were strengthened afterward. Other retained reference suites
were not all rerun in this batch. Final audits verify Python syntax, Fortran
formatting, diagnostic registry/driver correspondence, report/oracle and
certificate accounting, source hashes, retained licenses and git diff checks.

Original work remains MIT with retained sklearn BSD notices; no runtime
dependency or public extension was added. Inventory: 186 core types,
91 source modules, 94 native programs, 73 core reference suites and 231,103
retained core result groups, plus 28 separately counted NumPy RNG groups.


## Permutation importance - 2026-11-21

Added dense scalar and multiple-score permutation importance. Native Fortran
callbacks capture any fitted estimator and arbitrary targets, receiving
features, original one-based row identities and optional subset scoring
weights. The implementation follows source's full-data baseline, shared
worker seed, per-feature reset, all three subsampling branches and cumulative
column/index shuffles. Results expose raw importances and population means
and deviations. Integer and real32/real64 sample limits are distinguished;
a valid fraction rounded to zero forwards empty arrays to the scorer, as in
source. Quiet IEEE arithmetic handles NaN/infinite scores and overflow.
API.md describes callback contracts and the absence of Python scorer dispatch.

Completed gates on the existing Windows/fpm/GNU Fortran toolchain:

- All 95 native programs passed checked/runtime-trap and optimized release
  profiles, including deterministic repeats, one-repeat deviation, multiple
  metrics, missing features, empty subsamples and real32/int64 sample limits.
- Both profiles passed 17,820 source reference groups over 600 fixtures and
  five seeds. A multiple-output analytic scorer and fitted OLS, Ridge,
  logistic, Gaussian NB and KNN models exercise scalar and dictionary-of-score
  source paths versus their native callback equivalents.
- Every scorer call is checked for EXACT feature values, original row
  identities and scoring weights, in addition to numerical scores. This
  directly proves baseline/subset alignment, RNG consumption, sampling,
  cumulative permutations, callback count and shared multiple-score work.
- Cases include integer/fraction limits, full/single/empty subsets, single
  rows/features/repeats, constant columns, finite signed scoring weights,
  NaN features, NaN/infinite scores, and overflowing score differences and
  reductions. Raw importances and population summaries match source;
  emitted model state, responses and input arrays are preserved.
- The existing 192-group integration comparison passed. All 2,171 isolated
  invalid-input cases passed, including 21 new diagnostics. The checked
  permutation_importance example ran and verified that the informative
  feature has positive importance while the unused feature has zero importance.

Checked oracle counts: sklearn=16,230, input_identity=1,590.
Release oracle counts: sklearn=16,230, input_identity=1,590.
No source fixture was removed and ordinary tolerances were not widened.
Numeric production sources were frozen before the final full checked/native
and reference runs. Other retained source suites were not all rerun. Final
audits check Python syntax, Fortran formatting, diagnostic registry/driver
correspondence, report/oracle accounting, installed source hashes, retained
licenses and git diff consistency.

Original contributions remain MIT with retained sklearn BSD and NumPy
notices. No runtime dependency or public extension was added. There are
still 186 core estimator/transformer/splitter types; the importance result is
a container, not an estimator. Inventory: 92 source modules, 95 native
programs, 74 core reference suites and 248,923 retained core groups, plus
28 separately counted NumPy RNG groups.


## Brute partial dependence and ICE - 2026-11-21

Added callback-based brute partial dependence with averaged responses,
individual conditional expectation curves or both. Numeric/categorical grids,
Cunnane percentile endpoints (SciPy alphap=betap=.4), source unique-count
cutoffs, Cartesian feature order, repeated-feature substitutions and signed
weighted means follow the documented numerical contracts. Fixed-rank result
arrays flatten the grid dimensions; ragged axis values preserve their sizes
and ordering. Callbacks choose the resolved regression, probability or decision
response and preserve their output dimension. API.md describes this native
interface and its limits; tree-recursion estimation is not implemented.

Completed validation on the existing Windows/fpm/GNU Fortran toolchain:

- All 96 native programs passed checked/runtime-trap and optimized release
  profiles. Unit probes additionally cover weighted ICE invariance, copied
  result storage, categorical masks, duplicate features and NaN quantile grids.
- Both profiles passed 13,395 new source groups across 835 fixtures and five
  seeds, using an analytic multiple-output model, OLS, Ridge, logistic
  probabilities/decisions, Gaussian NB, KNN probabilities and LinearSVC.
- Every substituted feature matrix and predictor response is checked; untouched
  background columns are exact input identities. Axes, Cunnane interpolation,
  strict unique/resolution boundary, categorical logical/default-int64 selectors,
  unused categorical indices, duplicate/reordered/three-way axes, all result
  kinds, singleton/two-row backgrounds and signed/negative-total weights are
  covered. NaN axes and nonfinite prediction reductions retain quiet source
  semantics. Emitted fitted responses/parameters and inputs are preserved.
- Both profiles passed the existing 17,820-group permutation-importance suite.
  The original 192 integration comparisons and all 2,202 isolated invalid-input
  diagnostics passed, including 31 new cases. The checked partial_dependence
  example's grid and averaged response were compared with source Ridge.

Checked new-suite oracle counts: sklearn=10,830, input_identity=2,505, source_binary_shape_identity=60.
Release oracle counts: sklearn=10,830, input_identity=2,505, source_binary_shape_identity=60.

Each report retains 30 observed sklearn 1.6.1 binary-response reshape failures,
restricted to two grid points or two background rows and the expected cannot-
reshape diagnostic, with a complete prediction trace. Native callbacks expose
the response axis explicitly and return the complete mathematical curve.
The 60 affected average/individual groups are independently compared with
weighted means and individual responses from those actual source predictions,
and labeled source_binary_shape_identity. Ordinary source estimator tags and
response-method names are preserved by tracing; no source fixture was removed
and no ordinary comparison tolerance was widened.

Numerical production sources were frozen before the full checked/release
native and final reference runs. Other retained source suites were not all
rerun. Final audits verify syntax, formatting, diagnostic registration and
branches, source/error/oracle report accounting, source hashes, retained
licenses, exact example responses and git diff consistency.

Original work remains MIT. Retained sklearn BSD notices now also include the
SciPy source BSD notice for the percentile adaptation; no SciPy binary or
runtime is linked. No runtime dependency or public extension was added. The
result/axis helpers are containers, so core estimator/transformer/splitter
count remains 186. Inventory: 93 source modules, 96 native programs, 75 core
reference suites and 262,318 retained core groups, plus 28 separate NumPy RNG
groups.


## Weighted CART and standalone ExtraTrees (2026-11-21)

Added `extra_tree_classifier` and `extra_tree_regressor`, bringing core type
coverage to 188. Both standalone ExtraTree families use source float32 split
geometry and the legacy NumPy/xorshift random feature/threshold protocol.
DecisionTree fitting now accepts sample weights through the same weighted CART
builder. Node cardinalities exclude zero weights; separate learned weight mass,
weighted criteria, importance and scores follow source contracts. Finite signed
weights and degenerate zero mass retain quiet source NaN/Inf responses. The
older omitted-weight CART path preserves its existing real64/native RNG
conventions and now accepts zero-gain splits needed for XOR.

Voting, independent multioutput, classifier/regressor chains and multiclass
reductions accept ExtraTree descendants and forward tree training weights.
Nonfinite reductions use internal quiet-IEEE arithmetic; regression scoring
rejects undefined predictions. IsolationForest's custom tree builder now fills
the added node mass metadata without changing its splitting/scoring algorithm.
No Bagging estimator is included in this batch. Weighted single trees establish
its required foundation.

The new weighted/ExtraTree source suite passed 7,124 groups over 552 fixtures:
6,972 ordinary sklearn comparisons, 140 source pure-subtree identities, and
12 groups backed by five source-conditioned tree certificates. Ten fixtures
contain response-identical pure-node roundoff splits. The certificates replay
source random feature/threshold draws, use sklearn stumps to verify admissible
split costs and node responses, reconstruct query responses from those source
leaves, and independently check impurity importance. No fixture was dropped
and no ordinary tolerance was widened. Reports retain every group/oracle and
installed source SHA-256 hashes. These checks document floating-point tie
conventions rather than claiming universal bitwise topology parity.

All 97 native test programs passed with bounds/runtime/FPE checks. A final
focused checked/release weighted-tree test additionally verified nonuniform
weighted accuracy and R2 by hand, and checked weight forwarding through
composite estimators' learned root mass/probability. The checked public example
matched sklearn ExtraTree predictions and root cardinality/mass exactly.

Inventory: 94 source modules, 97 native programs, 76 core reference suites,
269,442 retained core groups plus 28 separate NumPy RNG groups. Original work
remains MIT; sklearn BSD source attribution is retained and expanded. No new
runtime dependency, imported repository or public extension was added.


Final batch gates: all 97 native programs also passed the full optimized
release invocation. The final weighted-tree unit passed separately against
both checked and release libraries after adding hand-computed weighted score
checks. Both source-suite profiles retained the same 552 fixtures, 7,124 groups,
oracle labels and source hashes. The existing checked reference suites passed:
voting 16,965; multiclass 27,490; independent multioutput 3,030; classifier
chains 8,665; regressor chains 3,211; IsolationForest 9,060; gradient boosting
classification 510; original integration 192. All 2,217 isolated invalid-input
cases passed, including 15 new weighted-tree cases. The invalid driver was
compiled directly against the fully rebuilt checked modules/archive with
`-fcheck=all -fbacktrace`, matching the diagnostic harness flags. Source suites
used those rebuilt checked executables; the release weighted reference driver
was compiled directly against the rebuilt release archive. Retained suites
not listed here were not rerun. Python syntax, Fortran formatting, report
accounting, retained license files and git whitespace checks passed.


## Bagging ensembles (2026-11-21)

Added `bagging_classifier` and `bagging_regressor`, exported through
sklearn_bagging, sklearn_ensemble and the main facade. Native bases are cloned;
weighted-capable bases receive bootstrap multiplicities or a subsampling mask,
while other native bases receive sampled rows. Source-compatible legacy NumPy
seed derivation and sampling support row/feature counts or fractions, repeated
features, nonreplacement tracking/permutation/reservoir selection, warm starts,
class probability alignment, hard-vote fallback, weighted scores and OOB
responses/scores. Actual one-based member draws and seeds remain inspectable.
No new dependency, imported repository or public extension was introduced.
Original work remains MIT with retained sklearn/NumPy upstream notices.

Existing voting fit/prediction dispatch and IsolationForest index sampling were
extracted unchanged into internal sklearn_base_dispatch and
sklearn_bagging_sampling modules. Dispatch adds capability/class metadata and
base seed helpers for bagging. The native base capability boundary is explicit;
standalone estimator signatures remain unchanged.

Source comparisons exposed a real weighted OLS issue: indexing duplicated
bootstrap rows can change the numerical rank cutoff relative to source's
weighted full training table. The exact failing fixture is retained in
validation/fixtures/bagging_ols_rank_deficient.json. Bagging now uses a native
weighted SVD adapter for exact OLS/Ridge bases, preserving the original row
count, weighted centering/rescaling and OLS source rank convention. Predictions,
coefficients, intercept and OLS rank are checked against source. Explicit
nonuniform weights and no-intercept/zero-weight identities are also covered.
Ordinary comparison tolerances were not widened and the failing fixture was
not removed.

The checked source suite passed 32,100 groups across 527 successful fixtures
plus one separately verified source OOB shape-error fixture. Oracle counts:
29,662 direct sklearn comparisons, 2,108 input identities, 183 groups backed
by 61 source-conditioned tree certificates, 146 conditioned member reductions,
and one source error. Certificates use the retained weighted-tree stump/RNG
oracle to verify actual member responses and importance before aggregating.
All 56 warm-start sample-property discrepancies are recorded: sklearn 1.6.1
exposes only the newest batch through estimators_samples_, while native records
retain every fitted member's actual draws. Source OOB single-class probability
broadcasting and incompatible-width failure are verified explicitly. Reports
retain source hashes, per-group oracle labels and split witnesses.

An initial incremental native build linked a cached forest test object from
03:50 against tree layout changes compiled at 05:45 and crashed. Freshly
compiling that unchanged test passed repeatedly. The previous build cache was
preserved under ignored validation/results/pre_bagging_build_20261121, and a
full clean checked build passed all 98 programs. After the weighted linear
adapter change, all 98 checked programs passed again; the final focused unit
also passed the added no-intercept and zero-weight OLS identities. The fresh
checked voting (16,965), IsolationForest (9,060) and original integration (192)
source suites passed. Their existing algorithms and ordinary oracles remain
unchanged.

Current inventory: 190 core estimator/transformer/splitter types, 97 source
modules, 98 native programs, 77 core reference suites and 301,542 retained core
groups plus 28 separate NumPy RNG groups. Public limits are documented in
API.md and COVERAGE.md; this is a dense practical subset, not a complete Python
estimator-protocol port.


The public Bagging example's exact Fortran-generated arrays are retained as
additional fixtures. Classifier predictions/scores match sklearn directly.
Three fully grown ExtraTree regression members choose different equal-cost
splits; their full random protocols, fitted leaves, predictions, importance
and OOB reductions pass the source-conditioned certificate. These are recorded
as members 16, 18 and 20 (zero-based) of example fixture 101. The checked
example scores are classification 1.0 / OOB 0.975, regression R2
0.99839820861863493 / OOB R2 0.98786969877112196. These small-fixture figures
are validation observations, not general performance guarantees.


Final Bagging gates: all 98 native programs passed both checked/runtime/FPE
and freshly built optimized release profiles. Both source profiles passed the
same 32,100 groups over 527 successful fixtures plus the expected OOB error,
with identical group/oracle records, primary source hashes and 56 warm-start
property observations. The final focused checked/release unit passed the
weighted OLS adapter, no-intercept/zero-weight identities, stochastic
base seed derivation and quiet zero-weight tree reductions. All 2,260 isolated
invalid-input diagnostics passed, including 43 new Bagging cases. The invalid
driver was compiled against the fresh checked modules/archive with the
canonical `-fcheck=all -fbacktrace` harness flags.

The fpm-selected release reference executable additionally passed 591 groups
on the retained OLS regression and exact public example arrays, including its
expected OOB error. `fpm run --profile release --example bagging_ensembles`
completed with process exit code zero and matched the recorded example scores.
The full source suites used drivers linked against the freshly built profile
archives; other retained source suites beyond the listed voting, isolation and
original integration checks were not rerun. Final audits verify extracted
code identity, Python syntax, Fortran line formatting, diagnostic registration,
report parity/accounting, retained license files and git whitespace. Numerical
production sources were unchanged during these final profile/source gates.


## Stacking ensembles and LOO RidgeCV (2026-11-21)

Added stacking_classifier and stacking_regressor with cloned named native
bases, dropped slots, source-order response selection, binary probability
column removal, passthrough, independent full/fold fits, custom partitioned CV
and prefit modes. Source first-appearance stratification handles minority
counts without the older native splitter's stricter per-class constraint.
Native introspection retains fitted fold models and actual OOF designs, making
training leakage and final-layer fitting independently inspectable. Default
classification uses strict native Newton logistic fitting; default regression
uses genuine source LOO/GCV RidgeCV rather than substituting K-fold selection.
The existing ridge_cv type adds cv=0 while keeping its historical cv=5 default.

Weighted linear fitting and composite scores were moved into internal shared
modules. RidgeCV LOO supports weighted ensemble dispatch, selected alphas,
candidate scores, wide designs and singleton/no-intercept behavior. The new
shared decision-response/class metadata dispatch supports stacking auto mode.
No new dependency, imported repository or public extension was added. Original
work remains MIT and sklearn/NumPy attribution is retained.

A weighted multinomial stacking fixture exposed false convergence in the
existing Newton line search: an objective decrease below floating-point
resolution triggered backtracking and then the small-step stopping test while
the unnormalized gradient was still 1.13e-8 at a requested 1e-10 tolerance.
The exact emitted meta design is retained in
validation/fixtures/stacking_logistic_roundoff.json. Armijo acceptance now
includes a bounded roundoff allowance, matching the existing native linear-SVM
solver's approach. The retained design now satisfies an independent gradient
check and source coefficient/probability comparisons. This improves the shared
logistic implementation, including existing composite estimators.

The checked stacking/LOO suite passed 3,569 groups across 193 fixtures: 128
stacking fits, 64 standalone LOO cases and the retained logistic regression.
The suite compares split order, response widths/methods, emitted OOF and full
transforms, final fitting/selection, predictions, probabilities, decisions,
weighted scores and input preservation. Source tree alternatives require the
existing CART/RNG certificate. LinearSVC alternatives require independent
squared-hinge stationarity/objective and strong-convexity score bounds. A
source Newton stopping difference is bounded using unprojected logistic
stationarity/objective checks and segment-Hessian curvature in the zero-mean
multinomial gauge; source optimizer warnings are retained. No ordinary
comparison tolerance was widened, and the failing meta design was preserved.

An initial GNU 17 unlimited-polymorphic allocation through a nested array
SOURCE expression crashed despite valid dimensions. Copying the original
estimator slot to a scalar prototype before fold cloning avoids that compiler
behavior and preserves the intended cloned configuration. Checked native
stacking tests cover copy/refit behavior, OOF versus full features, weighted
fits, prefit original labels, missing fold classes and custom split ordering.
All 99 native programs passed the checked/runtime/FPE profile after the solver
fix. All 2,304 isolated diagnostic cases passed, including 44 new stacking/LOO
cases. Stacking diagnostic keys use stk_ to avoid the existing st_ self-training
namespace; AST uniqueness and driver argument-length checks are enforced.

Current inventory: 192 core types, 100 source modules, 99 native programs and
78 core reference suites. This remains a dense practical subset with explicit
base capability and numerical conventions, documented in API.md/COVERAGE.md.


Both checked and optimized source runs passed identical 3,569 group/oracle
records over 193 fixtures. Counts: 2,721 direct sklearn groups, 294 final fits
on verified native meta designs, 512 input identities, 39 conditioned stacking
reductions, two logistic stationarity parameter groups and one independent
gradient regression check. Reports retain one CART alternative, 14 LinearSVC
and one logistic optimizer certificates, and two source line-search warnings.
Ordinary tolerance settings and retained failure fixtures are unchanged.
Retained core reference inventory is now 305,111 groups across 78 suites,
plus 28 separate NumPy RNG groups.

Final stacking batch gates: all 99 native test programs passed in checked and
optimized builds. The fpm stacking example exited successfully (accuracy 1,
regression R2 0.99999995908987271). All 2,304 invalid-input records passed.
Selected existing checked source suites passed: Bagging 32,100 groups,
weighted/streaming classifiers 16,536, multiclass 27,490, classifier chains
8,665, calibration 13,760, self-training 4,905, basic estimators 192, and voting
16,965. Optimized Bagging also passed all 32,100 groups. These are selected
regressions; the retained 78-suite inventory was not rerun in its entirety.

The initial cached checked voting reference executable stalled. Its executable,
object and input were preserved under ignored validation/results/stacking_probe;
a fresh compile against the current checked modules/archive passed the identical
input and the complete 1,200-fixture voting suite. The stale process was stopped.
The underlying cached-artifact cause was not established. Production sources
were unchanged during these final gates. Python syntax, Fortran line lengths
and whitespace, report parity, and git diff checks passed. MIT licensing and
retained third-party BSD attribution are unchanged.
