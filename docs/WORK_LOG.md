# Translation work log

## Pinball and D2 quantile implementation (2026-11-24)

Completed the real64 pinball/D2 quantile/absolute interfaces with all vector/
matrix rank pairs, modern multioutput keyword forms, legacy alpha/output_weights
positions and optional-array semantics, and owned raw helpers. Loss arithmetic
retains the source sign-product expression. D2 uses linear percentiles or
source-order weighted CDF selection with the existing Intel NumPy index helper,
then the exact zero/nonzero score masks. The sample-weight and singleton checks
follow the source's ordering. Scalar legacy raw outputs retain response width
for one sample; explicit raw helpers adapt Python's scalar NaN as [NaN].
The old regression implementation is retired in favor of a small public
compatibility facade over logarithmic, Tweedie and quantile kernels. Existing
imports remain. Original integration is MIT with retained BSD notices; no
dependency or ML extension is introduced.

The first checked reference probe passed 6,734 configurations / 20,034 retained
groups / 534 controlled rejections, with zero finite differences and 1,538
matching nonfinite masks. All 77 targeted guards passed. An independently
regenerated source inventory matches these counts. The full checked fpm batch
passed, including 33,120 earlier regression groups, 59 native programs, 192 core
groups, 77 new/25 earlier diagnostics, the accepted infinite-weight regression
and retained score-shape guard. The audit passed all 243 compiled units and
frozen hashes. Optimized verification subsequently passed; final results are in VALIDATION.md.

## General weighted Tweedie deviance/D2 implementation (2026-11-24)

Extended the general Tweedie family to all vector/one-column rank pairs and
source-compatible signed/nonfinite sample weights, preserving existing names,
keywords and module imports. The power-specific kernels now use source-order
arithmetic under temporary IEEE masks and NumPy buffered sums. D2 keeps the
source's one-sample return before weight-length/domain checks, scalar-mean null
model without revalidation, zero-null-deviance rejection and NaN propagation.
Poisson xlogy(0,NaN) is preserved for zero null predictions. Other regression
metric bodies and the main metrics facade are unchanged. Original integration
remains MIT with retained sklearn/NumPy BSD notices; no dependency or ML
extension is introduced.

The preliminary checked probe passed 7,734 configurations / 23,013 retained
value/identity/IEEE groups / 63 controlled rejections, with zero finite
differences and all 4,186 nonfinite masks matching. The incoming inexact flag
alternates clear/set to check both restoration paths. All 41 targeted guards
passed. The independently regenerated source inventory matches these counts.
Full fpm checked/optimized batches passed, including 33,120 earlier regression
groups, 58 native programs, 192 core groups, 41 new and 25 earlier diagnostics,
plus the retained score-shape guard. The old infinity-weight rejection migrated
to an accepted NaN regression checked against sklearn; its obsolete invalid
classification is removed from the inventory. The audit passed all 241 compiled
units per profile and frozen hashes. Final results are recorded separately.

## Weighted logarithmic and direct deviance implementation (2026-11-24)

Extended MSLE/RMSLE with sample weights, all vector/matrix rank pairs, output
aggregation and owned raw helpers. MSLE now permits valid (-1,0) targets and
uses compiler-runtime log1p to preserve tiny differences. RMSLE retains its
predicted/output_weights keywords, legacy positional layout and old module
import. Separate legacy entries keep optional/unallocated weight arguments
valid Fortran while modern multioutput forms retain unambiguous typed dispatch.
Both log metrics follow sklearn's finite, multiple-output custom-weight rules.
Poisson/Gamma direct deviance gains sample weights and all vector/column pairs,
source-domain guards, source-order kernels and quiet exceptional arithmetic.
Other regression metric bodies remain unchanged. Original integration remains
MIT with retained sklearn/NumPy BSD attribution and no new dependency/ML extension.

The preliminary checked reference probe passed 3,669 configurations / 11,224
value/identity/IEEE groups / 45 controlled rejections with zero finite score
differences and 1,741 matching nonfinite masks. All 79 targeted diagnostics
passed. The earlier regression-metric suite independently passed 33,120 groups
over 4,507 fixtures, with 26 diagnostics and 929 source-rejection records, using
the explicit legacy adapters. Native tests also check unallocated positional
and named legacy weights. The full checked fpm build/batch passed: 11,224 new
groups, 33,120 earlier regression groups, 57 native programs, 192 core groups,
79 new diagnostics, 26 earlier diagnostics and the retained score-shape guard.
The checked audit passed all 239 compiled-unit fingerprints and frozen hashes.
Optimized verification subsequently passed; final results are in VALIDATION.md.

An attempted documented fpm --list target-build shortcut returned names without
creating the new executables. The artifact check caught this; actual verification
uses the checksum-verified private fpm 0.14.0 build --tests -j 4 workflow.

## Weighted median and column maximum error implementation (2026-11-24)

Extended median absolute error with finite sample weights, matrix/vector rank
pairs, output weighting and owned raw results. Signed/zero/zero-total sample
weights follow sklearn's lower-percentile CDF selection. Sorting reuses the
unchanged scalar Intel index-order helper; cumulative weights and NaN-aware
binary search reproduce the pinned sklearn/NumPy semantics. The old doubled
middle term could spuriously overflow for huge odd-length medians; odd lengths
now select one observation while even midpoint overflow retains source behavior.
max_error accepts all vector/column rank pairs, rejects multioutput and quietly
propagates residual overflow. Original integration remains MIT with retained
sklearn/NumPy/Intel BSD attribution, without a new dependency or ML extension.

The first checked probe passed 5,632 configurations and 17,963 retained groups,
with 15 controlled source rejections, zero finite differences and exact matching
of all 325 nonfinite masks. All 28 targeted diagnostics passed. The expanded
source inventory adds direct signed infinite-output aggregation cases: 5,648
configurations, 18,011 groups, 15 rejections and 341 nonfinite masks. The checked
fpm build and full affected batch passed, including 56 native programs, 192 core
groups, 28 targeted diagnostics and the retained score-shape guard. The audit
passed all 236 compiled-unit fingerprints and frozen artifact hashes.
Optimized verification subsequently passed; final results are in VALIDATION.md.

## Weighted multioutput regression error implementation (2026-11-24)

Extended MSE, RMSE, MAE and MAPE with sample weights, matrix/vector rank pairs,
uniform/custom output aggregation and owned raw-score helpers. Existing vector
calls retain their names and result rank. RMSE roots are taken per output before
aggregation, and its singleton/nonfinite custom-weight rules follow sklearn's
wrapper rather than the stricter MSE/MAE/MAPE target checker. Signed weights,
nonfinite sample weights, MAPE's epsilon denominator and quiet exceptional
arithmetic match sklearn 1.6.1. Original integration remains MIT with retained
sklearn/NumPy BSD attribution; no dependency or extension algorithm is added.

The preliminary checked probe passed 4,688 configurations, 14,309 retained
value/identity/IEEE groups and 273 controlled source rejections. All finite
score differences were zero and all 2,030 nonfinite source masks matched.
All 77 targeted diagnostics passed, including the four static raw-result
adapter guards. An independently regenerated source inventory matches these
counts. Successful cases use a batched native driver; expected errors remain
isolated. The checked fpm --tests build and full affected batch passed, including
54 native programs, 192 core groups, 77 targeted diagnostics and the retained
score-shape guard. The audit passed all 233 compiled-unit fingerprints and
frozen artifact hashes. Optimized verification subsequently passed; final results
are recorded in VALIDATION.md.

## Weighted multioutput R2 and explained variance (2026-11-24)

Added sample weighting, uniform/variance/custom output aggregation and
`force_finite` to R2 and explained variance, with explicit vector/matrix adapters
and owned raw-output helpers. Existing vector score calls remain supported.
The implementation follows sklearn 1.6.1, including singleton R2 NaN, signed
and nonfinite sample weights, constant targets and IEEE exception preservation.
NumPy 2.2.6 C-order matrix and buffered vector reductions are identified in the
API. No dependency or extension algorithm was added; original integration
remains MIT with retained sklearn/NumPy BSD attribution.

At the implementation checkpoint the checked fpm build with tests passed,
as did 53 affected native programs, two examples, all 192 core reference groups,
20 targeted diagnostics and the retained EllipticEnvelope score-shape guard.
An initial 800-configuration metric probe passed (1,608 groups and 264
controlled source rejections). Full metric references and the optimized build
subsequently passed; final results are recorded in VALIDATION.md.

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


## Numeric TargetEncoder

Added sklearn.preprocessing.TargetEncoder as a native dense transformer with
real/integer vector targets, automatic type inference, binary/multiclass target
coding, fixed or empirical Bayes smoothing, NaN categories, unseen-category
fallback, optional explicit numeric category lists, and seeded KFold or
StratifiedKFold cross-fitting. Full-data learned encodings remain available
after fit_transform, while its output uses held-out training folds. The new
module depends only on existing common, resampling container and legacy RNG
modules; SciFort is not required. Public target_category records hold ragged
category/encoding vectors and do not add estimator or extension counts.

Source inspection and local probes identified two explicit-category edge cases
in sklearn 1.6.1: omitted training categories affect slot zero's statistics,
and unknown cross-fit rows are overwritten with the last fold's target mean.
Both are preserved and documented, with native unit assertions and source
fixtures. Additional probes confirmed that large integral-looking float
values outside int64 range infer continuous targets; explicit multiclass
fractional labels and explicit two-class multiclass are rejected. Fixed
smoothing uses a convex mixture rather than multiplying smooth by the target
mean; a documented target working-range bound protects variance arithmetic.

The reference harness compares each learned array, transform, cross-fit output,
fold assignment, parameter snapshot and refit result independently. It retains
source hashes and warnings with exact categorical/index checks and numerical
tolerances of rtol 2e-12 and atol 2e-13. Five deterministic seed recipes include
singleton/constant categories, rare classes, NaNs, unused/omitted categories,
large smoothing, cv=n, forced target modes and one-row fit-only inputs.
The checked native TargetEncoder program passed during development; final
source and full-build gates are recorded below when complete.


Checked source validation passed 12,170 result groups across 950 fixtures and
five seeds. All group oracles are direct sklearn comparisons, with no widened
numeric tolerances or alternative-solution certificates. The report retains
135 source warnings: 120 rare-class warnings (the independently constructed
splitter repeats the same warnings) and 15 NumPy int64 cast warnings for the
large floating-target inference fixtures. All 100 native test programs also
passed the final checked/runtime/FPE profile with the frozen implementation.
Current inventory: 193 core estimator/transformer/splitter types, 101 source
modules, 100 native test programs, and 79 retained core reference suites with
317,281 groups, plus the separate 28-group NumPy RNG suite.


The optimized source run passed the same 12,170 groups over 950 fixtures;
checked and optimized report records, source warnings and source hashes are
identical. All 100 native test programs passed in both checked and optimized
builds. The public fpm example exited successfully and its cross-fitted
training features, query encodings and unseen-category mean were independently
compared to Python TargetEncoder with the same recipe and tolerances.


Final diagnostic gate passed all 2,331 unique isolated invalid-input cases,
including 27 new TargetEncoder cases. Python syntax, Fortran 132-column and
whitespace checks, source hashes, checked/release report parity, facade exports,
and MIT/retained BSD attribution were verified. Production code remained
frozen throughout the final gates. Existing numerical implementations were
unchanged; the 79-suite reference inventory includes retained earlier results,
rather than a complete rerun of every older source suite. Only the new suite
was run in both profiles for this batch, alongside all 100 native programs.


## Numeric OrdinalEncoder

Added the sklearn OrdinalEncoder counterpart for numeric real64 category
matrices, with learned/manual categories, NaN missing values, unknown sentinels,
frequency-count/fraction grouping, max_categories and stable count-tie handling.
The native ragged category type aliases the existing TargetEncoder container;
its use does not change TargetEncoder's implementation. Query lookup uses binary
search after category discovery with the existing dense sort.

Inverse transforms return numeric values and optional allocatable logical
unknown/infrequent masks for source None/string values. Actual missing NaNs
remain distinct through both masks being false. Source's sentinel collision
precedence and negative/fractional NumPy indexing are reproduced. Source 1.6.1
has grouped/missing inverse index inconsistencies; the port preserves source
outputs and converts its index failures to checked input diagnostics. Unlike
the source inverse, the native function preserves its input array. These
representation choices and upstream edge cases are documented in API/COVERAGE.

The exploratory checked native unit passed, including refits, sentinel tags,
source grouping ties and admitted group/unknown collisions. An initial source
run passed 1,315 fixtures across five seeds, including 549 expected source
inverse IndexErrors. The final harness additionally checks learned category
counts through the source counting helper and genuine refitting after grouped
state. Final full-profile and diagnostic gates are recorded below when complete.


The final checked source comparison passed 16,214 result groups over 1,315
fixtures. Of these, 15,448 groups use sklearn directly (including 549 expected
source inverse failures), and 766 verify preservation of native inverse inputs.
All 549 source failures are IndexErrors; five source RuntimeWarnings arise from
NaN-to-int inverse-index conversion. Both failures and warnings are retained.
All 101 native programs passed the checked/runtime/FPE profile. The public fpm
example exited successfully, and its training/query codes and inverse tags
matched an independent sklearn recipe exactly.
Current inventory is 194 core types, 102 source modules, 101 native programs
and 80 retained core reference suites with 333,495 groups, plus 28 separate
NumPy RNG groups. No existing numerical estimator implementation was changed.


All 101 native programs also passed the optimized profile. The implementation
remained frozen through the checked and optimized gates; the source comparator
uses a separately resolved executable when overlapping a different profile's
build, so no fpm processes run concurrently.


The optimized source comparison passed the same 16,214 groups over 1,315
fixtures. Its group records, 549 expected source inverse errors, five source
warnings and two source hashes are identical to the checked report. No numeric
tolerances were widened; categorical and inverse results use exact equality.


Final diagnostic gate passed all 2,368 unique invalid-input cases, including
37 new ordinal cases. Final audits passed Python syntax, Fortran line-length/
whitespace checks, report parity, source hashes, facade exports, and MIT with
retained BSD attribution. All 101 native programs passed in checked and
optimized builds, the public example matched sklearn, and the new source suite
passed 16,214 groups in each profile. Existing numerical implementations were
unchanged; the 80-suite reference inventory includes retained earlier results
and does not imply that all older source suites were rerun in this batch.


## Refined sparse-covariance cross-validation

Added graphical_lasso_cv to sklearn_covariance and the sklearn facade, using
the existing coordinate-descent kernel and a source-compatible warm-start path.
Implemented automatic/refined and explicit grids, stable smaller-alpha tie
selection, exact default/custom split handling, failed-candidate recovery,
independent-centered path scores, full-data assume_centered semantics and the
separate unselected empirical endpoint. Learned endpoint matrices are native
inspection data, not new estimator/extension types. The normal scalar
GraphicalLasso fit is unchanged; its shared CD kernel now skips zero diagonal
Gram entries exactly as the source does.

Source probes confirmed single-point grids fail upstream, scalar zero-count
grids fail upstream, and an exactly diagonal covariance can produce an automatic
grid with zero and NaN alphas and select zero. The latter path is implemented,
including direct inverse failure propagation and its iteration/history shape.
One-row folds can fail every positive candidate yet leave a valid final fit;
negative-infinity ties choose the smaller alpha and reported std is NaN.
A scoped IEEE environment allows those candidate failures and restores all
three relevant halting modes and flags. Focused native FPE checks passed.

The exploratory source suite passed 5,980 groups over 255 fixtures with five
seeds. It includes p>n folds and covariance conditions around 1e10, rather than
removing ill-conditioned empirical endpoints. Twenty-six endpoint certificates
(six inverse-backward-error and twenty rank/determinant-roundoff certificates)
account for those conditioning differences; ordinary positive-alpha comparisons
and selected/final models passed at the unchanged tolerance. Full checked and
optimized builds, a standalone-family regression, the public example and all
input diagnostics are the final gates recorded below when complete.


The frozen checked source run passed 5,980 groups over 255 fixtures: 5,940 direct
sklearn groups and 40 source-conditioned empirical-endpoint groups. It retains
26 certificates (six inverse backward-error and twenty rank/determinant bounds)
and 338 source warnings (120 ConvergenceWarning, 143 RuntimeWarning and 75
UserWarning, including repeated source endpoint fits). Reports use strict JSON,
encoding nonfinite certificate values as strings. All 102 native programs passed
the checked/runtime/FPE profile. The existing GraphicalLasso source regression
passed all 144 groups. The public fpm example ran successfully and its selected
alpha, likelihood, grid size and precision row matched sklearn independently.
Current inventory: 195 core estimator/transformer/splitter types, 102 source
modules, 102 native tests and 81 core reference suites with 339,475 groups,
plus 28 separate NumPy RNG groups.


A final proof audit corrected the singular-determinant perturbation bound to
include both dimension factors in the ||L|| ||U|| matrix-norm estimate. The
checked 5,980-group comparison was rerun successfully with that corrected
certificate; the Fortran implementation stayed frozen. Its cached executable
was no longer present, so it was relinked from the validated checked object and
archive into a temporary directory. During optimized compilation, a live f951
process briefly showed almost no CPU time; it exited and compilation resumed
before inspection attached. No process was terminated or build restarted.


All 102 native programs passed in the optimized build. The optimized source
run passed all 5,980 groups; checked/optimized fixture, check/oracle, warning
and source-hash records are identical. Both profiles use six inverse and twenty
rank/determinant certificates. Floating certificate values vary across profiles;
the adaptive rank-deficient certificate triggers also differ for two folds
(seed 42/fold 2 in checked versus seed 0/fold 0 in release, cases 45/46).
Those changes are confined to the unselected singular empirical endpoint and
each profile independently satisfies its proof bounds. The corrected matrix-norm
bound is used in both final reports. No selected alpha or final-model group
uses an endpoint certificate.


Final gates passed all 2,401 unique isolated invalid-input cases (33 new), all
102 native programs in checked and optimized profiles, the 144-group existing
GraphicalLasso source suite, the public fpm example, and 5,980 new reference
groups per profile. Final audits passed Python syntax, Fortran line lengths/
whitespace, source hashes, strict JSON, per-profile oracle/certificate scope,
facade exports and MIT/retained BSD attribution. Production Fortran remained
frozen through the final gates. The 81-suite inventory includes retained earlier
results; all older source suites were not rerun for this batch.

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

A release rebuild exposed stale compiler metadata after the RNG interface
addition. Cache removal was blocked by policy; the generated release profile
directories were instead preserved under
build/hist_release_cache_backup_20261121A before a clean successful rebuild.
Production code and native tests were frozen for the final two-profile gates.

Final audits verified current upstream source and tested executable SHA256
hashes, complete unfiltered reports, and fresh objects for all 111 source
modules and 110 native programs in both profiles. Largest boosting output
absolute error was 3.56e-15 in either profile. Fortran formatting, Python
compilation and staged Git whitespace checks passed.


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

The generated build directory disappeared before final gate launch. No fpm
process was live and redirection failed before a build could start. Both
profiles were therefore rebuilt from scratch, with gate logs moved to the
system temporary directory fscikit_nca_20261121. No source files were lost.
The imported upstream License.txt retains its original trailing whitespace;
its verified pristine hash is preferred to cosmetic reformatting. The staged
whitespace audit excludes only that license file.


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

## Coordinate-descent NMF (2026-11-22)

Extended the existing `nmf` type with an explicit dense `solver='cd'` path from
sklearn 1.6.1. It implements alternating coordinate updates, dimension-scaled
L1/L2 penalties, zero-Hessian skips, seeded shuffling, projected-gradient stopping
and fixed-H transform. Reused the existing MiniBatchNMF initialization backend
through a small internal adapter for random, auto, custom and NNDSVD variants.
Added fitted dimensions and first/last violation diagnostics. Existing MU
arithmetic/defaults and other decomposition procedure bodies are unchanged.

Added a native test, custom rank-one example, stdin reference driver and validator.
The new reference suite retains 2,022 fixtures and 20,214 groups, checking public
sklearn fits and transforms, actual upstream Cython coordinate sweeps, objective
values, nonnegativity and reconstruction. Thirty-four isolated rejections cover
controls, initialization domains, query state and conservative real64 limits.
Stationary initialization and exact-zero stopping differences caused by reduction
rounding receive explicit arithmetic-scale and projected-gradient certificates;
the corresponding fixtures and all factor/objective comparisons are retained.

The added feature is an existing sklearn algorithm, with no new core type,
extension algorithm or runtime dependency. Documentation describes dense
Frobenius-only scope and native defaults differing from sklearn. MIT licensing
and retained sklearn/NumPy BSD attribution continue.

Final verification passed all 150 native tests in checked and release builds,
26,906 affected reference groups and 34 rejections per profile. Both explicit
fpm example runs matched sklearn (two iterations and zero reconstruction error).
The final audit verified 287 compiled-content digests per profile and complete
cross-profile fixture inventories. Reports and logs are archived in ignored
`validation/results/nmf_cd_final_audit.json` and companion files.

Inventory: 217 core public types, 129 core modules, 150 native tests, 128 core
reference suites and 837,309 retained groups; 37 invalid-input registries retain
3,551 cases. Only the affected references above were rerun. The separate 42-group
legacy RNG suite remains excluded and was not rerun.

## Beta-divergence NMF multiplicative updates (2026-11-22)

Replaced the basic NMF MU path with the dense sklearn 1.6.1 multiplicative/MM
algorithm, supporting numeric beta losses, shared random/custom/NNDSVD
initialization, auto component counts and dimension-scaled L1/L2 penalties.
Implemented source update order, MM exponents, epsilon floors/cutoffs,
ten-iteration convergence and fixed-H transform initialization. Matched the
observed sklearn 1.6.1 cached KL/L1 denominator mutation during transform.
Native checked arithmetic diagnoses unsupported real64 extremes before powered
operations, products, reductions or division. Default parameter values are
retained; seeded initialization and stopping now follow source MU semantics.

Added an internal MU module, native test, KL example, reference driver and
validator. Updated the CD invalid-input registry to reject non-Frobenius loss
after regularized MU became supported. CD numerical kernels, other decomposition
procedure bodies and the MiniBatchNMF source remain unchanged.

The new canonical sweep retains 2,090 fixtures and 20,844 groups, including
56 all-zero-H transform rejections. It checks public fits/transforms, actual
upstream updates, fitted metadata, initial/final/checkpoint errors and independent
objectives/nonnegativity/reconstruction. Six checked diagnostic certificates
bound divergence cancellation and independently verify the final divergence
using 60-digit Decimal arithmetic. No fixture or factor comparison is dropped.
Forty-four new isolated native rejection cases pass with trapped arithmetic.

A targeted subnormal counterexample exposed premature powered-value clipping.
The implementation now lets pow round representable subnormal results, matching
source. A native regression and 24 boundary fixtures across beta 2.5, 3 and 4
retain zero absolute tolerance for W/H factors. Both complete verification
profiles are rerun after the fix; initial pre-fix reports are archived separately.

No public estimator type, extension algorithm or runtime dependency is added.
Original interfaces and checked arithmetic remain MIT; sklearn/NumPy BSD
attribution and the NMF research references are retained.

Final post-fix verification passed all 151 native tests, 47,750 affected reference
groups and 78 rejection checks per profile in checked and release builds. Both
explicit fpm KL example runs matched sklearn (30 iterations and divergence
approximately 0.078976237024). The final audit verified all fixture fields,
report hashes and 290 compiled-content digests per profile; the six MU and sixteen
CD numerical certificates pass in each profile. Final reports/logs are archived
in ignored `validation/results`, including `nmf_mu_final_audit.json`.

Inventory: 217 core public types, 130 core modules, 151 native tests, 129 core
reference suites and 858,153 retained groups; 38 invalid-input registries retain
3,595 cases. Only the affected suites above were rerun. The separate 42-group
legacy RNG suite remains excluded and was not rerun.

## Positive linear-model constraints (2026-11-22)

Added `positive` to the shared coefficient-storage hierarchy, supporting the
sklearn option in LinearRegression, Ridge, ElasticNet, Lasso and coordinate-CV
wrappers. Unsupported inherited uses in Bayesian/ARD, OMP, Huber, RANSAC,
TheilSen, PA, GLMs and RidgeCV are diagnosed. Weighted OLS/Ridge fitting and
fitted composition clones preserve the option. The false path retains the
existing unconstrained SVD and coordinate methods.

Implemented a native Lawson-Hanson QR active set, using scaled positive-beta
Householder reflectors and Givens column removal. Positive Ridge uses an
augmented NNLS design. Positive coordinates are projected to zero in ordinary
and CV/refit paths. Unrestricted intercept recovery and checked predictions
remain separate from coefficient constraints. Positive OLS avoids SVD square-V
storage; a two-row, 10,000-feature fixture exercises the wide path.

Independent KKT checks found and reproduced a SciPy 1.15.3 defect: the final
below-pivot QR entry is left stale on insertion, which can corrupt later column
removal. The native implementation zeros that full tail. A deterministic wide
regression and BVLS/KKT certificates retain the correction. Exact positive
Ridge also receives strong-convexity certificates where public finite L-BFGS
stopping differs. Nonunique NNLS representatives and their query differences
are recorded explicitly rather than silently skipped.

Added two native tests, two reference drivers, a positive-model example and
core/composition validators. Existing coordinate-CV validation now accepts an
explicit executable to avoid nested fpm calls in regression gates. The core
sweep retains 1,044 fixtures/8,388 groups; the composition sweep retains 192
fixtures/1,608 groups. Thirty rejection cases check domains and unsupported
flags. Checked verification passed all 153 native tests and 55,764 affected
groups. Its audit verifies 296 compiled-content digests and all core/composition
fixture fields, including one QR-tail correction, two nonunique cases, thirteen
direct Ridge certificates and 216 certified composition Ridge children.

Original Fortran interfaces, range guards and validation are MIT contributions.
Existing sklearn/SciPy BSD attribution remains, with LAPACK 3.12.0 conventions
and its full BSD license retained in `LICENSES/lapack-LICENSE`. No new public
estimator type, extension algorithm or external runtime dependency is added.

Final verification passed all 153 native tests in checked and release profiles,
55,764 affected reference groups and 30 new rejection cases per profile. The
example matched sklearn in both profiles. Final audits verified 296 compiled
source-content digests per profile, complete field inventories and all solver
certificates. The initial composition comparison used public Ridge finite
optimizer coefficients; final reports retain certified exact constrained
solutions and the public prediction differences. Artifacts are archived in
ignored `validation/results`, including `positive_final_audit.json`.

Inventory: 217 core public types, 131 modules, 153 native tests, 131 core
reference suites and 868,149 groups; 39 invalid-input registries retain 3,625
cases. Only the affected suites above were rerun. The separate 42-group legacy
RNG suite remains excluded and was not rerun.

## Dense free nonnegative factorization

Implemented sklearn's `non_negative_factorization` in `sklearn_decomposition`
and exported it through the facade. The subroutine returns allocatable W/H
matrices and an iteration count. It forwards CD/MU controls, numeric beta loss,
initialization, automatic rank, regularization, seed and shuffling to existing
kernels. The fixed-H branch infers rank from H when requested, validates H,
initializes W according to the solver, ignores supplied W and returns H
unchanged. Existing transform arithmetic is unchanged; its private helper now
optionally returns the iteration count. Existing `nmf` defaults are preserved;
the function uses sklearn's CD/automatic-rank/200-iteration/`1e-4` defaults and
the documented native seed-zero convention.

Added a native test, stdin reference driver, fixed-component example and a
916-fixture public-sklearn comparison suite (3,664 groups, 12 diagnostic
cases). Tests include independent objectives, nonnegativity, fixed-H ownership,
invalid shapes/controls, ignored nonfinite W and overcomplete fixed dictionaries.
CD zero-tolerance stopping differences require both projected-gradient
residuals to satisfy an explicit roundoff bound; source results are retained.
Original Fortran adapters and tests remain MIT, with retained sklearn BSD
attribution. No dependency or public extension algorithm was added.

Checked verification passed all 154 native tests, 44,914 affected reference
groups and all 90 new/prior CD/MU diagnostics. Fourteen new-suite roundoff
certificates passed. The example matched sklearn. The checked audit verified
294 compiled-content digests, report/source/validator/executable hashes, new
field inventories and unchanged existing decomposition arithmetic.

Release verification passed the same 154 native tests, 44,914 affected reference
groups and 90 diagnostic cases. The optimized new suite retained 16 certified
stationarity-level iteration differences. Both examples matched sklearn. The
final audit verified all 294 compiled-content digests per profile, new field
inventories, report/executable/validator/source hashes, the stopping certificates
and unchanged existing decomposition arithmetic. Results and logs are archived
in ignored `validation/results`, including `nonnegative_final_audit.json`.

Inventory: 217 public types, 131 core modules, 154 native tests, 132 core reference
suites and 871,813 groups. Forty invalid-input registries retain 3,637 cases.
Only the affected suites above were rerun. The separate legacy RNG suite remains
excluded and was not rerun.

## Stateless text hashing

Implemented sklearn 1.6.1 HashingVectorizer with word/character/within-word
character n-grams, stop-word filtering, signed/binary counts, normalized CSR
output and explicit dense adapters. The text module and facade re-export the
new type. Existing MurmurHash3, FeatureHasher and CSR storage are reused without
arithmetic changes. Dynamic strings preserve exact content; padded character
arrays trim trailing spaces. Tokens use dynamic storage, and sparse output
supports the default million-feature width without a vocabulary or IDF state.

Added native tests, a stdin reference driver, a 1,779-fixture / 14,232-group
public-sklearn comparison suite with 13 diagnostic cases, and a sparse NB
classification example. Analyzer token order is checked before hashing,
including all English stop words, ASCII controls, long/empty text and maximum
integer n-gram ranges. A checked-DO boundary failure in the first development
run was corrected using explicit guarded loop advancement; the test is retained.
The obsolete build was stopped and restarted for the correction. The complete
corrected development run compares all 288,369 analyzer tokens exactly, with
maximum numeric differences below `4.5e-16` and no tolerance exceptions.

ASCII analysis is an explicit current limit; Unicode receives a diagnostic.
The original Fortran contributions remain MIT with retained sklearn BSD and
stop-list attribution. No dependency or extension algorithm was added.

Checked verification passed all 155 native tests, 17,369 affected reference
groups and 53 diagnostics (13 new and 40 prior). Existing feature-mapping/hash
regressions retain 2,945 groups and integration retains 192. All 288,369 analyzed
tokens and stored values match sklearn exactly; row-norm identity errors remain
below `4.5e-16`. The sparse NB example matches sklearn: shape `(4,1024)`, 16
stored entries and predictions `[0,1]`. The checked audit verifies 296 compiled
source-content digests, pinned stop words, all field/token inventories, hashes
and unchanged existing text/hash/CSR arithmetic.

Release verification passed all 155 native tests, the same 17,369 affected
reference groups and 53 diagnostics. Both profiles match every analyzer token
and stored value exactly, with row-norm identity errors below `4.5e-16`; both
examples match sklearn. The final audit verifies 296 compiled-content digests
per profile, all field/token inventories, source/validator/executable hashes,
the pinned English stop list and unchanged existing text/hash/CSR arithmetic.
Results, logs and audit scripts are archived in ignored `validation/results`,
including `hashing_text_final_audit.json` and the retained pre-fix logs.

Inventory: 218 public types, 132 core modules, 155 native tests, 133 core reference
suites and 886,045 groups. Forty-one invalid-input registries retain 3,650 cases.
Only the affected suites above were rerun. The separate legacy RNG suite remains
excluded and was not rerun.

## Normalizer, Binarizer and sparse normalization

Added dense/CSR stateless preprocessing estimators, one-to-one feature metadata,
dense protocol composition and dense/CSR free-function overloads. Normalization
supports both axes and returned norms, with source-specific dense near-zero
versus sparse nonzero handling. Sparse normalization/binarization preserve the
source distinctions among stored duplicates, canonical max reductions, explicit
zeros and threshold filtering. Quiet IEEE primitives retain source overflow/
NaN outcomes without native arithmetic traps. All outputs own their storage.

The reference checks identified that feature-name output requires fitting,
even though transforms work before fit. They also exposed a stale shared NB
weight restriction. The adapter now forwards existing sample weights for the
MultinomialNB hierarchy, including BernoulliNB and ComplementNB. New regression
cases cover weighted/unweighted pipelines, voting and bagging for all three.
The separate multioutput/chain dispatchers remain unchanged. Original Fortran
adapters remain MIT with retained sklearn/SciPy BSD attribution.

New reference suites retain 1,801 core fixtures / 19,386 groups, 29 diagnostics,
50 source nonfinite-result fields, and 864 composition fixtures / 5,328 groups.
Checked verification passes 157 native tests, 57,356 affected reference groups
and 130 diagnostics. Existing pipeline, column/union, voting, stacking and
integration suites were rerun. The example matches sklearn's normalized first
row `[0.6,0.8]` and weighted classifications `[1,1,-1,-1]`. The checked audit
verifies 304 compiled-content digests, complete field inventories, IEEE outcome
masks, source/validator/executable hashes and unchanged preprocessing/base-dispatch
arithmetic outside the intended additions and discrete NB forwarding/flag fix.

Release verification passed all 157 native tests, the same 57,356 affected
reference groups and 130 diagnostics. Both examples matched sklearn. The final
audit verified 304 compiled-content digests per profile, complete field
inventories, source finite/nonfinite outcomes, pinned executable/validator/
sklearn/SciPy hashes and unchanged surrounding procedure bodies. Artifacts are
archived in ignored `validation/results`, including `stateless_final_audit.json`.

Inventory: 220 public types, 133 core modules, 157 native tests, 135 core reference
suites / 910,759 groups and 42 invalid-input registries / 3,679 cases. Only the
affected suites above were rerun. The separate legacy RNG suite remains excluded
and was not rerun.

## PolynomialFeatures estimator, degree ranges and CSR expansion

Added a composable `polynomial_features` estimator while retaining the earlier
`polynomial_features(x,...)` array-call generic. It supports dense/CSR input,
minimum/maximum degree, interaction-only output, bias control, fitted widths and
names, lazy int64 powers/counts and dense protocol composition. Exact checked
binomials/Vandermonde counting avoids unnecessary large intermediates; exponent
iteration avoids allocating tuples proportional to enormous degrees. The old
array-call degree-12 cap is removed. No new dependency or extension is added.
MIT original adapters retain sklearn/SciPy BSD source attribution.

Dense suffix reuse follows source multiplication association. CSR expansion uses
canonicalized row storage and exact global column ranks, with source explicit-zero
rules and quiet IEEE multiplication. Validation exposed higher-degree sparse
`Inf * implicit-zero` NaNs, so the extreme-overflow path accounts for absent
feature positions. Ordinary sparse rows continue to enumerate stored features.
Native canonicalization corrects source duplicate-column product errors; empty
interaction output also corrects source metadata/sparse-stacking failures. Raw
source outcomes and exceptions remain in reports.

New references retain 1,985 core fixtures / 38,020 groups, 31 diagnostics,
984 noncanonical source records, 48 empty metadata exceptions, 36 empty sparse
transform exceptions and 243 nonfinite reference fields. Three huge single-feature
powers/names use symbolic identities, and one huge interaction fit-only count uses
the source's exact requested-range helper rather than its expensive unused full
auxiliary sum. The 480 composition fixtures / 2,880 groups cover normalization,
weighted/unweighted Ridge, weighted unions, selected columns, names and prototypes.
Transform tolerance is 3e-13 relative / 3e-14 absolute; independent Ridge SVDs use
3e-12 / 3e-13. Integer metadata and nonfinite masks compare exactly.

The native composition test initially used different Ridge alpha values for the
pipeline and direct oracle; the fixture now shares the same prototype settings.
All 159 checked native tests and the explicit quadratic/CSR example then passed.
Pre-fix logs are retained in ignored validation/results.

Checked verification passed all 159 native programs, 77,722 affected reference
groups and 161 diagnostics. The audit verified 307 compiled-content digests,
complete fixture/field inventories, source/validator/executable hashes and
unchanged surrounding preprocessing/shared-dispatch code. The explicit example
matched predictions `[0.75,1,2.75]`, CSR shape `(5,2)`, eight stored entries and
feature names `x0`, `x0^2`.

Repeated fresh-process checks later exposed an intermittent formatted-I/O crash
in exponent name conversion in both profiles, despite their earlier native passes.
Address mapping placed the failure in internal integer formatting while generating
names. Lazy names now convert nonnegative indices/exponents directly to decimal
text, without formatted-I/O state. Native regressions cover empty/UTF-8 names and
refitting. The core validator now repeats the original name/refit fixture in 128
fresh processes and compares every emitted field exactly; these repetitions are
reported separately and do not inflate the distinct-fixture/group inventory.
Both profiles and their affected comparisons were rebuilt and rerun after this
change; the final results below verify the corrected implementation.

Final checked and release verification each passes all 159 native tests, 77,722
affected reference groups and 161 diagnostics. Both explicit examples match source.
Each full core report includes 128 fresh-process name/refit repetitions, in
addition to a focused optimized final-source run of 1,000 repetitions. The final
audit verifies 307 compiled-content digests per profile, complete field/fixture
inventories, exact integer/nonfinite outcomes, pinned hashes and unchanged
surrounding preprocessing/shared-dispatch code. Artifacts and pre-fix logs are
retained in ignored `validation/results`, including `polynomial_final_audit.json`.

Inventory: 221 public core types, 134 source modules, 159 native tests, 137 core
reference suites / 951,659 groups, and 43 invalid-input registries / 3,710 cases.
Only affected reference suites were rerun; the separate legacy RNG suite remains
excluded and was not rerun. These types remain subsets of their sklearn APIs.

## Image/grid graph builders and subtraction boundary correction

Added `grid_to_graph` and `img_to_graph` to the image module/facade, backed by a
native sorted CSR builder. Voxel numbering and selected-mask compaction follow
NumPy C order. Grid graphs retain ones on axis edges/diagonal; image graphs
retain absolute adjacent differences and image-valued diagonals, including
stored zeros and nonfinite values. Scalar/vector/matrix/volume image adapters,
logical masks, empty images/selections and index/allocation checks are documented.
The builder uses linear sparse storage; a native 60,000-node grid regression
checks 299,000 stored entries and exact boundary/interior degree counts.
No estimator type, public extension or external dependency is added.

The reference suite preserves 560 sklearn 1.6.1 singleton-axis mask exceptions.
The source selects diagonal values from a squeezed image using an unsqueezed
mask. Native same-shaped masks are valid; the public source function with a mask
matched to its internal squeezed image supplies the intended graph oracle.
Ordinary successful raw source graphs are also compared. New coverage is 1,832
fixtures / 14,653 groups, 13 diagnostics, 352 nonfinite source fields and three
large sparse cases checked without dense adjacency.

The original development run failed on a near-maximum image-gradient fixture:
the shared quiet-subtraction guard could report infinity for a rounded finite
result, differently in the two edge orientations. The shared helper now scales
the extreme opposite-sign path by a power of two before testing true
overflow. Ordinary subtraction, signed zeros and nonfinite semantics are retained.
Independent NumPy checks pass 8,703 pairs / 17,406 groups bit for bit, including
258 cases exposing the previous guard. Other quiet primitive/reduction procedure
bodies and existing image patch procedures remain unchanged. Original native
storage/arithmetic remain MIT with retained sklearn/NumPy BSD attribution.

Checked verification passes all 161 native tests and the two new reference
suites. The public example matches a 20-node / 82-entry grid and the masked
three-node gradient matrix `[[1,0,7],[0,4,0],[7,0,8]]` with five stored entries.
The shared helper change is accompanied by image, structured-feature, histogram
boosting, sparse/smoothed NB, positive regression, preprocessing, weighted
classifier/tree and composition/integration regressions in both build profiles.
Final results below follow termination of both profiles and all regression groups.

The checked audit passes 161 native tests, 276,126 affected reference groups and
308 diagnostic cases. It verifies 321 compiled-content digests, complete new-suite
field inventories, exact subtraction bits, graph-mask outcomes, source/validator/
executable hashes and unchanged surrounding image-patch/quiet-primitive bodies.
The separate legacy RNG suite was not rerun and remains excluded.

Final checked and release verification each passes all 161 native programs,
276,126 affected reference groups and 308 diagnostic cases. Both examples match
source. The audit verifies 321 compiled-content digests per profile, complete
new-suite field inventories, exact finite subtraction bits/nonfinite outcomes,
graph mask corrections, source/validator/executable hashes, distinct optimized
executables and unchanged surrounding image-patch/quiet-primitive bodies.
Artifacts and pre-fix logs are retained in ignored `validation/results`, including
`image_graph_final_audit.json`.

Inventory: 221 public core types, 135 source modules, 161 native tests, 139 core
reference suites / 983,718 groups and 44 invalid-input registries / 3,723 cases.
Only affected suites were rerun. The separate legacy RNG suite remains excluded
and was not rerun. Public types remain subsets of their source APIs.

## Regression-tree criteria (2026-11-24)

Added scikit-learn Friedman MSE and Poisson criteria to the existing
`decision_tree_regressor` and standalone `extra_tree_regressor`. The default
remains squared error. Weighted means, forward/reverse incremental criterion
updates, mutable feature ordering, Poisson positive-child restrictions and
Friedman's distinct min_impurity_decrease scale follow sklearn 1.6.1.
The Poisson split proxy uses the source base-two logarithm; node deviance uses
the natural logarithm. GridSearchCV/RandomizedSearchCV can set `criterion`.
Forest-level criterion parameters, absolute-error fitting, training NaNs,
pruning and multioutput single-tree fitting remain outside this addition.

The new reference suite checks 1,247 fixtures / 16,211 groups, comparing all
13 fields directly against installed sklearn: predictions, leaf IDs,
importance, metadata, values, impurity, weighted mass, thresholds, children,
features, active-row counts and criterion preservation. Structures and integer
fields compare exactly; floating fields use rtol=2e-10/atol=2e-12 with matching
NaN/Inf outcomes. There are 89 nonfinite source values, and 16 isolated
source-matched invalid-input cases. The corpus includes both splitter families,
random feature selection, weighted/zero/signed/extreme masses, target scales,
constant features, minimum-improvement boundaries and missing query routing.
No alternate-tree certificate is used in this new suite.

Checked verification passed all 196 native test programs, the new suite and
9,984 groups / 768 fixtures from the existing weighted-tree suite. The latter
retains its existing source-conditioned and pure-subtree certificates.
Its separate invalid-input registries were not rerun. The facade example
matches sklearn: Poisson predictions [1,1,1,4,8.5,8.5] in five nodes and
Friedman ExtraTree predictions [1.75,1.75,1.75,1.75,8,9].

Cumulative inventory: 222 public core types, 151 source modules, 196 native
programs, 172 reference suites / 3,288,119 retained required groups, and
77 invalid-input registries / 4,443 cases. These counts describe API subsets.
MIT licensing and retained sklearn BSD attribution continue; no package
dependency or public extension algorithm was added.

Final verification: both the runtime-checked profile
(`-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow`) and optimized
release profile passed all 196 native programs, 26,195 affected reference
groups and 16 new diagnostics. Both examples match sklearn. The final audit
verified 354 compiled source units per profile using dependency-aware fpm
fingerprints, complete 13-field fixture inventories, stable source/validator/
executable/report hashes, unchanged installed tree binaries and distinct
checked/release reference executables. The private fpm 0.14.0 executable
checksum was verified; the global fpm installation was unchanged. Only the two
affected reference suites were rerun. Reports and audit artifacts remain
local under ignored `validation/results/tree_criteria*`, including
`tree_criteria_final_audit.json`.

## Weighted source-seeded regression forests (2026-11-24)

Expanded the existing random_forest_regressor and extra_trees_regressor
subsets with squared-error/Friedman MSE/Poisson criteria, optional fitting
and score weights, min_impurity_decrease, leaf-index matrices and learned
feature counts. Random forests accept integer max_samples with zero as
the unset/all-rows sentinel. Regression forests now use source NumPy tree
seeds and weighted bootstrap multiplicities instead of native random draws
and duplicated training rows. Active-row leaf-size rules therefore match
sklearn. The intrinsic Fortran RNG is preserved. Forest importance excludes
stumps and returns zeros when every tree is a stump. Search and SelectFromModel
forward regression-forest fit weights; criterion/max_samples/impurity-decrease
parameters are available in search.

The new comparison checks 1,210 forest fixtures / 20,538 result groups and
28 isolated source-matched input diagnostics against sklearn 1.6.1. Sixteen
mandatory fields include predictions, leaf IDs, forest importance, component
seeds, tree/forest metadata, every component node's values/impurity/mass/
threshold/children/feature/active count, criterion forwarding and intrinsic
RNG preservation. An additional score field covers 1,178 fixtures with valid
score inputs. Structures and integer fields compare exactly; floating results
use rtol=2e-10/atol=2e-12 with matching NaN/Inf outcomes. All 294 nonfinite
source values match. The suite uses direct source comparisons, with no
alternate-tree certificate. It includes signed/zero/extreme weights, sparse
count targets, extreme target scales, different bootstrap sizes, constants,
maximum supported integer seeds and NaN queries after finite training.

The checked build passed five affected native programs: regression forests,
trees/ensembles, search, model-based selection and cross-validation. It also
passed 28,494 reference groups: 20,538 new forest groups, 4,068 existing search
groups / 270 fixtures and 3,888 existing SelectFromModel groups / 432 fixtures.
The existing search/selector suites retain their documented rank/tree
certificates. Their separate invalid-input registries were not rerun. The
facade example matches weighted Poisson forest predictions and R2
0.88930868288274723, plus Friedman ExtraTrees predictions.

ExtraTrees retains the existing forced non-bootstrap fitting behavior.
Bootstrap-multiplied weights must remain finite. Fractional sample/feature
counts, OOB scores, warm starts, pruning, monotonic constraints, sparse input,
training NaNs, absolute-error criteria and multioutput forests remain outside
this subset. Classification forests retain their existing fitting algorithm.

Cumulative inventory: 222 public core types, 151 source modules, 197 native
programs, 173 reference suites / 3,308,657 retained required groups, and
78 invalid-input registries / 4,471 cases. These are API subsets. MIT licensing
and retained sklearn/NumPy BSD attribution continue. No dependency or public
extension algorithm was added.

Final regression-forest verification: four-job fpm 0.14.0 builds compiled
the package in checked (`-fcheck=all -fbacktrace
-ffpe-trap=invalid,zero,overflow`) and optimized release profiles. Each passed
the five focused native programs and 28,494 affected reference groups, plus
28 new source-matched input diagnostics. The other 192 native programs were
compiled but not executed in this batch. No other reference or invalid-input
suite is included in these rerun totals. Both facade examples match sklearn.

The combined audit verifies 164 compiled source units per profile with
dependency-aware fingerprints, complete 16/17-field forest inventories,
1,178 scored fixtures, stable source/binary/validator/executable/report hashes,
matching comparison inventories across profiles and distinct optimized
reference executables. It verifies unchanged unrelated source/test paths and
the private fpm checksum. The global fpm installation was unchanged. Reports
remain under ignored `validation/results/regression_forests*`, including
`regression_forests_final_audit.json`.

## Absolute-error trees and regression forests (2026-11-24)

Added criterion='absolute_error' to decision_tree_regressor,
extra_tree_regressor, random_forest_regressor and extra_trees_regressor.
Leaves use the sklearn weighted-median convention; split proxies and node
impurities use weighted absolute deviations. The existing size/feature/
impurity-decrease controls, fitting weights, forest bootstrap multiplicities
and criterion search grids apply. This adds one internal source module and
no public estimator type or extension algorithm.

The internal queue translation follows sklearn 1.6.1 tree/_utils.pyx and
tree/_criterion.pyx: stable sorted records, insert/remove/pop updates,
forward/reverse movement, and reset order. Initialized removed records are
retained, and the two buffers are reused across node resets for the entire
fit. Invalid accesses outside initialized storage return quiet NaN. Large
finite target averages can produce source Inf/NaN outcomes; tree importance
normalization now uses quiet scalar IEEE division to preserve those outcomes
under floating-point traps. Absolute-error predict/apply reject query NaNs,
as the source does. Training NaNs remain unsupported.

The new suite contains 400 single-tree and 400 forest fixtures / 11,989
required groups, plus 28 isolated source-matched input diagnostics. Of these,
795 fixtures / 11,915 groups compare directly against installed sklearn,
including exact structures, leaf IDs, component seeds, metadata and active-row
counts, plus floating predictions/values/impurity/mass/threshold/importance
fields at rtol=2e-10/atol=2e-12. All 96 nonfinite direct-source values match.
The 389 valid forest-score fixtures include weighted R2 comparisons.

Five fixtures / 74 groups separately verify empty-queue safety. Source
WeightedMedianCalculator.get_median reads indices k and k-1 when k=0; the
all-zero effective-weight case therefore reads invalid storage. The native
rule is a quiet-NaN root value/impurity, zero active rows and one leaf.
Forest averages containing that tree return NaN predictions; stump trees do
not contribute to importance. Nonempty components in a partially empty
bootstrap fixture still compare to source component trees. The new suite
labels these identities explicitly and makes no source-equivalence claim
for an invalid memory read. Other nonempty MAE fixtures use direct source
comparisons rather than alternate-tree certificates.

Checked verification passed all 198 native programs, 58,722 affected reference
groups and 72 diagnostics. These include 16,211 existing regression-tree
criterion groups, 20,538 existing regression-forest groups and 9,984 existing
weighted-tree groups. The weighted-tree suite retains its earlier documented
source-conditioned/pure-subtree certificates. Its separate input registries
were not rerun. The example matches median leaf 4 versus mean leaf 27 and
weighted median-forest predictions [0.5,1,2.5,76.75]. A separate private
128-tree storage probe also matched the source; it is excluded from retained
group totals.

Cumulative inventory: 222 public core types, 152 source modules, 198 native
programs, 174 reference suites / 3,320,646 retained required groups, and
79 invalid-input registries / 4,499 cases. These remain API subsets.
MIT licensing and retained sklearn BSD attribution continue. No dependency
or public extension algorithm was added.

Final absolute-error verification: checked and optimized release builds each
passed all 198 native programs, 58,722 affected reference/safety groups and
72 diagnostic cases. The new-suite totals are 11,915 direct-source groups
and 74 separately labeled safety/composed-component groups. Both examples
match sklearn, including [0.5,1,2.5,76.75] forest predictions. Four-job builds
used the checksum-verified private fpm 0.14.0; the global fpm was unchanged.

The combined audit verifies 358 compiled source units per profile using
dependency-aware fingerprints, complete single-tree/forest field inventories,
source/safety separation, source/binary/validator/executable/report hashes,
matching comparison inventories and distinct optimized reference executables.
Unrelated source/test/application paths remained unchanged. Only the four
affected reference suites and their reported diagnostic cases were rerun;
other retained suites/registries are excluded from these rerun totals.
Reports remain under ignored `validation/results/tree_absolute_error*`,
including `tree_absolute_error_final_audit.json`.

## Weighted classification forests (2026-11-24)

Expanded random_forest_classifier and extra_trees_classifier with optional
fit/score sample weights, Gini/entropy/log-loss criteria, impurity-decrease
controls, integer max_samples for random-forest bootstraps, leaf matrices,
log probabilities and learned feature/class counts. Class weighting accepts
none, balanced, balanced_subsample and an explicit class_weight_map using
original integer label keys. Balanced factors ignore fitting-weight magnitudes,
matching sklearn 1.6.1; balanced-subsample factors use bootstrap label counts.
Search and SelectFromModel now forward classifier-forest fitting weights.
Search also accepts criterion, class_weight, max_samples and impurity decrease.

Tree seeds and bootstrap draws follow NumPy's legacy RandomState contract.
Row multiplicities multiply weights on the full matrix instead of duplicating
rows. Intrinsic Fortran RNG state is preserved. The forest retains original
classes_, while component trees now expose zero-based encoded class indices,
matching sklearn forest storage. This changes the former native component
class representation and seeded classification-forest results. Forest
probability aggregation, NaN-aware argmax and quiet log probabilities match
source zero/signed-weight outcomes. All-stump importance is zero.

The new reference suite checks 3,480 fixtures / 73,044 groups directly against
installed sklearn 1.6.1. Twenty mandatory fields cover RNG preservation,
labels, probabilities/log probabilities, original/encoded classes, leaf IDs,
importance, seeds, forest/tree metadata and every component node's normalized
class values, impurity, weighted mass, threshold, children, feature and active
row count. An additional score field covers 3,444 valid weighted-accuracy
fixtures. Structures and integer fields compare exactly; floating fields use
rtol=2e-10/atol=2e-12, including all 33,170 nonfinite source values. No alternate
tree certificate is used in this new suite. Its 28 source-matched isolated
input diagnostics pass without runtime bounds errors or floating-point traps.

The corpus spans three criteria, both forest families, three class-weight
presets, full/reduced/disabled bootstraps, minimum-leaf/feature/impurity
controls, zero/signed/tiny/huge weights, dictionary defaults/unknown keys,
single classes, singleton data, NaN queries, integer-limit labels and the
maximum supported seed. ExtraTrees retains its forced non-bootstrap behavior.
Training NaNs, sparse inputs, OOB scores, warm starts, pruning, monotonic
constraints, fractional controls and multioutput forests remain unsupported.
Effective class/bootstrap-multiplied fitting weights must be finite.

Checked verification passed six focused native programs (classification
forests, tree ensembles, regression forests, search, model selection and
cross-validation) and 111,522 affected reference groups. In addition to the
73,044 new groups, those totals include 20,538 regression-forest, 9,984
weighted-tree, 4,068 search and 3,888 SelectFromModel groups. Existing suites
retain their documented certificates. Fifty-six input diagnostics are included
(new classification forests plus existing regression forests); separate
weighted-tree/search/selector invalid-input registries were not rerun.
The balanced facade example matches sklearn, including weighted accuracy
0.75, original label order [-7,9,42] and its forest/ExtraTrees predictions.

Cumulative inventory: 222 public core types, 152 source modules, 199 native
programs, 175 reference suites / 3,393,690 retained required groups, and
80 invalid-input registries / 4,527 cases. These remain API subsets. MIT
licensing and retained sklearn/NumPy BSD attribution continue; no dependency
or public extension algorithm was added.

Final classification-forest verification: runtime-checked and optimized
release builds each passed six focused native programs, 111,522 affected
reference groups and 56 diagnostic cases. The new suite's 73,044 groups all
use direct source comparisons. The other 193 native programs were compiled
but not executed in this batch. Only the five named reference suites and
reported diagnostic cases are included in the rerun totals.

Both facade examples match sklearn. The combined audit verifies 168 compiled
source units per profile with dependency-aware fpm fingerprints, complete
20/21-field inventories, 3,444 scored fixtures, source/binary/validator/
executable/report hashes, matching comparison inventories and distinct
optimized reference executables. It verifies unchanged unrelated source/test
paths and the checksum of private fpm 0.14.0; the global fpm was unchanged.
Artifacts remain under ignored `validation/results/classification_forests*`,
including `classification_forests_final_audit.json`.

## Minimal cost-complexity pruning (2026-11-24)

Added finite nonnegative ccp_alpha to all four standalone tree estimators
and all four random-forest/ExtraTrees classifier/regressor variants. Zero
skips pruning, preserving existing grown trees. Positive alpha uses sklearn
1.6.1's weakest-link risk, first-node tie ordering, inclusive stop threshold,
ancestor updates and preorder rebuilding. Node values/distributions, weighted
masses and sample counts are retained; leaf IDs and importance reflect the
pruned structure. Search adapters accept ccp_alpha, with weighted regression
and classification criterion/class-weight grids in native integration tests.

Standalone trees expose cost_complexity_pruning_path(x,y,sample_weight),
returning tree_pruning_path storage with ccp_alphas and impurities. A dynamic
clone refits at alpha zero so requesting a path preserves an unfitted or
fitted caller and resets its pruning parameter only in the clone. ExtraTree
dynamic fitting and criterion defaults are preserved. Forest components use
their existing weights, class weights and bootstrap multiplicities; forests
do not expose a single public path. This adds one internal module and a
result storage helper, with no new estimator type or extension algorithm.

The new comparison passes 2,682 fixtures / 53,640 direct-source groups and
24 source-matched invalid-alpha diagnostics against installed sklearn 1.6.1.
Twenty fields compare paths, predictions/probabilities, original/encoded
classes, unchanged caller state, forest/tree metadata, seeds, leaf IDs,
importance and every node's values, impurity, mass, thresholds, children,
features and active counts. Integer/structure fields compare exactly;
floating values use rtol=2e-10/atol=2e-12 with matching nonfinite outcomes
(312 source values). Cases include all eight estimator variants, seven
criteria, weighted/zero-weight data, class balancing, bootstrap trees,
constants and source alpha values immediately below/at/above pruning steps.
The new suite uses no alternate-tree certificate.

Checked verification passed all 200 native programs, 185,406 affected
reference/safety groups and 124 diagnostics across six suites. These include
16,211 existing regression-tree criterion, 11,989 absolute-error, 20,538
regression-forest, 73,044 classification-forest and 9,984 weighted-tree
groups. Existing suites retain their documented certificates/safety identities.
Other invalid-input registries were not rerun. The facade example matches
sklearn path alphas [0,1/12,1/6], risks [0,1/3,1/2], and an 11-to-3-node
reduction at alpha 0.1.

Quiet arithmetic preserves defined IEEE outcomes. If no comparable weakest
link exists, path construction rejects that undefined state rather than
using the source's uninitialized node index. Existing sparse/training-NaN,
monotonic, multioutput and fractional-control limits remain.

Cumulative inventory: 222 public core types, 153 source modules, 200 native
programs, 176 reference suites / 3,447,330 retained required groups, and
81 invalid-input registries / 4,551 cases. These remain API subsets. MIT
licensing and retained sklearn BSD attribution continue. No dependency or
public extension algorithm was added.

Final cost-complexity verification: runtime-checked and optimized release
builds each passed all 200 native programs, 185,406 affected reference/safety
groups and 124 diagnostic cases. The new suite contributes 53,640 direct
source groups and 24 alpha diagnostics; existing absolute-error safety groups
and weighted-tree certificates remain separately identified. Both facade
examples match sklearn, including exact-threshold pruning behavior.

The combined audit verifies 363 compiled source units per profile with
dependency-aware fingerprints, complete 20-field pruning inventories, stable
source/binary/validator/executable/report hashes, matching comparison
inventories and distinct optimized reference executables. Unrelated source,
test and application paths remained unchanged. Four-job builds used the
checksum-verified private fpm 0.14.0; the global fpm was unchanged. Only the
six named reference suites and their reported diagnostics were rerun. Reports
remain under ignored `validation/results/tree_pruning*`, including
`tree_pruning_final_audit.json`.

## Sparse decision paths and minimum weighted leaves (2026-11-24)

Added decision_path to all four standalone CART/ExtraTree models and four
random-forest/ExtraTrees classifier/regressor variants. Trees return native
real64 CSR indicators. Forests horizontally concatenate component paths and
optionally return zero-based cumulative node-column offsets, matching the
source/embedding offset convention. CSR row pointers and columns remain
one-based. Root and terminal nodes are included; pruned models use rebuilt
node indices. No dense sample-by-node matrix is formed. Integer totals are
checked in int64 before allocating stored entries and forest columns.

The source sklearn 1.6.1 _decision_path_dense comparison routes NaNs right,
even when apply/predict use the learned missing branch. The native methods
preserve that distinction and share fitted-state, feature-shape, infinity
and float32-range checks with apply. Absolute-error query NaNs are rejected.

All eight variants now accept finite min_weight_fraction_leaf in [0,0.5].
The threshold is the fraction times the effective training-weight total,
including forest bootstrap/class factors. A private NumPy-compatible pairwise
sum computes this total separately from splitter accumulation. Pruning-path
clones and parameter search preserve/forward the control. The native search
integration covers 16 weighted regression and 36 weighted classification
criterion/class-weight/pruning/fraction combinations.

The new suite checks 4,212 fixtures / 105,300 direct-source groups and
44 isolated source-matched diagnostics. Twenty-five fields include CSR shape,
pointers, columns, data and forest offsets plus fitted node structure,
weighted masses, impurity, values/probabilities, predictions, classes, seeds,
leaf IDs, importances, pruning paths and caller preservation. CSR/integer
fields compare exactly; floating fields use rtol=2e-10/atol=2e-12, including
372 nonfinite source values. Cases cover all eight estimators, seven criteria,
pruning, seven fraction values, zero/signed/tiny/huge weights, class balancing,
NaN queries and 137-row weight arrays exercising recursive pairwise reduction.
The new suite uses no alternate-structure certificate.

Checked verification passed all 201 native programs, 290,706 affected
reference/safety groups and 168 diagnostics across seven suites. Existing
absolute-error safety identities and weighted-tree certificates remain
separately identified. Other retained reference/input registries were not
rerun. The facade example matches CSR pointers [1,3,5,7,9], node columns
[1,2,1,2,1,2,1,3], NaN apply leaf 2 versus path [1,3], and forest offsets
[0,3,6,7] with indicator shape (4,7).

Cumulative inventory: 222 public core types, 153 source modules, 201 native
programs, 177 reference suites / 3,552,630 retained required groups, and
82 invalid-input registries / 4,595 cases. These remain API subsets. MIT
licensing and retained sklearn/NumPy BSD attribution continue; no dependency
or public extension algorithm was added.

Final sparse-path verification: checked and optimized release builds each
passed all 201 native programs, 290,706 affected reference/safety groups and
168 diagnostic cases. The new suite contributes 105,300 direct-source groups
and 44 diagnostics. Existing absolute-error safety identities and weighted-tree
certificates remain separately identified. Both facade examples match sklearn,
including CSR arrays, NaN routing and forest column offsets.

The combined audit verifies 365 compiled source units per profile using
dependency-aware fingerprints, complete 25-field path/constraint inventories,
stable source/binary/validator/executable/report hashes, matching comparison
inventories and distinct optimized reference executables. Unrelated source,
test and application paths remained unchanged. Four-job builds used the
checksum-verified private fpm 0.14.0; the global fpm was unchanged. Only the
seven named reference suites and their reported diagnostics were rerun.
Artifacts remain under ignored `validation/results/tree_paths*`, including
`tree_paths_final_audit.json`.

## Best-first tree growth and maximum leaf count

Added integer max_leaf_nodes to all eight standalone/forest tree families,
with zero representing sklearn None and values >=2 selecting best-first growth.
Source-priority heap ties, leaf records, child creation order and per-node
constant-feature discovery follow sklearn 1.6.1 BestFirstTreeBuilder.
Existing weighted criteria, bootstrap/class factors, minimum weighted leaves,
pruning, importance and sparse paths compose with the budget. Search grids
cover the new control. See docs/API.md and example/best_first_trees.f90.

The new direct-source reference suite compares 25 fields per fixture, including
complete structures, paths, pruning and caller state: 4,584 fixtures / 114,600
groups and 16 invalid-input diagnostics. It includes parity/zero-gain heaps,
equal priorities, all criteria, signed/zero/extreme weights and growth-only
nonfinite derived gains. Forty-four extreme-label cases skip pruning-path
construction, whose source kernel can enter undefined state; their growth
outputs still compare directly to source. Existing MAE undefined-empty-queue
safety identities remain separately recorded in the retained regression suite.

Inventory after this batch: 222 public core types, 153 source modules,
202 native programs, 178 reference suites / 3,667,230 retained required groups,
and 83 invalid-input registries / 4,611 cases. Types remain API subsets.
MIT licensing and retained sklearn/NumPy BSD attribution continue; no dependency
or public extension algorithm was added. Canonical verification is recorded below.

Final best-first verification: checked and optimized release builds each
passed all 202 native programs, 405,306 affected reference/safety groups and
184 diagnostic cases. The new suite contributes 114,600 direct-source groups
and 16 diagnostics. Both best-first example executables match sklearn's leaf
count, predictions and one-based apply indices.

The combined audit verifies 367 compiled source units per profile using
dependency-aware fingerprints, complete 25-field best-first inventories,
stable source/binary/validator/executable/report hashes, matching comparison
inventories and distinct optimized reference executables. Existing MAE safety
identities remain separately identified. Unrelated source/test/application
paths remained unchanged. Four-job builds used the checksum-verified private
fpm 0.14.0; the global fpm was unchanged. Only the eight named affected tree
reference suites and their reported diagnostics were rerun. Artifacts remain
under ignored validation/results/best_first*, including
best_first_final_audit.json. Implementation commit: a13efc7.

## Source-seeded Gaussian and sparse random projections

Replaced the intrinsic Fortran random stream for GaussianRandomProjection and
SparseRandomProjection with NumPy-compatible scalar-seeded components.
Row-major normal draws, source scaling, binomial row counts, automatic
tracking/permutation/reservoir sampling, deferred sign draws and the density-one
branch follow sklearn 1.6.1. Added n_features_in_ and finite parameter checks.
Fits preserve the intrinsic RNG and refits clear disabled inverse caches.
The binomial kernels moved unchanged into a shared internal source module;
bicluster generation is rechecked because it now imports that module.

The new direct-source suite contains 808 fixtures / 6,464 required groups and
43 invalid-input cases, covering five seeds, narrow/square/wide matrices,
automatic JL dimensions, automatic and explicit densities, tiny-density zero
matrices, both inverse-cache modes, component arrays, transforms, inverses,
metadata, refits and RNG isolation. Component matrices compare exactly;
transform and native SVD pseudoinverse comparisons use numerical tolerances.
The older regression/factor/projection suite retains injected-matrix algebra
checks and now accepts an explicit executable for canonical verification.

Inventory: 222 public core types, 154 source modules, 203 native programs,
179 reference suites / 3,673,694 retained groups and 84 invalid registries /
4,654 cases. Existing dense real64 projection storage and native SVD limits
remain. Earlier native seeds produce different matrices after this correction.
MIT licensing and BSD attribution remain; no dependency or extension is added.
Canonical checked/release verification is recorded below.

Final seeded-projection verification: checked and optimized release builds
both passed test_seeded_projections, test_factor_projection and
test_bicluster_datasets. These three native programs were run; the other 200
native programs were not executed in this batch. The three affected reference
suites each passed 17,768 total required groups and 57 diagnostics per profile:
6,464 new seeded-projection groups / 43 diagnostics, 11,100 retained bicluster
groups / 14 diagnostics, and 204 regression/factor/projection algebra groups.
Both facade examples match source components and transformed coordinates.

The combined audit verifies 165 compiled source units per profile, the unchanged
extracted binomial body, exact component arrays, all eight result fields for
all 808 new fixtures, stable sources/executables/validators/reports and matching
profile inventories. Source fixtures include 25 nontrivial tracking rows,
26 reservoir rows and 416 permutation rows, plus both inversion and BTPE
binomial branches. Native inverse/SVD comparisons retain explicit numerical
tolerances. No unrelated source/test/application path changed. Four-job builds
used the checksum-verified private fpm 0.14.0; the global fpm was unchanged.
Artifacts are under ignored validation/results/seeded_projections*, including
seeded_projections_final_audit.json. Implementation commit: 87923d7.

## CSR random projections and explicit sparse outputs

Both random projection types now accept CSR inputs for fitting, transforming
and inverse transforms. Shape-only fitting shares the unchanged seeded
component generators with dense fitting. CSR query/latent values multiply
directly without densifying inputs. SparseRandomProjection adds explicit
transform_sparse and fit_transform_sparse methods returning canonical CSR,
with one projected-row workspace and two-pass storage construction. Existing
methods keep dense outputs; inverse transforms always remain dense.

The direct reference suite covers 800 fixtures / 13,600 groups and 44
invalid-input cases. Cases include five seeds, narrow/square/wide geometry,
automatic JL dimensions, dense/sparse component branches, inverse caching,
empty rows, explicit zeros, unsorted duplicates, severe cancellation and CSR
preservation. Component arrays and all training/query projection values compare
exactly; sparse outputs compare after canonicalizing source storage. Native
SVD inverse tolerances remain numerical, with a cancellation-specific absolute
bound for 1e8 / -1e8 duplicate latent entries. The refit field separately labels
source repeatability plus the native inverse-cache-clearing contract.

Six structural diagnostics use SciPy CSR check_format as their source oracle;
38 use sklearn estimator validation. No malformed arrays enter source numeric
kernels. Native tests include a 100,000-by-10,000 zero CSR matrix (8 GB if
densified) and large sparse projected output. No dependency, public type or
extension algorithm was added. Dense component storage remains explicit.

Inventory: 222 public core types, 154 source modules, 204 native programs,
180 reference suites / 3,687,294 retained groups and 85 invalid registries /
4,698 cases. MIT licensing and retained sklearn/NumPy BSD attribution continue.
Canonical verification is recorded below.

Final CSR-projection verification: checked and optimized release builds each
passed test_csr_projections, test_seeded_projections and test_factor_projection.
These three native programs were run; the other 201 native programs were not
executed in this batch. Three affected reference suites each passed 20,268
required groups and 87 diagnostics per profile: 13,600 new CSR groups /
44 diagnostics, 6,464 retained seeded-projection groups / 43 diagnostics,
and 204 regression/factor/projection algebra groups. Both facade examples
match source projection/inverse values and canonical CSR pointer/index/data
arrays. Large sparse fits and sparse output tests passed in both profiles.

The combined audit verifies 165 compiled source units per profile, complete
17-field inventories for 800 new fixtures, exact component/transform/CSR values,
stable source/executable/validator/report hashes, matching profile inventories
and distinct optimized executables. Six new malformed-structure diagnostics
use SciPy check_format; 38 use sklearn validation. The 800 mixed refit/cache
contract groups remain separately labeled from 12,800 direct source groups.
Native SVD inverse comparisons use the documented numerical tolerances,
including cancellation-specific amplification. Only the three affected suites
and their diagnostics were rerun. Unrelated source/test/application paths
remained unchanged. Four-job builds used the checksum-verified private fpm
0.14.0; the global fpm was unchanged. Artifacts remain under ignored
validation/results/csr_projections*, including csr_projections_final_audit.json.
Implementation commit: ce1fdee.

## Source-seeded dense and CSR RBF features

RBFSampler now draws source-compatible row-major normals followed by uniform
phases, accepts CSR inputs and records n_features_in_. The scale branch follows
NumPy centered dense variance and SciPy CSR E[X^2]-E[X]^2, including implicit
zeros, unsorted duplicates, explicit zeros, row reduction order and the
zero-variance fallback. Existing scalar-seeded NumPy RNG and quiet IEEE helpers
are reused. Fits preserve the intrinsic RNG; native old seeded matrices change.
Transform normalization follows live n_components, matching source mutation.

The new suite contains 1,064 fixtures / 8,512 result groups and 33 diagnostics.
Weights, offsets and fitted gamma compare exactly to sklearn 1.6.1; transformed
features use numerical tolerances. Dense/CSR fit/query combinations cover five
seeds, multiple shapes/components, numeric/scaled gamma, zero/constant/near-
constant/tiny/huge data, duplicate cancellation, post-fit normalization changes,
refits and input/RNG preservation. The source returns 522 retained nonfinite
values; the native checked path preserves those outcomes without traps.
Three malformed-structure diagnostics use SciPy check_format; 30 use sklearn
validation. Native tests include large shape-only/variance CSR fits and queries
without allocating an 8 GB dense input. Existing kernel-approximation and chi2
native programs are rerun because they share the module.

Inventory: 222 public core types, 154 source modules, 205 native programs,
181 reference suites / 3,695,806 retained groups and 86 invalid registries /
4,731 cases. Dense output storage remains explicit. The old combined isotonic/
kernel algebra suite now accepts an explicit executable and retains its
separately injected-map checks. No new public type, dependency or extension
algorithm was added. MIT licensing and retained BSD attribution continue.
Canonical verification is recorded below.

A release comparison exposed a dense large-phase accumulation difference:
for 1e154 inputs, separate product rounding changed a phase bit and its cosine.
A phase-level probe identified fused accumulation matching the installed BLAS
reference. Large finite dense phases now use explicit standard ieee_fma;
ordinary bounded phases retain MATMUL, and nonfinite bounds retain quiet IEEE
composition. A native regression checks this finite phase case. The unchanged
1,064-fixture optimized probe passes all 8,512 groups, 33 diagnostics and 522
nonfinite reference values. No fixture or tolerance was relaxed or removed.

Final RBF-feature verification: checked and optimized release builds both
passed test_rbf_features, test_kernel_approximation and test_chi2_samplers,
including the fused finite-phase regression. These three native programs were
run; the other 202 native programs were not executed in this batch. The two
affected reference suites each passed 8,701 groups and 33 diagnostics per
profile: 8,512 new RBF groups / 33 diagnostics and 189 retained isotonic/kernel
algebra groups. Weights, offsets and gamma compare elementwise exactly, with
matching NaN/Inf semantics. All 522 source nonfinite values remain covered.
Both facade examples match source gamma, weights, offsets and mapped values.

The combined audit verifies 164 compiled source units per profile, complete
eight-field inventories for 1,064 new fixtures, stable source/executable/
validator/report hashes and matching checked/release inventories. Source CSR
preservation, seeded refits and global-RNG isolation identities are observed
alongside native counterparts. Three structural diagnostics use SciPy
check_format; 30 use sklearn estimator validation. The unchanged corpus and
tolerances passed after the phase fix; no failing case was removed or relaxed.
Only the two named affected reference suites and their diagnostics were rerun.
Unrelated source/test/application paths remained unchanged. Four-job builds
used the checksum-verified private fpm 0.14.0; the global fpm was unchanged.
Artifacts remain under ignored validation/results/rbf_features*, including
rbf_features_final_audit.json. Implementation commits: ad9019e, 1e49021 and
cf8d7b7. IEEE_FMA compiler support is documented in API.md.

## Source-seeded skewed chi-square features

SkewedChi2Sampler now uses NumPy-compatible scalar-seeded row-major uniforms,
source inverse-CDF operation order and subsequent phase offsets. The transform
uses rounded X+skewedness followed by log, preserving source overflow outcomes
rather than stabilizing the sum. Source-valid finite skew controls, transform
domains, fit-only negative data, live component/shift mutation, seeded refits
and intrinsic-RNG/input preservation are covered. RBF and skewed samplers share
the existing controlled dense Fourier dot implementation. Skewed input remains
dense, matching sklearn; no public type, dependency or extension was added.

The new direct-source suite contains 216 fixtures / 1,512 groups and 19 invalid
cases across five seeds, multiple shapes, positive/zero/negative shifts,
near-boundary/tiny/large/overflow inputs, mutations and fit-only domain cases.
Weights and offsets compare exactly; mapped values use numerical tolerances.
All 275 source nonfinite values remain compared. Temporary checked and optimized
probes pass the new suite, retained RBF suite and upgraded legacy chi-square
suite. The legacy suite's 204 groups now use real scalar-seeded source skewed
fits; its injected-draw shim and driver draw exports were removed. The additive
map implementation is unchanged. Native finite-control guards remain.

Inventory: 222 public core types, 154 source modules, 206 native programs,
182 reference suites / 3,697,318 retained groups and 87 invalid registries /
4,750 cases. MIT licensing and retained sklearn/NumPy BSD attribution continue.
Canonical verification is recorded below.

Final skewed-feature verification: checked and optimized release builds each
passed test_seeded_skewed_features, test_chi2_samplers, test_rbf_features and
test_kernel_approximation. These four native programs were executed; the other
202 native programs were not run in this batch. Three affected reference suites
each passed 10,228 groups and 52 diagnostics per profile: 1,512 new skewed groups /
19 diagnostics, 204 upgraded direct-source chi-square groups, and 8,512 retained
RBF groups / 33 diagnostics. The new suite retains all 275 nonfinite source
values; the RBF suite retains its 522. Both examples match source weights,
offsets, mapped values and live-normalization changes.

The combined audit verifies 166 compiled source units per profile, complete
seven-field inventories for 216 new fixtures, exact weights/offsets, stable
source/executable/validator/report hashes and matching checked/release
inventories. Shared Fourier helper changes preserve the existing RBF corpus,
including fused finite phases. Only the three named affected suites and their
diagnostics were rerun. The module's unrelated algorithm bodies remain unchanged;
only the named chi-square reference driver and new driver/test were changed in
application/test sources. Four-job builds used the checksum-verified private
fpm 0.14.0; the global fpm was unchanged. Artifacts remain under ignored
validation/results/skewed_features*, including skewed_features_final_audit.json.
Implementation commit: 1352c70. MIT licensing and retained BSD notices continue.

## Source-seeded Nystroem kernels and full SVD

Replaced intrinsic landmark sampling with NumPy-compatible permutations and
PSD eigendecomposition with a complete real Jacobi SVD adapted from MIT-licensed
SciFort. The source floor max(S,1e-12) follows full singular-vector construction;
null directions are completed and small nonzero values are retained. Added
sigmoid, laplacian, chi2, additive_chi2, typed callback and precomputed support,
alongside RBF/linear/poly/cosine. Kernels and parameters are live at query time.
Negative finite offsets are accepted; integer degrees are positive. The old
negative-offset invalid case was replaced with zero degree, preserving that
registry's count. Source precomputed row-only selection/shape behavior is
explicit, including rejection of smaller-than-sample landmark counts.

The new suite covers 408 fixtures, 5,760 required groups and 30 invalid cases.
152 regular cases compare normalization and feature coordinates directly;
256 singular/ill-conditioned cases compare singular values, normalization Gram
and realized train/query/cross kernel Grams. All cases verify SVD reconstruction
and both factor orthogonalities. Required inventories contain no unchecked
coordinate fields. The 5,760 groups include 4,944 direct/source-invariant groups
and 816 SVD certificates. Cases span five seeds, all kernels, singular/zero/
constant bases, negative coefficients, live parameters, callback asymmetry,
nonsymmetric precomputed input and widely separated nonzero singular values.
Checked and optimized temporary probes pass all affected reference suites.

SciFort's copyright and MIT license are retained; sklearn/NumPy BSD notices
continue. No fpm dependency or public extension algorithm was added. Dense
estimator inputs and integer degree are native API subsets. Inventory:
222 public core types, 155 source modules, 207 native programs, 183 reference
suites / 3,703,078 retained groups, and 88 invalid registries / 4,780 cases.
Canonical verification is recorded below.

Final Nystroem verification: checked and optimized release builds each passed
test_nystroem_svd, test_kernel_approximation, test_rbf_features,
test_seeded_skewed_features and test_chi2_samplers. These five native programs
were run; the other 202 native programs were not executed in this batch.
Five affected reference suites each passed 16,177 required groups: 5,760 new
Nystroem groups, 189 isotonic/kernel groups, 8,512 RBF groups, 1,512 skewed groups
and 204 direct chi-square groups. Diagnostics total 88 per profile: 30 new
Nystroem, 33 RBF, 19 skewed and six retained legacy Nystroem diagnostics. The
new square-precomputed/smaller-landmark rejection is observed in both profiles.

The combined audit verifies 171 compiled source units per profile, conditional
complete required inventories, source/executable/validator/report hashes,
matching profile inventories, SciFort source provenance and retained MIT
license/copyright. New required groups remain separated into 2,128 direct
regular-case comparisons, 2,816 singular-case source invariants and 816 SVD
certificates. Singular coordinates are explicitly not counted as direct
comparisons. Both examples match source landmark indices, normalization and
mapped values, including live gamma. Unrelated source/test/application paths
remain unchanged apart from the named new helper, driver/test and existing
legacy invalid driver. Only the five affected suites and reported diagnostics
were rerun. Four-job builds used the checksum-verified private fpm 0.14.0;
the global fpm was unchanged. Artifacts remain under ignored
validation/results/nystroem*, including nystroem_final_audit.json.
Implementation commit: d560ee6. MIT licensing and retained BSD notices continue.

## Dense and CSR additive chi-square feature maps

AdditiveChi2Sampler now accepts CSR fit/transform/fit_transform and returns a
CSR harmonic map, preserving raw stored-entry order, duplicates and zeros.
Stateless transforms keep width unfitted; fitted maps validate query width.
Both dense and sparse branches follow source intermediate arithmetic, replacing
the stabilized dense expression/zero-interval shortcut with quiet IEEE source
outcomes. Explicit CSR zeros retain NaN harmonic entries, unlike masked dense
zeros. CSR shape and stored counts use checked int64 totals before allocation.

The new suite covers 860 fixtures / 9,460 required groups and 39 diagnostics:
34 sklearn estimator checks, three SciPy CSR structural checks and two native
index-limit guards. Dense/CSR fit/query combinations, stateless/fitted maps,
default/custom/live intervals, duplicates, unsorted/explicit zeros, empty rows,
tiny/huge values and input/RNG preservation are checked. All 14,652 nonfinite
source values are compared. Raw CSR pointers/columns and controls compare
exactly; data use numerical tolerances. Native tests include a 100,000-by-10,000
sparse input whose 100,000-by-30,000 output would require 24 GB if densified.
Checked and optimized temporary probes pass all affected suites.

Inventory: 222 public core types, 155 source modules, 208 native programs,
184 reference suites / 3,712,538 retained groups and 89 invalid registries /
4,819 cases. No dependency, public type or extension algorithm was added.
MIT licensing and retained BSD attribution continue. Canonical verification
is recorded below.

Final additive-CSR verification: checked and optimized release builds each
passed test_additive_csr, test_nystroem_svd, test_kernel_approximation,
test_rbf_features, test_seeded_skewed_features and test_chi2_samplers. These six
native programs were executed; the other 202 native programs were not run in
this batch. Six affected suites each passed 25,637 groups per profile:
9,460 new additive groups, 5,760 Nystroem, 189 isotonic/kernel, 8,512 RBF,
1,512 skewed and 204 direct chi-square groups. All 14,652 new nonfinite source
values remain compared. Diagnostics total 127: 39 new cases, 30 Nystroem,
33 RBF, 19 skewed and six retained legacy Nystroem checks. The new registry
separates 34 sklearn checks, three SciPy structural checks and two native
CSR index-limit guards; Source numeric kernels are not invoked for the latter.

The combined audit verifies 173 compiled source units per profile, complete
11-field inventories for 860 new fixtures, exact CSR layouts/controls and
input/RNG identities, stable source/executable/validator/report hashes and
matching profile inventories. Both examples match source shape, pointer/index
arrays and stored values. Large sparse input/output tests passed in both
profiles. Existing Nystroem singular-case invariants and SVD certificates remain
separately classified. Only the six affected suites and reported diagnostics
were rerun. Unrelated source/test/application paths remained unchanged.
Four-job builds used the checksum-verified private fpm 0.14.0; the global fpm
was unchanged. Artifacts remain under ignored validation/results/additive_csr*,
including additive_csr_final_audit.json. Implementation commit: 8492c2f.
MIT licensing and retained BSD notices continue; no dependency was added.

## Nystroem CSR landmarks and mixed queries

Nystroem now accepts csr_matrix in fit, transform and fit_transform. CSR fits
retain selected rows in components_csr_; dense fits retain components_. Refit
clears the inactive representation. Both query representations produce dense
feature maps, using sparse products without copying the full input or landmark
matrix to dense storage. Callback kernels bridge one feature row at a time to
the existing real64 rank-one callback interface.

Sparse-supported kernels are rbf, linear, poly/polynomial, cosine, laplacian,
sigmoid and callbacks. Source chi2/additive_chi2 kernels require dense inputs.
Precomputed CSR fitting is rejected by the source SVD; source dense-precomputed
fit followed by CSR query yields an object array, which this numeric Fortran API
explicitly rejects. Dense precomputed behavior remains available.

Raw CSR duplicates, unsorted entries and explicit zeros are retained for dot
products and RBF/cosine norms, following source representation-sensitive
arithmetic. CSR cosine normalization treats only exact zero norms as zero;
the dense branch retains the source small-norm threshold. Laplacian evaluation
canonicalizes copies of caller inputs and persistently canonicalizes fitted
CSR landmarks. Caller-owned CSR arrays remain unchanged; sklearn/SciPy can
mutate caller CSR arrays in this path. Source unused trailing storage after
canonicalization is excluded from landmark comparisons using indptr(end).
Transform therefore takes the model with intent(inout), including dense
queries against a sparse basis; subsequent live kernel changes see the
canonicalized fitted basis.

The reference suite covers 870 fixtures / 15,018 required groups: five seeds,
nine kernel spellings/callbacks, three dense/CSR fit-query combinations,
canonical/unsorted/duplicate storage, zero/tiny values, capped components and
live gamma/kernel changes. The 366 regular bases compare feature coordinates
directly. Singular bases compare normalization and feature Grams, with separate
SVD orthogonality/reconstruction certificates. The normalization-Gram absolute
roundoff bound is 512*n*epsilon*max(1, spectral_norm(expected)); this handles
cancellation in the source 1e12 null-space projector. Other numerical fields
use rtol=atol=3e-8. Raw active CSR storage, landmarks and metadata compare
exactly. Native input/RNG preservation and deterministic refit are explicitly
classified separately from source comparisons. Source input mutations are
recorded rather than asserted to be absent.

The 29 diagnostics comprise 25 sklearn estimator errors, three SciPy CSR
structural errors and one explicit numeric-output guard for source object
arrays. Native tests include dense/CSR refits and a 100,000-by-100,000 CSR fit
whose dense input would require 80 GB. Inventory: 222 public core types,
155 source modules, 209 native programs, 185 reference suites / 3,727,556
retained groups, and 90 invalid registries / 4,848 cases. No public type,
extension algorithm or dependency was added. MIT licensing and retained BSD
attribution continue. Canonical build results are recorded below.

Final Nystroem-CSR verification: checked and optimized release profiles each
passed seven selected native programs: test_nystroem_csr, test_additive_csr,
test_nystroem_svd, test_kernel_approximation, test_rbf_features,
test_seeded_skewed_features and test_chi2_samplers. The other 202 native
programs were not executed in this batch. Each profile passed seven suites /
40,655 required groups: 15,018 new CSR Nystroem, 9,460 additive CSR, 5,760 dense
Nystroem, 189 isotonic/kernel, 8,512 RBF, 1,512 skewed and 204 chi-square groups.
All 156 diagnostics passed: 29 new cases plus 127 retained affected checks.

The new suite separates 5,490 direct source groups, 6,048 singular source
invariants, 1,740 SVD certificate groups, 870 native input/RNG identities and
870 native deterministic-refit identities. Source caller CSR mutation was
observed in 68 fixtures and recorded separately. The native caller inputs
remain unchanged. Callback fixtures use functions of their argument values;
stateful Python callback invocation order is outside the native contract.

The combined audit verified 175 compiled source units per profile, complete
conditional field inventories, stable source/executable/validator/report hashes,
matching checked/release configurations and distinct optimized executables.
The large sparse fit/query native tests passed in both profiles. The retained
additive-CSR example also matched source storage and values in both profiles.
Only the seven reported suites and diagnostics were rerun. Unrelated source,
test and application files remained unchanged. Four-job builds used the
checksum-verified private fpm 0.14.0; the global fpm was unchanged. Artifacts
remain under ignored validation/results/nystroem_csr*, including
nystroem_csr_final_audit.json. Implementation commit: 64046a8. MIT licensing
and retained upstream notices continue; no dependency or extension was added.

## PolynomialCountSketch CSR maps and IEEE Fourier products

PolynomialCountSketch now accepts csr_matrix in fit, transform and
fit_transform; any dense/CSR fit-query combination produces dense features.
Fits share the existing NumPy-compatible seeded hash initialization. Sparse
transforms stably group each row by feature, scale/sign stored entries before
summing duplicates, and add feature columns to hash bins in source order.
Neither the input matrix nor the full count-sketch tensor is densified: output
storage is n_samples*n_components, with a per-row degree*n_components scratch
matrix plus a sorted CSR copy. Input CSR arrays and the intrinsic RNG remain
unchanged. Dense and CSR refits replace both hash arrays and fitted dimensions.

The source live gamma, nonzero coef0, reduced degree and enlarged output-width
paths remain supported, while n_features_out_ retains fitted metadata. Existing
native finite/nonnegative gamma, finite coef0, positive integer controls and
fitted hash dimension/range guards remain in force. Source accepts negative
coef0 and returns NaNs; the native map preserves that result. Integer capacity
guards now check output and per-row scratch sizes before allocation. The old
pcs_transform_overflow diagnostic now identifies the earlier output-size guard.

Finite inputs can overflow count sums or Fourier products. The transform
quietly evaluates those Fourier operations and restores the caller's invalid,
divide-by-zero and overflow flags and halting modes. Complex products use the
installed NumPy 2.2.6 contiguous fused arithmetic. Inverse scaling separates
real and imaginary components; radix-2 unity/quarter rotations avoid spurious
0*Inf contamination. Small prime transforms through length 31 use paired
butterflies adapted to Fortran from the pocketfft revision shipped with SciPy
1.15.3, with retained BSD licensing. Larger lengths retain the existing
SciFort-derived radix-2/Bluestein path. No C++ or runtime dependency was added.

The new reference suite covers 2,944 fixtures / 26,496 required groups and
98 diagnostics: 92 sklearn estimator errors, three SciPy CSR structural checks
and three native integer-capacity guards. It checks five seeds, all four
fit/query representations, canonical/unsorted/duplicate/cancelling storage,
empty stored rows, constant/zero/tiny/huge maps, live controls, power-of-two and
prime lengths, refits, and all 63,736 nonfinite source values. Component counts
include 1, 2, 3, 4, 5, 7, 8, 11, 13, 16, 17, 19, 23, 29, 31, 67, 128, 257
and 1009; nonfinite fixtures cover degrees one and two at counts through 31. Active hashes,
metadata, input identities and refit hashes compare exactly; finite features
use rtol=atol=3e-8, with exact NaN/signed-infinity masks. Native IEEE environment
checks are classified separately from sklearn comparisons. Native tests also
compare small-prime FFTs with an independent direct DFT and fit a
100,000-by-100,000 sparse matrix whose dense input would require 80 GB.

Inventory: 222 public core types, 155 source modules, 210 native programs,
186 reference suites / 3,754,052 retained groups and 91 invalid registries /
4,946 cases. No public extension algorithm was added. Project integration
remains MIT licensed; the new pocketfft BSD notice joins retained sklearn,
NumPy and SciFort notices. Canonical profile verification is recorded below.

Final TensorSketch-CSR verification: checked and optimized release builds each
passed nine selected native programs: test_polynomial_csr,
test_polynomial_count_sketch, test_nystroem_csr, test_additive_csr,
test_nystroem_svd, test_kernel_approximation, test_rbf_features,
test_seeded_skewed_features and test_chi2_samplers. The other 201 native
programs were not executed in this batch. Each profile passed nine reference
suites / 70,101 required groups: 26,496 new TensorSketch CSR, 2,950 retained
dense TensorSketch, 15,018 CSR Nystroem, 9,460 additive CSR, 5,760 dense Nystroem,
189 isotonic/kernel, 8,512 RBF, 1,512 skewed and 204 chi-square groups.
All 290 diagnostics passed: 98 new cases, 36 retained TensorSketch cases and
156 retained affected kernel-map checks. The pcs_transform_overflow case now
checks the earlier output-capacity diagnostic; no legacy case was removed.

The new suite retained complete nine-field inventories for all 2,944 fixtures:
23,552 sklearn groups and 2,944 separately classified native IEEE-environment
groups. All 63,736 nonfinite source values were checked in each profile,
including exact NaN/signed-infinity masks. Fitted hashes, cached output metadata,
input/RNG identities and opposite-representation refit hashes remained exact.
Prime FFT direct-DFT/round-trip checks and the large sparse fit/query passed in
both profiles. The new CSR/dense-query example and retained additive-CSR example
both matched their source outputs.

The combined audit verified 180 compiled source units per profile, stable
source/executable/validator/report hashes, matching profile configurations,
field inventories and diagnostic registries, distinct optimized drivers and
the pinned pocketfft source/license notice. Only the nine reported suites and
diagnostics were rerun. Unrelated source/test/application paths remained
unchanged. Four-job builds used the checksum-verified private fpm 0.14.0;
the global fpm was unchanged. Artifacts remain under ignored
validation/results/polynomial_csr*, including polynomial_csr_final_audit.json.
Implementation commit: 08ce53c. Project integration remains MIT licensed with
retained BSD/SciFort notices; no runtime dependency or extension was added.

## MaxAbsScaler streaming, CSR and NaN handling

MaxAbsScaler now supports dense and CSR fit, partial_fit, fit_transform,
transform and inverse_transform, with max_abs_, scale_, n_features_in_ and
int64 n_samples_seen_. Full fit resets learned state; partial_fit accepts
mixed representations, checks fitted width and accumulates the sample count.
NaNs are excluded from batch extrema and retained in maps. As in sklearn
1.6.1, an all-NaN batch statistic propagates through subsequent partial-fit
maximum updates; a full refit clears that state. Implicit sparse zeros count
as observed zeros, including columns otherwise containing only stored NaNs.

Sparse fitting reduces coalesced feature coordinates in an owned CSR copy;
transforms preserve caller storage order, duplicate entries and explicit zeros.
Source sparse maps multiply by reciprocal scale; dense maps divide directly.
Inverse mapping uses source multiplication, including quiet 0*Inf outcomes.
Dense inverse accepts the source length-one learned-scale broadcast; it cannot
expand a one-column input to multiple fitted columns. CSR inverse requires
matching width. Infinite inputs are rejected; finite duplicate sums can produce
infinite learned scales, which retain their source IEEE behavior.

The new maxabs_scale function accepts dense matrices, rank-one vectors and CSR,
with axis zero or one. Sparse row scaling canonicalizes an owned copy, matching
the source CSC-transpose reduction's active output storage while preserving
caller arrays. Source/SciPy can mutate caller CSR storage through that transpose;
these source mutations are recorded separately. Dense/CSR inputs are never
converted to a full dense input or output matrix along the sparse path.
The native count has an explicit int64 overflow guard. In-place copy flags,
CSC containers, feature names and changes to other scaler classes are outside
this batch; existing composition wrappers keep their own validation rules.

The direct suite requires 55,644 groups across 2,932 fixtures, covering all
fit/partial/query representation combinations, both axes, one-dimensional
functions, NaNs and all-NaN streaming batches, implicit/explicit zeros,
unsorted/duplicate/cancelling storage, tiny/threshold/huge values, learned-scale
mutation, inverse broadcasting and refit resets. Numeric fields use 3e-8
absolute/relative tolerances with exact NaN/signed-infinity masks. Layouts,
metadata and preservation identities are exact. Native caller/RNG identities
are classified separately from source comparisons. All 14,272 nonfinite source
values are compared. The 101 diagnostics comprise 97 sklearn checks, three
SciPy structural errors and one native int64 counter guard.

The focused sklearn_maxabs_scaling module is re-exported by preprocessing and
the main facade. Inventory: 222 public core types, 156 source modules, 212
native programs, 187 reference suites / 3,809,696 retained groups and 92 invalid
registries / 5,047 cases. No dependency or public extension was added. MIT
integration and retained sklearn BSD notices continue. Canonical profile
verification is recorded below.

Final MaxAbs verification: checked and optimized release profiles each passed
15 selected native programs: test_maxabs_streaming, test_maxabs_integration,
test_preprocessing_metrics, test_stateless_preprocessing,
test_stateless_composition, test_pipeline, test_pipeline_supervised,
test_frozen, test_transformed_target, test_column_transformer,
test_feature_union, test_function_transformer, test_positive_composition,
test_polynomial_composition and test_target_encoding. The other 197 native
programs were not executed in this batch. Dedicated tests verify MaxAbs type
dispatch through Pipeline/FrozenEstimator, inverse delegation, feature width,
prototype preservation and the large sparse fit/query.

Each profile passed seven comparison runs / 81,306 groups: 55,644 new MaxAbs,
19,386 retained Normalizer/Binarizer, 5,328 stateless composition, 192 core,
108 MaxAbs column-transformer, 60 MaxAbs feature-union and 588 MaxAbs
transformed-target groups. The latter three are explicitly MaxAbs-only slices
of the original generators: 18, 12 and 84 fixtures, preserving their original
random-stream advancement. They are not full reruns of those integration suites.
All 130 diagnostics passed: 101 new cases and 29 retained stateless checks.
All 14,272 new nonfinite source values remain compared. Source free-row-scale
caller mutation occurred in 488 fixtures and is recorded separately from the
native immutable-input/RNG identity checks.

The combined audit verified 184 compiled source units per profile, complete
conditional field inventories for 2,932 new fixtures, exact CSR layouts and
metadata, stable source/executable/validator/report hashes and matching profile
configuration/check inventories. Optimized driver hashes differ from checked
drivers. Both example outputs matched source. It also verifies byte-identical
other preprocessing procedure bodies after removing the old MaxAbs type/methods
and adding the focused-module imports/exports. No unrelated source/test/app path
changed. Four-job builds used the checksum-verified private fpm 0.14.0; the
global fpm was unchanged. Artifacts remain under ignored validation/results/
maxabs*, including maxabs_final_audit.json. Implementation commit: 8c0157d.
MIT integration and retained upstream notices continue; no dependency or public
extension algorithm was added.

## MinMaxScaler streaming, NaNs and live clip bounds

MinMaxScaler now provides dense partial_fit and fit_transform, fitted
n_features_in_/int64 n_samples_seen_, and data_range_ alongside its existing
extrema, scale and offset. Full fit clears learned state; partial_fit combines
batch extrema and counts. NaNs are excluded from batch reductions and retained
in maps. Source all-NaN batch propagation through incremental minimum/maximum
updates is preserved. A full refit clears that history. Sparse inputs remain
unsupported, as in scikit-learn.

Mapping follows the source's separate multiply/add and subtract/divide steps
with quiet IEEE intermediate results. Finite ranges and endpoint spans can
overflow; their source infinities/NaNs are preserved with floating traps
enabled. Equal or reversed endpoints are rejected at fit/partial_fit; source
NaN/infinite endpoints are accepted. Query clip and range controls are live,
while scale/offset remain fitted. Clipping uses minimum(maximum(x,lower),upper),
including source behavior for post-fit reversed/NaN bounds. NaN mapped values
remain NaN. Inverse mapping supports a learned length-one scale/offset against
wider input, following source in-place broadcasting; input width cannot expand.

The new minmax_scale function supports dense matrices and rank-one vectors on
axis zero or one, with native optional feature_min/feature_max endpoints. Caller
arrays and intrinsic RNG state are preserved. In-place copy flags, sparse
containers and feature-name methods are outside the direct interface. Existing
composition wrappers retain their own validation rules. The focused
sklearn_minmax_scaling module is re-exported through preprocessing and the main
facade; other preprocessing procedure bodies remain unchanged.

The suite checks 596 fixtures / 9,532 required groups and 16 diagnostics (15
sklearn checks plus one native int64 counter-capacity guard). Fixtures include
five dataset seeds, three widths, streaming/refits, both free-function axes,
NaNs/all-NaN batches, constants, tiny/threshold/huge values, finite/nonfinite
endpoint ranges, live/reversed clip bounds, learned-scale mutation and inverse
broadcasting. All 8,544 nonfinite source values are compared. Numerical fields
use rtol=atol=3e-8 with exact NaN/signed-infinity masks; metadata and caller/RNG
identity checks are exact and identities are separately classified.

Inventory: 222 public core types, 157 source modules, 214 native programs,
188 reference suites / 3,819,228 retained groups and 93 invalid registries /
5,063 cases. No dependency or public extension was added. MIT integration and
retained sklearn BSD attribution continue. Canonical verification follows below.

Final MinMax verification: checked and optimized release profiles each passed
17 selected native programs: test_minmax_streaming, test_minmax_integration,
test_maxabs_streaming, test_maxabs_integration, test_preprocessing_metrics,
test_stateless_preprocessing, test_stateless_composition, test_pipeline,
test_pipeline_supervised, test_frozen, test_transformed_target,
test_column_transformer, test_feature_union, test_function_transformer,
test_positive_composition, test_polynomial_composition and test_target_encoding.
The other 197 native programs were not executed in this batch. Dedicated tests
verify MinMax/MaxAbs type dispatch through Pipeline/FrozenEstimator, inverse
maps, feature width and prototype preservation.

Each profile passed nine comparison runs / 93,628 groups: 9,532 new MinMax,
55,644 MaxAbs, 19,386 Normalizer/Binarizer, 5,328 stateless composition, 192 core,
216 column-transformer, 120 feature-union, 1,176 transformed-target and 2,034
pipeline groups. The last four are explicit integration slices: 36/24 column/
union fixtures with MinMax or MaxAbs, 168 target-transform fixtures with those
scalers, and 180 pipeline fixtures with MinMax stages (kinds 2, 8 and 11).
Original generator random streams are preserved; these are not full reruns of
the original integration suites. All 146 diagnostics passed: 16 new MinMax,
101 MaxAbs and 29 retained stateless checks. All 8,544 new nonfinite source
values remain compared, along with retained suites' nonfinite inventories.

The combined audit verifies 191 compiled source units per profile, complete
conditional inventories for 596 new fixtures, exact metadata and caller/RNG
identities, stable source/executable/validator/report hashes and matching
profile configuration/check inventories. New reports use strict JSON with
explicit string tags for NaN/infinite range endpoints. Both scaler examples
match source. The audit proves unchanged other preprocessing procedure bodies
after the focused MinMax extraction/imports/exports, and unchanged unrelated
source/test/application paths. Optimized driver hashes differ from checked
binaries. Four-job builds used the checksum-verified private fpm 0.14.0;
the global fpm was unchanged. Artifacts remain under ignored validation/results/
minmax*, including minmax_final_audit.json. Implementation commit: 7f5c285.
MIT integration and retained upstream notices continue; no dependency or public
extension algorithm was added.

## RobustScaler nullable statistics, NaNs, CSR and unit variance

RobustScaler now fits and maps dense matrices and native CSR, with NaN-aware
medians/linear quantiles, source-equal quantile endpoints, unit_variance,
fit_transform and n_features_in_. Disabled center_/scale_ statistics are
unallocated, representing source None. Dense transforms use live centering and
scaling flags and reject absent required fitted statistics. CSR fitting rejects
centering, while CSR queries/inverses ignore the centering flag as source does.
Raw CSR order, duplicates and explicit zeros are preserved during mapping.
Dense inverse permits learned scalar statistics to broadcast; disabled operations
allow changing inverse width. Sparse inverse checks width only when scaling.

Sparse fitting follows sklearn's raw CSC-column contract: stored entries form
the quantile sample and padding zeros fill the remaining row count. Duplicate
coordinates are not coalesced, and a raw column count above n_rows is rejected
by source. The native implementation groups stored values and represents padding
zeros by multiplicity, avoiding a dense column or full matrix allocation.
Free row scaling applies the same contract to raw CSR rows. The robust_scale
function also accepts dense matrices/rank-one vectors and both axes, with native
centering/scaling, quantile endpoint and unit-variance controls.

Median behavior follows the installed NumPy 2.2.6 dispatch: short 2-D axes below
600 rows average two middle values even for odd valid counts; larger axes use
1-D medians. Extreme finite median/quantile overflow and source zero-IQR/unit
adjustment NaNs/infinities are preserved with floating traps enabled. Quantile
interpolation uses source lower/upper-half operation order. Normal-reference
quantiles are computed natively by monotone log-tail inversion, including
subnormal probabilities and endpoint infinities, without a SciPy runtime.

The direct suite covers 1,332 fixtures / 31,944 required groups and 69 diagnostics:
65 sklearn checks, three SciPy CSR structural checks and one native integer
capacity guard. Cases cover three seeds, dense/CSR fit/query combinations,
optional statistics, both free-function axes, default/custom/equal/endpoint and
tail quantiles, missing/all-missing/zero/tiny/huge data, raw duplicates, live
flags, inverse broadcasting, refit replacement and the 599/600/601-row median
boundary. All 32,575 nonfinite source values are compared. Numerical fields use
3e-8 tolerances with exact NaN/signed-infinity masks; CSR layouts, metadata and
caller/RNG identities are exact, with identities classified separately.

The focused sklearn_robust_scaling module remains exported by preprocessing and
the facade. Frozen fitted-state/feature-width adapters now use robust metadata
so models without a scale vector remain supported. Other preprocessing methods
are unchanged. Native tests fit a 100,000-by-100,000 sparse matrix whose dense
storage would require 80 GB. No sparse centering, in-place copy flags or CSC
container API is introduced. Inventory: 222 public core types, 158 source modules,
216 native programs, 189 reference suites / 3,851,172 retained groups and 94
invalid registries / 5,132 cases. No dependency or public extension was added.
MIT integration and retained upstream notices continue. Canonical verification
is recorded below.

Final RobustScaler verification: checked and optimized release profiles each
passed 19 selected native programs: test_robust_scaling, test_robust_integration,
test_minmax_streaming, test_minmax_integration, test_maxabs_streaming,
test_maxabs_integration, test_preprocessing_metrics, test_stateless_preprocessing,
test_stateless_composition, test_pipeline, test_pipeline_supervised, test_frozen,
test_transformed_target, test_column_transformer, test_feature_union,
test_function_transformer, test_positive_composition,
test_polynomial_composition and test_target_encoding. The other 197 native
programs were not executed in this batch. Dedicated tests cover nullable robust
statistics through Pipeline/FrozenEstimator and the large sparse fit/map.

Each profile passed 11 comparison runs / 152,278 required groups: 31,944 new
robust, 25,272 full FrozenEstimator contracts, 9,532 MinMax, 55,644 MaxAbs,
19,386 Normalizer/Binarizer, 5,328 stateless composition, 192 core, 324 column-
transformer, 180 feature-union, 1,764 transformed-target and 2,712 pipeline groups.
The last four are explicit affected-scaler slices: 54/36 column/union fixtures
with MinMax/MaxAbs/Robust, 252 target-transform fixtures with those scalers, and
240 pipeline fixtures containing MinMax/Robust stages (kinds 2, 8, 10 and 11).
Original generator streams remain intact; these are not full integration-suite
reruns. All 227 diagnostics passed: 69 new robust, 12 FrozenEstimator, 16 MinMax,
101 MaxAbs and 29 stateless cases. All 32,575 new nonfinite source values remain
compared, alongside retained suites' nonfinite checks.

FrozenEstimator verifies 486 configurations / 1,944 stages. Its 15,552 underlying
base endpoint diagnostics remain separate from required wrapper contract groups.
Thirty-five logistic-regression endpoint differences remain (19 log-probability,
13 decision and three probability groups; no nonfinite-mask mismatches). The
largest absolute difference is about 2.95e-4 on these diagnostic endpoints.
They are not counted as contract passes and no logistic-regression code changed.

The combined audit verifies 197 compiled source units per profile, complete
conditional inventories for 1,332 new fixtures, exact CSR layouts/metadata and
caller/RNG identities, stable source/executable/validator/report hashes and
matching profile inventories. All three scaler examples match source. Other
preprocessing bodies remain byte-identical after robust extraction/imports;
frozen changes are exactly the two fitted-state/width clauses. No unrelated
source/test/application path changed. Optimized driver hashes differ from
checked binaries. Four-job builds used the checksum-verified private fpm 0.14.0;
the global fpm was unchanged. Artifacts remain under ignored validation/results/
robust*, including robust_final_audit.json. Implementation commit: 9164c4e.
MIT integration and retained notices continue; no dependency or public extension
algorithm was added. The broader translation goal remains unfinished.


## Verified weighted multioutput variance scores (2026-11-24)

Implementation checkpoint cf2fa4e passed the full checked and optimized batches.
Each profile retained 16,398 new reference/safety groups over 8,120 complete
configurations; 2,966 source rejections produced controlled diagnostics.
The source returned 3,085 nonfinite score values; all corresponding
NaN/infinity masks matched. Maximum finite absolute score difference was
0, within the 3e-12 absolute/relative comparison tolerance.
The suite covers all four explicit vector/matrix rank pairs, weighted and
multioutput paths, custom signed weights, constants, singleton early return,
nonfinite sample weights, overflow/subnormals and 8192-buffer boundaries.

Each profile also passed 53 selected native regression/model-selection programs,
192 retained core reference groups, 20 new diagnostics and the existing
EllipticEnvelope score-shape guard. Both examples ran; the variance example
matches sklearn's R2 0.0625, explained variance 1 and variance-weighted R2 0.4.
The combined audit checks 232 dependency-aware compiled units
per profile, complete matching configuration inventories, frozen input/source/
executable/validator/report hashes and different optimized driver binaries.
All other sklearn_metrics bodies match c67eff8 after removing the two superseded
primitive functions and adding imports/exports. No other prior source body was
changed. Artifacts remain under ignored validation/results/variance*.

Four-job --tests builds used the checksum-verified private fpm 0.14.0, GNU
Fortran 17.0.0 20260510, checked flags -fcheck=all -fbacktrace
-ffpe-trap=invalid,zero,overflow, and --profile release. The global fpm was
unchanged. References use sklearn 1.6.1 / NumPy 2.2.6 / Python 3.13.3.
MIT integration and retained BSD notices continue. No dependency, new core
type or public extension algorithm was added. The broader goal is unfinished.


## Verified weighted regression error family (2026-11-24)

Implementation checkpoint caf2b29 passed the full checked and optimized
batches. Each profile passed 14,309 retained value/identity/IEEE groups over
4,688 complete configurations, with 273 controlled source rejections. All
finite score differences were zero and all 2,030 nonfinite score masks matched.
The independently regenerated source inventory agrees with every configuration
ID, input hash, shape, rank pair, metric/mode and conditional result inventory.
Successful cases run in a batched native process; expected errors remain
isolated. Both profiles have matching full inventories and distinct binaries.

Each profile also passed 54 selected native regression/model-selection programs,
192 retained core groups, 77 targeted diagnostics and the existing
EllipticEnvelope score-shape guard. The targeted registry contains 73 source
rejections and four explicit native static raw-result adapter guards. Both
examples ran. The error example independently matches sklearn: MSE/MAE 1/6,
RMSE 1/(2*sqrt(3)), MAPE 1/12 and raw RMSE [1/sqrt(3),0].

The combined audit verifies 233 dependency-aware compiled-unit fingerprints per
profile, source/executable/validator/report hash stability and preservation of
all other sklearn_metrics bodies relative to 916d3b7. No other prior library
source changed. Artifacts remain under ignored validation/results/error*,
including error_final_audit.json. Four-job --tests builds used the verified
private fpm 0.14.0, checked flags -fcheck=all -fbacktrace
-ffpe-trap=invalid,zero,overflow, and --profile release. The global fpm was
unchanged. References use GNU Fortran 17.0.0 20260510, Python 3.13.3,
sklearn 1.6.1 and NumPy 2.2.6 with the 8192-element buffer and C-contiguous
float64 target matrices. Other layouts/settings may round differently.
Original integration remains MIT with retained sklearn/NumPy BSD notices.
No dependency, new core type or public extension algorithm was introduced.
The broader translation goal remains unfinished.


## Verified weighted median and column maximum errors (2026-11-24)

Implementation checkpoint ad2770f passed full checked and optimized
batches. Each profile retained 18,011 value/identity/IEEE groups over 5,648
complete configurations, with 15 controlled source rejections. All finite
differences were zero and all 341 NaN/positive-infinity/negative-infinity masks
matched. The independently regenerated source inventory agrees with each
configuration ID, data hash, rank, shape, metric/mode and conditional field.
Fixtures cover lower versus ordinary medians, signed/zero/zero-total CDFs,
overflowed CDFs, ties, huge odd/even medians, long arrays, custom output weights,
signed infinite aggregation and column maximum errors. Accepted cases run in
one native batch; expected source errors remain isolated.

Each profile also passed 56 selected native programs, including prior error/
variance metrics and the unchanged NumPy-ordering helper, 192 core groups,
28 targeted guards and the retained EllipticEnvelope score-shape guard. The
targeted registry has 27 source rejections and one explicit static raw-result
adapter guard. Both examples ran; the new example independently matches sklearn:
median 0, weighted median 1, raw weighted medians [1,0], custom-output median
0.25 and maximum 1. Native tests additionally verify RNG and owned raw arrays.

The combined audit verifies 236 dependency-aware compiled-unit fingerprints per
profile, matching complete inventories, stable source/executable/validator/
report hashes and distinct optimized drivers. All other sklearn_metrics bodies
match 46a6549, and the existing NumPy index-order source is unchanged. Artifacts
remain under ignored validation/results/order*, including order_final_audit.json.
Four-job --tests builds use the checksum-verified private fpm 0.14.0, checked
flags -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow and --profile release.
The global fpm is unchanged. GNU Fortran 17.0.0 20260510 / Python 3.13.3 /
sklearn 1.6.1 / NumPy 2.2.6 are the reference toolchain, with the existing pinned
Intel index-order backend and 8192-element reduction buffer. Source regression,
stats/extmath and native ordering-helper/Intel-license hashes are retained.
Original integration remains MIT with preserved sklearn/NumPy/Intel BSD notices.
No dependency, new core type or public extension algorithm was added. The broad
translation remains unfinished.


## Verified logarithmic errors and direct deviances (2026-11-24)

Implementation checkpoint 7053c3b passed both checked and optimized
fpm build --tests batches. Each profile passed 11,224 new value/identity/IEEE
groups over 3,669 complete configurations, with 45 controlled source rejections.
Finite differences were zero and all 1,741 NaN/positive-infinity/negative-infinity
masks matched. The independently regenerated inventory agrees with every
configuration ID, data hash, shape, rank, metric/mode and conditional result.
Coverage includes valid negative log targets, tiny log1p differences, roots
before aggregation, finite custom-output validation, signed/nonfinite sample
weights, direct deviance domains/columns, extreme ratios and buffered sums.

Each profile also passed 33,120 earlier regression groups over 4,507 complete
fixtures, including RMSLE's existing callers. There are 929 earlier source-
rejection records checked through 254 unique native probes and 26 earlier
targeted diagnostics. The 57 selected native programs, 192 core groups, 79 new
guards and retained EllipticEnvelope score-shape guard passed. Three new guards
are explicit native adapters (two scalar/raw-rank guards and RMSLE's mixed
keyword rejection); 76 correspond to source rejections. Native tests exercise
the legacy named and positional unallocated optional arrays, RNG preservation
and owned raw results. Two examples ran, with the new one independently
matching sklearn's MSLE/RMSLE/raw values and Poisson/Gamma examples.

The combined audit verifies 239 dependency-aware compiled-unit fingerprints per
profile, full matching inventories, stable source/executable/validator/report
hashes and distinct optimized drivers. All other sklearn_metrics bodies match
e3439cd after replacing three primitive functions and adding imports/exports.
All non-RMSLE sklearn_regression_metrics bodies match after import/export and
legacy-wrapper replacement. The general Tweedie/D2/pinball implementations are
unchanged. Artifacts remain under ignored validation/results/distribution*,
including distribution_final_audit.json. Four-job builds use the verified
private fpm 0.14.0, GNU Fortran 17.0.0 20260510, checked flags -fcheck=all
-fbacktrace -ffpe-trap=invalid,zero,overflow and --profile release. References
use Python 3.13.3, sklearn 1.6.1 and NumPy 2.2.6 with C-contiguous real64 arrays
and the default 8192-element buffer. Other math runtimes/layouts/settings can
round differently. The global fpm is unchanged. Original integration remains
MIT with retained sklearn/NumPy BSD notices. No dependency, new core type or
public extension algorithm was added. The broad translation remains unfinished.


## Verified general weighted Tweedie family (2026-11-24)

Implementation checkpoint 33cbe2d passed full checked and optimized
fpm build --tests batches. Each profile retained 23,013 value/identity/IEEE
groups over 7,734 complete configurations, with 63 controlled source rejections.
All finite differences were zero; all 4,186 nonfinite masks matched. The
independently regenerated inventory matches every configuration ID, data hash,
shape, rank, metric/power and conditional result. Fixtures cover ordinary,
near-boundary/wide powers, negative truth, signed/nonfinite weights, extreme
ratios, buffered sums, singleton shortcuts and nonpositive/nonfinite null
models. Incoming inexact flags alternate clear/set and are preserved.

Each profile also passed 33,120 earlier regression groups over 4,507 fixtures,
25 earlier diagnostics, one migrated accepted-infinity regression and 929 source-rejection records exercised via 254
unique native probes. All 58 selected native programs, 192 core groups, 41 new
source-aligned diagnostics and the EllipticEnvelope score-shape guard passed.
Both examples ran; the Tweedie example independently matches sklearn's normal,
Poisson and Gamma D2 scores and weighted compound deviance. Native tests also
check RNG and direct Poisson/Gamma identities. The combined audit verifies
241 dependency-aware compiled-unit fingerprints per profile, matching complete
inventories, stable source/executable/validator/report hashes and distinct
optimized drivers. The metrics facade is unchanged and all other regression
bodies match 7c89fff after replacing the two public vector functions with imports.

Artifacts remain under ignored validation/results/tweedie*, including
tweedie_final_audit.json. Four-job builds use verified private fpm 0.14.0,
GNU Fortran 17.0.0 20260510, checked flags -fcheck=all -fbacktrace
-ffpe-trap=invalid,zero,overflow and --profile release. References use Python
3.13.3, sklearn 1.6.1 and NumPy 2.2.6 with default 8192-element buffered sums.
Other math runtimes/buffer settings may round differently. The global fpm is
unchanged. Original integration remains MIT with preserved BSD attribution.
No dependency, core type or public ML extension was added. The broad translation
remains unfinished.


## Verified pinball and D2 quantile family (2026-11-24)

Implementation checkpoint eee4be4 passed full checked/optimized fpm
build --tests batches. Each profile retained 20,034 value/identity/IEEE groups
over 6,734 complete configurations, with 534 controlled source rejections.
All finite differences were zero; all 1,538 nonfinite masks matched. The
independent source inventory matches every configuration ID, data hash, rank,
shape, metric/mode/alpha and conditional result. Fixtures cover alpha endpoints,
ties, weighted CDF cancellation, signed/nonfinite weights, interpolation and
residual overflow, conditional score masks, singleton shortcuts and output
aggregation. Incoming inexact flags alternate clear/set and are preserved.

Each profile also passed 33,120 earlier regression groups over 4,507 fixtures,
25 earlier diagnostics, the accepted infinite-weight regression and 929 source
rejection records exercised via 254 unique native probes. All 59 selected
native programs, 192 core groups, 77 new diagnostics and the retained score-shape
guard passed. Six new guards are identified native adapters: scalar raw-result
and mixed-keyword checks for three metrics. Native tests also check legacy
optional unallocated arrays, singleton raw shapes, RNG and owned raw results.
Both examples ran; the new example independently matches sklearn: loss 1/6,
both D2 scores .5, custom loss 1/24 and raw losses [1/6,0].

The combined audit verifies 243 dependency-aware compiled-unit fingerprints per
profile, matching complete inventories, stable source/executable/validator/
report hashes and distinct optimized drivers. The metrics facade bodies match
fc7568c after raw imports/exports; the regression facade has exactly the intended
public reexports. Other library/ordering bodies are unchanged. Dead legacy
kernels are retired. Artifacts remain under ignored validation/results/
quantile_loss*, including quantile_loss_final_audit.json. Four-job builds use
verified private fpm 0.14.0, GNU Fortran 17.0.0 20260510, checked flags
-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow and --profile release.
References use Python 3.13.3, sklearn 1.6.1, NumPy 2.2.6, the pinned Intel index
backend, C-contiguous real64 arrays and the default 8192-element sum buffer.
Other sort/math backends or buffer settings can round differently. The global
fpm is unchanged. Original integration remains MIT with preserved BSD notices;
no dependency, core type or public ML extension was introduced. The broader
translation remains unfinished.


## Streaming nearest-pair reductions (2026-11-25, implementation checkpoint)

Added pairwise_distances_argmin and pairwise_distances_argmin_min with owned
1-based index/distance results, both axis directions, all four dense/CSR input
combinations and deterministic row callbacks. Eleven named metrics plus a
callback path stream through two row scratch buffers rather than forming the
pairwise matrix. Existing metrics bodies are unchanged; only public imports
and exports were added. Source contracts include reduced-distance selection,
ordered Gram arithmetic, finite heap sentinel, NaN fallback minima, sparse
normalization and the pinned Haversine CSR row-start coordinate convention.

Private checked probes passed 4,108 accepted configurations / 16,432 retained
groups, 88 source rejection cases, 26 targeted diagnostics and native tests.
The maximum scaled distance error was 4.31e-15. Full package checked/release
verification subsequently passed; final results follow below. Limits and the NaN-power native adapter are documented
in API.md. Original integration remains MIT with retained sklearn BSD notices;
no dependency, public core type or algorithm beyond sklearn was introduced.


## Verified streaming nearest-pair reductions (2026-11-25)

Implementation 4153392 passed full checked and optimized fpm build --tests
batches. In each profile, 4,108 accepted configurations retained 16,432 groups
checking exact nearest indices, distance arrays, nonfinite masks and combined
API agreement/input identity/IEEE state. All 40 source nonfinite masks matched;
the largest scaled finite distance error across both profiles was 4.302e-15
(rtol=atol=3e-12). All 88 source rejection cases and 26 targeted diagnostics
passed; the latter distinguish 25 source rejections from one native NaN-power
adapter guard. References use sklearn 1.6.1, NumPy 2.2.6, real64 C-contiguous
arrays and the default optimized backend with one OpenMP/BLAS thread.

Each profile also passed 16,729 existing pairwise groups: 16,709 strict source
comparisons and 20 retained large-offset roundoff-bound checks. All 26 earlier
pairwise diagnostics, 18 selected native programs and 192 core groups passed.
Other historical suites/native programs were not rerun in this batch. The
nearest-pair example matches an independent sklearn reference. The audit
verifies 201 dependency-aware source fingerprints per profile, stable source/
validator/executable hashes, distinct optimized drivers and complete independent
configuration/result inventories. Existing metric bodies and other library
sources match 3d1bf4f. The new module streams two rows and result vectors;
it does not allocate a pairwise matrix or densify complete CSR inputs.

Pinned source details include axis-zero callback orientation, upper-triangle
self-callback symmetry, reduced-distance selection and finite heap sentinels,
Gram versus l2 arithmetic, dense/CSR cosine normalization, NaN-overlap fallback
minima and the source's Haversine CSR row-start coordinate convention. The last
is explicitly documented because sparse and dense geographic answers can
differ. Supported names, native adapter limits and unexposed SciPy metrics/
per-feature parameters are listed in API.md. No dependency, public core type or
algorithm beyond sklearn was added. Original integration remains MIT with
retained sklearn BSD attribution. Verification artifacts remain ignored under
validation/results/pairwise_reductions*, including the combined final audit.

Inventory: 222 public core types, 175 source modules, 241 native programs,
214 reference suites / 4,180,483 retained groups and 113 invalid registries /
6,177 cases. The broader translation remains unfinished.


## Additional nearest-pair dissimilarities (2026-11-25, in progress)

Added twelve named numeric/Boolean distances to the streaming nearest-pair
APIs, including source optimized/fallback selection, numeric Hamming versus
Boolean matching, source constant/all-false NaNs, centered correlation and
Canberra's sparse stored-position rules. Existing named kernels and wrappers
remain intact. Closed-form native tests and 60 targeted source rejection guards
pass in private checked probes; broad accepted-case verification subsequently passed.

The pinned sparse Bray-Curtis tail loops access the exhausted opposite row
buffer, potentially outside allocated storage. The native CSR path therefore
uses sklearn's dense numerical contract on streamed logical rows. Incomplete
CSR configurations are explicitly marked as adaptations in validation; strict
sparse-source parity is not claimed there. This changes no algorithm scope.
SciPy distance copyright/BSD notices are retained alongside original MIT
integration. Full package checked/release verification subsequently passed, as recorded below.


## Verified additional nearest-pair dissimilarities (2026-11-25)

Implementation 51d8d5e passed full checked and optimized fpm build --tests
batches. Each profile passed 2,682 accepted configurations / 10,728 retained
groups: 2,538 strict sklearn configurations / 10,152 groups and 144 explicitly
identified sparse Bray-Curtis adaptations / 576 groups. The latter compare
logical sparse row values with sklearn's dense numerical contract; strict
parity with the source CSR exhausted-buffer accesses is not claimed. The
pinned source tail loops read from the exhausted opposite row, potentially beyond
allocated storage. The native adaptation streams rows without those reads and
preserves source nonempty-storage restrictions. It adds no new ML algorithm.

All accepted calls check exact nearest indices, distance arrays, nonfinite
masks, agreement of both APIs, input identity and IEEE flags/modes. All 482
reference nonfinite masks matched. The largest scaled finite error across
both profiles was 4.441e-16, at rtol=atol=3e-12. Every one of 3,582 source
rejection records passed, exercised via 588 representative native probes
per profile. Shared guards retain metric/storage/axis/shape/empty-storage/
keyword and incoming-flag parity; the initial uncached probe exercised all
3,582 separately. All 60 new targeted diagnostics are source rejections.

Each profile also passed 16,432 existing nearest-pair groups, their 88 source
rejections and 26 diagnostics; 16,729 earlier pairwise groups (16,709 strict
source comparisons and 20 retained roundoff-bound checks) and 26 diagnostics;
192 core groups and 19 selected native programs. Other historical suites/native
programs were not rerun. Both examples ran; the new example independently
matches sklearn. Source references use sklearn 1.6.1, NumPy 2.2.6 and SciPy
1.15.3, C-contiguous float64 arrays and single-thread OpenMP/BLAS settings.
The audit checks 203 compiled source-unit fingerprints per profile, fresh
configuration/result inventories, source/validator/executable/report hashes,
distinct release drivers and stable SciPy Python/binary oracle provenance.
Existing kernel bodies, public wrappers, facades and other library sources
remain unchanged from cad5935. Detailed artifacts are ignored under
validation/results/extra_dissimilarities*, including the final combined audit.

New names include Bray-Curtis, Canberra, correlation, numeric Hamming, Boolean
matching, Dice, Jaccard, Rogers-Tanimoto, Russell-Rao, Sokal-Michener,
Sokal-Sneath and Yule. Dense/CSR support follows source optimized/fallback
restrictions, apart from the identified sparse Bray-Curtis adaptation.
Per-feature weights and remaining metric parameters are still unexposed.
Original integration remains MIT; applicable sklearn/SciPy BSD terms and the
Damian Eads distance notice are retained. No dependency or public core type
was added. Inventory: 222 public core types, 175 source modules, 242 native
programs, 215 reference suites / 4,191,211 retained groups and 114 invalid
registries / 6,237 cases. The broader translation remains unfinished.


## Parameterized nearest-pair distances (2026-11-25, implementation checkpoint)

Added weighted Minkowski, standardized Euclidean, Mahalanobis and p/infinity
aliases. Typed optional w/V/VI parameters preserve earlier calls. The same-
storage case now follows fallback zero diagonals, including degenerate metrics,
while callbacks retain their own diagonal and fully missing NaN-Euclidean rows
remain NaN. Automatic Mahalanobis self covariance is streamed with ddof=1 and
pivoted inversion; explicit nonsymmetric/signed/nonfinite precision is accepted.
A local +infinity optimized heap after NaN is merged or retained according to
the default one-thread/256-row source strategy, tested at 1024/1025 queries.

Private checked probes passed 3,345 configurations / 13,380 groups, 1,406 source
rejection records via 274 native probes, and 33 targeted diagnostics. All 1,813
nonfinite masks matched; maximum scaled finite error was 4.07e-16. Seven targeted
checks are explicit native adapters: six unsafe-source precision shape guards
and weighted NaN-power rejection. Full package verification subsequently passed, as recorded below.
Integration remains MIT, with preserved sklearn/SciPy/NumPy BSD attribution.
No dependency, public core type or algorithm beyond sklearn was added.


## Verified parameterized nearest-pair distances (2026-11-25)

Implementation 25f596d passed full checked and optimized fpm build --tests
batches. Each profile passed 3,345 complete configurations / 13,380 retained
groups, comparing exact indices, distance/nonfinite arrays, agreement of both
APIs, target/parameter identity and IEEE flags/modes. All 1,813 source nonfinite
masks matched; maximum scaled finite error across both profiles was 4.063e-16
at rtol=atol=3e-12. All 1,406 source rejection records passed via 274 native
probes. Native probes share metric, storage, identity mode, shape, axis,
empty-storage and parameter-shape contracts; automatic self-covariance failures
retain the full data hash. All 33 targeted diagnostics passed: 26 source
rejections and seven identified native adapters. Six precision-shape checks
avoid unchecked source memory layouts, so that reference is not called;
weighted NaN-power rejection is the seventh adapter.

The APIs now accept weighted Minkowski, standardized Euclidean, Mahalanobis
and p/infinity aliases. Optional w/V/VI arguments preserve earlier forms and
unallocated optional-array forwarding. Source distinctions include finite
nonnegative Minkowski weights, general signed/nonfinite variance/precision,
reduced-distance clipping, sparse restrictions and explicit/default precision.
Automatic same-storage Mahalanobis streams ddof=1 covariance with a pivoted
inverse; singular and one-feature source failures are retained. Mahalanobis
adds O(features**2) storage/work per pair, without a full pairwise matrix.

Same-storage fallback calls now preserve the source zero diagonal, even for
degenerate correlation/Dice/Sokal-Sneath/Russell-Rao and overflowing sparse
Euclidean. Fully missing NaN-Euclidean rows remain NaN; off-diagonal NaNs still
win fallback minima. Callbacks retain their own diagonal values. Optimized
NaN-to-infinity heaps follow the pinned one-thread, 256-row auto-strategy merge,
verified on both sides of 1024/1025 queries. Other source thread/chunk/strategy
settings can change exceptional/tied results. Existing unweighted kernel bodies
are unchanged; earlier wrappers only gain optional parameters. Public facades
and other library bodies match c6119f4.

Each profile additionally passed 10,728 earlier dissimilarity groups (including
144 explicitly identified sparse Bray-Curtis adapter configurations), their
3,582 source rejection records and 60 diagnostics; 16,432 earlier nearest-pair
groups, 88 source rejections and 26 diagnostics; 16,729 earlier pairwise groups
(including 20 retained roundoff-bound checks) and 26 diagnostics; 192 core groups
and 20 selected native programs. Other historical suites/native programs were
not rerun. Three examples ran; the new one independently matches sklearn.
The combined audit verifies 206 compiled-unit fingerprints per profile,
complete fresh source configuration/result inventories, stable source/validator/
executable/report hashes and distinct release drivers. Reference metadata pins
sklearn 1.6.1, NumPy 2.2.6, SciPy 1.15.3, C-contiguous float64 arrays, NumPy's
8192-element buffer and the default optimized one-thread auto backend.
Artifacts remain ignored under validation/results/parameter_distances*,
including the final combined audit. Original integration remains MIT with
preserved sklearn/SciPy/NumPy BSD notices; no dependency, public core type or
algorithm beyond sklearn was added.

Inventory: 222 public core types, 175 source modules, 243 native programs,
216 reference suites / 4,204,591 retained groups and 115 invalid registries /
6,270 cases. The broader translation remains unfinished.


## Pairwise kernel dispatch (2026-11-25, implementation checkpoint)

Added pairwise_kernels named dispatch, all four dense/CSR combinations,
omitted/aliased Y, integer/real degrees, parameter filtering, row callbacks,
precomputed inputs and dense GP objects. Standalone kernel bodies are unchanged.
The source wrapper bypasses nested parameter bounds, so negative/nonfinite
applicable parameters retain source arithmetic. GP omitted Y becomes explicit
Y=X, including WhiteKernel's zero matrix. Typed result/row storage adaptations
are explicitly documented. Dense overflow follows scalar reconstructions of
the pinned OpenBLAS dot, GEMV and small-TN paths with retained BSD notices.

Private checked probes passed 2,005 configurations / 6,015 groups, 276 source
rejections, 34 targeted diagnostics and native integration tests. All 1,970
nonfinite masks matched; maximum scaled finite error was 4.45e-16. Full package
checked/optimized verification subsequently passed, as recorded below. Original integration remains MIT;
no dependency, public core type or ML extension was added.


## Verified pairwise kernel dispatch (2026-11-25)

Implementation be2ab7b passed full checked and optimized fpm build --tests
batches. Each profile passed 2,005 complete configurations / 6,015 retained
groups comparing matrix values, nonfinite masks and combined input identity/
IEEE state. All 1,970 source nonfinite masks matched; maximum scaled finite
error across both profiles was 4.441e-16, at rtol=atol=3e-12. All 276 controlled
source rejection cases and 34 targeted diagnostics passed. The accepted
inventory records four precomputed CSR results returned as owned dense native
matrices, explicitly identified as storage adaptation. Sparse callbacks receive
dense logical rows; no new ML algorithm is introduced.

Named kernels, all dense/CSR combinations, omitted/aliased Y, real/integer
degrees, filter_params, precomputed references, row callbacks and dense native
GP objects are covered. The source dispatcher's nested-validation bypass is
retained for applicable signed/nonfinite parameters. GP omitted Y is supplied
as explicit Y=X, so WhiteKernel contributes zero here. Callback self evaluation
uses upper triangle then diagonal. Dense accumulation reconstructs the pinned
OpenBLAS dot/GEMV/small-TN paths, including controlled overflow lane/tail cases.
No BLAS runtime dependency is added. Other hardware, backend layouts and thread
settings can round or propagate nonfinite values differently.

Each profile also passed 13,380 parameterized nearest-pair groups and their
1,406 source rejection records / 33 diagnostics; 10,728 earlier dissimilarity
groups (including 144 identified sparse Bray-Curtis adapter configurations) and
3,582 source rejection records / 60 diagnostics; 16,432 earlier nearest-pair
groups and 88 source rejections / 26 diagnostics; 16,729 earlier pairwise groups
(including 20 retained roundoff-bound checks) / 26 diagnostics; 192 core groups
and 21 selected native programs. Other historical suites/native programs were
not rerun. Four examples ran; the new one independently matches sklearn.
The combined audit verifies 210 compiled-unit fingerprints per profile,
complete fresh source configuration/result inventories, stable source/validator/
executable/report hashes and distinct optimized drivers. Existing metric
bodies, standalone kernels, GP bodies and other library sources match 13a94d5
apart from the new import/export. Reference provenance records sklearn 1.6.1,
NumPy 2.2.6, SciPy 1.15.3 and actual Python/BLAS/sparse binary hashes; NumPy's
OpenBLAS 0.3.29 SkylakeX backend uses one thread. Artifacts remain ignored under
validation/results/kernel_dispatch*, including the final combined audit.

Original integration remains MIT. Existing sklearn/NumPy/GP/OpenBLAS notices
are retained, with the 2021 OpenBLAS Project small-kernel BSD notice added in
LICENSES/OpenBLAS-small-kernel-BSD. No dependency or public core type was added.
Inventory: 222 public core types, 176 source modules, 244 native programs,
217 reference suites / 4,210,606 retained groups and 116 invalid registries /
6,304 cases. The broader translation remains unfinished.


## Gram LARS and graphical LARS verification (2026-11-25)

Implementation 13160bb adds lars_path_gram and mode='lars' to GraphicalLasso
and GraphicalLassoCV. Full checked and optimized fpm build --tests batches
passed using private fpm 0.14.0 and GNU Fortran 17. The checked flags include
-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow.

Each profile passed 1,440 Gram-path groups across 360 fixtures, including
25 source coefficient-dropout events. Cases cover LAR, Lasso, positive Lasso,
alpha interpolation, zero responses, iteration caps, duplicate columns and
wide matrices. Ninety duplicate-column fixtures certify merged coefficients
and every prediction knot because identical columns can select different
active representatives. Full wide paths use eps=0.1; their early knots use
machine epsilon. Rank-deficient full paths at machine epsilon can differ
from sklearn's incremental BLAS/LAPACK backend, including knot counts.
This existing backend limitation is retained explicitly in API and coverage
documentation, rather than claimed as exact parity. Input arrays are checked
unchanged by the native reference driver.

Both graphical solvers passed 144 GraphicalLasso result groups and 5,980
GraphicalLassoCV groups across 255 fixtures per mode per profile. These include
precomputed/centered covariance, iteration caps, warm starts, explicit folds,
refined and explicit grids, ill-conditioned empirical endpoints and failed
candidates. Automatic diagonal-covariance grids retain NaN-candidate failures.
The existing conditioned empirical-endpoint certificates are retained;
positive-alpha fits and selection use ordinary direct source comparisons.

Each profile also passed 730 existing least-angle groups, 192 core groups
and eight selected native programs. The 1,000 least-angle model-selection
groups passed directly in checked mode. Optimized mode passed 999 directly;
one rank-deficient LarsCV MSE group retains a source tolerance miss in one
entry (relative error about 4.3e-8 versus rtol=3e-8). A separately compiled
pre-change 5fe5c01 solver produces byte-identical output for the entire
200-group seed-2026 fixture, including that entry. An explicit optional
baseline-executable certificate records this existing limitation; ordinary
tolerances are unchanged and the default validator remains strict. Both
profiles passed
87 family input guards and 11 dedicated Gram guards. Two of the family guards
are new empty/shape checks; invalid-mode fixtures now use an unknown string
because LARS is accepted. Other historical suites/native programs were not
rerun. The combined audit checks 195 compiled-unit fingerprints per profile,
stable source/validator/executable manifests and distinct optimized Gram
drivers. It also confirms the old LARS arithmetic body is unchanged apart
from recoverable numerical status, and other library sources match 5fe5c01
except the three intended modules. Artifacts remain ignored under
validation/results/graphical_lars*, lars_gram* and graphical_cv*.

Original integration remains MIT with existing sklearn BSD attribution;
no dependency, public core type or algorithm outside sklearn was added.
Inventory: 222 public core types, 176 source modules, 245 native programs,
218 reference suites / 4,218,170 retained groups and 117 invalid registries /
6,317 cases. The broader translation remains unfinished.


## Direct and Gram OMP implementation (2026-11-25)

Added native orthogonal_mp and orthogonal_mp_gram with direct/Gram Cholesky
pursuit, residual tolerance, auto-precomputation, multiple targets and owned
full paths. Private checked and optimized probes each passed 16,170 groups
across 2,310 configurations. Native unit and 27 guard probes passed.
Independent certificates identify duplicate-atom allocation, zero-tolerance
warning roundoff and full-span rank-deficient tail steps. Source ragged-path
and missing-precomputed-iteration wrapper quirks are explicit adapters.
Full fpm checked/optimized build and regression verification subsequently passed,
as recorded below.
The existing OMP estimators retain their implementations. MIT and upstream
BSD attribution are retained.


## Verified direct and Gram OMP paths (2026-11-25)

Implementation b0256e6 passed full checked and optimized fpm build --tests
batches. Each profile passed 2,310 configurations / 16,170 retained groups,
including 154 ragged multi-target paths. Checks cover direct/Gram/auto routing,
vector/column/multiple responses, counts/defaults, finite nonnegative squared
residual tolerances, full paths and final coefficients, dependent/zero atoms,
zero responses, scaled/wide dictionaries, per-target iterations and input
retention. On directly compared coefficient/path entries the maximum scaled
error was 5.596e-15, at rtol=atol=3e-8.

The inventory identifies 462 duplicate-atom coefficient/path certificates
per profile: merged coefficients and predicted paths agree while atom
representatives may differ. Checked/optimized profiles respectively retain
4/4 full-span certificates. These verify common source/native path
prefixes, zero padding, final-path consistency and reconstruction to backward
error at every extra step after spanning a wide dictionary. They do not claim
identical final knot counts or nonunique coefficients. Exact-zero-tolerance
warning-state roundoff differs in 36/14 configurations; all corresponding
coefficient/path/iteration comparisons or full-span certificates still pass.
Premature flags describe native stopping and replace source warning objects.

Source 1.6.1 wrapper probes confirm missing requested iteration counts in the
precomputed orthogonal_mp route and IndexError for zero-first-target ragged
multi-target paths in both public functions. The native owned result retains
all independent target paths and iteration counts, using explicit zero padding.
These are identified storage/wrapper adaptations; reference column solves
supply each complete source target path. Gram vector norms are represented by
a native scalar, while matrix norms retain one entry per target. Source's
floor(.1*p) Gram default and its small-feature zero-count rejection are kept.

Each profile also passed 1,936 existing OMP CV groups (with its retained
numerical-equivalence certificates), 204 regression/factor/projection groups,
192 core groups, ten selected native programs, 27 diagnostics and the new
example. Eight diagnostics identify native finite/shape/nonpositive-pivot
contracts. Other historical suites/native programs were not rerun. The audit
checks 197 compiled-unit fingerprints per profile, stable source/validator/
executable/input inventories and distinct optimized drivers. Every old library
body matches e60457d; only linear_model import/export lines were added, alongside
the new module. Artifacts remain ignored under validation/results/omp_paths*.
The existing OMP estimator and CV estimator retain their original backends.

Original integration remains MIT with retained sklearn BSD attribution.
No dependency, public core estimator type or algorithm outside sklearn was
added. Inventory: 222 public core types, 177 source modules, 246 native
programs, 219 reference suites / 4,234,340 retained groups and 118 invalid
registries / 6,344 cases. The broader translation remains unfinished.


## Dense coordinate-descent path implementation (2026-11-25)

Added enet_path/lasso_path with scalar direct/Gram kernels and the existing
grouped multi-task backend. Paths support automatic/explicit alpha grids,
warm coefficients, random/cyclic coordinates, positivity, precomputation and
shared RNG continuity. Private checked and optimized probes each passed
27,684 groups across 3,076 configurations; analytic unit and 39 guard probes
passed. Coefficients and alpha grids compare directly to sklearn. Explicit
KKT/optimality certificates retain floating-point stopping differences at
zero models or zero L1 penalty. Full fpm builds and regressions subsequently passed, as recorded below.
Original integration remains MIT with sklearn BSD attribution retained.


## Verified dense elastic-net and Lasso paths (2026-11-25)

Implementation 42942d7 passed full checked and optimized fpm build --tests
batches. Each profile passed 3,076 configurations / 27,684 retained groups.
Coverage includes vector/column/multi-task responses, scalar direct and Gram
kernels, auto and supplied precomputation, sampled Gram checks, supplied and
discarded correlations, automatic/increasing/tiny grids, explicit sorting,
duplicates and empty paths, warm coefficients, shared random-coordinate RNG,
positivity, zero sweeps and zero/dependent/wide input columns. Alpha grids and
all coefficient entries remain direct sklearn comparisons at rtol=atol=3e-8.
The maximum scaled error on directly compared values across both profiles was
8.926e-12.

Stopping metadata is retained transparently when native and BLAS/Cython
roundoff selects different iterations, status or raw dual gaps. Checked and
optimized profiles respectively required 163/126
zero-model KKT certificates and 21/10 zero-L1 optimality certificates,
covering 159/111 configurations. Reports retain every affected alpha,
source/native count/gap/status and proof bounds. Zero-model checks bound
coefficients near machine zero and verify the KKT threshold. Zero-L1 checks
use independent ridge primal/dual gaps or least-squares/NNLS objective gaps,
within the requested tolerance or the recorded numerical floor. Coefficient
and grid tolerances are unchanged; raw metadata is not claimed identical in
these cases. Source warnings are retained, including discouraged zero-L1 fits.

The owned result keeps (target,feature,alpha) coefficients, normalized dual
gaps, iteration/convergence arrays and backend/tolerance diagnostics. Matrix
targets use the coupled L2,1 penalty, even with one column; only vectors
support positivity. Source's strict scalar versus non-strict group dual-gap
stopping, zero-sweep n_iter=1 convention and skipped zero-norm columns are
retained. Native inputs, including warm initial values, remain read-only.
The API is dense real64; sparse path solves and sparse weighting/centering
remain untranslated. This batch does not replace the existing estimator
backends or claim a complete Python API port.

Each profile also passed 192 existing multi-task groups, 7,960 multi-task
CV/random-coordinate groups (including retained numerical equivalences),
216 scalar coordinate-CV groups, 192 core groups, eleven selected native
programs, 39 guards and the new example. Thirteen guards identify native
finite/shape/convex-domain/denominator contracts. Other historical suites and
native programs were not rerun. The combined audit checks 200 compiled-unit
fingerprints per profile, stable source/validator/executable/input inventories
and distinct optimized drivers. Existing library algorithm bodies match
9cbc12e: only new import/exports and a shared multi-task RNG entry point were
added around the new module. Artifacts remain ignored under
validation/results/coordinate_paths*.

Original integration remains MIT with sklearn BSD attribution retained.
No runtime dependency, public core estimator type or algorithm outside sklearn
was added. Inventory: 222 public core types, 178 source modules, 247 native
programs, 220 reference suites / 4,262,024 retained groups and 119 invalid
registries / 6,383 cases. The broader translation remains unfinished.


## Sparse coordinate-path implementation (2026-11-25)

Added CSR vector overloads for enet_path/lasso_path using compressed-column
work storage, weighted residuals and implicit centering. Source precompute
forcing, raw alpha-grid conventions, signed/zero weights, duplicate storage
and unsigned implicit-zero counting are preserved. Private checked and
optimized probes passed direct source comparisons; analytic weighted/centered
and duplicate-counter tests and 33 guard probes passed. Large sparse storage
and ignored precomputation/scale inputs are retained in the reference corpus.
Full fpm builds and regressions subsequently passed, as recorded below. MIT and upstream attribution
are retained.


## Verified sparse weighted/centered coordinate paths (2026-11-25)

Implementation 47291a7 passed full checked and optimized fpm build --tests
batches. Each profile passed 2,168 configurations / 19,512 retained groups,
all directly compared to sklearn 1.6.1, with no numerical-equivalence
exceptions in the new suite. Maximum scaled error across both profiles was
7.698e-16, at rtol=atol=3e-8 for numeric arrays and exact equality for
shape, iteration, convergence, backend and input-retention groups.

Coverage includes canonical/unsorted/explicit-zero/duplicate CSR, raw CSC
conversion, scalar elastic-net/Lasso, automatic and explicit grids, repeated
alphas, initial coefficients, positivity, cyclic/random coordinates, zero
sweeps, missing/all-one/fractional/zero/signed sample weights and implicit
means/scales. Sparse precomputation is forced off, ignored Gram/Xy/scale
semantics are retained and all inputs are checked bit for bit, including
ignored NaN arrays. The automatic grid deliberately remains raw/unweighted/
uncentered; weights are not normalized. Explicit-grid paths avoid unused raw
correlations, including a zero-weight case whose raw X.T*y would overflow.

Raw duplicate storage follows the pinned source rather than silently
canonicalizing: stored-entry squares determine coordinate norms, and the
unweighted uint32 implicit-zero count wraps for overfull columns. Each
profile retains 180 configurations with such columns. Weighted accumulation
uses its separate source formula, so all-one weights need not equal omitted
weights for noncanonical data. Dedicated analytic tests distinguish both
normalizations. Conventional logical-matrix regression assumes canonical CSR;
finite signed weights are accepted as source behavior without a convexity claim.

The implementation transposes CSR once and uses compressed-column entries
plus O(n+p) working vectors, without a dense design or Gram allocation. A
32768-by-8192 design with one stored entry passed; its dense equivalent is
2 GiB. An empty-storage centered problem exercises implicit nonzero columns.
Requested output coefficient paths still use O(p*n_alphas) storage.

Each profile also passed 27,684 earlier dense path groups (including their
explicit numerical stopping certificates), 7,960 multi-task CV/random groups,
216 scalar coordinate-CV groups, 192 core groups, eleven selected native
programs and two examples. All 33 new diagnostics / 21 identified native
buffer/finite/mean/domain contracts and 39 dense diagnostics passed. Other
historical suites/native programs were not rerun. The combined audit checks
202 compiled-unit fingerprints per profile, stable source/validator/executable/
input inventories and distinct optimized drivers. Every old library algorithm
body matches 3b6276f; only the coordinate-path module gains the CSR overloads,
private CSC storage and sparse kernel. Artifacts remain ignored under
validation/results/sparse_coordinate*.

Original integration remains MIT with sklearn BSD attribution retained.
No runtime dependency, public core estimator type or algorithm outside sklearn
was added. Inventory: 222 public core types, 178 source modules, 248 native
programs, 221 reference suites / 4,281,536 retained groups and 120 invalid
registries / 6,416 cases. The broader translation remains unfinished.


## Fractional minimum sample counts (2026-11-27)

Implementation `23862e6` adds typed integer/real `set_min_samples_leaf` and
`set_min_samples_split` to DecisionTree, ExtraTree, RandomForest, ExtraTrees
and classic GradientBoosting classifiers/regressors. Fractions resolve with
ceil(full input row count), including zero-weight/out-of-bag rows; split counts
retain a minimum of two and require enough rows for both leaves. Leaf fractions
are in (0,1), split fractions in (0,1]. Search, clone resets, pruning paths,
warm growth and extracted component refits preserve the selected parameter
type. Integer setters reset fractional state; positive direct integer fields
override it. This adds sklearn compatibility without an extension algorithm.
Original typed integration is MIT with retained sklearn BSD attribution.

Checked and scoped optimized comparisons pass all 6,332 fitted configurations:
896 regression/672 classification trees, 768 regression/576 classification
forest warm snapshots, and 1,520 regression/1,900 classification boosting fits.
Complete tree state, pruning, predictions, scores, class probabilities, staged
outputs and OOB histories match actual sklearn 1.6.1. All 240 typed parameter
acceptance/rejection cases pass in both profiles. Integer fields remain exact;
floating comparisons retain rtol 2e-10 / atol 2e-12. No cases or numeric oracles
were removed or altered. The final fpm executables replay the preserved
Source-generated inputs/oracles; the optimized suites regenerate them
independently, and the cross-profile audit confirms their identities.

The full checked fpm build and 337-job package audit pass, including all 302
native programs, ten selected examples, fourteen earlier boosting suites, two
earlier tree suites, 510 legacy classifier groups and 192 core comparisons.
All 870 compiled fingerprints (204 library units) are verified. The optimized
scope rebuilds fifteen affected modules at O3, links matching checked
unchanged dependencies, and passes seven native tests plus all seven new
reference/guard suites. This is a scoped optimized audit, not a complete
optimized rebuild of the package.

Two initial audit commands used a mistyped validator filename. Both corrected
warm-start jobs pass; their failed command logs are retained. Earlier numeric
oracles and input identities are preserved. Python's process-dependent set
ordering in parameter-error messages is recorded separately; the preservation
audit compares identical set members, error types, phases and cases while
retaining the original raw messages. No library correction was needed.

GNU Fortran 15.2.0 verification uses the existing validation-only static
libgfortran/locale-copy wrapper for the separately reproduced Windows numeric
locale lifetime defect. Installed compilers remain unchanged. Reports, build
logs, original failed runner evidence, fingerprints and immutable oracle
provenance are retained under ignored `validation/results/min_samples_*`,
including `min_samples_package_audit.json` and
`min_samples_cross_profile_audit.json`. Both implementation and verification
commits are local; no push was made.


## Classic boosting validation stopping and NaN losses (2026-11-27)

Implementation `2d60731` adds validation stopping to both classic gradient
boosters through `n_iter_no_change`, `validation_fraction` and `tol`. Native
patience zero represents sklearn None. Regression uses seeded shuffle splits;
classification uses stratified splits, validates weighted class retention on
all rows before splitting, and checks that training rows retain every raw class.
Initializers and component controls use the training partition. The Source
circular window, retained stopping stage, trimmed histories and independent
split/tree RNG streams are preserved. Warm fits recreate the split and window
while retaining the tree RNG; validation starts at the first retained stage,
matching the sklearn 1.6.1 warm-generator convention. Search and clone resets
retain the new controls. Original integration remains MIT with retained BSD
attribution, without a new dependency or extension algorithm.

The checked build initially trapped on a Source-accepted NaN validation loss
with signed weights. Fix `8cc9229` makes this comparison false without raising
an invalid floating-point exception. A native regression test reproduces the
actual RandomState fixture. All 96 signed-weight configurations pass: 77
successful fits and 19 matching fit rejections, including seven regression
fits with NaN training/validation losses that stop after one stage. The initial
SIGFPE executable, inputs, raw output and Source oracles remain retained;
the final audit verifies that the reference data was not changed by the fix.

Nine new reference/guard suites pass all 1,968 configurations in checked and
scoped optimized profiles: 384 regression/396 classification cold fits, 512
warm snapshots per kind, 96 signed-weight fixtures, 28 fitted-template
composition fixtures and 40 parameter/split checks. Compositions include 25
numeric matches and three controlled rejections matching Source zero-total
OOB-weight errors. Source-undefined scores are recorded separately. Tree
structures, metadata and labels remain integer-exact; floating tolerances
remain rtol 2e-10 / atol 2e-12, including matching nonfinite masks.

The final checked fpm build and all 347 package jobs pass: all 303 native
programs, ten selected examples, fourteen earlier boosting suites, two earlier
tree suites, all 6,332 fractional-sample configurations and 240 parameter cases,
510 legacy classifier groups and 192 core comparisons. All 876 compiled-unit
fingerprints (204 library units) are verified. The optimized scope rebuilds
fifteen affected modules with matching checked dependencies and passes eight
native programs plus all nine new suites; this is not a full optimized package
rebuild. Final manifests verify current source, executable and object hashes.

Older ignored numeric JSON reports were missing when the initial audit tried
to replay them. Their raw inputs and native outputs were still available.
The final audit regenerates the six fractional-sample suites from installed
sklearn, verifies byte-identical retained inputs, and checks every retained
native output against the fresh Source arrays at the unchanged bounds. It
makes no claim to have compared unavailable historical Source JSON arrays.
Current-turn pre-fix Source arrays for fourteen boosting/two tree suites and
the original 384-case classification prefix are retained and compared directly.
Python set-order differences in error messages are recorded separately without
changing error types, phases, cases or allowed set members.

GNU Fortran 15.2.0 validation uses the existing static libgfortran/locale-copy
wrapper for the separately reproduced Windows numeric-locale lifetime defect;
installed compilers remain unchanged. Evidence is retained under ignored
`validation/results/gb_early_*`, including the final package/cross-profile
manifests, initial failures and compressed backup
`validation/results/gb_early_evidence_8cc9229.zip`. All commits are local;
no push was made. The overall sklearn translation remains incomplete.

### 2026-11-27 — AdaBoost tree controls and shared regression trees

Commit `05defcf` extends both AdaBoost estimators with typed integer/fractional
minimum-sample and feature controls, named feature budgets, leaf budgets,
minimum weighted leaf fractions and cost-complexity pruning. AdaBoost.R2 now
uses the shared DecisionTreeRegressor backend, including all four criteria and
both splitters. Extracted component estimators retain their parameter modes
when independently refitted on a new sample/feature shape. The public
unfitted prototype stays unfitted, and oversized integer feature budgets
follow sklearn acceptance.

AdaBoostRegressor now participates in the common fit/predict/weighted/seed
registry, nested estimator search parameters, pipelines, feature selection,
voting, bagging and stacking. Explicit CART-base seeding matches sklearn's
nested parameter order. Original MIT integration and BSD attribution are
retained, without a new runtime dependency or extension algorithm.

The checked implementation initially trapped on two NaN input comparisons in
AdaBoost.R2. Sequential finite and bound checks fix these actual failures;
all 158 Source parameter/sample-weight cases now pass. Initial inputs,
Source results, failing executable and diagnostics are retained. An overlapping
build and validator also caused a Windows linker lock and invalid executable;
those harness failures are retained, and the complete comparison was rerun
using stable executable copies after the build finished.

Both checked and scoped optimized profiles pass 1,344 new full-tree fixtures
(768 regression, 576 classification), 158 guards and 24 fitted-template
composition fixtures. The 36 original AdaBoost.R2 fixtures and all 1,649
comparison groups still pass at their original 3e-8 tolerance; their input
and Source arrays match the preserved pre-change oracle exactly. Both
profiles pass all 1,460 original SAMME classifier fixtures. Its former
native-only oversized-feature rejection is retained as Source acceptance;
the other thirteen diagnostics still reject. Raw new tree floats use
rtol 2e-11 / atol 2e-12, compositions retain 2e-10 / 2e-12, and integer tree
structure, labels and seeds remain exact. Source undefined zero-ensemble
importance is recorded explicitly with the existing native zero-vector
adaptation.

The final checked fpm build and all 355 package jobs pass: 304 native programs,
ten selected examples, fourteen previous boosting/two tree suites, 6,332
fractional-sample configurations and 240 parameter checks, the eight earlier
early-stopping suites plus 40 guards, seven AdaBoost suites, 510 legacy
classifier groups and 192 core comparisons. All 881 compiled-unit fingerprints
(204 library units) match the current sources. The optimized validation rebuilds
17 affected modules against matching checked dependencies and passes seven
native programs and all seven AdaBoost suites; it is a scoped optimized check.
Cross-profile manifests verify current source, executable and object hashes
and preserve the original Source oracles and inputs.

GNU Fortran 15.2.0 checks continue to use static libgfortran and the validation
locale-copy wrapper for the separately reproduced Windows runtime defect.
Evidence, initial failures and manifests are retained under ignored
`validation/results/ada_controls_*`, with a compressed backup in
`validation/results/ada_controls_evidence_05defcf.zip`. Commits remain local;
the overall sklearn translation remains incomplete.


### 2026-11-27 ? Weighted, sparse and warm-start Huber regression

Implementation `6971c92` replaces Huber's full-memory BFGS with the existing
bounded limited-memory L-BFGS-B dependency, using the Source weighted total
objective, scale floor, default function-value stopping and zero-iteration
counter convention. Dense/CSR features accept vector or one-column targets
and scalar/vector weights. CSR fitting traverses stored entries, including
unsorted duplicates and explicit zeros, without densifying the design.
Warm fits preserve Source's old-intercept initial-vector convention, including
when the public fit_intercept flag is false. Clone sites clear learned Huber
state while retaining controls. Common weighted dispatch, parameter search and
compositions now reach Huber-specific methods. The package remains MIT with
retained sklearn, SciPy, OpenBLAS and vendored L-BFGS-B BSD attribution; no new
estimator name, extension algorithm or runtime dependency is added.

The observed Windows norm-square updates are opt-in to the vendored solver;
existing callers retain their original default arithmetic path. The existing
Ridge transpose-vector helper is moved to shared kernel dispatch with its
finite arithmetic body unchanged after renaming. All 336 Ridge fixtures /
672 groups pass with maximum scaled error below 1.8e-14. A norm diagnostic
matches SciPy exactly on 3,355 vectors, including extreme magnitudes.

Source/native comparisons retain all 528 sequences / 744 fits at unchanged
rtol/atol 3e-8. The 522 non-wide sequences match; six wide, near-interpolating
fits capped at 100 iterations still fail strict comparison. Both Source and
native report nonconvergence on these cases. They remain failures, without
exclusions, changed expected arrays, relaxed bounds or fitted certificates.
Direct objectives/gradients match exactly at 1,442 real Source optimization
points plus 72 signed/negative/zero-weight probes, across C/F dense reduction
orders and canonical/raw CSR. This does not certify fitted-path parity.

The twelve fitted-template composition fixtures retain four strict numeric
mismatches in weighted CV, bagging/OOB or stacking, including Source inner fits
that report function-value convergence. Remaining errors in the retained grid
are below 4.9e-5. Eight fixtures match fully; integer rankings, selected models,
sample/feature indices and seeds match in every fixture. Correcting an initial
pipeline test-construction mismatch preserves the original Source oracle;
the native child now receives the same parameter-only clone as Python. The
internal bagging column-selection adapter reproduces NumPy's F-contiguous
BLAS route and reduces the initial large bagging prediction difference from
about 0.151 to 2.94e-7 in the unweighted eight-feature fixture. The adapter is
excluded from the public facade. All original failures remain retained.

All 70 actual Source parameter/weight/shape/query acceptance checks pass in
checked and scoped optimized profiles. Source cases run in isolated processes;
an earlier combined Source process terminated with a Windows heap error, whose
cause is not established. Its exit record is retained. A private probe also
encountered stale derived-type module metadata; complete dependency rebuilds
resolve it. No installed compiler or DLL was modified.

The final checked fpm build succeeds. The package audit completes 323 jobs:
305 native programs, eleven examples, 192 core comparisons, 213 original
robust/GLM groups, 672 Ridge groups, 1,514 objective/gradient cases and 70 guards
pass. Two strict fitted-comparison suites deliberately exit nonzero with the
six and four retained mismatches. Thus 321 jobs pass; strict Source parity is
false. All 888 compiled-unit fingerprints (205 library units) match current
sources. Scoped O3 validation rebuilds nine modules against matching checked
dependencies and runs thirteen native programs plus the same seven reference
suites, retaining the same strict mismatch inventories. It is not a full
optimized package rebuild.

The cross-profile audit verifies current source/executable/object hashes and
that every original fitted Source oracle and input, including all 213 legacy
robust/GLM groups, is unchanged. GNU Fortran 15.2.0 validation retains static
libgfortran and the locale-copy wrapper for the separately reproduced Windows
formatted-I/O lifetime defect. Redundant temporary executable copies from the
previous completed audit were byte-verified against their canonical build
before deletion, recovering about 4.13 GB; failure evidence was retained.

Evidence remains under ignored validation/results/huber_* with the verified
compressed backup huber_evidence_6971c92.zip. All commits are local. Remaining
strict fitted-path differences and the broader sklearn translation are still
unfinished work.


### 2026-11-27 ? RANSAC sampling, weighted CSR and joint targets

Implementation `24036fc` replaces RANSAC's global Fortran random stream with
the existing NumPy-compatible local MT19937 and auto sampling-without-replacement
backend. Tracking, permutation and reservoir paths preserve source subset
order. Typed minimum samples, skip limits, data/model validation callbacks and
custom row losses join weighted dense/CSR OLS subset fitting and final refits.
Vector, column and joint targets are supported through the existing OLS solver.
`estimator_` retains the fitted direct base, and inherited aliases retain native
compatibility. Common weighted fitting and scalar/nested search controls reach
RANSAC's own methods. Arbitrary base-estimator objects remain unsupported.
No new estimator name, extension algorithm or runtime dependency is added;
MIT integration retains sklearn and NumPy attribution.

The loop preserves Source's initial best inlier count, NaN score comparisons,
NaN custom-residual exclusion, dynamic trial cap, skip-before-sampling check,
last-fit counters and unweighted consensus scoring. Real min_samples=1.0 means
one sample, while fractions strictly below one use ceil(fraction*n). Typed
integer zero is rejected; clear_min_samples restores the native None sentinel.
Weighted scoring corresponds to Source's explicit metadata-routed base-score
request. CSR callbacks receive a dense selected subset; fitting itself preserves
CSR storage without constructing a full dense design. Native per-target
intercept vectors broadcast Source scalar zeros. Native coefficient aliases
correspond to Source SelectFromModel's explicit estimator_.coef_ getter.

A first native probe exposed an unallocated logical inlier mask. Allocating it
before iteration fixes the actual invalid-memory access; the failing binary,
source, input and debugger output are retained. Initial reference harness
errors around Source's score routing and nested coefficient getter are also
retained separately. Their fixes do not change fitted Source oracles. Every
fixture executes independently so one unexpected rejection cannot truncate
later comparisons.

The full reference inventory retains 714 configurations: the original 498-case
prefix plus 216 appended unsorted/duplicate/explicit-zero CSR cases. All 216
appended cases match. Source has 599 successful fits and 115 fit rejections;
697 configurations match completely. Seventeen dense zero-threshold cases
with a one-observation non-intercept OLS subset retain strict differences from
Source: ten Source rejections are accepted natively, one Source fit is rejected
natively, four fitted outputs and two counters differ. Exact-zero membership
is sensitive to underlying SVD rounding. These are retained failures at the
original rtol/atol 3e-8, without threshold changes, exclusions or replacement
expected values. Every nonzero-threshold configuration matches.

All 86 actual Source parameter/weight checks pass, including negative weights
that are only rejected if selected for OLS fitting. All eight fitted-template
composition fixtures match scores, predictions, sampled rows/features and
selected best models. Two strict CV-rank comparisons remain false for candidates
whose means differ by roughly 1e-15; their exact ranks are retained. Source
uses explicit metadata routing for weighted pipeline scoring. These adapters
are identified rather than being presented as full Python routing support.

The checked fpm build succeeds. The package audit completes 324 jobs: all 306
native programs, eleven selected examples, 192 core comparisons, all 213
original robust/GLM groups, all 672 Ridge groups, 1,514 Huber objective/gradient
checks and the 86 RANSAC guards pass. Two earlier Huber fitted suites retain
their unchanged six and four strict mismatch inventories, so 322 of these jobs
pass. The two new RANSAC fitted suites separately retain the seventeen and two
mismatches above. All 891 compiled-unit fingerprints (205 library units) match
current sources. Scoped O3 validation rebuilds nine modules against matching
checked dependencies, runs fourteen native programs and all nine reference
suites, and retains the same mismatch inventories. This is not a full optimized
package rebuild or a claim of complete Source parity.

The cross-profile audit verifies current source/executable/object hashes and
unchanged Source inputs, fitted oracles, the original 498-case prefix and all
213 legacy robust/GLM groups. GNU Fortran 15.2.0 checks retain static libgfortran
and the validation-only locale-copy wrapper for the separately reproduced
Windows formatted-I/O lifetime defect. Installed compilers are unchanged.
Evidence and failures remain under ignored validation/results/ransac_* with
the verified compressed backup ransac_evidence_24036fc.zip. Commits are local;
the strict numeric/rank differences and the broader translation remain open.


## Weighted GLMs, general Tweedie powers/links and warm starts (2026-11-27)

Extended the existing damped eigen/Newton GLMs with normalized sample weights,
scalar/vector fit weights, vector/one-column targets, warm coefficient reuse,
fitted feature/power metadata, weighted D2 and parameter-only clone resets.
Poisson and Gamma force their fixed loss/link through every dense overload.
Tweedie now accepts finite real powers and identity/log/auto links, including
negative powers, powers between zero and one, powers above two and log-link
squared error. Added identity/log gradient and curvature formulas, positive
trial checks, weighted initialization, and a gradient fallback when the
positive-eigenspace step is not a descent direction. The intercept remains
unpenalized and common nonzero weight scaling leaves the objective unchanged.

Search adapters expose the applicable GLM parameters, common base dispatch
accepts weighted GLMs, and Pipeline delegates weighted D2. Native one-column
score flattening is explicitly identified as a convenience extension: an
actual sklearn 1.6.1 probe rejects two-dimensional GLM scoring targets. No new
non-sklearn estimator was added. The MIT package license and BSD attribution
for the reference loss/contracts are retained.

Reference validation covers 158 configurations and 301 successful Source fits,
with scalar/vector/zero/signed weights, columns, two-step warm fits, explicit
links, capped fits and five zero-iteration rejections. The first run found
incorrect native acceptance of max_iter=0; the implementation now rejects it.
The original executable and failed comparison are retained. Twenty-four
separate Source/native invalid-input checks cover domains, controls, weights,
shape and warm feature-count mismatches. Native tests also exercise weight
scaling, row removal, clone reset, fitted-loss scoring and weighted Pipeline
fit/score. The strict 3e-8 comparison contract is unchanged.

Three configurations, indices 150, 151 and 153, retain D2 differences near
powers one and two. Their coefficients agree closely; cancellation in the
deviance evaluations makes scoring sensitive to evaluation order. These stay
strict failures. Source default L-BFGS, exact Newton-Cholesky trajectories,
warning/iteration-count parity, CSR fitting and extreme numerical domains
remain gaps. This is progress within the existing sklearn estimators, not a
claim that the broad translation is complete.


Scoped optimized validation rebuilt ten modules with O3 and loop unrolling,
ran sixteen native programs and eleven reference suites, and passed all native
programs. Six reference suites pass; five retain the documented GLM, Huber
and RANSAC numeric/rank differences. The new GLM failure inventory is exactly
[150,151,153], with all 24 guards passing. Existing Huber/RANSAC inventories
are unchanged. A byte comparison verifies that only the overridden linear
model and Pipeline objects changed between the stable optimization baseline
archive and the final checked library. This is scoped optimization, not a
full optimized package rebuild. GNU Fortran 15.2.0 validation retains static
libgfortran and the validation-only locale-copy wrapper; installed tools are
unchanged. Full checked package verification is recorded in the follow-up
verification entry.


### Weighted GLM checked verification and evidence

Implementation commit: 5955340. The final checked fpm build succeeds, and the
package audit completes 329 jobs with 324 passing. All 307 native programs and
eleven selected examples pass, together with core comparisons, all 672 Ridge
groups, all 213 original robust/GLM groups, 1,514 Huber objective/gradient
checks, 86 RANSAC guards and 24 new GLM guards. Five fitted reference suites
remain deliberately nonzero: the new GLM scores retain exactly three cases;
earlier Huber suites retain six/four cases, and RANSAC retains seventeen
numeric/acceptance/counter cases and two exact rank cases. These inventories
are unchanged across checked and scoped optimized builds. Tolerances, fixtures
and expected outputs were not loosened to make them pass.

All 894 audited build fingerprints match current sources, including 205
library units. The cross-profile audit verifies source/object hashes, matching
Source inputs and fitted oracles, the original legacy groups and all 195
non-overridden baseline archive members. The ten-module O3 build passes all
sixteen native programs and retains the same reference failure inventories.
Validation uses GNU Fortran 15.2.0, static libgfortran and the existing
validation-only Windows locale-copy wrapper; this does not establish stability
for an unwrapped Windows formatted-I/O runtime.

Reports, initial failure evidence, binaries, optimized objects, stable/final
libraries and source/license snapshots are retained under ignored
validation/results/glm_*. The compressed backup is
glm_evidence_5955340.zip: 218,687,787 bytes,
2,115 entries, SHA256
c550e628102d4f0c8758bfd9ccc57fb01205b853b9df7f6336de2eb397c844d5.
ZIP CRCs and critical source/report/binary hashes were verified. Commits remain
local. The three new score differences, existing mismatches, remaining solver
and sparse options, and the broader translation remain open.


## Source-default GLM L-BFGS and direct CSR fits (2026-11-27)

Added a dedicated weighted GLM loss/gradient and L-BFGS-B path, now selected by
`solver='lbfgs'` by default as in sklearn. It uses memory ten, gtol=tol,
ftol=64*epsilon and a 50-step line-search limit. The vendor's new optional
MaxLineSearch parameter retains the 20-step default for all existing callers.
All caller dependencies were rebuilt; native NCA, GPR/GPC, MLP, Ridge and Huber
regression checks remain part of the validation. Native termination messages,
convergence flags and iteration counts expose capped/abnormal outcomes without
turning Source convergence warnings into fatal fit errors.

Dense and direct CSR vector/column fits accept scalar/vector weights and warm
starts. CSR gradients/prediction/scoring retain raw duplicates, zeros and
storage order, with no dense design or feature-square Hessian. Clone resets
preserve the solver. The prior dense eigen/Newton backend is identified as the
`native-newton` extension; original Newton-specific fixtures select it
explicitly, preserving their data, tolerances and expected values. Source
Newton-Cholesky, universal optimizer trajectory parity, extreme numerical
ranges and Source score behavior outside the native positive prediction/mean
contract remain gaps. Native column-score flattening is an identified adapter.

The isolated objective comparison covers 192 actual sklearn LinearModelLoss
values/gradients, including finite/signed/zero weights, dense/canonical/raw CSR,
real Tweedie powers and nonfinite identity-link trial arithmetic. Values and
gradients pass the strict 3e-12 comparison. An initial IEEE-state preservation
bug was found and fixed by separating the flag/mode save and restore passes;
its report, executable and source are retained. All 192 IEEE-state checks now
pass. A preliminary 489-configuration fitted comparison matches Source
coefficients, predictions, iteration counts and convergence status except for
the known boundary-power scores: exactly nine strict cases, the earlier three
cases across three storage forms. Adversarial nonfinite line-search outcomes
also match Source rollback/status behavior.

The full checked build succeeds; all 308 native programs and eleven selected
examples pass. Scoped optimization rebuilds eleven modules and passes all
seventeen selected native programs. The optimized Poisson Pipeline comparison
passes 156 fixtures / 1,755 groups. Its 40,000-feature CSR memory probe peaks at
19,333,120 bytes, below 256 MiB; a 40001-square Hessian would require
12,800,640,008 bytes. The optimized Pipeline link initially omitted its test
support object; that validation setup was corrected before running it.

Three stale generic negative-input probes (power .5, power-zero/log link and
constant-target scores) were retired in favor of actual Source acceptance
comparisons. The four remaining legacy GLM diagnostics pass. The final fitted
suite additionally includes constant-target scores, and the invalid-input
suite expands to 32 dense/CSR Source/native guards. Full checked and optimized
reference inventories and evidence verification are recorded in the follow-up
entry. MIT licensing and BSD reference attribution are retained; commits are
local and the broader translation remains open.


### GLM L-BFGS checked/optimized verification and evidence

Implementation commit: a41644f. The checked fpm build and the refresh of the
two input-test executables succeed. The package audit completes 334 jobs, with
328 passing: all 308 native programs and eleven selected examples pass.
All 898 audited build fingerprints match current sources, including 206
library units. Six fitted reference suites deliberately retain strict failures:
the new L-BFGS suite has nine boundary-score cases, the older GLM Newton suite
has three, Huber retains six/four cases, and RANSAC retains seventeen cases and
two exact rank cases. These inventories match across checked and optimized
builds. No fixture geometry, tolerance or expected output was loosened.

The final L-BFGS comparison covers 492 configurations / 921 successful Source
fits. Coefficients, predictions, iteration counts and convergence status match;
only the nine documented scores differ (indices
450,451,452,453,454,455,459,460,461). All 32 actual dense/CSR Source/native guards
pass. Both profiles pass all 192 new loss/gradient/IEEE-state probes, all 156
Poisson Pipeline fixtures / 1,755 groups, all 213 original robust/GLM groups,
672 Ridge groups, 1,514 Huber objective checks and 86 RANSAC guards. Original
Source inputs and fitted oracles remain unchanged. Four surviving generic GLM
diagnostics pass after migrating the obsolete accepted cases.

Scoped O3 validation rebuilds eleven modules, passes seventeen native programs
and runs thirteen reference suites, with the same retained differences. Two
supplemental checks cover Pipeline and memory. Peak working set for the
40,000-feature sparse test is 20,099,072 bytes checked and 19,333,120 bytes
optimized; the 256 MiB cap passes. This is scoped optimization against checked
dependencies, not a full optimized package rebuild. The cross-profile audit
verifies source/object/report hashes and all 195 non-overridden baseline
archive members. Validation uses GNU Fortran 15.2.0 with static libgfortran and
the existing validation-only Windows locale-copy shim; installed tools are
unchanged, and unwrapped runtime stability is not established.

Evidence, initial IEEE failure artifacts, corrected binaries, optimized objects,
source/license snapshots and stable/final libraries are retained under ignored
validation/results/glm_*. The verified compressed archive is
glm_lbfgs_evidence_a41644f.zip: 267,719,582 bytes,
2,612 entries, SHA256
13764f64cb881a9d46bf08a2d449de08e43761bc2b4bf655d3428a023af297b3.
ZIP CRCs and critical source/report/binary hashes were verified. Commits remain
local. Source Newton-Cholesky, out-of-domain scoring, remaining numerical
mismatches and the broader sklearn translation remain open.


## Dense/CSR GLM Newton-Cholesky and shared L-BFGS corrections (2026-11-27)

Added solver='newton-cholesky' to Poisson, Gamma and Tweedie using the sklearn
1.6.1 Newton contract: weighted pointwise curvature, the weighted 25% warning
rule, absolute curvature in the Hessian, 21 backtracking attempts, incremental
raw predictions, gradient/decrement convergence checks, and L-BFGS fallback
with Source iteration accounting. Dense and raw CSR are supported without
densifying CSR designs; Hessian storage is quadratic in features as expected.
The native PSD linear solve uses Cholesky and an exact inverse one-norm
condition check; Source uses LAPACK symmetric factorization and a condition
estimate. This distinction and remaining numeric-path differences are explicit.
The existing default L-BFGS and native-newton extension remain available.

Direct Source comparisons pass 192 gradient/Hessian/curvature-warning/IEEE
probes and fifteen low-level fallback paths. The latter include exhausting all
21 line-search attempts, a zero remaining iteration budget, singular matrices
and negative curvature, without filtering through prediction-domain guards.
A preliminary 507-configuration public fit comparison retains twelve strict
cases: nine known boundary-power scores and three identity-link fallback
coefficient/iteration/public-score differences. Geometry, expectations and
3e-8 fitted / 3e-12 derivative tolerances remain unchanged.

The failure probes exposed a vendor projected-gradient NaN being collapsed by
MAX into apparent convergence. NaNs now propagate as in SciPy 1.15.3 projgr;
the Source BSD license and attribution are retained. A checked robust-regression
run subsequently trapped intermittently. Its signal was traced to Cauchy's
one-sided bound tests: an unused tl/tu local could be evaluated because Fortran
logical AND need not short circuit. A deterministic -finit-real=snan build and
debugger stopped at the exact upper-bound comparison with tu unset. The fix
evaluates distances only within the corresponding bound branch. A native test
covers free/lower/both/upper bounds. The hardened bound test and 32 concurrent
robust-regression replays all pass after the fix. Initial binaries, sources,
logs, debugger output and failed-audit evidence are retained, not dismissed as
an unexplained runtime/compiler issue. Earlier Windows locale-wrapper limits
remain separate.

The corrected full fpm build succeeds; the final package and optimized
reference audits are recorded below when complete. Scoped optimization has
passed all nineteen native programs, including the new tests and existing
optimizer callers. Source Newton controls reuse the typed solver, clone and
parameter adapters; no new external dependency is required. Package MIT
licensing and Source BSD notices are retained. This is progress toward the
broader sklearn translation, not a completion claim.


### Newton-Cholesky final verification and evidence

Implementation commit: 83c4b92. After correcting the demonstrated vendor bound
bug, the rebuilt checked package completes 340 audit jobs with 333 passing.
All 310 native programs and eleven selected examples pass, including the
robust-regression test that originally trapped. All 904 audited build
fingerprints match current sources, including 207 library units. Seven fitted
reference suites remain deliberately nonzero: the new Newton suite retains
twelve cases, L-BFGS nine scores, the legacy native-Newton suite three scores,
Huber six/four cases, and RANSAC seventeen cases plus two exact rank cases.
The inventories agree across checked and scoped optimized builds; fixtures,
expected outputs and tolerances remain unchanged.

The Newton public comparison covers 507 configurations / 936 successful Source
fits, retaining indices 450,451,452,453,454,455,459,460,461,486,487,488. Nine are
known boundary-power scores; the three identity-link fallback cases include
coefficient/iteration differences and one nonconverged CSR fit whose nonpositive
predictions are rejected by native scoring. Its partial fitted coefficients,
predictions and solver status are retained explicitly. These are unresolved
strict failures, not passing compatibility adapters.

Both profiles pass 192 new Hessian/gradient/curvature-warning/IEEE probes and
fifteen low-level fallback cases, including all 21 backtracking attempts and
zero remaining budget. All 32 GLM input guards pass for each solver. Existing
192 loss/gradient checks, 492 L-BFGS configurations / 921 Source fits, all 156
Poisson Pipeline fixtures / 1,755 groups, 213 original robust/GLM groups, 672
Ridge groups, 1,514 Huber objective checks and 86 RANSAC guards are rechecked.
Original Source inputs/oracles remain unchanged. Scoped optimization uses
twelve modules, passes nineteen selected native programs and runs seventeen
reference suites, plus Pipeline/memory supplements. Wide CSR peak working set
is 19,603,456 bytes checked / 19,394,560 bytes optimized. This is not a full
optimized package rebuild. All 195 non-overridden baseline library members
and source/object/report hashes pass cross-profile verification.

The deterministic signaling-NaN bound test fails before the correction and
passes afterward. The debugger identifies the unset upper-distance comparison;
32 hardened robust-regression replays pass after fixing it. Initial failing
binaries, vendor sources, debugger/replay logs and the failed first package
audit are preserved separately. GNU Fortran 15.2.0 checks retain static
libgfortran and the validation-only Windows locale-copy shim; installed tools
are unchanged. The demonstrated bound bug is separate from that runtime
workaround, and unwrapped Windows runtime stability is not claimed.

The verified compressed backup is glm_newton_evidence_83c4b92.zip,
317,423,537 bytes / 3,196 entries, SHA256
c4f3b72333ec35e563f67c8778d035f328d4fa987fae3dc8f2e45879eec29ace.
ZIP CRCs and critical source/report/binary hashes were verified. Evidence is
under ignored validation/results/glm_*. Commits remain local; the retained
numerical/domain gaps and broader translation remain open.


### BayesianRidge weighted fitting and evidence histories

Extended the sklearn BayesianRidge implementation with vector/scalar sample
weights, one-column targets, weighted scoring, compute_score and scores_, and
constructor-only cloning/search controls. Weighted centering and square-root
rescaling follow sklearn 1.6.1; precision updates retain the original row count,
including zero-weight rows. Both tall and wide coefficient paths and the final
posterior covariance follow the Source thin-SVD formulas. Nonnegative
hyperpriors are accepted; -1 is the documented native None sentinel for initial
precisions, and typed search None is supported. Pipeline dispatch accepts these
weighted fits. ARD is unchanged. MIT project licensing and Source BSD attribution
are retained in the implementation and third-party notices.

Added a 186-configuration Source comparison, twenty invalid-input cases, and a
native covariance/history/clone/Pipeline test. Preliminary checked comparisons
retain 26 strict failures: 21 wide covariance/standard-deviation cases with
nonunique thin-SVD null-space bases and five single-row likelihood histories.
Their indices are 63,64,65,66,67,68,69,70,71,75,76,77,78,79,80,81,82,83,87,88,89,
120,126,132,138,144. Coefficients, predictions, precisions and iteration counts
match in these cases. No tolerance or oracle was relaxed. The full checked
package builds successfully; the new native test and existing Bayesian and
IterativeImputer native tests also pass in the scoped optimized build. Full
package and cross-profile reference audits are in progress at this commit.

To make room for new evidence, removed only 164 generated executable copies
(2,390,870,194 bytes) after verifying each byte-for-byte against a preserved ZIP
entry and verifying the complete archive SHA256 against its earlier receipt.
The archives and canonical build remain intact; the deletion manifest is in
ignored validation/results/bayesian_archived_binary_cleanup.json. Installed
compilers and user files were untouched. Validation continues to use GNU
Fortran 15.2.0 with the existing validation-only Windows locale-copy shim.


### BayesianRidge final verification

Implementation commit: 81e3837. The checked package completes 344 audit jobs,
336 passing: all 311 native programs and eleven selected examples pass. Eight
strict reference suites intentionally remain nonzero, preserving the previous
seven inventories plus the new BayesianRidge 26-case inventory. All 908 build
fingerprints match current sources, including 208 library units. No fixture,
expected output or comparison tolerance was relaxed.

Checked and scoped optimized BayesianRidge comparisons cover 186 configurations
with identical Source inputs/oracles and the same 26 strict differences. All
twenty new input guards and six selected legacy guards pass. IterativeImputer
passes all 324 fixtures / 2,592 groups in each profile; Source file hashes agree.
Scoped optimization rebuilds fourteen modules, passes twenty-two selected
native programs and runs twenty reference suites, plus Pipeline and memory
supplements. This is not a full optimized package rebuild. Cross-profile
verification confirms all 194 non-overridden baseline archive members and
source/object/report hashes.

The existing GLM, Huber, RANSAC and Ridge checks retain their prior inventories.
The 156 Pipeline fixtures / 1,755 groups pass in both profiles. The 40,000-feature
CSR memory check peaks at 20,131,840 bytes checked / 19,423,232 bytes optimized.
All jobs are terminal; evidence and helper scripts are under ignored
validation/results/bayesian_*. Validation retains static libgfortran and the
validation-only locale-copy shim; unwrapped runtime stability is not claimed.
The retained covariance/history differences and the broader translation remain
open, and commits remain local.

The verified evidence backup is bayesian_evidence_81e3837.zip,
374,176,078 bytes / 3,796 entries, SHA256
fca6f2ed5787b3cc47bb54504174f892400b3b139cc4942f507916db43c9fe88.
ZIP CRCs and critical source/report hashes were verified. The receipt is
validation/results/bayesian_archive_receipt.json (ignored local evidence).


### ARDRegression posterior paths, likelihood history and controls

Replaced the ARD active-feature Cholesky inverse with the sklearn 1.6.1
Hermitian pseudoinverse and original-design tall/Woodbury branch. The native
Jacobi inverse uses SciPy pinvh's lower-triangle convention and n*eps cutoff;
it is an independent MIT numerical integration. Source update order, precision
updates, pruning and likelihood scores are preserved. Optional histories have
exactly n_iter_ entries; final posterior recomputation adds no extra score.
Added nonnegative hyperpriors/thresholds, two-sample validation, dense one-column
targets, fitted metadata, constructor-only clones and typed search controls.
Predictive uncertainty follows the raw Source expression without clipping.
ARD remains unweighted, as in Source. MIT licensing and BSD notices are retained.

An isolated checked probe covers 152 Source configurations, including tall,
square, wide, rank-deficient, two-sample, zero/constant-target, constant-design,
large-offset, zero-prior, zero-threshold and iteration-limit cases. Eight strict
failures remain at 54,59,66,71,102,107,114,119: centered wide/two-sample Woodbury
fits retain coefficient, precision, prediction, covariance/uncertainty and/or
history differences from Source LAPACK paths. Iteration counts and support
sizes match. No tolerance or oracle was relaxed. All sixteen actual Source/native
input-rejection cases pass. Full checked rebuilding and scoped optimization
are running; no final package verification is claimed at this commit.

A snapshot of the rebuilding fpm archive was detected to contain only the
8-byte archive header. It was replaced before any optimized link by the
previous complete baseline verified against its recorded SHA256. Final evidence
will verify all non-overridden library members against the completed build.
Removed 62 generated Bayesian checked/optimized executable copies only after
matching every file hash to its archived entry and verifying the full preserved
Bayesian evidence ZIP SHA256. The archive and library objects remain intact;
the cleanup manifest is validation/results/ard_archived_binary_cleanup.json.
Validation retains the private GNU Fortran 15.2.0 setup, static libgfortran and
validation-only Windows locale-copy shim; installed tools and user files are
untouched. The larger translation goal and retained numerical gaps remain open.


### ARDRegression final verification and coherent library baseline

Implementation commit: dc78194. The completed checked build passes all 312
native programs and eleven selected examples. Its 329-job focused audit has
327 passes; ARD and BayesianRidge strict fitted comparisons intentionally exit
nonzero. All 912 build fingerprints match current sources, including 209
library units. The 152 ARD configurations retain exactly eight failures in
both profiles: 54,59,66,71,102,107,114,119. Fitted support sizes, iteration counts,
input acceptance and controlled termination match throughout. All sixteen
Source/native ARD rejection cases and three selected legacy guards pass.
No fixture, oracle or tolerance was relaxed.

Both profiles preserve the previous BayesianRidge Source inputs/oracles and
its 186-configuration / 26-failure inventory; all twenty Bayesian input guards
pass. IterativeImputer passes all 324 fixtures / 2,592 groups in each profile,
with identical Source file hashes. Scoped optimization covers eight modules,
seven passing native programs and six reference suites. This is not a full
optimized package rebuild. Cross-profile verification confirms Source/object/
report hashes and all 201 non-overridden library members.

The first optimized baseline, although a complete prior archive, was not
coherent with the newly compiled type metadata. Direct member comparison found
31 differing old objects, including generated dispatch/type metadata in otherwise
unchanged sources. This caused two native failures: the Bayesian sparse test
reached an OLS-overload rejection and the search test reported an unfitted
forest. Both pass in the completed checked build and after optimized relinking
against its exact 209-member library, without changing implementation or tests.
The affected audit was stopped explicitly; its old archive, two failing binaries
and diagnostics are preserved in ard_stale_baseline_failure.zip with verified
CRCs/hashes. The final optimized helper now requires a successful terminal
checked build and a byte-identical baseline before compiling/linking. Prior
archive completeness alone is insufficient when derived-type metadata changes.

All build/audit jobs are terminal. The protected user files remain untouched;
the 62 archived executable copies removed earlier totaled 884,593,630 bytes.
Validation continues to use the private GNU Fortran 15.2.0 environment, static
libgfortran and validation-only Windows locale-copy shim. Runtime behavior
without that shim is not claimed. Evidence is in ignored validation/results/
ard_*. Commits remain local; strict numerical gaps and the larger sklearn
translation remain open.

The verified archive is ard_evidence_dc78194.zip,
147,811,161 bytes / 2,017 entries, SHA256
ecb2496642335a3a0b56bb4c9d46a33a846ab3b145f9254f179237a2c1643c15.
ZIP CRCs and critical source/report/object/binary hashes pass. The receipt is
validation/results/ard_archive_receipt.json (ignored local evidence).


### QuantileRegressor CSR designs, rank adapters and weight contracts

Added direct CSR fitting/prediction, dense/CSR vector and one-column targets,
scalar/vector fitting weights, and optional weighted R2 scores. Unsorted
stored columns, duplicate entries and explicit zeros are accumulated directly
into the existing simplex tableau without constructing a separate dense design.
Sparse prediction traverses stored entries. The Bland-rule simplex and its
dense O(n*(p+n)) tableau remain the native solver; HiGHS-scale sparse LP
performance is not claimed. Existing native tol/max_iter and certificate
attributes remain explicitly documented extensions.

Actual sklearn 1.6.1 accepts all-zero fitting weights and returns a zero model
with zero pivots. Implemented that result and removed only the obsolete
qr_weights_zero rejection expectation; the driver case remains callable as a
valid case. Weight-sum overflow now reaches a controlled finite-sum diagnostic
under floating-point traps, using IEEE-safe accumulation. Dense and CSR finite
input/shape checks remain in place. MIT licensing and Source BSD attribution
are retained; no HiGHS code is linked or adapted.

An isolated checked probe passes all 240 configurations / 3,720 groups against
Source, including twelve dense/CSR, vector/column, scalar/vector/default-weight
adapters across five designs and four quantile/penalty/intercept settings.
It covers all-zero weights and noncanonical CSR storage. Nonunique fits retain
the established objective/primal-dual comparison policy, while unique fits
also compare coefficients and predictions. All sixteen actual Source/native
CSR rejection cases pass. A new native test passes a wide sparse fit and a
40,000-by-4,000 query (1.28 GB if dense), plus input preservation, certificates,
weighted scoring and zero-weight reset. Full package rebuilding is in progress;
final checked/optimized and memory validation remains outstanding at this commit.
The legacy dense validator adds only an optional report destination and Source
input/oracle records; fixtures, comparisons and tolerances are unchanged.


### QuantileRegressor CSR final verification

Implementation commit: 6283d00. The completed checked package passes all 328
audit jobs: 313 native programs, eleven selected examples, three focused
reference suites and one memory check. All 915 build fingerprints match
current sources, including 209 library units. The scoped optimized build
recompiles the changed quantile module and passes both selected native tests,
all three reference suites and its memory check. This is not a full optimized
package rebuild. Its baseline is copied from the terminal completed checked
library and verified byte-for-byte before linking.

Both profiles pass all 240 new configurations / 3,720 groups, the original
120 dense configurations / 2,089 groups, and sixteen actual Source/native CSR
rejection cases. Source inputs/oracles agree across profiles; the new results
also agree with the isolated probe. Legacy fixtures, expected comparisons and
tolerances are unchanged. Nonunique solutions continue to use objective and
primal-dual checks; they do not imply identical arbitrary coefficients. All
22 remaining legacy QuantileRegressor guards pass. The retired zero-weight
case returns successfully, matching actual Source behavior. Additional checked
probes confirm controlled Source/native failures for overflowing weight sums
and overflowing duplicate CSR accumulation under floating-point traps.

The 40,000-by-4,000 query test peaks at 11,616,256 bytes checked / 11,046,912
bytes optimized, below its 1.28 GB dense representation. This demonstrates
sparse query storage; it does not remove the native fitting tableau's dense
O(n*(p+n)) cost. Cross-profile verification confirms all 208 non-overridden
library members and source/object/report hashes. All build/audit jobs are
terminal. Validation retains private GNU Fortran 15.2.0, static libgfortran and
the validation-only Windows locale-copy shim; unwrapped runtime stability is
not claimed. The broader project's previously documented numerical gaps remain
open, despite these focused quantile comparisons passing. Commits remain local.

The verified backup is quantile_csr_evidence_6283d00.zip,
47,340,609 bytes / 1,997 entries, SHA256
5652af1635467150504ba5ba99daf36eba8f39cef3c33226b126b2cc441b6cf0.
ZIP CRCs and critical source/report/object/binary hashes pass. The receipt is
validation/results/quantile_csr_archive_receipt.json (ignored local evidence).


### QuantileRegressor clones, CV, search and composite dispatch

Added QuantileRegressor to shared regressor recognition, weighted fitting,
prediction and fitted-state dispatch. Constructor-only cloning preserves its
five controls and drops learned coefficients, dual variables and diagnostics.
Typed search supports Source quantile/alpha/fit_intercept and the documented
native tol/max_iter extensions. Dense Pipeline final-estimator dispatch now
accepts weighted QuantileRegressor fits. A dedicated cross_val_score overload
fits a fresh constructor clone per KFold split, accepts scalar/vector fitting
weights and returns unweighted held-out R2, matching Source default routing.
Dense vector targets are supported here; direct CSR/column estimator methods
remain available separately. MIT licensing and Source BSD notices are retained.

An isolated checked comparison passes all 96 configurations across three
seeds, grid/randomized search, direct/Pipeline models, R2/MSE scoring and four
fit-weight modes (default, vector, scalar and all zero). It checks standalone
CV scores, explicit candidate parameter order, split/mean/std/train scores,
ranks, selected candidate, refit predictions, caller preservation and clone
reset state. All-zero scores cannot conceal incorrect sampled candidate order.
The new native integration test also passes. Full checked rebuilding and final
optimized/package validation are in progress at this commit.

Early partial-module probes exposed GNU module-cache coherence diagnostics
when mixing updated search metadata with cached facade/validation imports.
The probe now compiles its search dependency and uses explicit imports; those
diagnostics and logs are retained. Final validation will use the completed
coherent fpm library, including a separate public-facade CV smoke check.
Existing Source fixtures/tolerances are unchanged. The broader translation
and previously documented numerical gaps remain open; commits stay local.


### Legacy quantile validator file isolation

Concurrent checked and optimized legacy comparisons exposed shared temporary
input/output filenames in compare_quantile_regression.py. The optimized run
read 43 prediction rows while checking a 57-row fixture; this is a test-harness
race, not an estimator comparison result. All other optimized checks and the
checked comparisons completed, but neither overlapping legacy run is accepted
as independent profile evidence. Their initial audit/report/logs and the final
shared files are preserved under quantile_integration_*.

When --report is provided, the validator now derives its temporary input/output
paths from that report, isolating profiles. Default paths for old invocations
are retained. Fixtures, oracle values, comparison criteria and tolerances are
unchanged. Both legacy profiles will be rerun with isolated files; the scoped
optimized audit will be rerun and the checked audit will explicitly record
its replay and validator hash. Fortran sources and the completed build are
unchanged by this harness correction.


### QuantileRegressor integration final verification

Fortran implementation: ad80b80; validation file-isolation correction: 174b2ed.
The completed checked build passes all 314 native programs and eleven selected
examples. Its 330 defined audit jobs pass, with 331 actual checked executions
including the explicit isolated legacy replay. The initial audit and replay
provenance are retained; the overlapping legacy result is not accepted as
independent evidence. All 917 build fingerprints match current sources,
including 209 library units. No Fortran rebuild was needed for the Python-only
file-isolation correction.

The scoped optimized build covers seven modules, seven passing native programs
and five focused reference/memory jobs, all passing after the isolated rerun.
Both profiles pass 96 new clone/CV/grid/randomized/Pipeline configurations,
240 dense/CSR configurations / 3,720 groups, 120 legacy dense configurations /
2,089 groups, and sixteen Source/native CSR rejection cases. All 22 selected
legacy guards pass. The public use-sklearn facade smoke test accepts the new
scalar-weight cross_val_score overload. Candidate order, scores, ranks, refits,
constructor resets and caller preservation agree with actual sklearn 1.6.1.
Fit weights and unweighted held-out scoring follow its default routing.

The corrected legacy validator uses isolated per-report files. Source inputs
and oracles agree across fresh checked/optimized runs. Some earlier JSON reports
and the previous quantile CSR ZIP/receipt are absent on disk, including the
initial 96-case probe JSON; no availability or byte-preservation claim is made
for those missing artifacts. Instead, Git verifies that the original CSR
validator, CSR driver and quantile solver match 6283d00 exactly, and that the
legacy validator differs only in its two temporary-file paths. The new
integration fixture/driver also match ad80b80. Current comparisons rerun Source
and do not depend on unavailable historical outputs. Their recorded Source
hashes agree across profiles. The missing-artifact inventory and Git checks are
preserved in quantile_integration_historical_evidence.json.

Cross-profile checks pass source/object/report hashes, explicit replay hashes,
and all 202 non-overridden library members. The optimized baseline is verified
identical to the terminal completed checked library before linking. This is a
scoped optimized build, not a full optimized package rebuild. The wide-query
memory checks stay below 128 MiB for a 1.28 GB dense equivalent; fitting still
uses its documented dense simplex tableau. All build/audit jobs are terminal.
Validation retains the private GNU Fortran 15.2.0 setup, static libgfortran and
validation-only locale-copy shim; unwrapped Windows runtime stability is not
claimed. Evidence is under ignored validation/results/quantile_integration_*.
The broader translation and prior numerical gaps remain open; commits are local.

The newly verified archive is quantile_integration_evidence_174b2ed.zip,
115,305,671 bytes / 2,044 entries, SHA256
8b5d4100258643012ca35f274d0f6c7ed8e99534656679fd1223fbeaee2cac87.
ZIP CRCs and critical source/report/object/binary hashes pass. The receipt is
validation/results/quantile_integration_archive_receipt.json. This backup
contains current evidence and the preserved race diagnostics; it does not
restore the unavailable historical JSON/ZIP artifacts described above.


### Fresh-fold regression CV and fitting weights

Confirmed an existing cross_val_score defect: its linear-regression family
copied the caller once and reused that model across folds. With a previously
fitted warm-started caller and one solver iteration, Poisson scores differ
from cold CV by up to 2.6344492234931947 and Huber by 0.8575129053732354.
The before-fix program, executable and output are retained under cv_clone_*.

Each regression fold now allocates a separate copy and resets supported
warm-started estimator state using the existing constructor-only adapter.
Huber/GLM controls, including warm_start itself, are preserved; fitted caller
coefficients and preceding-fold state cannot seed optimizers. The caller is
unchanged. Added scalar/vector training-weight overloads for regression-family
models with native weighted fit; held-out scoring remains unweighted and uses
the estimator's own score (R2 or GLM deviance). Unsupported weighted estimators
retain controlled diagnostics. The original unweighted function signature is
preserved. Logistic CV also allocates a separate model for each stratified fold.

An isolated native regression test passes warm/cold invariance, caller-state
preservation and scalar/vector fit-weight equivalence. Actual sklearn 1.6.1
comparisons pass all 144 combinations of Huber/Poisson/Gamma/Tweedie, budgets
1/4/50, warm flag on/off, fitted/unfitted caller and default/vector/scalar
weights. Full checked package rebuilding is in progress; final profiles remain
to be verified. No Source fixture/tolerance was relaxed. MIT licensing and
Source BSD attribution are retained; this is progress toward the broader goal.


### Fresh-fold CV final verification

Implementation commit: 7968d25. All 328 checked audit jobs pass: 315 native
programs, eleven selected examples and two focused Source comparison suites.
All 919 build fingerprints match current sources, including 209 library units.
The public use-sklearn facade matches actual Source OLS/vector-weight and
Ridge/scalar-weight CV scores. The original before-fix harness, linked against
the completed corrected library, now reports exactly zero warm/cold differences
for both Poisson and Huber; its earlier differences were 2.6344492234931947 and
0.8575129053732354. Before/after sources, executables and logs are preserved.

Checked and scoped optimized builds pass all 144 new Huber/GLM CV configurations
and all 96 existing quantile clone/CV/grid/randomized/Pipeline configurations.
Source inputs, oracles and primary file hashes agree across profiles. Fitted
caller state is preserved. Scalar/vector fit weights and unweighted held-out
estimator scoring match Source. No fixture, comparison criterion or tolerance
was relaxed. The optimized build recompiles only sklearn_model_selection,
passes six selected native programs and both Source suites. This is not a full
optimized package rebuild. Other optimizer implementations are unchanged.

Cross-profile verification passes all source/object/report hashes and all 208
non-overridden baseline library members. The private optimized baseline is
verified byte-identical to the terminal completed checked library before any
link. All build/audit jobs are terminal. Validation retains private GNU Fortran
15.2.0, static libgfortran and the validation-only Windows locale-copy shim;
unwrapped runtime stability is not claimed. Evidence is under ignored
validation/results/cv_clone_*. The wider sklearn translation and documented
numerical/API gaps remain open; commits remain local.

The verified backup is cv_clone_evidence_7968d25.zip,
84,015,325 bytes / 2,013 entries, SHA256
b4359cace5fd47a25dd7ea7ee9e498742ecdb09138e90db049b70468eebe2307.
ZIP CRCs and critical source/report/object/binary hashes pass. Receipt:
validation/results/cv_clone_archive_receipt.json (ignored local evidence).


### Randomized KernelPCA eigensolving

Added eigen_solver='randomized' using the existing seeded randomized-SVD core,
Source ten-oversample defaults and sign(diag(Vt*U)) eigenvalue recovery.
Selected eigenvalues undergo Source PSD validation, deterministic orientation
and descending ordering before zero filtering and transform/inverse fitting.
The dense path also rejects an entirely negative selected spectrum, as Source
does. Added iterated_power (nonnegative or -1 for Source auto) and nonnegative
random_state controls. Native defaults remain dense and seed 0; Source auto/
None and ARPACK routing are explicitly unsupported. Both paths retain a dense
centered Gram matrix; the new path avoids a full dense eigenvector solve.
MIT integration and Source BSD attribution are retained.

The isolated checked comparison passes all 274 cases over six kernels, dense/
randomized solvers, auto/0/1/3 powers, integer seeds, clipped/all components,
zero designs, transforms and learned inverses. Signed indefinite precomputed
selection is compared against actual Source acceptance/rejection. All twelve
Source/native input guards pass. Native tests pass seeded repeats, changed-seed
behavior, convergence toward the dense spectrum/coordinates and inverse shapes.

An initial test incorrectly assumed approximate fit_transform(X) equals
transform(X). The Source algorithm returns scaled approximate eigenvectors for
the former and a kernel projection for the latter; they can differ at low power.
The test now checks that observable approximation behavior and separately checks
high-power convergence. The differential validator compares both outputs to
Source without equating them. The first failing test log is retained; no Source
fixture or tolerance was relaxed. Full checked rebuilding and final scoped
optimized/legacy-inverse validation are in progress at this commit. The larger
translation and previously documented gaps remain open; commits remain local.


### Randomized KernelPCA final verification

Implementation commit: 06e0306. All 330 checked audit jobs pass: 316 native
programs, eleven selected examples and three focused comparison/guard suites.
All 922 build fingerprints match current sources, including 209 library units.
Both profiles pass all 274 dense/randomized Source cases and twelve new input
rejections. The unchanged legacy inverse validator passes 452 fixtures / 4,520
groups, 452 ridge residual certificates and fourteen rejections in each profile.
It includes dense kernels, clipped/all/zero components and learned pre-images.
No Source fixture, comparison criterion or tolerance was relaxed.

Scoped optimization rebuilds the KernelPCA and randomized-SVD modules, passes
four selected native tests (including Isomap and the existing kernel tests),
and passes all three focused Source suites. This is not a full optimized package
rebuild. Source inputs/oracles and primary file hashes agree across profiles.
Cross-profile verification passes source/object/report hashes and all 207
non-overridden library members. Its baseline is a byte-verified copy of the
terminal completed checked library. The initial list/dictionary mismatch in
the audit helper was corrected; the original helper error log is preserved.

The new path follows Source randomized selection by eigenvalue magnitude,
including signed recovery and PSD rejection of indefinite selected spectra.
Approximate fit_transform coordinates and kernel-projected transform coordinates
are checked separately against Source. Native defaults remain explicitly dense
and seed 0; ARPACK/auto routing, sparse input and Source None RNG are not claimed.
Both solver paths retain dense Gram storage. Existing dense inverse behavior
and Isomap integration pass after the derived-type dependency rebuild.

All build/audit jobs are terminal. Private GNU Fortran 15.2.0 validation retains
static libgfortran and the validation-only Windows locale-copy shim; unwrapped
runtime stability is not claimed. Evidence is under ignored validation/results/
kernel_randomized_* plus current kernel_preimage_* reports. The larger sklearn
translation and previously documented gaps remain open. Commits remain local.

The verified backup is kernel_randomized_evidence_06e0306.zip,
39,586,825 bytes / 2,006 entries, SHA256
a3370d10a5fd3deb22b027f94fc7566661f909a8a1127869245ccb5110cad3cd.
ZIP CRCs and critical source/report/object/binary hashes pass. Receipt:
validation/results/kernel_randomized_archive_receipt.json (ignored evidence).


### CSR KernelPCA fits and mixed queries

Added dense/CSR fit and fit_transform overloads, and dense/CSR query transforms
after either fitted storage. Sparse fits own x_fit_csr_ and leave x_fit_
unallocated; dense/sparse refits clear the alternate cache. New fitted metadata
records original feature/sample counts. Shared centering/eigendecomposition
and projection preserve the dense and randomized paths. Kernel dispatch uses
bounded row scratch storage rather than a dense design; the Gram matrix remains
dense. Sparse inverse fitting and sparse precomputed inputs/queries are rejected,
matching actual sklearn 1.6.1 behavior. MIT/BSD attribution is retained.

The first 216-case differential probe exposed 72 strict failures. Source sparse
RBF and cosine row norms include stored duplicate entries, while the backend
had used coalesced logical rows for norms. Corrected the shared norm loops to
Source stored-entry semantics; dot products retain their existing logical sums.
The other failures were Source rejections of sparse precomputed KernelPCA calls,
which the first adapter accepted. Explicit matching restrictions now apply.
The initial failed report remains separate; no cases or tolerances were removed.
All 216 cases now pass across six kernels, two solvers, two datasets and nine
training/query storage combinations, including unsorted duplicates and zeros.

All twelve actual Source/native sparse-input guards pass. The native test passes
owned-cache checks, input preservation, dense/sparse refits and mixed queries.
A 1,000-by-50,000 CSR query (400 MB if dense) peaks at 9,912,320 bytes in the
isolated checked memory probe. Full checked rebuilding and scoped optimized,
legacy inverse and shared kernel-dispatch comparisons remain in progress at
this commit. The broader translation and documented gaps remain open; commits
stay local.


### Stored-entry cosine normalization order

An additional Source probe with two finite 1e308 duplicate CSR entries returns
cosine zero: normalization divides stored entries by an infinite row norm before
duplicate accumulation. Coalescing first would overflow and produce NaN after
division. The shared cosine path now normalizes each stored entry before logical
row accumulation, preserving Source operation order as well as its norm. Dense
rows are normalized before their dot product. A native regression asserts the
Source zero result. All 216 KernelPCA storage comparisons still pass.

The first edit applied normalization at the wrong loop and the native test
caught it; the probe log is preserved. It was corrected before validation.
These source/test changes occurred during the initial full build, so that
build will not be used as final evidence: a fresh incremental fpm rebuild will
follow its terminal completion. No Source fixture or tolerance was changed.

### CSR KernelPCA final verification

Implementation commits cb4c673, c83fd35 and e22d0e9 are verified against the
completed final checked build. All 317 native programs, 11 selected examples
and six focused reference/memory jobs pass (334 jobs). All 925 compiled source
fingerprints match the current worktree; the library contains 209 units.
The runtime configuration remains static libgfortran with the validation-only
Windows locale-copy shim; this does not establish unwrapped runtime stability.

A fresh scoped -O3 build covers kernel_pca, randomized_svd_core and
kernel_dispatch, using a byte-identical copy of the completed checked archive
(SHA256 73d2898d5fbf1188c801aaa78a33f26aa1d6a688ad1b120126c419b66c42648d).
All six optimized native regressions and six focused jobs pass. Cross-profile
checks confirm identical reference inputs/oracles, current source/object/report
hashes, and 206 unchanged archive members outside the three selected modules.
This is scoped optimized verification, not a full optimized package rebuild.
The first optimized invocation lacked the compiler runtime directory on PATH
and failed before compilation; the corrected invocation and original failure
logs are retained separately.

Both profiles pass all 216 sparse/mixed-storage configurations, 274 randomized
KernelPCA configurations, 12 sparse-input guards, and the existing 452 legacy
preimage fixtures / 4,520 groups / 452 ridge certificates / 14 rejections.
The shared dispatcher passes 2,005 configurations / 6,015 groups and 276 Source
rejection probes in both profiles, with maximum scaled error
3.3306690738754696e-16. No reference cases or tolerances were relaxed.

The final wide-query memory peaks are 12,890,112 bytes checked and 12,214,272
bytes optimized, below the 128 MiB limit. The query is 1,000 by 50,000 with two
stored entries per row; a dense query alone would occupy 400,000,000 bytes.
Training kernels remain dense and quadratic in training sample count.

Evidence: validation/results/kernel_sparse_package_audit.json,
kernel_sparse_optimized_audit.json and kernel_sparse_cross_profile_audit.json,
plus their focused reports and logs. Overall sklearn parity remains incomplete;
previously documented gaps remain open. MIT licensing and upstream BSD notices
are retained, and all commits remain local.

Verified evidence backup: kernel_sparse_evidence_e22d0e9.zip,
51,656,326 bytes / 2,059 entries, SHA256
537e4bac79717cc147decf6bf3ee414724faae94988bebeea3bcbb7a815117fc.
ZIP CRCs and critical source/report/object/binary hashes pass; receipt:
validation/results/kernel_sparse_archive_receipt.json (ignored evidence).

### IncrementalPCA CSR fits and batched queries

The preceding goal turn made verified progress: CSR KernelPCA implementation
and checked/optimized evidence were committed through bf0bbf4. Continued from
the authoritative clean tracked worktree, retaining the protected untracked
scratch files unchanged.

Added CSR fit/fit_transform and query overloads to IncrementalPCA, with the
Source dense-per-batch algorithm and matching sparse partial_fit rejection.
Finite stored data, nonempty shapes and query width are checked. Dense/CSR
refits reset the sample counter; input CSR buffers are not mutated. Duplicate
entries accumulate in stored order. Dense update/whitening formulas remain
the established native implementation, with its existing numerical limits.

The isolated checked native test and 324 actual sklearn configurations pass
at unchanged 3e-8 tolerances, covering all fitted outputs, projections/inverses,
whitening, batch/component settings, mixed storage and noncanonical CSR.
All twelve actual Source/native rejection checks pass. The million-by-50
query memory peak is 56,827,904 bytes against a 128 MiB limit and 400 MB dense
query equivalent. Full rebuilding is running; scoped optimized and complete
native/package audits remain pending. Commits stay local; MIT/BSD notices
are retained. The overall sklearn translation remains incomplete.

The legacy dense IncrementalPCA validator now accepts an explicit executable
and report path, preserving its default fpm route. Input/output scratch files
are derived from that report path so concurrent checked/optimized runs cannot
collide. The eight original dense cases per seed and all tolerances remain
unchanged. This lets both profiles verify the actual completed binaries
without triggering an unrelated fpm invocation during package audits.

### IncrementalPCA CSR final verification

Implementation 656acc8 and harness isolation 972b2b7 are verified against the
completed checked fpm build. All 318 native programs, 11 selected examples and
four focused jobs pass (333 jobs); 928 dependency-aware source fingerprints
match the current worktree, including all 209 library units. All four selected
optimized native programs and four focused jobs pass after rebuilding only
sklearn_incremental_pca with -O3 -funroll-loops. This is scoped optimized
verification, not a complete optimized rebuild. The copied checked archive
is byte-identical before/after copying, SHA256
4b6b35ec99fa73bebbd918ff8a9c5c1cdda2cf7d192a625beae8c7c0a9d1e5c5.
All 208 other archive members are unchanged.

Both profiles pass 324 CSR/mixed-storage configurations, twelve Source/native
rejection checks and the unchanged 264 legacy dense result groups (eight
cases per seed, three seeds, eleven fields). Reference inputs and Source
oracles match across profiles; source/object/report hashes are verified.
No cases or tolerances were removed or relaxed. The final million-by-50
sparse-query memory peaks are 59,256,832 bytes checked and 56,799,232 bytes
optimized, below 128 MiB and the 400,000,000-byte dense query equivalent.

Evidence: validation/results/incremental_sparse_package_audit.json,
incremental_sparse_optimized_audit.json and incremental_sparse_cross_profile_audit.json,
plus focused reports, build log and audit logs. The checked runtime remains
static libgfortran with the validation-only Windows locale-copy shim; these
results do not prove unwrapped runtime stability. The broader translation
and existing numerical/degenerate-data gaps remain open. Commits remain local.

Verified backup: incremental_sparse_evidence_972b2b7.zip,
48,216,112 bytes / 2,033 entries, SHA256
03a7df2125db1cf51b77380be667257d6e2e4803cdefc5767043595feac5ef86.
ZIP CRCs and critical source/report/object/binary hashes pass. Receipt:
validation/results/incremental_sparse_archive_receipt.json (ignored evidence).

### IncrementalPCA singleton and zero-variance fidelity

The preceding goal turn made verified progress through 82d1be8: CSR fits and
queries plus checked/optimized evidence were committed. Revalidated the clean
tracked worktree; protected untracked scratch files remain untouched.

Added a 192-case actual Source grid before edits. It recorded 162 strict failing
configurations, including rejected singleton first batches, zero-valued ratios
where Source has NaN, and shifted-data projection differences. The implementation
now accepts singleton initial fits/partial_fit, uses safe IEEE quotients for
explained variances/ratios, retains empty-spectrum noise NaN, corrects two-pass
batch variance and follows Source weighted-sum mean updates. Transform projects
before centering with the existing attributed scalar OpenBLAS reduction helpers.
Whitening preserves NaN scales and clips finite zero scales to epsilon.

The identical initial grid now has 176 passing configurations and sixteen
retained strict failures: 164,165,166,167,172,173,174,175,180,181,182,183,188,189,190,191.
They are all whitened large-offset projections; every fitted field passes the
unchanged 3e-8 tolerance. The intermediate IEEE/order probe and projection probe
are preserved separately. An initial generated driver declaration duplicated
mode; fixed before obtaining the baseline. Replaying Source with native fitted
parameters yields bit-identical train/query projections for all sixteen,
isolating the end-to-end residual to fitted-parameter last-bit differences and
their amplification. No fixture or tolerance was removed or relaxed, and strict
comparison continues to exit unsuccessfully on those differences.

The new native singleton, constant-data, CSR-singleton and streaming checks pass
under bounds/IEEE traps. Full checked rebuilding is running. Scoped optimized,
complete native and legacy dense/CSR reference audits remain pending. MIT/BSD
attribution is retained; commits remain local. Overall Source parity is open.

During native-test review, the pre-existing IncrementalPCA test was found to
assert zero ratios on constant data. That assertion encoded the old deviation
from Source. Updated it to require NaN ratios while retaining zero population
variance; the independent new native regression and actual Source grid prove
the expected behavior. No numerical tolerance was changed. The audit monitor
was stopped before it launched, using its identified process 24264; the active
fpm build is retained. After that build terminates, a final incremental rebuild
will include the updated assertion before the audits run.

### IncrementalPCA boundary final verification

Commits 29c53ef and d976c02 are verified against the final completed checked
build. The initial full build and final incremental test rebuild are retained
separately. All 319 native programs and eleven selected examples pass, as do
four existing focused jobs. Of 335 completed checked jobs, 334 pass; the boundary
comparison exits 1 on its sixteen explicitly retained strict differences.
All 930 dependency-aware fingerprints match current sources, including all
209 library units. No additional native failures are hidden by the strict-gap
accounting.

The scoped -O3 build recompiles sklearn_incremental_pca with -funroll-loops.
All five optimized native programs and four existing focused jobs pass; the
new boundary job retains exactly the same sixteen strict failures (nine of ten
jobs pass). Its baseline archive is byte-identical to the completed checked
library, SHA256 d35159a7195be51f9a4d7387d66e7f5febfe58f568d1ebd94e7e6b968a1e9f20;
all 208 other archive members match. This is not a full optimized rebuild.

Both profiles pass the existing 324 CSR configurations, twelve rejection checks
and 264 legacy dense groups. All baseline/new and cross-profile boundary inputs
and Source oracles match, with unchanged 3e-8 absolute/relative tolerances. The
192-case boundary suite improves from 162 strict failures to sixteen in both
profiles. All fitted fields pass; the largest remaining absolute projection
difference is 1.3174680406624617e-6. Replaying all sixteen with native fitted
parameters in actual Source gives bit-identical train/query projections in
both final profiles. Full end-to-end equality remains unproven and is explicitly
false in the cross-profile audit. No tolerance or fixture was changed.

The million-by-50 CSR query peaks at 57,430,016 bytes checked and 59,080,704 bytes
optimized, below 128 MiB and the 400,000,000-byte dense query equivalent.
The runtime remains static libgfortran with the validation-only Windows
locale-copy shim; no unwrapped-runtime stability claim is made.

Evidence: validation/results/incremental_boundary_package_audit.json,
incremental_boundary_optimized_audit.json, incremental_boundary_parameter_replay_final.json
and incremental_boundary_cross_profile_audit.json, plus reports and logs.
The broader translation and retained numerical gaps remain open; MIT licensing
and BSD attribution are preserved, and commits remain local.

Verified backup: incremental_boundary_evidence_d976c02.zip,
51,917,652 bytes / 2,077 entries, SHA256
9ceea9a6bfb67c0ebda1fa4506dc4cab57719e2d2e5781ab68fa6608ceaf46f9.
ZIP CRCs and critical source/report/object/binary hashes pass. Receipt:
validation/results/incremental_boundary_archive_receipt.json (ignored evidence).
Archive metadata explicitly retains false strict/focused Source parity.
