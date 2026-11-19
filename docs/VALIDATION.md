# Validation record

Initial package verified locally on Windows on 2026-11-17 (America/New_York).

Toolchain: fpm 0.12.0 alpha; GNU Fortran 17.0.0 20260510 experimental;
Python 3.13; scikit-learn 1.6.1.

| Check | Observed result |
| --- | --- |
| `fpm test --flag "-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow"` | All 14 native test programs passed |
| `fpm test --profile release` | All 14 native test programs passed |
| `python validation/compare_sklearn.py` | 192 result-group comparisons passed across three datasets |
| `python validation/check_invalid_inputs.py` | All 16 expected input-error cases passed |
| `fpm run --example classification` | Training accuracy 1.0 on illustrative fixture |
| `fpm run --example regression` | Held-out R2 0.9998243, RMSE 2.01117 on synthetic data |
| `fpm run --example text_classifier` | Both example documents classified as expected (0, 1) |
| `fpm run --example nonlinear` | Held-out moons accuracy 0.93333; SVC converged in 95 iterations |
| Fortran formatting audit | No lines over 132 columns or trailing whitespace |

Example results describe small synthetic/demonstration fixtures, not general
performance guarantees. See `validation/README.md` for reference tolerance,
tie-handling fixtures, random-stream differences and the scope of the checks.
Only this compiler/platform combination has been tested in this session.

The package has 23 source modules and 56 public estimator/transformer/splitter
types. No SciFort or other third-party runtime dependency was required.

Validation was performed on the working tree before the initial commit. An
initial read-only `.git` restriction was lifted by the user on 2026-11-18,
allowing the tested package to be committed.

## Graph, neighbor and outlier increment (2026-11-18)

Added eight scikit-learn counterpart types: `RadiusNeighborsClassifier`,
`RadiusNeighborsRegressor`, `NearestCentroid`, `LocalOutlierFactor`, `KernelPCA`,
`Isomap`, `LabelPropagation` and `LabelSpreading`.

| Check | Observed result |
| --- | --- |
| Native tests in release profile | All 16 programs passed |
| Native tests with bounds checking and invalid/zero/overflow traps | All 16 programs passed |
| Original reference suite | All 192 comparisons still passed |
| Graph/neighbor reference suite | All 120 added comparisons passed |
| Input-error suite | All 26 expected-error cases passed |
| `fpm run --example semi_supervised` | 20 of 100 labels supplied; transductive fixture accuracy 1.0; converged in 11 iterations |

The reference total is now 312 comparisons against scikit-learn 1.6.1. The new
suite checks disconnected Isomap graph completion, precomputed kernel
transforms, shrinking mean/median centroids, radius weighting, LOF training and
novelty scores, RBF/KNN diffusion probabilities, and diffusion iteration counts
and convergence status. One tightly converged reference fixture reaches its
iteration limit; matching that status is part of the check, not a claim that
all fits converge. See `validation/README.md` for details.

The package now has 27 source modules, 64 public estimator/transformer/splitter
types, and five examples. These additions are core counterparts, not public
extensions. MIT project licensing and retained scikit-learn notices apply as
documented in `THIRD_PARTY_NOTICES.md`; no new runtime dependency was added.

## Bayesian regression, factors and projections (2026-11-18)

Added six core counterpart types: BayesianRidge, ARDRegression,
OrthogonalMatchingPursuit, FactorAnalysis, GaussianRandomProjection and
SparseRandomProjection. The package now has 29 source modules, 70 public core
types, 18 native test programs and six examples.

| Check | Observed result |
| --- | --- |
| Bounds-checked native tests with invalid/zero/overflow traps | All 18 programs passed |
| Release native tests | All 18 programs passed |
| Bayesian/factor/projection reference suite | All 204 comparisons passed |
| Original reference suite | All 192 comparisons still passed |
| Graph/neighbor reference suite | All 120 comparisons still passed |
| Input-error suite | All 35 expected-error cases passed |
| `fpm run --example bayesian_regression` | Bayesian and ARD held-out R2 approximately 0.99999658 on the synthetic fixture |

The three reference suites now provide 516 result-group comparisons against
the installed scikit-learn 1.6.1 baseline.

The new suite covers tall and wide inputs, coefficient pruning, posterior
uncertainties, intercept-free and capped fits, OMP stopping modes, zero-factor
models, full-SVD factor likelihoods, projection algebra and pseudoinverses.
Native tests additionally check all-pruned ARD, dependent OMP dictionaries,
Bayesian cross-validation and covariance on the identifiable row space of
centered wide data, projection distribution moments and sparse density.

The reference suite uses `atol=rtol=2e-7`, except for wide ARD (`1e-5`) because
the active-feature Cholesky and sklearn Woodbury/SVD inverses have different
roundoff. That fixture uses a fitting tolerance of `1e-4` rather than comparing
roundoff-dominated stopping at `1e-8`. Wide Bayesian covariance is compared on
full-row-rank uncentered data; see `COVERAGE.md` for the nonunique null-space
basis in centered wide fits. Random projection reference checks use the native
matrix as input and do not assert identical NumPy random streams.

The project MIT license and retained sklearn BSD-3-Clause attribution continue
to apply. No public extension or runtime dependency was introduced.

## Robust regression and GLMs (2026-11-18)

Added HuberRegressor, RANSACRegressor, PoissonRegressor, GammaRegressor and
TweedieRegressor. The package now has 75 public core types in 29 source modules,
20 native test programs and seven examples.

| Check | Observed result |
| --- | --- |
| Bounds-checked native tests with invalid/zero/overflow traps | All 20 programs passed |
| Robust/GLM Python reference suite | All 213 comparisons passed |
| Release native tests | All 20 programs passed |
| Original Python reference suite | All 192 comparisons still passed |
| Graph/neighbor Python reference suite | All 120 comparisons still passed |
| Bayesian/factor/projection Python reference suite | All 204 comparisons still passed |
| Input-error suite | All 47 expected-error cases passed |
| `fpm run --example robust_regression` (bounds-checked) | Clean-signal R2: OLS 0.45298, Huber 0.9999878, RANSAC 0.99999997; 85 consensus inliers |

All four reference suites now provide 729 result-group comparisons against
the local scikit-learn 1.6.1 baseline.

The new native tests exercise coefficient recovery under contamination, Huber
target scaling and constrained scale at epsilon one, deterministic RANSAC
consensus and dynamic trial limits, singular unpenalized GLMs, Tweedie powers
near one/two, power-zero/ridge penalty equivalence, positive predictions,
cross-validation with D2 scoring and iteration-capped fits.

Reference comparisons use three separately seeded datasets. Huber checks use
`atol=rtol=1e-4` because the sklearn L-BFGS-B relative-loss and native BFGS
projected-gradient stopping rules differ. The objective, scale, coefficients,
predictions and outlier masks are checked. RANSAC comparisons use a known
separated consensus and compare final refits, not random-stream trial counts.
GLM references use sklearn's Newton-Cholesky solver and compare coefficients,
mean predictions, D2 and convergence/cap status with `atol=rtol=3e-8`.

These are documented dense sklearn subsets. MIT project licensing, retained
sklearn BSD-3-Clause attribution and the absence of runtime dependencies remain
unchanged.

## Mode/spectral clustering and density (2026-11-18)

Added MeanShift, AffinityPropagation, SpectralClustering, SpectralEmbedding and
KernelDensity. The package now has 80 public core types, 31 source modules,
22 native test programs and eight examples.

| Check | Observed result |
| --- | --- |
| Mode/spectral/density reference suite | All 231 added comparisons passed |
| Bounds-checked native tests with invalid/zero/overflow traps | All 22 programs passed |
| Release native tests | All 22 programs passed |
| Existing four reference suites | All 729 comparisons still passed |
| Input-error suite | All 62 expected-error cases passed |
| Bounds-checked clustering/density example | Three modes; center/between-cluster log densities -0.563706 and -49.258346 |

The five suites now provide 960 result-group comparisons against sklearn 1.6.1.

The reference suite uses three datasets with two, three and four features,
checking modes and exact center ordering, orphan/query semantics, bandwidths,
iteration limits, AP exemplars and working similarities,
spectral affinities/embedding Grams/partitions, and five normalized weighted
density kernels. It uses `atol=rtol=3e-8`; spectral partitions also match known
clusters exactly under adjusted Rand agreement. Gram comparisons remove
eigenvector signs/rotations. Weighted compact-kernel queries have positive
support to avoid tree-roundoff residue at mathematically zero density.

Native checks verify density integrals and zero-density negative infinities,
weighted sampling moments, seed repeatability, spectral degree-weighted
orthonormality, large-self-loop invariance and disconnected components, automatic zero MeanShift bandwidth,
equal similarities and AP fits without exemplars. These are supported dense
subsets, not complete API translations. Scope and limits are recorded in
`COVERAGE.md`; MIT project licensing and retained sklearn attribution apply.

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
