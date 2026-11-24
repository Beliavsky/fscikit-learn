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

## Coordinate-descent NMF (2026-11-22)

All 150 native test programs passed in both checked mode
(`-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow`) and release mode.
Each profile passed 26,906 affected reference groups: 20,214 new CD NMF groups
from 2,022 retained fixtures, 6,500 existing MiniBatchNMF groups and 192 integration
groups. All 34 new isolated rejection cases passed per profile. The native
rank-one example ran explicitly through fpm in both profiles and matched sklearn:
two iterations, zero reconstruction error and component `[1,2,3]`.

The new suite compares public sklearn 1.6.1 fit/transform results and replays the
actual upstream compiled coordinate kernel to certify first/last projected
violations. It independently checks nonnegativity, reconstruction and the
regularized objective. Six zero-data cases retain source-matching all-zero-H
transform rejections. Sixteen stopping certificates per profile bound rounding
differences in exact-zero decisions or already stationary initialization using
both the arithmetic scale and independently computed projected gradients. No
fixture or factor/objective comparison was dropped.

The final audit matched cross-profile field inventories, executable/validator/
upstream hashes and all 287 relevant compiled-content digests per profile
(129 core modules, four vendor modules, 150 native tests, three reference drivers
and the example). It verified that existing non-NMF decomposition, legacy MU
arithmetic and MiniBatchNMF procedure bodies are unchanged. Reports, native logs,
example checks and orchestration/audit scripts are archived in ignored
`validation/results`, including `nmf_cd_final_audit.json`.

Cumulative inventory: 217 public core types, 129 core modules, 150 native test
programs, 128 core reference suites and 837,309 retained groups. Thirty-seven
invalid-input registries retain 3,551 cases. Only the affected suites listed
above were rerun; these totals do not imply a full reference rerun. The separate
42-group legacy RNG suite remains excluded and was not rerun. MIT licensing and
retained sklearn/NumPy BSD attribution continue; no extension algorithm or
runtime dependency is added.

## Beta-divergence NMF multiplicative updates (2026-11-22)

Both corrected profiles passed all 151 native test programs: checked mode used
`-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow`, and release used fpm's
release profile. Each passed 47,750 affected reference groups: 20,844 new MU
groups, 20,214 CD regressions, 6,500 MiniBatchNMF groups and 192 integration
groups. All 44 new MU and 34 existing CD rejection checks passed per profile.
The CD registry now checks non-Frobenius-loss rejection in place of the earlier
regularized-MU rejection, since regularized MU is now supported.

The MU sweep retains 2,090 fixtures covering six initializers, nine primary beta
losses, regularization, one/thirteen/converged iteration limits, fixed-H queries,
auto/overcomplete counts, zero factors/entries, exact fits and epsilon-scale data.
Public sklearn results and the actual upstream MU update/stopping functions are
checked, along with independent objectives, nonnegativity and reconstruction.
All 56 source-matching zero-H transform failures are retained.

A targeted counterexample found powered results incorrectly clipped before
rounding to a positive subnormal. After fixing the implementation, both complete
profiles were rebuilt and rerun. A native regression and 24 boundary fixtures
across beta 2.5, 3 and 4 use zero absolute tolerance and 3e-12 relative tolerance
for W/H, preserving tiny nonzero coordinates. Initial pre-fix results are archived
separately and are not used by the final audit.

Six MU diagnostic certificates per profile bound cancellation in source
square-root divergence reporting and independently compute the final mathematical
divergence using 60-digit Decimal arithmetic from exact input floats. Sixteen
existing CD stopping certificates also pass per profile. No fixture, factor
comparison or objective comparison was dropped.

Both explicit fpm KL example runs matched sklearn: 30 fit iterations, square-root
divergence approximately 0.078976237024 and fixed-H code sum 10.859084336789.
The final audit verifies full cross-profile field inventories, executable/
validator/upstream hashes and 290 compiled-content digests per profile: 130 core
modules, four vendor modules, 151 native tests, four reference drivers and the
example. It also verifies unchanged CD numerical kernels, non-NMF decomposition
procedure bodies and MiniBatchNMF source. Reports, terminal logs, example checks
and orchestration/audit scripts are archived in ignored `validation/results`,
including `nmf_mu_final_audit.json`.

Cumulative inventory: 217 public core types, 130 core modules, 151 native tests,
129 core reference suites and 858,153 retained groups. Thirty-eight invalid-input
registries retain 3,595 cases. Only the affected references above were rerun;
cumulative totals do not imply a full reference rerun. The separate 42-group
legacy RNG suite remains excluded and was not rerun. Original interfaces and
checked arithmetic remain MIT with retained sklearn/NumPy BSD attribution.
No public extension algorithm or runtime dependency is added.

## Positive linear-model constraints (2026-11-22)

All 153 native test programs passed in checked mode
(`-fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow`) and release mode.
Each profile passed 55,764 affected reference groups: 8,388 positive fit/CV,
1,608 positive composition, 216 existing coordinate-CV, 32,100 bagging,
8,556 pipeline, 4,704 transformed-target and 192 integration groups. All 30 new
isolated rejection cases passed per profile. Only these affected reference
suites were rerun.

The core sweep retains 1,044 fixtures across OLS, Ridge, ElasticNet/Lasso and
their CV wrappers, with intercept/no-intercept, ordinary controls, weights,
wide/duplicate/zero/constant/scaled columns and negative targets. Nonnegativity,
objective and projected KKT conditions are checked independently. The 192
composition fixtures check positive options and fitted coefficients through
pipelines, transformed targets, bagging, independent multioutput and chains.

Each profile retains one certified correction of the stale final QR-tail row
in SciPy 1.15.3, checked against independent BVLS and a deterministic native KKT
regression. Two nonunique NNLS cases retain null-image/optimality proofs and
record differing query extrapolations. Thirteen direct Ridge cases retain
strong-convexity bounds for public finite L-BFGS optimizer differences. All 216
composition Ridge children receive corresponding constrained-solution proofs;
96 parent prediction comparisons retain public and exact results. No fixture
or objective comparison was discarded. Numerical deviations are documented in
the API rather than described as unqualified coefficient parity.

The native wide-OLS test fits a two-by-10,000 design without the unconstrained
square-V allocation. Positive and unconstrained refits, projected coordinates,
CV refits, weighted fitting and template-preserving fitted cloning are tested.
Unsupported inherited flags and invalid domains/ranges terminate with diagnostics
without trapped arithmetic. Both explicit fpm example runs match sklearn:
OLS coefficients `[0,3]`, unrestricted intercept `2`, Ridge coefficients `[0,1.5]`.

The final audit verifies full core/composition field inventories, executable/
validator/source hashes and all 296 compiled-content digests per profile:
131 core modules, four vendor modules, 153 tests, seven reference drivers and
the example. It also verifies the unchanged unconstrained SVD procedure body,
ordinary weighted SVD arithmetic suffix and unaffected regressor fit bodies
apart from the unsupported-option guard. Reports, logs and audit/orchestration
scripts are archived in ignored `validation/results`, including
`positive_final_audit.json`. The initial public-Ridge comparison failure in
composition is retained separately; final composition reports use the explicit
certified constrained-solution oracle.

Cumulative inventory: 217 public core types, 131 core modules, 153 native tests,
131 core reference suites and 868,149 retained groups. Thirty-nine invalid-input
registries retain 3,625 cases. Cumulative counts do not imply a full reference
rerun. The separate 42-group legacy RNG suite remains excluded and was not rerun.
Original adapters, checked arithmetic and tests remain MIT with retained
sklearn/SciPy/LAPACK BSD attribution. No public extension algorithm or additional
runtime dependency is introduced.

## Free nonnegative matrix factorization

Added the dense sklearn 1.6.1 `non_negative_factorization` function, exported
through `sklearn_decomposition` and the facade. The adapter reuses the existing
CD/MU kernels and adds fixed-H output iteration counts without changing solver
arithmetic. It preserves input arrays and returns separately owned W/H arrays.
The native defaults are CD, automatic rank, 200 iterations, `tol=1e-4` and seed
zero; Python's default seed remains process-state dependent. The existing
`nmf` type defaults are unchanged.

The new reference suite compares 916 fixtures / 3,664 groups and 12 diagnostics.
It covers initialization modes, both solvers, automatic/explicit ranks,
fixed/updated H, dimension-scaled regularization, shuffling, general beta losses,
overcomplete dictionaries, ignored nonfinite W and zero component columns.
Every fixture retains W, H, iterations and an independent objective plus
nonnegativity/fixed-H invariants. CD iteration differences at machine
stationarity require explicit projected-gradient certificates for both source
and native solutions; source and native iteration counts remain in the report.

Checked verification passes all 154 native tests and 44,914 affected reference
groups: 3,664 new free-function, 20,214 prior CD, 20,844 prior MU and 192
integration groups. All 12 new and 78 prior CD/MU diagnostic cases pass.
Fourteen new-suite iteration differences satisfy their roundoff certificates;
the largest W/H difference from source is about `3.1e-13`. The fixed-dictionary
example recovers codes `[2,1]`, `[1,3]`, `[4,2]` with a reconstruction error of
about `1.3e-10`. The checked audit verifies all report/executable/validator hashes,
full new-suite field inventories, 294 compiled-content digests and unchanged
existing decomposition arithmetic.

Release verification also passes all 154 native tests, the same 44,914 affected
reference groups and all 90 diagnostic cases. The optimized new suite retains
16 certified stationarity-level iteration differences and W/H errors below
`2e-13`. The release example matches sklearn. The final audit verifies 294
compiled-content digests per profile, full new-suite field inventories,
executable/validator/source hashes, all new stopping certificates and unchanged
existing decomposition arithmetic. Artifacts are archived in ignored
`validation/results`, including `nonnegative_final_audit.json`.

Cumulative inventory is 217 public types, 131 core modules, 154 native tests,
132 core reference suites / 871,813 retained groups and 40 invalid-input
registries / 3,637 cases. Only the affected suites above were rerun; the separate
42-group legacy RNG suite remains excluded and was not rerun.

## HashingVectorizer

Added sklearn 1.6.1's stateless HashingVectorizer in `sklearn_hashing_text`,
re-exported through the existing text module and facade. It supports ASCII
word/character/within-word character analysis; lowercasing; configurable n-grams
and positive minimum word length; pinned English/custom stop words; signed,
unsigned and binary counts; CSR output; explicit dense output; and L1/L2/no
normalization. The two document adapters distinguish exact dynamic strings from
right-padded Fortran character arrays. Empty documents are valid, and transform
works before fitting. Existing CountVectorizer/TF-IDF, FeatureHasher, MurmurHash
and CSR arithmetic are unchanged. No new runtime dependency was introduced.

The reference suite compares 1,779 fixtures / 14,232 groups to the public
HashingVectorizer and its built analyzers. It checks every ordered analyzer token,
exact CSR metadata/indices, numerical values, an independent reconstruction from
signed MurmurHash values, row norms and stateless/dense adapter invariants.
Native tests cover stored-zero collision binarization and sparse default width.
A CSR MultinomialNB example demonstrates downstream use with unsigned hashes.
All 318 English stop words, ASCII controls, long tokens, empty rows, short padded
words and maximum-integer n-gram requests are retained boundary cases.

The first development run caught GNU Fortran's checked-DO rejection when the
upper bound equals the maximum default integer, even with an immediate exit.
The char_wb n-gram loop now uses guarded explicit advancement, and the native
regression includes that maximum-bound request. The in-progress obsolete build
was stopped for this source correction; its log and the original failing
reference log remain archived separately. The corrected full development run
passes all fixtures, comparing 288,369 analyzer tokens exactly and normalized
values within `4.5e-16`. No tolerance exceptions or discarded fixtures are used.

Original Fortran adapters and tests remain MIT; sklearn's BSD terms and English
stop-list attribution are retained. Unicode analysis, accent/encoding options,
custom regex/callable analyzers and general text pipeline dispatch remain outside
the implemented interface and are explicitly documented.

Checked verification passes all 155 native tests, 17,369 affected reference
groups and 53 diagnostic cases: 14,232 new HashingVectorizer groups, 2,945 prior
feature-mapping/hash groups and 192 integration groups, plus 13 new and 40 prior
isolated diagnostics. The checked suite's stored numerical values match sklearn
exactly; row-norm identity errors are at most `4.5e-16`. CSR metadata and all
288,369 analyzer tokens match exactly. No numerical exceptions are accepted.
The example matches sklearn's `(4,1024)` CSR result with 16 stored entries and
predicted classes `[0,1]`.

The checked audit verifies 296 compiled source-content digests (132 core modules,
four vendor modules, 155 native tests, four drivers and the example), pinned
English stop words, complete result-group inventories, analyzer fingerprints,
source/validator/executable hashes and unchanged existing text/hash/CSR arithmetic.

Release verification passes the same 155 native tests, 17,369 affected reference
groups and 53 diagnostics. Stored values and every analyzer token match sklearn
exactly; row-norm identity errors remain below `4.5e-16`. The release example
matches the checked example and sklearn. The final audit verifies all 296
compiled-content digests per profile, complete group inventories, all analyzer
fingerprints, source/validator/executable hashes, the pinned English stop list,
and unchanged existing text/hash/CSR arithmetic. Artifacts are archived in
ignored `validation/results`, including `hashing_text_final_audit.json`; the
initial boundary failure and interrupted pre-fix build logs are retained.

Cumulative inventory: 218 public core types, 132 modules, 155 native test
programs, 133 core reference suites / 886,045 groups and 41 invalid-input
registries / 3,650 retained cases. Only the affected suites above were rerun.
The separate 42-group legacy RNG suite remains excluded and was not rerun.

## Normalizer and Binarizer

Implemented dense/CSR Normalizer and Binarizer, their free functions and typed
dense composition integration, following sklearn 1.6.1 `_data.py` and sparse
kernels. The free normalization function adds both axes and returned norms;
CSR output corresponds to source CSC converted back to CSR for column operations.
Dense near-zero scales use the source `10*epsilon` guard; sparse nonzero scales
do not. Sparse L1/L2 operate on stored entries, max canonicalizes duplicates,
and binarization removes zeros while retaining qualifying duplicates. Returned
sparse L1/L2 norms remain unsupported, matching source. Fitting records width
and optional names; transform can precede fit, but feature-name output requires
fit. Dense protocol inheritance integrates both estimators with Pipeline,
FeatureUnion and ColumnTransformer without modifying their dispatch arithmetic.

The shared base dispatcher now forwards the already implemented discrete NB
sample weights and recognizes that capability. New cases verify MultinomialNB,
BernoulliNB and ComplementNB in weighted/unweighted pipelines, bagging and voting.
Separate multioutput/chain adapters are unchanged. No new runtime dependency
or public extension algorithm was introduced. Original adapters are MIT with
retained sklearn/SciPy BSD attribution.

The core suite has 1,801 fixtures / 19,386 groups, 29 diagnostics and 50 explicitly
retained nonfinite-result fields. The composition suite has 864 fixtures / 5,328
groups. Inputs span dense/CSR, both axes, norms/thresholds, both fitted and
prefit transforms, width/name checks, zero/tiny/subnormal/large values, near-zero
scale boundaries, unsorted duplicates, cancellation and duplicate overflow.
NaN/infinity masks must match source. The development run corrected the fitted
feature-name guard and a harness use of optional unnamed pipeline metadata; it
also exposed the stale NB dispatch restriction, now corrected. No fixture is
discarded or numerical mismatch waived.

Checked verification passes all 157 native tests, 57,356 affected reference
groups and 130 diagnostics. The group total is 24,714 new core/composition,
8,556 pipeline, 3,360 column/union, 16,965 voting, 3,569 stacking and 192
integration. Diagnostics comprise 29 new, 58 existing pipeline and 43 existing
composition cases. Existing reference certificates remain retained by their
suites; the new suites require direct numerical agreement and matching IEEE
outcome masks. The checked example matches sklearn's normalized CSR row
`[0.6,0.8]` and weighted pipeline predictions `[1,1,-1,-1]`.

The checked audit verifies all 304 compiled-content digests (133 core modules,
four vendor modules, 157 tests, nine reference/rejection drivers and the example),
complete group inventories, scalar/array IEEE outcomes and pinned executable/
validator/sklearn/SciPy source hashes. It proves that the other preprocessing
procedure bodies are unchanged, and that shared base dispatch changes are
restricted to the discrete NB weight forwarding and capability flag.

Release verification passes the same 157 native tests, 57,356 affected reference
groups and 130 diagnostics. Both examples match sklearn. The final audit verifies
304 compiled-content digests per profile, complete independent group inventories,
matching finite/nonfinite source outcomes, pinned executable/validator/sklearn/
SciPy hashes and unchanged surrounding preprocessing/dispatch arithmetic. Logs,
reports and orchestration/audit scripts are archived in ignored
`validation/results`, including `stateless_final_audit.json`.

Cumulative inventory: 220 public core types, 133 source modules, 157 native tests,
135 core reference suites / 910,759 retained groups and 42 invalid-input registries
/ 3,679 cases. Only the affected suites above were rerun. The separate 42-group
legacy RNG suite remains excluded and was not rerun.

## PolynomialFeatures estimator and sparse expansion

The new estimator retains the original array-call API and adds degree ranges,
int64 counts/powers, lazy feature names, dense/CSR transformation and dense
composition. The old degree-12 cap is removed. Scope, allocation/index limits and
source corrections are documented in API.md. Original Fortran adapters remain
MIT with retained sklearn/SciPy BSD attribution.

The core validator passes 1,985 fixtures / 38,020 groups and 31 diagnostics;
composition adds 480 fixtures / 2,880 groups. The source reports preserve 984 raw
noncanonical outcomes, 48 empty metadata exceptions, 36 empty sparse-transform
exceptions and 243 nonfinite fields. Symbolic metadata for three huge one-feature
exponents and one requested-range source count helper are explicitly identified
oracle limits. Final reports require the complete fixture inventory starting at
zero, exact integer metadata and matching NaN/signed-infinity masks.

Checked verification passes all 159 native tests, 77,722 affected reference
groups and 161 diagnostics. Legacy preprocessing/composition, pipelines,
column/union and general integration suites were rerun. The explicit example
matches quadratic predictions `[0.75,1,2.75]`, sparse shape `(5,2)`, eight stored
entries and names `x0`, `x0^2`. The checked audit verifies 307 compiled-content
digests, full field inventories, hashes and unchanged surrounding code.

An intermittent internal-formatting failure exposed during fresh-process stress
checks was corrected by direct integer-to-text conversion for lazy names. Native
regressions cover empty/UTF-8 names and refits. The corrected checked and release
builds each pass all 159 native programs, 77,722 affected reference groups and
161 diagnostic cases, including 128 name/refit process repetitions per full core
report. A focused optimized final-source run additionally passed 1,000 repetitions.
The final audit verifies 307 compiled-content digests per profile, full fixture/
field inventories, hashes and unchanged surrounding preprocessing/dispatch code.
Both explicit examples match source. Reports and pre-fix logs are retained in
ignored `validation/results`, including `polynomial_final_audit.json`. Only
impacted suites were rerun; the separate legacy RNG suite remains excluded.

## Image/grid graphs and quiet-subtraction boundary verification

The new graph suite passes 1,832 fixtures / 14,653 groups and 13 diagnostic cases.
It compares source CSR structure/data/signs, dense views, stored-zero masks and
large sparse multiplication. Reports retain 560 raw source singleton-axis mask
exceptions and 352 nonfinite fields, with exact NaN/signed-infinity masks. The
contract identifies the consistent-mask public source oracle, dtype/output-class
limits and default-integer CSR limits.

The new subtraction suite passes 8,703 scalar pairs / 17,406 groups against NumPy,
matching every finite value bit and all non-NaN signs. Its anchor grid, random
finite bit patterns and explicit overflow-rounding cases expose 258 premature
infinities in the former shared guard. Scaling the extreme opposite-sign case
corrects the rounding boundary without changing floating-point halting modes.
Other quiet primitive/reduction procedure bodies remain unchanged.

The checked build passes all 161 native tests. Its public example matches the
20-node / 82-entry grid, three selected gradient nodes / five entries and matrix
`[[1,0,7],[0,4,0],[7,0,8]]`. The native sparse-scaling regression checks a
60,000-node grid with 299,000 entries without constructing dense adjacency.
Broader affected reference suites and release verification are recorded below.

Final checked and release verification each passes all 161 native programs,
276,126 affected reference groups and 308 diagnostics. Existing image-patch,
structured-feature, histogram-boosting, sparse/smoothed NB, positive-regression,
preprocessing, weighted-classifier/tree and pipeline/column/union/integration
suites were rerun because the subtraction helper is shared. Both examples match
source. The final audit verifies 321 compiled-content digests per profile,
complete new-suite field inventories, bit-exact finite subtraction, exact
nonfinite outcomes, graph-mask corrections, source/validator/executable hashes,
distinct optimized executables and unchanged surrounding procedure bodies.
Artifacts are retained in ignored `validation/results`, including
`image_graph_final_audit.json`. Only affected suites were rerun; the separate
legacy RNG suite remains excluded.

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
Canonical verification will be recorded below.

A release comparison exposed a dense large-phase accumulation difference:
for 1e154 inputs, separate product rounding changed a phase bit and its cosine.
A phase-level probe identified fused accumulation matching the installed BLAS
reference. Large finite dense phases now use explicit standard ieee_fma;
ordinary bounded phases retain MATMUL, and nonfinite bounds retain quiet IEEE
composition. A native regression checks this finite phase case. The unchanged
1,064-fixture optimized probe passes all 8,512 groups, 33 diagnostics and 522
nonfinite reference values. No fixture or tolerance was relaxed or removed.
