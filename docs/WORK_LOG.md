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
