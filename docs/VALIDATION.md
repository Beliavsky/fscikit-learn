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
