# Validation record

Verified locally on Windows on 2026-11-17 (America/New_York).

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
