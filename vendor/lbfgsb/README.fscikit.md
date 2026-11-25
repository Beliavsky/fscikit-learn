# Vendored L-BFGS-B numerical kernel

Sources from Jacob Williams lbfgsb 1.0.0, commit
bcb0ddc71ae1afdf5715ededaf9453aa704c2384. The original License.txt and fpm.toml
are retained. See the project THIRD_PARTY_NOTICES.md for authorship and scope.
PROVENANCE.json contains upstream file hashes after CRLF-to-LF normalization.
A local patch passes Iprint into lnsrlb so its ascent-direction diagnostic
respects Iprint<0. This print-control patch does not change numerical operations.
A second patch reconstructs real64 contiguous DDOT and DAXPY accumulation
from OpenBLAS SkylakeX, with retained OpenBLAS BSD notices. This changes
floating-point rounding to match the pinned SciPy backend more closely;
L-BFGS-B algorithm logic and termination tests are unchanged.
PROVENANCE.json retains upstream hashes and records the patched source hash.
This local fpm dependency supplies the optimizer for NCA and other estimators. It builds its own
Fortran BLAS/LINPACK helpers and requires no binary BLAS library or network.
