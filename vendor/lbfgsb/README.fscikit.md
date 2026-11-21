# Vendored L-BFGS-B numerical kernel

Unmodified sources from Jacob Williams lbfgsb 1.0.0, commit
bcb0ddc71ae1afdf5715ededaf9453aa704c2384. The original License.txt and fpm.toml
are retained. See the project THIRD_PARTY_NOTICES.md for authorship and scope.
PROVENANCE.json contains upstream file hashes after CRLF-to-LF normalization.
This local fpm dependency supplies the optimizer for NCA. It builds its own
Fortran BLAS/LINPACK helpers and requires no binary BLAS library or network.
