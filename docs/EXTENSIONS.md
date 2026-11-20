# Public extension inventory

**Implemented public extensions: none.** All public estimator, transformer
and splitter types have a scikit-learn counterpart. Their
supported subsets and behavioral differences are documented in
[COVERAGE.md](COVERAGE.md); these differences are not additional algorithms.

This inventory covers public algorithms and integrations beyond the declared
scikit-learn baseline. Internal numerical helpers and faster implementations
of core algorithms are tracked through coverage/provenance instead. See the
classification and admission rules in [SCOPE.md](SCOPE.md).

## Candidates, not implemented

| Candidate | Possible upstream source | Status | Reason for separate identification |
| --- | --- | --- | --- |
| Self-organizing maps | KohonenF08 | Deferred | Additional estimator outside the current baseline |
| ONNX-to-Fortran export/inference | roseNNa | Deferred | Additional deployment integration |
| CNN/transformer networks | neural-fortran or ATHENA | Deferred | Additional network families beyond the baseline MLP estimators |
| PyTorch/LibTorch bridge | FTorch | Deferred | Additional framework integration and external runtime |

No candidate above is a dependency, vendored implementation, supported feature
or commitment to implement it. Upstream license observations and inspection
revisions are recorded in [UPSTREAM_REUSE.md](UPSTREAM_REUSE.md).

## Required record for each admitted extension

Use a separate entry containing all of these fields:

| Field | Required content |
| --- | --- |
| Name and status | Proposed, experimental, supported or deprecated; specify whether code is present |
| Purpose | Concrete benefit and reason for extending the core scope |
| API | Explicit `fsklearn_extra_*` module, public symbols and relevant options |
| Baseline distinction | Why it is an extension and how it differs from related scikit-learn behavior |
| Origin | Original implementation or upstream repository, immutable revision and source paths |
| Attribution and license | Authors/copyright holders, retained license files, notice entry and modifications |
| Dependencies | Runtime/build requirements and how the default build remains independent of optional runtimes |
| Validation | Relevant tests, examples, reference/benchmark methodology and known limits |
| Maintenance | Maintenance responsibility and compatibility expectations |

An admitted extension must be labeled in its API documentation and examples,
remain outside the default `use sklearn` facade, and be counted separately from
scikit-learn coverage. Do not claim scikit-learn parity for extension-only output.
