# Project scope

Scikit-learn algorithms and their documented Fortran adaptations are the core
of fscikit-learn. Selective public extensions are allowed when they provide
clear value, have appropriate tests and attribution, and have a manageable
maintenance cost. An imported repository's contents do not automatically become
our public API.

The current compatibility baseline is **scikit-learn 1.6.1**, used by the local
reference validation. A future baseline change must be recorded explicitly.
Being a core algorithm means having a counterpart in that baseline; it does
not imply complete parameter or numerical parity. Current limits and behavior
differences belong in [COVERAGE.md](COVERAGE.md).

## Three categories

| Category | Definition | Public surface and validation |
| --- | --- | --- |
| Core | An estimator, transformer, metric, kernel, splitter or generator with a scikit-learn counterpart | `sklearn_*` modules and the `sklearn` facade. Identify the counterpart and supported options; compare with the reference implementation where applicable. |
| Internal implementation improvement | A numerical solver, search kernel, sorting routine, parallel backend or similar mechanism serving a core algorithm | Preserve the documented estimator contract and numerical tolerances. Keep implementation helpers private where practical; validate correctness and measure performance before claiming a speedup. |
| Public extension | A user-selectable algorithm, optimizer, estimator or integration without a counterpart in the baseline | Explicit `fsklearn_extra_*` modules, excluded from `use sklearn`. Register it in [EXTENSIONS.md](EXTENSIONS.md), document its purpose and differences, and provide independent tests. |

An implementation change can still alter results. Record meaningful deviations
such as tie handling, initialization, convergence rules and precision in the
coverage document; do not describe them as full scikit-learn equivalence.
A newly exposed optimizer or approximate-search mode without a baseline
counterpart is a public extension even if it is attached to a core estimator.

## Admission rules

Core coverage, correctness and existing documented gaps have priority. Before
adding a feature or importing code:

1. Identify its category and, for core work, the scikit-learn counterpart.
2. State the concrete benefit and avoid overlapping public frameworks.
3. Check precision, layout, dependencies, portability and maintenance burden.
4. Preserve the original license, copyright, source revision and attribution
   in accordance with [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
5. Add meaningful validation, examples and documentation for the supported
   behavior. Candidate code is not supported merely because it was inspected.
6. For an extension, add its inventory entry before exposing the API. Mark
   examples and results as extension-specific. Do not include extension results
   in counts of scikit-learn algorithms or claims of reference parity.

Prefer an isolated module or optional package for substantial extra runtimes.
Separate namespaces do not by themselves make an fpm dependency optional: the
default build must remain usable without an optional extension's external
runtime. Defer CNNs, transformers and framework bridges until a concrete need
justifies their integration.

## Examples of the distinction

- An exact ball-tree backend for core neighbor estimators is internal work;
  exposing a novel approximate-search algorithm is an extension.
- Minibatch SGD/Adam support for the existing MLP estimators is core work;
  exposing optimizers beyond the baseline requires an extension entry.
- Faster Lloyd updates for K-means are internal work.
- Self-organizing maps are potential public extensions, not core coverage.
- ONNX code generation or a LibTorch bridge is potential extension integration.

The MIT license applies to original project contributions. Third-party terms
continue to apply to their respective portions, in both core and extension code.
