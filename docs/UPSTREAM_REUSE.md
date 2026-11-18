# Recommended upstream Fortran code

Reviewed 2026-11-18 against the repositories linked from
[Beliavsky's neural-networks and machine-learning section](https://beliavsky.github.io/Fortran-code-on-GitHub/#neural-networks-and-machine-learning).
These are engineering recommendations based on current project gaps and
inspected upstream documentation/source, not adopted dependencies or measured
performance claims. No candidate source has been integrated by this review.

Apply the [scope policy](SCOPE.md) to every selected feature, not just to each
repository as a whole. Core scikit-learn counterparts have priority; public
extensions require explicit identification in [EXTENSIONS.md](EXTENSIONS.md)
and separate `fsklearn_extra_*` modules outside `use sklearn`.

## Recommended order

| Priority | Project and attribution | License | Suggested use |
| --- | --- | --- | --- |
| 1 | [tlux](https://github.com/tchlux/tlux), T.C.H. Lux | [MIT](https://github.com/tchlux/tlux/blob/f8b4f7d888360d5a47ab8cf41fa526d1268121ff/LICENSE.txt) | Adapt `tlux/approximate/balltree/{ball_tree,fast_select,fast_sort,prune,swap}.f90` behind our nearest-neighbor API. This targets brute-force distance and insertion-sort bottlenecks. Keep brute force as the correctness oracle. |
| 2 | [neural-fortran](https://github.com/modern-fortran/neural-fortran), Milan Curcic and neural-fortran contributors | [MIT](https://github.com/modern-fortran/neural-fortran/blob/17d01b55582e5df19557d2d70e03211f9b1820d9/LICENSE) | Focused adaptations for the baseline MLP training capabilities, such as minibatch SGD/momentum/Nesterov and Adam. Other exposed optimizers, network families and extra model-I/O APIs require separate extension review. Keep our sklearn-facing estimator API. |
| 3 | [FortLearner](https://github.com/NaoMatch/FortLearner), NaoMatch / FortLearner Authors | [MIT](https://github.com/NaoMatch/FortLearner/blob/83a7c1ed7e1b3ed89a6d78c43eba24dafa4f6411/LICENSE) | Evaluate K-means++ sampling, chunked distance and Lloyd-update kernels for large dense K-means. Reuse selected kernels rather than the Python packaging framework. |
| Extension candidate, deferred | [roseNNa](https://github.com/comp-physics/roseNNa), Ajay Bati, Spencer H. Bryngelson and contributors | [MIT](https://github.com/comp-physics/roseNNa/blob/7a3f57b694ef2faa53e0fb97f623ad94b7c19d04/LICENSE) | ONNX-to-Fortran inference/export would be an explicitly labeled public extension, outside the core facade. |
| Alternative | [NN_MOD](https://github.com/thchang/NN_MOD), Tyler H. Chang | [MIT](https://github.com/thchang/NN_MOD/blob/1f2b8991900dda36b40d997da77d287c25541bac/LICENSE) | Compact real-kind-based optimizer/regression modules as an alternative or comparison implementation. Avoid including two complete neural frameworks without a specific need. |

The tlux license's copyright line is literally `Copyright (c) 2018` without a
name. Preserve it verbatim and add repository/author attribution separately.

## Required engineering work before adoption

- **tlux:** the inspected ball tree uses `real32`, feature-by-sample arrays and
  explicit OpenMP runtime calls. Our API is `real64`, sample-by-feature. Port
  kinds/layout rather than silently downcasting. Audit helper-module licenses.
  Compare exact searches with brute force on duplicates, ties, near-coincident
  points and high-dimensional data. Exact search is the core target; a newly
  exposed approximate-search algorithm requires an extension entry.
- **neural-fortran:** fpm and serial/coarray builds are documented. Inspected
  optimizer source uses default `real`, not our `dp`; precision and layout need
  an adapter or audited adaptation. Validate loss scaling, initialization and
  minibatch behavior before exposing baseline capabilities through our MLP
  types. Do not expose all upstream optimizers or layers automatically.
- **FortLearner:** the current default branch inspected here focuses on
  **K-means**. The directory's broader description does not establish that tree,
  SGD and other classical estimators are currently present. Its build uses
  Make/OpenBLAS/OpenMP rather than fpm. Check Windows support and transitive
  modules, then validate inertia, empty clusters and explicit-center parity.
- Pin source revisions, preserve attribution, adapt a small module set, and
  compare correctness/performance before changing defaults. No candidate has
  been built or benchmarked in this project during this review.

## Other projects in the section

| Project | Review decision |
| --- | --- |
| [ATHENA](https://github.com/nedtaylor/athena), Ned Thaddeus Taylor (MIT) | Alternative for CNN/graph/physics-informed work. The inspected fpm manifest brings coreutils, diffstruc and graphstruc. Defer for this sklearn-oriented package; avoid adding it alongside neural-fortran by default. |
| [FKB](https://github.com/scientific-computing/FKB), [fnn](https://github.com/cerea-daml/fnn) (MIT top-level licenses) | Consider for a specific Keras interchange/operator requirement. Their scope overlaps a neural-network backend. |
| [FTorch](https://github.com/Cambridge-ICCS/FTorch) (MIT) | Optional PyTorch/LibTorch bridge, with an external runtime. Not a replacement for native classical estimators. |
| [KohonenF08](https://github.com/khaors/KohonenF08), Oscar Garcia-Cabrejo (MIT) | Self-organizing maps are a deferred public extension candidate, recorded separately in the extension inventory. |
| [gpf](https://github.com/ots22/gpf) (Apache-2.0) | Potential separately licensed GP backend, with LAPACK/NLopt dependencies. Do not relabel copied code MIT. Defer while extending our existing GP estimator. |
| [ENNUF](https://github.com/MetOffice/ennuf), [fortran-flann](https://github.com/ivan-pi/fortran-flann) (BSD-3-Clause) | Permissive alternatives with their own retained terms. FLANN also requires an external C++ library; prefer native search first. |
| [Inference-Engine](https://github.com/BerkeleyLab/inference-engine), now [fiats](https://github.com/BerkeleyLab/fiats) | The link redirects and license metadata is not a simple MIT identification. Defer pending file/dependency license review. |
| [ParallelForest](https://github.com/bert9bert/ParallelForest) | Inspected repository metadata did not identify a license. Establish the relevant package/file license before copying. |
| GPT/LLaMA/RWKV ports, chemistry-specific packages and framework bridges | Deferred extension territory; admit only for a concrete need and keep any public API explicitly separate from the core facade. |

Top-level license observations above were checked against upstream GitHub
metadata/license files on the review date. They are not substitutes for checking
the exact files and dependencies selected for incorporation.

## License and attribution policy

Original project contributions now use **MIT**. Existing scikit-learn-derived
portions retain **BSD-3-Clause** terms and notices. The MIT project license does
not erase upstream obligations. See `LICENSE`, `LICENSES/scikit-learn-COPYING`
and `THIRD_PARTY_NOTICES.md`. Primary explanations/texts:
[MIT](https://choosealicense.com/licenses/mit/) and
[BSD-3-Clause](https://choosealicense.com/licenses/bsd-3-clause/).

For each actual import or adaptation:

1. Record repository URL, immutable revision, original paths, authors/copyright
   holders and modifications in `THIRD_PARTY_NOTICES.md`.
2. Retain the exact upstream license under `LICENSES/` and preserve source
   copyright/license headers. Add source-level provenance and paper citations.
3. Keep the imported implementation behind our public wrapper, and ship the
   required notices in source/binary distributions.
4. Audit file-level exceptions and transitive dependencies. A top-level MIT
   badge does not establish the license of every bundled file.

No new runtime dependency or candidate source was added by this review. These
immutable snapshots identify inspection points, not adopted dependency pins:

| Repository | Inspected revision |
| --- | --- |
| tchlux/tlux | `f8b4f7d888360d5a47ab8cf41fa526d1268121ff` |
| modern-fortran/neural-fortran | `17d01b55582e5df19557d2d70e03211f9b1820d9` |
| NaoMatch/FortLearner | `83a7c1ed7e1b3ed89a6d78c43eba24dafa4f6411` |
| comp-physics/roseNNa | `7a3f57b694ef2faa53e0fb97f623ad94b7c19d04` |
| thchang/NN_MOD | `1f2b8991900dda36b40d997da77d287c25541bac` |
| nedtaylor/athena | `4b81aeac07113628c2c9a476648d5ac2a91b7c91` |
