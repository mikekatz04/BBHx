# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

**LISA Analysis Tools–wide conventions:** [`../LISAanalysistools/docs/conventions.md`](../LISAanalysistools/docs/conventions.md) (canonical). **This repo's map:** [`docs/codebase-map.md`](docs/codebase-map.md) (layout, backend structure, cross-repo dependencies, known defects in §7). **Units, frames, mass ratio and reference-phase conventions of the BBHx classes:** [`docs/source/user/conventions.rst`](docs/source/user/conventions.rst).

## Role and current state (2026-10)

BBHx is the **MBH + SOBBH-physics owner** in the LISA Analysis Tools layered
architecture. LISAanalysistools (LAT) owns the generic LISA infrastructure:
orbits, TDI configuration, the time-domain TDI-on-the-fly response, WDM / FD
domains and sensitivity curves. BBHx owns:

- the MBH frequency-domain pipeline: PhenomHM / PhenomD
  (`waveforms/phenomhm.py`), the fast FD response
  (`response/fastfdresponse.py`), `waveformbuild.BBHWaveformFD`, and the
  likelihoods in `likelihood.py`;
- the time-domain TDI-on-the-fly generators `mbhtdionfly.MBHTDIonFly` and
  `sobbhtdionfly.SOBBHTDIonFly`. `MBHTDIonFly` is fed by the external
  `phentax` `IMRPhenomTHM` model; phentax is not part of BBHx;
- the SOBBH native source class and kernel group
  (`cutils/sobbh_tdi_on_the_fly.{hh,cu}`), the routing subclass
  `sobbhcomps.SOBBHWDMComputations`, and the JAX source
  `jax/sources/sobbh.py`;
- `utils/transform.py`, the SSB <-> LISA frame maps used by the LAT MBH
  sky-mode move.

**Production reach (9mo/12mo global-fit runs).** The runs reach BBHx only
through these paths:

- `import bbhx`, which registers the backends;
- `utils.transform.SSB_to_LISA` / `LISA_to_SSB` and
  `utils.constants.ConstOmega`, called from LAT
  `sampling/moves/skymodehop.py`;
- the module load of `waveformbuild`;
- `SOBBHTDIonFly`, built lazily as the SOBBH move's cross-check / fallback
  generator.

No BBHx waveform, response or likelihood kernel runs there.

**Build infrastructure:**
- One nanobind module per backend flavor,
  `bbhx_backend_{cpu,cuda11x,cuda12x,cuda13x}.cbbhx`, built from
  `cutils/binding_bbhx.{hpp,cxx}` (`NB_MODULE(cbbhx, m)`). It holds
  `BBHxComputationWrap` (the MBH bound methods), `SOBBHTDIonTheFlyWrap` and
  `SOBBHComputationGroupWrap`.
- The binding went from pybind11 to nanobind at Phase 3M (2026-06-04).
  `nanobind==2.12.0` is pinned exactly in the build-system requires. This is a
  sprint-wide invariant: bump it in one sprint-wide PR only.
- All former Cython modules (phenomhm, likelihood, response, waveformbuild,
  newhdyn) were migrated into `cbbhx`; the cubic-spline solver of the old
  `interp` module now comes from GBT (`gbt_backend_<flavor>.interp`). No
  `.pyx` is left.
- The binding consumes LAT's canonical wrappers through
  `#include "binding_flr.hpp"` and follows the single-registrant rule (below).
- `cutils/__init__.py` combines three sets of symbols into one `BBHxBackend`
  per flavor: LAT's `LISAToolsBackend` (`lisatools_backend_<flavor>.pycppdetector`),
  GBT's `CubicSplineWrap` (`gbt_backend_<flavor>.interp`) and `cbbhx`.
- `LISATOOLS_CUTILS` / `GBT_CUTILS` local consumption: the build
  copy-compiles LAT's `Detector.cu` / `lat_tdi_on_the_fly.cu` and GBT's
  `Interpolate.cu` from the installed packages. There are no URL downloads.
  The versions are pinned through the `lisaanalysistools` / `gpubackendtools`
  build dependencies.
- Constants come from `lisaconstants` (LISA Analysis Tools–wide rule):
  - Python `utils/constants.py` imports it;
  - C++ `cutils/constants.h` reads the generated `cutils/lisaconstants_values.h`.
    Regenerate it with `python -m lisatools.utils.lisaconstants_header PATH`
    after a `lisaconstants` bump;
  - `tests/test_lisaconstants_values.py` guards both.

**Arrivals (for orientation):**
- **Phase 3G (from lisa-on-gpu):** `bbhx.jax.sources.sobbh.JaxSOBBHSource`.
  - It imports `JaxAmpPhaseSource` absolutely from
    `lisatools.jax.response.base`.
  - It loads the `sobbhtaylert3.py` PN prototype at first use. The prototype
    is not in BBHx: set `SOBBH_TAYLOR_T3_PATH` (LAT keeps a copy at
    `scripts/sobbh/sobbhtaylert3.py`).
  - It stores `t_ref` but does not apply it.
- **Phase 3L.8 (2026-06-04, from lisa-on-gpu):** the C++ `SOBBHTDIonTheFly` +
  `SOBBHComputationGroup` classes in `cutils/sobbh_tdi_on_the_fly.{hh,cu}`,
  bound in `cbbhx` (not in a separate module). Since 2026-07-30 the PN
  intrinsics are evaluated at `t - t_ref`.
- **2026-06-05 (from GBGPU):** `bbhx.sobbhcomps.SOBBHWDMComputations`, a thin
  routing subclass of `lisatools.chunked_het.WDMComputationsBase`, plus
  `examples/sobbh_tutorial.ipynb`. LAT reaches it only with
  `SOBBH_LIKELIHOOD=chunked`; the production default is the LAT `lookup`.
- **Signal-heterodyne port (2026-06-18/19):** the SOBBH duplicate of GBGPU's
  `gbfd_*` + `gb_signal_het_*` machinery.
  - Where it lives: `sobbh_tdi_on_the_fly.{hh,cu}`, bound as
    `SOBBHComputationGroupWrap.sobbh_signal_het_{get_ll,get_ll_sparse,get_ll_in_kernel,fill_global_in_kernel,get_ll_grad_in_kernel}`.
  - Python front end: none in the package yet. The `SOBBHSignalHetComputations`
    prototype is a LAT script
    (`LISAanalysistools/scripts/sobbh/sobbhsignalhetcomputations.py`).
  - JAX: there is no JAX SOBBH computation group.
  - The separate files once planned (`SOBBHSignalHet.{hh,cu}`,
    `SOBBHAbsoluteFD.{hh,cu}`, `binding_sobbhsignalhet.cxx`,
    `bbhx/sobbhsignalhetcomputations.py`) were not created.
  - The source-agnostic Python helpers live in LAT `signal_het.py` /
    `chunked_het.py`. The GB prototype is
    `LISAanalysistools/scripts/gb_chunked_het/gb_signal_het_wdm_v2.py`.
- **2026-06-10 (from LAT):** `bbhx.mbhtdionfly.MBHTDIonFly`, the time-domain
  MBH TDI-on-the-fly generator.
  - It was copied from the former `LISAanalysistools/scripts/mbh/mbhtdionfly.py`
    and converted to a library module: a `BBHxParallelModule` subclass with the
    backend chosen at instantiation via `force_backend`, per the LISA Analysis
    Tools–wide rule.
  - It composes a time-domain amp/phase mode generator
    (`phentax.waveform.IMRPhenomTHM`) with
    `lisatools.response.tdionfly.TDTDIonTheFly`. It is pure Python, with no new
    native code.
  - `phentax` (external, `asantini29/phentax`, NOT on PyPI) installs via the
    `phentax` extra (`pip install 'bbhx[phentax]'`) or directly from GitHub. The
    extra is a PEP 508 git direct reference, which must be stripped or pinned
    before any PyPI upload.
  - Docs: `docs/source/user/main.rst` (autoclass) and
    `examples/mbh_tdionfly_tutorial.ipynb`. Test: `tests/test_mbhtdionfly.py`,
    which skips the waveform test when phentax is missing.
  - It was validated bitwise-identical to the LAT script before the script was
    removed. LAT builds it in `stock/erebor/wrappers.get_mbh_tdionfly_gen`,
    only with `USE_TDIONFLY=1`.
- **2026-07-01 (from LAT settings files):** `bbhx.mbhphentax`, a set of legacy
  phentax adapters. They are TODO-tagged: the only importer is the legacy LAT
  `global_fit_input/mbh_phentax_only_global_fit_settings.py`, and the LAT stock
  recipes use their own copies in `stock/erebor/wrappers.py`.

**Phentax + sobbhx waveform expansion** (plan, not started): a
`waveforms/phentax/` subpackage for the PhenomTAX waveform and a
`waveforms/sobbhx/` subpackage for the sobbhx waveform, which today lives in
scripts across the LISA Analysis Tools repos. Each would get a
`waveforms/<name>/` Python frontend and a `cutils/<Name>Waveform.cu` kernel.
Neither subpackage exists yet: `waveforms/` holds `phenomhm.py` and
`ringdownphenomd.py`.

**Single-registrant rule (LISA Analysis Tools–wide)**: BBHx's binding TUs MUST NOT
register `OrbitsWrap`, `LISAResponseWrap`, `TDIConfigWrap`, or
`CubicSplineWrap`.

- The first three are owned by LAT's `pycppdetector`; `CubicSplineWrap` is
  owned by GBT's `interp` module (since 2026-06-10).
- The legacy `OrbitsWrap_responselisa` was deleted at Phase 3L.7p
  (2026-06-04) in favor of the canonical `OrbitsWrap`.
- `binding_bbhx.hpp` includes `lisatools_header_abi.hpp`, and
  `binding_bbhx.cxx` asserts `static_assert(!LISATOOLS_IS_WRAPPER_OWNER, ...)`.
  Any new binding TU must do the same.
- `LISAanalysistools/tools/check_single_registrant.sh` is the CI grep
  complement.

**Editable install**: the development stack installs BBHx (branch `dev`)
editably with `pip install --no-build-isolation -e .`, via
`LISAanalysistools/install.sh`. Python edits are then live. Re-run the
install after native-code, dependency or packaging changes.

## Backend implementation hierarchy (LISA Analysis Tools–wide rule)

When implementing or modifying an algorithm that exists across multiple
backends (GPU C++ / CPU C++ / JAX), follow this hierarchy:

1. **GPU C++ (CUDA) leads.** This is the canonical performance target
   and reference implementation. New algorithms and optimizations are
   designed for the GPU first; CPU and JAX paths follow.

2. **CPU C++ mirrors GPU C++ as closely as possible.** Same kernel
   structure, same algorithm, same data flow — use `#ifdef __CUDACC__`
   or shared compile-time macros (`CUDA_SHARED`, `THREAD_START`,
   `BLOCK_INCR`, …) to bridge platform differences. The CPU path
   exists primarily for testing and CPU-only environments; it must
   not diverge in algorithm or output beyond floating-point order of
   operations.

3. **CPU C++ must reproduce the overall lisatools computation.**
   Against the lisatools reference (e.g. `FDSignal.transform`,
   `TDSignal.transform`, `XYZ2SensitivityMatrix`), match to machine
   precision (≤ 1e-15 mismatch) in direct modes; cache/approximation
   modes have documented per-feature error budgets.

4. **JAX may diverge internally** — design it to be JAX-efficient.
   JAX-CPU and JAX-GPU compilation targets may even differ. Use
   JAX-native idioms (`jax.lax.scan`, `jax.vmap`, static-shape
   `dynamic_slice` + masks, functional carries) rather than
   mechanically translating CUDA shared memory / register caches.

5. **JAX must match C++ inner-product outputs.** End-to-end
   likelihood quantities (`<d|h>`, `<h|h>`, swap_ll 5 terms) must
   match the C++ to floating-point precision (reldiff ≲ 1e-12) on
   representative test cases. Intermediate quantities (raw templates,
   per-chunk WDM coefficients) may differ at FP precision due to
   summation order — validate at the inner-product level.

**Workflow for a new feature.** GPU C++ → CPU C++ via `#ifdef` → JAX
with JAX-native idioms → cross-backend inner-product validation.
