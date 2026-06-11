# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Sprint reorg state (post-Phase-3G, 2026-06-02)

BBHx is the **MBH + SOBBH-physics owner** in the sprint's layered
architecture. LISAanalysistools (LAT) owns generic LISA infrastructure;
BBHx owns MBH (PhenomHM, PhenomTAX) and SOBBH-specific physics.

**Build infrastructure (post-Phase-BBHx.pybind, 2026-06-03):**
- `src/bbhx/cutils/binding_bbhx.{hpp,cxx}` -- pybind11 module skeleton
  adapted from the original `origin/pybind` branch (commits 7c383e3,
  0543508, bd9e127) against the post-Phase-3L LAT setup. The
  branch-original `ReturnPointerBase` + `OrbitsWrap_bbhx` redeclarations
  were dropped; this TU now consumes LAT's canonical versions via
  `#include "binding_flr.hpp"` and asserts
  `static_assert(!LISATOOLS_IS_WRAPPER_OWNER, ...)` to enforce the
  single-registrant rule.
- Produces a new backend module `bbhx_backend_{cpu,cudaXXx}.cbbhx`
  alongside the existing Cython modules (phenomhm, interp, likelihood,
  response, waveformbuild, newhdyn). The skeleton ships an empty
  `BBHxComputationWrap`; subsequent commits populate it as each Cython
  module migrates to pybind11.
- `find_package(pybind11 CONFIG)` added to project-root `CMakeLists.txt`;
  `pybind11` added to `pyproject.toml`'s build-system requires.
- Same `LISATOOLS_CUTILS` / `GBT_CUTILS` local consumption pattern that
  `Detector.cu` etc. already use -- no URL downloads, version-pinned
  via lisaanalysistools build dep.

**Already received from lisa-on-gpu (Phase 3G):**
- `bbhx.jax.sources.sobbh` — `JaxSOBBHSource`. Imports
  `JaxAmpPhaseSource` absolutely from `lisatools.jax.response.base`.

**Pending arrival (future C++ TDIonTheFly carve-out session):**
- C++: `SOBBHTDIonTheFly` class + `SOBBHComputationGroup` class.
  Will land in `BBHx/src/bbhx/cutils/` with its own pybind11 module.
- JAX: `computation_group.py`'s `SOBBHComputationGroupWrapJAX`
  (currently in lisa-on-gpu's `fastlisaresponse.jax.wdm.computation_group`,
  mixed with GB; split during C++ carve-out).

**Pending arrival (V2 signal-heterodyne port, independent work item):**
- C++: `cutils/SOBBHSignalHet.{hh,cu}` (source-class entry) +
  `cutils/SOBBHAbsoluteFD.{hh,cu}` (PN inspiral FD formula, bin-by-bin) +
  `cutils/binding_sobbhsignalhet.cxx`.
- Python: `bbhx/sobbhsignalhetcomputations.py` —
  `SOBBHSignalHetComputations` parallels the GB version that lands in
  GBGPU. Same kernel-suite shape: `sobbh_signal_het_{fill_global,get_ll,
  swap_ll,get_ll_grad,hessian,get_fstat_ll}` (templated on
  `<SOBBHTDIonTheFly>` — same shape as GB version templated on
  `<GBTDIonTheFly>`).
- JAX: `bbhx/jax/wdm/sobbh_signal_het_*`.
- The generic polyphase + bin-fold + reconstruct primitives live in LAT
  (`lisatools/cutils/SignalHet*.hh,cu`); BBHx only owns the
  SOBBH-specific FD-bin producer.
- Full plan: `~/.claude/plans/yes-find-and-read-sprightly-garden.md`.
- Python prototype lives at
  `LISAanalysistools/scripts/gb_chunked_het/gb_signal_het_wdm_v2.py`
  (named for GB but the architecture is source-agnostic; the SOBBH
  variant follows the same pattern with `SOBBHAbsoluteFD` swapping in
  for `GBAbsoluteFD`).

**Received from LAT (2026-06-10): `bbhx.mbhtdionfly.MBHTDIonFly`** —
time-domain MBH TDI-on-the-fly generator, copied from
`LISAanalysistools/scripts/mbh/mbhtdionfly.py` and converted to a
library module (`BBHxParallelModule` subclass; backend at
instantiation via `force_backend`, sprint rule). It composes a
time-domain amp/phase mode generator (`phentax.waveform.IMRPhenomTHM`)
with `lisatools.response.tdionfly.TDTDIonTheFly`. Pure Python — no new
native code. `phentax` (external, `asantini29/phentax`, NOT on PyPI)
installs via the `phentax` extra (`pip install 'bbhx[phentax]'`, PEP
508 git direct reference — must be stripped/pinned before any PyPI
upload) or directly from GitHub. Docs: `docs/source/user/main.rst`
(autoclass) + `examples/mbh_tdionfly_tutorial.ipynb`. Test:
`tests/test_mbhtdionfly.py` (skips waveform test if phentax missing).
Validated bitwise-identical to the LAT script before the script's
removal.

**Phentax + sobbhx waveform expansion** (plan section, not yet started):
the existing `waveforms/phentax/` subpackage will house the PhenomTAX
waveform implementation; a new `waveforms/sobbhx/` will house the
sobbhx waveform currently in sprint-tree scripts. Each gets a
`waveforms/<name>/` Python frontend + a `cutils/<Name>Waveform.cu`
kernel.

**Single-registrant rule (sprint-wide)**: BBHx's binding TUs MUST NOT
register `OrbitsWrap`, `LISAResponseWrap`, `TDIConfigWrap`, or
`CubicSplineWrap`. The first three are owned by LAT's `pycppdetector`;
`CubicSplineWrap` is owned by GBT's `interp` module (2026-06-10). (The
legacy `OrbitsWrap_responselisa` was deleted at Phase 3L.7p 2026-06-04
in favor of the canonical `OrbitsWrap`.) When BBHx receives its tdionthefly
module, add `#include "lisatools_header_abi.hpp"` +
`static_assert(!LISATOOLS_IS_WRAPPER_OWNER, ...)` to its binding source
(see `lisa-on-gpu/src/fastlisaresponse/cutils/binding_tof.cxx` for the
pattern). Sprint-root `tools/check_single_registrant.sh` is the CI
grep complement.

**Editable-install requirement**: BBHx was installed to site-packages
before Phase 3G. The `bbhx.jax` subpackage that landed at Phase 3G is
ONLY visible if the package is installed editably (`pip install -e .`
from `BBHx/`). When developing Phase 3+, ensure the editable install
is current.

## Backend implementation hierarchy (sprint-wide rule)

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
