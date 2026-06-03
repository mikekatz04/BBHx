# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Sprint reorg state (post-Phase-3G, 2026-06-02)

BBHx is the **MBH + SOBBH-physics owner** in the sprint's layered
architecture. LISAanalysistools (LAT) owns generic LISA infrastructure;
BBHx owns MBH (PhenomHM, PhenomTAX) and SOBBH-specific physics.

**Already received from lisa-on-gpu (Phase 3G):**
- `bbhx.jax.sources.sobbh` — `JaxSOBBHSource`. Imports
  `JaxAmpPhaseSource` absolutely from `lisatools.jax.response.base`.

**Pending arrival (future C++ TDIonTheFly carve-out session):**
- C++: `SOBBHTDIonTheFly` class + `SOBBHComputationGroup` class.
  Will land in `BBHx/src/bbhx/cutils/` with its own pybind11 module.
- JAX: `computation_group.py`'s `SOBBHComputationGroupWrapJAX`
  (currently in lisa-on-gpu's `fastlisaresponse.jax.wdm.computation_group`,
  mixed with GB; split during C++ carve-out).

**Phentax + sobbhx waveform expansion** (plan section, not yet started):
the existing `waveforms/phentax/` subpackage will house the PhenomTAX
waveform implementation; a new `waveforms/sobbhx/` will house the
sobbhx waveform currently in sprint-tree scripts. Each gets a
`waveforms/<name>/` Python frontend + a `cutils/<Name>Waveform.cu`
kernel.

**Single-registrant rule (sprint-wide)**: BBHx's binding TUs MUST NOT
register `OrbitsWrap`, `LISAResponseWrap`, `TDIConfigWrap`,
`OrbitsWrap_responselisa`, or `CubicSplineWrap_responselisa`. Those are
owned by LAT's `pycppdetector`. When BBHx receives its tdionthefly
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
