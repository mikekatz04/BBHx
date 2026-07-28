_Last mapped: d5a54dc · 2026-07-10 · regenerate when structure changes_

# BBHx codebase map

## 1. What this is

`bbhx` is the LISA Analysis Tools **MBH (massive black-hole binary) + SOBBH (stellar-origin
BBH) physics owner**. It provides GPU/CPU-agnostic frequency-domain PhenomHM/PhenomD
waveform generation + fast LISA response + likelihoods for MBHs (the original
`arXiv:2005.01827` / `arXiv:2111.01064` package), plus two newer additions built
during the LISA Analysis Tools reorg: a time-domain "TDI-on-the-fly" generator for MBH
(via the external `phentax` IMRPhenomTHM model) and SOBBH, and a C++
chunked-heterodyne / signal-heterodyne likelihood pipeline for SOBBH that
mirrors GBGPU's galactic-binary machinery. Generic LISA infrastructure
(orbits, TDI config, response base classes, WDM/FD domains) is NOT owned
here — it lives in LISAanalysistools (LAT) and is consumed via LAT's
pybind11/nanobind module.

## 2. Layout

| Path | Role |
|---|---|
| `src/bbhx/waveformbuild.py` | `BBHWaveformFD` — the classic top-level FD waveform+response+interp pipeline; `TemplateInterpFD`. |
| `src/bbhx/waveforms/phenomhm.py` | `PhenomHMAmpPhase` — PhenomHM/PhenomD amplitude+phase generator (Python frontend over `waveform_amp_phase_wrap`). |
| `src/bbhx/waveforms/ringdownphenomd.py` | PhenomD ringdown/damping-frequency fit coefficients + helper math (3020 lines, mostly tables), imported by `phenomhm.py`. |
| `src/bbhx/response/fastfdresponse.py` | `LISATDIResponse` — fast FD LISA response (Marsat/Baghi-style), consumes `lisatools.detector.Orbits`. |
| `src/bbhx/likelihood.py` | `Likelihood`, `HeterodynedLikelihood`, `NewHeterodynedLikelihood` — FD likelihood classes for the classic MBH pipeline. |
| `src/bbhx/mbhtdionfly.py` | `MBHTDIonFly` — time-domain MBH TDI-on-the-fly generator (phentax amp/phase + `lisatools.response.tdionfly.TDTDIonTheFly`). |
| `src/bbhx/mbhphentax.py` | phentax response builders/adapters (`IMRPhenomTHMWaveform`, `MBHWaveWrap`, `MBHTDIonFlyWaveWrap`) + transform containers; carved out of LAT global-fit settings files (2026-07-01). |
| `src/bbhx/sobbhtdionfly.py` | `SOBBHTDIonFly` — time-domain SOBBH TDI-on-the-fly generator (Python-side PN amp/phase via `lisatools.sources.sobbh.waveform.SOBBHWaveform`). |
| `src/bbhx/sobbhcomps.py` | `SOBBHWDMComputations` — thin routing subclass of `lisatools.chunked_het.WDMComputationsBase` for SOBBH's C++ chunked-heterodyne likelihood. |
| `src/bbhx/jax/sources/sobbh.py` | `JaxSOBBHSource` — JAX PN amp/phase/frequency source, delegates to a dynamically-loaded `sobbhtaylert3.py` prototype file. |
| `src/bbhx/cutils/` | All native code: `.hh`/`.cu` kernels + the `binding_bbhx.{hpp,cxx}` nanobind module (`cbbhx`) + `CMakeLists.txt`. |
| `src/bbhx/cutils/PhenomHMWaveform.{hh,cu}`, `Response.{hh,cu}`, `Likelihood.{hh,cu}`, `WaveformBuild.{hh,cu}`, `SpecialLikelihood.{hh,cu}` | The classic MBH FD pipeline kernels (formerly six separate Cython `.pyx` modules, migrated to nanobind at Phase BBHx.pybind, 2026-06-03). |
| `src/bbhx/cutils/sobbh_tdi_on_the_fly.{hh,cu}` | `SOBBHTDIonTheFly` (PN source class) + `SOBBHComputationGroup` (chunked-het + signal-het kernel launchers); carved out of `lisa-on-gpu` at Phase 3L.8. |
| `src/bbhx/cutils/binding_bbhx.{hpp,cxx}` | The `cbbhx` nanobind module: `BBHxComputationWrap` (MBH bound methods) + `SOBBHTDIonTheFlyWrap` + `SOBBHComputationGroupWrap`. |
| `src/bbhx/cutils/__init__.py` | `BBHxBackend`/`BBHxBackendMethods`/`BBHx{Cpu,Cuda11x,Cuda12x,Cuda13x}Backend` — composes LAT's `LISAToolsBackend` with BBHx-native symbols. |
| `src/bbhx/utils/` | `parallelbase.py` (`BBHxParallelModule`), `transform.py` (MBH parameter transforms), `constants.py`, `exceptions.py`, `citations.py`, `config.py`, `utility.py`. |
| `src/bbhx/cutils/eob.pyx` | Orphaned Cython EOB-waveform stub — **not referenced by any CMakeLists**; dead code, not part of the build. |
| `src/PhenomHM.cu` (repo root, outside `src/bbhx/`) | Orphaned legacy pre-reorg PhenomHM source (3269 lines); not wired into `src/CMakeLists.txt` (which only does `add_subdirectory(bbhx)`). Dead file, not the active `src/bbhx/cutils/PhenomHMWaveform.cu`. |
| `tests/` | `test_bbhx.py` (classic FD pipeline smoke test), `test_mbhtdionfly.py` (MBH TDI-on-the-fly construction + optional phentax waveform test). |
| `examples/` | `bbhx_tutorial.ipynb`, `sobbh_tutorial.ipynb`, `mbh_tdionfly_tutorial.ipynb`. |
| `docs/source/user/` | Sphinx doc pages: `main.rst` (autoclass incl. `MBHTDIonFly`), `like.rst`, `response.rst`, `waveforms.rst`, `constants.rst`, `utils.rst`. |

## 3. Core abstractions

Two largely parallel pipelines, both subclassing `BBHxParallelModule`
(`utils/parallelbase.py`, backend prefix `"bbhx"`):

**A. Classic FD pipeline (MBH, PhenomHM/PhenomD):**

```
PhenomHMAmpPhase           LISATDIResponse            TemplateInterpFD
(amp/phase per mode)  -->  (FD response -> TDI)  -->  (cubic-spline interp
        |                          |                   to data freq grid)
        +----------- BBHWaveformFD (orchestrator) -----------+
                              |
                Likelihood / HeterodynedLikelihood / NewHeterodynedLikelihood
```
`BBHWaveformFD.__call__` drives `amp_phase_gen` -> `response_gen` ->
either `direct_sum_wrap` (no interp) or `interp_response` (cubic-spline
interp via `gpubackendtools.interpolate.CubicSplineInterpolant` +
`InterpTDI_wrap`), producing per-binary TDI channel arrays.

**B. TDI-on-the-fly pipelines (time-domain, MBH + SOBBH):**

```
MBHTDIonFly(wave_gen=phentax IMRPhenomTHM, orbits, tdi_config, ...)
SOBBHTDIonFly(wave_gen=SOBBHWaveform, orbits, tdi_config, ...)
        both feed amp/phase splines into:
lisatools.response.tdionfly.TDTDIonTheFly  (LAT-owned, generic on-the-fly response)
```
These are pure-Python classes (no new native code) that compose an
amp/phase mode generator with LAT's generic on-the-fly TDI response.
`mbhphentax.py` layers response-wrapper/domain-adapter classes
(`IMRPhenomTHMWaveform`, `MBHWaveWrap`, `MBHTDIonFlyWaveWrap`) plus
sampling<->waveform `TransformContainer`s on top, for use inside the
LAT global-fit stock recipes.

**C. SOBBH chunked-heterodyne / signal-heterodyne (C++, frequency-domain, WDM):**

```
SOBBHTDIonTheFly : public LISATDIonTheFly   (C++ PN source class; LAT base)
        |
SOBBHComputationGroup                        (kernel-launcher surface)
   sobbh_wdm_het_{fill_global,get_ll,swap_ll,get_fstat_ll}   <- chunked-het (WDM-domain)
   sobbh_signal_het_{get_ll,get_ll_sparse,get_ll_in_kernel,
                      fill_global_in_kernel,get_ll_grad_in_kernel} <- v2 polyphase signal-het
        |
SOBBHComputationGroupWrap / SOBBHTDIonTheFlyWrap   (nanobind wraps, binding_bbhx.hpp/.cxx)
        |
SOBBHWDMComputations(lisatools.chunked_het.WDMComputationsBase)   (Python routing, sobbhcomps.py)
```
`SOBBHTDIonTheFly` inherits LAT's `LISATDIonTheFly` base (shared with
GBGPU's `GBTDIonTheFly`); the templated `wdm_het_*_impl<SourceT>`
kernels and the generic polyphase/bin-fold machinery both live in LAT
headers (`lat_chunked_het_kernels.hh`, `lat_tdi_on_the_fly.hh`) and are
instantiated here with `SourceT = SOBBHTDIonTheFly`. `SOBBHWDMComputations`
is a ~30-line subclass of LAT's `WDMComputationsBase` that only sets
routing constants (`_BACKEND_PREFIX="bbhx"`, `_WRAP_ATTR=
"SOBBHComputationGroupWrap"`, `_METHOD_PREFIX="sobbh_wdm_het"`,
`_NPARAMS=11`, `_F0_PARAM_INDEX=5`) — all the chunk geometry / WDM
window / layer-grouping logic is inherited unchanged and shared with
GBGPU's `GBWDMComputations`.

`JaxSOBBHSource(lisatools.jax.response.base.JaxAmpPhaseSource)` is the
JAX-side mirror of the C++ `SOBBHTDIonTheFly` PN physics — both are
meant to be ported from the same prototype (`sobbhtaylert3.py`).

## 4. Public API / entry points

Users typically hit one of these top-level classes/functions:

- **`bbhx.waveformbuild.BBHWaveformFD(...)`** — classic call:
  `wave_gen(m1, m2, chi1z, chi2z, distance, phi_ref, f_ref, inc, lam, beta,
  psi, t_ref, freqs=..., modes=..., length=..., ...)` -> FD TDI channel
  arrays. Backed by `PhenomHMAmpPhase` + `LISATDIResponse` +
  `TemplateInterpFD`.
- **`bbhx.likelihood.Likelihood` / `HeterodynedLikelihood` /
  `NewHeterodynedLikelihood`** — FD log-likelihood given a
  `template_gen` (e.g. a configured `BBHWaveformFD`), `data_freqs`,
  `data_channels`, `psd`.
- **`bbhx.mbhtdionfly.MBHTDIonFly(wave_gen, orbits, tdi_config, dt, Tobs,
  t0, ...)`** — time-domain MBH TDI; `wave_gen` is typically
  `phentax.waveform.IMRPhenomTHM`. Called as
  `gen(m1, m2, s1z, s2z, distance, phi_ref, inclination, ra, dec, psi,
  t_merge, upsample_t_arr=..., combine=...)`.
- **`bbhx.sobbhtdionfly.SOBBHTDIonFly(wave_gen, orbits, tdi_config, dt,
  Tobs, t0, ...)`** — time-domain SOBBH TDI; `wave_gen` is a
  `lisatools.sources.sobbh.waveform.SOBBHWaveform`.
- **`bbhx.sobbhcomps.SOBBHWDMComputations`** — SOBBH chunked-heterodyne
  likelihood entry point (used from the LAT global-fit SOBBH move,
  same call surface as GBGPU's `GBWDMComputations`).
- **`bbhx.mbhphentax.get_mbh_phentax_response_wrapper(...)` /
  `get_mbh_tdionfly_gen(...)`** — cached builder functions used by the
  LAT global-fit stock recipes to construct the legacy `ResponseWrapper`
  or the `MBHTDIonFly` generator, plus `make_mbh_phentax_transform_container()`
  / `make_mbh_tdionfly_transform_container()` for sampling<->waveform bases.
- **`bbhx.get_backend(name)` / `bbhx.has_backend(name)` /
  `bbhx.get_first_backend(name)`** — backend resolution (see §5).
- Low-level building blocks (`PhenomHMAmpPhase`, `LISATDIResponse`,
  `TemplateInterpFD`) are importable directly for custom pipelines
  (used e.g. in `tests/test_bbhx.py::test_phenom_hm`).

## 5. Backend structure

- Every top-level class takes `force_backend=` at **construction** (never
  as a per-call kwarg — LISA Analysis Tools–wide rule) and exposes `self.xp` /
  `self.backend` via `BBHxParallelModule(gpubackendtools.ParallelModuleBase)`.
- Native code is CPU C++ / CUDA C++ only (no BBHx-owned JAX kernels for
  the MBH path; `bbhx.jax.sources.sobbh` is the one JAX source, mirroring
  the C++ SOBBH PN physics).
- The nanobind module is `cbbhx`, built per backend flavor as
  `bbhx_backend_{cpu,cuda11x,cuda12x,cuda13x}.cbbhx`, exposing
  `BBHxComputationWrap{CPU,GPU}` (MBH bound methods: `waveform_amp_phase_wrap`,
  `LISA_response_wrap`, `InterpTDI_wrap`, `direct_sum_wrap`, `hdyn_wrap`,
  `direct_like_wrap`, `speciallike` (GPU-only), `new_hdyn_{prep,like}`
  (GPU-only)) and `SOBBHTDIonTheFlyWrap{CPU,GPU}` +
  `SOBBHComputationGroupWrap{CPU,GPU}`.
- `src/bbhx/cutils/__init__.py` composes each flavor's LAT symbols
  (`lisatools_backend_<flavor>.pycppdetector`: `OrbitsWrap`, `TDIConfigWrap`,
  WDM/FD/Spline wraps, sensitivity/galactic-grid wraps) + GBT's
  `CubicSplineWrap` (`gbt_backend_<flavor>.interp`) + BBHx's own `cbbhx`
  symbols into one `BBHxBackend` object per flavor
  (`BBHxCpuBackend`, `BBHxCuda{11,12,13}xBackend`), registered in
  `bbhx/__init__.py` via `Globals().backends_manager.add_backends(...)`.
  Consumer code does `self.backend.OrbitsWrap`, `self.backend.SOBBHTDIonTheFlyWrap`,
  `self.backend.waveform_amp_phase_wrap(...)` all on the same object.
- CPU build compiles all `.cu` sources as `.cxx` (copied via
  `add_custom_command`) alongside a **copy-compiled** `Detector.cu` and
  `lat_tdi_on_the_fly.cu` pulled from the installed `lisatools` package
  (`LISATOOLS_CUTILS`) and `Interpolate.cu` pulled from `gpubackendtools`
  (`GBT_CUTILS`) — same "LAT/GBT must be compiled into every consuming
  .so for typeinfo" pattern used across LISA Analysis Tools. `SpecialLikelihood.cu` /
  `new_hdyn_*` are GPU-only (`#ifdef __CUDACC__`), so the CPU loader sets
  `speciallike=None`, `new_hdyn_prep=None`, `new_hdyn_like=None`.
  `__CUDA_COMPILATION__` is defined `PUBLIC` on GPU targets (bug fixed
  2026-06 — its absence previously caused `undefined symbol: _ZTV*CPU`
  link failures because `.cxx` binding TUs picked the wrong CPU/GPU class
  alias; see CPU/GPU class-name-aliasing rule below).
- CPU/GPU class-name aliasing: both the underlying C++ classes
  (`SOBBHTDIonTheFly{CPU,GPU}`, `SOBBHComputationGroup{CPU,GPU}`,
  `BBHxComputationWrap{CPU,GPU}`) and their nanobind wraps
  (`SOBBHTDIonTheFlyWrap{CPU,GPU}`, `SOBBHComputationGroupWrap{CPU,GPU}`)
  are `#define`-aliased at the top of `sobbh_tdi_on_the_fly.hh` /
  `binding_bbhx.hpp` per the LISA Analysis Tools–wide rule, so the CPU and CUDA
  plugin wheels never collide in pybind11/nanobind's global type
  registry.
- **Single-registrant rule**: `binding_bbhx.cxx` has
  `static_assert(!LISATOOLS_IS_WRAPPER_OWNER, ...)` — BBHx must never
  register `OrbitsWrap`/`TDIConfigWrap`/`LISAResponseWrap`/WDM-FD-Spline
  wraps (LAT owns those) or `CubicSplineWrap` (GBT owns that).

## 6. Cross-repo dependencies

**Imports from:**
- `gpubackendtools` (GBT) — `ParallelModuleBase`, `Globals`, `get_backend`/
  `has_backend`/`get_first_backend`, `Backend`, `CpuBackend`/`Cuda{11,12,13}xBackend`,
  `interpolate.CubicSplineInterpolant`, `gbt_backend_*.interp` (CubicSpline),
  build-time: `Interpolate.cu`, `cmake_functions.cmake`, `get_lapacke()`.
- `lisatools` (LAT) — `lisatools.cutils.{LISAToolsBackend,LISAToolsBackendMethods}`,
  `lisatools.detector.{Orbits,EqualArmlengthOrbits,ESAOrbits}`,
  `lisatools.response.{directresponse.ResponseWrapper,tdiconfig.TDIConfig,
  tdionfly.TDTDIonTheFly}`, `lisatools.domains.{TDSettings,TDSignal}`,
  `lisatools.sources.sobbh.waveform.SOBBHWaveform`,
  `lisatools.chunked_het.WDMComputationsBase`,
  `lisatools.jax.response.base.JaxAmpPhaseSource`,
  `lisatools.sensitivity.{SensitivityMatrix,AET1SensitivityMatrix}`;
  build-time: `Detector.{hpp,cu}`, `lat_tdi_on_the_fly.{hh,cu}`,
  `lat_chunked_het_kernels.hh`, `lisatools_header_abi.hpp`,
  `binding_flr.hpp`, `binding_lat_spline_tdi.hpp`, `binding_wdm_settings.hpp`,
  `fd_domain.hh`, `wdm_settings.hh`, `wdm_domain.hh`.
- `eryn` — `eryn.utils.transform.TransformContainer` (used in `mbhphentax.py`).
- `phentax` (external, `asantini29/phentax`, not on PyPI) — optional
  extra `bbhx[phentax]`; lazily imported so the package works without it.
- **`gbgpu`** — `src/bbhx/utils/citations.py` and `src/bbhx/utils/config.py`
  import `gbgpu.utils.exceptions.InvalidInputFile`, `gbgpu.__file__`,
  `gbgpu._is_editable`, `gbgpu.get_logger`. These two files are
  near-verbatim copies of GBGPU's `citations.py`/`config.py` that were
  never adapted for BBHx (see §7 — likely stale/broken, not a deliberate
  cross-dependency).

**Depends on BBHx (reverse):** primarily `LISAanalysistools`'s global-fit
layer — `lisatools/globalfit/{mbhglobal,mbhsearch,pipeline,recipe,plot,
buildcatalog,generatefuncs}.py`, `lisatools/globalfit/moves/mbhspecialmove.py`,
`lisatools/globalfit/stock/erebor/{transforms,wrappers,variants/all_sources}.py`,
`lisatools/sources/{bbh/waveform.py,sobbh/response.py}`,
`lisatools/sampling/moves/skymodehop.py`, plus many `LISAanalysistools/
scripts/{mbh,sobbh}/*.py` dev/validation scripts and
`LISAanalysistools/global_fit_input/*mbh*` compatibility stubs. No other
LISA Analysis Tools repo (GBGPU, FastEMRIWaveforms, Eryn, lisa-on-gpu) imports `bbhx`.

## 7. Non-obvious invariants / gotchas

- **BBHx's `CLAUDE.md` is stale (dated "post-Phase-3G, 2026-06-02")** — it
  describes `SOBBHTDIonTheFly`/`SOBBHComputationGroup` and the SOBBH
  signal-heterodyne kernels as "pending arrival"; both actually **shipped**
  at Phase 3L.8 (2026-06-04) and the sig-het port (2026-06-18/19) — see
  `binding_bbhx.hpp`/`.cxx` and `sobbh_tdi_on_the_fly.{hh,cu}`, all fully
  populated and registered. Trust the code over that file.
- **`utils/citations.py` and `utils/config.py` are copy-pasted from
  GBGPU and not adapted**: docstrings literally say "GBGPU"
  (`config.py:1`: `"""Implementation of a centralized configuration
  management for GBGPU."""`), `get_package_basepath()` imports and
  returns **`gbgpu`**'s install path (not bbhx's), config file search
  paths look for `gbgpu.ini`, and `citations.py` imports
  `gbgpu.utils.exceptions.InvalidInputFile` even though BBHx has its own
  `bbhx.utils.exceptions.InvalidInputFile`. `bbhx/__init__.py`'s
  `__all__` claims to export `get_logger`/`get_config`/
  `get_config_setter`/`get_file_manager`, but none of these names are
  actually defined or imported anywhere in `bbhx/__init__.py` — `from
  bbhx import *` would likely raise `AttributeError`. Whoever next
  touches config/citations should either finish the GBGPU->BBHx port or
  delete the unused surface.
- **`src/bbhx/cutils/eob.pyx`** (Cython) and **`src/PhenomHM.cu`**
  (repo-root, outside `src/bbhx/`) are orphaned dead files — neither is
  referenced by any `CMakeLists.txt`. The live PhenomHM kernel is
  `src/bbhx/cutils/PhenomHMWaveform.cu`.
- **Single-registrant rule / CPU-GPU aliasing** (LISA Analysis Tools–wide, see root
  `CLAUDE.md`): applies fully here — `binding_bbhx.cxx` static_asserts
  it, and both `SOBBHTDIonTheFly`/`SOBBHComputationGroup` and their
  Wraps carry the `#define ...GPU / ...CPU` alias blocks.
- **`__CUDA_COMPILATION__` must be `PUBLIC` on GPU CMake targets** —
  `src/bbhx/cutils/CMakeLists.txt` documents a fixed bug where its
  absence caused `undefined symbol: _ZTV*CPU` at GPU `.so` load time
  because `.cxx` TUs (no `__CUDACC__` from nvcc) picked the CPU class
  alias inside a GPU build.
- **MBH TDI-on-the-fly phase/reference-time convention**
  (`mbhtdionfly.py:145-158`): reference phase must be anchored at
  `t_ref = -t_merge` (not `t_ref=0`) to match the `pyResponse` /
  mojito convention; using `t_ref=0` leaves an ~11% edge-on mismatch
  from the higher modes.
- **MBH TDI-on-the-fly zero-tail padding** (`mbhtdionfly.py:168-194`):
  120 samples at `dt_tail=10s` are appended past the phentax grid end so
  the on-the-fly kernel's retarded-time reads (up to ~600s past the
  eval window edge for unlucky sky positions) never fall outside the
  spline; the physical post-ringdown amplitude there is ~1e-19 of peak.
- **SOBBH TDI-on-the-fly inclination convention differs from MBH**
  (`sobbhtdionfly.py:52-60`): MBH folds inclination into the per-mode
  spherical harmonic and passes `inc=0` to the response; SOBBH passes
  the *real* inclination because its amplitude is intrinsic
  (no-inclination) PN amplitude.
- **`phentax` coarse-graining scale** (`mbhphentax.py:41-47`):
  `MBH_TDIONFLY_COARSE_SCALE = 48.0` is required to resolve the
  merger/ringdown (~0.886s spacing); the phentax default of 12
  under-resolves it and leaves a spurious ~5e-22 mismatch floor.
- **SOBBH chunked-het param order** (`sobbhcomps.py`, matches
  `SOBBHTDIonTheFly`): `(m1, m2, s1, s2, distance, f_low, phi_c, inc, psi,
  lam, beta)` — 11 params, `f0`/carrier-frequency column is index 5
  (`_F0_PARAM_INDEX`), vs GB's 9-param / index layout.
- **`docs/source/user/main.rst`** documents `MBHTDIonFly` via autoclass;
  there is no equivalent doc page yet for `SOBBHTDIonFly` (SOBBH TDI-on-
  the-fly docs are examples-only, `examples/sobbh_tutorial.ipynb`).

## 8. Where to look for X

| Change / understand X | Start in |
|---|---|
| Classic MBH FD waveform generation (PhenomHM/PhenomD amp+phase) | `src/bbhx/waveforms/phenomhm.py`, `src/bbhx/cutils/PhenomHMWaveform.{hh,cu}` |
| Classic MBH FD LISA response | `src/bbhx/response/fastfdresponse.py`, `src/bbhx/cutils/Response.{hh,cu}` |
| Classic FD waveform orchestration / interpolation to data grid | `src/bbhx/waveformbuild.py`, `src/bbhx/cutils/WaveformBuild.{hh,cu}` |
| MBH FD likelihoods (direct, heterodyned) | `src/bbhx/likelihood.py`, `src/bbhx/cutils/Likelihood.{hh,cu}`, `src/bbhx/cutils/SpecialLikelihood.{hh,cu}` |
| MBH time-domain TDI-on-the-fly | `src/bbhx/mbhtdionfly.py`, `src/bbhx/mbhphentax.py` |
| SOBBH time-domain TDI-on-the-fly | `src/bbhx/sobbhtdionfly.py` |
| SOBBH PN source physics (C++) | `src/bbhx/cutils/sobbh_tdi_on_the_fly.{hh,cu}` (`SOBBHTDIonTheFly::sobbh_{amplitude,phase,f,fdot}`) |
| SOBBH PN source physics (JAX) | `src/bbhx/jax/sources/sobbh.py` + repo-root `sobbhtaylert3.py` prototype |
| SOBBH chunked-heterodyne likelihood (Python routing) | `src/bbhx/sobbhcomps.py` |
| SOBBH chunked-het / signal-het kernels (C++) | `src/bbhx/cutils/sobbh_tdi_on_the_fly.{hh,cu}` (`SOBBHComputationGroup`) |
| nanobind module surface / adding a new bound method | `src/bbhx/cutils/binding_bbhx.{hpp,cxx}` |
| Backend composition / what symbols a backend carries | `src/bbhx/cutils/__init__.py` |
| Native build (CMake), CPU/GPU target wiring, LAT/GBT header consumption | `src/bbhx/cutils/CMakeLists.txt`, root `CMakeLists.txt` |
| MBH parameter transforms (sampling<->waveform basis) | `src/bbhx/utils/transform.py`, `src/bbhx/mbhphentax.py` (transform containers) |
| Backend/parallel-module base class | `src/bbhx/utils/parallelbase.py` |
| Package-level backend registration / imports | `src/bbhx/__init__.py` |
| Tests / usage examples for the classic pipeline | `tests/test_bbhx.py` |
| Tests for MBH TDI-on-the-fly | `tests/test_mbhtdionfly.py` |
| Tutorial notebooks | `examples/bbhx_tutorial.ipynb`, `examples/sobbh_tutorial.ipynb`, `examples/mbh_tdionfly_tutorial.ipynb` |
