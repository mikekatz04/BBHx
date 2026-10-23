_Last mapped: d1b616b (+ the 2026-10-19 docstring/docs pass) · 2026-10-19 · regenerate when structure changes_

# BBHx codebase map

## 1. What this is

`bbhx` is the LISA Analysis Tools **MBH (massive black-hole binary) + SOBBH (stellar-origin
BBH) physics owner**. It provides:

- GPU/CPU-agnostic frequency-domain PhenomHM/PhenomD waveform generation, the fast LISA
  response and likelihoods for MBHs (the original `arXiv:2005.01827` / `arXiv:2111.01064`
  package);
- a time-domain "TDI-on-the-fly" generator for MBHs (via the external `phentax`
  IMRPhenomTHM model) and one for SOBBHs;
- a C++ chunked-heterodyne / signal-heterodyne likelihood pipeline for SOBBHs that mirrors
  GBGPU's galactic-binary machinery.

These two time-domain generators and the SOBBH pipeline were added during the LISA
Analysis Tools reorg. Generic LISA infrastructure (orbits, TDI config, response base
classes, WDM/FD domains, sensitivity curves) is NOT owned here. It lives in
LISAanalysistools (LAT) and is consumed through LAT's nanobind module
(`lisatools_backend_<flavor>.pycppdetector`) and its Python API.

**Production reach (9mo/12mo global-fit runs).** The runs touch bbhx only through these
paths:

- `import bbhx`, which registers the backends;
- `utils.transform.SSB_to_LISA` / `LISA_to_SSB` and `utils.constants.ConstOmega`, used by
  the LAT MBH sky-mode move (`lisatools/sampling/moves/skymodehop.py::LISAFrameMap`), which
  rotates the ICRS sky to the ecliptic first;
- the module load of `waveformbuild`, through `lisatools/sources/bbh/waveform.py` (not
  evaluated);
- `SOBBHTDIonFly`, built lazily as the SOBBH cross-check / fallback generator
  (`lisatools.globalfit.stock.erebor.wrappers.get_sobbh_tdionfly_gen`).

No bbhx waveform, response or likelihood kernel runs in production. `MBH_LIKELIHOOD`
scoring is phentax + the LAT response, and `SOBBH_LIKELIHOOD=lookup` is the LAT WDM
lookup (`SOBBHLookupRouter`).

## 2. Layout

| Path | Role |
|---|---|
| `src/bbhx/__init__.py` | Registers `bbhx_{cpu,cuda11x,cuda12x,cuda13x}` with `gpubackendtools`; `get_backend` / `has_backend` / `get_first_backend` wrappers (add the `bbhx_` prefix). |
| `src/bbhx/waveformbuild.py` | `BBHWaveformFD`, the classic top-level FD waveform + response + interpolation pipeline; `TemplateInterpFD`. |
| `src/bbhx/waveforms/phenomhm.py` | `PhenomHMAmpPhase`: PhenomHM/PhenomD amplitude + phase + t(f) generator (Python frontend over `waveform_amp_phase_wrap`). |
| `src/bbhx/waveforms/ringdownphenomd.py` | Pure data (about 3000 lines): the PhenomD QNM tables `QNMData_a`, `QNMData_fring`, `QNMData_fdamp`, `dspin`, star-imported by `phenomhm.py`. |
| `src/bbhx/response/fastfdresponse.py` | `LISATDIResponse`: fast FD LISA response (Marsat/Baghi), reads spacecraft positions from a `lisatools.detector.Orbits`. |
| `src/bbhx/likelihood.py` | `Likelihood` (direct), `HeterodynedLikelihood`, `NewHeterodynedLikelihood` (GPU-only multi-reference), `searchsorted2d_vec`. |
| `src/bbhx/mbhtdionfly.py` | `MBHTDIonFly`: time-domain MBH TDI on the fly (phentax amp/phase + `lisatools.response.tdionfly.TDTDIonTheFly`). |
| `src/bbhx/mbhphentax.py` | Legacy phentax adapters (`IMRPhenomTHMWaveform`, `MBHWaveWrap`, `MBHTDIonFlyWaveWrap`, builders, transform containers), moved here from LAT settings files on 2026-07-01. All are TODO-tagged: the only importer is LAT `global_fit_input/mbh_phentax_only_global_fit_settings.py`, and LAT's stock recipes use their own copies in `stock/erebor/wrappers.py`. |
| `src/bbhx/sobbhtdionfly.py` | `SOBBHTDIonFly`: time-domain SOBBH TDI on the fly (PN amp/phase from `lisatools.sources.sobbh.waveform.SOBBHWaveform`). |
| `src/bbhx/sobbhcomps.py` | `SOBBHWDMComputations`: thin routing subclass of `lisatools.chunked_het.WDMComputationsBase` for the SOBBH chunked-heterodyne likelihood. |
| `src/bbhx/jax/sources/sobbh.py` | `JaxSOBBHSource`: JAX PN amp/phase/frequency source; loads a `sobbhtaylert3.py` prototype at first use (not in BBHx; `$SOBBH_TAYLOR_T3_PATH`, LAT keeps `scripts/sobbh/sobbhtaylert3.py`). |
| `src/bbhx/utils/` | `parallelbase.py` (`BBHxParallelModule`), `transform.py` (SSB <-> LISA frame maps, sky-mode transform, `mT_q` with `q = m2/m1`), `constants.py` (from `lisaconstants`), `exceptions.py` (4 used base classes + FEW-copied classes nothing raises, TODO-tagged), `utility.py` (legacy Cython-era pointer helpers, TODO-tagged). |
| `src/bbhx/cutils/__init__.py` | `BBHxBackendMethods` / `BBHxBackend` / `BBHx{Cpu,Cuda11x,Cuda12x,Cuda13x}Backend`: compose LAT's `LISAToolsBackend` with BBHx-native symbols. |
| `src/bbhx/cutils/` (native) | `.hh`/`.cu` kernels + the `binding_bbhx.{hpp,cxx}` nanobind module (`cbbhx`) + `CMakeLists.txt`; `constants.h` reads the generated `lisaconstants_values.h`. |
| `src/bbhx/cutils/PhenomHMWaveform.{hh,cu}`, `Response.{hh,cu}`, `Likelihood.{hh,cu}`, `WaveformBuild.{hh,cu}`, `SpecialLikelihood.{hh,cu}` | The classic MBH FD pipeline kernels (formerly Cython `.pyx` modules, migrated to the binding module at Phase BBHx.pybind, 2026-06-03; nanobind since Phase 3M). |
| `src/bbhx/cutils/sobbh_tdi_on_the_fly.{hh,cu}` | `SOBBHTDIonTheFly` (PN source class) + `SOBBHComputationGroup` (chunked-het + signal-het kernel launchers); carved out of `lisa-on-gpu` at Phase 3L.8. |
| `src/bbhx/cutils/binding_bbhx.{hpp,cxx}` | The `cbbhx` nanobind module (`NB_MODULE(cbbhx, m)`): `BBHxComputationWrap` (MBH bound methods) + `SOBBHTDIonTheFlyWrap` + `SOBBHComputationGroupWrap` (+ `SOBBHTDIonTheFly`). |
| `tests/` | `test_bbhx.py` (full FD waveform, PhenomHM, FD response, direct and heterodyned likelihood), `test_mbhtdionfly.py` (construction + optional phentax waveform test), `test_frame_transform.py` (SSB <-> LISA frame maps, sky-mode tiling), `test_lisaconstants_values.py` (header and Python constants vs `lisaconstants`). |
| `examples/` | `bbhx_tutorial.ipynb`, `sobbh_tutorial.ipynb`, `mbh_tdionfly_tutorial.ipynb` (copied into `docs/source/` by `conf.py` at build time). |
| `docs/source/user/` | Sphinx pages: `main.rst` (`BBHWaveformFD`, `MBHTDIonFly`, `mbhphentax`), `conventions.rst` (units, frames, mass ratio, reference phase/time), `waveforms.rst`, `response.rst`, `like.rst`, `sobbh.rst` (`SOBBHTDIonFly`, `SOBBHWDMComputations`, `JaxSOBBHSource`), `utils.rst` (interpolation, `utils.transform`, legacy `utils.utility`), `constants.rst`, `backends.rst` (`get_backend` etc., `BBHxParallelModule`, `cutils`, exceptions). |

## 3. Core abstractions

Two largely parallel pipelines, both built on `BBHxParallelModule`
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

`BBHWaveformFD.__call__` drives `amp_phase_gen` -> `response_gen` -> either
`direct_sum_wrap` (no interpolation, `direct=True`) or `interp_response` (cubic-spline
interpolation via `gpubackendtools.interpolate.CubicSplineInterpolant` + `InterpTDI_wrap`).
It produces per-binary TDI channel arrays.

**B. TDI-on-the-fly pipelines (time domain, MBH + SOBBH):**

```
MBHTDIonFly(wave_gen=phentax IMRPhenomTHM, orbits, tdi_config, ...)
SOBBHTDIonFly(wave_gen=SOBBHWaveform, orbits, tdi_config, ...)
        both feed amp/phase splines into:
lisatools.response.tdionfly.TDTDIonTheFly  (LAT-owned, generic on-the-fly response)
```

These are pure-Python classes (no native code of their own). Each composes an amp/phase
mode generator with LAT's generic on-the-fly TDI response. The sky angles are read in the
frame of the orbit data. `mbhphentax.py` layers response-wrapper / domain-adapter classes
and sampling <-> waveform `TransformContainer`s on top. These are legacy: the LAT stock
recipes use their own copies.

**C. SOBBH chunked-heterodyne / signal-heterodyne (C++, WDM domain):**

```
SOBBHTDIonTheFly : public LISATDIonTheFly   (C++ PN source class; LAT base)
        |
SOBBHComputationGroup                        (kernel-launcher surface)
   sobbh_wdm_het_{fill_global,get_ll,swap_ll,get_fstat_ll}   <- chunked-het (WDM domain)
   sobbh_signal_het_{get_ll,get_ll_sparse,get_ll_in_kernel,
                      fill_global_in_kernel,get_ll_grad_in_kernel} <- polyphase signal-het
        |
SOBBHComputationGroupWrap / SOBBHTDIonTheFlyWrap   (nanobind wraps, binding_bbhx.hpp/.cxx)
        |
SOBBHWDMComputations(lisatools.chunked_het.WDMComputationsBase)   (Python routing, sobbhcomps.py)
```

`SOBBHTDIonTheFly` inherits LAT's `LISATDIonTheFly` base, which it shares with GBGPU's
`GBTDIonTheFly`. The templated `wdm_het_*_impl<SourceT>` kernels and the generic
polyphase / bin-fold machinery live in LAT headers (`lat_chunked_het_kernels.hh`,
`lat_tdi_on_the_fly.hh`) and are instantiated here with `SourceT = SOBBHTDIonTheFly`. The
PN intrinsics are evaluated at `t - t_ref`, where `f_low` and `phi_c` are defined (fix of
2026-07-30). The 11-parameter vector is
`(m1, m2, s1, s2, distance[pc], f_low, phi_c, inc, psi, lam, beta)`.

`SOBBHWDMComputations` only sets routing constants (`_BACKEND_PREFIX="bbhx"`,
`_WRAP_ATTR="SOBBHComputationGroupWrap"`, `_METHOD_PREFIX="sobbh_wdm_het"`,
`_NPARAMS=11`, `_F0_PARAM_INDEX=5`). All the chunk geometry / WDM window / layer-grouping
logic is inherited from LAT and shared with GBGPU's `GBWDMComputations`. The LAT Python
front end `lisatools.response.tdionfly.SOBBHTDIonTheFly` runs on a `bbhx_<flavor>` backend
(it uses `SOBBHTDIonTheFlyWrap`).

`JaxSOBBHSource(lisatools.jax.response.base.JaxAmpPhaseSource)` is the JAX-side mirror of
the C++ `SOBBHTDIonTheFly` PN physics. Both are meant to come from the same prototype
(`sobbhtaylert3.py`), but the JAX class stores `t_ref` without applying it, so the two
disagree for `t_ref != 0`.

## 4. Public API / entry points

- **`bbhx.waveformbuild.BBHWaveformFD(amp_phase_kwargs, response_kwargs, interp_kwargs,
  force_backend)`**, called as `wave_gen(m1, m2, chi1z, chi2z, distance[m], phi_ref,
  f_ref, inc, lam, beta, psi, t_ref, t_obs_start=..., t_obs_end=..., freqs=...,
  length=..., modes=..., direct=..., compress=..., squeeze=..., fill=..., combine=...)`.
  It returns FD TDI channel arrays. Pass `response_kwargs=dict(orbits=...)` explicitly.
- **`bbhx.likelihood.Likelihood(template_gen, data_freqs, data_channels, psd)`** and
  **`HeterodynedLikelihood(template_gen, data_freqs, data_channels,
  reference_template_params, length_f_het, ..., sens_mat=None, outer_buffer=1024)`**.
  `get_ll(params (num_params, num_bin_all), return_extracted_snr=..., phase_maximize=...)`.
  **`NewHeterodynedLikelihood`** is GPU-only, takes `psd` arrays and `(num_bin_all,
  num_params)` params, and needs an explicit CUDA `force_backend`.
- **`bbhx.mbhtdionfly.MBHTDIonFly(wave_gen, orbits, tdi_config, dt, Tobs, t0, dt_min=0.1,
  t_min=0.0, waveform_duration=None, force_backend=None)`**. `wave_gen` is typically
  `phentax.waveform.IMRPhenomTHM`. Called as `gen(m1, m2, s1z, s2z, distance[Mpc],
  phi_ref, inclination, ra, dec, psi, t_merge, upsample_t_arr=..., combine=...)`. The LAT
  builder is `stock/erebor/wrappers.get_mbh_tdionfly_gen` (only with `USE_TDIONFLY=1`).
- **`bbhx.sobbhtdionfly.SOBBHTDIonFly(wave_gen, orbits, tdi_config, dt, Tobs, t0,
  n_grid=2048, buffer_time=5000.0, force_backend=None)`**. `wave_gen` is a
  `lisatools.sources.sobbh.waveform.SOBBHWaveform`. Called as `gen(m1, m2, s1, s2,
  distance[Gpc], f_low, phi0, inc, ra, dec, psi, t_shift=0.0, upsample_t_arr=...,
  combine=True)`.
- **`bbhx.sobbhcomps.SOBBHWDMComputations`**: SOBBH chunked-heterodyne likelihood, with
  the same call surface as GBGPU's `GBWDMComputations` (constructor and methods from LAT
  `WDMComputationsBase`). LAT reaches it only with `SOBBH_LIKELIHOOD=chunked`
  (`stock/erebor/source_runtime.get_sobbh_chunked_comp`).
- **`bbhx.utils.transform`**: `SSB_to_LISA` / `LISA_to_SSB` (SSB-ecliptic angles,
  `t0` in years), `tSSBfromLframe` / `tLfromSSBframe`, `mbh_sky_mode_transform`
  (LISA-frame angles, source-major output layout), `mT_q` / `m1_m2_to_mT_q`
  (`q = m2/m1 <= 1`; LAT's stock MBH basis uses `Q = m1/m2 >= 1`), `modpi` / `mod2pi`.
- **`bbhx.get_backend(name)` / `bbhx.has_backend(name)`**: backend resolution (see §5).
  `bbhx.get_first_backend` does not work as written (see §7).
- **`bbhx.jax.sources.JaxSOBBHSource(t_ref)`**: JAX SOBBH source (needs `jax` and
  `sobbhtaylert3.py`).
- Low-level building blocks (`PhenomHMAmpPhase`, `LISATDIResponse`, `TemplateInterpFD`)
  are importable directly for custom pipelines (used e.g. in
  `tests/test_bbhx.py::test_phenom_hm`).

## 5. Backend structure

- Every compute class takes `force_backend=` at **construction**, never as a per-call
  kwarg (LISA Analysis Tools–wide rule). It exposes `self.xp` / `self.backend` via
  `BBHxParallelModule(gpubackendtools.ParallelModuleBase)`. A bare flavor (`"cpu"`,
  `"cuda12x"`, `"cuda"`, `"gpu"`) is resolved as `bbhx_<flavor>`. The default `None`
  picks the first loadable of `cuda13x`, `cuda12x`, `cuda11x`, `cpu`. The exception is
  `NewHeterodynedLikelihood`, which defines no `supported_backends`, so `None` raises
  `NotImplementedError`.
- Native code is CPU C++ / CUDA C++ only. BBHx registers no JAX backend.
  `bbhx.jax.sources.sobbh` is the one JAX source.
- The nanobind module is `cbbhx`, built per backend flavor as
  `bbhx_backend_{cpu,cuda11x,cuda12x,cuda13x}.cbbhx` (the GPU one is named after the CUDA
  toolkit's major version). It exposes:
  - `BBHxComputationWrap{CPU,GPU}`, with the MBH bound methods `waveform_amp_phase_wrap`,
    `get_phenom{hm,d}_ringdown_frequencies`, `LISA_response_wrap`, `InterpTDI_wrap`,
    `direct_sum_wrap`, `hdyn_wrap`, `prep_hdyn`, `direct_like_wrap`, and the GPU-only
    `speciallike` and `new_hdyn_{prep,like}`;
  - `SOBBHTDIonTheFlyWrap{CPU,GPU}` (`run_wave_tdi_wrap`, `get_buffer_size`);
  - `SOBBHComputationGroupWrap{CPU,GPU}`.
- `src/bbhx/cutils/__init__.py` composes, per flavor, into one `BBHxBackend` object
  (`BBHxCpuBackend`, `BBHxCuda{11,12,13}xBackend`):
  - LAT's symbols (`lisatools_backend_<flavor>.pycppdetector`: `OrbitsWrap`,
    `TDIConfigWrap`, the WDM/FD/Spline wraps, the sensitivity / galactic-grid wraps);
  - GBT's `CubicSplineWrap` (`gbt_backend_<flavor>.interp`);
  - BBHx's own `cbbhx` symbols.

  These backends are registered in `bbhx/__init__.py` via
  `Globals().backends_manager.add_backends(...)`. Consumer code calls
  `self.backend.OrbitsWrap`, `self.backend.SOBBHTDIonTheFlyWrap` and
  `self.backend.waveform_amp_phase_wrap(...)` on the same object. The native modules are
  imported when a backend is first loaded, not at `import bbhx`.
- The CPU build compiles all `.cu` sources as `.cxx` (copied via `add_custom_command`). It
  also copy-compiles `Detector.cu` and `lat_tdi_on_the_fly.cu` from the installed
  `lisatools` package (`LISATOOLS_CUTILS`) and `Interpolate.cu` from `gpubackendtools`
  (`GBT_CUTILS`). This is the same "LAT/GBT must be compiled into every consuming .so for
  typeinfo" pattern used across LISA Analysis Tools.
- `SpecialLikelihood.cu` and `new_hdyn_*` are GPU-only (`#ifdef __CUDACC__`), so the CPU
  loader sets `speciallike=None`, `new_hdyn_prep=None` and `new_hdyn_like=None`.
- `__CUDA_COMPILATION__` is defined `PUBLIC` on GPU targets. Without it, the `.cxx`
  binding TUs picked the wrong CPU/GPU class alias, which caused `undefined symbol:
  _ZTV*CPU` link failures (bug fixed 2026-06).
- CPU/GPU class-name aliasing: both the underlying C++ classes
  (`SOBBHTDIonTheFly{CPU,GPU}`, `SOBBHComputationGroup{CPU,GPU}`,
  `BBHxComputationWrap{CPU,GPU}`) and their nanobind wraps
  (`SOBBHTDIonTheFlyWrap{CPU,GPU}`, `SOBBHComputationGroupWrap{CPU,GPU}`) are
  `#define`-aliased at the top of `sobbh_tdi_on_the_fly.hh` / `binding_bbhx.hpp`, per the
  LISA Analysis Tools–wide rule. So the CPU and CUDA plugin wheels never collide in
  nanobind's global type registry.
- **Single-registrant rule**: `binding_bbhx.cxx` has
  `static_assert(!LISATOOLS_IS_WRAPPER_OWNER, ...)`. BBHx must never register
  `OrbitsWrap` / `TDIConfigWrap` / `LISAResponseWrap` / the WDM-FD-Spline wraps (LAT owns
  those) or `CubicSplineWrap` (GBT owns that).
- Constants: `cutils/constants.h` defines the repo-local names (`C_SI`, `MTSUN_SI`, ...)
  from the generated `lisaconstants_values.h`. Python `utils/constants.py` imports
  `lisaconstants`. `tests/test_lisaconstants_values.py` guards both.

## 6. Cross-repo dependencies

**Imports from:**
- `gpubackendtools` (GBT): `ParallelModuleBase`, `Globals`, `get_backend` /
  `has_backend` / `get_first_backend`, `Backend`, `CpuBackend` / `Cuda{11,12,13}xBackend`,
  `gpubackendtools.exceptions`, `interpolate.CubicSplineInterpolant`,
  `gbt_backend_*.interp` (CubicSpline). Build time: `Interpolate.cu`, `gbt_global.h`,
  `cmake_functions.cmake` / `get_lapacke()`.
- `lisatools` (LAT):
  - `lisatools.cutils.{LISAToolsBackend,LISAToolsBackendMethods}`;
  - `lisatools.detector.{Orbits,EqualArmlengthOrbits,ESAOrbits}`;
  - `lisatools.response.{directresponse.ResponseWrapper,tdiconfig.TDIConfig,
    tdionfly.TDTDIonTheFly}`;
  - `lisatools.domains.{TDSettings,TDSignal,FDSettings}`;
  - `lisatools.sources.sobbh.waveform.SOBBHWaveform`;
  - `lisatools.chunked_het.WDMComputationsBase`;
  - `lisatools.jax.response.base.JaxAmpPhaseSource`;
  - `lisatools.sensitivity.{SensitivityMatrix,AET1SensitivityMatrix}`;
  - `lisatools.utils.constants.YRSID_SI`.

  Build time: `Detector.{hpp,cu}`, `lat_tdi_on_the_fly.{hh,cu}`,
  `lat_chunked_het_kernels.hh`, `lisatools_header_abi.hpp`, `binding_flr.hpp`,
  `binding_lat_spline_tdi.hpp`, `binding_wdm_settings.hpp`, `fd_domain.hh`,
  `wdm_settings.hh`, `wdm_domain.hh`.
- `lisaconstants` (pinned `==2.0.2`): `utils/constants.py`.
- `eryn`: `eryn.utils.transform.TransformContainer` (only in `mbhphentax.py`; `eryn` is in
  the `sampling` extra, not a core dependency).
- `phentax` (external, `asantini29/phentax`, not on PyPI): optional extra `bbhx[phentax]`,
  imported lazily so the package works without it.
- No `gbgpu` import (the old GBGPU-copied `utils/citations.py` / `utils/config.py` were
  removed in `36a3419`).

**Depends on BBHx (reverse):** only LISAanalysistools, mainly:

- `lisatools/sampling/moves/skymodehop.py` (frame maps, `ConstOmega`);
- `lisatools/globalfit/stock/erebor/{transforms,wrappers,source_runtime}.py`;
- `lisatools/globalfit/recipe.py` (import of `SSB_to_LISA`);
- `lisatools/globalfit/moves/mbhspecialmove.py` and `lisatools/globalfit/mbhsearch.py`
  (FD likelihoods; not on the production path);
- `lisatools/sources/{bbh/waveform.py,sobbh/response.py}`;
- many `LISAanalysistools/scripts/{mbh,sobbh}/*.py` dev / validation scripts, `tests/`,
  and the `LISAanalysistools/global_fit_input/*mbh*` compatibility stubs.

`globalfit/{generatefuncs,plot,buildcatalog}.py` carry only commented-out bbhx imports. No
other LISA Analysis Tools repo (GBGPU, FastEMRIWaveforms, Eryn, lisa-on-gpu) imports
`bbhx`. GBGPU's `SOBBHTDIonTheFlyWrapJAX` (`gbgpu/jax/wrappers.py`) imports a
`gbgpu.jax.sources.sobbh` module that does not exist. LATW uses
`bbhx.likelihood.HeterodynedLikelihood` in `tutorials/further/X4_MBHB_MCMC.ipynb`.

## 7. Non-obvious invariants / gotchas

- **SOBBH signal-het has no package front end.** The native
  `sobbh_signal_het_*` kernels shipped (2026-06-18/19), but the only Python caller is
  the LAT script prototype `scripts/sobbh/sobbhsignalhetcomputations.py`
  (`SOBBHSignalHetComputations`). The separate files once planned in `CLAUDE.md` were
  not created (see the "Arrivals" list there).
- **Units differ per class.** The distance is in metres for `BBHWaveformFD` /
  `PhenomHMAmpPhase`, in Mpc for `MBHTDIonFly` with phentax, in Gpc for `SOBBHTDIonFly`,
  and in parsecs in the 11-parameter SOBBH vector (native kernel, `SOBBHWDMComputations`,
  `JaxSOBBHSource`). Masses are solar everywhere. See `docs/source/user/conventions.rst`.
- **Mass ratio:** `utils.transform.mT_q` uses `q = m2/m1 <= 1`, while LAT's stock MBH
  basis uses `Q = m1/m2 >= 1` (`stock/erebor/transforms.mT_Q`).
- **`f_ref` is ignored by the PhenomHM kernel.** `PhenomHMWaveform.cu::IMRPhenomHMCore`
  sets `Mf_ref = fmaxCalc` unconditionally (the `f_ref == 0` guard is commented out).
  `t_ref` / `phi_ref` therefore refer to the PhenomD (2,2) amplitude-peak frequency.
  `phi_ref` is applied by the response (`Y_lm(inc, phi_ref)`), not by the waveform kernel.
- **FD response frame:**
  - the transfer functions use the orbit file's spacecraft positions, while the
    orbital-delay phase uses an analytic circular ecliptic orbit (`Response.cu`,
    `// TODO: fix`); `lam` / `beta` agree with both only for ecliptic orbit data (the
    packaged LAT orbit files are SSB-ecliptic whatever their `frame="icrs"` label);
  - `phaseRdelay` is applied to both the transfer functions and the phases (open TODO);
  - `BBHWaveformFD.__call__` carries a TODO on the merger arrival time vs sky position;
  - `LISATDIResponse(orbits=None)` builds `EqualArmlengthOrbits()` with the default
    backend resolution, then asserts that its flavor matches, so pass orbits explicitly.
- **TDI-on-the-fly sky frame:** `MBHTDIonFly` / `SOBBHTDIonFly` read `(ra, dec, psi)` in
  the frame of the orbit data (`orbits.frame`). `SSB_to_LISA` / `LISA_to_SSB` take SSB
  ecliptic angles; the LAT sky-mode move rotates ICRS to ecliptic first.
- **`HeterodynedLikelihood` default reference is broken.** `BBHWaveformFD` with
  `fill=True, combine=False` ignores `squeeze` and returns `(1, 3, n)`, while
  `init_heterodyne_info` indexes it as `(3, n)`. The band search then picks up channel
  indices, and the setup fails (IndexError on CPU). Only `reference_gen_kwargs=
  dict(direct=True)` works. `tests/test_bbhx.py::test_het_likelihood`, the BBHx tutorial
  and LATW X4 use the default path. That test is expected to fail at HEAD (not run during
  the docs pass).
- **Every `citation` property raises `NameError`** (the `bbhx.utils.citations` module was
  removed, and its imports are commented out).
- **`bbhx.get_first_backend` is broken**: a `str` is iterated character by character, and
  a `list` fails the `"bbhx_" + ...` concatenation. `has_backend("cuda")` / `("gpu")`
  raise `ValueError`, because `bbhx_cuda` is not a registry name. Only `get_backend` and
  `force_backend` resolve the `cuda` / `gpu` aliases.
- **cutils exception classes:**
  - the CPU and CUDA module loaders raise
    `gpubackendtools.exceptions.BackendUnavailableException`, which is not the class the
    registry catches (`gpubackendtools.gpubackendtools.BackendUnavailableException`); a
    missing LAT/GBT native module therefore propagates out of `has_backend` instead of
    returning `False`;
  - the missing-`cupy` branch names `MissingDependencies`, which is not imported (so it
    raises `NameError`).
- **MBH TDI-on-the-fly reference phase** (`MBHTDIonFly.__call__`, comment above the
  `compute_strain_components_amp_phase` call): phentax is called with
  `t_ref = -t_merge` relative to merger, so `phi_ref` is the phase at absolute time `t0`.
  This matches the `pyResponse` / mojito convention. `t_ref = 0` left an ~11% edge-on
  mismatch from the higher modes.
- **MBH TDI-on-the-fly zero-tail padding** (`MBHTDIonFly.__call__`, `n_tail` /
  `dt_tail`): 120 samples at 10 s are appended past the phentax grid end. The on-the-fly
  kernel's retarded-time reads (up to ~600 s past the evaluation-window edge for unlucky
  sky positions) then never fall outside the spline. The physical post-ringdown amplitude
  there is ~1e-19 of peak. `dt_tdi_eval`, `Tobs` and `t_min` are accepted but unused.
- **SOBBH TDI-on-the-fly inclination convention differs from MBH** (`SOBBHTDIonFly` class
  docstring, "Convention note"). MBH folds the inclination into the per-mode spherical
  harmonic and passes `inc=0` to the response. SOBBH passes the *real* inclination,
  because its amplitude is the intrinsic (no-inclination) PN amplitude.
- **`phentax` coarse-graining scale** (`mbhphentax.MBH_TDIONFLY_COARSE_SCALE = 48.0`):
  this value is required to resolve the merger/ringdown (~0.886 s spacing). The phentax
  default of 12 under-resolves it and leaves a spurious ~5e-22 mismatch floor. The
  `mbh_tdionfly_tutorial` notebook uses the phentax default.
- **SOBBH chunked-het param order** (`sobbhcomps.py`, matches `SOBBHTDIonTheFly`):
  `(m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta)`, 11 params. The
  `f0` / carrier-frequency column is index 5 (`_F0_PARAM_INDEX`), unlike GB's 9-param
  layout. `SOBBHWDMComputations` inherits a `supported_backends` list with a `bbhx_jax`
  entry, but BBHx registers no JAX backend.
- **`NewHeterodynedLikelihood`** is GPU-only. Its `init_heterodyne_info` can stop at a
  `breakpoint()`. It supports single-mode templates only, and `reference_d_d` is indexed
  by reference index (correct only for one-to-one reference / data mapping).
- **`PhenomHMAmpPhase._sanity_check_params`** swaps `m1/m2` and the spins IN PLACE on the
  caller's arrays.
- **Docs build imports:** `docs/source/user/sobbh.rst` autodocs `JaxSOBBHSource` (needs
  `jax`), and `main.rst` autodocs `bbhx.mbhphentax` (needs `eryn`). Neither is in the `doc`
  extra.

## 8. Where to look for X

| Change / understand X | Start in |
|---|---|
| Classic MBH FD waveform generation (PhenomHM/PhenomD amp+phase) | `src/bbhx/waveforms/phenomhm.py`, `src/bbhx/cutils/PhenomHMWaveform.{hh,cu}` |
| Classic MBH FD LISA response | `src/bbhx/response/fastfdresponse.py`, `src/bbhx/cutils/Response.{hh,cu}` |
| Classic FD waveform orchestration / interpolation to data grid | `src/bbhx/waveformbuild.py`, `src/bbhx/cutils/WaveformBuild.{hh,cu}` |
| MBH FD likelihoods (direct, heterodyned) | `src/bbhx/likelihood.py`, `src/bbhx/cutils/Likelihood.{hh,cu}`, `src/bbhx/cutils/SpecialLikelihood.{hh,cu}` |
| MBH time-domain TDI-on-the-fly | `src/bbhx/mbhtdionfly.py` (LAT builder: `lisatools/globalfit/stock/erebor/wrappers.py`) |
| SOBBH time-domain TDI-on-the-fly | `src/bbhx/sobbhtdionfly.py` |
| SOBBH PN source physics (C++) | `src/bbhx/cutils/sobbh_tdi_on_the_fly.{hh,cu}` (`SOBBHTDIonTheFly::sobbh_{amplitude,phase,f,fdot}`) |
| SOBBH PN source physics (JAX) | `src/bbhx/jax/sources/sobbh.py` + the `sobbhtaylert3.py` prototype (LAT `scripts/sobbh/`) |
| SOBBH chunked-heterodyne likelihood (Python routing) | `src/bbhx/sobbhcomps.py` |
| SOBBH chunked-het / signal-het kernels (C++) | `src/bbhx/cutils/sobbh_tdi_on_the_fly.{hh,cu}` (`SOBBHComputationGroup`) |
| SSB <-> LISA frame maps, sky modes, mass ratio | `src/bbhx/utils/transform.py`, `tests/test_frame_transform.py` |
| Constants | `src/bbhx/utils/constants.py`, `src/bbhx/cutils/constants.h`, `src/bbhx/cutils/lisaconstants_values.h` |
| nanobind module surface / adding a new bound method | `src/bbhx/cutils/binding_bbhx.{hpp,cxx}` |
| Backend composition / what symbols a backend carries | `src/bbhx/cutils/__init__.py` |
| Native build (CMake), CPU/GPU target wiring, LAT/GBT header consumption | `src/bbhx/cutils/CMakeLists.txt`, root `CMakeLists.txt` |
| Backend/parallel-module base class | `src/bbhx/utils/parallelbase.py` |
| Package-level backend registration / imports | `src/bbhx/__init__.py` |
| Legacy phentax adapters / transform containers | `src/bbhx/mbhphentax.py` |
| Tests / usage examples for the classic pipeline | `tests/test_bbhx.py` |
| Tests for MBH TDI-on-the-fly | `tests/test_mbhtdionfly.py` |
| Sphinx docs | `docs/source/index.rst`, `docs/source/user/*.rst` |
| Tutorial notebooks | `examples/bbhx_tutorial.ipynb`, `examples/sobbh_tutorial.ipynb`, `examples/mbh_tdionfly_tutorial.ipynb` |
