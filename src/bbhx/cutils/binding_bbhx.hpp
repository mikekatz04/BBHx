#ifndef __BINDING_BBHX_HPP__
#define __BINDING_BBHX_HPP__

// BBHx pybind11 module surface.
//
// Adapted from origin/pybind branch (commits 7c383e3, 0543508, bd9e127)
// against the post-Phase-3L LAT setup: instead of redeclaring
// ReturnPointerBase + OrbitsWrap_bbhx + Detector here, we consume the
// canonical versions LAT registers in pycppdetector. This keeps the
// single-registrant rule intact -- LAT is the sole pybind11 registrant
// of OrbitsWrap, TDIConfigWrap, LISAResponseWrap, the
// LISATDIonTheFly base, and the FD/WDM/Spline Wraps. BBHx's cbbhx
// module is reserved for BBH-specific wrappers (PhenomHM, Response,
// WaveformBuild, Likelihood, Interpolate) and -- once Phase 3L.8 lands
// -- SOBBHTDIonTheFly + SOBBHComputationGroup.

// GBT's InterpolateDevice.hh defines the `CubicSpline` class that
// binding_flr.hpp's `CubicSplineWrap_responselisa` needs. Include it
// FIRST so that when binding_flr.hpp transitively pulls in `Interpolate.hh`
// (GBT's, which #includes InterpolateDevice.hh), the include guards are
// already satisfied and CubicSpline is visible. Without this, BBHx's
// local `Interpolate.hh` (same `__INTERPOLATE_HH__` guard, no CubicSpline)
// wins the include race and CubicSpline is undeclared.
#include "InterpolateDevice.hh"

// BBHx-specific waveform/response/likelihood headers. Each migrated
// Cython module's free functions get a method wrapper on
// BBHxComputationWrap, calling out to the underlying free function from
// the corresponding .hh below.
#include "PhenomHMWaveform.hh"  // waveform_amp_phase,
                                 // get_phenomhm_ringdown_frequencies_wrap,
                                 // get_phenomd_ringdown_frequencies_wrap
#include "Interpolate.hh"       // interpolate (BBHx's local, not GBT's)
#include "Likelihood.hh"        // hdyn, direct_like, prep_hdyn_wrap,
                                 // new_hdyn_prep_wrap, new_hdyn_like_wrap
#include "Response.hh"          // LISA_response
#include "WaveformBuild.hh"     // InterpTDI, direct_sum
// SpecialLikelihood.hh is GPU-only: its underlying SpecialLikelihood.cu has
// `cmplx trans_complex1 = 0.0;`-style initializers that only compile under
// the cuda_complex.hpp GPU branch (the CPU std::complex<double> typedef
// doesn't accept scalar 0.0). The corresponding `speciallike` wrapper is
// guarded by the same toggle below; the CPU __init__.py loader passes
// `speciallike=None` (matches the prior Cython-era behavior).
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#include "SpecialLikelihood.hh" // InterpTDILike (GPU only)
#endif

// LAT-canonical pybind11 base + array typedefs + Orbits + TDIConfig wrappers.
// binding_flr.hpp provides ReturnPointerBase and array_type<T>; consuming TUs
// MUST leave LISATOOLS_IS_WRAPPER_OWNER at its default (0) to satisfy the
// per-TU static_assert (BBHx never re-registers OrbitsWrap et al).
#include "lisatools_header_abi.hpp"
#include "binding_flr.hpp"
// Phase 3L.8 (2026-06-04): SOBBH-specific pybind11 Wraps for
// SOBBHTDIonTheFly + SOBBHComputationGroup. binding_lat_spline_tdi.hpp
// supplies the LISATDIonTheFlyWrap base that SOBBHTDIonTheFlyWrap
// inherits from. sobbh_tdi_on_the_fly.hh supplies the underlying C++
// classes + the sobbh_run_wave_tdi_wrap host launcher.
#include "binding_lat_spline_tdi.hpp" // LISATDIonTheFlyWrap (parent of SOBBHTDIonTheFlyWrap)
#include "wdm_settings.hh"            // WDMSettings (consumed by SOBBHComputationGroup methods)
#include "binding_wdm_settings.hpp"   // WDMSettingsWrap (constructor arg)
#include "sobbh_tdi_on_the_fly.hh"    // SOBBHTDIonTheFly + SOBBHComputationGroup + sobbh_run_wave_tdi_wrap

#include <string>
#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define BBHxComputationWrap BBHxComputationWrapGPU
// Phase 3L.8 aliases: SOBBH pybind11 Wraps carved from lisa-on-gpu.
#define SOBBHTDIonTheFlyWrap SOBBHTDIonTheFlyWrapGPU
#define SOBBHComputationGroupWrap SOBBHComputationGroupWrapGPU
#else
#define BBHxComputationWrap BBHxComputationWrapCPU
#define SOBBHTDIonTheFlyWrap SOBBHTDIonTheFlyWrapCPU
#define SOBBHComputationGroupWrap SOBBHComputationGroupWrapCPU
#endif

// Unified BBHx pybind11 wrapper. Inherits from ReturnPointerBase so
// every method body can use return_pointer_and_check_length /
// return_pointer (LAT-canonical implementations) to adapt
// array_type<T> -> raw pointer.
//
// Methods migrate in here one Cython module at a time. Already
// migrated:
// - phenomhm.pyx -> waveform_amp_phase_wrap,
//                   get_phenomhm_ringdown_frequencies,
//                   get_phenomd_ringdown_frequencies
//
// Pending migration:
// - interp.pyx -> interpolate
// - bbhlikelihood.pyx -> direct_like_wrap, hdyn_wrap
// - lisaresponse.pyx -> LISA_response_wrap, prep_hdyn,
//                       pyFastLISAResponse
// - bbhwaveformbuild.pyx -> direct_sum_wrap, InterpTDI_wrap
// - newhdynlike.pyx -> new_hdyn_prep, new_hdyn_like
//
// Phase 3L.8 lands SOBBHTDIonTheFly + SOBBHComputationGroup here.
class BBHxComputationWrap : public ReturnPointerBase {
  public:
    BBHxComputationWrap() = default;
    ~BBHxComputationWrap() = default;

    // ---- PhenomHMWaveform.hh wrappers (migrated from phenomhm.pyx) ----
    //
    // NOTE: the Cython phenomhm.pyx had no length checks (just cast
    // size_t pointers to T* and passed through). The migrated wrappers
    // below preserve that behavior using `return_pointer` everywhere
    // (no `_and_check_length` variant), since some callers pass
    // oversized buffers as scratch space -- e.g. PhenomHMAmpPhase.run_wave
    // sizes `fringdown` / `fdamp` as `num_modes * num_bin_all` but
    // get_phenomd_ringdown_frequencies only writes the first numBinAll
    // entries. Tightening to checked variants is a follow-up commit.

    void waveform_amp_phase_wrap(
        array_type<double> waveformOut,
        array_type<int>    ells_in,
        array_type<int>    mms_in,
        array_type<double> freqs,
        array_type<double> m1_SI,
        array_type<double> m2_SI,
        array_type<double> chi1z,
        array_type<double> chi2z,
        array_type<double> distance,
        array_type<double> f_ref,
        int numModes, int length, int numBinAll,
        array_type<double> Mf_RD_lm_all,
        array_type<double> Mf_DM_lm_all,
        int run_phenomd)
    {
        waveform_amp_phase(
            return_pointer(waveformOut,  "waveformOut"),
            return_pointer(ells_in,      "ells_in"),
            return_pointer(mms_in,       "mms_in"),
            return_pointer(freqs,        "freqs"),
            return_pointer(m1_SI,        "m1_SI"),
            return_pointer(m2_SI,        "m2_SI"),
            return_pointer(chi1z,        "chi1z"),
            return_pointer(chi2z,        "chi2z"),
            return_pointer(distance,     "distance"),
            return_pointer(f_ref,        "f_ref"),
            numModes, length, numBinAll,
            return_pointer(Mf_RD_lm_all, "Mf_RD_lm_all"),
            return_pointer(Mf_DM_lm_all, "Mf_DM_lm_all"),
            run_phenomd);
    }

    void get_phenomhm_ringdown_frequencies(
        array_type<double> fringdown,
        array_type<double> fdamp,
        array_type<double> m1,
        array_type<double> m2,
        array_type<double> chi1z,
        array_type<double> chi2z,
        array_type<int>    ells_in,
        array_type<int>    mm_in,
        int numModes, int numBinAll)
    {
        get_phenomhm_ringdown_frequencies_wrap(
            return_pointer(fringdown, "fringdown"),
            return_pointer(fdamp,     "fdamp"),
            return_pointer(m1,        "m1"),
            return_pointer(m2,        "m2"),
            return_pointer(chi1z,     "chi1z"),
            return_pointer(chi2z,     "chi2z"),
            return_pointer(ells_in,   "ells_in"),
            return_pointer(mm_in,     "mm_in"),
            numModes, numBinAll);
    }

    void get_phenomd_ringdown_frequencies(
        array_type<double> fringdown,
        array_type<double> fdamp,
        array_type<double> m1,
        array_type<double> m2,
        array_type<double> chi1z,
        array_type<double> chi2z,
        int numBinAll,
        array_type<double> y_rd_all,
        array_type<double> c1_rd_all,
        array_type<double> c2_rd_all,
        array_type<double> c3_rd_all,
        array_type<double> y_dm_all,
        array_type<double> c1_dm_all,
        array_type<double> c2_dm_all,
        array_type<double> c3_dm_all,
        double dspin,
        int num_segs)
    {
        get_phenomd_ringdown_frequencies_wrap(
            return_pointer(fringdown, "fringdown"),
            return_pointer(fdamp,     "fdamp"),
            return_pointer(m1,        "m1"),
            return_pointer(m2,        "m2"),
            return_pointer(chi1z,     "chi1z"),
            return_pointer(chi2z,     "chi2z"),
            numBinAll,
            return_pointer(y_rd_all,  "y_rd_all"),
            return_pointer(c1_rd_all, "c1_rd_all"),
            return_pointer(c2_rd_all, "c2_rd_all"),
            return_pointer(c3_rd_all, "c3_rd_all"),
            return_pointer(y_dm_all,  "y_dm_all"),
            return_pointer(c1_dm_all, "c1_dm_all"),
            return_pointer(c2_dm_all, "c2_dm_all"),
            return_pointer(c3_dm_all, "c3_dm_all"),
            dspin, num_segs);
    }

    // ---- Interpolate.hh wrapper (migrated from interp.pyx) ----

    void interpolate_wrap(
        array_type<double> freqs, array_type<double> propArrays,
        array_type<double> B, array_type<double> upper_diag,
        array_type<double> diag, array_type<double> lower_diag,
        int length, int numInterpParams, int numModes, int numBinAll)
    {
        interpolate(
            return_pointer(freqs,      "freqs"),
            return_pointer(propArrays, "propArrays"),
            return_pointer(B,          "B"),
            return_pointer(upper_diag, "upper_diag"),
            return_pointer(diag,       "diag"),
            return_pointer(lower_diag, "lower_diag"),
            length, numInterpParams, numModes, numBinAll);
    }

    // ---- Likelihood.hh wrappers (migrated from bbhlikelihood.pyx) ----

    void hdyn_wrap(
        array_type<std::complex<double>> likeOut1,
        array_type<std::complex<double>> likeOut2,
        array_type<std::complex<double>> templateChannels,
        array_type<std::complex<double>> dataConstants,
        array_type<double> dataFreqs,
        int numBinAll, int data_length, int nChannels)
    {
        hdyn(
            (cmplx*) return_pointer(likeOut1,         "likeOut1"),
            (cmplx*) return_pointer(likeOut2,         "likeOut2"),
            (cmplx*) return_pointer(templateChannels, "templateChannels"),
            (cmplx*) return_pointer(dataConstants,    "dataConstants"),
            return_pointer(dataFreqs, "dataFreqs"),
            numBinAll, data_length, nChannels);
    }

    void direct_like_wrap(
        array_type<std::complex<double>> d_h,
        array_type<std::complex<double>> h_h,
        array_type<std::complex<double>> dataChannels,
        array_type<double> noise_weight_times_df,
        array_type<long> templateChannels_ptrs,
        array_type<int> inds_start,
        array_type<int> ind_lengths,
        int data_stream_length, int numBinAll, int nChannels, int device)
    {
        direct_like(
            (cmplx*) return_pointer(d_h,          "d_h"),
            (cmplx*) return_pointer(h_h,          "h_h"),
            (cmplx*) return_pointer(dataChannels, "dataChannels"),
            return_pointer(noise_weight_times_df,    "noise_weight_times_df"),
            return_pointer(templateChannels_ptrs,    "templateChannels_ptrs"),
            return_pointer(inds_start,               "inds_start"),
            return_pointer(ind_lengths,              "ind_lengths"),
            data_stream_length, numBinAll, nChannels, device);
    }

    void prep_hdyn(
        array_type<std::complex<double>> A0_in,
        array_type<std::complex<double>> A1_in,
        array_type<std::complex<double>> B0_in,
        array_type<std::complex<double>> B1_in,
        array_type<std::complex<double>> d_arr,
        array_type<std::complex<double>> h0_arr,
        array_type<double> S_n_arr,
        double df,
        array_type<int> bins,
        array_type<double> f_dense,
        array_type<double> f_m_arr,
        int data_length, int nchannels, int length_f_rel)
    {
        prep_hdyn_wrap(
            (cmplx*) return_pointer(A0_in,  "A0_in"),
            (cmplx*) return_pointer(A1_in,  "A1_in"),
            (cmplx*) return_pointer(B0_in,  "B0_in"),
            (cmplx*) return_pointer(B1_in,  "B1_in"),
            (cmplx*) return_pointer(d_arr,  "d_arr"),
            (cmplx*) return_pointer(h0_arr, "h0_arr"),
            return_pointer(S_n_arr, "S_n_arr"),
            df,
            return_pointer(bins,    "bins"),
            return_pointer(f_dense, "f_dense"),
            return_pointer(f_m_arr, "f_m_arr"),
            data_length, nchannels, length_f_rel);
    }

    // ---- Response.hh wrapper (migrated from lisaresponse.pyx) ----
    //
    // `orbits` is the LAT-canonical OrbitsWrap* (the post-Phase-3E
    // wrapper from binding.hpp, NOT the OrbitsWrap from
    // binding_flr.hpp). pybind11's shared-type registry routes Python
    // `Orbits.pycppdetector` (= `self.backend.OrbitsWrap(*args)` in
    // lisatools/detector.py) through this signature. The Python frontend
    // in bbhx/response/fastfdresponse.py passes `self.orbits.pycppdetector`.
    void LISA_response_wrap(
        array_type<double> response_out,
        array_type<int>    ells_in,
        array_type<int>    mms_in,
        array_type<double> freqs,
        array_type<double> phi_ref,
        array_type<double> inc,
        array_type<double> lam,
        array_type<double> beta,
        array_type<double> psi,
        int TDItag, bool rescaled, bool tdi2, int order_fresnel_stencil,
        int numModes, int length, int numBinAll, int includesAmps,
        OrbitsWrap *orbits_wrap)
    {
        LISA_response(
            return_pointer(response_out, "response_out"),
            return_pointer(ells_in,      "ells_in"),
            return_pointer(mms_in,       "mms_in"),
            return_pointer(freqs,        "freqs"),
            return_pointer(phi_ref,      "phi_ref"),
            return_pointer(inc,          "inc"),
            return_pointer(lam,          "lam"),
            return_pointer(beta,         "beta"),
            return_pointer(psi,          "psi"),
            TDItag, rescaled, tdi2, order_fresnel_stencil,
            numModes, length, numBinAll, includesAmps,
            orbits_wrap->orbits);
    }

    // ---- WaveformBuild.hh wrappers (migrated from bbhwaveformbuild.pyx) ----

    void InterpTDI_wrap(
        array_type<long> templateChannels_ptrs,
        array_type<double> dataFreqs,
        array_type<double> freqs,
        array_type<double> propArrays,
        array_type<double> c1, array_type<double> c2, array_type<double> c3,
        array_type<double> t_start, array_type<double> t_end,
        int length, int data_length, int numBinAll, int numModes,
        array_type<long> inds_ptrs,
        array_type<int> inds_start,
        array_type<int> ind_lengths)
    {
        InterpTDI(
            return_pointer(templateChannels_ptrs, "templateChannels_ptrs"),
            return_pointer(dataFreqs,             "dataFreqs"),
            return_pointer(freqs,                 "freqs"),
            return_pointer(propArrays,            "propArrays"),
            return_pointer(c1,                    "c1"),
            return_pointer(c2,                    "c2"),
            return_pointer(c3,                    "c3"),
            return_pointer(t_start,               "t_start"),
            return_pointer(t_end,                 "t_end"),
            length, data_length, numBinAll, numModes,
            return_pointer(inds_ptrs,    "inds_ptrs"),
            return_pointer(inds_start,   "inds_start"),
            return_pointer(ind_lengths,  "ind_lengths"));
    }

    void direct_sum_wrap(
        array_type<std::complex<double>> templateChannels,
        array_type<double> bbh_buffer,
        int numBinAll, int data_length, int nChannels, int numModes,
        array_type<double> t_start, array_type<double> t_end)
    {
        direct_sum(
            (cmplx*) return_pointer(templateChannels, "templateChannels"),
            return_pointer(bbh_buffer, "bbh_buffer"),
            numBinAll, data_length, nChannels, numModes,
            return_pointer(t_start, "t_start"),
            return_pointer(t_end,   "t_end"));
    }

    // ---- Likelihood.hh extra (migrated from newhdynlike.pyx; GPU only) ----
    //
    // new_hdyn_prep_wrap / new_hdyn_like_wrap are inside the
    // `#ifdef __CUDACC__` block in Likelihood.cu -- they use GPU triple-
    // bracket kernel-launch syntax that doesn't survive the CPU compile.
    // The CPU __init__.py loader sets new_hdyn_* = None (matches the
    // prior Cython-era setup: bbhx_cpu_newhdyn was commented out, only
    // bbhx_gpu_newhdyn shipped).
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    void new_hdyn_like(
        array_type<std::complex<double>> likeOut1,
        array_type<std::complex<double>> likeOut2,
        array_type<std::complex<double>> templateChannels,
        array_type<std::complex<double>> dataConstants,
        array_type<double> dataFreqsIn,
        array_type<int> constants_index,
        int numBinAll, int length_f_rel, int nChannels, int num_constants)
    {
        new_hdyn_like_wrap(
            (cmplx*) return_pointer(likeOut1,         "likeOut1"),
            (cmplx*) return_pointer(likeOut2,         "likeOut2"),
            (cmplx*) return_pointer(templateChannels, "templateChannels"),
            (cmplx*) return_pointer(dataConstants,    "dataConstants"),
            return_pointer(dataFreqsIn,     "dataFreqsIn"),
            return_pointer(constants_index, "constants_index"),
            numBinAll, length_f_rel, nChannels, num_constants);
    }

    void new_hdyn_prep(
        array_type<std::complex<double>> A0_out,
        array_type<std::complex<double>> A1_out,
        array_type<std::complex<double>> B0_out,
        array_type<std::complex<double>> B1_out,
        array_type<std::complex<double>> h0_arr,
        array_type<std::complex<double>> data,
        array_type<double> psd,
        array_type<double> f_m_arr,
        double df,
        array_type<double> f_dense,
        array_type<int> data_index,
        array_type<int> noise_index,
        array_type<int> start_inds_all,
        array_type<int> num_points_seg,
        int length_f_rel, int num_bin, int data_length, int nchannels)
    {
        new_hdyn_prep_wrap(
            (cmplx*) return_pointer(A0_out, "A0_out"),
            (cmplx*) return_pointer(A1_out, "A1_out"),
            (cmplx*) return_pointer(B0_out, "B0_out"),
            (cmplx*) return_pointer(B1_out, "B1_out"),
            (cmplx*) return_pointer(h0_arr, "h0_arr"),
            (cmplx*) return_pointer(data,   "data"),
            return_pointer(psd,            "psd"),
            return_pointer(f_m_arr,        "f_m_arr"),
            df,
            return_pointer(f_dense,        "f_dense"),
            return_pointer(data_index,     "data_index"),
            return_pointer(noise_index,    "noise_index"),
            return_pointer(start_inds_all, "start_inds_all"),
            return_pointer(num_points_seg, "num_points_seg"),
            length_f_rel, num_bin, data_length, nchannels);
    }
#endif // __CUDACC__ (new_hdyn_*: GPU only)

    // ---- SpecialLikelihood.hh wrapper (migrated from gpuonlywaveformbuild.pyx) ----
    //
    // GPU only -- SpecialLikelihood.cu uses cuda_complex.hpp's GPU-side
    // scalar-to-cmplx conversions that don't translate to the CPU
    // typedef. Matches the prior Cython-era setup where
    // bbhx_cpu_speciallike was commented out and only bbhx_gpu_speciallike
    // shipped. The CPU __init__.py loader sets speciallike=None.
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    void speciallike(
        array_type<std::complex<double>> d_h,
        array_type<std::complex<double>> h_h,
        array_type<std::complex<double>> dataChannels,
        array_type<double> psd,
        array_type<double> dataFreqs,
        array_type<double> freqs,
        array_type<double> propArrays,
        array_type<double> c1, array_type<double> c2, array_type<double> c3,
        array_type<double> t_start, array_type<double> t_end,
        int length, int data_length, int numBinAll, int numModes,
        array_type<long> inds_ptrs,
        array_type<int> inds_start,
        array_type<int> ind_lengths,
        double df,
        array_type<int> data_index_all, int num_data_sets,
        array_type<int> noise_index_all, int num_noise_sets,
        int gpu)
    {
        InterpTDILike(
            (cmplx*) return_pointer(d_h,          "d_h"),
            (cmplx*) return_pointer(h_h,          "h_h"),
            (cmplx*) return_pointer(dataChannels, "dataChannels"),
            return_pointer(psd,        "psd"),
            return_pointer(dataFreqs,  "dataFreqs"),
            return_pointer(freqs,      "freqs"),
            return_pointer(propArrays, "propArrays"),
            return_pointer(c1, "c1"), return_pointer(c2, "c2"), return_pointer(c3, "c3"),
            return_pointer(t_start, "t_start"), return_pointer(t_end, "t_end"),
            length, data_length, numBinAll, numModes,
            return_pointer(inds_ptrs,    "inds_ptrs"),
            return_pointer(inds_start,   "inds_start"),
            return_pointer(ind_lengths,  "ind_lengths"),
            df,
            return_pointer(data_index_all,  "data_index_all"), num_data_sets,
            return_pointer(noise_index_all, "noise_index_all"), num_noise_sets,
            gpu);
    }
#endif // __CUDACC__ (speciallike: GPU only)
};

// ============================================================================
// Phase 3L.8 (2026-06-04): SOBBH carve-out from lisa-on-gpu's binding_tof.hpp.
//
// `class SOBBHTDIonTheFlyWrap` -- pybind11 wrapper around
//   `SOBBHTDIonTheFly` (declared in sobbh_tdi_on_the_fly.hh). Inherits
//   from LAT-owned `LISATDIonTheFlyWrap`. Exposes the time-domain
//   run_wave_tdi path; SOBBH has no FD/heterodyne run_wave_tdi
//   counterpart (those are GB-only and stay in GBGPU).
//
// `class SOBBHComputationGroupWrap` -- pybind11 wrapper around
//   `SOBBHComputationGroup`. Hosts the SOBBH chunked-heterodyne
//   likelihood path (sobbh_wdm_het_*).
// ============================================================================

class SOBBHTDIonTheFlyWrap : public LISATDIonTheFlyWrap {
  public:
    SOBBHTDIonTheFly *waveform;
    double T;
    double t_ref;

    SOBBHTDIonTheFlyWrap(OrbitsWrap *orbits_, TDIConfigWrap *tdi_config_, double T_, double t_ref_): LISATDIonTheFlyWrap(orbits_, tdi_config_)
    {
        T = T_;
        t_ref = t_ref_;
        waveform = new SOBBHTDIonTheFly(orbits_->orbits, tdi_config_->tdi_config, T_, t_ref_);
    };
    ~SOBBHTDIonTheFlyWrap(){
        delete waveform;
    };

    void run_wave_tdi_wrap(
        array_type<std::complex<double>>tdi_channels_arr,
        array_type<double>tdi_amp, array_type<double>tdi_phase, array_type<double>phi_ref,
        array_type<double>params, array_type<double>t_arr, int N, int num_bin, int n_params, int nchannels
    );

    int get_buffer_size(int N){return waveform->get_sobbh_buffer_size(N);};
};


// SOBBH chunked-heterodyne wrap. Same shape as GBComputationGroupWrap's
// gb_wdm_het_* family (routes to LAT's templated wdm_het_*_impl via
// SOBBHComputationGroup), sobbh_ prefix throughout.
class SOBBHComputationGroupWrap: public SOBBHComputationGroup, public ReturnPointerBase {
  public:
    void sobbh_wdm_het_fill_global(
        array_type<double> template_fill,
        OrbitsWrap *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        WDMSettingsWrap *wdm_settings_wrap,
        array_type<double> params_all, array_type<double> factors_all,
        array_type<double> chunk_t_starts,
        array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
        array_type<int> chunk_n_global_offset,
        array_type<double> wdm_window,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref,
        double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit);

    void sobbh_wdm_het_get_ll(
        array_type<double> d_h_out, array_type<double> h_h_out,
        OrbitsWrap *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        WDMSettingsWrap *wdm_settings_wrap,
        array_type<double> params_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        array_type<double> chunk_t_starts,
        array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
        array_type<int> chunk_n_global_offset,
        array_type<double> wdm_window,
        array_type<double> data_d, array_type<double> invC,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
        array_type<int> binary_perm, array_type<int> group_starts, array_type<int> group_ends,
        array_type<int> group_m_lo, array_type<int> group_m_hi, int n_groups);

    void sobbh_wdm_het_swap_ll(
        array_type<double> d_h_add_out, array_type<double> d_h_remove_out,
        array_type<double> add_add_out, array_type<double> remove_remove_out,
        array_type<double> add_remove_out,
        OrbitsWrap *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        WDMSettingsWrap *wdm_settings_wrap,
        array_type<double> params_add_all, array_type<double> params_remove_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        array_type<double> chunk_t_starts,
        array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
        array_type<int> chunk_n_global_offset,
        array_type<double> wdm_window,
        array_type<double> data_d, array_type<double> invC,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
        array_type<int> binary_perm, array_type<int> group_starts, array_type<int> group_ends,
        array_type<int> group_m_lo, array_type<int> group_m_hi, int n_groups,
        array_type<int> pair_m_lo_b, array_type<int> pair_m_hi_b);

    void sobbh_wdm_het_get_fstat_ll(
        array_type<double> N_arr_re_out, array_type<double> N_arr_im_out,
        array_type<double> M_mat_re_out, array_type<double> M_mat_im_out,
        OrbitsWrap *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        WDMSettingsWrap *wdm_settings_wrap,
        array_type<double> params_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        array_type<double> chunk_t_starts,
        array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
        array_type<int> chunk_n_global_offset,
        array_type<double> wdm_window,
        array_type<double> data_d, array_type<double> invC,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha, int grid_dim, int m_band_half_width);
};


// Module entry called from NB_MODULE(cbbhx, m) in binding_bbhx.cxx.
void bbhx_part(nb::module_ &m);

#endif // __BINDING_BBHX_HPP__
