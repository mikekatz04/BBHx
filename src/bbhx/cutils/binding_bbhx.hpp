#ifndef __BINDING_BBHX_HPP__
#define __BINDING_BBHX_HPP__

// BBHx pybind11 module surface.
//
// Adapted from origin/pybind branch (commits 7c383e3, 0543508, bd9e127)
// against the post-Phase-3L LAT setup: instead of redeclaring
// ReturnPointerBase + OrbitsWrap_bbhx + Detector here, we consume the
// canonical versions LAT registers in pycppdetector. This keeps the
// single-registrant rule intact -- LAT is the sole pybind11 registrant
// of OrbitsWrap_responselisa, TDIConfigWrap, LISAResponseWrap, the
// LISATDIonTheFly base, and the FD/WDM/Spline Wraps. BBHx's cbbhx
// module is reserved for BBH-specific wrappers (PhenomHM, Response,
// WaveformBuild, Likelihood, Interpolate) and -- once Phase 3L.8 lands
// -- SOBBHTDIonTheFly + SOBBHComputationGroup.

// BBHx-specific waveform/response/likelihood headers. As each Cython
// module migrates into BBHxComputationWrap, its corresponding header is
// added below. NOTE: the locally-defined Interpolate.hh shares the
// `__INTERPOLATE_HH__` guard with GBT's, so if/when wrapping BBHx's
// interp.pyx, use GBT's InterpolateDevice.hh for CubicSpline rather than
// BBHx's local Interpolate.hh.
#include "PhenomHMWaveform.hh"  // waveform_amp_phase,
                                 // get_phenomhm_ringdown_frequencies_wrap,
                                 // get_phenomd_ringdown_frequencies_wrap

// LAT-canonical pybind11 base + array typedefs + Orbits + TDIConfig wrappers.
// binding_flr.hpp provides ReturnPointerBase and array_type<T>; consuming TUs
// MUST leave LISATOOLS_IS_WRAPPER_OWNER at its default (0) to satisfy the
// per-TU static_assert (BBHx never re-registers OrbitsWrap_responselisa et al).
#include "lisatools_header_abi.hpp"
#include "binding_flr.hpp"

#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define BBHxComputationWrap BBHxComputationWrapGPU
#else
#define BBHxComputationWrap BBHxComputationWrapCPU
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
};

// Module entry called from PYBIND11_MODULE(cbbhx, m) in binding_bbhx.cxx.
void bbhx_part(py::module &m);

#endif // __BINDING_BBHX_HPP__
