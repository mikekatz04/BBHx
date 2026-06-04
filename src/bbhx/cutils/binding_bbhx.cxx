// BBHx pybind11 module entry.
//
// Adapted from origin/pybind branch (binding_bbhx.cxx) against the
// post-Phase-3L LAT setup. The originally-on-pybind-branch redeclaration
// of OrbitsWrap_bbhx / its pybind11 registration has been removed --
// LAT's pycppdetector is the sole registrant of OrbitsWrap_responselisa
// (and the rest of the shared wrapper family). BBHx's cbbhx module is
// reserved for BBH-specific wrappers + the (future) SOBBHTDIonTheFly +
// SOBBHComputationGroup carve-out from Phase 3L.8.

#include "binding_bbhx.hpp"

// Single-registrant rule: this TU must not be marked as the wrapper owner.
// LAT's binding.cxx sets the toggle to 1; every downstream binding TU
// leaves it at 0 and asserts via this:
static_assert(!LISATOOLS_IS_WRAPPER_OWNER,
    "Single-registrant rule: only LISAanalysistools may register "
    "OrbitsWrap / TDIConfigWrap / LISAResponseWrap / WDM*/FD*/Spline* "
    "with pybind11. BBHx's cbbhx module is for BBH-specific wrappers "
    "only (plus the SOBBHTDIonTheFly carve-out at Phase 3L.8). "
    "See plan it-is-time-to-delegated-peach.md "
    "and the existing pattern in "
    "lisa-on-gpu/src/fastlisaresponse/cutils/binding_tof.cxx.");

void bbhx_part(py::module &m) {
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<BBHxComputationWrap>(m, "BBHxComputationWrapGPU")
#else
    py::class_<BBHxComputationWrap>(m, "BBHxComputationWrapCPU")
#endif
        .def(py::init<>())
        // PhenomHMWaveform.hh (migrated from phenomhm.pyx)
        .def("waveform_amp_phase_wrap",
             &BBHxComputationWrap::waveform_amp_phase_wrap,
             "PhenomHM/PhenomD amp + phase generator.")
        .def("get_phenomhm_ringdown_frequencies",
             &BBHxComputationWrap::get_phenomhm_ringdown_frequencies,
             "PhenomHM ringdown + damping frequencies.")
        .def("get_phenomd_ringdown_frequencies",
             &BBHxComputationWrap::get_phenomd_ringdown_frequencies,
             "PhenomD ringdown + damping frequencies (spline-based).")
        // Interpolate.hh (migrated from interp.pyx)
        .def("interpolate_wrap",
             &BBHxComputationWrap::interpolate_wrap,
             "Cubic-spline interpolation of propArrays.")
        // Likelihood.hh (migrated from bbhlikelihood.pyx)
        .def("hdyn_wrap",
             &BBHxComputationWrap::hdyn_wrap,
             "Heterodyned likelihood.")
        .def("direct_like_wrap",
             &BBHxComputationWrap::direct_like_wrap,
             "Direct (full-FD) likelihood.")
        .def("prep_hdyn",
             &BBHxComputationWrap::prep_hdyn,
             "Prepare heterodyne coefficients (A0/A1/B0/B1).")
        // Response.hh (migrated from lisaresponse.pyx)
        .def("LISA_response_wrap",
             &BBHxComputationWrap::LISA_response_wrap,
             "FastFD LISA response (PhenomHM amp+phase -> TDI).")
        // WaveformBuild.hh (migrated from bbhwaveformbuild.pyx)
        .def("InterpTDI_wrap",
             &BBHxComputationWrap::InterpTDI_wrap,
             "Interpolate sparse PhenomHM modes onto dense TDI channels.")
        .def("direct_sum_wrap",
             &BBHxComputationWrap::direct_sum_wrap,
             "Sum PhenomHM modes -> TDI channels directly (no interp).")
        // Likelihood.hh extra (migrated from newhdynlike.pyx; GPU only)
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
        .def("new_hdyn_like",
             &BBHxComputationWrap::new_hdyn_like,
             "Hdyn likelihood with multiple constant-segment shifts.")
        .def("new_hdyn_prep",
             &BBHxComputationWrap::new_hdyn_prep,
             "Hdyn preparation with per-binary segment indexing.")
#endif
        // SpecialLikelihood.hh (migrated from gpuonlywaveformbuild.pyx; GPU only)
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
        .def("speciallike",
             &BBHxComputationWrap::speciallike,
             "InterpTDILike: fused interp + likelihood on the same kernel.")
#endif
        ;
}


PYBIND11_MODULE(cbbhx, m) {
    m.doc() = "BBHx pybind11 backend (skeleton). "
              "BBH waveforms + response + likelihood + interpolation will "
              "migrate into this module from their respective Cython "
              "submodules; SOBBHTDIonTheFly + SOBBHComputationGroup land "
              "here at Phase 3L.8.";
    bbhx_part(m);
}
