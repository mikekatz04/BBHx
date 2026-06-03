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
        // Method wrappers populated in subsequent commits as each
        // Cython .pyx module is migrated.
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
