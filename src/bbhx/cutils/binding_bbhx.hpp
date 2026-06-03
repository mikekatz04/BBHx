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

// BBHx-specific waveform/response/likelihood headers are deliberately NOT
// included by the skeleton -- BBHxComputationWrap has no method bodies yet,
// and the locally-defined Interpolate.hh collides with GBT's
// `__INTERPOLATE_HH__` guard, which would hide the CubicSpline definition
// that binding_flr.hpp (transitively) needs. As each Cython module migrates
// into BBHxComputationWrap, the corresponding BBHx header (e.g.
// PhenomHMWaveform.hh, Response.hh, ...) gets added below; reuse
// InterpolateDevice.hh (from GBT) rather than BBHx's Interpolate.hh when
// CubicSpline is needed.

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

// Skeleton wrapper. Phase BBHx.pybind ships the empty class + the
// pybind11 module entrypoint; subsequent commits populate
// BBHxComputationWrap with method wrappers as each Cython module
// (phenomhm.pyx, interp.pyx, bbhlikelihood.pyx, lisaresponse.pyx,
// bbhwaveformbuild.pyx, newhdynlike.pyx) is migrated to pybind11.
//
// Inherits from ReturnPointerBase so future method wrappers can use
// return_pointer_and_check_length / return_pointer with the LAT-side
// implementations.
class BBHxComputationWrap : public ReturnPointerBase {
  public:
    BBHxComputationWrap() = default;
    ~BBHxComputationWrap() = default;

    // Method wrappers are intentionally omitted in the skeleton commit.
    // When migrating a Cython module (e.g. phenomhm.pyx) here, add a
    // class member that adapts the array_type<T> -> raw pointer for
    // each existing free function (waveform_amp_phase,
    // get_phenomhm_ringdown_frequencies, ...).
};

// Module entry called from PYBIND11_MODULE(cbbhx, m) in binding_bbhx.cxx.
void bbhx_part(py::module &m);

#endif // __BINDING_BBHX_HPP__
