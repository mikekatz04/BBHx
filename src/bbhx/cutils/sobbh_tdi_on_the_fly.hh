#ifndef __SOBBH_TDI_ON_THE_FLY_HH__
#define __SOBBH_TDI_ON_THE_FLY_HH__

// SOBBH-specific TDI-on-the-fly machinery, carved out of lisa-on-gpu's
// `TDIonTheFly.{hh,cu}` + `binding_tof.{hpp,cxx}` at Phase 3L.8
// (2026-06-04). Mirrors GBGPU/src/gbgpu/cutils/gb_tdi_on_the_fly.hh.
//
// What lives here:
// - `class SOBBHTDIonTheFly : public LISATDIonTheFly` — stellar-origin
//   black-hole binary physics (PN amplitude/phase/frequency expansions
//   ported from sobbh_intrinsic_Ladeeda.cpp), `get_amp/phase/f/fdot`
//   virtual overrides, dtor, buffer-size helper.
// - SOBBH-specific kernel + host-wrapper functions:
//   - `sobbh_run_wave_tdi_kernel` + `sobbh_run_wave_tdi_wrap`
//     (time-domain TDI).
// - `class SOBBHComputationGroup` — Python-facing computation surface;
//   `sobbh_wdm_het_*_wrap` methods that instantiate LAT's templated
//   `wdm_het_*_impl<SOBBHTDIonTheFly>` launchers against the SOBBH
//   source class.
//
// What lives in LAT (post-Phase-3L.7a):
// - `LISATDIonTheFly` base + `OrbitsSplineCache` (Phase 3L.5)
// - Generic FD/WDM domains: `FDDomain`, `WDMSettings`, `WDMDomain`
//   (Phase 3L.1/2/4)
// - All source-agnostic chunked-het machinery: helpers + 4 templated
//   `wdm_het_*_kernel` bodies + 4 inline `wdm_het_*_impl<SourceT>`
//   launchers (Phase 3L.7a, in `lat_chunked_het_kernels.hh`)
//
// What lives in GBGPU (post-Phase-3L.7):
// - `GBTDIonTheFly` + `GBComputationGroup` mirrors (the GB-physics
//   counterpart of what's in this header).
//
// CPU/GPU class-name aliasing follows the sprint-wide rule — both the
// class and its `*Wrap` must be aliased so per-backend plugin wheels
// emit distinct C++ type names that pybind11 can register
// independently.

#include "Detector.hpp"               // Orbits + Vec
#include "LISAResponse.hh"            // TDIConfig
// CubicSpline + CUBIC_SPLINE_LINEAR_SPACING. We pull in the header-only
// InterpolateDevice.hh directly rather than going through GBT's
// Interpolate.hh -- both work post-2026-06-05 (BBHx no longer ships its
// own Interpolate.hh, so the local-shadow concern is gone), but the
// device-only header is what we actually need here.
#include "InterpolateDevice.hh"       // CubicSpline + CUBIC_SPLINE_LINEAR_SPACING
#include "fd_domain.hh"               // FDDomain
#include "wdm_settings.hh"            // WDMSettings
#include "wdm_domain.hh"              // WDMDomain
#include "lat_tdi_on_the_fly.hh"      // LISATDIonTheFly base + OrbitsSplineCache
#include "lat_chunked_het_kernels.hh" // wdm_het_*_impl<SourceT> + helpers
#include "gbt_global.h"               // cmplx + CUDA_DEVICE etc.


// CPU/GPU class-name aliasing -- one rule, both layers.
//
// (a) The C++ classes themselves (SOBBHTDIonTheFly + SOBBHComputationGroup):
//     both must be aliased so per-backend plugin wheels emit distinct
//     C++ type names. This is what allows the cuda12x + cpu plugin
//     wheels to coexist in the same Python interpreter without
//     pybind11 type-registry collisions.
// (b) The pybind11 wrappers (SOBBHTDIonTheFlyWrap +
//     SOBBHComputationGroupWrap) get their own aliasing in
//     `binding_bbhx.hpp` -- see the macro block there.
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define SOBBHTDIonTheFly      SOBBHTDIonTheFlyGPU
#define SOBBHComputationGroup SOBBHComputationGroupGPU
#else
#define SOBBHTDIonTheFly      SOBBHTDIonTheFlyCPU
#define SOBBHComputationGroup SOBBHComputationGroupCPU
#endif


// ============================================================================
// SOBBHTDIonTheFly
// ----------------------------------------------------------------------------
// LISATDIonTheFly subclass for stellar-origin black-hole binaries.
//
// Parameter layout (11 elements):
//   0: m1          (solar masses)
//   1: m2          (solar masses)
//   2: chi1z       (dimensionless aligned spin)
//   3: chi2z       (dimensionless aligned spin)
//   4: distance    (parsecs)
//   5: f_low       (Hz, "f0" alias for unified kernels)
//   6: phi_c       (rad)
//   7: iota        (rad)
//   8: psi         (rad)
//   9: lam         (rad, ecliptic longitude)
//  10: beta        (rad, ecliptic latitude)
// ============================================================================
class SOBBHTDIonTheFly : public LISATDIonTheFly{
    public:
        double T;
        double t_ref;
        int m1_index;
        int m2_index;
        int s1_index;
        int s2_index;
        int distance_index;
        int f_low_index;
        int phi_c_index;
        // f0_index is an ALIAS for f_low_index so source-class-agnostic
        // kernels (e.g. fast_wdm_inner_heterodyne) can read
        // ``src->f0_index`` uniformly across GB and SOBBH variants.
        int f0_index;

        CUDA_CALLABLE_MEMBER
        SOBBHTDIonTheFly(Orbits *orbits_, TDIConfig *tdi_config_, double T_, double t_ref_) : LISATDIonTheFly(orbits_, tdi_config_, 7, 8, 9, 10)
        {
            T = T_;
            t_ref = t_ref_;
            m1_index = 0;
            m2_index = 1;
            s1_index = 2;
            s2_index = 3;
            distance_index = 4;
            f_low_index = 5;
            phi_c_index = 6;
            f0_index = 5;          // alias of f_low_index for unified kernels
        };
        CUDA_CALLABLE_MEMBER
        ~SOBBHTDIonTheFly();
        // Intrinsic-quantity helpers (PN expansions ported from sobbh_intrinsic_Ladeeda.cpp).
        CUDA_DEVICE
        double sobbh_phase_fn(double x, double sigma, double delta, double eta, double s);
        CUDA_DEVICE
        double sobbh_time_to_merger_fn(double x, double sigma, double delta, double eta, double s);
        CUDA_DEVICE
        double sobbh_tau_to_x_fn(double tau, double sigma, double delta, double eta, double s);
        // Per-sample on-the-fly evaluators. amplitude/phase are GW-quadrupole
        // conventions: phase = 2 * (phi_c - phase_fn(x)), amp = 2 M eta x / D
        // (positive); get_hp_hc folds in -cos / -sin and the (1+cos^2 iota),
        // 2 cos(iota) factors.
        CUDA_DEVICE
        double sobbh_amplitude(double t, double *params);
        CUDA_DEVICE
        double sobbh_phase(double t, double *params);
        CUDA_DEVICE
        double sobbh_f(double t, double *params);
        CUDA_DEVICE
        double sobbh_fdot(double t, double *params);
        CUDA_CALLABLE_MEMBER
        int get_sobbh_buffer_size(int N);
        CUDA_DEVICE
        double get_amp(double t, double *params, int bin_i);
        CUDA_DEVICE
        double get_phase(double t, double *params, int bin_i);
        CUDA_DEVICE
        double get_f(double t, double *params, int bin_i);
        CUDA_DEVICE
        double get_fdot(double t, double *params, int bin_i);
};

void sobbh_run_wave_tdi_wrap(SOBBHTDIonTheFly *tdi_on_fly, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels);


// ============================================================================
// SOBBHComputationGroup
// ----------------------------------------------------------------------------
// Parallel to GBComputationGroup. Same chunked-heterodyne methods,
// sobbh_ prefix; routes to LAT's templated wdm_het_*_impl with
// SourceT = SOBBHTDIonTheFly.
//
// The pre-existing per-pixel-lookup path (gb_wdm_fill_global / get_ll /
// swap_ll) is GB-only and has no SOBBH equivalent; for SOBBH the
// chunked-heterodyne path IS the canonical entry point.
// ============================================================================
class SOBBHComputationGroup{
  public:
    void sobbh_wdm_het_fill_global_wrap(
        double *template_fill,
        Orbits *orbits, TDIConfig *tdi_config,
        WDMSettings *wdm_settings,
        double *params_all, double *factors_all,
        double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
        int *chunk_n_global_offset,
        double *wdm_window,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref,
        double tukey_alpha,
        int grid_dim, int N_cp_sig, int N_cp_orbit,
        int m_band_half_width);

    void sobbh_wdm_het_get_ll_wrap(
        double *d_h_out, double *h_h_out,
        Orbits *orbits, TDIConfig *tdi_config,
        WDMSettings *wdm_settings,
        double *params_all,
        int *data_index_all, int *noise_index_all,
        double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
        int *chunk_n_global_offset,
        double *wdm_window,
        double *data_d, double *invC,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha,
        int grid_dim, int N_cp_sig, int N_cp_orbit,
        int *binary_perm, int *group_starts, int *group_ends,
        int *group_m_lo, int *group_m_hi, int n_groups);

    void sobbh_wdm_het_swap_ll_wrap(
        double *d_h_add_out, double *d_h_remove_out,
        double *add_add_out, double *remove_remove_out, double *add_remove_out,
        Orbits *orbits, TDIConfig *tdi_config,
        WDMSettings *wdm_settings,
        double *params_add_all, double *params_remove_all,
        int *data_index_all, int *noise_index_all,
        double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
        int *chunk_n_global_offset,
        double *wdm_window,
        double *data_d, double *invC,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha,
        int grid_dim, int N_cp_sig, int N_cp_orbit,
        int *binary_perm, int *group_starts, int *group_ends,
        int *group_m_lo, int *group_m_hi, int n_groups,
        int *pair_m_lo_b, int *pair_m_hi_b);

    // F-stat (chunked-heterodyne); see GBComputationGroup::gb_wdm_het_get_fstat_ll_wrap.
    void sobbh_wdm_het_get_fstat_ll_wrap(
        double *N_arr_re_out, double *N_arr_im_out,
        double *M_mat_re_out, double *M_mat_im_out,
        Orbits *orbits, TDIConfig *tdi_config,
        WDMSettings *wdm_settings,
        double *params_all,
        int *data_index_all, int *noise_index_all,
        double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
        int *chunk_n_global_offset,
        double *wdm_window,
        double *data_d, double *invC,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha,
        int grid_dim, int m_band_half_width);
};

#endif // __SOBBH_TDI_ON_THE_FLY_HH__
