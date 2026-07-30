// BBHx pybind11 module entry.
//
// Adapted from origin/pybind branch (binding_bbhx.cxx) against the
// post-Phase-3L LAT setup. The originally-on-pybind-branch redeclaration
// of OrbitsWrap_bbhx / its pybind11 registration has been removed --
// LAT's pycppdetector is the sole registrant of OrbitsWrap
// (and the rest of the shared wrapper family). BBHx's cbbhx module is
// reserved for BBH-specific wrappers + the (future) SOBBHTDIonTheFly +
// SOBBHComputationGroup carve-out from Phase 3L.8.

#include "binding_bbhx.hpp"

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#endif

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

// ============================================================================
// Phase 3L.8 (2026-06-04): SOBBHTDIonTheFlyWrap + SOBBHComputationGroupWrap
// method bodies. Carved from
// lisa-on-gpu/src/fastlisaresponse/cutils/binding_tof.cxx (where they
// previously lived alongside the GB equivalents -- now in
// GBGPU/src/gbgpu/cutils/binding_gbgpu.cxx after Phase 3L.7g).
// ============================================================================

void SOBBHTDIonTheFlyWrap::run_wave_tdi_wrap(
    array_type<std::complex<double>>tdi_channels_arr,
    array_type<double>tdi_amp, array_type<double>tdi_phase, array_type<double>phi_ref,
    array_type<double>params, array_type<double>t_arr, int N, int num_bin, int n_params, int nchannels
)
{
    sobbh_run_wave_tdi_wrap(
        waveform,
        (cmplx*)return_pointer_and_check_length(tdi_channels_arr, "tdi_channels_arr", N, num_bin * nchannels),
        return_pointer_and_check_length(tdi_amp, "tdi_amp", N, num_bin * nchannels),
        return_pointer_and_check_length(tdi_phase, "tdi_phase", N, num_bin * nchannels),
        return_pointer_and_check_length(phi_ref, "phi_ref", N, num_bin),
        return_pointer_and_check_length(params, "params", n_params, num_bin),
        return_pointer_and_check_length(t_arr, "t_arr", N, num_bin),
        N, num_bin, n_params, nchannels
    );
}


// ---- SOBBH chunked-heterodyne pybind shims --------------------------------

void SOBBHComputationGroupWrap::sobbh_wdm_het_fill_global(
    array_type<double> template_fill,
    OrbitsWrap *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
    WDMSettingsWrap *wdm_settings_wrap,
    array_type<double> params_all, array_type<double> factors_all,
    array_type<int> data_index,
    array_type<double> chunk_t_starts,
    array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
    array_type<int> chunk_n_global_offset,
    array_type<double> wdm_window,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
    int m_band_half_width, bool active_band,
    int Nf_slab, array_type<int> slab_min_f)   // task-b per-band slab (0/empty = off)
{
    const int Nf = wdm_settings_wrap->wdm_settings->Nf;
    const int Nt = wdm_settings_wrap->wdm_settings->Nt;
    // Task-b: a narrow per-band slab covers Nf_slab layers instead of the full
    // Nf_active (only in active_band mode). Nf_slab<=0 keeps the full extent.
    const int slab_Nf = (Nf_slab > 0)
        ? Nf_slab : wdm_settings_wrap->wdm_settings->Nf_active;
    // per_template is one template slab; the buffer holds num_templates such
    // slabs and data_index[bin] routes each binary into its own slab (0 ->
    // offset 0, backward compatible). Mirrors GBGPU's gb_wdm_het_fill_global.
    const size_t per_template = active_band
        ? (size_t) nchannels * slab_Nf
                             * wdm_settings_wrap->wdm_settings->Nt_active
        : (size_t) nchannels * Nf * Nt;
    const size_t templ_total = template_fill.size();
    if (per_template == 0 || (templ_total % per_template) != 0) {
        throw std::invalid_argument(
            std::string("template_fill: length ") + std::to_string(templ_total)
            + " is not an integer multiple of one template slab ("
            + std::to_string(per_template) + ").");
    }
    const int num_templates = (int) (templ_total / per_template);
    sobbh_wdm_het_fill_global_wrap(
        return_pointer_and_check_length(template_fill, "template_fill",
                                        (int) per_template, num_templates),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config,
        wdm_settings_wrap->wdm_settings,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(factors_all, "factors_all", num_bin, 1),
        return_pointer_and_check_length(data_index, "data_index", num_bin, 1),
        return_pointer_and_check_length(chunk_t_starts, "chunk_t_starts", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_lo, "chunk_keep_lo", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_hi, "chunk_keep_hi", n_chunks, 1),
        return_pointer_and_check_length(chunk_n_global_offset, "chunk_n_global_offset", n_chunks, 1),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt_sub, 1),
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tukey_alpha, grid_dim, N_cp_sig, N_cp_orbit,
        m_band_half_width, active_band,
        Nf_slab,
        (slab_min_f.size() > 0
             ? return_pointer(slab_min_f, "slab_min_f") : nullptr));
}

void SOBBHComputationGroupWrap::sobbh_wdm_het_get_ll(
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
    array_type<int> group_m_lo, array_type<int> group_m_hi, int n_groups,
    int m_band_half_width,
    int Nf_slab, array_type<int> slab_min_f)   // task-b per-band slab (0/empty = off)
{
    const int gn = (n_groups > 0) ? n_groups : 1;
    // Task-b: per-band slab covers Nf_slab layers (full Nf_active when Nf_slab<=0).
    // The data_d/invC per-slab size checks below key off this extent.
    const int Nf_active = (Nf_slab > 0)
        ? Nf_slab : wdm_settings_wrap->wdm_settings->Nf_active;
    const int Nt_active = wdm_settings_wrap->wdm_settings->Nt_active;
    sobbh_wdm_het_get_ll_wrap(
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config,
        wdm_settings_wrap->wdm_settings,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        return_pointer_and_check_length(noise_index_all, "noise_index_all", num_bin, 1),
        return_pointer_and_check_length(chunk_t_starts, "chunk_t_starts", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_lo, "chunk_keep_lo", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_hi, "chunk_keep_hi", n_chunks, 1),
        return_pointer_and_check_length(chunk_n_global_offset, "chunk_n_global_offset", n_chunks, 1),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt_sub, 1),
        // data_d / invC may hold ANY whole number of (walker, band) slabs --
        // the kernels stride into them via data_index / noise_index. Require
        // whole slabs only (per-slab strides below).
        (data_d.size() % ((size_t) nchannels * Nf_active * Nt_active) == 0
             && data_d.size() > 0
             ? return_pointer(data_d, "data_d")
             : throw std::invalid_argument(
                   "data_d: length must be a positive multiple of "
                   "nchannels * Nf_active * Nt_active.")),
        (invC.size() % ((tdi_type == TDI_XYZ)
                            ? (size_t) nchannels * nchannels * Nf_active * Nt_active
                            : (size_t) nchannels * Nf_active * Nt_active) == 0
             && invC.size() > 0
             ? return_pointer(invC, "invC")
             : throw std::invalid_argument(
                   "invC: length must be a positive multiple of the "
                   "per-slab inverse-covariance size.")),
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, grid_dim, N_cp_sig, N_cp_orbit,
        return_pointer_and_check_length(binary_perm,  "binary_perm",  num_bin, 1),
        return_pointer_and_check_length(group_starts, "group_starts", gn, 1),
        return_pointer_and_check_length(group_ends,   "group_ends",   gn, 1),
        return_pointer_and_check_length(group_m_lo,   "group_m_lo",   gn, 1),
        return_pointer_and_check_length(group_m_hi,   "group_m_hi",   gn, 1),
        n_groups, m_band_half_width,
        Nf_slab,
        (slab_min_f.size() > 0
             ? return_pointer(slab_min_f, "slab_min_f") : nullptr));
}

void SOBBHComputationGroupWrap::sobbh_wdm_het_swap_ll(
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
    array_type<int> pair_m_lo_b, array_type<int> pair_m_hi_b,
    int m_band_half_width,
    int Nf_slab, array_type<int> slab_min_f)   // task-b per-band slab (0/empty = off)
{
    const int gn = (n_groups > 0) ? n_groups : 1;
    // Task-b: per-band slab covers Nf_slab layers (full Nf_active when Nf_slab<=0).
    const int Nf_active = (Nf_slab > 0)
        ? Nf_slab : wdm_settings_wrap->wdm_settings->Nf_active;
    const int Nt_active = wdm_settings_wrap->wdm_settings->Nt_active;
    sobbh_wdm_het_swap_ll_wrap(
        return_pointer_and_check_length(d_h_add_out,       "d_h_add_out",       num_bin, 1),
        return_pointer_and_check_length(d_h_remove_out,    "d_h_remove_out",    num_bin, 1),
        return_pointer_and_check_length(add_add_out,       "add_add_out",       num_bin, 1),
        return_pointer_and_check_length(remove_remove_out, "remove_remove_out", num_bin, 1),
        return_pointer_and_check_length(add_remove_out,    "add_remove_out",    num_bin, 1),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config,
        wdm_settings_wrap->wdm_settings,
        return_pointer_and_check_length(params_add_all,    "params_add_all",    nparams, num_bin),
        return_pointer_and_check_length(params_remove_all, "params_remove_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all,    "data_index_all",    num_bin, 1),
        return_pointer_and_check_length(noise_index_all,   "noise_index_all",   num_bin, 1),
        return_pointer_and_check_length(chunk_t_starts,    "chunk_t_starts",    n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_lo,     "chunk_keep_lo",     n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_hi,     "chunk_keep_hi",     n_chunks, 1),
        return_pointer_and_check_length(chunk_n_global_offset, "chunk_n_global_offset", n_chunks, 1),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt_sub, 1),
        // whole-multiple slab checks (see sobbh_wdm_het_get_ll)
        (data_d.size() % ((size_t) nchannels * Nf_active * Nt_active) == 0
             && data_d.size() > 0
             ? return_pointer(data_d, "data_d")
             : throw std::invalid_argument(
                   "data_d: length must be a positive multiple of "
                   "nchannels * Nf_active * Nt_active.")),
        (invC.size() % ((tdi_type == TDI_XYZ)
                            ? (size_t) nchannels * nchannels * Nf_active * Nt_active
                            : (size_t) nchannels * Nf_active * Nt_active) == 0
             && invC.size() > 0
             ? return_pointer(invC, "invC")
             : throw std::invalid_argument(
                   "invC: length must be a positive multiple of the "
                   "per-slab inverse-covariance size.")),
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, grid_dim, N_cp_sig, N_cp_orbit,
        return_pointer_and_check_length(binary_perm,  "binary_perm",  num_bin, 1),
        return_pointer_and_check_length(group_starts, "group_starts", gn, 1),
        return_pointer_and_check_length(group_ends,   "group_ends",   gn, 1),
        return_pointer_and_check_length(group_m_lo,   "group_m_lo",   gn, 1),
        return_pointer_and_check_length(group_m_hi,   "group_m_hi",   gn, 1),
        n_groups,
        return_pointer_and_check_length(pair_m_lo_b, "pair_m_lo_b", num_bin, 1),
        return_pointer_and_check_length(pair_m_hi_b, "pair_m_hi_b", num_bin, 1),
        m_band_half_width,
        Nf_slab,
        (slab_min_f.size() > 0
             ? return_pointer(slab_min_f, "slab_min_f") : nullptr));
}

void SOBBHComputationGroupWrap::sobbh_wdm_het_get_fstat_ll(
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
    double tukey_alpha, int grid_dim, int m_band_half_width,
    int Nf_slab, array_type<int> slab_min_f)   // task-b per-band slab (0/empty = off)
{
    // Task-b: per-band slab covers Nf_slab layers (full Nf_active when Nf_slab<=0).
    const int Nf_active = (Nf_slab > 0)
        ? Nf_slab : wdm_settings_wrap->wdm_settings->Nf_active;
    const int Nt_active = wdm_settings_wrap->wdm_settings->Nt_active;
    sobbh_wdm_het_get_fstat_ll_wrap(
        return_pointer_and_check_length(N_arr_re_out, "N_arr_re_out", num_bin, 4),
        return_pointer_and_check_length(N_arr_im_out, "N_arr_im_out", num_bin, 4),
        return_pointer_and_check_length(M_mat_re_out, "M_mat_re_out", num_bin, 10),
        return_pointer_and_check_length(M_mat_im_out, "M_mat_im_out", num_bin, 10),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config,
        wdm_settings_wrap->wdm_settings,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        return_pointer_and_check_length(noise_index_all, "noise_index_all", num_bin, 1),
        return_pointer_and_check_length(chunk_t_starts, "chunk_t_starts", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_lo, "chunk_keep_lo", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_hi, "chunk_keep_hi", n_chunks, 1),
        return_pointer_and_check_length(chunk_n_global_offset, "chunk_n_global_offset", n_chunks, 1),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt_sub, 1),
        // whole-multiple slab checks (see sobbh_wdm_het_get_ll)
        (data_d.size() % ((size_t) nchannels * Nf_active * Nt_active) == 0
             && data_d.size() > 0
             ? return_pointer(data_d, "data_d")
             : throw std::invalid_argument(
                   "data_d: length must be a positive multiple of "
                   "nchannels * Nf_active * Nt_active.")),
        (invC.size() % ((tdi_type == TDI_XYZ)
                            ? (size_t) nchannels * nchannels * Nf_active * Nt_active
                            : (size_t) nchannels * Nf_active * Nt_active) == 0
             && invC.size() > 0
             ? return_pointer(invC, "invC")
             : throw std::invalid_argument(
                   "invC: length must be a positive multiple of the "
                   "per-slab inverse-covariance size.")),
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type,
        tukey_alpha, grid_dim, m_band_half_width,
        Nf_slab,
        (slab_min_f.size() > 0
             ? return_pointer(slab_min_f, "slab_min_f") : nullptr));
}


// ---- SOBBH signal-heterodyne (v2 polyphase) pybind shims ------------------
// Mirror of GBComputationGroupWrap::gb_signal_het_* in binding_gbgpu.cxx.

void SOBBHComputationGroupWrap::sobbh_signal_het_get_ll(
    array_type<double> d_h_out, array_type<double> h_h_out,
    array_type<std::complex<double>> fd_rfft_all,
    array_type<std::complex<double>> c0_sparse_all,
    array_type<std::complex<double>> A0_all,
    array_type<std::complex<double>> A1_all,
    array_type<std::complex<double>> B0_all,
    array_type<std::complex<double>> B1_all,
    array_type<double> wdm_window,
    array_type<int> n_sparse_local_arr,
    array_type<double> params_cand_all,
    array_type<double> params_ref_all,
    array_type<int> data_index_all,
    int num_bin, int num_data,
    int nparams, int f0_idx, int fdot_idx,
    int Nf, int Nt, int Nf_active, int Nt_active,
    int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f,
    int m_active_half_width,
    double layer_df, double dt,
    int nchannels, int tdi_type,
    int n_rfft, double max_r)
{
    (void) Nt_layer;
    const size_t b_xyz  = (size_t) num_data * nchannels * nchannels * Nf_active * N_sparse_t;
    const size_t b_diag = (size_t) num_data * nchannels * Nf_active * N_sparse_t;
    sobbh_signal_het_get_ll_wrap(
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            fd_rfft_all, "fd_rfft_all", (size_t) num_bin * nchannels * n_rfft, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_sparse_all, "c0_sparse_all", (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A0_all, "A0_all", (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A1_all, "A1_all", (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0_all, "B0_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1_all, "B1_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt, 1),
        return_pointer_and_check_length(n_sparse_local_arr, "n_sparse_local", N_sparse_t, 1),
        return_pointer_and_check_length(params_cand_all, "params_cand_all", nparams, num_bin),
        return_pointer_and_check_length(params_ref_all, "params_ref_all", nparams, num_data),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        num_bin, num_data, nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f, m_active_half_width,
        layer_df, dt, nchannels, tdi_type, n_rfft, max_r);
}

void SOBBHComputationGroupWrap::sobbh_signal_het_get_ll_sparse(
    array_type<double> d_h_out, array_type<double> h_h_out,
    array_type<std::complex<double>> X_het_all,
    array_type<int> k_f0_all,
    array_type<std::complex<double>> c0_sparse_all,
    array_type<std::complex<double>> A0_all,
    array_type<std::complex<double>> A1_all,
    array_type<std::complex<double>> B0_all,
    array_type<std::complex<double>> B1_all,
    array_type<double> wdm_window,
    array_type<int> n_sparse_local_arr,
    array_type<double> params_cand_all,
    array_type<double> params_ref_all,
    array_type<int> data_index_all,
    int num_bin, int num_data,
    int nparams, int f0_idx, int fdot_idx,
    int Nf, int Nt, int Nf_active, int Nt_active,
    int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f,
    int m_active_half_width,
    double layer_df, double dt,
    int nchannels, int tdi_type,
    int N_sparse_fd, double max_r)
{
    (void) Nt_layer;
    const size_t b_xyz  = (size_t) num_data * nchannels * nchannels * Nf_active * N_sparse_t;
    const size_t b_diag = (size_t) num_data * nchannels * Nf_active * N_sparse_t;
    sobbh_signal_het_get_ll_sparse_wrap(
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            X_het_all, "X_het_all", (size_t) num_bin * nchannels * N_sparse_fd, 1)),
        return_pointer_and_check_length(k_f0_all, "k_f0_all", num_bin, 1),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_sparse_all, "c0_sparse_all", (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A0_all, "A0_all", (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A1_all, "A1_all", (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0_all, "B0_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1_all, "B1_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        nullptr, nullptr,   /* B0nc/B1nc: this validation path stays complex */
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt, 1),
        return_pointer_and_check_length(n_sparse_local_arr, "n_sparse_local", N_sparse_t, 1),
        return_pointer_and_check_length(params_cand_all, "params_cand_all", nparams, num_bin),
        return_pointer_and_check_length(params_ref_all, "params_ref_all", nparams, num_data),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        num_bin, num_data, nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f, m_active_half_width,
        layer_df, dt, nchannels, tdi_type, N_sparse_fd, max_r, /*project_real=*/0);
}

void SOBBHComputationGroupWrap::sobbh_signal_het_get_ll_in_kernel(
    SOBBHTDIonTheFlyWrap *tdi_wrap,
    array_type<double> d_h_out, array_type<double> h_h_out,
    array_type<std::complex<double>> c0_sparse_all,
    array_type<std::complex<double>> A0_all,
    array_type<std::complex<double>> A1_all,
    array_type<std::complex<double>> B0_all,
    array_type<std::complex<double>> B1_all,
    array_type<std::complex<double>> B0nc_all,
    array_type<std::complex<double>> B1nc_all,
    array_type<double> wdm_window,
    array_type<int> n_sparse_local_arr,
    array_type<double> params_cand_all,
    array_type<double> params_ref_all,
    array_type<int> data_index_all,
    int num_bin, int num_data,
    int nparams, int f0_idx, int fdot_idx,
    int Nf, int Nt, int Nf_active, int Nt_active,
    int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f,
    int m_active_half_width,
    double layer_df, double dt,
    double T_obs, double t_start,
    int nchannels, int tdi_type,
    int N_sparse_fd, double tukey_alpha, double max_r, int project_real)
{
    (void) Nt_layer;
    const size_t b_xyz  = (size_t) num_data * nchannels * nchannels * Nf_active * N_sparse_t;
    const size_t b_diag = (size_t) num_data * nchannels * Nf_active * N_sparse_t;
    sobbh_signal_het_get_ll_in_kernel_wrap(
        tdi_wrap->waveform,
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_sparse_all, "c0_sparse_all", (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A0_all, "A0_all", (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A1_all, "A1_all", (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0_all, "B0_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1_all, "B1_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0nc_all, "B0nc_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1nc_all, "B1nc_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt, 1),
        return_pointer_and_check_length(n_sparse_local_arr, "n_sparse_local", N_sparse_t, 1),
        return_pointer_and_check_length(params_cand_all, "params_cand_all", nparams, num_bin),
        return_pointer_and_check_length(params_ref_all, "params_ref_all", nparams, num_data),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        num_bin, num_data, nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f, m_active_half_width,
        layer_df, dt, T_obs, t_start,
        nchannels, tdi_type, N_sparse_fd, tukey_alpha, max_r, project_real);
}

void SOBBHComputationGroupWrap::sobbh_signal_het_fill_global_in_kernel(
    SOBBHTDIonTheFlyWrap *tdi_wrap,
    array_type<double> template_fill,
    array_type<std::complex<double>> c0_sparse_all,
    array_type<std::complex<double>> c0_dense_complex_all,
    array_type<double> wdm_window,
    array_type<int> n_sparse_local_arr,
    array_type<double> params_cand_all,
    array_type<double> params_ref_all,
    array_type<double> factors_all,
    array_type<int> data_index_all,
    int num_bin, int num_data,
    int nparams, int f0_idx, int fdot_idx,
    int Nf, int Nt, int Nf_active, int Nt_active,
    int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f,
    int m_active_half_width,
    double layer_df, double dt,
    double T_obs, double t_start,
    int nchannels,
    int N_sparse_fd, double tukey_alpha, double max_r)
{
    (void) Nt_layer;
    sobbh_signal_het_fill_global_in_kernel_wrap(
        tdi_wrap->waveform,
        return_pointer_and_check_length(template_fill, "template_fill",
            (size_t) num_data * nchannels * Nf * Nt, 1),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_sparse_all, "c0_sparse_all", (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_dense_complex_all, "c0_dense_complex_all", (size_t) num_data * nchannels * Nf_active * Nt_active, 1)),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt, 1),
        return_pointer_and_check_length(n_sparse_local_arr, "n_sparse_local", N_sparse_t, 1),
        return_pointer_and_check_length(params_cand_all, "params_cand_all", nparams, num_bin),
        return_pointer_and_check_length(params_ref_all, "params_ref_all", nparams, num_data),
        return_pointer_and_check_length(factors_all, "factors_all", num_bin, 1),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        num_bin, num_data, nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f, m_active_half_width,
        layer_df, dt, T_obs, t_start,
        nchannels, N_sparse_fd, tukey_alpha, max_r);
}

void SOBBHComputationGroupWrap::sobbh_signal_het_get_ll_grad_in_kernel(
    SOBBHTDIonTheFlyWrap *tdi_wrap,
    array_type<double> grad_out,
    array_type<double> d_h_central, array_type<double> h_h_central,
    array_type<std::complex<double>> c0_sparse_all,
    array_type<std::complex<double>> A0_all,
    array_type<std::complex<double>> A1_all,
    array_type<std::complex<double>> B0_all,
    array_type<std::complex<double>> B1_all,
    array_type<double> wdm_window,
    array_type<int> n_sparse_local_arr,
    array_type<double> params_cand_all,
    array_type<double> params_ref_all,
    array_type<int> data_index_all,
    array_type<double> param_eps,
    int num_bin, int num_data,
    int nparams, int f0_idx, int fdot_idx,
    int Nf, int Nt, int Nf_active, int Nt_active,
    int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f,
    int m_active_half_width,
    double layer_df, double dt,
    double T_obs, double t_start,
    int nchannels, int tdi_type,
    int N_sparse_fd, double tukey_alpha, double max_r)
{
    (void) Nt_layer;
    const size_t b_xyz  = (size_t) num_data * nchannels * nchannels * Nf_active * N_sparse_t;
    const size_t b_diag = (size_t) num_data * nchannels * Nf_active * N_sparse_t;
    sobbh_signal_het_get_ll_grad_in_kernel_wrap(
        tdi_wrap->waveform,
        return_pointer_and_check_length(grad_out, "grad_out", nparams, num_bin),
        return_pointer_and_check_length(d_h_central, "d_h_central", num_bin, 1),
        return_pointer_and_check_length(h_h_central, "h_h_central", num_bin, 1),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_sparse_all, "c0_sparse_all", (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A0_all, "A0_all", (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A1_all, "A1_all", (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0_all, "B0_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1_all, "B1_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt, 1),
        return_pointer_and_check_length(n_sparse_local_arr, "n_sparse_local", N_sparse_t, 1),
        return_pointer_and_check_length(params_cand_all, "params_cand_all", nparams, num_bin),
        return_pointer_and_check_length(params_ref_all, "params_ref_all", nparams, num_data),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        return_pointer_and_check_length(param_eps, "param_eps", nparams, 1),
        num_bin, num_data, nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f, m_active_half_width,
        layer_df, dt, T_obs, t_start,
        nchannels, tdi_type, N_sparse_fd, tukey_alpha, max_r);
}


void bbhx_part(nb::module_ &m) {
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<BBHxComputationWrap>(m, "BBHxComputationWrapGPU")
#else
    nb::class_<BBHxComputationWrap>(m, "BBHxComputationWrapCPU")
#endif
        .def(nb::init<>())
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
        // Interpolate.hh: no interpolate_wrap method here. BBHx's local
        // 3D-grid solver was deleted at the 2026-06-05 GBT-dedup pass;
        // the Python frontend uses gpubackendtools.interpolate
        // .CubicSplineInterpolant, which calls gbt_backend_*.interp's
        // interpolate_wrap directly.
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

    // ========================================================================
    // Phase 3L.8 (2026-06-04): SOBBHTDIonTheFlyWrap + SOBBHComputationGroupWrap
    // pybind11 registrations. Carved out of lisa-on-gpu's binding_tof.cxx,
    // mirror of the GB versions GBGPU registers in cgbgpu (Phase 3L.7g).
    // ========================================================================

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<SOBBHTDIonTheFlyWrap>(m, "SOBBHTDIonTheFlyWrapGPU")
#else
    nb::class_<SOBBHTDIonTheFlyWrap>(m, "SOBBHTDIonTheFlyWrapCPU")
#endif
    .def(nb::init<OrbitsWrap *, TDIConfigWrap *, double, double>(),
         nb::arg("orbits"), nb::arg("tdi_config"), nb::arg("Tobs"), nb::arg("t_ref"))
    .def("run_wave_tdi_wrap", &SOBBHTDIonTheFlyWrap::run_wave_tdi_wrap, "Run SOBBH TDI on the fly.")
    .def("get_buffer_size", &SOBBHTDIonTheFlyWrap::get_buffer_size, "Get needed buffer size.")
    .def_rw("orbits", &SOBBHTDIonTheFlyWrap::orbits)
    .def_rw("tdi_config", &SOBBHTDIonTheFlyWrap::tdi_config)
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<SOBBHTDIonTheFly>(m, "SOBBHTDIonTheFlyGPU")
#else
    nb::class_<SOBBHTDIonTheFly>(m, "SOBBHTDIonTheFlyCPU")
#endif
    .def(nb::init<Orbits *, TDIConfig*, double, double>(),
         nb::arg("orbits"), nb::arg("tdi_config"), nb::arg("Tobs"), nb::arg("t_ref"))
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<SOBBHComputationGroupWrap>(m, "SOBBHComputationGroupWrapGPU")
#else
    nb::class_<SOBBHComputationGroupWrap>(m, "SOBBHComputationGroupWrapCPU")
#endif
    .def(nb::init<>())
    .def("sobbh_wdm_het_fill_global", &SOBBHComputationGroupWrap::sobbh_wdm_het_fill_global,
         "SOBBH chunked-heterodyne fill_global. Same as gb_wdm_het_fill_global "
         "but with SOBBHTDIonTheFly as the source class. 11-parameter source "
         "vector (see SOBBHTDIonTheFly).")
    .def("sobbh_wdm_het_get_ll", &SOBBHComputationGroupWrap::sobbh_wdm_het_get_ll,
         "SOBBH chunked-heterodyne get_ll.")
    .def("sobbh_wdm_het_swap_ll", &SOBBHComputationGroupWrap::sobbh_wdm_het_swap_ll,
         "SOBBH chunked-heterodyne swap_ll.")
    .def("sobbh_wdm_het_get_fstat_ll", &SOBBHComputationGroupWrap::sobbh_wdm_het_get_fstat_ll,
         "SOBBH chunked-heterodyne F-stat (same N+M outputs as the GB variant).")
    // Signal-heterodyne (v2 polyphase) -- SOBBH duplicate of the GB family
    // (gb_signal_het_*). CPU validated; GPU authored via the dual-path macros
    // (block-per-binary, global scratch arena) but untested on this host.
    .def("sobbh_signal_het_get_ll", &SOBBHComputationGroupWrap::sobbh_signal_het_get_ll,
         "Signal-heterodyne get_ll from a precomputed rfft(Tukey*td) per "
         "binary + reference c0_sparse/A0/A1/B0/B1. SOBBH mirror of "
         "gb_signal_het_get_ll.")
    .def("sobbh_signal_het_get_ll_sparse", &SOBBHComputationGroupWrap::sobbh_signal_het_get_ll_sparse,
         "Sparse-FD signal-het get_ll. Consumes X_het (N_sparse_fd per binary "
         "per channel) + per-binary k_f0. SOBBH mirror of gb_signal_het_get_ll_sparse.")
    .def("sobbh_signal_het_get_ll_in_kernel", &SOBBHComputationGroupWrap::sobbh_signal_het_get_ll_in_kernel,
         "In-kernel sparse-FD signal-het get_ll. Fuses sobbh_run_fd_wave_tdi "
         "(from SOBBHTDIonTheFly) with the polyphase + bin-fold pipeline. Takes "
         "a SOBBHTDIonTheFlyWrap; tukey_alpha must match the dense-rfft window "
         "alpha; max_r caps |r| per channel-cell; project_real selects the "
         "real-WDM projection. SOBBH mirror of gb_signal_het_get_ll_in_kernel.")
    .def("sobbh_signal_het_fill_global_in_kernel",
         &SOBBHComputationGroupWrap::sobbh_signal_het_fill_global_in_kernel,
         "Signal-het fill_global. Reconstructs the dense candidate template via "
         "interp(r_sparse) re-rotation * c0_dense_complex, takes Re, and "
         "scatters factor * Re(c1_dense) into template_fill. SOBBH mirror of "
         "gb_signal_het_fill_global_in_kernel.")
    .def("sobbh_signal_het_get_ll_grad_in_kernel",
         &SOBBHComputationGroupWrap::sobbh_signal_het_get_ll_grad_in_kernel,
         "Signal-het central-difference gradient of logL = d_h - 0.5*h_h. "
         "param_eps[k] <= 0 freezes dimension k. SOBBH mirror of "
         "gb_signal_het_get_ll_grad_in_kernel.")
    ;
}


NB_MODULE(cbbhx, m) {
    m.doc() = "BBHx pybind11 backend. Hosts the BBH-specific waveform + "
              "response + likelihood + interp wrappers plus (Phase 3L.8, "
              "2026-06-04) the SOBBHTDIonTheFly + SOBBHComputationGroup "
              "machinery carved out of lisa-on-gpu.";
    m.attr("TDI_XYZ") = TDI_XYZ;
    m.attr("TDI_AET") = TDI_AET;
    m.attr("TDI_AE") = TDI_AE;
    bbhx_part(m);
}
