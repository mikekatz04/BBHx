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

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#include "pybind11_cuda_array_interface.hpp"
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
    OrbitsWrap_responselisa *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
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
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit)
{
    const int Nf = wdm_settings_wrap->wdm_settings->Nf;
    const int Nt = wdm_settings_wrap->wdm_settings->Nt;
    sobbh_wdm_het_fill_global_wrap(
        return_pointer_and_check_length(template_fill, "template_fill",
                                        (size_t) nchannels * Nf * Nt, 1),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config,
        wdm_settings_wrap->wdm_settings,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(factors_all, "factors_all", num_bin, 1),
        return_pointer_and_check_length(chunk_t_starts, "chunk_t_starts", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_lo, "chunk_keep_lo", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_hi, "chunk_keep_hi", n_chunks, 1),
        return_pointer_and_check_length(chunk_n_global_offset, "chunk_n_global_offset", n_chunks, 1),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt_sub, 1),
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tukey_alpha, grid_dim, N_cp_sig, N_cp_orbit);
}

void SOBBHComputationGroupWrap::sobbh_wdm_het_get_ll(
    array_type<double> d_h_out, array_type<double> h_h_out,
    OrbitsWrap_responselisa *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
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
    array_type<int> group_m_lo, array_type<int> group_m_hi, int n_groups)
{
    const int gn = (n_groups > 0) ? n_groups : 1;
    const int Nf_active = wdm_settings_wrap->wdm_settings->Nf_active;
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
        return_pointer_and_check_length(
            data_d, "data_d",
            (size_t) nchannels * Nf_active * Nt_active, 1),
        return_pointer_and_check_length(
            invC, "invC",
            ((tdi_type == TDI_XYZ)
                 ? (size_t) nchannels * nchannels * Nf_active * Nt_active
                 : (size_t) nchannels * Nf_active * Nt_active),
            1),
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
        n_groups);
}

void SOBBHComputationGroupWrap::sobbh_wdm_het_swap_ll(
    array_type<double> d_h_add_out, array_type<double> d_h_remove_out,
    array_type<double> add_add_out, array_type<double> remove_remove_out,
    array_type<double> add_remove_out,
    OrbitsWrap_responselisa *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
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
    array_type<int> pair_m_lo_b, array_type<int> pair_m_hi_b)
{
    const int gn = (n_groups > 0) ? n_groups : 1;
    const int Nf_active = wdm_settings_wrap->wdm_settings->Nf_active;
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
        return_pointer_and_check_length(
            data_d, "data_d",
            (size_t) nchannels * Nf_active * Nt_active, 1),
        return_pointer_and_check_length(
            invC, "invC",
            ((tdi_type == TDI_XYZ)
                 ? (size_t) nchannels * nchannels * Nf_active * Nt_active
                 : (size_t) nchannels * Nf_active * Nt_active),
            1),
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
        return_pointer_and_check_length(pair_m_hi_b, "pair_m_hi_b", num_bin, 1));
}

void SOBBHComputationGroupWrap::sobbh_wdm_het_get_fstat_ll(
    array_type<double> N_arr_re_out, array_type<double> N_arr_im_out,
    array_type<double> M_mat_re_out, array_type<double> M_mat_im_out,
    OrbitsWrap_responselisa *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
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
    double tukey_alpha, int grid_dim, int m_band_half_width)
{
    const int Nf_active = wdm_settings_wrap->wdm_settings->Nf_active;
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
        return_pointer_and_check_length(
            data_d, "data_d",
            (size_t) nchannels * Nf_active * Nt_active, 1),
        return_pointer_and_check_length(
            invC, "invC",
            ((tdi_type == TDI_XYZ)
                 ? (size_t) nchannels * nchannels * Nf_active * Nt_active
                 : (size_t) nchannels * Nf_active * Nt_active),
            1),
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type,
        tukey_alpha, grid_dim, m_band_half_width);
}


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

    // ========================================================================
    // Phase 3L.8 (2026-06-04): SOBBHTDIonTheFlyWrap + SOBBHComputationGroupWrap
    // pybind11 registrations. Carved out of lisa-on-gpu's binding_tof.cxx,
    // mirror of the GB versions GBGPU registers in cgbgpu (Phase 3L.7g).
    // ========================================================================

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<SOBBHTDIonTheFlyWrap>(m, "SOBBHTDIonTheFlyWrapGPU")
#else
    py::class_<SOBBHTDIonTheFlyWrap>(m, "SOBBHTDIonTheFlyWrapCPU")
#endif
    .def(py::init<OrbitsWrap_responselisa *, TDIConfigWrap *, double, double>(),
         py::arg("orbits"), py::arg("tdi_config"), py::arg("Tobs"), py::arg("t_ref"))
    .def("run_wave_tdi_wrap", &SOBBHTDIonTheFlyWrap::run_wave_tdi_wrap, "Run SOBBH TDI on the fly.")
    .def("get_buffer_size", &SOBBHTDIonTheFlyWrap::get_buffer_size, "Get needed buffer size.")
    .def_readwrite("orbits", &SOBBHTDIonTheFlyWrap::orbits)
    .def_readwrite("tdi_config", &SOBBHTDIonTheFlyWrap::tdi_config)
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<SOBBHTDIonTheFly>(m, "SOBBHTDIonTheFlyGPU")
#else
    py::class_<SOBBHTDIonTheFly>(m, "SOBBHTDIonTheFlyCPU")
#endif
    .def(py::init<Orbits *, TDIConfig*, double, double>(),
         py::arg("orbits"), py::arg("tdi_config"), py::arg("Tobs"), py::arg("t_ref"))
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<SOBBHComputationGroupWrap>(m, "SOBBHComputationGroupWrapGPU")
#else
    py::class_<SOBBHComputationGroupWrap>(m, "SOBBHComputationGroupWrapCPU")
#endif
    .def(py::init<>())
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
    ;
}


PYBIND11_MODULE(cbbhx, m) {
    m.doc() = "BBHx pybind11 backend. Hosts the BBH-specific waveform + "
              "response + likelihood + interp wrappers plus (Phase 3L.8, "
              "2026-06-04) the SOBBHTDIonTheFly + SOBBHComputationGroup "
              "machinery carved out of lisa-on-gpu.";
    m.attr("TDI_XYZ") = TDI_XYZ;
    m.attr("TDI_AET") = TDI_AET;
    m.attr("TDI_AE") = TDI_AE;
    bbhx_part(m);
}
