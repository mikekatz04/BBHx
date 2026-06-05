// Phase 3L.8 (2026-06-04): SOBBH-specific TDI-on-the-fly method bodies +
// kernels carved out of lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu.
// Mirror of GBGPU/src/gbgpu/cutils/gb_tdi_on_the_fly.cu.
//
// SOBBHTDIonTheFly PN intrinsic-quantity expressions (sobbh_phase_fn,
// sobbh_time_to_merger_fn, sobbh_tau_to_x_fn, sobbh_amplitude / phase /
// f / fdot, get_amp / get_phase / get_f / get_fdot, dtor,
// get_sobbh_buffer_size, sobbh_run_wave_tdi_kernel + sobbh_run_wave_tdi_wrap).
// SOBBHComputationGroup::sobbh_wdm_het_*_wrap methods that instantiate
// LAT's templated wdm_het_*_impl<SOBBHTDIonTheFly>.

#include "sobbh_tdi_on_the_fly.hh"
// constants.h provides C_SI (speed of light) used by the SOBBH PN
// expressions below. Lives in BBHx's local cutils alongside this file.
#include "constants.h"
// NUM_THREADS_HERE + CUDA_KERNEL / CUDA_SHARED / gpuErrchk all come from
// the LAT/GBT include chain pulled in via sobbh_tdi_on_the_fly.hh:
//   - NUM_THREADS_HERE: lat_chunked_het_kernels.hh
//   - CUDA_KERNEL / CUDA_SHARED / CUDA_DEVICE / gpuErrchk: gbt_global.h
// No additional preamble required.


// ---- carved from lisa-on-gpu TDIonTheFly.cu lines 5301-5786 ----
// ---------------------------------------------------------------------------
// Stellar-origin black-hole binary (SOBBH) TDI on the fly
// ---------------------------------------------------------------------------
//
// Ports the post-Newtonian intrinsic-quantity expressions from
// sobbh_intrinsic_Ladeeda.cpp into the LISATDIonTheFly framework. Only style
// has been adapted (CUDA_DEVICE decorators, no std:: qualifiers, no
// std::vector buffers); the PN coefficients themselves are bit-identical to
// the prototype.
//
// Conversions: m1, m2 in solar masses -> seconds via MTSUN_SOBBH;
// distance in parsecs -> seconds via PARSEC_SOBBH / C_SI. The amplitude
// returned by get_amp is the GW-quadrupole "scalar" piece a = 2 M eta x / D
// (positive); the inclination factor (1+cos^2 iota) for plus, 2 cos iota for
// cross, plus an overall minus sign and the cos/sin split, all live in
// LISATDIonTheFly::get_hp_hc and match the existing GB convention. The phase
// returned is the GW phase 2 * (phi_c - phase_fn(x)), with the factor of 2
// already folded in to match get_hp_hc's cos(phase) / sin(phase) pattern.

#define EULER_GAMMA_SOBBH 0.57721566490153286060
#define MTSUN_SOBBH      4.9254909476412675e-06
#define PARSEC_SOBBH     3.085677581491367e16

CUDA_DEVICE
double SOBBHTDIonTheFly::sobbh_phase_fn(double x, double sigma, double delta, double eta, double s)
{
    double x15 = pow(x, 1.5);
    double x20 = x * x;
    double x25 = pow(x, 2.5);
    double x30 = x * x * x;
    double x35 = pow(x, 3.5);
    double logx = log(x);

    double Phi_0_minus_phi =
        (
            1.0
            + x * (3.685515873015873 + (55.0 * eta) / 12.0)
            + x15 * (-10.0 * M_PI + (235.0 * s) / 6.0 + (125.0 * delta * sigma) / 8.0)
            + x20 * (
                15.051576475497606
                - 100.0 * s * s
                + (3085.0 * eta * eta) / 144.0
                - 100.0 * s * delta * sigma
                - (405.0 * sigma * sigma) / 16.0
                + eta * (26.92956349206349 + 100.0 * sigma * sigma)
            )
            + x35 * (
                (-9018232555.0 * s) / 6.096384e6
                + (125925.0 * s * s * s) / 224.0
                - (170978035.0 * delta * sigma) / 387072.0
                + (379805.0 * s * s * delta * sigma) / 448.0
                + (182755.0 * s * sigma * sigma) / 448.0
                + (1315.0 * delta * sigma * sigma * sigma) / 21.0
                + M_PI * (
                    37.93888721576594
                    - 200.0 * s * s
                    - 200.0 * s * delta * sigma
                    - (815.0 * sigma * sigma) / 16.0
                )
                + eta * eta * (
                    (-74045.0 * M_PI) / 6048.0
                    + (835.0 * s) / 288.0
                    + (7015.0 * delta * sigma) / 1152.0
                    + (285.0 * s * sigma * sigma) / 8.0
                    + (95.0 * delta * sigma * sigma * sigma) / 16.0
                )
                + eta * (
                    (3329545.0 * s) / 3024.0
                    - (95.0 * s * s * s) / 8.0
                    + (2909765.0 * delta * sigma) / 5376.0
                    - (285.0 * s * s * delta * sigma) / 16.0
                    - (385825.0 * s * sigma * sigma) / 224.0
                    - (130615.0 * delta * sigma * sigma * sigma) / 448.0
                    + M_PI * (31.292576058201057 + 200.0 * sigma * sigma)
                )
            )
            + x30 * (
                657.6504345051205
                - (1712.0 * EULER_GAMMA_SOBBH) / 21.0
                - (160.0 * M_PI * M_PI) / 3.0
                + (7915.0 * s * s) / 63.0
                - (127825.0 * eta * eta * eta) / 5184.0
                + (2645.0 * s * delta * sigma) / 56.0
                - (1645.0 * sigma * sigma) / 128.0
                + M_PI * ((940.0 * s) / 3.0 + (745.0 * delta * sigma) / 6.0)
                + eta * eta * (11.003327546296296 - 120.0 * sigma * sigma)
                + eta * (
                    -1290.7459270118156
                    + (2255.0 * M_PI * M_PI) / 48.0
                    + 120.0 * s * s
                    + 120.0 * s * delta * sigma
                    + (5875.0 * sigma * sigma) / 112.0
                )
                - (3424.0 * log(2.0)) / 21.0
                - (856.0 * logx) / 21.0
            )
            + x25 * (
                (38645.0 * M_PI) / 1344.0
                - (555605.0 * s) / 2016.0
                - (15.0 * s * s * s) / 4.0
                - (41745.0 * delta * sigma) / 448.0
                - (45.0 * s * s * delta * sigma) / 8.0
                - (45.0 * s * sigma * sigma) / 8.0
                - (15.0 * delta * sigma * sigma * sigma) / 8.0
                + eta * (
                    (-65.0 * M_PI) / 16.0
                    - (45.0 * s) / 8.0
                    + (5.0 * delta * sigma) / 2.0
                    + (45.0 * s * sigma * sigma) / 4.0
                    + (15.0 * delta * sigma * sigma * sigma) / 8.0
                )
            ) * logx
        ) / (32.0 * x25 * eta);

    return Phi_0_minus_phi;
}


CUDA_DEVICE
double SOBBHTDIonTheFly::sobbh_time_to_merger_fn(double x, double sigma, double delta, double eta, double s)
{
    double x15 = pow(x, 1.5);
    double x20 = x * x;
    double x25 = pow(x, 2.5);
    double x30 = x * x * x;
    double x40 = x20 * x20;
    double logx = log(x);

    double tc =
        (1.0 / eta) *
        (
            5.0 / (256.0 * x40)
            + (5.0 * (743.0 + 924.0 * eta)) / (64512.0 * x30)
            + (-48.0 * M_PI + 188.0 * s + 75.0 * delta * sigma) / (384.0 * x25)
            + (
                5.0 * (
                    -23187.0 * M_PI
                    + 221738.0 * s
                    + 3276.0 * M_PI * eta
                    + 5544.0 * s * eta
                    + 75141.0 * delta * sigma
                    - 1512.0 * delta * eta * sigma
                )
            ) / (193536.0 * x15)
            - (
                5.0 * (
                    -3058673.0
                    + 20321280.0 * s * s
                    - 5472432.0 * eta
                    - 4353552.0 * eta * eta
                    + 20321280.0 * s * delta * sigma
                    + 5143824.0 * sigma * sigma
                    - 20321280.0 * eta * sigma * sigma
                )
            ) / (1.30056192e8 * x20)
            + (
                -10052469856691.0
                + 1530761379840.0 * EULER_GAMMA_SOBBH
                + 1001432678400.0 * M_PI * M_PI
                - 5883416985600.0 * M_PI * s
                - 2359029657600.0 * s * s
                + 24236159077900.0 * eta
                - 882121363200.0 * M_PI * M_PI * eta
                - 2253223526400.0 * s * s * eta
                - 206607970800.0 * eta * eta
                + 462992376000.0 * eta * eta * eta
                - 2331460454400.0 * M_PI * delta * sigma
                - 886871462400.0 * s * delta * sigma
                - 2253223526400.0 * s * delta * eta * sigma
                - 492159175200.0 * sigma * sigma
                + 733471200000.0 * delta * delta * sigma * sigma
                + 1948937760000.0 * eta * sigma * sigma
                + 2253223526400.0 * eta * eta * sigma * sigma
                + 658084331520.0 * log(2.0)
                + 1201719214080.0 * log(4.0)
                + 765380689920.0 * logx
            ) / (1.20171921408e12 * x)
        );

    return tc;
}


CUDA_DEVICE
double SOBBHTDIonTheFly::sobbh_tau_to_x_fn(double tau, double sigma, double delta, double eta, double s)
{
    double tau_inv = 1.0 / tau;
    double tau_qrt = pow(tau, 0.25);
    double tau_inv_qrt = 1.0 / tau_qrt;
    double tau_inv_3_8 = pow(tau, -0.375);
    double tau_inv_5_8 = pow(tau, -0.625);
    double tau_inv_7_8 = pow(tau, -0.875);
    double tau_inv_3_4 = pow(tau, -0.75);
    double tau_inv_half = 1.0 / sqrt(tau);
    double logtau = log(tau);

    double x =
        (
            1.0
            + (
                (-113868647.0 * M_PI) / 4.3352064e8
                + (24532268147.0 * s) / 2.60112384e9
                + (21.0 * M_PI * s * s) / 16.0
                - (755.0 * s * s * s) / 192.0
                + (281190779.0 * delta * sigma) / 9.9090432e7
                + (21.0 * M_PI * s * delta * sigma) / 16.0
                - (4499.0 * s * s * delta * sigma) / 768.0
                + (1711.0 * M_PI * sigma * sigma) / 5120.0
                - (33929.0 * s * sigma * sigma) / 20480.0
                - (325.0 * s * delta * delta * sigma * sigma) / 256.0
                - (24007.0 * delta * sigma * sigma * sigma) / 49152.0
                + eta * eta * (
                    (294941.0 * M_PI) / 3.87072e6
                    + (3641.0 * s) / 122880.0
                    - (6169.0 * delta * sigma) / 294912.0
                )
                + eta * (
                    (-31821.0 * M_PI) / 143360.0
                    - (33704749.0 * s) / 5.16096e6
                    - (5756657.0 * delta * sigma) / 1.769472e6
                    - (21.0 * M_PI * sigma * sigma) / 16.0
                    + (1259.0 * s * sigma * sigma) / 192.0
                    + (493.0 * delta * sigma * sigma * sigma) / 256.0
                )
            ) * tau_inv_7_8
            + (
                (-11891.0 * M_PI) / 53760.0
                + (357923.0 * s) / 161280.0
                + (96473.0 * delta * sigma) / 129024.0
                + eta * (
                    (109.0 * M_PI) / 1920.0
                    - (187.0 * s) / 5760.0
                    - (79.0 * delta * sigma) / 1536.0
                )
            ) * tau_inv_5_8
            + (
                0.0770935689090451
                - (5.0 * s * s) / 8.0
                + (31.0 * eta * eta) / 288.0
                - (5.0 * s * delta * sigma) / 8.0
                - (81.0 * sigma * sigma) / 512.0
                + eta * (0.12607990244708994 + (5.0 * sigma * sigma) / 8.0)
            ) * tau_inv_half
            + (-0.2 * M_PI + (47.0 * s) / 60.0 + (5.0 * delta * sigma) / 16.0) * tau_inv_3_8
            + (0.18427579365079366 + (11.0 * eta) / 48.0) * tau_inv_qrt
            + (
                -1.6730147506856445
                + (107.0 * EULER_GAMMA_SOBBH) / 420.0
                + M_PI * M_PI / 6.0
                - (47.0 * M_PI * s) / 48.0
                - (1583.0 * s * s) / 4032.0
                + (25565.0 * eta * eta * eta) / 331776.0
                - (149.0 * M_PI * delta * sigma) / 384.0
                - (529.0 * s * delta * sigma) / 3584.0
                - (671.0 * sigma * sigma) / 8192.0
                + (125.0 * delta * delta * sigma * sigma) / 1024.0
                + eta * (
                    4.033581021911924
                    - (451.0 * M_PI * M_PI) / 3072.0
                    - (3.0 * s * s) / 8.0
                    - (3.0 * s * delta * sigma) / 8.0
                    + (2325.0 * sigma * sigma) / 7168.0
                )
                + eta * eta * (-0.03438539858217592 + (3.0 * sigma * sigma) / 8.0)
                + (107.0 * log(2.0)) / 420.0
                - (107.0 * logtau) / 3360.0
            ) * tau_inv_3_4
        ) / (4.0 * tau_qrt);

    return x;
}


// Computes (M, eta, sigma, delta, s, tc, tau, x) on the fly from the
// per-source parameter vector. Pre-merger only; post-merger callers must
// short-circuit via the caller's t < tc check (the per-sample wrappers below
// return amp == 0 and phase == 0 for t >= tc so the projection still
// produces a finite zero-amplitude template).
CUDA_DEVICE
double SOBBHTDIonTheFly::sobbh_amplitude(double t, double *params)
{
    double m1_sec = params[m1_index] * MTSUN_SOBBH;
    double m2_sec = params[m2_index] * MTSUN_SOBBH;
    double s1     = params[s1_index];
    double s2     = params[s2_index];
    double D_pc   = params[distance_index];
    double f_low  = params[f_low_index];

    double M = m1_sec + m2_sec;
    double eta   = (m1_sec * m2_sec) / (M * M);
    double sigma = (m2_sec * s2 - m1_sec * s1) / M;
    double s_pn  = (m1_sec * m1_sec * s1 + m2_sec * m2_sec * s2) / (M * M);
    double delta = (m1_sec - m2_sec) / M;

    double v0 = pow(M_PI * M * f_low, 1.0 / 3.0);
    double x0 = v0 * v0;
    double tc = sobbh_time_to_merger_fn(x0, sigma, delta, eta, s_pn) * M;

    if (t >= tc)
    {
        return 0.0;
    }

    double tau = eta * (tc - t) / (5.0 * M);
    double x   = sobbh_tau_to_x_fn(tau, sigma, delta, eta, s_pn);

    double D_sec = D_pc * PARSEC_SOBBH / C_SI;

    // Positive scalar amplitude: get_hp_hc folds in the overall minus sign
    // and the cos(2 phi) / sin(2 phi) split, matching the GB convention.
    return 2.0 * M * eta * x / D_sec;
}

CUDA_DEVICE
double SOBBHTDIonTheFly::sobbh_phase(double t, double *params)
{
    double m1_sec = params[m1_index] * MTSUN_SOBBH;
    double m2_sec = params[m2_index] * MTSUN_SOBBH;
    double s1     = params[s1_index];
    double s2     = params[s2_index];
    double f_low  = params[f_low_index];
    double phi_c  = params[phi_c_index];

    double M = m1_sec + m2_sec;
    double eta   = (m1_sec * m2_sec) / (M * M);
    double sigma = (m2_sec * s2 - m1_sec * s1) / M;
    double s_pn  = (m1_sec * m1_sec * s1 + m2_sec * m2_sec * s2) / (M * M);
    double delta = (m1_sec - m2_sec) / M;

    double v0 = pow(M_PI * M * f_low, 1.0 / 3.0);
    double x0 = v0 * v0;
    double tc = sobbh_time_to_merger_fn(x0, sigma, delta, eta, s_pn) * M;

    if (t >= tc)
    {
        return 0.0;
    }

    double tau = eta * (tc - t) / (5.0 * M);
    double x   = sobbh_tau_to_x_fn(tau, sigma, delta, eta, s_pn);

    double phi_orbital = phi_c - sobbh_phase_fn(x, sigma, delta, eta, s_pn);
    return 2.0 * phi_orbital;
}

CUDA_DEVICE
double SOBBHTDIonTheFly::sobbh_f(double t, double *params)
{
    // GW (quadrupolar) frequency: f_GW = 2 * f_orbital = 2 * x^(3/2) / (pi M).
    double m1_sec = params[m1_index] * MTSUN_SOBBH;
    double m2_sec = params[m2_index] * MTSUN_SOBBH;
    double s1     = params[s1_index];
    double s2     = params[s2_index];
    double f_low  = params[f_low_index];

    double M = m1_sec + m2_sec;
    double eta   = (m1_sec * m2_sec) / (M * M);
    double sigma = (m2_sec * s2 - m1_sec * s1) / M;
    double s_pn  = (m1_sec * m1_sec * s1 + m2_sec * m2_sec * s2) / (M * M);
    double delta = (m1_sec - m2_sec) / M;

    double v0 = pow(M_PI * M * f_low, 1.0 / 3.0);
    double x0 = v0 * v0;
    double tc = sobbh_time_to_merger_fn(x0, sigma, delta, eta, s_pn) * M;

    if (t >= tc)
    {
        return 0.0;
    }

    double tau = eta * (tc - t) / (5.0 * M);
    double x   = sobbh_tau_to_x_fn(tau, sigma, delta, eta, s_pn);
    return 2.0 * pow(x, 1.5) / (M_PI * M);
}

CUDA_DEVICE
double SOBBHTDIonTheFly::sobbh_fdot(double t, double *params)
{
    // Centered finite difference around t (small window scaled by chirp time).
    // The PN inspiral is smooth enough that the leading O(h^2) truncation is
    // well below the rest of the SOBBH pipeline error; tighter analytic
    // expressions can replace this if a downstream consumer ever needs them.
    double m1_sec = params[m1_index] * MTSUN_SOBBH;
    double m2_sec = params[m2_index] * MTSUN_SOBBH;
    double M = m1_sec + m2_sec;
    double dt_fd = 10.0 * M;
    if (dt_fd <= 0.0) dt_fd = 1.0;

    double f_plus  = sobbh_f(t + dt_fd, params);
    double f_minus = sobbh_f(t - dt_fd, params);
    return (f_plus - f_minus) / (2.0 * dt_fd);
}

CUDA_DEVICE
double SOBBHTDIonTheFly::get_amp(double t, double *params, int bin_i)
{
    return sobbh_amplitude(t, params);
}

CUDA_DEVICE
double SOBBHTDIonTheFly::get_phase(double t, double *params, int bin_i)
{
    return sobbh_phase(t, params);
}

CUDA_DEVICE
double SOBBHTDIonTheFly::get_f(double t, double *params, int bin_i)
{
    return sobbh_f(t, params);
}

CUDA_DEVICE
double SOBBHTDIonTheFly::get_fdot(double t, double *params, int bin_i)
{
    return sobbh_fdot(t, params);
}

CUDA_DEVICE
SOBBHTDIonTheFly::~SOBBHTDIonTheFly()
{
    return;
}

int SOBBHTDIonTheFly::get_sobbh_buffer_size(int N)
{
    return N * sizeof(double) + get_tdi_buffer_size(N);
}


#ifdef __CUDACC__
CUDA_KERNEL
void sobbh_run_wave_tdi_kernel(SOBBHTDIonTheFly *tdi_on_fly, int buffer_length, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
    extern CUDA_SHARED char shared_mem[];
    void *buffer = (void*)shared_mem;

    SOBBHTDIonTheFly tdi_on_fly_here(tdi_on_fly->orbits, tdi_on_fly->tdi_config, tdi_on_fly->T, tdi_on_fly->t_ref);
    tdi_on_fly_here.run_wave_tdi(buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
}
#endif

void sobbh_run_wave_tdi_wrap(SOBBHTDIonTheFly *tdi_on_fly, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
#ifdef __CUDACC__
    SOBBHTDIonTheFly *sobbh_here = new SOBBHTDIonTheFly(tdi_on_fly->orbits, tdi_on_fly->tdi_config, tdi_on_fly->T, tdi_on_fly->t_ref);
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, tdi_on_fly->orbits, sizeof(Orbits), cudaMemcpyHostToDevice));

    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_on_fly->tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));

    sobbh_here->orbits = d_orbits;
    sobbh_here->tdi_config = d_tdi_config;

    SOBBHTDIonTheFly *d_sobbh_here;
    cudaMalloc(&d_sobbh_here, sizeof(SOBBHTDIonTheFly));
    gpuErrchk(cudaMemcpy(d_sobbh_here, sobbh_here, sizeof(SOBBHTDIonTheFly), cudaMemcpyHostToDevice));

    int buffer_length = tdi_on_fly->get_sobbh_buffer_size(N);
    sobbh_run_wave_tdi_kernel<<<num_bin, NUM_THREADS_HERE, buffer_length>>>(d_sobbh_here, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_sobbh_here));
    delete sobbh_here;
#else
    int buffer_length = tdi_on_fly->get_sobbh_buffer_size(N);
    char *buffer = new char[buffer_length];
    tdi_on_fly->run_wave_tdi((void*)buffer, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);
    delete[] buffer;
#endif
}



// ---- carved from lisa-on-gpu TDIonTheFly.cu lines 6728-6855 ----

// ---- SOBBH-flavored wrappers ----------------------------------------------
void SOBBHComputationGroup::sobbh_wdm_het_fill_global_wrap(
    double *template_fill, Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_all, double *factors_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset, double *wdm_window,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit)
{
    wdm_het_fill_global_impl<SOBBHTDIonTheFly>(
        template_fill, orbits, tdi_config,
        wdm_settings,
        params_all, factors_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk, T_chunk, dt, T, t_ref, tukey_alpha,
        grid_dim, N_cp_sig, N_cp_orbit, /*m_band_half_width=*/1);
}

void SOBBHComputationGroup::sobbh_wdm_het_get_ll_wrap(
    double *d_h_out, double *h_h_out, Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_all, int *data_index_all, int *noise_index_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset, double *wdm_window,
    double *data_d, double *invC,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref, int tdi_type,
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
    int *binary_perm, int *group_starts, int *group_ends,
    int *group_m_lo, int *group_m_hi, int n_groups)
{
    wdm_het_get_ll_impl<SOBBHTDIonTheFly>(
        d_h_out, h_h_out, orbits, tdi_config,
        wdm_settings,
        params_all,
        data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC, n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha,
        grid_dim, N_cp_sig, N_cp_orbit,
        binary_perm, group_starts, group_ends,
        group_m_lo, group_m_hi, n_groups, /*m_band_half_width=*/1);
}

void SOBBHComputationGroup::sobbh_wdm_het_swap_ll_wrap(
    double *d_h_add_out, double *d_h_remove_out,
    double *add_add_out, double *remove_remove_out, double *add_remove_out,
    Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset, double *wdm_window,
    double *data_d, double *invC,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref, int tdi_type,
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
    int *binary_perm, int *group_starts, int *group_ends,
    int *group_m_lo, int *group_m_hi, int n_groups,
    int *pair_m_lo_b, int *pair_m_hi_b)
{
    wdm_het_swap_ll_impl<SOBBHTDIonTheFly>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        orbits, tdi_config,
        wdm_settings,
        params_add_all, params_remove_all,
        data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC, n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha,
        grid_dim, N_cp_sig, N_cp_orbit,
        binary_perm, group_starts, group_ends,
        group_m_lo, group_m_hi, n_groups,
        pair_m_lo_b, pair_m_hi_b, /*m_band_half_width=*/1);
}


void SOBBHComputationGroup::sobbh_wdm_het_get_fstat_ll_wrap(
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
    int grid_dim, int m_band_half_width)
{
    wdm_het_get_fstat_ll_impl<SOBBHTDIonTheFly>(
        N_arr_re_out, N_arr_im_out, M_mat_re_out, M_mat_im_out,
        orbits, tdi_config, wdm_settings,
        params_all, data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha,
        grid_dim, m_band_half_width);
}


