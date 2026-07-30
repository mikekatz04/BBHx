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
#include <vector>      // std::vector (signal-het CPU scratch)
#include <stdexcept>   // std::invalid_argument / std::runtime_error
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
    int *data_index_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset, double *wdm_window,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
    int m_band_half_width, bool active_band,
    int Nf_slab, int *slab_min_f)
{
    wdm_het_fill_global_impl<SOBBHTDIonTheFly>(
        template_fill, orbits, tdi_config,
        wdm_settings,
        params_all, factors_all,
        data_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk, T_chunk, dt, T, t_ref, tukey_alpha,
        grid_dim, N_cp_sig, N_cp_orbit, m_band_half_width, active_band,
        Nf_slab, slab_min_f);
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
    int *group_m_lo, int *group_m_hi, int n_groups,
    int m_band_half_width,
    int Nf_slab, int *slab_min_f)
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
        group_m_lo, group_m_hi, n_groups, m_band_half_width,
        Nf_slab, slab_min_f);
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
    int *pair_m_lo_b, int *pair_m_hi_b,
    int m_band_half_width,
    int Nf_slab, int *slab_min_f)
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
        pair_m_lo_b, pair_m_hi_b, m_band_half_width,
        Nf_slab, slab_min_f);
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
    int grid_dim, int m_band_half_width,
    int Nf_slab, int *slab_min_f)
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
        grid_dim, m_band_half_width,
        Nf_slab, slab_min_f);
}


// ============================================================================
// SOBBH heterodyne sparse-FD generator + signal-heterodyne (v2 polyphase).
//
// Duplicate of GBGPU/src/gbgpu/cutils/gb_tdi_on_the_fly.cu's gbfd_* +
// gb_signal_het_* machinery (the signal-het port, 2026-06-18), with
// GBTDIonTheFly -> SOBBHTDIonTheFly and gb_ -> sobbh_. The math is identical;
// only the source physics differs (and travels in polymorphically through
// the LISATDIonTheFly::get_tdi pipeline). Authored GPU-first using the
// GPUBackendTools dual-path macros: CUDA_DEVICE per-binary workers are shared
// between CPU and GPU; CUDA_KERNEL launchers are guarded by __CUDACC__ and the
// CPU branch of each wrap drives the same workers serially.
//
// NOTE: GB's gb_signal_het_* GPU branch is still a TODO/throw; the SOBBH port
// goes one step further and provides an actual (untested-on-this-host) GPU
// path -- block-per-binary, the per-binary math on THREAD_ZERO with working
// buffers in a launcher-allocated GLOBAL scratch arena (the per-binary
// polyphase/bin-fold buffers exceed the shared-memory ceiling at realistic
// grids, which is exactly why the GB version punted). The CPU path is the
// validated reference; the GPU path mirrors it line-for-line via the macros.
// ============================================================================

#ifndef N_PARAMS_MAX
#define N_PARAMS_MAX 20
#endif

int SOBBHTDIonTheFly::get_sobbh_fd_buffer_size(int N, int nchannels)
{
    // Shared-memory budget per source for the heterodyne FD kernel; exact
    // mirror of GBTDIonTheFly::get_gb_fd_buffer_size.
    return (int) (
          N_PARAMS_MAX * sizeof(double)
        + (size_t) N * sizeof(double)
        + (size_t) nchannels * (size_t) N * sizeof(cmplx)
        + 2 * (size_t) nchannels * (size_t) N * sizeof(double)
        + (size_t) N * sizeof(double)
        + (size_t) get_tdi_buffer_size(N)
    );
}


// ---------------------------------------------------------------------------
// FD helpers (mirror of gbfd_radix2_fft_inplace / gbfd_build_one_source /
// gbfd_run_one_source). Source-agnostic apart from the SOBBHTDIonTheFly type.
// ---------------------------------------------------------------------------
CUDA_DEVICE
void sobbhfd_radix2_fft_inplace(cmplx *a, int N, int log2N)
{
    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
    {
        int r = sobbhfd_bit_reverse(n, log2N);
        if (r > n)
        {
            cmplx t = a[n];
            a[n] = a[r];
            a[r] = t;
        }
    }
    CUDA_SYNC_THREADS;

    for (int s = 1; s <= log2N; ++s)
    {
        int m  = 1 << s;
        int mh = m >> 1;
        double base = -2.0 * M_PI / (double) m;  // forward FFT sign
        for (int k = THREAD_START_X; k < (N >> 1); k += BLOCK_INCR_X)
        {
            int g  = k / mh;
            int j  = k - g * mh;
            int i0 = g * m + j;
            int i1 = i0 + mh;
            double th = base * (double) j;
            cmplx w(cos(th), sin(th));
            cmplx u = a[i0];
            cmplx v = w * a[i1];
            a[i0] = u + v;
            a[i1] = u - v;
        }
        CUDA_SYNC_THREADS;
    }
}

CUDA_DEVICE
void sobbhfd_build_one_source(SOBBHTDIonTheFly *tof, void *shared_mem,
                              double *params_in, double t_start, double Tobs,
                              int N, int nchannels, int n_params, int bin_i,
                              int log2N,
                              cmplx **tdi_chan_out,
                              int *kf0_out, double *f0g_out, double *dts_out,
                              double tukey_alpha)
{
    char *cur = (char*) shared_mem;

    double *params_here = (double*) cur;
    cur += N_PARAMS_MAX * sizeof(double);

    double *t_arr_local = (double*) cur;
    cur += (size_t) N * sizeof(double);

    cmplx *tdi_chan = (cmplx*) cur;             // also slow + FFT buffer
    cur += (size_t) nchannels * N * sizeof(cmplx);

    double *tdi_amp = (double*) cur;
    cur += (size_t) nchannels * N * sizeof(double);

    double *tdi_phase = (double*) cur;
    cur += (size_t) nchannels * N * sizeof(double);

    double *phi_ref = (double*) cur;
    cur += (size_t) N * sizeof(double);

    void *get_tdi_scratch = (void*) cur;
    int   get_tdi_scratch_len = tof->get_tdi_buffer_size(N);

    for (int i = THREAD_START_X; i < n_params; i += BLOCK_INCR_X)
        params_here[i] = params_in[bin_i * n_params + i];
    CUDA_SYNC_THREADS;

    const double f0   = params_here[tof->f0_index];
    const double df   = 1.0 / Tobs;
    const int    kf0  = (int) llround(f0 / df);
    const double f0g  = (double) kf0 * df;
    const double dts  = Tobs / (double) N;

    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
        t_arr_local[n] = t_start + (double) n * dts;
    CUDA_SYNC_THREADS;

    tof->get_tdi(get_tdi_scratch, get_tdi_scratch_len,
                 tdi_chan, tdi_amp, tdi_phase, phi_ref,
                 params_here, t_arr_local, N, bin_i, nchannels);

    // slow positive-frequency complex signal with in-line Tukey taper.
    const double n_taper_fd = 0.5 * tukey_alpha * (double) (N - 1);
    const double dlast_fd   = (double) (N - 1);
    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
    {
        const double tau     = (double) n * dts;
        const double carrier = 2.0 * M_PI * f0g * tau;
        const double phref   = phi_ref[n];
        double w = 1.0;
        if (tukey_alpha > 0.0 && n_taper_fd > 0.0) {
            const double di = (double) n;
            if (di < n_taper_fd) {
                const double xn = di / n_taper_fd;
                w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
            } else if (di > dlast_fd - n_taper_fd) {
                const double xn = (dlast_fd - di) / n_taper_fd;
                w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
            }
        }
        for (int c = 0; c < nchannels; ++c)
        {
            const double th = tdi_phase[c * N + n] + phref - carrier;
            tdi_chan[c * N + n] =
                gcmplx::polar(tdi_amp[c * N + n] * w, th);
        }
    }
    CUDA_SYNC_THREADS;

    // NaN scrub (singular response geometry) before the in-place FFT.
    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
    {
        for (int c = 0; c < nchannels; ++c)
        {
            cmplx v = tdi_chan[c * N + n];
            if (!isfinite(v.real()) || !isfinite(v.imag()))
                tdi_chan[c * N + n] = cmplx(0.0, 0.0);
        }
    }
    CUDA_SYNC_THREADS;

    for (int c = 0; c < nchannels; ++c)
    {
        sobbhfd_radix2_fft_inplace(tdi_chan + (size_t) c * N, N, log2N);
        CUDA_SYNC_THREADS;
    }

    const double scale = 0.5 * dts;
    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
    {
        for (int c = 0; c < nchannels; ++c)
        {
            cmplx v = tdi_chan[c * N + n];
            tdi_chan[c * N + n] = cmplx(v.real() * scale, v.imag() * scale);
        }
    }
    CUDA_SYNC_THREADS;

    if (tdi_chan_out) *tdi_chan_out = tdi_chan;
    if (kf0_out)      *kf0_out      = kf0;
    if (f0g_out)      *f0g_out      = f0g;
    if (dts_out)      *dts_out      = dts;
}

CUDA_DEVICE
void sobbhfd_run_one_source(SOBBHTDIonTheFly *tof, void *shared_mem,
                            cmplx *X_het, int *k_f0_out, double *f0_grid_out,
                            double *params_in, double t_start, double Tobs,
                            int N, int nchannels, int n_params, int bin_i,
                            int log2N, double tukey_alpha)
{
    cmplx *tdi_chan = NULL;
    int    kf0      = 0;
    double f0g      = 0.0;
    double dts      = 0.0;
    sobbhfd_build_one_source(tof, shared_mem, params_in, t_start, Tobs,
                             N, nchannels, n_params, bin_i, log2N,
                             &tdi_chan, &kf0, &f0g, &dts, tukey_alpha);

    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
    {
        for (int c = 0; c < nchannels; ++c)
        {
            X_het[(size_t) bin_i * nchannels * N + (size_t) c * N + n] =
                tdi_chan[c * N + n];
        }
    }

    if (THREAD_ZERO)
    {
        k_f0_out[bin_i]    = kf0;
        f0_grid_out[bin_i] = f0g;
    }
    CUDA_SYNC_THREADS;
}

#ifdef __CUDACC__
CUDA_KERNEL
void sobbh_run_fd_wave_tdi_kernel(SOBBHTDIonTheFly *tdi_on_fly,
    cmplx *X_het, int *k_f0_out, double *f0_grid_out,
    double *params, double t_start, double Tobs,
    int N, int num_bin, int n_params, int nchannels, int log2N,
    double tukey_alpha)
{
    extern CUDA_SHARED char shared_mem[];
    SOBBHTDIonTheFly tof(tdi_on_fly->orbits, tdi_on_fly->tdi_config,
                         tdi_on_fly->T, tdi_on_fly->t_ref);
    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        sobbhfd_run_one_source(&tof, (void*) shared_mem,
                               X_het, k_f0_out, f0_grid_out,
                               params, t_start, Tobs,
                               N, nchannels, n_params, bin_i, log2N,
                               tukey_alpha);
    }
}
#endif

void sobbh_run_fd_wave_tdi_wrap(SOBBHTDIonTheFly *tdi_on_fly,
    cmplx *X_het, int *k_f0_out, double *f0_grid_out,
    double *params, double t_start, double Tobs,
    int N_sparse, int num_bin, int n_params, int nchannels,
    double tukey_alpha)
{
    int log2N = 0;
    {
        int m = N_sparse;
        while ((m & 1) == 0 && m > 1) { m >>= 1; ++log2N; }
#ifndef __CUDACC__
        if (m != 1) {
            throw std::invalid_argument(
                "sobbh_run_fd_wave_tdi_wrap: N_sparse must be a power of two.");
        }
#endif
    }

#ifdef __CUDACC__
    SOBBHTDIonTheFly *sobbh_host = new SOBBHTDIonTheFly(
        tdi_on_fly->orbits, tdi_on_fly->tdi_config,
        tdi_on_fly->T, tdi_on_fly->t_ref);

    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, tdi_on_fly->orbits, sizeof(Orbits),
                         cudaMemcpyHostToDevice));

    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_on_fly->tdi_config, sizeof(TDIConfig),
                         cudaMemcpyHostToDevice));

    sobbh_host->orbits     = d_orbits;
    sobbh_host->tdi_config = d_tdi_config;

    SOBBHTDIonTheFly *d_sobbh;
    cudaMalloc(&d_sobbh, sizeof(SOBBHTDIonTheFly));
    gpuErrchk(cudaMemcpy(d_sobbh, sobbh_host, sizeof(SOBBHTDIonTheFly),
                         cudaMemcpyHostToDevice));

    int shared_bytes =
        tdi_on_fly->get_sobbh_fd_buffer_size(N_sparse, nchannels);

    if (shared_bytes > 48 * 1024)
    {
        cudaFuncSetAttribute(
            sobbh_run_fd_wave_tdi_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes);
    }

    sobbh_run_fd_wave_tdi_kernel<<<num_bin, NUM_THREADS_HERE, shared_bytes>>>(
        d_sobbh, X_het, k_f0_out, f0_grid_out,
        params, t_start, Tobs,
        N_sparse, num_bin, n_params, nchannels, log2N, tukey_alpha);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_sobbh));
    delete sobbh_host;
#else
    const int shared_bytes =
        tdi_on_fly->get_sobbh_fd_buffer_size(N_sparse, nchannels);
    char *shared_mem = new char[shared_bytes];
    for (int bin_i = 0; bin_i < num_bin; ++bin_i)
    {
        sobbhfd_run_one_source(tdi_on_fly, (void*) shared_mem,
                               X_het, k_f0_out, f0_grid_out,
                               params, t_start, Tobs,
                               N_sparse, nchannels, n_params, bin_i, log2N,
                               tukey_alpha);
    }
    delete[] shared_mem;
#endif
}


// ===========================================================================
// Signal-heterodyne (v2 polyphase) device workers. One binary each; the
// per-binary working buffers are passed in as raw pointers (shared between the
// CPU serial loop and the GPU block-per-binary kernels). Direct transcription
// of the GBComputationGroup::gb_signal_het_* CPU bodies, made device-safe
// (raw buffers instead of std::vector; floor/fmax/sqrt instead of std::).
// ===========================================================================

// Max active-m-band width. GB uses 5 layers (half_width=2); SOBBHs CHIRP across
// many layers over the observation, so the reference-guided band must be WIDE
// enough to cover the carrier sweep (the moving-window fix, 2026-06-19). A 12 mHz
// SOBBH crosses ~66 layers/year; off-track layers contribute ~0 via the c0
// safe-divide floor, so a wide fixed band centred on the carrier recovers the
// dense logL (validated in scripts/sobbh/sobbh_moving_window_proto.py). The band
// half-width travels in as m_active_half_width; this caps M = 2*half+1.
#ifndef SOBBH_SIGHET_MAX_M
#define SOBBH_SIGHET_MAX_M 600
#endif

// Per-binary, single channel polyphase fold + iFFT (mirror of GB's
// anonymous-namespace signal_het_polyphase_one_channel). `weighted` (Nt) and
// `folded` (Nt_layer) are caller-provided scratch.
CUDA_DEVICE
void sobbh_sighet_polyphase_one_channel(
    const cmplx *fd_rfft_chan, const int *m_active, int m_active_layers,
    const double *window, int Nt, int Nt_layer, int N_sparse_t, int stride,
    int Nf, int ind_min_t, const int *n_sparse_local_arr, double dt, int n_rfft,
    cmplx *c1_sparse_out, cmplx *weighted, cmplx *folded)
{
    const int    N        = Nf * Nt;
    const int    half_Nt  = Nt / 2;
    const cmplx  I_c      = cmplx(0.0, 1.0);
    const double TWO_PI   = 2.0 * M_PI;
    const double kappa    = 2.0 * sqrt(M_PI * dt) / (double) Nf;
    const int    n_start  = ind_min_t + n_sparse_local_arr[0];

    for (int im = 0; im < m_active_layers; ++im) {
        const int m_global = m_active[im];
        const int centre   = m_global * half_Nt;

        for (int i = 0; i < Nt; ++i) {
            const int j_off = i - half_Nt;
            int k = centre + j_off;
            bool conj_flag = false;
            int k_use;
            if (k < 0)          { k_use = -k;     conj_flag = true; }
            else if (k > N / 2) { k_use = N - k;  conj_flag = true; }
            else                { k_use = k; }

            cmplx h(0.0, 0.0);
            if (k_use >= 0 && k_use < n_rfft) {
                h = fd_rfft_chan[k_use];
                if (conj_flag) h = gcmplx::conj(h);
            }
            const double phase_arg = TWO_PI * (double) j_off * (double) n_start / (double) Nt;
            const cmplx  prephase  = gcmplx::exp(I_c * phase_arg);
            weighted[i] = h * window[i] * prephase;
        }

        for (int r = 0; r < Nt_layer; ++r) folded[r] = cmplx(0.0, 0.0);
        for (int i = 0; i < Nt; ++i) {
            const int r = i % Nt_layer;
            folded[r] += weighted[i];
        }

        for (int n_layer = 0; n_layer < N_sparse_t; ++n_layer) {
            cmplx acc(0.0, 0.0);
            for (int r = 0; r < Nt_layer; ++r) {
                const double pa = TWO_PI * (double) r * (double) n_layer / (double) Nt_layer;
                acc += folded[r] * gcmplx::exp(I_c * pa);
            }
            acc *= (1.0 / (double) Nt_layer);

            const int n_global = n_start + n_layer * stride;
            const double sign_scale = ((n_global & 1) ? -1.0 : 1.0) / (double) stride;
            const cmplx after_ifft_lt = acc * sign_scale;
            const int  m_plus_n  = (m_global + n_global) & 1;
            const cmplx conj_cmn = (m_plus_n == 0) ? cmplx(1.0, 0.0) : cmplx(0.0, -1.0);
            const int  sign_mn_int = ((m_global + 1) * n_global) & 1;
            const double sign_mn   = sign_mn_int ? -1.0 : 1.0;
            const cmplx coef = kappa * sign_mn * conj_cmn;
            c1_sparse_out[(size_t) im * N_sparse_t + n_layer] = after_ifft_lt * coef;
        }
    }
}

// Shared helper: r = c1/c0 with safe-divide floor + max_r clip, then dr/dn.
CUDA_DEVICE
void sobbh_sighet_build_r_dr(
    const cmplx *c1_sparse, const cmplx *c0_sparse_all,
    const int *m_active, int M, int data_idx, int nchannels,
    int Nf_active, int N_sparse_t, int ind_min_f, int stride, double max_r,
    cmplx *r_sparse, cmplx *dr_sparse)
{
    const double FLOOR_EPS = 1e-12;
    for (int c = 0; c < nchannels; ++c) {
        for (int im = 0; im < M; ++im) {
            const int m_local = m_active[im] - ind_min_f;
            double max_mag = 0.0;
            for (int b = 0; b < N_sparse_t; ++b) {
                const cmplx c0v = c0_sparse_all[
                    ((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t
                    + (size_t) m_local * N_sparse_t + b];
                const double mag = gcmplx::abs(c0v);
                if (mag > max_mag) max_mag = mag;
            }
            const double floor_th = fmax(FLOOR_EPS * max_mag, 1e-300);
            for (int b = 0; b < N_sparse_t; ++b) {
                const cmplx c0v = c0_sparse_all[
                    ((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t
                    + (size_t) m_local * N_sparse_t + b];
                const cmplx c1v = c1_sparse[
                    (size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                const size_t r_idx = (size_t) c * M * N_sparse_t
                                   + (size_t) im * N_sparse_t + b;
                if (gcmplx::abs(c0v) > floor_th) {
                    cmplx r_val = c1v / c0v;
                    if (max_r > 0.0) {
                        const double abs_r = gcmplx::abs(r_val);
                        if (abs_r > max_r) r_val = r_val * (max_r / abs_r);
                    }
                    r_sparse[r_idx] = r_val;
                } else {
                    r_sparse[r_idx] = cmplx(0.0, 0.0);
                }
            }
        }
    }

    if (dr_sparse == NULL) return;
    const double Dn = (double) stride;
    for (int c = 0; c < nchannels; ++c) {
        for (int im = 0; im < M; ++im) {
            for (int b = 0; b < N_sparse_t; ++b) {
                const size_t i_cmb = (size_t) c * M * N_sparse_t
                                   + (size_t) im * N_sparse_t + b;
                cmplx d(0.0, 0.0);
                if (N_sparse_t >= 3) {
                    if (b == 0) d = (r_sparse[i_cmb + 1] - r_sparse[i_cmb]) / Dn;
                    else if (b == N_sparse_t - 1) d = (r_sparse[i_cmb] - r_sparse[i_cmb - 1]) / Dn;
                    else d = (r_sparse[i_cmb + 1] - r_sparse[i_cmb - 1]) / (2.0 * Dn);
                } else if (N_sparse_t == 2) {
                    const size_t i0 = (size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t;
                    d = (r_sparse[i0 + 1] - r_sparse[i0]) / Dn;
                }
                dr_sparse[i_cmb] = d;
            }
        }
    }
}

// Shared helper: bin-folded <d|h>, <h|h> from r/dr + A0/A1/B0/B1(/nc).
CUDA_DEVICE
void sobbh_sighet_innerprods(
    const cmplx *r_sparse, const cmplx *dr_sparse,
    const cmplx *A0_all, const cmplx *A1_all,
    const cmplx *B0_all, const cmplx *B1_all,
    const cmplx *B0nc_all, const cmplx *B1nc_all,
    const int *m_active, int M, int data_idx, int nchannels,
    int Nf_active, int N_sparse_t, int ind_min_f, int tdi_type, int project_real,
    double *d_h_out_bin, double *h_h_out_bin)
{
    cmplx d_h_raw(0.0, 0.0), h_h_raw(0.0, 0.0);
    for (int c = 0; c < nchannels; ++c) {
        for (int im = 0; im < M; ++im) {
            const int m_local = m_active[im] - ind_min_f;
            for (int b = 0; b < N_sparse_t; ++b) {
                const cmplx r  = r_sparse[ (size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                const cmplx dr = dr_sparse[(size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                const cmplx a0 = A0_all[((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                const cmplx a1 = A1_all[((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                d_h_raw += a0 * r + a1 * dr;
            }
        }
    }
    if (tdi_type == 0) {
        for (int c = 0; c < nchannels; ++c) for (int c2 = 0; c2 < nchannels; ++c2)
            for (int im = 0; im < M; ++im) {
                const int m_local = m_active[im] - ind_min_f;
                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx r_c  = r_sparse[(size_t) c  * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                    const cmplx r_c2 = r_sparse[(size_t) c2 * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                    const cmplx dr_c  = dr_sparse[(size_t) c  * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                    const cmplx dr_c2 = dr_sparse[(size_t) c2 * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                    const cmplx b0 = B0_all[(((size_t) data_idx * nchannels + c) * nchannels + c2) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                    const cmplx b1 = B1_all[(((size_t) data_idx * nchannels + c) * nchannels + c2) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                    const cmplx r_outer = gcmplx::conj(r_c) * r_c2;
                    const cmplx cross_drr = gcmplx::conj(r_c) * dr_c2 + gcmplx::conj(dr_c) * r_c2;
                    h_h_raw += b0 * r_outer + b1 * cross_drr;
                    if (project_real) {
                        const cmplx b0nc = B0nc_all[(((size_t) data_idx * nchannels + c) * nchannels + c2) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                        const cmplx b1nc = B1nc_all[(((size_t) data_idx * nchannels + c) * nchannels + c2) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                        h_h_raw += b0nc * (r_c * r_c2) + b1nc * (r_c * dr_c2 + dr_c * r_c2);
                    }
                }
            }
    } else {
        for (int c = 0; c < nchannels; ++c) for (int im = 0; im < M; ++im) {
            const int m_local = m_active[im] - ind_min_f;
            for (int b = 0; b < N_sparse_t; ++b) {
                const cmplx r  = r_sparse[ (size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                const cmplx dr = dr_sparse[(size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                const cmplx b0 = B0_all[((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                const cmplx b1 = B1_all[((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                const double rsq = (gcmplx::conj(r) * r).real();
                const cmplx cross_drr = gcmplx::conj(r) * dr + gcmplx::conj(dr) * r;
                h_h_raw += b0 * rsq + b1 * cross_drr;
                if (project_real) {
                    const cmplx b0nc = B0nc_all[((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                    const cmplx b1nc = B1nc_all[((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                    h_h_raw += b0nc * (r * r) + b1nc * (r * dr + dr * r);
                }
            }
        }
    }
    *d_h_out_bin = 0.5 * d_h_raw.real();
    *h_h_out_bin = 0.5 * h_h_raw.real();
}

// Compute m_active band for a binary (clip to active band).
CUDA_DEVICE
void sobbh_sighet_m_active(double f0_cand, double layer_df, int m_active_half_width,
                           int ind_min_f, int Nf_active, int M, int *m_active)
{
    const int m_floor = (int) floor(f0_cand / layer_df);
    const int Nf_active_idx_max = Nf_active - 1;
    for (int im = 0; im < M; ++im) {
        int m_g = m_floor + (im - m_active_half_width);
        if (m_g < ind_min_f) m_g = ind_min_f;
        if (m_g > ind_min_f + Nf_active_idx_max) m_g = ind_min_f + Nf_active_idx_max;
        m_active[im] = m_g;
    }
}

// ---- Worker #1: dense-rfft get_ll (one binary) ----------------------------
CUDA_DEVICE
void sobbh_sighet_get_ll_one_source(
    int bin, double *d_h_out, double *h_h_out,
    const cmplx *fd_rfft_all, const cmplx *c0_sparse_all,
    const cmplx *A0_all, const cmplx *A1_all, const cmplx *B0_all, const cmplx *B1_all,
    const double *wdm_window, const int *n_sparse_local_arr,
    const double *params_cand_all, const int *data_index_all,
    int nparams, int f0_idx,
    int Nf, int Nt, int Nf_active, int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f, int m_active_half_width,
    double layer_df, double dt, int nchannels, int tdi_type, int n_rfft, double max_r,
    cmplx *c1_sparse, cmplx *r_sparse, cmplx *dr_sparse,
    cmplx *weighted, cmplx *folded)
{
    const int M = 2 * m_active_half_width + 1;
    int m_active[SOBBH_SIGHET_MAX_M];
    const double f0_cand = params_cand_all[(size_t) bin * nparams + f0_idx];
    sobbh_sighet_m_active(f0_cand, layer_df, m_active_half_width, ind_min_f, Nf_active, M, m_active);
    const int data_idx = data_index_all[bin];

    for (int c = 0; c < nchannels; ++c) {
        const cmplx *fd_chan = fd_rfft_all + (size_t) bin * nchannels * n_rfft + (size_t) c * n_rfft;
        cmplx *c1_chan = c1_sparse + (size_t) c * M * N_sparse_t;
        sobbh_sighet_polyphase_one_channel(
            fd_chan, m_active, M, wdm_window, Nt, Nt_layer, N_sparse_t,
            stride, Nf, ind_min_t, n_sparse_local_arr, dt, n_rfft, c1_chan,
            weighted, folded);
    }
    sobbh_sighet_build_r_dr(c1_sparse, c0_sparse_all, m_active, M, data_idx,
        nchannels, Nf_active, N_sparse_t, ind_min_f, stride, max_r, r_sparse, dr_sparse);
    sobbh_sighet_innerprods(r_sparse, dr_sparse, A0_all, A1_all, B0_all, B1_all,
        NULL, NULL, m_active, M, data_idx, nchannels, Nf_active, N_sparse_t,
        ind_min_f, tdi_type, 0, &d_h_out[bin], &h_h_out[bin]);
}

// ---- Worker #2: sparse-FD get_ll (one binary) -----------------------------
CUDA_DEVICE
void sobbh_sighet_get_ll_sparse_one_source(
    int bin, double *d_h_out, double *h_h_out,
    const cmplx *X_het_all, const int *k_f0_all, const cmplx *c0_sparse_all,
    const cmplx *A0_all, const cmplx *A1_all, const cmplx *B0_all, const cmplx *B1_all,
    const cmplx *B0nc_all, const cmplx *B1nc_all,
    const double *wdm_window, const int *n_sparse_local_arr,
    const double *params_cand_all, const int *data_index_all,
    int nparams, int f0_idx,
    int Nf, int Nt, int Nf_active, int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f, int m_active_half_width,
    double layer_df, double dt, int nchannels, int tdi_type,
    int N_sparse_fd, double max_r, int project_real,
    cmplx *fold, cmplx *c1_sparse, cmplx *r_sparse, cmplx *dr_sparse)
{
    const int    M       = 2 * m_active_half_width + 1;
    const double TWO_PI  = 2.0 * M_PI;
    const cmplx  I_c     = cmplx(0.0, 1.0);
    const int    half_Nt = Nt / 2;
    const int    half_NS = N_sparse_fd / 2;
    const double kappa   = 2.0 * sqrt(M_PI * dt) / (double) Nf;
    const int    n_start = ind_min_t + n_sparse_local_arr[0];

    int m_active[SOBBH_SIGHET_MAX_M];
    const double f0_cand = params_cand_all[(size_t) bin * nparams + f0_idx];
    sobbh_sighet_m_active(f0_cand, layer_df, m_active_half_width, ind_min_f, Nf_active, M, m_active);
    const int data_idx = data_index_all[bin];
    const int k_f0     = k_f0_all[bin];

    for (int i = 0; i < nchannels * M * Nt_layer; ++i) fold[i] = cmplx(0.0, 0.0);
    for (int c = 0; c < nchannels; ++c) {
        const cmplx *X_chan = X_het_all + (size_t) bin * nchannels * N_sparse_fd + (size_t) c * N_sparse_fd;
        for (int i = 0; i < N_sparse_fd; ++i) {
            const cmplx Xi = X_chan[i];
            if (Xi.real() == 0.0 && Xi.imag() == 0.0) continue;
            const int k_abs = k_f0 + (i - half_NS);
            for (int im = 0; im < M; ++im) {
                const int j = k_abs - m_active[im] * half_Nt + half_Nt;
                if (j < 0 || j >= Nt) continue;
                const int j_off = j - half_Nt;
                const double phase_arg = TWO_PI * (double) j_off * (double) n_start / (double) Nt;
                const cmplx prephase = gcmplx::exp(I_c * phase_arg);
                const cmplx weighted = Xi * wdm_window[j] * prephase;
                const int r = j % Nt_layer;
                fold[(size_t) c * M * Nt_layer + (size_t) im * Nt_layer + r] += weighted;
            }
        }
    }

    for (int c = 0; c < nchannels; ++c) {
        for (int im = 0; im < M; ++im) {
            const cmplx *fold_cm = fold + (size_t) c * M * Nt_layer + (size_t) im * Nt_layer;
            for (int n_layer = 0; n_layer < N_sparse_t; ++n_layer) {
                cmplx acc(0.0, 0.0);
                for (int rr = 0; rr < Nt_layer; ++rr) {
                    const double pa = TWO_PI * (double) rr * (double) n_layer / (double) Nt_layer;
                    acc += fold_cm[rr] * gcmplx::exp(I_c * pa);
                }
                acc *= (1.0 / (double) Nt_layer);
                const int n_global = n_start + n_layer * stride;
                const double sign_scale = ((n_global & 1) ? -1.0 : 1.0) / (double) stride;
                const cmplx after_ifft_lt = acc * sign_scale;
                const int  m_global = m_active[im];
                const int  m_plus_n = (m_global + n_global) & 1;
                const cmplx conj_cmn = (m_plus_n == 0) ? cmplx(1.0, 0.0) : cmplx(0.0, -1.0);
                const int  sign_mn_int = ((m_global + 1) * n_global) & 1;
                const double sign_mn = sign_mn_int ? -1.0 : 1.0;
                const cmplx coef = kappa * sign_mn * conj_cmn;
                c1_sparse[(size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + n_layer] = after_ifft_lt * coef;
            }
        }
    }

    sobbh_sighet_build_r_dr(c1_sparse, c0_sparse_all, m_active, M, data_idx,
        nchannels, Nf_active, N_sparse_t, ind_min_f, stride, max_r, r_sparse, dr_sparse);
    sobbh_sighet_innerprods(r_sparse, dr_sparse, A0_all, A1_all, B0_all, B1_all,
        B0nc_all, B1nc_all, m_active, M, data_idx, nchannels, Nf_active, N_sparse_t,
        ind_min_f, tdi_type, project_real, &d_h_out[bin], &h_h_out[bin]);
}

// ---- Worker #4: sparse-FD fill_global (one binary) ------------------------
CUDA_DEVICE
void sobbh_sighet_fill_global_sparse_one_source(
    int bin, double *template_fill,
    const cmplx *X_het_all, const int *k_f0_all, const cmplx *c0_sparse_all,
    const cmplx *c0_dense_complex_all,
    const double *wdm_window, const int *n_sparse_local_arr,
    const double *params_cand_all, const double *params_ref_all, const double *factors_all,
    const int *data_index_all,
    int nparams, int f0_idx, int fdot_idx,
    int Nf, int Nt, int Nf_active, int Nt_active, int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f, int m_active_half_width,
    double layer_df, double dt, int nchannels, int N_sparse_fd, double max_r,
    cmplx *fold, cmplx *c1_sparse, cmplx *r_sparse)
{
    const int    M       = 2 * m_active_half_width + 1;
    const double TWO_PI  = 2.0 * M_PI;
    const cmplx  I_c     = cmplx(0.0, 1.0);
    const int    half_Nt = Nt / 2;
    const int    half_NS = N_sparse_fd / 2;
    const double kappa   = 2.0 * sqrt(M_PI * dt) / (double) Nf;
    const int    n_start = ind_min_t + n_sparse_local_arr[0];
    const double layer_dt = (double) Nf * dt;

    int m_active[SOBBH_SIGHET_MAX_M];
    const double f0_cand   = params_cand_all[(size_t) bin * nparams + f0_idx];
    const double fdot_cand = params_cand_all[(size_t) bin * nparams + fdot_idx];
    sobbh_sighet_m_active(f0_cand, layer_df, m_active_half_width, ind_min_f, Nf_active, M, m_active);
    const int    data_idx = data_index_all[bin];
    const int    k_f0     = k_f0_all[bin];
    const double factor   = factors_all[bin];
    const double f0_ref   = params_ref_all[(size_t) data_idx * nparams + f0_idx];
    const double fdot_ref = params_ref_all[(size_t) data_idx * nparams + fdot_idx];
    const double Df0      = f0_cand   - f0_ref;
    const double Dfdot    = fdot_cand - fdot_ref;

    for (int i = 0; i < nchannels * M * Nt_layer; ++i) fold[i] = cmplx(0.0, 0.0);
    for (int c = 0; c < nchannels; ++c) {
        const cmplx *X_chan = X_het_all + (size_t) bin * nchannels * N_sparse_fd + (size_t) c * N_sparse_fd;
        for (int i = 0; i < N_sparse_fd; ++i) {
            const cmplx Xi = X_chan[i];
            if (Xi.real() == 0.0 && Xi.imag() == 0.0) continue;
            const int k_abs = k_f0 + (i - half_NS);
            for (int im = 0; im < M; ++im) {
                const int j = k_abs - m_active[im] * half_Nt + half_Nt;
                if (j < 0 || j >= Nt) continue;
                const int j_off = j - half_Nt;
                const double phase_arg = TWO_PI * (double) j_off * (double) n_start / (double) Nt;
                const cmplx prephase = gcmplx::exp(I_c * phase_arg);
                const cmplx weighted = Xi * wdm_window[j] * prephase;
                const int r = j % Nt_layer;
                fold[(size_t) c * M * Nt_layer + (size_t) im * Nt_layer + r] += weighted;
            }
        }
    }

    for (int c = 0; c < nchannels; ++c) {
        for (int im = 0; im < M; ++im) {
            const cmplx *fold_cm = fold + (size_t) c * M * Nt_layer + (size_t) im * Nt_layer;
            for (int n_layer = 0; n_layer < N_sparse_t; ++n_layer) {
                cmplx acc(0.0, 0.0);
                for (int rr = 0; rr < Nt_layer; ++rr) {
                    const double pa = TWO_PI * (double) rr * (double) n_layer / (double) Nt_layer;
                    acc += fold_cm[rr] * gcmplx::exp(I_c * pa);
                }
                acc *= (1.0 / (double) Nt_layer);
                const int n_global = n_start + n_layer * stride;
                const double sign_scale = ((n_global & 1) ? -1.0 : 1.0) / (double) stride;
                const cmplx after_ifft_lt = acc * sign_scale;
                const int  m_global = m_active[im];
                const int  m_plus_n = (m_global + n_global) & 1;
                const cmplx conj_cmn = (m_plus_n == 0) ? cmplx(1.0, 0.0) : cmplx(0.0, -1.0);
                const int  sign_mn_int = ((m_global + 1) * n_global) & 1;
                const double sign_mn = sign_mn_int ? -1.0 : 1.0;
                const cmplx coef = kappa * sign_mn * conj_cmn;
                c1_sparse[(size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + n_layer] = after_ifft_lt * coef;
            }
        }
    }

    sobbh_sighet_build_r_dr(c1_sparse, c0_sparse_all, m_active, M, data_idx,
        nchannels, Nf_active, N_sparse_t, ind_min_f, stride, max_r, r_sparse, NULL);

    // carrier de-rotate r_sparse in place.
    for (int b = 0; b < N_sparse_t; ++b) {
        const int n_sparse_local = n_sparse_local_arr[b];
        const double t_n = (double)(ind_min_t + n_sparse_local) * layer_dt;
        const double phase_pred = TWO_PI * Df0 * t_n + M_PI * Dfdot * t_n * t_n;
        const cmplx rot = gcmplx::exp(I_c * (-phase_pred));
        for (int c = 0; c < nchannels; ++c)
            for (int im = 0; im < M; ++im) {
                const size_t idx = (size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b;
                r_sparse[idx] = r_sparse[idx] * rot;
            }
    }

    const int n_sparse_local_0 = n_sparse_local_arr[0];
    for (int n_dense = 0; n_dense < Nt_active; ++n_dense) {
        const int n_off = n_dense - n_sparse_local_0;
        int    b_lo  = n_off / stride;
        double frac = (double)(n_off - b_lo * stride) / (double) stride;
        if (b_lo < 0) { b_lo = 0; frac = 0.0; }
        if (b_lo >= N_sparse_t - 1) { b_lo = N_sparse_t - 1; frac = 0.0; }
        const int b_hi = (b_lo + 1 < N_sparse_t) ? (b_lo + 1) : b_lo;

        const double t_n_dense = (double)(ind_min_t + n_dense) * layer_dt;
        const double phase_dense = TWO_PI * Df0 * t_n_dense + M_PI * Dfdot * t_n_dense * t_n_dense;
        const cmplx rot_back = gcmplx::exp(I_c * phase_dense);
        const int n_global = ind_min_t + n_dense;

        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                const cmplx r_lo = r_sparse[(size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b_lo];
                const cmplx r_hi = r_sparse[(size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b_hi];
                const cmplx r_demod_dense = r_lo * (1.0 - frac) + r_hi * frac;
                const cmplx r_dense = r_demod_dense * rot_back;
                const int m_local = m_active[im] - ind_min_f;
                const cmplx c0v = c0_dense_complex_all[
                    ((size_t) data_idx * nchannels + c) * Nf_active * Nt_active
                    + (size_t) m_local * Nt_active + n_dense];
                const cmplx c1_dense = r_dense * c0v;
                const int m_global = m_active[im];
                const size_t out_idx = ((size_t) data_idx * nchannels + c) * Nf * Nt
                                     + (size_t) m_global * Nt + n_global;
                template_fill[out_idx] += factor * c1_dense.real();
            }
        }
    }
}

// fftshift + (1/dt) conversion of one binary's raw FD into the centered slice.
CUDA_DEVICE
void sobbh_sighet_fd_convert_one(int bin, const cmplx *X_het_raw, cmplx *X_het,
                                 int nchannels, int N_sparse_fd, double dt_inv)
{
    const int half_NS = N_sparse_fd / 2;
    for (int c = 0; c < nchannels; ++c) {
        const size_t base = ((size_t) bin * nchannels + c) * N_sparse_fd;
        for (int i = 0; i < N_sparse_fd; ++i) {
            const int m_signed = i - half_NS;
            const int m_fft = (m_signed >= 0) ? m_signed : (m_signed + N_sparse_fd);
            X_het[base + i] = X_het_raw[base + m_fft] * dt_inv;
        }
    }
}

#ifdef __CUDACC__
CUDA_KERNEL
void sobbh_sighet_get_ll_kernel(
    double *d_h_out, double *h_h_out,
    const cmplx *fd_rfft_all, const cmplx *c0_sparse_all,
    const cmplx *A0_all, const cmplx *A1_all, const cmplx *B0_all, const cmplx *B1_all,
    const double *wdm_window, const int *n_sparse_local_arr,
    const double *params_cand_all, const int *data_index_all,
    int num_bin, int nparams, int f0_idx,
    int Nf, int Nt, int Nf_active, int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f, int m_active_half_width,
    double layer_df, double dt, int nchannels, int tdi_type, int n_rfft, double max_r,
    cmplx *c1_s, cmplx *r_s, cmplx *dr_s, cmplx *wt_s, cmplx *fold_s)
{
    const int M = 2 * m_active_half_width + 1;
    const size_t cm = (size_t) nchannels * M * N_sparse_t;
    for (int bin = BLOCK_START_X; bin < num_bin; bin += GRID_INCR_X) {
        if (THREAD_ZERO)
            sobbh_sighet_get_ll_one_source(bin, d_h_out, h_h_out,
                fd_rfft_all, c0_sparse_all, A0_all, A1_all, B0_all, B1_all,
                wdm_window, n_sparse_local_arr, params_cand_all, data_index_all,
                nparams, f0_idx, Nf, Nt, Nf_active, Nt_layer, N_sparse_t, stride,
                ind_min_t, ind_min_f, m_active_half_width, layer_df, dt, nchannels,
                tdi_type, n_rfft, max_r,
                c1_s + (size_t) bin * cm, r_s + (size_t) bin * cm, dr_s + (size_t) bin * cm,
                wt_s + (size_t) bin * Nt, fold_s + (size_t) bin * Nt_layer);
    }
}

CUDA_KERNEL
void sobbh_sighet_get_ll_sparse_kernel(
    double *d_h_out, double *h_h_out,
    const cmplx *X_het_all, const int *k_f0_all, const cmplx *c0_sparse_all,
    const cmplx *A0_all, const cmplx *A1_all, const cmplx *B0_all, const cmplx *B1_all,
    const cmplx *B0nc_all, const cmplx *B1nc_all,
    const double *wdm_window, const int *n_sparse_local_arr,
    const double *params_cand_all, const int *data_index_all,
    int num_bin, int nparams, int f0_idx,
    int Nf, int Nt, int Nf_active, int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f, int m_active_half_width,
    double layer_df, double dt, int nchannels, int tdi_type,
    int N_sparse_fd, double max_r, int project_real,
    cmplx *fold_s, cmplx *c1_s, cmplx *r_s, cmplx *dr_s)
{
    const int M = 2 * m_active_half_width + 1;
    const size_t fl = (size_t) nchannels * M * Nt_layer;
    const size_t cm = (size_t) nchannels * M * N_sparse_t;
    for (int bin = BLOCK_START_X; bin < num_bin; bin += GRID_INCR_X) {
        if (THREAD_ZERO)
            sobbh_sighet_get_ll_sparse_one_source(bin, d_h_out, h_h_out,
                X_het_all, k_f0_all, c0_sparse_all, A0_all, A1_all, B0_all, B1_all,
                B0nc_all, B1nc_all, wdm_window, n_sparse_local_arr,
                params_cand_all, data_index_all, nparams, f0_idx,
                Nf, Nt, Nf_active, Nt_layer, N_sparse_t, stride,
                ind_min_t, ind_min_f, m_active_half_width, layer_df, dt, nchannels,
                tdi_type, N_sparse_fd, max_r, project_real,
                fold_s + (size_t) bin * fl, c1_s + (size_t) bin * cm,
                r_s + (size_t) bin * cm, dr_s + (size_t) bin * cm);
    }
}

CUDA_KERNEL
void sobbh_sighet_fill_global_sparse_kernel(
    double *template_fill,
    const cmplx *X_het_all, const int *k_f0_all, const cmplx *c0_sparse_all,
    const cmplx *c0_dense_complex_all,
    const double *wdm_window, const int *n_sparse_local_arr,
    const double *params_cand_all, const double *params_ref_all, const double *factors_all,
    const int *data_index_all,
    int num_bin, int nparams, int f0_idx, int fdot_idx,
    int Nf, int Nt, int Nf_active, int Nt_active, int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f, int m_active_half_width,
    double layer_df, double dt, int nchannels, int N_sparse_fd, double max_r,
    cmplx *fold_s, cmplx *c1_s, cmplx *r_s)
{
    const int M = 2 * m_active_half_width + 1;
    const size_t fl = (size_t) nchannels * M * Nt_layer;
    const size_t cm = (size_t) nchannels * M * N_sparse_t;
    for (int bin = BLOCK_START_X; bin < num_bin; bin += GRID_INCR_X) {
        if (THREAD_ZERO)
            sobbh_sighet_fill_global_sparse_one_source(bin, template_fill,
                X_het_all, k_f0_all, c0_sparse_all, c0_dense_complex_all,
                wdm_window, n_sparse_local_arr, params_cand_all, params_ref_all,
                factors_all, data_index_all, nparams, f0_idx, fdot_idx,
                Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
                ind_min_t, ind_min_f, m_active_half_width, layer_df, dt, nchannels,
                N_sparse_fd, max_r,
                fold_s + (size_t) bin * fl, c1_s + (size_t) bin * cm, r_s + (size_t) bin * cm);
    }
}

CUDA_KERNEL
void sobbh_sighet_fd_convert_kernel(const cmplx *X_het_raw, cmplx *X_het,
                                    int num_bin, int nchannels, int N_sparse_fd, double dt_inv)
{
    for (int bin = BLOCK_START_X; bin < num_bin; bin += GRID_INCR_X)
        if (THREAD_ZERO)
            sobbh_sighet_fd_convert_one(bin, X_het_raw, X_het, nchannels, N_sparse_fd, dt_inv);
}
#endif  // __CUDACC__


// ===========================================================================
// Signal-heterodyne wrap methods (dual-path launchers). GPU branch: global
// scratch arena + block-per-binary kernel launch. CPU branch: reused single
// slab + serial worker loop. Mirror of GBComputationGroup::gb_signal_het_*.
// ===========================================================================

void SOBBHComputationGroup::sobbh_signal_het_get_ll_wrap(
    double *d_h_out, double *h_h_out,
    cmplx  *fd_rfft_all,
    cmplx  *c0_sparse_all, cmplx *A0_all, cmplx *A1_all,
    cmplx  *B0_all, cmplx *B1_all,
    double *wdm_window, int *n_sparse_local_arr,
    double *params_cand_all, double *params_ref_all,
    int    *data_index_all,
    int     num_bin, int num_data,
    int     nparams, int f0_idx, int fdot_idx,
    int     Nf, int Nt, int Nf_active, int Nt_active,
    int     Nt_layer, int N_sparse_t, int stride,
    int     ind_min_t, int ind_min_f,
    int     m_active_half_width,
    double  layer_df, double dt,
    int     nchannels, int tdi_type,
    int     n_rfft, double max_r)
{
    (void) params_ref_all; (void) fdot_idx; (void) num_data; (void) Nt_active;
    const int M = 2 * m_active_half_width + 1;
    const size_t cm = (size_t) nchannels * M * N_sparse_t;

#ifdef __CUDACC__
    cmplx *c1_s, *r_s, *dr_s, *wt_s, *fold_s;
    gpuErrchk(cudaMalloc(&c1_s,  (size_t) num_bin * cm * sizeof(cmplx)));
    gpuErrchk(cudaMalloc(&r_s,   (size_t) num_bin * cm * sizeof(cmplx)));
    gpuErrchk(cudaMalloc(&dr_s,  (size_t) num_bin * cm * sizeof(cmplx)));
    gpuErrchk(cudaMalloc(&wt_s,  (size_t) num_bin * Nt * sizeof(cmplx)));
    gpuErrchk(cudaMalloc(&fold_s,(size_t) num_bin * Nt_layer * sizeof(cmplx)));
    sobbh_sighet_get_ll_kernel<<<num_bin, 1>>>(
        d_h_out, h_h_out, fd_rfft_all, c0_sparse_all, A0_all, A1_all, B0_all, B1_all,
        wdm_window, n_sparse_local_arr, params_cand_all, data_index_all,
        num_bin, nparams, f0_idx, Nf, Nt, Nf_active, Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f, m_active_half_width, layer_df, dt, nchannels,
        tdi_type, n_rfft, max_r, c1_s, r_s, dr_s, wt_s, fold_s);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaFree(c1_s)); gpuErrchk(cudaFree(r_s)); gpuErrchk(cudaFree(dr_s));
    gpuErrchk(cudaFree(wt_s)); gpuErrchk(cudaFree(fold_s));
#else
    std::vector<cmplx> c1(cm), r(cm), dr(cm), wt(Nt), fold(Nt_layer);
    for (int bin = 0; bin < num_bin; ++bin)
        sobbh_sighet_get_ll_one_source(bin, d_h_out, h_h_out,
            fd_rfft_all, c0_sparse_all, A0_all, A1_all, B0_all, B1_all,
            wdm_window, n_sparse_local_arr, params_cand_all, data_index_all,
            nparams, f0_idx, Nf, Nt, Nf_active, Nt_layer, N_sparse_t, stride,
            ind_min_t, ind_min_f, m_active_half_width, layer_df, dt, nchannels,
            tdi_type, n_rfft, max_r,
            c1.data(), r.data(), dr.data(), wt.data(), fold.data());
#endif
}

void SOBBHComputationGroup::sobbh_signal_het_get_ll_sparse_wrap(
    double *d_h_out, double *h_h_out,
    cmplx  *X_het_all, int *k_f0_all,
    cmplx  *c0_sparse_all, cmplx *A0_all, cmplx *A1_all,
    cmplx  *B0_all, cmplx *B1_all,
    cmplx  *B0nc_all, cmplx *B1nc_all,
    double *wdm_window, int *n_sparse_local_arr,
    double *params_cand_all, double *params_ref_all,
    int    *data_index_all,
    int     num_bin, int num_data,
    int     nparams, int f0_idx, int fdot_idx,
    int     Nf, int Nt, int Nf_active, int Nt_active,
    int     Nt_layer, int N_sparse_t, int stride,
    int     ind_min_t, int ind_min_f,
    int     m_active_half_width,
    double  layer_df, double dt,
    int     nchannels, int tdi_type,
    int     N_sparse_fd, double max_r, int project_real)
{
    (void) params_ref_all; (void) fdot_idx; (void) num_data; (void) Nt_active;
    const int M = 2 * m_active_half_width + 1;
    const size_t fl = (size_t) nchannels * M * Nt_layer;
    const size_t cm = (size_t) nchannels * M * N_sparse_t;

#ifdef __CUDACC__
    cmplx *fold_s, *c1_s, *r_s, *dr_s;
    gpuErrchk(cudaMalloc(&fold_s,(size_t) num_bin * fl * sizeof(cmplx)));
    gpuErrchk(cudaMalloc(&c1_s,  (size_t) num_bin * cm * sizeof(cmplx)));
    gpuErrchk(cudaMalloc(&r_s,   (size_t) num_bin * cm * sizeof(cmplx)));
    gpuErrchk(cudaMalloc(&dr_s,  (size_t) num_bin * cm * sizeof(cmplx)));
    sobbh_sighet_get_ll_sparse_kernel<<<num_bin, 1>>>(
        d_h_out, h_h_out, X_het_all, k_f0_all, c0_sparse_all,
        A0_all, A1_all, B0_all, B1_all, B0nc_all, B1nc_all,
        wdm_window, n_sparse_local_arr, params_cand_all, data_index_all,
        num_bin, nparams, f0_idx, Nf, Nt, Nf_active, Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f, m_active_half_width, layer_df, dt, nchannels,
        tdi_type, N_sparse_fd, max_r, project_real, fold_s, c1_s, r_s, dr_s);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaFree(fold_s)); gpuErrchk(cudaFree(c1_s));
    gpuErrchk(cudaFree(r_s)); gpuErrchk(cudaFree(dr_s));
#else
    std::vector<cmplx> fold(fl), c1(cm), r(cm), dr(cm);
    for (int bin = 0; bin < num_bin; ++bin)
        sobbh_sighet_get_ll_sparse_one_source(bin, d_h_out, h_h_out,
            X_het_all, k_f0_all, c0_sparse_all, A0_all, A1_all, B0_all, B1_all,
            B0nc_all, B1nc_all, wdm_window, n_sparse_local_arr,
            params_cand_all, data_index_all, nparams, f0_idx,
            Nf, Nt, Nf_active, Nt_layer, N_sparse_t, stride,
            ind_min_t, ind_min_f, m_active_half_width, layer_df, dt, nchannels,
            tdi_type, N_sparse_fd, max_r, project_real,
            fold.data(), c1.data(), r.data(), dr.data());
#endif
}

void SOBBHComputationGroup::sobbh_signal_het_get_ll_in_kernel_wrap(
    SOBBHTDIonTheFly *tdi_on_fly,
    double *d_h_out, double *h_h_out,
    cmplx  *c0_sparse_all,
    cmplx  *A0_all, cmplx *A1_all, cmplx *B0_all, cmplx *B1_all,
    cmplx  *B0nc_all, cmplx *B1nc_all,
    double *wdm_window, int *n_sparse_local_arr,
    double *params_cand_all, double *params_ref_all,
    int    *data_index_all,
    int     num_bin, int num_data,
    int     nparams, int f0_idx, int fdot_idx,
    int     Nf, int Nt, int Nf_active, int Nt_active,
    int     Nt_layer, int N_sparse_t, int stride,
    int     ind_min_t, int ind_min_f,
    int     m_active_half_width,
    double  layer_df, double dt,
    double  T_obs, double t_start,
    int     nchannels, int tdi_type,
    int     N_sparse_fd, double tukey_alpha, double max_r, int project_real)
{
    const size_t xlen = (size_t) num_bin * nchannels * N_sparse_fd;
    const double dt_inv = 1.0 / dt;

#ifdef __CUDACC__
    cmplx *X_het_raw, *X_het; int *k_f0; double *f0_grid;
    gpuErrchk(cudaMalloc(&X_het_raw, xlen * sizeof(cmplx)));
    gpuErrchk(cudaMalloc(&X_het,     xlen * sizeof(cmplx)));
    gpuErrchk(cudaMalloc(&k_f0,      (size_t) num_bin * sizeof(int)));
    gpuErrchk(cudaMalloc(&f0_grid,   (size_t) num_bin * sizeof(double)));

    sobbh_run_fd_wave_tdi_wrap(tdi_on_fly, X_het_raw, k_f0, f0_grid,
        params_cand_all, t_start, T_obs, N_sparse_fd, num_bin, nparams,
        nchannels, tukey_alpha);

    sobbh_sighet_fd_convert_kernel<<<num_bin, 1>>>(
        X_het_raw, X_het, num_bin, nchannels, N_sparse_fd, dt_inv);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    this->sobbh_signal_het_get_ll_sparse_wrap(
        d_h_out, h_h_out, X_het, k_f0, c0_sparse_all, A0_all, A1_all, B0_all, B1_all,
        B0nc_all, B1nc_all, wdm_window, n_sparse_local_arr,
        params_cand_all, params_ref_all, data_index_all, num_bin, num_data,
        nparams, f0_idx, fdot_idx, Nf, Nt, Nf_active, Nt_active,
        Nt_layer, N_sparse_t, stride, ind_min_t, ind_min_f, m_active_half_width,
        layer_df, dt, nchannels, tdi_type, N_sparse_fd, max_r, project_real);

    gpuErrchk(cudaFree(X_het_raw)); gpuErrchk(cudaFree(X_het));
    gpuErrchk(cudaFree(k_f0)); gpuErrchk(cudaFree(f0_grid));
#else
    std::vector<cmplx>  X_het_raw(xlen);
    std::vector<int>    k_f0_buf(num_bin);
    std::vector<double> f0_grid_buf(num_bin);
    sobbh_run_fd_wave_tdi_wrap(tdi_on_fly, X_het_raw.data(), k_f0_buf.data(),
        f0_grid_buf.data(), params_cand_all, t_start, T_obs, N_sparse_fd,
        num_bin, nparams, nchannels, tukey_alpha);

    std::vector<cmplx> X_het(xlen);
    for (int bin = 0; bin < num_bin; ++bin)
        sobbh_sighet_fd_convert_one(bin, X_het_raw.data(), X_het.data(),
                                    nchannels, N_sparse_fd, dt_inv);

    this->sobbh_signal_het_get_ll_sparse_wrap(
        d_h_out, h_h_out, X_het.data(), k_f0_buf.data(), c0_sparse_all,
        A0_all, A1_all, B0_all, B1_all, B0nc_all, B1nc_all,
        wdm_window, n_sparse_local_arr, params_cand_all, params_ref_all,
        data_index_all, num_bin, num_data, nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f, m_active_half_width, layer_df, dt,
        nchannels, tdi_type, N_sparse_fd, max_r, project_real);
#endif
}

void SOBBHComputationGroup::sobbh_signal_het_fill_global_sparse_wrap(
    double *template_fill,
    cmplx  *X_het_all, int *k_f0_all,
    cmplx  *c0_sparse_all, cmplx *c0_dense_complex_all,
    double *wdm_window, int *n_sparse_local_arr,
    double *params_cand_all, double *params_ref_all, double *factors_all,
    int    *data_index_all,
    int     num_bin, int num_data,
    int     nparams, int f0_idx, int fdot_idx,
    int     Nf, int Nt, int Nf_active, int Nt_active,
    int     Nt_layer, int N_sparse_t, int stride,
    int     ind_min_t, int ind_min_f,
    int     m_active_half_width,
    double  layer_df, double dt,
    int     nchannels,
    int     N_sparse_fd, double max_r)
{
    (void) num_data;
    const int M = 2 * m_active_half_width + 1;
    const size_t fl = (size_t) nchannels * M * Nt_layer;
    const size_t cm = (size_t) nchannels * M * N_sparse_t;

#ifdef __CUDACC__
    cmplx *fold_s, *c1_s, *r_s;
    gpuErrchk(cudaMalloc(&fold_s,(size_t) num_bin * fl * sizeof(cmplx)));
    gpuErrchk(cudaMalloc(&c1_s,  (size_t) num_bin * cm * sizeof(cmplx)));
    gpuErrchk(cudaMalloc(&r_s,   (size_t) num_bin * cm * sizeof(cmplx)));
    sobbh_sighet_fill_global_sparse_kernel<<<num_bin, 1>>>(
        template_fill, X_het_all, k_f0_all, c0_sparse_all, c0_dense_complex_all,
        wdm_window, n_sparse_local_arr, params_cand_all, params_ref_all,
        factors_all, data_index_all, num_bin, nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f, m_active_half_width, layer_df, dt, nchannels,
        N_sparse_fd, max_r, fold_s, c1_s, r_s);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaFree(fold_s)); gpuErrchk(cudaFree(c1_s)); gpuErrchk(cudaFree(r_s));
#else
    std::vector<cmplx> fold(fl), c1(cm), r(cm);
    for (int bin = 0; bin < num_bin; ++bin)
        sobbh_sighet_fill_global_sparse_one_source(bin, template_fill,
            X_het_all, k_f0_all, c0_sparse_all, c0_dense_complex_all,
            wdm_window, n_sparse_local_arr, params_cand_all, params_ref_all,
            factors_all, data_index_all, nparams, f0_idx, fdot_idx,
            Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
            ind_min_t, ind_min_f, m_active_half_width, layer_df, dt, nchannels,
            N_sparse_fd, max_r, fold.data(), c1.data(), r.data());
#endif
}

void SOBBHComputationGroup::sobbh_signal_het_fill_global_in_kernel_wrap(
    SOBBHTDIonTheFly *tdi_on_fly,
    double *template_fill,
    cmplx  *c0_sparse_all, cmplx *c0_dense_complex_all,
    double *wdm_window, int *n_sparse_local_arr,
    double *params_cand_all, double *params_ref_all, double *factors_all,
    int    *data_index_all,
    int     num_bin, int num_data,
    int     nparams, int f0_idx, int fdot_idx,
    int     Nf, int Nt, int Nf_active, int Nt_active,
    int     Nt_layer, int N_sparse_t, int stride,
    int     ind_min_t, int ind_min_f,
    int     m_active_half_width,
    double  layer_df, double dt,
    double  T_obs, double t_start,
    int     nchannels,
    int     N_sparse_fd, double tukey_alpha, double max_r)
{
    const size_t xlen = (size_t) num_bin * nchannels * N_sparse_fd;
    const double dt_inv = 1.0 / dt;

#ifdef __CUDACC__
    cmplx *X_het_raw, *X_het; int *k_f0; double *f0_grid;
    gpuErrchk(cudaMalloc(&X_het_raw, xlen * sizeof(cmplx)));
    gpuErrchk(cudaMalloc(&X_het,     xlen * sizeof(cmplx)));
    gpuErrchk(cudaMalloc(&k_f0,      (size_t) num_bin * sizeof(int)));
    gpuErrchk(cudaMalloc(&f0_grid,   (size_t) num_bin * sizeof(double)));
    sobbh_run_fd_wave_tdi_wrap(tdi_on_fly, X_het_raw, k_f0, f0_grid,
        params_cand_all, t_start, T_obs, N_sparse_fd, num_bin, nparams,
        nchannels, tukey_alpha);
    sobbh_sighet_fd_convert_kernel<<<num_bin, 1>>>(
        X_het_raw, X_het, num_bin, nchannels, N_sparse_fd, dt_inv);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    this->sobbh_signal_het_fill_global_sparse_wrap(
        template_fill, X_het, k_f0, c0_sparse_all, c0_dense_complex_all,
        wdm_window, n_sparse_local_arr, params_cand_all, params_ref_all,
        factors_all, data_index_all, num_bin, num_data, nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f, m_active_half_width, layer_df, dt, nchannels,
        N_sparse_fd, max_r);
    gpuErrchk(cudaFree(X_het_raw)); gpuErrchk(cudaFree(X_het));
    gpuErrchk(cudaFree(k_f0)); gpuErrchk(cudaFree(f0_grid));
#else
    std::vector<cmplx>  X_het_raw(xlen);
    std::vector<int>    k_f0_buf(num_bin);
    std::vector<double> f0_grid_buf(num_bin);
    sobbh_run_fd_wave_tdi_wrap(tdi_on_fly, X_het_raw.data(), k_f0_buf.data(),
        f0_grid_buf.data(), params_cand_all, t_start, T_obs, N_sparse_fd,
        num_bin, nparams, nchannels, tukey_alpha);
    std::vector<cmplx> X_het(xlen);
    for (int bin = 0; bin < num_bin; ++bin)
        sobbh_sighet_fd_convert_one(bin, X_het_raw.data(), X_het.data(),
                                    nchannels, N_sparse_fd, dt_inv);
    this->sobbh_signal_het_fill_global_sparse_wrap(
        template_fill, X_het.data(), k_f0_buf.data(), c0_sparse_all,
        c0_dense_complex_all, wdm_window, n_sparse_local_arr,
        params_cand_all, params_ref_all, factors_all, data_index_all,
        num_bin, num_data, nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f, m_active_half_width, layer_df, dt, nchannels,
        N_sparse_fd, max_r);
#endif
}

void SOBBHComputationGroup::sobbh_signal_het_get_ll_grad_in_kernel_wrap(
    SOBBHTDIonTheFly *tdi_on_fly,
    double *grad_out,
    double *d_h_central, double *h_h_central,
    cmplx  *c0_sparse_all,
    cmplx  *A0_all, cmplx *A1_all, cmplx *B0_all, cmplx *B1_all,
    double *wdm_window, int *n_sparse_local_arr,
    double *params_cand_all, double *params_ref_all,
    int    *data_index_all,
    double *param_eps,
    int     num_bin, int num_data,
    int     nparams, int f0_idx, int fdot_idx,
    int     Nf, int Nt, int Nf_active, int Nt_active,
    int     Nt_layer, int N_sparse_t, int stride,
    int     ind_min_t, int ind_min_f,
    int     m_active_half_width,
    double  layer_df, double dt,
    double  T_obs, double t_start,
    int     nchannels, int tdi_type,
    int     N_sparse_fd, double tukey_alpha, double max_r)
{
    // Per-binary central differences. Each evaluation re-uses the single-binary
    // get_ll_in_kernel path (which regenerates the FD for the perturbed params).
    // B0nc/B1nc are left null here -> the gradient uses the complex bin-fold
    // (matches the GB convention; project_real=0).
    std::vector<double> params_priv((size_t) nparams);
    for (int bin = 0; bin < num_bin; ++bin) {
        int data_idx_local = data_index_all[bin];

        for (int i = 0; i < nparams; ++i)
            params_priv[i] = params_cand_all[(size_t) bin * nparams + i];

        double d_h_C = 0.0, h_h_C = 0.0;
        this->sobbh_signal_het_get_ll_in_kernel_wrap(
            tdi_on_fly, &d_h_C, &h_h_C, c0_sparse_all,
            A0_all, A1_all, B0_all, B1_all, nullptr, nullptr,
            wdm_window, n_sparse_local_arr, params_priv.data(), params_ref_all,
            &data_idx_local, 1, num_data, nparams, f0_idx, fdot_idx,
            Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
            ind_min_t, ind_min_f, m_active_half_width, layer_df, dt,
            T_obs, t_start, nchannels, tdi_type, N_sparse_fd, tukey_alpha, max_r, 0);
        d_h_central[bin] = d_h_C;
        h_h_central[bin] = h_h_C;

        for (int k = 0; k < nparams; ++k) {
            const double eps = param_eps[k];
            if (eps <= 0.0) { grad_out[(size_t) bin * nparams + k] = 0.0; continue; }
            const double saved = params_priv[k];

            double d_h_P = 0.0, h_h_P = 0.0, d_h_M = 0.0, h_h_M = 0.0;
            params_priv[k] = saved + eps;
            this->sobbh_signal_het_get_ll_in_kernel_wrap(
                tdi_on_fly, &d_h_P, &h_h_P, c0_sparse_all,
                A0_all, A1_all, B0_all, B1_all, nullptr, nullptr,
                wdm_window, n_sparse_local_arr, params_priv.data(), params_ref_all,
                &data_idx_local, 1, num_data, nparams, f0_idx, fdot_idx,
                Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
                ind_min_t, ind_min_f, m_active_half_width, layer_df, dt,
                T_obs, t_start, nchannels, tdi_type, N_sparse_fd, tukey_alpha, max_r, 0);

            params_priv[k] = saved - eps;
            this->sobbh_signal_het_get_ll_in_kernel_wrap(
                tdi_on_fly, &d_h_M, &h_h_M, c0_sparse_all,
                A0_all, A1_all, B0_all, B1_all, nullptr, nullptr,
                wdm_window, n_sparse_local_arr, params_priv.data(), params_ref_all,
                &data_idx_local, 1, num_data, nparams, f0_idx, fdot_idx,
                Nf, Nt, Nf_active, Nt_active, Nt_layer, N_sparse_t, stride,
                ind_min_t, ind_min_f, m_active_half_width, layer_df, dt,
                T_obs, t_start, nchannels, tdi_type, N_sparse_fd, tukey_alpha, max_r, 0);

            params_priv[k] = saved;
            const double ll_P = d_h_P - 0.5 * h_h_P;
            const double ll_M = d_h_M - 0.5 * h_h_M;
            grad_out[(size_t) bin * nparams + k] = (ll_P - ll_M) / (2.0 * eps);
        }
    }
}


