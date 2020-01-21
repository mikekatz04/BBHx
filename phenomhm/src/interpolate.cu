/*  This code was created by Michael Katz.
 *  It is shared under the GNU license (see below).
 *  This code computes the interpolations for the GPU PhenomHM waveform.
 *
 *
 *  Copyright (C) 2019 Michael Katz
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with with program; see the file COPYING. If not, write to the
 *  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 *  MA  02111-1307  USA
 */

#include "manager.hh"
#include "stdio.h"
#include <assert.h>
#include "globalPhenomHM.h"
#include "interpolate.hh"

#ifdef __CUDACC__
#include "cusparse.h"
#else
#include "omp.h"
#include "lapacke.h"
#endif

#ifdef __CUDACC__
/*
GPU error checking
*/
#define gpuErrchk_here(ans) { gpuAssert_here((ans), __FILE__, __LINE__); }
inline void gpuAssert_here(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*
CuSparse error checking
*/
#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
                             fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
                             exit(-1);}} while(0)

#define CUDA_CALL(X) ERR_NE((X),cudaSuccess)
#define CUSPARSE_CALL(X) ERR_NE((X),CUSPARSE_STATUS_SUCCESS)

#endif


CUDA_CALLABLE_MEMBER void prep_splines(int i, int length, double *b, double *ud, double *diag, double *ld, double *x, double *y){
  double dx1, dx2, d, slope1, slope2;
  if (i == length - 1){
    dx1 = x[length - 2] - x[length - 3];
    dx2 = x[length - 1] - x[length - 2];
    d = x[length - 1] - x[length - 3];

    slope1 = (y[length - 2] - y[length - 3])/dx1;
    slope2 = (y[length - 1] - y[length - 2])/dx2;

    b[length - 1] = ((dx2*dx2*slope1 +
                             (2*d + dx2)*dx1*slope2) / d);
    diag[length - 1] = dx1;
    ld[length - 1] = d;
    ud[length - 1] = 0.0;

  } else if (i == 0){
      dx1 = x[1] - x[0];
      dx2 = x[2] - x[1];
      d = x[2] - x[0];

      //amp
      slope1 = (y[1] - y[0])/dx1;
      slope2 = (y[2] - y[1])/dx2;

      b[0] = ((dx1 + 2*d) * dx2 * slope1 +
                          dx1*dx1 * slope2) / d;
      diag[0] = dx2;
      ud[0] = d;
      ld[0] = 0.0;

  } else{
    dx1 = x[i] - x[i-1];
    dx2 = x[i+1] - x[i];

    //amp
    slope1 = (y[i] - y[i-1])/dx1;
    slope2 = (y[i+1] - y[i])/dx2;

    b[i] = 3.0* (dx2*slope1 + dx1*slope2);
    diag[i] = 2*(dx1 + dx2);
    ud[i] = dx1;
    ld[i] = dx2;
  }
}

/*
fill the B array on the GPU for response transfer functions.
*/
CUDA_CALLABLE_MEMBER
void fill_B_response(ModeContainer *mode_vals, double *B, double *freqs, double *upper_diag, double *diag, double *lower_diag, int f_length, int num_modes, int mode_i, int i){
    int num_pars = 8;
    int lead_ind;

    // phaseRdelay
    lead_ind = (0*num_modes*f_length) + mode_i*f_length;
    prep_splines(i, f_length, &B[lead_ind], &upper_diag[lead_ind], &diag[lead_ind], &lower_diag[lead_ind], freqs, mode_vals[mode_i].phaseRdelay);

    // transferL1_re
    lead_ind = (1*num_modes*f_length) + mode_i*f_length;
    prep_splines(i, f_length, &B[lead_ind], &upper_diag[lead_ind], &diag[lead_ind], &lower_diag[lead_ind], freqs, mode_vals[mode_i].transferL1_re);

    // transferL1_im
    lead_ind = (2*num_modes*f_length) + mode_i*f_length;
    prep_splines(i, f_length, &B[lead_ind], &upper_diag[lead_ind], &diag[lead_ind], &lower_diag[lead_ind], freqs, mode_vals[mode_i].transferL1_im);

    // transferL2_re
    lead_ind = (3*num_modes*f_length) + mode_i*f_length;
    prep_splines(i, f_length, &B[lead_ind], &upper_diag[lead_ind], &diag[lead_ind], &lower_diag[lead_ind], freqs, mode_vals[mode_i].transferL2_re);

    // transfer2_im
    lead_ind = (4*num_modes*f_length) + mode_i*f_length;
    prep_splines(i, f_length, &B[lead_ind], &upper_diag[lead_ind], &diag[lead_ind], &lower_diag[lead_ind], freqs, mode_vals[mode_i].transferL2_im);

    // transferL3_re
    lead_ind = (5*num_modes*f_length) + mode_i*f_length;
    prep_splines(i, f_length, &B[lead_ind], &upper_diag[lead_ind], &diag[lead_ind], &lower_diag[lead_ind], freqs, mode_vals[mode_i].transferL3_re);

    // transferL3_im
    lead_ind = (6*num_modes*f_length) + mode_i*f_length;
    prep_splines(i, f_length, &B[lead_ind], &upper_diag[lead_ind], &diag[lead_ind], &lower_diag[lead_ind], freqs, mode_vals[mode_i].transferL3_im);

    // time_freq_corr
    lead_ind = (7*num_modes*f_length) + mode_i*f_length;
    prep_splines(i, f_length, &B[lead_ind], &upper_diag[lead_ind], &diag[lead_ind], &lower_diag[lead_ind], freqs, mode_vals[mode_i].time_freq_corr);


}

#ifdef __CUDACC__
CUDA_KERNEL
void fill_B_response_wrap(ModeContainer *mode_vals, double *freqs, double *B, double *upper_diag, double *diag, double *lower_diag, int f_length, int num_modes, int nwalkers){
    int num_pars = 8;

    for (int walker_i = blockIdx.z * blockDim.z + threadIdx.z;
         walker_i < nwalkers;
         walker_i += blockDim.z * gridDim.z){

    for (int mode_i = blockIdx.y * blockDim.y + threadIdx.y;
         mode_i < num_modes;
         mode_i += blockDim.y * gridDim.y){

       for (int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < f_length;
            i += blockDim.x * gridDim.x){

              int mode_index = walker_i*num_modes + mode_i;

              fill_B_response(mode_vals, B, &freqs[walker_i*f_length], upper_diag, diag, lower_diag, f_length, num_modes*nwalkers, mode_index, i);

}
}
}
}
#else
void cpu_fill_B_response_wrap(ModeContainer *mode_vals, double *freqs, double *B, double *upper_diag, double *diag, double *lower_diag, int f_length, int num_modes, int nwalkers){
    int num_pars = 8;
    #pragma omp parallel for collapse(2)
    for (int walker_i = 0;
         walker_i < nwalkers;
         walker_i += 1){
    for (int mode_i = 0;
         mode_i < num_modes;
         mode_i += 1){

       for (int i = 0;
            i < f_length;
            i += 1){
              int mode_index = walker_i*num_modes + mode_i;

              fill_B_response(mode_vals, B, &freqs[walker_i*f_length], upper_diag, diag, lower_diag, f_length, num_modes*nwalkers, mode_index, i);

}
}
}
}
#endif


/*
fill B array on GPU for amp and phase
*/
CUDA_CALLABLE_MEMBER void fill_B_wave(ModeContainer *mode_vals, double *B, double *freqs, double *upper_diag, double *diag, double *lower_diag, int f_length, int num_modes, int mode_i, int i){
    int num_pars = 2;
    int lead_ind;

    // amp
    lead_ind = mode_i*f_length;
    prep_splines(i, f_length, &B[lead_ind], &upper_diag[lead_ind], &diag[lead_ind], &lower_diag[lead_ind], freqs, mode_vals[mode_i].amp);

    // phase
    lead_ind = (num_modes*f_length) + mode_i*f_length;
    prep_splines(i, f_length, &B[lead_ind], &upper_diag[lead_ind], &diag[lead_ind], &lower_diag[lead_ind], freqs, mode_vals[mode_i].phase);

}

#ifdef __CUDACC__
CUDA_KERNEL void fill_B_wave_wrap(ModeContainer *mode_vals, double *freqs, double *B, double *upper_diag, double *diag, double *lower_diag, int f_length, int num_modes, int nwalkers){
    int num_pars = 2;

    for (int walker_i = blockIdx.z * blockDim.z + threadIdx.z;
         walker_i < nwalkers;
         walker_i += blockDim.z * gridDim.z){

    for (int mode_i = blockIdx.y * blockDim.y + threadIdx.y;
         mode_i < num_modes;
         mode_i += blockDim.y * gridDim.y){

       for (int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < f_length;
            i += blockDim.x * gridDim.x){

              int mode_index = walker_i*num_modes + mode_i;

              fill_B_wave(mode_vals, B, &freqs[walker_i*f_length], upper_diag, diag, lower_diag, f_length, num_modes*nwalkers, mode_index, i);
}
}
}
}
#else
void cpu_fill_B_wave_wrap(ModeContainer *mode_vals, double *freqs, double *B, double *upper_diag, double *diag, double *lower_diag, int f_length, int num_modes, int nwalkers){

    int num_pars = 2;
    #pragma omp parallel for collapse(3)
    for (int walker_i = 0;
         walker_i < nwalkers;
         walker_i += 1){
    for (int mode_i = 0;
         mode_i < num_modes;
         mode_i += 1){

       for (int i = 0;
            i < f_length;
            i += 1){
              int mode_index = walker_i*num_modes + mode_i;

              fill_B_wave(mode_vals, B, &freqs[walker_i*f_length], upper_diag, diag, lower_diag, f_length, num_modes*nwalkers, mode_index, i);

}
}
}
}
#endif

CUDA_CALLABLE_MEMBER
void fill_coefficients(int i, int length, double *dydx, double dx, double *y, double *coeff1, double *coeff2, double *coeff3){
  double slope, t, dydx_i;

  slope = (y[i+1] - y[i])/dx;

  dydx_i = dydx[i];

  t = (dydx_i + dydx[i+1] - 2*slope)/dx;

  coeff1[i] = dydx_i;
  coeff2[i] = (slope - dydx_i) / dx - t;
  coeff3[i] = t/dx;
}

/*
find spline constants based on matrix solution for response transfer functions.
*/
CUDA_CALLABLE_MEMBER
void set_spline_constants_response(ModeContainer *mode_vals, double *B, int f_length, int num_modes, int mode_i, int i, double df){

    int lead_ind;

    // phaseRdelay
    lead_ind = (0*num_modes*f_length) + mode_i*f_length;
    fill_coefficients(i, f_length, &B[lead_ind], df, mode_vals[mode_i].phaseRdelay, mode_vals[mode_i].phaseRdelay_coeff_1, mode_vals[mode_i].phaseRdelay_coeff_2, mode_vals[mode_i].phaseRdelay_coeff_3);

    // transferL1_re
    lead_ind = (1*num_modes*f_length) + mode_i*f_length;
    fill_coefficients(i, f_length, &B[lead_ind], df, mode_vals[mode_i].transferL1_re, mode_vals[mode_i].transferL1_re_coeff_1, mode_vals[mode_i].transferL1_re_coeff_2, mode_vals[mode_i].transferL1_re_coeff_3);

    // transferL1_im
    lead_ind = (2*num_modes*f_length) + mode_i*f_length;
    fill_coefficients(i, f_length, &B[lead_ind], df, mode_vals[mode_i].transferL1_im, mode_vals[mode_i].transferL1_im_coeff_1, mode_vals[mode_i].transferL1_im_coeff_2, mode_vals[mode_i].transferL1_im_coeff_3);

    // transferL2_re
    lead_ind = (3*num_modes*f_length) + mode_i*f_length;
    fill_coefficients(i, f_length, &B[lead_ind], df, mode_vals[mode_i].transferL2_re, mode_vals[mode_i].transferL2_re_coeff_1, mode_vals[mode_i].transferL2_re_coeff_2, mode_vals[mode_i].transferL2_re_coeff_3);

    // transferL2_im
    lead_ind = (4*num_modes*f_length) + mode_i*f_length;
    fill_coefficients(i, f_length, &B[lead_ind], df, mode_vals[mode_i].transferL2_im, mode_vals[mode_i].transferL2_im_coeff_1, mode_vals[mode_i].transferL2_im_coeff_2, mode_vals[mode_i].transferL2_im_coeff_3);

    // transferL3_re
    lead_ind = (5*num_modes*f_length) + mode_i*f_length;
    fill_coefficients(i, f_length, &B[lead_ind], df, mode_vals[mode_i].transferL3_re, mode_vals[mode_i].transferL3_re_coeff_1, mode_vals[mode_i].transferL3_re_coeff_2, mode_vals[mode_i].transferL3_re_coeff_3);

    // transferL3_img
    lead_ind = (6*num_modes*f_length) + mode_i*f_length;
    fill_coefficients(i, f_length, &B[lead_ind], df, mode_vals[mode_i].transferL3_im, mode_vals[mode_i].transferL3_im_coeff_1, mode_vals[mode_i].transferL3_im_coeff_2, mode_vals[mode_i].transferL3_im_coeff_3);

    // time_freq_corr
    lead_ind = (7*num_modes*f_length) + mode_i*f_length;
    fill_coefficients(i, f_length, &B[lead_ind], df, mode_vals[mode_i].time_freq_corr, mode_vals[mode_i].time_freq_coeff_1, mode_vals[mode_i].time_freq_coeff_2, mode_vals[mode_i].time_freq_coeff_3);

}

#ifdef __CUDACC__
CUDA_KERNEL
void set_spline_constants_response_wrap(ModeContainer *mode_vals, double *B, int f_length, int nwalkers, int num_modes, double *freqs){
    double D_i, D_ip1, y_i, y_ip1;
    int num_pars = 8;
    int mode_index;
    double df;

    for (int walker_i = blockIdx.z * blockDim.z + threadIdx.z;
         walker_i < nwalkers;
         walker_i += blockDim.z * gridDim.z){

    for (int mode_i = blockIdx.y * blockDim.y + threadIdx.y;
         mode_i < num_modes;
         mode_i += blockDim.y * gridDim.y){

       mode_index = walker_i*num_modes + mode_i;

       for (int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < f_length-1;
            i += blockDim.x * gridDim.x){

              df = freqs[walker_i*f_length + i + 1] - freqs[walker_i*f_length + i];
              set_spline_constants_response(mode_vals, B, f_length, num_modes*nwalkers, mode_index, i, df);
  }
  }
}
}
#else
void cpu_set_spline_constants_response_wrap(ModeContainer *mode_vals, double *B, int f_length, int nwalkers, int num_modes, double *freqs){
    double D_i, D_ip1, y_i, y_ip1;
    int num_pars = 8;
    double df;
    int mode_index;

    #pragma omp parallel for collapse(3)
    for (int walker_i = 0;
         walker_i < nwalkers;
         walker_i += 1){
    for (int mode_i = 0;
         mode_i < num_modes;
         mode_i += 1){

       for (int i = 0;
            i < f_length-1;
            i += 1){

              mode_index = walker_i*num_modes + mode_i;
              df = freqs[walker_i*f_length + i + 1] - freqs[walker_i*f_length + i];

              set_spline_constants_response(mode_vals, B, f_length, num_modes*nwalkers, mode_index, i, df);
  }
  }
}
}
#endif

/*
Find spline coefficients after matrix calculation on GPU for amp and phase
*/

CUDA_CALLABLE_MEMBER void set_spline_constants_wave(ModeContainer *mode_vals, double *B, int f_length, int num_modes, int mode_i, int i, double df){

  int lead_ind;

  // amp
  lead_ind = mode_i*f_length;
  fill_coefficients(i, f_length, &B[lead_ind], df, mode_vals[mode_i].amp, mode_vals[mode_i].amp_coeff_1, mode_vals[mode_i].amp_coeff_2, mode_vals[mode_i].amp_coeff_3);

  // phase
  lead_ind = (num_modes*f_length) + mode_i*f_length;
  fill_coefficients(i, f_length, &B[lead_ind], df, mode_vals[mode_i].phase, mode_vals[mode_i].phase_coeff_1, mode_vals[mode_i].phase_coeff_2, mode_vals[mode_i].phase_coeff_3);
}

#ifdef __CUDACC__
CUDA_KERNEL void set_spline_constants_wave_wrap(ModeContainer *mode_vals, double *B, int f_length, int nwalkers, int num_modes, double *freqs){

    double D_i, D_ip1, y_i, y_ip1;
    int num_pars = 2;
    int mode_index;
    double df;

    for (int walker_i = blockIdx.z * blockDim.z + threadIdx.z;
         walker_i < nwalkers;
         walker_i += blockDim.z * gridDim.z){

     for (int mode_i = blockIdx.y * blockDim.y + threadIdx.y;
          mode_i < num_modes;
          mode_i += blockDim.y * gridDim.y){

          mode_index = walker_i*num_modes + mode_i;

        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < f_length-1;
             i += blockDim.x * gridDim.x){

              df = freqs[walker_i*f_length + i + 1] - freqs[walker_i*f_length + i];

               set_spline_constants_wave(mode_vals, B, f_length, num_modes*nwalkers, mode_index, i, df);
}
}
}
}
#else
void cpu_set_spline_constants_wave_wrap(ModeContainer *mode_vals, double *B, int f_length, int nwalkers, int num_modes, double *freqs){

    double D_i, D_ip1, y_i, y_ip1;
    int num_pars = 2;
    int mode_index;
    double df;

    #pragma omp parallel for collapse(3)
    for (int walker_i = 0;
         walker_i < nwalkers;
         walker_i += 1){
     for (int mode_i = 0;
          mode_i < num_modes;
          mode_i += 1){

        for (int i = 0;
             i < f_length-1;
             i += 1){

              mode_index = walker_i*num_modes + mode_i;
              df = freqs[walker_i*f_length + i + 1] - freqs[walker_i*f_length + i];
               set_spline_constants_wave(mode_vals, B, f_length, num_modes*nwalkers, mode_index, i, df);
}
}
}
}
#endif

/*
Interpolate amp, phase, and response transfer functions on GPU.
*/
CUDA_CALLABLE_MEMBER
void interpolate(agcmplx *channel1_out, agcmplx *channel2_out, agcmplx *channel3_out, ModeContainer* old_mode_vals,
    int num_modes, double d_log10f, double *old_freqs, int old_length, double *data_freqs, int data_length,
    double *channel1_ASDinv, double *channel2_ASDinv, double *channel3_ASDinv, int num_walkers,
    double f_min_limit, double f_max_limit, double t_break_start, double t_break_end, double t_obs_end, int walker_i, int i, double tc, double tShift){
    //int mode_i = blockIdx.y;

    double f, x, x2, x3, coeff_0, coeff_1, coeff_2, coeff_3;
    double time_check, amp, phase, phaseRdelay, phaseShift;
    double transferL1_re, transferL1_im, transferL2_re, transferL2_im, transferL3_re, transferL3_im;
    agcmplx ampphasefactor;
    //agcmplx I = agcmplx(0.0, 1.0);
    int old_ind_below;
    agcmplx trans_complex1 = agcmplx(0.0, 0.0);
    agcmplx trans_complex2 = agcmplx(0.0, 0.0);
    agcmplx trans_complex3 = agcmplx(0.0, 0.0);

    //if (mode_i >= num_modes) return;

    /*# if __CUDA_ARCH__>=200
    if (i == 200)
        printf("times: %e %e, %e, %e \n", t0, tRef, t_obs_start, t_break_start);

    #endif*/
    /*for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < data_length;
         i += blockDim.x * gridDim.x)
      {*/
      f = data_freqs[i];
      old_ind_below = floor((log10(f) - log10(old_freqs[walker_i*old_length + 0]))/d_log10f);

      if ((old_ind_below == old_length -1) || (f >= f_max_limit) || (f < f_min_limit) || (old_ind_below >= old_length)){
          return;
      }
      x = (f - old_freqs[walker_i*old_length + old_ind_below]);
      x2 = x*x;
      x3 = x*x2;

    ModeContainer *old_mode_vals_i;

    for (int mode_i=0; mode_i<num_modes; mode_i++){
            old_mode_vals_i = &old_mode_vals[walker_i*num_modes + mode_i];
            // interp time frequency to remove less than 0.0
            coeff_0 = old_mode_vals_i->time_freq_corr[old_ind_below];
            coeff_1 = old_mode_vals_i->time_freq_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals_i->time_freq_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals_i->time_freq_coeff_3[old_ind_below];

            time_check = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            if (time_check < t_break_start) {
                continue;
            }

            if ((t_obs_end > 0.0) && (time_check >= t_break_end)){
                continue;
            }

            // interp amplitude
            coeff_0 = old_mode_vals_i->amp[old_ind_below];
            coeff_1 = old_mode_vals_i->amp_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals_i->amp_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals_i->amp_coeff_3[old_ind_below];

            amp = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);
            if (amp < 1e-40){
                continue;
            }
            // interp phase
            coeff_0 = old_mode_vals_i->phase[old_ind_below];
            coeff_1 = old_mode_vals_i->phase_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals_i->phase_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals_i->phase_coeff_3[old_ind_below];

            phase  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals_i->phaseRdelay[old_ind_below];
            coeff_1 = old_mode_vals_i->phaseRdelay_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals_i->phaseRdelay_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals_i->phaseRdelay_coeff_3[old_ind_below];

            phaseRdelay  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);
            phaseShift = 2.0*PI*f*(tc + tShift); // tc is t0 and tShift if tRef_wave_frame
            ampphasefactor = amp*gcmplx::exp(agcmplx(0.0, phase + phaseRdelay + phaseShift));

            // X or A
            coeff_0 = old_mode_vals_i->transferL1_re[old_ind_below];
            coeff_1 = old_mode_vals_i->transferL1_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals_i->transferL1_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals_i->transferL1_re_coeff_3[old_ind_below];

            transferL1_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals_i->transferL1_im[old_ind_below];
            coeff_1 = old_mode_vals_i->transferL1_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals_i->transferL1_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals_i->transferL1_im_coeff_3[old_ind_below];

            transferL1_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            trans_complex1 += gcmplx::conj(agcmplx(transferL1_re, transferL1_im)* ampphasefactor * channel1_ASDinv[i]); //TODO may be faster to load as complex number with 0.0 for imaginary part

            // Y or E
            coeff_0 = old_mode_vals_i->transferL2_re[old_ind_below];
            coeff_1 = old_mode_vals_i->transferL2_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals_i->transferL2_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals_i->transferL2_re_coeff_3[old_ind_below];

            transferL2_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals_i->transferL2_im[old_ind_below];
            coeff_1 = old_mode_vals_i->transferL2_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals_i->transferL2_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals_i->transferL2_im_coeff_3[old_ind_below];

            transferL2_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            trans_complex2 += gcmplx::conj(agcmplx(transferL2_re, transferL2_im)* ampphasefactor * channel2_ASDinv[i]); //TODO may be faster to load as complex number with 0.0 for imaginary part

            // Z or T
            coeff_0 = old_mode_vals_i->transferL3_re[old_ind_below];
            coeff_1 = old_mode_vals_i->transferL3_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals_i->transferL3_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals_i->transferL3_re_coeff_3[old_ind_below];

            transferL3_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals_i->transferL3_im[old_ind_below];
            coeff_1 = old_mode_vals_i->transferL3_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals_i->transferL3_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals_i->transferL3_im_coeff_3[old_ind_below];

            transferL3_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            trans_complex3 += gcmplx::conj(agcmplx(transferL3_re, transferL3_im)* ampphasefactor * channel3_ASDinv[i]); //TODO may be faster to load as complex number with 0.0 for imaginary part

          }

        channel1_out[walker_i*data_length + i] = trans_complex1;
        channel2_out[walker_i*data_length + i] = trans_complex2;
        channel3_out[walker_i*data_length + i] = trans_complex3;
}

#ifdef __CUDACC__
CUDA_KERNEL
void interpolate_wrap(agcmplx *channel1_out, agcmplx *channel2_out, agcmplx *channel3_out, ModeContainer* old_mode_vals,
    int num_modes, double d_log10f, double *old_freqs, int old_length, double *data_freqs, int data_length, double* t0_arr, double* tRef_arr, double* tRef_wave_frame_arr, double *channel1_ASDinv,
    double *channel2_ASDinv, double *channel3_ASDinv, double t_obs_start, double t_obs_end, int num_walkers){
    //int mode_i = blockIdx.y;

    double f_min_limit, f_max_limit, t0, tRef, t_break_start, t_break_end, tRef_wave_frame;

    for (int walker_i = blockIdx.z * blockDim.z + threadIdx.z;
         walker_i < num_walkers;
         walker_i += blockDim.z * gridDim.z){

     f_min_limit = old_freqs[walker_i*old_length];
     f_max_limit = old_freqs[walker_i*old_length + old_length-1];
     t0 = t0_arr[walker_i];
     tRef = tRef_arr[walker_i];
     tRef_wave_frame = tRef_wave_frame_arr[walker_i];
     t_break_start = t0*YRSID_SI + tRef - t_obs_start*YRSID_SI; // t0 and t_obs_start in years. tRef in seconds.
     t_break_end = t0*YRSID_SI + tRef - t_obs_end*YRSID_SI;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < data_length;
         i += blockDim.x * gridDim.x){

            interpolate(channel1_out, channel2_out, channel3_out, old_mode_vals, num_modes, d_log10f, old_freqs, old_length,
                        data_freqs, data_length, channel1_ASDinv, channel2_ASDinv, channel3_ASDinv, num_walkers,
                        f_min_limit, f_max_limit, t_break_start, t_break_end, t_obs_end, walker_i, i, tRef_wave_frame, t0);

}
}
}

#else
void cpu_interpolate_wrap(agcmplx *channel1_out, agcmplx *channel2_out, agcmplx *channel3_out, ModeContainer* old_mode_vals,
    int num_modes, double d_log10f, double *old_freqs, int old_length, double *data_freqs, int data_length, double* t0_arr, double* tRef_arr, double* tRef_wave_frame_arr, double *channel1_ASDinv,
    double *channel2_ASDinv, double *channel3_ASDinv, double t_obs_start, double t_obs_end, int num_walkers){
    //int mode_i = blockIdx.y;

    double f_min_limit, f_max_limit, t0, tRef, t_break_start, t_break_end, tRef_wave_frame;

    #pragma omp parallel for collapse(2)
    for (int walker_i = 0;
         walker_i < num_walkers;
         walker_i += 1){

    for (int i = 0;
         i < data_length;
         i += 1){

           f_min_limit = old_freqs[walker_i*old_length];
           f_max_limit = old_freqs[walker_i*old_length + old_length-1];
           t0 = t0_arr[walker_i];
           tRef = tRef_arr[walker_i];
           tRef_wave_frame = tRef_wave_frame_arr[walker_i];
           t_break_start = t0*YRSID_SI + tRef - t_obs_start*YRSID_SI; // t0 and t_obs_start in years. tRef in seconds.
           t_break_end = t0*YRSID_SI + tRef - t_obs_end*YRSID_SI;

            interpolate(channel1_out, channel2_out, channel3_out, old_mode_vals, num_modes, d_log10f, old_freqs, old_length,
                        data_freqs, data_length, channel1_ASDinv, channel2_ASDinv, channel3_ASDinv, num_walkers,
                        f_min_limit, f_max_limit, t_break_start, t_break_end, t_obs_end, walker_i, i, tRef_wave_frame, t0);

}
}
}
#endif

/*
Interpolation class initializer
*/

Interpolate::Interpolate(){
    int pass = 0;
}

/*
allocate arrays for interpolation
*/

void Interpolate::alloc_arrays(int m, int n, double *d_B){
    w = new double[m*n];
    x = new double[m*n];

    #ifdef __CUDACC__
    gpuErrchk_here(cudaMalloc(&d_w, m*sizeof(double)));
    gpuErrchk_here(cudaMalloc(&d_x, m*n*sizeof(double)));
    #endif
}


CUDA_CALLABLE_MEMBER
void fit_constants_serial(int m, int n, double *w_in, double *a_in, double *b_in, double *c_in, double *d_in, double *x_in, int j){

  double *x, *d, *a, *b, *c, *w;

      d = &d_in[j*m];
      x = &x_in[j*m];
      a = &a_in[j*m];
      b = &b_in[j*m];
      c = &c_in[j*m];
      w = &w_in[j*m];



      for (int i=1; i<m; i++){
          w[i] = a[i]/b[i-1];
          b[i] = b[i] - w[i]*c[i-1];
          d[i] = d[i] - w[i]*d[i-1];
      }

      x[m-1] = d[m-1]/b[m-1];
      d[m-1] = x[m-1];
      for (int i=(m-2); i>=0; i--){
          x[i] = (d[i] - c[i]*x[i+1])/b[i];
          d[i] = x[i];
      }
}

#ifdef __CUDACC__
__global__
void printit(int index, double *a, double *b, double *c, double *d_in)
{

  printf("%e, %e, %e, %e\n", a[index], b[index], c[index], d_in[index]);

}
#endif

#ifdef __CUDACC__
void fit_constants_serial_wrap(int m, int n, double *w, double *a, double *b, double *c, double *d_in, double *x_in){

  void *pBuffer;
  cusparseStatus_t stat;
  cusparseHandle_t handle;

  size_t bufferSizeInBytes;

  CUSPARSE_CALL(cusparseCreate(&handle));
  CUSPARSE_CALL( cusparseDgtsv2StridedBatch_bufferSizeExt(handle, m, a, b, c, d_in, n, m, &bufferSizeInBytes));
  gpuErrchk_here(cudaMalloc(&pBuffer, bufferSizeInBytes));

    CUSPARSE_CALL(cusparseDgtsv2StridedBatch(handle,
                                              m,
                                              a, // dl
                                              b, //diag
                                              c, // du
                                              d_in,
                                              n,
                                              m,
                                              pBuffer));


CUSPARSE_CALL(cusparseDestroy(handle));
gpuErrchk_here(cudaFree(pBuffer));
}

#else
void cpu_fit_constants_serial_wrap(int m, int n, double *w, double *a, double *b, double *c, double *d_in, double *x_in){

    #pragma omp parallel for
    for (int j = 0;
         j < n;
         j += 1){
           //fit_constants_serial(m, n, w, a, b, c, d_in, x_in, j);
           int info = LAPACKE_dgtsv(LAPACK_COL_MAJOR, m, 1, &a[j*m + 1], &b[j*m], &c[j*m], &d_in[j*m], m);
           //if (info != m) printf("lapack info check: %d\n", info);

    }
}
#endif



/*
solve matrix solution for tridiagonal matrix for cublic spline.
*/
void Interpolate::prep(double *B, double *c, double *b, double *a, int m_, int n_, int to_gpu_){
    m = m_;
    n = n_;
    to_gpu = to_gpu_;

    #ifdef __CUDACC__
    int NUM_THREADS = 256;
    int num_blocks = std::ceil((n + NUM_THREADS -1)/NUM_THREADS);
    fit_constants_serial_wrap(m, n, w, a, b, c, B, x);
    #else

    cpu_fit_constants_serial_wrap(m, n, w, a, b, c, B, x);
    #endif
}


/*
Deallocate
*/
Interpolate::~Interpolate(){
  delete[] w;
  delete[] x;

  #ifdef __CUDACC__
    cudaFree(d_w);
    cudaFree(d_x);

  #endif
}
