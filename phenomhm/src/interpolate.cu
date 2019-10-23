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

#include "omp.h"

/*
GPU error checking
*/
#ifdef __CUDACC__
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

using namespace std;

/*
fill the B array on the GPU for response transfer functions.
*/
#ifdef __CUDACC__
__host__ __device__
#endif
void fill_B_response_inner(ModeContainer *mode_vals, double *B, int f_length, int num_modes, int mode_i, int i){
  if (i == f_length - 1){
      B[(0*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phaseRdelay[i] - mode_vals[mode_i].phaseRdelay[i-1]);
      B[(1*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_re[i] - mode_vals[mode_i].transferL1_re[i-1]);
      B[(2*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_im[i] - mode_vals[mode_i].transferL1_im[i-1]);
      B[(3*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_re[i] - mode_vals[mode_i].transferL2_re[i-1]);
      B[(4*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_im[i] - mode_vals[mode_i].transferL2_im[i-1]);
      B[(5*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_re[i] - mode_vals[mode_i].transferL3_re[i-1]);
      B[(6*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_im[i] - mode_vals[mode_i].transferL3_im[i-1]);
      B[(7*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].time_freq_corr[i] - mode_vals[mode_i].time_freq_corr[i-1]);

  } else if (i == 0){
      B[(0*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phaseRdelay[1] - mode_vals[mode_i].phaseRdelay[0]);
      B[(1*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_re[1] - mode_vals[mode_i].transferL1_re[0]);
      B[(2*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_im[1] - mode_vals[mode_i].transferL1_im[0]);
      B[(3*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_re[1] - mode_vals[mode_i].transferL2_re[0]);
      B[(4*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_im[1] - mode_vals[mode_i].transferL2_im[0]);
      B[(5*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_re[1] - mode_vals[mode_i].transferL3_re[0]);
      B[(6*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_im[1] - mode_vals[mode_i].transferL3_im[0]);
      B[(7*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].time_freq_corr[1] - mode_vals[mode_i].time_freq_corr[0]);
  } else{
      B[(0*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phaseRdelay[i+1] - mode_vals[mode_i].phaseRdelay[i-1]);
      B[(1*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_re[i+1] - mode_vals[mode_i].transferL1_re[i-1]);
      B[(2*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL1_im[i+1] - mode_vals[mode_i].transferL1_im[i-1]);
      B[(3*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_re[i+1] - mode_vals[mode_i].transferL2_re[i-1]);
      B[(4*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL2_im[i+1] - mode_vals[mode_i].transferL2_im[i-1]);
      B[(5*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_re[i+1] - mode_vals[mode_i].transferL3_re[i-1]);
      B[(6*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].transferL3_im[i+1] - mode_vals[mode_i].transferL3_im[i-1]);
      B[(7*num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].time_freq_corr[i+1] - mode_vals[mode_i].time_freq_corr[i-1]);
  }
}

#ifdef __CUDACC__
__host__
#endif
void cpu_fill_B_response(ModeContainer *mode_vals, double *B, int f_length, int num_modes, int mode_i){
       for (int i = 0;
            i < f_length;
            i += 1){
          fill_B_response_inner(mode_vals, B, f_length, num_modes, mode_i, i);
}
}

/*
fill B array on GPU for amp and phase
*/

#ifdef __CUDACC__
__host__ __device__
#endif
void fill_B_wave_inner(ModeContainer *mode_vals, double *B, int f_length, int num_modes, int mode_i, int i){
      if (i == f_length - 1){
          B[mode_i*f_length + i] = 3.0* (mode_vals[mode_i].amp[i] - mode_vals[mode_i].amp[i-1]);
          B[(num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phase[i] - mode_vals[mode_i].phase[i-1]);
      } else if (i == 0){
          B[mode_i*f_length + i] = 3.0* (mode_vals[mode_i].amp[1] - mode_vals[mode_i].amp[0]);
          B[(num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phase[1] - mode_vals[mode_i].phase[0]);
      } else{
          B[mode_i*f_length + i] = 3.0* (mode_vals[mode_i].amp[i+1] - mode_vals[mode_i].amp[i-1]);
          B[(num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phase[i+1] - mode_vals[mode_i].phase[i-1]);
      }
}


#ifdef __CUDACC__
__host__
#endif
void cpu_fill_B_wave(ModeContainer *mode_vals, double *B, int f_length, int num_modes, int mode_i){
       for (int i = 0;
            i < f_length;
            i += 1){

            fill_B_wave_inner(mode_vals, B, f_length, num_modes, mode_i, i);
}
}


/*
find spline constants based on matrix solution for response transfer functions.
*/
#ifdef __CUDACC__
__host__ __device__
#endif
void set_spline_constants_response_inner(ModeContainer *mode_vals, double *B, int f_length, int num_modes, int mode_i, int i){
      double D_i, D_ip1, y_i, y_ip1;
      D_i = B[(0*num_modes*f_length) + mode_i*f_length + i];
      D_ip1 = B[(0*num_modes*f_length) + mode_i*f_length + i + 1];
      y_i = mode_vals[mode_i].phaseRdelay[i];
      y_ip1 = mode_vals[mode_i].phaseRdelay[i+1];
      mode_vals[mode_i].phaseRdelay_coeff_1[i] = D_i;
      mode_vals[mode_i].phaseRdelay_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
      mode_vals[mode_i].phaseRdelay_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

      D_i = B[(1*num_modes*f_length) + mode_i*f_length + i];
      D_ip1 = B[(1*num_modes*f_length) + mode_i*f_length + i + 1];
      y_i = mode_vals[mode_i].transferL1_re[i];
      y_ip1 = mode_vals[mode_i].transferL1_re[i+1];
      mode_vals[mode_i].transferL1_re_coeff_1[i] = D_i;
      mode_vals[mode_i].transferL1_re_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
      mode_vals[mode_i].transferL1_re_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

      D_i = B[(2*num_modes*f_length) + mode_i*f_length + i];
      D_ip1 = B[(2*num_modes*f_length) + mode_i*f_length + i + 1];
      y_i = mode_vals[mode_i].transferL1_im[i];
      y_ip1 = mode_vals[mode_i].transferL1_im[i+1];
      mode_vals[mode_i].transferL1_im_coeff_1[i] = D_i;
      mode_vals[mode_i].transferL1_im_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
      mode_vals[mode_i].transferL1_im_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

      D_i = B[(3*num_modes*f_length) + mode_i*f_length + i];
      D_ip1 = B[(3*num_modes*f_length) + mode_i*f_length + i + 1];
      y_i = mode_vals[mode_i].transferL2_re[i];
      y_ip1 = mode_vals[mode_i].transferL2_re[i+1];
      mode_vals[mode_i].transferL2_re_coeff_1[i] = D_i;
      mode_vals[mode_i].transferL2_re_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
      mode_vals[mode_i].transferL2_re_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

      D_i = B[(4*num_modes*f_length) + mode_i*f_length + i];
      D_ip1 = B[(4*num_modes*f_length) + mode_i*f_length + i + 1];
      y_i = mode_vals[mode_i].transferL2_im[i];
      y_ip1 = mode_vals[mode_i].transferL2_im[i+1];
      mode_vals[mode_i].transferL2_im_coeff_1[i] = D_i;
      mode_vals[mode_i].transferL2_im_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
      mode_vals[mode_i].transferL2_im_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

      D_i = B[(5*num_modes*f_length) + mode_i*f_length + i];
      D_ip1 = B[(5*num_modes*f_length) + mode_i*f_length + i + 1];
      y_i = mode_vals[mode_i].transferL3_re[i];
      y_ip1 = mode_vals[mode_i].transferL3_re[i+1];
      mode_vals[mode_i].transferL3_re_coeff_1[i] = D_i;
      mode_vals[mode_i].transferL3_re_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
      mode_vals[mode_i].transferL3_re_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

      D_i = B[(6*num_modes*f_length) + mode_i*f_length + i];
      D_ip1 = B[(6*num_modes*f_length) + mode_i*f_length + i + 1];
      y_i = mode_vals[mode_i].transferL3_im[i];
      y_ip1 = mode_vals[mode_i].transferL3_im[i+1];
      mode_vals[mode_i].transferL3_im_coeff_1[i] = D_i;
      mode_vals[mode_i].transferL3_im_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
      mode_vals[mode_i].transferL3_im_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

      D_i = B[(7*num_modes*f_length) + mode_i*f_length + i];
      D_ip1 = B[(7*num_modes*f_length) + mode_i*f_length + i + 1];
      y_i = mode_vals[mode_i].time_freq_corr[i];
      y_ip1 = mode_vals[mode_i].time_freq_corr[i+1];
      mode_vals[mode_i].time_freq_coeff_1[i] = D_i;
      mode_vals[mode_i].time_freq_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
      mode_vals[mode_i].time_freq_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;
}

#ifdef __CUDACC__
__host__
#endif
void cpu_set_spline_constants_response(ModeContainer *mode_vals, double *B, int f_length, int num_modes, int mode_i){

       for (int i = 0;
            i < f_length-1;
            i += 1){

              set_spline_constants_response_inner(mode_vals, B, f_length, num_modes, mode_i, i);

}
}

/*
Find spline coefficients after matrix calculation on GPU for amp and phase
*/

#ifdef __CUDACC__
__host__ __device__
#endif
void set_spline_constants_wave_inner(ModeContainer *mode_vals, double *B, int f_length, int num_modes, int mode_i, int i){
        double D_i, D_ip1, y_i, y_ip1;
        D_i = B[mode_i*f_length + i];
        D_ip1 = B[mode_i*f_length + i + 1];
        y_i = mode_vals[mode_i].amp[i];
        y_ip1 = mode_vals[mode_i].amp[i+1];
        mode_vals[mode_i].amp_coeff_1[i] = D_i;
        mode_vals[mode_i].amp_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
        mode_vals[mode_i].amp_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

        D_i = B[(num_modes*f_length) + mode_i*f_length + i];
        D_ip1 = B[(num_modes*f_length) + mode_i*f_length + i + 1];
        y_i = mode_vals[mode_i].phase[i];
        y_ip1 = mode_vals[mode_i].phase[i+1];
        mode_vals[mode_i].phase_coeff_1[i] = D_i;
        mode_vals[mode_i].phase_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
        mode_vals[mode_i].phase_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;
}


#ifdef __CUDACC__
__host__
#endif
void cpu_set_spline_constants_wave(ModeContainer *mode_vals, double *B, int f_length, int num_modes, int mode_i){
        for (int i = 0;
             i < f_length - 1;
             i += 1){
              set_spline_constants_wave_inner(mode_vals, B, f_length, num_modes, mode_i, i);
}
}

#ifdef __CUDACC__
__global__ void fill_B_wave(ModeContainer *mode_vals, double *B, int f_length, int num_modes){
    for (int mode_i = blockIdx.y * blockDim.y + threadIdx.y;
         mode_i < num_modes;
         mode_i += blockDim.y * gridDim.y){

       for (int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < f_length;
            i += blockDim.x * gridDim.x){

            fill_B_wave_inner(mode_vals, B, f_length, num_modes, mode_i, i);
}
}
}


__global__
void fill_B_response(ModeContainer *mode_vals, double *B, int f_length, int num_modes){
    for (int mode_i = blockIdx.y * blockDim.y + threadIdx.y;
         mode_i < num_modes;
         mode_i += blockDim.y * gridDim.y){

       for (int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < f_length;
            i += blockDim.x * gridDim.x){

          fill_B_response_inner(mode_vals, B, f_length, num_modes, mode_i, i);
}
}
}

__global__ void set_spline_constants_wave(ModeContainer *mode_vals, double *B, int f_length, int num_modes){

     for (int mode_i = blockIdx.y * blockDim.y + threadIdx.y;
          mode_i < num_modes;
          mode_i += blockDim.y * gridDim.y){

        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < f_length-1;
             i += blockDim.x * gridDim.x){
              set_spline_constants_wave_inner(mode_vals, B, f_length, num_modes, mode_i, i);
}
}
}


__global__
void set_spline_constants_response(ModeContainer *mode_vals, double *B, int f_length, int num_modes){
    for (int mode_i = blockIdx.y * blockDim.y + threadIdx.y;
         mode_i < num_modes;
         mode_i += blockDim.y * gridDim.y){

       for (int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < f_length-1;
            i += blockDim.x * gridDim.x){

              set_spline_constants_response_inner(mode_vals, B, f_length, num_modes, mode_i, i);

}
}
}


/*
Interpolate amp, phase, and response transfer functions on GPU.
*/
__global__
void interpolate(cmplx *channel1_out, cmplx *channel2_out, cmplx *channel3_out, ModeContainer* old_mode_vals,
    int num_modes, double d_log10f, double *old_freqs, int old_length, double *data_freqs, int data_length, double* t0_arr, double* tRef_arr, double *channel1_ASDinv,
    double *channel2_ASDinv, double *channel3_ASDinv, double t_obs_dur, int num_walkers){
    //int mode_i = blockIdx.y;

    double f, x, x2, x3, coeff_0, coeff_1, coeff_2, coeff_3;
    double time_start, amp, phase, phaseRdelay, f_min_limit, f_max_limit, t0, tRef, t_break;
    double transferL1_re, transferL1_im, transferL2_re, transferL2_im, transferL3_re, transferL3_im;
    cmplx ampphasefactor;
    cmplx I = cmplx(0.0, 1.0);
    int old_ind_below;
    cmplx trans_complex;
    for (int walker_i = blockIdx.z * blockDim.z + threadIdx.z;
         walker_i < num_walkers;
         walker_i += blockDim.z * gridDim.z){

     f_min_limit = old_freqs[walker_i*old_length];
     f_max_limit = old_freqs[walker_i*old_length + old_length-1];
     t0 = t0_arr[walker_i];
     tRef = tRef_arr[walker_i];
     t_break = t0*YRSID_SI + tRef - t_obs_dur*YRSID_SI; // t0 and t_obs_dur in years. tRef in seconds.

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < data_length;
         i += blockDim.x * gridDim.x){
    //if (mode_i >= num_modes) return;

        channel1_out[walker_i*data_length + i] = cmplx(0.0, 0.0);
        channel2_out[walker_i*data_length + i] = cmplx(0.0, 0.0);
        channel3_out[walker_i*data_length + i] = cmplx(0.0, 0.0);


    /*# if __CUDA_ARCH__>=200
    if (i == 200)
        printf("times: %e %e, %e, %e \n", t0, tRef, t_obs_dur, t_break);

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
    x = (f - old_freqs[walker_i*old_length + old_ind_below])/(old_freqs[walker_i*old_length + old_ind_below+1] - old_freqs[walker_i*old_length + old_ind_below]);
    x2 = x*x;
    x3 = x*x2;
    for (int mode_i=0; mode_i<num_modes; mode_i++){

            // interp time frequency to remove less than 0.0
            coeff_0 = old_mode_vals[walker_i*num_modes + mode_i].time_freq_corr[old_ind_below];
            coeff_1 = old_mode_vals[walker_i*num_modes + mode_i].time_freq_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[walker_i*num_modes + mode_i].time_freq_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[walker_i*num_modes + mode_i].time_freq_coeff_3[old_ind_below];

            time_start = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            if (time_start < t_break) {
                continue;
            }

            // interp amplitude
            coeff_0 = old_mode_vals[walker_i*num_modes + mode_i].amp[old_ind_below];
            coeff_1 = old_mode_vals[walker_i*num_modes + mode_i].amp_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[walker_i*num_modes + mode_i].amp_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[walker_i*num_modes + mode_i].amp_coeff_3[old_ind_below];

            amp = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            if (amp < 1e-40){
                continue;
            }
            // interp phase
            coeff_0 = old_mode_vals[walker_i*num_modes + mode_i].phase[old_ind_below];
            coeff_1 = old_mode_vals[walker_i*num_modes + mode_i].phase_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[walker_i*num_modes + mode_i].phase_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[walker_i*num_modes + mode_i].phase_coeff_3[old_ind_below];

            phase  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[walker_i*num_modes + mode_i].phaseRdelay[old_ind_below];
            coeff_1 = old_mode_vals[walker_i*num_modes + mode_i].phaseRdelay_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[walker_i*num_modes + mode_i].phaseRdelay_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[walker_i*num_modes + mode_i].phaseRdelay_coeff_3[old_ind_below];

            phaseRdelay  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);
            ampphasefactor = amp* exp(cmplx(0.0, phase + phaseRdelay));

            // X or A
            coeff_0 = old_mode_vals[walker_i*num_modes + mode_i].transferL1_re[old_ind_below];
            coeff_1 = old_mode_vals[walker_i*num_modes + mode_i].transferL1_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[walker_i*num_modes + mode_i].transferL1_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[walker_i*num_modes + mode_i].transferL1_re_coeff_3[old_ind_below];

            transferL1_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            /*# if __CUDA_ARCH__>=200
            if (i == 15000)
                printf("times: %e, %d, %d, %d, %d, %e, %e, %e, %e, %e, %e\n", f, mode_i, walker_i, old_ind_below, old_length, time_start, t_break, t0, tRef, amp, transferL1_re);

            #endif //*/

            coeff_0 = old_mode_vals[walker_i*num_modes + mode_i].transferL1_im[old_ind_below];
            coeff_1 = old_mode_vals[walker_i*num_modes + mode_i].transferL1_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[walker_i*num_modes + mode_i].transferL1_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[walker_i*num_modes + mode_i].transferL1_im_coeff_3[old_ind_below];

            transferL1_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            trans_complex = cmplx(transferL1_re, transferL1_im)* ampphasefactor * channel1_ASDinv[i]; //TODO may be faster to load as complex number with 0.0 for imaginary part

            channel1_out[walker_i*data_length + i] = channel1_out[walker_i*data_length + i] + trans_complex;
            // Y or E
            coeff_0 = old_mode_vals[walker_i*num_modes + mode_i].transferL2_re[old_ind_below];
            coeff_1 = old_mode_vals[walker_i*num_modes + mode_i].transferL2_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[walker_i*num_modes + mode_i].transferL2_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[walker_i*num_modes + mode_i].transferL2_re_coeff_3[old_ind_below];

            transferL2_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[walker_i*num_modes + mode_i].transferL2_im[old_ind_below];
            coeff_1 = old_mode_vals[walker_i*num_modes + mode_i].transferL2_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[walker_i*num_modes + mode_i].transferL2_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[walker_i*num_modes + mode_i].transferL2_im_coeff_3[old_ind_below];

            transferL2_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            trans_complex = cmplx(transferL2_re, transferL2_im)* ampphasefactor * channel2_ASDinv[i];

            channel2_out[walker_i*data_length + i] = channel2_out[walker_i*data_length + i] + trans_complex;

            // Z or T
            coeff_0 = old_mode_vals[walker_i*num_modes + mode_i].transferL3_re[old_ind_below];
            coeff_1 = old_mode_vals[walker_i*num_modes + mode_i].transferL3_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[walker_i*num_modes + mode_i].transferL3_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[walker_i*num_modes + mode_i].transferL3_re_coeff_3[old_ind_below];

            transferL3_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[walker_i*num_modes + mode_i].transferL3_im[old_ind_below];
            coeff_1 = old_mode_vals[walker_i*num_modes + mode_i].transferL3_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[walker_i*num_modes + mode_i].transferL3_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[walker_i*num_modes + mode_i].transferL3_im_coeff_3[old_ind_below];

            transferL3_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            trans_complex = cmplx(transferL3_re, transferL3_im)* ampphasefactor * channel3_ASDinv[i];

            // add to this channel
            channel3_out[walker_i*data_length + i] = channel3_out[walker_i*data_length + i] + trans_complex;
    }
}
}
}

#endif


#ifdef __CUDACC__
__host__
#endif
void cpu_interpolate(cmplx *channel1_out, cmplx *channel2_out, cmplx *channel3_out, ModeContainer* old_mode_vals,
    int num_modes, double d_log10f, double *old_freqs, int old_length, double *data_freqs, int data_length, double* t0_arr, double* tRef_arr, double *channel1_ASDinv,
    double *channel2_ASDinv, double *channel3_ASDinv, double t_obs_dur, int num_walkers, int walker_i){
    //int mode_i = blockIdx.y;

    double f, x, x2, x3, coeff_0, coeff_1, coeff_2, coeff_3;
    double time_start, amp, phase, phaseRdelay, f_min_limit, f_max_limit, t0, tRef, t_break;
    double transferL1_re, transferL1_im, transferL2_re, transferL2_im, transferL3_re, transferL3_im;
    cmplx ampphasefactor;
    cmplx I = cmplx(0.0, 1.0);
    int old_ind_below;
    cmplx trans_complex;
    double current_f_low, next_f_low;

     f_min_limit = old_freqs[walker_i*old_length];
     f_max_limit = old_freqs[walker_i*old_length + old_length-1];
     t0 = t0_arr[walker_i];
     tRef = tRef_arr[walker_i];
     t_break = t0*YRSID_SI + tRef - t_obs_dur*YRSID_SI; // t0 and t_obs_dur in years. tRef in seconds.


    int i_start = 0;
    while(data_freqs[i_start] < f_min_limit) i_start++;

    f = data_freqs[i_start];
    old_ind_below = floor((log10(f) - log10(old_freqs[walker_i*old_length + 0]))/d_log10f);
    current_f_low = old_freqs[walker_i*old_length + old_ind_below];
    next_f_low = old_freqs[walker_i*old_length + old_ind_below];
    x = (f - current_f_low)/(next_f_low - current_f_low);
    x2 = x*x;
    x3 = x*x2;

    int num_coeffs = 4*num_modes;
    double amp_coeffs[4*num_modes], phase_coeffs[4*num_modes], phaseRdelay_coeffs[4*num_modes], time_freq_corr_coeffs[4*num_modes];
    double transferL1_re_coeffs[4*num_modes], transferL1_im_coeffs[4*num_modes];
    double transferL2_re_coeffs[4*num_modes], transferL2_im_coeffs[4*num_modes];
    double transferL3_re_coeffs[4*num_modes], transferL3_im_coeffs[4*num_modes];

    for (int i = i_start;
         i < data_length;
         i += 1){
    //if (mode_i >= num_modes) return;

        channel1_out[walker_i*data_length + i] = cmplx(0.0, 0.0);
        channel2_out[walker_i*data_length + i] = cmplx(0.0, 0.0);
        channel3_out[walker_i*data_length + i] = cmplx(0.0, 0.0);

    f = data_freqs[i];
    if (f>=next_f_low){
        old_ind_below += 1;
        if ((old_ind_below == old_length -1) || (f >= f_max_limit) || (f < f_min_limit) || (old_ind_below >= old_length)){
            continue;
        }
        current_f_low = next_f_low;
        next_f_low = old_freqs[walker_i*old_length + old_ind_below + 1];

        // interp time frequency to remove less than 0.0
        for (int mode_i=0; mode_i<num_modes; mode_i++){

          time_freq_corr_coeffs[mode_i*4 + 0] = old_mode_vals[walker_i*num_modes + mode_i].time_freq_corr[old_ind_below];
          time_freq_corr_coeffs[mode_i*4 + 1] = old_mode_vals[walker_i*num_modes + mode_i].time_freq_coeff_1[old_ind_below];
          time_freq_corr_coeffs[mode_i*4 + 2] = old_mode_vals[walker_i*num_modes + mode_i].time_freq_coeff_2[old_ind_below];
          time_freq_corr_coeffs[mode_i*4 + 3] = old_mode_vals[walker_i*num_modes + mode_i].time_freq_coeff_3[old_ind_below];

          // interp amplitude
          amp_coeffs[mode_i*4 + 0] = old_mode_vals[walker_i*num_modes + mode_i].amp[old_ind_below];
          amp_coeffs[mode_i*4 + 1] = old_mode_vals[walker_i*num_modes + mode_i].amp_coeff_1[old_ind_below];
          amp_coeffs[mode_i*4 + 2] = old_mode_vals[walker_i*num_modes + mode_i].amp_coeff_2[old_ind_below];
          amp_coeffs[mode_i*4 + 3] = old_mode_vals[walker_i*num_modes + mode_i].amp_coeff_3[old_ind_below];

          // interp phase
          phase_coeffs[mode_i*4 + 0] = old_mode_vals[walker_i*num_modes + mode_i].phase[old_ind_below];
          phase_coeffs[mode_i*4 + 1] = old_mode_vals[walker_i*num_modes + mode_i].phase_coeff_1[old_ind_below];
          phase_coeffs[mode_i*4 + 2] = old_mode_vals[walker_i*num_modes + mode_i].phase_coeff_2[old_ind_below];
          phase_coeffs[mode_i*4 + 3] = old_mode_vals[walker_i*num_modes + mode_i].phase_coeff_3[old_ind_below];

          transferL1_re_coeffs[mode_i*4 + 0] = old_mode_vals[walker_i*num_modes + mode_i].transferL1_re[old_ind_below];
          transferL1_re_coeffs[mode_i*4 + 1] = old_mode_vals[walker_i*num_modes + mode_i].transferL1_re_coeff_1[old_ind_below];
          transferL1_re_coeffs[mode_i*4 + 2] = old_mode_vals[walker_i*num_modes + mode_i].transferL1_re_coeff_2[old_ind_below];
          transferL1_re_coeffs[mode_i*4 + 3] = old_mode_vals[walker_i*num_modes + mode_i].transferL1_re_coeff_3[old_ind_below];

          transferL1_im_coeffs[mode_i*4 + 0] = old_mode_vals[walker_i*num_modes + mode_i].transferL1_im[old_ind_below];
          transferL1_im_coeffs[mode_i*4 + 1] = old_mode_vals[walker_i*num_modes + mode_i].transferL1_im_coeff_1[old_ind_below];
          transferL1_im_coeffs[mode_i*4 + 2] = old_mode_vals[walker_i*num_modes + mode_i].transferL1_im_coeff_2[old_ind_below];
          transferL1_im_coeffs[mode_i*4 + 3] = old_mode_vals[walker_i*num_modes + mode_i].transferL1_im_coeff_3[old_ind_below];

          transferL2_re_coeffs[mode_i*4 + 0] = old_mode_vals[walker_i*num_modes + mode_i].transferL2_re[old_ind_below];
          transferL2_re_coeffs[mode_i*4 + 1] = old_mode_vals[walker_i*num_modes + mode_i].transferL2_re_coeff_1[old_ind_below];
          transferL2_re_coeffs[mode_i*4 + 2] = old_mode_vals[walker_i*num_modes + mode_i].transferL2_re_coeff_2[old_ind_below];
          transferL2_re_coeffs[mode_i*4 + 3] = old_mode_vals[walker_i*num_modes + mode_i].transferL2_re_coeff_3[old_ind_below];

          transferL2_im_coeffs[mode_i*4 + 0] = old_mode_vals[walker_i*num_modes + mode_i].transferL2_im[old_ind_below];
          transferL2_im_coeffs[mode_i*4 + 1] = old_mode_vals[walker_i*num_modes + mode_i].transferL2_im_coeff_1[old_ind_below];
          transferL2_im_coeffs[mode_i*4 + 2] = old_mode_vals[walker_i*num_modes + mode_i].transferL2_im_coeff_2[old_ind_below];
          transferL2_im_coeffs[mode_i*4 + 3] = old_mode_vals[walker_i*num_modes + mode_i].transferL2_im_coeff_3[old_ind_below];

          transferL3_re_coeffs[mode_i*4 + 0] = old_mode_vals[walker_i*num_modes + mode_i].transferL3_re[old_ind_below];
          transferL3_re_coeffs[mode_i*4 + 1] = old_mode_vals[walker_i*num_modes + mode_i].transferL3_re_coeff_1[old_ind_below];
          transferL3_re_coeffs[mode_i*4 + 2] = old_mode_vals[walker_i*num_modes + mode_i].transferL3_re_coeff_2[old_ind_below];
          transferL3_re_coeffs[mode_i*4 + 3] = old_mode_vals[walker_i*num_modes + mode_i].transferL3_re_coeff_3[old_ind_below];

          transferL3_im_coeffs[mode_i*4 + 0] = old_mode_vals[walker_i*num_modes + mode_i].transferL3_im[old_ind_below];
          transferL3_im_coeffs[mode_i*4 + 1] = old_mode_vals[walker_i*num_modes + mode_i].transferL3_im_coeff_1[old_ind_below];
          transferL3_im_coeffs[mode_i*4 + 2] = old_mode_vals[walker_i*num_modes + mode_i].transferL3_im_coeff_2[old_ind_below];
          transferL3_im_coeffs[mode_i*4 + 3] = old_mode_vals[walker_i*num_modes + mode_i].transferL3_im_coeff_3[old_ind_below];
        }
    }
    x = (f - current_f_low)/(next_f_low - current_f_low);
    x2 = x*x;
    x3 = x*x2;

    for (int mode_i=0; mode_i<num_modes; mode_i++){

            time_start = time_freq_corr_coeffs[mode_i*4 + 0]
                          + (time_freq_corr_coeffs[mode_i*4 + 1]*x)
                          + (time_freq_corr_coeffs[mode_i*4 + 2]*x2)
                          + (time_freq_corr_coeffs[mode_i*4 + 3]*x3);

            if (time_start < t_break) {
                continue;
            }

            amp = amp_coeffs[mode_i*4 + 0]
                          + (amp_coeffs[mode_i*4 + 1]*x)
                          + (amp_coeffs[mode_i*4 + 2]*x2)
                          + (amp_coeffs[mode_i*4 + 3]*x3);

            if (amp < 1e-40){
                continue;
            }

            phase  = phase_coeffs[mode_i*4 + 0]
                          + (phase_coeffs[mode_i*4 + 1]*x)
                          + (phase_coeffs[mode_i*4 + 2]*x2)
                          + (phase_coeffs[mode_i*4 + 3]*x3);

            phaseRdelay  = phaseRdelay_coeffs[mode_i*4 + 0]
                          + (phaseRdelay_coeffs[mode_i*4 + 1]*x)
                          + (phaseRdelay_coeffs[mode_i*4 + 2]*x2)
                          + (phaseRdelay_coeffs[mode_i*4 + 3]*x3);

            ampphasefactor = amp* exp(cmplx(0.0, phase + phaseRdelay));

            // X or A
            transferL1_re  = transferL1_re_coeffs[mode_i*4 + 0]
                          + (transferL1_re_coeffs[mode_i*4 + 1]*x)
                          + (transferL1_re_coeffs[mode_i*4 + 2]*x2)
                          + (transferL1_re_coeffs[mode_i*4 + 3]*x3);

            transferL1_im  = transferL1_im_coeffs[mode_i*4 + 0]
                          + (transferL1_im_coeffs[mode_i*4 + 1]*x)
                          + (transferL1_im_coeffs[mode_i*4 + 2]*x2)
                          + (transferL1_im_coeffs[mode_i*4 + 3]*x3);

            trans_complex = cmplx(transferL1_re, transferL1_im)* ampphasefactor * channel1_ASDinv[i]; //TODO may be faster to load as complex number with 0.0 for imaginary part

            channel1_out[walker_i*data_length + i] = channel1_out[walker_i*data_length + i] + trans_complex;
            // Y or E
            transferL2_re  = transferL2_re_coeffs[mode_i*4 + 0]
                          + (transferL2_re_coeffs[mode_i*4 + 1]*x)
                          + (transferL2_re_coeffs[mode_i*4 + 2]*x2)
                          + (transferL2_re_coeffs[mode_i*4 + 3]*x3);

            transferL2_im  = transferL2_im_coeffs[mode_i*4 + 0]
                          + (transferL2_im_coeffs[mode_i*4 + 1]*x)
                          + (transferL2_im_coeffs[mode_i*4 + 2]*x2)
                          + (transferL2_im_coeffs[mode_i*4 + 3]*x3);

            trans_complex = cmplx(transferL2_re, transferL2_im)* ampphasefactor * channel2_ASDinv[i];

            channel2_out[walker_i*data_length + i] = channel2_out[walker_i*data_length + i] + trans_complex;

            // Z or T
            transferL3_re  = transferL3_re_coeffs[mode_i*4 + 0]
                          + (transferL3_re_coeffs[mode_i*4 + 1]*x)
                          + (transferL3_re_coeffs[mode_i*4 + 2]*x2)
                          + (transferL3_re_coeffs[mode_i*4 + 3]*x3);

            transferL3_im  = transferL3_im_coeffs[mode_i*4 + 0]
                          + (transferL3_im_coeffs[mode_i*4 + 1]*x)
                          + (transferL3_im_coeffs[mode_i*4 + 2]*x2)
                          + (transferL3_im_coeffs[mode_i*4 + 3]*x3);

            trans_complex = cmplx(transferL3_re, transferL3_im)* ampphasefactor * channel3_ASDinv[i];

            // add to this channel
            channel3_out[walker_i*data_length + i] = channel3_out[walker_i*data_length + i] + trans_complex;
    }
}
}


/*
Interpolation class initializer
*/

Interpolate::Interpolate(){
    int pass = 0;
}

/*
allocate arrays for interpolation
*/

#ifdef __CUDACC__
__host__
#endif
void Interpolate::alloc_arrays(int m, int n, double *d_B){
    double *w = new double[m];
    double *a = new double[m];
    double *b = new double[m];
    double *c = new double[m];

    a[0] = 0.0;
    b[0] = 2.0;
    c[0] = 1.0;

    a[m-1] = 1.0;
    b[m-1] = 2.0;
    c[m-1] = 0.0;

    for (int i = 1;
         i < m-1;
         i += 1){
     a[i] = 1.0;
     b[i] = 4.0;
     c[i] = 1.0;
 }

 for (int i=1; i<m; i++){
     w[i] = a[i]/b[i-1];
     b[i] = b[i] - w[i]*c[i-1];
 }

    #ifdef __CUDACC__
    gpuErrchk_here(cudaMalloc(&d_b, m*sizeof(double)));
    gpuErrchk_here(cudaMemcpy(d_b, b, m*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk_here(cudaMalloc(&d_c, m*sizeof(double)));
    gpuErrchk_here(cudaMemcpy(d_c, c, m*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk_here(cudaMalloc(&d_w, m*sizeof(double)));
    gpuErrchk_here(cudaMemcpy(d_w, w, m*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk_here(cudaMalloc(&d_x, m*n*sizeof(double)));

    #else
    x = new double[m*n];

    #endif

    //CUSPARSE_CALL( cusparseCreate(&handle) );
    //cusparseDgtsv2_nopivot_bufferSizeExt(handle, m, n, d_dl, d_d, d_du, d_B, m, &bufferSizeInBytes);
    //cusparseDestroy(handle);
    //printf("buffer: %d\n", bufferSizeInBytes);

    //cudaMalloc(&pBuffer, bufferSizeInBytes);
}

/*
setup tridiagonal matrix for interpolation solution
*/
/*
__global__
void setup_d_vals(double *dl, double *d, double *du, int m, int n){
    for (int j = blockIdx.y * blockDim.y + threadIdx.y;
         j < n;
         j += blockDim.y * gridDim.y){

       for (int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < m;
            i += blockDim.x * gridDim.x){
    if (i == 0){
        dl[j*m + 0] = 0.0;
        d[j*m + 0] = 2.0;
        du[j*m + 0] = 1.0;
    } else if (i == m - 1){
        dl[j*m + m-1] = 1.0;
        d[j*m + m-1] = 2.0;
        du[j*m + m-1] = 0.0;
    } else{
        dl[j*m + i] = 1.0;
        d[j*m + i] = 4.0;
        du[j*m + i] = 1.0;
    }
}
}
}
*/
/*
solve matrix solution for tridiagonal matrix for cublic spline.
*/
void Interpolate::prep(double *B, int m_, int n_, int walker_i){
    m = m_;
    n = n_;

    int NUM_THREADS = 256;

    #ifdef __CUDACC__
    int num_blocks = ceil((n + NUM_THREADS -1)/NUM_THREADS);
    //setup_d_vals<<<dim3(num_blocks, n), NUM_THREADS>>>(d_dl, d_d, d_du, m, n);
    cudaDeviceSynchronize();
    gpuErrchk_here(cudaGetLastError());
    Interpolate::gpu_fit_constants(B);

    #else
    Interpolate::cpu_fit_constants(B);
    #endif
}

#ifdef __CUDACC__
__host__ __device__
#endif
void fit_constants_serial(int m, int n, double *w, double *b, double *c, double *d_in, double *x_in, int j){
      double *x, *d;

      d = &d_in[j*m];
      x = &x_in[j*m];

      # pragma unroll
      for (int i=2; i<m; i++){
          //printf("%d\n", i);
          d[i] = d[i] - w[i]*d[i-1];
          //printf("%lf, %lf, %lf\n", w[i], d[i], b[i]);
      }

      x[m-1] = d[m-1]/b[m-1];
      d[m-1] = x[m-1];
      # pragma unroll
      for (int i=(m-2); i>=0; i--){
          x[i] = (d[i] - c[i]*x[i+1])/b[i];
          d[i] = x[i];
      }
}

#ifdef __CUDACC__
__global__
void gpu_fit_constants_serial(int m, int n, double *w, double *b, double *c, double *d_in, double *x_in){

    for (int j = blockIdx.x * blockDim.x + threadIdx.x;
         j < n;
         j += blockDim.x * gridDim.x){
           fit_constants_serial(m, n, w, b, c, d_in, x_in, j);
    }
}
#endif

#ifdef __CUDACC__
__host__
#endif
void Interpolate::cpu_fit_constants(double *B){
  int i, j, th_id, nthreads, walker_i;
  #pragma omp parallel private(th_id, i, j)
  {
  //for (int i=0; i<ndevices*nwalkers; i++){
      nthreads = omp_get_num_threads();
      th_id = omp_get_thread_num();
      for (int j=th_id; j<m; j+=nthreads){
           fit_constants_serial(m, n, w, b, c, B, x, j);
      }
  }
}

/*
Use cuSparse to perform matrix calcuation.
*/
#ifdef __CUDACC__
__host__
void Interpolate::gpu_fit_constants(double *B){
    //CUSPARSE_CALL( cusparseCreate(&handle) );
    //cusparseStatus_t status = cusparseDgtsv2_nopivot(handle, m, n, d_dl, d_d, d_du, B, m, pBuffer);
    //if (status !=  CUSPARSE_STATUS_SUCCESS) assert(0);
    //cusparseDestroy(handle);
    int NUM_THREADS = 256;
    int num_blocks = ceil((n + NUM_THREADS -1)/NUM_THREADS);
    gpu_fit_constants_serial<<<num_blocks, NUM_THREADS>>>(m, n, d_w, d_b, d_c, B, d_x);
    cudaDeviceSynchronize();
    gpuErrchk_here(cudaGetLastError());

}
#endif


/*
Deallocate
*/
#ifdef __CUDACC__
__host__
#endif

Interpolate::~Interpolate(){
    #ifdef __CUDACC__
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(pBuffer);
    #else
    delete[] x;
    #endif

    delete[] w;
    delete[] a;
    delete[] b;
    delete[] c;
}
