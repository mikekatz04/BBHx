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
#include <cusparse_v2.h>
#include "globalPhenomHM.h"

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
using namespace std;

/*
fill the B array on the GPU for response transfer functions.
*/
__global__
void fill_B_response(ModeContainer *mode_vals, double *B, int f_length, int num_modes){
    int mode_i = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f_length) return;
    if (mode_i >= num_modes) return;

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

/*
fill B array on GPU for amp and phase
*/
__global__ void fill_B_wave(ModeContainer *mode_vals, double *B, int f_length, int num_modes){
    int mode_i = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f_length) return;
    if (mode_i >= num_modes) return;
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


/*
find spline constants based on matrix solution for response transfer functions.
*/
__global__
void set_spline_constants_response(ModeContainer *mode_vals, double *B, int f_length, int num_modes){
    double D_i, D_ip1, y_i, y_ip1;
    int mode_i = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f_length) return;
    if (mode_i >= num_modes) return;

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

/*
Find spline coefficients after matrix calculation on GPU for amp and phase
*/

__global__ void set_spline_constants_wave(ModeContainer *mode_vals, double *B, int f_length, int num_modes){
    int mode_i = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f_length-1) return;
    if (mode_i >= num_modes) return;
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

/*
Auxillary functino for complex exponential on GPU
*/
__device__
cuDoubleComplex d_complex_exp (cuDoubleComplex arg)
{
   cuDoubleComplex res;
   double s, c;
   double e = exp(arg.x);
   sincos(arg.y, &s, &c);
   res.x = c * e;
   res.y = s * e;
   return res;
}

/*
Interpolate amp, phase, and response transfer functions on GPU.
*/
__global__
void interpolate(cuDoubleComplex *channel1_out, cuDoubleComplex *channel2_out, cuDoubleComplex *channel3_out, ModeContainer* old_mode_vals, int num_modes, double d_log10f, double *old_freqs, int old_length, double *data_freqs, int data_length, double t0, double tRef, double *channel1_ASDinv, double *channel2_ASDinv, double *channel3_ASDinv, double t_obs_start, double t_obs_end){
    //int mode_i = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= data_length) return;
    //if (mode_i >= num_modes) return;
    double f, x, x2, x3, coeff_0, coeff_1, coeff_2, coeff_3;
    double time, amp, phase, phaseRdelay;
    double transferL1_re, transferL1_im, transferL2_re, transferL2_im, transferL3_re, transferL3_im;
    double f_min_limit = old_freqs[0];
    double f_max_limit = old_freqs[old_length-1];
    cuDoubleComplex ampphasefactor;
    cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);
    int old_ind_below;
    cuDoubleComplex trans_complex;
    channel1_out[i] = make_cuDoubleComplex(0.0, 0.0);
    channel2_out[i] = make_cuDoubleComplex(0.0, 0.0);
    channel3_out[i] = make_cuDoubleComplex(0.0, 0.0);
    /*for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < data_length;
         i += blockDim.x * gridDim.x)
      {*/

    for (int mode_i=0; mode_i<num_modes; mode_i++){
            f = data_freqs[i];
            old_ind_below = floor((log10(f) - log10(old_freqs[0]))/d_log10f);
            if ((old_ind_below == old_length -1) || (f >= f_max_limit) || (f < f_min_limit)){
                return;
            }
            x = (f - old_freqs[old_ind_below])/(old_freqs[old_ind_below+1] - old_freqs[old_ind_below]);
            x2 = x*x;
            x3 = x*x2;
            // interp time frequency to remove less than 0.0
            coeff_0 = old_mode_vals[mode_i].time_freq_corr[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].time_freq_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].time_freq_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].time_freq_coeff_3[old_ind_below];

            time = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);
            /*# if __CUDA_ARCH__>=200
            if (i < 200)
                printf("times %d: %e, %e, %e \n", i, time, t_obs_start*YRSID_SI, t_obs_end*YRSID_SI);

            #endif //*/
            if ((time < t_obs_start*YRSID_SI) || (time > t_obs_end*YRSID_SI)) {
                continue;
            }

            // interp amplitude
            coeff_0 = old_mode_vals[mode_i].amp[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].amp_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].amp_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].amp_coeff_3[old_ind_below];

            amp = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);
            if (amp < 1e-40){
                continue;
            }
            // interp phase
            coeff_0 = old_mode_vals[mode_i].phase[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].phase_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].phase_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].phase_coeff_3[old_ind_below];

            phase  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[mode_i].phaseRdelay[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].phaseRdelay_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].phaseRdelay_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].phaseRdelay_coeff_3[old_ind_below];

            phaseRdelay  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);
            ampphasefactor = cuCmul(make_cuDoubleComplex(amp,0.0), d_complex_exp(make_cuDoubleComplex(0.0, phase + phaseRdelay)));

            // X or A
            coeff_0 = old_mode_vals[mode_i].transferL1_re[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL1_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL1_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL1_re_coeff_3[old_ind_below];

            transferL1_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[mode_i].transferL1_im[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL1_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL1_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL1_im_coeff_3[old_ind_below];

            transferL1_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            trans_complex = cuCmul(cuCmul(make_cuDoubleComplex(transferL1_re, transferL1_im), ampphasefactor), make_cuDoubleComplex(channel1_ASDinv[i], 0.0)); //TODO may be faster to load as complex number with 0.0 for imaginary part

            channel1_out[i] = cuCadd(channel1_out[i], trans_complex);
            // Y or E
            coeff_0 = old_mode_vals[mode_i].transferL2_re[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL2_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL2_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL2_re_coeff_3[old_ind_below];

            transferL2_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[mode_i].transferL2_im[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL2_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL2_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL2_im_coeff_3[old_ind_below];

            transferL2_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            trans_complex = cuCmul(cuCmul(make_cuDoubleComplex(transferL2_re, transferL2_im), ampphasefactor), make_cuDoubleComplex(channel2_ASDinv[i], 0.0));

            channel2_out[i] = cuCadd(channel2_out[i], trans_complex);

            // Z or T
            coeff_0 = old_mode_vals[mode_i].transferL3_re[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL3_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL3_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL3_re_coeff_3[old_ind_below];

            transferL3_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[mode_i].transferL3_im[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL3_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL3_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL3_im_coeff_3[old_ind_below];

            transferL3_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            trans_complex = cuCmul(cuCmul(make_cuDoubleComplex(transferL3_re, transferL3_im), ampphasefactor), make_cuDoubleComplex(channel3_ASDinv[i], 0.0));

            // add to this channel
            channel3_out[i] = cuCadd(channel3_out[i], trans_complex);
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

__host__
void Interpolate::alloc_arrays(int max_length_init){
    err = cudaMalloc(&d_dl, max_length_init*sizeof(double));
    assert(err == 0);
    err = cudaMalloc(&d_d, max_length_init*sizeof(double));
    assert(err == 0);
    err = cudaMalloc(&d_du, max_length_init*sizeof(double));
    assert(err == 0);
}

/*
setup tridiagonal matrix for interpolation solution
*/
__global__
void setup_d_vals(double *dl, double *d, double *du, int current_length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= current_length) return;
    if (i == 0){
        dl[0] = 0.0;
        d[0] = 2.0;
        du[0] = 1.0;
    } else if (i == current_length - 1){
        dl[current_length-1] = 1.0;
        d[current_length-1] = 2.0;
        du[current_length-1] = 0.0;
    } else{
        dl[i] = 1.0;
        d[i] = 4.0;
        du[i] = 1.0;
    }
}

/*
solve matrix solution for tridiagonal matrix for cublic spline.
*/
void Interpolate::prep(double *B, int m_, int n_, int to_gpu_){
    m = m_;
    n = n_;
    to_gpu = to_gpu_;
    int NUM_THREADS = 256;

    int num_blocks = std::ceil((m + NUM_THREADS -1)/NUM_THREADS);
    setup_d_vals<<<num_blocks, NUM_THREADS>>>(d_dl, d_d, d_du, m);
    cudaDeviceSynchronize();
    gpuErrchk_here(cudaGetLastError());
    Interpolate::gpu_fit_constants(B);
}

/*
Use cuSparse to perform matrix calcuation.
*/
__host__ void Interpolate::gpu_fit_constants(double *B){
    CUSPARSE_CALL( cusparseCreate(&handle) );
    cusparseStatus_t status = cusparseDgtsv_nopivot(handle, m, n, d_dl, d_d, d_du, B, m);
    if (status !=  CUSPARSE_STATUS_SUCCESS) assert(0);
    cusparseDestroy(handle);
}

/*
Deallocate
*/
__host__ Interpolate::~Interpolate(){
    cudaFree(d_dl);
    cudaFree(d_du);
    cudaFree(d_d);
}
