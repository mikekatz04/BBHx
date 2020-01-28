/*  This code was created by Michael Katz.
 *  It is shared under the GNU license (see below).
 *  This code computes the interpolations for the GPU PhenomHM waveform.
 *  This is implemented on the CPU to mirror the GPU program.
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

#ifndef __INTERPOLATE_H_
#define __INTERPOLATE_H_


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

#ifdef __CUDACC__
CUDA_KERNEL
void fill_B_response_wrap(ModeContainer *mode_vals, double *freqs, double *B, double *upper_diag, double *diag, double *lower_diag, int f_length, int num_modes, int nwalkers);
#else
void cpu_fill_B_response_wrap(ModeContainer *mode_vals, double *freqs, double *B, double *upper_diag, double *diag, double *lower_diag, int f_length, int num_modes, int nwalkers);
#endif


#ifdef __CUDACC__
CUDA_KERNEL void fill_B_wave_wrap(ModeContainer *mode_vals, double *freqs, double *B, double *upper_diag, double *diag, double *lower_diag, int f_length, int num_modes, int nwalkers);
#else
void cpu_fill_B_wave_wrap(ModeContainer *mode_vals, double *freqs, double *B, double *upper_diag, double *diag, double *lower_diag, int f_length, int num_modes, int nwalkers);
#endif


#ifdef __CUDACC__
CUDA_KERNEL
void set_spline_constants_response_wrap(ModeContainer *mode_vals, double *B, int f_length, int nwalkers, int num_modes, double *freqs);
#else
void cpu_set_spline_constants_response_wrap(ModeContainer *mode_vals, double *B, int f_length, int nwalkers, int num_modes, double *freqs);
#endif

#ifdef __CUDACC__
CUDA_KERNEL void set_spline_constants_wave_wrap(ModeContainer *mode_vals, double *B, int f_length, int nwalkers, int num_modes, double *freqs);
#else
void cpu_set_spline_constants_wave_wrap(ModeContainer *mode_vals, double *B, int f_length, int nwalkers, int num_modes, double *freqs);
#endif

#ifdef __CUDACC__
CUDA_KERNEL
void interpolate_wrap(agcmplx *channel1_out, agcmplx *channel2_out, agcmplx *channel3_out, ModeContainer* old_mode_vals,
    int num_modes, double d_log10f, double *old_freqs, int old_length, double *data_freqs, int data_length, double* t0_arr, double* tRef_arr, double* tRef_wave_frame_arr, double *channel1_ASDinv,
    double *channel2_ASDinv, double *channel3_ASDinv, double t_obs_start, double t_obs_end, int num_walkers);
#else
void cpu_interpolate_wrap(agcmplx *channel1_out, agcmplx *channel2_out, agcmplx *channel3_out, ModeContainer* old_mode_vals,
    int num_modes, double d_log10f, double *old_freqs, int old_length, double *data_freqs, int data_length, double* t0_arr, double* tRef_arr, double* tRef_wave_frame_arr, double *channel1_ASDinv,
    double *channel2_ASDinv, double *channel3_ASDinv, double t_obs_start, double t_obs_end, int num_walkers);
#endif

#ifdef __GLOBAL_FIT__
#ifdef __CUDACC__
__device__
#endif // __CUDACC__
#else
CUDA_CALLABLE_MEMBER
#endif // __GLOBAL_FIT__
agcmplx combine_information(agcmplx ampphasefactor, double trans_complex_re, double trans_complex_im);

#ifdef __GLOBAL_FIT__
#ifdef __CUDACC__
__device__
#endif // __CUDACC__
#else
CUDA_CALLABLE_MEMBER
#endif // __GLOBAL_FIT__
agcmplx get_ampphasefactor(double amp, double phase, double phaseRdelay, double phaseShift);

#ifdef __GLOBAL_FIT__
#ifdef __CUDACC__
__device__
#endif // __CUDACC__
#else
CUDA_CALLABLE_MEMBER
#endif // __GLOBAL_FIT__
void fill_templates(agcmplx *channel1_out, agcmplx *channel2_out, agcmplx *channel3_out, int i, int walker_i, int data_length,
                   agcmplx trans_complex1, agcmplx trans_complex2, agcmplx trans_complex3,
                   double *channel1_ASDinv, double *channel2_ASDinv, double *channel3_ASDinv);

class Interpolate{
    double *w;
    double *D;

    double *x;

    double *d_w;
    double *d_x;

    int m;
    int n;
    int to_gpu;

public:
    // FOR NOW WE ASSUME dLOGX is evenly spaced // TODO: allocate at the beginning
    Interpolate();

    void alloc_arrays(int m, int n, double *d_B);
   void prep(double *B, double *c, double *b, double *a, int m_, int n_, int to_gpu_);

   ~Interpolate(); //destructor

};

#endif //__INTERPOLATE_H_
