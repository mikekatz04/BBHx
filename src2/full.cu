/*  This code was edited by Michael Katz. It is originally from the LAL library.
 *  The original copyright and license is shown below. Michael Katz has edited
 *  the code for his purposes and removed dependencies on the LAL libraries. The code has been confirmed to match the LAL version.
 *  This code is distrbuted under the same GNU license it originally came with.
 *  The comments in the code have been left generally the same. A few comments
 *  have been made for the newer functions added.


 *  Copyright (C) 2017 Sebastian Khan, Francesco Pannarale, Lionel London
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
#include <math.h>
#include <complex>
#include <iostream>
#include "stdio.h"
#include <random>

#include "cuComplex.h"
#include "cublas_v2.h"

#include <stdbool.h>
#include "full.h"

#include "cusparse_v2.h"

#include "constants.h"
#include "global.h"
// #include "PhenomHM.hh"

#define  NUM_THREADS 256
#define  NUM_THREADS2 64
#define  NUM_THREADS3 256
#define  NUM_THREADS4 256




__device__
void prep_splines(int i, int length, int interp_i, int ninterps, int num_intermediates, double *b, double *ud, double *diag, double *ld, double *x, double *y, int numBinAll, int param, int nsub, int sub_i){
  double dx1, dx2, d, slope1, slope2;
  int ind0x, ind1x, ind2x, ind0y, ind1y, ind2y, ind_out;

  double xval0, xval1, xval2, yval1;

  int numFreqarrs = int(ninterps / num_intermediates);
  int freqArr_i = int(interp_i / num_intermediates);

  //if ((threadIdx.x == 10) && (blockIdx.x == 1)) printf("numFreqarrs %d %d %d %d %d\n", ninterps, interp_i, num_intermediates, numFreqarrs, freqArr_i);
  if (i == length - 1){
    ind0y = (param * nsub + sub_i) * length + (length - 3);
    ind1y = (param * nsub + sub_i) * length + (length - 2);
    ind2y = (param * nsub + sub_i) * length + (length - 1);

    ind0x = freqArr_i * length + (length - 3);
    ind1x = freqArr_i * length + (length - 2);
    ind2x = freqArr_i * length + (length - 1);

    ind_out = (param * nsub + sub_i) * length + (length - 1);

    xval0 = x[ind0x];
    xval1 = x[ind1x];
    xval2 = x[ind2x];

    dx1 = xval1 - xval0;
    dx2 = xval2 - xval1;
    d = xval2 - xval0;

    yval1 = y[ind1y];

    slope1 = (yval1 - y[ind0y])/dx1;
    slope2 = (y[ind2y] - yval1)/dx2;

    b[ind_out] = ((dx2*dx2*slope1 +
                             (2*d + dx2)*dx1*slope2) / d);
    diag[ind_out] = dx1;
    ld[ind_out] = d;
    ud[ind_out] = 0.0;

  } else if (i == 0){

      ind0y = (param * nsub + sub_i) * length + 0;
      ind1y = (param * nsub + sub_i) * length + 1;
      ind2y = (param * nsub + sub_i) * length + 2;

      ind0x = freqArr_i * length + 0;
      ind1x = freqArr_i * length + 1;
      ind2x = freqArr_i * length + 2;

      ind_out = (param * nsub + sub_i) * length + 0;

      xval0 = x[ind0x];
      xval1 = x[ind1x];
      xval2 = x[ind2x];


      dx1 = xval1 - xval0;
      dx2 = xval2 - xval1;
      d = xval2 - xval0;

      yval1 = y[ind1y];

      //amp
      slope1 = (yval1 - y[ind0y])/dx1;
      slope2 = (y[ind2y] - yval1)/dx2;

      b[ind_out] = ((dx1 + 2*d) * dx2 * slope1 +
                          dx1*dx1 * slope2) / d;
    ud[ind_out] = d;
    ld[ind_out] = 0.0;
      diag[ind_out] = dx2;

  } else{

      ind0y = (param * nsub + sub_i) * length + (i - 1);
      ind1y = (param * nsub + sub_i) * length + (i + 0);
      ind2y = (param * nsub + sub_i) * length + (i + 1);

      ind0x = freqArr_i * length + (i - 1);
      ind1x = freqArr_i * length + (i - 0);
      ind2x = freqArr_i * length + (i + 1);

      ind_out = (param * nsub + sub_i) * length + i;

      xval0 = x[ind0x];
      xval1 = x[ind1x];
      xval2 = x[ind2x];

    dx1 = xval1 - xval0;
    dx2 = xval2 - xval1;

    yval1 = y[ind1y];

    //amp
    slope1 = (yval1 - y[ind0y])/dx1;
    slope2 = (y[ind2y] - yval1)/dx2;

    b[ind_out] = 3.0* (dx2*slope1 + dx1*slope2);
    diag[ind_out] = 2*(dx1 + dx2);
    ud[ind_out] = dx1;
    ld[ind_out] = dx2;
  }

}



CUDA_KERNEL
void fill_B(double *freqs_arr, double *y_all, double *B, double *upper_diag, double *diag, double *lower_diag,
                      int ninterps, int length, int num_intermediates, int numModes, int numBinAll){

    int param = 0;
    int nsub = 0;
    int sub_i = 0;
    #ifdef __CUDACC__

    int start1 = blockIdx.x;
    int end1 = ninterps;
    int diff1 = gridDim.x;

    #else

    int start1 = 0;
    int end1 = ninterps;
    int diff1 = 1;

    #endif
    for (int interp_i = start1;
         interp_i<end1; // 2 for re and im
         interp_i+= diff1){

         #ifdef __CUDACC__

         int start2 = threadIdx.x;
         int end2 = length;
         int diff2 = blockDim.x;

         #else

         int start2 = 0;
         int end2 = length;
         int diff2 = 1;

         #endif

        param = int((double) interp_i/(numModes * numBinAll));
        nsub = numModes * numBinAll;
        sub_i = interp_i % (numModes * numBinAll);

       for (int i = start2;
            i < end2;
            i += diff2){

            int lead_ind = interp_i*length;
            prep_splines(i, length, interp_i, ninterps, num_intermediates, B, upper_diag, diag, lower_diag, freqs_arr, y_all, numBinAll, param, nsub, sub_i);

}
}
}

/*
CuSparse error checking
*/
#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
                             fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
                             exit(-1);}} while(0)

#define CUSPARSE_CALL(X) ERR_NE((X),CUSPARSE_STATUS_SUCCESS)

void interpolate_kern(int m, int n, double *a, double *b, double *c, double *d_in)
{
        #ifdef __CUDACC__
        size_t bufferSizeInBytes;

        cusparseHandle_t handle;
        void *pBuffer;

        CUSPARSE_CALL(cusparseCreate(&handle));
        CUSPARSE_CALL( cusparseDgtsv2StridedBatch_bufferSizeExt(handle, m, a, b, c, d_in, n, m, &bufferSizeInBytes));
        gpuErrchk(cudaMalloc(&pBuffer, bufferSizeInBytes));

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
      gpuErrchk(cudaFree(pBuffer));

      #else

    #ifdef __USE_OMP__
    #pragma omp parallel for
    #endif
    for (int j = 0;
         j < n;
         j += 1){
           //fit_constants_serial(m, n, w, a, b, c, d_in, x_in, j);
           int info = LAPACKE_dgtsv(LAPACK_COL_MAJOR, m, 1, &a[j*m + 1], &b[j*m], &c[j*m], &d_in[j*m], m);
           //if (info != m) printf("lapack info check: %d\n", info);

       }

      #endif


    /*
    int interp_i = threadIdx.x + blockDim.x * blockIdx.x;

    int param = (int) (interp_i / (numModes * numBinAll));
    int nsub = numBinAll * numModes;
    int sub_i = interp_i % (numModes * numBinAll);

    int ind_i, ind_im1, ind_ip1;
    if (interp_i < ninterps)
    {

        double w = 0.0;
        for (int i = 1; i < n; i += 1)
        {
            ind_i = (param * n + i) * nsub + sub_i;
            ind_im1 = (param * n + (i-1)) * nsub + sub_i;


            w = a[ind_i]/b[ind_im1];
            b[ind_i] = b[ind_i] - w * c[ind_im1];
            d[ind_i] = d[ind_i] - w * d[ind_im1];
        }

        ind_i = (param * n + (n-1)) * nsub + sub_i;

        d[ind_i] = d[ind_i]/b[ind_i];
        for (int i = n - 2; i >= 0; i -= 1)
        {
            ind_i = (param * n + i) * nsub + sub_i;
            ind_ip1 = (param * n + (i+1)) * nsub + sub_i;

            d[ind_i] = (d[ind_i] - c[ind_i] * d[ind_ip1])/b[ind_i];

        }
    }
    */
}


CUDA_CALLABLE_MEMBER
void fill_coefficients(int i, int length, int sub_i, int nsub, double *dydx, double dx, double *y, double *coeff1, double *coeff2, double *coeff3, int param){
  double slope, t, dydx_i;

  int ind_i = (param * nsub + sub_i) * length + i;
  int ind_ip1 = (param * nsub + sub_i) * length + (i + 1);

  slope = (y[ind_ip1] - y[ind_i])/dx;

  dydx_i = dydx[ind_i];

  t = (dydx_i + dydx[ind_ip1] - 2*slope)/dx;

  coeff1[ind_i] = dydx_i;
  coeff2[ind_i] = (slope - dydx_i) / dx - t;
  coeff3[ind_i] = t/dx;

  //if ((param == 1) && (i == length - 3) && (sub_i == 0)) printf("freq check: %d %d %d %d %d\n", i, dydx[ind_i], dydx[ind_ip1]);


}

CUDA_KERNEL
void set_spline_constants(double *f_arr, double* y, double *c1, double* c2, double* c3, double *B,
                      int ninterps, int length, int num_intermediates, int numBinAll, int numModes){

    double df;
    #ifdef __CUDACC__
    int start1 = blockIdx.x;
    int end1 = ninterps;
    int diff1 = gridDim.x;
    #else

    int start1 = 0;
    int end1 = ninterps;
    int diff1 = 1;

    #endif

    for (int interp_i= start1;
         interp_i<end1; // 2 for re and im
         interp_i+= diff1){

     int numFreqarrs = int(ninterps / num_intermediates);
     int freqArr_i = int(interp_i / num_intermediates);

     int param = (int) (interp_i / (numModes * numBinAll));
     int nsub = numBinAll * numModes;
     int sub_i = interp_i % (numModes * numBinAll);

     #ifdef __CUDACC__
     int start2 = threadIdx.x;
     int end2 = length - 1;
     int diff2 = blockDim.x;
     #else

     int start2 = 0;
     int end2 = length - 1;
     int diff2 = 1;

     #endif
     for (int i = start2;
            i < end2;
            i += diff2){

                // TODO: check if there is faster way to do this
              df = f_arr[freqArr_i * length + (i + 1)] - f_arr[freqArr_i * length + i];

              int lead_ind = interp_i*length;
              fill_coefficients(i, length, sub_i, nsub, B, df,
                                y,
                                c1,
                                c2,
                                c3, param);

}
}
}


__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long* address_as_ull =
                              (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__device__ void atomicAddComplex(cmplx* a, cmplx b){
  //transform the addresses of real and imag. parts to double pointers
  double *x = (double*)a;
  double *y = x+1;
  //use atomicAdd for double variables
  atomicAddDouble(x, b.real());
  atomicAddDouble(y, b.imag());
}



#define  DATA_BLOCK 128
#define  NUM_INTERPS 9

__device__
cmplx get_ampphasefactor(double amp, double phase, double phaseShift){
    return amp*gcmplx::exp(cmplx(0.0, phase + phaseShift));
}

__device__
cmplx combine_information(cmplx* channel1, cmplx* channel2, cmplx* channel3, double amp, double phase, double tf, cmplx transferL1, cmplx transferL2, cmplx transferL3, double t_start, double t_end)
{
    // TODO: make sure the end of the ringdown is included
    if ((tf >= t_start) && ((tf <= t_end) || (t_end <= 0.0)) && (amp > 1e-40))
    {
        cmplx amp_phase_term = amp*gcmplx::exp(cmplx(0.0, -phase));  // add phase shift

        *channel1 = gcmplx::conj(transferL1 * amp_phase_term);
        *channel2 = gcmplx::conj(transferL2 * amp_phase_term);
        *channel3 = gcmplx::conj(transferL3 * amp_phase_term);

    }
}

#define  NUM_TERMS 4

#define  MAX_NUM_COEFF_TERMS 1200

CUDA_KERNEL
void TDI(cmplx* templateChannels, double* dataFreqsIn, double dlog10f, double* freqsOld, double* propArrays, double* c1In, double* c2In, double* c3In, double t_mrg, int old_length, int data_length, int numBinAll, int numModes, double t_obs_start, double t_obs_end, int* inds, int ind_start, int ind_length, int bin_i)
{

    int num_params = 9;
    //int mode_i = blockIdx.y;

    int numAll = numBinAll * numModes * old_length;

    double tempLike, addLike, time_check, phaseShift;
    cmplx trans_complex1, trans_complex2, trans_complex3, ampphasefactor;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= ind_length) return;

    double f = dataFreqsIn[i + ind_start];

    int ind_here = inds[i];

    double f_old = freqsOld[ind_here];

    double x = f - f_old;
    double x2 = x * x;
    double x3 = x * x2;

    trans_complex1 = 0.0; trans_complex2 = 0.0; trans_complex3 = 0.0;

    for (int mode_i = 0; mode_i < numModes; mode_i += 1)
    {
        int int_shared = ((0 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
        double amp = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

        //if ((i == 100) || (i == 101)) printf("%d %d %d %e %e %e %e %e %e\n", window_i, mode_i, i, amp, f, f_old, y[int_shared], c1[int_shared], c2[int_shared]);

        int_shared = ((1 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
        double phase = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

        int_shared = ((2 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
        double tf = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

        int_shared = ((3 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
        double transferL1_re = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

        int_shared = ((4 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
        double transferL1_im = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

        int_shared = ((5 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
        double transferL2_re = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

        int_shared = ((6 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
        double transferL2_im = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

        int_shared = ((7 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
        double transferL3_re = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

        int_shared = ((8 * numBinAll + bin_i) * numModes + mode_i) * old_length + ind_here;
        double transferL3_im = propArrays[int_shared] + c1In[int_shared] * x + c2In[int_shared] * x2 + c3In[int_shared] * x3;

        cmplx channel1(0.0, 0.0);
        cmplx channel2(0.0, 0.0);
        cmplx channel3(0.0, 0.0);

        combine_information(&channel1, &channel2, &channel3, amp, phase, tf, cmplx(transferL1_re, transferL1_im), cmplx(transferL2_re, transferL2_im), cmplx(transferL3_re, transferL3_im), t_obs_start, t_obs_end);

        trans_complex1 += channel1;
        trans_complex2 += channel2;
        trans_complex3 += channel3;
    }

    atomicAddComplex(&templateChannels[0 * ind_length + i], trans_complex1);
    atomicAddComplex(&templateChannels[1 * ind_length + i], trans_complex2);
    atomicAddComplex(&templateChannels[2 * ind_length + i], trans_complex3);


    /*
    __shared__ double y[MAX_NUM_COEFF_TERMS];
    __shared__ double c1[MAX_NUM_COEFF_TERMS];
    __shared__ double c2[MAX_NUM_COEFF_TERMS];
    __shared__ double c3[MAX_NUM_COEFF_TERMS];
    __shared__ double freqs_shared[MAX_NUM_COEFF_TERMS];

    int num_params = 9;
    int mode_i = blockIdx.y;

    int numAll = numBinAll * numModes * old_length;

    double amp, phase, tfCorr, transferL1_re, transferL1_im, transferL2_re, transferL2_im, transferL3_re, transferL3_im;
    double x, x2, x3, tempLike, addLike, time_check, phaseShift;
    cmplx trans_complex1, trans_complex2, trans_complex3, ampphasefactor;

    __shared__ int start_ind, end_ind;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    bool run = true;
    if (i >= ind_length) run = false;

    __syncthreads();

    if (threadIdx.x == 0)
    {
        start_ind = inds[i];
    }
    int max_thread_num = (ind_length - blockDim.x*blockIdx.x > NUM_THREADS3) ? NUM_THREADS3 : ind_length - blockDim.x*blockIdx.x;

    if (threadIdx.x == max_thread_num - 1)
    {
        end_ind = inds[i];
    }

    __syncthreads();

    int num_windows = end_ind - start_ind + 1;

    int nsub = numModes * numBinAll;

    //if (run) printf("%d %d %d %d %d %d %d %d\n", max_thread_num, threadIdx.x, blockDim.x, NUM_THREADS3, i, ind_length, start_ind, end_ind);

    for (int j = threadIdx.x; j < num_windows; j += blockDim.x)
    {
        int window_i = j;

        int old_ind = start_ind + window_i;

        if ((old_ind < 0) || (old_ind >= old_length))
        {
            continue;
        }

        freqs_shared[window_i] = freqsOld[old_ind];

        //if ((blockIdx.x == 0) && (blockIdx.y == 0)) printf("%d %d %e %e\n", old_ind, window_i, freqs_shared[window_i], freqsOld[old_ind]);
    }

    __syncthreads();

    for (int j = threadIdx.x; j < num_params * num_windows; j += blockDim.x)
    {
        int window_i = j % num_windows;
        int param_i = (int) (j / num_windows);

        int old_ind = start_ind + window_i;

        if ((old_ind < 0) || (old_ind >= old_length))
        {
            continue;
        }

        int ind = ((param_i * numBinAll + bin_i) * numModes + mode_i) * old_length + old_ind;
        int ind_shared = window_i * num_params + param_i;

        y[ind_shared] = propArrays[ind];
        c1[ind_shared] = c1In[ind];
        c2[ind_shared] = c2In[ind];
        c3[ind_shared] = c3In[ind];

        if (ind_shared > MAX_NUM_COEFF_TERMS) printf("BAD %d %d\n", ind_shared, window_i);
    }

    __syncthreads();

    if (run)
    {
        double f = dataFreqsIn[i + ind_start];

        int ind_here = inds[i];

        int window_i = ind_here - start_ind;

        double f_old = freqs_shared[window_i];

        double x = f - f_old;
        double x2 = x * x;
        double x3 = x * x2;

        int int_shared = window_i * num_params + 0;
        double amp = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        //if ((i == 100) || (i == 101)) printf("%d %d %d %e %e %e %e %e %e\n", window_i, mode_i, i, amp, f, f_old, y[int_shared], c1[int_shared], c2[int_shared]);

        int_shared = window_i * num_params + 1;
        double phase = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        int_shared = window_i * num_params + 2;
        double tf = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        int_shared = window_i * num_params + 3;
        double transferL1_re = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        int_shared = window_i * num_params + 4;
        double transferL1_im = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        int_shared = window_i * num_params + 5;
        double transferL2_re = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        int_shared = window_i * num_params + 6;
        double transferL2_im = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        int_shared = window_i * num_params + 7;
        double transferL3_re = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        int_shared = window_i * num_params + 8;
        double transferL3_im = y[int_shared] + c1[int_shared] * x + c2[int_shared] * x2 + c3[int_shared] * x3;

        cmplx channel1(0.0, 0.0);
        cmplx channel2(0.0, 0.0);
        cmplx channel3(0.0, 0.0);

        combine_information(&channel1, &channel2, &channel3, amp, phase, tf, cmplx(transferL1_re, transferL1_im), cmplx(transferL2_re, transferL2_im), cmplx(transferL3_re, transferL3_im), t_obs_start, t_obs_end);

        atomicAddComplex(&templateChannels[0 * ind_length + i], channel1);
        atomicAddComplex(&templateChannels[1 * ind_length + i], channel2);
        atomicAddComplex(&templateChannels[2 * ind_length + i], channel3);

    }
    */
}

CUDA_KERNEL
void fill_waveform(cmplx* templateChannels,
                double* bbh_buffer,
                int numBinAll, int data_length, int nChannels, int numModes, double* t_start, double* t_end)
{

    cmplx I(0.0, 1.0);

    cmplx temp_channel1 = 0.0, temp_channel2 = 0.0, temp_channel3 = 0.0;
    for (int bin_i = blockIdx.x; bin_i < numBinAll; bin_i += gridDim.x)
    {

        double t_start_bin = t_start[bin_i];
        double t_end_bin = t_end[bin_i];

        for (int i = threadIdx.x; i < data_length; i += blockDim.x)
        {
            cmplx temp_channel1 = 0.0, temp_channel2 = 0.0, temp_channel3 = 0.0;
            for (int mode_i = 0; mode_i < numModes; mode_i += 1)
            {

                int ind = ((0 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double amp = bbh_buffer[ind];

                ind = ((1 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double phase = bbh_buffer[ind];

                ind = ((2 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double tf = bbh_buffer[ind];

                ind = ((3 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double transferL1_re = bbh_buffer[ind];

                ind = ((4 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double transferL1_im = bbh_buffer[ind];

                ind = ((5 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double transferL2_re = bbh_buffer[ind];

                ind = ((6 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double transferL2_im = bbh_buffer[ind];

                ind = ((7 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double transferL3_re = bbh_buffer[ind];

                ind = ((8 * numBinAll + bin_i) * numModes + mode_i) * data_length + i;
                double transferL3_im = bbh_buffer[ind];

                cmplx channel1 = 0.0 + 0.0 * I;
                cmplx channel2 = 0.0 + 0.0 * I;
                cmplx channel3 = 0.0 + 0.0 * I;

                combine_information(&channel1, &channel2, &channel3, amp, phase, tf, cmplx(transferL1_re, transferL1_im), cmplx(transferL2_re, transferL2_im), cmplx(transferL3_re, transferL3_im), t_start_bin, t_end_bin);

                temp_channel1 += channel1;
                temp_channel2 += channel2;
                temp_channel3 += channel3;
            }

            templateChannels[(bin_i * 3 + 0) * data_length + i] = temp_channel1;
            templateChannels[(bin_i * 3 + 1) * data_length + i] = temp_channel2;
            templateChannels[(bin_i * 3 + 2) * data_length + i] = temp_channel3;

        }
    }
}

void direct_sum(cmplx* templateChannels,
                double* bbh_buffer,
                int numBinAll, int data_length, int nChannels, int numModes, double* t_start, double* t_end)
{

    int nblocks5 = numBinAll; // std::ceil((numBinAll + NUM_THREADS4 -1)/NUM_THREADS4);

    fill_waveform<<<nblocks5, NUM_THREADS4>>>(templateChannels, bbh_buffer, numBinAll, data_length, nChannels, numModes, t_start, t_end);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}


void InterpTDI(long* templateChannels_ptrs, double* dataFreqs, double dlog10f, double* freqs, double* propArrays, double* c1, double* c2, double* c3, double* t_mrg_in, double* t_start_in, double* t_end_in, int length, int data_length, int numBinAll, int numModes, double t_obs_start, double t_obs_end, long* inds_ptrs, int* inds_start, int* ind_lengths)
{

    cudaStream_t streams[numBinAll];

    #pragma omp parallel for
    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        int length_bin_i = ind_lengths[bin_i];
        int ind_start = inds_start[bin_i];
        int* inds = (int*) inds_ptrs[bin_i];

        double t_mrg = t_mrg_in[bin_i];
        double t_start = t_start_in[bin_i];
        double t_end = t_end_in[bin_i];

        cmplx* templateChannels = (cmplx*) templateChannels_ptrs[bin_i];

        int nblocks3 = std::ceil((length_bin_i + NUM_THREADS3 -1)/NUM_THREADS3);
        cudaStreamCreate(&streams[bin_i]);

        dim3 gridDim(nblocks3, 1);
        TDI<<<gridDim, NUM_THREADS3, 0, streams[bin_i]>>>(templateChannels, dataFreqs, dlog10f, freqs, propArrays, c1, c2, c3, t_mrg, length, data_length, numBinAll, numModes, t_start, t_end, inds, ind_start, length_bin_i, bin_i);

    }

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    #pragma omp parallel for
    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        //destroy the streams
        cudaStreamDestroy(streams[bin_i]);
    }
}

#define  DATA_BLOCK2 512
CUDA_KERNEL
void hdynLikelihood(cmplx* likeOut1, cmplx* likeOut2,
                    cmplx* templateChannels, cmplx* dataConstants,
                    double* dataFreqsIn,
                    int numBinAll, int data_length, int nChannels)
{
    __shared__ cmplx A0temp[DATA_BLOCK2];
    __shared__ cmplx A1temp[DATA_BLOCK2];
    __shared__ cmplx B0temp[DATA_BLOCK2];
    __shared__ cmplx B1temp[DATA_BLOCK2];
    __shared__ double dataFreqs[DATA_BLOCK2];

    cmplx A0, A1, B0, B1;

    cmplx trans_complex(0.0, 0.0);
    cmplx prev_trans_complex(0.0, 0.0);
    double prevFreq = 0.0;
    double freq = 0.0;

    int currentStart = 0;

    cmplx r0, r1, r1Conj, tempLike1, tempLike2;
    double mag_r0, midFreq;

    int binNum = threadIdx.x + blockDim.x * blockIdx.x;

    if (true) // for (int binNum = threadIdx.x + blockDim.x * blockIdx.x; binNum < numBinAll; binNum += blockDim.x * gridDim.x)
    {
        tempLike1 = 0.0;
        tempLike2 = 0.0;
        for (int channel = 0; channel < nChannels; channel += 1)
        {
            prevFreq = 0.0;
            currentStart = 0;
            while (currentStart < data_length)
            {
                __syncthreads();
                for (int jj = threadIdx.x; jj < DATA_BLOCK2; jj += blockDim.x)
                {
                    if ((jj + currentStart) >= data_length) continue;
                    A0temp[jj] = dataConstants[(0 * nChannels + channel) * data_length + currentStart + jj];
                    A1temp[jj] = dataConstants[(1 * nChannels + channel) * data_length + currentStart + jj];
                    B0temp[jj] = dataConstants[(2 * nChannels + channel) * data_length + currentStart + jj];
                    B1temp[jj] = dataConstants[(3 * nChannels + channel) * data_length + currentStart + jj];

                    dataFreqs[jj] = dataFreqsIn[currentStart + jj];

                    //if ((jj + currentStart < 3) && (binNum == 0) & (channel == 0))
                    //    printf("check %e %e, %e %e, %e %e, %e %e, %e \n", A0temp[jj], A1temp[jj], B0temp[jj], B1temp[jj], dataFreqs[jj]);

                }
                __syncthreads();
                if (binNum < numBinAll)
                {
                    for (int jj = 0; jj < DATA_BLOCK2; jj += 1)
                    {
                        if ((jj + currentStart) >= data_length) continue;
                        freq = dataFreqs[jj];
                        trans_complex = templateChannels[((jj + currentStart) * nChannels + channel) * numBinAll + binNum];

                        if ((prevFreq != 0.0) && (jj + currentStart > 0))
                        {
                            A0 = A0temp[jj]; // constants will need to be aligned with 1..n-1 because there are data_length - 1 bins
                            A1 = A1temp[jj];
                            B0 = B0temp[jj];
                            B1 = B1temp[jj];

                            r1 = (trans_complex - prev_trans_complex)/(freq - prevFreq);
                            midFreq = (freq + prevFreq)/2.0;

                            r0 = trans_complex - r1 * (freq - midFreq);

                            //if (((binNum == 767) || (binNum == 768)) & (channel == 0))
                            //    printf("CHECK2: %d %d %d %e %e\n", jj + currentStart, binNum, jj, A0); // , %e %e, %e %e, %e %e, %e %e,  %e %e,  %e %e , %e\n", ind, binNum, jj + currentStart, A0, A1, B0, B1, freq, prevFreq, trans_complex, prev_trans_complex, midFreq);

                            r1Conj = gcmplx::conj(r1);

                            tempLike1 += A0 * gcmplx::conj(r0) + A1 * r1Conj;

                            mag_r0 = gcmplx::abs(r0);
                            tempLike2 += B0 * (mag_r0 * mag_r0) + 2. * B1 * gcmplx::real(r0 * r1Conj);
                        }

                        prev_trans_complex = trans_complex;
                        prevFreq = freq;
                    }
                }
                currentStart += DATA_BLOCK2;
            }
        }
        if (binNum < numBinAll)
        {
            likeOut1[binNum] = tempLike1;
            likeOut2[binNum] = tempLike2;
        }
    }
}




void interpolate(double* freqs, double* propArrays,
                 double* B, double* upper_diag, double* diag, double* lower_diag,
                 int length, int numInterpParams, int numModes, int numBinAll)
{

    int num_intermediates = numModes * numInterpParams;
    int ninterps = numModes * numInterpParams * numBinAll;

    int nblocks = std::ceil((ninterps + NUM_THREADS -1)/NUM_THREADS);

    double* c1 = upper_diag; //&interp_array[0 * numInterpParams * amp_phase_size];
    double* c2 = diag; //&interp_array[1 * numInterpParams * amp_phase_size];
    double* c3 = lower_diag; //&interp_array[2 * numInterpParams * amp_phase_size];

    //printf("%d after response, %d\n", jj, nblocks2);

     fill_B<<<nblocks, NUM_THREADS>>>(freqs, propArrays, B, upper_diag, diag, lower_diag, ninterps, length, num_intermediates, numModes, numBinAll);
     cudaDeviceSynchronize();
     gpuErrchk(cudaGetLastError());

     //printf("%d after fill b\n", jj);
     interpolate_kern(length, ninterps, lower_diag, diag, upper_diag, B);


  set_spline_constants<<<nblocks, NUM_THREADS>>>(freqs, propArrays, c1, c2, c3, B,
                    ninterps, length, num_intermediates, numBinAll, numModes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    //printf("%d after set spline\n", jj);
}

void hdyn(cmplx* likeOut1, cmplx* likeOut2,
                    cmplx* templateChannels, cmplx* dataConstants,
                    double* dataFreqs,
                    int numBinAll, int data_length, int nChannels)
{

    int nblocks4 = std::ceil((numBinAll + NUM_THREADS4 -1)/NUM_THREADS4);

    hdynLikelihood<<<nblocks4, NUM_THREADS4>>>(likeOut1, likeOut2, templateChannels, dataConstants, dataFreqs, numBinAll, data_length, nChannels);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

CUDA_KERNEL
void noiseweight_template(cmplx* templateChannels, double* noise_weight_times_df, int ind_start, int length, int data_stream_length)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < length; i += gridDim.x * blockDim.x)
    {
        for (int j = 0; j < 3; j+= 1)
        {
            templateChannels[j * length + i] = templateChannels[j * length + i] * noise_weight_times_df[j * data_stream_length + ind_start + i];
        }
    }
}

#define NUM_THREADS_LIKE 256

void direct_like(double* d_h, double* h_h, cmplx* dataChannels, double* noise_weight_times_df, long* templateChannels_ptrs, int* inds_start, int* ind_lengths, int data_stream_length, int numBinAll)
{

    cudaStream_t streams[numBinAll];
    cublasHandle_t handle;

    cuDoubleComplex result_d_h[numBinAll];
    cuDoubleComplex result_h_h[numBinAll];

    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
      printf ("CUBLAS initialization failed\n");
      exit(0);
    }

    #pragma omp parallel for
    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        int length_bin_i = ind_lengths[bin_i];
        int ind_start = inds_start[bin_i];

        cmplx* templateChannels = (cmplx*) templateChannels_ptrs[bin_i];

        int nblocks = std::ceil((length_bin_i + NUM_THREADS_LIKE -1)/NUM_THREADS_LIKE);
        cudaStreamCreate(&streams[bin_i]);

        noiseweight_template<<<nblocks, NUM_THREADS_LIKE, 0, streams[bin_i]>>>(templateChannels, noise_weight_times_df, ind_start, length_bin_i, data_stream_length);
        cudaStreamSynchronize(streams[bin_i]);

        for (int j = 0; j < 3; j += 1)
        {

            cublasSetStream(handle, streams[bin_i]);
            stat = cublasZdotc(handle, length_bin_i,
                              (cuDoubleComplex*)&dataChannels[j * data_stream_length + ind_start], 1,
                              (cuDoubleComplex*)&templateChannels[j * length_bin_i], 1,
                              &result_d_h[bin_i]);
            cudaStreamSynchronize(streams[bin_i]);
            if (stat != CUBLAS_STATUS_SUCCESS)
            {
                exit(0);
            }

            d_h[bin_i] += 4.0 * cuCreal(result_d_h[bin_i]);

            cublasSetStream(handle, streams[bin_i]);
            stat = cublasZdotc(handle, length_bin_i,
                              (cuDoubleComplex*)&templateChannels[j * length_bin_i], 1,
                              (cuDoubleComplex*)&templateChannels[j * length_bin_i], 1,
                              &result_h_h[bin_i]);
            cudaStreamSynchronize(streams[bin_i]);
            if (stat != CUBLAS_STATUS_SUCCESS)
            {
                exit(0);
            }
            h_h[bin_i] += 4.0 * cuCreal(result_h_h[bin_i]);

        }
    }

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    #pragma omp parallel for
    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        //destroy the streams
        cudaStreamDestroy(streams[bin_i]);
    }
    cublasDestroy(handle);
}
/*
int main()
{

    int TDItag = 1;
    int order_fresnel_stencil = 0;
    double tBase = 1.0;

    int numBinAll = 5000;
    int numModes = 6;
    int length = 1024;
    int data_length = 4096;

    int *ells_in, *mms_in;

    gpuErrchk(cudaMallocManaged(&ells_in, numModes * sizeof(int)));
    gpuErrchk(cudaMallocManaged(&mms_in, numModes * sizeof(int)));

    ells_in[0] = 2;
    ells_in[1] = 3;
    ells_in[2] = 4;

    ells_in[3] = 2;
    ells_in[4] = 3;
    ells_in[5] = 4;

    mms_in[0] = 2;
    mms_in[1] = 3;
    mms_in[2] = 4;

    mms_in[3] = 1;
    mms_in[4] = 2;
    mms_in[5] = 3;

    double *amps, *phases, *phases_deriv, *freqs, *m1_SI, *m2_SI, *chi1z, *chi2z, *distance, *phiRef, *fRef;
    double *inc, *lam, *beta, *psi, *tRef_wave_frame, *tRef_sampling_frame;
    double *response_out;
    double *B, *interp_array; // plays roll of upper lower diag, and then coefficients 1, 2, 3

    size_t amp_phase_size = numBinAll * numModes * length *sizeof(double);
    size_t freqs_size = numBinAll * length * sizeof(double);
    size_t bin_size = numBinAll * sizeof(double);

    int numInterpParams = 9;

    gpuErrchk(cudaMallocManaged(&amps, numInterpParams * amp_phase_size));

    response_out = &amps[1 * numBinAll * numModes * length];

    double *upper_diag, *diag, *lower_diag;
    gpuErrchk(cudaMallocManaged(&B, numInterpParams * amp_phase_size));
    gpuErrchk(cudaMallocManaged(&upper_diag, numInterpParams * amp_phase_size));
    gpuErrchk(cudaMallocManaged(&diag, numInterpParams * amp_phase_size));
    gpuErrchk(cudaMallocManaged(&lower_diag, numInterpParams * amp_phase_size));

    //double* upper_diag = &interp_array[0 * numInterpParams * amp_phase_size];
    //double* diag = &interp_array[1 * numInterpParams * amp_phase_size];
    //double* lower_diag = &interp_array[2 * numInterpParams * amp_phase_size];

    double* propArrays = amps;

    gpuErrchk(cudaMallocManaged(&freqs, freqs_size));

    gpuErrchk(cudaMallocManaged(&m1_SI, bin_size));
    gpuErrchk(cudaMallocManaged(&m2_SI, bin_size));
    gpuErrchk(cudaMallocManaged(&chi1z, bin_size));
    gpuErrchk(cudaMallocManaged(&chi2z, bin_size));
    gpuErrchk(cudaMallocManaged(&distance, bin_size));
    gpuErrchk(cudaMallocManaged(&phiRef, bin_size));
    gpuErrchk(cudaMallocManaged(&fRef, bin_size));

    gpuErrchk(cudaMallocManaged(&inc, bin_size));
    gpuErrchk(cudaMallocManaged(&lam, bin_size));
    gpuErrchk(cudaMallocManaged(&beta, bin_size));
    gpuErrchk(cudaMallocManaged(&psi, bin_size));
    gpuErrchk(cudaMallocManaged(&tRef_wave_frame, bin_size));
    gpuErrchk(cudaMallocManaged(&tRef_sampling_frame, bin_size));

    double m1 = 2e6; // solar
    double m2 = 1e6;
    double a1 = 0.8;
    double a2 = 0.8;
    double dist = 30.0; // Gpc
    double phi_ref = 0.0;
    double f_ref = 0.0;
    double inc_in = PI/3.;
    double lam_in = 0.4;
    double beta_in = 0.24;
    double psi_in = 1.0;
    double tRef_wave_frame_in = 10.0;
    double tRef_sampling_frame_in = 50.0;

    double Msec = (m1 + m2) * MTSUN_SI;

    double log10f_start = log10(1e-4/Msec);
    double log10f_end = log10(0.6/Msec);

    double dlog10f = (log10f_end - log10f_start)/(length - 1);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    for (int bin_i = 0; bin_i < numBinAll; bin_i += 1)
    {
        m1_SI[bin_i] = (1e6 * MSUN_SI) * (1 + distribution(generator));
        m2_SI[bin_i] = (4e5 * MSUN_SI) * (1 + distribution(generator));

        chi1z[bin_i] = (distribution(generator))* 0.9;
        chi2z[bin_i] = (distribution(generator))* 0.9;

        distance[bin_i] = (35) * (1 + distribution(generator)) * 1e9 * PC_SI;
        phiRef[bin_i] = (1 + distribution(generator));
        fRef[bin_i] = f_ref;

        inc[bin_i] = (distribution(generator)) * 0.25 + 0.25;
        lam[bin_i] = (distribution(generator)) * 0.25 + 0.25;
        beta[bin_i] = (distribution(generator)) * 0.25 + 0.25;
        psi[bin_i] = (distribution(generator)) * 0.25 + 0.25;
        tRef_wave_frame[bin_i] = (1 + distribution(generator)) * 20.0;
        tRef_sampling_frame[bin_i] = (1 + distribution(generator)) * 20.0;

        for (int i = 0; i < length; i += 1)
        {
            freqs[i * numBinAll + bin_i] = pow(10.0, log10f_start + i * dlog10f);
        }
    }

    cmplx *dataChannels, *templateChannels, *dataConstants;
    double *dataFreqs;
    int nChannels = 3;

    double t_obs_start = 1.0;
    double t_obs_end = 0.0;

    gpuErrchk(cudaMallocManaged(&dataChannels, nChannels * data_length * sizeof(cmplx)));
    gpuErrchk(cudaMallocManaged(&dataConstants, NUM_TERMS * nChannels * data_length * sizeof(cmplx)));
    gpuErrchk(cudaMallocManaged(&templateChannels, numBinAll * nChannels * data_length * sizeof(cmplx)));
    gpuErrchk(cudaMallocManaged(&dataFreqs, data_length * sizeof(double)));

    double dlog10fData = (log10f_end - log10f_start)/(data_length - 1);

    for (int i = 0; i < data_length; i += 1)
    {
        dataFreqs[i] = pow(10.0, log10f_start + i * dlog10fData);

        for (int channel = 0; channel < nChannels; channel += 1)
        {
            dataChannels[channel * data_length + i] = cmplx(1.0, 1.0);

            for (int constant = 0; constant < NUM_TERMS; constant += 1)
            {
                dataConstants[(constant * nChannels + channel) * data_length + i] = cmplx(1.0, 1.0);
            }
        }
    }

    cmplx *likeOut1;
    gpuErrchk(cudaMallocManaged(&likeOut1, numBinAll * sizeof(cmplx)));

    cmplx *likeOut2;
    gpuErrchk(cudaMallocManaged(&likeOut2, numBinAll * sizeof(cmplx)));

    double *c1, *c2, *c3;
    int numIter = 10;

    for (int jj = 0; jj < numIter; jj += 1)
    {

        //printf("%d begin\n", jj);
        waveform_amp_phase(
        amps, ///**< [out] Frequency-domain waveform hx
        ells_in,
        mms_in,
        freqs,               ///**< Frequency points at which to evaluate the waveform (Hz)
        m1_SI,                       // /**< mass of companion 1 (kg)
        m2_SI,                        ///**< mass of companion 2 (kg)
        chi1z,                        ///**< z-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1)
        chi2z,                        ///**< z-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1)
        distance,               ///**< distance of source (m)
        phiRef,                 ///**< reference orbital phase (rad)
        fRef,                      //  /**< Reference frequency
        numModes,
        length,
        numBinAll
   );

   int includesAmps = 0;
   LISA_response(
       response_out,
       ells_in,
       mms_in,
       freqs,               ///**< Frequency points at which to evaluate the waveform (Hz)
       phiRef,                // /**< reference orbital phase (rad)
       fRef,                    //    /**< Reference frequency
       inc,
       lam,
       beta,
       psi,
       tRef_wave_frame,
       tRef_sampling_frame,
       tBase, TDItag, order_fresnel_stencil,
       numModes,
       length,
       numBinAll,
       includesAmps
  );

  interpolate(freqs, propArrays,
                   B, upper_diag, diag, lower_diag,
                 length, numInterpParams, numModes, numBinAll);

    //printf("%d middle\n", jj);

    c1 = upper_diag; //&interp_array[0 * numInterpParams * amp_phase_size];
    c2 = diag; //&interp_array[1 * numInterpParams * amp_phase_size];
    c3 = lower_diag; //&interp_array[2 * numInterpParams * amp_phase_size];


    InterpTDI(templateChannels, dataChannels, dataFreqs, freqs, propArrays, c1, c2, c3, tBase, tRef_sampling_frame, tRef_wave_frame, length, data_length,   numBinAll, numModes, t_obs_start, t_obs_end);

    hdyn(likeOut1, likeOut2, templateChannels, dataConstants, dataFreqs, numBinAll, data_length, nChannels);
    }

    int binNum = 1000;
    int mode_i = 0;
    for (int i = 0; i < 5; i += 1) printf("%d %e %e\n", i, c1[(i * numModes + 0) * numBinAll + 0], c2[(i * numModes + 0) * numBinAll + 0]);

    return 0;
}

*/

/*
__device__
void fill_coefficients(int i, int length, int mode_i, int numModes, int interp_i, int ninterps, double *dydx, double dx, double *y, double *coeff1, double *coeff2, double *coeff3){
  double slope, t, dydx_i;

  int indip1 = ((i + 1) * numModes + mode_i) * ninterps + interp_i;
  int indi = ((i) * numModes + mode_i) * ninterps + interp_i;

  slope = (y[indip1] - y[indi])/dx;

  dydx_i = dydx[indi];

  t = (dydx_i + dydx[indip1] - 2*slope)/dx;

  coeff1[indi] = dydx_i;
  coeff2[indi] = (slope - dydx_i) / dx - t;
  coeff3[indi] = t/dx;
}




__device__
void prep_splines(int i, int length, int mode_i, int numModes, int interp_i, int ninterps,  double *b, double *ud, double *diag, double *ld, double *x, double *y){
  double dx1, dx2, d, slope1, slope2;
  int ind1x, ind2x, ind3x, ind1y, ind2y, ind3y;
  if (i == length - 1){

     ind1x = (length - 2) * ninterps + interp_i;
     ind2x = (length - 3) * ninterps + interp_i;
     ind3x = (length - 1) * ninterps + interp_i;

     ind1y = ((length - 2) * numModes + mode_i) * ninterps + interp_i;
     ind2y = ((length - 3) * numModes + mode_i) * ninterps + interp_i;
     ind3y = ((length - 1) * numModes + mode_i) * ninterps + interp_i;


  } else if (i == 0){

      ind1x = 1 * ninterps + interp_i;
      ind2x = 0 * ninterps + interp_i;
      ind3x = 2 * ninterps + interp_i;

      ind1y = (1 * numModes + mode_i) * ninterps + interp_i;
      ind2y = (0 * numModes + mode_i) * ninterps + interp_i;
      ind3y = (2 * numModes + mode_i) * ninterps + interp_i;


  } else{

      ind1x = (i) * ninterps + interp_i;
      ind2x = (i-1) * ninterps + interp_i;
      ind3x = (i+1) * ninterps + interp_i;

      ind1y = ((i) * numModes + mode_i) * ninterps + interp_i;
      ind2y = ((i-1) * numModes + mode_i) * ninterps + interp_i;
      ind3y = ((i+1) * numModes + mode_i) * ninterps + interp_i;
  }

    dx1 = x[ind1x] - x[ind2x];
    dx2 = x[ind3x] - x[ind1x];

    //amp
    slope1 = (y[ind1y] - y[ind2y])/dx1;
    slope2 = (y[ind3y] - y[ind1y])/dx2;

    b[ind1y] = 3.0* (dx2*slope1 + dx1*slope2);
    diag[ind1y] = 2*(dx1 + dx2);
    ud[ind1y] = dx1;
    ld[ind1y] = dx2;
}



CUDA_KERNEL
void fill_B(double *x_arr, double *y_all, double *B, double *upper_diag, double *diag, double *lower_diag,
                      int ninterps, int length, int numModes){


    int start1 = blockIdx.x*blockDim.x + threadIdx.x;
    int end1 = ninterps;
    int diff1 = blockDim.x*gridDim.x;

    for (int interp_i= start1;
         interp_i<end1; // 2 for re and im
         interp_i+= diff1)
        {

       for (int mode_i = 0; mode_i < numModes; mode_i += 1)
       {
           for (int i = start2;
                i < end2;
                i += diff2)
                {
                    prep_splines(i, length, mode_i, numModes, interp_i, ninterps,  B, upper_diag, diag, lower_diag, x_arr, y_all);

                }
       }

    }
}



CUDA_KERNEL
void set_spline_constants(double *x_arr, double *interp_array, double *B,
                      int ninterps, int length, int numModes){

    double dx;
    InterpContainer mode_vals;

    int start1 = blockIdx.x*blockDim.x + threadIdx.x;
    int end1 = ninterps;
    int diff1 = blockDim.x*gridDim.x;

    int npts = ninterps * length * numModes;

    for (int interp_i= start1;
         interp_i<end1; // 2 for re and im
         interp_i+= diff1){

             for (int mode_i = 0; mode_i < numModes; mode_i += 1)
             {
                 for (int i = start2;
                      i < end2;
                      i += diff2)
                      {
                          dx = x_arr[i + 1] - x_arr[i];

                          int lead_ind = interp_i*length;
                          fill_coefficients(i, length, mode_i, numModes, interp_i, ninterps, B, dx,
                                            &interp_array[0 * npts],
                                            &interp_array[1 * npts],
                                            &interp_array[2 * npts],
                                            &interp_array[3 * npts]);

                      }
             }
}



void fit_wrap(int m, int n, double *a, double *b, double *c, double *d_in){

    #ifdef __CUDACC__
    size_t bufferSizeInBytes;

    cusparseHandle_t handle;
    void *pBuffer;

    CUSPARSE_CALL(cusparseCreate(&handle));
    CUSPARSE_CALL( cusparseDgtsv2StridedBatch_bufferSizeExt(handle, m, a, b, c, d_in, n, m, &bufferSizeInBytes));
    gpuErrchk(cudaMalloc(&pBuffer, bufferSizeInBytes));

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
  gpuErrchk(cudaFree(pBuffer));

  #else

#ifdef __USE_OMP__
#pragma omp parallel for
#endif
for (int j = 0;
     j < n;
     j += 1){
       //fit_constants_serial(m, n, w, a, b, c, d_in, x_in, j);
       int info = LAPACKE_dgtsv(LAPACK_COL_MAJOR, m, 1, &a[j*m + 1], &b[j*m], &c[j*m], &d_in[j*m], m);
       //if (info != m) printf("lapack info check: %d\n", info);

   }

  #endif

}
*/
