#include "global.h"
#include "constants.h"
#include "Interpolate.hh"

#ifdef __CUDACC__
#include "cusparse_v2.h"
#else
#include "lapacke.h"
#endif

#define  NUM_THREADS_INTERPOLATE 256

 // See scipy CubicSpline implementation, it matches that
CUDA_CALLABLE_MEMBER
void prep_splines(int i, int length, int interp_i, int ninterps, int num_intermediates, double *b, double *ud, double *diag, double *ld, double *x, double *y, int numBinAll, int param, int nsub, int sub_i){
  double dx1, dx2, d, slope1, slope2;
  int ind0x, ind1x, ind2x, ind0y, ind1y, ind2y, ind_out;

  double xval0, xval1, xval2, yval1;

  // get proper frequency array since it is given once for all modes
  int freqArr_i = int(sub_i / num_intermediates);

  // fill values in spline initial computations
  // get indices into the 1D arrays
  // compute necessary quantities
  // fill the diagonals
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

 // See scipy CubicSpline implementation, it matches that
 // this is for solving the banded matrix equation
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

    // use lapack on CPU
    #ifdef __USE_OMP__
    #pragma omp parallel for
    #endif
    for (int j = 0;
        j < n;
        j += 1)
    {
        int info = LAPACKE_dgtsv(LAPACK_COL_MAJOR, m, 1, &a[j*m + 1], &b[j*m], &c[j*m], &d_in[j*m], m);
        //if (info != m) printf("lapack info check: %d\n", info);
    }

    #endif
}

// See Scipy CubicSpline for more information
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

     int param = (int) (interp_i / (numModes * numBinAll));
     int nsub = numBinAll * numModes;
     int sub_i = interp_i % (numModes * numBinAll);

     int freqArr_i = int(sub_i / num_intermediates);

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

void interpolate(double* freqs, double* propArrays,
                 double* B, double* upper_diag, double* diag, double* lower_diag,
                 int length, int numInterpParams, int numModes, int numBinAll)
{

    int num_intermediates = numModes;
    int ninterps = numModes * numInterpParams * numBinAll;

    int nblocks = std::ceil((ninterps + NUM_THREADS_INTERPOLATE -1)/NUM_THREADS_INTERPOLATE);

    // these are used for both coefficients and diagonals because they are the same size and
    // this reduces the total memory needed
    double* c1 = upper_diag;
    double* c2 = diag;
    double* c3 = lower_diag;

    // process is fill the B matrix which is banded.
    // solve banded matrix equation for spline coefficients
    // Fill the spline coefficients properly

    #ifdef __CUDACC__
    fill_B<<<nblocks, NUM_THREADS_INTERPOLATE>>>(freqs, propArrays, B, upper_diag, diag, lower_diag, ninterps, length, num_intermediates, numModes, numBinAll);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    interpolate_kern(length, ninterps, lower_diag, diag, upper_diag, B);

    set_spline_constants<<<nblocks, NUM_THREADS_INTERPOLATE>>>(freqs, propArrays, c1, c2, c3, B,
                ninterps, length, num_intermediates, numBinAll, numModes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    #else
    fill_B(freqs, propArrays, B, upper_diag, diag, lower_diag, ninterps, length, num_intermediates, numModes, numBinAll);

    interpolate_kern(length, ninterps, lower_diag, diag, upper_diag, B);


    set_spline_constants(freqs, propArrays, c1, c2, c3, B,
               ninterps, length, num_intermediates, numBinAll, numModes);
    #endif
}
