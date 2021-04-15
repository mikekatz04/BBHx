#include "global.h"
#include "constants.h"
#include "Interpolate.hh"

#ifdef __CUDACC__
#include "cusparse_v2.h"
#else
#include "lapacke.h"
#endif

#define  NUM_THREADS_INTERPOLATE 256

CUDA_CALLABLE_MEMBER
void prep_splines(int i, int length, int interp_i, int ninterps, int num_intermediates, double *b, double *ud, double *diag, double *ld, double *x, double *y, int numBinAll, int param, int nsub, int sub_i){
  double dx1, dx2, d, slope1, slope2;
  int ind0x, ind1x, ind2x, ind0y, ind1y, ind2y, ind_out;

  double xval0, xval1, xval2, yval1;

  //int numFreqarrs = int(ninterps / num_intermediates);
  int freqArr_i = int(sub_i / num_intermediates);

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

  //if ((param < 3) && (i == 10) && ((sub_i == 0) || (sub_i == 6))) printf("%d %d %d %e %e %e %e\n", param, sub_i, freqArr_i, b[ind_out], xval1, xval2, yval1);
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

     //int numFreqarrs = int(ninterps / num_intermediates);


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

void interpolate(double* freqs, double* propArrays,
                 double* B, double* upper_diag, double* diag, double* lower_diag,
                 int length, int numInterpParams, int numModes, int numBinAll)
{

    int num_intermediates = numModes;
    int ninterps = numModes * numInterpParams * numBinAll;

    int nblocks = std::ceil((ninterps + NUM_THREADS_INTERPOLATE -1)/NUM_THREADS_INTERPOLATE);

    double* c1 = upper_diag; //&interp_array[0 * numInterpParams * amp_phase_size];
    double* c2 = diag; //&interp_array[1 * numInterpParams * amp_phase_size];
    double* c3 = lower_diag; //&interp_array[2 * numInterpParams * amp_phase_size];

    //printf("%d after response, %d\n", jj, nblocks2);

     #ifdef __CUDACC__
     fill_B<<<nblocks, NUM_THREADS_INTERPOLATE>>>(freqs, propArrays, B, upper_diag, diag, lower_diag, ninterps, length, num_intermediates, numModes, numBinAll);
     cudaDeviceSynchronize();
     gpuErrchk(cudaGetLastError());

     //printf("%d after fill b\n", jj);
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
    //printf("%d after set spline\n", jj);
}













CUDA_CALLABLE_MEMBER
void prep_splines_TD(int i, int length, int interp_i, int ninterp, double *b, double *ud, double *diag, double *ld, double *x, double *y, int numBinAll, int bin_i){
  double dx1, dx2, d, slope1, slope2;
  int ind0x, ind1x, ind2x, ind0y, ind1y, ind2y, ind_out;

  double xval0, xval1, xval2, yval1;

  //int numFreqarrs = int(ninterps / num_intermediates);

  //if ((threadIdx.x == 10) && (blockIdx.x == 1)) printf("numFreqarrs %d %d %d %d %d\n", ninterps, interp_i, num_intermediates, numFreqarrs, tArr_i);
  if (i == length - 1){
    ind0y = (length - 3) * ninterp + interp_i;
    ind1y = (length - 2) * ninterp + interp_i;
    ind2y = (length - 1) * ninterp + interp_i;

    ind0x = (length - 3) * numBinAll + bin_i;
    ind1x = (length - 2) * numBinAll + bin_i;
    ind2x = (length - 1) * numBinAll + bin_i;

    ind_out = (length - 1) * ninterp + interp_i;

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

      ind0y = 0 * ninterp + interp_i;
      ind1y = 1 * ninterp + interp_i;
      ind2y = 2 * ninterp + interp_i;

      ind0x = 0 * numBinAll + bin_i;
      ind1x = 1 * numBinAll + bin_i;
      ind2x = 2 * numBinAll + bin_i;

      ind_out = 0 * ninterp + interp_i;

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

      ind0y = (i - 1) * ninterp + interp_i;
      ind1y = (i + 0) * ninterp + interp_i;
      ind2y = (i + 1) * ninterp + interp_i;

      ind0x = (i - 1) * numBinAll + bin_i;
      ind1x = (i + 0) * numBinAll + bin_i;
      ind2x = (i + 1) * numBinAll + bin_i;

      ind_out = i * ninterp + interp_i;

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

  //if ((param < 3) && (i == 10) && ((sub_i == 0) || (sub_i == 6))) printf("%d %d %d %e %e %e %e\n", param, sub_i, freqArr_i, b[ind_out], xval1, xval2, yval1);
}



CUDA_KERNEL
void fill_B_TD(double *t_arr, double *y_all, double *B, double *upper_diag, double *diag, double *lower_diag,
                      int ninterps, int numBinAll, int* lengths, int nsubs){

    #ifdef __CUDACC__

    int start1 = blockIdx.x * blockDim.x + threadIdx.x;
    int end1 = ninterps;
    int diff1 = gridDim.x * blockDim.x;

    #else

    int start1 = 0;
    int end1 = ninterps;
    int diff1 = 1;

    #endif
    for (int interp_i = start1;
         interp_i<end1; // 2 for re and im
         interp_i+= diff1){

        int bin_i = int(interp_i / nsubs);
        int length = lengths[bin_i];

        int start2 = 0;
        int diff2 = 1;
           for (int i = start2;
                i < length;
                i += diff2){


            prep_splines_TD(i, length, interp_i, ninterps, B, upper_diag, diag, lower_diag, t_arr, y_all, numBinAll, bin_i);

}
}
}

CUDA_KERNEL
void interpolate_kern_TD(int* mAll, int n, double *a, double *b, double *c, double *d, int nsubs)
{
    #ifdef __CUDACC__

    int start1 = threadIdx.x + blockDim.x * blockIdx.x;;
    int diff1 = blockDim.x * gridDim.x;

    #else

    int start1 = 0;
    int diff1 = 1;

    #endif
    for (int interp_i = start1;
         interp_i<n; // 2 for re and im
         interp_i+= diff1)
    {
        int bin_i = int(interp_i / nsubs);

        int m = mAll[bin_i];
        int ind_i, ind_im1, ind_ip1;
        double w = 0.0;
        for (int i = 1; i < m; i += 1)
        {
            ind_i = i * n + interp_i;
            ind_im1 = (i - 1) * n + interp_i;


            w = a[ind_i]/b[ind_im1];
            b[ind_i] = b[ind_i] - w * c[ind_im1];
            d[ind_i] = d[ind_i] - w * d[ind_im1];
        }

        ind_i = (m-1) * n + interp_i;

        d[ind_i] = d[ind_i]/b[ind_i];
        for (int i = m - 2; i >= 0; i -= 1)
        {
            ind_i = i * n + interp_i;
            ind_ip1 = (i + 1) * n + interp_i;

            d[ind_i] = (d[ind_i] - c[ind_i] * d[ind_ip1])/b[ind_i];

        }
    }
}


CUDA_CALLABLE_MEMBER
void fill_coefficients_td(int i, int interp_i, int ninterps, double *dydx, double dx, double *y, double *coeff1, double *coeff2, double *coeff3){
  double slope, t, dydx_i;

  int ind_i = i * ninterps + interp_i;
  int ind_ip1 = (i + 1) * ninterps + interp_i;

  slope = (y[ind_ip1] - y[ind_i])/dx;

  dydx_i = dydx[ind_i];

  t = (dydx_i + dydx[ind_ip1] - 2*slope)/dx;

  coeff1[ind_i] = dydx_i;
  coeff2[ind_i] = (slope - dydx_i) / dx - t;
  coeff3[ind_i] = t/dx;

  //if ((param == 1) && (i == length - 3) && (sub_i == 0)) printf("freq check: %d %d %d %d %d\n", i, dydx[ind_i], dydx[ind_ip1]);


}

CUDA_KERNEL
void set_spline_constants_TD(double *t_arr, double* y, double *c1, double* c2, double* c3, double *B,
                      int ninterps, int* lengths, int numBinAll, int nsubs){

      #ifdef __CUDACC__

      int start1 = blockIdx.x * blockDim.x + threadIdx.x;
      int end1 = ninterps;
      int diff1 = gridDim.x * blockDim.x;

      #else

      int start1 = 0;
      int end1 = ninterps;
      int diff1 = 1;

      #endif
    for (int interp_i= start1;
         interp_i<ninterps; // 2 for re and im
         interp_i+= diff1){

    int bin_i = int(interp_i / nsubs);
    int length = lengths[bin_i];

     int start2 = 0;
     int diff2 = 1;
     for (int i = start2;
            i < length - 1;
            i += diff2){

                // TODO: check if there is faster way to do this
              double dt = t_arr[(i + 1) * numBinAll + bin_i] - t_arr[(i) * numBinAll + bin_i];

              fill_coefficients_td(i, interp_i, ninterps, B, dt, y, c1, c2, c3);

}
}
}

void interpolate_TD(double* t_arr, double* propArrays,
                 double* B, double* upper_diag, double* diag, double* lower_diag,
                 int* lengths, int numBinAll, int nsubs)
{

    int ninterps = numBinAll * nsubs;

    int nblocks = std::ceil((ninterps + NUM_THREADS_INTERPOLATE -1)/NUM_THREADS_INTERPOLATE);

    double* c1 = upper_diag; //&interp_array[0 * numInterpParams * amp_phase_size];
    double* c2 = diag; //&interp_array[1 * numInterpParams * amp_phase_size];
    double* c3 = lower_diag; //&interp_array[2 * numInterpParams * amp_phase_size];

    //printf("%d after response, %d\n", jj, nblocks2);

     #ifdef __CUDACC__
     fill_B_TD<<<nblocks, NUM_THREADS_INTERPOLATE>>>(t_arr, propArrays, B, upper_diag, diag, lower_diag, ninterps, numBinAll, lengths, nsubs);
     cudaDeviceSynchronize();
     gpuErrchk(cudaGetLastError());

     //printf("%d after fill b\n", jj);
     interpolate_kern_TD<<<nblocks, NUM_THREADS_INTERPOLATE>>>(lengths, ninterps,lower_diag, diag, upper_diag, B, nsubs);
     cudaDeviceSynchronize();
     gpuErrchk(cudaGetLastError());

    set_spline_constants_TD<<<nblocks, NUM_THREADS_INTERPOLATE>>>(t_arr, propArrays, c1, c2, c3, B,
                    ninterps, lengths, numBinAll, nsubs);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    #else
    fill_B_TD(t_arr, propArrays, B, upper_diag, diag, lower_diag, ninterps, numBinAll, lengths, nsubs);

    //printf("%d after fill b\n", jj);
    interpolate_kern_TD(lengths, ninterps,lower_diag, diag, upper_diag, B, nsubs);

    set_spline_constants_TD(t_arr, propArrays, c1, c2, c3, B,
                   ninterps, lengths, numBinAll, nsubs);
    #endif
    //printf("%d after set spline\n", jj);
}
