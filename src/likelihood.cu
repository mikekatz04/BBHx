#ifdef __CUDACC__
#include "cuComplex.h"
#include "cublas_v2.h"
#endif

#include <assert.h>
#include <iostream>
#include "globalPhenomHM.h"
#include <complex>

#include "cuda_complex.hpp"
#include "likelihood.hh"

#ifdef __CUDACC__
#else
#include "omp.h"
#endif

#ifdef __CUDACC__
__device__
void print_stuff()
{
  printf("PRINT STUFF yayay check it \n");
}
#endif


#ifdef __CUDACC__
void gpu_likelihood(double *d_h_arr, double *h_h_arr, agcmplx **d_data_channel1, agcmplx **d_data_channel2, agcmplx **d_data_channel3,
                                agcmplx **d_template_channel1, agcmplx **d_template_channel2, agcmplx **d_template_channel3,
                                 int data_stream_length, int nwalkers, int ndevices, cublasHandle_t *handle){

     //#pragma omp parallel
     //{
     //for (int i=0; i<nwalkers; i++){
        int j, i, th_id, nthreads;
         double d_h = 0.0;
         double h_h = 0.0;
         char * status;
         double res;
         cuDoubleComplex result;
         cublasStatus_t stat;
         //nthreads = omp_get_num_threads();
         //th_id = omp_get_thread_num();
         for (int j=0; j<ndevices; j+=1){
             cudaSetDevice(j);
             for (int i=0; i<nwalkers; i++){
                 d_h = 0.0;
                 h_h = 0.0;
                 // get data - template terms
                  stat = cublasZdotc(handle[j], data_stream_length,
                          (cuDoubleComplex*)&d_template_channel1[j][data_stream_length*i], 1,
                          (cuDoubleComplex*)d_data_channel1[j], 1,
                          &result);
                  status = _cudaGetErrorEnum(stat);
                   cudaDeviceSynchronize();

                   if (stat != CUBLAS_STATUS_SUCCESS) {
                           exit(0);
                       }
                  d_h += cuCreal(result);
                  //printf("channel1 d_h: %e\n", cuCreal(result));

                  stat = cublasZdotc(handle[j], data_stream_length,
                          (cuDoubleComplex*)&d_template_channel2[j][data_stream_length*i], 1,
                          (cuDoubleComplex*)d_data_channel2[j], 1,
                          &result);
                  status = _cudaGetErrorEnum(stat);
                   cudaDeviceSynchronize();

                   if (stat != CUBLAS_STATUS_SUCCESS) {
                           exit(0);
                       }
                  d_h += cuCreal(result);
                  //printf("channel2 d_h: %e\n", cuCreal(result));

                  stat = cublasZdotc(handle[j], data_stream_length,
                          (cuDoubleComplex*)&d_template_channel3[j][data_stream_length*i], 1,
                          (cuDoubleComplex*)d_data_channel3[j], 1,
                          &result);
                  status = _cudaGetErrorEnum(stat);
                   cudaDeviceSynchronize();

                   if (stat != CUBLAS_STATUS_SUCCESS) {
                           exit(0);
                       }
                  d_h += cuCreal(result);
                  //printf("channel3 d_h: %e\n", cuCreal(result));


                  // get template template terms
                 stat = cublasZdotc(handle[j], data_stream_length,
                              (cuDoubleComplex*)&d_template_channel1[j][data_stream_length*i], 1,
                              (cuDoubleComplex*)&d_template_channel1[j][data_stream_length*i], 1,
                              &result);
                      status = _cudaGetErrorEnum(stat);
                       cudaDeviceSynchronize();

                       if (stat != CUBLAS_STATUS_SUCCESS) {
                               exit(0);
                           }
                      h_h += cuCreal(result);
                      //printf("channel1 h_h: %e\n", cuCreal(result));

                      stat = cublasZdotc(handle[j], data_stream_length,
                              (cuDoubleComplex*)&d_template_channel2[j][data_stream_length*i], 1,
                              (cuDoubleComplex*)&d_template_channel2[j][data_stream_length*i], 1,
                              &result);
                      status = _cudaGetErrorEnum(stat);
                       cudaDeviceSynchronize();

                       if (stat != CUBLAS_STATUS_SUCCESS) {
                               exit(0);
                           }
                      h_h += cuCreal(result);
                      //printf("channel2 h_h: %e\n", cuCreal(result));

                      stat = cublasZdotc(handle[j], data_stream_length,
                              (cuDoubleComplex*)&d_template_channel3[j][data_stream_length*i], 1,
                              (cuDoubleComplex*)&d_template_channel3[j][data_stream_length*i], 1,
                              &result);
                      status = _cudaGetErrorEnum(stat);
                       cudaDeviceSynchronize();

                       if (stat != CUBLAS_STATUS_SUCCESS) {
                               exit(0);
                           }
                      h_h += cuCreal(result);
                      //printf("channel3 h_h: %e\n", cuCreal(result));
                  d_h_arr[j*nwalkers + i] = 4*d_h;
                  h_h_arr[j*nwalkers + i] = 4*h_h;
             }
        }
}
#else
agcmplx complex_dot_product(agcmplx *arr1, agcmplx *arr2, int n){
    agcmplx out(0.0, 0.0);

    for (int i=0; i<n; i++){
        out += gcmplx::conj(arr1[i])*arr2[i];
    }
    return out;
}

void cpu_likelihood(double *d_h_arr, double *h_h_arr, agcmplx *h_data_channel1, agcmplx *h_data_channel2, agcmplx *h_data_channel3,
                                agcmplx *template_channel1, agcmplx *template_channel2, agcmplx *template_channel3,
                                 int data_stream_length, int nwalkers, int ndevices){

     //#pragma omp parallel
     //{
     //for (int i=0; i<nwalkers; i++){
        //int j, i, th_id, nthreads;
         double d_h = 0.0;
         double h_h = 0.0;
         char * status;
         double res;
         agcmplx temp;
         int i=0;
         //nthreads = omp_get_num_threads();
         //th_id = omp_get_thread_num();
         #pragma omp parallel for private(d_h, h_h, temp, i)
         for (i=0; i<nwalkers; i++){
                 d_h = 0.0;
                 h_h = 0.0;
                 // get data - template terms

                  temp = complex_dot_product(&template_channel1[data_stream_length*i], h_data_channel1, data_stream_length);

                  d_h += temp.real();
                  //printf("channel1 d_h: %e\n", cuCreal(result));

                  temp = complex_dot_product(&template_channel2[data_stream_length*i], h_data_channel2, data_stream_length);

                  d_h += temp.real();
                  //printf("channel2 d_h: %e\n", cuCreal(result));

                  temp = complex_dot_product(&template_channel3[data_stream_length*i], h_data_channel3, data_stream_length);

                  d_h += temp.real();

                  temp = complex_dot_product(&template_channel1[data_stream_length*i], &template_channel1[data_stream_length*i], data_stream_length);

                  h_h += temp.real();

                  temp = complex_dot_product(&template_channel2[data_stream_length*i], &template_channel2[data_stream_length*i], data_stream_length);

                  h_h += temp.real();

                  temp = complex_dot_product(&template_channel3[data_stream_length*i], &template_channel3[data_stream_length*i], data_stream_length);

                  h_h += temp.real();

                  d_h_arr[i] = 4*d_h;
                  h_h_arr[i] = 4*h_h;
        }
}
#endif
