#include "omp.h"
#include "globalPhenomHM.h"
#include "likelihood.hh"

#ifdef __CUDACC__
#include "cuComplex.h"
#include "cublas_v2.h"

__host__
void GetLikelihood_GPU (double *d_h_arr, double *h_h_arr, int nwalkers, int ndevices, cublasHandle_t *handle,
                cmplx **d_template_channel1, cmplx **d_data_channel1,
                cmplx **d_template_channel2, cmplx **d_data_channel2,
                cmplx **d_template_channel3, cmplx **d_data_channel3,
                int data_stream_length){

    //printf("like mem\n");
    //print_mem_info();

     //#pragma omp parallel
     //{
     //for (int i=0; i<nwalkers; i++){
        int j, i, th_id, nthreads;
         double d_h = 0.0;
         double h_h = 0.0;
         char * status;
         cublasStatus_t stat;
         double res;
         cuDoubleComplex result;
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
    //}
}
#endif

cmplx complex_dot_product(cmplx *arr1, cmplx *arr2, int n){
  cmplx sum(0.0, 0.0);
  for (int i=0; i<n; i++){
      sum += conj(arr1[i])*arr2[i];
  }
  return sum;
}


void GetLikelihood_CPU(double *d_h_arr, double *h_h_arr, int nwalkers,
                cmplx *template_channel1, cmplx *data_channel1,
                cmplx *template_channel2, cmplx *data_channel2,
                cmplx *template_channel3, cmplx *data_channel3,
                int data_stream_length){


      int i, j, th_id, nthreads;
      double d_h, h_h;
      cmplx res;
      //# pragma omp parallel private(i, j, th_id, d_h, h_h, res)
      //{
      //    nthreads = omp_get_num_threads();
      //    th_id = omp_get_thread_num();
          for (int i=0; i<nwalkers; i+=1){
              h_h = 0.0;
              d_h = 0.0;

              res = complex_dot_product(data_channel1, &template_channel1[i*data_stream_length], data_stream_length);
              d_h += real(res);

              res = complex_dot_product(data_channel2, &template_channel2[i*data_stream_length], data_stream_length);
              d_h += real(res);

              res = complex_dot_product(data_channel3, &template_channel3[i*data_stream_length], data_stream_length);
              d_h += real(res);

              res = complex_dot_product(&template_channel1[i*data_stream_length], &template_channel1[i*data_stream_length], data_stream_length);
              h_h += real(res);

              res = complex_dot_product(&template_channel2[i*data_stream_length], &template_channel2[i*data_stream_length], data_stream_length);
              h_h += real(res);

              res = complex_dot_product(&template_channel3[i*data_stream_length], &template_channel3[i*data_stream_length], data_stream_length);
              h_h += real(res);

              d_h_arr[i] = 4.*d_h;
              h_h_arr[i] = 4.*h_h;
              //printf("%lf, %lf\n", d_h_arr[i], h_h_arr[i]);
          }
      //}

}
