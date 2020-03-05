#ifndef __LIKELIHOOD_H__
#define __LIKELIHOOD_H__

#ifdef __CUDACC__
#include "cuComplex.h"
#include "cublas_v2.h"
#endif

#include <assert.h>
#include <iostream>
#include "globalPhenomHM.h"
#include <complex>
#include "cuda_complex.hpp"

#ifdef __CUDACC__
void gpu_likelihood(double *d_h_arr, double *h_h_arr, agcmplx **d_data_channel1, agcmplx **d_data_channel2, agcmplx **d_data_channel3,
                                agcmplx **d_template_channel1, agcmplx **d_template_channel2, agcmplx **d_template_channel3,
                                 int data_stream_length, int nwalkers, int ndevices, cublasHandle_t *handle, int* first_inds, int* last_inds);
#else
void cpu_likelihood(double *d_h_arr, double *h_h_arr, agcmplx *h_data_channel1, agcmplx *h_data_channel2, agcmplx *h_data_channel3,
                                agcmplx *template_channel1, agcmplx *template_channel2, agcmplx *template_channel3,
                                 int data_stream_length, int nwalkers, int ndevices, int* first_inds, int* last_inds);
#endif

#ifdef __CUDACC__
static char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif

#endif //__LIKELIHOOD_H__
