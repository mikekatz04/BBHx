#ifdef __LIKELIHOOD_H__
#define __LIKELIHOOD_H__

#include "omp.h"
#include "globalPhenomHM.h"
#include "cuComplex.h"
#include "cublas_v2.h"

__host__
void GetLikelihood_GPU (double *d_h_arr, double *h_h_arr, int nwalkers, int ndevices, cublasHandle_t *handle,
                cmplx **d_template_channel1, cmplx **d_data_channel1,
                cmplx **d_template_channel2, cmplx **d_data_channel2,
                cmplx **d_template_channel3, cmplx **d_data_channel3,
                int data_stream_length);

cmplx complex_dot_product(cmplx *arr1, cmplx *arr2, int n);

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

#endif //__LIKELIHOOD_H__
