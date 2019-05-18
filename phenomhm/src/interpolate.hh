#ifndef __INTERPOLATE_H_
#define __INTERPOLATE_H_
#include <cusparse_v2.h>

class Interpolate{
    double *w;
    double *D;

    double *dl;
    double *d;
    double *du;
    double *d_dl;
    double *d_d;
    double *d_du;
    cusparseHandle_t  handle;
    cudaError_t err;
    int m;
    int n;
    int to_gpu;

public:
    // FOR NOW WE ASSUME dLOGX is evenly spaced // TODO: allocate at the beginning
    Interpolate();

    __host__ void alloc_arrays(int max_length_init);
    __host__ void prep(double *B, int m_, int n_, int to_gpu_);

    __host__ ~Interpolate(); //destructor

    __host__ void gpu_fit_constants(double *B);
    __host__ void fit_constants(double *B);
};

#endif //__INTERPOLATE_H_
