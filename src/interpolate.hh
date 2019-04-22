#ifndef __INTERPOLATE_H_
#define __INTERPOLATE_H_
#include <cusparse_v2.h>

class Interpolate{
    double *x_old;
    double *y_old;
    double dx_old;
    double *a;
    double *b;
    double *c;

    double *w;
    double *D;
    int N;

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

    double *coeff_1;
    double *coeff_2;
    double *coeff_3;

    double *dev_coeff_1;
    double *dev_coeff_2;
    double *dev_coeff_3;
    double *dev_x_old;
    double *dev_y_old;

    int has_been_transferred;

public:
    // FOR NOW WE ASSUME dLOGX is evenly spaced // TODO: allocate at the beginning
    Interpolate();

    __host__ void prep(double *B, int m_, int n_, int to_gpu_);

    __host__ ~Interpolate(); //destructor

    __host__ void gpu_fit_constants(double *B);
    __host__ void fit_constants(double *B);
    __host__ void transferToDevice();
    __device__ double call(double x_new);
    __host__ double cpu_call(double x_new);
};

__global__ void wave_interpolate(double *f_new, double *amp_new, double *phase_new, int num_modes, int length, Interpolate *interp_all);
#endif //__INTERPOLATE_H_
