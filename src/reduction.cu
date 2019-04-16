#include "cuComplex.h"
static const int blockSize = 256;

__global__ void sumCommSingleBlock(const cuDoubleComplex *a, double *out, int arraySize) {
    int idx = threadIdx.x;
    //static const int blockSize = blockDim.x;
    double sum_re = 0.0;
    double sum_im = 0.0;
    for (int i = idx; i < arraySize; i += blockSize){
        sum_re += cuCreal(a[i]);
        sum_im += cuCimag(a[i]);
    }
    __shared__ double r[blockSize];
    __shared__ double im[blockSize];
    r[idx] = sum_re;
    im[idx] = sum_im;
    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { //uniform
        if (idx<size){
            r[idx] += r[idx+size];
            im[idx] += im[idx+size];
        }
        __syncthreads();
    }
    if (idx == 0){
        out[0] = r[0];
        out[1] = im[0];
    }
}


__global__ void sumCommMultiBlock(const cuDoubleComplex *gArr, int arraySize, cuDoubleComplex *gOut, int first_run) {
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*blockSize;
    const int gridSize = blockSize*gridDim.x;
    double sum1 = 0;
    double sum2 = 0;
    cuDoubleComplex trans;
    for (int i = gthIdx; i < arraySize; i += gridSize){
        if (first_run == 1) trans = cuCmul(cuConj(gArr[i]), gArr[i]);
        else trans = gArr[i];
        sum1 += cuCreal(trans);
        sum2 += cuCimag(trans);
    }
    __shared__ double shArr1[blockSize];
    __shared__ double shArr2[blockSize];
    shArr1[thIdx] = sum1;
    shArr2[thIdx] = sum2;
    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { //uniform
        if (thIdx<size){
            shArr1[thIdx] += shArr1[thIdx+size];
            shArr2[thIdx] += shArr2[thIdx+size];
        }

        __syncthreads();
    }
    if (thIdx == 0)
        gOut[blockIdx.x] = make_cuDoubleComplex(shArr1[0], shArr2[0]);
}
