#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#include "cuda_complex.hpp"
#include "stdio.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
#define CUDA_KERNEL __global__
#define CUDA_SHARED __shared__
#define CUDA_SYNC_THREADS __syncthreads();
#define THREAD_ZERO (threadIdx.x == 0)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_KERNEL
#define CUDA_SHARED
#define CUDA_SYNC_THREADS
#define THREAD_ZERO (true)
#endif

typedef gcmplx::complex<double> cmplx;

#endif // __GLOBAL_H__
