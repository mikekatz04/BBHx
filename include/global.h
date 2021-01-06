#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#include "cuda_complex.hpp"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_KERNEL __global__

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
#endif

typedef gcmplx::complex<double> cmplx;

#endif // __GLOBAL_H__
