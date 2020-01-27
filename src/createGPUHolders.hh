#ifndef _CREATE_
#define _CREATE_

#include "globalPhenomHM.h"
#include "stdio.h"

ModeContainer * gpu_create_modes(int num_modes, int num_walkers, unsigned int *l_vals, unsigned int *m_vals, int max_length, int to_gpu, int to_interp);
void gpu_destroy_modes(ModeContainer * mode_vals);

/*
Function for gpu Error checking.
//*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#endif // _CREATE_
