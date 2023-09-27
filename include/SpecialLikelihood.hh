#ifndef __SPECIAL_LIKE__
#define __SPECIAL_LIKE__

#include "global.h"

//#ifdef __CUDACC__
void InterpTDILike(cmplx* d_h, cmplx* h_h, cmplx* dataChannels, double* psd, double* dataFreqs, double* freqs, double* propArrays, double* c1, double* c2, double* c3, double* t_start_in, double* t_end_in, int length, int data_length, int numBinAll, int numModes, long* inds_ptrs, int* inds_start, int* ind_lengths, double df, int* data_index_all, int num_data_sets, int* noise_index_all, int num_noise_sets, int gpu);
//#endif

#endif // __SPECIAL_LIKE__
