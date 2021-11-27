#ifndef __WAVEFORM_BUILD_HH__
#define __WAVEFORM_BUILD_HH__

#include "global.h"

void InterpTDI(long* templateChannels_ptrs, double* dataFreqs, double* freqs, double* propArrays, double* c1, double* c2, double* c3, double* t_start, double* t_end, int length, int data_length, int numBinAll, int numModes, long* inds_ptrs, int* inds_start, int* ind_lengths);

void direct_sum(cmplx* templateChannels,
                double* bbh_buffer,
                int numBinAll, int data_length, int nChannels, int numModes, double* t_start, double* t_end);

#endif // __WAVEFORM_BUILD_HH__
