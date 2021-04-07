#ifndef __WAVEFORM_BUILD_HH__
#define __WAVEFORM_BUILD_HH__

#include "global.h"

void InterpTDI(long* templateChannels_ptrs, double* dataFreqs, double dlog10f, double* freqs, double* propArrays, double* c1, double* c2, double* c3, double* t_mrg, double* t_start, double* t_end, int length, int data_length, int numBinAll, int numModes, double t_obs_start, double t_obs_end, long* inds_ptrs, int* inds_start, int* ind_lengths);

void direct_sum(cmplx* templateChannels,
                double* bbh_buffer,
                int numBinAll, int data_length, int nChannels, int numModes, double* t_start, double* t_end);

void TDInterp(long* templateChannels_ptrs, double* dataTime, long* tsAll, long* propArraysAll, long* c1All, long* c2All, long* c3All, double* Fplus_in, double* Fcross_in, int* old_lengths, int data_length, int numBinAll, int numModes, int* ls, int* ms, long* inds_ptrs, int* inds_start, int* ind_lengths, int numChannels);

void TDInterp2(cmplx* templateChannels, double* dataTime, double* tsAll, double* propArraysAll, double* c1All, double* c2All, double* c3All, double* Fplus_in, double* Fcross_in, int old_length, int* old_lengths, int data_length, int numBinAll, int numModes, int* ls, int* ms, int* inds, int* lengths, int max_length, int numChannels);

#endif // __WAVEFORM_BUILD_HH__
