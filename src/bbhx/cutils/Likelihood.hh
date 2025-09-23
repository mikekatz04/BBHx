#ifndef __LIKELIHOOD_HH__
#define __LIKELIHOOD_HH__

#include "global.h"

void hdyn(cmplx* likeOut1, cmplx* likeOut2,
                    cmplx* templateChannels, cmplx* dataConstants,
                    double* dataFreqs,
                    int numBinAll, int data_length, int nChannels);

void direct_like(cmplx* d_h, cmplx* h_h, cmplx* dataChannels, double* noise_weight_times_df, long* templateChannels_ptrs, int* inds_start, int* ind_lengths, int data_stream_length, int numBinAll);

void prep_hdyn_wrap(cmplx* A0_in, cmplx* A1_in, cmplx* B0_in, cmplx* B1_in, cmplx* d_arr, cmplx* h0_arr, double* S_n_arr, double df, int* bins, double* f_dense, double* f_m_arr, int data_length, int nchannels, int length_f_rel);

#endif // __LIKELIHOOD_HH__
