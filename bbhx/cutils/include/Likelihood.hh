#ifndef __LIKELIHOOD_HH__
#define __LIKELIHOOD_HH__

#include "global.h"

void hdyn(cmplx* likeOut1, cmplx* likeOut2,
                    cmplx* templateChannels, cmplx* dataConstants,
                    double* dataFreqs,
                    int numBinAll, int data_length, int nChannels);

void direct_like(cmplx* d_h, cmplx* h_h, cmplx* dataChannels, double* noise_weight_times_df, long* templateChannels_ptrs, int* inds_start, int* ind_lengths, int data_stream_length, int numBinAll, int device);

void prep_hdyn_wrap(cmplx* A0_in, cmplx* A1_in, cmplx* B0_in, cmplx* B1_in, cmplx* d_arr, cmplx* h0_arr, double* S_n_arr, double df, int* bins, double* f_dense, double* f_m_arr, int data_length, int nchannels, int length_f_rel);

void new_hdyn_prep_wrap(cmplx *A0_out, cmplx *A1_out, cmplx *B0_out, cmplx *B1_out,
    cmplx *h0_arr, cmplx *data, double *psd, double *f_m_arr, double df, double *f_dense, int *data_index, int *noise_index,
    int *start_inds_all, int *num_points_seg, int length_f_rel, int num_bin, int data_length, int nchannels);

void new_hdyn_like_wrap(cmplx* likeOut1, cmplx* likeOut2,
                    cmplx* templateChannels, cmplx* dataConstants,
                    double* dataFreqsIn, int *constants_index,
                    int numBinAll, int length_f_rel, int nChannels, int num_constants);

#endif // __LIKELIHOOD_HH__
