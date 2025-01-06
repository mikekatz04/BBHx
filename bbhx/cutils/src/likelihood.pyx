import numpy as np
cimport numpy as np
from libcpp cimport bool

from bbhx.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "Likelihood.hh":
    ctypedef void* cmplx 'cmplx'

    void hdyn(cmplx* likeOut1, cmplx* likeOut2,
                        cmplx* templateChannels, cmplx* dataConstants,
                        double* dataFreqs,
                        int numBinAll, int data_length, int nChannels);

    void direct_like(cmplx* d_h, cmplx* h_h, cmplx* dataChannels, double* noise_weight_times_df, long* templateChannels_ptrs, int* inds_start, int* ind_lengths, int data_stream_length, int numBinAll);

    void prep_hdyn_wrap(cmplx* A0_in, cmplx* A1_in, cmplx* B0_in, cmplx* B1_in, cmplx* d_arr, cmplx* h0_arr, double* S_n_arr, double df, int* bins, double* f_dense, double* f_m_arr, int data_length, int nchannels, int length_f_rel);

@pointer_adjust
def hdyn_wrap(likeOut1, likeOut2,
                    templateChannels, dataConstants,
                    dataFreqs,
                    numBinAll, data_length, nChannels):

    cdef size_t likeOut1_in = likeOut1
    cdef size_t likeOut2_in = likeOut2
    cdef size_t templateChannels_in = templateChannels
    cdef size_t dataConstants_in = dataConstants
    cdef size_t dataFreqs_in = dataFreqs

    hdyn(<cmplx*> likeOut1_in, <cmplx*> likeOut2_in,
            <cmplx*> templateChannels_in, <cmplx*> dataConstants_in,
            <double*> dataFreqs_in,
            numBinAll, data_length, nChannels);

@pointer_adjust
def direct_like_wrap(d_h, h_h, dataChannels, noise_weight_times_df, templateChannels_ptrs, inds_start, ind_lengths, data_stream_length, numBinAll):

    cdef size_t d_h_in = d_h
    cdef size_t h_h_in = h_h
    cdef size_t dataChannels_in = dataChannels
    cdef size_t noise_weight_times_df_in = noise_weight_times_df
    cdef size_t templateChannels_ptrs_in = templateChannels_ptrs
    cdef size_t inds_start_in = inds_start
    cdef size_t ind_lengths_in = ind_lengths

    direct_like(<cmplx*> d_h_in, <cmplx*> h_h_in, <cmplx*> dataChannels_in, <double*> noise_weight_times_df_in, <long*> templateChannels_ptrs_in, <int*> inds_start_in, <int*> ind_lengths_in, data_stream_length, numBinAll)


@pointer_adjust
def prep_hdyn(A0_in, A1_in, B0_in, B1_in, d_arr, h0_arr, S_n_arr, df, bins, f_dense, f_m_arr, data_length, nchannels, length_f_rel):

    cdef size_t A0_in_in = A0_in
    cdef size_t A1_in_in = A1_in
    cdef size_t B0_in_in = B0_in
    cdef size_t B1_in_in = B1_in
    cdef size_t d_arr_in = d_arr
    cdef size_t h0_arr_in = h0_arr
    cdef size_t S_n_arr_in = S_n_arr
    cdef size_t f_dense_in = f_dense
    cdef size_t f_m_arr_in = f_m_arr
    cdef size_t bins_in = bins

    prep_hdyn_wrap(<cmplx*> A0_in_in, <cmplx*> A1_in_in, <cmplx*> B0_in_in, <cmplx*> B1_in_in, <cmplx*> d_arr_in, <cmplx*> h0_arr_in, <double*> S_n_arr_in, df, <int*> bins_in, <double*> f_dense_in, <double*> f_m_arr_in, data_length, nchannels, length_f_rel)
