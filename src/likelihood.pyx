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
                        int numBinAll, int data_length, int nChannels, bool full);

    void direct_like(double* d_h, double* h_h, cmplx* dataChannels, double* noise_weight_times_df, long* templateChannels_ptrs, int* inds_start, int* ind_lengths, int data_stream_length, int numBinAll);

@pointer_adjust
def hdyn_wrap(likeOut1, likeOut2,
                    templateChannels, dataConstants,
                    dataFreqs,
                    numBinAll, data_length, nChannels, full):

    cdef size_t likeOut1_in = likeOut1
    cdef size_t likeOut2_in = likeOut2
    cdef size_t templateChannels_in = templateChannels
    cdef size_t dataConstants_in = dataConstants
    cdef size_t dataFreqs_in = dataFreqs

    hdyn(<cmplx*> likeOut1_in, <cmplx*> likeOut2_in,
            <cmplx*> templateChannels_in, <cmplx*> dataConstants_in,
            <double*> dataFreqs_in,
            numBinAll, data_length, nChannels, full);

@pointer_adjust
def direct_like_wrap(d_h, h_h, dataChannels, noise_weight_times_df, templateChannels_ptrs, inds_start, ind_lengths, data_stream_length, numBinAll, full):

    cdef size_t d_h_in = d_h
    cdef size_t h_h_in = h_h
    cdef size_t dataChannels_in = dataChannels
    cdef size_t noise_weight_times_df_in = noise_weight_times_df
    cdef size_t templateChannels_ptrs_in = templateChannels_ptrs
    cdef size_t inds_start_in = inds_start
    cdef size_t ind_lengths_in = ind_lengths

    direct_like(<double*> d_h_in, <double*> h_h_in, <cmplx*> dataChannels_in, <double*> noise_weight_times_df_in, <long*> templateChannels_ptrs_in, <int*> inds_start_in, <int*> ind_lengths_in, data_stream_length, numBinAll)
