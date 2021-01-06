import numpy as np
cimport numpy as np

from bbhx.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "full.h":
    ctypedef void* cmplx 'cmplx'

    void InterpTDI(long* templateChannels_ptrs, double* dataFreqs, double dlog10f, double* freqs, double* propArrays, double* c1, double* c2, double* c3, double* t_mrg, double* t_start, double* t_end, int length, int data_length, int numBinAll, int numModes, double t_obs_start, double t_obs_end, long* inds_ptrs, int* inds_start, int* ind_lengths);

    void hdyn(cmplx* likeOut1, cmplx* likeOut2,
                        cmplx* templateChannels, cmplx* dataConstants,
                        double* dataFreqs,
                        int numBinAll, int data_length, int nChannels);

    void direct_sum(cmplx* templateChannels,
                    double* bbh_buffer,
                    int numBinAll, int data_length, int nChannels, int numModes, double* t_start, double* t_end)

    void direct_like(double* d_h, double* h_h, cmplx* dataChannels, double* noise_weight_times_df, long* templateChannels_ptrs, int* inds_start, int* ind_lengths, int data_stream_length, int numBinAll);

@pointer_adjust
def InterpTDI_wrap(templateChannels_ptrs, dataFreqs, dlog10f, freqs, propArrays, c1, c2, c3, t_mrg, t_start, t_end, length, data_length, numBinAll, numModes, t_obs_start, t_obs_end, inds_ptrs, inds_start, ind_lengths):

    cdef size_t freqs_in = freqs
    cdef size_t propArrays_in = propArrays
    cdef size_t templateChannels_ptrs_in = templateChannels_ptrs
    cdef size_t dataFreqs_in = dataFreqs
    cdef size_t c1_in = c1
    cdef size_t c2_in = c2
    cdef size_t c3_in = c3
    cdef size_t t_mrg_in = t_mrg
    cdef size_t t_start_in = t_start
    cdef size_t t_end_in = t_end
    cdef size_t inds_ptrs_in = inds_ptrs
    cdef size_t inds_start_in = inds_start
    cdef size_t ind_lengths_in = ind_lengths

    InterpTDI(<long*> templateChannels_ptrs_in, <double*> dataFreqs_in, dlog10f, <double*> freqs_in, <double*> propArrays_in, <double*> c1_in, <double*> c2_in, <double*> c3_in, <double*> t_mrg_in, <double*> t_start_in, <double*> t_end_in, length, data_length, numBinAll, numModes, t_obs_start, t_obs_end, <long*> inds_ptrs_in, <int*> inds_start_in, <int*> ind_lengths_in);


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
def direct_sum_wrap(templateChannels,
                bbh_buffer,
                numBinAll, data_length, nChannels, numModes, t_start, t_end):

    cdef size_t templateChannels_in = templateChannels
    cdef size_t bbh_buffer_in = bbh_buffer
    cdef size_t t_start_in = t_start
    cdef size_t t_end_in = t_end

    direct_sum(<cmplx*> templateChannels_in,
                    <double*> bbh_buffer_in,
                    numBinAll, data_length, nChannels, numModes, <double*> t_start_in, <double*> t_end_in)


@pointer_adjust
def direct_like_wrap(d_h, h_h, dataChannels, noise_weight_times_df, templateChannels_ptrs, inds_start, ind_lengths, data_stream_length, numBinAll):

    cdef size_t d_h_in = d_h
    cdef size_t h_h_in = h_h
    cdef size_t dataChannels_in = dataChannels
    cdef size_t noise_weight_times_df_in = noise_weight_times_df
    cdef size_t templateChannels_ptrs_in = templateChannels_ptrs
    cdef size_t inds_start_in = inds_start
    cdef size_t ind_lengths_in = ind_lengths

    direct_like(<double*> d_h_in, <double*> h_h_in, <cmplx*> dataChannels_in, <double*> noise_weight_times_df_in, <long*> templateChannels_ptrs_in, <int*> inds_start_in, <int*> ind_lengths_in, data_stream_length, numBinAll)
