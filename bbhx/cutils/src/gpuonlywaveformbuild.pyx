import numpy as np
cimport numpy as np

from bbhx.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "SpecialLikelihood.hh":
    ctypedef void* cmplx 'cmplx'

    void InterpTDILike(cmplx* d_h, cmplx* h_h, cmplx* dataChannels, double* psd, double* dataFreqs, double* freqs, double* propArrays, double* c1, double* c2, double* c3, double* t_start_in, double* t_end_in, int length, int data_length, int numBinAll, int numModes, long* inds_ptrs, int* inds_start, int* ind_lengths, double df, int* data_index_all, int num_data_sets, int* noise_index_all, int num_noise_sets) except+

@pointer_adjust
def InterpTDILike_wrap(d_h, h_h, dataChannels, psd, dataFreqs, freqs, propArrays, c1, c2, c3, t_start, t_end, length, data_length, numBinAll, numModes, inds_ptrs, inds_start, ind_lengths, df, data_index_all, num_data_sets, noise_index_all, num_noise_sets):

    cdef size_t d_h_in = d_h
    cdef size_t h_h_in = h_h
    cdef size_t dataChannels_in = dataChannels
    cdef size_t psd_in = psd
    cdef size_t freqs_in = freqs
    cdef size_t propArrays_in = propArrays
    cdef size_t dataFreqs_in = dataFreqs
    cdef size_t c1_in = c1
    cdef size_t c2_in = c2
    cdef size_t c3_in = c3
    cdef size_t t_start_in = t_start
    cdef size_t t_end_in = t_end
    cdef size_t inds_ptrs_in = inds_ptrs
    cdef size_t inds_start_in = inds_start
    cdef size_t ind_lengths_in = ind_lengths
    cdef size_t data_index_all_in = data_index_all
    cdef size_t noise_index_all_in = noise_index_all

    InterpTDILike(<cmplx*> d_h_in, <cmplx*> h_h_in, <cmplx*> dataChannels_in, <double*> psd_in, <double*> dataFreqs_in, <double*> freqs_in, <double*> propArrays_in, <double*> c1_in, <double*> c2_in, <double*> c3_in, <double*> t_start_in, <double*> t_end_in, length, data_length, numBinAll, numModes, <long*> inds_ptrs_in, <int*> inds_start_in, <int*> ind_lengths_in, df, <int*> data_index_all_in, num_data_sets, <int*> noise_index_all_in, num_noise_sets);
