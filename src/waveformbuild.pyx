import numpy as np
cimport numpy as np

from bbhx.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "WaveformBuild.hh":
    ctypedef void* cmplx 'cmplx'

    void InterpTDI(long* templateChannels_ptrs, double* dataFreqs, double* freqs, double* propArrays, double* c1, double* c2, double* c3, double* t_start, double* t_end, int length, int data_length, int numBinAll, int numModes, long* inds_ptrs, int* inds_start, int* ind_lengths);

    void direct_sum(cmplx* templateChannels,
                    double* bbh_buffer,
                    int numBinAll, int data_length, int nChannels, int numModes, double* t_start, double* t_end)


@pointer_adjust
def InterpTDI_wrap(templateChannels_ptrs, dataFreqs, freqs, propArrays, c1, c2, c3, t_start, t_end, length, data_length, numBinAll, numModes, inds_ptrs, inds_start, ind_lengths):

    cdef size_t freqs_in = freqs
    cdef size_t propArrays_in = propArrays
    cdef size_t templateChannels_ptrs_in = templateChannels_ptrs
    cdef size_t dataFreqs_in = dataFreqs
    cdef size_t c1_in = c1
    cdef size_t c2_in = c2
    cdef size_t c3_in = c3
    cdef size_t t_start_in = t_start
    cdef size_t t_end_in = t_end
    cdef size_t inds_ptrs_in = inds_ptrs
    cdef size_t inds_start_in = inds_start
    cdef size_t ind_lengths_in = ind_lengths

    InterpTDI(<long*> templateChannels_ptrs_in, <double*> dataFreqs_in, <double*> freqs_in, <double*> propArrays_in, <double*> c1_in, <double*> c2_in, <double*> c3_in, <double*> t_start_in, <double*> t_end_in, length, data_length, numBinAll, numModes, <long*> inds_ptrs_in, <int*> inds_start_in, <int*> ind_lengths_in);

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
