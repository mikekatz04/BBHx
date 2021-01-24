import numpy as np
cimport numpy as np

from bbhx.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "WaveformBuild.hh":
    ctypedef void* cmplx 'cmplx'

    void InterpTDI(long* templateChannels_ptrs, double* dataFreqs, double dlog10f, double* freqs, double* propArrays, double* c1, double* c2, double* c3, double* t_mrg, double* t_start, double* t_end, int length, int data_length, int numBinAll, int numModes, double t_obs_start, double t_obs_end, long* inds_ptrs, int* inds_start, int* ind_lengths);

    void direct_sum(cmplx* templateChannels,
                    double* bbh_buffer,
                    int numBinAll, int data_length, int nChannels, int numModes, double* t_start, double* t_end)

    void TDInterp(long* templateChannels_ptrs, double* dataTime, long* tsAll, long* propArraysAll, long* c1All, long* c2All, long* c3All, double* Fplus_in, double* Fcross_in, int* old_lengths, int data_length, int numBinAll, int numModes, long* inds_ptrs, int* inds_start, int* ind_lengths, int numChannels);


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
def TDInterp_wrap(templateChannels_ptrs, dataTime, ts, propArrays, c1, c2, c3, Fplus_in, Fcross_in, old_lengths, data_length, numBinAll, numModes, inds_ptrs, inds_start, ind_lengths, numChannels):

    cdef size_t ts_in = ts
    cdef size_t propArrays_in = propArrays
    cdef size_t templateChannels_ptrs_in = templateChannels_ptrs
    cdef size_t dataTime_in = dataTime
    cdef size_t c1_in = c1
    cdef size_t c2_in = c2
    cdef size_t c3_in = c3
    cdef size_t Fplus_in_in = Fplus_in
    cdef size_t Fcross_in_in = Fcross_in
    cdef size_t inds_ptrs_in = inds_ptrs
    cdef size_t inds_start_in = inds_start
    cdef size_t ind_lengths_in = ind_lengths
    cdef size_t old_lengths_in = old_lengths

    TDInterp(<long*> templateChannels_ptrs_in, <double*> dataTime_in, <long*> ts_in, <long*> propArrays_in, <long*> c1_in, <long*> c2_in, <long*> c3_in, <double*> Fplus_in_in, <double*> Fcross_in_in, <int*> old_lengths_in, data_length, numBinAll, numModes, <long*> inds_ptrs_in, <int*> inds_start_in, <int*> ind_lengths_in, numChannels)
