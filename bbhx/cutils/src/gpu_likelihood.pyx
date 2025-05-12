import numpy as np
cimport numpy as np
from libcpp cimport bool

from bbhx.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "Likelihood.hh":
    ctypedef void* cmplx 'cmplx'

    void new_hdyn_prep_wrap(cmplx *A0_out, cmplx *A1_out, cmplx *B0_out, cmplx *B1_out,
        cmplx *h0_arr, cmplx *data, double *psd, double *f_m_arr, double df, double *f_dense, int *data_index, int *noise_index,
        int *start_inds_all, int *num_points_seg, int length_f_rel, int num_bin, int data_length, int nchannels) except+

    void new_hdyn_like_wrap(cmplx* likeOut1, cmplx* likeOut2,
        cmplx* templateChannels, cmplx* dataConstants,
        double* dataFreqsIn, int *constants_index,
        int numBinAll, int length_f_rel, int nChannels, int num_constants) except+


@pointer_adjust
def new_hdyn_like(likeOut1, likeOut2,
        templateChannels, dataConstants,
         dataFreqsIn, constants_index,
        numBinAll, length_f_rel, nChannels, num_constants):

    cdef size_t likeOut1_in = likeOut1
    cdef size_t likeOut2_in = likeOut2
    cdef size_t templateChannels_in = templateChannels
    cdef size_t dataConstants_in = dataConstants
    cdef size_t dataFreqsIn_in = dataFreqsIn
    cdef size_t constants_index_in = constants_index

    new_hdyn_like_wrap(<cmplx *>likeOut1_in, <cmplx *>likeOut2_in,
        <cmplx *>templateChannels_in, <cmplx *>dataConstants_in,
        <double *> dataFreqsIn_in, <int *> constants_index_in,
        numBinAll, length_f_rel, nChannels, num_constants)


@pointer_adjust
def new_hdyn_prep(A0_out, A1_out, B0_out, B1_out,
        h0_arr, data, psd, f_m_arr, df, f_dense, data_index, noise_index,
        start_inds_all, num_points_seg, length_f_rel, num_bin, data_length, nchannels):

    cdef size_t A0_out_in = A0_out
    cdef size_t A1_out_in = A1_out
    cdef size_t B0_out_in = B0_out
    cdef size_t B1_out_in = B1_out
    cdef size_t h0_arr_in = h0_arr
    cdef size_t data_in = data
    cdef size_t psd_in = psd
    cdef size_t f_m_arr_in = f_m_arr
    cdef size_t f_dense_in = f_dense
    cdef size_t data_index_in = data_index
    cdef size_t noise_index_in = noise_index
    cdef size_t start_inds_all_in = start_inds_all
    cdef size_t num_points_seg_in = num_points_seg

    new_hdyn_prep_wrap(<cmplx *>A0_out_in, <cmplx *>A1_out_in, <cmplx *>B0_out_in, <cmplx *>B1_out_in,
        <cmplx *>h0_arr_in, <cmplx *>data_in, <double *>psd_in, <double *>f_m_arr_in, df, <double *>f_dense_in, <int *>data_index_in, <int *>noise_index_in,
        <int *>start_inds_all_in, <int *>num_points_seg_in, length_f_rel, num_bin, data_length, nchannels)

