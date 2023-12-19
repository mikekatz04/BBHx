import numpy as np
cimport numpy as np

from bbhx.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "tmp_adjust.hh":
    ctypedef void* cmplx 'cmplx'

    void fast_snr_calculator_wrap(
        double *snr_out, 
        const double *params, 
        const int num_binaries,
        const int N,
        double *y_rd_all,
        double *c1_rd_all,
        double *c2_rd_all,
        double *c3_rd_all,
        double *y_dm_all,
        double *c1_dm_all,
        double *c2_dm_all,
        double *c3_dm_all,
        double dspin,
        int num_segs,
        double *psd,
        double df_psd,
        int num_psd
    ) except+

@pointer_adjust
def fast_snr_calculator(
    snr_out, 
    params, 
    num_binaries,
    N,
    y_rd_all,
    c1_rd_all,
    c2_rd_all,
    c3_rd_all,
    y_dm_all,
    c1_dm_all,
    c2_dm_all,
    c3_dm_all,
    dspin,
    num_segs,
    psd,
    df_psd,
    num_psd
):
    cdef size_t snr_out_in = snr_out
    cdef size_t params_in = params
    cdef size_t y_rd_all_in = y_rd_all
    cdef size_t c1_rd_all_in = c1_rd_all
    cdef size_t c2_rd_all_in = c2_rd_all
    cdef size_t c3_rd_all_in = c3_rd_all
    cdef size_t y_dm_all_in = y_dm_all
    cdef size_t c1_dm_all_in = c1_dm_all
    cdef size_t c2_dm_all_in = c2_dm_all
    cdef size_t c3_dm_all_in = c3_dm_all
    cdef size_t psd_in = psd

    fast_snr_calculator_wrap(
        <double *>snr_out_in, 
        <double *>params_in, 
        num_binaries,
        N,
        <double *>y_rd_all_in,
        <double *>c1_rd_all_in,
        <double *>c2_rd_all_in,
        <double *>c3_rd_all_in,
        <double *>y_dm_all_in,
        <double *>c1_dm_all_in,
        <double *>c2_dm_all_in,
        <double *>c3_dm_all_in,
        dspin,
        num_segs,
        <double *>psd_in,
        df_psd,
        num_psd
    )

