import numpy as np
cimport numpy as np

from bbhx.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "Response.hh":
    ctypedef void* cmplx 'cmplx'

    void LISA_response(
        double* response_out,
        int* ells_in,
        int* mms_in,
        double* freqs,
        double* phiRef,
        double* f_ref,
        double* inc,
        double* lam,
        double* beta,
        double* psi,
        double* tRef_wave_frame,
        double* tRef_sampling_frame,
        double tBase, int TDItag, int order_fresnel_stencil,
        int numModes,
        int length,
        int numBinAll,
        int includesAmps
    );

@pointer_adjust
def LISA_response_wrap(
     response_out,
     ells,
     mms,
     freqs,
     phiRef,
     f_ref,
     inc,
     lam,
     beta,
     psi,
     tRef_wave_frame,
     tRef_sampling_frame,
    tBase, TDItag, order_fresnel_stencil,
    numModes,
    length,
    numBinAll,
    includesAmps
):

    cdef size_t response_out_in = response_out
    cdef size_t ells_in = ells
    cdef size_t mms_in = mms
    cdef size_t freqs_in = freqs
    cdef size_t inc_in = inc
    cdef size_t lam_in = lam
    cdef size_t beta_in = beta
    cdef size_t psi_in = psi
    cdef size_t tRef_wave_frame_in = tRef_wave_frame
    cdef size_t tRef_sampling_frame_in = tRef_sampling_frame
    cdef size_t phiRef_in = phiRef
    cdef size_t f_ref_in = f_ref

    LISA_response(
        <double*> response_out_in,
        <int*> ells_in,
        <int*> mms_in,
        <double*> freqs_in,
        <double*> phiRef_in,
        <double*> f_ref_in,
        <double*> inc_in,
        <double*> lam_in,
        <double*> beta_in,
        <double*> psi_in,
        <double*> tRef_wave_frame_in,
        <double*> tRef_sampling_frame_in,
        tBase, TDItag, order_fresnel_stencil,
        numModes,
        length,
        numBinAll,
        includesAmps
    )
