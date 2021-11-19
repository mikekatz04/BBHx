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
        double* inc,
        double* lam,
        double* beta,
        double* psi,
        double* t_ref,
        int TDItag, int order_fresnel_stencil,
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
     inc,
     lam,
     beta,
     psi,
     t_ref,
    TDItag, order_fresnel_stencil,
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
    cdef size_t t_ref_in = t_ref
    cdef size_t phiRef_in = phiRef

    LISA_response(
        <double*> response_out_in,
        <int*> ells_in,
        <int*> mms_in,
        <double*> freqs_in,
        <double*> phiRef_in,
        <double*> inc_in,
        <double*> lam_in,
        <double*> beta_in,
        <double*> psi_in,
        <double*> t_ref_in,
        TDItag, order_fresnel_stencil,
        numModes,
        length,
        numBinAll,
        includesAmps
    )
