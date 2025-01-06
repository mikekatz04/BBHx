import numpy as np
cimport numpy as np
from libcpp cimport bool

from bbhx.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "Detector.hpp":
    ctypedef void* Orbits 'Orbits'

cdef extern from "Response.hh":
    ctypedef void* cmplx 'cmplx'

    void LISA_response(
        double* response_out,
        int* ells_in,
        int* mms_in,
        double* freqs,
        double* phi_ref,
        double* inc,
        double* lam,
        double* beta,
        double* psi,
        int TDItag, bool rescaled, bool tdi2, int order_fresnel_stencil,
        int numModes,
        int length,
        int numBinAll,
        int includesAmps,
        Orbits *orbits
    );

@pointer_adjust
def LISA_response_wrap(
     response_out,
     ells,
     mms,
     freqs,
     phi_ref,
     inc,
     lam,
     beta,
     psi,
    TDItag, rescaled, tdi2, order_fresnel_stencil,
    numModes,
    length,
    numBinAll,
    includesAmps,
    orbits
):

    cdef size_t response_out_in = response_out
    cdef size_t ells_in = ells
    cdef size_t mms_in = mms
    cdef size_t freqs_in = freqs
    cdef size_t inc_in = inc
    cdef size_t lam_in = lam
    cdef size_t beta_in = beta
    cdef size_t psi_in = psi
    cdef size_t phi_ref_in = phi_ref
    cdef size_t orbits_in = orbits

    LISA_response(
        <double*> response_out_in,
        <int*> ells_in,
        <int*> mms_in,
        <double*> freqs_in,
        <double*> phi_ref_in,
        <double*> inc_in,
        <double*> lam_in,
        <double*> beta_in,
        <double*> psi_in,
        TDItag, rescaled, tdi2, order_fresnel_stencil,
        numModes,
        length,
        numBinAll,
        includesAmps,
        <Orbits*> orbits_in
    )
