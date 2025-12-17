import numpy as np
cimport numpy as np
from libcpp cimport bool

from gpubackendtools import wrapper

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "Response.hh":
    ctypedef void* cmplx 'cmplx'
    cdef cppclass FastLISAResponseWrap "FastLISAResponse":
        void add_orbit_information(double dt_, int N_, double *n_arr_, double *L_arr_, double *x_arr_, int *links_, int *sc_r_, int *sc_e_, double armlength_) except+
        void dealloc() except+
        
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
            int includesAmps
        ) except+



cdef class pyFastLISAResponse:
    cdef FastLISAResponseWrap *g

    def __cinit__(self):
        
        self.g = new FastLISAResponseWrap()

    def add_orbit_information(self, *args, **kwargs):
        (
            dt,
            N, 
            n_arr,
            L_arr, 
            x_arr,
            links,
            sc_r, 
            sc_e,
            armlength
        ), tkwargs = wrapper(*args, **kwargs)

        cdef size_t n_arr_in = n_arr
        cdef size_t L_arr_in = L_arr
        cdef size_t x_arr_in = x_arr
        cdef size_t links_in = links
        cdef size_t sc_r_in = sc_r
        cdef size_t sc_e_in = sc_e
        
        self.g.add_orbit_information(
            dt,
            N,
            <double*> n_arr_in,
            <double*> L_arr_in, 
            <double*> x_arr_in, 
            <int*> links_in, 
            <int*> sc_r_in, 
            <int*> sc_e_in,
            armlength
        )

    def __dealloc__(self):
        self.g.dealloc()
        if self.g:
            del self.g

    def LISA_response_wrap(self, *args, **kwargs):
        (
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
            includesAmps
        ), tkwargs = wrapper(*args, **kwargs)

        cdef size_t response_out_in = response_out
        cdef size_t ells_in = ells
        cdef size_t mms_in = mms
        cdef size_t freqs_in = freqs
        cdef size_t inc_in = inc
        cdef size_t lam_in = lam
        cdef size_t beta_in = beta
        cdef size_t psi_in = psi
        cdef size_t phi_ref_in = phi_ref

        self.g.LISA_response(
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
            includesAmps
        )
