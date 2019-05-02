import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/manager.hh":
    cdef cppclass PhenomHMwrap "PhenomHM":
        PhenomHMwrap(int,
        np.uint32_t *,
        np.uint32_t *,
        int, np.float64_t*,
        np.complex128_t *, int, np.float64_t*, np.float64_t*, np.float64_t*)

        void gen_amp_phase(np.float64_t *, int,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double)

        void setup_interp_wave()

        void perform_interp(double, double, int)

        void LISAresponseFD(double, double, double, double, double, double, double, int)

        void setup_interp_response()

        void Likelihood(np.float64_t*)
        void GetTDI(np.complex128_t*, np.complex128_t*, np.complex128_t*)
        void GetAmpPhase(np.float64_t*, np.float64_t*)

cdef class PhenomHM:
    cdef PhenomHMwrap* g
    cdef int num_modes
    cdef int f_dim
    cdef int data_length

    def __cinit__(self, max_length_init,
     np.ndarray[ndim=1, dtype=np.uint32_t] l_vals,
     np.ndarray[ndim=1, dtype=np.uint32_t] m_vals,
     np.ndarray[ndim=1, dtype=np.float64_t] data_freqs,
     np.ndarray[ndim=1, dtype=np.complex128_t] data_stream,
     np.ndarray[ndim=1, dtype=np.float64_t] X_ASDinv,
     np.ndarray[ndim=1, dtype=np.float64_t] Y_ASDinv,
     np.ndarray[ndim=1, dtype=np.float64_t] Z_ASDinv):
        self.num_modes = len(l_vals)
        self.data_length = len(data_stream)
        self.g = new PhenomHMwrap(max_length_init,
        &l_vals[0],
        &m_vals[0],
        self.num_modes, &data_freqs[0],
        &data_stream[0], self.data_length, &X_ASDinv[0], &Y_ASDinv[0], &Z_ASDinv[0])

    def gen_amp_phase(self, np.ndarray[ndim=1, dtype=np.float64_t] freqs,
                        m1, #solar masses
                        m2, #solar masses
                        chi1z,
                        chi2z,
                        distance,
                        phiRef,
                        f_ref):

        self.f_dim = len(freqs)
        self.g.gen_amp_phase(&freqs[0], self.f_dim,
                                m1, #solar masses
                                m2, #solar masses
                                chi1z,
                                chi2z,
                                distance,
                                phiRef,
                                f_ref)

    def setup_interp_wave(self):
        self.g.setup_interp_wave()
        return

    def LISAresponseFD(self, inc, lam, beta, psi, t0, tRef, merger_freq, TDItag):
        self.g.LISAresponseFD(inc, lam, beta, psi, t0, tRef, merger_freq, TDItag)
        return

    def setup_interp_response(self):
        self.g.setup_interp_response()
        return

    def perform_interp(self, f_min, df, length_new):
        self.g.perform_interp(f_min, df, length_new)
        return

    def Likelihood(self):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] like_out_ = np.zeros((2,), dtype=np.float64)
        self.g.Likelihood(&like_out_[0])
        return like_out_

    def GetTDI(self):
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] X_ = np.zeros((self.data_length*self.num_modes,), dtype=np.complex128)
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] Y_ = np.zeros((self.data_length*self.num_modes,), dtype=np.complex128)
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] Z_ = np.zeros((self.data_length*self.num_modes,), dtype=np.complex128)

        self.g.GetTDI(&X_[0], &Y_[0], &Z_[0])

        return (X_.reshape(self.num_modes, self.data_length), Y_.reshape(self.num_modes, self.data_length), Z_.reshape(self.num_modes, self.data_length))

    def GetAmpPhase(self):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] amp_ = np.zeros((self.f_dim*self.num_modes,), dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] phase_ = np.zeros((self.f_dim*self.num_modes,), dtype=np.float64)

        self.g.GetAmpPhase(&amp_[0], &phase_[0])

        return (amp_.reshape(self.num_modes, self.f_dim), phase_.reshape(self.num_modes, self.f_dim))

    def WaveformThroughLikelihood(self, np.ndarray[ndim=1, dtype=np.float64_t] freqs,
                        m1, #solar masses
                        m2, #solar masses
                        chi1z,
                        chi2z,
                        distance,
                        phiRef,
                        f_ref, inc, lam, beta, psi, t0, tRef, merger_freq, TDItag, f_min, df, length_new):
        self.gen_amp_phase(freqs,
                            m1, #solar masses
                            m2, #solar masses
                            chi1z,
                            chi2z,
                            distance,
                            phiRef,
                            f_ref)
        self.setup_interp_wave()
        self.LISAresponseFD(inc, lam, beta, psi, t0, tRef, merger_freq, TDItag)
        self.setup_interp_response()
        self.perform_interp(f_min, df, length_new)
        return self.Likelihood()
