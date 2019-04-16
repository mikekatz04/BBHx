import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/c_manager.h":
    cdef cppclass PhenomHMwrap "PhenomHM":
        PhenomHMwrap(np.float64_t *, int,
        np.uint32_t *,
        np.uint32_t *,
        int)
        void gen_PhenomHM(double,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double)

        double Likelihood()
        void Get_Waveform(np.complex128_t*, np.complex128_t*)

cdef class PhenomHM:
    cdef PhenomHMwrap* g
    cdef int num_modes
    cdef int f_dim

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.float64_t] freqs,
     np.ndarray[ndim=1, dtype=np.uint32_t] l_vals,
     np.ndarray[ndim=1, dtype=np.uint32_t] m_vals):
        self.f_dim = len(freqs)
        self.num_modes = len(l_vals)
        self.g = new PhenomHMwrap(&freqs[0], self.f_dim,
        &l_vals[0],
        &m_vals[0],
        self.num_modes)

    def gen_PhenomHM(self,
                        m1, #solar masses
                        m2, #solar masses
                        chi1z,
                        chi2z,
                        distance,
                        inclination,
                        phiRef,
                        deltaF,
                        f_ref):

        self.g.gen_PhenomHM(m1, #solar masses
                                m2, #solar masses
                                chi1z,
                                chi2z,
                                distance,
                                inclination,
                                phiRef,
                                deltaF,
                                f_ref)

    def Likelihood(self):
        return self.g.Likelihood()

    def Get_Waveform(self):
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] hptilde_ = np.zeros(self.f_dim*self.num_modes, dtype=np.complex128)

        cdef np.ndarray[ndim=1, dtype=np.complex128_t] hctilde_ = np.zeros(self.f_dim*self.num_modes, dtype=np.complex128)

        self.g.Get_Waveform(&hptilde_[0], &hctilde_[0])

        return (hptilde_.reshape(self.num_modes, self.f_dim), hctilde_.reshape(self.num_modes, self.f_dim))
