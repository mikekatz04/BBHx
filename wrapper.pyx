import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/manager.hh":
    cdef cppclass GPUPhenomHMwrap "GPUPhenomHM":
        GPUPhenomHMwrap(int,
        np.uint32_t *,
        np.uint32_t *,
        int,
        int,
        int,
        np.complex128_t *, int)
        void cpu_gen_PhenomHM(np.float64_t *, int,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double)

        void gpu_gen_PhenomHM(np.float64_t *, int,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double)

        void add_interp(int)
        void cpu_interp_wave(double, double, int)
        void interp_wave(double, double, int)
        double Likelihood(int)
        void Get_Waveform(np.complex128_t*)
        void gpu_Get_Waveform(np.complex128_t*)

cdef class GPUPhenomHM:
    cdef GPUPhenomHMwrap* g
    cdef int num_modes
    cdef int f_dim
    cdef int data_length
    cdef int interp_length

    def __cinit__(self, max_length,
     np.ndarray[ndim=1, dtype=np.uint32_t] l_vals,
     np.ndarray[ndim=1, dtype=np.uint32_t] m_vals,
     to_gpu, to_interp, np.ndarray[ndim=1, dtype=np.complex128_t] data_stream):
        self.num_modes = len(l_vals)
        self.data_length = len(data_stream)
        self.g = new GPUPhenomHMwrap(max_length,
        &l_vals[0],
        &m_vals[0],
        self.num_modes,
        to_gpu, to_interp, &data_stream[0], self.data_length)

    def cpu_gen_PhenomHM(self, np.ndarray[ndim=1, dtype=np.float64_t] freqs,
                        m1, #solar masses
                        m2, #solar masses
                        chi1z,
                        chi2z,
                        distance,
                        inclination,
                        phiRef,
                        deltaF,
                        f_ref):

        self.f_dim = len(freqs)
        self.g.cpu_gen_PhenomHM(&freqs[0], self.f_dim,
                                m1, #solar masses
                                m2, #solar masses
                                chi1z,
                                chi2z,
                                distance,
                                inclination,
                                phiRef,
                                deltaF,
                                f_ref)

    def gpu_gen_PhenomHM(self, np.ndarray[ndim=1, dtype=np.float64_t] freqs,
                        m1, #solar masses
                        m2, #solar masses
                        chi1z,
                        chi2z,
                        distance,
                        inclination,
                        phiRef,
                        deltaF,
                        f_ref):

        self.f_dim = len(freqs)
        self.g.gpu_gen_PhenomHM(&freqs[0], self.f_dim,
                                m1, #solar masses
                                m2, #solar masses
                                chi1z,
                                chi2z,
                                distance,
                                inclination,
                                phiRef,
                                deltaF,
                                f_ref)

    def add_interp(self, interp_length):
        self.interp_length = interp_length
        self.g.add_interp(interp_length)
        return

    def interp_wave(self, f_min, df, length_new):
        self.g.interp_wave(f_min, df, length_new)
        return

    def cpu_interp_wave(self, f_min, df, length_new):
        self.g.cpu_interp_wave(f_min, df, length_new)
        return

    def Likelihood(self, length):
        return self.g.Likelihood(length)

    def Get_Waveform(self):
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] hI_ = np.zeros((self.interp_length*self.num_modes,), dtype=np.complex128)

        self.g.Get_Waveform(&hI_[0])

        return hI_.reshape(self.num_modes, self.interp_length)

    def gpu_Get_Waveform(self):
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] hI_ = np.zeros((self.interp_length*self.num_modes,), dtype=np.complex128)

        self.g.gpu_Get_Waveform(&hI_[0])

        return hI_.reshape(self.num_modes, self.interp_length)
