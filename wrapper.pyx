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
        int)
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
        void interp_wave(np.float64_t*, np.float64_t*)
        double Likelihood()
        void Get_Waveform(int, np.float64_t*, np.float64_t*)
        void gpu_Get_Waveform(int, np.float64_t*, np.float64_t*)

cdef class GPUPhenomHM:
    cdef GPUPhenomHMwrap* g
    cdef int num_modes
    cdef int f_dim

    def __cinit__(self, max_length,
     np.ndarray[ndim=1, dtype=np.uint32_t] l_vals,
     np.ndarray[ndim=1, dtype=np.uint32_t] m_vals,
     to_gpu, to_interp):
        self.num_modes = len(l_vals)
        self.g = new GPUPhenomHMwrap(max_length,
        &l_vals[0],
        &m_vals[0],
        self.num_modes,
        to_gpu, to_interp)

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
        self.g.add_interp(interp_length)
        return

    def interp_wave(self, np.ndarray[ndim=1, dtype=np.float64_t] amp, np.ndarray[ndim=1, dtype=np.float64_t] phase):
        self.g.interp_wave(&amp[0], &phase[0])
        return

    def Likelihood(self):
        return self.g.Likelihood()

    def Get_Waveform(self):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] amp_ = np.zeros((self.f_dim,), dtype=np.float64)

        cdef np.ndarray[ndim=1, dtype=np.float64_t] phase_ = np.zeros((self.f_dim,), dtype=np.float64)

        amp_out = np.zeros((self.num_modes, self.f_dim), dtype=np.float64)
        phase_out = np.zeros((self.num_modes, self.f_dim), dtype=np.float64)
        for mode_i in range(self.num_modes):
            self.g.Get_Waveform(mode_i, &amp_[0], &phase_[0])
            amp_out[mode_i] = amp_
            phase_out[mode_i] = phase_

        return (amp_, phase_)

    def gpu_Get_Waveform(self):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] amp_ = np.zeros((self.f_dim,), dtype=np.float64)

        cdef np.ndarray[ndim=1, dtype=np.float64_t] phase_ = np.zeros((self.f_dim,), dtype=np.float64)

        amp_out = np.zeros((self.num_modes, self.f_dim), dtype=np.float64)
        phase_out = np.zeros((self.num_modes, self.f_dim), dtype=np.float64)
        for mode_i in range(self.num_modes):
            self.g.gpu_Get_Waveform(mode_i, &amp_[0], &phase_[0])
            amp_out[mode_i] = amp_
            phase_out[mode_i] = phase_

        return (amp_, phase_)
