import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/c_manager.h":
    cdef cppclass PhenomHMwrap "PhenomHM":
        PhenomHMwrap(int,
        np.uint32_t *,
        np.uint32_t *,
        int,
        np.float64_t*,
        np.complex128_t *, np.complex128_t *, np.complex128_t *, int, np.float64_t*, np.float64_t*, np.float64_t*, int, double)

        void gen_amp_phase(np.float64_t *, int,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double)

        void setup_interp_wave()
        void GetAmpPhase(np.float64_t*, np.float64_t*)
        void LISAresponseFD(double, double, double, double, double, double, double, double)
        void setup_interp_response()
        void perform_interp()
        void Likelihood(np.float64_t*)
        void Combine()
        void GetTDI(np.complex128_t*, np.complex128_t*, np.complex128_t*)

cdef class PhenomHM:
    cdef PhenomHMwrap* g
    cdef int num_modes
    cdef int data_length
    cdef data_channel1
    cdef data_channel2
    cdef data_channel3
    cdef int current_length

    def __cinit__(self, max_length_init,
     np.ndarray[ndim=1, dtype=np.uint32_t] l_vals,
     np.ndarray[ndim=1, dtype=np.uint32_t] m_vals,
     np.ndarray[ndim=1, dtype=np.float64_t] data_freqs,
     np.ndarray[ndim=1, dtype=np.complex128_t] data_channel1,
     np.ndarray[ndim=1, dtype=np.complex128_t] data_channel2,
     np.ndarray[ndim=1, dtype=np.complex128_t] data_channel3,
     np.ndarray[ndim=1, dtype=np.float64_t] X_ASDinv,
     np.ndarray[ndim=1, dtype=np.float64_t] Y_ASDinv,
     np.ndarray[ndim=1, dtype=np.float64_t] Z_ASDinv,
     TDItag, t_obs_dur):
        self.num_modes = len(l_vals)
        self.data_channel1 = data_channel1
        self.data_channel2 = data_channel2
        self.data_channel3 = data_channel3
        self.data_length = len(data_channel1)
        self.g = new PhenomHMwrap(max_length_init,
        &l_vals[0],
        &m_vals[0],
        self.num_modes, &data_freqs[0],
        &data_channel1[0], &data_channel2[0], &data_channel3[0],
        self.data_length, &X_ASDinv[0], &Y_ASDinv[0], &Z_ASDinv[0], TDItag, t_obs_dur)

    def gen_amp_phase(self, np.ndarray[ndim=1, dtype=np.float64_t] freqs,
                        m1, #solar masses
                        m2, #solar masses
                        chi1z,
                        chi2z,
                        distance,
                        phiRef,
                        f_ref):

        self.current_length = len(freqs)
        self.g.gen_amp_phase(&freqs[0], self.current_length,
                                m1, #solar masses
                                m2, #solar masses
                                chi1z,
                                chi2z,
                                distance,
                                phiRef,
                                f_ref)

    def setup_interp_wave(self):
        self.g.setup_interp_wave()

    def LISAresponseFD(self, inc, lam, beta, psi, t0_epoch, tRef_wave_frame, tRef_sampling_frame, merger_freq):
        self.g.LISAresponseFD(inc, lam, beta, psi, t0_epoch, tRef_wave_frame, tRef_sampling_frame, merger_freq)

    def setup_interp_response(self):
        self.g.setup_interp_response()

    def perform_interp(self):
        self.g.perform_interp()

    def Combine(self):
        self.g.Combine()
        return

    def Likelihood(self):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] like_out_ = np.zeros((2,), dtype=np.float64)
        self.g.Likelihood(&like_out_[0])
        return like_out_

    def GetAmpPhase(self):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] amp_
        cdef np.ndarray[ndim=1, dtype=np.float64_t] phase_

        amp_ = np.zeros((self.num_modes*self.current_length,), dtype=np.float64)
        phase_ = np.zeros((self.num_modes*self.current_length,), dtype=np.float64)

        self.g.GetAmpPhase(&amp_[0], &phase_[0])
        return (amp_.reshape(self.num_modes, self.current_length), phase_.reshape(self.num_modes, self.current_length))

    def GetTDI(self):
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] data_channel1_
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] data_channel2_
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] data_channel3_

        data_channel1_ = np.zeros((self.data_length,), dtype=np.complex128)
        data_channel2_ = np.zeros((self.data_length,), dtype=np.complex128)
        data_channel3_ = np.zeros((self.data_length,), dtype=np.complex128)

        self.g.GetTDI(&data_channel1_[0], &data_channel2_[0], &data_channel3_[0])

        return (data_channel1_, data_channel2_, data_channel3_)

    def WaveformThroughLikelihood(self, np.ndarray[ndim=1, dtype=np.float64_t] freqs,
                        m1, #solar masses
                        m2, #solar masses
                        chi1z,
                        chi2z,
                        distance,
                        phiRef,
                        f_ref, inc, lam, beta, psi, t0, tRef_wave_frame, tRef_sampling_frame, merger_freq, return_amp_phase=False, return_TDI=False):
        self.gen_amp_phase(freqs,
                            m1, #solar masses
                            m2, #solar masses
                            chi1z,
                            chi2z,
                            distance,
                            phiRef,
                            f_ref)

        print(return_amp_phase)
        if return_amp_phase:
            return self.GetAmpPhase()

        self.LISAresponseFD(inc, lam, beta, psi, t0, tRef_wave_frame, tRef_sampling_frame, merger_freq)
        self.setup_interp_wave()
        self.setup_interp_response()
        self.perform_interp()

        print(return_TDI)
        if return_TDI:
            return self.GetTDI()

        return self.Likelihood()
