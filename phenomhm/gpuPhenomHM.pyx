import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/manager.hh":
    cdef int GetDeviceCount();

    cdef cppclass PhenomHMwrap "PhenomHM":
        PhenomHMwrap(int,
        np.uint32_t *,
        np.uint32_t *,
        int, int, int, double, int, int)

        void gen_amp_phase(np.float64_t *, int,
                            np.float64_t *,
                            np.float64_t *,
                            np.float64_t *,
                            np.float64_t *,
                            np.float64_t *,
                            np.float64_t *,
                            np.float64_t *)

        void setup_interp_wave()

        void perform_interp()

        void LISAresponseFD(np.float64_t *, np.float64_t *, np.float64_t *, np.float64_t *, np.float64_t *, np.float64_t *, np.float64_t *, np.float64_t *)

        void setup_interp_response()

        void input_data(np.float64_t *data_freqs, np.float64_t *,
                          np.float64_t *, np.float64_t *,
                          np.float64_t *, np.float64_t *,
                          np.float64_t *, int)

        void Likelihood(np.float64_t*, np.float64_t*)
        void GetTDI(np.float64_t*, np.float64_t*, np.float64_t*)
        void GetAmpPhase(np.float64_t*, np.float64_t*)

cdef class PhenomHM:
    cdef PhenomHMwrap* g
    cdef int num_modes
    cdef int f_dim
    cdef int data_length
    cdef int nwalkers
    cdef max_length_init
    cdef int ndevices

    def __cinit__(self, max_length_init, data_length,
     np.ndarray[ndim=1, dtype=np.uint32_t] l_vals,
     np.ndarray[ndim=1, dtype=np.uint32_t] m_vals,
     TDItag,
     t_obs_dur,
     nwalkers,
     ndevices):

        self.nwalkers = nwalkers
        self.num_modes = len(l_vals)
        self.ndevices = ndevices
        self.data_length = data_length
        self.max_length_init = max_length_init
        self.g = new PhenomHMwrap(max_length_init,
        &l_vals[0],
        &m_vals[0],
        self.num_modes, self.data_length, TDItag, t_obs_dur, nwalkers, ndevices)

    def gen_amp_phase(self, np.ndarray[ndim=1, dtype=np.float64_t] freqs,
                        np.ndarray[ndim=1, dtype=np.float64_t] m1, #solar masses
                        np.ndarray[ndim=1, dtype=np.float64_t] m2, #solar masses
                        np.ndarray[ndim=1, dtype=np.float64_t] chi1z,
                        np.ndarray[ndim=1, dtype=np.float64_t] chi2z,
                        np.ndarray[ndim=1, dtype=np.float64_t] distance,
                        np.ndarray[ndim=1, dtype=np.float64_t] phiRef,
                        np.ndarray[ndim=1, dtype=np.float64_t] f_ref):

        self.f_dim = self.max_length_init
        self.g.gen_amp_phase(&freqs[0], self.f_dim,
                                &m1[0], #solar masses
                                &m2[0], #solar masses
                                &chi1z[0],
                                &chi2z[0],
                                &distance[0],
                                &phiRef[0],
                                &f_ref[0])

    def input_data(self, np.ndarray[ndim=1, dtype=np.float64_t] data_freqs,
                            np.ndarray[ndim=1, dtype=np.complex128_t] data_channel1,
                            np.ndarray[ndim=1, dtype=np.complex128_t] data_channel2,
                            np.ndarray[ndim=1, dtype=np.complex128_t] data_channel3,
                            np.ndarray[ndim=1, dtype=np.float64_t] channel1_ASDinv,
                            np.ndarray[ndim=1, dtype=np.float64_t] channel2_ASDinv,
                            np.ndarray[ndim=1, dtype=np.float64_t] channel3_ASDinv):

        cdef np.ndarray[ndim=1, dtype=np.float64_t] data_channel1_in = np.zeros(data_channel1.shape[0]*2).astype(np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] data_channel2_in = np.zeros(data_channel2.shape[0]*2).astype(np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] data_channel3_in = np.zeros(data_channel3.shape[0]*2).astype(np.float64)

        if len(data_freqs) != self.data_length:
            raise ValueError("data freqs length not equal to data length input.")

        data_channel1_in[0::2] = data_channel1.real
        data_channel1_in[1::2] = data_channel1.imag
        data_channel2_in[0::2] = data_channel2.real
        data_channel2_in[1::2] = data_channel2.imag
        data_channel3_in[0::2] = data_channel3.real
        data_channel3_in[1::2] = data_channel3.imag


        self.g.input_data(&data_freqs[0], &data_channel1_in[0],
                            &data_channel2_in[0], &data_channel3_in[0],
                            &channel1_ASDinv[0], &channel2_ASDinv[0],
                            &channel3_ASDinv[0], len(data_freqs))

    def setup_interp_wave(self):
        self.g.setup_interp_wave()
        return

    def LISAresponseFD(self,
                       np.ndarray[ndim=1, dtype=np.float64_t] inc,
                       np.ndarray[ndim=1, dtype=np.float64_t] lam,
                       np.ndarray[ndim=1, dtype=np.float64_t] beta,
                       np.ndarray[ndim=1, dtype=np.float64_t] psi,
                       np.ndarray[ndim=1, dtype=np.float64_t] t0,
                       np.ndarray[ndim=1, dtype=np.float64_t] tRef_wave_frame,
                       np.ndarray[ndim=1, dtype=np.float64_t] tRef_sampling_frame,
                       np.ndarray[ndim=1, dtype=np.float64_t] merger_freq):


        self.g.LISAresponseFD(&inc[0], &lam[0], &beta[0], &psi[0], &t0[0], &tRef_wave_frame[0], &tRef_sampling_frame[0], &merger_freq[0])
        return

    def setup_interp_response(self):
        self.g.setup_interp_response()
        return

    def perform_interp(self):
        self.g.perform_interp()
        return

    def Likelihood(self):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] d_h_arr = np.zeros((self.ndevices*self.nwalkers,), dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] h_h_arr = np.zeros((self.ndevices*self.nwalkers,), dtype=np.float64)
        self.g.Likelihood(&d_h_arr[0], &h_h_arr[0])
        return d_h_arr, h_h_arr

    def GetTDI(self):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] X_ = np.zeros(((self.data_length*2)*self.nwalkers*self.ndevices,), dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] Y_ = np.zeros(((self.data_length*2)*self.nwalkers*self.ndevices,), dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] Z_ = np.zeros(((self.data_length*2)*self.nwalkers*self.ndevices,), dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] X_out = np.zeros(((self.data_length)*self.nwalkers*self.ndevices,), dtype=np.complex128)
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] Y_out = np.zeros(((self.data_length)*self.nwalkers*self.ndevices,), dtype=np.complex128)
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] Z_out = np.zeros(((self.data_length)*self.nwalkers*self.ndevices,), dtype=np.complex128)

        self.g.GetTDI(&X_[0], &Y_[0], &Z_[0])

        X_out = X_[0::2] + 1j*X_[1::2]
        Y_out = Y_[0::2] + 1j*Y_[1::2]
        Z_out = Z_[0::2] + 1j*Z_[1::2]

        return (X_out.reshape(self.nwalkers*self.ndevices, -1), Y_out.reshape(self.nwalkers*self.ndevices, -1), Z_out.reshape(self.nwalkers*self.ndevices, -1))

    def GetAmpPhase(self):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] amp_ = np.zeros((self.f_dim*self.num_modes*self.nwalkers*self.ndevices,), dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] phase_ = np.zeros((self.f_dim*self.num_modes*self.nwalkers*self.ndevices,), dtype=np.float64)

        self.g.GetAmpPhase(&amp_[0], &phase_[0])

        return (amp_.reshape(self.nwalkers*self.ndevices, self.num_modes, self.f_dim), phase_.reshape(self.nwalkers*self.ndevices, self.num_modes, self.f_dim))

    def WaveformThroughLikelihood(self, np.ndarray[ndim=1, dtype=np.float64_t] freqs,
                        np.ndarray[ndim=1, dtype=np.float64_t] m1, #solar masses
                        np.ndarray[ndim=1, dtype=np.float64_t] m2, #solar masses
                        np.ndarray[ndim=1, dtype=np.float64_t] chi1z,
                        np.ndarray[ndim=1, dtype=np.float64_t] chi2z,
                        np.ndarray[ndim=1, dtype=np.float64_t] distance,
                        np.ndarray[ndim=1, dtype=np.float64_t] phiRef,
                        np.ndarray[ndim=1, dtype=np.float64_t] f_ref,
                        np.ndarray[ndim=1, dtype=np.float64_t] inc,
                        np.ndarray[ndim=1, dtype=np.float64_t] lam,
                        np.ndarray[ndim=1, dtype=np.float64_t] beta,
                        np.ndarray[ndim=1, dtype=np.float64_t] psi,
                        np.ndarray[ndim=1, dtype=np.float64_t] t0,
                        np.ndarray[ndim=1, dtype=np.float64_t] tRef_wave_frame,
                        np.ndarray[ndim=1, dtype=np.float64_t] tRef_sampling_frame,
                        np.ndarray[ndim=1, dtype=np.float64_t] merger_freq,
                        return_amp_phase=False, return_TDI=False):

        self.gen_amp_phase(freqs,
                            m1, #solar masses
                            m2, #solar masses
                            chi1z,
                            chi2z,
                            distance,
                            phiRef,
                            f_ref)


        if return_amp_phase:
            return self.GetAmpPhase()

        self.LISAresponseFD(inc, lam, beta, psi, t0, tRef_wave_frame, tRef_sampling_frame, merger_freq)
        self.setup_interp_wave()
        self.setup_interp_response()
        self.perform_interp()

        if return_TDI:
            return self.GetTDI()

        return self.Likelihood()

def getDeviceCount():
    return GetDeviceCount()
