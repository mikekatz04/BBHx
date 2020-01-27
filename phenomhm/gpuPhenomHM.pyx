import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "../src/manager.hh":
    cdef int GetDeviceCount();

    cdef cppclass PhenomHMwrap "PhenomHM":
        PhenomHMwrap(int max_length_init_,
            unsigned int *l_vals_,
            unsigned int *m_vals_,
            int num_modes_,
            int data_stream_length_,
            int TDItag,
            double t_obs_start_,
            double t_obs_end_,
            int nwalkers_,
            int ndevices_)

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

        void input_data(np.float64_t *data_freqs, np.complex128_t *,
                          np.complex128_t *, np.complex128_t *,
                          np.float64_t *, np.float64_t *,
                          np.float64_t *, int)

        void input_global_data(np.int_t ptr_data_freqs_,
                                          np.int_t ptr_data_channel1_,
                                    np.int_t ptr_data_channel2_, np.int_t ptr_data_channel3_, int data_stream_length_)

        void Likelihood(np.float64_t*, np.float64_t*)
        void ResetGlobalTemplate()
        void GetTDI(np.complex128_t*, np.complex128_t*, np.complex128_t*)
        void GetResponse(np.complex128_t* transferL1_, np.complex128_t* transferL2_, np.complex128_t* transferL3_,
                          np.float64_t* phaseRdelay_, np.float64_t* time_freq_corr_)
        void GetAmpPhase(np.float64_t*, np.float64_t*)
        void GetPhaseSpline(np.float64_t* phase, np.float64_t* coeff1_, np.float64_t* coeff2_, np.float64_t* coeff3_)

cdef class PhenomHM:
    cdef PhenomHMwrap* g
    cdef int num_modes
    cdef int f_dim
    cdef int data_length
    cdef int nwalkers
    cdef max_length_init
    cdef int ndevices

    def __cinit__(self, max_length_init,
     np.ndarray[ndim=1, dtype=np.uint32_t] l_vals,
     np.ndarray[ndim=1, dtype=np.uint32_t] m_vals,
     data_stream_length,
     TDItag,
     t_obs_start,
     t_obs_end,
     nwalkers,
     ndevices):

        self.nwalkers = nwalkers
        self.num_modes = len(l_vals)
        self.ndevices = ndevices
        self.data_length = data_stream_length
        self.max_length_init = max_length_init
        self.g = new PhenomHMwrap(max_length_init,
        &l_vals[0],
        &m_vals[0],
        self.num_modes,
        self.data_length, TDItag,
        t_obs_start,
        t_obs_end, nwalkers, ndevices)

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

        self.g.input_data(&data_freqs[0], &data_channel1[0],
                            &data_channel2[0], &data_channel3[0],
                            &channel1_ASDinv[0], &channel2_ASDinv[0],
                            &channel3_ASDinv[0], len(data_freqs))

    def input_global_data(self, data_freqs,
                            template_channel1,
                            template_channel2,
                            template_channel3,
                            ):

        cdef np.int_t ptr_data_freqs
        cdef np.int_t ptr_template_channel1
        cdef np.int_t ptr_template_channel2
        cdef np.int_t ptr_template_channel3

        if isinstance(data_freqs, np.ndarray):
            ptr_data_freqs = data_freqs.__array_interface__.get('data')[0]
            ptr_template_channel1 = template_channel1.__array_interface__.get('data')[0]
            ptr_template_channel2 = template_channel2.__array_interface__.get('data')[0]
            ptr_template_channel3 = template_channel3.__array_interface__.get('data')[0]

        else:  # assumes cupy array then
            ptr_data_freqs = data_freqs.data.mem.ptr
            ptr_template_channel1 = template_channel1.data.mem.ptr
            ptr_template_channel2 = template_channel2.data.mem.ptr
            ptr_template_channel3 = template_channel3.data.mem.ptr

        self.g.input_global_data(ptr_data_freqs,
                                ptr_template_channel1,
                                ptr_template_channel2,
                                ptr_template_channel3,
                                len(data_freqs))

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
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] X_ = np.zeros((self.data_length*self.nwalkers*self.ndevices,), dtype=np.complex128)
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] Y_ = np.zeros((self.data_length*self.nwalkers*self.ndevices,), dtype=np.complex128)
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] Z_ = np.zeros((self.data_length*self.nwalkers*self.ndevices,), dtype=np.complex128)

        self.g.GetTDI(&X_[0], &Y_[0], &Z_[0])

        return (X_.reshape(self.nwalkers*self.ndevices, -1), Y_.reshape(self.nwalkers*self.ndevices, -1), Z_.reshape(self.nwalkers*self.ndevices, -1))

    def GetAmpPhase(self):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] amp_ = np.zeros((self.f_dim*self.num_modes*self.nwalkers*self.ndevices,), dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] phase_ = np.zeros((self.f_dim*self.num_modes*self.nwalkers*self.ndevices,), dtype=np.float64)

        self.g.GetAmpPhase(&amp_[0], &phase_[0])

        return (amp_.reshape(self.nwalkers*self.ndevices, self.num_modes, self.f_dim), phase_.reshape(self.nwalkers*self.ndevices, self.num_modes, self.f_dim))

    def GetPhaseSpline(self):
        cdef np.ndarray[ndim=1, dtype=np.float64_t] phase_ = np.zeros((self.f_dim*self.num_modes*self.nwalkers*self.ndevices,), dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] coeff1_ = np.zeros(((self.f_dim-1)*self.num_modes*self.nwalkers*self.ndevices,), dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] coeff2_ = np.zeros(((self.f_dim-1)*self.num_modes*self.nwalkers*self.ndevices,), dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] coeff3_ = np.zeros(((self.f_dim-1)*self.num_modes*self.nwalkers*self.ndevices,), dtype=np.float64)

        self.g.GetPhaseSpline(&phase_[0], &coeff1_[0], &coeff2_[0], &coeff3_[0])

        return (phase_.reshape(self.nwalkers*self.ndevices, self.num_modes, self.f_dim),
                coeff1_.reshape(self.nwalkers*self.ndevices, self.num_modes, (self.f_dim-1)),
                coeff2_.reshape(self.nwalkers*self.ndevices, self.num_modes, (self.f_dim-1)),
                coeff3_.reshape(self.nwalkers*self.ndevices, self.num_modes, (self.f_dim-1)))

    def GetResponse(self):
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] transferL1_ = np.zeros((self.f_dim*self.num_modes*self.nwalkers*self.ndevices,), dtype=np.complex128)
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] transferL2_ = np.zeros((self.f_dim*self.num_modes*self.nwalkers*self.ndevices,), dtype=np.complex128)
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] transferL3_ = np.zeros((self.f_dim*self.num_modes*self.nwalkers*self.ndevices,), dtype=np.complex128)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] phaseRdelay_ = np.zeros((self.f_dim*self.num_modes*self.nwalkers*self.ndevices,), dtype=np.float64)
        cdef np.ndarray[ndim=1, dtype=np.float64_t] time_freq_corr_ = np.zeros((self.f_dim*self.num_modes*self.nwalkers*self.ndevices,), dtype=np.float64)

        self.g.GetResponse(&transferL1_[0], &transferL2_[0], &transferL3_[0], &phaseRdelay_[0], &time_freq_corr_[0])

        return (transferL1_.reshape(self.nwalkers*self.ndevices, self.num_modes, self.f_dim),
                transferL2_.reshape(self.nwalkers*self.ndevices, self.num_modes, self.f_dim),
                transferL3_.reshape(self.nwalkers*self.ndevices, self.num_modes, self.f_dim),
                phaseRdelay_.reshape(self.nwalkers*self.ndevices, self.num_modes, self.f_dim),
                time_freq_corr_.reshape(self.nwalkers*self.ndevices, self.num_modes, self.f_dim))

    def ResetGlobalTemplate(self):
        self.g.ResetGlobalTemplate()

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
                        return_amp_phase=False, return_TDI=False, return_response=False, return_phase_spline=False):

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

        self.setup_interp_wave()
        if return_phase_spline:
            return self.GetPhaseSpline()

        self.LISAresponseFD(inc, lam, beta, psi, t0, tRef_wave_frame, tRef_sampling_frame, merger_freq)
        if return_response:
            return self.GetResponse()

        self.setup_interp_response()
        self.perform_interp()

        if return_TDI:
            return self.GetTDI()

        return self.Likelihood()


def getDeviceCount():
    return GetDeviceCount()
