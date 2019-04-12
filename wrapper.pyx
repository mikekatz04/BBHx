import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/manager.hh":
    cdef cppclass C_GPUPhenomHM "GPUPhenomHM":
        C_GPUPhenomHM(np.int32_t*, int,
        np.float64_t *, int,
        np.uint32_t *,
        np.uint32_t *,
        int,
        int)
        void increment()
        void cpu_gen_PhenomHM(double,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double)

        void gpu_gen_PhenomHM(double,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double,
                            double)

        void retreive()
        void Get_Waveform(np.complex128_t*, np.complex128_t*)
        void retreive_to(np.int32_t*, int)

cdef class GPUPhenomHM:
    cdef C_GPUPhenomHM* g
    cdef int dim1
    cdef int num_modes
    cdef int f_dim

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.int32_t] arr,
     np.ndarray[ndim=1, dtype=np.float64_t] freqs,
     np.ndarray[ndim=1, dtype=np.uint32_t] l_vals,
     np.ndarray[ndim=1, dtype=np.uint32_t] m_vals,
     to_gpu):
        self.dim1 = len(arr)
        self.f_dim = len(freqs)
        self.num_modes = len(l_vals)
        self.g = new C_GPUPhenomHM(&arr[0], self.dim1,
        &freqs[0], self.f_dim,
        &l_vals[0],
        &m_vals[0],
        self.num_modes,
        to_gpu)


    def increment(self):
        self.g.increment()

    def cpu_gen_PhenomHM(self,
                        m1, #solar masses
                        m2, #solar masses
                        chi1z,
                        chi2z,
                        distance,
                        inclination,
                        phiRef,
                        deltaF,
                        f_ref):

        self.g.cpu_gen_PhenomHM(m1, #solar masses
                                m2, #solar masses
                                chi1z,
                                chi2z,
                                distance,
                                inclination,
                                phiRef,
                                deltaF,
                                f_ref)

    def gpu_gen_PhenomHM(self,
                        m1, #solar masses
                        m2, #solar masses
                        chi1z,
                        chi2z,
                        distance,
                        inclination,
                        phiRef,
                        deltaF,
                        f_ref):

        self.g.gpu_gen_PhenomHM(m1, #solar masses
                                m2, #solar masses
                                chi1z,
                                chi2z,
                                distance,
                                inclination,
                                phiRef,
                                deltaF,
                                f_ref)

    def retreive_inplace(self):
        self.g.retreive()

    def Get_Waveform(self):
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] hptilde_ = np.zeros(self.f_dim*self.num_modes, dtype=np.complex128)

        cdef np.ndarray[ndim=1, dtype=np.complex128_t] hctilde_ = np.zeros(self.f_dim*self.num_modes, dtype=np.complex128)

        self.g.Get_Waveform(&hptilde_[0], &hctilde_[0])

        return (hptilde_, hctilde_)

    def retreive(self):
        cdef np.ndarray[ndim=1, dtype=np.int32_t] a = np.zeros(self.dim1, dtype=np.int32)

        self.g.retreive_to(&a[0], self.dim1)

        return a
