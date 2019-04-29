import gpuPhenomHM
import numpy as np
import numpy.testing as npt
from astropy.cosmology import Planck15 as cosmo
from scipy import constants as ct
import time
import tdi
import pdb

def test():
    df = 1e-4
    freq, phiRef, f_ref, m1, m2, chi1z, chi2z, distance, deltaF, inclination = np.logspace(-5, -1, 16384), 0.0, 1e-3, 1e6, 5e6, 0.8, 0.8, cosmo.luminosity_distance(3.0).value*1e6*ct.parsec, -1.0, np.pi/3.

    inc = inclination
    lam = 0.3
    beta = 0.2
    psi = np.pi/3.
    tc = 0.9*ct.Julian_year
    tShift = 0.0
    TDItag = 2

    l_vals = np.array([2, 3, 4, 4, 3], dtype=np.uint32) #
    m_vals = np.array([2, 3, 4, 3, 2], dtype=np.uint32) #,

    df = 1e-8

    # FIXME core dump from python is happening at 2e5 - 3e5 ish
    data = np.fft.rfft(np.sin(2*np.pi*1e-3 * np.arange(65536)*0.1))

    interp_freq = 1e-5+np.arange(len(data))*1e-8

    AET_ASDinv = 1./np.sqrt(tdi.noisepsd_AE(interp_freq, model='SciRDv1'))
    to_gpu=0
    to_interp = 1
    cpu_phenomHM = gpuPhenomHM.GPUPhenomHM(len(freq),
     l_vals,
     m_vals,
     to_gpu, to_interp, data, AET_ASDinv)

    cpu_phenomHM.add_interp(len(interp_freq))

    #cpu_phenomHM.add_interp(interp_freq, len(interp_freq))
    for i in range(1):
        st = time.perf_counter()
        cpu_phenomHM.cpu_gen_PhenomHM(freq, m1,  # solar masses
                     m2,  # solar masses
                     chi1z,
                     chi2z,
                     distance,
                     inclination,
                     phiRef,
                     deltaF,
                     f_ref)
        cpu_phenomHM.cpu_setup_interp_wave()
        cpu_phenomHM.cpu_LISAresponseFD(inc, lam, beta, psi, tc, tShift, TDItag)
        cpu_phenomHM.cpu_setup_interp_response()
        cpu_phenomHM.cpu_perform_interp(1e-5, 1e-7, len(interp_freq))

    cpu_X, cpu_Y, cpu_Z = cpu_phenomHM.Get_Waveform()

    #amp = np.abs(cpu_amp).flatten()
    #phase = np.unwrap(np.arctan2(cpu_amp.real, cpu_amp.imag)).flatten()
    #cpu_phenomHM.interp_wave(amp, phase)
    print('cpu', time.perf_counter() - st)
    #print(np.sum(np.real(cpu_amp.conj()*cpu_amp)))

    #for i in range(4):
    #    st = time.perf_counter()
    #    spline = np.interp(interp_freq, freq, amp[:len(freq)])
    #    print('scipy', time.perf_counter() - st)
    """
    to_gpu=1
    to_interp = 0
    gpu_phenomHM = gpuPhenomHM.GPUPhenomHM(len(freq),
     l_vals,
     m_vals,
     to_gpu, to_interp, data)


    num = 100
    #for _ in range(5):
    st = time.perf_counter()
    for i in range(num):

        gpu_phenomHM.gpu_gen_PhenomHM(freq, m1,  # solar masses
                     m2,  # solar masses
                     chi1z,
                     chi2z,
                     distance,
                     inclination,
                     phiRef,
                     deltaF,
                     f_ref)

        check = gpu_phenomHM.Likelihood(len(freq))
    t = time.perf_counter() - st
    print('gpu whole:', t/num)
    """
    #import pdb; pdb.set_trace()
    #gpu_amp, gpu_phase = gpu_phenomHM.gpu_Get_Waveform()
    #assert(np.allclose(cpu_amp, gpu_amp))
    #assert(np.allclose(cpu_phase, gpu_phase))
    #print('CPU MATCHES GPU!!!!')

    to_gpu=1
    to_interp = 1
    gpu_phenomHM = gpuPhenomHM.GPUPhenomHM(len(freq),
     l_vals,
     m_vals,
     to_gpu, to_interp, data, AET_ASDinv)

    gpu_phenomHM.add_interp(len(interp_freq))
    num = 1000
    st = time.perf_counter()
    for i in range(num):

        gpu_phenomHM.gpu_gen_PhenomHM(freq, m1,  # solar masses
                     m2,  # solar masses
                     chi1z,
                     chi2z,
                     distance,
                     inclination,
                     phiRef,
                     deltaF,
                     f_ref)
        gpu_phenomHM.gpu_setup_interp_wave()
        gpu_phenomHM.gpu_LISAresponseFD(inc, lam, beta, psi, tc, tShift, TDItag)
        gpu_phenomHM.gpu_setup_interp_response()
        gpu_phenomHM.gpu_perform_interp(1e-5, 1e-8, len(interp_freq))
        like = gpu_phenomHM.Likelihood(len(interp_freq))

    t = time.perf_counter() - st
    print('gpu per waveform:', t/num)
    print(like)
    gpu_hI = gpu_phenomHM.gpu_Get_Waveform()
    #print('2gpu per waveform:', t/num)
    #import pdb; pdb.set_trace()


if __name__ == "__main__":
    test()
