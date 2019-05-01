import sys; sys.settrace
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
    freq, phiRef, f_ref, m1, m2, chi1z, chi2z, distance, deltaF, inclination = np.logspace(-4.3, 0, 16384), 0.0, 1e-3, 1e5, 5e5, 0.8, 0.8, cosmo.luminosity_distance(3.0).value*1e6*ct.parsec, -1.0, np.pi/3.

    #freq = np.load('freqs.npy')

    inc = inclination
    lam = 0.3
    beta = 0.2
    psi = np.pi/3.
    t0 = 0.9*ct.Julian_year
    tRef = 5.0*60.0  # minutes to seconds
    merger_freq = 0.018/((m1+m2)*1.989e30*ct.G/ct.c**3)
    print('merger_freq', merger_freq)
    TDItag = 2

    l_vals = np.array([2, 3, 4, 4, 3], dtype=np.uint32) #
    m_vals = np.array([2, 3, 4, 3, 2], dtype=np.uint32) #,

    df = 1e-5

    # FIXME core dump from python is happening at 2e5 - 3e5 ish
    data = np.fft.rfft(np.sin(2*np.pi*1e-3 * np.arange(1e5)*0.1))

    interp_freq = 1e-5+np.arange(len(data))*df

    AE_ASDinv = 1./np.sqrt(tdi.noisepsd_AE(interp_freq, model='SciRDv1'))
    AE_ASDinv = 1./np.sqrt(tdi.noisepsd_AE(interp_freq, model='SciRDv1'))
    T_ASDinv = 1./np.sqrt(tdi.noisepsd_T(interp_freq, model='SciRDv1'))

    phenomHM = gpuPhenomHM.PhenomHM(len(freq),
     l_vals,
     m_vals, data, AE_ASDinv, AE_ASDinv, T_ASDinv)

    num = 100
    st = time.perf_counter()
    for i in range(num):

        phenomHM.gen_amp_phase(freq, m1,  # solar masses
                     m2,  # solar masses
                     chi1z,
                     chi2z,
                     distance,
                     phiRef,
                     f_ref)
        phenomHM.setup_interp_wave()
        phenomHM.LISAresponseFD(inc, lam, beta, psi, t0, tRef, merger_freq, TDItag)
        phenomHM.setup_interp_response()
        phenomHM.perform_interp(1e-5, df, len(interp_freq))
        like = phenomHM.Likelihood()

    t = time.perf_counter() - st
    print('gpu per waveform:', t/num)
    print(like)
    gpu_X, gpu_Y, gpu_Z = phenomHM.GetTDI()
    gpu_amp, gpu_phase = phenomHM.GetAmpPhase()
    np.save('wave_test', np.asarray([gpu_X, gpu_Y, gpu_Z]))
    #print('2gpu per waveform:', t/num)
    import pdb; pdb.set_trace()

    """
    to_gpu=0
    to_interp = 1
    cpu_phenomHM = gpuPhenomHM.GPUPhenomHM(len(freq),
     l_vals,
     m_vals,
     to_gpu, to_interp, data, AE_ASDinv, AE_ASDinv, T_ASDinv)

    cpu_phenomHM.add_interp(len(interp_freq))
    num = 100
    st = time.perf_counter()
    for i in range(num):
        print(i)
        cpu_phenomHM.cpu_gen_PhenomHM(freq, m1,  # solar masses
                     m2,  # solar masses
                     chi1z,
                     chi2z,
                     distance,
                     inclination,
                     phiRef,
                     deltaF,
                     f_ref)
        #cpu_phenomHM.cpu_setup_interp_wave()
        #cpu_phenomHM.cpu_LISAresponseFD(inc, lam, beta, psi, t0, tRef, merger_freq, TDItag)
        #cpu_phenomHM.cpu_setup_interp_response()
        #cpu_phenomHM.cpu_perform_interp(1e-5, df, len(interp_freq))

    t = time.perf_counter() - st
    print('gpu per waveform:', t/num)
    #print(like)
    #cpu_X, cpu_Y, cpu_Z = cpu_phenomHM.Get_Waveform()
    #np.save('wave_test_cpu', np.asarray([cpu_X, cpu_Y, cpu_Z]))
    #amp = np.abs(cpu_amp).flatten()
    #phase = np.unwrap(np.arctan2(cpu_amp.real, cpu_amp.imag)).flatten()
    #cpu_phenomHM.interp_wave(amp, phase)
    print('cpu', time.perf_counter() - st)
    """
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


if __name__ == "__main__":
    test()
