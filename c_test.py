import numpy as np
import numpy.testing as npt
from astropy.cosmology import Planck15 as cosmo
from scipy import constants as ct
import time
import tdi
import pdb
from phenomhm.utils.convert import Converter, Recycler

try:
    import gpuPhenomHM as PhenomHM
except ImportError:
    import PhenomHM


def test():
    df = 1e-5
    data_length = int(1.2e5)
    # FIXME core dump from python is happening at 2e5 - 3e5 ish
    data = np.fft.rfft(np.sin(2*np.pi*1e-3 * np.arange(data_length)*0.1))

    df = 1e-4
    q = 1/3.00000000e+00
    freq, phiRef, f_ref, m1, m2, chi1z, chi2z, distance, deltaF = np.logspace(-6, 0, int(2**11)), 1.654545, 1e-3, 2.00000000e+06/(1+q), 4e6*q/(1+q), 0.0, 0.0,  3.65943000e+04*1e6*ct.parsec, -1. # cosmo.luminosity_distance(2.0).value*1e6*ct.parsec, -1.0

    #freq = np.load('freqs.npy')

    inc =  - 1.04719755
    lam = -2.43647481e-02
    beta = -6.24341583e-01
    psi = 5*np.pi/4.
    t0 = 1.0
    tRef = 10.0# minutes to seconds
    merger_freq = 0.018/((m1+m2)*1.989e30*ct.G/ct.c**3)
    f_ref = 0.0
    TDItag = 2

    l_vals = np.array([2, 3, 4, 2, 3, 4], dtype=np.uint32) #
    m_vals = np.array([2, 3, 4, 1, 2, 3], dtype=np.uint32) #,

    key_order = ['inc', 'lam', 'beta', 'psi', 'ln_tRef']

    recycler = Recycler(key_order)

    converter = Converter(key_order, tLtoSSB=True)

    array = np.array([inc, lam, beta, psi, np.log(tRef)])

    array = recycler.recycle(array)
    array = converter.convert(array)
    inc, lam, beta, psi, tRef = array

    print('init:', inc, lam, beta, psi, tRef)


    #data_freqs = np.fft.rfftfreq(data_length, d=0.1)
    #data_freqs[0] = 1e-8
    data_freqs = np.logspace(-6, -1, len(data))

    deltaF = np.zeros_like(data_freqs)
    deltaF[1:] = np.diff(data_freqs)
    deltaF[0] = deltaF[1]
    AE_ASDinv = 1./np.sqrt(tdi.noisepsd_AE(data_freqs, model='SciRDv1'))*np.sqrt(deltaF)
    AE_ASDinv = 1./np.sqrt(tdi.noisepsd_AE(data_freqs, model='SciRDv1'))*np.sqrt(deltaF)
    T_ASDinv = 1./np.sqrt(tdi.noisepsd_T(data_freqs, model='SciRDv1'))*np.sqrt(deltaF)

    phenomHM = PhenomHM.PhenomHM(len(freq),
     l_vals,
     m_vals, data_freqs, data, data, data, AE_ASDinv, AE_ASDinv, T_ASDinv, TDItag)

    num = 10
    st = time.perf_counter()
    for i in range(num):
        """phenomHM.gen_amp_phase(freq, m1,  # solar masses
                     m2,  # solar masses
                     chi1z,
                     chi2z,
                     distance,
                     phiRef,
                     f_ref)
        #phenomHM.Combine()
        phenomHM.setup_interp_wave()
        phenomHM.LISAresponseFD(inc, lam, beta, psi, t0, tRef, merger_freq)
        phenomHM.setup_interp_response()
        phenomHM.perform_interp()
        like = phenomHM.Likelihood()"""

        like2 = phenomHM.WaveformThroughLikelihood(freq, m1, m2,  chi1z, chi2z, distance, phiRef,f_ref, inc, lam, beta, psi, t0, tRef, merger_freq)
        print(like2)
        """
        phenomHM.LISAresponseFD(inc, lam, beta, psi, t0, tRef, merger_freq)

        phenomHM.setup_interp_response()

        phenomHM.perform_interp()

        like = phenomHM.Likelihood()

        like2 = phenomHM.WaveformThroughLikelihood(freq, m1,  # solar masses
                     m2,  # solar masses
                     chi1z,
                     chi2z,
                     distance,
                     phiRef,
                     f_ref, inc, lam, beta, psi, t0, tRef, merger_freq)

        assert(all(like == like2))
        """

    t = time.perf_counter() - st
    print('gpu per waveform:', t/num)
    #print(like)
    #gpu_X, gpu_Y, gpu_Z = phenomHM.GetTDI()
    amp, phase = phenomHM.GetAmpPhase()
    A, E, T = phenomHM.GetTDI()
    #np.save('wave_test', np.asarray([gpu_X, gpu_Y, gpu_Z]))
    #print('2gpu per waveform:', t/num)
    #import pdb; pdb.set_trace()
    """
    import matplotlib.pyplot as plt
    for ampi in amp:
        plt.loglog(freq, ampi)
    plt.show()
    plt.close()
    for i, phasei in enumerate(phase):
        plt.semilogx(freq, phasei, label=str(i))
    plt.legend()
    plt.show()
    plt.close()
    for i, Ai in enumerate(A):
        plt.loglog(data_freqs, np.abs(Ai), label=str(i))
    plt.show()
    plt.close()
    plt.loglog(data_freqs, np.abs(A[0]))
    plt.loglog(data_freqs, np.abs(E[0]))
    plt.loglog(data_freqs, np.abs(T[0]))
    plt.show()

    h_h_all = 0
    d_h_all = 0
    d_d_all = 0
    for chan in [A, E, T]:
        for i in range(len(m_vals)):
            for j in range(len(m_vals)):
                h_h = np.dot(chan[i].conj(), chan[j]).real
                h_h_all += h_h
        data = np.sum(chan, axis=0)
        d_d = np.dot(data.conj(), data).real
        d_d_all += d_d
        d_h_t = np.asarray([np.dot(data.conj(), chani) for chani in chan])
        d_h = np.sum(d_h_t).real
        d_h_all += d_h
    """
    """
    A_out, E_out, T_out = [], [], []
    for i in [2, 3, 4]:
        phenomHM = PhenomHM.PhenomHM(len(freq),
         np.array([i], dtype=np.uint32),
         np.array([i], dtype=np.uint32), data_freqs, data, data, data, AE_ASDinv, AE_ASDinv, T_ASDinv, TDItag)
        A, E, T = phenomHM.WaveformThroughLikelihood(freq, m1,  # solar masses
                     m2,  # solar masses
                     chi1z,
                     chi2z,
                     distance,
                     phiRef,
                     f_ref, inc, lam, beta, psi, t0, tRef, merger_freq, return_TDI=True)
        A_out.append(A)
        E_out.append(E)
        T_out.append(T)
    """
    np.save('TDI', np.array([A, E, T]))
    pdb.set_trace()

if __name__ == "__main__":
    test()
