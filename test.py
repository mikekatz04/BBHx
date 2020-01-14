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
    import cpuPhenomHM as PhenomHM


def test():
    df = 1e-5
    length = int(2 ** 21)
    data_length = int(2 * length)
    # FIXME core dump from python is happening at 2e5 - 3e5 ish
    data = np.fft.rfft(np.sin(2 * np.pi * 1e-3 * np.arange(data_length) * 0.1))

    key_order = [
        "ln_mT",
        "mr",
        "a1",
        "a2",
        "ln_distance",  # Gpc z=2
        "phiRef",
        "cos_inc",
        "lam",
        "sin_beta",
        "psi",
        "ln_tRef",
    ]

    initial_point = {
        "ln_mT": np.log(2.00000000e06),
        "mr": 1 / 3.00000000e00,
        "a1": 0.0,
        "a2": 0.0,
        "ln_distance": np.log(3.65943000e01),  # Gpc z=2
        "phiRef": 2.13954125e00,
        "cos_inc": np.cos(1.04719755e00),
        "lam": -2.43647481e-02,
        "sin_beta": np.sin(6.24341583e-01),
        "psi": 2.02958790e00,
        "ln_tRef": np.log(5.02462348e01),
    }
    M = 4e7
    q = 0.2
    freq, phiRef, f_ref, m1, m2, chi1z, chi2z, distance, deltaF = (
        np.logspace(-6, 0, int(2 ** 10)),
        3.09823412789,
        0.0,
        M / (1 + q),
        M * q / (1 + q),
        0.0,
        0.0,
        15.93461637 * 1e3 * 1e6 * ct.parsec,
        -1.0,
    )  # cosmo.luminosity_distance(2.0).value*1e6*ct.parsec, -1.0

    # freq = np.load('freqs.npy')
    t0 = 1.0
    tRef = 2.39284219993e1  # minutes to seconds
    merger_freq = 0.018 / ((m1 + m2) * 1.989e30 * ct.G / ct.c ** 3)
    Msec = (m1 + m2) * 1.989e30 * ct.G / ct.c ** 3
    f_ref = 0.0
    TDItag = 2
    l_vals = np.array([2, 3, 4, 2, 3, 4], dtype=np.uint32)  #
    m_vals = np.array([2, 3, 4, 1, 2, 3], dtype=np.uint32)  # ,

    Msec = (m1 + m2) * 1.989e30 * ct.G / ct.c ** 3
    upper_freq = 0.5 / Msec
    lower_freq = 1e-4 / Msec
    freqs = np.logspace(np.log10(lower_freq), np.log10(upper_freq), len(freq))
    data_freqs = np.logspace(np.log10(lower_freq), np.log10(upper_freq), length)

    data = data[:length]

    deltaF = np.zeros_like(data_freqs)
    deltaF[1:] = np.diff(data_freqs)
    deltaF[0] = deltaF[1]

    """AE_noise = np.genfromtxt('SnAE2017.dat').T
    T_noise = np.genfromtxt('SnAE2017.dat').T

    from scipy.interpolate import CubicSpline

    AE_noise = CubicSpline(AE_noise[0], AE_noise[1])
    T_noise = CubicSpline(T_noise[0], T_noise[1])

    AE_ASDinv = 1./np.sqrt(AE_noise(data_freqs))*np.sqrt(deltaF)
    AE_ASDinv = 1./np.sqrt(AE_noise(data_freqs))*np.sqrt(deltaF)
    T_ASDinv = 1./np.sqrt(T_noise(data_freqs))*np.sqrt(deltaF)"""

    AE_ASDinv = (
        1.0
        / np.sqrt(tdi.noisepsd_AE(data_freqs, model="SciRDv1", includewd=3))
        * np.sqrt(deltaF)
    )
    AE_ASDinv = (
        1.0
        / np.sqrt(tdi.noisepsd_AE(data_freqs, model="SciRDv1", includewd=3))
        * np.sqrt(deltaF)
    )
    T_ASDinv = (
        1.0 / np.sqrt(tdi.noisepsd_T(data_freqs, model="SciRDv1")) * np.sqrt(deltaF)
    )

    # AE_ASDinv = np.ones_like(data_freqs)
    # T_ASDinv = np.ones_like(data_freqs)

    t_obs_start = 0.9
    t_obs_end = 0.0

    inc = np.cos(2.98553920)
    lam = 5.900332547
    beta = np.sin(-1.3748820938)
    psi = 0.139820023

    key_order = ["cos_inc", "lam", "sin_beta", "psi", "ln_tRef"]
    recycler = Recycler(key_order)

    converter = Converter(key_order, tLtoSSB=True)

    tRef_sampling_frame = tRef

    array = np.array([inc, lam, beta, psi, np.log(tRef)])

    array = recycler.recycle(array)
    array = converter.convert(array)
    inc, lam, beta, psi, tRef_wave_frame = array
    print("init:", inc, lam, beta, psi, tRef_wave_frame)
    nwalkers = 20
    ndevices = 1

    # data_A, data_E, data_T = np.load('data_set.npy')
    data_A, data_E, data_T = data, data, data
    phenomHM = PhenomHM.PhenomHM(
        len(freq),
        l_vals,
        m_vals,
        data_freqs,
        data_A,
        data_E,
        data_T,
        AE_ASDinv,
        AE_ASDinv,
        T_ASDinv,
        TDItag,
        t_obs_start,
        t_obs_end,
        nwalkers,
        ndevices,
    )

    num = 1
    st = time.perf_counter()
    # phiRef = np.linspace(0.0, 2*np.pi, num)
    # snrs = np.zeros_like(phiRef)

    print(
        m1,
        m2,
        chi1z,
        chi1z,
        distance,
        phiRef,
        inc,
        lam,
        beta,
        psi,
        tRef_wave_frame,
        tRef_sampling_frame,
    )

    nwalkers = nwalkers * ndevices
    freqs_in = np.tile(freqs, nwalkers)
    m1_in = np.full(nwalkers, m1)
    m2_in = np.full(nwalkers, m2)
    chi1z_in = np.full(nwalkers, chi1z)
    chi2z_in = np.full(nwalkers, chi1z)
    distance_in = np.full(nwalkers, distance)
    phiRef_in = np.full(nwalkers, phiRef)
    f_ref_in = np.full(nwalkers, f_ref)
    inc_in = np.full(nwalkers, inc)
    lam_in = np.full(nwalkers, lam)
    beta_in = np.full(nwalkers, beta)
    psi_in = np.full(nwalkers, psi)
    t0_in = np.full(nwalkers, t0)
    tRef_wave_frame_in = np.full(nwalkers, tRef_wave_frame)
    tRef_sampling_frame_in = np.full(nwalkers, tRef_sampling_frame)
    merger_freq_in = np.full(nwalkers, merger_freq)

    like2 = phenomHM.WaveformThroughLikelihood(
        freqs_in,
        m1_in,
        m2_in,
        chi1z_in,
        chi2z_in,
        distance_in,
        phiRef_in,
        f_ref_in,
        inc_in,
        lam_in,
        beta_in,
        psi_in,
        t0_in,
        tRef_wave_frame_in,
        tRef_sampling_frame_in,
        merger_freq_in,
        return_TDI=False,
    )

    phenomHM.input_data(
        data_freqs, data_A, data_E, data_T, AE_ASDinv, AE_ASDinv, T_ASDinv
    )

    for i in range(num):
        """
        phenomHM.gen_amp_phase(
            freqs_in, m1_in, m2_in, chi1z_in, chi2z_in, distance_in, phiRef_in, f_ref_in
        )

        # phenomHM.Combine()

        phenomHM.LISAresponseFD(
            inc_in,
            lam_in,
            beta_in,
            psi_in,
            t0_in,
            tRef_wave_frame_in,
            tRef_sampling_frame_in,
            merger_freq_in,
        )


        phenomHM.setup_interp_wave()

        phenomHM.setup_interp_response()

        phenomHM.perform_interp()

        like = phenomHM.Likelihood()

        """

        like2 = phenomHM.WaveformThroughLikelihood(
            freqs_in,
            m1_in,
            m2_in,
            chi1z_in,
            chi2z_in,
            distance_in,
            phiRef_in,
            f_ref_in,
            inc_in,
            lam_in,
            beta_in,
            psi_in,
            t0_in,
            tRef_wave_frame_in,
            tRef_sampling_frame_in,
            merger_freq_in,
            return_TDI=False,
        )

        """try:
            assert(np.allclose(like2[1], like2[1][0]))
            print('\n\n\n GOOD \n\n\n')
        except AssertionError:
            import pdb; pdb.set_trace()"""
        # snrs[i] = like2[1]
        # print(like2**(1/2))

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

    # np.save('snrs', np.array([phiRef, snrs]))
    t = time.perf_counter() - st
    print("gpu per waveform:", t / (num * nwalkers))
    # print(like)
    # gpu_X, gpu_Y, gpu_Z = phenomHM.GetTDI()
    amp, phase = phenomHM.GetAmpPhase()
    import pdb

    pdb.set_trace()
    # A, E, T = phenomHM.GetTDI()
    # np.save('wave_test', np.asarray([gpu_X, gpu_Y, gpu_Z]))
    # print('2gpu per waveform:', t/num)
    # import pdb; pdb.set_trace()
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

    # np.save('amp-phase', np.concatenate([np.array([freq]), amp, phase], axis=0))
    # np.save('TDI', np.array([data_freqs, A, E, T]))
    pdb.set_trace()


if __name__ == "__main__":
    test()
