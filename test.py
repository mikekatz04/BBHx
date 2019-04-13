import gpuPhenomHM
import numpy as np
import numpy.testing as npt
from astropy.cosmology import Planck15 as cosmo
from scipy import constants as ct
import time

def test():
    df = 1e-7
    freq, phiRef, f_ref, m1, m2, chi1z, chi2z, distance, deltaF, inclination = np.arange(1e-5, 1e-1 + df, df), 0.0, 1e-5, 1e5, 5e5, 0.8, 0.8, cosmo.luminosity_distance(3.0).value*1e6*ct.parsec, -1.0, np.pi/3.

    l_vals = np.array([2, 2, 3, 4, 4, 3], dtype=np.uint32)
    m_vals = np.array([2, 1, 3, 4, 3, 2], dtype=np.uint32)

    to_gpu = 2

    to_gpu=1
    gpu_phenomHM = gpuPhenomHM.GPUPhenomHM(freq,
     l_vals,
     m_vals,
     to_gpu)

    #for _ in range(5):
    for i in range(10):
        st = time.perf_counter()
        gpu_phenomHM.gpu_gen_PhenomHM(m1,  # solar masses
                     m2,  # solar masses
                     chi1z,
                     chi2z,
                     distance,
                     inclination,
                     phiRef,
                     deltaF,
                     f_ref)
        print('gpu', time.perf_counter() - st)
    gpu_hp, gpu_hc = gpu_phenomHM.gpu_Get_Waveform()

    to_gpu=0
    cpu_phenomHM = gpuPhenomHM.GPUPhenomHM(freq,
     l_vals,
     m_vals,
     to_gpu)
    st = time.perf_counter()
    cpu_phenomHM.cpu_gen_PhenomHM(m1,  # solar masses
                 m2,  # solar masses
                 chi1z,
                 chi2z,
                 distance,
                 inclination,
                 phiRef,
                 deltaF,
                 f_ref)
    print('cpu', time.perf_counter() - st)
    cpu_hp, cpu_hc = cpu_phenomHM.Get_Waveform()

    assert(np.allclose(cpu_hp, gpu_hp))
    assert(np.allclose(cpu_hc, gpu_hc))
    print('CPU MATCHES GPU!!!!')


if __name__ == "__main__":
    test()
