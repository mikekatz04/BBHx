import GPUPhenomHM
import numpy as np
import numpy.testing as npt
from astropy.cosmology import Planck15 as cosmo
from scipy import constants as ct
import time
import pdb
import matplotlib.pyplot as plt

def test():
    num_freqs = 1024
    freq, phiRef, f_ref, m1, m2, chi1z, chi2z, distance, deltaF, inclination = np.logspace(-6, -1, num_freqs), 0.0, 1e-5, 1e5, 5e5, 0.8, 0.8, cosmo.luminosity_distance(3.0).value*1e6*ct.parsec, -1.0, np.pi/3.

    l_vals = np.array([2, 2, 3, 4, 4, 3], dtype=np.uint32)
    m_vals = np.array([2, 1, 3, 4, 3, 2], dtype=np.uint32)



    phenomHM = GPUPhenomHM.GPUPhenomHM(num_freqs,
     l_vals,
     m_vals)

    #for _ in range(5):
    st = time.perf_counter()
    phenomHM.cpu_gen_PhenomHM(freq, m1,  # solar masses
                 m2,  # solar masses
                 chi1z,
                 chi2z,
                 distance,
                 inclination,
                 phiRef,
                 deltaF,
                 f_ref)
    print('cpu', time.perf_counter() - st)

    amp, phase = phenomHM.Get_Waveform()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for i in range(6):#len(m_vals)):
        ax1.loglog(freq, amp[i])
        ax2.semilogx(freq, phase[i])
    plt.show()
    pdb.set_trace()




if __name__ == "__main__":
    test()
