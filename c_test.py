import PhenomHM
import numpy as np
import numpy.testing as npt
from astropy.cosmology import Planck15 as cosmo
from scipy import constants as ct
import time
import pdb
import matplotlib.pyplot as plt

def test():
    df = 1e-7
    freq, phiRef, f_ref, m1, m2, chi1z, chi2z, distance, deltaF, inclination = np.arange(1e-5, 1e-1 + df, df), 0.0, 1e-5, 1e5, 5e5, 0.8, 0.8, cosmo.luminosity_distance(3.0).value*1e6*ct.parsec, -1.0, np.pi/3.

    l_vals = np.array([2, 2, 3, 4, 4, 3], dtype=np.uint32)
    m_vals = np.array([2, 1, 3, 4, 3, 2], dtype=np.uint32)


    phenomHM = PhenomHM.PhenomHM(freq,
     l_vals,
     m_vals)

    #for _ in range(5):
    st = time.perf_counter()
    phenomHM.gen_PhenomHM(m1,  # solar masses
                 m2,  # solar masses
                 chi1z,
                 chi2z,
                 distance,
                 inclination,
                 phiRef,
                 deltaF,
                 f_ref)
    print('cpu', time.perf_counter() - st)
    check = phenomHM.Likelihood()
    print('inner produce base check', check)

    hp, hc = phenomHM.Get_Waveform()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for i in range(len(m_vals)):
        ax1.loglog(freq, np.abs(hp[i]))
        ax2.semilogx(freq, np.unwrap(np.angle(hp[i])))
    plt.show()




if __name__ == "__main__":
    test()
