import gpuadder
import numpy as np
import numpy.testing as npt
from astropy.cosmology import Planck15 as cosmo
from scipy import constants as ct

def test():
    freq, phiRef, f_ref, m1, m2, chi1z, chi2z, distance, deltaF, inclination = np.arange(1, int(1e4)+1, 1)*1e-5, 0.0, 1e-5, 1e5, 5e5, 0.8, 0.8, cosmo.luminosity_distance(3.0).value*1e6*ct.parsec, -1.0, 0.0

    l_vals = np.array([2, 2, 3, 4, 4, 3], dtype=np.uint32)
    m_vals = np.array([2, 1, 3, 4, 3, 2], dtype=np.uint32)

    to_gpu = 0

    arr = np.array([1,2,2,2], dtype=np.int32)
    adder = gpuadder.GPUPhenomHM(arr,
     freq,
     l_vals,
     m_vals,
     to_gpu)


    for _ in range(5):
        adder.cpu_gen_PhenomHM(m1,  # solar masses
                     m2,  # solar masses
                     chi1z,
                     chi2z,
                     distance,
                     inclination,
                     phiRef,
                     deltaF,
                     f_ref)
        print(adder.Get_Waveform())

    adder.increment()

    adder.retreive_inplace()
    results2 = adder.retreive()

    npt.assert_array_equal(arr, [2,3,3,3])
    npt.assert_array_equal(results2, [2,3,3,3])

    #import pdb; pdb.set_trace()

if __name__ == "__main__":
    test()
