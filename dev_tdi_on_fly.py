import numpy as np
from bbhx.waveformbuild import BBHWaveformFD
from lisatools.utils.constants import *
from lisatools.detector import EqualArmlengthOrbits
import matplotlib.pyplot as plt
from bbhx.utils.interpolate import CubicSplineInterpolant


if __name__ == "__main__":

    Tobs = YRSID_SI
    
    # set parameters
    f_ref = 0.0  # let phenom codes set f_ref -> fmax = max(f^2A(f))
    phi_ref = 0.0  # phase at f_ref
    m1 = 1e6
    m2 = 5e5
    a1 = 0.2
    a2 = 0.4
    dist = 18e3 * PC_SI * 1e6  # 3e3 in Mpc
    inc = np.pi / 3.0
    beta = np.pi / 4.0  # ecliptic latitude
    lam = np.pi / 5.0  # ecliptic longitude
    psi = np.pi / 6.0  # polarization angle
    t_ref = 0.8 * YRSID_SI  # t_ref  (in the SSB reference frame)

    # frequencies to interpolate to
    freq_new = np.logspace(-4, 0, 10000)
    modes = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]
    orbits = EqualArmlengthOrbits()
    orbits.configure(linear_interp_setup=True)

    from bbhx.response.tdionfly import BBHTDIonTheFly
    wave_gen = BBHWaveformFD(
        amp_phase_kwargs=dict(run_phenomd=False),
        response_kwargs=dict(orbits=orbits),
        response_class=BBHTDIonTheFly,
    )

    wave = wave_gen(
        m1,
        m2,
        a1,
        a2,
        dist,
        phi_ref,
        f_ref,
        inc,
        lam,
        beta,
        psi,
        t_ref,
        freqs=freq_new,
        modes=modes,
        direct=False,
        output_splines=True,
        tdi_in_amp_phase=True,
        length=1024,
    )# [0]

    breakpoint()
