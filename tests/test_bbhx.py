# Tests

# Copyright (C) 2021 Michael L. Katz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import unittest
import numpy as np

from bbhx.waveformbuild import BBHWaveformFD
from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
from bbhx.response.fastfdresponse import LISATDIResponse
from bbhx.likelihood import Likelihood, HeterodynedLikelihood
from bbhx.utils.constants import *
from bbhx.utils.transform import *

from lisatools.sensitivity import get_sensitivity
from lisatools.detector import EqualArmlengthOrbits

try:
    import cupy as xp

    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    gpu_available = False
    import numpy as xp

np.random.seed(111222)


force_backend = "gpu" if gpu_available else "cpu"

class WaveformTest(unittest.TestCase):
    def test_full_waveform(self):
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
        freq_new = xp.logspace(-4, 0, 10000)
        modes = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]
        orbits = EqualArmlengthOrbits(force_backend=force_backend)
        orbits.configure(linear_interp_setup=True)

        wave_gen = BBHWaveformFD(
            amp_phase_kwargs=dict(run_phenomd=False),
            response_kwargs=dict(orbits=orbits),
            force_backend=force_backend,
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
            fill=True,
            squeeze=True,
            length=1024,
        )[0]

        self.assertTrue(np.all(~np.isnan(wave)))

    def test_phenom_hm(self):
        
        phenomhm = PhenomHMAmpPhase(force_backend=force_backend, run_phenomd=False)
        f_ref = 0.0  # let phenom codes set f_ref -> fmax = max(f^2A(f))
        phi_ref = 0.0  # phase at f_ref
        m1 = 1e6
        m2 = 5e5
        a1 = 0.2
        a2 = 0.4
        dist = 18e3 * PC_SI * 1e6  # 3e3 in Mpc
        t_ref = 0.8 * YRSID_SI

        phenomhm(m1, m2, a1, a2, dist, phi_ref, f_ref, t_ref, 1024)

        # get important quantities
        freqs = phenomhm.freqs_shaped  # shape (num_bin_all, length)
        amps = phenomhm.amp  # shape (num_bin_all, num_modes, length)
        phase = phenomhm.phase  # shape (num_bin_all, num_modes, length)
        tf = phenomhm.tf  # shape (num_bin_all, num_modes, length)
        self.assertTrue(np.all(~np.isnan(freqs)))
        self.assertTrue(np.all(~np.isnan(amps)))
        self.assertTrue(np.all(~np.isnan(phase)))
        self.assertTrue(np.all(~np.isnan(tf)))

    def test_fast_fd_response(self):

        phenomhm = PhenomHMAmpPhase(force_backend=force_backend, run_phenomd=False)
        f_ref = 0.0  # let phenom codes set f_ref -> fmax = max(f^2A(f))
        phi_ref = 0.0  # phase at f_ref
        m1 = 1e6
        m2 = 5e5
        a1 = 0.2
        a2 = 0.4
        dist = 18e3 * PC_SI * 1e6  # 3e3 in Mpc
        t_ref = 0.8 * YRSID_SI

        phenomhm(m1, m2, a1, a2, dist, phi_ref, f_ref, t_ref, 1024)

        # use phase/tf information from last waveform run
        freqs = phenomhm.freqs.copy()
        phase = phenomhm.phase.copy()
        tf = phenomhm.tf.copy()
        modes = phenomhm.modes

        phi_ref = 0.0
        inc = np.pi / 4
        beta = np.pi / 5
        lam = np.pi / 6
        psi = np.pi / 7

        length = phenomhm.freqs_shaped.shape[-1]

        orbits = EqualArmlengthOrbits(force_backend=force_backend)
        orbits.configure(linear_interp_setup=True)

        response = LISATDIResponse(orbits=orbits, force_backend=force_backend)

        response(
            freqs, inc, lam, beta, psi, phi_ref, length, phase=phase, tf=tf, modes=modes
        )

        self.assertTrue(np.all(~np.isnan(response.transferL1)))
        self.assertTrue(np.all(~np.isnan(response.transferL2)))
        self.assertTrue(np.all(~np.isnan(response.transferL3)))
        self.assertTrue(np.all(~np.isnan(response.phase)))
        self.assertTrue(np.all(~np.isnan(response.tf)))

    def test_direct_likelihood(self):
        orbits = EqualArmlengthOrbits(force_backend=force_backend)
        orbits.configure(linear_interp_setup=True)

        wave_gen = BBHWaveformFD(
            amp_phase_kwargs=dict(run_phenomd=False),
            response_kwargs=dict(orbits=orbits),
            force_backend=force_backend,
        )

        ######## generate data
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

        T_obs = 1.2  # years
        dt = 10.0

        n = int(T_obs * YRSID_SI / dt)
        data_freqs = xp.fft.rfftfreq(n, dt)[1:]  # remove DC

        # frequencies to interpolate to
        modes = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]
        waveform_kwargs = dict(
            modes=modes, direct=False, fill=True, squeeze=True, length=1024
        )

        data_channels = wave_gen(
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
            freqs=data_freqs,
            **waveform_kwargs
        )[0]

        try:
            data_freqs_cpu = data_freqs.get()
        except AttributeError:
            data_freqs_cpu = data_freqs

        ######## get noise information (need lisatools)
        PSD_A = get_sensitivity(data_freqs_cpu, sens_fn="A1TDISens")
        PSD_E = get_sensitivity(data_freqs_cpu, sens_fn="E1TDISens")
        PSD_T = get_sensitivity(data_freqs_cpu, sens_fn="T1TDISens")

        df = data_freqs[1] - data_freqs[0]

        psd = xp.asarray([PSD_A, PSD_E, PSD_T])

        # initialize Likelihood
        like = Likelihood(
            wave_gen,
            data_freqs,
            data_channels,
            psd,
            force_backend=force_backend,
        )

        # get params
        num_bins = 10
        params_in = np.tile(
            np.array(
                [m1, m2, a1, a2, dist, phi_ref, f_ref, inc, lam, beta, psi, t_ref]
            ),
            (num_bins, 1),
        )

        # change masses for test
        params_in[:, 0] *= 1 + 1e-4 * np.random.randn(num_bins)

        # get_ll and not __call__ to work with lisatools
        ll = like.get_ll(params_in.T, **waveform_kwargs)

        self.assertTrue(np.all(~np.isnan(ll)))

    def test_het_likelihood(self):

        orbits = EqualArmlengthOrbits(force_backend=force_backend)
        orbits.configure(linear_interp_setup=True)

        wave_gen = BBHWaveformFD(
            amp_phase_kwargs=dict(run_phenomd=False),
            response_kwargs=dict(orbits=orbits),
            force_backend=force_backend,
        )

        ######## generate data
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

        T_obs = 1.2  # years
        dt = 10.0

        n = int(T_obs * YRSID_SI / dt)
        data_freqs = xp.fft.rfftfreq(n, dt)[1:]  # remove DC

        # frequencies to interpolate to
        modes = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]
        waveform_kwargs = dict(
            modes=modes, direct=False, fill=True, squeeze=True, length=1024
        )

        data_channels = wave_gen(
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
            freqs=data_freqs,
            **waveform_kwargs
        )[0]

        try:
            data_freqs_cpu = data_freqs.get()
        except AttributeError:
            data_freqs_cpu = data_freqs

        ######## get noise information (need lisatools)
        PSD_A = get_sensitivity(data_freqs_cpu, sens_fn="A1TDISens")
        PSD_E = get_sensitivity(data_freqs_cpu, sens_fn="E1TDISens")
        PSD_T = get_sensitivity(data_freqs_cpu, sens_fn="T1TDISens")

        df = data_freqs[1] - data_freqs[0]

        psd = xp.asarray([PSD_A, PSD_E, PSD_T])

        # initialize Likelihood
        like = Likelihood(
            wave_gen,
            data_freqs,
            data_channels,
            psd,
            force_backend=force_backend,
        )

        # get params
        num_bins = 10
        params_in = np.tile(
            np.array(
                [m1, m2, a1, a2, dist, phi_ref, f_ref, inc, lam, beta, psi, t_ref]
            ),
            (num_bins, 1),
        )

        # change masses for test
        params_in[:, 0] *= 1 + 1e-4 * np.random.randn(num_bins)

        # get_ll and not __call__ to work with lisatools
        ll = like.get_ll(params_in.T, **waveform_kwargs)

        self.assertTrue(np.all(~np.isnan(ll)))

        reference_index = ll.argmax()

        reference_params = params_in[reference_index]

        # how many frequencies to use
        length_f_het = 128

        # initialize Likelihood
        like_het = HeterodynedLikelihood(
            wave_gen,
            data_freqs,
            data_channels,
            reference_params,
            length_f_het,
            force_backend=force_backend,
        )

        # get_ll and not __call__ to work with lisatools
        ll_het = like_het.get_ll(params_in.T, **waveform_kwargs)

        self.assertTrue(np.all(~np.isnan(ll)))
