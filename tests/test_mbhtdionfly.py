# Tests for bbhx.mbhtdionfly (time-domain MBH TDI on the fly)

# Copyright (C) 2026 Michael L. Katz
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

from bbhx.mbhtdionfly import MBHTDIonFly

try:
    from phentax.waveform import IMRPhenomTHM

    phentax_available = True
except (ImportError, ModuleNotFoundError):
    phentax_available = False


class MBHTDIonFlyConstructionTest(unittest.TestCase):
    def test_construction_and_backend(self):
        # construction does not touch the waveform generator, so a
        # placeholder is fine here
        gen = MBHTDIonFly(None, None, None, 2.5, 1.0e6, 0.0, force_backend="cpu")
        self.assertEqual(gen.backend.name, "bbhx_cpu")
        self.assertEqual(gen.response_backend, "cpu")
        self.assertAlmostEqual(gen.sampling_frequency, 1 / 2.5)


@unittest.skipUnless(phentax_available, "phentax not installed (pip install 'bbhx[phentax]')")
class MBHTDIonFlyWaveformTest(unittest.TestCase):
    def test_time_domain_tdi(self):
        from lisaconstants import ASTRONOMICAL_YEAR
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.response.tdiconfig import TDIConfig

        Tobs = ASTRONOMICAL_YEAR / 12
        dt = 2.5
        tol = 1e-12

        wave_gen = IMRPhenomTHM(
            higher_modes=[21, 33, 44],
            include_negative_modes=True,
            t_low_fit=True,
            coarse_grain=True,
            atol=tol,
            rtol=tol,
            T=Tobs,
        )

        gen = MBHTDIonFly(
            wave_gen,
            EqualArmlengthOrbits(force_backend="cpu"),
            TDIConfig("2nd generation"),
            dt,
            Tobs,
            0.0,
            force_backend="cpu",
        )

        params = dict(
            m1=2.0e6,
            m2=8.0e5,
            s1z=0.3,
            s2z=0.1,
            distance=3.0e4,
            phi_ref=1.2,
            inclination=0.7,
            ra=1.0,
            dec=0.3,
            psi=0.5,
        )
        t_merge = 2.0e6

        upsample_t_arr = np.arange(int(Tobs / dt)) * dt
        tdi = gen(**params, t_merge=t_merge, upsample_t_arr=upsample_t_arr, combine=True)

        self.assertEqual(tdi.shape, (3, len(upsample_t_arr)))
        self.assertTrue(np.all(np.isfinite(tdi)))
        # signal must be non-trivial in every TDI channel
        for i in range(3):
            self.assertGreater(np.count_nonzero(tdi[i]), 0)

        # per-mode (uncombined) output stacks the modes on the leading axis
        tdi_modes = gen(**params, t_merge=t_merge, upsample_t_arr=upsample_t_arr, combine=False)
        self.assertEqual(tdi_modes.shape[1:], (3, len(upsample_t_arr)))
        np.testing.assert_allclose(
            np.asarray(tdi_modes.sum(axis=0)), np.asarray(tdi), rtol=1e-13, atol=0.0
        )


if __name__ == "__main__":
    unittest.main()
