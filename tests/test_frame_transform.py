"""SSB <-> LISA-frame transforms (``bbhx.utils.transform``).

Checks the frame definition against geometry that is independent of the
closed-form expressions: the constellation normal and center position read
off the stock equal-arm orbit file, an explicit rotation-matrix construction
of the LISA frame, and the local-north polarization rotation.
"""

import unittest
import warnings

import numpy as np

from bbhx.utils.constants import C_SI, ConstOmega, YRSID_SI
from bbhx.utils.transform import (
    LISA_to_SSB,
    SSB_to_LISA,
    tLfromSSBframe,
    tSSBfromLframe,
    mbh_sky_mode_transform,
)

ZETA = np.pi / 3.0


def _unit(lam, beta):
    return np.stack(
        [np.cos(beta) * np.cos(lam), np.cos(beta) * np.sin(lam), np.sin(beta)], axis=-1
    )


def _lisa_frame_axes(alpha):
    """x_L, y_L, z_L of R_z(alpha) R_y(-zeta) R_z(-alpha) applied to the SSB axes."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cz, sz = np.cos(ZETA), np.sin(ZETA)
    z_l = np.stack([-sz * ca, -sz * sa, cz * np.ones_like(ca)], axis=-1)
    x_l = np.stack([cz * ca**2 + sa**2, -(1 - cz) * sa * ca, sz * ca], axis=-1)
    y_l = np.cross(z_l, x_l)
    return x_l, y_l, z_l


def _wrap(x, period):
    return (x + period / 2) % period - period / 2


class FrameTransformTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        n = 500
        cls.t = rng.uniform(0.05, 1.9, n) * YRSID_SI
        cls.lam = rng.uniform(0, 2 * np.pi, n)
        cls.beta = np.arcsin(rng.uniform(-0.98, 0.98, n))
        cls.psi = rng.uniform(0, np.pi, n)

    def test_round_trip(self):
        for t0 in (0.0, 0.37):
            tL, lamL, betaL, psiL = SSB_to_LISA(self.t, self.lam, self.beta, self.psi, t0=t0)
            t2, lam2, beta2, psi2 = LISA_to_SSB(tL, lamL, betaL, psiL, t0=t0)
            self.assertLess(np.abs(t2 - self.t).max(), 1e-4)
            self.assertLess(np.abs(_wrap(lam2 - self.lam, 2 * np.pi)).max(), 1e-10)
            self.assertLess(np.abs(beta2 - self.beta).max(), 1e-10)
            self.assertLess(np.abs(_wrap(psi2 - self.psi, np.pi)).max(), 1e-10)

    def test_per_row_t0_broadcasts(self):
        t0 = np.random.default_rng(3).uniform(-0.5, 0.5, self.t.size)
        tL, lamL, betaL, psiL = SSB_to_LISA(self.t, self.lam, self.beta, self.psi, t0=t0)
        # row i with its own t0 == the scalar call at that t0
        for i in (0, 17, 400):
            ref = SSB_to_LISA(self.t[i], self.lam[i], self.beta[i], self.psi[i], t0=float(t0[i]))
            for a, b in zip((tL[i], lamL[i], betaL[i], psiL[i]), ref):
                self.assertAlmostEqual(float(a), float(b), places=10)
        t2, _, beta2, _ = LISA_to_SSB(tL, lamL, betaL, psiL, t0=t0)
        self.assertLess(np.abs(t2 - self.t).max(), 1e-4)
        self.assertLess(np.abs(beta2 - self.beta).max(), 1e-10)

    def test_time_retardation_is_n_dot_rc_over_c(self):
        alpha = ConstOmega * self.t
        r_c = 1.495978707e11 * np.stack([np.cos(alpha), np.sin(alpha), 0 * alpha], axis=-1)
        n = _unit(self.lam, self.beta)
        tL = tLfromSSBframe(self.t, self.lam, self.beta)
        self.assertLess(np.abs((self.t - np.einsum("ij,ij->i", n, r_c) / C_SI) - tL).max(), 1e-6)
        # inverse to first order in Omega R / c
        self.assertLess(np.abs(tSSBfromLframe(tL, self.lam, self.beta) - self.t).max(), 1e-4)

    def test_sky_is_the_rotation_matrix_frame(self):
        tL, lamL, betaL, psiL = SSB_to_LISA(self.t, self.lam, self.beta, self.psi)
        # alpha is evaluated at the SSB time of each row
        x_l, y_l, z_l = _lisa_frame_axes(ConstOmega * self.t)
        n = _unit(self.lam, self.beta)
        self.assertLess(np.abs(np.sin(betaL) - np.einsum("ij,ij->i", n, z_l)).max(), 1e-12)
        lam_geom = np.arctan2(np.einsum("ij,ij->i", n, y_l), np.einsum("ij,ij->i", n, x_l))
        self.assertLess(np.abs(_wrap(lamL - lam_geom, 2 * np.pi)).max(), 1e-12)

    def test_psi_rotation_is_local_north_to_local_north(self):
        tL, lamL, betaL, psiL = SSB_to_LISA(self.t, self.lam, self.beta, self.psi)
        _, _, z_l = _lisa_frame_axes(ConstOmega * self.t)
        n = _unit(self.lam, self.beta)

        def north(z):
            v = z - np.einsum("ij,ij->i", z, n)[:, None] * n
            return v / np.linalg.norm(v, axis=1)[:, None]

        n_ssb = north(np.tile([0.0, 0.0, 1.0], (n.shape[0], 1)))
        n_lisa = north(z_l)
        chi = np.arctan2(
            np.einsum("ij,ij->i", np.cross(n_ssb, n_lisa), n),
            np.einsum("ij,ij->i", n_ssb, n_lisa),
        )
        self.assertLess(np.abs(_wrap(psiL - (self.psi + chi), np.pi)).max(), 1e-12)

    def test_z_axis_is_the_equal_arm_orbit_normal(self):
        try:
            from lisatools.detector import EqualArmlengthOrbits
        except ImportError:  # pragma: no cover
            self.skipTest("lisatools orbits unavailable")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orb = EqualArmlengthOrbits(force_backend="cpu", frame="ecliptic")
            t = self.t[:50]
            p = np.stack([np.atleast_2d(orb.get_pos(t, sc=i)) for i in (1, 2, 3)], axis=1)
        normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        normal *= np.sign(normal[:, 2:3])
        _, _, z_l = _lisa_frame_axes(ConstOmega * t)
        self.assertLess(np.abs(z_l - normal).max(), 1e-4)
        # and the retardation uses the actual center position
        n = _unit(self.lam[:50], self.beta[:50])
        tL = tLfromSSBframe(t, self.lam[:50], self.beta[:50])
        self.assertLess(
            np.abs(tL - (t - np.einsum("ij,ij->i", n, p.mean(axis=1)) / C_SI)).max(), 0.05
        )


class SkyModeTransformTest(unittest.TestCase):
    def test_eight_modes_tile_and_wrap(self):
        rng = np.random.default_rng(5)
        coords = rng.uniform(0, 1, (3, 11))
        coords[:, 8] *= 2 * np.pi
        coords[:, 9] = rng.uniform(-1, 1, 3)
        coords[:, 10] *= np.pi
        out = mbh_sky_mode_transform(coords, ind_map=dict(inc=7, lam=8, beta=9, psi=10), kind="both", cos_i=True)
        self.assertEqual(out.shape, (24, 11))
        base = out[0::8]
        np.testing.assert_allclose(base, coords)
        lat = out[1::8]
        np.testing.assert_allclose(lat[:, 9], -coords[:, 9])
        np.testing.assert_allclose(lat[:, 7], -coords[:, 7])
        np.testing.assert_allclose(lat[:, 10], (np.pi - coords[:, 10]))
        for k in range(1, 4):
            lon = out[2 * k::8]
            np.testing.assert_allclose(_wrap(lon[:, 8] - coords[:, 8] - k * np.pi / 2, 2 * np.pi), 0, atol=1e-12)
            np.testing.assert_allclose(_wrap(lon[:, 10] - coords[:, 10] - k * np.pi / 2, np.pi), 0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
