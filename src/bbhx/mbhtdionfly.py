# MBH TDI-on-the-fly waveform generation (time domain)

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

"""Time-domain MBH TDI-on-the-fly generator (:class:`MBHTDIonFly`).

Feeds the per-mode amplitude / phase of a time-domain mode generator (phentax
``IMRPhenomTHM``) through the LISA Analysis Tools TDI-on-the-fly response
(:class:`lisatools.response.tdionfly.TDTDIonTheFly`). Used by LISA Analysis Tools only when
``USE_TDIONFLY=1`` (``lisatools.globalfit.stock.erebor.wrappers.get_mbh_tdionfly_gen``),
which the production launchers do not set.
"""

from typing import Optional

import numpy as np
from lisatools.response.tdionfly import TDTDIonTheFly

from .utils.parallelbase import BBHxParallelModule


class MBHTDIonFly(BBHxParallelModule):
    """Generate MBH TDI observables on the fly in the time domain.

    This class combines a time-domain amplitude/phase mode generator
    (e.g. ``phentax.waveform.IMRPhenomTHM``) with the LISA
    TDI-on-the-fly response
    (:class:`lisatools.response.tdionfly.TDTDIonTheFly`). The waveform
    generator produces per-mode amplitude and phase on an adaptive time
    grid; this class feeds those splines through the on-the-fly TDI
    machinery to produce the TDI observables directly in the time
    domain, optionally upsampled onto a user-provided time array.

    Note:
        ``phentax`` is not published on PyPI. Install it alongside
        ``bbhx`` with the ``phentax`` extra
        (``pip install 'bbhx[phentax]'``) or directly via
        ``pip install git+https://github.com/asantini29/phentax.git``.

    Args:
        wave_gen (object): Time-domain mode generator. Must expose
            ``compute_strain_components_amp_phase(m1, m2, s1z, s2z,
            distance, phi_ref, inclination, psi, delta_t=..., t_min=...,
            t_ref=...)`` returning ``(times, mask, amp, phase)`` (amp / phase of shape
            ``(num_binaries, num_modes, num_times)``, merger at ``t = 0``) and a
            ``num_modes`` attribute (e.g. ``phentax.waveform.IMRPhenomTHM``).
        orbits (:class:`lisatools.detector.Orbits`): Configured orbits
            instance. The sky angles given at call time are read in its frame
            (``orbits.frame``; ICRS for the global-fit orbits).
        tdi_config (:class:`lisatools.response.tdiconfig.TDIConfig`):
            TDI configuration (e.g. ``TDIConfig("2nd generation")``).
        dt (float): Sampling cadence of the data stream (sec); the
            ``sampling_frequency`` of the response, and the unit of the ``1000 / dt``
            node margin trimmed from each end of the evaluation grid.
        Tobs (float): Observation time (sec); stored as ``T``, not used by
            :meth:`__call__`.
        t0 (float): Reference time offset added to the waveform time
            grid (sec): the merger lands at absolute time ``t0 + t_merge``.
        dt_min (float, optional): Densest cadence used when generating
            the waveform modes (sec; the ``delta_t`` handed to ``wave_gen``). Default
            ``0.1``.
        t_min (float, optional): Minimum time (sec); stored, not used by
            :meth:`__call__`. Default ``0.0``.
        waveform_duration (float, optional): Duration of waveform to
            generate before merger (sec). If ``None``, the merger time
            passed at call time is used (the waveform starts at ``t0``). Default
            ``None``.
        force_backend (str, optional): ``"cpu"``, ``"gpu"``, ``"cuda"``,
            ``"cuda11x"``, ``"cuda12x"``, or ``"cuda13x"`` (see
            :class:`~bbhx.utils.parallelbase.BBHxParallelModule`); the same flavor is
            handed to the LISA Analysis Tools response. Default ``None``.

    Example:
        .. code-block:: python

            import numpy as np
            from phentax.waveform import IMRPhenomTHM
            from lisatools.detector import ESAOrbits
            from lisatools.response.tdiconfig import TDIConfig
            from bbhx.mbhtdionfly import MBHTDIonFly

            dur = 3.0e6  # s
            wave = IMRPhenomTHM(
                higher_modes=[21, 33, 44], include_negative_modes=True,
                coarse_grain=True, coarse_graining_scale_factor=48.0, T=dur,
            )  # as in lisatools.globalfit.stock.erebor.wrappers, minus t_low_fit / atol / rtol
            gen = MBHTDIonFly(
                wave, ESAOrbits(force_backend="cpu"), TDIConfig("2nd generation"),
                dt=5.0, Tobs=dur, t0=0.0, force_backend="cpu",
            )
            # m1, m2 [Msun], s1z, s2z, distance [Mpc], phi_ref, inc, ra, dec, psi,
            # t_merge [s after t0]
            tdi = gen(1e6, 5e5, 0.2, 0.4, 1.0e4, 0.0, 0.8, 1.0, 0.3, 0.5, 2.5e6,
                      upsample_t_arr=np.arange(0.0, dur, 5.0), combine=True)  # (3, N)
    """

    def __init__(
        self,
        wave_gen,
        orbits,
        tdi_config,
        dt,
        Tobs,
        t0,
        dt_min=0.1,
        t_min=0.0,
        waveform_duration=None,
        force_backend=None,
    ):
        super().__init__(force_backend=force_backend)
        self.wave_gen = wave_gen
        self.orbits = orbits
        self.tdi_config = tdi_config
        self.dt = dt
        self.dt_min = dt_min
        self.t_min = t_min
        self.T = Tobs
        self.t0 = t0
        self.waveform_duration = waveform_duration

    @classmethod
    def supported_backends(cls) -> list:
        """Preference order: ``bbhx_`` + ``cuda13x``, ``cuda12x``, ``cuda11x``, ``cpu``."""
        return ["bbhx_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    @property
    def xp(self) -> object:
        """Numpy or Cupy"""
        return self.backend.xp

    @property
    def response_backend(self) -> str:
        """Backend flavor handed to the lisatools response classes."""
        return self.backend.name.split("_")[-1]

    @property
    def dt(self) -> float:
        """dt value from data (s); setting it also sets ``sampling_frequency = 1 / dt``."""
        return self._dt

    @dt.setter
    def dt(self, dt: float):
        self._dt = dt
        self.sampling_frequency = 1 / dt

    def __call__(
        self,
        m1,
        m2,
        s1z,
        s2z,
        distance,
        phi_ref,
        inclination,
        ra,
        dec,
        psi,
        t_merge,
        upsample_t_arr: np.ndarray = None,
        combine: bool = False,
        dt_tdi_eval=10.0,  # tunable: denser than the adaptive grid, sparser than dt=2.5s
        *args: Optional[tuple],
        **kwargs: Optional[dict],
    ):
        """Generate the time-domain TDI observables of one MBH binary.

        The modes are generated on ``wave_gen``'s adaptive grid from
        ``-waveform_duration`` to just after merger (merger at ``t = 0``, reference phase at
        ``t_ref = -t_merge``, i.e. at absolute time ``t0``, which is the start of the window
        when ``waveform_duration`` is ``None``), shifted to absolute times
        ``t + t_merge + t0``, converted to the response convention (amplitude ``/ 2``,
        phase ``pi - phase``) and padded with a 1200 s zero-amplitude tail. They are then
        projected with :class:`lisatools.response.tdionfly.TDTDIonTheFly` with ``inc = 0``
        (the inclination is already in the spin-weighted spherical harmonics). Only the
        first binary of the mode generator's output is used.

        Args:
            m1 (float): Mass of body 1 (solar masses).
            m2 (float): Mass of body 2 (solar masses).
            s1z (float): Dimensionless aligned spin of body 1.
            s2z (float): Dimensionless aligned spin of body 2.
            distance (float): Luminosity distance in the unit ``wave_gen`` expects (Mpc
                for phentax ``IMRPhenomTHM``).
            phi_ref (float): Reference phase (rad) at absolute time ``t0``.
            inclination (float): Inclination (rad).
            ra (float): Sky longitude (rad) in the frame of ``orbits`` (right ascension
                for ICRS orbits).
            dec (float): Sky latitude (rad) in the same frame (declination for ICRS).
            psi (float): Polarization angle (rad) in the same frame.
            t_merge (float): Merger time (s) after ``t0``.
            upsample_t_arr: Absolute times (s) on which to evaluate the TDI. Default
                ``None`` returns the response output object instead.
            combine: Sum the modes. Only used with ``upsample_t_arr``. Default ``False``.
            dt_tdi_eval (float, optional): Accepted but not used. Default ``10.0``.
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            lisatools.response.tdionfly.TDTDIOutput or xp.ndarray: Without
            ``upsample_t_arr``, the spline-backed output of the response (one sub-signal
            per mode). Otherwise the channels on ``upsample_t_arr``: shape
            ``(num_modes, 3, len(upsample_t_arr))``, or ``(3, len(upsample_t_arr))`` with
            ``combine=True``; times outside the output span are 0.
        """
        xp = self.xp

        if self.waveform_duration is None:
            waveform_duration = t_merge
        else:
            waveform_duration = self.waveform_duration

        # Reference phase is anchored at the reference epoch (t = -t_merge
        # relative to merger), NOT at merger (t_ref=0).  This matches the
        # PhenomTHMTDIWaveform / pyResponse convention (use_reference_time ->
        # t_ref = -merger_time) and the mojito merger-vs-reference-epoch phase
        # convention.  t_ref=0.0 left a ~sin^2(inc), frequency-growing
        # mismatch vs pyResponse (~11% edge-on, dominated by the higher modes);
        # t_ref=-t_merge brings the signal-band agreement to ~3e-4.
        new_times, new_mask, sc_amp, sc_phase = self.wave_gen.compute_strain_components_amp_phase(
            m1, m2, s1z, s2z, distance, phi_ref, inclination, psi,
            delta_t=self.dt_min, t_min=-waveform_duration, t_ref=-t_merge,
        )

        mode_amp = sc_amp / 2.0          # AMP_FACTOR = 1/2
        mode_phase = np.pi - sc_phase    # negate and add π

        # this will be close by an integer multiple of dt_min
        _new_times = xp.asarray(new_times[new_mask] + t_merge + self.t0)

        nmodes = self.wave_gen.num_modes

        amp = xp.asarray(mode_amp[0][:, new_mask[0]])
        phase = xp.asarray(mode_phase[0][:, new_mask[0]])

        # Zero-amplitude / constant-phase tail appended past the last
        # waveform node. The waveform grid ends ~1400 s after the SSB merger
        # time, but the merger burst ARRIVES at the constellation at
        # t_merge + k.x/c (up to ~+500 s, sky-sign dependent), and the
        # on-the-fly TDI reads the amp/phase splines at retarded times up to
        # another ~|k.x|/c + delay-chain (~600 s) past each eval point. For
        # unlucky sky positions those reads land past the node top
        # ("Outside spline" on MBHB src1 at +0.09 s). The post-ringdown
        # amplitude at the node top is ~1e-19 of peak, so a zero tail is the
        # physical continuation; it is never inside the eval window itself.
        n_tail = 120
        dt_tail = 10.0
        tail_t = _new_times[-1] + dt_tail * xp.arange(1, n_tail + 1)
        _new_times_pad = xp.concatenate([_new_times, tail_t])
        amp = xp.concatenate([amp, xp.zeros((amp.shape[0], n_tail))], axis=1)
        phase = xp.concatenate([phase, xp.repeat(phase[:, -1:], n_tail, axis=1)], axis=1)

        new_times_arr = xp.repeat(_new_times_pad[None, :], nmodes, axis=0)

        sampling_frequency = 1 / self.dt

        # Eval window: same slice of the ORIGINAL (unpadded) grid as always
        # -- [tdi_buffer : N_orig - tdi_buffer] samples. The appended tail
        # only serves the spline reads beyond the eval window.
        tdi_buffer = int(1000 / self.dt)  # samples (~1000 s at the coarse ends)

        eval_t_arr = new_times_arr[:, tdi_buffer:-(tdi_buffer + n_tail)]

        tdi_gen = TDTDIonTheFly(
            eval_t_arr,
            amp,
            phase,
            sampling_frequency=sampling_frequency,
            num_sub=nmodes,
            t_input=new_times_arr,
            tdi_config=self.tdi_config,
            orbits=self.orbits,
            force_backend=self.response_backend,
        )

        inc = xp.full(nmodes, 0.0)  # inclination is already applied in the spherical harmonic
        polarization = xp.full(nmodes, psi)
        ra_arr = xp.full(nmodes, ra)
        dec_arr = xp.full(nmodes, dec)

        output = tdi_gen(inc, polarization, ra_arr, dec_arr, return_spline=True)

        if upsample_t_arr is None:
            return output

        upsample_t_arr = xp.asarray(upsample_t_arr)
        new_tdi = xp.zeros((output.t_arr.shape[0], 3, upsample_t_arr.shape[-1]))
        keep = (upsample_t_arr >= output.t_arr.min().item()) & (upsample_t_arr <= output.t_arr.max().item())
        new_tdi[:, :, keep] = output.eval_tdi(upsample_t_arr[keep])

        if combine:
            return new_tdi.sum(axis=0)

        return new_tdi
