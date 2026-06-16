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
            t_ref=...)`` returning ``(times, mask, amp, phase)`` and a
            ``num_modes`` attribute (e.g.
            ``phentax.waveform.IMRPhenomTHM``).
        orbits (:class:`lisatools.detector.Orbits`): Configured orbits
            instance.
        tdi_config (:class:`lisatools.response.tdiconfig.TDIConfig`):
            TDI configuration (e.g. ``TDIConfig("2nd generation")``).
        dt (float): Sampling cadence of the data stream (sec).
        Tobs (float): Observation time (sec).
        t0 (float): Reference time offset added to the waveform time
            grid (sec).
        dt_min (float, optional): Densest cadence used when generating
            the waveform modes (sec). (default: ``0.1``)
        t_min (float, optional): Minimum time (sec). (default: ``0.0``)
        waveform_duration (float, optional): Duration of waveform to
            generate before merger (sec). If ``None``, the merger time
            passed at call time is used. (default: ``None``)
        force_backend (str, optional): ``"cpu"``, ``"gpu"``, ``"cuda"``,
            ``"cuda11x"``, ``"cuda12x"``, or ``"cuda13x"``.
            (default: ``None``)

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
        """dt value from data."""
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
        new_times_arr = xp.repeat(_new_times[None, :], nmodes, axis=0)

        amp = xp.asarray(mode_amp[0][:, new_mask[0]])
        phase = xp.asarray(mode_phase[0][:, new_mask[0]])

        sampling_frequency = 1 / self.dt

        tdi_buffer = int(1000 / self.dt)  # seconds #todo @Mike: how many samples do we have to discard here? I kept getting out of splines error for smaller values

        eval_t_arr = new_times_arr[:, tdi_buffer:-tdi_buffer]

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
