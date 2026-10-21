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

import time
from typing import Optional

import numpy as np
from lisatools.response.tdionfly import TDTDIonTheFly

from .utils.parallelbase import BBHxParallelModule

# Optional device-synced stage timing (off unless MBHTDIONFLY_TIMING=1).  See
# lisatools.utils.stagetimer -- shared with the response-side timers so one
# report covers the whole MBH evaluation path.
from lisatools.utils.stagetimer import (  # noqa: E402
    TIMER_COUNTS,
    TIMERS,
    TIMING,
    report_timing,
)
from lisatools.utils.stagetimer import stage as _stage  # noqa: E402


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

    # ------------------------------------------------------------------
    # Batched evaluation
    # ------------------------------------------------------------------
    def call_batch(
        self,
        m1, m2, s1z, s2z, distance, phi_ref, inclination, ra, dec, psi, t_merge,
        upsample_t_arr: np.ndarray = None,
        combine: bool = False,
        **kwargs,
    ):
        """Evaluate W binaries in ONE TDI kernel launch.

        The scalar :meth:`__call__` launches
        ``td_spline_run_wave_tdi_kernel<<<num_bin, 128>>>`` with
        ``num_bin = nmodes`` (8) -- 8 blocks on an 84-SM GPU, ~10% occupancy --
        and the global fit calls it once per walker, serially. Stacking W
        binaries into ``num_bin = W * nmodes`` is supported by the kernel as-is
        (``run_wave_tdi`` grid-strides ``bin_i`` by ``gridDim.x`` and slices its
        scratch by ``blockIdx.x``). Measured 3.5x at W=4, 5.9x at W=32.

        ``phentax`` already batches: every parameter is typed ``float | Array``
        and the return is rectangular ``(W, nmodes, ntimes)``, so the waveform
        stage costs one call. Members differ only in mask and time values; the
        kernel takes per-bin ``t_arr``, so only the spline LENGTH must be
        common. We pad every member to the batch-max valid count with the same
        zero-amplitude / constant-phase tail the scalar path already appends
        (~7% spread across walkers in practice).

        Args:
            m1..t_merge: array-like, each of length W.
            upsample_t_arr: shared output time grid.
            combine: sum over modes (per member), as in :meth:`__call__`.

        Returns:
            ``(W, 3, len(upsample_t_arr))`` when ``combine``, else
            ``(W, nmodes, 3, len(upsample_t_arr))``.
        """
        xp = self.xp
        t_total0 = time.perf_counter() if TIMING else None

        p = [np.atleast_1d(np.asarray(v, dtype=float)) for v in
             (m1, m2, s1z, s2z, distance, phi_ref, inclination, ra, dec, psi,
              t_merge)]
        W = max(len(v) for v in p)
        p = [np.broadcast_to(v, (W,)) for v in p]
        (m1, m2, s1z, s2z, distance, phi_ref, inclination, ra, dec, psi,
         t_merge) = p

        if self.waveform_duration is None:
            waveform_duration = t_merge
        else:
            waveform_duration = np.full(W, self.waveform_duration)

        with _stage("1_wave_gen_amp_phase", xp):
            new_times, new_mask, sc_amp, sc_phase = (
                self.wave_gen.compute_strain_components_amp_phase(
                    m1, m2, s1z, s2z, distance, phi_ref, inclination, psi,
                    delta_t=self.dt_min, t_min=-waveform_duration,
                    t_ref=-t_merge,
                )
            )

        with _stage("2_array_prep_pad", xp):
            mode_amp = sc_amp / 2.0          # AMP_FACTOR = 1/2
            mode_phase = np.pi - sc_phase    # negate and add pi

            nmodes = self.wave_gen.num_modes
            new_times = np.asarray(new_times)
            new_mask = np.asarray(new_mask)

            # Ragged valid counts -> pad every member up to the batch max, so
            # all bins share one spline length N (the kernel's single `N` arg).
            counts = new_mask.reshape(W, -1).sum(axis=1).astype(int)
            n_valid = int(counts.max())

            n_tail = 120
            dt_tail = 10.0
            n_spline = n_valid + n_tail

            t_stack = xp.zeros((W, nmodes, n_spline))
            amp_stack = xp.zeros((W, nmodes, n_spline))
            phase_stack = xp.zeros((W, nmodes, n_spline))

            for w in range(W):
                mask_w = new_mask[w]
                t_w = xp.asarray(new_times[w][mask_w] + t_merge[w] + self.t0)
                a_w = xp.asarray(mode_amp[w][:, mask_w])
                ph_w = xp.asarray(mode_phase[w][:, mask_w])

                # Pad short members past their own node top, keeping the grid
                # strictly increasing; amplitude 0 / phase held, exactly as the
                # scalar path's tail. n_pad_here covers both the ragged
                # shortfall and the shared tail.
                n_pad_here = n_spline - int(counts[w])
                pad_t = t_w[-1] + dt_tail * xp.arange(1, n_pad_here + 1)
                t_full = xp.concatenate([t_w, pad_t])
                a_full = xp.concatenate(
                    [a_w, xp.zeros((nmodes, n_pad_here))], axis=1)
                ph_full = xp.concatenate(
                    [ph_w, xp.repeat(ph_w[:, -1:], n_pad_here, axis=1)], axis=1)

                t_stack[w] = t_full
                amp_stack[w] = a_full
                phase_stack[w] = ph_full

            # (W, nmodes, n_spline) -> (W*nmodes, n_spline): one bin per
            # (walker, mode) pair.
            nbins = W * nmodes
            t_input = t_stack.reshape(nbins, n_spline)
            amp = amp_stack.reshape(nbins, n_spline)
            phase = phase_stack.reshape(nbins, n_spline)

            sampling_frequency = 1 / self.dt
            tdi_buffer = int(1000 / self.dt)
            eval_t_arr = t_input[:, tdi_buffer:-(tdi_buffer + n_tail)]

        with _stage("3_tdigen_construct", xp):
            tdi_gen = TDTDIonTheFly(
                eval_t_arr,
                amp,
                phase,
                sampling_frequency=sampling_frequency,
                num_sub=nbins,
                t_input=t_input,
                tdi_config=self.tdi_config,
                orbits=self.orbits,
                force_backend=self.response_backend,
            )

        # Inclination is already applied in the spherical harmonic; the sky /
        # polarization angles are per WALKER, repeated across that walker's modes.
        inc = xp.zeros(nbins)
        polarization = xp.repeat(xp.asarray(psi), nmodes)
        ra_arr = xp.repeat(xp.asarray(ra), nmodes)
        dec_arr = xp.repeat(xp.asarray(dec), nmodes)

        with _stage("4_tdi_kernel", xp):
            output = tdi_gen(inc, polarization, ra_arr, dec_arr,
                             return_spline=True)

        if upsample_t_arr is None:
            if TIMING:
                TIMERS["total"] += time.perf_counter() - t_total0
                TIMER_COUNTS["total"] += 1
            return output

        with _stage("5_upsample_eval_tdi", xp):
            upsample_t_arr = xp.asarray(upsample_t_arr)
            nt = upsample_t_arr.shape[-1]

            # Each walker covers its own time span, so "in range" is per-bin.
            # eval_spline_vals accepts a 2-D (nbins, n_new) grid of per-bin
            # evaluation times, so instead of looping we build one: in-range
            # points take the upsample value, out-of-range points are CLAMPED
            # to that bin's own window top purely to keep the spline argument
            # in range (their results are zeroed immediately below, exactly as
            # the scalar path leaves out-of-range samples at zero).
            t_win = output.t_arr                       # (nbins, N_eval)
            lo = t_win[:, :1]                          # (nbins, 1)
            hi = t_win[:, -1:]
            in_range = (upsample_t_arr[None, :] >= lo) & (
                upsample_t_arr[None, :] <= hi)         # (nbins, nt)
            t_eval = xp.where(in_range, upsample_t_arr[None, :], hi)

            vals = output.eval_tdi(t_eval)             # (nbins, 3, nt)
            new_tdi = xp.where(in_range[:, None, :], vals, 0.0)

            new_tdi = new_tdi.reshape(W, nmodes, 3, nt)
            if combine:
                new_tdi = new_tdi.sum(axis=1)

        if TIMING:
            TIMERS["total"] += time.perf_counter() - t_total0
            TIMER_COUNTS["total"] += 1

        return new_tdi

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
        t_total0 = time.perf_counter() if TIMING else None

        with _stage("1_wave_gen_amp_phase", xp):
            new_times, new_mask, sc_amp, sc_phase = self.wave_gen.compute_strain_components_amp_phase(
                m1, m2, s1z, s2z, distance, phi_ref, inclination, psi,
                delta_t=self.dt_min, t_min=-waveform_duration, t_ref=-t_merge,
            )

        with _stage("2_array_prep_pad", xp):
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

        with _stage("3_tdigen_construct", xp):
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

        with _stage("4_tdi_kernel", xp):
            output = tdi_gen(inc, polarization, ra_arr, dec_arr, return_spline=True)

        if upsample_t_arr is None:
            if TIMING:
                TIMERS["total"] += time.perf_counter() - t_total0
                TIMER_COUNTS["total"] += 1
            return output

        with _stage("5_upsample_eval_tdi", xp):
            upsample_t_arr = xp.asarray(upsample_t_arr)
            new_tdi = xp.zeros((output.t_arr.shape[0], 3, upsample_t_arr.shape[-1]))
            keep = (upsample_t_arr >= output.t_arr.min().item()) & (upsample_t_arr <= output.t_arr.max().item())
            new_tdi[:, :, keep] = output.eval_tdi(upsample_t_arr[keep])

            if combine:
                new_tdi = new_tdi.sum(axis=0)

        if TIMING:
            TIMERS["total"] += time.perf_counter() - t_total0
            TIMER_COUNTS["total"] += 1

        return new_tdi
