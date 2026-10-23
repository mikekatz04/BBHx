"""Fast frequency-domain LISA response for MBH harmonics (:class:`LISATDIResponse`).

Computes the TDI transfer functions of each ``(l, m)`` mode along its time-frequency track
t(f) (arXiv:1806.10734, arXiv:2003.00357) with the native ``LISA_response_wrap`` kernel.
It is the response stage of :class:`bbhx.waveformbuild.BBHWaveformFD`; the global-fit
production runs do not evaluate it.
"""
from __future__ import annotations
# LISA Response Functions

# Copyright (C) 2021 Michael L. Katz, Sylvain Marsat
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
from __future__ import annotations

import numpy as np

from ..utils.constants import *

from lisatools.detector import EqualArmlengthOrbits, Orbits
from ..utils.parallelbase import BBHxParallelModule

class LISATDIResponse(BBHxParallelModule):
    r"""Evaluate the fast frequency domain response function.

    The response function is the fast frequency domain response function
    from `arXiv:1806.10734 <https://arxiv.org/abs/1806.10734>`_ and
    `arXiv:2003.00357 <https://arxiv.org/abs/2003.00357>`_. Please cite
    these papers if this class is used.

    For each mode and frequency the kernel evaluates the single-link responses at time
    t(f) with the spacecraft positions and link unit vectors of ``orbits``, combines
    them into first-generation TDI with the constant armlength ``orbits.armlength`` (a
    fixed, non-breathing armlength), multiplies in the spin-weighted spherical harmonics
    of ``(inc, phi_ref)`` and the polarization ``psi``, and adds the orbital-delay phase
    :math:`\phi_R = 2\pi f\, k \cdot p_0 / c` to the mode phase (the transfer functions are
    also multiplied by :math:`e^{-i\phi_R}`), where ``p0`` is an analytic circular orbit of
    radius ``AU_SI`` in the ecliptic plane at angle ``Omega0 * t``. ``lam`` / ``beta``
    must therefore be expressed in the frame of the orbit data, and that frame must be
    the ecliptic for the analytic delay to agree with it (true for the packaged LISA
    Analysis Tools orbit files, whose positions are SSB-ecliptic whatever their ``frame``
    label). See the TODOs on the orbital-delay convention in ``Response.cu`` and
    :meth:`bbhx.waveformbuild.BBHWaveformFD.__call__`.

    This class has GPU capability.

    Args:
        TDItag (str, optional): TDI channels to generate. Options are ``"XYZ"`` and
            ``"AET"``; any value other than ``"XYZ"`` gives ``"AET"``. Default ``"AET"``.
        orbits: LISA Analysis Tools orbit object (:class:`~lisatools.detector.Orbits`);
            its backend flavor must match this class's (asserted). Default
            ``None`` builds :class:`~lisatools.detector.EqualArmlengthOrbits` with default
            backend resolution, which can pick a different flavor than ``force_backend``;
            pass orbits explicitly (e.g. ``ESAOrbits(force_backend="cpu")``).
        rescaled: If ``True``, drop the overall TDI prefactors (``2i sin(2x) e^{2ix}`` for
            XYZ; the A/E and T factors for AET) to avoid infinities at high frequency.
            Default ``False``.
        tdi2: If ``True``, multiply the AET output by :math:`-2i \sin(4x) e^{4ix}`,
            :math:`x = \pi f L / c`: the TDI 1 -> TDI 2 conversion for equal armlengths.
            Ignored for ``TDItag="XYZ"``. Default ``False``.
        order_fresnel_stencil (int, optional): Order of the Fresnel stencil in the
            response. Currently, anything above 0 is not implemented (raises
            ``NotImplementedError``). This is left in for future compatibility.
            Default ``0``.
        force_backend (str, optional): Backend name (``"cpu"``, ``"cuda12x"``, ``"cuda"``,
            ``"gpu"``, ...), see :class:`~bbhx.utils.parallelbase.BBHxParallelModule`.
            Default ``None`` picks the first available of ``cuda13x``, ``cuda12x``,
            ``cuda11x``, ``cpu``.

    Attributes:
        allowable_modes (list): Allowed list of mode tuple pairs ``(l,m)`` for
            the chosen waveform model.
        ells_default (xp.ndarray): Default values for the ``l`` index of the harmonic.
        mms_default (xp.ndarray): Default values for the ``m`` index of the harmonic.
        buffer (float): Tolerance (s, class attribute ``1e3``) by which t(f) may leave
            the orbit time span; values within it are clamped to the span.
        includes_amps (int): ``1`` if the buffer of the last call holds amplitudes in its
            first slot (9 quantities), ``0`` otherwise (8).
        order_fresnel_stencil (int): See Args.
        TDItag (str): TDI channels to generate. Either ``"XYZ"`` or ``"AET"``.
        rescaled (bool): See Args.
        tdi2 (bool): See Args.
        modes (list): Modes of the last call.
        response_carrier (xp.ndarray): Flat buffer of the last call (the caller's
            ``out_buffer`` when given).

    Example:
        .. code-block:: python

            from lisatools.detector import ESAOrbits
            from bbhx.response.fastfdresponse import LISATDIResponse
            from bbhx.waveforms.phenomhm import PhenomHMAmpPhase

            phenomhm = PhenomHMAmpPhase(force_backend="cpu")
            phenomhm(1e6, 5e5, 0.2, 0.4, 5.5e26, 0.0, 0.0, 3e6, 1024)
            response = LISATDIResponse(orbits=ESAOrbits(force_backend="cpu"), force_backend="cpu")
            # freqs, inc, lam, beta, psi, phi_ref, length
            response(
                phenomhm.freqs_shaped.copy(), 0.8, 0.5, 0.6, 0.9, 0.0, 1024,
                phase=phenomhm.phase.copy(), tf=phenomhm.tf.copy(), modes=phenomhm.modes,
            )
            A = response.transferL1  # (1, 6, 1024) complex transfer function
    """

    buffer = 1e3  # seconds
    def __init__(
        self,
        TDItag="AET",
        orbits: Orbits | None = None,
        rescaled: bool = False,
        tdi2: bool = False,
        order_fresnel_stencil=0,
        force_backend=None,
    ):
        self.rescaled = rescaled
        self.tdi2 = tdi2

        # gpu setup
        super().__init__(force_backend=force_backend)

        if order_fresnel_stencil > 0:
            raise NotImplementedError

        self.order_fresnel_stencil = order_fresnel_stencil

        # TDI setup
        self.TDItag = TDItag
        if TDItag == "XYZ":
            self.TDItag_int = 1

        else:
            self.TDItag_int = 2

        # PhenomHM modes
        self.allowable_modes = [(2, 2), (3, 3), (4, 4), (2, 1), (3, 2), (4, 3)]

        self.ells_default = self.xp.array([2, 3, 4, 2, 3, 4], dtype=self.xp.int32)

        self.mms_default = self.xp.array([2, 3, 4, 1, 2, 3], dtype=self.xp.int32)

        self.orbits = orbits
    
    @classmethod
    def supported_backends(cls) -> list:
        """Preference order: ``bbhx_`` + ``cuda13x``, ``cuda12x``, ``cuda11x``, ``cpu``."""
        return ["bbhx_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    @property
    def response_gen(self):
        """Native ``LISA_response_wrap`` kernel (CPU or GPU)."""
        return self.backend.LISA_response_wrap

    @property
    def xp(self):
        """Numpy or Cupy"""
        return self.backend.xp

    @property
    def orbits(self) -> Orbits:
        """Orbit object read by the response kernel (through ``orbits.pycppdetector``).

        Setting ``None`` builds :class:`~lisatools.detector.EqualArmlengthOrbits`; a value
        that is not a :class:`~lisatools.detector.Orbits` raises ``ValueError``. The setter
        configures the orbits (``_ensure_configured``) and asserts that their backend
        flavor equals this object's.
        """
        return self._orbits

    @orbits.setter
    def orbits(self, orbits: Orbits) -> None:
        if orbits is None:
            self._orbits = EqualArmlengthOrbits()
        elif not isinstance(orbits, Orbits):
            raise ValueError(
                "Input orbits must be of type Orbits (from LISA Analysis Tools)."
            )
        else:
            self._orbits = orbits
        self._orbits._ensure_configured()
        assert self._orbits.backend.name.split("_")[-1] == self.backend.name.split("_")[-1]

    @property
    def citation(self):
        """Citation strings for this class.

        Warning:
            Raises ``NameError``: the citation strings (``katz_citations``, ...) are no
            longer defined in this module (``bbhx.utils.citations`` was removed).
        """
        return katz_citations + marsat_1 + marsat_2

    def _sanity_check_modes(self, ells, mms):
        """Raise ``ValueError`` if an ``(ell, mm)`` pair is not in :attr:`allowable_modes`."""
        for ell, mm in zip(ells, mms):
            if (ell, mm) not in self.allowable_modes:
                raise ValueError(
                    "Requested mode [(l,m) = ({},{})] is not available. Allowable modes include {}".format(
                        ell, mm, self.allowable_modes
                    )
                )

    def _initialize_response_container(self):
        """Allocate a zero :attr:`response_carrier` of size ``nparams * num_per_param``."""
        self.response_carrier = self.xp.zeros(
            (self.nparams * self.length * self.num_modes * self.num_bin_all),
            dtype=self.xp.float64,
        )

    @property
    def transferL1(self):
        """TransferL1 term in response. Shape: ``(num_bin_all, num_modes, length)``"""
        temp = self.response_carrier[
            (self.includes_amps + 2)
            * self.num_per_param : (self.includes_amps + 3)
            * self.num_per_param
        ].reshape(
            self.num_bin_all, self.num_modes, self.length
        ) + 1j * self.response_carrier[
            (self.includes_amps + 3)
            * self.num_per_param : (self.includes_amps + 4)
            * self.num_per_param
        ].reshape(
            self.num_bin_all, self.num_modes, self.length
        )

        return temp

    @property
    def transferL2(self):
        """TransferL2 term in response. Shape: ``(num_bin_all, num_modes, length)``"""
        temp = self.response_carrier[
            (self.includes_amps + 4)
            * self.num_per_param : (self.includes_amps + 5)
            * self.num_per_param
        ].reshape(
            self.num_bin_all, self.num_modes, self.length
        ) + 1j * self.response_carrier[
            (self.includes_amps + 5)
            * self.num_per_param : (self.includes_amps + 6)
            * self.num_per_param
        ].reshape(
            self.num_bin_all, self.num_modes, self.length
        )

        return temp

    @property
    def transferL3(self):
        """TransferL3 term in response. Shape: ``(num_bin_all, num_modes, length)``"""
        temp = self.response_carrier[
            (self.includes_amps + 6)
            * self.num_per_param : (self.includes_amps + 7)
            * self.num_per_param
        ].reshape(
            self.num_bin_all, self.num_modes, self.length
        ) + 1j * self.response_carrier[
            (self.includes_amps + 7)
            * self.num_per_param : (self.includes_amps + 8)
            * self.num_per_param
        ].reshape(
            self.num_bin_all, self.num_modes, self.length
        )

        return temp

    @property
    def phase(self):
        """Get updated phase info. Shape: ``(num_bin_all, num_modes, length)``"""
        phase = self.response_carrier[
            (self.includes_amps + 0)
            * self.num_per_param : (self.includes_amps + 1)
            * self.num_per_param
        ].reshape(self.num_bin_all, self.num_modes, self.length)

        return phase

    @property
    def tf(self):
        """Get tf info. Shape: ``(num_bin_all, num_modes, length)``"""
        tf = self.response_carrier[
            (self.includes_amps + 1)
            * self.num_per_param : (self.includes_amps + 2)
            * self.num_per_param
        ].reshape(self.num_bin_all, self.num_modes, self.length)

        return tf

    def __call__(
        self,
        freqs,
        inc,
        lam,
        beta,
        psi,
        phi_ref,
        length,
        modes=None,
        phase=None,
        tf=None,
        out_buffer=None,
        adjust_phase=True,
        direct=False,
    ):
        """Evaluate the response function.

        Fills the transfer functions (read through :attr:`transferL1`, :attr:`transferL2`,
        :attr:`transferL3`) and adds the orbital-delay phase to the mode phases (read
        through :attr:`phase`; :attr:`tf` is the time-frequency track used). Returns
        ``None``.

        Args:
            freqs (xp.ndarray): Frequencies (Hz) at which the response is evaluated:
                shape ``(num_bin_all, num_modes, length)`` or flat with
                ``num_bin_all * num_modes * length`` entries.
            inc (float or xp.ndarray): Inclination (rad), one per binary.
            lam (float or xp.ndarray): Longitude (rad) in the frame of the orbit data
                (conventionally SSB ecliptic, see the class docstring), one per binary.
            beta (float or xp.ndarray): Latitude (rad), same frame, one per binary.
            psi (float or xp.ndarray): Polarization angle (rad), one per binary.
            phi_ref (float or xp.ndarray): Reference phase. **Note**:
                The response function rotates the source by ``phi_ref`` (it enters the
                spin-weighted spherical harmonics ``Y_lm(inc, phi_ref)``). For this reason,
                the main waveform functions (e.g. :class:`bbhx.waveformbuild.BBHWaveformFD`)
                provide ``phi_ref = 0.0`` into the source-frame scaled waveform generators
                (e.g. :class:`bbhx.waveforms.phenomhm.PhenomHMAmpPhase`). This allows
                the reference phase to be applied here in the response.
            length (int): Number of frequencies per binary and mode (the chunk size of the
                flat arrays).
            modes (list, optional): ``(l, m)`` tuples, a subset of :attr:`allowable_modes`.
                Default ``None``: the PhenomHM modes
                ``[(2,2), (3,3), (4,4), (2,1), (3,2), (4,3)]``.
            phase (xp.ndarray, optional): Waveform phase, shape
                ``(num_bin_all, num_modes, length)`` or flat. The kernel adds the
                orbital-delay phase ``phaseRdelay`` to it. Required, with ``tf``, when
                ``out_buffer`` is not given.
            tf (xp.ndarray, optional): Waveform time-frequency correspondence (s), same
                shape as ``phase``: where the constellation is at each frequency. Values
                within :attr:`buffer` of the orbit time span are clamped to it IN PLACE.
            out_buffer (xp.ndarray, optional): Flat buffer holding the waveform and filled by
                the response, ``nparams * num_bin_all * num_modes * length`` entries with
                ``nparams = 9`` (amplitude first, the layout of
                :class:`~bbhx.waveformbuild.BBHWaveformFD`) or ``8`` (no amplitude). When
                ``phase`` / ``tf`` are not given, they are read from it. Default ``None``.
            adjust_phase (bool, optional): If ``True``, write the delay-adjusted phase back
                into the given ``phase`` array (only when ``phase`` and ``tf`` are given).
                Default ``True``.
            direct (bool, optional): If ``True``, skip the check / clamping of t(f) against
                the orbit time span. Default ``False``.

        Raises:
            ValueError: Wrong array sizes or shapes, ``out_buffer`` not of 8 or 9
                quantities, only one of ``phase`` / ``tf`` given, neither ``out_buffer`` nor
                ``phase`` / ``tf`` given, an unavailable mode, or (``direct=False``, ``phase``
                / ``tf`` given) t(f) more than :attr:`buffer` outside the orbit time span.
            AttributeError: Same out-of-span case when t(f) is read from ``out_buffer``
                (the error message formats ``tf.min()`` with ``tf=None``).
        """

        # to cupy if needed
        inc = self.xp.asarray(self.xp.atleast_1d(inc)).copy()
        lam = self.xp.asarray(self.xp.atleast_1d(lam)).copy()
        beta = self.xp.asarray(self.xp.atleast_1d(beta)).copy()
        psi = self.xp.asarray(self.xp.atleast_1d(psi)).copy()
        phi_ref = self.xp.asarray(self.xp.atleast_1d(phi_ref)).copy()

        # mode setup
        if modes is not None:
            ells = self.xp.asarray([ell for ell, mm in modes], dtype=self.xp.int32)
            mms = self.xp.asarray([mm for ell, mm in modes], dtype=self.xp.int32)
            self._sanity_check_modes(ells, mms)

        else:
            ells = self.ells_default
            mms = self.mms_default

        self.modes = [(ell, mm) for ell, mm in zip(ells, mms)]

        num_modes = len(ells)
        num_bin_all = len(inc)

        # store all info
        self.length = length
        self.num_modes = num_modes
        self.num_bin_all = num_bin_all
        self.num_per_param = length * num_modes * num_bin_all
        self.num_per_bin = length * num_modes

        # number of respones-specific parameters
        self.nresponse_params = 6

        if out_buffer is None and phase is None and tf is None:
            raise ValueError("Must provide either out_buffer or both phase and tf.")

        # setup out_buffer based on inputs
        if out_buffer is None:
            includes_amps = 0
            self.nparams = includes_amps + 2 + self.nresponse_params
            self._initialize_response_container()

        else:
            # use other shape information already known to
            # make sure the given out_buffer is the right length
            # and has the right number of parameters (8 or 9 (including amps))
            nparams_empirical = len(out_buffer) / (
                self.num_bin_all * self.num_modes * self.length
            )

            # indicate if there is an integer number of params in the out_buffer
            if np.allclose(nparams_empirical, np.round(nparams_empirical)) and int(
                nparams_empirical
            ) in [8, 9]:
                self.nparams = int(nparams_empirical)
            else:
                raise ValueError(
                    f"out_buffer incorrect length. The length should be equivalent to (8 or 9) * {self.num_bin_all * self.num_modes * self.length}. Given length is {len(out_buffer)}."
                )

            includes_amps = 1 if self.nparams == 9 else 0
            self.response_carrier = out_buffer

        # if amps are included they are in teh first slot in the array
        self.includes_amps = includes_amps

        # setup and check frequency dimensions
        self.freqs = freqs
        if self.freqs.ndim > 1:
            if self.freqs.shape != (self.num_bin_all, self.num_modes, self.length):
                raise ValueError(
                    f"freqs have incorrect shape. Shape should be {(self.num_bin_all, self.num_modes, self.length)}. Current shape is {freqs.shape}."
                )
            self.freqs = self.freqs.flatten()
        else:
            if len(freqs) != self.num_bin_all * self.num_modes * self.length:
                raise ValueError(
                    f"freqs incorrect length. The length should be equivalent to {self.num_bin_all * self.num_modes * self.length}. Given length is {len(freqs)}."
                )

        # if using phase/tf
        if phase is not None and tf is not None:
            use_phase_tf = True

            if not direct:
                if (
                    self.xp.any((tf - self.orbits.t_base.min()) < -self.buffer)
                    or self.xp.any((tf - self.orbits.t_base.max()) > self.buffer)
                ):
                    raise ValueError(
                        f"Orbital information does not cover minimum ({tf.min()}) and maximum ({tf.max()}) tf. Orbital information begins at {self.orbits.t_base.min()} and ends at {self.orbits.t_base.max()}. It is also greater than the buffer {self.buffer}."
                    )
                
                # if we make it here, it is close enough under the buffer to just rewrite the value.
                if self.xp.any(tf > self.orbits.t_base.max()):
                    # if we make it here, it is close enough under the buffer to just rewrite the value.
                    tf[tf > self.orbits.t_base.max()] = self.orbits.t_base.max() 
                if self.xp.any(tf < self.orbits.t_base.min()):
                    # if we make it here, it is close enough under the buffer to just rewrite the value.
                    tf[tf < self.orbits.t_base.min()] = self.orbits.t_base.min() 

            if phase.shape != tf.shape:
                raise ValueError(
                    "Shape of phase array and tf array need to be the same shape."
                )

            if phase.ndim > 1:
                if phase.shape != (self.num_bin_all, self.num_modes, self.length):
                    raise ValueError(
                        f"phase have incorrect shape. Shape should be {(self.num_bin_all, self.num_modes, self.length)}. Current shape is {phase.shape}."
                    )
                # will need to write the phase to original array later if adjust_phase == True
                first = phase.copy().flatten()
                second = tf.copy().flatten()

            else:
                if len(phase) != self.num_bin_all * self.num_modes * self.length:
                    raise ValueError(
                        f"phase incorrect length. The length should be equivalent to {self.num_bin_all * self.num_modes * self.length}. Given length is {len(phase)}."
                    )

                # will need to write the phase to original array later if adjust_phase == True
                first = phase
                second = tf

            # fill the phase into the buffer (which is flat)
            self.response_carrier[
                (includes_amps + 0)
                * length
                * num_modes
                * num_bin_all : (includes_amps + 1)
                * length
                * num_modes
                * num_bin_all
            ] = first

            # fill tf in the buffer (which is flat)
            self.response_carrier[
                (includes_amps + 1)
                * length
                * num_modes
                * num_bin_all : (includes_amps + 2)
                * length
                * num_modes
                * num_bin_all
            ] = second

        elif phase is not None or tf is not None:
            raise ValueError("If provided phase or tf, need to provide both.")

        else:
            use_phase_tf = False
            
            if not direct:
                if (
                    self.xp.any((self.tf - self.orbits.t_base.min()) < -self.buffer)
                    or self.xp.any((self.tf - self.orbits.t_base.max()) > self.buffer)
                ):
                    raise ValueError(
                        f"Orbital information does not cover minimum ({tf.min()}) and maximum ({tf.max()}) tf. Orbital information begins at {self.orbits.t_base.min()} and ends at {self.orbits.t_base.max()}. It is also greater than the buffer {self.buffer}."
                    )
                
                # if we make it here, it is close enough under the buffer to just rewrite the value.
                if self.xp.any(self.tf > self.orbits.t_base.max()):
                    # if we make it here, it is close enough under the buffer to just rewrite the value.
                    self.tf[self.tf > self.orbits.t_base.max()] = self.orbits.t_base.max() 
                if self.xp.any(self.tf < self.orbits.t_base.min()):
                    # if we make it here, it is close enough under the buffer to just rewrite the value.
                    self.tf[self.tf < self.orbits.t_base.min()] = self.orbits.t_base.min() 

        # run response code in C/CUDA
        self.response_gen(
            self.response_carrier,
            ells,
            mms,
            self.freqs,
            phi_ref,
            inc,
            lam,
            beta,
            psi,
            self.TDItag_int,
            self.rescaled,
            self.tdi2,
            self.order_fresnel_stencil,
            num_modes,
            length,
            num_bin_all,
            includes_amps,
            # LISA_response_wrap (pybind11, post-Phase-BBHx.pybind.bulk) takes the
            # OrbitsWrap pybind11 instance, not the high-level Python
            # Orbits class -- the wrap is exposed as `self.orbits.pycppdetector`.
            # Prior Cython binding used gpubackendtools.wrapper which extracted
            # `.ptr` from the Python Orbits object.
            self.orbits.pycppdetector,
        )

        # adjust input phase arrays in-place
        if use_phase_tf and adjust_phase:
            output = self.response_carrier[
                (includes_amps + 0)
                * length
                * num_modes
                * num_bin_all : (includes_amps + 1)
                * length
                * num_modes
                * num_bin_all
            ]

            if phase.ndim > 1:
                phase[:] = output.reshape(phase.shape)
            else:
                phase[:] = output
