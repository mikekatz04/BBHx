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
    """Evaluate the fast frequency domain response function

    The response function is the fast frequency domain response function
    from `arXiv:1806.10734 <https://arxiv.org/abs/1806.10734>`_ and
    `arXiv:2003.00357 <https://arxiv.org/abs/2003.00357>`_. Please cite
    these papers if this class is used. This response assumes a fixed,
    non-breathing armlength for the LISA constellation.

    This class has GPU capability.

    Args:
        TDItag (str, optional): TDI channels to generate. Options are ``"XYZ"`` and
            ``"AET"``. If ``"XYZ"`` is not given, it will default to ``"AET"``.
            (Default: ``"AET"``)
        orbits (Orbits, optional): Orbit class. If ``None``, orbits is set to
            :class:`EqualArmlengthOrbits`. (Default: ``None``)
        rescaled (bool, optional): If ``True``, rescale TDI functions to avoid
            infinities at high frequency. (Default: ``False``)
        tdi2 (bool, optional): If ``True``, apply a factor of :math:`-2i \\sin{(4x)}e^{i4x})`
            to tdi1 output. This is a conversion from TDI 1 to TDI 2 under the assumption of equal armlengt orbits.
            (Default: ``False``)
        order_fresnel_stencil (int, optional): Order of the Fresnel stencil in the
            response. Currently, anything above 0 is not implemented. This is left
            in for future compatibility. (Default: ``0``)
        force_backend (str, optional): ``"cpu"'', ``"gpu"'', ``"cuda"'', ``"cuda12x"'', or ``"cuda11x"''.

    Attributes:
        allowable_modes (list): Allowed list of mode tuple pairs ``(l,m)`` for
            the chosen waveform model.
        ells_default (np.ndarray): Default values for the ``l`` index of the harmonic.
        mms_default (np.ndarray): Default values for the ``m`` index of the harmonic.
        includes_amps (bool): If ``True``, the ``out_buffer`` contains the first
            entry for amplitudes.
        order_fresnel_stencil (int): Order of the Fresnel stencil in the
            response. Currently, anything above 0 is not implemented. This is left
            in for future compatibility.
        TDItag (str): TDI channels to generate. Either ``"XYZ"`` or ``"AET"``.

    """

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

    @property
    def response_gen(self):
        """C function on GPU/CPU"""
        return self.backend.LISA_response_wrap

    @property
    def xp(self):
        """Numpy or Cupy"""
        return self.backend.xp

    @property
    def orbits(self) -> Orbits:
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
        self._orbits.configure(linear_interp_setup=True)
        assert self._orbits.backend.name.split("_")[-1] == self.backend.name.split("_")[-1]

    @property
    def citation(self):
        """Return citations for this class"""
        return katz_citations + marsat_1 + marsat_2

    def _sanity_check_modes(self, ells, mms):
        """Make sure modes are allowed"""
        for ell, mm in zip(ells, mms):
            if (ell, mm) not in self.allowable_modes:
                raise ValueError(
                    "Requested mode [(l,m) = ({},{})] is not available. Allowable modes include {}".format(
                        ell, mm, self.allowable_modes
                    )
                )

    def _initialize_response_container(self):
        """setup reponse container if needed"""
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
        """Evaluate respones function

        Args:
            freqs (1D or 2D xp.ndarray): Frequency at which the response is evaluated.
                2D shape is ``(num_bin_all, length)``. If given as a 1D array,
                it should be of length ``num_bin_all * length``.
            inc (scalar or 1D xp.ndarray): Inclination of BBH system in radians.
            lam (scalar or 1D xp.ndarray): Ecliptic longitude in SSB frame in radians.
            beta (scalar or 1D xp.ndarray): Ecliptic latitude in SSB frame in radians.
            psi (scalar or 1D xp.ndarray): Polarization angle of the system in radians.
            phi_ref (scalar or 1D xp.ndarray): Reference phase. **Note**:
                The response function rotates the source by ``phi_ref``. For this reason,
                the main waveform functions (e.g. :class:`bbhx.waveform.BBHWaveformFD`)
                provide ``phi_ref = 0.0`` into the source-frame scaled waveform generators
                (e.g. :class:`bbhx.waveforms.phenomhm.PhenomHMAmpPhase`). This allows
                the reference phase to be applied here in the response.
            length (int): The length of the individual frequency arrays. This is required
                because of the options for putting in 1D arrays into this function.
                The length tells the chunk size in a 1D array.
            modes (list, optional): Harmonic modes to use. If not given, they will
                default to those available in the waveform model PhenomHM:
                ``[(2,2), (3,3), (4,4), (2,1), (3,2), (4,3)]``. (Default: ``None``)
            phase (xp.ndarray, optional): Waveform phase. This is adjusted by the ``phaseRdelay``
                quantity in the code. If more than 1D, the shape should be
                ``(num_bin_all, num_modes, length)``. If 1D, its total length
                should be equivalent to ``num_bin_all * num_modes * length``.
                If ``out_buffer`` is not provided, ``phase`` and ``tf`` are required.
            tf (xp.ndarray, optional): Waveform time-frequency correspondence. This tells the
                response where the LISA constellation is at each frequency.
                If more than 1D, the shape should be
                ``(num_bin_all, num_modes, length)``. If 1D, its total length
                should be equivalent to ``num_bin_all * num_modes * length``.
                If ``out_buffer`` is not provided, ``phase`` and ``tf`` are required.
            out_buffer (xp.ndarray, optional): 1D array initialized to contain all computations
                from the inital waveform and response function. If providing ``out_buffer``,
                the response fills it directly. To make this happen easily in GPU/CPU
                agnostic manner, out_buffer needs to be a 1D array with length
                equivalent to ``nparams * num_bin_all * num_modes * length``.
                ``nparams`` can be 8 if the buffer does not include the amplitudes
                (which are not needed at all for the response computation) or 9
                if it includes the amplitudes. (Default: ``None``)
            adjust_phase (bool, optional): If ``True`` adjust the phase array in-place
                inside the response code. **Note**: This only applies when
                inputing ``phase`` and ``tf``. (Default: ``True``)

        Raises:
            ValueError: Incorrect dimensions for the arrays.

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

            if not direct and (
                tf.min() < self.orbits.t_base.min()
                or tf.max() > self.orbits.t_base.max()
            ):
                raise ValueError(
                    f"Orbital information does not cover minimum ({tf.min()}) and maximum ({tf.max()}) tf. Orbital information begins at {self.orbits.t_base.min()} and ends at {self.orbits.t_base.max()}."
                )

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
            if not direct and (
                self.tf.min() < self.orbits.t_base.min()
                or self.tf.max() > self.orbits.t_base.max()
            ):
                breakpoint()
                raise ValueError(
                    f"Orbital information does not cover minimum ({self.tf.min()}) and maximum ({self.tf.max()}) tf. Orbital information begins at {self.orbits.t_base.min()} and ends at {self.orbits.t_base.max()}."
                )

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
            self.orbits,
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
