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
from fastlisaresponse.tdionfly import FDTDIonTheFly

class BBHTDIonTheFly(BBHxParallelModule):
    """

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
        
        self.fd_gen = FDTDIonTheFly

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
        return ["bbhx_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

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
        return_spline=False,  # TODO: docs!
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

        # TODO: maybe get rid of this since this will be the "newer structure"
        response_carrier_shaped = self.response_carrier.reshape(self.nparams, self.num_bin_all, self.num_modes, self.length)
        freqs_shaped = self.freqs.reshape(self.num_bin_all, self.num_modes, self.length)
        t_shaped = response_carrier_shaped[2]

        # TODO: check this
        print("CHANGE THIS!!!! Right now requires same number of points. May need to adjust.")
        _tmp1 = self.xp.argwhere(t_shaped < 1000.0)
        start_index = _tmp1[:, 2].max().item() + 1
        _tmp2 = self.xp.argwhere(t_shaped > t_shaped.max(axis=-1)[:, :, None] - 1000.0)
        end_index = _tmp2[:, 2].min().item() - 1
        inds_keep = self.xp.arange(start_index, end_index + 1)
        if start_index + 1 > end_index:
            breakpoint()

        t_tdi = t_shaped[:, :, inds_keep].reshape(-1, end_index - start_index + 1)
        # SOME CHOICES:
        #   * RESAMPLE ON NEW T KEEPING SAME OVERALL LENGTH
        #   * ONCE (OR WE CAN NOW) THE SPLINES ARE UPDATED TO HAVE VARIABLE NUMBER OF COMPONENTS,
        #    WE COULD USE THAT. 
        amp_shaped = response_carrier_shaped[0]

        # TODO: !!!! What do with turnover in time? !!!!!

        # _ind_keep = np.where(np.diff(t_arr) < 0.0)[0]
        # if len(_ind_keep) > 0:
        #     ind_keep = _ind_keep[0] + 1
        # else:
        #     ind_keep = t_arr.shape[0]
        from gpubackendtools.interpolate import CubicSplineInterpolant
        amp_spl = CubicSplineInterpolant(t_shaped, amp_shaped, force_backend=self.backend.name.split("_")[-1])
        freq_spl = CubicSplineInterpolant(t_shaped, freqs_shaped, force_backend=self.backend.name.split("_")[-1])

        from fastlisaresponse.tdionfly import FDTDIonTheFly

        dt = 10000.0
    
        fs = 0.1
        CUBIC_SPLINE_LINEAR_SPACING = 1
        CUBIC_SPLINE_LOG10_SPACING = 2
        CUBIC_SPLINE_GENERAL_SPACING = 3
        # 11 is nparams / will not affect spline
        fd_wave_gen = self.fd_gen(t_tdi, amp_spl, freq_spl, fs, self.num_bin_all * self.num_modes, 11, spline_type=CUBIC_SPLINE_GENERAL_SPACING)
        _inc = self.xp.repeat(inc, self.num_modes)
        _psi = self.xp.repeat(psi, self.num_modes)
        _lam = self.xp.repeat(lam, self.num_modes)
        _beta = self.xp.repeat(beta, self.num_modes)
        wave_output = fd_wave_gen(_inc, _psi, _lam, _beta, return_spline=return_spline)
        
        
        import matplotlib.pyplot as plt
        plt.plot(wave_output.t.T, wave_output.X.real.T)
        plt.plot(wave_output.t.T, wave_output.X.real.T)
        plt.show()
        plt.plot(wave_output.t.T, wave_output.Y.real.T)
        plt.plot(wave_output.t.T, wave_output.Y.real.T)
        plt.show()
        plt.plot(wave_output.t.T, wave_output.Z.real.T)
        plt.plot(wave_output.t.T, wave_output.Z.real.T)
        plt.show()
        breakpoint()
        inds_tmp = (self.xp.tile(inds_keep, (self.num_bin_all * self.num_modes, 1)) + self.length * self.xp.repeat(self.xp.arange(self.num_bin_all * self.num_modes)[:, None], len(inds_keep), axis=-1)).flatten()
        # need to vectorize response setup

        self.response_carrier[
            (includes_amps + 2) * length * num_modes * num_bin_all + inds_tmp
        ] = wave_output.X.real.flatten()

        self.response_carrier[
            (includes_amps + 3) * length * num_modes * num_bin_all + inds_tmp
        ] = wave_output.X.imag.flatten()

        self.response_carrier[
            (includes_amps + 4) * length * num_modes * num_bin_all + inds_tmp
        ] = wave_output.Y.real.flatten()

        self.response_carrier[
            (includes_amps + 5) * length * num_modes * num_bin_all + inds_tmp
        ] = wave_output.Y.imag.flatten()

        self.response_carrier[
            (includes_amps + 6) * length * num_modes * num_bin_all + inds_tmp
        ] = wave_output.Z.real.flatten()

        self.response_carrier[
            (includes_amps + 7) * length * num_modes * num_bin_all + inds_tmp
        ] = wave_output.Z.imag.flatten()

        return wave_output

        # adjust input phase arrays in-place
        # if use_phase_tf and adjust_phase:
        #     output = self.response_carrier[
        #         (includes_amps + 0)
        #         * length
        #         * num_modes
        #         * num_bin_all : (includes_amps + 1)
        #         * length
        #         * num_modes
        #         * num_bin_all
        #     ]

        #     if phase.ndim > 1:
        #         phase[:] = output.reshape(phase.shape)
        #     else:
        #         phase[:] = output
