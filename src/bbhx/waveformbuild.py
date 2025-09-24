# Main waveform class location

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

import numpy as np
import warnings

# import gpu stuff
try:
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    print("No CuPy")

from .waveforms.phenomhm import PhenomHMAmpPhase
from .response.fastfdresponse import LISATDIResponse
from .utils.transform import tSSBfromLframe, tLfromSSBframe
from .utils.interpolate import CubicSplineInterpolant
from .utils.constants import *
from .utils.citations import *
from .utils.parallelbase import BBHxParallelModule

class TemplateInterpFD(BBHxParallelModule):
    """Interpolate frequency domain template.

    This class wraps :class:`CubicSplineInterpolant <bbhx.utils.interpolate.CubicSplineInterpolant>` so
    that it fits into this specific waveform production method.

    This class has GPU capabilities.

    Args:
        force_backend (str, optional): ``"cpu"'', ``"gpu"'', ``"cuda"'', ``"cuda12x"'', or ``"cuda11x"''.

    Attributes:
        data_length (int): Length of data. This class interpolates to this length.
        length (int): Length of original frequency array.
        num_bin_all (int): Number of binaries.
        num_channels (int): Number of channels in data.
        num_modes (int): Number of harmonics.
        template_carrier (complex128 xp.ndarray): Carrier for output templates.
            Templates can be accessed through the ``template_channels`` property.
        force_backend (str): Backend string indicator.

    """

    def __init__(self, force_backend=None):
        super().__init__(force_backend=force_backend)

    @property
    def xp(self) -> object:
        """Numpy or Cupy"""
        return self.backend.xp

    @property
    def template_gen(self) -> callable:
        """C/CUDA wrapped function for computing interpolated waveforms"""
        return self.backend.InterpTDI_wrap

    @property
    def template_channels(self):
        """Get template channels from ``self.template_carrier``."""
        return [
            self.template_carrier[i].reshape(self.num_channels, self.lengths[i])
            for i in range(self.num_bin_all)
        ]

    @property
    def citation(self):
        """citations for this class"""
        return katz_citations

    def __call__(
        self,
        data_freqs,
        interp_container,
        t_start,
        t_end,
        length,
        num_modes,
        num_channels,
    ):
        """Generate frequency domain template via interpolation.

        This class takes all waveform and response information as sparse arrays
        and then interpolates to the proper frequency array.

        This class acts in a unique why by passing arrays of pointers from python
        into C++/CUDA.

        Args:
            data_freqs (double xp.ndarray): Frequencies to interpolate to.
            interp_container (obj): ``container`` attribute from the interpolant
                class: :class:`CubicSplineInterpolant <bbhx.utils.interpolate.CubicSplineInterpolant>`.
            t_start (double xp.ndarray): Array of start times (sec) for each binary.
            t_end (double xp.ndarray): Array of end times (sec) for each binary.
            length (int): Length of original frequency array.
            num_modes (int): Number of harmonics.
            num_channels (int): Number of channels in data.

        Returns:
            list: List of template arrays for all binaries.
                shape of each array: ``(self.num_channels, self.data_length)``

        """

        # fill important quantities
        num_bin_all = len(t_start)
        self.length = length
        self.num_modes = num_modes
        self.num_bin_all = num_bin_all
        self.data_length = len(data_freqs)
        self.num_channels = num_channels

        # unpack interp_container
        (freqs, y, c1, c2, c3) = interp_container

        freqs_shaped = freqs.reshape(self.num_bin_all, self.num_modes, -1)

        # find where each binary's signal starts and ends in the data array

        # lowest frequencies in all the modes. Highest frequencies in all the modes.
        start_and_end = self.xp.asarray(
            [
                freqs_shaped.min(axis=(1, 2)),
                freqs_shaped.max(axis=(1, 2)),
            ]
        ).T

        if self.backend.name != "bbhx_cpu" and not isinstance(data_freqs, self.xp.ndarray):
            raise ValueError(
                "Make sure if using Cupy or Numpy, the input freqs array is of the same type."
            )

        inds_start_and_end = self.xp.asarray(
            [
                self.xp.searchsorted(data_freqs, temp, side="right")
                for temp in start_and_end
            ]
        )

        # find proper interpolation window for each point in data stream
        inds = [
            np.concatenate(
                [
                    self.xp.searchsorted(
                        freqs_shaped[i, j], data_freqs[st:et], side="right"
                    )
                    - 1
                    for j in range(self.num_modes)
                ]
            )
            .astype(self.xp.int32)
            .copy()
            for i, (st, et) in enumerate(inds_start_and_end)
        ]

        # lengths of the signals in frequency domain
        self.lengths = lengths = np.asarray(
            [int(len(inds_i) / self.num_modes) for inds_i in inds], dtype=self.xp.int32
        )

        # make sure have this quantity available on CPU
        try:
            temp_inds = inds_start_and_end[:, 0].get()
        except AttributeError:
            temp_inds = inds_start_and_end[:, 0]

        # where to start filling waveform
        self.start_inds = start_inds = (temp_inds.copy()).astype(np.int32)

        # get pointers to all the index arrays of different length
        try:
            self.ptrs = ptrs = np.asarray([ind_i.data.ptr for ind_i in inds])
        except AttributeError:
            self.ptrs = ptrs = np.asarray(
                [ind_i.__array_interface__["data"][0] for ind_i in inds]
            )

        # initialize template information
        self.template_carrier = [
            self.xp.zeros(
                int(self.num_channels * temp_length),
                dtype=self.xp.complex128,
            )
            for temp_length in lengths
        ]

        # get pointers to template carriers so they can be run in streams
        try:
            template_carrier_ptrs = np.asarray(
                [temp_carrier.data.ptr for temp_carrier in self.template_carrier]
            )
        except AttributeError:
            template_carrier_ptrs = np.asarray(
                [
                    temp_carrier.__array_interface__["data"][0]
                    for temp_carrier in self.template_carrier
                ]
            )

        # fill templates
        self.template_gen(
            template_carrier_ptrs,
            data_freqs,
            freqs,
            y,
            c1,
            c2,
            c3,
            t_start,
            t_end,
            self.length,
            self.data_length,
            self.num_bin_all,
            self.num_modes,
            ptrs,
            start_inds,
            lengths,
        )

        # return templates in the right shape
        return self.template_channels


class BBHWaveformFD(BBHxParallelModule):
    """Generate waveforms put through response functions

    This class generates waveforms put through the LISA response function. In the
    future, ground-based analysis may be added. Therefore, it currently
    returns the TDI variables according the response keyword arguments given.

    If you use this class, please cite `arXiv:2005.01827 <https://arxiv.org/abs/2005.01827>`_
    and `arXiv:2111.01064 <https://arxiv.org/abs/2111.01064>`_, as well as the papers
    listed for the waveform and response given just below.

    Right now, it is hard coded to produce the waveform with
    :class:`PhenomHMAmpPhase <bbhx.waveforms.phenomhm.PhenomHMAmpPhase>`. This can also be used
    to produce PhenomD. See the docs for that waveform. The papers describing PhenomHM/PhenomD
    waveforms are here: `arXiv:1708.00404 <https://arxiv.org/abs/1708.00404>`_,
    `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_, and
    `arXiv:1508.07253 <https://arxiv.org/abs/1508.07253>`_.

    The response function is the fast frequency domain response function
    from `arXiv:1806.10734 <https://arxiv.org/abs/1806.10734>`_ and
    `arXiv:2003.00357 <https://arxiv.org/abs/2003.00357>`_. It is implemented in
    :class:`LISATDIResponse <bbhx.response.fastfdresponse.LISATDIResponse`.

    This class is GPU accelerated.

    Args:
        amp_phase_kwargs (dict, optional): Keyword arguments for the
            initialization of the ampltidue-phase waveform class: :class:`PhenomHMAmpPhase <bbhx.waveforms.phenomhm.PhenomHMAmpPhase>`.
        response_kwargs (dict, optional): Keyword arguments for the initialization
            of the response class: :class:`LISATDIResponse <bbhx.response.fastfdresponse.LISATDIResponse`.
        interp_kwargs (dict, optional): Keyword arguments for the initialization
            of the interpolation class: :class:`TemplateInterpFD`.
        force_backend (str, optional): ``"cpu"'', ``"gpu"'', ``"cuda"'', ``"cuda12x"'', or ``"cuda11x"''.

    Attributes:
        amp_phase_gen (obj): Waveform generation class.
        data_length (int): Length of the final output data.
        interp_response (obj): Interpolation class.
        length (int): Length of initial evaluations of waveform and response.
        num_bin_all (int): Total number of binaries analyzed.
        num_interp_params (int): Number of parameters to interpolate (9).
        num_modes (int): Number of harmonic modes.
        out_buffer_final (xp.ndarray): Array with buffer information with shape:
            ``(self.num_interp_params, self.num_bin_all, self.num_modes, self.length)``.
            The order of the parameters is amplitude, phase, t-f, transferL1re, transferL1im,
            transferL2re, transferL2im, transferL3re, transferL3im.

    """

    def __init__(
        self,
        amp_phase_kwargs={},
        response_kwargs={},
        interp_kwargs={},
        force_backend=None,
    ):
        super().__init__(force_backend=force_backend)
        self.force_backend = force_backend
        # initialize waveform and response funtions
        self.amp_phase_gen = PhenomHMAmpPhase(**amp_phase_kwargs, force_backend=force_backend)
        self.response_gen = LISATDIResponse(**response_kwargs, force_backend=force_backend)

        self.num_interp_params = 9

        # setup the final interpolant
        self.interp_response = TemplateInterpFD(**interp_kwargs, force_backend=force_backend)

    @property
    def xp(self) -> object:
        """Numpy or Cupy"""
        return self.backend.xp

    @property
    def waveform_gen(self) -> callable:
        """C/CUDA wrapped function for computing waveforms"""
        return self.backend.direct_sum_wrap

    @property
    def citation(self):
        """Citations for this class"""
        return (
            katz_citations + marsat_1 + marsat_2 + phenomhm_citation + phenomd_citations
        )

    def __call__(
        self,
        m1,
        m2,
        chi1z,
        chi2z,
        distance,
        phi_ref,
        f_ref,
        inc,
        lam,
        beta,
        psi,
        t_ref,
        t_obs_start=0.0,
        t_obs_end=1.0,
        freqs=None,
        length=None,
        modes=None,
        shift_t_limits=True,
        direct=False,
        compress=True,
        squeeze=False,
        fill=False,
        combine=False,
    ):
        """Generate the binary black hole frequency-domain TDI waveforms


        Args:
            m1 (double scalar or np.ndarray): Mass 1 in Solar Masses :math:`(m1 > m2)`.
            m2 (double or np.ndarray): Mass 2 in Solar Masses :math:`(m1 > m2)`.
            chi1z (double or np.ndarray): Dimensionless spin 1 (for Mass 1) in Solar Masses.
            chi2z (double or np.ndarray): Dimensionless spin 2 (for Mass 1) in Solar Masses.
            distance (double or np.ndarray): Luminosity distance in m.
            phi_ref (double or np.ndarray): Phase at ``f_ref``.
            f_ref (double or np.ndarray): Reference frequency at which ``phi_ref`` and ``t_ref`` are set.
                If ``f_ref == 0``, it will be set internally by the PhenomHM code
                to :math:`f_\\text{max} = \\text{max}(f^2A_{22}(f))`.
            inc (double or np.ndarray): Inclination of the binary in radians :math:`(\iota\in[0.0, \pi])`.
            lam (double or np.ndarray): Ecliptic longitude :math:`(\lambda\in[0.0, 2\pi])`.
            beta (double or np.ndarray): Ecliptic latitude :math:`(\\beta\in[-\pi/2, \pi/2])`.
            psi (double or np.ndarray): Polarization angle in radians :math:`(\psi\in[0.0, \pi])`.
            t_ref (double or np.ndarray): Reference time in seconds. It is set at ``f_ref``.
            t_obs_start (double, optional): Start time of observation in years
                in the LISA constellation reference frame. This is with reference to :math:`t=0`.
                (Default: 0.0)
            t_obs_end (double, optional): End time of observation in years in the
                LISA constellation reference frame. This is with reference to :math:`t=0`.
                (Default: 1.0)
            freqs (np.ndarray, optional): Frequencies at which to evaluate the final waveform.
                If ``length`` is also given, the interpolants interpolate to these
                frequencies. If ``length`` is not given, the waveform amplitude, phase,
                and response will be directly evaluated at these frequencies. In this case,
                a 2D np.ndarray can also be provided. (Default: ``None``)
            length (int, optional): Number of frequencies to use in sparse array for
                interpolation.
            modes (list, optional): Harmonic modes to use. If not given, they will
                default to those available in the waveform model. For PhenomHM:
                [(2,2), (3,3), (4,4), (2,1), (3,2), (4,3)]. For PhenomD: [(2,2)].
                (Default: ``None``)
            shift_t_limits (bool, optional): If ``False``, ``t_obs_start`` and ``t_obs_end``
                are relative to ``t_ref`` counting backwards in time. If ``True``,
                those quantities are relative to :math:`t=0`. (Default: ``False``)
            direct (bool, optional): If ``True``, directly compute the waveform without
                interpolation. (Default: ``False``)
            compress (bool, optional): If ``True``, combine harmonics into single channel
                waveforms. (Default: ``True``)
            squeeze (bool, optional): If ``True``, remove any axes of length 1 from the
                final return information. (Default: ``False``)
            fill (bool, optional): If ``True``, fill data streams according to the ``combine``
                keyword argument. If ``False, returns information for the fast likelihood functions.
            combine (bool, optional): If ``True``, combine all waveforms into the same output
                data stream. (Default: ``False``)


        Returns:
            xp.ndarray: Shape ``(3, self.length, self.num_bin_all)``.
                Final waveform for each binary. If ``direct==True`` and ``compress==True``.
                # TODO: switch dimensions?
            xp.ndarray:  Shape ``(3, self.num_modes, self.length, self.num_bin_all)``.
                Final waveform for each binary. If ``direct==True`` and ``compress==False``.
            xp.ndarray:  Shape ``(3, self.data_length)``.
                Final waveform of all binaries in the same data stream.
                If ``fill==True`` and ``combine==True``.
            xp.ndarray:  Shape ``(self.num_bin_all, 3, self.data_length)``.
                Final waveform of all binaries in the same data stream.
                If ``fill==True`` and ``combine==False``.
            tuple: Information for fast likelihood functions.
                First entry is ``template_channels`` property from :class:`TemplateInterpFD`.
                Second entry is ``start_inds`` attribute from ``self.interp_response``.
                Third entry is ``lengths`` attribute from ``self.interp_response``.

        Raises:
            ValueError: ``length`` and ``freqs`` not given. Modes are given but not in a list.


        """

        # make sure everything is at least a 1D array
        m1 = np.atleast_1d(m1)
        m2 = np.atleast_1d(m2)
        chi1z = np.atleast_1d(chi1z)
        chi2z = np.atleast_1d(chi2z)
        distance = np.atleast_1d(distance)
        phi_ref = np.atleast_1d(phi_ref)
        inc = np.atleast_1d(inc)
        lam = np.atleast_1d(lam)
        beta = np.atleast_1d(beta)
        psi = np.atleast_1d(psi)
        t_ref = np.atleast_1d(t_ref)

        self.num_bin_all = len(m1)

        # TODO: add sanity checks for t_start, t_end
        # how to set up time limits
        if shift_t_limits is False:
            warnings.warn(
                "Deprecated: shift_t_limits. Previously shift_t_limits defaulted to False. This option is now removed and permanently set to shift_t_limits=True."
            )
            # t_ref_L = tLfromSSBframe(t_ref, lam, beta)

            # # start and end times are defined in the LISA reference frame
            # t_obs_start_L = t_ref_L - t_obs_start * YRSID_SI
            # t_obs_end_L = t_ref_L - t_obs_end * YRSID_SI

            # # convert to SSB frame
            # t_obs_start_SSB = tSSBfromLframe(t_obs_start_L, lam, beta, 0.0)
            # t_obs_end_SSB = tSSBfromLframe(t_obs_end_L, lam, beta, 0.0)

            # # fix zeros and less than zero
            # t_start = (
            #     t_obs_start_SSB if t_obs_start > 0.0 else np.zeros(self.num_bin_all)
            # )
            # t_end = t_obs_end_SSB if t_obs_end > 0.0 else np.zeros_like(t_start)

        # else:
        # start and end times are defined in the LISA reference frame
        t_obs_start_L = t_obs_start * YRSID_SI
        t_obs_end_L = t_obs_end * YRSID_SI

        # convert to SSB frame
        t_obs_start_SSB = tSSBfromLframe(t_obs_start_L, lam, beta, 0.0)
        t_obs_end_SSB = tSSBfromLframe(t_obs_end_L, lam, beta, 0.0)
        t_start = np.atleast_1d(t_obs_start_SSB)
        t_end = np.atleast_1d(t_obs_end_SSB)

        if freqs is None and length is None:
            raise ValueError("Must input freqs or length.")

        # if computing directly, set info
        elif freqs is not None and direct is True:
            if freqs.ndim == 1:
                # set the length of the input
                length = len(freqs)
            else:
                length = freqs.shape[1]

        # this means the frequencies are what needs to be interpolated to
        elif direct is False:
            self.data_length = len(freqs)
            if length is None:
                raise ValueError("If direct is False, length parameter must be given.")

        self.length = length

        # setup harmonic modes
        if modes is None:
            # default mode setup
            self.num_modes = len(self.amp_phase_gen.allowable_modes)
        else:
            if not isinstance(modes, list):
                raise ValueError("modes must be a list.")
            self.num_modes = len(modes)

        self.num_bin_all = len(m1)

        out_buffer = self.xp.zeros(
            (self.num_interp_params * self.length * self.num_modes * self.num_bin_all)
        )

        freqs_temp = freqs if direct else None

        phi_ref_amp_phase = np.zeros_like(m1)

        Tobs = t_end - t_start
        self.amp_phase_gen(
            m1,
            m2,
            chi1z,
            chi2z,
            distance,
            phi_ref_amp_phase,
            f_ref,
            t_ref,
            length,
            freqs=freqs_temp,
            out_buffer=out_buffer,
            modes=modes,
            Tobs=Tobs,
            direct=direct,
        )

        # setup buffer to carry around all the quantities of interest
        # params are amp, phase, tf, transferL1re, transferL1im, transferL2re, transferL2im, transferL3re, transferL3im
        out_buffer = out_buffer.reshape(
            self.num_interp_params, self.num_bin_all, self.num_modes, self.length
        )

        out_buffer = out_buffer.flatten().copy()

        # compute response function
        self.response_gen(
            self.amp_phase_gen.freqs,
            inc,
            lam,
            beta,
            psi,
            phi_ref,
            length,
            out_buffer=out_buffer,  # fill into this buffer
            modes=self.amp_phase_gen.modes,
            direct=direct,
        )

        # for checking
        self.out_buffer_final = out_buffer.reshape(
            9, self.num_bin_all, self.num_modes, self.length
        ).copy()

        # direct computation from buffer
        # + compressing all harmonics into a single data stream by direct combination
        if direct and compress:
            # setup template
            templateChannels = self.xp.zeros(
                (self.num_bin_all * 3 * self.length), dtype=self.xp.complex128
            )

            # direct computation of 3 channel waveform
            self.waveform_gen(
                templateChannels,
                out_buffer,
                self.num_bin_all,
                self.length,
                3,
                self.num_modes,
                self.xp.asarray(t_start),
                self.xp.asarray(t_end),
            )

            out = templateChannels.reshape(self.num_bin_all, 3, self.length)

            if squeeze:
                out = out.squeeze()

            return out

        elif direct:
            out = self.xp.zeros(
                (self.num_bin_all, 3, self.num_modes, self.length),
                dtype=self.xp.complex128,
            )
            for mode_i in range(self.num_modes):
                # setup template
                templateChannels = self.xp.zeros(
                    (self.num_bin_all * 3 * self.length), dtype=self.xp.complex128
                )

                out_buffer_temp = (
                    self.out_buffer_final[:, :, mode_i, :].copy().flatten()
                )
                # direct computation of 3 channel waveform
                self.waveform_gen(
                    templateChannels,
                    out_buffer_temp,
                    self.num_bin_all,
                    self.length,
                    3,
                    1,  # num_modes
                    self.xp.asarray(t_start),
                    self.xp.asarray(t_end),
                )

                out[:, :, mode_i, :] += templateChannels.reshape(
                    self.num_bin_all, 3, self.length
                )

            if squeeze:
                out = out.squeeze()

            return out

        else:

            # setup interpolant
            spline = CubicSplineInterpolant(
                self.amp_phase_gen.freqs,
                out_buffer,
                length=self.length,
                num_interp_params=self.num_interp_params,
                num_modes=self.num_modes,
                num_bin_all=self.num_bin_all,
                force_backend=self.force_backend,
            )

            # TODO: try single block reduction for likelihood (will probably be worse for smaller batch, but maybe better for larger batch)?
            template_channels = self.interp_response(
                freqs,
                spline.container,
                t_start,
                t_end,
                self.length,
                self.num_modes,
                3,
            )

            # fill the data stream
            if fill:
                if combine:
                    # combine into one data stream
                    data_out = self.xp.zeros((3, len(freqs)), dtype=self.xp.complex128)
                    for temp, start_i, length_i in zip(
                        template_channels,
                        self.interp_response.start_inds,
                        self.interp_response.lengths,
                    ):
                        data_out[:, start_i : start_i + length_i] += temp

                    if squeeze:
                        return data_out.squeeze()

                    return data_out

                else:
                    # put in separate data streams
                    data_out = self.xp.zeros(
                        (self.num_bin_all, 3, len(freqs)), dtype=self.xp.complex128
                    )
                    for bin_i, (temp, start_i, length_i) in enumerate(
                        zip(
                            template_channels,
                            self.interp_response.start_inds,
                            self.interp_response.lengths,
                        )
                    ):
                        data_out[bin_i, :, start_i : start_i + length_i] = temp

                return data_out
            else:
                # return information for the fast likelihood functions
                return (
                    template_channels,
                    self.interp_response.start_inds,
                    self.interp_response.lengths,
                )
