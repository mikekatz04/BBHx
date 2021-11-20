import numpy as np

try:
    import cupy as xp
    from pyWaveformBuild import direct_sum_wrap as direct_sum_wrap_gpu
    from pyWaveformBuild import InterpTDI_wrap as InterpTDI_wrap_gpu
    from pyWaveformBuild import TDInterp_wrap2 as TDInterp_wrap_gpu

except (ImportError, ModuleNotFoundError) as e:
    print("No CuPy")
    import numpy as xp

from lisatools.utils.transform import tSSBfromLframe

from pyWaveformBuild_cpu import direct_sum_wrap as direct_sum_wrap_cpu
from pyWaveformBuild_cpu import InterpTDI_wrap as InterpTDI_wrap_cpu
from pyWaveformBuild_cpu import TDInterp_wrap2 as TDInterp_wrap_cpu

from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
from bbhx.response.fastfdresponse import LISATDIResponse
from bbhx.utils.interpolate import CubicSplineInterpolant, CubicSplineInterpolantTD
from bbhx.utils.constants import *


class TemplateInterp:
    def __init__(self, max_init_len=-1, use_gpu=False):

        if use_gpu:
            self.template_gen = InterpTDI_wrap_gpu
            self.xp = xp

        else:
            self.template_gen = InterpTDI_wrap_cpu
            self.xp = np

        if max_init_len > 0:
            self.use_buffers = True
            raise NotImplementedError

        else:
            self.use_buffers = False

    def _initialize_template_container(self):
        self.template_carrier = self.xp.zeros(
            (self.num_bin_all * self.nChannels * self.data_length),
            dtype=self.xp.complex128,
        )

    @property
    def template_channels(self):
        return [
            self.template_carrier[i].reshape(self.nChannels, self.lengths[i])
            for i in range(self.num_bin_all)
        ]

    def __call__(
        self,
        dataFreqs,
        interp_container,
        t_start,
        t_end,
        length,
        data_length,
        num_modes,
        t_obs_start,
        t_obs_end,
        nChannels,
    ):

        numBinAll = len(t_start)
        self.length = length
        self.num_modes = num_modes
        self.num_bin_all = numBinAll
        self.data_length = data_length
        self.nChannels = nChannels

        # self._initialize_template_container()

        (freqs, y, c1, c2, c3) = interp_container

        freqs_shaped = freqs.reshape(self.num_bin_all, -1)

        start_and_end = self.xp.asarray([freqs_shaped[:, 0], freqs_shaped[:, -1],]).T

        inds_start_and_end = self.xp.asarray(
            [
                self.xp.searchsorted(dataFreqs, temp, side="right")
                for temp in start_and_end
            ]
        )

        inds = [
            self.xp.searchsorted(
                freqs_shaped[i], dataFreqs[st:et], side="right"
            ).astype(self.xp.int32)
            - 1
            for i, (st, et) in enumerate(inds_start_and_end)
        ]

        self.lengths = lengths = np.asarray(
            [len(inds_i) for inds_i in inds], dtype=self.xp.int32
        )

        try:
            temp_inds = inds_start_and_end[:, 0].get()
        except AttributeError:
            temp_inds = inds_start_and_end[:, 0]

        self.start_inds = start_inds = (temp_inds.copy()).astype(np.int32)

        try:
            self.ptrs = ptrs = np.asarray([ind_i.data.ptr for ind_i in inds])
        except AttributeError:
            self.ptrs = ptrs = np.asarray(
                [ind_i.__array_interface__["data"][0] for ind_i in inds]
            )

        self.template_carrier = [
            self.xp.zeros(int(self.nChannels * temp_length), dtype=self.xp.complex128,)
            for temp_length in lengths
        ]

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

        dlog10f = 1.0

        self.template_gen(
            template_carrier_ptrs,
            dataFreqs,
            dlog10f,
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
            t_obs_start,
            t_obs_end,
            ptrs,
            start_inds,
            lengths,
        )

        return self.template_channels


class BBHWaveformFD:
    """Generate waveforms put through response functions

    This class generates waveforms put through the LISA response function. In the
    future, ground-based analysis may be added. Therefore, it currently
    returns the TDI variables according the response keyword arguments given.

    If you use this class, please cite `arXiv:2005.01827 <https://arxiv.org/abs/2005.01827>`_
    and `arXiv:2111.01064 <https://arxiv.org/abs/2111.01064>`_, as well as the papers
    listed for the waveform and response given just below.

    Right now, it is hard coded to produce the waveform with
    :class:`PhenomHMAmpPhase`. This can also be used
    to produce PhenomD. See the docs for that waveform. The papers describing PhenomHM/PhenomD
    waveforms are here: `arXiv:1708.00404 <https://arxiv.org/abs/1708.00404>`_,
    `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_, and
    `arXiv:1508.07253 <https://arxiv.org/abs/1508.07253>`_.

    The response function is the fast frequency domain response function
    from `arXiv:1806.10734 <https://arxiv.org/abs/1806.10734>`_ and
    `arXiv:2003.00357 <https://arxiv.org/abs/2003.00357>`_. It is implemented in
    :class:`LISATDIResponse`.

    This class is GPU accelerated.

    Args:
        amp_phase_kwargs (dict, optional): Keyword arguments for the
            initialization of the ampltidue-phase waveform class: :class:`PhenomHMAmpPhase`.
        response_kwargs (dict, optional): Keyword arguments for the initialization
            of the response class: :class:`LISATDIResponse`.
        interp_kwargs (dict, optional): Keyword arguments for the initialization
            of the interpolation class: :class:`TemplateInterp`.
        use_gpu (bool, optional): If ``True``, use a GPU. (Default: ``False``)

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
        response_gen (obj): Response generation class.
        use_gpu (bool): A GPU is being used if ``use_gpu==True``.
        waveform_gen (obj): Direct summation waveform generation class.
        xp (obj): Either ``numpy`` or ``cupy``.

    """

    def __init__(
        self, amp_phase_kwargs={}, response_kwargs={}, interp_kwargs={}, use_gpu=False,
    ):

        # initialize waveform and response funtions
        self.amp_phase_gen = PhenomHMAmpPhase(**amp_phase_kwargs, use_gpu=use_gpu)
        self.response_gen = LISATDIResponse(**response_kwargs, use_gpu=use_gpu)

        self.use_gpu = use_gpu
        if use_gpu:
            self.xp = xp
            self.waveform_gen = direct_sum_wrap_gpu
        else:
            self.xp = np
            self.waveform_gen = direct_sum_wrap_cpu

        self.num_interp_params = 9

        # setup the final interpolant
        self.interp_response = TemplateInterp(**interp_kwargs, use_gpu=use_gpu)

    def __call__(
        self,
        m1,
        m2,
        chi1z,
        chi2z,
        distance,
        phiRef,
        f_ref,
        inc,
        lam,
        beta,
        psi,
        t_ref,
        t_obs_start=1.0,
        t_obs_end=0.0,
        freqs=None,
        length=None,
        modes=None,
        shift_t_limits=False,
        direct=False,
        compress=True,
        squeeze=False,
        fill=False,
        combine=False,
    ):
        """Generate the binary black hole frequency-domain TDI waveforms


        Args:
            m1 (double): Mass 1 in Solar Masses :math:`(m1 > m2)`.
            m2 (double): Mass 2 in Solar Masses :math:`(m1 > m2)`.
            chi1z (double): Dimensionless spin 1 (for Mass 1) in Solar Masses.
            chi2z (double): Dimensionless spin 2 (for Mass 1) in Solar Masses.
            distance (double): Luminosity distance in m.
            phiRef (double): Phase at ``f_ref``.
            f_ref (double): Reference frequency at which ``phi_ref`` and ``t_ref`` are set.
                If ``f_ref == 0``, it will be set internally by the PhenomHM code
                to :math:`f_\\text{max} = \\text{max}(f^2A_{22}(f))`.
            inc (double): Inclination of the binary in radians :math:`(\iota\in[0.0, \pi])`.
            lam (double): Ecliptic longitude :math:`(\lambda\in[0.0, 2\pi])`.
            beta (double): Ecliptic latitude :math:`(\\beta\in[-\pi/2, \pi/2])`.
            psi (double): Polarization angle in radians :math:`(\psi\in[0.0, \pi])`.
            t_ref (double): Reference time in seconds. It is set at ``f_ref``.
            t_obs_start (double, optional): Start time of observation in years
                in the LISA constellation reference frame. If ``shift_t_limits==True``,
                this is with reference to :math:`t=0`. If ``shift_t_limits==False`` this is
                with reference to ``t_ref`` and works backwards. So, in this case,
                ``t_obs_start`` gives how much time back from merger to start the waveform.
                (Default: 1.0)
            t_obs_end (double, optional): End time of observation in years in the
                LISA constellation reference frame. If
                ``shift_t_limits==True``, this is with reference to :math:`t=0`.
                If ``shift_t_limits==False`` this is with reference to ``t_ref``
                and works backwards. So, in this case, ``t_obs_end`` gives how much time
                back from merger to start the waveform. If the value is zero, it takes
                everything after the merger as well. (Default: 0.0)
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
                First entry is ``template_channels`` property from :class:`TemplateInterp`.
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
        phiRef = np.atleast_1d(phiRef)
        inc = np.atleast_1d(inc)
        lam = np.atleast_1d(lam)
        beta = np.atleast_1d(beta)
        psi = np.atleast_1d(psi)
        t_ref = np.atleast_1d(t_ref)

        self.num_bin_all = len(m1)

        # TODO: add sanity checks for t_start, t_end
        # how to set up time limits
        if shift_t_limits is False:
            # start and end times are defined in the LISA reference frame
            t_obs_start_L = t_ref - t_obs_start * YRSID_SI
            t_obs_end_L = t_ref - t_obs_end * YRSID_SI

            # convert to SSB frame
            t_obs_start_SSB = tSSBfromLframe(t_obs_start_L, lam, beta, 0.0)
            t_obs_end_SSB = tSSBfromLframe(t_obs_end_L, lam, beta, 0.0)

            # fix zeros and less than zero
            t_start = (
                t_obs_start_SSB if t_obs_start > 0.0 else np.zeros(self.num_bin_all)
            )
            t_end = t_obs_end_SSB if t_obs_end > 0.0 else np.zeros_like(t_start)

        else:
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

        phiRef_amp_phase = np.zeros_like(m1)

        self.amp_phase_gen(
            m1,
            m2,
            chi1z,
            chi2z,
            distance,
            phiRef_amp_phase,
            f_ref,
            length,
            freqs=freqs_temp,
            out_buffer=out_buffer,
            modes=modes,
        )

        # setup buffer to carry around all the quantities of interest
        # params are amp, phase, tf, transferL1re, transferL1im, transferL2re, transferL2im, transferL3re, transferL3im
        out_buffer = out_buffer.reshape(
            self.num_interp_params, self.num_bin_all, self.num_modes, self.length
        )

        # adjust phases based on shift from t_ref
        phases = out_buffer[1].copy()
        phases = (
            self.amp_phase_gen.freqs.reshape(self.num_bin_all, -1)
            * self.xp.asarray(t_ref[:, self.xp.newaxis])
            * 2
            * np.pi
        )[:, self.xp.newaxis, :] + phases

        # TODO: can switch between spline derivatives and actual derivatives
        out_buffer[1] = phases.copy()

        # adjust t-f for shift of t_ref
        out_buffer[2] += self.xp.asarray(t_ref[:, self.xp.newaxis, self.xp.newaxis])

        out_buffer = out_buffer.flatten().copy()

        # compute response function
        self.response_gen(
            self.amp_phase_gen.freqs,
            inc,
            lam,
            beta,
            psi,
            t_ref,
            phiRef,
            length,
            includes_amps=True,
            out_buffer=out_buffer,  # fill into this buffer
            modes=self.amp_phase_gen.modes,
        )

        # for checking
        self.out_buffer_final = out_buffer.reshape(
            9, self.num_bin_all, self.num_modes, self.length
        ).copy()

        # direct computation from buffer
        # + compressing all harmonics into a single data stream by diret combination
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

            out = templateChannels.reshape(self.num_bin_all, 3, self.length).transpose(
                (1, 2, 0)
            )

            if squeeze:
                out = out.squeeze()

            return out

        if direct:
            temp = self.xp.swapaxes(
                out_buffer.reshape(
                    self.num_interp_params,
                    self.num_bin_all,
                    self.num_modes,
                    self.length,
                ),
                1,
                3,
            )

            amp = temp[0]
            phase = temp[1]

            transfer_L1 = temp[3] + 1j * temp[4]
            transfer_L2 = temp[5] + 1j * temp[6]
            transfer_L3 = temp[7] + 1j * temp[8]

            # TODO: check this combination
            # TODO: produce combination as same in CUDA
            amp_phase = amp * self.xp.exp(1j * phase)

            # if t_obs_end <= 0.0:
            #    test = np.full_like(amp_phase.get(), True)

            # else:
            #    test = np.full_like(amp_phase.get(), False)

            # temp2 = ((tf <= t_end[0]) + (test)).astype(bool)
            # inds = (tf >= t_start[0]) & temp2 & (amp.get() > 1e-40)

            out = self.xp.asarray(
                [
                    amp_phase * transfer_L1,
                    amp_phase * transfer_L2,
                    amp_phase * transfer_L3,
                ]
            )

            breakpoint()

            # out[:, ~inds] = 0.0
            if squeeze:
                out = out.squeeze()

            return out

        else:

            # setup interpolant
            spline = CubicSplineInterpolant(
                self.amp_phase_gen.freqs,
                out_buffer,
                self.length,
                self.num_interp_params,
                self.num_modes,
                self.num_bin_all,
                use_gpu=self.use_gpu,
            )

            # TODO: try single block reduction for likelihood (will probably be worse for smaller batch, but maybe better for larger batch)?

            template_channels = self.interp_response(
                freqs,
                spline.container,
                t_start,
                t_end,
                self.length,
                self.data_length,
                self.num_modes,
                t_obs_start,
                t_obs_end,
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
                        data_out[:, start_i : start_i + length_i] = temp
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
