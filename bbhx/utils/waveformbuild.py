import numpy as np

try:
    import cupy as xp
    from pyWaveformBuild import direct_sum_wrap as direct_sum_wrap_gpu
    from pyWaveformBuild import InterpTDI_wrap as InterpTDI_wrap_gpu

except (ImportError, ModuleNotFoundError) as e:
    print("No CuPy")
    import numpy as xp

from pyWaveformBuild_cpu import direct_sum_wrap as direct_sum_wrap_cpu
from pyWaveformBuild_cpu import InterpTDI_wrap as InterpTDI_wrap_cpu

from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
from bbhx.response.fastlisaresponse import LISATDIResponse
from bbhx.utils.interpolate import CubicSplineInterpolant
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
        t_mrg,
        t_start,
        t_end,
        length,
        data_length,
        num_modes,
        t_obs_start,
        t_obs_end,
        nChannels,
    ):

        numBinAll = len(t_mrg)
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
            t_mrg,
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


class BBHWaveform:
    def __init__(
        self, amp_phase_kwargs={}, response_kwargs={}, interp_kwargs={}, use_gpu=False
    ):

        self.response_gen = LISATDIResponse(**response_kwargs, use_gpu=use_gpu)

        self.amp_phase_gen = PhenomHMAmpPhase(**amp_phase_kwargs, use_gpu=use_gpu)

        self.interp_tdi = TemplateInterp(**interp_kwargs, use_gpu=use_gpu)

        self.use_gpu = use_gpu
        if use_gpu:
            self.waveform_gen = direct_sum_wrap_gpu
            self.xp = xp
        else:
            self.waveform_gen = direct_sum_wrap_cpu
            self.xp = np

        self.num_interp_params = 9

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
        tRef_wave_frame,
        tRef_sampling_frame,
        tBase=0.0,
        t_obs_start=1.0,
        t_obs_end=0.0,
        freqs=None,
        length=None,
        modes=None,
        direct=True,
        compress=True,
        squeeze=True,
        shift_t_limits=False,
        fill=False,
    ):

        # TODO: if t_obs_end = t_mrg

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
        tRef_wave_frame = np.atleast_1d(tRef_wave_frame)
        tRef_sampling_frame = np.atleast_1d(tRef_sampling_frame)

        t_mrg = tRef_sampling_frame + tBase * YRSID_SI

        if shift_t_limits is False:
            t_start = tRef_sampling_frame + tBase * YRSID_SI - t_obs_start * YRSID_SI
            t_end = (
                tRef_sampling_frame + tBase * YRSID_SI - t_obs_end * YRSID_SI
                if t_obs_end > 0.0
                else np.zeros_like(t_start)
            )

        else:
            t_start = np.atleast_1d(t_obs_start)
            t_end = np.atleast_1d(t_obs_end)

        self.num_bin_all = len(m1)

        if freqs is None and length is None:
            raise ValueError("Must input freqs or length.")

        elif freqs is not None and direct is True:
            length = len(freqs)

        elif direct is False:
            self.data_length = len(freqs)

        self.length = length

        if modes is None:
            self.num_modes = len(self.amp_phase_gen.allowable_modes)
        else:
            self.num_modes = len(modes)

        self.num_bin_all = len(m1)

        out_buffer = self.xp.zeros(
            (self.num_interp_params * self.length * self.num_modes * self.num_bin_all)
        )

        freqs_temp = freqs if direct else None

        self.amp_phase_gen(
            m1,
            m2,
            chi1z,
            chi2z,
            distance,
            phiRef,
            f_ref,
            length,
            freqs=freqs_temp,
            out_buffer=out_buffer,
            modes=modes,
        )

        breakpoint()
        self.response_gen(
            self.amp_phase_gen.freqs,
            inc,
            lam,
            beta,
            psi,
            tRef_wave_frame,
            tRef_sampling_frame,
            phiRef,
            f_ref,
            tBase,
            length,
            includes_amps=True,
            out_buffer=out_buffer,
            modes=modes,
        )
        breakpoint()

        if direct and compress:

            templateChannels = self.xp.zeros(
                (self.num_bin_all * 3 * self.length), dtype=self.xp.complex128
            )

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
            # tf = temp[2].get()
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

            # out[:, ~inds] = 0.0
            if squeeze:
                out = out.squeeze()

            return out

        else:

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
            template_channels = self.interp_tdi(
                freqs,
                spline.container,
                t_mrg,
                t_start,
                t_end,
                self.length,
                self.data_length,
                self.num_modes,
                t_obs_start,
                t_obs_end,
                3,
            )

            if fill:
                data_out = self.xp.zeros((3, len(freqs)), dtype=self.xp.complex128)
                for temp, start_i, length_i in zip(
                    template_channels,
                    self.interp_tdi.start_inds,
                    self.interp_tdi.lengths,
                ):
                    data_out[:, start_i : start_i + length_i] = temp

                return data_out
            else:

                return (
                    template_channels,
                    self.interp_tdi.start_inds,
                    self.interp_tdi.lengths,
                )
