import numpy as np

try:
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    print("No CuPy")
    import numpy as xp

from pyHdynBBH import *

from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
from bbhx.response.fastlisaresponse import LISATDIResponse
from bbhx.utils.interpolate import CubicSplineInterpolant
from lisatools.sensitivity import get_sensitivity

# TODO: deal with zeros in amplitude

from bbhx.utils.constants import *


use_gpu = True


class TemplateInterp:
    def __init__(self, max_init_len=-1, use_gpu=False):

        if use_gpu:
            self.xp = xp

        else:
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

        self.start_inds = start_inds = (inds_start_and_end[:, 0].get().copy()).astype(
            np.int32
        )

        self.ptrs = ptrs = np.asarray([ind_i.data.ptr for ind_i in inds])

        self.template_carrier = [
            self.xp.zeros(int(self.nChannels * temp_length), dtype=self.xp.complex128,)
            for temp_length in lengths
        ]

        template_carrier_ptrs = np.asarray(
            [temp_carrier.data.ptr for temp_carrier in self.template_carrier]
        )

        dlog10f = 1.0

        InterpTDI_wrap(
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
            self.xp = xp
        else:
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
        f_Ref,
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

        if direct and compress:

            templateChannels = self.xp.zeros(
                (self.num_bin_all * 3 * self.length), dtype=self.xp.complex128
            )

            direct_sum_wrap(
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
            amp_phase = amp * self.xp.exp(-1j * phase)

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

            if compress:
                out = out.sum(axis=2)

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
                use_gpu=use_gpu,
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


class Likelihood:
    def __init__(
        self, waveform_model, dataFreqs, dataChannels, noiseFactors, use_gpu=False
    ):

        self.use_gpu = use_gpu

        if use_gpu:
            self.xp = xp

        else:
            self.xp = np

        self.dataFreqs = dataFreqs
        self.dataChannels = dataChannels.flatten()
        self.noiseFactors = noiseFactors.flatten()
        self.waveform_gen = waveform_model
        self.data_stream_length = len(dataFreqs)

        # assumes dataChannels is already factored by noiseFactors
        self.d_d = (
            4 * self.xp.sum((self.dataChannels.conj() * self.dataChannels)).real
        ).item()

    def __call__(self, params, **waveform_kwargs):

        templateChannels, inds_start, ind_lengths = self.waveform_gen(
            *params, **waveform_kwargs
        )

        templateChannels = [tc.flatten() for tc in templateChannels]

        templateChannels_ptrs = np.asarray(
            [tc.data.ptr for tc in templateChannels], dtype=np.int64
        )

        d_h = np.zeros(self.waveform_gen.num_bin_all)
        h_h = np.zeros(self.waveform_gen.num_bin_all)

        # TODO: if filling multiple signals into stream, need to adjust this for that in terms of inds start / ind_lengths
        direct_like_wrap(
            d_h,
            h_h,
            self.dataChannels,
            self.noiseFactors,
            templateChannels_ptrs,
            inds_start,
            ind_lengths,
            self.data_stream_length,
            self.waveform_gen.num_bin_all,
        )

        return 1 / 2 * (self.d_d + h_h - 2 * d_h)


class RelativeBinning:
    def __init__(
        self,
        template_gen,
        f_dense,
        d,
        template_gen_args,
        length_f_rel,
        template_gen_kwargs={},
        use_gpu=False,
    ):

        self.template_gen = template_gen
        self.f_dense = f_dense
        self.d = d
        self.length_f_rel = length_f_rel

        self.use_gpu = use_gpu
        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

        self._init_rel_bin_info(template_gen_args, template_gen_kwargs)

    def _init_rel_bin_info(self, template_gen_args, template_gen_kwargs={}):

        template_gen_kwargs["squeeze"] = True
        template_gen_kwargs["compress"] = True
        template_gen_kwargs["direct"] = True

        minF = self.f_dense.min()
        maxF = self.f_dense.max()

        freqs = xp.logspace(xp.log10(minF), np.log10(maxF), self.length_f_rel)

        h0 = self.template_gen(
            *template_gen_args, freqs=self.f_dense, **template_gen_kwargs
        )

        h0_temp = self.template_gen(
            *template_gen_args, freqs=freqs, **template_gen_kwargs
        )[0]

        freqs_keep = freqs[~(self.xp.abs(h0_temp) == 0.0)]

        freqs = xp.logspace(
            xp.log10(freqs_keep[0]), np.log10(freqs_keep[-1]), self.length_f_rel
        )

        self.h0_short = self.template_gen(
            *template_gen_args, freqs=freqs, **template_gen_kwargs
        )[:, :, xp.newaxis]

        bins = xp.searchsorted(freqs, self.f_dense, "right") - 1

        f_m = (freqs[1:] + freqs[:-1]) / 2

        df = self.f_dense[1] - self.f_dense[0]

        # TODO: make adjustable
        S_n = xp.asarray(get_sensitivity(self.f_dense.get(), sens_fn="noisepsd_AE"))

        A0_flat = 4 * (h0.conj() * self.d) / S_n * df
        A1_flat = 4 * (h0.conj() * self.d) / S_n * df * (self.f_dense - f_m[bins])

        B0_flat = 4 * (h0.conj() * h0) / S_n * df
        B1_flat = 4 * (h0.conj() * h0) / S_n * df * (self.f_dense - f_m[bins])

        self.A0 = xp.zeros((3, self.length_f_rel - 1), dtype=np.complex128)
        self.A1 = xp.zeros_like(self.A0)
        self.B0 = xp.zeros_like(self.A0)
        self.B1 = xp.zeros_like(self.A0)

        A0_in = xp.zeros((3, self.length_f_rel), dtype=np.complex128)
        A1_in = xp.zeros_like(A0_in)
        B0_in = xp.zeros_like(A0_in)
        B1_in = xp.zeros_like(A0_in)

        for ind in xp.unique(bins[:-1]):
            inds_keep = bins == ind

            # TODO: check this
            inds_keep[-1] = False

            self.A0[:, ind] = xp.sum(A0_flat[:, inds_keep], axis=1)
            self.A1[:, ind] = xp.sum(A1_flat[:, inds_keep], axis=1)
            self.B0[:, ind] = xp.sum(B0_flat[:, inds_keep], axis=1)
            self.B1[:, ind] = xp.sum(B1_flat[:, inds_keep], axis=1)

            A0_in[:, ind + 1] = xp.sum(A0_flat[:, inds_keep], axis=1)
            A1_in[:, ind + 1] = xp.sum(A1_flat[:, inds_keep], axis=1)
            B0_in[:, ind + 1] = xp.sum(B0_flat[:, inds_keep], axis=1)
            B1_in[:, ind + 1] = xp.sum(B1_flat[:, inds_keep], axis=1)

        # PAD As with a zero in the front
        self.dataConstants = self.xp.concatenate(
            [A0_in.flatten(), A1_in.flatten(), B0_in.flatten(), B1_in.flatten()]
        )

        self.base_d_d = xp.sum(4 * (self.d.conj() * self.d) / S_n * df).real

        self.base_h_h = xp.sum(B0_flat).real

        self.base_d_h = xp.sum(A0_flat).real

        self.base_ll = 1 / 2 * (self.base_d_d + self.base_h_h - 2 * self.base_d_h)

        self.freqs = freqs
        self.f_m = f_m

    def __call__(self, params, **waveform_kwargs):

        waveform_kwargs["direct"] = True
        waveform_kwargs["compress"] = True
        waveform_kwargs["squeeze"] = False

        waveform_kwargs["freqs"] = self.freqs

        self.h_short = self.template_gen(*params, **waveform_kwargs)

        r = self.h_short / self.h0_short

        """
        r1 = (r[:, 1:] - r[:, :-1]) / (
            self.freqs[1:][xp.newaxis, :, xp.newaxis]
            - self.freqs[:-1][xp.newaxis, :, xp.newaxis]
        )

        r0 = r[:, :-1] - r1 * (
            self.freqs[:-1][xp.newaxis, :, xp.newaxis]
            - self.f_m[xp.newaxis, :, xp.newaxis]
        )

        self.Z_d_h = xp.sum(
            (
                xp.conj(r0) * self.A0[:, :, xp.newaxis]
                + xp.conj(r1) * self.A1[:, :, xp.newaxis]
            ),
            axis=(0, 1),
        )

        self.Z_h_h = xp.sum(
            (
                self.B0[:, :, xp.newaxis] * np.abs(r0) ** 2
                + 2 * self.B1[:, :, xp.newaxis] * xp.real(r0 * xp.conj(r1))
            ),
            axis=(0, 1),
        )

        test_like = 1 / 2 * (self.base_d_d + self.Z_h_h - 2 * self.Z_d_h).real
        """
        self.hdyn_d_h = self.xp.zeros(
            self.template_gen.num_bin_all, dtype=self.xp.complex128
        )
        self.hdyn_h_h = self.xp.zeros(
            self.template_gen.num_bin_all, dtype=self.xp.complex128
        )

        templates_in = r.transpose((1, 0, 2)).flatten()

        hdyn_wrap(
            self.hdyn_d_h,
            self.hdyn_h_h,
            templates_in,
            self.dataConstants,
            self.freqs,
            self.template_gen.num_bin_all,
            len(self.freqs),
            3,
        )

        like = 1 / 2.0 * (self.base_d_d + self.hdyn_h_h - 2 * self.hdyn_d_h).real

        return like


def test_phenomhm(
    m1,
    m2,
    chi1z,
    chi2z,
    distance,
    phiRef,
    f_ref,
    length,
    inc,
    lam,
    beta,
    psi,
    tRef_wave_frame,
    tRef_sampling_frame,
    tBase,
    t_obs_start,
    t_obs_end,
):

    m1_test = m1 * 1.00001
    m1_temp = m1 * 1.00003
    nChannels = 3
    data_length = 2 ** 15

    num_bin_all = len(m1)

    amp_phase_kwargs = dict()
    response_kwargs = dict(max_init_len=-1, TDItag="AET", order_fresnel_stencil=0)

    bbh = BBHWaveform(
        response_kwargs=response_kwargs,
        amp_phase_kwargs=amp_phase_kwargs,
        use_gpu=use_gpu,
    )

    df = 1.0 / YRSID_SI

    f_n = xp.arange(1e-6, 1e-1 + df, df)

    modes = None  # [(2, 5), (3, 3)]

    S_n = xp.asarray(
        [
            get_sensitivity(f_n.get(), sens_fn="noisepsd_AE"),
            get_sensitivity(f_n.get(), sens_fn="noisepsd_AE"),
            get_sensitivity(f_n.get(), sens_fn="noisepsd_T"),
        ]
    )

    data_length = len(f_n)

    import time

    data = bbh(
        m1[:1],
        m2[:1],
        chi1z[:1],
        chi2z[:1],
        distance[:1],
        phiRef[:1],
        f_ref[:1],
        inc[:1],
        lam[:1],
        beta[:1],
        psi[:1],
        tRef_wave_frame[:1],
        tRef_sampling_frame[:1],
        tBase=tBase,
        t_obs_start=t_obs_start,
        t_obs_end=t_obs_end,
        freqs=f_n,
        length=2048,
        modes=modes,
        direct=False,
        compress=True,
        fill=True,
    )

    num = 100

    noise_weight_times_df = xp.sqrt(1 / S_n * df)
    data_stream_length = len(f_n)

    data_scaled = data * noise_weight_times_df

    like = Likelihood(bbh, f_n, data_scaled, noise_weight_times_df, use_gpu=True)

    numBinAll = 32

    st = time.perf_counter()
    for _ in range(num):
        ll = like(
            [
                m1_test[:numBinAll],
                m2[:numBinAll],
                chi1z[:numBinAll],
                chi2z[:numBinAll],
                distance[:numBinAll],
                phiRef[:numBinAll],
                f_ref[:numBinAll],
                inc[:numBinAll],
                lam[:numBinAll],
                beta[:numBinAll],
                psi[:numBinAll],
                tRef_wave_frame[:numBinAll],
                tRef_sampling_frame[:numBinAll],
            ],
            tBase=tBase,
            t_obs_start=t_obs_start,
            t_obs_end=t_obs_end,
            freqs=f_n,
            length=4096,
            modes=modes,
            direct=False,
            compress=True,
            fill=False,
        )

    et = time.perf_counter()

    print((et - st) / num / numBinAll)

    ll_template = like(
        [
            m1_temp[:1],
            m2[:1],
            chi1z[:1],
            chi2z[:1],
            distance[:1],
            phiRef[:1],
            f_ref[:1],
            inc[:1],
            lam[:1],
            beta[:1],
            psi[:1],
            tRef_wave_frame[:1],
            tRef_sampling_frame[:1],
        ],
        tBase=tBase,
        t_obs_start=t_obs_start,
        t_obs_end=t_obs_end,
        freqs=f_n,
        length=4096,
        modes=modes,
        direct=False,
        compress=True,
        fill=False,
    )

    d = data.reshape(3, -1)

    template_gen_args = (
        m1_temp[:1],
        m2[:1],
        chi1z[:1],
        chi2z[:1],
        distance[:1],
        phiRef[:1],
        f_ref[:1],
        inc[:1],
        lam[:1],
        beta[:1],
        psi[:1],
        tRef_wave_frame[:1],
        tRef_sampling_frame[:1],
    )

    template_gen_kwargs = dict(
        tBase=tBase,
        t_obs_start=t_obs_start,
        t_obs_end=t_obs_end,
        length=None,
        modes=modes,
        direct=True,
        compress=True,
    )

    relbin = RelativeBinning(
        bbh,
        f_n,
        d,
        template_gen_args,
        length,
        template_gen_kwargs=template_gen_kwargs,
        use_gpu=use_gpu,
    )

    d_d = relbin.base_d_d

    import time

    st = time.perf_counter()
    num = 100

    for _ in range(num):
        ll_res = relbin(
            [
                m1_test,
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
            ],
            tBase=tBase,
            t_obs_start=t_obs_start,
            t_obs_end=t_obs_end,
            freqs=None,
            length=None,
            modes=modes,
            direct=True,
            compress=True,
            squeeze=False,
        )
    et = time.perf_counter()
    print((et - st) / num / num_bin_all, num, num_bin_all)

    breakpoint()


if __name__ == "__main__":

    num_bin_all = 10000
    length = 256

    m1 = np.full(num_bin_all, 1.000000e5)
    # m1[1:] += np.random.randn(num_bin_all - 1) * 100
    m2 = np.full_like(m1, 9e4)
    chi1z = np.full_like(m1, 0.2)
    chi2z = np.full_like(m1, 0.2)
    distance = np.full_like(m1, 10e9 * PC_SI)
    phiRef = np.full_like(m1, 0.0)
    f_ref = np.full_like(m1, 0.0)

    inc = np.full_like(m1, 0.2)
    lam = np.full_like(m1, 0.0)
    beta = np.full_like(m1, 0.3)
    psi = np.full_like(m1, 0.4)
    tRef_wave_frame = np.full_like(m1, 100.0)
    tRef_sampling_frame = np.full_like(m1, 120.0)
    tBase = 1.0

    t_obs_start = 1.0
    t_obs_end = 0.0

    test_phenomhm(
        m1,
        m2,
        chi1z,
        chi2z,
        distance,
        phiRef,
        f_ref,
        length,
        inc,
        lam,
        beta,
        psi,
        tRef_wave_frame,
        tRef_sampling_frame,
        tBase,
        t_obs_start,
        t_obs_end,
    )
