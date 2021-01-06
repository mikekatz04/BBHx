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
from bbhx.utils.waveformbuild import TemplateInterp, BBHWaveform

# TODO: deal with zeros in amplitude

from bbhx.utils.constants import *


use_gpu = True


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
