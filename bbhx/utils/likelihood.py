import numpy as np

try:
    import cupy as xp
    from pyLikelihood import hdyn_wrap as hdyn_wrap_gpu
    from pyLikelihood import direct_like_wrap as direct_like_wrap_gpu

except (ImportError, ModuleNotFoundError) as e:
    print("No CuPy")
    import numpy as xp

from pyLikelihood_cpu import hdyn_wrap as hdyn_wrap_cpu
from pyLikelihood_cpu import direct_like_wrap as direct_like_wrap_cpu

from bbhx.utils.constants import *

from lisatools.sensitivity import get_sensitivity


class Likelihood:
    def __init__(
        self, waveform_model, dataFreqs, dataChannels, noiseFactors, use_gpu=False
    ):

        self.use_gpu = use_gpu

        if use_gpu:
            self.like_gen = direct_like_wrap_gpu
            self.xp = xp

        else:
            self.like_gen = direct_like_wrap_cpu
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

    def get_ll(self, params, **waveform_kwargs):

        templateChannels, inds_start, ind_lengths = self.waveform_gen(
            *params, **waveform_kwargs
        )

        templateChannels = [tc.flatten() for tc in templateChannels]

        try:
            templateChannels_ptrs = np.asarray(
                [tc.data.ptr for tc in templateChannels], dtype=np.int64
            )
        except AttributeError:
            templateChannels_ptrs = np.asarray(
                [tc.__array_interface__["data"][0] for tc in templateChannels],
                dtype=np.int64,
            )

        d_h = np.zeros(self.waveform_gen.num_bin_all)
        h_h = np.zeros(self.waveform_gen.num_bin_all)

        # TODO: if filling multiple signals into stream, need to adjust this for that in terms of inds start / ind_lengths
        self.like_gen(
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

        out = 1 / 2 * (self.d_d + h_h - 2 * d_h)
        try:
            return out.get()

        except AttributeError:
            return out


class RelativeBinning:
    def __init__(
        self,
        template_gen,
        f_dense,
        d,
        template_gen_args,
        length_f_rel,
        template_gen_kwargs={},
        noise_kwargs_AE={},
        noise_kwargs_T={},
        use_gpu=False,
    ):

        self.template_gen = template_gen
        self.f_dense = f_dense
        self.d = d
        self.length_f_rel = length_f_rel

        self.use_gpu = use_gpu
        if use_gpu:
            self.like_gen = hdyn_wrap_gpu
            self.xp = xp
        else:
            self.like_gen = hdyn_wrap_cpu
            self.xp = np

        self._init_rel_bin_info(
            template_gen_args,
            template_gen_kwargs=template_gen_kwargs,
            noise_kwargs_AE=noise_kwargs_AE,
            noise_kwargs_T=noise_kwargs_T,
        )

    def _init_rel_bin_info(
        self,
        template_gen_args,
        template_gen_kwargs={},
        noise_kwargs_AE={},
        noise_kwargs_T={},
    ):

        template_gen_kwargs["squeeze"] = True
        template_gen_kwargs["compress"] = True
        template_gen_kwargs["direct"] = True

        minF = self.f_dense.min() * 0.999999999999
        maxF = self.f_dense.max() * 1.000000000001

        freqs = xp.logspace(xp.log10(minF), np.log10(maxF), self.length_f_rel)

        h0 = self.template_gen(
            *template_gen_args, freqs=self.f_dense, **template_gen_kwargs
        )

        h0_temp = self.template_gen(
            *template_gen_args, freqs=freqs, **template_gen_kwargs
        )[0]

        freqs_keep = freqs[~(self.xp.abs(h0_temp) == 0.0)]

        freqs = self.xp.logspace(
            self.xp.log10(freqs_keep[0]),
            self.xp.log10(freqs_keep[-1]),
            self.length_f_rel,
        )

        self.h0_short = self.template_gen(
            *template_gen_args, freqs=freqs, **template_gen_kwargs
        )[:, :, xp.newaxis]

        inds = (self.f_dense >= freqs[0]) & (self.f_dense <= freqs[-1])

        self.f_dense = self.f_dense[inds]
        self.d = self.d[:, inds]
        h0 = h0[:, inds]

        bins = self.xp.searchsorted(freqs, self.f_dense, "right") - 1

        f_m = (freqs[1:] + freqs[:-1]) / 2

        df = self.f_dense[1] - self.f_dense[0]

        # TODO: make adjustable
        try:
            f_n_host = self.f_dense.get()
        except AttributeError:
            f_n_host = self.f_dense

        S_n = xp.asarray(
            [
                get_sensitivity(f_n_host, sens_fn="noisepsd_AE", **noise_kwargs_AE),
                get_sensitivity(f_n_host, sens_fn="noisepsd_AE", **noise_kwargs_AE),
                get_sensitivity(f_n_host, sens_fn="noisepsd_T", **noise_kwargs_T),
            ]
        )

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

    def get_ll(self, params, **waveform_kwargs):

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

        self.like_gen(
            self.hdyn_d_h,
            self.hdyn_h_h,
            templates_in,
            self.dataConstants,
            self.freqs,
            self.template_gen.num_bin_all,
            len(self.freqs),
            3,
        )

        out = 1 / 2.0 * (self.base_d_d + self.hdyn_h_h - 2 * self.hdyn_d_h).real

        try:
            return out.get()

        except AttributeError:
            return out
