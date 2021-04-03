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

        self.d_h = np.zeros(self.waveform_gen.num_bin_all)
        self.h_h = np.zeros(self.waveform_gen.num_bin_all)

        # TODO: if filling multiple signals into stream, need to adjust this for that in terms of inds start / ind_lengths
        self.like_gen(
            self.d_h,
            self.h_h,
            self.dataChannels,
            self.noiseFactors,
            templateChannels_ptrs,
            inds_start,
            ind_lengths,
            self.data_stream_length,
            self.waveform_gen.num_bin_all,
        )

        out = 1 / 2 * (self.d_d + self.h_h - 2 * self.d_h)
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
        length_f_rel,
        template_gen_args=None,
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

        if template_gen_args is not None:
            self.setup(
                template_gen_args,
                template_gen_kwargs=template_gen_kwargs,
                noise_kwargs_AE=noise_kwargs_AE,
                noise_kwargs_T=noise_kwargs_T,
            )

    def setup(
        self,
        template_gen_args,
        template_gen_kwargs={},
        noise_kwargs_AE={},
        noise_kwargs_T={},
    ):

        template_gen_kwargs["squeeze"] = False
        template_gen_kwargs["compress"] = True
        template_gen_kwargs["direct"] = True

        minF = self.f_dense.min() * 0.999999999999
        maxF = self.f_dense.max() * 1.000000000001

        freqs = self.xp.logspace(
            self.xp.log10(minF), self.xp.log10(maxF), self.length_f_rel
        )

        h0 = self.xp.atleast_3d(
            self.template_gen(
                *template_gen_args, freqs=self.f_dense, **template_gen_kwargs
            )
        ).transpose((2, 0, 1))

        h0_temp = self.xp.atleast_3d(
            self.template_gen(*template_gen_args, freqs=freqs, **template_gen_kwargs)
        ).transpose(
            (2, 0, 1)
        )  # [0]

        self.num_bin_all, channels, length = h0_temp.shape

        inds_not_zero = h0_temp[:, 0] != 0.0
        inds_arange = self.xp.tile(
            self.xp.arange(self.length_f_rel), (self.num_bin_all, 1)
        )

        indicator = inds_arange * (inds_not_zero) + int(1e8) * (~inds_not_zero)
        start_inds = self.xp.argmin(indicator, axis=1)

        indicator = inds_arange * (inds_not_zero) - int(1e8) * (~inds_not_zero)
        end_inds = self.xp.argmax(indicator, axis=1)

        start_freqs = freqs[start_inds]
        end_freqs = freqs[end_inds]

        freqs = self.xp.logspace(
            self.xp.log10(start_freqs), self.xp.log10(end_freqs), self.length_f_rel,
        ).T

        self.h0_short = self.template_gen(
            *template_gen_args, freqs=freqs, **template_gen_kwargs
        )

        # inds = (self.f_dense >= freqs[0]) & (self.f_dense <= freqs[-1])

        # self.f_dense = self.f_dense[inds]
        # self.d = self.d[:, inds]
        # h0 = h0[:, inds]
        A0_in = self.xp.zeros(
            (3, self.length_f_rel, self.num_bin_all), dtype=np.complex128
        )
        A1_in = self.xp.zeros_like(A0_in)
        B0_in = self.xp.zeros_like(A0_in)
        B1_in = self.xp.zeros_like(A0_in)

        self.base_d_d = self.xp.zeros(self.num_bin_all)
        self.base_d_h = self.xp.zeros(self.num_bin_all)
        self.base_h_h = self.xp.zeros(self.num_bin_all)

        for i, (freqs_i, h0_i) in enumerate(zip(freqs, h0)):
            f_m = (freqs_i[1:] + freqs_i[:-1]) / 2

            inds = (self.f_dense >= freqs_i[0]) & (self.f_dense <= freqs_i[-1])
            temp_f_dense = self.f_dense[inds]
            bins = self.xp.searchsorted(freqs_i, temp_f_dense, "right") - 1
            temp_d = self.d[:, inds]
            temp_h0_i = h0_i[:, inds]
            df = self.f_dense[1] - self.f_dense[0]

            # TODO: make adjustable
            try:
                f_n_host = temp_f_dense.get()
            except AttributeError:
                f_n_host = temp_f_dense

            S_n = self.xp.asarray(
                [
                    get_sensitivity(f_n_host, sens_fn="noisepsd_AE", **noise_kwargs_AE),
                    get_sensitivity(f_n_host, sens_fn="noisepsd_AE", **noise_kwargs_AE),
                    get_sensitivity(f_n_host, sens_fn="noisepsd_T", **noise_kwargs_T),
                ]
            )
            A0_flat = 4 * (temp_h0_i.conj() * temp_d) / S_n * df
            A1_flat = (
                4 * (temp_h0_i.conj() * temp_d) / S_n * df * (temp_f_dense - f_m[bins])
            )

            B0_flat = 4 * (temp_h0_i.conj() * temp_h0_i) / S_n * df
            B1_flat = (
                4
                * (temp_h0_i.conj() * temp_h0_i)
                / S_n
                * df
                * (temp_f_dense - f_m[bins])
            )

            # self.A0 = self.xp.zeros((3, self.length_f_rel - 1), dtype=np.complex128)
            # self.A1 = self.xp.zeros_like(self.A0)
            # self.B0 = self.xp.zeros_like(self.A0)
            # self.B1 = self.xp.zeros_like(self.A0)

            for ind in self.xp.unique(bins[:-1]):
                inds_keep = bins == ind

                # TODO: check this
                inds_keep[-1] = False

                # self.A0[:, ind] = self.xp.sum(A0_flat[:, inds_keep], axis=1)
                # self.A1[:, ind] = self.xp.sum(A1_flat[:, inds_keep], axis=1)
                # self.B0[:, ind] = self.xp.sum(B0_flat[:, inds_keep], axis=1)
                # self.B1[:, ind] = self.xp.sum(B1_flat[:, inds_keep], axis=1)

                A0_in[:, ind + 1, i] = self.xp.sum(A0_flat[:, inds_keep], axis=1)
                A1_in[:, ind + 1, i] = self.xp.sum(A1_flat[:, inds_keep], axis=1)
                B0_in[:, ind + 1, i] = self.xp.sum(B0_flat[:, inds_keep], axis=1)
                B1_in[:, ind + 1, i] = self.xp.sum(B1_flat[:, inds_keep], axis=1)

            self.base_d_d[i] = self.xp.sum(4 * (temp_d.conj() * temp_d) / S_n * df).real
            self.base_h_h[i] = self.xp.sum(B0_flat).real
            self.base_d_h[i] = self.xp.sum(A0_flat).real

        # PAD As with a zero in the front
        self.dataConstants = self.xp.concatenate(
            [
                A0_in.transpose(1, 0, 2).flatten(),
                A1_in.transpose(1, 0, 2).flatten(),
                B0_in.transpose(1, 0, 2).flatten(),
                B1_in.transpose(1, 0, 2).flatten(),
            ]
        )

        self.base_ll = 1 / 2 * (self.base_d_d + self.base_h_h - 2 * self.base_d_h)

        self.freqs = freqs.squeeze()
        self.freqs_flat = freqs.T.flatten()
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
            self.freqs[1:][self.xp.newaxis, :, self.xp.newaxis]
            - self.freqs[:-1][self.xp.newaxis, :, self.xp.newaxis]
        )

        r0 = r[:, :-1] - r1 * (
            self.freqs[:-1][self.xp.newaxis, :, self.xp.newaxis]
            - self.f_m[self.xp.newaxis, :, self.xp.newaxis]
        )

        self.Z_d_h = self.xp.sum(
            (
                self.xp.conj(r0) * self.A0[:, :, self.xp.newaxis]
                + self.xp.conj(r1) * self.A1[:, :, self.xp.newaxis]
            ),
            axis=(0, 1),
        )

        self.Z_h_h = self.xp.sum(
            (
                self.B0[:, :, self.xp.newaxis] * np.abs(r0) ** 2
                + 2 * self.B1[:, :, self.xp.newaxis] * self.xp.real(r0 * self.xp.conj(r1))
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

        if self.num_bin_all > 1 and (self.template_gen.num_bin_all != self.num_bin_all):
            raise NotImplementedError

        templates_in = r.transpose((1, 0, 2)).flatten()

        full = False if self.num_bin_all == 1 else True
        self.like_gen(
            self.hdyn_d_h,
            self.hdyn_h_h,
            templates_in,
            self.dataConstants,
            self.freqs_flat,
            self.template_gen.num_bin_all,
            self.length_f_rel,
            3,
            full,
        )

        out = 1 / 2.0 * (self.base_d_d + self.hdyn_h_h - 2 * self.hdyn_d_h).real

        try:
            self.h_h = self.hdyn_h_h.get()
            self.d_h = self.hdyn_d_h.get()
        except AttributeError:
            self.h_h = self.hdyn_h_h
            self.d_h = self.hdyn_d_h

        try:
            return out.get()

        except AttributeError:
            return out
