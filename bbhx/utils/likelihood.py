import numpy as np

try:
    import cupy as xp
    from pyLikelihood import hdyn_wrap as hdyn_wrap_gpu
    from pyLikelihood import direct_like_wrap as direct_like_wrap_gpu
    from pyLikelihood import prep_hdyn as prep_hdyn_gpu

except (ImportError, ModuleNotFoundError) as e:
    print("No CuPy")
    import numpy as xp

from pyLikelihood_cpu import prep_hdyn as prep_hdyn_cpu
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
        use_gpu=False,
        **kwargs
    ):

        self.template_gen = template_gen
        self.f_dense = f_dense
        # TODO: make adjustable
        try:
            self.f_dense_cpu = self.f_dense.get()
        except AttributeError:
            self.f_dense_cpu = self.f_dense

        self.d = d
        self.length_f_rel = length_f_rel

        self.use_gpu = use_gpu
        if use_gpu:
            self.like_gen = hdyn_wrap_gpu
            self.prepare_hdyn = prep_hdyn_gpu
            self.xp = xp
        else:
            self.like_gen = hdyn_wrap_cpu
            self.prepare_hdyn = prep_hdyn_cpu
            self.xp = np

        if template_gen_args is not None:
            self.setup(template_gen_args, **kwargs)

    def setup(
        self,
        template_gen_args,
        template_gen_kwargs={},
        noise_kwargs_AE={},
        noise_kwargs_T={},
        batch_size=1,
    ):

        template_gen_kwargs["squeeze"] = False
        template_gen_kwargs["compress"] = True
        template_gen_kwargs["direct"] = True

        template_gen_kwargs_full = template_gen_kwargs.copy()
        template_gen_kwargs_full["direct"] = False
        template_gen_kwargs_full["length"] = 1024
        template_gen_kwargs_full["combine"] = False
        template_gen_kwargs_full["fill"] = True

        minF = self.f_dense.min() * 0.999999999999
        maxF = self.f_dense.max() * 1.000000000001

        S_n_all = self.xp.asarray(
            [
                get_sensitivity(
                    self.f_dense_cpu, sens_fn="noisepsd_AE", **noise_kwargs_AE
                ),
                get_sensitivity(
                    self.f_dense_cpu, sens_fn="noisepsd_AE", **noise_kwargs_AE
                ),
                get_sensitivity(
                    self.f_dense_cpu, sens_fn="noisepsd_T", **noise_kwargs_T
                ),
            ]
        )

        freqs_init = self.xp.logspace(
            self.xp.log10(minF), self.xp.log10(maxF), self.length_f_rel
        )

        if template_gen_args.ndim == 1:
            template_gen_args = np.atleast_2d(template_gen_args).T

        self.num_bin_all = len(template_gen_args[0])

        num_batches = self.num_bin_all // batch_size
        num_batches = (
            num_batches if (self.num_bin_all % batch_size) == 0 else num_batches + 1
        )

        A0_in = self.xp.zeros(
            (3, self.length_f_rel, self.num_bin_all), dtype=np.complex128
        )
        A1_in = self.xp.zeros_like(A0_in)
        B0_in = self.xp.zeros_like(A0_in)
        B1_in = self.xp.zeros_like(A0_in)

        self.base_d_d = self.xp.zeros(self.num_bin_all)
        self.base_d_h = self.xp.zeros(self.num_bin_all)
        self.base_h_h = self.xp.zeros(self.num_bin_all)

        all_freqs = self.xp.zeros((self.num_bin_all, self.length_f_rel))
        self.h0_short = self.xp.zeros(
            (3, self.length_f_rel, self.num_bin_all), dtype=self.xp.complex128
        )

        for batch_i in range(num_batches):
            end = (
                (batch_i + 1) * batch_size
                if (batch_i + 1) * batch_size <= self.num_bin_all
                else self.num_bin_all
            )
            slicin = np.arange(batch_i * batch_size, end)
            slicin_gc = self.xp.arange(batch_i * batch_size, end)

            h0 = self.template_gen(
                *template_gen_args[:, slicin],
                freqs=self.f_dense,
                **template_gen_kwargs_full
            )

            h0_temp = self.xp.atleast_3d(
                self.template_gen(
                    *template_gen_args[:, slicin],
                    freqs=freqs_init,
                    **template_gen_kwargs
                )
            ).transpose(
                (2, 0, 1)
            )  # [0]

            num_bin_here, channels, length = h0_temp.shape

            inds_not_zero = h0[:, 0] != 0.0
            inds_arange = self.xp.tile(
                self.xp.arange(len(self.f_dense)), (num_bin_here, 1)
            )

            indicator = inds_arange * (inds_not_zero) + int(1e8) * (~inds_not_zero)
            start_inds = self.xp.argmin(indicator, axis=1)

            indicator = inds_arange * (inds_not_zero) - int(1e8) * (~inds_not_zero)
            end_inds = self.xp.argmax(indicator, axis=1)

            start_freqs = self.f_dense[start_inds] * 1.00001
            end_freqs = self.f_dense[end_inds] * 0.99999

            # TODO: change limits to full signal
            freqs = self.xp.logspace(
                self.xp.log10(start_freqs), self.xp.log10(end_freqs), self.length_f_rel,
            ).T

            all_freqs[slicin_gc, :] = freqs

            self.h0_short[:, :, slicin_gc] = self.template_gen(
                *template_gen_args[:, slicin], freqs=freqs, **template_gen_kwargs
            )
            # inds = (self.f_dense >= freqs[0]) & (self.f_dense <= freqs[-1])

            # self.f_dense = self.f_dense[inds]
            # self.d = self.d[:, inds]
            # h0 = h0[:, inds]

            for (i, freqs_i, h0_i) in zip(slicin, freqs, h0):
                f_m = (freqs_i[1:] + freqs_i[:-1]) / 2

                inds = (self.f_dense >= freqs_i[0]) & (self.f_dense <= freqs_i[-1])
                temp_f_dense = self.f_dense[inds]
                S_n = S_n_all[:, inds].copy()
                data_length = len(temp_f_dense)
                bins = (
                    self.xp.searchsorted(freqs_i, temp_f_dense, "right").astype(
                        self.xp.int32
                    )
                    - 1
                )
                temp_d = self.d[:, inds].copy()
                temp_h0_i = h0_i[:, inds].copy()
                df = (self.f_dense[1] - self.f_dense[0]).item()

                A0_flat = self.xp.zeros(3 * self.length_f_rel, dtype=self.xp.complex128)
                A1_flat = self.xp.zeros_like(A0_flat)
                B0_flat = self.xp.zeros_like(A0_flat)
                B1_flat = self.xp.zeros_like(A0_flat)

                self.prepare_hdyn(
                    A0_flat,
                    A1_flat,
                    B0_flat,
                    B1_flat,
                    temp_d,
                    temp_h0_i,
                    S_n.flatten(),
                    df,
                    bins,
                    temp_f_dense,
                    f_m,
                    data_length,
                    3,
                    self.length_f_rel,
                )
                """
                # self.A0 = self.xp.zeros((3, self.length_f_rel - 1), dtype=np.complex128)
                # self.A1 = self.xp.zeros_like(self.A0)
                # self.B0 = self.xp.zeros_like(self.A0)
                # self.B1 = self.xp.zeros_like(self.A0)

                for ind in self.xp.unique(bins[:-1]):
                    ind = ind.item()
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
                """
                A0_in[:, :, i] = A0_flat.reshape(3, -1)
                A1_in[:, :, i] = A1_flat.reshape(3, -1)
                B0_in[:, :, i] = B0_flat.reshape(3, -1)
                B1_in[:, :, i] = B1_flat.reshape(3, -1)

                self.base_d_d[i] = self.xp.sum(
                    4 * (temp_d.conj() * temp_d) / S_n * df
                ).real
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

        self.freqs = all_freqs.squeeze()
        self.freqs_flat = all_freqs.T.flatten()
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
