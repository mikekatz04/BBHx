"""
Wrapper code for gpuPhenomHM. Helps to calculate likelihoods
for samplers. Author: Michael Katz

Calculates phenomHM waveforms, puts them through the LISA response
and calculates likelihood.
"""

import numpy as np
from scipy import constants as ct

from katzsamplertools.utils.convert import Converter
from katzsamplertools.utils.constants import *
from katzsamplertools.utils.generatenoise import (
    generate_noise_frequencies,
    generate_noise_single_channel,
)

import tdi

try:
    import cupy as xp
except ImportError:
    import numpy as xp

    print("No cupy")

try:
    from gpuPhenomHM import PhenomHM
    from gpuPhenomHM import getDeviceCount

    num_gpus = getDeviceCount()
    if num_gpus > 0:
        import_cpu = False
    else:
        import_cpu = True

except ImportError:
    import_cpu = True

if import_cpu:
    from cpuPhenomHM import PhenomHM

import time


class pyPhenomHM(Converter):
    def __init__(
        self,
        injection,
        max_length_init,
        nwalkers,
        ndevices,
        l_vals,
        m_vals,
        data_freqs,
        data_stream,
        t0,
        key_order,
        t_obs_start,
        t_obs_end=0.0,
        **kwargs
    ):
        """
        data_stream (dict): keys X, Y, Z or A, E, T
        """
        prop_defaults = {
            "fRef": 0.0,
            "TDItag": "AET",  # AET or XYZ
            "max_dimensionless_freq": 0.5,
            "min_dimensionless_freq": 1e-4,
            "data_stream_whitened": True,
            "data_params": {},
            "log_scaled_likelihood": True,
            "eps": 1e-6,
            "num_params": 11,
            "num_data_points": int(2 ** 19),
            "df": None,
            "tLtoSSB": True,
            "noise_kwargs": {"model": "SciRDv1", "includewd": 1},
            "add_noise": None,  # if added should be dict with fs
        }

        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
            # TODO: check this
            kwargs[prop] = kwargs.get(prop, default)

        self.ndim = len(injection)
        self.nwalkers, self.ndevices = nwalkers, ndevices
        self.t0 = np.full(nwalkers * ndevices, t0)
        self.t_obs_start = t_obs_start
        self.t_obs_end = t_obs_end
        self.fRef = np.full(nwalkers * ndevices, self.fRef)
        self.max_length_init = max_length_init
        self.l_vals, self.m_vals = l_vals, m_vals
        self.data_freqs, self.data_stream = data_freqs, data_stream
        self.injection = injection

        if self.tLtoSSB is True:
            transform_frame = "tLtoSSB"
        else:
            transform_frame = None

        self.converter = Converter(
            "mbh", key_order, t0=t0, transform_frame=transform_frame
        )

        self.injection_array = np.array([injection[key] for key in key_order])

        self.injection_transformed = self.converter.recycle(self.injection_array.copy())
        self.injection_transformed = self.converter.convert(self.injection_transformed)

        self.determine_freqs_noise(**kwargs)

        if import_cpu is False:
            self.device_data_freqs = xp.asarray(self.data_freqs)

        if self.TDItag not in ["AET", "XYZ"]:
            raise ValueError("TDItag must be AET or XYZ.")

        else:
            if self.TDItag == "AET":
                self.TDItag_in = 2
            else:
                self.TDItag_in = 1

        self.generator = PhenomHM(
            max_length_init,
            l_vals,
            m_vals,
            len(self.data_freqs),
            self.TDItag_in,
            t_obs_start,
            t_obs_end,
            nwalkers,
            ndevices,
        )

        if self.data_stream is {} or self.data_stream is None:
            if self.injection == {}:
                raise ValueError(
                    "If data_stream is empty dict or None,"
                    + "user must supply data_params kwarg as "
                    + "dict with params for data stream."
                )

            self.inject_signal()

            self.data_stream_whitened = False

        self.create_input_data(**kwargs)

    def create_input_data(self, **kwargs):
        for i, channel in enumerate(self.TDItag):
            if channel not in self.data_stream:
                raise KeyError("{} not in TDItag {}.".format(channel, self.TDItag))

            setattr(self, "data_channel{}".format(i + 1), self.data_stream[channel])

        additional_factor = np.ones_like(self.data_freqs)
        if self.log_scaled_likelihood:
            additional_factor[1:] = np.sqrt(np.diff(self.data_freqs))
            additional_factor[0] = additional_factor[1]
        else:
            df = self.data_freqs[1] - self.data_freqs[0]
            additional_factor = np.sqrt(df)

        if self.TDItag == "AET":
            self.TDItag_in = 2

            self.channel1_ASDinv = (
                1.0
                / np.sqrt(tdi.noisepsd_AE(self.data_freqs, **self.noise_kwargs))
                * additional_factor
            )
            self.channel2_ASDinv = (
                1.0
                / np.sqrt(tdi.noisepsd_AE(self.data_freqs, **self.noise_kwargs))
                * additional_factor
            )
            self.channel3_ASDinv = (
                1.0
                / np.sqrt(
                    tdi.noisepsd_T(
                        self.data_freqs, model=kwargs["noise_kwargs"]["model"]
                    )
                )
                * additional_factor
            )

        elif self.TDItag == "XYZ":
            self.TDItag_in = 1
            for i in range(1, 4):
                temp = (
                    np.sqrt(tdi.noisepsd_XYZ(self.data_freqs, **self.noise_kwargs))
                    * additional_factor
                )
                setattr(self, "channel{}_ASDinv".format(i), temp)

        if self.data_stream_whitened is False:
            for i in range(1, 4):
                temp = getattr(self, "data_channel{}".format(i)) * getattr(
                    self, "channel{}_ASDinv".format(i)
                )
                setattr(self, "data_channel{}".format(i), temp)

        self.d_d = 4 * np.sum(
            [
                np.abs(self.data_channel1) ** 2,
                np.abs(self.data_channel2) ** 2,
                np.abs(self.data_channel3) ** 2,
            ]
        )

        self.generator.input_data(
            self.data_freqs,
            self.data_channel1,
            self.data_channel2,
            self.data_channel3,
            self.channel1_ASDinv,
            self.channel2_ASDinv,
            self.channel3_ASDinv,
        )

    def inject_signal(self):
        data_channel = np.ones_like(self.data_freqs, dtype=np.complex128)
        channel_ASDinv = np.ones_like(self.data_freqs)

        self.generator.input_data(
            self.data_freqs,
            data_channel,
            data_channel,
            data_channel,
            channel_ASDinv,
            channel_ASDinv,
            channel_ASDinv,
        )

        tiled_injection = np.tile(
            self.injection_array, (self.nwalkers * self.ndevices, 1)
        )

        data_stream_out = self.getNLL(tiled_injection.T, return_TDI=True)

        self.data_stream = {
            key: val[0] + an
            for key, val, an in zip(self.TDItag, data_stream_out, self.added_noise)
        }

    def determine_freqs_noise(self, **kwargs):
        self.added_noise = [0.0 for _ in range(3)]
        if self.data_freqs is None:
            if self.add_noise is not None:
                fs = kwargs["add_noise"]["fs"]
                Tobs = (self.t_obs_start - self.t_obs_end) * YRSID_SI
                noise_freqs = generate_noise_frequencies(Tobs, fs)

                self.data_freqs = data_freqs = noise_freqs[
                    noise_freqs >= kwargs["add_noise"]["min_freq"]
                ]

                df = data_freqs[1] - data_freqs[0]

                self.added_noise[0] = generate_noise_single_channel(
                    tdi.noisepsd_AE, [], self.noise_kwargs, df, data_freqs
                )

                self.added_noise[1] = generate_noise_single_channel(
                    tdi.noisepsd_AE, [], self.noise_kwargs, df, data_freqs
                )

                self.added_noise[2] = generate_noise_single_channel(
                    tdi.noisepsd_T,
                    [],
                    dict(model=kwargs["noise_kwargs"]["model"]),
                    df,
                    data_freqs,
                )

            else:
                # assumes m1 and m2 are first two entries in converted array
                m1 = self.injection_transformed[0]
                m2 = self.injection_transformed[1]
                Msec = (m1 + m2) * MTSUN
                upper_freq = self.max_dimensionless_freq / Msec
                lower_freq = self.min_dimensionless_freq / Msec
                self.data_freqs = data_freqs = np.logspace(
                    np.log10(lower_freq), np.log10(upper_freq), self.num_data_points
                )

    def NLL(
        self,
        m1,
        m2,
        a1,
        a2,
        distance,
        phiRef,
        inc,
        lam,
        beta,
        psi,
        tRef_wave_frame,
        tRef_sampling_frame,
        freqs=None,
        return_amp_phase=False,
        return_TDI=False,
        return_snr=False,
        return_response=False,
        return_phase_spline=False,
    ):

        Msec = (m1 + m2) * MTSUN

        if freqs is None:
            upper_freq = self.max_dimensionless_freq / Msec
            lower_freq = self.min_dimensionless_freq / Msec
            freqs = np.asarray(
                [
                    np.logspace(np.log10(lf), np.log10(uf), self.max_length_init)
                    for lf, uf in zip(lower_freq, upper_freq)
                ]
            )

        first_freqs = freqs[:, 0]
        last_freqs = freqs[:, -1]

        if import_cpu is False:
            first_inds = xp.searchsorted(
                self.device_data_freqs, xp.asarray(first_freqs), side="left"
            ).get()
            last_inds = xp.searchsorted(
                self.device_data_freqs, xp.asarray(last_freqs), side="right"
            ).get()
        else:
            first_inds = np.searchsorted(self.data_freqs, first_freqs, side="left")
            last_inds = np.searchsorted(self.data_freqs, last_freqs, side="right")

        merger_freq = np.zeros_like(m1)
        out = self.generator.WaveformThroughLikelihood(
            freqs.flatten(),
            m1,
            m2,  # solar masses
            a1,
            a2,
            distance,
            phiRef,
            self.fRef,
            inc,
            lam,
            beta,
            psi,
            self.t0,
            tRef_wave_frame,
            tRef_sampling_frame,
            merger_freq,
            first_inds.astype(np.int32),
            last_inds.astype(np.int32),
            return_amp_phase=return_amp_phase,
            return_TDI=return_TDI,
            return_response=return_response,
            return_phase_spline=return_phase_spline,
        )

        if return_amp_phase:
            return (freqs.reshape(len(lower_freq), -1),) + out

        if return_phase_spline:
            return (freqs.reshape(len(lower_freq), -1),) + out

        if return_response:
            return (freqs.reshape(len(lower_freq), -1),) + out

        if return_TDI:
            return out

        d_h, h_h = out

        if return_snr:
            return np.sqrt(d_h), np.sqrt(h_h)

        # 1/2<d-h|d-h> = 1/2(<d|d> + <h|h> - 2<d|h>)
        return 1.0 / 2.0 * (self.d_d + h_h - 2 * d_h)

    def getNLL(self, x, **kwargs):
        # changes parameters to in range in actual array (not copy)
        x = self.converter.recycle(x)

        # need tRef in the sampling frame
        tRef_sampling_frame = np.exp(x[10])

        # converts parameters in copy, not original array
        x_in = self.converter.convert(x.copy())

        m1, m2, a1, a2, distance, phiRef, inc, lam, beta, psi, tRef_wave_frame = x_in

        return self.NLL(
            m1,
            m2,
            a1,
            a2,
            distance,
            phiRef,
            inc,
            lam,
            beta,
            psi,
            tRef_wave_frame,
            tRef_sampling_frame,
            **kwargs
        )

    def get_Fisher(self, x):
        Mij = np.zeros((self.ndim, self.ndim), dtype=x.dtype)
        if self.nwalkers * self.ndevices < 2 * self.ndim:
            raise ValueError("num walkers must be greater than 2*ndim")
        x_in = np.tile(x, (self.nwalkers * self.ndevices, 1))

        for i in range(self.ndim):
            x_in[2 * i, i] += self.eps
            x_in[2 * i + 1, i] -= self.eps

        A, E, T = self.getNLL(x_in.T, return_TDI=True)

        for i in range(self.ndim):
            Ai_up, Ei_up, Ti_up = A[2 * i + 1], E[2 * i + 1], T[2 * i + 1]
            Ai_down, Ei_down, Ti_down = A[2 * i], E[2 * i], T[2 * i]

            hi_A = (Ai_up - Ai_down) / (2 * self.eps)
            hi_E = (Ei_up - Ei_down) / (2 * self.eps)
            hi_T = (Ti_up - Ti_down) / (2 * self.eps)

            for j in range(i, self.ndim):
                Aj_up, Ej_up, Tj_up = A[2 * j + 1], E[2 * j + 1], T[2 * j + 1]
                Aj_down, Ej_down, Tj_down = A[2 * j], E[2 * j], T[2 * j]

                hj_A = (Aj_up - Aj_down) / (2 * self.eps)
                hj_E = (Ej_up - Ej_down) / (2 * self.eps)
                hj_T = (Tj_up - Tj_down) / (2 * self.eps)

                inner_product = 4 * np.real(
                    (
                        np.dot(hi_A.conj(), hj_A)
                        + np.dot(hi_E.conj(), hj_E)
                        + np.dot(hi_T.conj(), hj_T)
                    )
                )

                Mij[i][j] = inner_product
                Mij[j][i] = inner_product

        return Mij
