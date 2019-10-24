"""
Wrapper code for gpuPhenomHM. Helps to calculate likelihoods
for samplers. Author: Michael Katz

Calculates phenomHM waveforms, puts them through the LISA response
and calculates likelihood.
"""

import numpy as np
from scipy import constants as ct
from phenomhm.utils.convert import Converter, Recycler

import tdi

try:
    from gpuPhenomHM import PhenomHM
except ImportError:
    from cpuPhenomHM import PhenomHM

import time

MTSUN = 1.989e30 * ct.G / ct.c ** 3


class pyPhenomHM(Converter):
    def __init__(
        self,
        max_length_init,
        nwalkers,
        ndevices,
        l_vals,
        m_vals,
        data_freqs,
        data_stream,
        t0,
        t_obs_dur,
        key_order,
        **kwargs
    ):
        """
        data_stream (dict): keys X, Y, Z or A, E, T
        """
        prop_defaults = {
            "TDItag": "AET",  # AET or XYZ
            "max_dimensionless_freq": 0.5,
            "min_dimensionless_freq": 1e-4,
            "data_stream_whitened": True,
            "data_params": {},
            "log_scaled_likelihood": True,
            "eps": 1e-6,
            "test_inds": None,
            "num_params": 11,
            "num_data_points": int(2 ** 19),
            "df": None,
            "tLtoSSB": True,
            "noise_kwargs": {"model": "SciRDv1"},
        }

        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
            # TODO: check this
            kwargs[prop] = kwargs.get(prop, default)

        self.nwalkers, self.ndevices = nwalkers, ndevices
        self.converter = Converter(key_order, tLtoSSB=self.tLtoSSB)
        self.recycler = Recycler(key_order, tLtoSSB=self.tLtoSSB)

        self.generator = None
        self.t0 = np.full(nwalkers * ndevices, t0)
        self.t_obs_dur = t_obs_dur
        self.max_length_init = max_length_init
        self.l_vals, self.m_vals = l_vals, m_vals
        self.data_freqs, self.data_stream = data_freqs, data_stream

        if self.test_inds is None:
            self.test_inds = np.arange(self.num_params)

        if self.TDItag not in ["AET", "XYZ"]:
            raise ValueError("TDItag must be AET or XYZ.")

        if self.data_stream is {} or self.data_stream is None:
            if self.data_params == {}:
                raise ValueError(
                    "If data_stream is empty dict or None,"
                    + "user must supply data_params kwarg as "
                    + "dict with params for data stream."
                )
            kwargs["data_params"]["t0"] = t0
            kwargs["data_params"]["t_obs_dur"] = t_obs_dur

            self.data_freqs, self.data_stream, self.generator = create_data_set(
                nwalkers,
                ndevices,
                l_vals,
                m_vals,
                t0,
                self.data_params,
                self.converter,
                self.recycler,
                num_generate_points=max_length_init,
                data_freqs=data_freqs,
                **kwargs
            )
            self.data_stream_whitened = False

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

            """AE_noise = np.genfromtxt('SnAE2017.dat').T
            T_noise = np.genfromtxt('SnAE2017.dat').T

            from scipy.interpolate import CubicSpline

            AE_noise = CubicSpline(AE_noise[0], AE_noise[1])
            T_noise = CubicSpline(T_noise[0], T_noise[1])

            self.channel1_ASDinv = 1./np.sqrt(AE_noise(self.data_freqs))*additional_factor
            self.channel2_ASDinv = 1./np.sqrt(AE_noise(self.data_freqs))*additional_factor
            self.channel3_ASDinv = 1./np.sqrt(T_noise(self.data_freqs))*additional_factor"""

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
                / np.sqrt(tdi.noisepsd_T(self.data_freqs, **self.noise_kwargs))
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

        if self.generator is None:
            self.generator = PhenomHM(
                self.max_length_init,
                len(self.data_freqs),
                self.l_vals,
                self.m_vals,
                self.TDItag_in,
                self.t_obs_dur,
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

        self.fRef = np.zeros(nwalkers * ndevices)

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
    ):

        Msec = (m1 + m2) * MTSUN
        # merger frequency for 22 mode amplitude in phenomD
        merger_freq = 0.018 / Msec

        if freqs is None:
            upper_freq = self.max_dimensionless_freq / Msec
            lower_freq = self.min_dimensionless_freq / Msec
            freqs = np.asarray(
                [
                    np.logspace(np.log10(lf), np.log10(uf), self.max_length_init)
                    for lf, uf in zip(lower_freq, upper_freq)
                ]
            ).flatten()

        out = self.generator.WaveformThroughLikelihood(
            freqs,
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
            return_amp_phase=return_amp_phase,
            return_TDI=return_TDI,
        )

        if return_amp_phase or return_TDI:
            return out

        d_h, h_h = out

        if return_snr:
            return np.sqrt(d_h), np.sqrt(h_h)

        return self.d_d + h_h - 2 * d_h

    def getNLL(self, x, **kwargs):
        # changes parameters to in range in actual array (not copy)
        x = self.recycler.recycle(x)

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
        Mij = np.zeros((len(self.test_inds), len(self.test_inds)), dtype=x.dtype)
        if self.nwalkers * self.ndevices < 2 * len(self.test_inds):
            raise ValueError("num walkers must be greater than 2*ndim")
        x_in = np.tile(x, (self.nwalkers * self.ndevices, 1))

        for i in range(len(self.test_inds)):
            x_in[2 * i, i] += self.eps
            x_in[2 * i + 1, i] -= self.eps

        A, E, T = self.getNLL(x_in.T, return_TDI=True)

        for i in range(len(self.test_inds)):
            Ai_up, Ei_up, Ti_up = A[2 * i + 1], E[2 * i + 1], T[2 * i + 1]
            Ai_down, Ei_down, Ti_down = A[2 * i], E[2 * i], T[2 * i]

            hi_A = (Ai_up - Ai_down) / (2 * self.eps)
            hi_E = (Ei_up - Ei_down) / (2 * self.eps)
            hi_T = (Ti_up - Ti_down) / (2 * self.eps)

            for j in range(i, len(self.test_inds)):
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

    def gradNLL(self, x):
        grad = np.zeros_like(x)
        for i in range(len(self.test_inds)):
            x_in = x.copy()

            x_in[i] = x[i] + self.eps
            like_up = self.getNLL(x_in)

            x_in[i] = x[i] - self.eps
            like_down = self.getNLL(x_in)

            grad[i] = (like_up - like_down) / (2 * self.eps)
        return grad

    def deriv_2_of_NLL(self, x):
        Mij = np.zeros_like(x)
        f_x = self.getNLL(x)
        for i in range(len(self.test_inds)):
            x_in = x.copy()

            x_in[i] = x[i] + 2 * self.eps
            like_up = self.getNLL(x_in)

            x_in[i] = x[i] - 2 * self.eps
            like_down = self.getNLL(x_in)

            Mij[i] = (like_up - 2 * f_x + like_down) / (4 * self.eps ** 2)

        return Mij


def create_data_set(
    nwalkers,
    ndevices,
    l_vals,
    m_vals,
    t0,
    waveform_params,
    converter,
    recycler,
    data_freqs=None,
    TDItag="AET",
    num_data_points=int(2 ** 19),
    num_generate_points=int(2 ** 18),
    df=None,
    min_dimensionless_freq=1e-4,
    max_dimensionless_freq=1.0,
    add_noise=None,
    **kwargs
):
    key_list = list(waveform_params.keys())

    vals = np.array([waveform_params[key] for key in key_list])

    tRef_sampling_frame = np.exp(vals[10])

    vals = recycler.recycle(vals)
    vals = converter.convert(vals)

    waveform_params = {key: vals[i] for i, key in enumerate(key_list)}

    waveform_params["tRef_sampling_frame"] = tRef_sampling_frame

    if "ln_m1" in waveform_params:
        waveform_params["m1"] = waveform_params["ln_m1"]
        waveform_params["m2"] = waveform_params["ln_m2"]
    if "ln_mT" in waveform_params:
        # has been converted
        waveform_params["m1"] = waveform_params["ln_mT"]
        waveform_params["m2"] = waveform_params["mr"]

    if "chi_s" in waveform_params:
        waveform_params["a1"] = waveform_params["chi_s"]
        waveform_params["a2"] = waveform_params["chi_a"]

    if "cos_inc" in waveform_params:
        waveform_params["inc"] = waveform_params["cos_inc"]

    if "sin_beta" in waveform_params:
        waveform_params["beta"] = waveform_params["sin_beta"]

    waveform_params["distance"] = waveform_params["ln_distance"]
    waveform_params["tRef_wave_frame"] = waveform_params["ln_tRef"]
    waveform_params["fRef"] = 0.0

    m1 = waveform_params["m1"]
    m2 = waveform_params["m2"]
    Msec = (m1 + m2) * MTSUN
    merger_freq = 0.018 / Msec

    if data_freqs is None:
        if add_noise is not None:
            fs = add_noise["fs"]
            t_obs_dur = waveform_params["t_obs_dur"]
            df = 1.0 / (t_obs_dur * ct.Julian_year)
            num_data_points = int(t_obs_dur * ct.Julian_year * fs)
            noise_freqs = np.fft.rfftfreq(num_data_points, 1 / fs)
            data_freqs = noise_freqs[noise_freqs >= add_noise["min_freq"]]

        else:
            m1 = waveform_params["m1"]
            m2 = waveform_params["m2"]
            Msec = (m1 + m2) * MTSUN
            upper_freq = max_dimensionless_freq / Msec
            lower_freq = min_dimensionless_freq / Msec
            merger_freq = 0.018 / Msec
            if df is None:
                data_freqs = np.logspace(
                    np.log10(lower_freq), np.log10(upper_freq), num_data_points
                )
            else:
                data_freqs = np.arange(fmin, fmax + df, df)

    if add_noise is not None:

        norm1 = 0.5 * (1.0 / df) ** 0.5
        re = np.random.normal(0, norm1, size=(3,) + data_freqs.shape)
        im = np.random.normal(0, norm1, size=(3,) + data_freqs.shape)
        htilde = re + 1j * im

        # the 0.125 is 1/8 to match LDC data #FIXME
        if TDItag == "AET":
            # assumes gaussian noise
            noise_channel1 = (
                np.sqrt(tdi.noisepsd_AE(data_freqs, **kwargs["noise_kwargs"]))
                * htilde[0]
                * 0.125
            )
            noise_channel2 = (
                np.sqrt(tdi.noisepsd_AE(data_freqs, **kwargs["noise_kwargs"]))
                * htilde[1]
                * 0.125
            )
            noise_channel3 = (
                np.sqrt(tdi.noisepsd_T(data_freqs, **kwargs["noise_kwargs"]))
                * htilde[2]
                * 0.125
            )

        else:
            # assumes gaussian noise
            noise_channel1 = (
                np.sqrt(tdi.noisepsd_XYZ(data_freqs, **kwargs["noise_kwargs"]))
                * htilde[0]
                * 0.125
            )
            noise_channel2 = (
                np.sqrt(tdi.noisepsd_XYZ(data_freqs, **kwargs["noise_kwargs"]))
                * htilde[1]
                * 0.125
            )
            noise_channel3 = (
                np.sqrt(tdi.noisepsd_XYZ(data_freqs, **kwargs["noise_kwargs"]))
                * htilde[2]
                * 0.125
            )

    generate_freqs = np.logspace(
        np.log10(data_freqs.min()), np.log10(data_freqs.max()), num_generate_points
    )

    fake_data = np.zeros_like(data_freqs, dtype=np.complex128)
    fake_ASD = np.ones_like(data_freqs)

    if TDItag == "AET":
        TDItag_in = 2

    elif TDItag == "XYZ":
        TDItag_in = 1

    phenomHM = PhenomHM(
        len(generate_freqs),
        len(data_freqs),
        l_vals,
        m_vals,
        TDItag_in,
        waveform_params["t_obs_dur"],
        nwalkers,
        ndevices,
    )

    phenomHM.input_data(
        data_freqs, fake_data, fake_data, fake_data, fake_ASD, fake_ASD, fake_ASD
    )

    freqs = np.tile(generate_freqs, nwalkers * ndevices)
    m1 = np.full(nwalkers * ndevices, waveform_params["m1"])
    m2 = np.full(nwalkers * ndevices, waveform_params["m2"])
    a1 = np.full(nwalkers * ndevices, waveform_params["a1"])
    a2 = np.full(nwalkers * ndevices, waveform_params["a2"])
    distance = np.full(nwalkers * ndevices, waveform_params["distance"])
    phiRef = np.full(nwalkers * ndevices, waveform_params["phiRef"])
    fRef = np.full(nwalkers * ndevices, waveform_params["fRef"])
    inc = np.full(nwalkers * ndevices, waveform_params["inc"])
    lam = np.full(nwalkers * ndevices, waveform_params["lam"])
    beta = np.full(nwalkers * ndevices, waveform_params["beta"])
    psi = np.full(nwalkers * ndevices, waveform_params["psi"])
    t0 = np.full(nwalkers * ndevices, waveform_params["t0"])
    tRef_wave_frame = np.full(nwalkers * ndevices, waveform_params["tRef_wave_frame"])
    tRef_sampling_frame = np.full(
        nwalkers * ndevices, waveform_params["tRef_sampling_frame"]
    )
    merger_freq = np.full(nwalkers * ndevices, merger_freq)

    channel1, channel2, channel3 = phenomHM.WaveformThroughLikelihood(
        freqs,
        m1,
        m2,
        a1,
        a2,
        distance,
        phiRef,
        fRef,
        inc,
        lam,
        beta,
        psi,
        t0,
        tRef_wave_frame,
        tRef_sampling_frame,
        merger_freq,
        return_TDI=True,
    )

    channel1, channel2, channel3 = channel1[0], channel2[0], channel3[0]

    if add_noise is not None:
        channel1 = channel1 + noise_channel1
        channel2 = channel2 + noise_channel2
        channel3 = channel3 + noise_channel3

    data_stream = {TDItag[0]: channel1, TDItag[1]: channel2, TDItag[2]: channel3}
    return data_freqs, data_stream, phenomHM
