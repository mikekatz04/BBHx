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

MTSUN = 1.988546954961461467461011951140572744e30 * ct.G / ct.c ** 3


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
            "test_inds": None,
            "num_params": 11,
            "num_data_points": int(2 ** 19),
            "df": None,
            "tLtoSSB": True,
            "noise_kwargs": {"model": "SciRDv1", "includewd": 1},
        }

        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
            # TODO: check this
            kwargs[prop] = kwargs.get(prop, default)

        self.nwalkers, self.ndevices = nwalkers, ndevices
        self.converter = Converter(key_order, t0, tLtoSSB=self.tLtoSSB)
        self.recycler = Recycler(key_order, tLtoSSB=self.tLtoSSB)

        self.generator = None
        self.t0 = np.full(nwalkers * ndevices, t0)
        self.t_obs_start = t_obs_start
        self.t_obs_end = t_obs_end
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
            kwargs["data_params"]["t_obs_start"] = t_obs_start
            kwargs["data_params"]["t_obs_end"] = t_obs_end

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

        if self.generator is None:
            self.generator = PhenomHM(
                self.max_length_init,
                self.l_vals,
                self.m_vals,
                len(self.data_freqs),
                self.TDItag_in,
                self.t_obs_start,
                self.t_obs_end,
                self.nwalkers,
                self.ndevices,
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

        if isinstance(self.fRef, str):
            if self.fRef == "merger freq":
                self.fRef_merger_freq = True
            else:
                raise ValueError("If fRef kwarg is a string, it must be merger freq.")

        else:
            self.fRef = np.full(nwalkers * ndevices, self.fRef)
            self.fRef_merger_freq = False

        if import_cpu is False:
            self.device_data_freqs = xp.asarray(self.data_freqs)

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
        # merger frequency for 22 mode amplitude in phenomD
        merger_freq = 0.018 / Msec

        if self.fRef_merger_freq:
            self.fRef = self.fRef_merger_freq

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
    fRef=0.0,
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

    m1 = waveform_params["m1"]
    m2 = waveform_params["m2"]
    Msec = (m1 + m2) * MTSUN
    merger_freq = 0.018 / Msec

    if isinstance(fRef, str):
        if fRef == "merger freq":
            waveform_params["fRef"] = merger_freq
        else:
            raise ValueError("If fRef kwarg is a string, it must be merger freq.")

    else:
        waveform_params["fRef"] = fRef

    if data_freqs is None:
        if add_noise is not None:

            sampling_frequency = add_noise["fs"]
            t_obs_start = waveform_params["t_obs_start"]
            t_obs_end = waveform_params["t_obs_end"]
            duration = (t_obs_start - t_obs_end) * ct.Julian_year
            df = 1.0 / duration
            number_of_samples = int(np.round(duration * sampling_frequency))
            number_of_frequencies = int(np.round(number_of_samples / 2) + 1)

            noise_freqs = np.linspace(
                start=0, stop=sampling_frequency / 2, num=number_of_frequencies
            )

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
                data_freqs = np.arange(lower_freq, upper_freq + df, df)

    if add_noise is not None:

        # following bilby convention (?)
        norm1 = 0.5 * (1.0 / df) ** 0.5
        re = np.random.normal(0, norm1, size=(3,) + data_freqs.shape)
        im = np.random.normal(0, norm1, size=(3,) + data_freqs.shape)
        htilde = re + 1j * im

        if TDItag == "AET":
            # assumes gaussian noise
            noise_channel1 = (
                (
                    np.sqrt(tdi.noisepsd_AE(data_freqs, **kwargs["noise_kwargs"]))
                    * htilde[0]
                )
                * 2
                * df ** 0.5
            )

            noise_channel2 = (
                np.sqrt(tdi.noisepsd_AE(data_freqs, **kwargs["noise_kwargs"]))
                * htilde[1]
                * 2
                * df ** 0.5
            )
            noise_channel3 = (
                np.sqrt(
                    tdi.noisepsd_T(data_freqs, model=kwargs["noise_kwargs"]["model"])
                )
                * htilde[2]
                * 2
                * df ** 0.5
            )

        else:
            # assumes gaussian noise
            noise_channel1 = (
                np.sqrt(tdi.noisepsd_XYZ(data_freqs, **kwargs["noise_kwargs"]))
                * htilde[0]
                * 2
                * df ** 0.5
            )
            noise_channel2 = (
                np.sqrt(tdi.noisepsd_XYZ(data_freqs, **kwargs["noise_kwargs"]))
                * htilde[1]
                * 2
                * df ** 0.5
            )
            noise_channel3 = (
                np.sqrt(tdi.noisepsd_XYZ(data_freqs, **kwargs["noise_kwargs"]))
                * htilde[2]
                * 2
                * df ** 0.5
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
        l_vals,
        m_vals,
        len(data_freqs),
        TDItag_in,
        waveform_params["t_obs_start"],
        waveform_params["t_obs_end"],
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

    first_inds = np.full(nwalkers * ndevices, 0, dtype=np.int32)
    last_inds = np.full(nwalkers * ndevices, len(data_freqs), dtype=np.int32)
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
        first_inds.astype(np.int32),
        last_inds.astype(np.int32),
        return_TDI=True,
    )

    channel1, channel2, channel3 = channel1[0], channel2[0], channel3[0]

    if add_noise is not None:
        channel1 = channel1 + noise_channel1
        channel2 = channel2 + noise_channel2
        channel3 = channel3 + noise_channel3

    data_stream = {TDItag[0]: channel1, TDItag[1]: channel2, TDItag[2]: channel3}
    return data_freqs, data_stream, phenomHM
