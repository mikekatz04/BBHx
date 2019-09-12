"""
Wrapper code for gpuPhenomHM. Helps to calculate likelihoods
for samplers. Author: Michael Katz

Calculates phenomHM waveforms, puts them through the LISA response
and calculates likelihood.
"""

import numpy as np
from scipy import constants as ct
from .utils.convert import Converter, Recycler

import tdi

try:
    from gpuPhenomHM import PhenomHM
except ImportError:
    from PhenomHM import PhenomHM

import time

MTSUN = 1.989e30*ct.G/ct.c**3


class pyPhenomHM(Converter):
    def __init__(self, max_length_init, l_vals,  m_vals, data_freqs, data_stream, t0, t_obs_dur, key_order, **kwargs):
        """
        data_stream (dict): keys X, Y, Z or A, E, T
        """
        prop_defaults = {
            'TDItag': 'AET',  # AET or XYZ
            'max_dimensionless_freq': 0.5,
            'min_dimensionless_freq': 1e-4,
            'data_stream_whitened': True,
            'data_params': {},
            'log_scaled_likelihood': True,
            'eps': 1e-6,
            'test_inds': None,
            'num_params': 12,
            'num_data_points': int(2**19),
            'num_generate_points': int(2**18),
            'df': None,
            'tLtoSSB': True,
            'noise_kwargs': {'model': 'SciRDv1'},
        }

        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
            #TODO: check this
            kwargs[prop] = kwargs.get(prop, default)

        self.converter = Converter(key_order, tLtoSSB=self.tLtoSSB)
        self.recycler = Recycler(key_order, tLtoSSB=self.tLtoSSB)

        self.t0 = t0
        self.t_obs_dur = t_obs_dur
        self.max_length_init = max_length_init
        self.l_vals, self.m_vals = l_vals, m_vals
        self.data_freqs, self.data_stream = data_freqs, data_stream

        if self.test_inds is None:
            self.test_inds = np.arange(self.num_params)

        if self.TDItag not in ['AET', 'XYZ']:
            raise ValueError('TDItag must be AET or XYZ.')

        if self.data_stream is {} or self.data_stream is None:
            if self.data_params is {}:
                raise ValueError('If data_stream is empty dict or None,'
                                 + 'user must supply data_params kwarg as'
                                 + 'dict with params for data stream.')
            kwargs['data_params']['t0'] = t0
            kwargs['data_params']['t_obs_dur'] = t_obs_dur

            self.data_freqs, self.data_stream = (create_data_set(l_vals,  m_vals, t0,
                                self.data_params, data_freqs=data_freqs, **kwargs))
            self.data_stream_whitened = False

        for i, channel in enumerate(self.TDItag):
            if channel not in self.data_stream:
                raise KeyError('{} not in TDItag {}.'.format(channel, self.TDItag))

            setattr(self, 'data_channel{}'.format(i+1), self.data_stream[channel])
        additional_factor = np.ones_like(self.data_freqs)
        if self.log_scaled_likelihood:
            additional_factor[1:] = np.sqrt(np.diff(self.data_freqs))
            additional_factor[0] = additional_factor[1]
        else:
            df = self.data_freqs[1] - self.data_freqs[0]
            additional_factor = np.sqrt(df)

        if self.TDItag == 'AET':
            self.TDItag_in = 2

            """AE_noise = np.genfromtxt('SnAE2017.dat').T
            T_noise = np.genfromtxt('SnAE2017.dat').T

            from scipy.interpolate import CubicSpline

            AE_noise = CubicSpline(AE_noise[0], AE_noise[1])
            T_noise = CubicSpline(T_noise[0], T_noise[1])

            self.channel1_ASDinv = 1./np.sqrt(AE_noise(self.data_freqs))*additional_factor
            self.channel2_ASDinv = 1./np.sqrt(AE_noise(self.data_freqs))*additional_factor
            self.channel3_ASDinv = 1./np.sqrt(T_noise(self.data_freqs))*additional_factor"""

            self.channel1_ASDinv = 1./np.sqrt(tdi.noisepsd_AE(self.data_freqs, **self.noise_kwargs))*additional_factor
            self.channel2_ASDinv = 1./np.sqrt(tdi.noisepsd_AE(self.data_freqs, **self.noise_kwargs))*additional_factor
            self.channel3_ASDinv = 1./np.sqrt(tdi.noisepsd_T(self.data_freqs, **self.noise_kwargs))*additional_factor

        elif self.TDItag == 'XYZ':
            self.TDItag_in = 1
            for i in range(1, 4):
                temp = np.sqrt(tdi.noisepsd_XYZ(self.data_freqs, **self.noise_kwargs))*additional_factor
                setattr(self, 'channel{}_ASDinv'.format(i), temp)

        if self.data_stream_whitened is False:
            for i in range(1, 4):
                temp = (getattr(self, 'data_channel{}'.format(i)) *
                        getattr(self, 'channel{}_ASDinv'.format(i)))
                setattr(self, 'data_channel{}'.format(i), temp)

        self.d_d = 4*np.sum([np.abs(self.data_channel1)**2, np.abs(self.data_channel2)**2, np.abs(self.data_channel3)**2])

        self.generator = PhenomHM(self.max_length_init,
                          self.l_vals, self.m_vals,
                          self.data_freqs, self.data_channel1,
                          self.data_channel2, self.data_channel3,
                          self.channel1_ASDinv, self.channel2_ASDinv, self.channel3_ASDinv,
                          self.TDItag_in, self.t_obs_dur)

    def NLL(self, m1, m2, a1, a2, distance,
                 phiRef, inc, lam, beta,
                 psi, tRef_wave_frame, tRef_sampling_frame, freqs=None, return_amp_phase=False, return_TDI=False, return_snr=False):

        Msec = (m1+m2)*MTSUN
        # merger frequency for 22 mode amplitude in phenomD
        merger_freq = 0.018/Msec
        fRef = 0.0

        if freqs is None:
            upper_freq = self.max_dimensionless_freq/Msec
            lower_freq = self.min_dimensionless_freq/Msec
            freqs = np.logspace(np.log10(lower_freq), np.log10(upper_freq), self.max_length_init)

        out = self.generator.WaveformThroughLikelihood(freqs,
                                              m1, m2,  # solar masses
                                              a1, a2,
                                              distance, phiRef, fRef,
                                              inc, lam, beta, psi,
                                              self.t0, tRef_wave_frame, tRef_sampling_frame, merger_freq,
                                              return_amp_phase=return_amp_phase,
                                              return_TDI=return_TDI)

        if return_amp_phase or return_TDI:
            return out

        d_h, h_h = out

        if return_snr:
            return np.sqrt(d_h), np.sqrt(h_h)

        return self.d_d + h_h - 2*d_h


def create_data_set(l_vals,  m_vals, t0, waveform_params, data_freqs=None, TDItag='AET', num_data_points=int(2**19), num_generate_points=int(2**18), df=None, min_dimensionless_freq=1e-4, max_dimensionless_freq=1.0, add_noise=None, **kwargs):
    key_list = list(waveform_params.keys())
    converter = Converter(key_list, **kwargs)
    recycler = Recycler(key_list, **kwargs)

    vals = np.array([waveform_params[key] for key in key_list])

    tRef_sampling_frame = np.exp(vals[10])

    vals = recycler.recycle(vals)
    vals = converter.convert(vals)

    waveform_params = {key: vals[i] for i, key in enumerate(key_list)}

    waveform_params['tRef_sampling_frame'] = tRef_sampling_frame

    if 'ln_m1' in waveform_params:
        waveform_params['m1'] = waveform_params['ln_m1']
        waveform_params['m2'] = waveform_params['ln_m2']
    if 'ln_mT' in waveform_params:
        # has been converted
        waveform_params['m1'] = waveform_params['ln_mT']
        waveform_params['m2'] = waveform_params['mr']

    if 'chi_s' in waveform_params:
        waveform_params['a1'] = waveform_params['chi_s']
        waveform_params['a2'] = waveform_params['chi_a']

    if 'cos_inc' in waveform_params:
        waveform_params['inc'] = waveform_params['cos_inc']

    if 'sin_beta' in waveform_params:
        waveform_params['beta'] = waveform_params['sin_beta']

    waveform_params['distance'] = waveform_params['ln_distance']
    waveform_params['tRef_wave_frame'] = waveform_params['ln_tRef']
    waveform_params['fRef'] = 0.0

    m1 = waveform_params['m1']
    m2 = waveform_params['m2']
    Msec = (m1+m2)*MTSUN
    merger_freq = 0.018/Msec

    if data_freqs is None:
        if add_noise is not None:
            fs = add_noise['fs']
            t_obs_dur = waveform_params['t_obs_dur']
            df = 1./(t_obs_dur*ct.Julian_year)
            num_data_points = int(t_obs_dur*ct.Julian_year*fs)
            noise_freqs = np.fft.rfftfreq(num_data_points, 1/fs)
            data_freqs = noise_freqs[noise_freqs >= add_noise['min_freq']]

        else:
            m1 = waveform_params['m1']
            m2 = waveform_params['m2']
            Msec = (m1+m2)*MTSUN
            upper_freq = max_dimensionless_freq/Msec
            lower_freq = min_dimensionless_freq/Msec
            merger_freq = 0.018/Msec
            if df is None:
                data_freqs = np.logspace(np.log10(lower_freq), np.log10(upper_freq), num_data_points)
            else:
                data_freqs = np.arange(fmin, fmax+df, df)

    if add_noise is not None:

        norm1 = 0.5 * (1. / df)**0.5
        re = np.random.normal(0, norm1, size=(3,) + data_freqs.shape)
        im = np.random.normal(0, norm1, size=(3,) + data_freqs.shape)
        htilde = re + 1j * im

        # the 0.125 is 1/8 to match LDC data #FIXME
        if TDItag == 'AET':
            # assumes gaussian noise
            noise_channel1 = np.sqrt(tdi.noisepsd_AE(data_freqs, **kwargs['noise_kwargs']))*htilde[0]*0.125
            noise_channel2 = np.sqrt(tdi.noisepsd_AE(data_freqs, **kwargs['noise_kwargs']))*htilde[1]*0.125
            noise_channel3 = np.sqrt(tdi.noisepsd_T(data_freqs, **kwargs['noise_kwargs']))*htilde[2]*0.125

        else:
            # assumes gaussian noise
            noise_channel1 = np.sqrt(tdi.noisepsd_XYZ(data_freqs, **kwargs['noise_kwargs']))*htilde[0]*0.125
            noise_channel2 = np.sqrt(tdi.noisepsd_XYZ(data_freqs, **kwargs['noise_kwargs']))*htilde[1]*0.125
            noise_channel3 = np.sqrt(tdi.noisepsd_XYZ(data_freqs, **kwargs['noise_kwargs']))*htilde[2]*0.125

    generate_freqs = np.logspace(np.log10(data_freqs.min()), np.log10(data_freqs.max()), num_generate_points)

    fake_data = np.zeros_like(data_freqs, dtype=np.complex128)
    fake_ASD = np.ones_like(data_freqs)

    if TDItag == 'AET':
        TDItag_in = 2

    elif TDItag == 'XYZ':
        TDItag_in = 1

    phenomHM = PhenomHM(len(generate_freqs), l_vals, m_vals, data_freqs, fake_data, fake_data, fake_data, fake_ASD, fake_ASD, fake_ASD, TDItag_in, waveform_params['t_obs_dur'])

    phenomHM.gen_amp_phase(generate_freqs, waveform_params['m1'],  # solar masses
                 waveform_params['m2'],  # solar masses
                 waveform_params['a1'],
                 waveform_params['a2'],
                 waveform_params['distance'],
                 waveform_params['phiRef'],
                 waveform_params['fRef'])

    phenomHM.LISAresponseFD(waveform_params['inc'], waveform_params['lam'], waveform_params['beta'], waveform_params['psi'], waveform_params['t0'], waveform_params['tRef_wave_frame'], waveform_params['tRef_sampling_frame'], merger_freq)
    phenomHM.setup_interp_wave()
    phenomHM.setup_interp_response()
    phenomHM.perform_interp()

    channel1, channel2, channel3 = phenomHM.GetTDI()

    if channel1.ndim > 1:
        channel1, channel2, channel3 = channel1.sum(axis=0), channel2.sum(axis=0), channel3.sum(axis=0)

    if add_noise is not None:
        channel1 = channel1 + noise_channel1
        channel2 = channel2 + noise_channel2
        channel3 = channel3 + noise_channel3

    data_stream = {TDItag[0]: channel1, TDItag[1]: channel2, TDItag[2]: channel3}
    return data_freqs, data_stream
