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
    def __init__(self, max_length_init, l_vals,  m_vals, data_freqs, data_stream, t0, key_order, **kwargs):
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
        }

        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        self.converter = Converter(key_order, tLtoSSB=self.tLtoSSB)
        self.recycler = Recycler(key_order, tLtoSSB=self.tLtoSSB)

        self.t0 = t0
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

        if self.TDItag == 'AET':
            self.TDItag_in = 2
            self.channel1_ASDinv = 1./np.sqrt(tdi.noisepsd_AE(self.data_freqs, model='SciRDv1'))*additional_factor
            self.channel2_ASDinv = 1./np.sqrt(tdi.noisepsd_AE(self.data_freqs, model='SciRDv1'))*additional_factor
            self.channel3_ASDinv = 1./np.sqrt(tdi.noisepsd_T(self.data_freqs, model='SciRDv1'))*additional_factor

        elif self.TDItag == 'XYZ':
            self.TDItag_in = 1
            for i in range(1, 4):
                temp = np.sqrt(tdi.noisepsd_XYZ(self.data_freqs, model='SciRDv1'))*additional_factor
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
                          self.TDItag_in)

    def NLL(self, m1, m2, a1, a2, distance,
                 phiRef, inc, lam, beta,
                 psi, tRef, freqs=None, return_amp_phase=False, return_TDI=False):

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
                                              self.t0, tRef, merger_freq,
                                              return_amp_phase=return_amp_phase,
                                              return_TDI=return_TDI)

        if return_amp_phase or return_TDI:
            return out

        d_h, h_h = out

        return self.d_d + h_h - 2*d_h


def create_data_set(l_vals,  m_vals, t0, waveform_params, data_freqs=None, TDItag='AET', num_data_points=int(2**19), num_generate_points=int(2**18), df=None, min_dimensionless_freq=1e-4, max_dimensionless_freq=1.0, **kwargs):
    key_list = list(waveform_params.keys())
    converter = Converter(key_list, **kwargs)
    recycler = Recycler(key_list, **kwargs)

    vals = np.array([waveform_params[key] for key in key_list])

    vals = recycler.recycle(vals)
    vals = converter.convert(vals)

    waveform_params = {key: vals[i] for i, key in enumerate(key_list)}

    if 'ln_m1' in waveform_params:
        waveform_params['m1'] = waveform_params['ln_m1']
        waveform_params['m2'] = waveform_params['ln_m2']
    if 'ln_mT' in waveform_params:
        # has been converted
        waveform_params['m1'] = waveform_params['ln_mT']
        waveform_params['m2'] = waveform_params['mr']

    waveform_params['distance'] = waveform_params['ln_distance']
    waveform_params['tRef'] = waveform_params['ln_tRef']

    if data_freqs is None:
        m1 = waveform_params['m1']
        m2 = waveform_params['m2']
        Msec = (m1+m2)*MTSUN
        upper_freq = max_dimensionless_freq/Msec
        lower_freq = min_dimensionless_freq/Msec
        merger_freq = 0.018/Msec
        waveform_params['fRef'] = 0.0
        if df is None:
            data_freqs = np.logspace(np.log10(lower_freq), np.log10(upper_freq), num_data_points)
        else:
            data_freqs = np.arange(fmin, fmax+df, df)

    generate_freqs = np.logspace(np.log10(data_freqs.min()), np.log10(data_freqs.max()), num_generate_points)

    fake_data = np.zeros_like(data_freqs, dtype=np.complex128)
    fake_ASD = np.ones_like(data_freqs)

    if TDItag == 'AET':
        TDItag_in = 2

    elif TDItag == 'XYZ':
        TDItag_in = 1

    phenomHM = PhenomHM(len(generate_freqs), l_vals, m_vals, data_freqs, fake_data, fake_data, fake_data, fake_ASD, fake_ASD, fake_ASD, TDItag_in)

    phenomHM.gen_amp_phase(generate_freqs, waveform_params['m1'],  # solar masses
                 waveform_params['m2'],  # solar masses
                 waveform_params['a1'],
                 waveform_params['a2'],
                 waveform_params['distance'],
                 waveform_params['phiRef'],
                 waveform_params['fRef'])

    phenomHM.LISAresponseFD(waveform_params['inc'], waveform_params['lam'], waveform_params['beta'], waveform_params['psi'], waveform_params['t0'], waveform_params['tRef'], merger_freq)
    phenomHM.setup_interp_wave()
    phenomHM.setup_interp_response()
    phenomHM.perform_interp()

    channel1, channel2, channel3 = phenomHM.GetTDI()

    if channel1.ndim > 1:
        channel1, channel2, channel3 = channel1.sum(axis=0), channel2.sum(axis=0), channel3.sum(axis=0)

    data_stream = {TDItag[0]: channel1, TDItag[1]: channel2, TDItag[2]: channel3}
    return data_freqs, data_stream
