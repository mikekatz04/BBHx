from phenomhm.phenomhm import pyPhenomHM
import numpy as np

if __name__ == "__main__":
    prop_defaults = {
        'TDItag': 'AET',  # AET or XYZ
        'max_dimensionless_freq': 0.5,
        'min_dimensionless_freq': 1e-4,
        'data_stream_whitened': True,
        'data_params': {},
        'log_scaled_likelihood': True,
        'eps': 1e-6,
        'test_inds': None,
        'num_params': 11,
        'num_data_points': int(2**19),
        'df': None,
        'tLtoSSB': True,
        'noise_kwargs': {'model': 'SciRDv1'},
    }

    max_length_init = 2**12
    nwalkers, ndevices = 20, 2
    l_vals = np.array([2, 3, 4, 2, 3, 4], dtype=np.uint32)
    m_vals = np.array([2, 3, 4, 1, 2, 3], dtype=np.uint32)
    data_freqs, data_stream = None, None
    t0 = 1.0
    t_obs_dur = 0.9

    data_params = {
        'ln_mT': np.log(4e7),
        'mr': 0.2,
        'a1': 0.0,
        'a2': 0.0,
        'ln_distance': np.log(15.93461637),  # Gpc z=2
        'phiRef': 3.09823412789,
        'cos_inc': np.cos(2.98553920),
        'lam': 5.900332547,
        'sin_beta': np.sin(-1.3748820938),
        'psi': 0.139820023,
        'ln_tRef': np.log(2.39284219993e1)
    }

    data_params = {
        'ln_mT': np.log(2.00000000e+06),
        'mr': 1/3.00000000e+00,
        'a1': 0.0,
        'a2': 0.0,
        'ln_distance': np.log(3.65943000e+01),  # Gpc z=2
        'phiRef': 2.13954125e+00,
        'cos_inc': np.cos(1.04719755e+00),
        'lam': -2.43647481e-02,
        'sin_beta': np.sin(6.24341583e-01),
        'psi': 2.02958790e+00,
        'ln_tRef': np.log(5.02462348e+01)
    }

    prop_defaults['data_params'] = data_params

    key_order = [
        'ln_mT',
        'mr',
        'a1',
        'a2',
        'ln_distance',  # Gpc z=2
        'phiRef',
        'cos_inc',
        'lam',
        'sin_beta',
        'psi',
        'ln_tRef'
    ]

    phenomhm = pyPhenomHM(max_length_init, nwalkers, ndevices, l_vals,  m_vals, data_freqs, data_stream, t0, t_obs_dur, key_order, **prop_defaults)

    waveform_params = np.tile(np.array([data_params[key] for key in key_order]), (nwalkers, 1))

    data_params2 = {
        'ln_mT': np.log(2.00000000e+06),
        'mr': 1/3.00000000e+00,
        'a1': 0.0,
        'a2': 0.0,
        'ln_distance': np.log(3.65943000e+01),  # Gpc z=2
        'phiRef': 2.13954125e+00,
        'cos_inc': np.cos(1.04719755e+00),
        'lam': -2.43647481e-02,
        'sin_beta': np.sin(6.24341583e-01),
        'psi': 2.02958790e+00,
        'ln_tRef': np.log(5.02462348e+01)
    }

    waveform_params = np.tile(np.array([data_params[key] for key in key_order]), (nwalkers*ndevices, 1))

    waveform_params[0:ndevices*nwalkers:2, 6] *= -1
    waveform_params[0:ndevices*nwalkers:2, 8] *= -1
    waveform_params[0:ndevices*nwalkers:2, 9] = np.pi - waveform_params[0:ndevices*nwalkers:2, 9]

    for i in range(4):
        waveform_params[(i)*int(ndevices*nwalkers/4):(i+1)*int(ndevices*nwalkers/4), 7] += i*np.pi/2
        waveform_params[(i)*int(ndevices*nwalkers/4):(i+1)*int(ndevices*nwalkers/4), 9] += i*np.pi/2

    check = phenomhm.getNLL(waveform_params.T)
    fishder = phenomhm.get_Fisher(waveform_params[0])
    import pdb; pdb.set_trace()
