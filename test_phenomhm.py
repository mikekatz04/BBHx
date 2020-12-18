from phenomhm.phenomhm import pyPhenomHM
import numpy as np
import argparse
import time
import scipy.constants as ct

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--time", "-t", default=0, type=int)
    args = parser.parse_args()

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
        "num_data_points": int(2 ** 16),
        "df": None,
        "tLtoSSB": True,
        "noise_kwargs": {"model": "SciRDv1", "includewd": None},
        # "add_noise": {"fs": 0.1, "min_freq": 1e-7},
    }

    max_length_init = 2 ** 14
    nwalkers, ndevices = 24, 1
    l_vals = np.array([2], dtype=np.uint32)  # , 3, 4, 2, 3, 4], dtype=np.uint32)
    m_vals = np.array([2], dtype=np.uint32)  # , 3, 4, 1, 2, 3], dtype=np.uint32)
    data_freqs, data_stream = None, None
    t0 = 0.0
    t_obs_start = 24959512.543446492
    t_obs_end = 0.0

    data_params = {
        "ln_mT": np.log(4e7),
        "q": 1.0 / 5.0,
        "a1": 0.0,
        "a2": 0.0,
        "ln_distance": np.log(15.93461637),  # Gpc z=2
        "phiRef": 3.09823412789,
        "cos_inc": np.cos(1.342937847),
        "lam": 4.21039847344,
        "sin_beta": np.sin(-0.7328920398),
        "psi": 0.139820023,
        "ln_tRef": np.log(2.39284219993e1),
    }

    data_params = {
        "ln_mT": np.log(2.00000000e06),
        "q": 1 / 3.00000000e00,
        "a1": 0.0,
        "a2": 0.0,
        "ln_distance": np.log(3.65943000e01),  # Gpc z=2
        "phiRef": 2.13954125e00,
        "cos_inc": np.cos(1.04719755e00),
        "lam": -2.43647481e-02,
        "sin_beta": np.sin(6.24341583e-01),
        "psi": 2.02958790e00,
        "ln_tRef": np.log(5.02462348e01),
    }

    """


    data_params = {
        "ln_mT": 13.364379218989889,
        "q": 0.9010856895246949,
        "a1": 0.182299389588786,
        "a2": 0.5678791057168875,
        "ln_distance": 1.5097807282180846,
        "phiRef": 1.121525591993405,
        "cos_inc": 0.988442340258988,
        "lam": 6.091956125810419,
        "sin_beta": 0.10988800651854277,
        "psi": 3.318620395313955,
        "ln_tRef": 17.867701059562503,
    }
    """

    data_params = {
        "ln_mT": np.log(2599137.035 + 1242860.685),
        "q": 1242860.685 / 2599137.035,
        "a1": 0.7534821857057837,
        "a2": 0.6215875279643664,
        "ln_distance": np.log(56005.783662877526 / 1e3),
        "cos_inc": np.cos(1.2245321255939288),
        "phiRef": 6.247897265570264,
        "lam": -2.5765925991650085,
        "sin_beta": np.sin(0.05294026120170111),
        "psi": 0.8346797841575135,
        "ln_tRef": np.log(24959512.543446492),
    }

    prop_defaults["data_params"] = data_params

    key_order = [
        "ln_mT",
        "q",
        "a1",
        "a2",
        "ln_distance",  # Gpc z=2
        "phiRef",
        "cos_inc",
        "lam",
        "sin_beta",
        "psi",
        "ln_tRef",
    ]

    phenomhm = pyPhenomHM(
        data_params,
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
        t_obs_end=t_obs_end,
        **prop_defaults
    )

    orig_params = np.array([data_params[key] for key in key_order])
    waveform_params = np.tile(
        np.array([data_params[key] for key in key_order]), (nwalkers, 1)
    )

    waveform_params = np.tile(
        np.array([data_params[key] for key in key_order]), (nwalkers * ndevices, 1)
    )

    waveform_params[0 : ndevices * nwalkers : 2, 6] *= -1
    waveform_params[0 : ndevices * nwalkers : 2, 8] *= -1
    waveform_params[0 : ndevices * nwalkers : 2, 9] = (
        np.pi - waveform_params[0 : ndevices * nwalkers : 2, 9]
    )

    for i in range(4):
        waveform_params[
            (i) * int(ndevices * nwalkers / 4) : (i + 1) * int(ndevices * nwalkers / 4),
            7,
        ] += (i * np.pi / 2)
        waveform_params[
            (i) * int(ndevices * nwalkers / 4) : (i + 1) * int(ndevices * nwalkers / 4),
            9,
        ] += (i * np.pi / 2)

    check = phenomhm.getNLL(waveform_params.T)

    if args.time:
        st = time.perf_counter()
        check = phenomhm.getNLL(waveform_params.T)
        for _ in range(args.time):
            check = phenomhm.getNLL(waveform_params.T)
        et = time.perf_counter()

        print("Number of evals:", args.time)
        print("ndevices:", ndevices, "nwalkers:", nwalkers)
        print("total time:", et - st)
        print("time per group likelihood call:", (et - st) / args.time)
        print(
            "time per individual likelihood call:",
            (et - st) / (args.time * nwalkers * ndevices),
        )

    check = phenomhm.getNLL(waveform_params.T)
    fisher = phenomhm.get_Fisher(waveform_params[0])

    snr_check = phenomhm.getNLL(waveform_params.T, return_snr=True)
    print(check[0:3])

    new_pars = np.array(
        [
            np.log(2599137.035 + 1242860.685),
            1242860.685 / 2599137.035,
            0.7534821857057837,
            0.6215875279643664,
            np.log(164401.62761742485 / 1e3),
            6.247897265570264,
            np.cos(0.0),
            -2.5676962200986537,
            np.sin(0.08599977626833821),
            0.8346797841575135,
            np.log(24959512.543446492),
        ]
    )

    waveform_params[0] = orig_params
    waveform_params[1] = new_pars
    snr_check = phenomhm.getNLL(waveform_params.T, return_snr=True)
    check_new = phenomhm.getNLL(waveform_params.T)
    import pdb

    pdb.set_trace()
