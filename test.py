import numpy as np

try:
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    print("No CuPy")
    import numpy as xp

from pyHdynBBH import *

from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
from bbhx.response.fastlisaresponse import LISATDIResponse
from bbhx.utils.interpolate import CubicSplineInterpolant
from lisatools.sensitivity import get_sensitivity
from bbhx.utils.waveformbuild import TemplateInterp, BBHWaveform
from bbhx.utils.likelihood import RelativeBinning, Likelihood

# TODO: deal with zeros in amplitude

from bbhx.utils.constants import *


use_gpu = True


def test_phenomhm(
    m1,
    m2,
    chi1z,
    chi2z,
    distance,
    phiRef,
    f_ref,
    length,
    inc,
    lam,
    beta,
    psi,
    tRef_wave_frame,
    tRef_sampling_frame,
    tBase,
    t_obs_start,
    t_obs_end,
):

    m1_test = m1 * 1.00001
    m1_temp = m1 * 1.00003
    nChannels = 3
    data_length = 2 ** 15

    num_bin_all = len(m1)

    amp_phase_kwargs = dict()
    response_kwargs = dict(max_init_len=-1, TDItag="AET", order_fresnel_stencil=0)

    bbh = BBHWaveform(
        response_kwargs=response_kwargs,
        amp_phase_kwargs=amp_phase_kwargs,
        use_gpu=use_gpu,
    )

    df = 1.0 / YRSID_SI

    f_n = xp.arange(1e-6, 1e-1 + df, df)

    modes = None  # [(2, 5), (3, 3)]

    S_n = xp.asarray(
        [
            get_sensitivity(f_n.get(), sens_fn="noisepsd_AE"),
            get_sensitivity(f_n.get(), sens_fn="noisepsd_AE"),
            get_sensitivity(f_n.get(), sens_fn="noisepsd_T"),
        ]
    )

    data_length = len(f_n)

    import time

    data = bbh(
        m1[:1],
        m2[:1],
        chi1z[:1],
        chi2z[:1],
        distance[:1],
        phiRef[:1],
        f_ref[:1],
        inc[:1],
        lam[:1],
        beta[:1],
        psi[:1],
        tRef_wave_frame[:1],
        tRef_sampling_frame[:1],
        tBase=tBase,
        t_obs_start=t_obs_start,
        t_obs_end=t_obs_end,
        freqs=f_n,
        length=2048,
        modes=modes,
        direct=False,
        compress=True,
        fill=True,
    )

    num = 100

    noise_weight_times_df = xp.sqrt(1 / S_n * df)
    data_stream_length = len(f_n)

    data_scaled = data * noise_weight_times_df

    like = Likelihood(bbh, f_n, data_scaled, noise_weight_times_df, use_gpu=True)

    numBinAll = 32

    st = time.perf_counter()
    for _ in range(num):
        ll = like(
            [
                m1_test[:numBinAll],
                m2[:numBinAll],
                chi1z[:numBinAll],
                chi2z[:numBinAll],
                distance[:numBinAll],
                phiRef[:numBinAll],
                f_ref[:numBinAll],
                inc[:numBinAll],
                lam[:numBinAll],
                beta[:numBinAll],
                psi[:numBinAll],
                tRef_wave_frame[:numBinAll],
                tRef_sampling_frame[:numBinAll],
            ],
            tBase=tBase,
            t_obs_start=t_obs_start,
            t_obs_end=t_obs_end,
            freqs=f_n,
            length=4096,
            modes=modes,
            direct=False,
            compress=True,
            fill=False,
        )

    et = time.perf_counter()

    print((et - st) / num / numBinAll)

    ll_template = like(
        [
            m1_temp[:1],
            m2[:1],
            chi1z[:1],
            chi2z[:1],
            distance[:1],
            phiRef[:1],
            f_ref[:1],
            inc[:1],
            lam[:1],
            beta[:1],
            psi[:1],
            tRef_wave_frame[:1],
            tRef_sampling_frame[:1],
        ],
        tBase=tBase,
        t_obs_start=t_obs_start,
        t_obs_end=t_obs_end,
        freqs=f_n,
        length=4096,
        modes=modes,
        direct=False,
        compress=True,
        fill=False,
    )

    d = data.reshape(3, -1)

    template_gen_args = (
        m1_temp[:1],
        m2[:1],
        chi1z[:1],
        chi2z[:1],
        distance[:1],
        phiRef[:1],
        f_ref[:1],
        inc[:1],
        lam[:1],
        beta[:1],
        psi[:1],
        tRef_wave_frame[:1],
        tRef_sampling_frame[:1],
    )

    template_gen_kwargs = dict(
        tBase=tBase,
        t_obs_start=t_obs_start,
        t_obs_end=t_obs_end,
        length=None,
        modes=modes,
        direct=True,
        compress=True,
    )

    relbin = RelativeBinning(
        bbh,
        f_n,
        d,
        template_gen_args,
        length,
        template_gen_kwargs=template_gen_kwargs,
        use_gpu=use_gpu,
    )

    d_d = relbin.base_d_d

    import time

    st = time.perf_counter()
    num = 100

    for _ in range(num):
        ll_res = relbin(
            [
                m1_test,
                m2,
                chi1z,
                chi2z,
                distance,
                phiRef,
                f_ref,
                inc,
                lam,
                beta,
                psi,
                tRef_wave_frame,
                tRef_sampling_frame,
            ],
            tBase=tBase,
            t_obs_start=t_obs_start,
            t_obs_end=t_obs_end,
            freqs=None,
            length=None,
            modes=modes,
            direct=True,
            compress=True,
            squeeze=False,
        )
    et = time.perf_counter()
    print((et - st) / num / num_bin_all, num, num_bin_all)

    breakpoint()


if __name__ == "__main__":

    num_bin_all = 10000
    length = 256

    m1 = np.full(num_bin_all, 1.000000e5)
    # m1[1:] += np.random.randn(num_bin_all - 1) * 100
    m2 = np.full_like(m1, 9e4)
    chi1z = np.full_like(m1, 0.2)
    chi2z = np.full_like(m1, 0.2)
    distance = np.full_like(m1, 10e9 * PC_SI)
    phiRef = np.full_like(m1, 0.0)
    f_ref = np.full_like(m1, 0.0)

    inc = np.full_like(m1, 0.2)
    lam = np.full_like(m1, 0.0)
    beta = np.full_like(m1, 0.3)
    psi = np.full_like(m1, 0.4)
    tRef_wave_frame = np.full_like(m1, 100.0)
    tRef_sampling_frame = np.full_like(m1, 120.0)
    tBase = 1.0

    t_obs_start = 1.0
    t_obs_end = 0.0

    test_phenomhm(
        m1,
        m2,
        chi1z,
        chi2z,
        distance,
        phiRef,
        f_ref,
        length,
        inc,
        lam,
        beta,
        psi,
        tRef_wave_frame,
        tRef_sampling_frame,
        tBase,
        t_obs_start,
        t_obs_end,
    )
