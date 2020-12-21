import numpy as np

try:
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    print("No CuPy")
    import numpy as xp

from pyHdynBBH import *

from lisatools.sensitivity import get_sensitivity

# TODO: deal with zeros in amplitude

PI = 3.141592653589793238462643383279502884
TWOPI = 6.283185307179586476925286766559005768
PI_2 = 1.570796326794896619231321691639751442
PI_4 = 0.785398163397448309615660845819875721
MRSUN_SI = 1.476625061404649406193430731479084713e3
MTSUN_SI = 4.925491025543575903411922162094833998e-6
MSUN_SI = 1.988546954961461467461011951140572744e30

GAMMA = 0.577215664901532860606512090082402431
PC_SI = 3.085677581491367278913937957796471611e16

YRSID_SI = 31558149.763545600

F0 = 3.168753578687779e-08
Omega0 = 1.9909865927683788e-07

ua = 149597870700.0
R_SI = 149597870700.0
AU_SI = 149597870700.0
aorbit = 149597870700.0

clight = 299792458.0
sqrt3 = 1.7320508075688772
invsqrt3 = 0.5773502691896258
invsqrt6 = 0.4082482904638631
sqrt2 = 1.4142135623730951
L_SI = 2.5e9
eorbit = 0.004824185218078991
C_SI = 299792458.0

use_gpu = True


class PhenomHMAmpPhase:
    def __init__(self, max_init_len=-1, use_gpu=False):

        if use_gpu:
            self.xp = xp

        else:
            self.xp = np

        if max_init_len > 0:
            self.use_buffers = True
            raise NotImplementedError

        else:
            self.use_buffers = False

        self.allowable_modes = [(2, 2), (3, 3), (4, 4), (2, 1), (3, 2), (4, 3)]

        self.ells_default = self.xp.array([2, 3, 4, 2, 3, 4], dtype=self.xp.int32)

        self.mms_default = self.xp.array([2, 3, 4, 1, 2, 3], dtype=self.xp.int32)

        self.Mf_min = 1e-4
        self.Mf_max = 0.6

    def _sanity_check_modes(self, ells, mms):

        for (ell, mm) in zip(ells, mms):
            if (ell, mm) not in self.allowable_modes:
                raise ValueError(
                    "Requested mode [(l,m) = ({},{})] is not available. Allowable modes include {}".format(
                        ell, mm, self.allowable_modes
                    )
                )

    def _initialize_waveform_container(self):

        self.waveform_carrier = self.xp.zeros(
            (self.length * self.num_modes * self.num_bin_all * self.nparams),
            dtype=self.xp.float64,
        )

    def _initialize_freqs(self, m1, m2):
        M_tot_sec = (m1 + m2) * MTSUN_SI

        base_freqs = self.xp.logspace(
            self.xp.log10(self.Mf_min), self.xp.log10(self.Mf_max), self.length
        )

        self.freqs = (
            base_freqs[:, self.xp.newaxis] / M_tot_sec[self.xp.newaxis, :]
        ).flatten()

    @property
    def amp(self):
        amps = self.waveform_carrier[
            0 * self.num_per_param : 1 * self.num_per_param
        ].reshape(self.length, self.num_modes, self.num_bin_all)

        amps = self.xp.transpose(amps, axes=(2, 1, 0))
        return amps

    @property
    def phase(self):
        phase = self.waveform_carrier[
            1 * self.num_per_param : 2 * self.num_per_param
        ].reshape(self.length, self.num_modes, self.num_bin_all)

        phase = self.xp.transpose(phase, axes=(2, 1, 0))
        return phase

    @property
    def phase_deriv(self):
        phase_deriv = self.waveform_carrier[
            2 * self.num_per_param : 3 * self.num_per_param
        ].reshape(self.length, self.num_modes, self.num_bin_all)

        phase_deriv = self.xp.transpose(phase_deriv, axes=(2, 1, 0))
        return phase_deriv

    @property
    def freqs_shaped(self):
        return self._freqs.reshape(self.length, self.num_bin_all).T

    @property
    def freqs(self):
        return self._freqs

    @freqs.setter
    def freqs(self, f):
        if f.ndim > 1:
            self._freqs = f.flatten()

        else:
            self._freqs = f

    def __call__(
        self,
        m1,
        m2,
        chi1z,
        chi2z,
        distance,
        phiRef,
        f_ref,
        length,
        freqs=None,
        out_buffer=None,
        modes=None,
    ):

        if modes is not None:
            ells, mms = self.xp.asarray([[ell, mm] for ell, mm in modes]).T

            self._sanity_check_modes(ells, mms)

        else:
            ells = self.ells_default
            mms = self.mms_default

        num_modes = len(ells)
        num_bin_all = len(m1)

        self.length = length
        self.num_modes = num_modes
        self.num_bin_all = num_bin_all
        self.nparams = 3
        self.num_per_param = length * num_modes * num_bin_all
        self.num_per_bin = length * num_modes

        m1 = self.xp.asarray(m1)
        m2 = self.xp.asarray(m2)
        chi1z = self.xp.asarray(chi1z)
        chi2z = self.xp.asarray(chi2z)
        distance = self.xp.asarray(distance)
        phiRef = self.xp.asarray(phiRef)
        f_ref = self.xp.asarray(f_ref)

        if out_buffer is None:
            self._initialize_waveform_container()

        else:
            self.waveform_carrier = out_buffer

        if freqs is None:
            self._initialize_freqs(m1, m2)

        else:
            self.freqs = freqs

        m1_SI = m1 * MSUN_SI
        m2_SI = m2 * MSUN_SI

        waveform_amp_phase_wrap(
            self.waveform_carrier,
            ells,
            mms,
            self.freqs,
            m1_SI,
            m2_SI,
            chi1z,
            chi2z,
            distance,
            phiRef,
            f_ref,
            num_modes,
            length,
            num_bin_all,
        )


class LISATDIResponse:
    def __init__(
        self, max_init_len=-1, TDItag="AET", order_fresnel_stencil=0, use_gpu=False
    ):

        if use_gpu:
            self.xp = xp

        else:
            self.xp = np

        if max_init_len > 0:
            self.use_buffers = True
            raise NotImplementedError

        else:
            self.use_buffers = False

        if order_fresnel_stencil > 0:
            raise NotImplementedError

        self.order_fresnel_stencil = order_fresnel_stencil

        self.TDItag = TDItag
        if TDItag == "XYZ":
            self.TDItag_int = 1

        else:
            self.TDItag_int = 2

        self.allowable_modes = [(2, 2), (3, 3), (4, 4), (2, 1), (3, 2), (4, 3)]

        self.ells_default = self.xp.array([2, 3, 4, 2, 3, 4], dtype=self.xp.int32)

        self.mms_default = self.xp.array([2, 3, 4, 1, 2, 3], dtype=self.xp.int32)

    def _sanity_check_modes(self, ells, mms):

        for (ell, mm) in zip(ells, mms):
            if (ell, mm) not in self.allowable_modes:
                raise ValueError(
                    "Requested mode [(l,m) = ({},{})] is not available. Allowable modes include {}".format(
                        ell, mm, self.allowable_modes
                    )
                )

    def _initialize_response_container(self):

        self.response_carrier = self.xp.zeros(
            (self.length * self.num_modes * self.num_bin_all * self.nparams),
            dtype=self.xp.float64,
        )

    @property
    def transferL1(self):
        temp = self.response_carrier[
            (self.includes_amps + 2)
            * self.num_per_param : (self.includes_amps + 3)
            * self.num_per_param
        ].reshape(
            self.length, self.num_modes, self.num_bin_all
        ) + 1j * self.response_carrier[
            (self.includes_amps + 3)
            * self.num_per_param : (self.includes_amps + 4)
            * self.num_per_param
        ].reshape(
            self.length, self.num_modes, self.num_bin_all
        )

        temp = self.xp.transpose(temp, axes=(2, 1, 0))
        return temp

    @property
    def transferL2(self):
        temp = self.response_carrier[
            (self.includes_amps + 4)
            * self.num_per_param : (self.includes_amps + 5)
            * self.num_per_param
        ].reshape(
            self.length, self.num_modes, self.num_bin_all
        ) + 1j * self.response_carrier[
            (self.includes_amps + 5)
            * self.num_per_param : (self.includes_amps + 6)
            * self.num_per_param
        ].reshape(
            self.length, self.num_modes, self.num_bin_all
        )

        temp = self.xp.transpose(temp, axes=(2, 1, 0))
        return temp

    @property
    def transferL3(self):
        temp = self.response_carrier[
            (self.includes_amps + 6)
            * self.num_per_param : (self.includes_amps + 7)
            * self.num_per_param
        ].reshape(
            self.length, self.num_modes, self.num_bin_all
        ) + 1j * self.response_carrier[
            (self.includes_amps + 7)
            * self.num_per_param : (self.includes_amps + 8)
            * self.num_per_param
        ].reshape(
            self.length, self.num_modes, self.num_bin_all
        )

        temp = self.xp.transpose(temp, axes=(2, 1, 0))
        return temp

    @property
    def phase(self):
        phase = self.response_carrier[
            (self.includes_amps + 0)
            * self.num_per_param : (self.includes_amps + 1)
            * self.num_per_param
        ].reshape(self.length, self.num_modes, self.num_bin_all)

        phase = self.xp.transpose(phase, axes=(2, 1, 0))
        return phase

    @property
    def phase_deriv(self):
        phase_deriv = self.response_carrier[
            (self.includes_amps + 1)
            * self.num_per_param : (self.includes_amps + 2)
            * self.num_per_param
        ].reshape(self.length, self.num_modes, self.num_bin_all)

        phase_deriv = self.xp.transpose(phase_deriv, axes=(2, 1, 0))
        return phase_deriv

    def __call__(
        self,
        freqs,
        inc,
        lam,
        beta,
        psi,
        tRef_wave_frame,
        tRef_sampling_frame,
        phiRef,
        f_ref,
        tBase,
        length,
        modes=None,
        includes_amps=False,
        phases=None,
        phases_deriv=None,
        need_flatten=False,
        xp=None,
        out_buffer=None,
    ):

        if xp is not None:
            self.xp = xp

        if modes is not None:
            ells = self.xp.asarray([ell for ell, mm in modes])
            mms = ells = self.xp.asarray([mm for ell, mm in modes])

            self._sanity_check_modes(ells, mms)

        else:
            ells = self.ells_default
            mms = self.mms_default

        num_modes = len(ells)
        num_bin_all = len(inc)

        self.length = length
        self.num_modes = num_modes
        self.num_bin_all = num_bin_all
        self.num_per_param = length * num_modes * num_bin_all
        self.num_per_bin = length * num_modes

        self.nresponse_params = 6
        self.nparams = includes_amps + 2 + self.nresponse_params

        inc = self.xp.asarray(inc)
        lam = self.xp.asarray(lam)
        beta = self.xp.asarray(beta)
        psi = self.xp.asarray(psi)
        tRef_wave_frame = self.xp.asarray(tRef_wave_frame)
        tRef_sampling_frame = self.xp.asarray(tRef_sampling_frame)
        phiRef = self.xp.asarray(phiRef)
        f_ref = self.xp.asarray(f_ref)

        if out_buffer is None:
            includes_amps = 0
            self._initialize_response_container()

        else:
            includes_amps = includes_amps
            self.response_carrier = out_buffer

        self.includes_amps = includes_amps

        self.freqs = freqs

        if phases is not None and phases_deriv is not None:

            first = phases if not need_flatten else phases.flatten()
            second = phases_deriv if not need_flatten else phases_deriv.flatten()

            out_buffer[
                (includes_amps + 0)
                * length
                * num_modes
                * num_bin_all : (includes_amps + 1)
                * length
                * num_modes
                * num_bin_all
            ] = first

            out_buffer[
                (includes_amps + 1)
                * length
                * num_modes
                * num_bin_all : (includes_amps + 2)
                * length
                * num_modes
                * num_bin_all
            ] = second

        elif phases is not None or phases_deriv is not None:
            raise ValueError(
                "If provided phases or phases_deriv, need to provide both."
            )

        LISA_response_wrap(
            self.response_carrier,
            ells,
            mms,
            self.freqs,
            phiRef,
            f_ref,
            inc,
            lam,
            beta,
            psi,
            tRef_wave_frame,
            tRef_sampling_frame,
            tBase,
            self.TDItag_int,
            self.order_fresnel_stencil,
            num_modes,
            length,
            num_bin_all,
            includes_amps,
        )


class CubicSplineInterpolant:
    """GPU-accelerated Multiple Cubic Splines

    This class produces multiple cubic splines on a GPU. It has a CPU option
    as well. The cubic splines are produced with "not-a-knot" boundary
    conditions.

    This class can be run on GPUs and CPUs.

    args:
        t (1D double xp.ndarray): t values as input for the spline.
        y_all (2D double xp.ndarray): y values for the spline.
            Shape: (ninterps, length).
        use_gpu (bool, optional): If True, prepare arrays for a GPU. Default is
            False.

    """

    def __init__(
        self, x, y_all, length, num_interp_params, num_modes, num_bin_all, use_gpu=False
    ):

        if use_gpu:
            self.xp = xp
            self.interpolate_arrays = interpolate_wrap

        else:
            self.xp = np
            self.interpolate_arrays = interpolate_wrap

        ninterps = num_modes * num_interp_params * num_bin_all
        self.degree = 3

        self.reshape_shape = (num_interp_params, length, num_modes, num_bin_all)

        B = self.xp.zeros((ninterps * length,))
        self.c1 = upper_diag = self.xp.zeros_like(B)
        self.c2 = diag = self.xp.zeros_like(B)
        self.c3 = lower_diag = self.xp.zeros_like(B)
        self.y = y_all

        self.interpolate_arrays(
            x,
            y_all,
            B,
            upper_diag,
            diag,
            lower_diag,
            length,
            num_interp_params,
            num_modes,
            num_bin_all,
        )

        self.x = x

    @property
    def y_shaped(self):
        return self.xp.transpose(self.y.reshape(self.reshape_shape), axes=(0, 3, 2, 1))

    @property
    def c1_shaped(self):
        return self.xp.transpose(self.c1.reshape(self.reshape_shape), axes=(0, 3, 2, 1))

    @property
    def c2_shaped(self):
        return self.xp.transpose(self.c2.reshape(self.reshape_shape), axes=(0, 3, 2, 1))

    @property
    def c3_shaped(self):
        return self.xp.transpose(self.c3.reshape(self.reshape_shape), axes=(0, 3, 2, 1))

    @property
    def container(self):
        return [self.x, self.y, self.c1, self.c2, self.c3]


class TemplateInterp:
    def __init__(self, max_init_len=-1, use_gpu=False):

        if use_gpu:
            self.xp = xp

        else:
            self.xp = np

        if max_init_len > 0:
            self.use_buffers = True
            raise NotImplementedError

        else:
            self.use_buffers = False

    def _initialize_template_container(self):

        self.template_carrier = self.xp.zeros(
            (self.data_length * self.num_bin_all * self.nChannels),
            dtype=self.xp.complex128,
        )

    @property
    def template_channels(self):
        return self.xp.transpose(
            self.template_carrier.reshape(
                self.data_length, self.nChannels, self.num_bin_all
            ),
            axes=(1, 2, 0),
        )

    def __call__(
        self,
        dataChannels,
        dataFreqs,
        interp_container,
        tBase,
        tRef_sampling_frame,
        tRef_wave_frame,
        length,
        data_length,
        num_modes,
        t_obs_start,
        t_obs_end,
        nChannels,
    ):

        numBinAll = len(tRef_sampling_frame)
        self.length = length
        self.num_modes = num_modes
        self.num_bin_all = numBinAll
        self.data_length = data_length
        self.nChannels = nChannels

        self._initialize_template_container()

        (freqs, y, c1, c2, c3) = interp_container

        tRef_sampling_frame = self.xp.asarray(tRef_sampling_frame)
        tRef_wave_frame = self.xp.asarray(tRef_wave_frame)

        InterpTDI_wrap(
            self.template_carrier,
            dataChannels,
            dataFreqs,
            freqs,
            y,
            c1,
            c2,
            c3,
            tBase,
            tRef_sampling_frame,
            tRef_wave_frame,
            self.length,
            self.data_length,
            self.num_bin_all,
            self.num_modes,
            t_obs_start,
            t_obs_end,
        )


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

    nChannels = 3
    data_length = 2 ** 15

    num_interp_params = 9
    num_modes = 6

    phenomhm = PhenomHMAmpPhase(use_gpu=use_gpu)
    response = LISATDIResponse(
        max_init_len=-1, TDItag="AET", order_fresnel_stencil=0, use_gpu=use_gpu
    )

    df = 1.0 / YRSID_SI

    f_n = xp.arange(1e-3, 1e-1 + df * 100, df * 100)

    data_length = len(f_n)
    out_buffer = xp.zeros((num_interp_params * data_length * num_modes * 1))

    test_phen = phenomhm(
        m1[:1],
        m2[:1],
        chi1z[:1],
        chi2z[:1],
        distance[:1],
        phiRef[:1],
        f_ref[:1],
        len(f_n),
        freqs=f_n,
        out_buffer=out_buffer,
    )

    minF = phenomhm.freqs_shaped[0].min()
    maxF = phenomhm.freqs_shaped[0].max()

    test_resp = response(
        f_n,
        inc[:1],
        lam[:1],
        beta[:1],
        psi[:1],
        tRef_wave_frame[:1],
        tRef_sampling_frame[:1],
        phiRef[:1],
        f_ref[:1],
        tBase,
        len(f_n),
        includes_amps=True,
        out_buffer=out_buffer,
    )

    data_info = out_buffer.reshape(num_interp_params, data_length, num_modes, 1)

    amp = data_info[0]
    phase = data_info[1]
    phase_deriv = data_info[2]
    transfer_L1 = data_info[3] + 1j * data_info[4]
    transfer_L2 = data_info[5] + 1j * data_info[6]
    transfer_L3 = data_info[7] + 1j * data_info[8]

    d = xp.sum(
        xp.asarray(
            [
                amp * xp.exp(-1j * phase) * transfer_L1,
                amp * xp.exp(-1j * phase) * transfer_L2,
                amp * xp.exp(-1j * phase) * transfer_L3,
            ]
        ),
        axis=2,
    ).squeeze()

    m1 *= 1.00001

    test_phen = phenomhm(
        m1[:1],
        m2[:1],
        chi1z[:1],
        chi2z[:1],
        distance[:1],
        phiRef[:1],
        f_ref[:1],
        len(f_n),
        freqs=f_n,
        out_buffer=out_buffer,
    )

    test_resp = response(
        f_n,
        inc[:1],
        lam[:1],
        beta[:1],
        psi[:1],
        tRef_wave_frame[:1],
        tRef_sampling_frame[:1],
        phiRef[:1],
        f_ref[:1],
        tBase,
        len(f_n),
        includes_amps=True,
        out_buffer=out_buffer,
    )

    data_info = out_buffer.reshape(num_interp_params, data_length, num_modes, 1)

    amp = data_info[0]
    phase = data_info[1]
    phase_deriv = data_info[2]
    transfer_L1 = data_info[3] + 1j * data_info[4]
    transfer_L2 = data_info[5] + 1j * data_info[6]
    transfer_L3 = data_info[7] + 1j * data_info[8]

    h0 = xp.sum(
        xp.asarray(
            [
                amp * xp.exp(-1j * phase) * transfer_L1,
                amp * xp.exp(-1j * phase) * transfer_L2,
                amp * xp.exp(-1j * phase) * transfer_L3,
            ]
        ),
        axis=2,
    ).squeeze()

    from lisatools.diagnostic import inner_product

    freqs = xp.logspace(xp.log10(minF), np.log10(maxF), length)

    bins = xp.searchsorted(freqs, f_n, "right") - 1

    f_m = (freqs[1:] + freqs[:-1]) / 2

    df = f_n[1] - f_n[0]

    S_n = xp.asarray(get_sensitivity(f_n.get(), sens_fn="noisepsd_AE"))

    A0_flat = 4 * (h0.conj() * d) / S_n * df
    A1_flat = 4 * (h0.conj() * d) / S_n * df * (f_n - f_m[bins])

    B0_flat = 4 * (h0.conj() * h0) / S_n * df
    B1_flat = 4 * (h0.conj() * h0) / S_n * df * (f_n - f_m[bins])

    A0 = xp.zeros((3, length - 1), dtype=np.complex128)
    A1 = xp.zeros_like(A0)
    B0 = xp.zeros_like(A0)
    B1 = xp.zeros_like(A0)

    for ind in xp.unique(bins[:-1]):
        inds_keep = bins == ind

        inds_keep[-1] = False

        A0[:, ind] = xp.sum(A0_flat[:, inds_keep], axis=1)
        A1[:, ind] = xp.sum(A1_flat[:, inds_keep], axis=1)
        B0[:, ind] = xp.sum(B0_flat[:, inds_keep], axis=1)
        B1[:, ind] = xp.sum(B1_flat[:, inds_keep], axis=1)

    d_d = xp.sum(4 * (d.conj() * d) / S_n * df).real

    h_h = xp.sum(B0_flat).real

    d_h = xp.sum(A0_flat).real

    d_h_2 = xp.sum(
        xp.asarray(
            [inner_product(d[i], h0[i], f_arr=f_n, PSD="noisepsd_AE") for i in range(3)]
        )
    )
    d_d_2 = xp.sum(
        xp.asarray(
            [inner_product(d[i], d[i], f_arr=f_n, PSD="noisepsd_AE") for i in range(3)]
        )
    )
    h_h_2 = xp.sum(
        xp.asarray(
            [
                inner_product(h0[i], h0[i], f_arr=f_n, PSD="noisepsd_AE")
                for i in range(3)
            ]
        )
    )

    base_ll = 1 / 2 * (d_d + h_h - 2 * d_h)

    num_bin_all = len(m1)

    out_buffer = xp.zeros((num_interp_params * length * num_modes * num_bin_all))

    freqs = xp.logspace(xp.log10(minF), np.log10(maxF), length)

    phenomhm(
        m1[:1],
        m2[:1],
        chi1z[:1],
        chi2z[:1],
        distance[:1],
        phiRef[:1],
        f_ref[:1],
        length,
        freqs=freqs,
        out_buffer=out_buffer,
    )

    response(
        freqs,
        inc[:1],
        lam[:1],
        beta[:1],
        psi[:1],
        tRef_wave_frame[:1],
        tRef_sampling_frame[:1],
        phiRef[:1],
        f_ref[:1],
        tBase,
        length,
        includes_amps=True,
        out_buffer=out_buffer,
    )

    data_info = out_buffer.reshape(num_interp_params, length, num_modes, num_bin_all)

    amp = data_info[0]
    phase = data_info[1]
    phase_deriv = data_info[2]
    transfer_L1 = data_info[3] + 1j * data_info[4]
    transfer_L2 = data_info[5] + 1j * data_info[6]
    transfer_L3 = data_info[7] + 1j * data_info[8]

    h0_short = xp.sum(
        xp.asarray(
            [
                amp * xp.exp(-1j * phase) * transfer_L1,
                amp * xp.exp(-1j * phase) * transfer_L2,
                amp * xp.exp(-1j * phase) * transfer_L3,
            ]
        ),
        axis=2,
    ).squeeze()

    m1 *= 1.0001
    m2 *= 1.001
    phenomhm(
        m1,
        m2,
        chi1z,
        chi2z,
        distance,
        phiRef,
        f_ref,
        length,
        freqs=freqs,
        out_buffer=out_buffer,
    )

    response(
        freqs,
        inc,
        lam,
        beta,
        psi,
        tRef_wave_frame,
        tRef_sampling_frame,
        phiRef,
        f_ref,
        tBase,
        length,
        includes_amps=True,
        out_buffer=out_buffer,
    )

    data_info = out_buffer.reshape(num_interp_params, length, num_modes, num_bin_all)

    amp = data_info[0]
    phase = data_info[1]
    phase_deriv = data_info[2]
    transfer_L1 = data_info[3] + 1j * data_info[4]
    transfer_L2 = data_info[5] + 1j * data_info[6]
    transfer_L3 = data_info[7] + 1j * data_info[8]

    h_short = xp.sum(
        xp.asarray(
            [
                amp * xp.exp(-1j * phase) * transfer_L1,
                amp * xp.exp(-1j * phase) * transfer_L2,
                amp * xp.exp(-1j * phase) * transfer_L3,
            ]
        ),
        axis=2,
    )

    r = h_short / h0_short[:, :, xp.newaxis]

    r1 = (r[:, 1:] - r[:, :-1]) / (
        freqs[1:][xp.newaxis, :, xp.newaxis] - freqs[:-1][xp.newaxis, :, xp.newaxis]
    )

    r0 = r[:, :-1] - r1 * (
        freqs[:-1][xp.newaxis, :, xp.newaxis] - f_m[xp.newaxis, :, xp.newaxis]
    )

    Z_d_h = xp.sum(
        (xp.conj(r0) * A0[:, :, xp.newaxis] + xp.conj(r1) * A1[:, :, xp.newaxis]),
        axis=(0, 1),
    )

    Z_h_h = xp.sum(
        (
            B0[:, :, xp.newaxis] * np.abs(r0) ** 2
            + 2 * B1[:, :, xp.newaxis] * xp.real(r0 * xp.conj(r1))
        ),
        axis=(0, 1),
    )

    out_buffer = xp.zeros((num_interp_params * data_length * num_modes * 1))

    test_phen = phenomhm(
        m1[:1],
        m2[:1],
        chi1z[:1],
        chi2z[:1],
        distance[:1],
        phiRef[:1],
        f_ref[:1],
        len(f_n),
        freqs=f_n,
        out_buffer=out_buffer,
    )

    test_resp = response(
        f_n,
        inc[:1],
        lam[:1],
        beta[:1],
        psi[:1],
        tRef_wave_frame[:1],
        tRef_sampling_frame[:1],
        phiRef[:1],
        f_ref[:1],
        tBase,
        len(f_n),
        includes_amps=True,
        out_buffer=out_buffer,
    )

    data_info = out_buffer.reshape(num_interp_params, data_length, num_modes, 1)

    amp = data_info[0]
    phase = data_info[1]
    phase_deriv = data_info[2]
    transfer_L1 = data_info[3] + 1j * data_info[4]
    transfer_L2 = data_info[5] + 1j * data_info[6]
    transfer_L3 = data_info[7] + 1j * data_info[8]

    h_test = xp.sum(
        xp.asarray(
            [
                amp * xp.exp(-1j * phase) * transfer_L1,
                amp * xp.exp(-1j * phase) * transfer_L2,
                amp * xp.exp(-1j * phase) * transfer_L3,
            ]
        ),
        axis=2,
    ).squeeze()

    d_h_test = xp.sum(4 * (h_test.conj() * d) / S_n * df).real

    h_h_test = xp.sum(4 * (h_test.conj() * h_test) / S_n * df).real

    ll_res = 1 / 2 * (d_d + Z_h_h - 2 * Z_d_h)
    ll_test = 1 / 2 * (d_d + h_h_test - 2 * d_h_test)
    breakpoint()

    """
    spline = CubicSplineInterpolant(
        freqs,
        out_buffer,
        length,
        num_interp_params,
        num_modes,
        num_bin_all,
        use_gpu=use_gpu,
    )

    template_gen = TemplateInterp(max_init_len=-1, use_gpu=use_gpu)

    template_gen(
        dataChannels,
        dataFreqs,
        spline.container,
        tBase,
        tRef_sampling_frame,
        tRef_wave_frame,
        length,
        data_length,
        num_modes,
        t_obs_start,
        t_obs_end,
        nChannels,
    )
    """


if __name__ == "__main__":

    num_bin_all = 3
    length = 128

    m1 = np.array([4.000000e6])  # , 4.01e6, 4.0001e6])
    m2 = np.full_like(m1, 1e6)
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

    for _ in range(10):
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
