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
        ).T.flatten()

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
        return self._freqs.reshape(self.num_bin_all, self.length)

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
            self.freqs = self.xp.tile(freqs, (self.num_bin_all, 1)).flatten()

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
            ells = self.xp.asarray([ell for ell, mm in modes], dtype=self.xp.int32)
            mms = ells = self.xp.asarray([mm for ell, mm in modes], dtype=self.xp.int32)

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

        self.length = length

        self.reshape_shape = (num_interp_params, num_bin_all, num_modes, length)

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

        # TODO: need to fix last point
        self.x = x

    @property
    def y_shaped(self):
        return self.y.reshape(self.reshape_shape)

    @property
    def c1_shaped(self):
        return self.c1.reshape(self.reshape_shape)

    @property
    def c2_shaped(self):
        return self.c2.reshape(self.reshape_shape)

    @property
    def c3_shaped(self):
        return self.c3.reshape(self.reshape_shape)

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
            (self.num_bin_all * self.nChannels * self.data_length),
            dtype=self.xp.complex128,
        )

    @property
    def template_channels(self):
        return [
            self.template_carrier[i].reshape(self.nChannels, self.lengths[i])
            for i in range(self.num_bin_all)
        ]

    def __call__(
        self,
        dataFreqs,
        interp_container,
        t_mrg,
        t_start,
        t_end,
        length,
        data_length,
        num_modes,
        t_obs_start,
        t_obs_end,
        nChannels,
    ):

        numBinAll = len(t_mrg)
        self.length = length
        self.num_modes = num_modes
        self.num_bin_all = numBinAll
        self.data_length = data_length
        self.nChannels = nChannels

        # self._initialize_template_container()

        (freqs, y, c1, c2, c3) = interp_container

        freqs_shaped = freqs.reshape(self.num_bin_all, -1)

        start_and_end = self.xp.asarray([freqs_shaped[:, 0], freqs_shaped[:, -1],]).T

        inds_start_and_end = self.xp.asarray(
            [
                self.xp.searchsorted(dataFreqs, temp, side="right")
                for temp in start_and_end
            ]
        )

        inds = [
            self.xp.searchsorted(
                freqs_shaped[i], dataFreqs[st:et], side="right"
            ).astype(self.xp.int32)
            - 1
            for i, (st, et) in enumerate(inds_start_and_end)
        ]

        self.lengths = lengths = np.asarray(
            [len(inds_i) for inds_i in inds], dtype=self.xp.int32
        )

        self.start_inds = start_inds = (inds_start_and_end[:, 0].get().copy()).astype(
            np.int32
        )

        self.ptrs = ptrs = np.asarray([ind_i.data.ptr for ind_i in inds])

        self.template_carrier = [
            self.xp.zeros(int(self.nChannels * temp_length), dtype=self.xp.complex128,)
            for temp_length in lengths
        ]

        template_carrier_ptrs = np.asarray(
            [temp_carrier.data.ptr for temp_carrier in self.template_carrier]
        )

        dlog10f = 1.0

        InterpTDI_wrap(
            template_carrier_ptrs,
            dataFreqs,
            dlog10f,
            freqs,
            y,
            c1,
            c2,
            c3,
            t_mrg,
            t_start,
            t_end,
            self.length,
            self.data_length,
            self.num_bin_all,
            self.num_modes,
            t_obs_start,
            t_obs_end,
            ptrs,
            start_inds,
            lengths,
        )

        return self.template_channels


class BBHWaveform:
    def __init__(
        self, amp_phase_kwargs={}, response_kwargs={}, interp_kwargs={}, use_gpu=False
    ):

        self.response_gen = LISATDIResponse(**response_kwargs, use_gpu=use_gpu)

        self.amp_phase_gen = PhenomHMAmpPhase(**amp_phase_kwargs, use_gpu=use_gpu)

        self.interp_tdi = TemplateInterp(**interp_kwargs, use_gpu=use_gpu)

        self.use_gpu = use_gpu
        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

        self.num_interp_params = 9

    def __call__(
        self,
        m1,
        m2,
        chi1z,
        chi2z,
        distance,
        phiRef,
        f_Ref,
        inc,
        lam,
        beta,
        psi,
        tRef_wave_frame,
        tRef_sampling_frame,
        tBase=0.0,
        t_obs_start=1.0,
        t_obs_end=0.0,
        freqs=None,
        length=None,
        modes=None,
        direct=True,
        compress=True,
        squeeze=True,
        shift_t_limits=False,
        fill=False,
    ):

        # TODO: if t_obs_end = t_mrg

        m1 = np.atleast_1d(m1)
        m2 = np.atleast_1d(m2)
        chi1z = np.atleast_1d(chi1z)
        chi2z = np.atleast_1d(chi2z)
        distance = np.atleast_1d(distance)
        phiRef = np.atleast_1d(phiRef)
        inc = np.atleast_1d(inc)
        lam = np.atleast_1d(lam)
        beta = np.atleast_1d(beta)
        psi = np.atleast_1d(psi)
        tRef_wave_frame = np.atleast_1d(tRef_wave_frame)
        tRef_sampling_frame = np.atleast_1d(tRef_sampling_frame)

        t_mrg = tRef_sampling_frame + tBase * YRSID_SI

        if shift_t_limits is False:
            t_start = tRef_sampling_frame + tBase * YRSID_SI - t_obs_start * YRSID_SI
            t_end = (
                tRef_sampling_frame + tBase * YRSID_SI - t_obs_end * YRSID_SI
                if t_obs_end > 0.0
                else np.zeros_like(t_start)
            )

        else:
            t_start = np.atleast_1d(t_obs_start)
            t_end = np.atleast_1d(t_obs_end)

        self.num_bin_all = len(m1)

        if freqs is None and length is None:
            raise ValueError("Must input freqs or length.")

        elif freqs is not None and direct is True:
            length = len(freqs)

        elif direct is False:
            self.data_length = len(freqs)

        self.length = length

        if modes is None:
            self.num_modes = len(self.amp_phase_gen.allowable_modes)
        else:
            self.num_modes = len(modes)

        self.num_bin_all = len(m1)

        out_buffer = self.xp.zeros(
            (self.num_interp_params * self.length * self.num_modes * self.num_bin_all)
        )

        freqs_temp = freqs if direct else None

        self.amp_phase_gen(
            m1,
            m2,
            chi1z,
            chi2z,
            distance,
            phiRef,
            f_ref,
            length,
            freqs=freqs_temp,
            out_buffer=out_buffer,
        )

        self.response_gen(
            self.amp_phase_gen.freqs,
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

        if direct and compress:

            templateChannels = self.xp.zeros(
                (self.num_bin_all * 3 * self.length), dtype=self.xp.complex128
            )

            direct_sum_wrap(
                templateChannels,
                out_buffer,
                self.num_bin_all,
                self.length,
                3,
                self.num_modes,
                self.xp.asarray(t_start),
                self.xp.asarray(t_end),
            )

            out = templateChannels.reshape(self.num_bin_all, 3, self.length).transpose(
                (1, 2, 0)
            )

            if squeeze:
                out = out.squeeze()

            return out

        if direct:
            temp = self.xp.swapaxes(
                out_buffer.reshape(
                    self.num_interp_params,
                    self.num_bin_all,
                    self.num_modes,
                    self.length,
                ),
                1,
                3,
            )

            amp = temp[0]
            phase = temp[1]
            # tf = temp[2].get()
            transfer_L1 = temp[3] + 1j * temp[4]
            transfer_L2 = temp[5] + 1j * temp[6]
            transfer_L3 = temp[7] + 1j * temp[8]

            # TODO: check this combination
            # TODO: produce combination as same in CUDA
            amp_phase = amp * self.xp.exp(-1j * phase)

            # if t_obs_end <= 0.0:
            #    test = np.full_like(amp_phase.get(), True)

            # else:
            #    test = np.full_like(amp_phase.get(), False)

            # temp2 = ((tf <= t_end[0]) + (test)).astype(bool)
            # inds = (tf >= t_start[0]) & temp2 & (amp.get() > 1e-40)

            out = self.xp.asarray(
                [
                    amp_phase * transfer_L1,
                    amp_phase * transfer_L2,
                    amp_phase * transfer_L3,
                ]
            )

            # out[:, ~inds] = 0.0

            if compress:
                out = out.sum(axis=2)

            if squeeze:
                out = out.squeeze()

            return out

        else:

            spline = CubicSplineInterpolant(
                self.amp_phase_gen.freqs,
                out_buffer,
                self.length,
                self.num_interp_params,
                self.num_modes,
                self.num_bin_all,
                use_gpu=use_gpu,
            )

            # TODO: try single block reduction for likelihood (will probably be worse for smaller batch, but maybe better for larger batch)?
            template_channels = self.interp_tdi(
                freqs,
                spline.container,
                t_mrg,
                t_start,
                t_end,
                self.length,
                self.data_length,
                self.num_modes,
                t_obs_start,
                t_obs_end,
                3,
            )

            if fill:
                data_out = self.xp.zeros((3, len(freqs)), dtype=self.xp.complex128)
                for temp, start_i, length_i in zip(
                    template_channels,
                    self.interp_tdi.start_inds,
                    self.interp_tdi.lengths,
                ):
                    data_out[:, start_i : start_i + length_i] = temp

                return data_out
            else:

                return (
                    template_channels,
                    self.interp_tdi.start_inds,
                    self.interp_tdi.lengths,
                )


class Likelihood:
    def __init__(
        self, waveform_model, dataFreqs, dataChannels, noiseFactors, use_gpu=False
    ):

        self.use_gpu = use_gpu

        if use_gpu:
            self.xp = xp

        else:
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

    def __call__(self, params, **waveform_kwargs):

        templateChannels, inds_start, ind_lengths = self.waveform_gen(
            *params, **waveform_kwargs
        )

        templateChannels = [tc.flatten() for tc in templateChannels]

        templateChannels_ptrs = np.asarray(
            [tc.data.ptr for tc in templateChannels], dtype=np.int64
        )

        d_h = np.zeros(self.waveform_gen.num_bin_all)
        h_h = np.zeros(self.waveform_gen.num_bin_all)

        # TODO: if filling multiple signals into stream, need to adjust this for that in terms of inds start / ind_lengths
        direct_like_wrap(
            d_h,
            h_h,
            self.dataChannels,
            self.noiseFactors,
            templateChannels_ptrs,
            inds_start,
            ind_lengths,
            self.data_stream_length,
            self.waveform_gen.num_bin_all,
        )

        return 1 / 2 * (self.d_d + h_h - 2 * d_h)


class RelativeBinning:
    def __init__(
        self,
        template_gen,
        f_dense,
        d,
        template_gen_args,
        length_f_rel,
        template_gen_kwargs={},
        use_gpu=False,
    ):

        self.template_gen = template_gen
        self.f_dense = f_dense
        self.d = d
        self.length_f_rel = length_f_rel

        self.use_gpu = use_gpu
        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

        self._init_rel_bin_info(template_gen_args, template_gen_kwargs)

    def _init_rel_bin_info(self, template_gen_args, template_gen_kwargs={}):

        template_gen_kwargs["squeeze"] = True
        template_gen_kwargs["compress"] = True
        template_gen_kwargs["direct"] = True

        minF = self.f_dense.min()
        maxF = self.f_dense.max()

        freqs = xp.logspace(xp.log10(minF), np.log10(maxF), self.length_f_rel)

        h0 = self.template_gen(
            *template_gen_args, freqs=self.f_dense, **template_gen_kwargs
        )

        h0_temp = self.template_gen(
            *template_gen_args, freqs=freqs, **template_gen_kwargs
        )[0]

        freqs_keep = freqs[~(self.xp.abs(h0_temp) == 0.0)]

        freqs = xp.logspace(
            xp.log10(freqs_keep[0]), np.log10(freqs_keep[-1]), self.length_f_rel
        )

        self.h0_short = self.template_gen(
            *template_gen_args, freqs=freqs, **template_gen_kwargs
        )[:, :, xp.newaxis]

        bins = xp.searchsorted(freqs, self.f_dense, "right") - 1

        f_m = (freqs[1:] + freqs[:-1]) / 2

        df = self.f_dense[1] - self.f_dense[0]

        # TODO: make adjustable
        S_n = xp.asarray(get_sensitivity(self.f_dense.get(), sens_fn="noisepsd_AE"))

        A0_flat = 4 * (h0.conj() * self.d) / S_n * df
        A1_flat = 4 * (h0.conj() * self.d) / S_n * df * (self.f_dense - f_m[bins])

        B0_flat = 4 * (h0.conj() * h0) / S_n * df
        B1_flat = 4 * (h0.conj() * h0) / S_n * df * (self.f_dense - f_m[bins])

        self.A0 = xp.zeros((3, self.length_f_rel - 1), dtype=np.complex128)
        self.A1 = xp.zeros_like(self.A0)
        self.B0 = xp.zeros_like(self.A0)
        self.B1 = xp.zeros_like(self.A0)

        A0_in = xp.zeros((3, self.length_f_rel), dtype=np.complex128)
        A1_in = xp.zeros_like(A0_in)
        B0_in = xp.zeros_like(A0_in)
        B1_in = xp.zeros_like(A0_in)

        for ind in xp.unique(bins[:-1]):
            inds_keep = bins == ind

            # TODO: check this
            inds_keep[-1] = False

            self.A0[:, ind] = xp.sum(A0_flat[:, inds_keep], axis=1)
            self.A1[:, ind] = xp.sum(A1_flat[:, inds_keep], axis=1)
            self.B0[:, ind] = xp.sum(B0_flat[:, inds_keep], axis=1)
            self.B1[:, ind] = xp.sum(B1_flat[:, inds_keep], axis=1)

            A0_in[:, ind + 1] = xp.sum(A0_flat[:, inds_keep], axis=1)
            A1_in[:, ind + 1] = xp.sum(A1_flat[:, inds_keep], axis=1)
            B0_in[:, ind + 1] = xp.sum(B0_flat[:, inds_keep], axis=1)
            B1_in[:, ind + 1] = xp.sum(B1_flat[:, inds_keep], axis=1)

        # PAD As with a zero in the front
        self.dataConstants = self.xp.concatenate(
            [A0_in.flatten(), A1_in.flatten(), B0_in.flatten(), B1_in.flatten()]
        )

        self.base_d_d = xp.sum(4 * (self.d.conj() * self.d) / S_n * df).real

        self.base_h_h = xp.sum(B0_flat).real

        self.base_d_h = xp.sum(A0_flat).real

        self.base_ll = 1 / 2 * (self.base_d_d + self.base_h_h - 2 * self.base_d_h)

        self.freqs = freqs
        self.f_m = f_m

    def __call__(self, params, **waveform_kwargs):

        waveform_kwargs["direct"] = True
        waveform_kwargs["compress"] = True
        waveform_kwargs["squeeze"] = False

        waveform_kwargs["freqs"] = self.freqs

        self.h_short = self.template_gen(*params, **waveform_kwargs)

        r = self.h_short / self.h0_short

        """
        r1 = (r[:, 1:] - r[:, :-1]) / (
            self.freqs[1:][xp.newaxis, :, xp.newaxis]
            - self.freqs[:-1][xp.newaxis, :, xp.newaxis]
        )

        r0 = r[:, :-1] - r1 * (
            self.freqs[:-1][xp.newaxis, :, xp.newaxis]
            - self.f_m[xp.newaxis, :, xp.newaxis]
        )

        self.Z_d_h = xp.sum(
            (
                xp.conj(r0) * self.A0[:, :, xp.newaxis]
                + xp.conj(r1) * self.A1[:, :, xp.newaxis]
            ),
            axis=(0, 1),
        )

        self.Z_h_h = xp.sum(
            (
                self.B0[:, :, xp.newaxis] * np.abs(r0) ** 2
                + 2 * self.B1[:, :, xp.newaxis] * xp.real(r0 * xp.conj(r1))
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

        templates_in = r.transpose((1, 0, 2)).flatten()

        hdyn_wrap(
            self.hdyn_d_h,
            self.hdyn_h_h,
            templates_in,
            self.dataConstants,
            self.freqs,
            self.template_gen.num_bin_all,
            len(self.freqs),
            3,
        )

        like = 1 / 2.0 * (self.base_d_d + self.hdyn_h_h - 2 * self.hdyn_d_h)
        return like


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
    num_bin_all = len(m1)

    amp_phase_kwargs = dict()
    response_kwargs = dict(max_init_len=-1, TDItag="AET", order_fresnel_stencil=0)

    phenomhm = PhenomHMAmpPhase(use_gpu=use_gpu, **amp_phase_kwargs)
    response = LISATDIResponse(use_gpu=use_gpu, **response_kwargs)

    bbh = BBHWaveform(
        response_kwargs=response_kwargs,
        amp_phase_kwargs=amp_phase_kwargs,
        use_gpu=use_gpu,
    )

    df = 1.0 / YRSID_SI

    f_n = xp.arange(1e-6, 1e-1 + df, df)

    S_n = xp.asarray(
        [
            get_sensitivity(f_n.get(), sens_fn="noisepsd_AE"),
            get_sensitivity(f_n.get(), sens_fn="noisepsd_AE"),
            get_sensitivity(f_n.get(), sens_fn="noisepsd_T"),
        ]
    )

    data_length = len(f_n)

    import time

    st = time.perf_counter()

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
        length=1024,
        modes=None,
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

    for _ in range(num):
        ll = like(
            [
                m1[:numBinAll],
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
            length=1024,
            modes=None,
            direct=False,
            compress=True,
            fill=False,
        )

    et = time.perf_counter()

    print((et - st) / num / numBinAll)

    d = data.reshape(3, -1)
    m1 *= 1.00001

    template_gen_args = (
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
    )

    template_gen_kwargs = dict(
        tBase=tBase,
        t_obs_start=t_obs_start,
        t_obs_end=t_obs_end,
        length=None,
        modes=None,
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
                m1,
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
            modes=None,
            direct=True,
            compress=True,
            squeeze=False,
        )
    et = time.perf_counter()
    print((et - st) / num / num_bin_all, num, num_bin_all)

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

    num_bin_all = 100
    length = 256

    m1 = np.full(num_bin_all, 4.000000e6)
    # m1[1:] += np.random.randn(num_bin_all - 1) * 100
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
