import numpy as np

try:
    import cupy as xp
    from pyResponse import LISA_response_wrap as LISA_response_wrap_gpu

except (ImportError, ModuleNotFoundError) as e:
    print("No CuPy or GPU response available.")
    import numpy as xp

from pyResponse_cpu import LISA_response_wrap as LISA_response_wrap_cpu
from bbhx.utils.constants import *


class LISATDIResponse:
    def __init__(
        self, max_init_len=-1, TDItag="AET", order_fresnel_stencil=0, use_gpu=False
    ):

        if use_gpu:
            self.response_gen = LISA_response_wrap_gpu
            self.xp = xp

        else:
            self.response_gen = LISA_response_wrap_cpu
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
        t_ref,
        phiRef,
        length,
        modes=None,
        includes_amps=False,
        phases=None,
        phases_deriv=None,
        need_flatten=False,
        xp=None,
        out_buffer=None,
    ):

        # TODO: prevent mode count from being different from input buffer
        if xp is not None:
            self.xp = xp

        if modes is not None:
            ells = self.xp.asarray([ell for ell, mm in modes], dtype=self.xp.int32)
            mms = self.xp.asarray([mm for ell, mm in modes], dtype=self.xp.int32)

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

        inc = self.xp.asarray(inc).copy()
        lam = self.xp.asarray(lam).copy()
        beta = self.xp.asarray(beta).copy()
        psi = self.xp.asarray(psi).copy()
        t_ref = self.xp.asarray(t_ref).copy()
        phiRef = self.xp.asarray(phiRef).copy()

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

        self.response_gen(
            self.response_carrier,
            ells,
            mms,
            self.freqs,
            phiRef,
            inc,
            lam,
            beta,
            psi,
            t_ref,
            self.TDItag_int,
            self.order_fresnel_stencil,
            num_modes,
            length,
            num_bin_all,
            includes_amps,
        )
