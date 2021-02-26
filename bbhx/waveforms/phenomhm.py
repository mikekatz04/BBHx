import numpy as np

try:
    from pyPhenomHM import waveform_amp_phase_wrap as waveform_amp_phase_wrap_gpu
    from pyPhenomHM import (
        get_phenomhm_ringdown_frequencies as get_phenomhm_ringdown_frequencies_gpu,
    )
    import cupy as xp

except (ImportError, ModuleNotFoundError) as e:
    print("No CuPy or GPU PhenomHM module.")
    import numpy as xp

from pyPhenomHM_cpu import waveform_amp_phase_wrap as waveform_amp_phase_wrap_cpu
from pyPhenomHM_cpu import (
    get_phenomhm_ringdown_frequencies as get_phenomhm_ringdown_frequencies_cpu,
)
from bbhx.utils.constants import *


class PhenomHMAmpPhase:
    def __init__(self, max_init_len=-1, use_gpu=False):

        if use_gpu:
            self.xp = xp
            self.waveform_gen = waveform_amp_phase_wrap_gpu
            self.phenomhm_ringdown_freqs = get_phenomhm_ringdown_frequencies_gpu

        else:
            self.xp = np
            self.waveform_gen = waveform_amp_phase_wrap_cpu
            self.phenomhm_ringdown_freqs = get_phenomhm_ringdown_frequencies_cpu

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
        run_phenomd=False,
    ):

        if modes is not None:
            ells = self.xp.asarray([ell for ell, mm in modes], dtype=self.xp.int32)
            mms = self.xp.asarray([mm for ell, mm in modes], dtype=self.xp.int32)

            self._sanity_check_modes(ells, mms)

        else:
            ells = self.ells_default
            mms = self.mms_default

        if run_phenomd:
            ells = self.xp.asarray([2], dtype=self.xp.int32)
            mms = self.xp.asarray([2], dtype=self.xp.int32)

        num_modes = len(ells)
        num_bin_all = len(m1)

        self.length = length
        self.num_modes = num_modes
        self.num_bin_all = num_bin_all
        self.nparams = 3
        self.num_per_param = length * num_modes * num_bin_all
        self.num_per_bin = length * num_modes

        m1 = self.xp.asarray(m1).copy()
        m2 = self.xp.asarray(m2).copy()
        chi1z = self.xp.asarray(chi1z).copy()
        chi2z = self.xp.asarray(chi2z).copy()
        distance = self.xp.asarray(distance).copy()
        phiRef = self.xp.asarray(phiRef).copy()
        f_ref = self.xp.asarray(f_ref).copy()

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

        self.fringdown = self.xp.zeros(self.num_modes * self.num_bin_all)
        self.fdamp = self.xp.zeros(self.num_modes * self.num_bin_all)

        self.phenomhm_ringdown_freqs(
            self.fringdown,
            self.fdamp,
            m1,
            m2,
            chi1z,
            chi2z,
            ells,
            mms,
            self.num_modes,
            self.num_bin_all,
        )

        self.waveform_gen(
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
            self.fringdown,
            self.fdamp,
            run_phenomd,
        )
