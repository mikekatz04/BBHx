# Wraps codes for PhenomHM originally from
# arXiv:1708.00404
# arXiv:1508.07250
# arXiv:1508.07253
# but now adjusted for GPU usage.

# Copyright (C) 2021 Michael L. Katz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
from scipy.interpolate import CubicSpline

# import GPU stuff
try:
    from pyPhenomHM import waveform_amp_phase_wrap as waveform_amp_phase_wrap_gpu
    from pyPhenomHM import (
        get_phenomhm_ringdown_frequencies as get_phenomhm_ringdown_frequencies_gpu,
    )
    from pyPhenomHM import (
        get_phenomd_ringdown_frequencies as get_phenomd_ringdown_frequencies_gpu,
    )
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    print("No CuPy or GPU PhenomHM module.")

from pyPhenomHM_cpu import waveform_amp_phase_wrap as waveform_amp_phase_wrap_cpu
from pyPhenomHM_cpu import (
    get_phenomhm_ringdown_frequencies as get_phenomhm_ringdown_frequencies_cpu,
)
from pyPhenomHM_cpu import (
    get_phenomd_ringdown_frequencies as get_phenomd_ringdown_frequencies_cpu,
)

from ..utils.constants import *
from ..waveforms.ringdownphenomd import *


class PhenomHMAmpPhase:
    """Produce PhenomHM in the amplitude and phase representation

    This class implements PhenomD and PhenomHM in a GPU-accelerated form.
    If you use this class, please cite
    `arXiv:2005.01827 <https://arxiv.org/abs/2005.01827>`_,
    `arXiv:2111.01064 <https://arxiv.org/abs/2111.01064>`_
    `arXiv:1708.00404 <https://arxiv.org/abs/1708.00404>`_,
    `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_, and
    `arXiv:1508.07253 <https://arxiv.org/abs/1508.07253>_.

    Args:
        use_gpu (bool, optional): If ``True``, run on the GPU.
        run_phenomd (bool, optional): If ``True``, run the PhenomD
            waveform rather than PhenomHM. Really this is the same
            as choosing ``modes=[(2,2)]`` in the PhenomHM waveform.
        mf_min (double, optional): Dimensionless minimum frequency to use when performing
            interpolation. (Default: ``1e-4``)
        mf_max (double, optional): Dimensionless maximum frequency to use when performing
            interpolation. (Default: ``6e-1``)
        initial_t_val (double, optional): Time at the start of the
            time window. This shifts the phase accordingly but does
            not shift the tf correspondence so that the response
            is still accurately reflected. (Default: ``0.0``)

    Attributes:
        allowable_modes (list): Allowed list of mode tuple pairs ``(l,m)`` for
            the chosen waveform model.
        ells_default (np.ndarray): Default values for the ``l`` index of the harmonic.
        mms_default (np.ndarray): Default values for the ``m`` index of the harmonic.
        mf_max (double): Dimensionless maximum frequency to use when performing
            interpolation.
        mf_min (double): Dimensionless minimum frequency to use when performing
            interpolation.
        phenomhm_ringdown_freqs (obj): Ringdown frequency determination in PhenomHM.
        phenomd_ringdown_freqs (obj): Ringdown frequency determination in PhenomD.
        run_phenomd (bool): If ``True``, run the PhenomD
            waveform rather than PhenomHM. Really this is the same
            as choosing ``modes=[(2,2)]`` in the PhenomHM waveform.
        y_rd (xp.ndarray): Y-values for PhenomD ringdown frequncy for Cubic Spline.
        c1_rd (xp.ndarray): Cubic Spline c1 values for PhenomD ringdown frequency.
        c2_rd (xp.ndarray): Cubic Spline c2 values for PhenomD ringdown frequency.
        c3_rd (xp.ndarray): Cubic Spline c3 values for PhenomD ringdown frequency.
        y_dm (xp.ndarray): Y-values for PhenomD damping frequncy for Cubic Spline.
        c1_dm (xp.ndarray): Cubic Spline c1 values for PhenomD damping frequency.
        c2_dm (xp.ndarray): Cubic Spline c2 values for PhenomD damping frequency.
        c3_dm (xp.ndarray): Cubic Spline c3 values for PhenomD damping frequency.
        waveform_carrier (xp.ndarray): Carrier for amplitude, phase, and tf information.

    """

    def __init__(
        self,
        use_gpu=False,
        run_phenomd=False,
        mf_min=1e-4,
        mf_max=0.6,
        initial_t_val=0.0,
    ):
        self.use_gpu = use_gpu

        self.run_phenomd = run_phenomd

        self.allowable_modes = [(2, 2), (3, 3), (4, 4), (2, 1), (3, 2), (4, 3)]

        self.ells_default = self.xp.array([2, 3, 4, 2, 3, 4], dtype=self.xp.int32)

        self.mms_default = self.xp.array([2, 3, 4, 1, 2, 3], dtype=self.xp.int32)

        # fix allowable quantities if you are running phenomd
        if self.run_phenomd:
            self.allowable_modes = [(2, 2)]

            self.ells_default = self.xp.array([2], dtype=self.xp.int32)

            self.mms_default = self.xp.array([2], dtype=self.xp.int32)

        # max/min dimensionless frequencies evaluated when
        # freqs are not given, but a length is given
        self.mf_min = mf_min
        self.mf_max = mf_max

        self.initial_t_val = initial_t_val

        # prepare the PhenomD spline info for fRD and fDM
        self._init_phenomd_fring_spline()

    @property
    def use_gpu(self) -> bool:
        """Whether to use a GPU."""
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, use_gpu: bool) -> None:
        """Set ``use_gpu``."""
        assert isinstance(use_gpu, bool)
        self._use_gpu = use_gpu

    @property
    def xp(self) -> object:
        """Numpy or Cupy"""
        return cp if self.use_gpu else np

    @property
    def waveform_gen(self) -> callable:
        """C/CUDA wrapped function for computing interpolation."""
        return (
            waveform_amp_phase_wrap_gpu if self.use_gpu else waveform_amp_phase_wrap_cpu
        )

    @property
    def phenomhm_ringdown_freqs(self) -> callable:
        """C/CUDA wrapped function for computing PhenomHM Ringdown frequencies."""
        return (
            get_phenomhm_ringdown_frequencies_gpu
            if self.use_gpu
            else get_phenomhm_ringdown_frequencies_cpu
        )

    @property
    def phenomd_ringdown_freqs(self) -> callable:
        """C/CUDA wrapped function for computing PhenomD Ringdown frequencies."""
        return (
            get_phenomd_ringdown_frequencies_gpu
            if self.use_gpu
            else get_phenomd_ringdown_frequencies_cpu
        )

    @property
    def citation(self):
        """Return citations for this class"""
        return katz_citations + phenomhm_citation + phenomd_citations

    def _init_phenomd_fring_spline(self):
        """Prepare PhenomD fring and fdamp splines"""
        # setup splines
        spl_ring = CubicSpline(QNMData_a, QNMData_fring)
        spl_damp = CubicSpline(QNMData_a, QNMData_fdamp)

        # store the coefficients
        self.y_rd = self.xp.asarray(QNMData_fring).copy()
        self.c1_rd = self.xp.asarray(spl_ring.c[-2]).copy()
        self.c2_rd = self.xp.asarray(spl_ring.c[-3]).copy()
        self.c3_rd = self.xp.asarray(spl_ring.c[-4]).copy()

        self.y_dm = self.xp.asarray(QNMData_fdamp).copy()
        self.c1_dm = self.xp.asarray(spl_damp.c[-2]).copy()
        self.c2_dm = self.xp.asarray(spl_damp.c[-3]).copy()
        self.c3_dm = self.xp.asarray(spl_damp.c[-4]).copy()

    def _sanity_check_modes(self, ells, mms):
        """Make sure ell and mm combinations are available"""
        for ell, mm in zip(ells, mms):
            if (ell, mm) not in self.allowable_modes:
                raise ValueError(
                    "Requested mode [(l,m) = ({},{})] is not available. Allowable modes include {}".format(
                        ell, mm, self.allowable_modes
                    )
                )

    def _sanity_check_params(self, m1, m2, chi1z, chi2z):
        """Make sure input parameters are okay"""
        if np.any(np.abs(chi1z) > 1.0):
            raise ValueError(
                "chi1z array contains values with abs(chi1z) > 1.0 which is not allowed."
            )

        if np.any(np.abs(chi2z) > 1.0):
            raise ValueError(
                "chi2z array contains values with abs(chi2z) > 1.0 which is not allowed."
            )

        # ensure m1 > m2
        switch = m1 < m2

        m1_temp = m2[switch]
        m2[switch] = m1[switch]
        m1[switch] = m1_temp

        # adjust chi values if masses are switched
        chi1z_temp = chi2z[switch]
        chi2z[switch] = chi1z[switch]
        chi1z[switch] = chi1z_temp

        return (m1, m2, chi1z, chi2z)

    def _initialize_waveform_container(self):
        """Initialize the waveform container based on input dimensions"""
        self.waveform_carrier = self.xp.zeros(
            (self.nparams * self.length * self.num_modes * self.num_bin_all),
            dtype=self.xp.float64,
        )

    def _initialize_freqs(self, m1, m2):
        """Setup frequencies when not given by user"""
        M_tot_sec = (m1 + m2) * MTSUN_SI

        # dimensionless freqs
        base_freqs = self.xp.logspace(
            self.xp.log10(self.mf_min), self.xp.log10(self.mf_max), self.length
        )

        # adjust them for each binary Mass
        # flatten to prepare for C computations
        self.freqs = (
            base_freqs[:, self.xp.newaxis] / M_tot_sec[self.xp.newaxis, :]
        ).T.flatten()

    @property
    def amp(self):
        """Get the amplitude array with shape ``(num_bin_all, num_modes, length)``"""
        amps = self.waveform_carrier[
            0 * self.num_per_param : 1 * self.num_per_param
        ].reshape(self.num_bin_all, self.num_modes, self.length)
        return amps

    @property
    def phase(self):
        """Get the phase array with shape ``(num_bin_all, num_modes, length)``"""
        phase = self.waveform_carrier[
            1 * self.num_per_param : 2 * self.num_per_param
        ].reshape(self.num_bin_all, self.num_modes, self.length)
        return phase

    @property
    def tf(self):
        """Get the tf array with shape ``(num_bin_all, num_modes, length)``"""
        tf = self.waveform_carrier[
            2 * self.num_per_param : 3 * self.num_per_param
        ].reshape(self.num_bin_all, self.num_modes, self.length)
        return tf

    @property
    def freqs_shaped(self):
        """Get the freqs array with shape ``(num_bin_all, length)``"""
        return self._freqs.reshape(self.num_bin_all, self.length)

    @property
    def freqs(self):
        """Get the flat freqs array"""
        return self._freqs

    @freqs.setter
    def freqs(self, f):
        """Prepare frequencies properly for C"""
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
        phi_ref,
        f_ref,
        t_ref,
        length,
        freqs=None,
        out_buffer=None,
        modes=None,
    ):
        """Generate PhenomHM/D waveforms

        Generate PhenomHM/PhenomD waveforms based on user given quantitites
        in the Amplitude-Phase representation.

        Args:
            m1 (double scalar or np.ndarray): Mass 1 in Solar Masses :math:`(m1 > m2)`.
            m2 (double or np.ndarray): Mass 2 in Solar Masses :math:`(m1 > m2)`.
            chi1z (double or np.ndarray): Dimensionless spin 1 (for Mass 1) in Solar Masses.
            chi2z (double or np.ndarray): Dimensionless spin 2 (for Mass 1) in Solar Masses.
            distance (double or np.ndarray): Luminosity distance in m.
            phi_ref (double or np.ndarray): Phase at ``f_ref``.
            f_ref (double or np.ndarray): Reference frequency at which ``phi_ref`` and ``t_ref`` are set.
                If ``f_ref == 0``, it will be set internally by the PhenomHM code
                to :math:`f_\\text{max} = \\text{max}(f^2A_{22}(f))`.
            t_ref (double or np.ndarray): Reference time in seconds. It is set at ``f_ref``.
            length (int): Length of the frequency array over which the waveform is created.
            freqs (1D or 2D xp.ndarray, optional): If ``None``, the class will generate the
                frequency array over which the waveform is evaluated. If 1D xp.ndarray,
                this array will be copied for all binaries evaluated. If 2D,
                it must have shape ``(num_bin_all, length)``. (Default: ``None``)
            out_buffer (xp.ndarray, optional): If ``None``, a buffer array will be created.
                If provided, it should be flattened from shape
                ``(nparams, length, num_modes, num_bin_all)``. ``nparams`` can
                be 3 if just evaluating PhenomHM/D. If using the same buffer for
                the response it must be 9. (Default: ``None``)
            modes (list, optional): Harmonic modes to use. If not given, they will
                default to those available in the waveform model. For PhenomHM:
                [(2,2), (3,3), (4,4), (2,1), (3,2), (4,3)]. For PhenomD: [(2,2)].
                (Default: ``None``)

        """

        # cast to 1D arrays if scalars
        m1 = np.atleast_1d(m1)
        m2 = np.atleast_1d(m2)
        chi1z = np.atleast_1d(chi1z)
        chi2z = np.atleast_1d(chi2z)
        distance = np.atleast_1d(distance)
        phi_ref = np.atleast_1d(phi_ref)
        f_ref = np.atleast_1d(f_ref)
        t_ref = np.atleast_1d(t_ref)

        # make sure parameters are okay and ordered so m1 > m2
        m1, m2, chi1z, chi2z = self._sanity_check_params(m1, m2, chi1z, chi2z)

        # set modes
        if modes is not None:
            ells = self.xp.asarray([ell for ell, mm in modes], dtype=self.xp.int32)
            mms = self.xp.asarray([mm for ell, mm in modes], dtype=self.xp.int32)
            self.modes = modes
            self._sanity_check_modes(ells, mms)

        else:
            self.modes = self.allowable_modes
            ells = self.ells_default
            mms = self.mms_default

        # adjust for phenomD
        if self.run_phenomd:
            ells = self.xp.asarray([2], dtype=self.xp.int32)
            mms = self.xp.asarray([2], dtype=self.xp.int32)
            self.modes = self.allowable_modes

        num_modes = len(ells)
        num_bin_all = len(m1)

        # store necessary parameters
        # here we evaluate 3 parameters: amp, phase, tf
        self.length = length
        self.num_modes = num_modes
        self.num_bin_all = num_bin_all
        self.nparams = 3
        self.num_per_param = length * num_modes * num_bin_all
        self.num_per_bin = length * num_modes

        # cast to GPU if needed
        m1 = self.xp.asarray(m1).copy()
        m2 = self.xp.asarray(m2).copy()
        chi1z = self.xp.asarray(chi1z).copy()
        chi2z = self.xp.asarray(chi2z).copy()
        distance = self.xp.asarray(distance).copy()
        phi_ref = self.xp.asarray(phi_ref).copy()
        f_ref = self.xp.asarray(f_ref).copy()

        # setup out_buffer if not given
        if out_buffer is None:
            self._initialize_waveform_container()

        else:
            # TODO: add sanity checks for buffer size like response
            self.waveform_carrier = out_buffer

        # initialize frequencies if not given
        if freqs is None:
            self._initialize_freqs(m1, m2)

        elif freqs.ndim == 1:
            self.freqs = self.xp.tile(freqs, (self.num_bin_all, 1)).flatten()
        else:
            self.freqs = freqs.flatten().copy()

        # convert to SI units for the mass
        m1_SI = m1 * MSUN_SI
        m2_SI = m2 * MSUN_SI

        # prepare for phenomD fring and fdamp
        self.fringdown = self.xp.zeros(self.num_modes * self.num_bin_all)
        self.fdamp = self.xp.zeros(self.num_modes * self.num_bin_all)

        # get phenomD freq info
        self.phenomd_ringdown_freqs(
            self.fringdown,
            self.fdamp,
            m1,
            m2,
            chi1z,
            chi2z,
            self.num_bin_all,
            self.y_rd,
            self.c1_rd,
            self.c2_rd,
            self.c3_rd,
            self.y_dm,
            self.c1_dm,
            self.c2_dm,
            self.c3_dm,
            dspin,
        )

        if not self.run_phenomd:
            # move phenomD results to the last entry in the array after
            # phenomhm frequencies
            append_phenomd_frd = self.fringdown[: self.num_bin_all].copy()
            append_phenomd_fdm = self.fdamp[: self.num_bin_all].copy()

            # get phenomhm frequencies
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

            # this adds the phenomD frequencies to keep everything consistent
            self.fringdown = (
                self.xp.concatenate(
                    [
                        self.fringdown.reshape(-1, num_modes),
                        self.xp.array([append_phenomd_frd]).T,
                    ],
                    axis=1,
                )
                .flatten()
                .copy()
            )
            self.fdamp = (
                self.xp.concatenate(
                    [
                        self.fdamp.reshape(-1, num_modes),
                        self.xp.array([append_phenomd_fdm]).T,
                    ],
                    axis=1,
                )
                .flatten()
                .copy()
            )

        # inside this code, t_ref is zero and phi_ref is zero
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
            f_ref,
            num_modes,
            length,
            num_bin_all,
            self.fringdown,
            self.fdamp,
            self.run_phenomd,
        )

        # adjust phases based on shift from t_ref
        # do this inplace
        temp = (
            self.freqs.reshape(self.num_bin_all, -1)
            * self.xp.asarray(t_ref[:, self.xp.newaxis] - self.initial_t_val)
            * 2
            * np.pi
        )

        # phases = self.waveform_carrier[1]  (waveform carrier is flat)
        self.waveform_carrier[
            1 * self.num_per_param : 2 * self.num_per_param
        ] += self.xp.tile(temp[:, None, :], (1, self.num_modes, 1)).flatten()

        # adjust t-f for shift of t_ref
        # t_ref array = self.waveform_carrier[2] (waveform carrier is flat)
        self.waveform_carrier[
            2 * self.num_per_param : 3 * self.num_per_param
        ] += self.xp.tile(
            self.xp.asarray(t_ref)[:, None, None], (1, self.num_modes, self.length)
        ).flatten()
