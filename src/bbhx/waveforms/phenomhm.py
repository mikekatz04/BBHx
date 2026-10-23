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

"""PhenomHM / PhenomD frequency-domain waveforms in the amplitude-phase representation.

:class:`PhenomHMAmpPhase` evaluates, per binary, per harmonic and per frequency, the
amplitude, the phase and the time-frequency correspondence t(f) of PhenomHM (or PhenomD,
the (2,2) mode alone) with the native ``waveform_amp_phase_wrap`` kernel. It is the source
stage of :class:`bbhx.waveformbuild.BBHWaveformFD`. The PhenomD ringdown / damping
frequencies come from the QNM tables of :mod:`bbhx.waveforms.ringdownphenomd`.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from ..utils.constants import *
from ..waveforms.ringdownphenomd import *
from ..utils.parallelbase import BBHxParallelModule


class PhenomHMAmpPhase(BBHxParallelModule):
    r"""Produce PhenomHM in the amplitude and phase representation.

    This class implements PhenomD and PhenomHM in a GPU-accelerated form.
    If you use this class, please cite
    `arXiv:2005.01827 <https://arxiv.org/abs/2005.01827>`_,
    `arXiv:2111.01064 <https://arxiv.org/abs/2111.01064>`_,
    `arXiv:1708.00404 <https://arxiv.org/abs/1708.00404>`_,
    `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_, and
    `arXiv:1508.07253 <https://arxiv.org/abs/1508.07253>`_.

    For each binary, mode and frequency it computes the amplitude :math:`A_{lm}(f)`, the
    phase :math:`\phi_{lm}(f)` (convention
    :math:`\tilde{h}_{lm}(f) = A_{lm}(f) e^{-i\phi_{lm}(f)}`, see the BBHx docs) and the
    time-frequency correspondence t(f) (s). They are stored flat in
    :attr:`waveform_carrier` (quantity-major, then
    ``(num_bin_all, num_modes, length)``) and read through :attr:`amp`, :attr:`phase`,
    :attr:`tf` and :attr:`freqs_shaped`.

    Args:
        force_backend (str, optional): Backend name (``"cpu"``, ``"cuda12x"``, ``"cuda"``,
            ``"gpu"``, ...), see :class:`~bbhx.utils.parallelbase.BBHxParallelModule`.
            Default ``None`` picks the first available of ``cuda13x``, ``cuda12x``,
            ``cuda11x``, ``cpu``.
        run_phenomd (bool, optional): If ``True``, run the PhenomD
            waveform rather than PhenomHM. Really this is the same
            as choosing ``modes=[(2,2)]`` in the PhenomHM waveform. Default ``False``.
        mf_min (float, optional): Lowest dimensionless frequency ``M f`` (``M`` the total
            mass in seconds) of the internal grid used when no ``freqs`` are given; mode
            ``(l, m)`` uses ``M f`` times ``m / 2``. Default ``1e-4``.
        mf_max (float, optional): Highest dimensionless frequency of that grid. Default
            ``0.6``.
        initial_t_val (float or np.ndarray, optional): Time origin of the phase: the phase
            is shifted by ``2 pi f (t_ref - initial_t_val)`` while t(f) is still shifted by
            ``t_ref``, so the response still sees the right times. A float, a
            ``(num_bin_all,)`` array or a ``(num_bin_all, num_modes)`` numpy array (an
            ``int`` is not accepted). Default ``0.0``.

    Attributes:
        allowable_modes (list): Allowed list of mode tuple pairs ``(l,m)`` for
            the chosen waveform model.
        ells_default (xp.ndarray): Default values for the ``l`` index of the harmonic.
        mms_default (xp.ndarray): Default values for the ``m`` index of the harmonic.
        mf_max (float): See Args.
        mf_min (float): See Args.
        initial_t_val (float or np.ndarray): See Args (may be reassigned between calls).
        run_phenomd (bool): See Args.
        modes (list): Modes used by the last call.
        y_rd (xp.ndarray): Y-values for PhenomD ringdown frequency for Cubic Spline.
        c1_rd (xp.ndarray): Cubic Spline c1 values for PhenomD ringdown frequency.
        c2_rd (xp.ndarray): Cubic Spline c2 values for PhenomD ringdown frequency.
        c3_rd (xp.ndarray): Cubic Spline c3 values for PhenomD ringdown frequency.
        y_dm (xp.ndarray): Y-values for PhenomD damping frequency for Cubic Spline.
        c1_dm (xp.ndarray): Cubic Spline c1 values for PhenomD damping frequency.
        c2_dm (xp.ndarray): Cubic Spline c2 values for PhenomD damping frequency.
        c3_dm (xp.ndarray): Cubic Spline c3 values for PhenomD damping frequency.
        fringdown (xp.ndarray): Ringdown frequencies of the last call (per binary and
            mode; PhenomHM appends the PhenomD (2,2) value per binary).
        fdamp (xp.ndarray): Damping frequencies of the last call, same layout.
        waveform_carrier (xp.ndarray): Flat carrier for amplitude, phase, and tf
            information (or the caller's ``out_buffer``).

    Example:
        .. code-block:: python

            from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
            from bbhx.utils.constants import PC_SI, YRSID_SI

            phenomhm = PhenomHMAmpPhase(run_phenomd=False, force_backend="cpu")
            # m1, m2 [Msun], chi1z, chi2z, distance [m], phi_ref, f_ref, t_ref [s], length
            phenomhm(1e6, 5e5, 0.2, 0.4, 18e3 * PC_SI * 1e6, 0.0, 0.0, 0.95 * YRSID_SI, 1024)
            amp, phase, tf = phenomhm.amp, phenomhm.phase, phenomhm.tf  # (1, 6, 1024)
            freqs = phenomhm.freqs_shaped  # (1, 6, 1024)
    """

    def __init__(
        self,
        force_backend=None,
        run_phenomd=False,
        mf_min=1e-4,
        mf_max=0.6,
        initial_t_val=0.0,
    ):
        super().__init__(force_backend=force_backend)

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
    def xp(self) -> object:
        """Numpy or Cupy"""
        return self.backend.xp
    
    @classmethod
    def supported_backends(cls) -> list:
        """Preference order: ``bbhx_`` + ``cuda13x``, ``cuda12x``, ``cuda11x``, ``cpu``."""
        return ["bbhx_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    @property
    def waveform_gen(self) -> callable:
        """Native ``waveform_amp_phase_wrap`` kernel (PhenomHM / PhenomD amplitude-phase)."""
        return self.backend.waveform_amp_phase_wrap

    @property
    def phenomhm_ringdown_freqs(self) -> callable:
        """C/CUDA wrapped function for computing PhenomHM Ringdown frequencies."""
        return self.backend.get_phenomhm_ringdown_frequencies

    @property
    def phenomd_ringdown_freqs(self) -> callable:
        """C/CUDA wrapped function for computing PhenomD Ringdown frequencies."""
        return self.backend.get_phenomd_ringdown_frequencies

    @property
    def citation(self):
        """Citation strings for this class.

        Warning:
            Raises ``NameError``: the citation strings (``katz_citations``, ...) are no
            longer defined in this module (``bbhx.utils.citations`` was removed).
        """
        return katz_citations + phenomhm_citation + phenomd_citations

    def _init_phenomd_fring_spline(self):
        """Prepare PhenomD fring and fdamp splines.

        Fits scipy cubic splines of ``QNMData_fring`` and ``QNMData_fdamp`` against the
        final spin ``QNMData_a`` and stores the values and coefficients
        (``y_rd, c1_rd, c2_rd, c3_rd`` and ``y_dm, c1_dm, c2_dm, c3_dm``, ``xp`` arrays) read
        by the PhenomD ringdown-frequency kernel.
        """
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
        """Raise ``ValueError`` if an ``(ell, mm)`` pair is not in :attr:`allowable_modes`."""
        for ell, mm in zip(ells, mms):
            if (ell, mm) not in self.allowable_modes:
                raise ValueError(
                    "Requested mode [(l,m) = ({},{})] is not available. Allowable modes include {}".format(
                        ell, mm, self.allowable_modes
                    )
                )

    def _sanity_check_params(self, m1, m2, chi1z, chi2z):
        """Check the spins and order the bodies so that ``m1 >= m2``.

        Raises ``ValueError`` if any ``|chi1z|`` or ``|chi2z|`` exceeds 1. Where
        ``m1 < m2`` the masses and spins are swapped IN PLACE on the given arrays.

        Returns:
            tuple: ``(m1, m2, chi1z, chi2z)`` (the same array objects).
        """
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
        """Allocate a zero :attr:`waveform_carrier` of size ``nparams * num_per_param``."""
        self.waveform_carrier = self.xp.zeros(
            (self.nparams * self.length * self.num_modes * self.num_bin_all),
            dtype=self.xp.float64,
        )

    def _initialize_freqs(self, m1, m2, mms):
        """Build the internal frequency grid when the user gives none.

        ``length`` log-spaced points in ``[mf_min, mf_max]`` divided by the total mass in
        seconds and multiplied by ``m / 2`` per mode; stored flat in :attr:`freqs`
        (layout ``(num_bin_all, num_modes, length)``).
        """
        M_tot_sec = (m1 + m2) * MTSUN_SI

        # dimensionless freqs
        base_freqs = self.xp.logspace(
            self.xp.log10(self.mf_min), self.xp.log10(self.mf_max), self.length
        )

        # adjust them for each binary Mass
        # flatten to prepare for C computations
        self.freqs = (
            base_freqs[self.xp.newaxis, self.xp.newaxis, :]
            / M_tot_sec[:, self.xp.newaxis, self.xp.newaxis]
            * (mms[self.xp.newaxis, :, self.xp.newaxis])
            / 2
        ).flatten()
        # mms / 2 sets ratio versus 22 mode

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
        """Get the freqs array with shape ``(num_bin_all, num_modes, length)``"""
        return self._freqs.reshape(self.num_bin_all, self.num_modes, self.length)

    @property
    def freqs(self):
        """Get the flat freqs array (Hz), layout ``(num_bin_all, num_modes, length)``"""
        return self._freqs

    @freqs.setter
    def freqs(self, f):
        """Store frequencies flat for C (arrays with ``ndim > 1`` are flattened)."""
        if f.ndim > 1:
            self._freqs = f.flatten()

        else:
            self._freqs = f

    def run_wave(
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
        """Generate PhenomHM/D waveforms.

        Generate PhenomHM/PhenomD waveforms based on user given quantities
        in the Amplitude-Phase representation. The kernel produces the waveform with the
        (2,2) mode at ``t(f) = 0`` and zero phase at the reference frequency; this method
        then adds
        ``2 pi f (t_ref - initial_t_val)`` to the phase and ``t_ref`` to t(f). Results are
        read through :attr:`amp`, :attr:`phase`, :attr:`tf` and :attr:`freqs_shaped`.

        Binaries whose PhenomD ringdown frequency comes out negative get
        ``fringdown = fdamp = 1e-2`` and their distance multiplied by ``1e20`` (amplitude
        effectively zero), and a ``FIX FIX FIX FREQ`` line with their parameters is
        printed. This works with a single mode only: with several modes the mask has
        ``num_modes * num_bin_all`` entries while ``distance`` and the masses have
        ``num_bin_all``, so the branch raises ``IndexError``.

        Args:
            m1 (float or np.ndarray): Mass of body 1 in solar masses. Bodies (and spins) are
                swapped in place where ``m1 < m2`` (see :meth:`_sanity_check_params`).
            m2 (float or np.ndarray): Mass of body 2 in solar masses.
            chi1z (float or np.ndarray): Dimensionless aligned spin of body 1,
                ``|chi1z| <= 1``.
            chi2z (float or np.ndarray): Dimensionless aligned spin of body 2,
                ``|chi2z| <= 1``.
            distance (float or np.ndarray): Luminosity distance in metres.
            phi_ref (float or np.ndarray): Accepted but NOT used: the kernel runs with zero
                reference phase (:class:`~bbhx.waveformbuild.BBHWaveformFD` applies
                ``phi_ref`` in the response).
            f_ref (float or np.ndarray): Reference frequency (Hz). Passed to the kernel but
                currently IGNORED: the kernel always uses the peak frequency of the PhenomD
                (2,2) merger-ringdown amplitude ansatz (``fmaxCalc``, Eq. 20 of
                arXiv:1508.07253).
            t_ref (float or np.ndarray): Time (s) at which the (2,2) mode reaches that
                reference frequency (t(f) of every mode is shifted by ``t_ref``).
            length (int): Number of frequencies per mode.
            freqs (xp.ndarray, optional): Evaluation frequencies (Hz): 1D ``(length,)``
                (tiled to every binary and mode) or 3D ``(num_bin_all, num_modes, length)``.
                Default ``None`` builds the internal grid from ``mf_min`` / ``mf_max``.
            out_buffer (xp.ndarray, optional): Flat buffer to fill instead of a new
                :attr:`waveform_carrier`, layout ``(nparams, num_bin_all, num_modes,
                length)``; the first 3 slots (amplitude, phase, t(f)) are filled. Use
                ``nparams = 9`` to share it with
                :class:`~bbhx.response.fastfdresponse.LISATDIResponse`. Its size is not
                checked. Default ``None``.
            modes (list, optional): ``(l, m)`` tuples, a subset of
                :attr:`allowable_modes`. Default ``None``: PhenomHM
                ``[(2,2), (3,3), (4,4), (2,1), (3,2), (4,3)]``, PhenomD ``[(2,2)]``
                (``run_phenomd=True`` always uses ``[(2,2)]``).

        Raises:
            ValueError: ``|chi| > 1``, an unavailable mode, or ``freqs`` neither 1D nor 3D.
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
            self._initialize_freqs(m1, m2, mms)

        elif freqs.ndim == 1:
            self.freqs = self.xp.tile(
                freqs, (self.num_bin_all, self.num_modes, 1)
            ).flatten()
        elif freqs.ndim == 3:
            self.freqs = freqs.flatten().copy()
        else:
            raise ValueError(
                "If providing freqs, must be 1D array (length,) or 3D array (length, num_modes, length)."
            )

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
            len(self.y_rd)
        )

        # deal with issues in frequency determination
        fix_ring_f = self.fringdown < 0.0

        if self.xp.any(fix_ring_f):
            self.fringdown[fix_ring_f] = 1e-2
            self.fdamp[fix_ring_f] = 1e-2
            distance[fix_ring_f] *= 1e20

            print(
                "FIX FIX FIX FREQ:", 
                m1[fix_ring_f],
                m2[fix_ring_f],
                chi1z[fix_ring_f],
                chi2z[fix_ring_f]
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

        if isinstance(self.initial_t_val, float):
            _initial_t_val = np.array([self.initial_t_val])[:, None, None]
        elif self.initial_t_val.ndim == 1:
            _initial_t_val = self.initial_t_val[:, None, None]
        elif self.initial_t_val.ndim == 2:
            _initial_t_val = self.initial_t_val[:, :, None]

        # adjust phases based on shift from t_ref
        # do this inplace
        temp = (
            self.freqs.reshape(self.num_bin_all, self.num_modes, -1)
            * self.xp.asarray(
                t_ref[:, self.xp.newaxis, self.xp.newaxis] - _initial_t_val
            )
            * 2
            * np.pi
        )
        
        # phases = self.waveform_carrier[1]  (waveform carrier is flat)
        self.waveform_carrier[
            1 * self.num_per_param : 2 * self.num_per_param
        ] += temp.flatten()

        # adjust t-f for shift of t_ref
        # t_ref array = self.waveform_carrier[2] (waveform carrier is flat)
        self.waveform_carrier[
            2 * self.num_per_param : 3 * self.num_per_param
        ] += self.xp.tile(
            self.xp.asarray(t_ref)[:, None, None], (1, self.num_modes, self.length)
        ).flatten()

    def __call__(self, *args, t_orbit_end=None, direct=False, **kwargs):
        """Generate the waveform, restricting the frequency band to the orbit-covered times.

        Runs :meth:`run_wave`. With ``direct=False`` it then finds, per binary and mode, the
        frequencies where t(f) crosses ``500`` s and ``t_orbit_end - 500`` s (keeping the
        band edges where t(f) stays inside), and runs :meth:`run_wave` again on
        ``length`` log-spaced frequencies between them (a 3D ``freqs`` grid). This keeps
        the response inside the orbit data.

        Args:
            *args: Positional arguments of :meth:`run_wave`
                (``m1, m2, chi1z, chi2z, distance, phi_ref, f_ref, t_ref, length``).
            t_orbit_end (float or np.ndarray, optional): End time (s) of the orbit data: a
                float, a numpy ``(num_bin_all,)`` array or a numpy
                ``(num_bin_all, num_modes)`` array. Default ``None`` means no upper cut.
            direct (bool, optional): If ``True``, run :meth:`run_wave` once on the given
                frequencies, without band adjustment. Default ``False``.
            **kwargs: Keyword arguments of :meth:`run_wave` (``freqs``, ``out_buffer``,
                ``modes``).

        Raises:
            ValueError: The t(f) crossing is not found exactly once per binary and mode
                (``t_ref`` too close to 0 or to ``t_orbit_end``), or ``t_orbit_end`` has an
                unsupported type.
        """

        self.run_wave(*args, **kwargs)

        # interpolating. Need to adjust frequency bounds
        if not direct:
            # TODO: must check if this is accurate enough!!!!!!!

            # PURPOSE: NEED TO REMOVE FREQUENCIES WHERE ORBIT INFO DOES NOT EXIST
            # get start of available information
            buffer = 500.0
            fix = self.tf.min(axis=-1) < buffer
            f_min = self.freqs_shaped.min(axis=-1)

            if self.xp.any(fix):
                windows_of_interest = self.xp.argwhere(
                    (buffer < self.tf[fix][:, 1:]) & (buffer > self.tf[fix][:, :-1])
                )

                if windows_of_interest.shape[0] != fix.sum():
                    raise ValueError(
                        "The search for a window at x_min did not run correctly. If t_ref is close to zero, the time-frequency correspondence may be crossing zero twice."
                    )

                # linear interp for minimum frequency
                segment_inds = windows_of_interest[:, 1]

                y1 = self.xp.take_along_axis(
                    self.freqs_shaped[fix], segment_inds[:, None], axis=-1
                )[:, 0]
                y2 = self.xp.take_along_axis(
                    self.freqs_shaped[fix], segment_inds[:, None] + 1, axis=-1
                )[:, 0]
                x1 = self.xp.take_along_axis(
                    self.tf[fix], segment_inds[:, None], axis=-1
                )[:, 0]
                x2 = self.xp.take_along_axis(
                    self.tf[fix], segment_inds[:, None] + 1, axis=-1
                )[:, 0]

                m = (y2 - y1) / (x2 - x1)
                b = y1

                f_min_new = m * (buffer - x1) + b
                f_min[fix] = f_min_new

            if t_orbit_end is None:
                # will default to tf bound
                t_orbit_end = self.xp.full((self.num_bin_all, self.num_modes), 1e300)
            elif isinstance(t_orbit_end, float):
                t_orbit_end = self.xp.full((self.num_bin_all, self.num_modes), t_orbit_end)
            elif isinstance(t_orbit_end, np.ndarray) and t_orbit_end.ndim == 1:
                assert t_orbit_end.shape[0] == self.num_bin_all
                t_orbit_end = self.xp.repeat(
                    self.xp.asarray(t_orbit_end)[:, None], self.num_modes, axis=-1
                )
            elif isinstance(t_orbit_end, np.ndarray) and t_orbit_end.ndim == 2:
                t_orbit_end = self.xp.asarray(t_orbit_end)
                assert t_orbit_end.shape == (self.num_bin_all, self.num_modes)
            else:
                raise ValueError("t_orbit_end entered incorrectly.")

            fix = self.tf.max(axis=-1) > t_orbit_end - buffer
            f_max = self.freqs_shaped.max(axis=-1)

            if self.xp.any(fix):
                windows_of_interest = self.xp.argwhere(
                    (t_orbit_end[fix][:, None] - buffer < self.tf[fix, 1:])
                    & (t_orbit_end[fix][:, None] - buffer > self.tf[fix, :-1])
                )

                if windows_of_interest.shape[0] != fix.sum():
                    raise ValueError(
                        "The search for a window at x_max did not run correctly. If t_ref is close to the observation time, the time-frequency correspondence may be crossing zero twice."
                    )

                # linear interp for minimum frequency
                segment_inds = windows_of_interest[:, 1]

                y1 = self.xp.take_along_axis(
                    self.freqs_shaped[fix], segment_inds[:, None], axis=-1
                )[:, 0]
                y2 = self.xp.take_along_axis(
                    self.freqs_shaped[fix], segment_inds[:, None] + 1, axis=-1
                )[:, 0]
                x1 = self.xp.take_along_axis(
                    self.tf[fix], segment_inds[:, None], axis=-1
                )[:, 0]
                x2 = self.xp.take_along_axis(
                    self.tf[fix], segment_inds[:, None] + 1, axis=-1
                )[:, 0]

                m = (y2 - y1) / (x2 - x1)
                b = y1

                f_max_new = m * ((t_orbit_end[fix] - buffer) - x1) + b

                f_max[fix] = f_max_new

            new_freqs = self.xp.logspace(
                self.xp.log10(f_min), self.xp.log10(f_max), self.length, axis=-1
            )

            tmp_kwargs = kwargs.copy()
            tmp_kwargs["freqs"] = new_freqs
            self.run_wave(*args, **tmp_kwargs)
