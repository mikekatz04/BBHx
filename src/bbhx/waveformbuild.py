# Main waveform class location

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

"""Frequency-domain MBH TDI waveform builder.

:class:`BBHWaveformFD` chains :class:`~bbhx.waveforms.phenomhm.PhenomHMAmpPhase` (per-mode
amplitude, phase and t(f) on a sparse frequency grid),
:class:`~bbhx.response.fastfdresponse.LISATDIResponse` (fast FD transfer functions) and
:class:`TemplateInterpFD` (spline interpolation onto the data frequencies) to produce TDI
channels of PhenomHM / PhenomD signals. Its templates feed
:class:`~bbhx.likelihood.Likelihood` and :class:`~bbhx.likelihood.HeterodynedLikelihood`.

The global-fit production runs import this module (through
``lisatools.sources.bbh.waveform``) but do not evaluate it: their MBH waveform is phentax
plus the LISA Analysis Tools response. Importing the module prints ``No CuPy`` when ``cupy``
is not installed.
"""

import numpy as np
import warnings

# import gpu stuff
try:
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    print("No CuPy")

from .waveforms.phenomhm import PhenomHMAmpPhase
from .response.fastfdresponse import LISATDIResponse
from .utils.transform import tSSBfromLframe, tLfromSSBframe
from gpubackendtools.interpolate import CubicSplineInterpolant
from .utils.constants import *
# from .utils.citations import *
from .utils.parallelbase import BBHxParallelModule

class TemplateInterpFD(BBHxParallelModule):
    """Interpolate sparse waveform + response splines onto the data frequencies.

    Wraps the native ``InterpTDI_wrap`` kernel. Given the cubic-spline values and
    coefficients (from :class:`gpubackendtools.interpolate.CubicSplineInterpolant`) of the
    nine per-mode quantities on each binary's sparse frequency grid (amplitude, phase,
    t(f), and the real / imaginary parts of the three transfer functions), it evaluates the
    TDI channels at every data frequency inside that binary's band, sums the modes, and zeros
    the frequencies whose t(f) falls outside ``[t_start, t_end]``. Used by
    :class:`BBHWaveformFD` when ``direct=False``. The attributes below are set by
    :meth:`__call__`.

    This class has GPU capabilities.

    Args:
        force_backend (str, optional): Backend name (``"cpu"``, ``"cuda12x"``, ``"cuda"``,
            ``"gpu"``, ...), see :class:`~bbhx.utils.parallelbase.BBHxParallelModule`.
            Default ``None`` picks the first available of ``cuda13x``, ``cuda12x``,
            ``cuda11x``, ``cpu``.

    Attributes:
        data_length (int): Number of data frequencies interpolated onto.
        length (int): Length of each sparse frequency grid.
        num_bin_all (int): Number of binaries.
        num_channels (int): Number of TDI channels.
        num_modes (int): Number of harmonics.
        lengths (np.ndarray): ``int32``, shape ``(num_bin_all,)``: number of data
            frequencies inside each binary's band.
        start_inds (np.ndarray): ``int32``, shape ``(num_bin_all,)``: index into the data
            frequencies of the first frequency of each binary's band.
        template_carrier (list[xp.ndarray]): One flat ``complex128`` array of size
            ``num_channels * lengths[i]`` per binary; reshaped views are returned by
            :attr:`template_channels`.
    """

    def __init__(self, force_backend=None):
        super().__init__(force_backend=force_backend)

    @property
    def xp(self) -> object:
        """Numpy or Cupy"""
        return self.backend.xp
    
    @classmethod
    def supported_backends(cls) -> list:
        """Preference order: ``bbhx_`` + ``cuda13x``, ``cuda12x``, ``cuda11x``, ``cpu``."""
        return ["bbhx_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    @property
    def template_gen(self) -> callable:
        """Native ``InterpTDI_wrap`` kernel that interpolates and sums the templates."""
        return self.backend.InterpTDI_wrap

    @property
    def template_channels(self):
        """Per-binary templates, ``template_carrier[i]`` shaped ``(num_channels, lengths[i])``."""
        return [
            self.template_carrier[i].reshape(self.num_channels, self.lengths[i])
            for i in range(self.num_bin_all)
        ]

    @property
    def citation(self):
        """Citation strings for this class.

        Warning:
            Raises ``NameError``: ``katz_citations`` is no longer defined in this module (the
            ``bbhx.utils.citations`` import is commented out).
        """
        return katz_citations

    def __call__(
        self,
        data_freqs,
        interp_container,
        t_start,
        t_end,
        length,
        num_modes,
        num_channels,
    ):
        """Generate frequency domain template via interpolation.

        Takes all waveform and response information as sparse spline arrays and
        interpolates them onto ``data_freqs``. Each binary's band runs from the lowest to
        the highest sparse frequency over all its modes; the per-binary index windows are
        passed to C++/CUDA as arrays of pointers.

        Args:
            data_freqs (xp.ndarray): Data frequencies (Hz), 1D and increasing. Must be an
                ``xp`` array of this backend on GPU.
            interp_container (list): ``[freqs, y, c1, c2, c3]``: ``freqs`` is the flat
                sparse grid of shape ``(num_bin_all * num_modes * length,)``; ``y``, ``c1``,
                ``c2``, ``c3`` are the flat spline values and coefficients of the 9
                quantities, ordered ``(param, binary, mode, frequency)`` with ``param`` =
                amplitude, phase, t(f), transferL1 re / im, transferL2 re / im,
                transferL3 re / im.
            t_start (xp.ndarray): Per-binary start time (s); frequencies whose t(f) is
                earlier are zeroed.
            t_end (xp.ndarray): Per-binary end time (s); frequencies whose t(f) is later
                are zeroed (a value ``<= 0`` disables this cut).
            length (int): Length of each sparse frequency grid.
            num_modes (int): Number of harmonics.
            num_channels (int): Number of TDI channels.

        Returns:
            list[xp.ndarray]: :attr:`template_channels`: entry ``i`` is ``complex128`` of
            shape ``(num_channels, lengths[i])`` and holds binary ``i`` on
            ``data_freqs[start_inds[i]:start_inds[i] + lengths[i]]``.

        Raises:
            ValueError: On a GPU backend, when ``data_freqs`` is not a ``cupy`` array.
        """

        # fill important quantities
        num_bin_all = len(t_start)
        self.length = length
        self.num_modes = num_modes
        self.num_bin_all = num_bin_all
        self.data_length = len(data_freqs)
        self.num_channels = num_channels

        # unpack interp_container
        (freqs, y, c1, c2, c3) = interp_container

        freqs_shaped = freqs.reshape(self.num_bin_all, self.num_modes, -1)

        # find where each binary's signal starts and ends in the data array

        # lowest frequencies in all the modes. Highest frequencies in all the modes.
        start_and_end = self.xp.asarray(
            [
                freqs_shaped.min(axis=(1, 2)),
                freqs_shaped.max(axis=(1, 2)),
            ]
        ).T

        if self.backend.name != "bbhx_cpu" and not isinstance(data_freqs, self.xp.ndarray):
            raise ValueError(
                "Make sure if using Cupy or Numpy, the input freqs array is of the same type."
            )

        inds_start_and_end = self.xp.asarray(
            [
                self.xp.searchsorted(data_freqs, temp, side="right")
                for temp in start_and_end
            ]
        )

        # find proper interpolation window for each point in data stream
        inds = [
            np.concatenate(
                [
                    self.xp.searchsorted(
                        freqs_shaped[i, j], data_freqs[st:et], side="right"
                    )
                    - 1
                    for j in range(self.num_modes)
                ]
            )
            .astype(self.xp.int32)
            .copy()
            for i, (st, et) in enumerate(inds_start_and_end)
        ]

        # lengths of the signals in frequency domain
        self.lengths = lengths = np.asarray(
            [int(len(inds_i) / self.num_modes) for inds_i in inds], dtype=self.xp.int32
        )

        # make sure have this quantity available on CPU
        try:
            temp_inds = inds_start_and_end[:, 0].get()
        except AttributeError:
            temp_inds = inds_start_and_end[:, 0]

        # where to start filling waveform
        self.start_inds = start_inds = (temp_inds.copy()).astype(np.int32)

        # get pointers to all the index arrays of different length
        try:
            self.ptrs = ptrs = np.asarray([ind_i.data.ptr for ind_i in inds])
        except AttributeError:
            self.ptrs = ptrs = np.asarray(
                [ind_i.__array_interface__["data"][0] for ind_i in inds]
            )

        # initialize template information
        self.template_carrier = [
            self.xp.zeros(
                int(self.num_channels * temp_length),
                dtype=self.xp.complex128,
            )
            for temp_length in lengths
        ]

        # get pointers to template carriers so they can be run in streams
        try:
            template_carrier_ptrs = np.asarray(
                [temp_carrier.data.ptr for temp_carrier in self.template_carrier]
            )
        except AttributeError:
            template_carrier_ptrs = np.asarray(
                [
                    temp_carrier.__array_interface__["data"][0]
                    for temp_carrier in self.template_carrier
                ]
            )

        # fill templates
        self.template_gen(
            template_carrier_ptrs,
            data_freqs,
            freqs,
            y,
            c1,
            c2,
            c3,
            t_start,
            t_end,
            self.length,
            self.data_length,
            self.num_bin_all,
            self.num_modes,
            ptrs,
            start_inds,
            lengths,
        )

        # return templates in the right shape
        return self.template_channels


class BBHWaveformFD(BBHxParallelModule):
    """Generate waveforms put through response functions.

    This class generates waveforms put through the LISA response function. In the
    future, ground-based analysis may be added. Therefore, it currently
    returns the TDI variables according the response keyword arguments given.

    If you use this class, please cite `arXiv:2005.01827 <https://arxiv.org/abs/2005.01827>`_
    and `arXiv:2111.01064 <https://arxiv.org/abs/2111.01064>`_, as well as the papers
    listed for the waveform and response given just below.

    Right now, it is hard coded to produce the waveform with
    :class:`PhenomHMAmpPhase <bbhx.waveforms.phenomhm.PhenomHMAmpPhase>`. This can also be used
    to produce PhenomD. See the docs for that waveform. The papers describing PhenomHM/PhenomD
    waveforms are here: `arXiv:1708.00404 <https://arxiv.org/abs/1708.00404>`_,
    `arXiv:1508.07250 <https://arxiv.org/abs/1508.07250>`_, and
    `arXiv:1508.07253 <https://arxiv.org/abs/1508.07253>`_.

    The response function is the fast frequency domain response function
    from `arXiv:1806.10734 <https://arxiv.org/abs/1806.10734>`_ and
    `arXiv:2003.00357 <https://arxiv.org/abs/2003.00357>`_. It is implemented in
    :class:`LISATDIResponse <bbhx.response.fastfdresponse.LISATDIResponse>`.

    This class is GPU accelerated. The three sub-generators are built with the same
    ``force_backend``. It is not used by the global-fit production runs.

    Args:
        amp_phase_kwargs (dict, optional): Keyword arguments for
            :class:`PhenomHMAmpPhase <bbhx.waveforms.phenomhm.PhenomHMAmpPhase>`
            (``run_phenomd``, ``mf_min``, ``mf_max``, ``initial_t_val``). Default ``{}``.
        response_kwargs (dict, optional): Keyword arguments for
            :class:`LISATDIResponse <bbhx.response.fastfdresponse.LISATDIResponse>`
            (``TDItag``, ``orbits``, ``rescaled``, ``tdi2``, ``order_fresnel_stencil``).
            Default ``{}`` (AET, TDI 1, the response's default orbits).
        interp_kwargs (dict, optional): Keyword arguments for :class:`TemplateInterpFD`,
            which takes none besides ``force_backend``. Default ``{}``.
        force_backend (str, optional): Backend name (``"cpu"``, ``"cuda12x"``, ``"cuda"``,
            ``"gpu"``, ...), see :class:`~bbhx.utils.parallelbase.BBHxParallelModule`.
            Default ``None`` picks the first available of ``cuda13x``, ``cuda12x``,
            ``cuda11x``, ``cpu``.

    Attributes:
        amp_phase_gen (PhenomHMAmpPhase): Amplitude / phase generator.
        response_gen (LISATDIResponse): Response generator.
        interp_response (TemplateInterpFD): Interpolation stage.
        interp_like (callable or None): Native ``speciallike`` kernel used by
            :meth:`get_direct_ll` (``None`` on CPU).
        force_backend (str or None): The ``force_backend`` argument as given.
        data_length (int): Number of data frequencies (set when ``direct=False``).
        length (int): Number of frequencies of the sparse (or direct) evaluation grid.
        num_bin_all (int): Number of binaries of the last call.
        num_interp_params (int): Number of per-mode quantities in the buffer (9).
        num_modes (int): Number of harmonic modes of the last call.
        t_start (np.ndarray): Per-binary observation start, SSB time (s), of the last call.
        t_end (np.ndarray): Per-binary observation end, SSB time (s), of the last call.
        out_buffer_final (xp.ndarray): Copy of the filled buffer of the last call, shape
            ``(num_interp_params, num_bin_all, num_modes, length)``. The order of the
            quantities is amplitude, phase, t(f), transferL1re, transferL1im, transferL2re,
            transferL2im, transferL3re, transferL3im.

    Example:
        .. code-block:: python

            import numpy as np
            from lisatools.detector import ESAOrbits
            from bbhx.waveformbuild import BBHWaveformFD
            from bbhx.utils.constants import PC_SI, YRSID_SI

            wave_gen = BBHWaveformFD(
                response_kwargs=dict(orbits=ESAOrbits(force_backend="cpu")),
                force_backend="cpu",
            )
            freqs = np.logspace(-4, 0, 10000)
            # m1, m2 [Msun], chi1z, chi2z, distance [m], phi_ref, f_ref, inc,
            # lam, beta (SSB ecliptic), psi, t_ref [s]
            AET = wave_gen(
                1e6, 5e5, 0.2, 0.4, 18e3 * PC_SI * 1e6, 0.0, 0.0, np.pi / 3,
                np.pi / 5, np.pi / 4, np.pi / 6, 0.5 * YRSID_SI,
                freqs=freqs, length=1024, fill=True,
            )[0]  # (3, len(freqs)): A, E, T on ``freqs``
    """

    def __init__(
        self,
        amp_phase_kwargs={},
        response_kwargs={},
        interp_kwargs={},
        force_backend=None,
    ):
        super().__init__(force_backend=force_backend)
        self.force_backend = force_backend
        # initialize waveform and response funtions
        self.amp_phase_gen = PhenomHMAmpPhase(**amp_phase_kwargs, force_backend=force_backend)
        self.response_gen = LISATDIResponse(**response_kwargs, force_backend=force_backend)

        self.num_interp_params = 9

        # setup the final interpolant
        self.interp_like = self.backend.speciallike
        self.interp_response = TemplateInterpFD(**interp_kwargs, force_backend=force_backend)

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
        """Native ``direct_sum_wrap`` kernel (direct, non-interpolated TDI build)."""
        return self.backend.direct_sum_wrap

    @property
    def citation(self):
        """Citation strings for this class.

        Warning:
            Raises ``NameError``: the citation strings (``katz_citations``, ...) are no
            longer defined in this module (the ``bbhx.utils.citations`` import is commented
            out).
        """
        return (
            katz_citations + marsat_1 + marsat_2 + phenomhm_citation + phenomd_citations
        )

    def __call__(
        self,
        m1,
        m2,
        chi1z,
        chi2z,
        distance,
        phi_ref,
        f_ref,
        inc,
        lam,
        beta,
        psi,
        t_ref,
        t_obs_start=0.0,
        t_obs_end=1.0,
        freqs=None,
        length=None,
        modes=None,
        shift_t_limits=True,
        direct=False,
        compress=True,
        squeeze=False,
        fill=False,
        combine=False,
        return_splines=False,
    ):
        """Generate the binary black hole frequency-domain TDI waveforms.

        :class:`~bbhx.waveforms.phenomhm.PhenomHMAmpPhase` computes the per-mode amplitude,
        phase and t(f) (on a sparse grid of ``length`` frequencies per mode when
        ``direct=False``, on ``freqs`` when ``direct=True``);
        :class:`~bbhx.response.fastfdresponse.LISATDIResponse` adds the transfer functions
        and the reference phase; the TDI channels are then summed directly on that grid
        (``direct=True``) or spline-interpolated onto ``freqs`` (:class:`TemplateInterpFD`).
        With ``direct=False`` each mode's sparse grid is log-spaced over the model's default
        band (``mf_min`` to ``mf_max`` in units of the total mass), clipped to the
        frequencies whose t(f) lies between 500 s and 500 s before the end of the orbit
        data. Frequencies whose t(f) falls outside the observation window are zeroed.

        The source parameters accept scalars or 1D arrays of length ``num_bin_all`` (one
        entry per binary).

        Args:
            m1 (float or np.ndarray): Mass of body 1 in solar masses. When ``m1 < m2`` the
                bodies (and spins) are swapped so that ``m1 >= m2`` (see Note).
            m2 (float or np.ndarray): Mass of body 2 in solar masses.
            chi1z (float or np.ndarray): Dimensionless aligned spin of body 1,
                ``|chi1z| <= 1``.
            chi2z (float or np.ndarray): Dimensionless aligned spin of body 2,
                ``|chi2z| <= 1``.
            distance (float or np.ndarray): Luminosity distance in metres.
            phi_ref (float or np.ndarray): Reference phase (rad). It is applied by the
                response; the waveform kernel runs with zero phase.
            f_ref (float or np.ndarray): Reference frequency (Hz). Passed to the kernel but
                currently IGNORED: the kernel always uses the peak frequency of the PhenomD
                (2,2) merger-ringdown amplitude ansatz (``fmaxCalc``, Eq. 20 of
                arXiv:1508.07253) as reference frequency, so ``t_ref`` and ``phi_ref``
                refer to that frequency. ``0.0`` is the conventional value.
            inc (float or np.ndarray): Inclination (rad), in ``[0, pi]``.
            lam (float or np.ndarray): Ecliptic longitude (rad), SSB frame (see Warning).
            beta (float or np.ndarray): Ecliptic latitude (rad), SSB frame.
            psi (float or np.ndarray): Polarization angle (rad), in ``[0, pi]``.
            t_ref (float or np.ndarray): Time (s) at which the (2,2) mode reaches the
                reference frequency above (about the merger time); every mode's t(f) is
                shifted by ``t_ref``. It is read on the time axis of
                ``response_gen.orbits``. Intended as an SSB-frame time; see Warning.
            t_obs_start (float or np.ndarray, optional): Start of the observation in years,
                a LISA-frame time measured from ``t = 0`` (scalar or one per binary);
                converted per binary to an SSB time with
                :func:`~bbhx.utils.transform.tSSBfromLframe` (orbital phase ``t0 = 0``).
                Default ``0.0``.
            t_obs_end (float or np.ndarray, optional): End of the observation in years,
                same convention. Default ``1.0``.
            freqs (np.ndarray, optional): With ``direct=False``: data frequencies (Hz, 1D,
                increasing) to interpolate onto; required unless ``return_splines=True``.
                With ``direct=True``: the evaluation frequencies, 1D ``(length,)`` (shared by
                all binaries and modes) or 3D ``(num_bin_all, num_modes, length)``; when
                ``None``, the internal grid ``amp_phase_gen.freqs`` is used. Default
                ``None``.
            length (int, optional): Number of sparse frequencies per mode; required when
                ``direct=False``. With ``direct=True`` it is taken from ``freqs``, or, when
                ``freqs`` is ``None``, it is required and sets the size of the internal grid.
            modes (list, optional): ``(l, m)`` tuples, a subset of
                ``[(2,2), (3,3), (4,4), (2,1), (3,2), (4,3)]``. Default ``None``: all modes of
                the model (PhenomD: ``[(2,2)]``).
            shift_t_limits (bool, optional): Deprecated. ``False`` only emits a warning;
                ``t_obs_start`` / ``t_obs_end`` are always measured from ``t = 0``. Default
                ``True``.
            direct (bool, optional): Evaluate directly on ``freqs`` without interpolation.
                Default ``False``.
            compress (bool, optional): With ``direct=True``, sum the modes into the TDI
                channels; ``False`` keeps a mode axis. Default ``True``.
            squeeze (bool, optional): Remove length-1 axes from the output. Applied to the
                ``direct=True`` outputs and to ``fill=True, combine=True``; NOT applied to
                ``fill=True, combine=False`` or to the template tuple. Default ``False``.
            fill (bool, optional): With ``direct=False``, return data-length arrays instead
                of the per-binary template tuple. Default ``False``.
            combine (bool, optional): With ``fill=True``, sum all binaries into one data
                stream. Default ``False``.
            return_splines (bool, optional): With ``direct=False``, return the
                :class:`~gpubackendtools.interpolate.CubicSplineInterpolant` built over the 9
                sparse quantities instead of interpolating. Default ``False``.

        Returns:
            xp.ndarray or tuple or CubicSplineInterpolant: ``complex128`` arrays with the 3
            channels of the response's ``TDItag``:

            - ``direct=True, compress=True``: shape ``(num_bin_all, 3, length)``.
            - ``direct=True, compress=False``: shape ``(num_bin_all, 3, num_modes, length)``.
            - ``fill=True, combine=True``: shape ``(3, len(freqs))``.
            - ``fill=True, combine=False``: shape ``(num_bin_all, 3, len(freqs))``.
            - ``fill=False`` (default): ``(template_channels, start_inds, lengths)`` from
              :class:`TemplateInterpFD`; ``template_channels[i]`` has shape
              ``(3, lengths[i])`` and covers ``freqs[start_inds[i]:start_inds[i] + lengths[i]]``.
            - ``return_splines=True``: the spline object.

        Raises:
            ValueError: Neither ``freqs`` nor ``length`` is given; ``direct=False`` without
                ``length``; ``modes`` is not a list or holds an unavailable mode;
                ``|chi| > 1``; the t(f) window search fails (``t_ref`` too close to 0 or
                to the end of the orbit data).

        Note:
            ``m1``, ``m2``, ``chi1z``, ``chi2z`` passed as ``np.ndarray`` are modified in
            place when bodies are swapped (``np.atleast_1d`` does not copy).

        Warning:
            ``lam`` / ``beta`` enter the transfer functions together with the spacecraft
            positions of ``response_gen.orbits`` (they must be in that orbit data's frame)
            and the orbital-delay phase through an analytic circular orbit in the ecliptic
            plane, so the two agree only for ecliptic orbit data (the packaged LISA
            Analysis Tools orbit files hold SSB-ecliptic positions whatever their
            ``frame`` label; Mojito ``L1Orbits`` loaded with ``frame="icrs"`` do not).
            The merger arrival time does not follow the sky position at fixed
            ``t_ref`` the way the LISA Analysis Tools time-domain response does (open TODO
            in the method body).
        """

        # TODO(2026-10-12, sky-mode verification): the merger ARRIVAL time of
        # this FD response does not follow the sky position at fixed ``t_ref``
        # the way the LISA Analysis Tools time-domain response (pyResponseTDI)
        # does. Measured with the 22 mode on the stock equal-arm orbits:
        #   * two skies at the same t_ref: the arrival shifts by about MINUS
        #     TWICE the physical retardation difference (n2 - n1) . r_c / c
        #     (expected: plus one times it);
        #   * a LISA-frame sky-mode hop (SSB_to_LISA -> hop -> LISA_to_SSB,
        #     which moves t_ref by dt_hop so the DETECTOR time is fixed) only
        #     matches the original after shifting the hopped template back by
        #     -dt_hop, i.e. ``t_ref`` behaves like a LISA-frame time here.
        # Suspect: Response.cu applies exp(-i phaseRdelay) to the transfer
        # functions AND adds +phaseRdelay to ``phases`` (response_modes), and
        # WaveformBuild.cu conjugates the product -- the two orbital-delay
        # factors do not cancel/combine as intended. Not on the production
        # (time-domain) path; the SSB<->LISA transforms themselves are verified
        # against pyResponseTDI (tests/test_frame_transform.py and LAT
        # tests/test_skymodehop.py). Decide the intended t_ref convention and
        # add a two-sky arrival-time test when this is fixed.
        # make sure everything is at least a 1D array
        m1 = np.atleast_1d(m1)
        m2 = np.atleast_1d(m2)
        chi1z = np.atleast_1d(chi1z)
        chi2z = np.atleast_1d(chi2z)
        distance = np.atleast_1d(distance)
        phi_ref = np.atleast_1d(phi_ref)
        inc = np.atleast_1d(inc)
        lam = np.atleast_1d(lam)
        beta = np.atleast_1d(beta)
        psi = np.atleast_1d(psi)
        t_ref = np.atleast_1d(t_ref)

        self.num_bin_all = len(m1)

        # TODO: add sanity checks for t_start, t_end
        # how to set up time limits
        if shift_t_limits is False:
            warnings.warn(
                "Deprecated: shift_t_limits. Previously shift_t_limits defaulted to False. This option is now removed and permanently set to shift_t_limits=True."
            )
            # t_ref_L = tLfromSSBframe(t_ref, lam, beta)

            # # start and end times are defined in the LISA reference frame
            # t_obs_start_L = t_ref_L - t_obs_start * YRSID_SI
            # t_obs_end_L = t_ref_L - t_obs_end * YRSID_SI

            # # convert to SSB frame
            # t_obs_start_SSB = tSSBfromLframe(t_obs_start_L, lam, beta, 0.0)
            # t_obs_end_SSB = tSSBfromLframe(t_obs_end_L, lam, beta, 0.0)

            # # fix zeros and less than zero
            # t_start = (
            #     t_obs_start_SSB if t_obs_start > 0.0 else np.zeros(self.num_bin_all)
            # )
            # t_end = t_obs_end_SSB if t_obs_end > 0.0 else np.zeros_like(t_start)

        # else:
        # start and end times are defined in the LISA reference frame
        t_obs_start_L = t_obs_start * YRSID_SI
        t_obs_end_L = t_obs_end * YRSID_SI

        # convert to SSB frame
        t_obs_start_SSB = tSSBfromLframe(t_obs_start_L, lam, beta, 0.0)
        t_obs_end_SSB = tSSBfromLframe(t_obs_end_L, lam, beta, 0.0)
        t_start = np.atleast_1d(t_obs_start_SSB)
        t_end = np.atleast_1d(t_obs_end_SSB)

        self.t_start = t_start
        self.t_end = t_end

        if freqs is None and length is None:
            raise ValueError("Must input freqs or length.")

        # if computing directly, set info
        elif freqs is not None and direct is True:
            if freqs.ndim == 1:
                # set the length of the input
                length = len(freqs)
            else:
                length = freqs.shape[-1]

        # this means the frequencies are what needs to be interpolated to
        elif direct is False and return_splines is False:
            self.data_length = len(freqs)
            if length is None:
                raise ValueError("If direct is False, length parameter must be given.")

        self.length = length

        # setup harmonic modes
        if modes is None:
            # default mode setup
            self.num_modes = len(self.amp_phase_gen.allowable_modes)
        else:
            if not isinstance(modes, list):
                raise ValueError("modes must be a list.")
            self.num_modes = len(modes)

        self.num_bin_all = len(m1)

        out_buffer = self.xp.zeros(
            (self.num_interp_params * self.length * self.num_modes * self.num_bin_all)
        )

        freqs_temp = freqs if direct else None

        phi_ref_amp_phase = np.zeros_like(m1)

        Tobs = t_end - t_start
        t_orbit_end = self.response_gen.orbits.t.max().item()
        self.amp_phase_gen(
            m1,
            m2,
            chi1z,
            chi2z,
            distance,
            phi_ref_amp_phase,
            f_ref,
            t_ref,
            length,
            freqs=freqs_temp,
            out_buffer=out_buffer,
            modes=modes,
            t_orbit_end=t_orbit_end,
            direct=direct,
        )

        # setup buffer to carry around all the quantities of interest
        # params are amp, phase, tf, transferL1re, transferL1im, transferL2re, transferL2im, transferL3re, transferL3im
        out_buffer = out_buffer.reshape(
            self.num_interp_params, self.num_bin_all, self.num_modes, self.length
        )

        out_buffer = out_buffer.flatten().copy()

        # compute response function
        self.response_gen(
            self.amp_phase_gen.freqs,
            inc,
            lam,
            beta,
            psi,
            phi_ref,
            length,
            out_buffer=out_buffer,  # fill into this buffer
            modes=self.amp_phase_gen.modes,
            direct=direct,
        )

        # for checking
        self.out_buffer_final = out_buffer.reshape(
            9, self.num_bin_all, self.num_modes, self.length
        ).copy()

        # direct computation from buffer
        # + compressing all harmonics into a single data stream by direct combination
        if direct and compress:
            # setup template
            templateChannels = self.xp.zeros(
                (self.num_bin_all * 3 * self.length), dtype=self.xp.complex128
            )

            # direct computation of 3 channel waveform
            self.waveform_gen(
                templateChannels,
                out_buffer,
                self.num_bin_all,
                self.length,
                3,
                self.num_modes,
                self.xp.asarray(t_start),
                self.xp.asarray(t_end),
            )

            out = templateChannels.reshape(self.num_bin_all, 3, self.length)

            if squeeze:
                out = out.squeeze()

            return out

        elif direct:
            out = self.xp.zeros(
                (self.num_bin_all, 3, self.num_modes, self.length),
                dtype=self.xp.complex128,
            )
            for mode_i in range(self.num_modes):
                # setup template
                templateChannels = self.xp.zeros(
                    (self.num_bin_all * 3 * self.length), dtype=self.xp.complex128
                )

                out_buffer_temp = (
                    self.out_buffer_final[:, :, mode_i, :].copy().flatten()
                )
                # direct computation of 3 channel waveform
                self.waveform_gen(
                    templateChannels,
                    out_buffer_temp,
                    self.num_bin_all,
                    self.length,
                    3,
                    1,  # num_modes
                    self.xp.asarray(t_start),
                    self.xp.asarray(t_end),
                )

                out[:, :, mode_i, :] += templateChannels.reshape(
                    self.num_bin_all, 3, self.length
                )

            if squeeze:
                out = out.squeeze()

            return out

        else:

            # GBT's CubicSplineInterpolant treats x and y as one spline per
            # row of an (ninterps, length) grid. BBHx's spline grid is 4D --
            # (num_interp_params, num_bin_all, num_modes, length) for y but
            # only (num_bin_all, num_modes, length) for x, because the
            # frequency axis is shared across the interpolation parameters.
            # Tile x to match y's leading dimension, then build the spline
            # with ninterps = num_interp_params * num_bin_all * num_modes.
            ninterps = self.num_interp_params * self.num_bin_all * self.num_modes
            x_tiled = self.xp.tile(
                self.amp_phase_gen.freqs.ravel(), self.num_interp_params
            )
            spline = CubicSplineInterpolant(
                x_tiled,
                out_buffer.ravel(),
                ninterps=ninterps,
                length=self.length,
                force_backend=self.force_backend,
            )

            if return_splines:
                return spline

            # InterpTDI expects x with the un-tiled (num_bin_all, num_modes,
            # length) layout (shared across params); reuse the original
            # freqs buffer rather than spline.container[0] (which holds the
            # tiled copy).
            template_channels = self.interp_response(
                freqs,
                [
                    self.amp_phase_gen.freqs,
                    out_buffer,
                    spline.c1_flat,
                    spline.c2_flat,
                    spline.c3_flat,
                ],
                t_start,
                t_end,
                self.length,
                self.num_modes,
                3,
            )

            # fill the data stream
            if fill:
                if combine:
                    # combine into one data stream
                    data_out = self.xp.zeros((3, len(freqs)), dtype=self.xp.complex128)
                    for temp, start_i, length_i in zip(
                        template_channels,
                        self.interp_response.start_inds,
                        self.interp_response.lengths,
                    ):
                        data_out[:, start_i : start_i + length_i] += temp

                    if squeeze:
                        return data_out.squeeze()

                    return data_out

                else:
                    # put in separate data streams
                    data_out = self.xp.zeros(
                        (self.num_bin_all, 3, len(freqs)), dtype=self.xp.complex128
                    )
                    for bin_i, (temp, start_i, length_i) in enumerate(
                        zip(
                            template_channels,
                            self.interp_response.start_inds,
                            self.interp_response.lengths,
                        )
                    ):
                        data_out[bin_i, :, start_i : start_i + length_i] = temp

                return data_out
            else:
                # return information for the fast likelihood functions
                return (
                    template_channels,
                    self.interp_response.start_inds,
                    self.interp_response.lengths,
                )
    
    def get_direct_ll(
        self,
        data_freqs,
        dataChannels,
        psd,
        df,
        *waveform_args,
        data_index=None,
        noise_index=None,
        phase_maximize=False,
        **waveform_kwargs,
    ):
        """Direct log-likelihood of interpolated templates against one or more data sets.

        GPU only. Builds the sparse splines (:meth:`__call__` with ``return_splines=True``)
        and hands them to the fused ``speciallike`` kernel, which interpolates every binary
        onto ``data_freqs`` and accumulates ``<d|h>`` and ``<h|h>`` without materializing
        the templates. Binary ``i`` is compared with data set ``data_index[i]`` and PSD set
        ``noise_index[i]``. The caller must set ``self.d_d`` first (a float or an array of
        length ``num_bin_all``). Only caller: ``search_likelihood_wrap`` in LISA Analysis
        Tools' ``globalfit/mbhsearch.py`` (a development search script, not on the
        production path).

        Args:
            data_freqs (xp.ndarray): Data frequencies (Hz), 1D of length ``data_length``.
            dataChannels (xp.ndarray): Flat complex data of
                ``num_data_sets * 3 * data_length`` entries (``(num_data_sets, 3,
                data_length)`` flattened).
            psd (xp.ndarray): Flat PSD of ``num_noise_sets * 3 * data_length`` entries.
            df (float): Frequency spacing of the data (Hz).
            *waveform_args: Source parameters forwarded to :meth:`__call__`
                (``m1, m2, chi1z, chi2z, distance, phi_ref, f_ref, inc, lam, beta, psi,
                t_ref``).
            data_index (np.ndarray, optional): ``int32`` numpy array of length
                ``num_bin_all``: data set of each binary. Default ``None`` means all 0.
            noise_index (np.ndarray, optional): ``int32`` numpy array of length
                ``num_bin_all``: PSD set of each binary. Default ``None`` means all 0.
            phase_maximize (bool, optional): Use ``|<d|h>|`` (phase maximization); the
                complex value is kept in ``self.non_marg_d_h``. Default ``False``.
            **waveform_kwargs: Forwarded to :meth:`__call__`; must contain ``length``.
                ``direct``, ``compress``, ``squeeze``, ``fill``, ``combine`` and
                ``return_splines`` are overwritten.

        Returns:
            xp.ndarray: Complex, shape ``(num_bin_all,)``:
            ``-1/2 (d_d + <h|h> - 2 <d|h>)`` (take ``.real``). ``<d|h>`` and ``<h|h>`` are
            stored in ``self.d_h`` and ``self.h_h``.

        Raises:
            AssertionError: ``length`` missing, index arrays not ``int32`` numpy arrays, or
                data / PSD sizes not multiples of ``3 * data_length`` covering the indices.
            ValueError: ``self.d_d`` is not set.
            NameError: On a CPU backend (the CPU branch raises the undefined
                ``NotImplementedErrors``).
        """

        assert "length" in waveform_kwargs and waveform_kwargs["length"] is not None

        waveform_kwargs["direct"] = False
        waveform_kwargs["compress"] = False
        waveform_kwargs["squeeze"] = False
        waveform_kwargs["fill"] = False
        waveform_kwargs["combine"] = False

        waveform_kwargs["return_splines"] = True

        splines = self.__call__(*waveform_args, **waveform_kwargs)

        t_start = self.t_start
        t_end = self.t_end
        
        # fill important quantities
        self.data_length = len(data_freqs)
        self.num_channels = 3

        # unpack interp_container -- splines.container[0] holds the tiled
        # x (size num_interp_params * num_bin_all * num_modes * length) we
        # built for GBT's ninterps-flat layout; downstream interp_like
        # expects the un-tiled (num_bin_all, num_modes, length) layout
        # used by the original amp/phase grid. Slice the leading copy.
        spline_x_size = self.num_bin_all * self.num_modes * self.length
        freqs = splines.x_flat[:spline_x_size]
        y = splines.y_flat
        c1 = splines.c1_flat
        c2 = splines.c2_flat
        c3 = splines.c3_flat

        freqs_shaped = freqs.reshape(self.num_bin_all, -1)

        # find where each binary's signal starts and ends in the data array

        start_and_end = self.xp.asarray([freqs_shaped[:, 0], freqs_shaped[:, -1],]).T

        # TODO: use 2D version
        inds_start_and_end = self.xp.asarray(
            [
                self.xp.searchsorted(data_freqs, temp, side="right")
                for temp in start_and_end
            ]
        )

        # find proper interpolation window for each point in data stream
        inds = [
            self.xp.searchsorted(
                freqs_shaped[i], data_freqs[st:et], side="right"
            ).astype(self.xp.int32)
            - 1
            for i, (st, et) in enumerate(inds_start_and_end)
        ]

        # lengths of the signals in frequency domain
        self.lengths = lengths = np.asarray(
            [len(inds_i) for inds_i in inds], dtype=self.xp.int32
        )

        # make sure have this quantity available on CPU
        try:
            temp_inds = inds_start_and_end[:, 0].get()
        except AttributeError:
            temp_inds = inds_start_and_end[:, 0]

        # where to start filling waveform
        self.start_inds = start_inds = (temp_inds.copy()).astype(np.int32)

        # get pointers to all the index arrays of different length
        try:
            self.ptrs = ptrs = np.asarray([ind_i.data.ptr for ind_i in inds])
        except AttributeError:
            self.ptrs = ptrs = np.asarray(
                [ind_i.__array_interface__["data"][0] for ind_i in inds]
            )

        d_h = self.xp.zeros(self.num_bin_all, dtype=complex)
        h_h = self.xp.zeros(self.num_bin_all, dtype=complex)

        if data_index is None:
            data_index = np.zeros(self.num_bin_all, dtype=np.int32)
        if noise_index is None:
            noise_index = np.zeros(self.num_bin_all, dtype=np.int32)

        assert (data_index.max() + 1) * self.data_length * 3 <= dataChannels.shape[0]
        assert (noise_index.max() + 1) * self.data_length * 3 <= psd.shape[0]
        assert data_index.dtype == np.int32
        assert noise_index.dtype == np.int32
        assert isinstance(data_index, np.ndarray)
        assert isinstance(noise_index, np.ndarray)

        tmp = dataChannels.shape[0] / 3 / self.data_length
        assert tmp == float(int(tmp))
        num_data_sets = int(tmp)

        tmp = psd.shape[0] / 3 / self.data_length
        assert tmp == float(int(tmp))
        num_noise_sets = int(tmp)

        if not hasattr(self, "d_d"):
            raise ValueError("Need to set d_d term for self.")
        
        if self.backend.uses_gpu:
            # TODO: add get_cuda_device to gpubackendtools
            gpu = self.backend.xp.cuda.runtime.getDevice()
        else:
            raise NotImplementedErrors
        
        self.interp_like(
            d_h, 
            h_h, 
            dataChannels, 
            psd,
            data_freqs,
            freqs,
            y,
            c1,
            c2,
            c3,
            t_start,
            t_end,
            self.length,
            self.data_length,
            self.num_bin_all,
            self.num_modes,
            ptrs,
            start_inds,
            lengths,
            df, 
            data_index, 
            num_data_sets, 
            noise_index, 
            num_noise_sets,
            gpu
        )

        if phase_maximize:
            self.non_marg_d_h = d_h.copy()
            d_h = self.xp.abs(d_h)

        like = -1/2 * (self.d_d + h_h - 2 * d_h)

        self.h_h = h_h
        self.d_h = d_h
        # return templates in the right shape
        return like





