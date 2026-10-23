# Fast Likelihood functions

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

"""Fast MBH log-likelihoods for frequency-domain BBHx templates.

- :class:`Likelihood`: direct noise-weighted sum over the data frequencies where each
  template is non-zero.
- :class:`HeterodynedLikelihood`: heterodyned (relative-binning) likelihood against a
  reference template, on a sparse frequency grid.
- :class:`NewHeterodynedLikelihood`: GPU-only multi-reference / multi-data-set variant.
- :func:`searchsorted2d_vec`: row-wise ``searchsorted`` helper.

The templates come from :class:`bbhx.waveformbuild.BBHWaveformFD` (or any generator with
the same call signature and output layout). Not used by the global-fit production runs.
"""

import numpy as np

from .utils.constants import *
from .utils.parallelbase import BBHxParallelModule

from lisatools.sensitivity import SensitivityMatrix, AET1SensitivityMatrix


class Likelihood(BBHxParallelModule):
    r"""Fast Base Likelihood Class for MBHBs.

    This class computes the gravitational wave Likelihood as a direct sum over
    frequency bins. It only sums over the frequencies where the MBHB signal
    exists. Therefore, larger mass waveforms are faster because there are less
    frequencies. This class computes:

    .. math::

        \mathcal{L}\propto-\frac{1}{2}\langle d-h|d-h\rangle
        =-\frac{1}{2}\left(\langle d|d\rangle + \langle h|h\rangle
        - 2\langle d|h\rangle\right),

    with :math:`\langle a|b\rangle = 4\sum_f a^*(f) b(f) \Delta f / S_n(f)` summed over
    channels. The data are whitened once at construction
    (``data_channels * sqrt(delta_f / psd)``); ``<d|d>`` is computed then.

    This class has GPU capability.

    Args:
        template_gen (obj): Waveform generator called as
            ``template_gen(*params, freqs=data_freqs, fill=False, direct=False, **kw)``
            that returns ``(template_channels, start_inds, lengths)`` and exposes
            ``num_bin_all``, e.g. :class:`bbhx.waveformbuild.BBHWaveformFD`.
        data_freqs (xp.ndarray): Frequencies for the data stream (Hz), 1D. ``data_freqs``
            should be a numpy (cupy) array if running on the CPU (GPU).
        data_channels (xp.ndarray): Complex data stream, shape
            ``(nchannels, len(data_freqs))`` (3 channels with BBHx templates), NOT whitened.
            ``data_channels`` should be a numpy (cupy) array if running on the CPU (GPU).
        psd (np.ndarray or cupy.ndarray): One-sided noise PSD :math:`S_n(f)`, shape
            ``(nchannels, len(data_freqs))``; copied to the host.
        force_backend (str, optional): Backend name (``"cpu"``, ``"cuda12x"``, ``"cuda"``,
            ``"gpu"``, ...), see :class:`~bbhx.utils.parallelbase.BBHxParallelModule`.
            Default ``None`` picks the first available of ``cuda13x``, ``cuda12x``,
            ``cuda11x``, ``cpu``.

    Attributes:
        d_d (float): :math:`\langle d|d\rangle` inner product value.
        data_channels (xp.ndarray): Whitened data ``noise_factors * data_channels``, flat
            ``(nchannels * len(data_freqs),)``.
        data_freqs (xp.ndarray): Frequencies for the data stream (1D).
        data_stream_length (int): Length of data.
        delta_f (np.ndarray): Frequency spacing per bin, shape of ``psd`` (the first bin
            repeats the second).
        nchannels (int): Number of channels (``psd.shape[0]``).
        noise_factors (xp.ndarray): :math:`\sqrt{\Delta f / S_n(f)}`, flat
            ``(nchannels * len(data_freqs),)``.
        psd (np.ndarray): Power Spectral Density :math:`S_n(f)` (host copy), shape
            ``(nchannels, len(data_freqs))``.
        waveform_gen (obj): The ``template_gen`` argument.
        d_h (np.ndarray): Complex ``<d|h>`` per binary of the last :meth:`get_ll` call.
        h_h (np.ndarray): Complex ``<h|h>`` per binary of the last :meth:`get_ll` call.
        phase_maximize (bool): If ``True``, compute the phase-maximized
            log-Likelihood (and snr if ``return_extracted_snr==True``).
        return_extracted_snr (bool): Return the snr in addition to the Likelihood.

    Example:
        .. code-block:: python

            from bbhx.likelihood import Likelihood

            # wave_gen: a BBHWaveformFD; data_freqs (n,); data (3, n) A, E, T; psd (3, n)
            like = Likelihood(wave_gen, data_freqs, data, psd, force_backend="cpu")
            # params: (12, num_bin_all) in BBHWaveformFD.__call__ order
            ll = like.get_ll(params, length=1024)
    """

    def __init__(
        self,
        template_gen,
        data_freqs,
        data_channels,
        psd,
        force_backend=None,
    ):

        super().__init__(force_backend=force_backend)

        # store required information
        self.data_freqs = data_freqs

        try:
            data_freqs_cpu = data_freqs.get()
        except AttributeError:
            data_freqs_cpu = data_freqs

        # will not store psd or delta_f on GPU if being used
        try:
            psd = psd.get()
        except AttributeError:
            pass

        self.psd = np.asarray(psd)
        self.delta_f = np.zeros_like(self.psd)
        self.delta_f[:, 1:] = np.diff(data_freqs_cpu)
        self.delta_f[:, 0] = self.delta_f[:, 1]

        self.noise_factors = self.xp.asarray(
            np.sqrt(1.0 / psd * self.delta_f)
        ).flatten()

        # store and adjust data_channels
        self.data_channels = self.noise_factors * self.xp.asarray(
            data_channels.flatten()
        )

        self.waveform_gen = template_gen
        self.data_stream_length = len(data_freqs)
        # psd is shape (nchannels, n_freqs); infer the channel count.
        # (Mirrors the HeterodynedLikelihood pattern below at line 923.)
        self.nchannels = self.psd.shape[0]

        # assumes data_channels is already factored by psd
        self.d_d = (
            4 * self.xp.sum((self.data_channels.conj() * self.data_channels)).real
        ).item()

    @classmethod
    def supported_backends(cls) -> list:
        """Preference order: ``bbhx_`` + ``cuda13x``, ``cuda12x``, ``cuda11x``, ``cpu``."""
        return ["bbhx_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    @property
    def like_gen(self):
        """Native ``direct_like_wrap`` kernel (CPU or GPU)."""
        return self.backend.direct_like_wrap

    @property
    def xp(self):
        """Cupy or Numpy"""
        return self.backend.xp

    @property
    def citation(self):
        """Citation strings for this class.

        Warning:
            Raises ``NameError``: ``katz_citations`` is no longer defined in this module
            (``bbhx.utils.citations`` was removed).
        """
        return katz_citations

    def get_ll(
        self,
        params,
        return_extracted_snr=False,
        phase_maximize=False,
        **waveform_kwargs
    ):
        r"""Compute the log-Likelihood.

        Generates the templates on the data frequencies (``fill=False``, ``direct=False``)
        and accumulates ``<d|h>`` and ``<h|h>`` with the native kernel over each
        template's band only.

        Args:
            params (np.ndarray): Parameters for evaluating log-Likelihood, unpacked
                row by row into ``template_gen``: shape ``(num_params,)`` for one binary
                or ``(num_params, num_bin_all)`` for several (for
                :class:`~bbhx.waveformbuild.BBHWaveformFD`: the 12 rows ``m1, m2, chi1z,
                chi2z, distance, phi_ref, f_ref, inc, lam, beta, psi, t_ref``).
            return_extracted_snr (bool, optional): If ``True``, also return
                :math:`\mathrm{Re}\langle d|h\rangle / \sqrt{\mathrm{Re}\langle h|h\rangle}`
                as a second column. Default ``False``.
            phase_maximize (bool, optional): If ``True``, use :math:`|\langle d|h\rangle|`
                (the phase-maximized log-Likelihood, and snr if
                ``return_extracted_snr=True``). Default ``False``.
            **waveform_kwargs: Keyword arguments for ``template_gen`` (for
                :class:`~bbhx.waveformbuild.BBHWaveformFD`: ``length`` is required;
                ``modes``, ``t_obs_start``, ...). ``freqs``, ``fill`` and ``direct`` are
                overwritten.

        Returns:
            np.ndarray: Log-likelihoods, shape ``(num_bin_all,)``, or
            ``(num_bin_all, 2)`` columns ``[log-Likelihood, snr]`` with
            ``return_extracted_snr=True``.
        """

        # store info
        self.phase_maximize = phase_maximize
        self.return_extracted_snr = return_extracted_snr

        # setup kwargs properly
        waveform_kwargs["freqs"] = self.data_freqs
        waveform_kwargs["fill"] = False
        waveform_kwargs["direct"] = False

        # get information from waveform generators
        templateChannels, inds_start, ind_lengths = self.waveform_gen(
            *params, **waveform_kwargs
        )

        # get flattened templates to remove channel dimension
        templateChannels = [tc.flatten() for tc in templateChannels]

        # adjust for cupy vs numpy
        try:
            templateChannels_ptrs = np.asarray(
                [tc.data.ptr for tc in templateChannels], dtype=np.int64
            )
        except AttributeError:
            templateChannels_ptrs = np.asarray(
                [tc.__array_interface__["data"][0] for tc in templateChannels],
                dtype=np.int64,
            )

        # initialize inner product info
        self.d_h = np.zeros(self.waveform_gen.num_bin_all, dtype=self.xp.complex128)
        self.h_h = np.zeros(self.waveform_gen.num_bin_all, dtype=self.xp.complex128)

        # Get current CUDA device when on cupy backend; CPU mode has no
        # device, fall back to -1 (the C kernel uses this to take the
        # CPU code path).
        if hasattr(self.xp, "cuda"):
            device = self.xp.cuda.runtime.getDevice()
        else:
            device = -1
        self.like_gen(
            self.d_h,
            self.h_h,
            self.data_channels,
            self.noise_factors,
            templateChannels_ptrs,
            inds_start,
            ind_lengths,
            self.data_stream_length,
            self.waveform_gen.num_bin_all,
            self.nchannels,
            device
        )

        # phase marginalize in d_h term
        d_h_temp = self.d_h if not self.phase_maximize else self.xp.abs(self.d_h)
        out = -1 / 2 * (self.d_d + self.h_h - 2 * d_h_temp).real
        # get out of cupy if needed
        try:
            out = out.get()

        except AttributeError:
            pass

        if self.return_extracted_snr:
            return np.array([out, d_h_temp.real / np.sqrt(self.h_h.real)]).T
        else:
            return out


class HeterodynedLikelihood(BBHxParallelModule):
    r"""Compute the Heterodyned log-Likelihood.

    Heterodyning involves separating the fast and slow evolutions when comparing
    signals. This involves comparing a reference template to the data stream and determining
    various quantities at the full frequency resolution of the data stream. Then, during
    online computation, the log-Likelihood is determined by computing a new waveform
    on a sparse frequency grid and comparing it to the reference waveform on the same
    sparse grid. The practical aspect is the computation of a smaller number of frequency
    points which lowers the required  memory and increases the speed of the computation.
    More information on the general method can be found in
    `arXiv:2109.02728 <https://arxiv.org/abs/2109.02728>`_. We implement the method
    as described in `arXiv:1806.08792 <https://arxiv.org/abs/1806.08792>`_.

    This class also works with higher order harmonic modes, but it has not been tested
    extensively. It only does a direct summation over the modes rather than heterodyning
    per mode. So, it is less reliable, but in practice it produces a solid posterior
    distribution.

    Setup (:meth:`init_heterodyne_info`, run by the constructor): the reference template
    ``h0`` is generated on the dense data grid; the sparse grid is ``length_f_het``
    log-spaced frequencies between the first and last data frequency where ``h0`` is
    non-zero; the summary constants A0, A1, B0, B1 are accumulated per sparse bin with the
    PSD of ``sens_mat`` evaluated on the data grid. Up to ``outer_buffer // 2 + 1`` dense
    bins on each side of the reference band (clipped at the grid edges) are handled
    directly (no heterodyning) by :meth:`get_ll`; for a template whose ``Re <h|h>`` or
    ``Re <d|h>`` contribution from the outermost bin of that buffer exceeds ``0.001``, the
    outer terms are set to ``-1e300``, which gives a log-likelihood of about ``-5e299``
    without phase maximization but about ``+1.5e300`` with ``phase_maximize=True`` (the
    absolute value of ``<d|h>`` flips the sign).

    This class has GPU capabilities.

    Args:
        template_gen (obj): Waveform generator with the
            :class:`~bbhx.waveformbuild.BBHWaveformFD` call signature and output layout
            (it must expose ``num_bin_all``).
        data_freqs (xp.ndarray): Frequencies for the data stream (Hz), 1D and evenly
            spaced. ``data_freqs`` should be a numpy (cupy) array if running on the CPU
            (GPU).
        data_channels (xp.ndarray): Complex data stream, shape ``(3, len(data_freqs))``
            (A, E, T matching ``sens_mat``). ``data_channels`` should be a numpy (cupy)
            array if running on the CPU (GPU).
        reference_template_params (np.ndarray): Parameters of the reference template, one
            binary, unpacked into ``template_gen`` (``(num_params,)``).
        length_f_het (int): Number of sparse (heterodyne) frequencies. Must be smaller than
            the number of data frequencies inside the reference band.
        template_gen_kwargs (dict, optional): Keywords arguments for generating the
            template with ``template_gen``. It will automatically add/change ``direct``
            and ``compress`` keyword arguments to ``True``. The keyword argument
            ``squeeze`` is automatically set to ``True`` for the initial setup and then
            automatically set to ``False`` for online computations. If you so choose (not
            recommended), you can change these kwargs for online running using the
            ``**waveform_kwargs`` setup in the ``self.get_ll`` method. However,
            if you include in this adjustment ``direct``, ``compress``, or ``squeeze``
            will still automatically be overwritten. The dict is modified in place.
            Default ``{}``.
        reference_gen_kwargs (dict, optional): Keywords arguments for generating the
            reference template with ``template_gen``. It will automatically add/change
            ``fill``, ``compress``, and ``squeeze`` keyword arguments to ``True``.
            These waveforms can be produced without interpolation with ``direct = True``.
            If ``length`` keyword argument is given and ``direct=False``, the
            waveform will be interpolated. If ``length`` and ``direct`` are not given,
            it will interpolate the signal with ``length=8096``. The dict is modified in
            place. Default ``{}``. See the Warning on the interpolated reference.
        sens_mat (SensitivityMatrix, optional): LISA Analysis Tools
            :class:`~lisatools.sensitivity.SensitivityMatrix` with a 1D (uncorrelated)
            channel layout, whose entries ``[0]``, ``[1]``, ``[2]`` are the PSDs of the three
            data channels; it is re-evaluated on the data grid (``update_basis_settings``).
            Default ``None`` uses :class:`~lisatools.sensitivity.AET1SensitivityMatrix`
            (the analytic equal-arm TDI 1.5 A/E/T PSDs, not the global-fit noise model).
        outer_buffer (int, optional): Number of dense bins (split evenly below and above
            the reference band) handled directly in :meth:`get_ll`. Default ``1024``.
        force_backend (str, optional): Backend name (``"cpu"``, ``"cuda12x"``, ``"cuda"``,
            ``"gpu"``, ...), see :class:`~bbhx.utils.parallelbase.BBHxParallelModule`.
            Default ``None`` picks the first available of ``cuda13x``, ``cuda12x``,
            ``cuda11x``, ``cpu``.

    Attributes:
        reference_d_d (float): :math:`\langle d|d\rangle` inner product value.
        reference_h_h (float): :math:`\langle h|h\rangle` inner product value
            for the reference template.
        reference_d_h (float): :math:`\langle d|h\rangle` inner product value
            for the reference template.
        reference_ll (float): log-Likelihood value for the reference template.
        hdyn_d_h (np.ndarray): Heterodyned complex :math:`\langle d|h\rangle`
            inner product values for the test templates (last call).
        hdyn_h_h (np.ndarray): Heterodyned complex :math:`\langle h|h\rangle`
            inner product values for the test templates (last call).
        total_d_h (np.ndarray): ``hdyn_d_h`` plus the direct outer-buffer part.
        total_h_h (np.ndarray): ``hdyn_h_h`` plus the direct outer-buffer part.
        h0_sparse (xp.ndarray): Sparse reference waveform, shape ``(1, 3, length_f_het)``.
        h_sparse (xp.ndarray): Sparse waveforms for the test parameters (last call).
        d (xp.ndarray): Data stream, shape ``(3, len(data_freqs))``.
        data_constants (xp.ndarray): Flattened array container holding all heterodyning
            constants needed: A0, A1, B0, B1 (each ``(3, length_f_het)``).
        f_dense (xp.ndarray): Frequencies for the data stream (1D).
        freqs (xp.ndarray): Frequencies for sparse arrays.
        f_m (xp.ndarray): Frequency of mid-point in each sparse bin.
        length_f_het (int): Length of sparse array.
        outer_freqs (xp.ndarray): Dense frequencies of the outer buffer.
        return_extracted_snr (bool): Return the snr in addition to the Likelihood.
        phase_maximize (bool): If ``True``, compute the phase-maximized
            log-Likelihood (and snr if ``return_extracted_snr==True``).

    Warning:
        With the default interpolated reference (``reference_gen_kwargs`` without
        ``direct=True``), :class:`~bbhx.waveformbuild.BBHWaveformFD` returns the dense
        reference with shape ``(1, 3, n)`` (``fill=True, combine=False`` ignores
        ``squeeze``), while :meth:`init_heterodyne_info` indexes it as ``(3, n)``: the
        band search then picks up channel indices instead of frequency indices, the
        sparse grid collapses onto the first three data frequencies, and the setup fails
        (``IndexError`` on CPU). Only a ``direct=True`` reference has the expected shape.

    Example:
        .. code-block:: python

            from bbhx.likelihood import HeterodynedLikelihood

            # wave_gen: a BBHWaveformFD; data_freqs (n,) evenly spaced; data (3, n) A, E, T
            # direct=True reference: see the Warning on the default interpolated one
            like_het = HeterodynedLikelihood(
                wave_gen, data_freqs, data, reference_params, 128,
                reference_gen_kwargs=dict(direct=True), force_backend="cpu",
            )
            # params: (12, num_bin_all) in BBHWaveformFD.__call__ order
            ll = like_het.get_ll(params)
            like_het.init_heterodyne_info(new_reference_params)  # re-centre when needed
    """

    def __init__(
        self,
        template_gen,
        data_freqs,
        data_channels,
        reference_template_params,
        length_f_het,
        template_gen_kwargs={},
        reference_gen_kwargs={},
        sens_mat=None,
        outer_buffer=1024,
        force_backend=None,
    ):

        self.buffer = outer_buffer
        # store all input information
        self.template_gen = template_gen
        self.f_dense = data_freqs
        self.d = data_channels
        self.length_f_het = length_f_het  #  + self.buffer
        self.half_buffer = int(self.buffer / 2)

        # direct based on GPU usage
        super().__init__(force_backend=force_backend)

        self.sens_mat = sens_mat

        # calculate all quantites related to the reference template
        self.init_heterodyne_info(
            reference_template_params,
            template_gen_kwargs=template_gen_kwargs,
            reference_gen_kwargs=reference_gen_kwargs,
        )

    @classmethod
    def supported_backends(cls) -> list:
        """Preference order: ``bbhx_`` + ``cuda13x``, ``cuda12x``, ``cuda11x``, ``cpu``."""
        return ["bbhx_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    @property
    def sens_mat(self):
        """Sensitivity matrix of the data channels (see the ``sens_mat`` argument).

        Setting ``None`` stores a placeholder
        :class:`~lisatools.sensitivity.AET1SensitivityMatrix` on a dummy
        ``FDSettings(N=1001, df=1e-4)`` grid; :meth:`init_heterodyne_info` re-evaluates the
        matrix on the data grid. Other values must be
        :class:`~lisatools.sensitivity.SensitivityMatrix` instances (asserted).
        """
        return self._sens_mat

    @sens_mat.setter
    def sens_mat(self, sens_mat):
        if sens_mat is None:
            # Post-Phase-3 LAT refactor: SensitivityMatrix subclasses take
            # a DomainSettingsBase, not a raw frequency array. This is a
            # placeholder built only to satisfy the constructor; it is
            # immediately overwritten downstream when the user supplies
            # real psd data on the het grid (set_data path).
            from lisatools.domains import FDSettings
            _settings = FDSettings(N=1001, df=1e-4, min_freq=0.0, max_freq=1e-1)
            sens_mat = AET1SensitivityMatrix(_settings)
        assert isinstance(sens_mat, SensitivityMatrix)
        self._sens_mat = sens_mat

    @property
    def like_gen(self):
        """Native ``hdyn_wrap`` kernel (CPU or GPU)."""
        return self.backend.hdyn_wrap

    @property
    def xp(self):
        """Numpy or Cupy"""
        return self.backend.xp

    @property
    def citation(self):
        """Citation strings for this class.

        Warning:
            Raises ``NameError``: the citation strings (``katz_citations``, ...) are no
            longer defined in this module (``bbhx.utils.citations`` was removed).
        """
        return katz_citations + Cornish_Heterodyning + Rel_Bin_citation

    def init_heterodyne_info(
        self,
        reference_template_params,
        template_gen_kwargs={},
        reference_gen_kwargs={},
    ):
        """Prepare all information for Heterodyning.

        Run by the constructor; call it again to re-centre on a new reference template
        (the data, ``length_f_het`` and ``sens_mat`` are kept). Builds the dense
        reference, the sparse grid, the outer-buffer indices and the A0, A1, B0, B1
        summary constants, and sets the ``reference_*`` attributes (see the class
        docstring). The kwargs given to the constructor are not reused: pass them again
        when re-centring.

        Args:
            reference_template_params (np.ndarray): Parameters for the reference template for
                ``template_gen`` (one binary, ``(num_params,)``).
            template_gen_kwargs (dict, optional): Keywords arguments for generating the
                template with ``template_gen``. It will automatically add/change ``direct``
                and ``compress`` keyword arguments to ``True``. The keyword argument
                ``squeeze`` is automatically set to ``True`` for the initial setup and then
                automatically set to ``False`` for online computations. If you so choose (not
                recommended), you can change these kwargs for online running using the
                ``**waveform_kwargs`` setup in the ``self.get_ll`` method. However,
                if you include in this adjustment ``direct``, ``compress``, or ``squeeze``
                will still automatically be overwritten. Modified in place and stored as
                ``self.template_gen_kwargs``. Default ``{}``.
            reference_gen_kwargs (dict, optional): Keywords arguments for generating the
                reference template with ``template_gen``. It will automatically add/change
                ``fill``, ``compress``, and ``squeeze`` keyword arguments to ``True``.
                These waveforms can be produced without interpolation with ``direct = True``.
                If ``length`` keyword argument is given and ``direct=False``, the
                waveform will be interpolated. If ``length`` and ``direct`` are not given,
                it will interpolate the signal with ``length=8096``. Modified in place.
                Default ``{}``.

        Raises:
            AssertionError: The reference band holds no more than ``length_f_het`` data
                frequencies, or it starts at data index 1 / ends at index ``n - 2``
                (degenerate outer buffer).
        """

        # add the necessary kwargs for the initial template generation process.
        template_gen_kwargs["squeeze"] = True
        template_gen_kwargs["compress"] = True
        template_gen_kwargs["direct"] = True

        # need to be just outside the values of the waveform
        minF = self.f_dense.min() * 0.999999999999
        maxF = self.f_dense.max() * 1.000000000001

        if minF == 0.0:
            minF = 1e-6

        reference_gen_kwargs["squeeze"] = True
        reference_gen_kwargs["compress"] = True
        reference_gen_kwargs["fill"] = True

        # setup for reference template generation
        if "direct" in reference_gen_kwargs and reference_gen_kwargs["direct"] is True:
            pass
        else:
            reference_gen_kwargs["direct"] = False
            if "length" not in reference_gen_kwargs:
                reference_gen_kwargs["length"] = 8096

        # Heterodyning grid
        freqs = self.xp.logspace(
            self.xp.log10(minF), self.xp.log10(maxF), self.length_f_het
        )

        self.reference_template_params = reference_template_params.copy()
        self.reference_gen_kwargs = reference_gen_kwargs
        
        # generate dense reference template
        h0 = self.template_gen(*reference_template_params, freqs=self.f_dense, **reference_gen_kwargs)  # [0]

        # generate sparse reference template
        # h0_temp = self.template_gen(
        #     *reference_template_params, freqs=freqs, **template_gen_kwargs
        # )[0]

        
        # # get rid of places where freqs are zero and narrow boundaries
        inds_freq_keep = self.xp.where(~(self.xp.abs(h0[0]) == 0.0))[0]

        self.freqs_keep = self.f_dense[inds_freq_keep]
        min_freq_ind_lower = inds_freq_keep[0] - (self.half_buffer + 1)
        if min_freq_ind_lower < 0:
            min_freq_ind_lower = 0
        max_freq_ind_lower = inds_freq_keep[0] - 1
        assert min_freq_ind_lower != max_freq_ind_lower

        min_freq_ind_upper = inds_freq_keep[-1] + 1
        max_freq_ind_upper = inds_freq_keep[-1] + (self.half_buffer + 1)
        if max_freq_ind_upper > self.f_dense.shape[0] - 1:
            max_freq_ind_upper = self.f_dense.shape[0] - 1
        assert min_freq_ind_upper != max_freq_ind_upper

        self.outer_inds = self.xp.concatenate([self.xp.arange(self.f_dense.shape[0])[min_freq_ind_lower:max_freq_ind_lower + 1], self.xp.arange(self.f_dense.shape[0])[min_freq_ind_upper:max_freq_ind_upper + 1]]) 
        self.outer_freqs = self.xp.concatenate([self.f_dense[min_freq_ind_lower:max_freq_ind_lower + 1], self.f_dense[min_freq_ind_upper:max_freq_ind_upper + 1]]) 
        
        assert len(self.freqs_keep) > self.length_f_het
        freqs_middle = self.xp.logspace(
            self.xp.log10(self.freqs_keep[0]),
            self.xp.log10(self.freqs_keep[-1]),
            self.length_f_het,
        )

        # freqs = self.xp.concatenate([self.freqs_keep[:self.half_buffer], freqs_middle, self.freqs_keep[-self.half_buffer:]])
        freqs = freqs_middle

        # regenerate at only non-zero values of the waveform
        self.h0_sparse = self.template_gen(
            *reference_template_params, freqs=freqs, **template_gen_kwargs
        )[self.xp.newaxis, :, :]

        # find which frequencies in the dense array in contained in the sparse array
        # inds = (self.f_dense >= freqs[0]) & (self.f_dense <= freqs[-1])

        self.h0_outer = self.template_gen(
            *reference_template_params, freqs=self.outer_freqs, **reference_gen_kwargs
        )[0]

        # narrow the dense arrays to these indices
        self.f_dense = self.f_dense  # [inds]
        self.d = self.d  # [:, inds]
        self.d_outer = self.d[:, self.outer_inds]
        h0 = h0  # [:, inds]

        # find which sparse array bins the dense frequencies fit into
        bins = self.xp.searchsorted(freqs, self.f_dense, "right") - 1

        # get frequency at middle of the bin
        f_m = (freqs[1:] + freqs[:-1]) / 2

        df = self.f_dense[1] - self.f_dense[0]
        self.df = df
        # should be on CPU for sensitivity computation
        try:
            f_n_host = self.f_dense.get()
        except AttributeError:
            f_n_host = self.f_dense

        # Post-Phase-3 LAT refactor: SensitivityMatrix.update_frequency_arr
        # became SensitivityMatrix.update_basis_settings, which takes a
        # DomainSettingsBase. Wrap f_n_host in an FDSettings that covers
        # the same active band.
        from lisatools.domains import FDSettings
        _df_dense = float(f_n_host[1] - f_n_host[0])
        _fd_settings_dense = FDSettings(
            N=len(f_n_host) + 1,  # +1 to account for the DC bin
            df=_df_dense,
            min_freq=float(f_n_host[0]),
            max_freq=float(f_n_host[-1]),
        )
        self.sens_mat.update_basis_settings(_fd_settings_dense)

        # compute sensitivity at dense frequencies
        S_n = self.xp.asarray([self.sens_mat[0], self.sens_mat[1], self.sens_mat[2]])
        self.S_n_outer = S_n[:, self.outer_inds].copy()

        # compute the individual frequency contributions to A0, A1 (see paper)
        A0_flat = 4 * (h0.conj() * self.d) / S_n * df
        A1_flat = 4 * (h0.conj() * self.d) / S_n * df * (self.f_dense - f_m[bins])

        # compute the individual frequency contributions to B0, B1 (see paper)
        B0_flat = 4 * (h0.conj() * h0) / S_n * df
        B1_flat = 4 * (h0.conj() * h0) / S_n * df * (self.f_dense - f_m[bins])

        # initialize containers for sparse sums of A0, A1, B0, B1
        A0_in = self.xp.zeros((3, self.length_f_het), dtype=np.complex128)
        A1_in = self.xp.zeros_like(A0_in)
        B0_in = self.xp.zeros_like(A0_in)
        B1_in = self.xp.zeros_like(A0_in)

        for ind in self.xp.unique(bins[:-1]):
            inds_keep = bins == ind

            # TODO: check this
            inds_keep[-1] = False

            # +1 allows for zero as the first entry (for C compatibility)
            A0_in[:, ind + 1] = self.xp.sum(A0_flat[:, inds_keep], axis=1)
            A1_in[:, ind + 1] = self.xp.sum(A1_flat[:, inds_keep], axis=1)
            B0_in[:, ind + 1] = self.xp.sum(B0_flat[:, inds_keep], axis=1)
            B1_in[:, ind + 1] = self.xp.sum(B1_flat[:, inds_keep], axis=1)

        # compute stored array of all coefficients
        self.data_constants = self.xp.concatenate(
            [A0_in.flatten(), A1_in.flatten(), B0_in.flatten(), B1_in.flatten()]
        )

        # reference quantities
        self.reference_d_d = self.xp.sum(4 * (self.d.conj() * self.d) / S_n * df).real

        self.reference_h_h = self.xp.sum(B0_flat).real

        self.reference_d_h = self.xp.sum(A0_flat).real

        self.reference_ll = (
            -1 / 2 * (self.reference_d_d + self.reference_h_h - 2 * self.reference_d_h)
        )

        # sparse frequencies
        self.freqs = freqs
        
        # middle bin frequencies
        self.f_m = f_m

        # prepare kwargs for online evaluation
        template_gen_kwargs["squeeze"] = False
        self.template_gen_kwargs = template_gen_kwargs

    def get_ll(
        self,
        params,
        return_extracted_snr=False,
        phase_maximize=False,
        **waveform_kwargs
    ):
        r"""Compute the log-Likelihood.

        Evaluates each template directly on the sparse grid, forms the ratio
        ``h / h0`` and combines it with the summary constants in the native kernel, then
        adds the directly computed outer-buffer terms. Uses ``reference_d_d`` for
        ``<d|d>``.

        Args:
            params (np.ndarray): Parameters for evaluating log-Likelihood, unpacked
                row by row into ``template_gen``: shape ``(num_params,)`` for one binary
                or ``(num_params, num_bin_all)`` for several (BBHWaveformFD order
                ``m1, m2, chi1z, chi2z, distance, phi_ref, f_ref, inc, lam, beta, psi,
                t_ref``).
            return_extracted_snr (bool, optional): If ``True``, also return
                :math:`\mathrm{Re}\langle d|h\rangle / \sqrt{\mathrm{Re}\langle h|h\rangle}`
                as a second column. Default ``False``.
            phase_maximize (bool, optional): If ``True``, use :math:`|\langle d|h\rangle|`
                (the phase-maximized log-Likelihood, and snr if
                ``return_extracted_snr=True``). Default ``False``.
            **waveform_kwargs: Keyword arguments for ``template_gen``. ``direct``,
                ``compress`` and ``squeeze`` are always taken from
                ``self.template_gen_kwargs`` and ``freqs`` is overwritten; other keys of
                ``self.template_gen_kwargs`` fill in when absent.

        Returns:
            np.ndarray: Log-likelihoods, shape ``(num_bin_all,)``, or
            ``(num_bin_all, 2)`` columns ``[log-Likelihood, snr]`` with
            ``return_extracted_snr=True``.
        """

        # store info
        self.phase_maximize = phase_maximize
        self.return_extracted_snr = return_extracted_snr

        # setup kwargs
        all_kwargs_keys = list(
            set(list(waveform_kwargs.keys()) + list(self.template_gen_kwargs.keys()))
        )

        for key in all_kwargs_keys:
            if key in ["direct", "compress", "squeeze"]:
                waveform_kwargs[key] = self.template_gen_kwargs[key]
            else:
                if key in self.template_gen_kwargs:
                    waveform_kwargs[key] = waveform_kwargs.get(
                        key, self.template_gen_kwargs[key]
                    )

        # set the frequencies at which the waveform is evaluated
        waveform_kwargs["freqs"] = self.freqs

        # compute the new sparse template
        self.h_sparse = self.template_gen(*params, **waveform_kwargs)

        # compute complex residual
        r = self.h_sparse / self.h0_sparse

        # initialize container for inner products term
        self.hdyn_d_h = self.xp.zeros(
            self.template_gen.num_bin_all, dtype=self.xp.complex128
        )
        self.hdyn_h_h = self.xp.zeros(
            self.template_gen.num_bin_all, dtype=self.xp.complex128
        )

        # adjust the residuals for entry into C
        residuals_in = r.transpose((2, 1, 0)).flatten()

        self.like_gen(
            self.hdyn_d_h,
            self.hdyn_h_h,
            residuals_in,
            self.data_constants,
            self.freqs,
            self.template_gen.num_bin_all,
            len(self.freqs),
            3,
        )

                # need to add any contributions to d_h and h_h outside of the heterodyne in the buffer area
        outer_freqs = self.outer_freqs
        waveform_kwargs["freqs"] = outer_freqs
        waveform_kwargs_tmp = waveform_kwargs.copy()
        # waveform_kwargs_tmp["direct"] = False

        # TODO: maybe don't do this direct?
        self.h_outer = self.template_gen(*params, **waveform_kwargs_tmp)

        # TODO: trapz?
        # if self.xp.any(self.h_outer):
        #     breakpoint()

        self.outer_h_h = 4 * self.df * self.xp.sum((self.h_outer.conj() * self.h_outer) / self.S_n_outer, axis=(-1, -2))
        self.outer_d_h = 4 * self.df * self.xp.sum((self.h_outer.conj() * self.d_outer) / self.S_n_outer, axis=(-1, -2))
        
        tmp_outer_lower_edge_h_h = 4 * self.df * self.xp.sum((self.h_outer[:, :, 0].conj() * self.h_outer[:, :, 0]) / self.S_n_outer[None, :, 0], axis=-1)
        tmp_outer_lower_edge_d_h = 4 * self.df * self.xp.sum((self.h_outer[:, :, 0].conj() * self.d_outer[None, :, 0]) / self.S_n_outer[None, :, 0], axis=-1)
        
        tmp_outer_upper_edge_h_h = 4 * self.df * self.xp.sum((self.h_outer[:, :, -1].conj() * self.h_outer[:, :, -1]) / self.S_n_outer[None, :, -1], axis=-1)
        tmp_outer_upper_edge_d_h = 4 * self.df * self.xp.sum((self.h_outer[:, :, -1].conj() * self.d_outer[None, :, -1]) / self.S_n_outer[None, :, -1], axis=-1)
        
        fix = (tmp_outer_lower_edge_h_h.real > 0.001) | (tmp_outer_upper_edge_h_h.real > 0.001) | (tmp_outer_lower_edge_d_h.real > 0.001) | (tmp_outer_upper_edge_d_h.real > 0.001)
        if self.xp.any(fix):
            # TODO: need to look at this further and relate to the buffer
            self.outer_h_h[fix] = -1e300
            self.outer_d_h[fix] = -1e300
            
        self.total_h_h = self.hdyn_h_h + self.outer_h_h
        self.total_d_h = self.hdyn_d_h + self.outer_d_h

        # if phase marginalize
        d_h_temp = (
            self.total_d_h if not self.phase_maximize else self.xp.abs(self.total_d_h)
        )

        # log-Likelihood
        out = -1 / 2.0 * (self.reference_d_d + self.total_h_h - 2 * d_h_temp).real

        # move to CPU if needed
        try:
            self.hdyn_h_h = self.hdyn_h_h.get()
            self.hdyn_d_h = self.hdyn_d_h.get()
            self.total_h_h = self.total_h_h.get()
            self.total_d_h = self.total_d_h.get()
            d_h_temp = d_h_temp.get()
            out = out.get()

        except AttributeError:
            pass

        if self.return_extracted_snr:
            return np.array([out, d_h_temp.real / np.sqrt(self.total_h_h.real)]).T
        else:
            return out


def searchsorted2d_vec(a, b, xp=None, **kwargs):
    """Row-wise ``searchsorted`` in one vectorized call.

    Returns, for every row ``i``, ``xp.searchsorted(a[i], b_i, **kwargs)`` where ``b_i``
    is ``b`` (1D, shared by all rows) or ``b[i]`` (2D). Each row is offset by
    ``i * max_num`` (``max_num`` = the larger value range of ``a`` and ``b``, plus 1) so a
    single flat ``searchsorted`` suffices; this assumes ``a`` and ``b`` cover comparable
    value ranges (as for frequency grids of one band). Used by
    :meth:`NewHeterodynedLikelihood.init_heterodyne_info`; copies of this helper also live
    in gpubackendtools and LISA Analysis Tools.

    Args:
        a (xp.ndarray): Shape ``(m, n)``, each row sorted ascending.
        b (xp.ndarray): Values to locate, shape ``(k,)`` or ``(m, k)``.
        xp (module, optional): ``numpy`` or ``cupy``. Default ``None`` means ``numpy``.
        **kwargs: Forwarded to ``xp.searchsorted`` (e.g. ``side="right"``).

    Returns:
        xp.ndarray: Integer indices of shape ``(m, k)``, each in ``[0, n]`` relative to
        its row.
    """
    if xp is None:
        xp = np
    m, n = a.shape
    max_num = xp.maximum(a.max() - a.min(), b.max() - b.min()) + 1
    r = max_num * xp.arange(a.shape[0])[:, None]
    p = xp.searchsorted((a + r).ravel(), (b + r).ravel(), **kwargs).reshape(m, -1)
    return p - n * (xp.arange(m)[:, None])


class NewHeterodynedLikelihood(BBHxParallelModule):
    r"""Compute the Heterodyned log-Likelihood (multi-reference, GPU only).

    Heterodyning involves separating the fast and slow evolutions when comparing
    signals. This involves comparing a reference template to the data stream and determining
    various quantities at the full frequency resolution of the data stream. Then, during
    online computation, the log-Likelihood is determined by computing a new waveform
    on a sparse frequency grid and comparing it to the reference waveform on the same
    sparse grid. The practical aspect is the computation of a smaller number of frequency
    points which lowers the required  memory and increases the speed of the computation.
    More information on the general method can be found in
    `arXiv:2109.02728 <https://arxiv.org/abs/2109.02728>`_. We implement the method
    as described in `arXiv:1806.08792 <https://arxiv.org/abs/1806.08792>`_.

    GPU-only multi-reference variant of :class:`HeterodynedLikelihood`. Each of
    ``num_bin`` reference templates gets its own sparse grid (``length_f_het`` log-spaced
    frequencies over the band where that reference is non-zero) and its own summary
    constants A0, A1, B0, B1, computed against data set ``data_index[j]`` and PSD set
    ``noise_index[j]`` by the native ``new_hdyn_prep`` kernel; :meth:`get_ll` then picks a
    constants set per template (``constants_index``). There is no outer buffer. The
    native kernels exist only on CUDA backends (``None`` on CPU) and the setup frees the
    cupy memory pool. The only known caller is ``MBHSpecialMove`` in LISA Analysis Tools
    (not on the production path).

    This class has GPU capabilities.

    Args:
        template_gen (obj): Waveform generator with the
            :class:`~bbhx.waveformbuild.BBHWaveformFD` call signature and output layout
            (it must expose ``num_bin_all``).
        data_freqs (xp.ndarray): Frequencies for the data stream (Hz), 1D and evenly
            spaced (cupy).
        data_channels (xp.ndarray): Complex data, shape
            ``(num_data_sets, nchannels, len(data_freqs))``.
        psd (xp.ndarray): PSDs, shape ``(num_noise_sets, nchannels, len(data_freqs))``.
        reference_template_params (np.ndarray): Reference parameters, one ROW per
            reference: shape ``(num_bin, num_params)`` (a 1D vector is one reference).
        length_f_het (int): Number of sparse frequencies per reference.
        data_index (xp.ndarray, optional): Data set of each reference, length
            ``num_bin``; must already be ``int32``. Default ``None`` means all 0.
        noise_index (xp.ndarray, optional): PSD set of each reference, length ``num_bin``
            (cast to ``int32``). Default ``None`` means all 0.
        template_gen_kwargs (dict, optional): Keyword arguments for the sparse templates;
            ``direct=True``, ``compress=True`` and ``squeeze=False`` are forced. Modified
            in place. Default ``{}``.
        reference_gen_kwargs (dict, optional): Keyword arguments for the dense references;
            ``fill=True``, ``compress=True`` and ``squeeze=False`` are forced, and
            ``direct`` becomes ``False`` (with ``length`` defaulting to ``8096``) unless
            ``direct=True`` is given. Modified in place. Default ``{}``.
        noise_kwargs_AE (dict, optional): Accepted and ignored. Default ``{}``.
        noise_kwargs_T (dict, optional): Accepted and ignored. Default ``{}``.
        force_backend (str, optional): Backend name; must resolve to a CUDA backend
            (``"cuda13x"``, ``"cuda"``, ...), see
            :class:`~bbhx.utils.parallelbase.BBHxParallelModule`. Must be given: this
            class defines no ``supported_backends``, so the default ``None`` raises
            ``NotImplementedError``.

    Attributes:
        reference_d_d (xp.ndarray): :math:`\langle d|d\rangle` of each data set,
            shape ``(num_data_sets,)``.
        reference_h_h (xp.ndarray): :math:`\langle h|h\rangle` of each reference,
            shape ``(num_bin,)``.
        reference_d_h (xp.ndarray): :math:`\langle d|h\rangle` of each reference.
        reference_ll (xp.ndarray): Log-likelihood of each reference.
        num_constants_sets (int): Number of constants sets (``len(reference_ll)``).
        hdyn_d_h (np.ndarray): Heterodyned complex :math:`\langle d|h\rangle` of the
            test templates (last call).
        hdyn_h_h (np.ndarray): Heterodyned complex :math:`\langle h|h\rangle` of the
            test templates (last call).
        h0_sparse (xp.ndarray): Sparse references, shape
            ``(num_bin, nchannels, length_f_het)``.
        h_sparse (xp.ndarray): Sparse test templates (last call).
        d (xp.ndarray): The data.
        psd (xp.ndarray): The PSDs.
        data_length (int): Number of data frequencies.
        nchannels (int): Number of channels (``data_channels.shape[1]``).
        A0_out (xp.ndarray): Summary constant A0, shape
            ``(num_bin, nchannels, length_f_het)``; ``A1_out``, ``B0_out`` and ``B1_out``
            hold A1, B0 and B1 the same way.
        data_constants (xp.ndarray): The four arrays above, stacked and flattened.
        f_dense (xp.ndarray): Frequencies for the data stream (1D).
        freqs (xp.ndarray): Sparse frequencies, shape ``(num_bin, length_f_het)``.
        f_m (xp.ndarray): Bin mid-point frequencies, same shape (first column 0).
        length_f_het (int): Length of each sparse grid.
        template_gen (obj): The waveform generator.
        return_extracted_snr (bool): Return the snr in addition to the Likelihood.
        phase_maximize (bool): If ``True``, compute the phase-maximized
            log-Likelihood (and snr if ``return_extracted_snr==True``).
    """

    def __init__(
        self,
        template_gen,
        data_freqs,
        data_channels,
        psd,
        reference_template_params,
        length_f_het,
        data_index=None, 
        noise_index=None,
        template_gen_kwargs={},
        reference_gen_kwargs={},
        noise_kwargs_AE={},
        noise_kwargs_T={},
        force_backend=None,
    ):

        # direct based on GPU usage
        super().__init__(force_backend=force_backend)

        # store all input information
        self.template_gen = template_gen
        self.f_dense = data_freqs
        self.d = data_channels
        self.psd = psd
        self.length_f_het = length_f_het

        # calculate all quantites related to the reference template
        self.init_heterodyne_info(
            reference_template_params,
            template_gen_kwargs=template_gen_kwargs,
            reference_gen_kwargs=reference_gen_kwargs,
            data_index=data_index,
            noise_index=noise_index,
        )

    @property
    def citation(self):
        """Citation strings for this class.

        Warning:
            Raises ``NameError``: the citation strings (``katz_citations``, ...) are no
            longer defined in this module (``bbhx.utils.citations`` was removed).
        """
        return katz_citations + Cornish_Heterodyning + Rel_Bin_citation

    @property
    def xp(self) -> object:
        """Numpy or Cupy"""
        return self.backend.xp

    @property
    def like_gen(self) -> callable:
        """Native ``new_hdyn_like`` kernel (GPU only; ``None`` on CPU)."""
        return self.backend.new_hdyn_like
    
    @property
    def prep_gen(self) -> callable:
        """Native ``new_hdyn_prep`` kernel (GPU only; ``None`` on CPU)."""
        return self.backend.new_hdyn_prep

    def init_heterodyne_info(
        self,
        reference_template_params,
        template_gen_kwargs={},
        reference_gen_kwargs={},
        data_index=None, 
        noise_index=None,
    ):
        """Prepare all information for Heterodyning.

        Run by the constructor; call it again to re-centre (the data and PSDs are kept).
        Builds per-reference sparse grids, the summary constants (native ``prep_gen``
        kernel) and the ``reference_*`` attributes (see the class docstring), then frees
        the cupy memory pool.

        Args:
            reference_template_params (np.ndarray): Reference parameters, one row per
                reference, shape ``(num_bin, num_params)`` (passed through
                ``np.atleast_2d``).
            template_gen_kwargs (dict, optional): Keyword arguments for the sparse
                templates; ``squeeze=False``, ``compress=True`` and ``direct=True`` are
                forced. Modified in place and stored as ``self.template_gen_kwargs``.
                Default ``{}``.
            reference_gen_kwargs (dict, optional): Keyword arguments for the dense
                references; ``squeeze=False``, ``compress=True`` and ``fill=True`` are
                forced; ``direct`` becomes ``False`` (with ``length`` defaulting to
                ``8096``) unless ``direct=True`` is given. Modified in place. Default
                ``{}``.
            data_index (xp.ndarray, optional): ``int32`` data set of each reference.
                Default ``None`` means all 0.
            noise_index (xp.ndarray, optional): PSD set of each reference (cast to
                ``int32``). Default ``None`` means all 0.

        Raises:
            AssertionError: Index arrays of the wrong length (or a non-``int32``
                ``data_index``), or inconsistent sparse-bin bookkeeping.

        Warning:
            A sparse-bin index above ``length_f_het`` but below ``1e4`` triggers a
            ``breakpoint()`` (debugger stop) left in the code.
        """

        # TODO: need this?
        # if self.backend.uses_cupy:
        #     self.xp.cuda.runtime.setDevice(self.gpu)
            
        reference_template_params = np.atleast_2d(reference_template_params)
        # add the necessary kwargs for the initial template generation process.
        template_gen_kwargs["squeeze"] = False
        template_gen_kwargs["compress"] = True
        template_gen_kwargs["direct"] = True

        num_bin = len(reference_template_params)
        self.data_length = data_length = self.d.shape[-1]
        self.nchannels = nchannels = self.d.shape[1]

        # need to be just outside the values of the waveform
        minF = self.f_dense.min() * 0.999999999999
        maxF = self.f_dense.max() * 1.000000000001

        if minF == 0.0:
            minF = 1e-6
            
        reference_gen_kwargs["squeeze"] = False
        reference_gen_kwargs["compress"] = True
        reference_gen_kwargs["fill"] = True

        # setup for reference template generation
        if "direct" in reference_gen_kwargs and reference_gen_kwargs["direct"] is True:
            pass
        else:
            reference_gen_kwargs["direct"] = False
            if "length" not in reference_gen_kwargs:
                reference_gen_kwargs["length"] = 8096

        # Heterodyning grid
        freqs = self.xp.logspace(
            self.xp.log10(minF), self.xp.log10(maxF), self.length_f_het
        )

        # generate dense reference template
        h0 = self.template_gen(
            *reference_template_params.T, freqs=self.f_dense, **reference_gen_kwargs
        )[:, :nchannels].flatten().copy().reshape(reference_template_params.shape[0], nchannels, data_length)

        # generate sparse reference template
        h0_temp = self.template_gen(
            *reference_template_params.T, freqs=freqs, **template_gen_kwargs
        )[:, :nchannels]

        # get rid of places where freqs are zero and narrow boundaries
        freqs_tmp = np.tile(freqs, (len(reference_template_params), 1)).copy()

        freqs_tmp[(self.xp.abs(h0_temp[:, 0]) == 0.0)] = 1e300
        min_freqs = freqs_tmp.min(axis=-1)

        freqs_tmp = np.tile(freqs, (len(reference_template_params), 1)).copy()
        freqs_tmp[(self.xp.abs(h0_temp[:, 0]) == 0.0)] = -1e300
        max_freqs = freqs_tmp.max(axis=-1)

        freqs = self.xp.logspace(
            self.xp.log10(min_freqs),
            self.xp.log10(max_freqs),
            self.length_f_het,
        ).T.copy()

        # regenerate at only non-zero values of the waveform
        self.h0_sparse = self.template_gen(
            *reference_template_params.T, freqs=freqs[:, None, :], **template_gen_kwargs
        )[:, :nchannels]

        # find which sparse array bins the dense frequencies fit into
        bins = searchsorted2d_vec(freqs, self.f_dense, xp=self.xp, side="right") - 1

        special_bins = self.xp.arange(num_bin)[:, None] * int(1e6) + bins

        # make sure sorted
        assert not self.xp.any(self.xp.diff(special_bins, axis=-1) < 0)
        uni_special_bins, uni_start_index_special_bins, uni_count_special_bins = np.unique(special_bins.flatten(), return_counts=True, return_index=True)

        binary_ind = np.floor(uni_special_bins / 1e6).astype(int)
        het_bin_ind = uni_special_bins - binary_ind * int(1e6)

        if np.any((het_bin_ind > self.length_f_het) & (het_bin_ind < int(1e4))):
            breakpoint()

        het_bin_ind[het_bin_ind > int(1e4)] = -1
        binary_ind[het_bin_ind == -1] = -1

        assert het_bin_ind.min() == -1 and het_bin_ind.max() < self.length_f_het

        inds_all = -self.xp.ones((num_bin, self.length_f_het), dtype=self.xp.int32)
        inds_all[(binary_ind[binary_ind != -1], het_bin_ind[het_bin_ind != -1])] = uni_start_index_special_bins[het_bin_ind >= 0]
        inds_all[inds_all >= 0] = (inds_all - self.xp.arange(num_bin)[:, None] * special_bins.shape[-1])[inds_all >= 0]
        
        assert self.xp.all(inds_all < special_bins.shape[1])

        inds_all[:, -1] = special_bins.shape[1]

        start_inds_all = self.xp.zeros_like(inds_all)
        end_inds_all = self.xp.zeros_like(inds_all)

        start_inds_all[:, 1:] = inds_all[:, :-1].copy()
        end_inds_all[:, 1:] = inds_all[:, 1:].copy()

        lengths = (end_inds_all - start_inds_all).copy()

        f_m = self.xp.zeros_like(freqs)
        # get frequency at middle of the bin
        f_m[:, 1:] = (freqs[:, 1:] + freqs[:, :-1]) / 2

        df = (self.f_dense[1] - self.f_dense[0]).item()

        self.A0_out = self.xp.zeros((num_bin, nchannels, (self.length_f_het)), dtype=complex)
        self.A1_out = self.xp.zeros((num_bin, nchannels, (self.length_f_het)), dtype=complex)
        self.B0_out = self.xp.zeros((num_bin, nchannels, (self.length_f_het)), dtype=complex)
        self.B1_out = self.xp.zeros((num_bin, nchannels, (self.length_f_het)), dtype=complex)
        
        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            assert len(data_index) == num_bin
            assert data_index.dtype == self.xp.int32

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            assert len(data_index) == num_bin
            data_index = data_index.astype(self.xp.int32)

        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            assert len(noise_index) == num_bin
            noise_index = noise_index.astype(self.xp.int32)
        
        self.prep_gen(self.A0_out, self.A1_out, self.B0_out, self.B1_out,
        h0, self.d, self.psd, f_m, df, self.f_dense, data_index, noise_index,
        start_inds_all, lengths, self.length_f_het, num_bin, data_length, nchannels)

        self.data_constants = self.xp.array([self.A0_out, self.A1_out, self.B0_out, self.B1_out]).flatten()
        # reference quantities

        self.reference_d_d = self.xp.zeros(self.d.shape[0], dtype=self.xp.float64)
        for i in range(self.d.shape[0]):
            self.reference_d_d[i] = self.xp.sum(4 * (self.d[i].conj() * self.d[i]) / self.psd[i] * df).real

        self.reference_h_h = self.xp.sum(self.B0_out, axis=(1, 2)).real

        self.reference_d_h = self.xp.sum(self.A0_out, axis=(1, 2)).real

        self.reference_ll = (
            -1 / 2 * (self.reference_d_d + self.reference_h_h - 2 * self.reference_d_h)
        )

        self.num_constants_sets = len(self.reference_ll)

        # sparse frequencies
        self.freqs = freqs

        # middle bin frequencies
        self.f_m = f_m

        # prepare kwargs for online evaluation
        template_gen_kwargs["squeeze"] = False
        self.template_gen_kwargs = template_gen_kwargs
        self.xp.get_default_memory_pool().free_all_blocks()
        return

    def get_ll(
        self,
        params,
        return_extracted_snr=False,
        phase_maximize=False,
        constants_index=None,
        **waveform_kwargs
    ):
        r"""Compute the log-Likelihood.

        Template ``i`` is evaluated on the sparse grid of constants set
        ``constants_index[i]`` (``freqs`` of shape ``(num_bin_all, 1, length_f_het)``, so a
        single-mode template generator is assumed), divided by that set's sparse reference
        and combined with its constants by the native kernel. ``<d|d>`` is
        ``reference_d_d[constants_index]``: ``reference_d_d`` holds one value per DATA set,
        so this is right only when reference ``j`` uses data set ``j`` (or there is a
        single data set and ``constants_index`` is all 0).

        Args:
            params (np.ndarray): Parameters, one ROW per template: shape
                ``(num_bin_all, num_params)`` (transposed into ``template_gen``; note the
                opposite layout to :meth:`HeterodynedLikelihood.get_ll`).
            return_extracted_snr (bool, optional): If ``True``, also return
                :math:`\mathrm{Re}\langle d|h\rangle / \sqrt{\mathrm{Re}\langle h|h\rangle}`
                as a second column. Default ``False``.
            phase_maximize (bool, optional): If ``True``, use :math:`|\langle d|h\rangle|`
                (the phase-maximized log-Likelihood, and snr if
                ``return_extracted_snr=True``). Default ``False``.
            constants_index (xp.ndarray, optional): Constants set (reference) of each
                template, length ``num_bin_all`` (cast to ``int32``; must be below
                ``num_constants_sets``). Default ``None`` means all 0.
            **waveform_kwargs: Keyword arguments for ``template_gen``. ``direct``,
                ``compress`` and ``squeeze`` are always taken from
                ``self.template_gen_kwargs`` and ``freqs`` is overwritten; other keys of
                ``self.template_gen_kwargs`` fill in when absent.

        Returns:
            np.ndarray: Log-likelihoods, shape ``(num_bin_all,)``, or
            ``(num_bin_all, 2)`` columns ``[log-Likelihood, snr]`` with
            ``return_extracted_snr=True``.
        """

        # store info
        self.phase_maximize = phase_maximize
        self.return_extracted_snr = return_extracted_snr

        # setup kwargs
        all_kwargs_keys = list(
            set(list(waveform_kwargs.keys()) + list(self.template_gen_kwargs.keys()))
        )

        for key in all_kwargs_keys:
            if key in ["direct", "compress", "squeeze"]:
                waveform_kwargs[key] = self.template_gen_kwargs[key]
            else:
                if key in self.template_gen_kwargs:
                    waveform_kwargs[key] = waveform_kwargs.get(
                        key, self.template_gen_kwargs[key]
                    )

        if constants_index is None:
            constants_index = self.xp.zeros(params.shape[0], dtype=self.xp.int32)
        else:
            assert len(constants_index) == params.shape[0]
            constants_index = constants_index.astype(self.xp.int32)
            assert np.all(constants_index < self.num_constants_sets)

        # set the frequencies at which the waveform is evaluated
        waveform_kwargs["freqs"] = self.freqs[constants_index][:, None, :]

        # compute the new sparse template
        self.h_sparse = self.template_gen(*params.T, **waveform_kwargs)[:, :self.nchannels]

        # compute complex residual
        r = (self.h_sparse / self.h0_sparse[constants_index])  # .flatten()
        freqs = self.freqs  # .flatten().copy()

        # initialize container for inner products term
        self.hdyn_d_h = self.xp.zeros(
            self.template_gen.num_bin_all, dtype=self.xp.complex128
        )
        self.hdyn_h_h = self.xp.zeros(
            self.template_gen.num_bin_all, dtype=self.xp.complex128
        )

        # adjust the residuals for entry into C
        # residuals_in = r.transpose((2, 1, 0)).flatten()

        self.like_gen(
            self.hdyn_d_h, 
            self.hdyn_h_h,
            r.flatten().copy(), 
            self.data_constants,
            freqs.flatten().copy(),
            constants_index,
            self.template_gen.num_bin_all,
            self.length_f_het,
            self.nchannels,
            self.num_constants_sets,  # num constants sets
        )

        # if phase marginalize
        d_h_temp = (
            self.hdyn_d_h if not self.phase_maximize else self.xp.abs(self.hdyn_d_h)
        )

        # log-Likelihood
        out = -1 / 2.0 * (self.reference_d_d[constants_index] + self.hdyn_h_h - 2 * d_h_temp).real

        # move to CPU if needed
        try:
            self.hdyn_h_h = self.hdyn_h_h.get()
            self.hdyn_d_h = self.hdyn_d_h.get()
            d_h_temp = d_h_temp.get()
            out = out.get()

        except AttributeError:
            pass

        if self.return_extracted_snr:
            return np.array([out, d_h_temp.real / np.sqrt(self.hdyn_h_h.real)]).T
        else:
            return out
