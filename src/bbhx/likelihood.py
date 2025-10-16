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

import numpy as np

from .utils.constants import *
from .utils.parallelbase import BBHxParallelModule

from lisatools.sensitivity import SensitivityMatrix, AET1SensitivityMatrix


class Likelihood(BBHxParallelModule):
    """Fast Base Likelihood Class for MBHBs

    This class computes the graitational wave Likelihood as a direct sum over
    frequecy bins. It only sums over the frequencies where the MBHB signal
    exists. Therefore, larger mass waveforms are faster because there are less
    frequencies. This class computes:
    :math:`\\mathcal{L}\\propto-1/2\\langle d-h|d-h\\rangle=-1/2\\left(\\langle d|d\\rangle \\langle h|h\\rangle - 2\\langle d|h\\rangle\\right)`.

    This class has GPU capability.

    Args:
        template_gen (obj): Waveform generation class that returns a tuple of
            (list of template arrays, start indices, lengths). See
            :class:`bbhx.waveform.BBHWaveformFD` for more information on this
            return type.
        data_freqs (double xp.ndarray): Frequencies for the data stream. ``data_freqs``
            should be a numpy (cupy) array if running on the CPU (GPU).
        data_channels (complex128 xp.ndarray): Data stream. 2D array of shape: ``(3, len(data_freqs))``.
            It is assumed there are 3 channels. ``data_channels``
            should be a numpy (cupy) array if running on the CPU (GPU).
        psd (double xp.ndarray): Power Spectral Density in the noise:math:`S_n(f)`.
            2D array of shape: ``(3, len(data_freqs))``.
            It is assumed there are 3 channels. ``psd``
            should be a numpy (cupy) array if running on the CPU (GPU).
        force_backend (str, optional): ``"cpu"'', ``"gpu"'', ``"cuda"'', ``"cuda12x"'', or ``"cuda11x"''.

    Attributes:
        d_d (double): :math:`\\langle d|d\\rangle` inner product value.
        data_channels (complex128 np.ndarray): Data stream. 1D flattened array
            of shape: ``(3, len(data_freqs))``. **Note** ``data_channels`` should
            be multiplied by ``psd`` before input into this class.
        data_freqs (double np.ndarray): Frequencies for the data stream (1D).
        data_stream_length (int): Length of data.
        noise_factors (double xp.ndarray): :math:`\\sqrt{\\frac{\\Delta f}{S_n(f)}}`.
            1D flattened array of shape: ``(3, len(data_freqs))``.
        psd (double xp.ndarray): Power Spectral Density in the noise:math:`S_n(f)`.
            1D flattened array of shape: ``(3, len(data_freqs))``.
        template_gen (obj): Waveform generation class that returns a tuple of
            (list of template arrays, start indices, lengths). See
            :class:`bbhx.waveform.BBHWaveformFD` for more information on this
            return type.
        phase_marginalize (bool): If ``True``, compute the phase-marginalized
            log-Likelihood (and snr if ``return_extracted_snr==True``).
        return_extracted_snr (bool): Return the snr in addition to the Likeilihood.


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

        # assumes data_channels is already factored by psd
        self.d_d = (
            4 * self.xp.sum((self.data_channels.conj() * self.data_channels)).real
        ).item()

    @classmethod
    def supported_backends(cls) -> list:
        return ["bbhx_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    @property
    def like_gen(self):
        """Likelihood for either GPU or CPU."""
        return self.backend.direct_like_wrap

    @property
    def xp(self):
        """Cupy or Numpy"""
        return self.backend.xp

    @property
    def citation(self):
        return katz_citations

    def get_ll(
        self,
        params,
        return_extracted_snr=False,
        phase_marginalize=False,
        **waveform_kwargs
    ):
        """Compute the log-Likelihood

        params (double np.ndarray): Parameters for evaluating log-Likelihood.
            ``params.shape=(num_params,)`` if 1D or
            ``params.shape=(num_params, num_bin_all)`` if 2D for more than
            one binary.
        return_extracted_snr (bool, optional): If ``True``, return
            :math:`\\langle d|h\\rangle\\ / \\sqrt{\\langle h|h\\rangle}` as a second entry
            of the return array. This produces a return array of
            ``xp.array([log likelihood, snr]).T``. If ``False``, just return
            the log-Likelihood array.
        phase_marginalize (bool, optional): If ``True``, compute the phase-marginalized
            log-Likelihood (and snr if ``return_extracted_snr==True``).
        **waveform_kwargs (dict, optional): Keyword arguments for waveform
            generator.

        Returns:
            np.ndarray: log-Likelihoods or ``np.array([log-Likelihoods, snr]).T``

        """

        # store info
        self.phase_marginalize = phase_marginalize
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

        device = self.xp.cuda.runtime.getDevice()
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
        d_h_temp = self.d_h if not self.phase_marginalize else self.xp.abs(self.d_h)
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
    """Compute the Heterodyned log-Likelihood

    Heterdyning involves separating the fast and slow evolutions when comparing
    signals. This involves comparing a reference template to the data stream and determining
    various quantities at the full frequency resolution of the data stream. Then, during
    online computation, the log-Likelihood is determined by computing a new waveform
    on a sparse frequency grid and comparing it to the reference waveform on the same
    sparse grid. The practical aspect is the computation of a smaller number of frequency
    points which lowers the required  memory and increases the speed of the computation.
    More information on the general method can be found in
    `arXiv:2109.02728 <https://arxiv.org/abs/2109.02728>`_. We implement the method
    as described in `arXiv:1806.08792 <https://arxiv.org/abs/1806.08792>`_.

    This class also works with higher order harmonic modes, but it has not been tested extensivally.
    It only does a direct summation over the modes rather than heterodyning per mode. So, it is less reliable,
    but in practice it produces a solid posterior distribution.

    This class has GPU capabilities.

    Args:
        template_gen (obj): Waveform generation class that returns a tuple of
            (list of template arrays, start indices, lengths). See
            :class:`bbhx.waveform.BBHWaveformFD` for more information on this
            return type.
        data_freqs (double xp.ndarray): Frequencies for the data stream. ``data_freqs``
            should be a numpy (cupy) array if running on the CPU (GPU).
        data_channels (complex128 xp.ndarray): Data stream. 2D array of shape: ``(3, len(data_freqs))``.
            It is assumed there are 3 channels. ``data_channels``
            should be a numpy (cupy) array if running on the CPU (GPU).
        reference_template_params (np.ndarray): Parameters for the reference template for
            ``template_gen``.
        template_gen_kwargs (dict, optional): Keywords arguments for generating the
            template with ``template_gen``. It will automatically add/change ``direct``
            and ``compress`` keyword arguments to ``True``. The keyword argument
            ``squeeze`` is automatically set to ``True`` for the initial setup and then
            automatically set to ``False`` for online computations. If you so choose (not
            recomended), you can change these kwargs for online running using the
            ``**waveform_kwargs`` setup in the ``self.get_ll`` method. However,
            if you include in this adjustment ``direct``, ``compress``, or ``squeeze``
            will still automatically be overwritten.
            (Default: ``{}``)
        reference_gen_kwargs (dict, optional): Keywords arguments for generating the
            reference template with ``template_gen``. It will automatically add/change
            ``fill``, ``compress``, and ``squeeze`` keyword arguments to ``True``.
            These waveforms can be produced without interpolation with ``direct = True``.
            If ``length`` keyword argument is given and ``direct=False``, the
            waveform will be interpolated. If ``length`` and ``direct`` are not given,
            it will interpolate the signal with ``length=8096``.
            (Default: ``{}``)
        sens_mat (SensitivityMatrix, optional): :class:`SensitivityMatrix` object representing the AET channels.
            If ``None``, defaults to class:`AET1SensitivityMatrix`. (default: ``None``)
        force_backend (str, optional): ``"cpu"'', ``"gpu"'', ``"cuda"'', ``"cuda12x"'', or ``"cuda11x"''.

    Attributes:
        reference_d_d (double): :math:`\\langle d|d\\rangle` inner product value.
        reference_h_h(double): :math:`\\langle h|h\\rangle` inner product value
            for the reference template.
        reference_d_h (double): :math:`\\langle d|h\\rangle` inner product value
            for the reference template.
        reference_ll (double): log-Likelihood value for the reference template.
        hdyn_d_h (complex128 xp.ndarray): Heterodyned :math:`\\langle d|h\\rangle`
            inner product values for the test templates.
        hdyn_h_h (complex128 xp.ndarray): Heterodyned :math:`\\langle d|h\\rangle`
            inner product values for the test templates.
        h0_sparse (xp.ndarray): Array with sparse waveform for reference parameters.
        h_sparse (xp.ndarray): Array with sparse waveform for test parameters.
        d (complex128 xp.ndarray): Data stream. 1D flattened array
            of shape: ``(3, len(data_freqs))``.
        data_stream_length (int): Length of data.
        data_constants (xp.ndarray): Flattened array container holding all heterodyning
            constants needed: A0, A1, B0, B1.
        f_dense (xp.ndarray): Frequencies for the data stream (1D).
        freqs (xp.ndarray): Frequencies for sparse arrays.
        f_m (xp.ndarray): Frequency of mid-point in each sparse bin.
        length_f_het (int): Length of sparse array.
        psd (double xp.ndarray): :math:`\\sqrt{\\frac{\\Delta f}{S_n(f)}}`.
            1D flattened array of shape: ``(3, len(data_freqs))``.
        return_extracted_snr (bool): Return the snr in addition to the Likeilihood.
        phase_marginalize (bool): If ``True``, compute the phase-marginalized
            log-Likelihood (and snr if ``return_extracted_snr==True``).

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
        return ["bbhx_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    @property
    def sens_mat(self):
        """Sensitivity Matrix"""
        return self._sens_mat

    @sens_mat.setter
    def sens_mat(self, sens_mat):
        if sens_mat is None:
            _f_not_needed = np.logspace(-5, -1, 1000)
            sens_mat = AET1SensitivityMatrix(_f_not_needed)
        assert isinstance(sens_mat, SensitivityMatrix)
        self._sens_mat = sens_mat

    @property
    def like_gen(self):
        """C function on GPU/CPU"""
        return self.backend.hdyn_wrap

    @property
    def xp(self):
        """Numpy or Cupy"""
        return self.backend.xp

    @property
    def citation(self):
        """Citations for this class"""
        return katz_citations + Cornish_Heterodyning + Rel_Bin_citation

    def init_heterodyne_info(
        self,
        reference_template_params,
        template_gen_kwargs={},
        reference_gen_kwargs={},
    ):
        """Prepare all information for Heterdyning

        Args:
            reference_template_params (np.ndarray): Parameters for the reference template for
                ``template_gen``.
            template_gen_kwargs (dict, optional): Keywords arguments for generating the
                template with ``template_gen``. It will automatically add/change ``direct``
                and ``compress`` keyword arguments to ``True``. The keyword argument
                ``squeeze`` is automatically set to ``True`` for the initial setup and then
                automatically set to ``False`` for online computations. If you so choose (not
                recomended), you can change these kwargs for online running using the
                ``**waveform_kwargs`` setup in the ``self.get_ll`` method. However,
                if you include in this adjustment ``direct``, ``compress``, or ``squeeze``
                will still automatically be overwritten.
                (Default: ``{}``)
            reference_gen_kwargs (dict, optional): Keywords arguments for generating the
                reference template with ``template_gen``. It will automatically add/change
                ``fill``, ``compress``, and ``squeeze`` keyword arguments to ``True``.
                These waveforms can be produced without interpolation with ``direct = True``.
                If ``length`` keyword argument is given and ``direct=False``, the
                waveform will be interpolated. If ``length`` and ``direct`` are not given,
                it will interpolate the signal with ``length=8096``.
                (Default: ``{}``)

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

        self.sens_mat.update_frequency_arr(f_n_host)

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
        phase_marginalize=False,
        **waveform_kwargs
    ):
        """Compute the log-Likelihood

        params (double np.ndarray): Parameters for evaluating log-Likelihood.
            ``params.shape=(num_params,)`` if 1D or
            ``params.shape=(num_params, num_bin_all)`` if 2D for more than
            one binary.
        return_extracted_snr (bool, optional): If ``True``, return
            :math:`\\langle d|h\\rangle\\ / \\sqrt{\\langle h|h\\rangle}` as a second entry
            of the return array. This produces a return array of
            ``xp.array([log likelihood, snr]).T``. If ``False``, just return
            the log-Likelihood array.
        phase_marginalize (bool, optional): If ``True``, compute the phase-marginalized
            log-Likelihood (and snr if ``return_extracted_snr==True``).
        **waveform_kwargs (dict, optional): Keyword arguments for waveform
            generator. Some may be overwritten. See the main class docstring.

        Returns:
            np.ndarray: log-Likelihoods or ``np.array([log-Likelihoods, snr]).T``

        """

        # store info
        self.phase_marginalize = phase_marginalize
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
            self.total_d_h if not self.phase_marginalize else self.xp.abs(self.total_d_h)
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
    if xp is None:
        xp = np
    m, n = a.shape
    max_num = xp.maximum(a.max() - a.min(), b.max() - b.min()) + 1
    r = max_num * xp.arange(a.shape[0])[:, None]
    p = xp.searchsorted((a + r).ravel(), (b + r).ravel(), **kwargs).reshape(m, -1)
    return p - n * (xp.arange(m)[:, None])


class NewHeterodynedLikelihood(BBHxParallelModule):
    """Compute the Heterodyned log-Likelihood

    Heterdyning involves separating the fast and slow evolutions when comparing
    signals. This involves comparing a reference template to the data stream and determining
    various quantities at the full frequency resolution of the data stream. Then, during
    online computation, the log-Likelihood is determined by computing a new waveform
    on a sparse frequency grid and comparing it to the reference waveform on the same
    sparse grid. The practical aspect is the computation of a smaller number of frequency
    points which lowers the required  memory and increases the speed of the computation.
    More information on the general method can be found in
    `arXiv:2109.02728 <https://arxiv.org/abs/2109.02728>`_. We implement the method
    as described in `arXiv:1806.08792 <https://arxiv.org/abs/1806.08792>`_.

    This class also works with higher order harmonic modes. TODO: fill in.

    This class has GPU capabilities.

    Args:
        template_gen (obj): Waveform generation class that returns a tuple of
            (list of template arrays, start indices, lengths). See
            :class:`bbhx.waveform.BBHWaveformFD` for more information on this
            return type.
        data_freqs (double xp.ndarray): Frequencies for the data stream. ``data_freqs``
            should be a numpy (cupy) array if running on the CPU (GPU).
        data_channels (complex128 xp.ndarray): Data stream. 2D array of shape: ``(3, len(data_freqs))``.
            It is assumed there are 3 channels. ``data_channels``
            should be a numpy (cupy) array if running on the CPU (GPU).
        reference_template_params (np.ndarray): Parameters for the reference template for
            ``template_gen``.
        template_gen_kwargs (dict, optional): Keywords arguments for generating the
            template with ``template_gen``. It will automatically add/change ``direct``
            and ``compress`` keyword arguments to ``True``. The keyword argument
            ``squeeze`` is automatically set to ``True`` for the initial setup and then
            automatically set to ``False`` for online computations. If you so choose (not
            recomended), you can change these kwargs for online running using the
            ``**waveform_kwargs`` setup in the ``self.get_ll`` method. However,
            if you include in this adjustment ``direct``, ``compress``, or ``squeeze``
            will still automatically be overwritten.
            (Default: ``{}``)
        reference_gen_kwargs (dict, optional): Keywords arguments for generating the
            reference template with ``template_gen``. It will automatically add/change
            ``fill``, ``compress``, and ``squeeze`` keyword arguments to ``True``.
            These waveforms can be produced without interpolation with ``direct = True``.
            If ``length`` keyword argument is given and ``direct=False``, the
            waveform will be interpolated. If ``length`` and ``direct`` are not given,
            it will interpolate the signal with ``length=8096``.
            (Default: ``{}``)
        noise_kwargs_AE (dict, optional): Keywords arguments for generating the
            noise with ``noisepsd_AE`` (A & E Channel) option in the ``get_sensitivity`` function
            from ``lisatools`` TODO: add link. (Default: ``{}``)
        noise_kwargs_T (dict, optional): Keywords arguments for generating the
            noise with ``noisepsd_T`` (T Channel) option in the ``get_sensitivity`` function
            from ``lisatools`` TODO: add link. (Default: ``{}``)
        use_gpu (bool, optional): If ``True``, use GPU.

    Attributes:
        reference_d_d (double): :math:`\langle d|d\\rangle` inner product value.
        reference_h_h(double): :math:`\langle h|h\\rangle` inner product value
            for the reference template.
        reference_d_h (double): :math:`\langle d|h\\rangle` inner product value
            for the reference template.
        reference_ll (double): log-Likelihood value for the reference template.
        hdyn_d_h (complex128 xp.ndarray): Heterodyned :math:`\langle d|h\\rangle`
            inner product values for the test templates.
        hdyn_h_h (complex128 xp.ndarray): Heterodyned :math:`\langle d|h\\rangle`
            inner product values for the test templates.
        h0_sparse (xp.ndarray): Array with sparse waveform for reference parameters.
        h_sparse (xp.ndarray): Array with sparse waveform for test parameters.
        d (complex128 xp.ndarray): Data stream. 1D flattened array
            of shape: ``(3, len(data_freqs))``.
        data_stream_length (int): Length of data.
        data_constants (xp.ndarray): Flattened array container holding all heterodyning
            constants needed: A0, A1, B0, B1.
        f_dense (xp.ndarray): Frequencies for the data stream (1D).
        freqs (xp.ndarray): Frequencies for sparse arrays.
        f_m (xp.ndarray): Frequency of mid-point in each sparse bin.
        length_f_het (int): Length of sparse array.
        like_gen (obj): C/CUDA implementation of likelihood compuation.
        psd (double xp.ndarray): :math:`\\sqrt{\\frac{\\Delta f}{S_n(f)}}`.
            1D flattened array of shape: ``(3, len(data_freqs))``.
        template_gen (obj): Waveform generation class that returns a tuple of
            (list of template arrays, start indices, lengths). See
            :class:`bbhx.waveform.BBHWaveformFD` for more information on this
            return type.
        return_extracted_snr (bool): Return the snr in addition to the Likeilihood.
        phase_marginalize (bool): If ``True``, compute the phase-marginalized
            log-Likelihood (and snr if ``return_extracted_snr==True``).
        use_gpu (bool): If True, using GPU.
        xp (obj): Either numpy or cupy.



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
        """Citations for this class"""
        return katz_citations + Cornish_Heterodyning + Rel_Bin_citation

    @property
    def xp(self) -> object:
        return self.backend.xp

    @property
    def like_gen(self) -> callable:
        return self.backend.new_hdyn_like
    
    @property
    def prep_gen(self) -> callable:
        return self.backend.new_hdyn_prep

    def init_heterodyne_info(
        self,
        reference_template_params,
        template_gen_kwargs={},
        reference_gen_kwargs={},
        data_index=None, 
        noise_index=None,
    ):
        """Prepare all information for Heterdyning

        Args:
            reference_template_params (np.ndarray): Parameters for the reference template for
                ``template_gen``.
            template_gen_kwargs (dict, optional): Keywords arguments for generating the
                template with ``template_gen``. It will automatically add/change ``direct``
                and ``compress`` keyword arguments to ``True``. The keyword argument
                ``squeeze`` is automatically set to ``True`` for the initial setup and then
                automatically set to ``False`` for online computations. If you so choose (not
                recomended), you can change these kwargs for online running using the
                ``**waveform_kwargs`` setup in the ``self.get_ll`` method. However,
                if you include in this adjustment ``direct``, ``compress``, or ``squeeze``
                will still automatically be overwritten.
                (Default: ``{}``)
            reference_gen_kwargs (dict, optional): Keywords arguments for generating the
                reference template with ``template_gen``. It will automatically add/change
                ``fill``, ``compress``, and ``squeeze`` keyword arguments to ``True``.
                These waveforms can be produced without interpolation with ``direct = True``.
                If ``length`` keyword argument is given and ``direct=False``, the
                waveform will be interpolated. If ``length`` and ``direct`` are not given,
                it will interpolate the signal with ``length=8096``.
                (Default: ``{}``)
            noise_kwargs_AE (dict, optional): Keywords arguments for generating the
                noise with ``noisepsd_AE`` (A & E Channel) option in the ``get_sensitivity`` function
                from ``lisatools`` TODO: add link. (Default: ``{}``)
            noise_kwargs_T (dict, optional): Keywords arguments for generating the
                noise with ``noisepsd_T`` (T Channel) option in the ``get_sensitivity`` function
                from ``lisatools`` TODO: add link. (Default: ``{}``)

        """

        if self.use_gpu:
            self.xp.cuda.runtime.setDevice(self.gpu)
            
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
        phase_marginalize=False,
        constants_index=None,
        **waveform_kwargs
    ):
        """Compute the log-Likelihood

        params (double np.ndarray): Parameters for evaluating log-Likelihood.
            ``params.shape=(num_params,)`` if 1D or
            ``params.shape=(num_params, num_bin_all)`` if 2D for more than
            one binary.
        return_extracted_snr (bool, optional): If ``True``, return
            :math:`\langle d|h\\rangle\ / \sqrt{\langle h|h\\rangle}` as a second entry
            of the return array. This produces a return array of
            ``xp.array([log likelihood, snr]).T``. If ``False``, just return
            the log-Likelihood array.
        phase_marginalize (bool, optional): If ``True``, compute the phase-marginalized
            log-Likelihood (and snr if ``return_extracted_snr==True``).
        **waveform_kwargs (dict, optional): Keyword arguments for waveform
            generator. Some may be overwritten. See the main class docstring.

        Returns:
            np.ndarray: log-Likelihoods or ``np.array([log-Likelihoods, snr]).T``

        """

        # store info
        self.phase_marginalize = phase_marginalize
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
        breakpoint()
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
            self.hdyn_d_h if not self.phase_marginalize else self.xp.abs(self.hdyn_d_h)
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
