Likelihood Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

The likelihood classes take a template generator with the
:class:`~bbhx.waveformbuild.BBHWaveformFD` call signature. Their ``get_ll`` method
evaluates a parameter array in the parameter order of that generator. The array has
shape ``(num_params, num_bin_all)`` for :class:`~bbhx.likelihood.Likelihood` and
:class:`~bbhx.likelihood.HeterodynedLikelihood`, and ``(num_bin_all, num_params)`` for
:class:`~bbhx.likelihood.NewHeterodynedLikelihood`.

``phase_maximize=True`` gives the phase-maximized log-likelihood (it uses
:math:`|\langle d|h\rangle|`). ``return_extracted_snr=True`` adds the extracted SNR as a
second column.

Direct Likelihood Computation
*******************************

.. autoclass:: bbhx.likelihood.Likelihood
    :members:
    :show-inheritance:
    :inherited-members:

Heterodyned Likelihood Computation
*************************************

.. autoclass:: bbhx.likelihood.HeterodynedLikelihood
    :members:
    :show-inheritance:
    :inherited-members:

Multi-reference Heterodyned Likelihood (GPU only)
***************************************************

:class:`~bbhx.likelihood.NewHeterodynedLikelihood` gives each reference template its own
sparse grid and summary constants. It runs only on a CUDA backend, which must be passed
explicitly (``force_backend``).

.. autoclass:: bbhx.likelihood.NewHeterodynedLikelihood
    :members:
    :show-inheritance:

Helper
*******

.. autofunction:: bbhx.likelihood.searchsorted2d_vec
