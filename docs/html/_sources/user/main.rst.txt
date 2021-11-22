Full TDI Waveforms
~~~~~~~~~~~~~~~~~~~~~~~~~~

TDI observable waveforms are produced in three main parts:

    1. Produce a waveform in the frequency-domain amplitude (:math:`A(f)`) and phase (:math:`\phi(f)`) representation at the solar system barycenter (SSB). This also must include the time-frequency correspondence. This gives:

    .. math::
        \tilde{h}_{lm}(f) = A_{lm}(f)e^{-i\phi(f)}.

    2. Use the time frequency correspondence to compute the LISA response transfer functions: :math:`\mathcal{T}(f, t_{lm}(f))`, where :math:`t_{lm}(f)` is the time-frequency correspondence.

    3. :math:`A(f)`, :math:`\phi(f)`, and :math:`\mathcal{T}(f, t_{lm}(f))` are computed on the same sparse grid in frequencies. The final step is interpolating to the frequencies of the Fourier transform of the data stream, producing the TDI observables (:math:`AET`):

    .. math::
        \tilde{h}^{AET}(f) = \sum_{lm}\mathcal{T}^{AET}(f, t_{lm}(f))\tilde{h}_{lm}(f).


.. autoclass:: bbhx.waveformbuild.BBHWaveformFD
    :members:
    :show-inheritance:
    :inherited-members:
