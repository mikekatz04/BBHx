Full TDI Waveforms
~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~bbhx.waveformbuild.BBHWaveformFD` produces frequency-domain TDI observables in
three steps:

1. :class:`~bbhx.waveforms.phenomhm.PhenomHMAmpPhase` produces each harmonic
   :math:`(l, m)` at the solar system barycenter (SSB) in the frequency-domain
   amplitude (:math:`A_{lm}(f)`) and phase (:math:`\phi_{lm}(f)`) representation. It
   also gives the time-frequency correspondence :math:`t_{lm}(f)`:

   .. math::
       \tilde{h}_{lm}(f) = A_{lm}(f)e^{-i\phi_{lm}(f)}.

2. :class:`~bbhx.response.fastfdresponse.LISATDIResponse` uses the time-frequency
   correspondence to compute the LISA response transfer functions
   :math:`\mathcal{T}(f, t_{lm}(f))`.

3. By default (``direct=False``), :math:`A_{lm}(f)`, :math:`\phi_{lm}(f)` and
   :math:`\mathcal{T}(f, t_{lm}(f))` are computed on a sparse frequency grid per mode.
   :class:`~bbhx.waveformbuild.TemplateInterpFD` then spline-interpolates them onto
   the frequencies of the Fourier transform of the data stream. With ``direct=True``,
   the waveform is instead evaluated directly on the requested frequencies. The modes
   are summed into the TDI observables: A, E, T by default, or X, Y, Z with
   ``response_kwargs=dict(TDItag="XYZ")``:

   .. math::
       \tilde{h}^{AET}(f) = \sum_{lm}\mathcal{T}^{AET}(f, t_{lm}(f))\tilde{h}_{lm}(f).

Units, frames and the reference frequency / phase / time conventions are summarized in
:doc:`conventions`.

.. autoclass:: bbhx.waveformbuild.BBHWaveformFD
    :members:
    :show-inheritance:
    :inherited-members:


Time-domain TDI on the fly
~~~~~~~~~~~~~~~~~~~~~~~~~~

As an alternative to the frequency-domain pipeline above, MBH TDI observables can be
generated directly in the time domain. A time-domain mode generator (currently
`phentax <https://github.com/asantini29/phentax>`_'s ``IMRPhenomTHM``) produces per-mode
amplitude and phase on an adaptive time grid. These are splined and fed through the LISA
TDI-on-the-fly response (:class:`lisatools.response.tdionfly.TDTDIonTheFly`), which
produces the TDI channels without an intermediate Fourier transform.

The sky angles are read in the frame of the orbit data, and the distance is in the unit
of the mode generator (Mpc for ``IMRPhenomTHM``). The stellar-origin counterpart,
:class:`~bbhx.sobbhtdionfly.SOBBHTDIonFly`, is documented in :doc:`sobbh`.

.. note::
    ``phentax`` is not published on PyPI. Install it with the
    ``phentax`` extra (``pip install 'bbhx[phentax]'``) or directly via
    ``pip install git+https://github.com/asantini29/phentax.git``.

.. autoclass:: bbhx.mbhtdionfly.MBHTDIonFly
    :members:
    :show-inheritance:
    :inherited-members:

phentax adapters (legacy)
**************************

:mod:`bbhx.mbhphentax` holds phentax response builders, domain adapters and
sampling-to-waveform transform containers. They were moved here from older LISA Analysis
Tools global-fit settings files. The production global-fit runs do not use them. LISA
Analysis Tools keeps its own copies of the TDI-on-the-fly builder and adapter in
``lisatools.globalfit.stock.erebor.wrappers``. Importing the module needs ``eryn`` (the
``sampling`` extra). ``phentax`` is imported only when a waveform is built.

.. automodule:: bbhx.mbhphentax
    :members:
