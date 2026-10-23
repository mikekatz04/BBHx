Stellar-Origin Black Hole Binaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BBHx holds the SOBBH-specific parts of the LISA Analysis Tools stack: the native PN
source class ``SOBBHTDIonTheFly`` and the kernel group ``SOBBHComputationGroup``
(chunked-heterodyne and signal-heterodyne WDM likelihood kernels). Both are compiled into
``bbhx_backend_<flavor>.cbbhx`` and exposed as ``SOBBHTDIonTheFlyWrap`` /
``SOBBHComputationGroupWrap`` on every BBHx backend. The generic response and WDM
machinery stays in LISA Analysis Tools. Its SOBBH front end
``lisatools.response.tdionfly.SOBBHTDIonTheFly`` runs on a ``bbhx_<flavor>`` backend.

The 11-parameter SOBBH vector of the native kernel is
``(m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta)``:

- masses in solar masses;
- distance in parsecs;
- ``f_low`` (Hz) and ``phi_c`` (rad) defined at the reference time ``t_ref``.

See :doc:`conventions` for the units and parameter order of the other classes.

Time-domain TDI on the fly
***************************

:class:`~bbhx.sobbhtdionfly.SOBBHTDIonFly` takes the PN amplitude and phase from
:class:`lisatools.sources.sobbh.waveform.SOBBHWaveform` (distance in Gpc). It passes
them through the LISA Analysis Tools time-domain TDI-on-the-fly response, in the same
way :class:`~bbhx.mbhtdionfly.MBHTDIonFly` does for MBHs.

.. autoclass:: bbhx.sobbhtdionfly.SOBBHTDIonFly
    :members:
    :show-inheritance:
    :inherited-members:

Chunked-heterodyne WDM likelihood
**********************************

:class:`~bbhx.sobbhcomps.SOBBHWDMComputations` only sets the SOBBH routing constants:

- backend family ``bbhx``;
- wrap ``SOBBHComputationGroupWrap``;
- kernel prefix ``sobbh_wdm_het``;
- 11 parameters, with ``f_low`` at index 5.

The constructor and the methods (``fill_global_wdm``, ``get_ll_wdm``,
``get_swap_ll_wdm``, ...) are those of
:class:`lisatools.chunked_het.WDMComputationsBase`. The GB counterpart is
``gbgpu.gbcomps.GBWDMComputations``.

.. autoclass:: bbhx.sobbhcomps.SOBBHWDMComputations
    :show-inheritance:

JAX source
***********

:class:`~bbhx.jax.sources.sobbh.JaxSOBBHSource` is a source plugin for the LISA Analysis
Tools JAX TDI-on-the-fly engine. Importing it needs ``jax``. Evaluating it needs the PN
prototype ``sobbhtaylert3.py``, which is not installed with BBHx. Point
``SOBBH_TAYLOR_T3_PATH`` at a copy (LISA Analysis Tools keeps one at
``scripts/sobbh/sobbhtaylert3.py``). ``t_ref`` is stored but not applied, while the native
kernel evaluates the PN expressions at ``t - t_ref``. So the two agree only for
``t_ref = 0``.

.. autoclass:: bbhx.jax.sources.sobbh.JaxSOBBHSource
    :members:
    :show-inheritance:
