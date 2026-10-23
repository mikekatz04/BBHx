Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~

The BBHx generators differ in units, frames and parameter order. This page collects
these conventions. The class docstrings give the full details.

Parameter order and units
*************************

Masses are in solar masses everywhere. Spins are dimensionless aligned spins. Angles are
in radians.

.. list-table::
   :header-rows: 1
   :widths: 22 46 12 20

   * - Class
     - Call parameters, in order
     - Distance
     - Sky angles
   * - :class:`~bbhx.waveformbuild.BBHWaveformFD` (and the ``params`` of
       :class:`~bbhx.likelihood.Likelihood` / :class:`~bbhx.likelihood.HeterodynedLikelihood`)
     - ``m1, m2, chi1z, chi2z, distance, phi_ref, f_ref, inc, lam, beta, psi, t_ref``
     - metres
     - frame of the orbit data (see below)
   * - :class:`~bbhx.waveforms.phenomhm.PhenomHMAmpPhase`
     - ``m1, m2, chi1z, chi2z, distance, phi_ref, f_ref, t_ref, length``
     - metres
     - none (source frame)
   * - :class:`~bbhx.mbhtdionfly.MBHTDIonFly`
     - ``m1, m2, s1z, s2z, distance, phi_ref, inclination, ra, dec, psi, t_merge``
     - unit of ``wave_gen`` (Mpc for phentax ``IMRPhenomTHM``)
     - frame of ``orbits``
   * - :class:`~bbhx.sobbhtdionfly.SOBBHTDIonFly`
     - ``m1, m2, s1, s2, distance, f_low, phi0, inc, ra, dec, psi``
     - Gpc
     - frame of ``orbits``
   * - 11-parameter SOBBH vector: the native ``SOBBHTDIonTheFly`` kernel (and its
       LISA Analysis Tools front end ``lisatools.response.tdionfly.SOBBHTDIonTheFly``),
       :class:`~bbhx.sobbhcomps.SOBBHWDMComputations`,
       :class:`~bbhx.jax.sources.sobbh.JaxSOBBHSource`
     - ``m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta``
     - parsecs
     - frame of the orbit data used by the response

Times are in seconds, with two exceptions. ``t_obs_start`` / ``t_obs_end`` of
:meth:`BBHWaveformFD.__call__ <bbhx.waveformbuild.BBHWaveformFD.__call__>` are LISA-frame
times in years, measured from ``t = 0``. The ``t0`` orbital-phase offset of
:func:`~bbhx.utils.transform.SSB_to_LISA` / :func:`~bbhx.utils.transform.LISA_to_SSB` is
also in years.

Mass ratio
**********

:func:`~bbhx.utils.transform.mT_q` and :func:`~bbhx.utils.transform.m1_m2_to_mT_q` use
the mass ratio ``q = m2 / m1 <= 1``, with ``m1`` the heavier body. The stock LISA
Analysis Tools MBH sampling basis uses the inverse ratio ``Q = m1 / m2 >= 1``
(``lisatools.globalfit.stock.erebor.transforms.mT_Q``). If ``m1 < m2``,
:class:`~bbhx.waveformbuild.BBHWaveformFD` swaps the two bodies and their spins.
This swap happens in place on ``np.ndarray`` inputs.

Reference frequency, phase and time (frequency-domain waveforms)
******************************************************************

- The phase convention is :math:`\tilde{h}_{lm}(f) = A_{lm}(f) e^{-i\phi_{lm}(f)}`.
- ``f_ref`` is passed to the PhenomHM kernel, but the kernel ignores it. It always uses
  the peak frequency of the PhenomD (2,2) merger-ringdown amplitude ansatz as the
  reference frequency. ``0.0`` is the conventional value.
- The kernel builds the (2,2) mode with zero phase and ``t(f) = 0`` at that reference
  frequency. The waveform is then shifted in time: ``t_ref`` is added to t(f) and
  ``2 pi f (t_ref - initial_t_val)`` to the phase (``initial_t_val`` is a
  :class:`~bbhx.waveforms.phenomhm.PhenomHMAmpPhase` argument, default ``0.0``). So
  ``t_ref`` (s) is the time at which the (2,2) mode reaches the reference frequency,
  close to the merger time. It is read on the time axis of the response's orbits.
- ``phi_ref`` is applied by the response, through the spin-weighted spherical harmonics
  :math:`{}_{-2}Y_{lm}(\iota, \phi_{\rm ref})`.
  :class:`~bbhx.waveformbuild.BBHWaveformFD` runs the amplitude-phase kernel with zero
  phase.

Frames
******

- :func:`~bbhx.utils.transform.SSB_to_LISA` and :func:`~bbhx.utils.transform.LISA_to_SSB`
  map between SSB-ecliptic angles and the LISA constellation frame. A sky given in
  ICRS (right ascension, declination, ICRS polarization) must first be rotated to the
  ecliptic. The LISA Analysis Tools MBH sky-mode move does this rotation before it calls
  them. :func:`~bbhx.utils.transform.mbh_sky_mode_transform` takes LISA-frame angles.
- In the fast FD response (:class:`~bbhx.response.fastfdresponse.LISATDIResponse`, and so
  :class:`~bbhx.waveformbuild.BBHWaveformFD`), ``lam`` / ``beta`` are used in two places:

  - the transfer functions, together with the spacecraft positions of the orbit data;
  - the orbital-delay phase, together with an analytic circular orbit in the ecliptic
    plane.

  The two agree only for ecliptic orbit data. The orbit files packaged with LISA
  Analysis Tools (:class:`lisatools.detector.EqualArmlengthOrbits`,
  :class:`lisatools.detector.ESAOrbits`) hold SSB-ecliptic positions, whatever their
  ``frame`` label says. Mojito ``L1Orbits`` loaded with ``frame="icrs"`` hold ICRS
  positions.
- The time-domain TDI-on-the-fly generators (:class:`~bbhx.mbhtdionfly.MBHTDIonFly`,
  :class:`~bbhx.sobbhtdionfly.SOBBHTDIonFly`, and LISA Analysis Tools'
  ``SOBBHTDIonTheFly``) read the sky angles in the frame of the orbit data. With ICRS
  orbits they are right ascension, declination and ICRS polarization. ICRS is the
  global-fit run frame.

Orbits and TDI
**************

Pass the orbits explicitly, with the same backend flavor as the class that uses them,
e.g. ``LISATDIResponse(orbits=ESAOrbits(force_backend="cpu"), force_backend="cpu")``.
``orbits=None`` builds :class:`lisatools.detector.EqualArmlengthOrbits` with the default
backend resolution, which can pick a different flavor and then fails the flavor check.

The fast FD response builds TDI with the constant armlength ``orbits.armlength``.
``tdi2=True`` applies the equal-armlength TDI 1 to TDI 2 conversion to the A, E, T
channels only; it has no effect for ``TDItag="XYZ"``. The time-domain TDI-on-the-fly
generators take the TDI generation from their
:class:`lisatools.response.tdiconfig.TDIConfig` (e.g. ``TDIConfig("2nd generation")``).

Backends
********

Every compute class chooses its backend once, at construction, through
``force_backend``. There is no per-call backend argument. See :doc:`backends`.
