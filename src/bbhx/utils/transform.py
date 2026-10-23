# Useful MBHB-related Transformations

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

"""Frame and parameter transforms for massive black hole binaries.

- :func:`SSB_to_LISA` / :func:`LISA_to_SSB`: reference time, sky position and polarization
  between the solar-system-barycenter (SSB) ECLIPTIC frame and the LISA constellation frame
  (analytic circular-orbit model, arXiv:2003.00357). In the global-fit runs they are called
  by the LISA Analysis Tools MBH sky-mode move
  (``lisatools.sampling.moves.skymodehop.LISAFrameMap``), which first rotates the run's
  ICRS sky to ecliptic.
- :func:`tLfromSSBframe` / :func:`tSSBfromLframe`: the time retardation alone.
- :func:`mbh_sky_mode_transform`: the 8 LISA-frame sky modes of an MBH posterior.
- :func:`mT_q` / :func:`m1_m2_to_mT_q`: total mass and mass ratio ``q = m2 / m1 <= 1`` <->
  component masses.
- :func:`modpi` / :func:`mod2pi`: angle wrapping.

All functions are NumPy-only (no backend) and broadcast over array inputs. Angles are in
radians and times in seconds, except the ``t0`` orbital-phase offset of
:func:`SSB_to_LISA` / :func:`LISA_to_SSB`, which is in years. ``OrbitR`` (= ``AU_SI``, m)
is the orbital radius of the constellation centre used by the retardation.
"""

import numpy as np
from scipy import constants as ct

from .constants import *
# from .citations import *

OrbitR = AU_SI


def mT_q(mT, q):
    """Convert total mass and mass ratio to component masses.

    Convention: ``q = m2 / m1 <= 1``, so ``m1 = mT / (1 + q) >= m2 = mT q / (1 + q)``. A
    ``q > 1`` returns the two masses in swapped order (``m1 < m2``). The stock LISA Analysis
    Tools MBH sampling basis uses the inverse ratio ``Q = m1 / m2 >= 1`` instead
    (``lisatools.globalfit.stock.erebor.transforms.mT_Q``).

    Args:
        mT (float or np.ndarray): Total mass ``m1 + m2`` (solar masses in BBHx; any unit
            works).
        q (float or np.ndarray): Mass ratio ``m2 / m1``.

    Returns:
        tuple: ``(m1, m2)``, in the unit and broadcast shape of the inputs.

    Example:
        >>> from bbhx.utils.transform import mT_q
        >>> mT_q(3.0, 0.5)
        (2.0, 1.0)
    """
    return (mT / (1 + q), mT * q / (1 + q))


def m1_m2_to_mT_q(m1, m2):
    """Convert ``m1, m2`` to total mass and mass ratio (inverse of :func:`mT_q`).

    Args:
        m1 (float or np.ndarray): Primary mass.
        m2 (float or np.ndarray): Secondary mass.

    Returns:
        tuple: ``(mT, q)`` with ``mT = m1 + m2`` and ``q = m2 / m1`` (``q <= 1`` when
        ``m1 >= m2``).

    Example:
        >>> from bbhx.utils.transform import m1_m2_to_mT_q
        >>> m1_m2_to_mT_q(2.0, 1.0)
        (3.0, 0.5)
    """
    return (m1 + m2, m2 / m1)


def modpi(phase):
    """Wrap an angle into ``[0, pi)``.

    Computes ``phase - floor(phase / pi) * pi``. Originally from Sylvain Marsat.

    Args:
        phase (float or np.ndarray): Angle (rad).

    Returns:
        float or np.ndarray: ``phase`` modulo ``pi``, in ``[0, pi)``.
    """
    # from sylvain
    return phase - np.floor(phase / np.pi) * np.pi


def mod2pi(phase):
    """Wrap an angle into ``[0, 2 pi)``.

    Computes ``phase - floor(phase / (2 pi)) * 2 pi``. Originally from Sylvain Marsat.

    Args:
        phase (float or np.ndarray): Angle (rad).

    Returns:
        float or np.ndarray: ``phase`` modulo ``2 pi``, in ``[0, 2 pi)``.
    """
    # from sylvain
    return phase - np.floor(phase / (2 * np.pi)) * 2 * np.pi


def tSSBfromLframe(tL, lambdaSSB, betaSSB, t0=0.0):
    r"""Get time in SSB frame from time in LISA-frame.

    Compute Solar System Barycenter time ``tSSB`` from retarded time at the center
    of the LISA constellation ``tL``. **NOTE**: depends on the sky position
    given in the solar system barycenter (SSB) ecliptic frame.

    The constellation center sits at ecliptic longitude
    :math:`\alpha(t) = \Omega t + \Omega t_0` (circular 1 AU orbit), so a
    wavefront that reaches the SSB at ``tSSB`` reaches the constellation at
    ``tL = tSSB - (R/c) cos(beta) cos(alpha - lambda)``
    (:func:`tLfromSSBframe`). This is the inverse of that relation, solved
    to first order in :math:`\Omega R / c` (the returned value carries the
    second-order term :math:`-\tfrac12 \Omega (R\cos\beta/c)^2 \sin 2(\alpha - \lambda)`,
    with :math:`\alpha` evaluated at ``tL``).

    Originally from Sylvain Marsat. For more information and citation, see
    `arXiv:2003.00357 <https://arxiv.org/abs/2003.00357>`_.

    Args:
        tL (float or np.ndarray): Time in the LISA constellation frame (s).
        lambdaSSB (float or np.ndarray): Ecliptic longitude in the SSB frame (rad).
        betaSSB (float or np.ndarray): Ecliptic latitude in the SSB frame (rad).
        t0 (float or np.ndarray, optional): Orbital-phase offset in **seconds**
            (``alpha = Omega * (t + t0)``). **Note** the public :func:`LISA_to_SSB` /
            :func:`SSB_to_LISA` take ``t0`` in years and convert before calling this.
            Broadcasts against the other arguments. Default ``0.0``.

    Returns:
        float or np.ndarray: Time in the SSB frame (s).
    """
    ConstPhi0 = ConstOmega * t0
    phase = ConstOmega * tL + ConstPhi0 - lambdaSSB
    RoC = OrbitR / C_SI
    return (
        tL
        + RoC * np.cos(betaSSB) * np.cos(phase)
        - 1.0 / 2 * ConstOmega * pow(RoC * np.cos(betaSSB), 2) * np.sin(2.0 * phase)
    )


def tLfromSSBframe(tSSB, lambdaSSB, betaSSB, t0=0.0):
    """Get time in LISA frame from time in SSB-frame.

    Compute retarded time at the center of the LISA constellation frame ``tL`` from
    the time in the SSB frame ``tSSB``. **NOTE**: depends on the sky position
    given in the solar system barycenter (SSB) ecliptic frame.

    ``tL = tSSB - n . r_c(tSSB) / c`` with ``n`` the unit vector towards the
    source and ``r_c = R (cos alpha, sin alpha, 0)`` the constellation
    center (``R = OrbitR = AU_SI``), ``alpha = Omega * (tSSB + t0)``.

    Originally from Sylvain Marsat. For more information and citation, see
    `arXiv:2003.00357 <https://arxiv.org/abs/2003.00357>`_.

    Args:
        tSSB (float or np.ndarray): Time in the SSB frame (s).
        lambdaSSB (float or np.ndarray): Ecliptic longitude in the SSB frame (rad).
        betaSSB (float or np.ndarray): Ecliptic latitude in the SSB frame (rad).
        t0 (float or np.ndarray, optional): Orbital-phase offset in **seconds** (see
            :func:`tSSBfromLframe`). Default ``0.0``.

    Returns:
        float or np.ndarray: Time in the LISA frame (s).
    """
    ConstPhi0 = ConstOmega * t0
    phase = ConstOmega * tSSB + ConstPhi0 - lambdaSSB
    RoC = OrbitR / C_SI
    return tSSB - RoC * np.cos(betaSSB) * np.cos(phase)


def LISA_to_SSB(tL, lambdaL, betaL, psiL, t0=0.0):
    """Convert sky position, polarization and time from the LISA frame to the SSB frame.

    Convert the sky and orientation parameters from the center of the LISA
    constellation reference to the SSB ecliptic reference frame. Exact inverse of
    :func:`SSB_to_LISA` (see there for the frame definition); the SSB time
    is recovered by a 3-step fixed-point iteration on the retardation,
    which converges to ~1e-6 s.

    The parameters that are converted are the reference time, ecliptic latitude,
    ecliptic longitude, and polarization angle. The returned angles are SSB
    ECLIPTIC; rotate them to ICRS (``lisatools.sources.utils.ecliptic_to_icrs``) for the
    global-fit run frame.

    Originally from Sylvain Marsat. For more information and citation, see
    `arXiv:2003.00357 <https://arxiv.org/abs/2003.00357>`_.

    Args:
        tL (float or np.ndarray): Time in the LISA constellation frame (s).
        lambdaL (float or np.ndarray): Longitude in the LISA frame (rad).
        betaL (float or np.ndarray): Latitude in the LISA frame (rad).
        psiL (float or np.ndarray): Polarization angle in the LISA frame (rad).
        t0 (float or np.ndarray, optional): Orbital-phase offset in **years**: the
            constellation center is at ecliptic longitude
            ``alpha = Omega * (tSSB + t0 * YRSID_SI)``. Broadcasts against the other
            inputs, so a per-source array pins each row's constellation phase
            independently. Default ``0.0``.

    Returns:
        tuple: ``(tSSB, lambdaSSB, betaSSB, psiSSB)``: SSB time (s), ecliptic longitude in
        ``[0, 2 pi)``, ecliptic latitude in ``[-pi/2, pi/2]`` and polarization in
        ``[0, pi)`` (rad).

    Example:
        >>> from bbhx.utils.transform import LISA_to_SSB, SSB_to_LISA
        >>> tL, lamL, betaL, psiL = SSB_to_LISA(1.0e7, 1.0, 0.3, 0.5)
        >>> tSSB, lam, beta, psi = LISA_to_SSB(tL, lamL, betaL, psiL)
        >>> round(float(lam), 6), round(float(beta), 6), round(float(psi), 6)
        (1.0, 0.3, 0.5)
    """

    t0 = t0 * YRSID_SI

    ConstPhi0 = ConstOmega * t0
    coszeta = np.cos(np.pi / 3.0)
    sinzeta = np.sin(np.pi / 3.0)
    coslambdaL = np.cos(lambdaL)
    sinlambdaL = np.sin(lambdaL)
    cosbetaL = np.cos(betaL)
    sinbetaL = np.sin(betaL)
    lambdaSSB_approx = 0.0
    betaSSB_approx = 0.0
    # Initially, approximate alpha using tL instead of tSSB - then iterate */
    tSSB_approx = tL
    for k in range(3):
        alpha = ConstOmega * tSSB_approx + ConstPhi0
        cosalpha = np.cos(alpha)
        sinalpha = np.sin(alpha)
        lambdaSSB_approx = np.arctan2(
            cosalpha * cosalpha * cosbetaL * sinlambdaL
            - sinalpha * sinbetaL * sinzeta
            + cosbetaL * coszeta * sinalpha * sinalpha * sinlambdaL
            - cosalpha * cosbetaL * coslambdaL * sinalpha
            + cosalpha * cosbetaL * coszeta * coslambdaL * sinalpha,
            cosbetaL * coslambdaL * sinalpha * sinalpha
            - cosalpha * sinbetaL * sinzeta
            + cosalpha * cosalpha * cosbetaL * coszeta * coslambdaL
            - cosalpha * cosbetaL * sinalpha * sinlambdaL
            + cosalpha * cosbetaL * coszeta * sinalpha * sinlambdaL,
        )
        betaSSB_approx = np.arcsin(
            coszeta * sinbetaL
            + cosalpha * cosbetaL * coslambdaL * sinzeta
            + cosbetaL * sinalpha * sinzeta * sinlambdaL
        )
        tSSB_approx = tSSBfromLframe(tL, lambdaSSB_approx, betaSSB_approx, t0)

    lambdaSSB_approx = lambdaSSB_approx % (2 * np.pi)
    #  /* Polarization */
    psiSSB = modpi(
        psiL
        + np.arctan2(
            cosalpha * sinzeta * sinlambdaL - coslambdaL * sinalpha * sinzeta,
            cosbetaL * coszeta
            - cosalpha * coslambdaL * sinbetaL * sinzeta
            - sinalpha * sinbetaL * sinzeta * sinlambdaL,
        )
    )

    return (tSSB_approx, lambdaSSB_approx, betaSSB_approx, psiSSB)


def SSB_to_LISA(tSSB, lambdaSSB, betaSSB, psiSSB, t0=0.0):
    r"""Convert sky position, polarization and time from the SSB frame to the LISA frame.

    Convert the sky and orientation parameters from the SSB ecliptic reference frame to
    the center of the LISA constellation reference frame. The parameters that are
    converted are the reference time, ecliptic latitude, ecliptic longitude, and
    polarization angle.

    The inputs are SSB ECLIPTIC angles. A sky given in ICRS (the global-fit run frame:
    right ascension, declination, ICRS polarization) must be rotated to ecliptic first, as
    the LISA Analysis Tools MBH sky-mode move does with
    ``lisatools.sources.utils.icrs_to_ecliptic`` before calling this function.

    **Frame definition** (verified 2026-10 against the LISA Analysis Tools
    time-domain response on the stock equal-arm and ESA orbits):

    - the constellation center is at ecliptic longitude
      :math:`\alpha = \Omega (t_{SSB} + t_0)` on a circular 1 AU orbit;
    - the LISA frame is the SSB ecliptic frame rotated by the tilt angle
      :math:`\zeta = 60^\circ` about the orbital tangent
      :math:`(-\sin\alpha, \cos\alpha, 0)`, i.e.
      :math:`R_z(\alpha) R_y(-\zeta) R_z(-\alpha)`. Its ``z`` axis
      :math:`(-\sin\zeta\cos\alpha, -\sin\zeta\sin\alpha, \cos\zeta)`
      is the normal of the constellation plane; the in-plane (cartwheel)
      rotation is deliberately NOT tracked -- it only shifts
      :math:`\lambda_L, \psi_L` by a constant, which the 4-fold
      longitudinal sky-mode symmetry absorbs;
    - ``psiL = psiSSB + chi`` where ``chi`` is the rotation (about the line
      of sight) from the ecliptic local north to the LISA-frame local north
      at the source -- the same convention as the ICRS <-> ecliptic
      polarization rotation in ``lisatools.sources.utils``;
    - ``tL`` is the retarded time at the constellation center
      (:func:`tLfromSSBframe`).

    Originally from Sylvain Marsat. For more information and citation, see
    `arXiv:2003.00357 <https://arxiv.org/abs/2003.00357>`_.
    **Note**: no transformation of the phase -- approximant-dependence.

    Args:
        tSSB (float or np.ndarray): Time in the SSB frame (s).
        lambdaSSB (float or np.ndarray): Ecliptic longitude in the SSB frame (rad).
        betaSSB (float or np.ndarray): Ecliptic latitude in the SSB frame (rad).
        psiSSB (float or np.ndarray): Polarization angle in the SSB ecliptic frame (rad).
        t0 (float or np.ndarray, optional): Orbital-phase offset in **years**
            (``alpha = Omega * (tSSB + t0 * YRSID_SI)``). Broadcasts against the other
            inputs, so a per-source array pins each row's constellation phase
            independently (e.g. to the phase read off a realistic orbit file at that
            row's time). Default ``0.0``.

    Returns:
        tuple: ``(tL, lambdaL, betaL, psiL)``: LISA-frame time (s), longitude in
        ``(-pi, pi]`` (``arctan2``, not wrapped), latitude in ``[-pi/2, pi/2]`` and
        polarization in ``[0, pi)`` (rad).

    Example:
        >>> from bbhx.utils.transform import SSB_to_LISA
        >>> tL, lamL, betaL, psiL = SSB_to_LISA(1.0e7, 1.0, 0.3, 0.5)
    """
    t0 = t0 * YRSID_SI

    ConstPhi0 = ConstOmega * t0
    coszeta = np.cos(np.pi / 3.0)
    sinzeta = np.sin(np.pi / 3.0)
    coslambda = np.cos(lambdaSSB)
    sinlambda = np.sin(lambdaSSB)
    cosbeta = np.cos(betaSSB)
    sinbeta = np.sin(betaSSB)
    alpha = ConstOmega * tSSB + ConstPhi0
    cosalpha = np.cos(alpha)
    sinalpha = np.sin(alpha)
    tL = tLfromSSBframe(tSSB, lambdaSSB, betaSSB, t0)
    lambdaL = np.arctan2(
        cosalpha * cosalpha * cosbeta * sinlambda
        + sinalpha * sinbeta * sinzeta
        + cosbeta * coszeta * sinalpha * sinalpha * sinlambda
        - cosalpha * cosbeta * coslambda * sinalpha
        + cosalpha * cosbeta * coszeta * coslambda * sinalpha,
        cosalpha * sinbeta * sinzeta
        + cosbeta * coslambda * sinalpha * sinalpha
        + cosalpha * cosalpha * cosbeta * coszeta * coslambda
        - cosalpha * cosbeta * sinalpha * sinlambda
        + cosalpha * cosbeta * coszeta * sinalpha * sinlambda,
    )
    betaL = np.arcsin(
        coszeta * sinbeta
        - cosalpha * cosbeta * coslambda * sinzeta
        - cosbeta * sinalpha * sinzeta * sinlambda
    )
    psiL = modpi(
        psiSSB
        + np.arctan2(
            coslambda * sinalpha * sinzeta - cosalpha * sinzeta * sinlambda,
            cosbeta * coszeta
            + cosalpha * coslambda * sinbeta * sinzeta
            + sinalpha * sinbeta * sinzeta * sinlambda,
        )
    )

    return (tL, lambdaL, betaL, psiL)


def mbh_sky_mode_transform(
    coords, ind_map=None, kind="both", inplace=False, cos_i=False
):
    r"""Sky-mode transformation in the LISA reference frame.

    In the LISA constellation reference frame, the sky localization for a
    source generally has 8 sky modes: 4 longitudinal and 2 latitudinal.
    Longitudinal modes involve :math:`\lambda + (0, 1, 2, 3)\times\pi/2` and
    :math:`\psi + (0, 1, 2, 3)\times\pi/2`, where :math:`\lambda` is the
    LISA-frame longitude and :math:`\psi` is the polarization angle. The
    latitudinal modes are at :math:`(\pm\beta, \pm\cos{\iota}, \pi - \psi)`
    where :math:`\beta` and :math:`\iota` are the LISA-frame latitude and inclination,
    respectively. The input angles must therefore be LISA-frame values (see
    :func:`SSB_to_LISA`). The longitudinally shifted rows have their longitude wrapped to
    ``[0, 2 pi)`` and their polarization to ``[0, pi)``; unshifted rows are not wrapped.

    It is generally wise to start MBH PE runs in all eight sky modes. Using
    a mode-hopping proposal is also a good idea and/or parallel tempering.
    For MBHBs at higher frequency, the 8 modes reduce only to the latitudinal
    modes located at the true ecliptic longitude.

    **Layout**: the output is source-major -- the ``factor`` modes of binary
    ``i`` occupy rows ``i * factor .. (i + 1) * factor - 1`` (mode index
    ``2 * k + j``: longitudinal shift ``k``, latitudinal flip ``j``; for
    ``kind="long"`` the index is ``k``, for ``kind="lat"`` it is ``j``). The
    same layout is assumed of the input when ``inplace=True``.

    Args:
        coords (np.ndarray): 2D array with shape ``(num_bin_all, ndim)`` (a 1D
            ``(ndim,)`` row is accepted when ``inplace=False``).
            ``num_bin_all`` is the total number of binaries. ``ndim`` is the
            number of parameters given. If the inclination is given as :math:`\cos{\iota}`,
            that needs to be indicated with ``cos_i=True``. It does not matter if
            :math:`\beta` or :math:`\sin{\beta}` is given because the transforms
            involve multiplying the input by -1.
        ind_map (dict, optional): Keys are parameter names and values are their index
            into the coords array. Must include the keys ``inc, lam, beta, psi``.
            Default ``None`` means ``dict(inc=7, lam=8, beta=9, psi=10)``.
        kind (str, optional): Which transform: ``"both"`` (all 8 modes, ``factor = 8``),
            ``"long"`` (the 4 longitudinal modes, ``factor = 4``) or ``"lat"`` (the 2
            latitudinal modes, ``factor = 2``). Default ``"both"``.
        inplace (bool, optional): If ``True``, overwrite ``coords`` in place, treating each
            consecutive block of ``factor`` rows as the modes of one binary; the number of
            rows must then be a multiple of ``factor``. If ``False``, each row is first
            repeated ``factor`` times (a new array; ``coords`` is left unchanged). Default
            ``False``.
        cos_i (bool, optional): If ``True``, the inclination is input
            as the cosine of the inclination. Default ``False``.

    Returns:
        np.ndarray: Shape ``(num_bin_all * factor, ndim)`` when ``inplace=False``; the
        modified ``coords`` itself (same shape as the input) when ``inplace=True``.

    Raises:
        ValueError: ``ind_map`` is not a dict, ``kind`` is not one of the three options, or
            (``inplace=True``) the number of rows is not a multiple of ``factor``.

    Example:
        >>> import numpy as np
        >>> from bbhx.utils.transform import mbh_sky_mode_transform
        >>> coords = np.array([[0.4, 3.42, 0.6, 1.0]])  # inc, lam_L, beta_L, psi_L
        >>> ind_map = dict(inc=0, lam=1, beta=2, psi=3)
        >>> mbh_sky_mode_transform(coords, ind_map=ind_map, kind="both").shape
        (8, 4)
    """

    # initialize ind_map
    if ind_map is None:
        ind_map = dict(inc=7, lam=8, beta=9, psi=10)

    elif isinstance(ind_map, dict) is False:
        raise ValueError("If providing the ind_map kwarg, it must be a dict.")

    if kind not in ["both", "lat", "long"]:
        raise ValueError(
            "The kwarg 'kind' must be lat for latitudinal transformation, long for longitudinal transformation, or both for both."
        )

    # factors for transformations
    elif kind == "both":
        factor = 8

    elif kind == "long":
        factor = 4

    elif kind == "lat":
        factor = 2

    if inplace:
        if (coords.shape[0] % factor) != 0:
            raise ValueError(
                "If performing an inplace transformation, the coords provided must have a first dimension size divisible by {} for a '{}' transformation.".format(
                    factor, kind
                )
            )

    else:
        # source-major (each binary's modes contiguous): the strided mode
        # indexing below assumes this. (``np.tile`` -- the pre-2026-10 layout
        # -- interleaved binaries and only covered all modes for a single
        # binary.)
        coords = np.repeat(np.atleast_2d(coords), factor, axis=0)

    if kind == "both" or kind == "lat":

        # inclination
        if cos_i:
            coords[1::2, ind_map["inc"]] *= -1

        else:
            coords[1::2, ind_map["inc"]] = np.pi - coords[1::2, ind_map["inc"]]

        # beta
        coords[1::2, ind_map["beta"]] *= -1

        # psi
        coords[1::2, ind_map["psi"]] = np.pi - coords[1::2, ind_map["psi"]]

    if kind == "long":
        for i in range(1, 4):
            # lambda
            coords[i::4, ind_map["lam"]] = (
                coords[i::4, ind_map["lam"]] + np.pi / 2 * i
            ) % (2 * np.pi)

            # psi
            coords[i::4, ind_map["psi"]] = (
                coords[i::4, ind_map["psi"]] + np.pi / 2 * i
            ) % (np.pi)

    if kind == "both":
        # indexing needs to be different in this case for longitudinal transforms
        num = coords.shape[0]
        for i in range(1, 4):
            for j in range(2):
                # lambda
                coords[2 * i + j :: 8, ind_map["lam"]] = (
                    coords[2 * i + j :: 8, ind_map["lam"]] + np.pi / 2 * i
                ) % (2 * np.pi)

                # psi
                coords[2 * i + j :: 8, ind_map["psi"]] = (
                    coords[2 * i + j :: 8, ind_map["psi"]] + np.pi / 2 * i
                ) % (np.pi)

    return coords
