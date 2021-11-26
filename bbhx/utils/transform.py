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


import numpy as np
from scipy import constants as ct

from .constants import *
from .citations import *

OrbitR = AU_SI


def mT_q(mT, q):
    """Convert total mass and mass ratio to m1,m2

    Args:
        mT (scalar or np.ndarray): Total mass of system.
        q (scalar or np.ndarray): Mass ratio of system with :math:`q<1`.

    Returns:
        tuple: First entry is ``m1``. Second entry is ``m2``.


    """
    return (mT / (1 + q), mT * q / (1 + q))


def modpi(phase):
    """Modulus with pi as the period

    Originally from Sylvain Marsat.

    Args:
        phase (scalar or np.ndarray): Phase angle.

    Returns:
        scalar or np.ndarray: Phase angle modulus by pi.

    """
    # from sylvain
    return phase - np.floor(phase / np.pi) * np.pi


def mod2pi(phase):
    """Modulus with 2pi as the period

    Originally from Sylvain Marsat.

    Args:
        phase (scalar or np.ndarray): Phase angle.

    Returns:
        scalar or np.ndarray: Phase angle modulus by 2pi.

    """
    # from sylvain
    return phase - np.floor(phase / (2 * np.pi)) * 2 * np.pi


def tSSBfromLframe(tL, lambdaSSB, betaSSB, t0=0.0):
    """Get time in SSB frame from time in LISA-frame.

    Compute Solar System Barycenter time ``tSSB`` from retarded time at the center
    of the LISA constellation ``tL``. **NOTE**: depends on the sky position
    given in solar system barycenter (SSB) frame.

    Originally from Sylvain Marsat. For more information and citation, see
    `arXiv:2003.00357 <https://arxiv.org/abs/2003.00357>`_.

    Args:
        tL (scalar or np.ndarray): Time in LISA constellation reference frame.
        lambdaSSB (scalar or np.ndarray): Ecliptic longitude in
            SSB reference frame.
        betaSSB (scalar or np.ndarray): Ecliptic latitude in SSB reference frame.
        t0 (double, optional): Initial start time point away from zero.
            (Default: ``0.0``)

    Returns:
        scalar or np.ndarray: Time in the SSB frame.

    """
    ConstPhi0 = ConstOmega * t0
    phase = ConstOmega * tL + ConstPhi0 - lambdaSSB
    RoC = OrbitR / C_SI
    return (
        tL
        + RoC * np.cos(betaSSB) * np.cos(phase)
        - 1.0 / 2 * ConstOmega * pow(RoC * np.cos(betaSSB), 2) * np.sin(2.0 * phase)
    )


# Compute retarded time at the center of the LISA constellation tL from Solar System Barycenter time tSSB */
def tLfromSSBframe(tSSB, lambdaSSB, betaSSB, t0=0.0):
    """Get time in LISA frame from time in SSB-frame.

    Compute retarded time at the center of the LISA constellation frame ``tL`` from
    the time in the SSB frame ``tSSB``. **NOTE**: depends on the sky position
    given in solar system barycenter (SSB) frame.

    Originally from Sylvain Marsat. For more information and citation, see
    `arXiv:2003.00357 <https://arxiv.org/abs/2003.00357>`_.

    Args:
        tSSB (scalar or np.ndarray): Time in LISA constellation reference frame.
        lambdaSSB (scalar or np.ndarray): Ecliptic longitude in
            SSB reference frame.
        betaSSB (scalar or np.ndarray): Time in LISA constellation reference frame.
        t0 (double, optional): Initial start time point away from zero.
            (Default: ``0.0``)

    Returns:
        scalar or np.ndarray: Time in the LISA frame.

    """
    ConstPhi0 = ConstOmega * t0
    phase = ConstOmega * tSSB + ConstPhi0 - lambdaSSB
    RoC = OrbitR / C_SI
    return tSSB - RoC * np.cos(betaSSB) * np.cos(phase)


def LISA_to_SSB(tL, lambdaL, betaL, psiL, t0=0.0):
    """Convert sky/orientation from LISA frame to SSB frame.

    Convert the sky and orientation parameters from the center of the LISA
    constellation reference to the SSB reference frame.

    The parameters that are converted are the reference time, ecliptic latitude,
    ecliptic longitude, and polarization angle.

    Originally from Sylvain Marsat. For more information and citation, see
    `arXiv:2003.00357 <https://arxiv.org/abs/2003.00357>`_.

    Args:
        tL (scalar or np.ndarray): Time in LISA constellation reference frame.
        lambdaL (scalar or np.ndarray): Ecliptic longitude in
            LISA reference frame.
        betaL (scalar or np.ndarray): Ecliptic latitude in LISA reference frame.
        psiL (scalar or np.ndarray): Polarization angle in LISA reference frame.
        t0 (double, optional): Initial start time point away from zero.
            (Default: ``0.0``)

    Returns:
        Tuple: (``tSSB``, ``lambdaSSB``, ``betaSSB``, ``psiSSB``)


    """

    t0 = t0 * YRSID_SI

    ConstPhi0 = ConstOmega * t0
    coszeta = np.cos(np.pi / 3.0)
    sinzeta = np.sin(np.pi / 3.0)
    coslambdaL = np.cos(lambdaL)
    sinlambdaL = np.sin(lambdaL)
    cosbetaL = np.cos(betaL)
    sinbetaL = np.sin(betaL)
    cospsiL = np.cos(psiL)
    sinpsiL = np.sin(psiL)
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
    """Convert sky/orientation from SSB frame to LISA frame.

    Convert the sky and orientation parameters from the SSB reference frame to the center of the LISA
    constellation reference frame.

    The parameters that are converted are the reference time, ecliptic latitude,
    ecliptic longitude, and polarization angle.

    Originally from Sylvain Marsat. For more information and citation, see
    `arXiv:2003.00357 <https://arxiv.org/abs/2003.00357>`_.
    **Note**: no transformation of the phase -- approximant-dependence.

    Args:
        tSSB (scalar or np.ndarray): Time in SSB reference frame.
        lambdaSSB (scalar or np.ndarray): Ecliptic longitude in
            SSB reference frame.
        betaSSB (scalar or np.ndarray): Ecliptic latitude in SSB reference frame.
        psiSSB (scalar or np.ndarray): Polarization angle in SSB reference frame.
        t0 (double, optional): Initial start time point away from zero in years.
            (Default: ``0.0``)

    Returns:
        Tuple: (``tL``, ``lambdaL``, ``betaL``, ``psiL``)

    """
    t0 = t0 * YRSID_SI

    ConstPhi0 = ConstOmega * t0
    alpha = 0.0
    cosalpha = 0
    sinalpha = 0.0
    coslambda = 0
    sinlambda = 0.0
    cosbeta = 0.0
    sinbeta = 0.0
    cospsi = 0.0
    sinpsi = 0.0
    coszeta = np.cos(np.pi / 3.0)
    sinzeta = np.sin(np.pi / 3.0)
    coslambda = np.cos(lambdaSSB)
    sinlambda = np.sin(lambdaSSB)
    cosbeta = np.cos(betaSSB)
    sinbeta = np.sin(betaSSB)
    cospsi = np.cos(psiSSB)
    sinpsi = np.sin(psiSSB)
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
    """Sky-mode transformation in the LISA reference frame

    In the LISA constellation referenence frame, the sky localization for a
    source generally has 8 sky modes: 4 longitudinal and 2 latitudinal.
    Longitudinal modes involve :math:`\lambda + (0, 1, 2, 3)\\times\pi/2` and
    :math:`\psi + (0, 1, 2, 3)\\times\pi/2`, where :math:`\lambda` is the
    ecliptic longitude and :math:`\psi` is the polarization angle. The
    latitudinal modes are at :math:`(\pm\\beta, \pm\cos{\iota}, \pm\cos{\psi})`
    where :math:`\\beta` and :math:`\iota` are the ecliptic latitude and inclination,
    respectively.

    It is generally wise to start MBH PE runs in all eight sky modes. Using
    a mode-hopping proposal is also a good idea and/or parallel tempering.
    For MBHBs at higher frequency, the 8 modes reduce only to the latitudinal
    modes located at the true ecliptic longitude.

    Args:
        coords (np.ndarray): 2D array with shape: ``(num_bin_all, ndim)``.
            ``num_bin_all`` is the total number of binaries. ``ndim`` is the
            number of parameters given. If the inclination is given as :math:`\cos{\iota}`,
            that needs to be indicated with ``cos_i=True``. It does not matter if
            :math:`\\beta` or :math:`\sin{\\beta}` is given because the transforms
            involve multiplying the input by -1.
        ind_map (dict, optional): Keys are parameter names and values are there index
            into the coords array. Must include the keys ``inc, lam, beta, psi``.
            If ``None``, it will be ``dict(inc=7, lam=8, beta=9, psi=10)``.
            (Default: ``None``)
        kind (str, optional): String indicating which transform is needed.
            If ``kind=="both"`` (``kind=="long") [``kind=="lat"``],
            the transformation will be to all (longitudinal) [latitudinal] sky modes.
            If ``inplace==False``, the output array 1st dimension will be a factor
            of 8 (4) [2] longer. (Default: ``"both"``)
        inplace (bool, optional): If ``True``, adjust ``coords`` in place. In this
            case, the user needs to ensure the number of sets of binary Parameters
            is an integer multiple of the number of sky modes associated with the
            trasnformation. If ``kind=="both"`` (``kind=="long") [``kind=="lat"``],
            this must be a mutliple of 8 (4) [2].
            (Default: ``False``)
        cos_i (bool, optional): If ``True``, the inclination is input
            as the cosine of the inclination.
            (Default: ``False``)

    Returns:
        np.ndarray: 2D array with shape: ``(num_bin_all * factor, ndim)``.
            The factor is 1 if ``inplace==False``. The factor is 8 (4) [2]
            if ``kind=="both"`` (``kind=="long") [``kind=="lat"``].

    Raises:
        ValueError: Input arguments are not properly given.

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
        coords = np.tile(coords, (factor, 1))

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
