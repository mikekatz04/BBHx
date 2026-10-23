
# Physical and mathematical constants of the bbhx package

# Copyright (C) 2020 Michael L. Katz
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

"""Physical and mathematical constants used throughout BBHx (SI units).

The physical values come from ``lisaconstants`` (pinned ``lisaconstants==2.0.2``):
``MSUN_SI`` (kg), ``YRSID_SI`` (astronomical year, s), ``AU_SI`` (m), ``C_SI`` (m/s),
``G_SI`` (m^3 kg^-1 s^-2), ``GMSUN`` (solar mass parameter, m^3/s^2), ``PC_SI`` (m), and the
derived ``MTSUN_SI = GMSUN / C_SI**3`` (s) and ``MRSUN_SI = GMSUN / C_SI**2`` (m). The C++
kernels read the same numbers from ``cutils/lisaconstants_values.h``.

Mathematical constants: ``PI``, ``PI_2``, ``PI_3``, ``PI_4``, ``SQRTPI``, ``SQRTTWOPI``,
``INVSQRTPI``, ``INVSQRTTWOPI``, ``GAMMA`` (Euler's constant), ``SQRT2``, ``SQRT3``,
``SQRT6``, ``INVSQRT2``, ``INVSQRT3``, ``INVSQRT6``.

LISA orbit constants of the analytic circular-orbit model used by
:mod:`bbhx.utils.transform` and the fast FD response: ``F0 = 1 / YRSID_SI`` (Hz),
``Omega0 = 2 PI / YRSID_SI`` (rad/s), ``ConstOmega`` (= ``Omega0``; read by LAT's MBH
sky-mode move), ``L_SI = 2.5e9`` (nominal arm length, m) and
``eorbit = L_SI / (2 SQRT3 AU_SI)`` (orbital eccentricity).
"""

import lisaconstants as _lc

# Physical constants from lisaconstants (2026-10-08; were hard-coded older
# values, e.g. GMSUN 1.3271244210789466e20). The C side reads the same numbers
# from cutils/lisaconstants_values.h via cutils/constants.h.
MSUN_SI = _lc.SOLAR_MASS
YRSID_SI = _lc.ASTRONOMICAL_YEAR
AU_SI = _lc.ASTRONOMICAL_UNIT
C_SI = _lc.SPEED_OF_LIGHT
G_SI = _lc.GRAVITATIONAL_CONSTANT
GMSUN = _lc.SOLAR_MASS_PARAMETER
MTSUN_SI = GMSUN / C_SI**3
MRSUN_SI = GMSUN / C_SI**2
PC_SI = _lc.PARSEC
PI = 3.141592653589793238462643383279502884
PI_2 = 1.570796326794896619231321691639751442
PI_3 = 1.047197551196597746154214461093167628
PI_4 = 0.785398163397448309615660845819875721
SQRTPI = 1.772453850905516027298167483341145183
SQRTTWOPI = 2.506628274631000502415765284811045253
INVSQRTPI = 0.564189583547756286948079451560772585
INVSQRTTWOPI = 0.398942280401432677939946059934381868
GAMMA = 0.577215664901532860606512090082402431
SQRT2 = 1.414213562373095048801688724209698079
SQRT3 = 1.732050807568877293527446341505872367
SQRT6 = 2.449489742783178098197284074705891392
INVSQRT2 = 0.707106781186547524400844362104849039
INVSQRT3 = 0.577350269189625764509148780501957455
INVSQRT6 = 0.408248290463863016366214012450981898
F0 = 1.0 / YRSID_SI
Omega0 = 2.0 * PI / YRSID_SI
L_SI = 2.5e9
eorbit = L_SI / (2.0 * SQRT3 * AU_SI)
ConstOmega = Omega0
