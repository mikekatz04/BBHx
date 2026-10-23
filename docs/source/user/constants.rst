Constants
~~~~~~~~~~~~~~~~~~~~~~~~~~

The module :code:`bbhx.utils.constants` holds the physical and mathematical constants used
throughout the package (SI units).

The physical constants are not written out in BBHx: following the LISA Analysis Tools
convention, they come from `lisaconstants <https://pypi.org/project/lisaconstants/>`_
(pinned ``lisaconstants==2.0.2``).

- Python: :code:`bbhx.utils.constants` imports :code:`lisaconstants` (``MSUN_SI``,
  ``YRSID_SI``, ``AU_SI``, ``C_SI``, ``G_SI``, ``GMSUN``, ``PC_SI``) and derives
  ``MTSUN_SI`` and ``MRSUN_SI`` from them.
- C++ / CUDA: the kernels read the same numbers from the generated header
  ``src/bbhx/cutils/lisaconstants_values.h`` (through ``cutils/constants.h``).
- ``tests/test_lisaconstants_values.py`` checks both against the installed
  ``lisaconstants`` package.

To see the values in use, print them, e.g.
:code:`from bbhx.utils.constants import GMSUN; print(GMSUN)`.

The orbit constants ``F0``, ``Omega0`` and ``ConstOmega`` (``= Omega0``) are derived from
``YRSID_SI``, and ``eorbit`` from ``L_SI`` and ``AU_SI``. They belong to the analytic
circular-orbit model used by :mod:`bbhx.utils.transform` and by the orbital-delay phase of
the fast FD response.
``L_SI`` is the nominal arm length, 2.5e9 m. It is not a ``lisaconstants`` value, and
realistic armlengths come from the orbit file.

.. automodule:: bbhx.utils.constants
