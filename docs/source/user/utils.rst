Utility Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

Interpolation Utilities
************************

.. autoclass:: bbhx.waveformbuild.TemplateInterpFD
    :members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: gpubackendtools.interpolate.CubicSplineInterpolant
    :members:
    :show-inheritance:
    :inherited-members:

.. note::
   ``CubicSplineInterpolant`` moved to ``gpubackendtools.interpolate``
   at the BBHx GBT-dedup (2026-06-05). The old
   ``bbhx.utils.interpolate`` module no longer exists.

Useful Transformation Functions
********************************

These transforms are useful when working with MBHBs. They map the sky position,
polarization and time between the SSB ecliptic frame and the LISA constellation frame,
and they generate the other probable LISA-frame sky modes during sampling. The inputs of
:func:`~bbhx.utils.transform.SSB_to_LISA` are SSB-ecliptic angles, so an ICRS sky must
be rotated to the ecliptic first (see :doc:`conventions`).

.. automodule:: bbhx.utils.transform
    :members:
    :show-inheritance:
    :inherited-members:

Constants are documented in :doc:`constants`.

Legacy pointer helpers
***********************

These helpers come from the old Cython build. BBHx no longer uses them: its native code is
bound with nanobind and takes arrays directly. The maintained copies are in
``gpubackendtools.pointeradjust``.

.. automodule:: bbhx.utils.utility
    :members:
