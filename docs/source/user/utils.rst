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

When working with MBHBs, there are some useful transformations. Mainly to and from the LISA and SSB reference frames. Also, transforming to other probable sky locations during sampling.

.. automodule:: bbhx.utils.transform
    :members:
    :show-inheritance:
    :inherited-members:


.. include:: constants.rst


Citation Module
****************

.. automodule:: bbhx.utils.citations
    :members:

Cython GPU/CPU Agnostic Tools
*******************************

.. automodule:: bbhx.utils.utility
    :members:
