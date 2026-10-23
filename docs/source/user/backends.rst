Backends
~~~~~~~~~~~~~~~~~~~~~~~~~~

Importing ``bbhx`` registers four backends with ``gpubackendtools``: ``bbhx_cpu``,
``bbhx_cuda11x``, ``bbhx_cuda12x`` and ``bbhx_cuda13x``. Each backend combines three
native modules:

- the LISA Analysis Tools native module (``lisatools_backend_<flavor>.pycppdetector``);
- the GPUBackendTools cubic-spline module (``gbt_backend_<flavor>.interp``);
- BBHx's own nanobind module (``bbhx_backend_<flavor>.cbbhx``).

The native modules are imported the first time a backend is loaded, not at
``import bbhx``.

The CPU backend ships with the ``bbhx`` package. The ``bbhx-cuda11x`` and
``bbhx-cuda12x`` plugin wheels add the CUDA backends. A source build with a CUDA toolkit
builds the backend of that toolkit's major version (``cuda11x``, ``cuda12x`` or
``cuda13x``).

Every compute class chooses its backend once, at construction, with ``force_backend``:

- ``"cpu"``, ``"cuda11x"``, ``"cuda12x"`` or ``"cuda13x"``;
- ``"cuda"`` or ``"gpu"``: the first available CUDA flavor;
- a full ``"bbhx_<flavor>"`` name.

The default ``None`` takes the first loadable of ``cuda13x``, ``cuda12x``, ``cuda11x``
and ``cpu``. The exception is :class:`~bbhx.likelihood.NewHeterodynedLikelihood`, which
needs an explicit CUDA backend. There is no per-call backend argument.

Package-level functions
************************

.. autofunction:: bbhx.get_backend

.. autofunction:: bbhx.has_backend

.. autofunction:: bbhx.get_first_backend

Base class of the compute classes
**********************************

.. autoclass:: bbhx.utils.parallelbase.BBHxParallelModule
    :members:
    :show-inheritance:

Backend classes
****************

.. automodule:: bbhx.cutils
    :members:
    :show-inheritance:

Exceptions
***********

.. automodule:: bbhx.utils.exceptions

.. autoexception:: bbhx.utils.exceptions.BBHXException
    :show-inheritance:

.. autoexception:: bbhx.utils.exceptions.CudaException
    :show-inheritance:

.. autoexception:: bbhx.utils.exceptions.CuPyException
    :show-inheritance:

.. autoexception:: bbhx.utils.exceptions.MissingDependency
    :show-inheritance:
