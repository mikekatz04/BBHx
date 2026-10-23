"""BBHx: massive black hole binary waveforms, LISA response and likelihoods (CPU and GPU).

Main entry points:

- :class:`bbhx.waveformbuild.BBHWaveformFD`: PhenomHM / PhenomD frequency-domain TDI waveforms
  (:class:`~bbhx.waveforms.phenomhm.PhenomHMAmpPhase` amplitude/phase +
  :class:`~bbhx.response.fastfdresponse.LISATDIResponse` fast FD response + interpolation).
- :class:`bbhx.likelihood.Likelihood` and :class:`bbhx.likelihood.HeterodynedLikelihood`:
  direct and heterodyned (relative-binning) log-likelihoods for those waveforms.
- :class:`bbhx.mbhtdionfly.MBHTDIonFly` and :class:`bbhx.sobbhtdionfly.SOBBHTDIonFly`:
  time-domain TDI-on-the-fly generators for MBHs (phentax modes) and stellar-origin BBHs.
- :class:`bbhx.sobbhcomps.SOBBHWDMComputations`: SOBBH chunked-heterodyne WDM likelihood.
- :mod:`bbhx.utils.transform`: SSB <-> LISA frame maps and MBH sky-mode transforms.

Importing the package registers the ``bbhx_cpu``, ``bbhx_cuda11x``, ``bbhx_cuda12x`` and
``bbhx_cuda13x`` backends with ``gpubackendtools`` (each composes the LISA Analysis Tools
native backend with BBHx's own kernels, see :mod:`bbhx.cutils`). Compute classes pick their
backend once, at construction, through the ``force_backend`` argument (see
:class:`~bbhx.utils.parallelbase.BBHxParallelModule`).
"""

# ruff: noqa: E402
try:
    from bbhx._version import (  # pylint: disable=E0401,E0611
        __version__,
        __version_tuple__,
    )

except ModuleNotFoundError:
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

    try:
        __version__ = version(__name__)
        __version_tuple__ = tuple(__version__.split("."))
    except PackageNotFoundError:  # pragma: no cover
        __version__ = "unknown"
        __version_tuple__ = (0, 0, 0, "unknown")
    finally:
        del version, PackageNotFoundError

_is_editable: bool
try:
    from . import _editable

    _is_editable = True
    del _editable
except (ModuleNotFoundError, ImportError):
    _is_editable = False

from . import cutils, utils

from gpubackendtools import Globals
from .cutils import (
    BBHxCpuBackend,
    BBHxCuda11xBackend,
    BBHxCuda12xBackend,
    BBHxCuda13xBackend,
)


# Phase 3L.7k (2026-06-04): BBHx backends now compose LAT's
# LISAToolsBackend surface with SOBBH + BBH-specific Wraps. The previous
# fastlisaresponse_<flavor> backend family (lisa-on-gpu) is being
# retired and consumers can use these directly via
# ``bbhx.get_backend(...)``.
add_backends = {
    "bbhx_cpu": BBHxCpuBackend,
    "bbhx_cuda11x": BBHxCuda11xBackend,
    "bbhx_cuda12x": BBHxCuda12xBackend,
    "bbhx_cuda13x": BBHxCuda13xBackend,
}

Globals().backends_manager.add_backends(add_backends)


from gpubackendtools import get_backend as _get_backend
from gpubackendtools import has_backend as _has_backend
from gpubackendtools import get_first_backend as _get_first_backend
from gpubackendtools.gpubackendtools import Backend


def get_backend(backend: str) -> Backend:
    """Return a BBHx compute backend by name.

    A name that does not contain ``"bbhx"`` gets the ``"bbhx_"`` prefix, then
    :func:`gpubackendtools.get_backend` resolves it against the backends registered by this
    package at import. Accepted names:

    - ``"cpu"``, ``"cuda11x"``, ``"cuda12x"``, ``"cuda13x"``: that backend.
    - ``"cuda"``: the first loadable of ``cuda13x``, ``cuda12x``, ``cuda11x`` (in that order).
    - ``"gpu"``: alias of ``"cuda"``.
    - a full registry name containing ``"bbhx"`` (e.g. ``"bbhx_cpu"``), passed through
      unchanged.

    BBHx registers no JAX backend.

    Args:
        backend: Backend name, see above.

    Returns:
        Backend: The loaded backend (a :class:`~bbhx.cutils.BBHxBackend` subclass).
        ``backend.xp`` is ``numpy`` (CPU) or ``cupy`` (CUDA).

    Raises:
        gpubackendtools.gpubackendtools.BackendAccessException: When the backend (or, for
            ``"cuda"`` / ``"gpu"``, every CUDA backend) is unavailable or disabled.
        ValueError: When the name is not a registered backend.

    Example:
        >>> import bbhx
        >>> bbhx.get_backend("cpu").name
        'bbhx_cpu'
    """
    # Assigns a local name only (it does not set the function's __doc__); the
    # docstring above is what help() shows.
    __doc__ = _get_backend.__doc__
    if "bbhx" not in backend:
        return _get_backend("bbhx_" + backend)
    else:
        return _get_backend(backend)


def has_backend(backend: str) -> Backend:
    """Report whether a BBHx backend can be loaded, without raising.

    The name is prefixed with ``"bbhx_"`` (unless it already contains it) and passed to
    :func:`gpubackendtools.has_backend`, which tries to load the backend once and caches the
    result.

    Accepted names: ``"cpu"``, ``"cuda11x"``, ``"cuda12x"``, ``"cuda13x"``, or a full
    ``"bbhx_<name>"`` registry name. The ``"cuda"`` and ``"gpu"`` aliases are honoured by
    :func:`get_backend` but not here: the prefixed ``"bbhx_cuda"`` is not a registered name.

    Args:
        backend: Backend name, see above.

    Returns:
        bool: ``True`` when the backend's native modules (and ``cupy`` for CUDA) load. The
        ``-> Backend`` annotation in the signature is inaccurate.

    Raises:
        ValueError: When the name is not a registered backend (including ``"cuda"`` and
            ``"gpu"``, see above).

    Example:
        >>> import bbhx
        >>> bbhx.has_backend("cpu")
        True
    """
    # Assigns a local name only (it does not set the function's __doc__); the
    # docstring above is what help() shows.
    __doc__ = _has_backend.__doc__
    if "bbhx_" not in backend:
        return _has_backend("bbhx_" + backend)
    else:
        return _has_backend(backend)

        
def get_first_backend(backend: str) -> Backend:
    """Return the first loadable backend of a preference list (thin wrapper, see Warning).

    Prefixes ``"bbhx_"`` onto ``backend`` (unless it already contains it) and forwards to
    :func:`gpubackendtools.get_first_backend`, which expects a SEQUENCE of names.

    Args:
        backend: Backend name(s), see Warning.

    Returns:
        Backend: The first loadable backend.

    Raises:
        gpubackendtools.gpubackendtools.BackendAccessException: When none of the names can be
            loaded.
        ValueError: For a ``str`` argument, see Warning.
        TypeError: For a ``list`` argument, see Warning.

    Warning:
        As written this wrapper does not work: a ``str`` argument becomes one prefixed string
        that ``gpubackendtools`` iterates character by character, so the first character
        fails as an unknown backend name (``ValueError``); a ``list`` argument fails the
        string concatenation (``TypeError``). BBHx classes resolve their default backend
        through ``supported_backends()`` and :func:`gpubackendtools.get_first_backend`
        instead. Use :func:`get_backend` with ``"cuda"`` for "first available CUDA backend".
    """
    # Assigns a local name only (it does not set the function's __doc__); the
    # docstring above is what help() shows.
    __doc__ = _get_first_backend.__doc__
    if "bbhx_" not in backend:
        return _get_first_backend("bbhx_" + backend)
    else:
        return _get_first_backend(backend)

from . import response, waveforms

__all__ = [
    "__version__",
    "__version_tuple__",
    "_is_editable",
    "get_backend",
    "has_backend",
    "get_first_backend",
    "response",
    "waveforms",
]
