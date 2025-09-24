"""BBHx."""

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
from .cutils import BBHxCpuBackend, BBHxCuda11xBackend, BBHxCuda12xBackend

add_backends = {
    "bbhx_cpu": BBHxCpuBackend,
    "bbhx_cuda11x": BBHxCuda11xBackend,
    "bbhx_cuda12x": BBHxCuda12xBackend,
}

Globals().backends_manager.add_backends(add_backends)


from gpubackendtools import get_backend as _get_backend
from gpubackendtools import has_backend as _has_backend
from gpubackendtools import get_first_backend as _get_first_backend
from gpubackendtools.gpubackendtools import Backend


def get_backend(backend: str) -> Backend:
    __doc__ = _get_backend.__doc__
    if "bbhx" not in backend:
        return _get_backend("bbhx_" + backend)
    else:
        return _get_backend(backend)


def has_backend(backend: str) -> Backend:
    __doc__ = _has_backend.__doc__
    if "bbhx_" not in backend:
        return _has_backend("bbhx_" + backend)
    else:
        return _has_backend(backend)

        
def get_first_backend(backend: str) -> Backend:
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
    "get_logger",
    "get_config",
    "get_config_setter",
    "get_backend",
    "get_file_manager",
    "has_backend",
    "response",
    "waveforms",
]
