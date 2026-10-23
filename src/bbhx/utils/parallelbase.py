"""BBHx base class for backend-dispatching compute objects (:class:`BBHxParallelModule`)."""

from typing import Optional, Sequence, TypeVar, Union
import types


from gpubackendtools import ParallelModuleBase


class BBHxParallelModule(ParallelModuleBase):
    """Base of every BBHx compute class; adds the ``"bbhx_"`` prefix to bare backend names.

    The backend is chosen once, at construction, and every later array operation goes
    through ``self.xp`` / ``self.backend`` (see
    :class:`gpubackendtools.parallelbase.ParallelModuleBase`). ``force_backend`` resolution:

    - ``None``: the first loadable entry of the subclass's ``supported_backends()`` (BBHx
      classes return ``["bbhx_" + f for f in cls.GPU_RECOMMENDED()]``, i.e. ``cuda13x``,
      ``cuda12x``, ``cuda11x``, then ``cpu``);
    - a ``str`` not starting with ``"bbhx_"`` (``"cpu"``, ``"cuda12x"``, ``"cuda"``,
      ``"gpu"``): turned into ``("bbhx", name)`` and resolved as ``"bbhx_<name>"``
      (``"cuda"`` / ``"gpu"`` pick the first available CUDA flavor);
    - a full ``"bbhx_<flavor>"`` name, a ``Backend`` instance or a ``(package, flavor)``
      tuple: passed to the base class unchanged.

    Subclasses must implement the classmethod ``supported_backends``.

    Args:
        force_backend (str | Backend | tuple[str, str], optional): Backend to use, see above.
            Default ``None``.

    Raises:
        ValueError: If ``force_backend`` has an unsupported type or names an unregistered
            backend.
        BackendAccessException: If the requested backend (or every supported one) is
            unavailable.
    """

    _BACKEND_PREFIX = "bbhx"

    def __init__(self, force_backend=None):
        if isinstance(force_backend, str) and not force_backend.startswith(
            self._BACKEND_PREFIX + "_"
        ):
            force_backend = (self._BACKEND_PREFIX, force_backend)
        super().__init__(force_backend)
