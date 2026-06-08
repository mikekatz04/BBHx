from typing import Optional, Sequence, TypeVar, Union
import types


from gpubackendtools import ParallelModuleBase


class BBHxParallelModule(ParallelModuleBase):
    _BACKEND_PREFIX = "bbhx"

    def __init__(self, force_backend=None):
        if isinstance(force_backend, str) and not force_backend.startswith(
            self._BACKEND_PREFIX + "_"
        ):
            force_backend = (self._BACKEND_PREFIX, force_backend)
        super().__init__(force_backend)
