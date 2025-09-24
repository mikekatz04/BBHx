from typing import Optional, Sequence, TypeVar, Union
import types


from gpubackendtools import ParallelModuleBase


class BBHxParallelModule(ParallelModuleBase):
    def __init__(self, force_backend=None):
        force_backend_in = ('bbhx', force_backend) if isinstance(force_backend, str) else force_backend
        super().__init__(force_backend_in)
