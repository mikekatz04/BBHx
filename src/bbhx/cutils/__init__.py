from __future__ import annotations
import dataclasses
import enum
import types
import typing
import abc
from typing import Optional, Sequence, TypeVar, Union
from ..utils.exceptions import *

from gpubackendtools.gpubackendtools import BackendMethods, CpuBackend, Cuda11xBackend, Cuda12xBackend
from gpubackendtools.exceptions import *

@dataclasses.dataclass
class BBHxBackendMethods(BackendMethods):
    hdyn_wrap: typing.Callable[(...), None]
    direct_like_wrap: typing.Callable[(...), None]
    direct_sum_wrap: typing.Callable[(...), None]
    InterpTDI_wrap: typing.Callable[(...), None]
    LISA_response_wrap: typing.Callable[(...), None]
    waveform_amp_phase_wrap: typing.Callable[(...), None]
    get_phenomhm_ringdown_frequencies: typing.Callable[(...), None]
    get_phenomd_ringdown_frequencies: typing.Callable[(...), None]
    interpolate_wrap: typing.Callable[(...), None]
    speciallike: typing.Callable[(...), None]
    new_hdyn_prep: typing.Callable[(...), None]
    new_hdyn_like: typing.Callable[(...), None]

class BBHxBackend:
    hdyn_wrap: typing.Callable[(...), None]
    direct_like_wrap: typing.Callable[(...), None]
    direct_sum_wrap: typing.Callable[(...), None]
    InterpTDI_wrap: typing.Callable[(...), None]
    LISA_response_wrap: typing.Callable[(...), None]
    waveform_amp_phase_wrap: typing.Callable[(...), None]
    get_phenomhm_ringdown_frequencies: typing.Callable[(...), None]
    interpolate_wrap: typing.Callable[(...), None]
    speciallike: typing.Callable[(...), None]
    new_hdyn_prep: typing.Callable[(...), None]
    new_hdyn_like: typing.Callable[(...), None]

    def __init__(self, bbhx_backend_methods):

        # set direct bbhx methods
        # pass rest to general backend
        assert isinstance(bbhx_backend_methods, BBHxBackendMethods)

        self.hdyn_wrap = bbhx_backend_methods.hdyn_wrap
        self.direct_like_wrap = bbhx_backend_methods.direct_like_wrap
        self.direct_sum_wrap = bbhx_backend_methods.direct_sum_wrap
        self.InterpTDI_wrap = bbhx_backend_methods.InterpTDI_wrap
        self.LISA_response_wrap = bbhx_backend_methods.LISA_response_wrap
        self.waveform_amp_phase_wrap = bbhx_backend_methods.waveform_amp_phase_wrap
        self.get_phenomhm_ringdown_frequencies = bbhx_backend_methods.get_phenomhm_ringdown_frequencies
        self.get_phenomd_ringdown_frequencies = bbhx_backend_methods.get_phenomd_ringdown_frequencies
        self.interpolate_wrap = bbhx_backend_methods.interpolate_wrap
        self.speciallike = bbhx_backend_methods.speciallike
        self.new_hdyn_prep = bbhx_backend_methods.new_hdyn_prep
        self.new_hdyn_like = bbhx_backend_methods.new_hdyn_like
    

class BBHxCpuBackend(CpuBackend, BBHxBackend):
    """Implementation of the CPU backend"""
    
    _backend_name = "bbhx_backend_cpu"
    _name = "bbhx_cpu"
    def __init__(self, *args, **kwargs):
        CpuBackend.__init__(self, *args, **kwargs)
        BBHxBackend.__init__(self, self.cpu_methods_loader())

    @staticmethod
    def cpu_methods_loader() -> BBHxBackendMethods:
        try:
            import bbhx_backend_cpu.cbbhx  # Phase BBHx.pybind.bulk: sole BBHx backend module
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cpu' backend could not be imported."
            ) from e

        numpy = BBHxCpuBackend.check_numpy()

        # Single-instance BBHxComputationWrap holds every migrated wrapper.
        # Bound methods are exposed below as `waveform_amp_phase_wrap` etc.
        # so the user-facing API (self.backend.<name>(...)) is unchanged
        # from the prior Cython-only implementation.
        _cbbhx = bbhx_backend_cpu.cbbhx.BBHxComputationWrapCPU()

        return BBHxBackendMethods(
            hdyn_wrap=_cbbhx.hdyn_wrap,
            direct_like_wrap=_cbbhx.direct_like_wrap,
            direct_sum_wrap=_cbbhx.direct_sum_wrap,
            InterpTDI_wrap=_cbbhx.InterpTDI_wrap,
            LISA_response_wrap=_cbbhx.LISA_response_wrap,
            waveform_amp_phase_wrap=_cbbhx.waveform_amp_phase_wrap,
            get_phenomhm_ringdown_frequencies=_cbbhx.get_phenomhm_ringdown_frequencies,
            get_phenomd_ringdown_frequencies=_cbbhx.get_phenomd_ringdown_frequencies,
            interpolate_wrap=_cbbhx.interpolate_wrap,
            # speciallike + new_hdyn_* are GPU-only (their .cu sources use
            # GPU kernel-launch syntax that doesn't compile on CPU);
            # matches the prior Cython-era setup.
            speciallike=None,
            new_hdyn_prep=None,
            new_hdyn_like=None,
            xp=numpy,
        )


class BBHxCuda11xBackend(Cuda11xBackend, BBHxBackend):

    """Implementation of CUDA 11.x backend"""
    _backend_name : str = "bbhx_backend_cuda11x"
    _name = "bbhx_cuda11x"

    def __init__(self, *args, **kwargs):
        Cuda11xBackend.__init__(self, *args, **kwargs)
        BBHxBackend.__init__(self, self.cuda11x_module_loader())
        
    @staticmethod
    def cuda11x_module_loader():
        try:
            import bbhx_backend_cuda11x.cbbhx  # Phase BBHx.pybind.bulk: sole BBHx backend module
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda11x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda11x' backend requires cupy", pip_deps=["cupy-cuda11x"]
            ) from e

        _cbbhx = bbhx_backend_cuda11x.cbbhx.BBHxComputationWrapGPU()

        return BBHxBackendMethods(
            hdyn_wrap=_cbbhx.hdyn_wrap,
            direct_like_wrap=_cbbhx.direct_like_wrap,
            direct_sum_wrap=_cbbhx.direct_sum_wrap,
            InterpTDI_wrap=_cbbhx.InterpTDI_wrap,
            LISA_response_wrap=_cbbhx.LISA_response_wrap,
            waveform_amp_phase_wrap=_cbbhx.waveform_amp_phase_wrap,
            get_phenomhm_ringdown_frequencies=_cbbhx.get_phenomhm_ringdown_frequencies,
            get_phenomd_ringdown_frequencies=_cbbhx.get_phenomd_ringdown_frequencies,
            interpolate_wrap=_cbbhx.interpolate_wrap,
            speciallike=_cbbhx.speciallike,
            new_hdyn_prep=_cbbhx.new_hdyn_prep,
            new_hdyn_like=_cbbhx.new_hdyn_like,
            xp=cupy,
        )

class BBHxCuda12xBackend(Cuda12xBackend, BBHxBackend):
    """Implementation of CUDA 12.x backend"""
    _backend_name : str = "bbhx_backend_cuda12x"
    _name = "bbhx_cuda12x"
    
    def __init__(self, *args, **kwargs):
        Cuda12xBackend.__init__(self, *args, **kwargs)
        BBHxBackend.__init__(self, self.cuda12x_module_loader())
        
    @staticmethod
    def cuda12x_module_loader():
        try:
            import bbhx_backend_cuda12x.cbbhx  # Phase BBHx.pybind.bulk: sole BBHx backend module
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda12x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda12x' backend requires cupy", pip_deps=["cupy-cuda12x"]
            ) from e

        _cbbhx = bbhx_backend_cuda12x.cbbhx.BBHxComputationWrapGPU()

        return BBHxBackendMethods(
            hdyn_wrap=_cbbhx.hdyn_wrap,
            direct_like_wrap=_cbbhx.direct_like_wrap,
            direct_sum_wrap=_cbbhx.direct_sum_wrap,
            InterpTDI_wrap=_cbbhx.InterpTDI_wrap,
            LISA_response_wrap=_cbbhx.LISA_response_wrap,
            waveform_amp_phase_wrap=_cbbhx.waveform_amp_phase_wrap,
            get_phenomhm_ringdown_frequencies=_cbbhx.get_phenomhm_ringdown_frequencies,
            get_phenomd_ringdown_frequencies=_cbbhx.get_phenomd_ringdown_frequencies,
            interpolate_wrap=_cbbhx.interpolate_wrap,
            speciallike=_cbbhx.speciallike,
            new_hdyn_prep=_cbbhx.new_hdyn_prep,
            new_hdyn_like=_cbbhx.new_hdyn_like,
            xp=cupy,
        )

"""List of existing backends, per default order of preference."""
# TODO: __all__ ?


