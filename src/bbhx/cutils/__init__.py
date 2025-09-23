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

class BBHxBackend:
    hdyn_wrap: typing.Callable[(...), None]
    direct_like_wrap: typing.Callable[(...), None]
    direct_sum_wrap: typing.Callable[(...), None]
    InterpTDI_wrap: typing.Callable[(...), None]
    LISA_response_wrap: typing.Callable[(...), None]
    waveform_amp_phase_wrap: typing.Callable[(...), None]
    get_phenomhm_ringdown_frequencies: typing.Callable[(...), None]
    interpolate_wrap: typing.Callable[(...), None]

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
            import bbhx_backend_cpu.likelihood
            import bbhx_backend_cpu.waveformbuild
            import bbhx_backend_cpu.response
            import bbhx_backend_cpu.phenomhm
            import bbhx_backend_cpu.interp
            
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cpu' backend could not be imported."
            ) from e

        numpy = BBHxCpuBackend.check_numpy()

        return BBHxBackendMethods(
            hdyn_wrap=bbhx_backend_cpu.likelihood.hdyn_wrap,
            direct_like_wrap=bbhx_backend_cpu.likelihood.direct_like_wrap,
            direct_sum_wrap=bbhx_backend_cpu.waveformbuild.direct_sum_wrap,
            InterpTDI_wrap=bbhx_backend_cpu.waveformbuild.InterpTDI_wrap,
            LISA_response_wrap=bbhx_backend_cpu.response.LISA_response_wrap,
            waveform_amp_phase_wrap=bbhx_backend_cpu.phenomhm.waveform_amp_phase_wrap,
            get_phenomhm_ringdown_frequencies=bbhx_backend_cpu.phenomhm.get_phenomhm_ringdown_frequencies,
            get_phenomd_ringdown_frequencies=bbhx_backend_cpu.phenomhm.get_phenomd_ringdown_frequencies,
            interpolate_wrap=bbhx_backend_cpu.interp.interpolate_wrap,
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
            import bbhx_backend_cuda11x.likelihood
            import bbhx_backend_cuda11x.waveformbuild
            import bbhx_backend_cuda11x.response
            import bbhx_backend_cuda11x.phenomhm
            import bbhx_backend_cuda11x.interp

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

        return BBHxBackendMethods(
            hdyn_wrap=bbhx_backend_cuda11x.likelihood.hdyn_wrap,
            direct_like_wrap=bbhx_backend_cuda11x.likelihood.direct_like_wrap,
            direct_sum_wrap=bbhx_backend_cuda11x.waveformbuild.direct_sum_wrap,
            InterpTDI_wrap=bbhx_backend_cuda11x.waveformbuild.InterpTDI_wrap,
            LISA_response_wrap=bbhx_backend_cuda11x.response.LISA_response_wrap,
            waveform_amp_phase_wrap=bbhx_backend_cuda11x.phenomhm.waveform_amp_phase_wrap,
            get_phenomhm_ringdown_frequencies=bbhx_backend_cuda11x.phenomhm.get_phenomhm_ringdown_frequencies,
            get_phenomd_ringdown_frequencies=bbhx_backend_cuda11x.phenomhm.get_phenomd_ringdown_frequencies,
            interpolate_wrap=bbhx_backend_cuda11x.interp.interpolate_wrap,
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
            import bbhx_backend_cuda12x.likelihood
            import bbhx_backend_cuda12x.waveformbuild
            import bbhx_backend_cuda12x.response
            import bbhx_backend_cuda12x.phenomhm
            import bbhx_backend_cuda12x.interp

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

        return BBHxBackendMethods(
            hdyn_wrap=bbhx_backend_cuda12x.likelihood.hdyn_wrap,
            direct_like_wrap=bbhx_backend_cuda12x.likelihood.direct_like_wrap,
            direct_sum_wrap=bbhx_backend_cuda12x.waveformbuild.direct_sum_wrap,
            InterpTDI_wrap=bbhx_backend_cuda12x.waveformbuild.InterpTDI_wrap,
            LISA_response_wrap=bbhx_backend_cuda12x.response.LISA_response_wrap,
            waveform_amp_phase_wrap=bbhx_backend_cuda12x.phenomhm.waveform_amp_phase_wrap,
            get_phenomhm_ringdown_frequencies=bbhx_backend_cuda12x.phenomhm.get_phenomhm_ringdown_frequencies,
            get_phenomd_ringdown_frequencies=bbhx_backend_cuda12x.phenomhm.get_phenomd_ringdown_frequencies,
            interpolate_wrap=bbhx_backend_cuda12x.interp.interpolate_wrap,
            xp=cupy,
        )

"""List of existing backends, per default order of preference."""
# TODO: __all__ ?


