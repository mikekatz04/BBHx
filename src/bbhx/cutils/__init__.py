"""BBHx backend definitions.

Phase 3L.7k (2026-06-04): BBHx backends now compose the LAT
``LISAToolsBackend`` surface (Orbits / WDM / FD / Spline / response
Wraps) with BBHx's own native symbols
(`SOBBHTDIonTheFlyWrap`, `SOBBHComputationGroupWrap`, plus the
PhenomHM / Response / WaveformBuild / Likelihood / Interpolate bound
methods). Each module loader imports both
``lisatools_backend_<flavor>.pycppdetector`` and
``bbhx_backend_<flavor>.cbbhx`` so a single ``bbhx.get_backend(name)``
call returns a backend object carrying every native symbol the BBH
pipelines need.

This is the BBHx-side counterpart to the post-Phase-3L.7k LAT backend
extension; the previously-separate ``fastlisaresponse_<flavor>``
backend family (defined in lisa-on-gpu) is being retired in favor of
this composition.
"""

from __future__ import annotations

import dataclasses
import typing

from gpubackendtools.exceptions import *
from gpubackendtools.gpubackendtools import (
    CpuBackend,
    Cuda11xBackend,
    Cuda12xBackend,
    Cuda13xBackend,
)
from lisatools.cutils import (
    LISAToolsBackend,
    LISAToolsBackendMethods,
)

from ..utils.exceptions import *


@dataclasses.dataclass
class BBHxBackendMethods(LISAToolsBackendMethods):
    """Container of native symbols exposed by a BBHx backend.

    Extends :class:`lisatools.cutils.LISAToolsBackendMethods` with:

    1. The BBH-specific bound methods that previously lived on this
       class pre-Phase-3L.7k (waveform / response / likelihood / interp
       wraps from BBHxComputationWrap).
    2. The SOBBH source-class Wraps carved to BBHx at Phase 3L.8
       (SOBBHTDIonTheFlyWrap + SOBBHComputationGroupWrap).
    """

    # ---- BBHx native methods (BBHxComputationWrap; pre-3L.7k) ----
    # NB: interpolate_wrap was retired here at the 2026-06-05 GBT-dedup
    # pass; cubic-spline construction routes through
    # gpubackendtools.interpolate.CubicSplineInterpolant (gbt_backend_*).
    hdyn_wrap: typing.Callable[(...), None]
    direct_like_wrap: typing.Callable[(...), None]
    direct_sum_wrap: typing.Callable[(...), None]
    InterpTDI_wrap: typing.Callable[(...), None]
    LISA_response_wrap: typing.Callable[(...), None]
    waveform_amp_phase_wrap: typing.Callable[(...), None]
    get_phenomhm_ringdown_frequencies: typing.Callable[(...), None]
    get_phenomd_ringdown_frequencies: typing.Callable[(...), None]
    speciallike: typing.Callable[(...), None]
    new_hdyn_prep: typing.Callable[(...), None]
    new_hdyn_like: typing.Callable[(...), None]

    # ---- SOBBH source-class Wraps (Phase 3L.8) ----
    SOBBHTDIonTheFlyWrap: object
    SOBBHComputationGroupWrap: object


class BBHxBackend(LISAToolsBackend):
    """Mixin attaching BBHx-specific symbols on top of a LAT backend.

    Inherits :class:`lisatools.cutils.LISAToolsBackend` so concrete
    BBHx backends expose every LAT native symbol (OrbitsWrap,
    TDIConfigWrap, WDM/FD/Spline wraps, TDITypeDict) plus the
    BBH-specific bound methods and SOBBH Wraps. Consumer code can do
    ``self.backend.OrbitsWrap``,
    ``self.backend.SOBBHTDIonTheFlyWrap``, and
    ``self.backend.waveform_amp_phase_wrap(...)`` on the same object.
    """

    # BBH-specific bound methods
    hdyn_wrap: typing.Callable[(...), None]
    direct_like_wrap: typing.Callable[(...), None]
    direct_sum_wrap: typing.Callable[(...), None]
    InterpTDI_wrap: typing.Callable[(...), None]
    LISA_response_wrap: typing.Callable[(...), None]
    waveform_amp_phase_wrap: typing.Callable[(...), None]
    get_phenomhm_ringdown_frequencies: typing.Callable[(...), None]
    get_phenomd_ringdown_frequencies: typing.Callable[(...), None]
    speciallike: typing.Callable[(...), None]
    new_hdyn_prep: typing.Callable[(...), None]
    new_hdyn_like: typing.Callable[(...), None]
    # SOBBH source-class Wraps
    SOBBHTDIonTheFlyWrap: object
    SOBBHComputationGroupWrap: object

    def __init__(self, bbhx_backend_methods):
        # Populate LAT-side fields first.
        assert isinstance(bbhx_backend_methods, BBHxBackendMethods)
        LISAToolsBackend.__init__(self, bbhx_backend_methods)

        # BBH-specific.
        self.hdyn_wrap = bbhx_backend_methods.hdyn_wrap
        self.direct_like_wrap = bbhx_backend_methods.direct_like_wrap
        self.direct_sum_wrap = bbhx_backend_methods.direct_sum_wrap
        self.InterpTDI_wrap = bbhx_backend_methods.InterpTDI_wrap
        self.LISA_response_wrap = bbhx_backend_methods.LISA_response_wrap
        self.waveform_amp_phase_wrap = bbhx_backend_methods.waveform_amp_phase_wrap
        self.get_phenomhm_ringdown_frequencies = bbhx_backend_methods.get_phenomhm_ringdown_frequencies
        self.get_phenomd_ringdown_frequencies = bbhx_backend_methods.get_phenomd_ringdown_frequencies
        self.speciallike = bbhx_backend_methods.speciallike
        self.new_hdyn_prep = bbhx_backend_methods.new_hdyn_prep
        self.new_hdyn_like = bbhx_backend_methods.new_hdyn_like
        # SOBBH.
        self.SOBBHTDIonTheFlyWrap = bbhx_backend_methods.SOBBHTDIonTheFlyWrap
        self.SOBBHComputationGroupWrap = bbhx_backend_methods.SOBBHComputationGroupWrap


# --- module-loader helpers -----------------------------------------------


def _lat_methods_from_pycppdetector(_lat_pd, *, gpu: bool, xp):
    """Build the LAT-side portion of BBHxBackendMethods."""
    suffix = "GPU" if gpu else "CPU"
    return {
        "OrbitsWrap": getattr(_lat_pd, f"OrbitsWrap{suffix}"),
        "Orbits": getattr(_lat_pd, f"Orbits{suffix}"),
        "check_orbits": _lat_pd.check_orbits,
        "TDSplineTDIWaveformWrap": getattr(_lat_pd, f"TDSplineTDIWaveformWrap{suffix}"),
        "FDSplineTDIWaveformWrap": getattr(_lat_pd, f"FDSplineTDIWaveformWrap{suffix}"),
        "LISAResponseWrap": getattr(_lat_pd, f"LISAResponseWrap{suffix}"),
        "LISAResponse": getattr(_lat_pd, f"LISAResponse{suffix}"),
        "TDIConfigWrap": getattr(_lat_pd, f"TDIConfigWrap{suffix}"),
        "TDIConfig": getattr(_lat_pd, f"TDIConfig{suffix}"),
        "CubicSplineWrap_responselisa": getattr(_lat_pd, f"CubicSplineWrap{suffix}_responselisa"),
        "WDMSettingsWrap": getattr(_lat_pd, f"WDMSettingsWrap{suffix}"),
        "WDMDomainWrap": getattr(_lat_pd, f"WDMDomainWrap{suffix}"),
        "FDDomainWrap": getattr(_lat_pd, f"FDDomainWrap{suffix}"),
        "TDITypeDict": {"XYZ": _lat_pd.TDI_XYZ, "AET": _lat_pd.TDI_AET, "AE": _lat_pd.TDI_AE},
        "xp": xp,
    }


def _bbhx_methods_from_cbbhx(_cbbhx, *, gpu: bool):
    """Build the BBHx-specific portion of BBHxBackendMethods."""
    suffix = "GPU" if gpu else "CPU"
    _bbcomp = getattr(_cbbhx, f"BBHxComputationWrap{suffix}")()
    methods = {
        "hdyn_wrap": _bbcomp.hdyn_wrap,
        "direct_like_wrap": _bbcomp.direct_like_wrap,
        "direct_sum_wrap": _bbcomp.direct_sum_wrap,
        "InterpTDI_wrap": _bbcomp.InterpTDI_wrap,
        "LISA_response_wrap": _bbcomp.LISA_response_wrap,
        "waveform_amp_phase_wrap": _bbcomp.waveform_amp_phase_wrap,
        "get_phenomhm_ringdown_frequencies": _bbcomp.get_phenomhm_ringdown_frequencies,
        "get_phenomd_ringdown_frequencies": _bbcomp.get_phenomd_ringdown_frequencies,
        # SOBBH source-class Wraps (Phase 3L.8).
        "SOBBHTDIonTheFlyWrap": getattr(_cbbhx, f"SOBBHTDIonTheFlyWrap{suffix}"),
        "SOBBHComputationGroupWrap": getattr(_cbbhx, f"SOBBHComputationGroupWrap{suffix}"),
    }
    # speciallike + new_hdyn_* are GPU-only (their .cu sources use GPU
    # kernel-launch syntax that doesn't compile on CPU); matches the
    # prior Cython-era setup.
    if gpu:
        methods["speciallike"] = _bbcomp.speciallike
        methods["new_hdyn_prep"] = _bbcomp.new_hdyn_prep
        methods["new_hdyn_like"] = _bbcomp.new_hdyn_like
    else:
        methods["speciallike"] = None
        methods["new_hdyn_prep"] = None
        methods["new_hdyn_like"] = None
    return methods


# --- concrete backends ---------------------------------------------------


class BBHxCpuBackend(CpuBackend, BBHxBackend):
    """CPU backend backed by ``lisatools_backend_cpu.pycppdetector`` +
    ``bbhx_backend_cpu.cbbhx``."""

    _backend_name = "bbhx_backend_cpu"
    _name = "bbhx_cpu"

    def __init__(self, *args, **kwargs):
        CpuBackend.__init__(self, *args, **kwargs)
        BBHxBackend.__init__(self, self.cpu_methods_loader())

    @staticmethod
    def cpu_methods_loader() -> BBHxBackendMethods:
        try:
            import bbhx_backend_cpu.cbbhx
            import lisatools_backend_cpu.pycppdetector
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException("'cpu' backend could not be imported.") from e

        numpy = BBHxCpuBackend.check_numpy()
        return BBHxBackendMethods(
            **_lat_methods_from_pycppdetector(
                lisatools_backend_cpu.pycppdetector, gpu=False, xp=numpy
            ),
            **_bbhx_methods_from_cbbhx(bbhx_backend_cpu.cbbhx, gpu=False),
        )


def _make_cuda_loader(flavor):
    """Generate a flavor-specific cuda module loader function."""

    def _loader() -> BBHxBackendMethods:
        bb_mod_name = f"bbhx_backend_{flavor}"
        lat_mod_name = f"lisatools_backend_{flavor}"
        try:
            import importlib

            bb_mod = importlib.import_module(f"{bb_mod_name}.cbbhx")
            lat_mod = importlib.import_module(f"{lat_mod_name}.pycppdetector")
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                f"'{flavor}' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                f"'{flavor}' backend requires cupy",
                pip_deps=[f"cupy-{flavor}"],
            ) from e

        return BBHxBackendMethods(
            **_lat_methods_from_pycppdetector(lat_mod, gpu=True, xp=cupy),
            **_bbhx_methods_from_cbbhx(bb_mod, gpu=True),
        )

    return _loader


class BBHxCuda11xBackend(Cuda11xBackend, BBHxBackend):
    """CUDA 11.x backend."""

    _backend_name: str = "bbhx_backend_cuda11x"
    _name = "bbhx_cuda11x"
    cuda11x_module_loader = staticmethod(_make_cuda_loader("cuda11x"))

    def __init__(self, *args, **kwargs):
        Cuda11xBackend.__init__(self, *args, **kwargs)
        BBHxBackend.__init__(self, self.cuda11x_module_loader())


class BBHxCuda12xBackend(Cuda12xBackend, BBHxBackend):
    """CUDA 12.x backend."""

    _backend_name: str = "bbhx_backend_cuda12x"
    _name = "bbhx_cuda12x"
    cuda12x_module_loader = staticmethod(_make_cuda_loader("cuda12x"))

    def __init__(self, *args, **kwargs):
        Cuda12xBackend.__init__(self, *args, **kwargs)
        BBHxBackend.__init__(self, self.cuda12x_module_loader())


class BBHxCuda13xBackend(Cuda13xBackend, BBHxBackend):
    """CUDA 13.x backend."""

    _backend_name: str = "bbhx_backend_cuda13x"
    _name = "bbhx_cuda13x"
    cuda13x_module_loader = staticmethod(_make_cuda_loader("cuda13x"))

    def __init__(self, *args, **kwargs):
        Cuda13xBackend.__init__(self, *args, **kwargs)
        BBHxBackend.__init__(self, self.cuda13x_module_loader())
