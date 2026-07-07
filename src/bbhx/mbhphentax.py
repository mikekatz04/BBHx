"""MBH phentax LISA-response builders + domain-projection adapters.

Carved out of the LISA global-fit settings files (2026-07-01) into BBHx (the
sprint's MBH-physics owner, home of :class:`bbhx.mbhtdionfly.MBHTDIonFly` and the
``phentax`` extra). Provides both:

* the **legacy pyResponse path** — a :class:`IMRPhenomTHMWaveform` adapter around
  ``phentax.waveform.IMRPhenomTHM`` (returns ``hp + 1j*hx`` for
  ``lisatools ResponseWrapper``), plus ``get_mbh_phentax_response_wrapper`` and
  the ``MBHWaveWrap`` domain adapter; and
* the **TDI-on-the-fly path** — ``get_mbh_tdionfly_gen`` (builds a
  :class:`bbhx.mbhtdionfly.MBHTDIonFly` fed by phentax coarse-grained amp/phase)
  and the ``MBHTDIonFlyWaveWrap`` domain adapter.

Plus the phentax sampling<->waveform transform containers.

``phentax`` (external, ``asantini29/phentax``, not on PyPI) installs via the
``phentax`` extra (``pip install 'bbhx[phentax]'``); it is imported lazily so
this module is importable without it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from eryn.utils.transform import TransformContainer

from lisatools.detector import ESAOrbits, EqualArmlengthOrbits, Orbits
from lisatools.domains import TDSettings, TDSignal
from lisatools.response.directresponse import ResponseWrapper
from lisatools.response.tdiconfig import TDIConfig
from lisatools.utils.constants import YRSID_SI

from .utils.transform import m1_m2_to_mT_q, mT_q  # (mT, q) <-> (m1, m2), q = m2/m1

# Process-wide caches (injection + template paths share one instance).
_MBH_WAVE_GEN_CACHE: dict = {}
_MBH_TDIONFLY_GEN_CACHE: dict = {}

# phentax IMRPhenomTHM TDI-on-the-fly defaults (validated 2026-06-15).
# coarse_graining_scale_factor=48 resolves the merger/ringdown (~0.886 s spacing
# at merger) and drops the legacy<->on-the-fly mismatch to mm 6.3e-7 (>1mHz);
# the phentax default 12 under-resolves it and leaves a spurious ~5e-22 floor.
MBH_TDIONFLY_HIGHER_MODES = (21, 33, 44)
MBH_TDIONFLY_TOL = 1e-12
MBH_TDIONFLY_COARSE_SCALE = 48.0


class IMRPhenomTHMWaveform:
    """LISA-side IMRPhenomTHM (phentax) waveform generator.

    Mirrors the call-signature contract that ``lisatools ResponseWrapper``
    expects (and that :class:`lisatools.sources.sobbh.SOBBHWaveform` implements
    for SOBBHs):

    * ``__init__(Tobs, dt, t0, ...)`` builds the internal uniform time grid +
      caches a single :class:`phentax.IMRPhenomTHM` instance.
    * ``__call__(m1, m2, s1z, s2z, dist, phi_ref, inc, psi, lam, beta,
      t_plunge, **kwargs) -> complex Array`` evaluates the polarisation-rotated
      waveform on the internal time grid and returns ``hp + 1j*hx``.

    ``lam`` and ``beta`` are accepted in the call signature so ``ResponseWrapper``
    can read them off (via ``index_lambda`` / ``index_beta``) and apply the LISA
    sky projection; the waveform itself does not use them.

    The merger is placed at ``t = t_plunge`` within the observation window.
    phentax produces its waveform on a grid centred at the merger (``t = 0``),
    so we ask it for ``t_min = -t_plunge`` and ``T = Tobs``; the returned
    ``num_steps = ceil(Tobs/dt)`` samples then align one-to-one with the target
    grid because the spacing is ``delta_t = dt`` on both.
    """

    def __init__(
        self,
        Tobs: float,
        dt: float,
        t0: float = 0.0,
        higher_modes: Optional[object] = "all",
        include_negative_modes: bool = True,
        coarse_grain: bool = False,
        force_backend: str = "cpu",
        pad_zeros: bool = True,
    ):
        # Lazy-import phentax so the module-level import-time cost is paid only
        # when the waveform is actually constructed.
        from phentax.waveform import IMRPhenomTHM

        self.Tobs = Tobs
        self.dt = dt
        self.t0 = t0
        self.force_backend = force_backend
        self.pad_zeros = pad_zeros
        N = int(round(Tobs / dt))
        self._N = N
        self._times_np = np.arange(N) * dt + t0

        self._phentax = IMRPhenomTHM(
            higher_modes=higher_modes,
            include_negative_modes=include_negative_modes,
            coarse_grain=coarse_grain,
            T=Tobs,
        )

    @property
    def N(self) -> int:
        return self._N

    @property
    def times(self) -> np.ndarray:
        return self._times_np

    def __call__(
        self,
        m1,
        m2,
        s1z,
        s2z,
        dist,
        phi_ref,
        inc,
        psi,
        lam,
        beta,
        t_plunge,
        **kwargs,
    ):
        """Evaluate the polarisation-rotated waveform on the internal time grid.

        Returns a complex array ``h = hp + 1j*hx`` so the call signature matches
        what ``lisatools ResponseWrapper`` expects (it reads ``h.real`` /
        ``h.imag`` to get the two polarisations). Set ``flip_hx=True`` on
        ``ResponseWrapper`` to get the standard ``h.real - 1j*h.imag`` convention.

        ``dist`` is in Gpc; converted to Mpc inside (phentax convention).
        ``lam`` / ``beta`` are accepted but unused here — they're forwarded
        through ``ResponseWrapper`` for sky-projection. ``t_plunge`` is the merger
        time relative to the start of the observation; phentax internally uses
        merger == 0.
        """
        del lam, beta  # consumed by ResponseWrapper, not the source waveform
        # ResponseWrapper injects ``T`` and ``dt`` into kwargs; ignore.
        kwargs.pop("T", None)
        kwargs.pop("dt", None)
        kwargs.pop("convert_to_ra_dec", None)

        dist_mpc = float(dist) * 1.0e3  # Gpc -> Mpc (phentax convention)
        t_plunge_f = float(t_plunge)

        # phentax compute_polarizations returns (times, mask, h_plus, h_cross)
        # with merger at t=0. Setting ``t_min = -t_plunge`` and ``T = Tobs``
        # makes the returned ``ceil(T/dt)`` samples align with our target
        # ``[0, Tobs)`` grid sample-for-sample.
        _, mask_phen, hp_phen, hx_phen = self._phentax.compute_polarizations(
            m1=float(m1),
            m2=float(m2),
            chi1z=float(s1z),
            chi2z=float(s2z),
            distance=dist_mpc,
            phi_ref=float(phi_ref),
            inclination=float(inc),
            psi=float(psi),
            delta_t=float(self.dt),
            t_min=-t_plunge_f,
            t_ref=-t_plunge_f,
            T=float(self.Tobs),
        )

        # phentax always returns a leading batch axis (shape (1, N) for a
        # single-source call); squeeze it so the wrapped ``ResponseWrapper``
        # (which uses ``len(h)`` to size the projection) sees a 1D array.
        hp = np.asarray(hp_phen, dtype=np.float64).reshape(-1)
        hx = np.asarray(hx_phen, dtype=np.float64).reshape(-1)
        mask = np.asarray(mask_phen, dtype=bool).reshape(-1)

        # Zero out invalid samples flagged by phentax's adaptive/uniform grid.
        hp = np.where(mask, hp, 0.0)
        hx = np.where(mask, hx, 0.0)

        # Pad with zeros (or clip) to exactly N so ``ResponseWrapper`` and the
        # WDM transform downstream see the right shape.
        if hp.shape[-1] < self._N:
            if self.pad_zeros:
                pad = self._N - hp.shape[-1]
                hp = np.pad(hp, (0, pad), mode="constant")
                hx = np.pad(hx, (0, pad), mode="constant")
            else:
                raise ValueError(
                    f"phentax produced {hp.shape[-1]} samples but N={self._N}; "
                    "either set pad_zeros=True or increase the input grid."
                )
        elif hp.shape[-1] > self._N:
            hp = hp[: self._N]
            hx = hx[: self._N]

        return hp + 1j * hx


def get_mbh_phentax_response_wrapper(
    *,
    Tobs: float,
    dt: float,
    t_start: float,
    tdi_config: TDIConfig,
    tdi_chan: str = "XYZ",
    role: str = "template",
    order: int = 40,
    t_buffer: float = 3e4,
    higher_modes: Optional[object] = "all",
    orbits: Optional[Orbits] = None,
    force_backend: str = "cpu",
):
    """Build (and cache) a :class:`ResponseWrapper` around :class:`IMRPhenomTHMWaveform`.

    A single generator is reused between the synthetic-injection data loader and
    the template move so the slow phentax setup runs once per
    ``(Tobs, dt, t_start, tdi_chan, order, force_backend, str(higher_modes))``
    key (``role`` is intentionally excluded).
    """
    key = (Tobs, dt, t_start, tdi_chan, order, force_backend, str(higher_modes))
    if key in _MBH_WAVE_GEN_CACHE:
        return _MBH_WAVE_GEN_CACHE[key]

    waveform_gen = IMRPhenomTHMWaveform(
        Tobs=Tobs,
        dt=dt,
        t0=t_start,
        higher_modes=higher_modes,
        force_backend=force_backend,
    )

    response_kwargs = {
        "Tobs": Tobs / YRSID_SI,
        "dt": dt,
        # Indices match IMRPhenomTHMWaveform.__call__:
        #   (m1, m2, s1z, s2z, dist, phi_ref, inc, psi, lam, beta, t_plunge)
        #              0   1   2    3    4     5    6    7    8     9     10
        "index_lambda": 8,
        "index_beta": 9,
        "flip_hx": True,
        "force_backend": force_backend,
        "tdi": tdi_config,
        "tdi_chan": tdi_chan,
        "order": order,
        "remove_garbage": "zero",
        "is_ecliptic_latitude": True,
        "t_buffer": t_buffer,
    }

    if orbits is None:
        orbits = EqualArmlengthOrbits(force_backend=force_backend)
    wave_gen = ResponseWrapper(
        waveform_gen,
        orbits=orbits,
        t0=t_start,
        **response_kwargs,
    )
    _MBH_WAVE_GEN_CACHE[key] = wave_gen
    return wave_gen


class MBHWaveWrap:
    """Run the cached MBH ResponseWrapper and project to the run's domain.

    Output is a :class:`~lisatools.domains.DomainBase` subclass (FDSignal /
    WDMSignal / ...) so ACA dispatch and the MBH move's ``get_waveform_here``
    land on the right kernels. Sky coords pass through in the orbits frame
    directly (sprint convention: SSB ecliptic) — no per-call frame conversion.
    """

    def __init__(
        self,
        wave_gen,
        td_settings: TDSettings,
        target_domain,
        td_window=None,
        runtime_kwargs: Optional[dict] = None,
        nchannels: Optional[int] = None,
    ):
        self.wave_gen = wave_gen
        self.td_settings = td_settings
        self.target_domain = target_domain
        self.td_window = td_window
        self.runtime_kwargs = runtime_kwargs or {}
        self.nchannels = nchannels

    def __call__(self, *params, **kwargs):
        call_kwargs = dict(self.runtime_kwargs)
        call_kwargs.update(kwargs)
        arr = np.asarray(self.wave_gen(*params, **call_kwargs))
        if self.nchannels is not None:
            arr = arr[: self.nchannels]
        return TDSignal(arr, self.td_settings).transform(
            self.target_domain, window=self.td_window
        )


def get_mbh_tdionfly_gen(
    *,
    dt: float,
    t_start: float,
    dur_s: float,
    tdi_config: TDIConfig,
    orbits: Optional[Orbits] = None,
    waveform_duration: Optional[float] = None,
    higher_modes: Sequence[int] = MBH_TDIONFLY_HIGHER_MODES,
    coarse_scale: float = MBH_TDIONFLY_COARSE_SCALE,
    tol: float = MBH_TDIONFLY_TOL,
    dt_min: float = 0.1,
    force_backend: str = "cpu",
):
    """Build (and cache) a :class:`bbhx.mbhtdionfly.MBHTDIonFly`.

    Mirrors the validated recipe in ``scripts/mbh/mbh_likelihood_compare.py``: a
    phentax ``IMRPhenomTHM`` with coarse-grained amp/phase output feeding the
    LISA TDI-on-the-fly response. ``t_start`` is the epoch the merger time is
    referenced to (the merger lands at ``t_start + t_merge``); ``dur_s`` is both
    the phentax ``T`` and the generator's observation window.
    """
    higher_modes = tuple(higher_modes)
    key = (
        dt, t_start, dur_s, force_backend, waveform_duration, higher_modes,
        coarse_scale, tol, dt_min, id(orbits),
    )
    if key in _MBH_TDIONFLY_GEN_CACHE:
        return _MBH_TDIONFLY_GEN_CACHE[key]

    from phentax.waveform import IMRPhenomTHM

    from .mbhtdionfly import MBHTDIonFly

    wave_gen = IMRPhenomTHM(
        higher_modes=list(higher_modes),
        include_negative_modes=True,
        t_low_fit=True,
        coarse_grain=True,
        coarse_graining_scale_factor=coarse_scale,
        atol=tol,
        rtol=tol,
        T=dur_s,
    )

    if orbits is None:
        orbits = ESAOrbits(force_backend=force_backend)

    gen = MBHTDIonFly(
        wave_gen,
        orbits,
        tdi_config,
        dt,
        dur_s,
        t0=t_start,
        dt_min=dt_min,
        waveform_duration=waveform_duration if waveform_duration is not None else dur_s,
        force_backend=force_backend,
    )
    _MBH_TDIONFLY_GEN_CACHE[key] = gen
    return gen


class MBHTDIonFlyWaveWrap:
    """Adapter: :class:`bbhx.mbhtdionfly.MBHTDIonFly` TD output -> run-domain signal.

    Consumes the MBH **waveform** basis produced by
    ``make_mbh_transform_container().both_transforms`` —
    ``(m1, m2, s1z, s2z, dist[Mpc], phi_ref, inc, psi, ra, dec, t_plunge)``
    (psi before ra/dec) — and reorders it to the ``MBHTDIonFly`` call order
    ``(..., inc, ra, dec, psi, t_merge)`` before evaluating on the data time grid
    (``t_arr``) with ``combine=True``.
    """

    def __init__(
        self,
        wave_gen,
        t_arr: np.ndarray,
        td_settings: TDSettings,
        target_domain,
        td_window=None,
        runtime_kwargs: Optional[dict] = None,
        nchannels: Optional[int] = None,
    ):
        self.wave_gen = wave_gen
        self.t_arr = t_arr
        self.td_settings = td_settings
        self.target_domain = target_domain
        self.td_window = td_window
        self.runtime_kwargs = runtime_kwargs or {}
        self.nchannels = nchannels

    def raw_td(self, *params, **kwargs):
        """Combined TD TDI channels on the data grid (before domain projection)."""
        call_kwargs = dict(self.runtime_kwargs)
        call_kwargs.update(kwargs)
        # The TDI-on-the-fly generator reads (ra, dec) in the orbits frame
        # directly; drop the legacy ResponseWrapper kwarg if present.
        call_kwargs.pop("convert_to_ra_dec", None)
        m1, m2, s1z, s2z, dist, phi_ref, inc, psi, ra, dec, t_plunge = params
        arr = np.asarray(
            self.wave_gen(
                m1, m2, s1z, s2z, dist, phi_ref, inc, ra, dec, psi, t_plunge,
                upsample_t_arr=self.t_arr, combine=True, **call_kwargs,
            )
        )
        if self.nchannels is not None:
            arr = arr[: self.nchannels]
        return arr

    def __call__(self, *params, **kwargs):
        arr = self.raw_td(*params, **kwargs)
        return TDSignal(arr, self.td_settings).transform(
            self.target_domain, window=self.td_window
        )


def make_mbh_phentax_transform_container() -> TransformContainer:
    """Sampling-basis -> waveform-basis transform for the legacy phentax MBH path.

    In-place transform over the shared basis names: after it fires, the value at
    ``logM`` is ``m1`` (via ``(logM, q) -> mT_q``, ``q = m2/m1``), ``q`` is
    ``m2``, ``cos_iota`` is ``inc``, ``sin_beta`` is ``beta``.
    :class:`IMRPhenomTHMWaveform.__call__` then receives them positionally as
    ``(m1, m2, s1z, s2z, dist, phi_ref, inc, psi, lam, beta, t_plunge)``.

    Unlike the legacy bbhx MBH transform this does NOT apply ``LISA_to_SSB``
    (phentax + :class:`ResponseWrapper` consume ecliptic ``(lam, beta)`` directly
    with ``is_ecliptic_latitude=True``) nor ``gpc_to_mpc``
    (:class:`IMRPhenomTHMWaveform` converts Gpc -> Mpc inside its ``__call__``).
    """
    basis = [
        "logM", "q", "s1z", "s2z", "dist", "phi_ref",
        "cos_iota", "psi", "lam", "sin_beta", "t_plunge",
    ]
    return TransformContainer(
        input_basis=basis,
        output_basis=basis,
        parameter_transforms={
            "logM": np.exp,
            ("logM", "q"): mT_q,  # (M_total, q) -> (m1, m2)
            "cos_iota": np.arccos,  # cos_iota -> inc
            "sin_beta": np.arcsin,  # sin_beta -> beta
        },
        fill_dict={},
        inverse_parameter_transforms={
            "logM": np.log,
            ("logM", "q"): m1_m2_to_mT_q,  # (m1, m2) -> (M_total, q)
            "cos_iota": np.cos,  # inc -> cos_iota
            "sin_beta": np.sin,  # beta -> sin_beta
        },
    )


# Waveform (full) + sampling basis for the TDI-on-the-fly path. NOTE the order
# differs from the legacy ResponseWrapper basis: (lam, beta) come before psi,
# dist is Mpc, and the merger time is named t_merger.
MBH_TDIONFLY_FULL_BASIS = [
    "m1", "m2", "s1z", "s2z", "dist", "phi_ref",
    "inc", "lam", "beta", "psi", "t_merger",
]
MBH_TDIONFLY_SAMPLED_BASIS = [
    "mT", "q", "s1z", "s2z", "dist", "phi_ref",
    "cosinc", "lam", "sinbeta", "psi", "t_merger",
]


def make_mbh_tdionfly_transform_container() -> TransformContainer:
    """Sampling-basis -> waveform-basis transform for the TDI-on-the-fly path.

    ``(mT, q)`` with ``q = m2/m1`` maps to ``(m1, m2)``; ``cosinc`` / ``sinbeta``
    invert to the angles. ``dist`` passes through in Mpc.
    """
    return TransformContainer(
        input_basis=MBH_TDIONFLY_SAMPLED_BASIS,
        output_basis=MBH_TDIONFLY_FULL_BASIS,
        parameter_transforms={
            ("mT", "q"): mT_q,
            "cosinc": np.arccos,
            "sinbeta": np.arcsin,
        },
        key_map={"mT": "m1", "q": "m2", "cosinc": "inc", "sinbeta": "beta"},
        inverse_parameter_transforms={
            ("mT", "q"): m1_m2_to_mT_q,
            "cosinc": np.cos,
            "sinbeta": np.sin,
        },
    )
