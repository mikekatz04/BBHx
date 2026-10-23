"""JAX SOBBH (stellar-origin black-hole binary) source.

Thin class wrapper around the module-level ``amplitude(t, params)`` /
``gw_phase(t, params)`` / ``gw_frequency_t(t, params)`` functions exposed by the prototype
``sobbhtaylert3.py``. That file is the single source of truth for the PN expressions; the
C++ ``SOBBHTDIonTheFly`` kernel (now ``bbhx/cutils/sobbh_tdi_on_the_fly.{hh,cu}``) was
ported from the same place, so the JAX path and the C++ path stay in sync as long as both
load the same PN code.

``sobbhtaylert3.py`` is not part of the installed package: it is looked up in up to six
parent directories of this file, or at ``$SOBBH_TAYLOR_T3_PATH``, on the first call of
:meth:`JaxSOBBHSource.amplitude` / ``phase`` / ``f`` (``FileNotFoundError`` when absent;
it is not present in the BBHx checkout, while LISA Analysis Tools keeps a copy at
``scripts/sobbh/sobbhtaylert3.py`` that ``SOBBH_TAYLOR_T3_PATH`` can point at). Importing
the module needs ``jax`` and LISA Analysis Tools' ``lisatools.jax.response.base``.

Parameter layout matches the C++ class header
(``bbhx/cutils/sobbh_tdi_on_the_fly.hh``):

    params[0]  = m1            (solar masses)
    params[1]  = m2            (solar masses)
    params[2]  = s1            (dimensionless spin, primary)
    params[3]  = s2            (dimensionless spin, secondary)
    params[4]  = distance      (parsecs)
    params[5]  = f_low         (Hz, GW frequency at t_ref)
    params[6]  = phi_c         (rad, reference orbital phase)
    params[7]  = inc           (rad, inclination)
    params[8]  = psi           (rad, polarization)
    params[9]  = lam           (rad, ecliptic longitude)
    params[10] = beta          (rad, ecliptic latitude)
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Tuple

import jax.numpy as jnp

from lisatools.jax.response.base import JaxAmpPhaseSource


def _load_sobbh_pn():
    """Locate and import ``sobbhtaylert3.py`` from the repo root.

    The file lives outside the installed package (it is the reference
    prototype kept at the repo root). We resolve it relative to the
    package install location by walking up parents looking for the
    file. The env var ``SOBBH_TAYLOR_T3_PATH`` overrides this.
    """
    env = os.environ.get("SOBBH_TAYLOR_T3_PATH")
    if env and Path(env).is_file():
        target = Path(env)
    else:
        here = Path(__file__).resolve()
        target = None
        for parent in here.parents[:6]:
            candidate = parent / "sobbhtaylert3.py"
            if candidate.is_file():
                target = candidate
                break
        if target is None:
            raise FileNotFoundError(
                "Could not locate sobbhtaylert3.py. Set SOBBH_TAYLOR_T3_PATH "
                "to point at it explicitly."
            )

    spec = importlib.util.spec_from_file_location("_sobbh_pn", str(target))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_pn = None


def _pn_mod():
    """Return the PN module, loading it with :func:`_load_sobbh_pn` on first use (cached)."""
    global _pn
    if _pn is None:
        _pn = _load_sobbh_pn()
    return _pn


class JaxSOBBHSource(JaxAmpPhaseSource):
    """Stellar-origin binary black-hole TaylorT3-style PN source.

    Plugin for the LISA Analysis Tools JAX TDI-on-the-fly engine
    (:class:`lisatools.jax.response.base.JaxAmpPhaseSource`): 11 parameters per source in
    the layout of the module docstring, with the sky angles at ``inc_index = 7``,
    ``psi_index = 8``, ``lam_index = 9``, ``beta_index = 10``. Shown in the BBHx SOBBH
    tutorial; no code in the workspace instantiates this copy (GBGPU's
    ``SOBBHTDIonTheFlyWrapJAX`` imports a ``gbgpu.jax.sources.sobbh`` module that GBGPU
    does not contain).

    Args:
        t_ref: Reference time at which ``f_low`` is the GW frequency
            (seconds). Same as ``SOBBHTDIonTheFly::t_ref`` in C++. Stored only: the
            amplitude, phase and frequency are evaluated at ``t`` directly (see
            :meth:`amplitude`).

    Example:
        .. code-block:: python

            import jax.numpy as jnp
            from bbhx.jax.sources import JaxSOBBHSource

            src = JaxSOBBHSource(t_ref=0.0)
            # m1, m2, s1, s2, distance [pc], f_low [Hz], phi_c, inc, psi, lam, beta
            params = jnp.array([30.0, 25.0, 0.1, 0.2, 5e8, 0.012, 0.3, 0.8, 0.5, 1.0, 0.3])
            amp = src.amplitude(jnp.linspace(0.0, 1e6, 100), params)  # needs sobbhtaylert3.py
    """

    n_params: int = 11
    param_names: Tuple[str, ...] = (
        "m1", "m2", "s1", "s2",
        "distance", "f_low", "phi_c",
        "inc", "psi", "lam", "beta",
    )
    # Sky-angle indices for SOBBH (shifted by +2 vs UCB).
    inc_index: int = 7
    psi_index: int = 8
    lam_index: int = 9
    beta_index: int = 10

    def __init__(self, t_ref: float):
        self.t_ref = float(t_ref)

    def amplitude(self, t: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """Delegate to ``sobbhtaylert3.amplitude(t, params)``.

        Note: ``t_ref`` is implicit in the SOBBH PN model here -- ``f_low``
        is the GW frequency at the time origin, and this method evaluates at ``t``
        directly (not ``t - t_ref``); ``t_ref`` does not enter the formula. The C++
        ``SOBBHTDIonTheFly`` kernel evaluates at ``t - t_ref`` since 2026-07-30, so the two
        agree only for ``t_ref = 0``. The same holds for :meth:`phase` and :meth:`f`.

        Args:
            t: Time(s) (s).
            params: 1D parameter vector of length 11.

        Returns:
            jnp.ndarray: Amplitude, broadcasting with ``t``.
        """
        return _pn_mod().amplitude(t, params)

    def phase(self, t: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """Delegate to ``sobbhtaylert3.gw_phase(t, params)``: GW phase (rad) at ``t``."""
        return _pn_mod().gw_phase(t, params)

    def f(self, t: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """Delegate to ``sobbhtaylert3.gw_frequency_t(t, params)``: GW frequency (Hz) at ``t``."""
        return _pn_mod().gw_frequency_t(t, params)
