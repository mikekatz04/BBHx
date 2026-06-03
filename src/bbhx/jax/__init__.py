"""JAX implementations of BBH / SOBBH-specific source classes.

Absorbed from ``fastlisaresponse.jax`` at Phase 3G of the sprint reorg.
Generic LISA-physics JAX (response, orbits, WDM) lives in
:mod:`lisatools.jax`. Subpackage import is gated on ``import jax``.
"""
from __future__ import annotations

try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    _HAS_JAX = True
except (ImportError, ModuleNotFoundError):
    _HAS_JAX = False

if _HAS_JAX:
    from . import sources

    __all__ = ["sources"]
else:
    __all__ = []
