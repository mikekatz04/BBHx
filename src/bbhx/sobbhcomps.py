"""SOBBH chunked-heterodyne Python frontend (BBHx-side wrapper): :class:`SOBBHWDMComputations`.

Phase 3L.7p-followup (2026-06-06): SOBBH-specific code lives in BBHx,
GB-specific code lives in GBGPU, and the shared chunked-heterodyne
machinery they both build on lives in
:class:`lisatools.chunked_het.WDMComputationsBase`. This file is the
SOBBH side of that split -- a thin sub-class that only sets the
SOBBH-specific routing constants. Everything else (chunk geometry,
WDM window, layer-grouping, ``fill_global_wdm`` / ``get_ll_wdm`` /
``get_swap_ll_wdm`` / ``get_ll_grad_wdm``) is inherited from the LAT
base.

Pre-2026-06-06 this class lived at ``gbgpu.gbcomps.SOBBHWDMComputations``
-- importing it from there now emits ``ImportError``. Update your
import to ``from bbhx.sobbhcomps import SOBBHWDMComputations``.
"""
from __future__ import annotations

from lisatools.chunked_het import WDMComputationsBase


class SOBBHWDMComputations(WDMComputationsBase):
    """Stellar-origin BBH analog of ``GBWDMComputations``.

    Same chunked-heterodyne pipeline as the GB version, only the routing
    constants differ:

    * Backend family: ``bbhx_<flavor>`` (carries ``SOBBHComputationGroupWrap``).
    * Backend wrap class: ``SOBBHComputationGroupWrap`` instead of
      ``GBComputationGroupWrap``.
    * Kernel-name family: ``sobbh_wdm_het_{fill_global, get_ll, swap_ll, ...}``.
    * Per-source parameter count: ``11`` (vs 9 for GB).
    * Carrier-frequency column for layer-grouping: ``params[:, 5] = f_low``.

    Source intrinsic amp / phase come from the C++ side's
    ``SOBBHTDIonTheFly`` pointer (a JAX path would use
    :class:`bbhx.jax.sources.sobbh.JaxSOBBHSource`, but BBHx registers no ``bbhx_jax``
    backend, so the ``jax`` entry of the inherited ``supported_backends`` is never
    available), so the
    only difference at the Python layer is which backend wrap class is
    invoked and how the parameter vector is interpreted -- everything else
    (chunk geometry, WDM window, layer-grouping, narrow-band dispatch,
    gradient hooks, ``fill_global``) is inherited unchanged.

    Param order (matches ``lisatools.response.tdionfly.SOBBHTDIonTheFly``):

        ``(m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta)``.

    Constructor arguments are those of :class:`lisatools.chunked_het.WDMComputationsBase`
    (``wdm_settings``, ``t_ref``, ``Nt_sub``, ``n_pad``, ``N_sparse``, ``tukey_alpha``,
    ``use_tukey``, ``N_cp_sig``, ``N_cp_orbit``, ``orbits``, ``tdi_config``,
    ``force_backend``, ``d_d``, ``tdi_type``, ``t_obs_start``); a bare ``force_backend``
    flavor resolves to a ``bbhx_<flavor>`` backend. The global-fit production runs do not
    build this class (``SOBBH_LIKELIHOOD=lookup`` uses the LISA Analysis Tools
    ``SOBBHLookupRouter``; this class is reached only with ``SOBBH_LIKELIHOOD=chunked``).

    Example:
        .. code-block:: python

            from bbhx.sobbhcomps import SOBBHWDMComputations

            # wdm_set: a lisatools.domains.WDMSettings; orbits: a configured Orbits
            chunked_sobbh = SOBBHWDMComputations(
                wdm_set, t_ref=t_ref, Nt_sub=256, n_pad=32, N_sparse=256,
                orbits=orbits, tdi_config="2nd generation", d_d=0.0, tdi_type="XYZ",
            )
    """

    # SOBBH chunked-het dispatches through bbhx_<flavor> backends, which
    # carry SOBBHComputationGroupWrap (+ SOBBHTDIonTheFlyWrap + the
    # BBHx-specific bound methods + the LAT-side LISA-response Wraps via
    # the BBHxBackend composition).
    _BACKEND_PREFIX = "bbhx"
    _WRAP_ATTR = "SOBBHComputationGroupWrap"
    _METHOD_PREFIX = "sobbh_wdm_het"
    _NPARAMS = 11
    _F0_PARAM_INDEX = 5   # SOBBHTDIonTheFly: params[5] = f_low
