"""Extension points of the aqueous model.

The aqueous residual Helmholtz energy of ms-eCPA is the sum of a physical
(SRK/Huron-Vidal) term, Wertheim association, Debye-Hueckel and the
Maribo-Mogensen permittivity closure. A downstream package can add a
Helmholtz contribution, and modify the permittivity closure, through the
extension object registered here. With nothing registered -- the state of
this package on import -- every extension quantity takes its neutral value
and the kernels evaluate exactly the expressions they would evaluate
without this module.

An extension object is any object providing some of these methods; a
missing method means "neutral".

``aqueous_terms(T, rho, x1w, x2w, x3w, xiw) -> AqueousTerms``
    Called once per evaluation of the aqueous state, in both
    ``lnphi._aq_core_state`` and ``eos.residual_elv``, only when ions are
    present. ``rho`` is the molar density [mol/m3] at the raw EoS volume;
    ``xiw`` are the ion mole fractions. Arguments may be complex (the inner
    Newton differentiates by complex step), so an implementation must be
    analytic in them.

``extra_aqueous_seeds(Zw_liq) -> sequence of (Zw, eps_r, chi1w)``
    Further starting points for the aqueous inner Newton, tried after every
    built-in seed has failed.

``colligative_dielectric(n, V, T) -> (density_factor, dipole_weight)``
``colligative_helmholtz(n, V, T) -> float``
    The same two modifications for the total-mole-basis assembly in
    ``scripts/total_mole_basis.py`` (extensive A/RT), with
    ``colligative_cache_key()`` returning a hashable that changes whenever
    the extension's own parameters change.

``AqueousTerms`` fields, all at fixed T:

    density_factor   r: the permittivity closure (Clausius-Mossotti M and
                     the dipole term) is evaluated at density rho*r
                     instead of rho. V-chains of those factors carry r
                     (it is assumed that d(V/r)/dV = 1 at fixed N and
                     d(V/r)/dN_k = 0 for the three partitioning species).
    dipole_weight    w: multiplies the water dipole term of the closure.
    V_dw_dV          V (dw/dV) at fixed N.
    N_dw_dNw         N (dw/dN_water) at fixed V.
    Z                additive contribution to the compressibility factor.
    lnphi            additive contributions to (ln phi_H2O, ln phi_CH4,
                     ln phi_CO2).
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AqueousTerms", "NEUTRAL", "register", "unregister", "get",
           "active"]


@dataclass(frozen=True)
class AqueousTerms:
    density_factor: object = 1.0
    dipole_weight: object = 1.0
    V_dw_dV: object = 0.0
    N_dw_dNw: object = 0.0
    Z: object = 0.0
    lnphi: tuple = (0.0, 0.0, 0.0)


#: The values the kernels use when no extension is registered.
NEUTRAL = AqueousTerms()

_EXT = None


def register(ext) -> None:
    """Install ``ext`` as THE extension (replacing any previous one)."""
    global _EXT
    _EXT = ext


def unregister() -> None:
    global _EXT
    _EXT = None


def get():
    """The registered extension object, or None."""
    return _EXT


def active() -> bool:
    """True when an extension is registered and reports itself enabled.

    An extension may define ``enabled()``; without it, registration alone
    counts as active. Consumers that hold their own copy of the model (the
    compiled backend) use this to decide whether they can still answer.
    """
    if _EXT is None:
        return False
    f = getattr(_EXT, "enabled", None)
    return True if f is None else bool(f())


def aqueous_terms(T, rho, x1w, x2w, x3w, xiw):
    """Kernel entry: the registered extension's terms, or NEUTRAL."""
    if _EXT is None:
        return NEUTRAL
    f = getattr(_EXT, "aqueous_terms", None)
    if f is None:
        return NEUTRAL
    out = f(T, rho, x1w, x2w, x3w, xiw)
    return NEUTRAL if out is None else out


def extra_aqueous_seeds(Zw_liq):
    if _EXT is None:
        return ()
    f = getattr(_EXT, "extra_aqueous_seeds", None)
    return () if f is None else tuple(f(Zw_liq))
