"""Three-phase (vapor + aqueous + CO2-rich liquid) stability and flash.

Builds on the two-phase functions in ``ms_ecpa.flash`` and
``ms_ecpa.stability`` and does not change them.

Physical setting
----------------
Below the CO2 critical temperature (~304 K) and near/above the CO2
saturation pressure, the CO2-rich phase condenses: a vapor (V), an
aqueous liquid (Laq), and a CO2-rich liquid (L_CO2) can coexist.  For
the CO2--H2O binary the three-phase locus is a line in (T, P) hugging
the pure-CO2 vapor-pressure curve (Wendland et al. 1999); with CH4 in
the feed it opens into a finite two-dimensional region.  Ions remain
in the aqueous phase only.

Formulation
-----------
Aqueous phase is the reference.  Two K-vectors on the partition basis
(H2O, CH4, CO2):

    Kg_i = y_i / x_part_i        (vapor / aqueous)
    Kl_i = xl_i / x_part_i       (CO2-rich liquid / aqueous)

Both non-aqueous phases are evaluated with the existing gas kernel
(``ms_ecpa.lnphi._gas_residual``); the CO2-rich liquid is simply
the dense root, selected by seeding the inner Newton with
``Z ~ 2 B_mix`` (the same dense seed the two-phase fallback ladders
already use).

The multiphase Rachford-Rice for the phase fractions ``(beta_g,
beta_l)`` is solved as the stationary point of Michelsen's convex
function ``F(beta) = -sum_i z_i ln D_i`` with
``D_i = 1 + beta_g (Kg_i - 1) + beta_l (Kl_i - 1)``, by a damped
Newton restricted to ``D_i > 0``.  A minimiser outside the phase
simplex means a phase vanishes (negative flash); the caller then
falls back to the two-phase solution.

Ion bookkeeping is identical to the two-phase flash: ions never enter
Rachford-Rice; they are reinstated through the affine aqueous closure,
with the chain convention's vapor fraction generalised to
``beta_g + beta_l`` (the closure only needs the *aqueous* fraction).

Entry points
------------
* ``rachford_rice_3p``  — two-K-vector Rachford-Rice.
* ``flash3_kv``         — three-phase SSI + Jex + Newton polish.
* ``stability_flash3``  — hierarchical driver: two-phase
  ``stability_flash`` first, then a tangent-plane test of the
  converged two-phase state against dense CO2-rich (and vapor)
  trials, then ``flash3_kv`` if a third phase is called for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time

import numpy as np

from .parameters import Mw, R, b1, b2, b3
from .flash import (
    wilson_k,
    rachford_rice,
    compute_aqueous_x1w_chain,
    FlashKVResult,
)
from .lnphi import (
    LIQUID_EPSR_MIN,
    lnphi_aq,
    lnphi_gas,
    _aq_residual_3var,
    _aq_lnphi_evaluate_3var,
    _gas_residual,
    _gas_lnphi_evaluate,
    _gas_root_is_physical,
)
from .newton import newton_solve
from .stability import stability_flash, StabilityFlashResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b_mix(w):
    """Co-volume of a 3-species (H2O, CH4, CO2) composition [m3/mol]."""
    w = np.asarray(w, dtype=float)
    return float(b1 * w[0] + b2 * w[1] + b3 * w[2])


def _dense_z_seeds(w, T, P_Pa):
    """Ladder of dense (liquid-like) Z seeds for the gas kernel.

    Ascending from just above the reduced co-volume ``B``: the damped
    Newton walks uphill to the NEAREST root, so starting low finds the
    dense root first.  Starting high (e.g. ``2B``) can land in the
    middle (mechanically unstable) root's basin at low temperature —
    observed at 274 K, 44 bar, where a ``2B`` seed converged to the
    saddle root Z=0.29 instead of the liquid root Z=0.09.
    """
    B = _b_mix(w) * float(P_Pa) / (R * float(T))
    return (max(1.05 * B, B + 0.005), 1.2 * B + 0.02,
            1.5 * B, 2.0 * B, 3.0 * B)


def _solve_gas_kernel(w, T, P_Pa, v_warm, *, dense=False):
    """Inner Newton on the gas kernel with vapor- or dense-root seeding.

    Returns ``(v, lnphi_tuple)`` or raises RuntimeError.  ``v_warm`` is a
    one-element list holding the previous converged ``[Z, chi]`` (or
    None); it is an extra seed for the ladder, not a short circuit — the
    branch the returned root sits on is re-established on every call.
    """
    w = np.asarray(w, dtype=float)

    def _accept(v, ok):
        if not ok:
            return False
        if not (np.isfinite(v[0]) and np.isfinite(v[1])):
            return False
        # chi1h may legitimately exceed 1 (by a lot at water-lean, dense,
        # low-T states) where the H2O-CO2 cross-association strength
        # S13(T) turns negative — the extended Wertheim domain; the
        # production SSI accepts these states too.  Only chi <= 0 is
        # inadmissible (lnphi carries 4 ln chi).  _gas_root_is_physical
        # encodes exactly that rule, and additionally screens chi2h/chi3h
        # (which carry their own 4 ln chi and can go negative when S13<0)
        # and the Zh <= B_mix case that makes log(Zh - B) undefined.
        return (v[0] > 0.0) and (v[1] > 0.0) and _gas_root_is_physical(
            v, w, T, P_Pa)

    # The Newton's seed->root basins are unreliable (a low dense seed
    # can be captured by the MIDDLE, mechanically unstable root —
    # observed at 274 K, 44 bar), so the root is selected structurally:
    # run the whole ladder, collect the distinct converged roots, and
    # pick the extremal one — smallest Z for a dense phase, largest Z
    # for a vapor.
    #
    # The ladder runs on every call, warm or cold.  A single Newton from
    # the previous step's root is accepted by `_accept` on physicality
    # alone, and physicality says nothing about which density branch the
    # root is on — so a slot that once walked across into the other
    # branch is perfectly self-consistent there and stays for every
    # later step (measured at 274-283 K, 38-60 bar: dense slots
    # returning Z ~ 0.55-0.74 where the ladder's dense root is
    # Z ~ 0.09-0.11, the drift persisting to 1e-10 between steps), which
    # voids the `dense` contract this function exists to enforce.  The
    # previous root therefore enters as a continuity rung — it keeps the
    # returned state on the branch the last step converged to whenever
    # that root is still the extremal one — but it no longer decides.
    if dense:
        seeds = [np.array([Zs, c]) for Zs in _dense_z_seeds(w, T, P_Pa)
                 for c in (0.90,)]
        seeds.append(np.array([0.1, 0.99]))
    else:
        B = _b_mix(w) * float(P_Pa) / (R * float(T))
        seeds = [np.array([1.0, 0.99]),
                 np.array([max(B * 1.2 + 0.3, 1.0), 0.99]),
                 np.array([0.7, 0.95]),
                 np.array([2.0, 0.99])]
    if v_warm[0] is not None:
        seeds.insert(0, np.asarray(v_warm[0], dtype=float).copy())
    roots = []
    for v0 in seeds:
        v, ok, _ = newton_solve(_gas_residual, v0, args=(w, T, P_Pa),
                                tol=1e-10, maxiter=80)
        if _accept(v, ok) and not any(abs(v[0] - r[0]) < 1e-7
                                      for r in roots):
            roots.append(v)
    if roots:
        v = (min(roots, key=lambda r: r[0]) if dense
             else max(roots, key=lambda r: r[0]))
        v_warm[0] = v
        l = _gas_lnphi_evaluate(v, w, T, P_Pa)
        return v, (float(np.real(l[0])), float(np.real(l[1])),
                   float(np.real(l[2])))
    raise RuntimeError(
        f"gas-kernel inner Newton failed ({'dense' if dense else 'vapor'} "
        f"seeding) at w={w.tolist()}, T={T:.2f} K, P={P_Pa/1e5:.2f} bar")


def _solve_aq_kernel(x_aq, mi_aq, T, P_Pa, v_warm):
    """Inner Newton on the aqueous kernel (warm-started); returns (v, lnphi)."""
    x_aq = np.asarray(x_aq, dtype=float)
    _Zw_liq = float(P_Pa) * 18.015e-6 / (8.314 * float(T))
    seeds = []
    if v_warm[0] is not None:
        seeds.append(np.asarray(v_warm[0], dtype=float).copy())
    seeds += [np.array([_Zw_liq, 60.0, 0.07]),
              np.array([0.05, 50.0, 0.10]),
              np.array([_Zw_liq, 50.0, 0.80]),
              np.array([_Zw_liq, 78.0, 0.90]),
              np.array([_Zw_liq, 80.0, 0.97]),
              np.array([_Zw_liq, 30.0, 0.10])]
    for v0 in seeds:
        v, ok, _ = newton_solve(_aq_residual_3var, v0,
                                args=(x_aq, np.asarray(mi_aq, dtype=float),
                                      T, P_Pa),
                                tol=1e-10, maxiter=80)
        # Physical guard, the same rule the two-phase kernel applies at
        # ms_ecpa.lnphi (both the primary and the ladder accept tests):
        # chi1w (v[2]) enters lnphi as 4 ln chi, and a relative
        # permittivity (v[1]) is bounded below by that of vacuum, so a
        # non-positive value is the Onsager-Kirkwood closure solved
        # outside its domain rather than a state of any medium.  The 3x3
        # Newton can settle there and satisfy the residual — at 298 K,
        # 1 bar, x_aq=(0.10, 0.85, 0.05) it converges to eps_r = -0.50
        # with chi1w = 0.993 — and v_aq[1] is published directly as
        # ``eps_r``.  Reject and try the next seed.
        #
        # The bound is LIQUID_EPSR_MIN rather than zero, for the reason
        # given with that constant: eps_r > 0 admits a VAPOUR root
        # (Zw ~ 0.5, eps_r ~ 1.0003) as though it were the brine.  This is
        # the AQUEOUS solve; both non-aqueous phases here go through the
        # gas kernel, so nothing in this guard can reject a liquid
        # hydrocarbon, whose permittivity is near 2.
        if ok and (v[2] <= 0 or v[1] <= LIQUID_EPSR_MIN):
            ok = False
        if ok:
            v_warm[0] = v
            l = _aq_lnphi_evaluate_3var(v, x_aq,
                                        np.asarray(mi_aq, dtype=float),
                                        T, P_Pa)
            return v, (float(np.real(l[0])), float(np.real(l[1])),
                       float(np.real(l[2])))
    raise RuntimeError(
        f"aqueous inner Newton failed at x_aq={x_aq.tolist()}, "
        f"T={T:.2f} K, P={P_Pa/1e5:.2f} bar")


# ---------------------------------------------------------------------------
# Two-K-vector Rachford-Rice
# ---------------------------------------------------------------------------

def rachford_rice_3p(Kg, Kl, z, *, beta_init=None, tol=1e-12, maxiter=60):
    """Solve the three-phase Rachford-Rice equations for two phase fractions.

    Minimises the convex Michelsen function

    .. math::
        F(\\beta_g, \\beta_l) = -\\sum_i z_i \\ln D_i, \\qquad
        D_i = 1 + \\beta_g (K^g_i - 1) + \\beta_l (K^l_i - 1)

    over the open region ``D_i > 0`` with a damped Newton (the Hessian
    is positive semi-definite, so the interior stationary point is the
    global minimum).  No simplex constraint is imposed during the solve
    (negative-flash convention); the returned ``status`` classifies the
    minimiser.

    Parameters
    ----------
    Kg : array_like, shape (3,)
        K-values of the vapour relative to the aqueous phase,
        ``y_i / x_part,i``, on the partitioning basis (H2O, CH4, CO2).
        NaN marks an absent species.
    Kl : array_like, shape (3,)
        K-values of the CO2-rich liquid relative to the aqueous phase,
        ``xl_i / x_part,i``.  NaN marks an absent species.
    z : array_like, shape (3,)
        Feed mole fractions of the partitioning species (H2O, CH4, CO2).
        ``z_i == 0`` marks an absent species.
    beta_init : tuple of float, optional
        Starting ``(beta_g, beta_l)``.  By default, half of each two-phase
        Rachford-Rice solution.
    tol : float
        Convergence tolerance on the largest component of the gradient.
    maxiter : int
        Maximum number of Newton iterations.

    Returns
    -------
    beta_g : float
        Vapour mole fraction on the partitioning basis.
    beta_l : float
        CO2-rich-liquid mole fraction on the partitioning basis.  The
        aqueous fraction is ``1 - beta_g - beta_l``.
    status : str
        ``"interior"``: a three-phase solution inside the simplex.
        ``"no-gas"``, ``"no-liquid"`` or ``"no-aqueous"``: the minimiser
        lies outside the simplex across the face of the named phase, so
        that phase vanishes and a two-phase flash applies.
        ``"failed"``: the Newton iteration did not converge.
    """
    Kg = np.asarray(Kg, dtype=float)
    Kl = np.asarray(Kl, dtype=float)
    z = np.asarray(z, dtype=float)
    present = (z > 0) & np.isfinite(Kg) & np.isfinite(Kl)
    zp = z[present]
    ug = Kg[present] - 1.0
    ul = Kl[present] - 1.0

    def _D(bg, bl):
        return 1.0 + bg * ug + bl * ul

    def _grad(bg, bl):
        D = _D(bg, bl)
        return np.array([-np.sum(zp * ug / D), -np.sum(zp * ul / D)])

    def _hess(bg, bl):
        D2 = _D(bg, bl) ** 2
        return np.array([
            [np.sum(zp * ug * ug / D2), np.sum(zp * ug * ul / D2)],
            [np.sum(zp * ug * ul / D2), np.sum(zp * ul * ul / D2)],
        ])

    # Start from the midpoint of the two 2-phase solutions if not given.
    if beta_init is None:
        bg0 = rachford_rice(Kg, z)
        bl0 = rachford_rice(Kl, z)
        b = np.array([0.5 * bg0, 0.5 * bl0])
        # Nudge strictly inside the feasible region.
        if np.any(_D(b[0], b[1]) <= 0):
            b = np.array([0.25 * bg0, 0.25 * bl0])
    else:
        b = np.array([float(beta_init[0]), float(beta_init[1])])
    if np.any(_D(b[0], b[1]) <= 0):
        b = np.array([0.0, 0.0])  # feed point is always feasible (D=1)

    # Negative-flash box: allow moderately negative / super-unity phase
    # fractions so a vanishing phase is detected by the face the
    # minimiser pins to, while preventing runaway along near-null
    # Hessian directions (Kg ≈ Kl makes F flat along beta_g - beta_l).
    BOX_LO, BOX_HI = -0.5, 1.5

    def _classify(bg, bl, pinned):
        bw = 1.0 - bg - bl
        eps = 1e-10
        if bg < -eps or (pinned and bg <= BOX_LO + 1e-9):
            return "no-gas"
        if bl < -eps or (pinned and bl <= BOX_LO + 1e-9):
            return "no-liquid"
        if bw < -eps or (pinned and bg + bl >= 2 * BOX_HI - 1e-9):
            return "no-aqueous"
        return "interior"

    conv = False
    pinned = False
    for _ in range(maxiter):
        g = _grad(b[0], b[1])
        if float(np.max(np.abs(g))) < tol:
            conv = True
            break
        H = _hess(b[0], b[1])
        ridge = 1e-12 * max(float(np.trace(H)), 1.0)
        try:
            step = np.linalg.solve(H + ridge * np.eye(2), -g)
        except np.linalg.LinAlgError:
            break
        # Cap the raw step; phase fractions live in O(1).
        s_norm = float(np.max(np.abs(step)))
        if s_norm > 2.0:
            step *= 2.0 / s_norm
        # Feasibility + descent backtracking on D_i > 0, inside the box.
        lam = 1.0
        f0 = -np.sum(zp * np.log(_D(b[0], b[1])))
        accepted = False
        for _ls in range(40):
            b_try = np.clip(b + lam * step, BOX_LO, BOX_HI)
            D_try = _D(b_try[0], b_try[1])
            if np.all(D_try > 1e-12) and np.any(b_try != b):
                f_try = -np.sum(zp * np.log(D_try))
                if f_try <= f0 + 1e-14:
                    pinned = bool(np.any(b_try <= BOX_LO + 1e-12)
                                  or np.any(b_try >= BOX_HI - 1e-12))
                    b = b_try
                    accepted = True
                    break
            lam *= 0.5
        if not accepted:
            break
    if not conv:
        g = _grad(b[0], b[1])
        if float(np.max(np.abs(g))) < 1e-9:
            conv = True  # accept near-machine stationarity
    bg, bl = float(b[0]), float(b[1])
    if not conv:
        # Not stationary: if the minimiser is pressed against a box face
        # (or clearly outside the simplex), the face names the vanishing
        # phase; otherwise report failure.
        #
        # "Clearly" is the point of DROP_MARGIN.  The eps in _classify is
        # the noise floor of a *stationary* point, where a phase fraction
        # a hair below zero is a real minimiser just outside the simplex.
        # A non-stationary iterate carries no such meaning: a line search
        # that stalled at bg = -5e-10 on its way back inside would be
        # returned as an established "no-gas", and the caller (via
        # _PhaseDrop -> drop_phase -> stability_flash3) treats a definite
        # drop as a physical fact about the state.  Off-face iterates
        # must therefore be a margin outside the simplex before their
        # sign is read as a phase-boundary crossing.
        DROP_MARGIN = 1e-6
        cls = _classify(bg, bl, pinned)
        if (cls != "interior" and not pinned
                and min(bg, bl, 1.0 - bg - bl) > -DROP_MARGIN):
            cls = "failed"
        return bg, bl, (cls if cls != "interior" else "failed")
    return bg, bl, _classify(bg, bl, pinned)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Flash3Result:
    """Result of a three-phase K-value flash (returned by ``flash3_kv``).

    Attributes
    ----------
    T : float
        Temperature [K].
    P : float
        Pressure [bar].
    z : tuple of float, length 3
        Feed mole fractions of the partitioning species (H2O, CH4, CO2).
    mi : tuple of float, length 6
        Input ion molalities (Na+, Cl-, K+, Ca2+, Mg2+, SO4 2-) [mol/kg
        water].
    converged : bool
        True when the fugacity residual is below ``tol``.
    message : str
        "converged", or the reason the flash stopped.
    Kg : tuple of float, length 3
        K-values vapour/aqueous on the partitioning basis
        (``y_i / x_part,i``); NaN for absent species.
    Kl : tuple of float, length 3
        K-values CO2-rich liquid/aqueous on the partitioning basis.
    x_aq : tuple of float, length 3
        Aqueous mole fractions (H2O, CH4, CO2), ions included in the total.
    y_gas : tuple of float, length 3
        Vapour mole fractions (H2O, CH4, CO2).
    x_lco2 : tuple of float, length 3
        CO2-rich-liquid mole fractions (H2O, CH4, CO2).
    beta_g : float
        Vapour mole fraction on the partitioning basis.
    beta_l : float
        CO2-rich-liquid mole fraction on the partitioning basis.
    x1w : float
        Aqueous H2O mole fraction, equal to ``x_aq[0]``.
    Z_aq : float
        Compressibility factor of the aqueous phase.
    Z_gas : float
        Compressibility factor of the vapour.  After convergence the phase
        labels are set so that ``Z_gas >= Z_lco2``.
    Z_lco2 : float
        Compressibility factor of the CO2-rich liquid.
    eps_r : float
        Relative permittivity of the aqueous phase.
    n_iter_ssi : int
        Number of successive-substitution iterations.
    n_iter_newton : int
        Number of Newton iterations on ln K after the substitution phase.
    max_residual : float
        ``max|g|`` over present species and both K-vectors, with ``g`` the
        change of ``ln K`` from one substitution step; ``inf`` on failure.
    fugacities_aq : tuple of float, length 3
        Fugacities (H2O, CH4, CO2) in the aqueous phase [bar].
    fugacities_gas : tuple of float, length 3
        Fugacities in the vapour [bar].
    fugacities_lco2 : tuple of float, length 3
        Fugacities in the CO2-rich liquid [bar].
    mi_aq : tuple of float, length 6
        Equilibrium aqueous ion molalities [mol/kg water]; equal to ``mi``
        under ``"as-aqueous"``.
    salinity_convention : str
        ``"as-aqueous"`` or ``"chain"`` (see ``ms_ecpa.flash.flash_kv``).
    chi_H2O_aq : float
        Fraction of H2O association sites not bonded, aqueous phase.
    chi_H2O_gas : float
        Same quantity in the vapour.
    chi_H2O_lco2 : float
        Same quantity in the CO2-rich liquid.
    drop_phase : str
        Empty unless the flash stopped because a phase vanished:
        ``"no-gas"``, ``"no-liquid"`` or ``"no-aqueous"`` (from
        ``rachford_rice_3p``), ``"failed"`` (the three-phase Rachford-Rice
        did not converge), or ``"merged"`` (the vapour and the liquid
        reached the same state).
    """
    T: float
    P: float
    z: tuple
    mi: tuple
    converged: bool
    message: str
    Kg: tuple                 # vapor/aqueous K (partition basis)
    Kl: tuple                 # CO2-rich-liquid/aqueous K (partition basis)
    x_aq: tuple               # true aqueous mole fractions (H2O, CH4, CO2)
    y_gas: tuple              # vapor mole fractions
    x_lco2: tuple             # CO2-rich-liquid mole fractions
    beta_g: float             # vapor fraction (partition basis)
    beta_l: float             # CO2-rich-liquid fraction (partition basis)
    x1w: float
    Z_aq: float
    Z_gas: float
    Z_lco2: float
    eps_r: float
    n_iter_ssi: int
    n_iter_newton: int
    max_residual: float
    fugacities_aq: tuple      # (f_H2O, f_CH4, f_CO2) [bar]
    fugacities_gas: tuple
    fugacities_lco2: tuple
    mi_aq: tuple = (0.0,) * 6
    salinity_convention: str = "as-aqueous"
    chi_H2O_aq: float = float("nan")
    chi_H2O_gas: float = float("nan")
    chi_H2O_lco2: float = float("nan")
    drop_phase: str = ""      # set when RR3 says a phase vanishes


@dataclass
class StabilityFlash3Result:
    """Result of the three-phase stability flash (from ``stability_flash3``).

    Attributes
    ----------
    phases : int
        Number of phases reported: 1, 2 or 3.
    two_phase : StabilityFlashResult
        Result of the two-phase ``stability_flash`` (stage 1, always run).
        When the two-phase state was replaced by a re-flash (see
        ``stability_flash3``), this holds the re-flashed result, with
        ``route == -1``.
    flash3 : Flash3Result or None
        The three-phase flash when ``phases == 3``; otherwise None.
    tpd3_min : float
        Most negative TPD of the stage-2 trials against the two-phase
        equilibrium.  NaN when stage 2 was skipped or no trial converged.
    message : str
        Short summary of the outcome.
    incumbent_gas_root : str
        ``"vapor"`` or ``"dense"``: the density root of the two-phase
        non-aqueous phase.  Empty when stage 2 did not reach that step.
    attempts : list of str
        One line per step of stages 2 and 3.
    t_stab3_ms : float
        Wall time of the stage-2 test [ms].
    t_flash3_ms : float
        Wall time of the stage-3 flash, including any re-flash [ms].
    t_total_ms : float
        Wall time of the whole call, including stage 1 [ms].
    """
    phases: int                                # 1, 2 or 3
    two_phase: StabilityFlashResult            # stage-1 result (always run)
    flash3: Optional[Flash3Result]             # populated when phases == 3
    tpd3_min: float                            # min TPD of stage-2 trials
    message: str
    incumbent_gas_root: str = ""               # "vapor" | "dense" | ""
    attempts: list = field(default_factory=list)
    t_stab3_ms: float = 0.0
    t_flash3_ms: float = 0.0
    t_total_ms: float = 0.0


# ---------------------------------------------------------------------------
# Three-phase K-value flash
# ---------------------------------------------------------------------------

class _PhaseDrop(Exception):
    """Internal signal: the multiphase RR says a phase leaves the simplex."""

    def __init__(self, which, bg, bl):
        super().__init__(which)
        self.which = which
        self.bg = bg
        self.bl = bl


def flash3_kv(
    T, P, z, mi,
    *,
    Kg_init,
    Kl_init,
    tol=1e-8,
    maxiter=120,
    accelerated=True,
    use_newton=True,
    newton_tol=1e-4,
    max_newton=8,
    salinity_convention="as-aqueous",
    verbose=False,
):
    """Run the three-phase K-value flash (vapour, aqueous, CO2-rich liquid).

    The iteration is that of ``ms_ecpa.flash.flash_kv`` (successive
    substitution on ln K with the acceleration of Jex et al. 2024, then
    Newton iterations), applied to two K-vectors.  The initial K-values
    must be given; ``stability_flash3`` computes them from its stability
    stage.  The vapour is evaluated on the low-density root of the gas
    equation of state and the CO2-rich liquid on the dense root.

    Parameters
    ----------
    T : float
        Temperature [K].
    P : float
        Pressure [bar].
    z : array_like, shape (3,)
        Feed mole fractions of the partitioning species (H2O, CH4, CO2),
        ions excluded; sum to 1.
    mi : array_like, shape (6,)
        Ion molalities (Na+, Cl-, K+, Ca2+, Mg2+, SO4 2-) [mol/kg water].
    Kg_init : array_like, shape (3,)
        Initial K-values vapour/aqueous on the partitioning basis.
    Kl_init : array_like, shape (3,)
        Initial K-values CO2-rich liquid/aqueous on the partitioning basis.
    tol : float
        Convergence tolerance on ``max|g|`` over both K-vectors.
    maxiter : int
        Maximum number of substitution iterations.
    accelerated : bool
        Apply the acceleration of Jex et al. (2024).
    use_newton : bool
        Switch to Newton iterations on ln K once ``max|g| < newton_tol``.
    newton_tol : float
        Residual below which the Newton iterations start.
    max_newton : int
        Maximum number of Newton iterations.
    salinity_convention : {"as-aqueous", "chain"}
        As in ``flash_kv``; under ``"chain"`` the sum ``beta_g + beta_l``
        takes the place of the two-phase vapour fraction.
    verbose : bool
        Print one line per iteration.

    Returns
    -------
    Flash3Result
        Compositions and fractions of the three phases.  If the
        three-phase Rachford-Rice solution leaves the simplex, or the
        vapour and liquid become identical, the flash stops with
        ``converged=False`` and ``drop_phase`` set; ``stability_flash3``
        then uses a two-phase result.

    Raises
    ------
    ValueError
        If ``z`` does not have 3 entries or ``mi`` does not have 6 entries.
    """
    z = np.asarray(z, dtype=float)
    mi = np.asarray(mi, dtype=float)
    if z.shape != (3,):
        raise ValueError("z must have 3 entries: (z_H2O, z_CH4, z_CO2)")
    if mi.shape != (6,):
        raise ValueError("mi must have 6 entries: (Na, Cl, K, Ca, Mg, SO4)")
    present = z > 0
    idx_present = np.where(present)[0]
    n_p = idx_present.size
    P_Pa = P * 1e5

    Kg = np.where(present, np.asarray(Kg_init, dtype=float), np.nan)
    Kl = np.where(present, np.asarray(Kl_init, dtype=float), np.nan)

    v_aq_warm = [None]
    v_gas_warm = [None]
    v_liq_warm = [None]
    beta_warm = [None]

    def _ssi3_step(Kg_in, Kl_in):
        # 1) Two-fraction Rachford-Rice.
        bg, bl, status = rachford_rice_3p(Kg_in, Kl_in, z,
                                          beta_init=beta_warm[0])
        if status != "interior":
            raise _PhaseDrop(status, bg, bl)
        beta_warm[0] = (bg, bl)
        bw = 1.0 - bg - bl

        # 2) Partition-basis compositions.
        with np.errstate(divide="ignore", invalid="ignore"):
            D = 1.0 + bg * (Kg_in - 1.0) + bl * (Kl_in - 1.0)
            x_part = np.where(present, z / D, 0.0)
            y_part = np.where(present, Kg_in * x_part, 0.0)
            l_part = np.where(present, Kl_in * x_part, 0.0)
        for arr in (x_part, y_part, l_part):
            s = float(np.nansum(arr))
            if s > 0:
                arr /= s

        # 3) Aqueous closure (ions).  The chain closure needs only the
        # total non-aqueous fraction; beta_g + beta_l plays the role the
        # single beta played in the two-phase flash.
        if salinity_convention == "chain":
            x1w_true, scale, mi_aq_step, _sig = compute_aqueous_x1w_chain(
                x_part[0], bg + bl, float(z[0]), mi)
            if not np.isfinite(x1w_true) or x1w_true <= 0.0 or scale <= 0.0:
                raise RuntimeError(
                    f"chain closure invalid (bg={bg:.4f}, bl={bl:.4f})")
        else:
            m_tot = float(np.sum(mi))
            x1w_true = (float(x_part[0] / (1.0 + Mw * m_tot * x_part[0]))
                        if x_part[0] > 0 else 0.0)
            scale = 1.0 - Mw * m_tot * x1w_true
            mi_aq_step = mi
        x_aq_loc = (x1w_true,
                    float(x_part[1] * scale) if present[1] else 0.0,
                    float(x_part[2] * scale) if present[2] else 0.0)
        y_loc = tuple(float(v) for v in y_part)
        l_loc = tuple(float(v) for v in l_part)

        if present[0] and x1w_true < 0.02:
            raise RuntimeError(
                f"degenerate-aqueous: x1w={x1w_true:.4f}")

        # 4) lnphi in all three phases (aqueous / vapor root / dense root).
        v_aq, la = _solve_aq_kernel(x_aq_loc, mi_aq_step, T, P_Pa, v_aq_warm)
        v_g, lg = _solve_gas_kernel(y_loc, T, P_Pa, v_gas_warm, dense=False)
        v_l, ll = _solve_gas_kernel(l_loc, T, P_Pa, v_liq_warm, dense=True)

        # Merge guard: if the vapor and liquid states coincide, the
        # three-phase system has collapsed to two phases.
        if (abs(v_g[0] - v_l[0]) < 1e-6
                and float(np.max(np.abs(np.asarray(y_loc) -
                                        np.asarray(l_loc)))) < 1e-6):
            raise _PhaseDrop("merged", bg, bl)

        # 5) K updates.  Same partition-basis correction as flash_kv:
        # K_part = scale * exp(lnphi_aq - lnphi_other).
        lnKg_new = np.full(3, np.nan)
        lnKl_new = np.full(3, np.nan)
        for i in range(3):
            if not present[i]:
                continue
            lnKg_new[i] = np.log(scale) + (la[i] - lg[i])
            lnKl_new[i] = np.log(scale) + (la[i] - ll[i])
        g_loc = np.concatenate([
            (lnKg_new - np.log(np.where(present, Kg_in, 1.0)))[idx_present],
            (lnKl_new - np.log(np.where(present, Kl_in, 1.0)))[idx_present],
        ])
        return dict(
            Kg_new=np.exp(lnKg_new), Kl_new=np.exp(lnKl_new),
            x_aq=x_aq_loc, y_gas=y_loc, x_lco2=l_loc,
            beta_g=bg, beta_l=bl, x1w=x1w_true,
            g=g_loc, la=la, lg=lg, ll=ll, scale=scale,
            v_aq=v_aq, v_g=v_g, v_l=v_l, mi_aq=np.asarray(mi_aq_step),
        )

    def _fail(msg, drop="", it=0, ev=None):
        return Flash3Result(
            T=float(T), P=float(P), z=tuple(z), mi=tuple(mi),
            converged=False, message=msg,
            Kg=tuple(Kg), Kl=tuple(Kl),
            x_aq=(np.nan,) * 3 if ev is None else ev["x_aq"],
            y_gas=(np.nan,) * 3 if ev is None else ev["y_gas"],
            x_lco2=(np.nan,) * 3 if ev is None else ev["x_lco2"],
            beta_g=np.nan if ev is None else ev["beta_g"],
            beta_l=np.nan if ev is None else ev["beta_l"],
            x1w=np.nan if ev is None else ev["x1w"],
            Z_aq=np.nan, Z_gas=np.nan, Z_lco2=np.nan, eps_r=np.nan,
            n_iter_ssi=it, n_iter_newton=0, max_residual=np.inf,
            fugacities_aq=(np.nan,) * 3, fugacities_gas=(np.nan,) * 3,
            fugacities_lco2=(np.nan,) * 3,
            mi_aq=tuple(float(m) for m in mi),
            salinity_convention=str(salinity_convention),
            drop_phase=drop,
        )

    # ── SSI loop ─────────────────────────────────────────────────────────
    g_prev = None
    m_jex = 1.0
    n_iter_ssi = 0
    converged = False
    max_resid = np.inf
    ev = None
    for it in range(maxiter):
        try:
            ev = _ssi3_step(Kg, Kl)
        except _PhaseDrop as pd:
            return _fail(f"phase drop at it={it}: {pd.which} "
                         f"(bg={pd.bg:.4f}, bl={pd.bl:.4f})",
                         drop=pd.which, it=it)
        except RuntimeError as exc:
            return _fail(f"SSI failed at it={it}: {exc}", it=it)

        g = ev["g"]
        max_resid = float(np.max(np.abs(g)))
        if verbose:
            print(f"[KV3] it={it+1:3d} bg={ev['beta_g']:.4f} "
                  f"bl={ev['beta_l']:.4f} x1w={ev['x1w']:.4f} "
                  f"max|g|={max_resid:.3e}")
        if max_resid < tol:
            Kg, Kl = ev["Kg_new"], ev["Kl_new"]
            n_iter_ssi = it + 1
            converged = True
            break
        if use_newton and max_resid < newton_tol:
            n_iter_ssi = it + 1
            break

        # Jex acceleration on the concatenated lnK vector.
        if accelerated and g_prev is not None:
            num_a = float(np.dot(g_prev, g_prev))
            den_a = float(np.dot(g_prev, g_prev - g))
            if abs(den_a) > 1e-30:
                m_jex = float(np.clip(abs(num_a / den_a * m_jex), 1.0, 10.0))
            else:
                m_jex = 1.0
        g_prev = g.copy()

        lnKg = np.log(np.where(present, Kg, 1.0))
        lnKl = np.log(np.where(present, Kl, 1.0))
        lnKg[idx_present] += m_jex * g[:n_p]
        lnKl[idx_present] += m_jex * g[n_p:]
        lnKg = np.clip(lnKg, -30.0, 30.0)
        lnKl = np.clip(lnKl, -30.0, 30.0)
        Kg = np.where(present, np.exp(lnKg), np.nan)
        Kl = np.where(present, np.exp(lnKl), np.nan)
        n_iter_ssi = it + 1

    # ── Newton polish on the concatenated lnK ────────────────────────────
    n_iter_newton = 0
    if (use_newton and not converged and np.isfinite(max_resid)
            and max_resid < newton_tol):
        for _ in range(max_newton):
            try:
                base = _ssi3_step(Kg, Kl)
            except (_PhaseDrop, RuntimeError):
                break
            base_g = base["g"]
            h = 1e-5
            n2 = 2 * n_p
            J = np.zeros((n2, n2))
            failed = False
            for j in range(n2):
                Kg_p, Kl_p = Kg.copy(), Kl.copy()
                if j < n_p:
                    Kg_p[idx_present[j]] *= np.exp(h)
                else:
                    Kl_p[idx_present[j - n_p]] *= np.exp(h)
                try:
                    ev_p = _ssi3_step(Kg_p, Kl_p)
                except (_PhaseDrop, RuntimeError):
                    failed = True
                    break
                J[:, j] = (ev_p["g"] - base_g) / h
            if failed:
                break
            try:
                dlnK = np.linalg.solve(J, -base_g)
            except np.linalg.LinAlgError:
                break
            dlnK = np.clip(dlnK, -5.0, 5.0)
            Kg_n, Kl_n = Kg.copy(), Kl.copy()
            Kg_n[idx_present] *= np.exp(dlnK[:n_p])
            Kl_n[idx_present] *= np.exp(dlnK[n_p:])
            try:
                ev_n = _ssi3_step(Kg_n, Kl_n)
            except (_PhaseDrop, RuntimeError):
                break
            new_resid = float(np.max(np.abs(ev_n["g"])))
            if new_resid > max_resid:
                break
            Kg, Kl, ev = Kg_n, Kl_n, ev_n
            max_resid = new_resid
            n_iter_newton += 1
            if max_resid < tol:
                converged = True
                break

    if ev is None:
        return _fail("no successful SSI step")

    # ── Label hygiene: vapor must be the lighter phase ───────────────────
    Z_g = float(ev["v_g"][0])
    Z_l = float(ev["v_l"][0])
    if Z_l > Z_g:
        # The "liquid" slot converged to the lighter root: swap labels.
        Kg, Kl = Kl, Kg
        ev["Kg_new"], ev["Kl_new"] = ev["Kl_new"], ev["Kg_new"]
        ev["y_gas"], ev["x_lco2"] = ev["x_lco2"], ev["y_gas"]
        ev["beta_g"], ev["beta_l"] = ev["beta_l"], ev["beta_g"]
        ev["lg"], ev["ll"] = ev["ll"], ev["lg"]
        ev["v_g"], ev["v_l"] = ev["v_l"], ev["v_g"]
        Z_g, Z_l = Z_l, Z_g

    # ── Equilibrium aqueous molality (chain diagnostic) ──────────────────
    bg, bl = float(ev["beta_g"]), float(ev["beta_l"])
    x1w = float(ev["x1w"])
    if (salinity_convention == "chain" and np.isfinite(bg + bl)
            and (1.0 - bg - bl) > 0 and x1w > 0):
        # partition-basis water fraction = x1w * (1 + S); see
        # flash.compute_aqueous_x1w_chain.
        _S = Mw * float(z[0]) * float(np.sum(mi)) / (1.0 - bg - bl)
        mi_aq_final = mi * (
            float(z[0]) / ((1.0 - bg - bl) * x1w * (1.0 + _S)))
    else:
        mi_aq_final = mi

    la, lg, ll = ev["la"], ev["lg"], ev["ll"]
    x_aq = ev["x_aq"]
    y_gas = ev["y_gas"]
    x_lco2 = ev["x_lco2"]
    fug_aq = tuple(float(np.exp(la[i]) * x_aq[i] * P) for i in range(3))
    fug_gas = tuple(float(np.exp(lg[i]) * y_gas[i] * P) for i in range(3))
    fug_lco2 = tuple(float(np.exp(ll[i]) * x_lco2[i] * P) for i in range(3))

    msg = ("converged" if converged
           else f"did not converge (max|g|={max_resid:.3e})")
    return Flash3Result(
        T=float(T), P=float(P), z=tuple(float(v) for v in z),
        mi=tuple(float(m) for m in mi),
        converged=bool(converged), message=msg,
        Kg=tuple(float(k) for k in Kg), Kl=tuple(float(k) for k in Kl),
        x_aq=tuple(float(v) for v in x_aq),
        y_gas=tuple(float(v) for v in y_gas),
        x_lco2=tuple(float(v) for v in x_lco2),
        beta_g=bg, beta_l=bl, x1w=x1w,
        Z_aq=float(ev["v_aq"][0]), Z_gas=Z_g, Z_lco2=Z_l,
        eps_r=float(ev["v_aq"][1]),
        n_iter_ssi=int(n_iter_ssi), n_iter_newton=int(n_iter_newton),
        max_residual=float(max_resid),
        fugacities_aq=fug_aq, fugacities_gas=fug_gas,
        fugacities_lco2=fug_lco2,
        mi_aq=tuple(float(m) for m in mi_aq_final),
        salinity_convention=str(salinity_convention),
        chi_H2O_aq=float(ev["v_aq"][2]),
        chi_H2O_gas=float(ev["v_g"][1]),
        chi_H2O_lco2=float(ev["v_l"][1]),
    )


# ---------------------------------------------------------------------------
# Two-phase flash pinned to the DENSE non-aqueous root
# ---------------------------------------------------------------------------

def _flash2_dense(T, P, z, mi, K_init, *,
                  salinity_convention="as-aqueous",
                  tol=1e-8, maxiter=120):
    """Two-phase K-value flash whose non-aqueous phase is held on the
    dense (liquid) density root.

    The production ``ms_ecpa.flash.flash_kv`` cannot guarantee a
    density root: its ``gas_Z_init`` is only a Newton seed, and the
    kernel's seed-to-root basins are unreliable (a dense seed can
    converge onto the vapor root and reproduce the metastable
    equilibrium).  This clone of the SSI uses
    ``_solve_gas_kernel`` with ``dense=True`` — collect-all cold
    start picking the smallest-Z root, warm-start continuity after —
    so the liquid root is structural, not seeded.  Used by
    ``stability_flash3`` to replace a metastable vapor--aqueous
    split above the three-phase pressure.  Returns a fully populated
    ``ms_ecpa.flash.FlashKVResult``.
    """
    z = np.asarray(z, dtype=float)
    mi = np.asarray(mi, dtype=float)
    present = z > 0
    idx = np.where(present)[0]
    P_Pa = P * 1e5
    K = np.where(present, np.asarray(K_init, dtype=float), np.nan)
    v_aq_warm = [None]
    v_g_warm = [None]
    g_prev = None
    m_jex = 1.0
    converged = False
    max_resid = np.inf
    ev = None

    def _step(K_in):
        beta = rachford_rice(K_in, z)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = 1.0 + beta * (K_in - 1.0)
            x_part = np.where(present, z / denom, 0.0)
            y_part = np.where(present, K_in * x_part, 0.0)
        for arr in (x_part, y_part):
            s = float(np.nansum(arr))
            if s > 0:
                arr /= s
        if salinity_convention == "chain":
            x1w, scale, mi_aq, _sig = compute_aqueous_x1w_chain(
                x_part[0], beta, float(z[0]), mi)
            if not np.isfinite(x1w) or x1w <= 0.0 or scale <= 0.0:
                raise RuntimeError("chain closure invalid")
        else:
            m_tot = float(np.sum(mi))
            x1w = (float(x_part[0] / (1.0 + Mw * m_tot * x_part[0]))
                   if x_part[0] > 0 else 0.0)
            scale = 1.0 - Mw * m_tot * x1w
            mi_aq = mi
        x_aq = (x1w,
                float(x_part[1] * scale) if present[1] else 0.0,
                float(x_part[2] * scale) if present[2] else 0.0)
        y = tuple(float(v) for v in y_part)
        if present[0] and x1w < 0.02:
            raise RuntimeError("degenerate aqueous")
        v_aq, la = _solve_aq_kernel(x_aq, mi_aq, T, P_Pa, v_aq_warm)
        v_g, lg = _solve_gas_kernel(y, T, P_Pa, v_g_warm, dense=True)
        lnK_new = np.full(3, np.nan)
        for i in range(3):
            if present[i]:
                lnK_new[i] = np.log(scale) + (la[i] - lg[i])
        g = (lnK_new - np.log(np.where(present, K_in, 1.0)))[idx]
        return dict(K_new=np.exp(lnK_new), x_aq=x_aq, y=y, beta=beta,
                    x1w=x1w, g=g, la=la, lg=lg, v_aq=v_aq, v_g=v_g,
                    mi_aq=np.asarray(mi_aq, dtype=float))

    n_iter = 0
    for it in range(maxiter):
        try:
            ev = _step(K)
        except RuntimeError as exc:
            return FlashKVResult(
                T=float(T), P=float(P), z=tuple(z), mi=tuple(mi),
                converged=False, message=f"dense flash failed: {exc}",
                K=tuple(K), x_aq=(np.nan,) * 3, y_gas=(np.nan,) * 3,
                beta=np.nan, x1w=np.nan, Z_aq=np.nan, Z_gas=np.nan,
                eps_r=np.nan, n_iter_ssi=it, n_iter_newton=0,
                max_residual=np.inf, fugacities_aq=(np.nan,) * 3,
                fugacities_gas=(np.nan,) * 3,
                mi_aq=tuple(float(m) for m in mi),
                salinity_convention=str(salinity_convention))
        g = ev["g"]
        max_resid = float(np.max(np.abs(g)))
        n_iter = it + 1
        if max_resid < tol:
            K = ev["K_new"]
            converged = True
            break
        if g_prev is not None:
            den = float(np.dot(g_prev, g_prev - g))
            if abs(den) > 1e-30:
                m_jex = float(np.clip(
                    abs(float(np.dot(g_prev, g_prev)) / den * m_jex),
                    1.0, 10.0))
            else:
                m_jex = 1.0
        g_prev = g.copy()
        lnK = np.log(np.where(present, K, 1.0))
        lnK[idx] += m_jex * g
        K = np.where(present, np.exp(np.clip(lnK, -30, 30)), np.nan)

    # The message below asserts the non-aqueous phase sits on the dense
    # root, and the caller adopts the result on the strength of it.  The
    # SSI re-establishes the branch at every step, but that is a property
    # of the path; verify it on the state actually being returned.  The
    # check is a fresh (cold) dense ladder at the converged y: if it
    # finds a strictly smaller root, the returned state is not the dense
    # one — the usual reason being that the dense root ceased to exist
    # along the way and the ladder's min-of-found-roots is a vapor root.
    root_note = ""
    if converged:
        try:
            v_chk, _ = _solve_gas_kernel(ev["y"], T, P_Pa, [None], dense=True)
        except RuntimeError as exc:
            converged = False
            root_note = f"dense-root check failed: {exc}"
        else:
            if float(v_chk[0]) < float(ev["v_g"][0]) - 1e-7:
                converged = False
                root_note = (
                    f"non-aqueous phase is not on the dense root "
                    f"(Z={float(ev['v_g'][0]):.4f}, "
                    f"dense root Z={float(v_chk[0]):.4f})")

    la, lg = ev["la"], ev["lg"]
    x_aq, y = ev["x_aq"], ev["y"]
    fug_aq = tuple(float(np.exp(la[i]) * x_aq[i] * P) for i in range(3))
    fug_gas = tuple(float(np.exp(lg[i]) * y[i] * P) for i in range(3))
    return FlashKVResult(
        T=float(T), P=float(P), z=tuple(float(v) for v in z),
        mi=tuple(float(m) for m in mi),
        converged=bool(converged),
        message=("converged (dense non-aqueous root)" if converged else
                 root_note or
                 f"did not converge (max|g|={max_resid:.3e})"),
        K=tuple(float(k) for k in K),
        x_aq=tuple(float(v) for v in x_aq),
        y_gas=tuple(float(v) for v in y),
        beta=float(ev["beta"]), x1w=float(ev["x1w"]),
        Z_aq=float(ev["v_aq"][0]), Z_gas=float(ev["v_g"][0]),
        eps_r=float(ev["v_aq"][1]),
        n_iter_ssi=int(n_iter), n_iter_newton=0,
        max_residual=float(max_resid),
        fugacities_aq=fug_aq, fugacities_gas=fug_gas,
        mi_aq=tuple(float(m) for m in ev["mi_aq"]),
        salinity_convention=str(salinity_convention),
        chi_H2O_aq=float(ev["v_aq"][2]),
        chi_H2O_gas=float(ev["v_g"][1]),
    )


# ---------------------------------------------------------------------------
# Stage-2 stability: test the converged TWO-phase state for a third phase
# ---------------------------------------------------------------------------

def _tpd_trial_3p(d_i, w_init, T, P_Pa, mi, kind, *,
                  tol=1e-8, maxiter=80):
    """TPD SSI of a single trial phase against a two-phase equilibrium.

    ``kind`` selects the trial-phase kernel: ``"vapor"`` (gas root),
    ``"dense"`` (liquid-CO2 root), or ``"aq"``.  Log-space SSI with the
    same Jex acceleration/clipping as ``ms_ecpa.stability.tpd_ssi``.
    Returns ``(tpd, w, Z_trial, n_iter, converged)`` — ``Z_trial`` is
    the trial kernel's converged compressibility factor (NaN for the
    aqueous kernel), needed by the caller to tell a dense root from a
    vapor root at (near-)identical composition.
    """
    W = np.asarray(w_init, dtype=float).copy()
    present = W > 0
    idx = np.where(present)[0]
    lnW = np.full(3, -np.inf)
    lnW[idx] = np.log(np.clip(W[idx], 1e-300, None))
    d_p = np.asarray(d_i, dtype=float)[idx]
    v_warm = [None]
    g_prev = None
    m_jex = 1.0
    tpd = np.nan
    Z_trial = float("nan")
    for it in range(maxiter):
        s = float(np.sum(np.exp(lnW[idx])))
        w = np.zeros(3)
        w[idx] = np.exp(lnW[idx]) / s
        try:
            if kind == "aq":
                lp = lnphi_aq(tuple(w), mi, T, P_Pa)
            else:
                v_k, lp = _solve_gas_kernel(w, T, P_Pa, v_warm,
                                            dense=(kind == "dense"))
                Z_trial = float(v_k[0])
        except RuntimeError:
            return np.nan, w, Z_trial, it, False
        lnW_new_p = d_p - np.asarray(lp)[idx]
        g_p = lnW_new_p - lnW[idx]
        g_max = float(np.max(np.abs(g_p)))
        W_sum = float(np.sum(np.exp(lnW_new_p)))
        tpd = 1.0 - W_sum
        if g_max < tol:
            w = np.zeros(3)
            w[idx] = np.exp(lnW_new_p) / W_sum
            return float(tpd), w, Z_trial, it + 1, True
        if g_prev is not None:
            den = float(np.dot(g_prev, g_prev - g_p))
            if abs(den) > 1e-30:
                m_jex = float(np.clip(
                    abs(float(np.dot(g_prev, g_prev)) / den * m_jex),
                    1.0, 10.0))
            else:
                m_jex = 1.0
        g_prev = g_p.copy()
        lnW[idx] = lnW[idx] + np.clip(m_jex * g_p, -5.0, 5.0)
    w = np.zeros(3)
    w[idx] = np.exp(lnW[idx])
    return float(tpd), w, Z_trial, maxiter, False


def _condensable_slot(z, present, T):
    """Index (1 or 2) of the condensable gas slot, or None.

    A gas can form its own liquid phase below its critical temperature.
    The Wilson critical constants in ``ms_ecpa.flash`` are the
    per-slot source of truth (they track slot rebinds, e.g. a pilot
    swapping CH4 -> C2H6/C3H8).  With two condensable gases present,
    the one dominant in the feed wins.
    """
    from .flash import _TC
    cands = [i for i in (1, 2) if present[i] and float(T) < float(_TC[i])]
    if not cands:
        return None
    return max(cands, key=lambda i: z[i])


def _third_phase_trials(z, present, y_eq, incumbent_dense, c_slot):
    """Trial list for the stage-2 (third-phase) tangent-plane test.

    Each entry: ``(label, kind, w_init)`` with ``kind`` in
    {"vapor", "dense"}.  The first — and physically decisive — trial is
    the *other density root at the incumbent gas composition*: in a
    gas--H2O binary the vapor and the gas-rich liquid differ almost
    entirely by density, not composition.  Compositions rich in the
    condensable gas (slot ``c_slot``) on both roots back it up for
    multicomponent feeds where the third phase differs in composition
    too.
    """
    other = "vapor" if incumbent_dense else "dense"
    trials = [(f"{other} root at incumbent gas composition", other,
               np.asarray(y_eq, dtype=float).copy())]
    o_slot = 1 if c_slot == 2 else 2  # the other (non-condensable) gas
    for xw in (0.005, 0.02):
        w = np.zeros(3)
        w[0] = xw
        if present[o_slot]:
            w[o_slot] = 0.02
        w[c_slot] = 1.0 - float(np.sum(w))
        trials.append((f"dense slot{c_slot}-rich (x_w={xw})", "dense", w))
        trials.append((f"vapor slot{c_slot}-rich (x_w={xw})", "vapor",
                       w.copy()))
    if present[o_slot]:
        w = np.zeros(3)
        w[0] = 0.02
        w[c_slot] = 0.05
        w[o_slot] = 1.0 - float(np.sum(w))
        trials.append((f"vapor slot{o_slot}-rich", "vapor", w))
    return trials


def stability_flash3(
    T, P, z, mi,
    *,
    tpd3_threshold=-1e-4,
    salinity_convention="as-aqueous",
    stability_kwargs=None,
    flash3_kwargs=None,
    **flash_kwargs,
):
    """Run the two-phase stability flash, then test for and solve a third phase.

    The calculation has three stages:

    1. ``ms_ecpa.stability.stability_flash`` (two-phase).
    2. If stage 1 gave a converged two-phase state and a gas is below its
       critical temperature, a tangent-plane test of that state against
       dense (liquid-CO2) and vapour trials.
    3. If a trial gives ``TPD < tpd3_threshold``, ``flash3_kv`` started
       from the two-phase K-values and the K-values of that trial.

    If the three-phase flash does not converge, or a phase fraction is at
    the boundary, the two-phase result is kept.  When the flash reports
    that the vapour or the liquid vanishes (``drop_phase`` "no-gas",
    "no-liquid" or "merged"), a two-phase flash is run from the K-values of
    the new phase; if it converges to a different split, that split
    replaces the stage-1 result, because the stage-1 non-aqueous phase was
    metastable.

    Parameters
    ----------
    T : float
        Temperature [K].
    P : float
        Pressure [bar].
    z : array_like, shape (3,)
        Feed mole fractions of the partitioning species (H2O, CH4, CO2),
        ions excluded; sum to 1.
    mi : array_like, shape (6,)
        Ion molalities (Na+, Cl-, K+, Ca2+, Mg2+, SO4 2-) [mol/kg water].
    tpd3_threshold : float
        A stage-2 trial with ``TPD < tpd3_threshold``, whose state differs
        from both phases of the two-phase result, starts the three-phase
        flash.  The value is closer to zero than the two-phase threshold
        because the stage-2 trials are evaluated against a converged
        equilibrium (residual about 1e-8), not against the feed.
    salinity_convention : {"as-aqueous", "chain"}
        Passed to all flashes (see ``ms_ecpa.flash.flash_kv``).
    stability_kwargs : dict, optional
        Keyword arguments passed to ``michelsen_tpd`` through
        ``stability_flash``.
    flash3_kwargs : dict, optional
        Keyword arguments passed to ``flash3_kv``.
    **flash_kwargs
        Keyword arguments passed to ``stability_flash`` (and from there to
        ``flash_kv``).

    Returns
    -------
    StabilityFlash3Result
        Number of phases, the stage-1 result in ``two_phase``, and the
        three-phase flash in ``flash3`` when ``phases == 3``.
    """
    t0 = time.perf_counter()
    z = np.asarray(z, dtype=float)
    mi = np.asarray(mi, dtype=float)
    present = z > 0
    attempts = []

    r2 = stability_flash(T, P, z, mi,
                         stability_kwargs=stability_kwargs,
                         salinity_convention=salinity_convention,
                         **flash_kwargs)

    inc_root = [""]

    def _wrap(phases, flash3, tpd3_min, msg, t_stab3, t_flash3):
        return StabilityFlash3Result(
            phases=phases, two_phase=r2, flash3=flash3,
            tpd3_min=tpd3_min, message=msg,
            incumbent_gas_root=inc_root[0], attempts=attempts,
            t_stab3_ms=t_stab3, t_flash3_ms=t_flash3,
            t_total_ms=(time.perf_counter() - t0) * 1e3,
        )

    # A third phase needs a condensable gas (one below its critical
    # temperature) and a converged two-phase state to test against.
    fr = r2.flash_result
    if r2.phases != 2 or fr is None or not fr.converged:
        attempts.append("stage2: skipped (no converged two-phase state)")
        return _wrap(r2.phases, None, np.nan,
                     f"two-phase driver: {r2.message}", 0.0, 0.0)
    c_slot = _condensable_slot(z, present, T)
    if c_slot is None:
        attempts.append("stage2: skipped (no condensable gas at this T)")
        return _wrap(2, None, np.nan, "two-phase (no condensable gas)",
                     0.0, 0.0)

    # ── Stage 2: tangent-plane test of the 2-phase equilibrium ──────────
    t_s2 = time.perf_counter()
    P_Pa = P * 1e5
    y_eq = np.asarray(fr.y_gas, dtype=float)
    x_aq_eq = np.asarray(fr.x_aq, dtype=float)
    mi_aq_eq = np.asarray(fr.mi_aq, dtype=float)

    # Incumbent partition-basis aqueous composition (for distinctness
    # checks and the K seeds).
    s_part = float(np.sum(x_aq_eq))
    x_part_eq = x_aq_eq / s_part if s_part > 0 else x_aq_eq

    # Identify the incumbent gas-phase density root by isofugacity with
    # the aqueous phase: solve the kernel from a vapor seed and a dense
    # seed, keep the root that satisfies ln x_aq + lnphi_aq = ln y +
    # lnphi_gas (the converged equilibrium), and remember the other
    # root — it is the natural third-phase candidate.
    try:
        la_eq = lnphi_aq(tuple(x_aq_eq), mi_aq_eq, T, P_Pa)
    except RuntimeError as exc:
        attempts.append(f"stage2: aqueous reference failed ({exc})")
        return _wrap(2, None, np.nan, "two-phase (stage-2 reference failed)",
                     (time.perf_counter() - t_s2) * 1e3, 0.0)
    idx_p = np.where(present)[0]
    lhs = (np.log(np.clip(x_aq_eq, 1e-300, None))
           + np.asarray(la_eq))[idx_p]

    roots = {}
    for kind, dense in (("vapor", False), ("dense", True)):
        vw = [None]
        try:
            v_k, lp_k = _solve_gas_kernel(y_eq, T, P_Pa, vw, dense=dense)
        except RuntimeError:
            continue
        resid = float(np.max(np.abs(
            lhs - (np.log(np.clip(y_eq, 1e-300, None))
                   + np.asarray(lp_k))[idx_p])))
        roots[kind] = (float(v_k[0]), lp_k, resid)
    # The two ladders return the EXTREMAL roots, and the root the
    # two-phase flash converged on need not be one of them: where the gas
    # cubic has three physical roots the equilibrium can sit on the
    # middle one (measured on the CO2-H2O binary at 284 K/55 bar,
    # 290 K/60 bar and 296 K/65 bar: equilibrium Z = 0.386 / 0.390 /
    # 0.365, between a dense 0.121 / 0.138 / 0.159 and a vapor 0.492 /
    # 0.472 / 0.476).  ``fr`` carries that root — flash_kv reports
    # Z_gas/chi_H2O_gas from the root the SSI actually used — so let it
    # compete for the reference.  Without it the tangent plane below is
    # built from a state displaced from the equilibrium by up to 0.24 in
    # ln f, thousands of times the tpd3_threshold the trials are then
    # measured against, and Z_ref names a phase the incumbent is not on.
    if np.isfinite(fr.Z_gas) and np.isfinite(fr.chi_H2O_gas):
        v_i = np.array([float(fr.Z_gas), float(fr.chi_H2O_gas)])
        if _gas_root_is_physical(v_i, y_eq, T, P_Pa):
            l_i = _gas_lnphi_evaluate(v_i, y_eq, T, P_Pa)
            lp_i = (float(np.real(l_i[0])), float(np.real(l_i[1])),
                    float(np.real(l_i[2])))
            resid_i = float(np.max(np.abs(
                lhs - (np.log(np.clip(y_eq, 1e-300, None))
                       + np.asarray(lp_i))[idx_p])))
            roots["incumbent"] = (float(v_i[0]), lp_i, resid_i)
    if not roots:
        attempts.append("stage2: gas reference evaluation failed")
        return _wrap(2, None, np.nan, "two-phase (stage-2 reference failed)",
                     (time.perf_counter() - t_s2) * 1e3, 0.0)
    inc_kind = min(roots, key=lambda k: roots[k][2])
    Z_ref, lp_ref, resid_ref = roots[inc_kind]
    # Everything downstream (the first, physically decisive trial in
    # _third_phase_trials; the reported incumbent_gas_root, which the
    # propane-locus pilot reads to sign its indicator) means vapor-like
    # vs dense-like, so the flash's own root is labelled by the ladder
    # root it lies nearest.
    if inc_kind == "incumbent":
        # Nearest ladder root by Z, with the isofugacity residual as the
        # tie-break: where the cubic has a single root both ladders
        # return it and the two distances differ only by Newton
        # termination noise, which must not be what decides the label.
        ladder = [k for k in roots if k != "incumbent"]
        tolZ = 1e-6 * max(1.0, Z_ref)

        def _rank(k):
            dZ = abs(roots[k][0] - Z_ref)
            return (0.0 if dZ < tolZ else dZ, roots[k][2])

        inc_label = min(ladder, key=_rank) if ladder else "vapor"
    else:
        inc_label = inc_kind
    incumbent_dense = (inc_label == "dense")
    inc_root[0] = inc_label
    attempts.append(
        f"stage2: incumbent gas root = {inc_label} "
        f"({'from flash' if inc_kind == 'incumbent' else 'ladder'}, "
        f"Z={Z_ref:.4f}, isofug resid={resid_ref:.2e})")
    # The tangent plane is only the equilibrium's if the reference state
    # IS the equilibrium.  That is the premise the tight tpd3_threshold
    # rests on: a converged two-phase state reproduces isofugacity to
    # ~1e-11, so a residual anywhere near the trigger means no available
    # root reproduces it and the plane cannot decide a third phase either
    # way.  Report that instead of a verdict measured against a plane
    # displaced by more than the thing being measured.
    RESID_REF_TOL = 1e-6
    if not (resid_ref < RESID_REF_TOL):
        attempts.append(
            f"stage2: no gas root reproduces the incumbent equilibrium "
            f"(best isofug resid={resid_ref:.2e}); third-phase test not "
            f"decidable")
        return _wrap(2, None, np.nan,
                     "two-phase (stage-2 reference not at equilibrium)",
                     (time.perf_counter() - t_s2) * 1e3, 0.0)
    d_i = np.where(present,
                   np.log(np.clip(y_eq, 1e-300, None)) + np.asarray(lp_ref),
                   0.0)

    tpd3_min = np.inf
    best = None  # (label, kind, tpd, w)
    n_tested = 0  # trials that actually produced a TPD
    for label, kind, w0 in _third_phase_trials(z, present, y_eq,
                                               incumbent_dense, c_slot):
        tpd, w, Z_t, n_it, ok = _tpd_trial_3p(d_i, w0, T, P_Pa, mi_aq_eq,
                                              kind)
        if not ok or not np.isfinite(tpd):
            attempts.append(f"stage2 trial [{label}]: no convergence")
            continue
        n_tested += 1
        # Distinctness: a trial duplicates the incumbent gas phase only
        # if BOTH the composition and the density root coincide; in the
        # binary the vapor and the CO2-rich liquid share (nearly) the
        # same composition and differ only in Z.
        same_comp = float(np.max(np.abs(w - y_eq))) < 5e-3
        same_root = (np.isfinite(Z_t)
                     and abs(Z_t - Z_ref) < 1e-3 * max(1.0, Z_ref))
        if same_comp and same_root:
            attempts.append(
                f"stage2 trial [{label}]: tpd={tpd:.3e} (incumbent; skip)")
            continue
        if float(np.max(np.abs(w - x_part_eq))) < 5e-3:
            attempts.append(
                f"stage2 trial [{label}]: tpd={tpd:.3e} (aqueous; skip)")
            continue
        attempts.append(
            f"stage2 trial [{label}]: tpd={tpd:.3e} Z={Z_t:.4f} it={n_it}")
        if tpd < tpd3_min:
            tpd3_min = tpd
            best = (label, kind, tpd, w)
    if not np.isfinite(tpd3_min):
        tpd3_min = np.nan
    t_stab3 = (time.perf_counter() - t_s2) * 1e3

    if best is None or not (tpd3_min < tpd3_threshold):
        if n_tested == 0:
            # No trial produced a TPD at all — every trial kernel failed
            # to converge, and the per-trial notes for that live only in
            # `attempts`.  Stability with respect to a third phase was
            # not established here, so do not say it was: `phases` and
            # `message` are what a sweep aggregates, and this state would
            # otherwise be indistinguishable from a tested-stable one.
            return _wrap(2, None, tpd3_min,
                         "two-phase (third-phase test inconclusive: "
                         "no trial converged)", t_stab3, 0.0)
        return _wrap(2, None, tpd3_min,
                     "two-phase (stable wrt third phase)", t_stab3, 0.0)

    # ── Stage 3: three-phase flash ───────────────────────────────────────
    t_f3 = time.perf_counter()
    label, kind, tpd, w_new = best
    eps = 1e-12
    # Incumbent gas K on the partition basis: the converged flash K.
    Kg0 = np.asarray(fr.K, dtype=float)
    # New-phase K from isofugacity with the incumbent aqueous phase at
    # the trial's own density root: K_i = scale * exp(lnphi_aq_i -
    # lnphi_new_i).  A composition-ratio seed (w/x_part) would be nearly
    # parallel to Kg0 in the binary — the two non-aqueous phases share
    # (almost) one composition and differ in the H2O fugacity channel,
    # which only the lnphi difference resolves.
    try:
        vw_new = [None]
        _, ll_new = _solve_gas_kernel(w_new, T, P_Pa, vw_new,
                                      dense=(kind == "dense"))
        Kl0 = np.where(present,
                       s_part * np.exp(np.asarray(la_eq)
                                       - np.asarray(ll_new)),
                       np.nan)
    except RuntimeError:
        Kl0 = np.where(present,
                       np.clip(w_new, eps, None)
                       / np.clip(x_part_eq, eps, None),
                       np.nan)
    Kg0 = np.where(present, np.clip(Kg0, 1e-6, 1e6), np.nan)
    Kl0 = np.where(present, np.clip(Kl0, 1e-6, 1e6), np.nan)
    # flash3_kv's "gas" slot is vapor-seeded and its "liquid" slot is
    # dense-seeded; when the incumbent is itself the dense phase, the
    # new trial phase is the vapor — swap the seeds accordingly.
    if incumbent_dense and kind == "vapor":
        Kg0, Kl0 = Kl0, Kg0
    attempts.append(
        f"stage3: flash3 seeded from [{label}] (tpd={tpd:.3e})")

    f3kw = dict(flash3_kwargs or {})
    r3 = flash3_kv(T, P, z, mi, Kg_init=Kg0, Kl_init=Kl0,
                   salinity_convention=salinity_convention, **f3kw)
    t_flash3 = (time.perf_counter() - t_f3) * 1e3

    if r3.converged:
        # Sanity: all three phase fractions strictly interior.
        bw = 1.0 - r3.beta_g - r3.beta_l
        if min(r3.beta_g, r3.beta_l, bw) > 1e-6:
            return _wrap(3, r3, tpd3_min,
                         "three-phase (V + Laq + L_CO2)", t_stab3, t_flash3)
        attempts.append(
            f"stage3: converged but boundary fractions "
            f"(bg={r3.beta_g:.3e}, bl={r3.beta_l:.3e}); keeping two-phase")
        return _wrap(2, None, tpd3_min,
                     "two-phase (three-phase split degenerate)",
                     t_stab3, t_flash3)

    # Phase-drop fallback: the negative TPD was real, so the *incumbent*
    # non-aqueous phase may be the metastable one (e.g. a vapor above
    # the three-phase pressure, where the stable split is Laq + L_CO2).
    # Re-flash two-phase seeded from the NEW phase's K and adopt that
    # answer when it converges to a real split at lower Gibbs energy
    # (the TPD test already established the new phase is below the
    # incumbent tangent plane).
    if r3.drop_phase in ("no-gas", "no-liquid", "merged"):
        if r3.drop_phase == "no-gas":
            # The vapor vanishes: the stable split is Laq + L_dense.
            # Re-flash with the non-aqueous phase PINNED to the dense
            # root — a seed on the production flash is not enough, the
            # kernel's seed-to-root basins can fall back to the vapor
            # root and reproduce the metastable equilibrium.
            fr_new = _flash2_dense(T, P, z, mi, Kl0,
                                   salinity_convention=salinity_convention)
        else:
            from .flash import flash_kv
            fr_new = flash_kv(T, P, z, mi, K_init=Kg0,
                              salinity_convention=salinity_convention)
        moved = (fr_new.converged and float(np.max(np.abs(
            np.asarray(fr_new.x_aq) - np.asarray(fr.x_aq)))) > 1e-9)
        if (moved and np.isfinite(fr_new.beta)
                and 1e-4 < fr_new.beta < 1.0 - 1e-4):
            attempts.append(
                f"stage3: {r3.drop_phase} -> re-flashed two-phase from "
                "the new phase's K (incumbent was metastable)")
            r2_new = StabilityFlashResult(
                phases=2, tpd_min=r2.tpd_min, stable=False,
                flash_result=fr_new, stability=r2.stability,
                attempts=list(r2.attempts) + list(attempts),
                message="two-phase (re-flashed: incumbent non-aqueous "
                        "phase was metastable)",
                T=r2.T, P=r2.P, z=r2.z, mi=r2.mi,
                t_stab_ms=r2.t_stab_ms, t_flash_ms=r2.t_flash_ms,
                t_total_ms=r2.t_total_ms,
            )
            return StabilityFlash3Result(
                phases=2, two_phase=r2_new, flash3=None,
                tpd3_min=tpd3_min,
                message="two-phase (stable split replaces metastable "
                        "incumbent)",
                incumbent_gas_root=inc_root[0], attempts=attempts,
                t_stab3_ms=t_stab3, t_flash3_ms=t_flash3,
                t_total_ms=(time.perf_counter() - t0) * 1e3,
            )
    attempts.append(f"stage3: flash3 not converged ({r3.message}); "
                    "keeping two-phase")
    return _wrap(2, None, tpd3_min,
                 f"two-phase (flash3 fallback: {r3.drop_phase or 'no-conv'})",
                 t_stab3, t_flash3)
