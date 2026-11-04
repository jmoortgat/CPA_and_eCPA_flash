"""Multiphase K-value flash for the multi-salt eCPA.

Architecture
------------
The flash is a successive-substitution iteration (SSI) on the
log-K-vector for the three *partitioning* species (H2O, CH4, CO2),
with Jex 2024 acceleration and an optional Newton polish.  Ions are
non-partitioning (``K_ion ≡ 0``) and enter only via the aqueous-phase
ion mole-fraction sum.

The functions in this module:

* ``aqueous_closure`` — residual of the multi-salt form of the scalar
  equation for ``x1w`` used in Paper I (``ecpa.flash``).  It reduces to the
  NaCl form when the ions are (Na⁺, Cl⁻) only; a unit test checks this.
* ``compute_aqueous_x1w_from_KP`` — direct algebraic closure for
  ``x1w_aq`` given current ``K`` and partition-basis composition
  ``x_partition_aq``.  No iteration required.
* ``rachford_rice`` — scalar SSI inner loop on β (vapor fraction
  on the partitioning-component basis).
* ``flash_kv`` — the production multi-gas / multi-salt K-value
  flash; see its docstring for the full signature.

The lnφ kernels live in ``ms_ecpa.lnphi``; both phases are
self-consistently solved there.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .parameters import Mw, R, Tc1, Tc2, Tc3, Pc1, Pc2, Pc3, w2, b1, b2, b3
from .lnphi import lnphi_aq, lnphi_gas, _gas_root_is_physical


# Wilson omegas for the three partitioning species (literature values).
# Water: 0.3443; methane: 0.01142 (already in parameters.py as `w2`);
# CO2: 0.22394.
_OMEGA = np.array([0.3443, w2, 0.22394])
_TC = np.array([Tc1, Tc2, Tc3])
_PC_BAR = np.array([Pc1, Pc2, Pc3]) / 1e5  # bar


# ---------------------------------------------------------------------------
# Wilson K
# ---------------------------------------------------------------------------

def wilson_k(T, P_bar):
    """Return the Wilson K-values of H2O, CH4 and CO2 at temperature T and pressure P.

    ``K_i = (P_c,i / P) * exp(5.373 * (1 + ω_i) * (1 − T_c,i / T))``

    These are the default ``K_init`` of ``flash_kv``.

    Parameters
    ----------
    T : float
        Temperature [K].
    P_bar : float
        Pressure [bar].

    Returns
    -------
    ndarray, shape (3,)
        K-values ``y_i / x_i`` (H2O, CH4, CO2).
    """
    return _PC_BAR / float(P_bar) * np.exp(5.373 * (1 + _OMEGA) * (1 - _TC / T))


# ---------------------------------------------------------------------------
# Aqueous mass-balance closure (multi-salt, multi-gas)
# ---------------------------------------------------------------------------

def aqueous_closure(x1w, K, y, mi):
    """Residual of the multi-salt aqueous-phase normalisation equation.

    The aqueous phase carries water (``x1w``), the two partitioning gases
    (CH4: ``x2w``, CO2: ``x3w``), and the six ions (``xiw``).  The
    normalisation constraint ``x1w + x2w + x3w + Σ xiw = 1`` re-expressed
    via current K-values and gas-phase mole fractions yields:

    .. math::

        R(x1w; K, y, m_i)
            = x_{1w}\\,(1 + M_w\\,m_{\\text{tot}})
              + y_{\\text{CH4}}/K_{\\text{CH4}}
              + y_{\\text{CO2}}/K_{\\text{CO2}}
              - 1

    Here:

    * Ion mole fraction in aqueous phase: ``x_{i,ion} = M_w · x1w · m_i``
      (per-kg-water convention), so ``Σ xiw = M_w · x1w · m_tot``.
    * Each non-water partitioning species follows isofugacity:
      ``x_k^{aq} = y_k / K_k``.

    Parameters
    ----------
    x1w : float
        Trial water mole fraction in the aqueous phase.
    K : array_like, shape (3,)
        K-values ``(K_H2O, K_CH4, K_CO2)``.
    y : array_like, shape (3,)
        Gas-phase mole fractions ``(y_H2O, y_CH4, y_CO2)``; must sum to 1.
    mi : array_like, shape (6,)
        Ion molalities (Na+, Cl-, K+, Ca2+, Mg2+, SO4^2-) [mol/kg-water].

    Returns
    -------
    R : float
        Residual; zero at the aqueous closure.

    Notes
    -----
    The closure reduces to the NaCl equation for ``x1w`` of Paper I
    (``ecpa.flash``) when ``mi = (m_NaCl, m_NaCl, 0, 0, 0, 0)`` and
    ``K_CH4`` is unused (CO2 as the only gas).  See ``tests/test_aqueous_closure.py``
    for the algebraic reduction.
    """
    K = np.asarray(K, dtype=float)
    y = np.asarray(y, dtype=float)
    mi = np.asarray(mi, dtype=float)
    m_tot = float(np.sum(mi))
    x_ion_sum = Mw * x1w * m_tot
    # Each present gas contributes y_k / K_k; skip absent gases (K_k=NaN, y_k=0)
    x_ch4_aq = 0.0 if (y[1] == 0.0 or not np.isfinite(K[1])) else y[1] / K[1]
    x_co2_aq = 0.0 if (y[2] == 0.0 or not np.isfinite(K[2])) else y[2] / K[2]
    return x1w + x_ion_sum + x_ch4_aq + x_co2_aq - 1.0


def compute_aqueous_x1w_from_KP(K, y, mi):
    """Return aqueous ``x1w`` in closed form, with ``mi`` the aqueous molality.

    Solves ``aqueous_closure`` for ``x1w`` directly (the equation is
    affine in ``x1w``), so no Newton is required:

    .. math::

        x_{1w} = \\frac{1 - y_{\\text{CH4}}/K_{\\text{CH4}}
                          - y_{\\text{CO2}}/K_{\\text{CO2}}}
                       {1 + M_w\\,m_{\\text{tot}}}

    This routine treats ``mi`` as the equilibrium aqueous-phase molality
    (the "as-aqueous" convention; salt conservation is not applied).  For
    the salt-conservation treatment of Paper I, generalised to several
    salts, see ``compute_aqueous_x1w_chain``.

    Parameters
    ----------
    K : array_like, shape (3,)
        K-values ``(K_H2O, K_CH4, K_CO2)``; NaN for absent species.
    y : array_like, shape (3,)
        Gas-phase mole fractions ``(y_H2O, y_CH4, y_CO2)``.
    mi : array_like, shape (6,)
        Ion molalities (Na+, Cl-, K+, Ca2+, Mg2+, SO4 2-) [mol/kg water].

    Returns
    -------
    x1w : float
        Aqueous H2O mole fraction (ions included in the total).
    x_ch4 : float
        Aqueous CH4 mole fraction ``y_CH4 / K_CH4``.
    x_co2 : float
        Aqueous CO2 mole fraction ``y_CO2 / K_CO2``.
    """
    K = np.asarray(K, dtype=float)
    y = np.asarray(y, dtype=float)
    mi = np.asarray(mi, dtype=float)
    m_tot = float(np.sum(mi))
    x_ch4 = 0.0 if (y[1] == 0.0 or not np.isfinite(K[1])) else y[1] / K[1]
    x_co2 = 0.0 if (y[2] == 0.0 or not np.isfinite(K[2])) else y[2] / K[2]
    x1w = (1.0 - x_ch4 - x_co2) / (1.0 + Mw * m_tot)
    return x1w, x_ch4, x_co2


def compute_aqueous_x1w_chain(x_part_w, beta, z_w, mi_feed):
    """Return aqueous x_w and ion molalities with salt conservation applied.

    This is the salt-conservation treatment of Paper I, generalised to
    several salts and two gases.

    Given the partition-basis aqueous water mole fraction ``x_part_w``
    from Rachford-Rice, the partition-basis vapour fraction ``beta``, the
    volatile-basis H2O feed mole fraction ``z_w``, and the prepared-brine
    ion molality vector ``mi_feed`` (mol per kg of pre-gas water),
    the chain prescribes:

    .. math::
        m_{i,\\text{aq}}^{*} = m_{i,\\text{feed}}
            \\frac{z_w}{(1-\\beta)\\,x_{\\text{part},w}}.

    **The denominator is the PARTITION-basis (salt-free) water mole
    fraction.**  Molality is per kilogram of water, and the moles of
    water in the aqueous phase are ``(1-beta) * x_part_w`` per mole of
    partitioning feed.  This is exactly the H2O balance
    ``z_w = (1-beta) * x_w_aq`` of Paper I, whose
    ``x_w_aq`` is likewise salt-free.  Using the ion-inclusive
    ``x1w_true`` here instead increases every molality by a factor
    ``1/(1 - Sigma_x)`` and violates salt conservation (+3.7 % at 1 molal,
    +21.8 % at 5 molal).

    With ``S = M_w z_w m_tot_feed / (1 - beta)``, the ion mole-fraction
    sum follows from ``Sigma_x = M_w x_w_aq m_tot_aq``:

    .. math::
        \\Sigma_x = \\frac{S}{1+S}, \\qquad
        x_{w,\\text{aq}} = \\frac{x_{\\text{part},w}}{1+S}.

    :math:`\\Sigma_x` is still independent of ``x_{w,aq}``, so the
    partition-to-true conversion stays affine and no Brent solve is
    needed.  Note ``0 < 1/(1+S) <= 1`` for every physical input, so the
    closure does not produce an unphysical state.

    Parameters
    ----------
    x_part_w : float
        Partition-basis aqueous H2O mole fraction from RR/H2O balance,
        ``x_part_w = z_w / (1 + beta·(K_w - 1))``.
    beta : float
        Vapour fraction on the partitioning-component basis (0..1).
    z_w : float
        Volatile-basis H2O feed mole fraction (= ``z[0]`` in the
        partition-component vector used by ``flash_kv``).
    mi_feed : array_like, shape (6,)
        Prepared-brine ion molalities (Na, Cl, K, Ca, Mg, SO4)
        [mol / kg pre-gas water].

    Returns
    -------
    x1w_true : float
        True aqueous H2O mole fraction (ions included in the denominator).
        NaN when ``beta >= 1`` or the inputs give no valid state.
    scale : float
        Conversion factor ``x_aq_i = x_part_i · scale`` for the volatile
        species (= ``1 - Sigma_x``).
    mi_aq : ndarray, shape (6,)
        Per-ion aqueous-phase molality at equilibrium [mol / kg aq water].
    sigma_x : float
        Total ion mole fraction in the aqueous phase (= ``M_w · z_w ·
        m_tot,feed / (1 - beta)``).

    Notes
    -----
    Salt-free limit (``mi_feed = 0``): ``S = 0``, ``sigma_x = 0``,
    ``scale = 1``, ``x1w_true = x_part_w``, ``mi_aq = 0`` --- recovers
    the binary flash exactly.  Single-NaCl limit
    (``mi_feed = (m_s, m_s, 0, 0, 0, 0)``, ``z_CH4 = 0``):
    ``m_i,aq* = m_s * z_w / [(1-beta) * x_part_w]`` --- matches Eq. (9)
    of Paper I per ion, with that paper's salt-free water mole fraction.

    Salt conservation is the invariant to test: the ion moles in the
    aqueous phase, ``m_i_aq * (1-beta) * x_part_w * M_w``, must equal the
    prepared ion moles ``m_i_feed * z_w * M_w`` exactly.
    """
    mi_feed = np.asarray(mi_feed, dtype=float)
    m_tot_feed = float(np.sum(mi_feed))
    one_minus_beta = 1.0 - float(beta)
    if one_minus_beta <= 0.0:
        # β → 1: no aqueous phase; sigma_x diverges. Degenerate flash.
        return float("nan"), float("nan"), np.full(6, float("nan")), float("inf")
    # S = ion moles per mole of aqueous PARTITIONING species.
    S = Mw * float(z_w) * m_tot_feed / one_minus_beta
    if not np.isfinite(S) or S < 0.0:
        return float("nan"), float("nan"), np.full(6, float("nan")), float("nan")
    # Sigma_x = S/(1+S) and scale = 1/(1+S) follow from
    # Sigma_x = M_w * x_w_aq * m_tot_aq with x_w_aq = x_part_w * scale.
    # Both lie in (0, 1] for every physical input.
    scale = 1.0 / (1.0 + S)
    sigma_x = S * scale
    x1w_true = float(x_part_w) * scale
    if x1w_true <= 0.0:
        return float("nan"), float(scale), np.full(6, float("nan")), float(sigma_x)
    # Molality is per kg of WATER, so divide by the partition-basis
    # water mole fraction, not by the ion-inclusive x1w_true.
    mi_aq = mi_feed * (float(z_w) / (one_minus_beta * float(x_part_w)))
    return x1w_true, scale, mi_aq, sigma_x


# ---------------------------------------------------------------------------
# Rachford-Rice
# ---------------------------------------------------------------------------

def rachford_rice(K, z, *, beta_lo=None, beta_hi=None):
    """Solve the Rachford-Rice equation for the vapour fraction β.

    .. math::

        F(\\beta) = \\sum_i \\frac{z_i (K_i - 1)}{1 + \\beta (K_i - 1)} = 0

    Only present species enter the sum; species with ``z_i == 0`` or a
    non-finite ``K_i`` are dropped.  The root is found with Brent's method
    on a bracket where ``F`` changes sign.

    Parameters
    ----------
    K : array_like, shape (3,)
        K-values ``y_i / x_i`` on the partitioning basis (H2O, CH4, CO2).
    z : array_like, shape (3,)
        Feed mole fractions of the partitioning species (H2O, CH4, CO2).
    beta_lo : float, optional
        Lower end of the bracket.  Defaults to the pole bound
        ``1 / (1 - max K) + 1e-12``, and is never below 0.
    beta_hi : float, optional
        Upper end of the bracket.  Defaults to the pole bound
        ``1 / (1 - min K) - 1e-12``, and is never above 1.

    Returns
    -------
    float
        Vapour mole fraction β on the partitioning basis.  If no root lies
        in (0, 1), the nearer bound is returned: 0 when all ``K_i <= 1``
        (or no species is present), 1 when all ``K_i >= 1``, and otherwise
        the end of the bracket where ``|F|`` is smaller.
    """
    K = np.asarray(K, dtype=float)
    z = np.asarray(z, dtype=float)
    present = (z > 0) & np.isfinite(K) & np.isfinite(z)
    K_p = K[present]
    z_p = z[present]
    if K_p.size == 0:
        return 0.0
    if np.all(K_p <= 1.0):
        return 0.0
    if np.all(K_p >= 1.0):
        return 1.0

    def F(beta):
        return float(np.sum(z_p * (K_p - 1.0) / (1.0 + beta * (K_p - 1.0))))

    eps = 1e-12
    if beta_lo is None:
        beta_lo = max(1.0 / (1.0 - float(np.max(K_p))) + eps, 0.0)
    if beta_hi is None:
        beta_hi = min(1.0 / (1.0 - float(np.min(K_p))) - eps, 1.0)
    beta_lo = max(beta_lo, 0.0)
    beta_hi = min(beta_hi, 1.0)

    f_lo, f_hi = F(beta_lo), F(beta_hi)
    if f_lo * f_hi > 0:
        # No sign change — pick the closer boundary
        return 0.0 if abs(f_lo) < abs(f_hi) else 1.0
    return float(brentq(F, beta_lo, beta_hi, xtol=1e-12, rtol=1e-12, maxiter=200))


# ---------------------------------------------------------------------------
# K-value flash result
# ---------------------------------------------------------------------------

@dataclass
class FlashKVResult:
    """Result of a K-value flash (returned by ``flash_kv`` and ``dew_point_flash``).

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
        water]: the feed molality under ``"chain"``, the aqueous molality
        under ``"as-aqueous"``.
    converged : bool
        True when the fugacity residual is below the tolerance.
    message : str
        "converged", or the reason the flash stopped.
    K : tuple of float, length 3
        K-values (H2O, CH4, CO2); NaN for absent species.  From
        ``flash_kv`` they are on the partitioning basis, ``y_i / x_part,i``,
        where ``x_part`` is the aqueous composition without the ions
        (renormalised to sum to 1).  From ``dew_point_flash`` they are
        ``y_i / x_aq,i``.  The two agree in the salt-free case.
    x_aq : tuple of float, length 3
        Aqueous mole fractions (H2O, CH4, CO2) with the ions included in the
        total, so they sum to ``1 - sum(x_ion)``.
    y_gas : tuple of float, length 3
        Gas-phase mole fractions (H2O, CH4, CO2); sum to 1.
    beta : float
        Vapour mole fraction on the partitioning basis (ions excluded).
    x1w : float
        Aqueous H2O mole fraction, equal to ``x_aq[0]``.
    Z_aq : float
        Compressibility factor ``PV/(nRT)`` of the aqueous phase; NaN if not
        available.
    Z_gas : float
        Compressibility factor of the gas phase; NaN if not available.
    eps_r : float
        Relative permittivity of the aqueous phase; NaN if not available.
    n_iter_ssi : int
        Number of successive-substitution iterations (0 for
        ``dew_point_flash``).
    n_iter_newton : int
        Number of Newton iterations on ln K after the substitution phase
        (``dew_point_flash`` reports 1).
    max_residual : float
        Final residual.  From ``flash_kv``: ``max|g|`` over present species,
        with ``g_i`` the change of ``ln K_i`` from one substitution step.
        From ``dew_point_flash``: the largest relative fugacity difference
        ``|f_aq - f_gas| / max(f_aq, f_gas)``.
    fugacities_aq : tuple of float, length 3
        Fugacities (H2O, CH4, CO2) in the aqueous phase [bar].
    fugacities_gas : tuple of float, length 3
        Fugacities (H2O, CH4, CO2) in the gas phase [bar].
    mi_aq : tuple of float, length 6
        Equilibrium aqueous ion molalities [mol/kg water].  Equal to ``mi``
        under ``"as-aqueous"``; computed from salt conservation under
        ``"chain"``.
    salinity_convention : str
        ``"as-aqueous"`` or ``"chain"`` (see ``flash_kv``).
    chi_H2O_aq : float
        Fraction of H2O association sites not bonded (Wertheim monomer
        fraction) in the aqueous phase; NaN if not available.
    chi_H2O_gas : float
        Same quantity in the gas phase; NaN if not available.
    n_iter_aq_inner : int
        Newton iterations of the final aqueous-phase solve at the returned
        state (used to compute ``Z_aq``, ``eps_r`` and ``chi_H2O_aq``).
    n_iter_gas_inner : int
        Newton iterations of the final gas-phase solve at the returned
        state.
    """
    T: float
    P: float
    z: tuple
    mi: tuple                # input mi (= m_feed under "chain", or m_aq under "as-aqueous")
    converged: bool
    message: str
    K: tuple                 # (K_H2O, K_CH4, K_CO2); NaN for absent species
    x_aq: tuple              # (x1w, x2w, x3w) aqueous mole fractions
    y_gas: tuple             # (y1, y2, y3) gas-phase mole fractions
    beta: float              # vapor fraction (partitioning-component basis)
    x1w: float
    Z_aq: float
    Z_gas: float
    eps_r: float
    n_iter_ssi: int
    n_iter_newton: int
    max_residual: float
    fugacities_aq: tuple     # (f_h2o, f_ch4, f_co2) [bar], aqueous
    fugacities_gas: tuple    # (f_h2o, f_ch4, f_co2) [bar], gas
    mi_aq: tuple = (0.0,)*6  # equilibrium aqueous-phase molality (chain output)
    salinity_convention: str = "as-aqueous"
    # Inner Newton state at convergence -- everything needed to warm-start
    # a future call from this point.
    chi_H2O_aq: float = float("nan")   # aqueous H2O association fraction
    chi_H2O_gas: float = float("nan")  # gas-phase H2O association fraction
    n_iter_aq_inner: int = 0           # cumulative inner Newton iter count, aqueous
    n_iter_gas_inner: int = 0          # cumulative inner Newton iter count, gas


# ---------------------------------------------------------------------------
# K-value flash
# ---------------------------------------------------------------------------

def flash_kv(
    T, P, z, mi,
    *,
    K_init=None,
    tol=1e-8,
    maxiter=80,
    accelerated=True,
    use_newton=True,
    newton_tol=1e-4,
    max_newton=5,
    salinity_convention="as-aqueous",
    gas_Z_init=None,
    verbose=False,
):
    """Run the two-phase K-value flash of the multi-salt eCPA at given T, P and feed.

    The flash is a successive substitution on ln K of the partitioning
    species, with the acceleration of Jex et al. (2024) and an optional
    Newton step on ln K once the residual is small.  Ions stay in the
    aqueous phase.  The flash does not test stability; use
    ``ms_ecpa.stability.stability_flash`` for that.

    Parameters
    ----------
    T : float
        Temperature [K].
    P : float
        Pressure [bar].
    z : array_like, shape (3,)
        Feed mole fractions of the partitioning species (H2O, CH4, CO2),
        ions excluded; sum to 1.  Set ``z[1] = 0`` for the CO2-water
        system and ``z[2] = 0`` for the CH4-water system.
    mi : array_like, shape (6,)
        Ion molalities (Na+, Cl-, K+, Ca2+, Mg2+, SO4 2-) [mol/kg water].
        See ``salinity_convention`` for how they are read.
    K_init : array_like, optional
        Initial K-vector for ``(H2O, CH4, CO2)``.  Defaults to Wilson.
        Components corresponding to absent species (``z_i = 0``) are
        ignored.
    tol : float
        Convergence tolerance on ``max |g|``, where
        ``g_i = ln s + lnφ_i^aq − lnφ_i^gas − ln K_i`` and ``s`` is the
        ion-closure factor ``1 - sum(x_ion)``.
    maxiter : int
        Maximum number of substitution iterations.
    accelerated : bool
        Apply the acceleration of Jex et al. (2024) to the ln K update.
    use_newton : bool
        Switch to Newton iterations on ln K once ``max|g| < newton_tol``.
    newton_tol : float
        Residual below which the Newton iterations start.
    max_newton : int
        Maximum number of Newton iterations.
    salinity_convention : {"as-aqueous", "chain"}
        Interpretation of the input ``mi`` and the aqueous closure.
        ``"as-aqueous"`` (default) treats ``mi`` as the equilibrium
        aqueous-phase molality directly (no salt-conservation
        amplification). This matches how literature CO2-brine and
        CH4-brine experimental papers back-calculate solubility (using
        the prepared-brine water fraction throughout), so it is the
        like-for-like convention for validation against published data
        sets. ``"chain"`` is the salt-conservation treatment
        of Paper I, generalised here per ion to several salts and two
        gases: it treats ``mi`` as the prepared-brine FEED
        molality and applies ``m_aq* = m_feed · z_w / [(1-β) · x_w_aq]``
        at every SSI step. ``"chain"`` requires the user to commit to
        a gas-to-brine cell-volume convention (encoded implicitly by
        the chosen feed ``z``) and should be used when the application
        --- e.g. a reservoir-simulation grid cell with a known
        composition --- supplies a physically meaningful ``z``.
    gas_Z_init : float, optional
        Initial compressibility-factor seed for the FIRST gas-side inner
        Newton (subsequent iterations warm-start from the previous
        converged state).  The default ``None`` uses the vapour-root seed
        ``Z = 1.0``.  Pass a dense estimate (``Z ~ 2 B_mix``) to steer the non-aqueous
        phase onto the liquid-CO2 root below the CO2 critical
        temperature — used by the three-phase driver's metastable-vapor
        re-flash (``ms_ecpa.flash3.stability_flash3``).
    verbose : bool
        Print one line per iteration.

    Returns
    -------
    FlashKVResult
        Phase compositions, vapour fraction, K-values, fugacities and
        convergence information.  Check ``converged``: a failed flash is
        returned with ``converged=False`` and the reason in ``message``,
        not raised.

    Raises
    ------
    ValueError
        If ``z`` does not have 3 entries summing to 1, or ``mi`` does not
        have 6 entries.
    """
    z = np.asarray(z, dtype=float)
    mi = np.asarray(mi, dtype=float)
    if z.shape != (3,):
        raise ValueError("z must have 3 entries: (z_H2O, z_CH4, z_CO2)")
    if mi.shape != (6,):
        raise ValueError("mi must have 6 entries: (Na, Cl, K, Ca, Mg, SO4)")
    if abs(np.sum(z) - 1.0) > 1e-8:
        raise ValueError(f"z must sum to 1; got Σz = {float(np.sum(z)):.6f}")

    present = z > 0  # boolean mask of partitioning species actually present
    P_Pa = P * 1e5

    # ── Initial K ─────────────────────────────────────────────────────────
    if K_init is None:
        K = wilson_k(T, P)
    else:
        K = np.asarray(K_init, dtype=float).copy()
    # Absent species: mark K = NaN so RR / closure skip them.
    K = np.where(present, K, np.nan)

    # ── Initial state for the SSI loop ────────────────────────────────────
    lnK_present = np.log(np.where(present, K, 1.0))  # only present entries matter
    g_prev = None
    m_jex = 1.0
    n_iter_ssi = 0
    converged = False
    max_resid = np.inf
    x_aq = (np.nan, np.nan, np.nan)
    y_gas = (np.nan, np.nan, np.nan)
    beta = np.nan
    x1w = np.nan

    # Warm-start hidden states for the lnphi inner Newtons; updated each
    # successful evaluation so the next iteration starts near the previous
    # converged (Zw, epsr, chi1w) / (Zh, chi1h).  None ⇒ kernel cold start.
    v_aq_warm = [None]
    v_gas_warm = [None]
    if gas_Z_init is not None:
        # Steer the first gas-side inner Newton onto a chosen density
        # root (e.g. liquid CO2); warm-start chaining takes over after
        # the first successful step.
        v_gas_warm[0] = np.array([float(gas_Z_init), 0.95])

    def _ssi_step(K_in):
        """Single SSI evaluation: K_in → (x_aq, y_gas, K_new, g, ...)."""
        # 1) Rachford-Rice on present species
        beta_loc = rachford_rice(K_in, z)
        # 2) Partition-basis compositions (only present species are meaningful)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = 1.0 + beta_loc * (K_in - 1.0)
            x_part = np.where(present, z / denom, 0.0)   # partition aqueous mol fracs
            y_part = np.where(present, K_in * x_part, 0.0)
        # Renormalise (RR ensures sum-to-one for present species, but
        # numerical drift is possible)
        sx = float(np.nansum(x_part))
        sy = float(np.nansum(y_part))
        if sx > 0:
            x_part = x_part / sx
        if sy > 0:
            y_part = y_part / sy

        # 3) Multi-salt aqueous closure: rescale x_part to true aqueous
        # mole fractions by accounting for the ion fraction.
        if salinity_convention == "chain":
            # Rigorous treatment (Moortgat2025iecr, generalised to multi-salt):
            # mi is the prepared-brine FEED molality; the equilibrium aqueous
            # molality m_aq* is amplified by salt conservation as H2O
            # partitions into the gas phase. The aqueous ion mole-fraction
            # sum sigma_x = S/(1+S), S = M_w · z_w · m_tot_feed / (1-beta),
            # is independent of x1w_true, so the partition-basis to
            # true-basis conversion stays affine: x_aq_i = x_part_i · scale
            # with scale = 1/(1+S).
            x1w_true, scale, mi_aq_step, _sigma = compute_aqueous_x1w_chain(
                x_part[0], beta_loc, float(z[0]), mi
            )
            if not np.isfinite(x1w_true) or x1w_true <= 0.0 or scale <= 0.0:
                raise RuntimeError(
                    f"chain closure invalid (β={beta_loc:.4f}, "
                    f"z_w={float(z[0]):.4f}, m_tot_feed="
                    f"{float(np.sum(mi)):.4f}); aqueous phase cannot hold the "
                    "salt mass implied by the feed/phase-split combination."
                )
        else:
            # Legacy "as-aqueous" convention: mi treated as the equilibrium
            # aqueous molality directly (no salt-conservation amplification).
            # x1w_true = x_part[0] / (1 + M_w·m_tot·x_part[0])  (derived from
            # x_part[0] = x1w_true / (1 − Σx_ion) and Σx_ion = M_w·m_tot·x1w_true)
            m_tot = float(np.sum(mi))
            x1w_true = float(x_part[0] / (1.0 + Mw * m_tot * x_part[0])) if x_part[0] > 0 else 0.0
            scale = 1.0 - Mw * m_tot * x1w_true
            mi_aq_step = np.asarray(mi, dtype=float)

        x_aq_loc = (x1w_true,
                    float(x_part[1] * scale) if present[1] else 0.0,
                    float(x_part[2] * scale) if present[2] else 0.0)
        y_gas_loc = (float(y_part[0]),
                     float(y_part[1]) if present[1] else 0.0,
                     float(y_part[2]) if present[2] else 0.0)

        # Guard: fire only when x_H2O is genuinely negligible (x1w < 0.02).
        # The original threshold was 0.10, which fired on transient SSI states
        # for water-poor binary feeds (z_H2O=0.05–0.10) where Jex transiently
        # drives K_H2O→1, giving x1w≈z_H2O≈0.05–0.10.  At TRUE equilibrium
        # these feeds have x_H2O_aq≈0.25–0.98, so the guard was mis-firing.
        # Lowering to 0.02 reserves the K-reset for the genuinely unphysical
        # x_H2O≈0 cases while letting water-poor feeds converge naturally.
        # The dissolved_gas arm was removed in Iteration 4 for the same reason.
        dissolved_gas = x_aq_loc[1] + x_aq_loc[2]
        if present[0] and x1w_true < 0.02:
            raise RuntimeError(
                f"degenerate-aqueous: x1w={x1w_true:.4f} dissolved={dissolved_gas:.4f} "
                f"(K_H2O={K_in[0]:.4f}); K-reset required"
            )

        # 4) lnφ evaluations (warm-started from last successful step)
        from .lnphi import (_aq_residual_3var, _gas_residual,
                            _aq_lnphi_evaluate_3var, _gas_lnphi_evaluate,
                            _gas_root_is_physical)
        from .newton import newton_solve as _ns

        # Aqueous: warm Newton on the 3-var residual, then evaluate lnφ.
        # mi_aq_step is the salt-conservation-chain-amplified equilibrium
        # aqueous molality vector under "chain" convention, or simply the
        # input mi under "as-aqueous" (legacy) convention.
        v0_aq = v_aq_warm[0]
        if v0_aq is None:
            # Cold start on the SAME physics-scaled seed lnphi_aq uses by
            # default: Zw ~ P*V_m/(R*T) with V_m = 18.015 cm^3/mol. The fixed
            # Zw = 0.05 that used to sit here is the seed lnphi_aq's own
            # docstring records as landing outside the liquid basin at low
            # pressure, and the root it finds there is taken as authoritative
            # below, so the two entry points have to agree on where to look.
            v0_aq = np.array([float(P_Pa) * 18.015e-6 / (8.314 * float(T)),
                              50.0, 0.10])
        v_aq, ok_aq, _ = _ns(_aq_residual_3var, v0_aq,
                             args=(np.asarray(x_aq_loc), mi_aq_step, T, P_Pa),
                             tol=1e-10, maxiter=80)
        # Same acceptance test lnphi_aq applies: a non-positive monomer
        # fraction or a negative relative permittivity is not a state.
        if ok_aq and (v_aq[2] <= 0 or v_aq[1] <= 0):
            ok_aq = False
        warm_ok = True
        if not ok_aq:
            # lnphi_aq has its own, longer seed ladder. Run it: this `la` is
            # the authoritative aqueous lnphi and, if we get here, it has
            # already succeeded.
            la = lnphi_aq(x_aq_loc, mi_aq_step, T, P_Pa)
            # Everything below only re-derives v_aq for the WARM START of the
            # next SSI step. It is an optimisation, not a correctness
            # requirement, and its ladder is a hand-copy of lnphi_aq's that
            # has drifted from the original.
            _Zw_liq = float(P_Pa) * 18.015e-6 / (8.314 * float(T))
            for alt in (np.array([_Zw_liq, 60.0, 0.07]),
                        np.array([_Zw_liq, 50.0, 0.80]),
                        np.array([_Zw_liq, 50.0, 0.90]),
                        np.array([0.07,   60.0, 0.13]),
                        np.array([0.05,   50.0, 0.10]),
                        np.array([0.10,   30.0, 0.05]),
                        np.array([0.30,   70.0, 0.20]),
                        np.array([0.20,   40.0, 0.10]),
                        np.array([0.50,   50.0, 0.10]),
                        np.array([0.50,   80.0, 0.50]),
                        np.array([_Zw_liq, 78.0, 0.85]),
                        np.array([_Zw_liq, 78.0, 0.95]),
                        # Iter 6 Fix 1: high-P eps_r seeds
                        # (literature eps_r reaches ~100-130 at
                        # P=1341-2000 bar).
                        np.array([_Zw_liq,  90.0, 0.07]),
                        np.array([_Zw_liq, 100.0, 0.07]),
                        np.array([_Zw_liq, 110.0, 0.07]),
                        np.array([_Zw_liq,  90.0, 0.85]),
                        np.array([_Zw_liq, 100.0, 0.90]),
                        # Iter 6 Fix 3: chi1w≈0.97 seeds for 3 m brines.
                        np.array([_Zw_liq, 80.0, 0.97]),
                        np.array([_Zw_liq, 85.0, 0.95]),
                        # Iter 7 Fix B: water-poor aqueous seeds
                        # (aqueous phase carrying >50% dissolved gas,
                        # weak association from H2O sparsity).
                        np.array([_Zw_liq, 30.0, 0.10]),
                        np.array([_Zw_liq, 40.0, 0.20]),
                        np.array([_Zw_liq, 50.0, 0.30]),
                        # Iter 8 Fix E: targeted seeds from v7 failure
                        # x_aq vectors -- water-rich+3m, water-rich+
                        # mid-salt, and very water-poor (eps_r dragged
                        # toward gas value).
                        np.array([_Zw_liq, 75.0, 0.98]),
                        np.array([_Zw_liq, 70.0, 0.99]),
                        np.array([_Zw_liq, 70.0, 0.93]),
                        np.array([_Zw_liq, 65.0, 0.95]),
                        np.array([_Zw_liq,  5.0, 0.05]),
                        np.array([_Zw_liq, 10.0, 0.05]),
                        np.array([_Zw_liq, 15.0, 0.10])):
                v_aq, ok_aq, _ = _ns(_aq_residual_3var, alt,
                                     args=(np.asarray(x_aq_loc), mi_aq_step, T, P_Pa),
                                     tol=1e-10, maxiter=80)
                if ok_aq and v_aq[2] <= 0:
                    ok_aq = False  # reject spurious chi1w<0 fixed point
                if ok_aq:
                    break
            if ok_aq:
                # The stored root is a warm start, but the NEXT SSI step
                # evaluates lnφ from it directly (the `else` branch above),
                # so it has to be the root `la` came from. The aqueous
                # residual admits a spurious non-liquid root at low T/P that
                # passes the chi1w>0 test -- lnphi_aq's own notes record it
                # returning f_water ~4 bar instead of ~0.23 bar -- and this
                # ladder's rung order has drifted from lnphi_aq's, so the two
                # can land on different branches. Where they agree nothing
                # changes; where they do not, drop the warm state so the next
                # step cold-starts on the same physics-scaled seed lnphi_aq
                # uses, rather than silently switching branch mid-flash.
                la_chk = _aq_lnphi_evaluate_3var(v_aq, np.asarray(x_aq_loc),
                                                 mi_aq_step, T, P_Pa)
                la_chk = tuple(float(np.real(c)) for c in la_chk)
                if not all(abs(la_chk[i] - float(la[i])) < 1e-8 for i in range(3)):
                    ok_aq = False
            if not ok_aq:
                # Do NOT raise. `la` above is valid -- lnphi_aq converged --
                # and raising here aborted a flash that had actually solved,
                # purely because this shorter duplicate ladder missed the
                # warm-start root. Discard the warm state instead and let the
                # next step cold-start, exactly as the degenerate-step branch
                # below already does (`v_aq_warm[0] = None`). v_aq currently
                # holds the last failed attempt, so it must not be stored.
                warm_ok = False
        else:
            la = _aq_lnphi_evaluate_3var(v_aq, np.asarray(x_aq_loc), mi_aq_step, T, P_Pa)
            la = (float(np.real(la[0])), float(np.real(la[1])), float(np.real(la[2])))
        v_aq_warm[0] = v_aq if warm_ok else None

        # Gas: warm Newton on the 2-var residual, then evaluate lnφ.
        v0_g = v_gas_warm[0]
        if v0_g is None:
            v0_g = np.array([1.0, 0.99])
        v_g, ok_g, _ = _ns(_gas_residual, v0_g,
                           args=(np.asarray(y_gas_loc), T, P_Pa),
                           tol=1e-10, maxiter=80)
        if ok_g and not _gas_root_is_physical(v_g, np.asarray(y_gas_loc),
                                              T, P_Pa):
            ok_g = False  # reject spurious chi<=0 gas fixed point
        if not ok_g:
            lg = lnphi_gas(y_gas_loc, T, P_Pa)
            for alt in (np.array([1.0, 0.99]),
                        np.array([0.7, 0.95]),
                        np.array([0.3, 0.90])):
                v_g, ok_g, _ = _ns(_gas_residual, alt,
                                   args=(np.asarray(y_gas_loc), T, P_Pa),
                                   tol=1e-10, maxiter=80)
                if ok_g and not _gas_root_is_physical(
                        v_g, np.asarray(y_gas_loc), T, P_Pa):
                    ok_g = False  # reject spurious chi<=0 gas fixed point
                if ok_g:
                    break
            if ok_g:
                # Same warm-start/authoritative-root agreement test as on the
                # aqueous side: this 3-rung ladder is not lnphi_gas's, the gas
                # cubic carries up to three physical roots, and
                # _gas_root_is_physical screens chi, not the density branch.
                # Storing a root `lg` did not come from would have the next
                # step evaluate lnφ on a different branch than this one did.
                lg_chk = _gas_lnphi_evaluate(v_g, np.asarray(y_gas_loc), T, P_Pa)
                lg_chk = tuple(float(np.real(c)) for c in lg_chk)
                if not all(abs(lg_chk[i] - float(lg[i])) < 1e-8 for i in range(3)):
                    # Clear rather than merely skip the store: leaving the
                    # previous step's root in place would seed the next
                    # Newton back onto the branch this test just rejected.
                    ok_g = False
                    v_gas_warm[0] = None
        else:
            lg = _gas_lnphi_evaluate(v_g, np.asarray(y_gas_loc), T, P_Pa)
            lg = (float(np.real(lg[0])), float(np.real(lg[1])), float(np.real(lg[2])))
        if ok_g:
            v_gas_warm[0] = v_g

        # 5) K_new on the partition basis: f_i^aq = f_i^gas
        # lnK_i = lnφ_i^aq − lnφ_i^gas.  But this gives K relating *aqueous*
        # mole fraction to *gas* mole fraction; we want K on the partition
        # basis (Rachford-Rice convention), where x_part = x_aq / scale.
        # In log form: ln(y/x_part) = ln(y) − ln(x_part) = ln(y) − ln(x_aq) + ln(scale)
        # = lnK_aq + ln(scale).
        lnK_new = np.full(3, np.nan)
        for i in range(3):
            if not present[i]:
                continue
            # f_i^aq = exp(la[i]) · x_aq_i · P,  f_i^gas = exp(lg[i]) · y_i · P
            # Equality => exp(la_i − lg_i) = y_i / x_aq_i = (y_i / x_part_i) / scale
            # => K_part_i = y_i / x_part_i = scale · exp(la_i − lg_i)
            lnK_new[i] = np.log(scale) + (la[i] - lg[i])
        g_loc = lnK_new - np.log(K_in)
        # absent species: g = 0
        for i in range(3):
            if not present[i]:
                g_loc[i] = 0.0
        return dict(K_new=np.exp(lnK_new), x_aq=x_aq_loc, y_gas=y_gas_loc,
                    beta=beta_loc, x1w=x1w_true, g=g_loc, la=la, lg=lg)

    # ── SSI loop ──────────────────────────────────────────────────────────
    _n_degenerate_resets = 0   # guard against infinite reset loops
    for it in range(maxiter):
        try:
            ev = _ssi_step(K)
        except RuntimeError as exc:
            exc_msg = str(exc)
            # Degenerate-aqueous: K-values have driven the aqueous composition
            # outside the eCPA model's valid domain (dissolved gas > water).
            # Reset K to Wilson (gas-biased) and clear the Jex accumulator so
            # the first post-reset step is plain substitution.
            if exc_msg.startswith("degenerate-aqueous:") and _n_degenerate_resets < 5:
                _n_degenerate_resets += 1
                K_wil_now = wilson_k(T, P)
                if present[0]:
                    if np.isfinite(K[0]) and K[0] > 0:
                        K[0] = min(K[0], K_wil_now[0])
                    else:
                        # K_H2O is NaN/inf/negative — chi1w→NaN propagated.
                        # min(NaN, Wilson)=NaN, so we must assign directly.
                        K[0] = float(K_wil_now[0])
                for i in (1, 2):
                    if present[i]:
                        # Use the max of Wilson K and a floor of 5 so that
                        # RR gives β > 0 even when Wilson K is sub-unity.
                        K[i] = max(float(K_wil_now[i]), 5.0)
                v_aq_warm[0] = None   # discard warm state from degenerate step
                g_prev = None         # reset Jex accumulator; will be off for one step
                m_jex = 1.0
                continue
            return FlashKVResult(
                T=T, P=P, z=tuple(z), mi=tuple(mi),
                converged=False, message=f"SSI failed at it={it}: {exc_msg}",
                K=tuple(K), x_aq=x_aq, y_gas=y_gas,
                beta=beta, x1w=x1w,
                Z_aq=np.nan, Z_gas=np.nan, eps_r=np.nan,
                n_iter_ssi=it, n_iter_newton=0, max_residual=np.inf,
                fugacities_aq=(np.nan,)*3, fugacities_gas=(np.nan,)*3,
                mi_aq=tuple(float(m) for m in np.asarray(mi, dtype=float)),
                salinity_convention=str(salinity_convention),
                chi_H2O_aq=float("nan"), chi_H2O_gas=float("nan"),
                n_iter_aq_inner=0, n_iter_gas_inner=0,
            )

        x_aq = ev["x_aq"]; y_gas = ev["y_gas"]; beta = ev["beta"]; x1w = ev["x1w"]
        g = ev["g"]
        K_new = ev["K_new"]
        # Restrict residual to present species
        g_present = g[present]
        max_resid = float(np.max(np.abs(g_present)))
        if verbose:
            print(f"[KV] it={it+1:2d}  beta={beta:.4f}  x1w={x1w:.4f}  "
                  f"max|g|={max_resid:.3e}")

        if max_resid < tol:
            K = K_new
            n_iter_ssi = it + 1
            converged = True
            break
        if use_newton and max_resid < newton_tol:
            n_iter_ssi = it + 1
            break  # → Newton polish

        # Jex 2024 acceleration on lnK.  g_prev is None on the first step
        # and immediately after a degenerate-aqueous K-reset, so Jex stays
        # off for exactly one step following a reset without any special counter.
        lnK_now = np.log(np.where(present, K, 1.0))
        if accelerated and g_prev is not None:
            num_a = float(np.dot(g_prev, g_prev))
            denom_a = float(np.dot(g_prev, g_prev - g_present))
            if abs(denom_a) > 1e-30:
                m_jex = abs(num_a / denom_a * m_jex)
                m_jex = float(np.clip(m_jex, 1.0, 10.0))
            else:
                m_jex = 1.0
        g_prev = g_present.copy()

        # Update lnK only for present species
        lnK_new_vec = lnK_now.copy()
        idx_present = np.where(present)[0]
        lnK_new_vec[idx_present] = lnK_now[idx_present] + m_jex * g_present
        # Clip large excursions
        lnK_new_vec = np.clip(lnK_new_vec, -30.0, 30.0)

        # ── Iter 7 Fix A: K_H2O upper clamp ──────────────────────────────
        # If the Jex step would push K_H2O so high that the implied
        # partition-basis aqueous water fraction (from RR with the
        # current beta) falls below a small floor, cap K_H2O.  This
        # prevents the "x_H2O,aq drift to zero" cycle that produces
        # K_H2O=NaN under chi1w propagation and exhausts the
        # degenerate-aqueous K-resets.  Mathematically, requiring
        # x_part_w = z[0] / (1 + beta·(K_H2O - 1)) >= X1W_FLOOR
        # gives K_H2O_max = (z[0] / X1W_FLOOR - 1) / beta + 1.  Only
        # active when H2O is present, beta is bounded away from 0/1,
        # and the implied bound is meaningful (K_H2O_max > 1).
        if present[0] and 1e-4 < beta < 1.0 - 1e-4:
            # Iter 8 Fix C: tighten X1W_FLOOR from 0.05 -> 0.10.  v7
            # showed the original clamp leaks (failures at it=29 with
            # x1w=0.0145, well below the 0.05 floor) because the next
            # RR's beta drifts above the beta used to compute
            # K_H2O_max.  Doubling the floor restores a safety margin
            # without changing the algorithm structure.
            _X1W_FLOOR = 0.10
            z_H2O = float(z[0])
            if z_H2O > _X1W_FLOOR:
                K_H2O_max = (z_H2O / _X1W_FLOOR - 1.0) / beta + 1.0
                if K_H2O_max > 1.0:
                    lnK_new_vec[0] = min(lnK_new_vec[0],
                                         float(np.log(K_H2O_max)))

        K = np.where(present, np.exp(lnK_new_vec), np.nan)
        n_iter_ssi = it + 1

    # ── Optional Newton polish ────────────────────────────────────────────
    n_iter_newton = 0
    if use_newton and not converged and np.isfinite(max_resid) and max_resid < newton_tol:
        idx_present = np.where(present)[0]
        n_p = idx_present.size
        for _ in range(max_newton):
            lnK_now = np.log(K[idx_present])
            # Forward-difference Jacobian on lnK
            h = 1e-5
            J = np.zeros((n_p, n_p))
            base_ev = _ssi_step(K)
            base_g = base_ev["g"][idx_present]
            for j in range(n_p):
                K_pert = K.copy()
                K_pert[idx_present[j]] = K[idx_present[j]] * np.exp(h)
                try:
                    ev_p = _ssi_step(K_pert)
                except RuntimeError:
                    J = None
                    break
                J[:, j] = (ev_p["g"][idx_present] - base_g) / h
            if J is None:
                break
            try:
                dlnK = np.linalg.solve(J, -base_g)
            except np.linalg.LinAlgError:
                break
            dlnK = np.clip(dlnK, -5.0, 5.0)
            K_new_full = K.copy()
            K_new_full[idx_present] = K[idx_present] * np.exp(dlnK)
            try:
                ev = _ssi_step(K_new_full)
            except RuntimeError:
                break
            new_resid = float(np.max(np.abs(ev["g"][idx_present])))
            if not np.isfinite(new_resid) or new_resid > max_resid:
                # Revert on divergence AND on a non-finite residual: `nan >
                # max_resid` is False, so without the isfinite test a step
                # whose g carries a NaN reads as an improvement, overwrites
                # the sub-newton_tol SSI state with it, and leaves max_resid
                # NaN -- which then disables this same test for every later
                # polish iteration.  Mirrors the entry condition above.
                break  # diverging or non-finite; revert
            K = K_new_full
            x_aq = ev["x_aq"]; y_gas = ev["y_gas"]; beta = ev["beta"]; x1w = ev["x1w"]
            max_resid = new_resid
            n_iter_newton += 1
            if max_resid < tol:
                converged = True
                break

    # ── Equilibrium aqueous-phase molality (chain output) ─────────────────
    # Under "chain", recompute m_aq* from the converged (β, x1w) for
    # diagnostic reporting and to use in the post-convergence lnφ_aq
    # evaluations below. Under "as-aqueous", mi_aq == input mi.
    if salinity_convention == "chain" and np.isfinite(beta) and 0.0 < (1.0 - beta) and x1w > 0:
        # x1w is the ion-INCLUSIVE aqueous water fraction; molality needs
        # the partition-basis one, x_part_w = x1w * (1 + S).  See
        # compute_aqueous_x1w_chain.
        _S = Mw * float(z[0]) * float(np.sum(mi)) / (1.0 - float(beta))
        mi_aq_final = np.asarray(mi, dtype=float) * (
            float(z[0]) / ((1.0 - float(beta)) * float(x1w) * (1.0 + _S))
        )
    else:
        mi_aq_final = np.asarray(mi, dtype=float)

    # ── Diagnostics from the converged state ──────────────────────────────
    # Re-run lnphi to recover Z_aq, eps_r, chi_H2O_aq, Z_gas, chi_H2O_gas.
    # Warm-start from the SSI loop's last-converged inner state when it
    # exists -- cold seeds (0.05, 50, 0.10) often fail to converge from
    # scratch at states the SSI itself reached via warm-start carry-over.
    Z_aq = Z_gas = eps_r = np.nan
    chi_H2O_aq = chi_H2O_gas = float("nan")
    n_aq_inner_diag = n_gas_inner_diag = 0
    fug_aq = fug_gas = (np.nan,)*3
    try:
        # Re-call kernels and harvest internal state by calling the
        # 3-variable Newton and gas Newton directly.
        from .lnphi import (_aq_residual_3var, _gas_residual,
                            _aq_lnphi_evaluate_3var, _gas_lnphi_evaluate)
        from .newton import newton_solve
        v_aq_init = (np.asarray(v_aq_warm[0], dtype=float).copy()
                     if v_aq_warm[0] is not None
                     else np.array([float(P_Pa) * 18.015e-6 / (8.314 * float(T)),
                                    50.0, 0.10]))
        v_aq, ok_aq, n_aq_inner_diag = newton_solve(
            _aq_residual_3var, v_aq_init,
            args=(np.asarray(x_aq), mi_aq_final, T, P_Pa),
            tol=1e-10, maxiter=80)
        # Same acceptance the equilibrium path uses. These are diagnostics,
        # but they are reported alongside the fugacities as though they
        # described the same state, so they must come from a root the rest
        # of the package would accept -- not merely from one the Newton
        # stopped at. Left as NaN otherwise, which is readable as "not
        # available" in a way a wrong number is not.
        if ok_aq and v_aq[2] > 0 and v_aq[1] > 0:
            Z_aq = float(v_aq[0])
            eps_r = float(v_aq[1])
            chi_H2O_aq = float(v_aq[2])
        def _accept_v_gas(v, ok):
            # Delegate to the one predicate the rest of the package uses.
            #
            # This used to carry its own rule -- Z > 0 and chi in [0, 1+eps]
            # -- which had drifted from the SSI step, lnphi_gas and
            # flash3._solve_gas_kernel, all of which accept roots via
            # lnphi._gas_root_is_physical.  The upper bound chi <= 1 is a
            # theorem only when every association strength is non-negative,
            # and this model lets the H2O-CO2 solvation strength S13(T) go
            # negative below about 290 K.  So the two rules disagreed in
            # both directions, and at low T the flash would converge with
            # correct fugacities while these diagnostics rejected the very
            # root the SSI had just used -- returning NaN for Z_gas,
            # chi_H2O_gas and (derived from Z_gas) rho_gas.
            if not ok:
                return False
            return _gas_root_is_physical(v, np.asarray(y_gas), T, P_Pa)

        v_g_init = (np.asarray(v_gas_warm[0], dtype=float).copy()
                    if v_gas_warm[0] is not None
                    else np.array([1.0, 0.99]))
        v_g, ok_g, n_gas_inner_diag = newton_solve(
            _gas_residual, v_g_init,
            args=(np.asarray(y_gas), T, P_Pa),
            tol=1e-10, maxiter=80)
        if not _accept_v_gas(v_g, ok_g):
            # Fallback ladder for the gas-side post-convergence Newton.
            # At low T (<310K) and moderate P (~50-100 bar) the warm-start
            # Zh=1.0 falls outside the basin for the dense-gas root and
            # the Newton stalls.  Seed Zh from B_mix scaling (analogous
            # to the ladder inside lnphi_gas) and from the typical
            # χ_H2O_gas regimes (low-density gas ~0.99, dense gas near
            # ~0.5 when water content is appreciable).
            P_loc = float(P_Pa)
            b_mix_g = b1*float(y_gas[0]) + b2*float(y_gas[1]) + b3*float(y_gas[2])
            B_mix_g = b_mix_g * P_loc / R / float(T)
            Zh_B = max(B_mix_g * 1.2 + 0.3, 1.0)
            Zh_liq = max(B_mix_g * 2.0, 0.4)
            ok_g = False
            for Zh_alt, chi_alt in [
                (Zh_B, 0.99), (Zh_B*1.3, 0.99), (3.0, 0.99),
                (Zh_liq, 0.5), (Zh_liq*1.2, 0.7),
                (0.5, 0.99), (0.3, 0.99),
                (B_mix_g * 1.5, 0.9), (B_mix_g * 3.0, 0.7),
                # Unfloored dense rungs.  Every seed above is either >= 0.4
                # (Zh_liq carries a max(..., 0.4) floor) or scaled >= 1.5x
                # B_mix, which sits above the dense-gas root at low T where
                # S13 < 0 and chi may exceed 1.
                (B_mix_g * 1.2, 0.99), (B_mix_g * 1.2, 1.3),
            ]:
                v_try, ok_try, n_try = newton_solve(
                    _gas_residual, np.array([Zh_alt, chi_alt]),
                    args=(np.asarray(y_gas), T, P_Pa),
                    tol=1e-10, maxiter=80)
                if _accept_v_gas(v_try, ok_try):
                    v_g = v_try; ok_g = True; n_gas_inner_diag = n_try
                    break
        if ok_g and _accept_v_gas(v_g, True):
            Z_gas = float(v_g[0])
            chi_H2O_gas = float(v_g[1])
        la = lnphi_aq(x_aq, mi_aq_final, T, P_Pa)
        lg = lnphi_gas(y_gas, T, P_Pa)
        # Report the fugacities on the SAME density root as the (Z, eps_r,
        # chi) reported beside them.  Both kernels are called cold here --
        # lnphi_gas from its default vapour seed Z=1.0, lnphi_aq from its
        # liquid seed -- while Z_gas/Z_aq above come from the root the SSI
        # actually converged on, carried in through the warm-start chain (or
        # steered there by `gas_Z_init`).  Where the gas cubic has three
        # physical roots those are not the same state: at T=290 K, P=52 bar
        # with gas_Z_init=0.12 the reported Z_gas is the dense CO2 root while
        # the cold call returns the vapour root's lnphi, so the fugacity
        # column and the density column describe different phases and
        # fugacities_gas no longer matches fugacities_aq.  The re-evaluation
        # below is additive: it fires only when the cold root and the
        # reported root disagree by far more than Newton termination noise
        # (~1e-10 in lnphi), so a state whose two roots already agree returns
        # exactly the value it returned before.
        _ROOT_LNPHI_TOL = 1e-8
        if np.isfinite(Z_aq) and np.isfinite(eps_r) and np.isfinite(chi_H2O_aq):
            la_root = _aq_lnphi_evaluate_3var(
                np.array([Z_aq, eps_r, chi_H2O_aq]),
                np.asarray(x_aq), mi_aq_final, T, P_Pa)
            la_root = tuple(float(np.real(c)) for c in la_root)
            if max(abs(la_root[i] - float(la[i])) for i in range(3)) > _ROOT_LNPHI_TOL:
                la = la_root
        if np.isfinite(Z_gas) and np.isfinite(chi_H2O_gas):
            lg_root = _gas_lnphi_evaluate(np.array([Z_gas, chi_H2O_gas]),
                                          np.asarray(y_gas), T, P_Pa)
            lg_root = tuple(float(np.real(c)) for c in lg_root)
            if max(abs(lg_root[i] - float(lg[i])) for i in range(3)) > _ROOT_LNPHI_TOL:
                lg = lg_root
        fug_aq = tuple(float(np.exp(la[i]) * x_aq[i] * P) for i in range(3))
        fug_gas = tuple(float(np.exp(lg[i]) * y_gas[i] * P) for i in range(3))
    except Exception:
        pass

    msg = "converged" if converged else f"did not converge (max|g|={max_resid:.3e})"
    return FlashKVResult(
        T=float(T), P=float(P), z=tuple(float(zi) for zi in z),
        mi=tuple(float(m) for m in mi),
        converged=bool(converged), message=msg,
        K=tuple(float(k) for k in K),
        x_aq=tuple(float(x) for x in x_aq),
        y_gas=tuple(float(y) for y in y_gas),
        beta=float(beta), x1w=float(x1w),
        Z_aq=Z_aq, Z_gas=Z_gas, eps_r=eps_r,
        n_iter_ssi=int(n_iter_ssi),
        n_iter_newton=int(n_iter_newton),
        max_residual=float(max_resid),
        fugacities_aq=fug_aq, fugacities_gas=fug_gas,
        mi_aq=tuple(float(m) for m in mi_aq_final),
        salinity_convention=str(salinity_convention),
        chi_H2O_aq=float(chi_H2O_aq),
        chi_H2O_gas=float(chi_H2O_gas),
        n_iter_aq_inner=int(n_aq_inner_diag),
        n_iter_gas_inner=int(n_gas_inner_diag),
    )


# ---------------------------------------------------------------------------
# Dew-point / bubble-point rescue flash
# ---------------------------------------------------------------------------


#: Dissolved-gas refinement inside the dew-point rescue. The map is a
#: successive substitution on a trace composition, so it is damped and capped
#: rather than iterated to machine precision; convergence here is typically a
#: handful of sweeps.
DEW_REFINE_ITERS = 40
DEW_REFINE_DAMP = 0.7
DEW_REFINE_TOL = 1e-12

#: Largest RELATIVE fugacity imbalance a rescued state may carry. Relative,
#: because the species that matters (dissolved CO2) has a fugacity orders of
#: magnitude below water's, and an absolute tolerance would wave it through --
#: which is exactly how these states came to be 96 % wrong while reporting
#: max_residual = 0.
DEW_FUGACITY_TOL = 1e-6


def dew_point_flash(
    T, P, z, mi, x_aq_fixed,
    *,
    salinity_convention="as-aqueous",
    beta_tol=1e-8,
):
    """Solve the two-phase flash near the water dew point from a TPD trial.

    The equations are the same as in ``flash_kv``: the same equation of
    state, fugacity equality for all three components, and the same mixture
    model.  Only the iteration scheme differs.  Near the water saturation
    line the K-value successive substitution approaches its trivial fixed
    point (K = 1) and converges slowly or not at all.  Here the aqueous
    composition is iterated on the component fugacity equalities, and the
    phase fraction :math:`\\beta` follows from the water balance through a
    bracketed scalar solve, which cannot move onto the trivial solution in
    one step.  This is not an approximation or a reduced model.

    ``stability_flash`` calls this function (route 6) when every K-value
    flash failed, ``tpd_min < -1e-3``, and the best trial is aqueous.

    Parameters
    ----------
    T : float
        Temperature [K].
    P : float
        Pressure [bar].
    z : array_like, shape (3,)
        Feed mole fractions of the partitioning species (H2O, CH4, CO2);
        sum to 1.
    mi : array_like, shape (6,)
        Ion molalities (Na+, Cl-, K+, Ca2+, Mg2+, SO4 2-) [mol/kg water].
    x_aq_fixed : array_like, shape (3,)
        Starting aqueous composition on the partitioning basis (H2O, CH4,
        CO2), normally the composition ``w`` of the aqueous TPD trial.  It
        is converted to the ion-inclusive basis and then solved for; it is
        not held fixed.  Because the stability test found it as a
        composition distinct from the feed, the iteration starts away from
        the trivial solution.
    salinity_convention : {"as-aqueous", "chain"}
        As in ``flash_kv``.  Here it affects only the reported ``mi_aq``.
    beta_tol : float
        Tolerance on β in the Brent solve.

    Returns
    -------
    FlashKVResult or None
        A result with ``converged=True``, or None if no β can be bracketed,
        a fugacity evaluation fails, or the largest relative fugacity
        difference over present components exceeds ``DEW_FUGACITY_TOL``.
        ``max_residual`` holds that relative difference.

    Notes
    -----
    The phase fraction follows from the water fugacity balance

    .. math::

        x_{\\rm H_2O}^{\\rm aq}\\,\\varphi_{\\rm H_2O}^{\\rm aq}(x_{\\rm aq})
        = y_{\\rm H_2O}(\\beta)\\,\\varphi_{\\rm H_2O}^{\\rm gas}(y_{\\rm gas}(\\beta)),

    where the gas composition ``y_gas(β)`` is fixed by overall mass
    balance ``z = β y_gas + (1−β) x_aq``.  Brent's method on this
    single scalar equation gives β in the interior of (0, 1), away from
    the trivial solution K = 1.

    The water fugacity balance fixes β.  The dissolved-gas fractions are
    then updated by damped successive substitution,
    ``x_i = y_i φ_i^gas / φ_i^aq`` for CH4 and CO2, with water as the
    balance and β re-solved after each sweep (at most
    ``DEW_REFINE_ITERS`` sweeps, damping ``DEW_REFINE_DAMP``).  This step
    is required for the dissolved-gas fractions to be correct: x_CO2 is
    itself of order 1e-5, so an absolute fugacity residual of 1e-5 is a
    relative error of order 100 % in x_CO2.  The residual reported is
    therefore relative, per component.
    """
    from scipy.optimize import brentq
    from .lnphi import (lnphi_aq, lnphi_gas, _aq_residual_3var, _gas_residual,
                        _gas_accept_or_repair)
    from .newton import newton_solve

    z = np.asarray(z, dtype=float)
    mi = np.asarray(mi, dtype=float)
    x_aq = np.asarray(x_aq_fixed, dtype=float)
    P_Pa = float(P) * 1e5

    # Put the trial composition on the SAME basis the rest of the flash uses.
    # x_aq_fixed arrives on the partition basis (the three partitioning
    # species, summing to 1). The aqueous phase, however, also contains the
    # ions, so the closure the aqueous kernel assumes is
    #     x1w + x2w + x3w + sum(x_ion) = 1,  x_ion = x1w * Mw * mi,
    # i.e. the three species sum to 1 - sum(x_ion), not to 1. _ssi_step
    # (flash_kv) builds x_aq on exactly this basis; converting here keeps the
    # two paths on one basis, which matters because both feed the same
    # lnphi_aq and it consumes x_aq[0] as given.
    _m_tot = float(np.sum(mi))
    if x_aq.sum() > 0:
        x_part = x_aq / x_aq.sum()
        _x1w_true = (float(x_part[0] / (1.0 + Mw * _m_tot * x_part[0]))
                     if x_part[0] > 0 else 0.0)
        _scale = 1.0 - Mw * _m_tot * _x1w_true
        x_aq = np.array([_x1w_true,
                         float(x_part[1]) * _scale,
                         float(x_part[2]) * _scale])

    # Aqueous-side fugacity of water is constant (x_aq fixed).
    try:
        ln_aq = lnphi_aq(tuple(x_aq), mi, T, P_Pa)
    except Exception:
        return None
    phi_w_aq = float(np.exp(ln_aq[0]))
    f_w_aq = x_aq[0] * phi_w_aq * float(P)   # bar

    # β lower bound: y_gas_i ≥ 0 for every present component.
    lo_candidates = [1e-8]
    for i in range(3):
        if x_aq[i] > 1e-12:
            lo_candidates.append(1.0 - float(z[i]) / float(x_aq[i]))
    beta_lo = max(lo_candidates) + 1e-8
    beta_hi = 1.0 - 1e-8
    if beta_lo >= beta_hi:
        return None

    la_cache = ln_aq

    def y_gas_of_beta(beta, x_aq=x_aq):
        y = (z - (1.0 - beta) * x_aq) / beta
        y = np.clip(y, 0.0, None)
        s = y.sum()
        return y / s if s > 0 else y

    def _solve_beta(x_aq, f_w_aq):
        """Brent on water fugacity balance at a FIXED aqueous composition."""
        lo_c = [1e-8]
        for i in range(3):
            if x_aq[i] > 1e-12:
                lo_c.append(1.0 - float(z[i]) / float(x_aq[i]))
        b_lo, b_hi = max(lo_c) + 1e-8, 1.0 - 1e-8
        if b_lo >= b_hi:
            return None

        def water_residual(beta):
            y_gas = y_gas_of_beta(beta, x_aq)
            try:
                ln_gas = lnphi_gas(tuple(y_gas), T, P_Pa)
            except Exception:
                return np.nan
            return f_w_aq - y_gas[0] * float(np.exp(ln_gas[0])) * float(P)

        r_lo, r_hi = water_residual(b_lo), water_residual(b_hi)
        if not (np.isfinite(r_lo) and np.isfinite(r_hi)) or r_lo * r_hi > 0:
            return None
        try:
            return float(brentq(water_residual, b_lo, b_hi,
                                xtol=beta_tol, maxiter=200))
        except Exception:
            return None

    beta = _solve_beta(x_aq, f_w_aq)
    if beta is None:
        return None

    # ---- refine the dissolved-gas fractions -----------------------------
    # Successive substitution on the gas components only: fugacity equality
    # gives x_i = y_i * phi_i^gas / phi_i^aq, with water taken as the balance
    # so the aqueous composition stays normalised. beta is re-solved each
    # sweep because the mass balance moves with x_aq. Damped: x_gas_aq is
    # small and the map can overshoot into negative water.
    for _ in range(DEW_REFINE_ITERS):
        y_gas = y_gas_of_beta(beta, x_aq)
        try:
            ln_gas = lnphi_gas(tuple(y_gas), T, P_Pa)
            ln_aq_i = lnphi_aq(tuple(x_aq), mi, T, P_Pa)
        except Exception:
            return None
        x_new = np.array(x_aq, dtype=float)
        for i in (1, 2):
            if z[i] <= 0.0:
                x_new[i] = 0.0
                continue
            x_new[i] = y_gas[i] * float(np.exp(ln_gas[i] - ln_aq_i[i]))
        gas_sum = float(x_new[1] + x_new[2])
        if not np.isfinite(gas_sum) or gas_sum >= 1.0:
            return None
        # Water is the balance, but the balance is against the ion closure
        # above: x1w (1 + Mw m_tot) + gas_sum = 1. At m_tot = 0 this is the
        # plain 1 - gas_sum, which is why salt-free states are unaffected.
        x_new[0] = (1.0 - gas_sum) / (1.0 + Mw * _m_tot)
        step = float(np.max(np.abs(x_new - np.asarray(x_aq, float))))
        x_aq = ((1.0 - DEW_REFINE_DAMP) * np.asarray(x_aq, float)
                + DEW_REFINE_DAMP * x_new)
        try:
            ln_aq_i = lnphi_aq(tuple(x_aq), mi, T, P_Pa)
        except Exception:
            return None
        f_w_aq = x_aq[0] * float(np.exp(ln_aq_i[0])) * float(P)
        b_new = _solve_beta(x_aq, f_w_aq)
        if b_new is None:
            return None
        beta = b_new
        if step < DEW_REFINE_TOL:
            break

    try:
        la_cache = lnphi_aq(tuple(x_aq), mi, T, P_Pa)
    except Exception:
        return None

    y_gas = y_gas_of_beta(beta, x_aq)
    ln_gas = lnphi_gas(tuple(y_gas), T, P_Pa)

    # The honest residual, across every component actually present, measured
    # RELATIVE to the fugacity scale so a trace species is judged on its own
    # terms rather than against water's.
    # Accumulated with np.max, not the builtin: the builtin keeps its
    # incumbent whenever the comparison is False, and `nan > x` is False, so
    # a non-finite component term would be dropped and the state would be
    # certified on the remaining components -- with the isfinite test below
    # never able to fire. np.max propagates the NaN into `resid`, which is
    # what that test is there to catch.
    terms = []
    for i in range(3):
        if z[i] <= 0.0:
            continue
        fa = float(np.exp(la_cache[i]) * x_aq[i] * float(P))
        fg = float(np.exp(ln_gas[i]) * y_gas[i] * float(P))
        scale = max(abs(fa), abs(fg), 1e-300)
        terms.append(abs(fa - fg) / scale)
    resid = float(np.max(terms)) if terms else 0.0
    if not np.isfinite(resid) or resid > DEW_FUGACITY_TOL:
        # Previously returned as converged with max_residual = 0.0. Refusing
        # is what lets the caller fall back to something that works.
        return None

    # K = y/x for the partition-basis components with x > 0.
    present = z > 0.0
    K = np.array([
        y_gas[i] / x_aq[i] if (present[i] and x_aq[i] > 1e-12) else float("nan")
        for i in range(3)
    ])

    # Fill in the diagnostic warm-start state (Z_aq, eps_r, chi_aq, Z_gas,
    # chi_gas).  Use the standard v_init discipline; failures leave NaN.
    try:
        v_aq0 = np.array([float(P_Pa) * 18.015e-6 / (8.314 * float(T)), 50.0, 0.10])
        v_aq, ok_aq, n_aq = newton_solve(_aq_residual_3var, v_aq0,
                                         args=(x_aq, mi, T, P_Pa),
                                         tol=1e-10, maxiter=80)
        # Same acceptance the equilibrium paths apply (lnphi_aq and
        # flash_kv's SSI and diagnostics): newton_solve's `converged` is a
        # residual test only, and the aqueous closure has fixed points with a
        # non-positive monomer fraction or permittivity that satisfy it. NaN
        # reads as "not available"; a plausible-looking negative chi does not.
        ok_aq = bool(ok_aq) and v_aq[2] > 0 and v_aq[1] > 0
        Z_aq = float(v_aq[0]) if ok_aq else float("nan")
        eps_r = float(v_aq[1]) if ok_aq else float("nan")
        chi_H2O_aq = float(v_aq[2]) if ok_aq else float("nan")
        n_iter_aq_inner = int(n_aq)
    except Exception:
        Z_aq = eps_r = chi_H2O_aq = float("nan"); n_iter_aq_inner = 0
    try:
        v_g0 = np.array([1.0, 0.99])
        v_g, ok_g, n_g = newton_solve(_gas_residual, v_g0,
                                      args=(np.asarray(y_gas), T, P_Pa),
                                      tol=1e-10, maxiter=80)
        # _gas_residual has no logarithms, so the negative-chi branch of the
        # association closure is an ordinary fixed point: this Newton reports
        # it as converged and it would be published as chi_H2O_gas (measured
        # as low as -11 at 308 K / 899 bar) beside fugacities that came from
        # lnphi_gas's repaired root. Run the same accept-or-repair lnphi_gas
        # runs so the two describe one state; a root that is already physical
        # passes through untouched.
        v_g, ok_g = _gas_accept_or_repair(v_g, ok_g, np.asarray(y_gas),
                                          T, P_Pa, 1e-10, 80)
        Z_gas = float(v_g[0]) if ok_g else float("nan")
        chi_H2O_gas = float(v_g[1]) if ok_g else float("nan")
        n_iter_gas_inner = int(n_g)
    except Exception:
        Z_gas = chi_H2O_gas = float("nan"); n_iter_gas_inner = 0

    fug_aq = tuple(float(np.exp(la_cache[i]) * x_aq[i] * float(P)) for i in range(3))
    fug_gas = tuple(float(np.exp(ln_gas[i]) * y_gas[i] * float(P)) for i in range(3))

    # Under 'chain' salinity, recompute mi_aq at the converged β/x1w.
    if salinity_convention == "chain" and 0.0 < (1.0 - beta) and x_aq[0] > 0:
        # partition-basis water fraction = x_aq[0] * (1 + S); see
        # compute_aqueous_x1w_chain.
        _S = Mw * float(z[0]) * float(np.sum(mi)) / (1.0 - float(beta))
        mi_aq_final = np.asarray(mi, dtype=float) * (
            float(z[0]) / ((1.0 - float(beta)) * float(x_aq[0]) * (1.0 + _S)))
    else:
        mi_aq_final = np.asarray(mi, dtype=float)

    return FlashKVResult(
        T=float(T), P=float(P),
        z=tuple(float(zi) for zi in z),
        mi=tuple(float(m) for m in mi),
        converged=True,
        message="two-phase (dew-point rescue: fixed x_aq, Brent on β)",
        K=tuple(float(k) for k in K),
        x_aq=tuple(float(x) for x in x_aq),
        y_gas=tuple(float(y) for y in y_gas),
        beta=float(beta),
        x1w=float(x_aq[0]),
        Z_aq=Z_aq, Z_gas=Z_gas, eps_r=eps_r,
        n_iter_ssi=0,          # no K-space SSI; this is a specialised rescue
        n_iter_newton=1,        # one Brent solve
        max_residual=float(resid),
        fugacities_aq=fug_aq, fugacities_gas=fug_gas,
        mi_aq=tuple(float(m) for m in mi_aq_final),
        salinity_convention=str(salinity_convention),
        chi_H2O_aq=chi_H2O_aq, chi_H2O_gas=chi_H2O_gas,
        n_iter_aq_inner=n_iter_aq_inner,
        n_iter_gas_inner=n_iter_gas_inner,
    )
