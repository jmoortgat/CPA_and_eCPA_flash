"""Michelsen tangent-plane-distance (TPD) phase-stability test for the
multi-salt eCPA, plus the combined stability + flash driver.

Architecture
------------
The TPD primitive ``tpd_ssi`` runs a single Michelsen successive-substitution
trial in log-space::

    ln W_i^{k+1} = d_i - lnφ_trial_i(w^k, ...)
    w^k = W^k / Σ_j W_j^k
    tm* = 1 - Σ_i W_i     (at the stationary point)

with the standard convention ``tpd = 1 - sum(W_i)`` (so ``tpd < 0`` ⇒
unstable).  The reference vector ``d_i`` is built from the **gas-phase EoS
evaluated at the partitioning-component feed composition** ``z``.  This
choice does not favour either gas; it generalises the reference of Paper I
(the CO₂-rich EoS at the feed composition), which assumed a single gas.
See ``michelsen_tpd`` for the trial schedule.

Jex 2024 acceleration is applied to the ``lnW`` step using the same
diagonal approximation used in the K-value flash, with a cap
``m ∈ [1, 10]``.

The functions of this module are:

* ``tpd_ssi``: one TPD trial; returns ``TPDResult``.
* ``michelsen_tpd``: runs up to 8 trials and returns the most negative
  ``tpd_min``; returns ``StabilityResult``.
* ``make_k_init_from_trial``: builds initial K-values for ``flash_kv``
  from a trial.
* ``stability_flash``: the TPD test followed by ``flash_kv`` from a
  sequence of initial K-values, with single-phase fallbacks.  Returns
  ``StabilityFlashResult``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .lnphi import lnphi_aq, lnphi_gas
from .parameters import Mw
from .flash import flash_kv, wilson_k, FlashKVResult


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class TPDResult:
    """Result of one Michelsen TPD trial (returned by ``tpd_ssi``).

    Attributes
    ----------
    tpd : float
        Tangent-plane distance ``1 - sum(W)`` at the last iterate.  A
        negative value indicates that the feed is unstable.
    W : tuple of float, length 3
        Final trial mole numbers (H2O, CH4, CO2), not normalised.
    w : tuple of float, length 3
        Final trial mole fractions ``W / sum(W)`` (H2O, CH4, CO2), on the
        partitioning basis (ions excluded).
    lnphi_trial : tuple of float, length 3
        ln of the fugacity coefficients in the trial phase at ``w``.  On the
        aqueous side this includes the ion-closure term ``ln s``.
    n_iter : int
        Number of successive-substitution iterations run.
    converged : bool
        True when ``max|g| < tol``.
    message : str
        "converged", or the reason the iteration stopped.
    side : str
        "aq" or "gas": the fugacity-coefficient kernel used for the trial
        phase.
    label : str
        Trial label used in diagnostic messages.
    early_terminated : bool
        Always False in the current code (early termination was removed).
    max_residual : float
        Stationarity residual ``max|g|``, with
        ``g_i = ln W_i + lnphi_i - d_i``, at the last iterate evaluated.
        Same meaning as ``FlashKVResult.max_residual``.  NaN only when the
        loop did not run (``maxiter <= 0``).  It distinguishes a trial that
        missed ``tol`` narrowly from one whose ``tpd`` is not meaningful.
    """
    tpd: float
    W: tuple             # final mole-number vector (3,)
    w: tuple             # final normalised mole fractions (3,)
    lnphi_trial: tuple   # final lnφ in trial phase (3,)
    n_iter: int
    converged: bool
    message: str
    side: str = "?"      # "aq" or "gas" (which lnφ kernel was used)
    label: str = ""      # human-readable trial label
    early_terminated: bool = False
    # Stationarity residual max|g| at the last iterate the trial evaluated,
    # in the same sense as FlashKVResult.max_residual / Flash3Result.
    # max_residual.  NaN only when the loop never ran (maxiter <= 0).  A
    # trial that stops at maxiter is not necessarily far from a stationary
    # point, and without this number a caller cannot tell a trial that
    # missed `tol` by a hair from one whose tpd is noise.
    max_residual: float = float("nan")


@dataclass
class StabilityResult:
    """Result of the Michelsen TPD trial search (returned by ``michelsen_tpd``).

    Attributes
    ----------
    stable : bool
        True when ``tpd_min > -tpd_threshold``, or when no trial gave a
        negative TPD.  ``message`` starts with "INCONCLUSIVE" when every
        trial converged to the feed composition or failed.
    tpd_min : float
        Most negative TPD over the converged, non-trivial trials; 0.0 when
        none was negative.
    trials : list of TPDResult
        Every trial run, in order (the standard panel, then the wide panel
        when it was needed).
    best_trial_idx : int
        Index in ``trials`` of the trial that gave ``tpd_min``; -1 when no
        trial gave a negative TPD.
    message : str
        Summary of the search, including warnings when the reference
        fugacities had to be evaluated on the aqueous side.
    d_ref : tuple of float, length 3
        Reference vector ``d_i = ln z_i + lnphi_ref_i(z)`` (H2O, CH4, CO2);
        0.0 for components absent from the feed.
    z : tuple of float, length 3
        Feed mole fractions (H2O, CH4, CO2) on the partitioning basis.
    """
    stable: bool
    tpd_min: float
    trials: List[TPDResult]
    best_trial_idx: int
    message: str
    d_ref: tuple                # reference d-vector (3,) = ln(z) + lnφ_ref
    z: tuple                    # the partition-basis feed used


@dataclass
class StabilityFlashResult:
    """Result of the stability test and flash (from ``stability_flash``).

    Attributes
    ----------
    phases : int
        Number of phases reported: 1 or 2.
    tpd_min : float
        Most negative TPD found by the stability test (``stability.tpd_min``).
    stable : bool
        True when the result is single-phase (``phases == 1``).
    flash_result : FlashKVResult or None
        The accepted two-phase flash when ``phases == 2``.  For route 7 it is
        the converged flash whose phase fraction lies at 0 or 1; for route 9
        it is the last failed attempt, or None.  None for routes 0 and 8.
    stability : StabilityResult
        Full result of ``michelsen_tpd``.
    attempts : list of str
        One line per step taken (stability test and each flash attempt).
    message : str
        Short summary of the outcome.
    T : float
        Temperature [K].
    P : float
        Pressure [bar].
    z : tuple of float, length 3
        Feed mole fractions (H2O, CH4, CO2) on the partitioning basis.
    mi : tuple of float, length 6
        Ion molalities (Na+, Cl-, K+, Ca2+, Mg2+, SO4 2-) [mol/kg water].
    route : int
        Step of the algorithm that produced the result; the codes are
        listed under Notes.
    t_stab_ms : float
        Wall time of the stability test [ms].
    t_flash_ms : float
        Total wall time of all flash attempts run, including rejected ones
        [ms].
    t_total_ms : float
        Wall time of the whole call [ms]; ``t_stab_ms + t_flash_ms`` plus a
        small overhead.

    Notes
    -----
    The ``route`` codes are the same as those returned by the Fortran
    library (``ms_ecpa.backend``):

    * 0: single-phase, ``tpd_min`` above ``tpd_threshold``.
    * 1: two-phase, flash started from the K-values of the best trial.
    * 2: two-phase, flash started from Wilson K-values.
    * 3: two-phase, flash started from water-biased K-values.
    * 4: two-phase, successive substitution only (no acceleration, no
      Newton step).
    * 5: two-phase, flash started from K-values built from the best
      aqueous and the best gas trial.
    * 6: two-phase, dew-point flash (``dew_point_flash``).
    * 7: single-phase, every converged flash gave a phase fraction at 0
      or 1.
    * 8: single-phase, every flash failed and ``tpd_min`` is above
      ``tpd_soft_threshold``; no flash result is attached.
    * 9: failure; no flash converged and ``tpd_min`` is below
      ``tpd_soft_threshold`` (``phases`` is 2).
    * -1: not recorded (results assembled elsewhere, for example by the
      re-flash in ``ms_ecpa.flash3``).

    Routes 7 and 8 are single-phase results obtained after the TPD test
    found the feed unstable.  Their ``phases`` and ``stable`` fields are the
    same as for route 0, so use ``route`` to tell them apart.
    """
    phases: int                                  # 1 or 2
    tpd_min: float
    stable: bool                                 # True ⇔ single-phase
    flash_result: Optional[FlashKVResult]        # populated when phases == 2
    stability: StabilityResult
    attempts: List[str]                          # human-readable attempt log
    message: str
    T: float
    P: float
    z: tuple
    mi: tuple
    # Which rung of the ladder produced this answer, as an integer so that
    # callers do not have to parse ``message`` to find out.  The codes are
    # the ones the Fortran engine already returns (see ms_ecpa.backend._ROUTE):
    #   0 single-phase (hard TPD guard)      5 two-phase (dual-trial K-init)
    #   1 two-phase (trial K-init)           6 two-phase (dew-point rescue)
    #   2 two-phase (Wilson K-init)          7 single-phase (β-degenerate)
    #   3 two-phase (water-biased K-init)    8 single-phase (soft fallback)
    #   4 two-phase (pure-SSI)               9 hard failure
    # Routes 7 and 8 are single-phase classifications reached *after* the
    # TPD test called the feed unstable, so they are not interchangeable
    # with route 0 even though the returned fields look alike; route 8 in
    # particular carries no flash result at all.  -1 means "not recorded":
    # results assembled elsewhere (e.g. flash3's re-flash) leave it unset.
    route: int = -1
    # Split timing (ms) so reservoir-simulator readers can separate the
    # stability-test cost — which a warm-started simulator can skip
    # entirely between timesteps — from the flash cost, which is the
    # per-cell VLE work analogous to a non-flash equation-of-state solve.
    # t_stab_ms is the Michelsen TPD wall time; t_flash_ms sums every
    # flash_kv attempt actually run (including retries when a candidate
    # split was rejected as β-degenerate or phase-swapped); t_total_ms
    # is the end-to-end stability_flash wall time and equals
    # t_stab_ms + t_flash_ms + a small classification/dispatch overhead.
    t_stab_ms: float = 0.0
    t_flash_ms: float = 0.0
    t_total_ms: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _present_mask(z):
    """Boolean mask of partitioning components actually present in the feed."""
    return np.asarray(z, dtype=float) > 0.0


def _safe_log(x, floor=1e-300):
    """Log clamped from below to avoid -inf at zero-composition components."""
    return float(np.log(max(float(x), floor)))


#: Evaluate an aqueous trial on the composition basis the FLASH uses.
#:
#: The flash carries its aqueous composition on the partitioning basis and
#: converts to true aqueous mole fractions through the ion closure,
#: ``x_aq = s x`` with ``s = 1 - sum_j x_j``, so a species' fugacity in the
#: brine is ``f_i = s x_i phi_i(s x) P`` and the matching ``ln s`` appears in
#: the flash's K-update.  The trial as first implemented passed the
#: partition-basis ``w`` to the aqueous kernel directly and carried no
#: ``ln s``.  Normally that term would cancel against the reference, but the
#: reference here is built on the gas side, which is ion-free, so the closure
#: factor has nothing to cancel against and does not drop out.  In a salt-free
#: system ``s == 1`` and the two conventions coincide.
#:
#: Set False to recover the original convention.  Kept because the difference
#: changes reported classifications near the phase boundary and the comparison
#: has to stay runnable.
AQ_TRIAL_ON_FLASH_BASIS = True


def _aq_closure_scale(w_full, mi):
    """Ion-closure factor ``s`` for a partition-basis aqueous composition.

    Mirrors the flash exactly (``ms_ecpa.flash``): with ``m_tot`` the total ion
    molality and ``x_part`` the normalised partition-basis composition,
    ``x1w_true = x_part_w / (1 + Mw m_tot x_part_w)`` and
    ``s = 1 - Mw m_tot x1w_true``, whence ``x_aq = s x_part``.
    """
    m_tot = float(np.sum(np.asarray(mi, dtype=float)))
    if m_tot <= 0:
        return 1.0
    xw = float(np.asarray(w_full, dtype=float)[0])
    if xw <= 0:
        return 1.0
    x1w_true = xw / (1.0 + Mw * m_tot * xw)
    return float(1.0 - Mw * m_tot * x1w_true)


def _evaluate_lnphi(side, w_full, mi, T, P_Pa):
    """Dispatch lnφ evaluation to the requested side.

    Both kernels expect a 3-vector (H2O, CH4, CO2) and return a 3-tuple.
    ``mi`` is the ion molality vector — used only on the aqueous side.
    The gas side is ion-free by construction (K_ion ≡ 0).

    Returns ``(lnphi, ln_s)``.  ``ln_s`` is the ion-closure term the caller
    must add to ``g_i``; it is zero on the gas side and, under the original
    convention, on the aqueous side too.
    """
    w_full = np.asarray(w_full, dtype=float)
    if side == "aq":
        if AQ_TRIAL_ON_FLASH_BASIS:
            s = _aq_closure_scale(w_full, mi)
            return lnphi_aq(tuple(s * w_full), mi, T, P_Pa), float(np.log(s))
        return lnphi_aq(tuple(w_full), mi, T, P_Pa), 0.0
    elif side == "gas":
        return lnphi_gas(tuple(w_full), T, P_Pa), 0.0
    else:
        raise ValueError(f"side must be 'aq' or 'gas', got {side!r}")


# ---------------------------------------------------------------------------
# Single-trial TPD SSI
# ---------------------------------------------------------------------------

def tpd_ssi(
    d_i,
    trial_init,
    T,
    P,
    mi,
    side,
    *,
    present=None,
    accelerated=True,
    tol=1e-8,
    maxiter=80,
    early_tpd=-0.01,
    min_iter_before_early=3,
    label="",
):
    """Run one Michelsen TPD trial by successive substitution in ln W.

    Parameters
    ----------
    d_i : array_like, shape (3,)
        Reference d-vector ``ln(z_i) + lnφ_ref_i(z)`` for the partitioning
        components (H2O, CH4, CO2).  The "ref" lnφ is the gas-phase EoS
        evaluated at the renormalised partitioning-component feed (see
        ``michelsen_tpd``).  Entries corresponding to absent species
        (``z_i = 0``) may be set to anything — they are masked out via
        ``present``.
    trial_init : array_like, shape (3,)
        Initial mole-number vector ``W_i^0`` for the partitioning components
        (not normalised; the iteration finds W and its sum).
    T : float
        Temperature [K].
    P : float
        Pressure [bar].
    mi : array_like, shape (6,)
        Ion molalities (Na+, Cl-, K+, Ca2+, Mg2+, SO4 2-) [mol/kg water].
        Used only when ``side == "aq"`` (the gas phase contains no ions).
    side : {"aq", "gas"}
        Selects which lnφ kernel is called.
    present : array_like of bool, shape (3,), optional
        Mask of partitioning components actually present in the feed.
        Defaults to ``np.array([True, True, True])``.
    accelerated : bool
        Apply the acceleration of Jex et al. (2024) to the ln W step.
    tol : float
        Convergence tolerance on ``max|g|``, where
        ``g_i = ln W_i + lnφ_i - d_i``.
    maxiter : int
        Maximum number of iterations.
    early_tpd : float
        Accepted for compatibility and not used: early termination on
        ``tpd < early_tpd`` was removed, because near dew points and
        critical points the iterate can pass through negative TPD values on
        its way to the trivial solution ``w = z``.
    min_iter_before_early : int
        Accepted for compatibility and not used (see ``early_tpd``).
    label : str
        Trial label used in diagnostic messages.

    Returns
    -------
    TPDResult
        Final trial composition, its TPD, and the convergence information.
        ``converged`` is False when ``maxiter`` was reached or a
        fugacity-coefficient evaluation failed; ``message`` gives the
        reason.
    """
    d_i = np.asarray(d_i, dtype=float)
    W = np.asarray(trial_init, dtype=float).copy()
    if present is None:
        present = np.array([True, True, True])
    present = np.asarray(present, dtype=bool)

    # Absent components: force W_i = 0 throughout the iteration.
    W = np.where(present, np.clip(W, 1e-300, None), 0.0)
    P_Pa = float(P) * 1e5

    g_prev = None
    m_acc = 1.0
    n_iter = 0
    converged = False
    # Bound before the loop so the non-convergence message and the returned
    # residual are defined even when the loop body never runs (maxiter <= 0).
    max_g = float("nan")
    early = False
    lnphi_t = np.array([np.nan, np.nan, np.nan])
    last_msg = ""
    sum_W = float(W.sum())
    tpd = 1.0 - sum_W

    for it in range(maxiter):
        # Normalised trial composition w
        sum_W = float(W.sum())
        if not np.isfinite(sum_W) or sum_W <= 0:
            last_msg = f"non-positive sum_W={sum_W}"
            break
        w = np.where(present, W / sum_W, 0.0)
        # Numerical safety: keep components present in the trial > tiny floor
        # so lnφ inner Newtons see a non-degenerate composition.
        w = np.where(present, np.clip(w, 1e-12, None), 0.0)
        # Re-normalise after clip
        sw = float(w.sum())
        if sw <= 0:
            last_msg = f"degenerate w after clip"
            break
        w = w / sw

        # lnφ at trial composition
        try:
            (l1, l2, l3), ln_s = _evaluate_lnphi(side, w, mi, T, P_Pa)
        except Exception as exc:
            last_msg = f"lnphi failure at it={it}: {exc}"
            break
        # ln_s is the ion-closure term; zero on the gas side and under the
        # original aqueous convention.
        lnphi_t = np.array([float(l1), float(l2), float(l3)]) + ln_s

        # Stationarity residual g_i = ln W_i + lnφ_i - d_i (present species only).
        lnW = np.where(present,
                       np.log(np.where(present, np.clip(W, 1e-300, None), 1.0)),
                       0.0)
        g_full = lnW + lnphi_t - d_i
        g = np.where(present, g_full, 0.0)
        g_p = g[present]
        max_g = float(np.max(np.abs(g_p))) if g_p.size else 0.0

        # Direct-substitution update in log space
        lnW_new = d_i - lnphi_t  # only present entries are meaningful
        lnW_step = lnW_new - lnW
        lnW_step = np.where(present, lnW_step, 0.0)
        # Clip per-step excursion (matches prior project)
        lnW_step = np.clip(lnW_step, -5.0, 5.0)

        # Jex 2024 acceleration on lnW step (diagonal U_j^{-1} ≈ I)
        if accelerated and g_prev is not None and g_prev.shape == g_p.shape:
            num_a = float(np.dot(g_prev, g_prev))
            denom_a = float(np.dot(g_prev, g_prev - g_p))
            if abs(denom_a) > 1e-30:
                m_acc = abs(num_a / denom_a * m_acc)
                m_acc = float(np.clip(m_acc, 1.0, 10.0))
            else:
                m_acc = 1.0
        g_prev = g_p.copy()

        # Apply update
        lnW_next = lnW + m_acc * lnW_step
        lnW_next = np.clip(lnW_next, -500.0, 500.0)
        W = np.where(present, np.exp(lnW_next), 0.0)

        sum_W = float(W.sum())
        tpd = 1.0 - sum_W if np.isfinite(sum_W) else np.nan
        n_iter = it + 1

        # Convergence test
        if max_g < tol:
            converged = True
            last_msg = "converged"
            break

        # Early-termination on ``tpd < early_tpd`` was removed here: at
        # near-dew / near-critical conditions the SSI iterate transiently
        # dives to strongly negative tpd values while descending toward
        # the *trivial* fixed point w=z (tpd=0, single-phase).  Stopping
        # at that dive gave a false unstable classification and drove
        # the downstream flash into the β-degenerate trap.  Instead we
        # let SSI run to full residual-norm convergence; the trivial-w
        # guard in ``michelsen_tpd`` still rejects the final result
        # when it settles on the feed composition, and the trial-search
        # early exit still fires on genuinely converged negative tpd.

    if not converged and not last_msg:
        last_msg = (f"did not converge in {maxiter} iters "
                    f"(max|g|={max_g:.3e})")

    sum_W = float(W.sum()) if np.all(np.isfinite(W)) else np.nan
    tpd_final = 1.0 - sum_W if np.isfinite(sum_W) else np.nan
    w_final = (W / sum_W) if (np.isfinite(sum_W) and sum_W > 0) else np.zeros(3)

    return TPDResult(
        tpd=float(tpd_final),
        W=tuple(float(x) for x in W),
        w=tuple(float(x) for x in w_final),
        lnphi_trial=tuple(float(x) for x in lnphi_t),
        n_iter=int(n_iter),
        converged=bool(converged),
        message=last_msg,
        side=side,
        label=label,
        early_terminated=bool(early),
        max_residual=float(max_g),
    )


# ---------------------------------------------------------------------------
# 8-trial Michelsen TPD search
# ---------------------------------------------------------------------------

def _build_trial_list(z, present, T, P, mi):
    """Construct the 8-trial schedule.

    Each entry is ``(label, side, W0)`` where ``W0`` is a length-3
    mole-number vector seeded so that ``W0/sum(W0)`` matches the spec's
    trial composition.  The actual ``sum(W0)`` is unimportant at the start
    of SSI (the iteration finds it), so we use ``sum(W0) = 1`` to leave
    log-space well-conditioned.

    Trials skipped automatically when degenerate (component absent in z).
    """
    z = np.asarray(z, dtype=float)
    K_wil = wilson_k(T, P)  # 3-vector (H2O, CH4, CO2); same length as present

    trials = []

    # 1, 2: CO2-rich gas trials
    if present[2]:
        for x_h2o, lab in [(0.01, "gas CO2-rich (x_H2O=0.01)"),
                            (0.10, "gas CO2-rich (x_H2O=0.10)")]:
            w = np.array([x_h2o, 0.0, 1.0 - x_h2o])
            trials.append((lab, "gas", w.copy()))

    # 3, 4: CH4-rich gas trials
    if present[1]:
        for x_h2o, lab in [(0.01, "gas CH4-rich (x_H2O=0.01)"),
                            (0.10, "gas CH4-rich (x_H2O=0.10)")]:
            w = np.array([x_h2o, 1.0 - x_h2o, 0.0])
            trials.append((lab, "gas", w.copy()))

    # 5, 6: aqueous trials (water-rich)
    # 5 — pure-water aqueous: x_H2O = 0.99, ε for present gases
    eps = 1e-4
    w_aq1 = np.zeros(3)
    w_aq1[0] = 0.99
    n_present_gas = int(present[1]) + int(present[2])
    if n_present_gas > 0:
        gas_share = (1.0 - w_aq1[0]) / n_present_gas
        if present[1]:
            w_aq1[1] = gas_share
        if present[2]:
            w_aq1[2] = gas_share
    else:
        w_aq1[0] = 1.0
    trials.append(("aq water-rich (x_H2O=0.99)", "aq", w_aq1))

    # 6 — moderate aqueous: gas-side molar share ~0.05 each (or ~0.1 if only
    # one gas present); biased toward the gas the feed actually contains.
    w_aq2 = np.zeros(3)
    if n_present_gas == 2:
        # bias slightly by the feed: heavier-weighted on the larger z
        z_gases = np.array([z[1], z[2]])
        if z_gases.sum() > 0:
            split = z_gases / z_gases.sum()
        else:
            split = np.array([0.5, 0.5])
        gas_total = 0.10
        w_aq2[1] = gas_total * split[0]
        w_aq2[2] = gas_total * split[1]
        w_aq2[0] = 1.0 - gas_total
    elif n_present_gas == 1:
        gas_total = 0.05
        if present[1]:
            w_aq2[1] = gas_total
        if present[2]:
            w_aq2[2] = gas_total
        w_aq2[0] = 1.0 - gas_total
    else:
        w_aq2[0] = 1.0
    trials.append(("aq moderate (x_gas≈5–10%)", "aq", w_aq2))

    # 7, 8: Wilson-K trials.
    # 7 (aq-like): w_i ∝ z_i / K_i
    # 8 (gas-like): w_i ∝ K_i * z_i
    z_p = np.where(present, z, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_aq_wil = np.where(present, z_p / np.where(present, K_wil, 1.0), 0.0)
        w_gas_wil = np.where(present, K_wil * z_p, 0.0)
    s_aq = float(w_aq_wil.sum())
    s_gas = float(w_gas_wil.sum())
    if s_aq > 0:
        w_aq_wil = w_aq_wil / s_aq
        trials.append(("Wilson aq-like (z/K)", "aq", w_aq_wil))
    if s_gas > 0:
        w_gas_wil = w_gas_wil / s_gas
        trials.append(("Wilson gas-like (K*z)", "gas", w_gas_wil))

    return trials


def _build_wide_trial_list(z, present):
    """A deliberately spread-out trial panel, for when the normal one collapses.

    The standard schedule clusters near pure water, near the feed, and on the
    Wilson K estimate. Near the mixture critical region all three sit inside
    the basin of the TRIVIAL solution W = z: every trial converges back to the
    feed, every TPD comes back as numerical noise around zero, and the search
    reports stability having looked nowhere. Measured for CO2-H2O at 573 K /
    575 bar, where Todheide's data has two phases: six of six trials converged
    to |w - z| ~ 1e-16 with TPD ~ 1e-12.

    These march across the composition range instead, so that at least one
    starts inside the other phase's basin when a second phase exists. Built
    only after the primary panel has found nothing, so the cost falls on the
    few states that need it.
    """
    out = []
    gas_slots = [k for k in (1, 2) if present[k]]
    if not gas_slots or not present[0]:
        return out
    for k in gas_slots:
        nm = "CH4" if k == 1 else "CO2"
        for f in (0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95):
            w = np.zeros(3)
            w[k] = f
            w[0] = 1.0 - f
            out.append((f"wide {nm}={f:.2f}", "gas" if f >= 0.5 else "aq", w))
    if len(gas_slots) == 2:
        for f in (0.30, 0.60, 0.90):
            w = np.zeros(3)
            w[1] = w[2] = 0.5 * f
            w[0] = 1.0 - f
            out.append((f"wide CH4+CO2={f:.2f}", "gas" if f >= 0.5 else "aq", w))
    return out


def michelsen_tpd(
    z,
    mi,
    T,
    P,
    *,
    tpd_threshold=1e-3,
    accelerated=True,
    tol=1e-8,
    maxiter=80,
    early_tpd=-0.5,
    trivial_w_atol=5e-3,
):
    """Run the Michelsen TPD stability test of the multi-salt eCPA at one state.

    Runs ``tpd_ssi`` for a panel of up to 8 trial compositions (2 CO₂-rich,
    2 CH₄-rich, 2 aqueous, 2 from Wilson K-values) and returns the most
    negative TPD.  Trials for a gas absent from the feed (for example the
    CH₄-rich trials when ``z_CH4 = 0``) are skipped.

    A trial counts only when it converged and its composition differs from
    the feed by at least ``trivial_w_atol`` in some component.  If no trial
    of the standard panel counts, a wider panel that spans the composition
    range is run.  The search stops early when a trial gives
    ``tpd < early_tpd``.

    The reference vector is computed with the **gas-phase EoS** at the feed
    composition::

        d_i = ln(z_i) + lnφ_gas_i(z, T, P)

    This choice does not favour either gas.  It generalises the reference of
    Paper I (the CO₂-rich EoS at the feed composition), which applies only to
    the CO₂-H₂O system.  If no gas-phase root is found, the aqueous EoS is
    used at the feed composition, and then at pure brine.  Absent components
    (``z_i = 0``) are excluded from the reference and from the trials.

    Parameters
    ----------
    z : array_like, shape (3,)
        Feed mole fractions of the partitioning species (H2O, CH4, CO2),
        ions excluded; sum to 1.
    mi : array_like, shape (6,)
        Ion molalities (Na+, Cl-, K+, Ca2+, Mg2+, SO4 2-) [mol/kg water].
    T : float
        Temperature [K].
    P : float
        Pressure [bar].
    tpd_threshold : float
        Classification threshold: the feed is reported ``stable`` when
        ``tpd_min > -tpd_threshold``.  ``stability_flash`` passes
        ``-tpd_threshold`` of its own (default 0.005) unless the caller
        sets this value.
    accelerated : bool
        Passed to ``tpd_ssi``.
    tol : float
        Passed to ``tpd_ssi``.
    maxiter : int
        Passed to ``tpd_ssi``.
    early_tpd : float
        The search over trials stops as soon as a converged, non-trivial
        trial gives ``tpd < early_tpd``.
    trivial_w_atol : float
        A converged trial whose composition lies within this distance of
        the feed (in every component) is the trivial solution and is not
        counted.

    Returns
    -------
    StabilityResult
        ``stable``, ``tpd_min``, all trials, and the index of the trial that
        gave ``tpd_min``.

    Raises
    ------
    ValueError
        If ``z`` does not have 3 entries summing to 1, or ``mi`` does not
        have 6 entries.
    RuntimeError
        If the reference fugacity coefficients cannot be evaluated on the
        gas side or on the aqueous side.
    """
    z = np.asarray(z, dtype=float)
    if z.shape != (3,):
        raise ValueError("z must have 3 entries (H2O, CH4, CO2)")
    if abs(float(z.sum()) - 1.0) > 1e-8:
        raise ValueError(f"z must sum to 1; got {float(z.sum()):.6f}")
    mi = np.asarray(mi, dtype=float)
    if mi.shape != (6,):
        raise ValueError("mi must have 6 entries (Na, Cl, K, Ca, Mg, SO4)")

    present = _present_mask(z)
    P_Pa = float(P) * 1e5

    # ── Reference d-vector ────────────────────────────────────────────────
    # Use the gas-phase EoS at the partitioning-basis feed.  Renormalise z
    # over present components only (it already sums to 1 there, but be
    # defensive).
    z_ref = np.where(present, z, 0.0)
    s = float(z_ref.sum())
    if s <= 0:
        raise ValueError("z has no present components")
    z_ref = z_ref / s
    # Try several Z_init hints for the gas-phase reference: 1.0 (light gas
    # root) → 0.7 (mid) → B_mix·1.5 (heavier root, supercritical-dense feed).
    # This generalizes the prior project's "CO2-rich EoS" reference: when the
    # feed is itself liquid-like (case A: 323K/500bar with y_CO2 ≈ 0.4), the
    # gas-like Z_init=1 root doesn't exist and we need the dense root.
    from .parameters import R, b1, b2, b3, Mw as _Mw
    b_mix = b1*z_ref[0] + b2*z_ref[1] + b3*z_ref[2]
    B_mix = b_mix * P_Pa / R / float(T)
    # Include a B-scaled seed so the Newton finds the dense-gas root when
    # B_mix > 1 (P > ~500 bar for CO2 / CH4), where Zh_seed=1.0 falls inside
    # the unphysical Zh < B_mix region.
    Zh_B = max(B_mix * 1.2 + 0.3, 1.0)
    Z_hints = (1.0, Zh_B, max(Zh_B * 1.3, 2.0), 0.7, 0.5, max(B_mix*1.5, 0.4), 0.3)
    lnphi_ref = None
    last_exc = None
    for Zh in Z_hints:
        try:
            cand = lnphi_gas(tuple(z_ref), T, P_Pa, Z_init=float(Zh))
            if all(np.isfinite(c) for c in cand):
                lnphi_ref = cand
                break
        except Exception as exc:
            last_exc = exc
            continue
    if lnphi_ref is None:
        # Aqueous-side reference fallback (when gas-side has no valid root,
        # e.g. water-rich feeds at T/P where water is liquid).
        # Step 1: apply the ion closure to convert the partition-basis z_ref
        # to true aqueous mole fractions (so the sum including ions = 1).
        m_tot_ref = float(np.sum(mi))
        if m_tot_ref > 0:
            x1w_ref = z_ref[0] / (1.0 + _Mw * m_tot_ref * z_ref[0])
            sc_ref = 1.0 - _Mw * m_tot_ref * x1w_ref
        else:
            x1w_ref = z_ref[0]
            sc_ref = 1.0
        x_aq_ref = (x1w_ref, float(z_ref[1]) * sc_ref, float(z_ref[2]) * sc_ref)
        try:
            lnphi_ref = lnphi_aq(x_aq_ref, mi, T, P_Pa)
            fallback_msg = f" [WARN: gas-side ref failed ({last_exc}); used aq ref]"
        except Exception as exc2:
            # Step 2: pure-brine reference (no dissolved gas) — always has a
            # valid Newton solution regardless of z_CO2 / z_CH4 content.
            x1w_brine = 1.0 / (1.0 + _Mw * m_tot_ref) if m_tot_ref > 0 else 1.0
            try:
                lnphi_ref = lnphi_aq((x1w_brine, 0.0, 0.0), mi, T, P_Pa)
                fallback_msg = (f" [WARN: gas-side ref failed ({last_exc}); "
                                f"aq ref failed ({exc2}); used pure-brine ref]")
            except Exception as exc3:
                raise RuntimeError(
                    f"michelsen_tpd: reference lnφ failed on all sides "
                    f"(gas: {last_exc}; aq-closure: {exc2}; aq-brine: {exc3})"
                )
    else:
        fallback_msg = ""
    d_ref = np.array([_safe_log(z_ref[i]) + float(lnphi_ref[i]) for i in range(3)])

    # Absent components: d_i is irrelevant (masked out), but set to a finite
    # value so vector arithmetic doesn't propagate -inf.
    d_ref = np.where(present, d_ref, 0.0)

    # ── Trial schedule ────────────────────────────────────────────────────
    trial_specs = _build_trial_list(z, present, T, P, mi)

    results: List[TPDResult] = []
    tpd_min = 0.0
    best_idx = -1
    n_informative = 0        # trials that converged somewhere OTHER than z
    z_present = np.where(present, z, 0.0)

    def _run_panel(specs):
        """Evaluate a panel of trials, updating the running best."""
        nonlocal tpd_min, best_idx, n_informative
        for label, side, w0 in specs:
            i_t = len(results)
            # W0 = w0 (normalised) ⇒ initial sum_W = 1.
            W0 = np.where(present, w0, 0.0)
            # Floor present entries to avoid log(0) on first iteration.
            W0 = np.where(present, np.clip(W0, 1e-12, None), 0.0)
            res = tpd_ssi(
                d_ref, W0, T, P, mi, side,
                present=present,
                accelerated=accelerated, tol=tol, maxiter=maxiter,
                early_tpd=early_tpd, label=label,
            )
            results.append(res)
            # Trivial-fixed-point guard: SSI can spiral toward w ≈ z (the
            # trivial solution of the TPD stationarity condition), where the
            # residual 1 - sum_W picks up a noise-level negative value that's
            # not a real instability signal. Reject trials whose converged
            # composition is within trivial_w_atol (component-wise) of the feed.
            w_arr = np.asarray(res.w, dtype=float)
            trivial = bool(np.max(np.abs(w_arr - z_present)) < trivial_w_atol)
            # A trial that did not converge has not found a stationary point,
            # so its tpd is wherever the iteration happened to stop. It must
            # not be allowed to set tpd_min or to count as informative.
            if not getattr(res, "converged", True):
                continue
            if (not trivial) and np.isfinite(res.tpd):
                n_informative += 1
                if res.tpd < tpd_min:
                    tpd_min = float(res.tpd)
                    best_idx = i_t
                # Early-out only when a non-trivial trial saw strongly
                # negative TPD (cheaper to skip the remaining trials).
                if res.tpd < early_tpd:
                    return True
        return False

    _run_panel(trial_specs)

    # A panel in which EVERY trial collapsed to the trivial solution, or
    # failed, has not established stability -- it has established nothing. The
    # two outcomes used to be reported identically, as "stable". Widen the
    # search before concluding anything.
    escalated = False
    if n_informative == 0:
        wide = _build_wide_trial_list(z, present)
        if wide:
            escalated = True
            _run_panel(wide)

    if best_idx < 0:
        stable = True
        if n_informative == 0:
            # Still nothing informative. Say so: this is an unproven stable,
            # not a demonstrated one.
            msg = ("INCONCLUSIVE: every trial collapsed to the trivial "
                   "solution or failed" +
                   (" (wide panel also exhausted)" if escalated else "") +
                   fallback_msg)
        else:
            msg = "all trials gave tpd >= 0 (stable)" + fallback_msg
    else:
        stable = tpd_min > -float(tpd_threshold)
        # Report the winning trial's stationarity residual alongside its tpd:
        # every flash attempt in the stability_flash log carries its max|g|,
        # and the trial that set tpd_min is read from the same log.
        msg = (f"tpd_min={tpd_min:.3e} from trial '{results[best_idx].label}' "
               f"({results[best_idx].side}, "
               f"max|g|={results[best_idx].max_residual:.2e})" + fallback_msg)

    return StabilityResult(
        stable=bool(stable),
        tpd_min=float(tpd_min),
        trials=results,
        best_trial_idx=int(best_idx) if best_idx >= 0 else -1,
        message=msg,
        d_ref=tuple(float(x) for x in d_ref),
        z=tuple(float(x) for x in z),
    )


# ---------------------------------------------------------------------------
# K-init from best trial
# ---------------------------------------------------------------------------

def make_k_init_from_trial(trial_result: TPDResult, z, T, P):
    """Build initial K-values for ``flash_kv`` from a stability trial.

    The same approach is used in Paper I
    (``ecpa.stability._make_elv_guess_from_trial``).

    The converged trial composition ``w`` is taken as one phase (gas if
    ``side == "gas"``, aqueous if ``side == "aq"``), and the feed ``z`` as
    an estimate of the other phase:

    * gas trial: ``K_i = w_i / z_i``;
    * aqueous trial: ``K_i = z_i / w_i``.

    K-values are clipped to ``[1e-6, 1e6]``.  If ``K_H2O > 2``, which
    happens when an aqueous trial has almost no water, the Wilson value is
    used for H2O instead.

    Parameters
    ----------
    trial_result : TPDResult
        The trial, normally ``stability.trials[stability.best_trial_idx]``.
    z : array_like, shape (3,)
        Feed mole fractions of the partitioning species (H2O, CH4, CO2);
        sum to 1.
    T : float
        Temperature [K].
    P : float
        Pressure [bar].

    Returns
    -------
    ndarray, shape (3,)
        K-values ``y_i / x_i`` (H2O, CH4, CO2).  NaN for components absent
        from the feed.
    """
    z = np.asarray(z, dtype=float)
    present = _present_mask(z)
    w = np.asarray(trial_result.w, dtype=float)
    K_wil = wilson_k(T, P)

    K = np.where(present, K_wil, np.nan)  # default to Wilson; will overwrite

    eps = 1e-12
    for i in range(3):
        if not present[i]:
            continue
        if trial_result.side == "gas":
            denom = max(float(z[i]), eps)
            num = max(float(w[i]), eps)
            K[i] = num / denom
        else:  # "aq"
            num = max(float(z[i]), eps)
            denom = max(float(w[i]), eps)
            K[i] = num / denom

    K = np.where(present, np.clip(K, 1e-6, 1e6), np.nan)

    # Physical guard: when a degenerate aq-side trial has near-zero w_H2O,
    # the formula K_H2O = z_H2O / w_H2O blows up.  K_H2O > 2 is almost
    # always unphysical for our brine systems (T ≤ 623 K): it would force
    # water into the gas phase and cause immediate SSI divergence.  Cap at
    # Wilson K when the trial K_H2O is implausibly large.
    if present[0] and np.isfinite(K[0]) and K[0] > 2.0:
        K[0] = K_wil[0]

    return K


# ---------------------------------------------------------------------------
# Combined stability + flash driver
# ---------------------------------------------------------------------------

def _is_real_split(fr, eps_beta=1e-4):
    """A flash result counts as a true two-phase split only when β is
    bounded away from 0 and 1.  β at the boundary means Rachford-Rice
    found no mass-balance-consistent split given the converged K, so the
    feed is effectively single-phase even if the underlying TPD test was
    negative (which can happen at near-dew/near-bubble points where the
    K-values implied by lnφ_aq/lnφ_gas all sit on the same side of 1).
    """
    if fr is None or not getattr(fr, "converged", False):
        return False
    b = float(getattr(fr, "beta", float("nan")))
    if not np.isfinite(b):
        return False
    return (eps_beta < b < 1.0 - eps_beta)


def _is_phase_swapped(fr):
    """Detect a flash result whose phase labels are inverted: the kernel
    converged to a valid split but the "aqueous"-side EoS solution lives
    on the gas-like branch (low density, low permittivity, weak
    Wertheim association) and the "gas"-side solution lives on the
    aqueous-like branch.  Happens at intermediate T/P near the
    critical region of the lighter component (~T=448K, P~120 bar for
    CO2-rich feeds) when the inner-Newton ladder accepts the
    light-density root for the aqueous side.

    The strongest, model-agnostic signal is composition: a real
    aqueous phase carries the bulk of the water, a real gas phase the
    bulk of the partitioning gases.  Use that as the primary check,
    then fall back to Z and χ when composition alone is ambiguous.
    """
    if fr is None or not getattr(fr, "converged", False):
        return False
    x_aq = getattr(fr, "x_aq", None)
    y_gas = getattr(fr, "y_gas", None)
    if x_aq is None or y_gas is None:
        return False
    x_w_aq = float(x_aq[0]); y_w_gas = float(y_gas[0])
    if not (np.isfinite(x_w_aq) and np.isfinite(y_w_gas)):
        return False
    # Primary composition criterion: if the "gas" phase carries far more
    # water than the "aqueous" phase, labels are swapped.  A 2x ratio
    # with y_w_gas > 0.5 catches the dominant CO2-ternary failure mode
    # at T~448K P~120 bar (the aqueous side gets a 10% water gas-like
    # root while the gas side gets a 99% water aqueous-like root).
    if (y_w_gas > 0.5) and (y_w_gas > 2.0 * x_w_aq):
        return True
    # Secondary Z criterion when both compressibilities are finite.
    Z_aq = float(getattr(fr, "Z_aq", float("nan")))
    Z_gas = float(getattr(fr, "Z_gas", float("nan")))
    if np.isfinite(Z_aq) and np.isfinite(Z_gas) and (Z_aq > Z_gas + 0.1):
        return True
    # Tertiary χ_H2O criterion: water in a real aqueous phase is
    # strongly hydrogen-bonded (χ small).  A "weakly associated"
    # aqueous side with appreciable water content is a swap signal.
    chi_aq = float(getattr(fr, "chi_H2O_aq", float("nan")))
    if np.isfinite(chi_aq) and x_w_aq > 0.1 and chi_aq > 0.8:
        return True
    return False


def _swap_phases(fr):
    """Return a copy of ``fr`` with aqueous/gas labels and derived
    quantities exchanged.  Used to relabel results flagged by
    ``_is_phase_swapped``.

    Mole-fraction columns and EoS-side state swap directly; K-values
    invert (K_i = y/x ⇒ swapped K_i = x/y = 1/K_i); β complements to
    1 - β (the new gas-phase mass fraction); fugacities and ion
    molalities swap correspondingly.  Note that mi_aq under the
    "as-aqueous" convention is identical for both sides so it is left
    unchanged.
    """
    if fr is None:
        return fr
    from dataclasses import replace
    K_new = tuple(
        (1.0 / k) if (k is not None and np.isfinite(k) and k != 0.0)
        else float("nan")
        for k in fr.K
    )
    beta_new = 1.0 - float(fr.beta) if np.isfinite(fr.beta) else fr.beta
    # x1w on the new aqueous side is the previous gas-side water fraction.
    x1w_new = float(fr.y_gas[0]) if getattr(fr, "y_gas", None) is not None else fr.x1w
    return replace(
        fr,
        K=K_new,
        x_aq=tuple(float(y) for y in fr.y_gas),
        y_gas=tuple(float(x) for x in fr.x_aq),
        beta=beta_new,
        x1w=x1w_new,
        Z_aq=float(fr.Z_gas), Z_gas=float(fr.Z_aq),
        chi_H2O_aq=float(fr.chi_H2O_gas), chi_H2O_gas=float(fr.chi_H2O_aq),
        fugacities_aq=tuple(float(f) for f in fr.fugacities_gas),
        fugacities_gas=tuple(float(f) for f in fr.fugacities_aq),
        message=(fr.message or "") + " [phases relabeled: light↔dense]",
    )


def stability_flash(
    T, P, z, mi,
    *,
    tpd_threshold=-0.005,
    tpd_soft_threshold=-0.1,
    stability_kwargs=None,
    **flash_kwargs,
):
    """Run the stability test and, if the feed is unstable, the two-phase flash.

    This is the hierarchical algorithm of Paper I
    (``ecpa.stability.ecpa_stability_flash``), extended to several salts and
    two gases.  The steps are:

    1. Run ``michelsen_tpd``.
    2. If ``tpd_min > tpd_threshold``, return single-phase (route 0).
    3. Run ``flash_kv`` from a sequence of initial K-values, in this order:
       from the best trial (route 1), Wilson (route 2), water-biased
       (route 3), Wilson without acceleration or Newton steps (route 4),
       and from the best aqueous and gas trials together (route 5).  The
       first result that converges with a phase fraction strictly between
       0 and 1, and whose phases are not interchanged, is returned.
    4. If ``tpd_min < -1e-3`` and the best trial is aqueous, run
       ``dew_point_flash`` (route 6).
    5. If a flash converged but its phase fraction is at 0 or 1, return
       single-phase (route 7).
    6. If ``tpd_min > tpd_soft_threshold``, return single-phase (route 8).
    7. Otherwise return a failure (route 9).

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
    tpd_threshold : float
        Single-phase threshold: ``tpd_min > tpd_threshold`` gives a
        single-phase result without a flash.  The value was chosen for the
        multi-salt system.  It is closer to zero than the -0.02 of Jex et
        al. (2024), because near critical points the eCPA can give slightly
        negative TPD values for feeds that are stable.
    tpd_soft_threshold : float
        Threshold for step 6.  When every flash attempt has failed and
        ``tpd_min`` is above this value, the feed is reported single-phase
        (``route == 8``, ``flash_result`` None).  This is a heuristic, not a
        second test: no further calculation is made, and the TPD test had
        found the feed unstable.  To separate this case from a single-phase
        result of the TPD test, check ``route == 8``; ``phases`` and
        ``stable`` are the same as for route 0.
    stability_kwargs : dict, optional
        Keyword arguments passed to ``michelsen_tpd``.
    **flash_kwargs
        Keyword arguments passed to ``flash_kv`` (for example
        ``salinity_convention``).

    Returns
    -------
    StabilityFlashResult
        Number of phases, the flash result when two-phase, the stability
        result, the ``route`` code of the step that produced the answer, and
        timings.  Phase compositions are in ``flash_result.x_aq`` and
        ``flash_result.y_gas``.
    """
    stab_kw = dict(stability_kwargs or {})
    # Pass the same hard threshold into the stability classifier so its
    # `.stable` flag is consistent with our combined-driver convention.
    if "tpd_threshold" not in stab_kw:
        stab_kw["tpd_threshold"] = float(-tpd_threshold)

    # ── Split-timing bookkeeping ──────────────────────────────────────
    # t_start: total-call wall-clock reference; overhead outside the
    # stab/flash kernels (RR closure, phase-swap/β-degenerate checks,
    # dispatch) is measured as t_total - t_stab - t_flash by the caller.
    # t_flash_ms accumulates across every flash_kv attempt actually run,
    # including retries that end up rejected as β-degenerate or
    # phase-swapped — reservoir simulators pay this cost too, so it
    # belongs in the reported flash budget.
    t_start = time.perf_counter()
    _t0 = time.perf_counter()
    stab = michelsen_tpd(z, mi, T, P, **stab_kw)
    t_stab_ms = (time.perf_counter() - _t0) * 1000.0
    t_flash_ms = 0.0

    attempts: List[str] = [f"stability: {stab.message}"]

    # ── Hard guard: tpd above the hard threshold ⇒ single-phase ────────
    if stab.tpd_min > tpd_threshold:
        return StabilityFlashResult(
            phases=1,
            tpd_min=float(stab.tpd_min),
            stable=True,
            flash_result=None,
            stability=stab,
            attempts=attempts + [
                f"single-phase: tpd_min={stab.tpd_min:.3e} > {tpd_threshold}"
            ],
            route=0,
            message=f"single-phase (tpd_min={stab.tpd_min:.3e})",
            T=float(T), P=float(P),
            z=tuple(float(zi) for zi in z),
            mi=tuple(float(m) for m in mi),
            t_stab_ms=float(t_stab_ms),
            t_flash_ms=float(t_flash_ms),
            t_total_ms=float((time.perf_counter() - t_start) * 1000.0),
        )

    # Any converged-but-β-degenerate flash result we see along the way;
    # used by the soft fallback to detect "flash had nothing to split
    # toward, even with TPD strongly negative".
    last_degenerate_fr = None

    # ── Attempt 1: K_init from best trial ─────────────────────────────
    K_trial = None
    if stab.best_trial_idx >= 0:
        try:
            K_trial = make_k_init_from_trial(
                stab.trials[stab.best_trial_idx], z, T, P)
            _t0 = time.perf_counter()
            fr = flash_kv(T, P, z, mi, K_init=K_trial, **flash_kwargs)
            t_flash_ms += (time.perf_counter() - _t0) * 1000.0
            attempts.append(
                f"flash_kv(K_trial): converged={fr.converged}, "
                f"beta={fr.beta:.4g}, "
                f"SSI={fr.n_iter_ssi}, Newton={fr.n_iter_newton}, "
                f"max|g|={fr.max_residual:.2e}"
            )
            if _is_real_split(fr):
                if _is_phase_swapped(fr):
                    attempts.append(
                        "  flash_kv(K_trial): phases swapped "
                        f"(Z_aq={fr.Z_aq:.3f}, Z_gas={fr.Z_gas:.3f}); skipping"
                    )
                else:
                    return StabilityFlashResult(
                        phases=2,
                        tpd_min=float(stab.tpd_min),
                        stable=False,
                        flash_result=fr,
                        stability=stab,
                        attempts=attempts,
                        route=1,
                        message="two-phase (trial K-init)",
                        T=float(T), P=float(P),
                        z=tuple(float(zi) for zi in z),
                        mi=tuple(float(m) for m in mi),
                        t_stab_ms=float(t_stab_ms),
                        t_flash_ms=float(t_flash_ms),
                        t_total_ms=float((time.perf_counter() - t_start) * 1000.0),
                    )
            if fr is not None and fr.converged:
                last_degenerate_fr = fr
        except Exception as exc:
            attempts.append(f"flash_kv(K_trial): exception {exc}")

    # ── Attempt 2: Wilson K_init ──────────────────────────────────────
    try:
        K_wil = wilson_k(T, P)
        _t0 = time.perf_counter()
        fr2 = flash_kv(T, P, z, mi, K_init=K_wil, **flash_kwargs)
        t_flash_ms += (time.perf_counter() - _t0) * 1000.0
        attempts.append(
            f"flash_kv(Wilson): converged={fr2.converged}, "
            f"beta={fr2.beta:.4g}, "
            f"SSI={fr2.n_iter_ssi}, Newton={fr2.n_iter_newton}, "
            f"max|g|={fr2.max_residual:.2e}"
        )
        if _is_real_split(fr2):
            if _is_phase_swapped(fr2):
                attempts.append(
                    "  flash_kv(Wilson): phases swapped "
                    f"(Z_aq={fr2.Z_aq:.3f}, Z_gas={fr2.Z_gas:.3f}); skipping"
                )
            else:
                return StabilityFlashResult(
                    phases=2,
                    tpd_min=float(stab.tpd_min),
                    stable=False,
                    flash_result=fr2,
                    stability=stab,
                    attempts=attempts,
                    route=2,
                    message="two-phase (Wilson K-init)",
                    T=float(T), P=float(P),
                    z=tuple(float(zi) for zi in z),
                    mi=tuple(float(m) for m in mi),
                    t_stab_ms=float(t_stab_ms),
                    t_flash_ms=float(t_flash_ms),
                    t_total_ms=float((time.perf_counter() - t_start) * 1000.0),
                )
        if fr2 is not None and fr2.converged:
            last_degenerate_fr = fr2
        last_fr = fr2
    except Exception as exc:
        attempts.append(f"flash_kv(Wilson): exception {exc}")
        last_fr = None

    # ── Attempt 3: water-biased K (forced aqueous regime) ────────────
    # When the Wilson / trial K-values put CH4 or CO2 into the "aqueous"
    # phase (K < 1 at extreme P), both earlier attempts produce degenerate
    # aqueous compositions and fail.  A water-biased K init forces water
    # to the aqueous phase and gases to the gas phase regardless of Wilson.
    try:
        K_wb = np.where(np.asarray(z) > 0, wilson_k(T, P), np.nan)
        # Force H2O strongly aqueous; gases strongly into gas phase
        if np.asarray(z)[0] > 0:
            K_wb[0] = min(K_wb[0], 1e-3)
        for i in (1, 2):
            if np.asarray(z)[i] > 0:
                K_wb[i] = max(K_wb[i], 5.0)   # moderate (not extreme) preference
        _t0 = time.perf_counter()
        fr3 = flash_kv(T, P, z, mi, K_init=K_wb, **flash_kwargs)
        t_flash_ms += (time.perf_counter() - _t0) * 1000.0
        attempts.append(
            f"flash_kv(water-biased): converged={fr3.converged}, "
            f"beta={fr3.beta:.4g}, "
            f"SSI={fr3.n_iter_ssi}, Newton={fr3.n_iter_newton}, "
            f"max|g|={fr3.max_residual:.2e}"
        )
        if _is_real_split(fr3):
            if _is_phase_swapped(fr3):
                attempts.append(
                    "  flash_kv(water-biased): phases swapped "
                    f"(Z_aq={fr3.Z_aq:.3f}, Z_gas={fr3.Z_gas:.3f}); skipping"
                )
            else:
                return StabilityFlashResult(
                    phases=2,
                    tpd_min=float(stab.tpd_min),
                    stable=False,
                    flash_result=fr3,
                    stability=stab,
                    attempts=attempts,
                    route=3,
                    message="two-phase (water-biased K-init)",
                    T=float(T), P=float(P),
                    z=tuple(float(zi) for zi in z),
                    mi=tuple(float(m) for m in mi),
                    t_stab_ms=float(t_stab_ms),
                    t_flash_ms=float(t_flash_ms),
                    t_total_ms=float((time.perf_counter() - t_start) * 1000.0),
                )
        if fr3 is not None and fr3.converged:
            last_degenerate_fr = fr3
        if last_fr is None or (fr3.n_iter_ssi > (last_fr.n_iter_ssi if last_fr else 0)):
            last_fr = fr3
    except Exception as exc:
        attempts.append(f"flash_kv(water-biased): exception {exc}")

    # ── Attempt 4: pure SSI, no acceleration, no Newton polish ────────
    # Iter 6 Fix 2.  When Attempts 1-3 all fail, the failure modes seen
    # in standard-sweep v5 are dominated by inner-Newton pathologies in
    # the Jex-accelerated SSI: chi1w drifts negative or to NaN, the
    # Newton polish diverges, K-resets get stuck oscillating.  Pure SSI
    # at maxiter=200 sidesteps every one of those: no Jex (cannot
    # overshoot), no Newton polish (cannot diverge on a bad J), no
    # degenerate-aqueous K-reset cycle.  This attempt can never
    # introduce a regression -- it only fires when nothing else worked.
    try:
        fkw_pure = {k: v for k, v in flash_kwargs.items()
                    if k not in ("accelerated", "use_newton", "maxiter")}
        K_pure = wilson_k(T, P)        # local re-derive: K_wil scope guarded
        _t0 = time.perf_counter()
        fr4 = flash_kv(
            T, P, z, mi, K_init=K_pure,
            accelerated=False, use_newton=False, maxiter=200,
            **fkw_pure,
        )
        t_flash_ms += (time.perf_counter() - _t0) * 1000.0
        attempts.append(
            f"flash_kv(pure-SSI): converged={fr4.converged}, "
            f"beta={fr4.beta:.4g}, "
            f"SSI={fr4.n_iter_ssi}, Newton={fr4.n_iter_newton}, "
            f"max|g|={fr4.max_residual:.2e}"
        )
        if _is_real_split(fr4):
            if _is_phase_swapped(fr4):
                attempts.append(
                    "  flash_kv(pure-SSI): phases swapped "
                    f"(Z_aq={fr4.Z_aq:.3f}, Z_gas={fr4.Z_gas:.3f}); skipping"
                )
            else:
                return StabilityFlashResult(
                    phases=2,
                    tpd_min=float(stab.tpd_min),
                    stable=False,
                    flash_result=fr4,
                    stability=stab,
                    attempts=attempts,
                    route=4,
                    message="two-phase (pure-SSI fallback)",
                    T=float(T), P=float(P),
                    z=tuple(float(zi) for zi in z),
                    mi=tuple(float(m) for m in mi),
                    t_stab_ms=float(t_stab_ms),
                    t_flash_ms=float(t_flash_ms),
                    t_total_ms=float((time.perf_counter() - t_start) * 1000.0),
                )
        if fr4 is not None and fr4.converged:
            last_degenerate_fr = fr4
        if last_fr is None or (fr4.n_iter_ssi > (last_fr.n_iter_ssi if last_fr else 0)):
            last_fr = fr4
    except Exception as exc:
        attempts.append(f"flash_kv(pure-SSI): exception {exc}")

    # ── Attempt 4b: dual-trial K-init ─────────────────────────────────
    # Near a miscibility locus the feed lies between two compositionally
    # similar phases, and every K built against the FEED (best-trial K,
    # Wilson K, water-biased K) is a poor stand-in for the phase ratio:
    # the SSI walks x_H2O to zero and dies in degenerate K-resets (the
    # single dense-sweep failure at 516 K / 329 bar, z_CO2 = 0.25).  But
    # the stability scan usually holds the answer already: when BOTH an
    # aq-side and a gas-side trial converged to negative TPD, their
    # compositions approximate the two coexisting phases, and
    # K_i = w_i^gas / w_i^aq is a far sharper seed than any feed-based
    # ratio.  Runs only after Attempts 1-4 failed, so every previously
    # converging state is bit-identical by construction.
    try:
        best_gas = best_aq = None
        for tr in stab.trials:
            if not (tr.converged and np.isfinite(tr.tpd) and tr.tpd < 0):
                continue
            if tr.side == "gas" and (best_gas is None or tr.tpd < best_gas.tpd):
                best_gas = tr
            if tr.side == "aq" and (best_aq is None or tr.tpd < best_aq.tpd):
                best_aq = tr
        if best_gas is not None and best_aq is not None:
            eps = 1e-12
            _pres = _present_mask(z)
            w_g = np.asarray(best_gas.w, dtype=float)
            w_a = np.asarray(best_aq.w, dtype=float)
            K_dual = np.where(_pres,
                              np.clip(w_g, eps, None) / np.clip(w_a, eps, None),
                              np.nan)
            K_dual = np.where(_pres, np.clip(K_dual, 1e-6, 1e6), np.nan)
            _t0 = time.perf_counter()
            fr4b = flash_kv(T, P, z, mi, K_init=K_dual, **flash_kwargs)
            t_flash_ms += (time.perf_counter() - _t0) * 1000.0
            attempts.append(
                f"flash_kv(dual-trial K): converged={fr4b.converged}, "
                f"beta={fr4b.beta:.4g}, "
                f"SSI={fr4b.n_iter_ssi}, Newton={fr4b.n_iter_newton}, "
                f"max|g|={fr4b.max_residual:.2e}"
            )
            if _is_real_split(fr4b):
                if _is_phase_swapped(fr4b):
                    attempts.append(
                        "  flash_kv(dual-trial K): phases swapped "
                        f"(Z_aq={fr4b.Z_aq:.3f}, Z_gas={fr4b.Z_gas:.3f}); "
                        "skipping"
                    )
                else:
                    return StabilityFlashResult(
                        phases=2,
                        tpd_min=float(stab.tpd_min),
                        stable=False,
                        flash_result=fr4b,
                        stability=stab,
                        attempts=attempts,
                        route=5,
                        message="two-phase (dual-trial K-init)",
                        T=float(T), P=float(P),
                        z=tuple(float(zi) for zi in z),
                        mi=tuple(float(m) for m in mi),
                        t_stab_ms=float(t_stab_ms),
                        t_flash_ms=float(t_flash_ms),
                        t_total_ms=float(
                            (time.perf_counter() - t_start) * 1000.0),
                    )
            if fr4b is not None and fr4b.converged:
                last_degenerate_fr = fr4b
        else:
            attempts.append(
                "flash_kv(dual-trial K): skipped (need converged negative "
                "trials on both sides)")
    except Exception as exc:
        attempts.append(f"flash_kv(dual-trial K): exception {exc}")

    # ── Attempt 5: dew-point rescue flash ─────────────────────────────
    # When TPD is strongly negative (feed thermodynamically unstable)
    # but every K-value SSI attempt above collapsed to the trivial
    # K = 1 fixed point, we are almost always in a near-dew-line
    # regime: the aqueous phase composition is essentially the TPD
    # aqueous-trial (near-pure water) and the SSI just cannot escape
    # its trivial attractor from any Wilson-scale K-init.  Fix
    # x_aq at the TPD aqueous trial's converged composition and
    # solve for β via Brent on the water-fugacity balance.  This
    # single scalar solve avoids the K-space attractor and gives a
    # physical β in the interior of (0, 1).
    if stab.tpd_min < -1e-3 and stab.best_trial_idx >= 0:
        try:
            best_trial = stab.trials[stab.best_trial_idx]
            if best_trial.side == "aq" and best_trial.w[0] > 0.5:
                _t0 = time.perf_counter()
                x_aq_TPD = np.asarray(best_trial.w, dtype=float)
                x_aq_TPD = x_aq_TPD / x_aq_TPD.sum()
                from .flash import dew_point_flash
                fr5 = dew_point_flash(
                    T, P, z, mi, x_aq_TPD,
                    salinity_convention=flash_kwargs.get(
                        "salinity_convention", "as-aqueous"),
                )
                t_flash_ms += (time.perf_counter() - _t0) * 1000.0
                if fr5 is not None and _is_real_split(fr5):
                    attempts.append(
                        f"dew-point rescue: β={fr5.beta:.4g}, water fugacity "
                        f"balance converged")
                    return StabilityFlashResult(
                        phases=2,
                        tpd_min=float(stab.tpd_min),
                        stable=False,
                        flash_result=fr5,
                        stability=stab,
                        attempts=attempts,
                        route=6,
                        message="two-phase (dew-point rescue)",
                        T=float(T), P=float(P),
                        z=tuple(float(zi) for zi in z),
                        mi=tuple(float(m) for m in mi),
                        t_stab_ms=float(t_stab_ms),
                        t_flash_ms=float(t_flash_ms),
                        t_total_ms=float((time.perf_counter() - t_start) * 1000.0),
                    )
                elif fr5 is None:
                    attempts.append("dew-point rescue: no β bracket found")
                else:
                    attempts.append(
                        f"dew-point rescue: β={fr5.beta:.4g} still at boundary; "
                        "skipping")
        except Exception as exc:
            attempts.append(f"dew-point rescue: exception {exc}")

    # ── β-degenerate fallback ─────────────────────────────────────────
    # All flash attempts that DID converge settled with β at 0 or 1
    # (i.e., Rachford-Rice found no mass-balance-consistent split given
    # the converged K).  Even when TPD is strongly negative, this is the
    # honest classification: numerically the feed has no resolvable
    # second phase, so report phase=1 and surface the degenerate fr for
    # downstream diagnostics.
    if last_degenerate_fr is not None:
        b_deg = float(last_degenerate_fr.beta)
        return StabilityFlashResult(
            phases=1,
            tpd_min=float(stab.tpd_min),
            stable=True,
            flash_result=last_degenerate_fr,
            stability=stab,
            attempts=attempts + [
                f"flash β-degenerate (β={b_deg:.4g}); reclassified single-phase "
                f"(tpd_min={stab.tpd_min:.3e})"
            ],
            route=7,
            message=(f"single-phase (β-degenerate flash, β={b_deg:.4g}; "
                     f"tpd_min={stab.tpd_min:.3e})"),
            T=float(T), P=float(P),
            z=tuple(float(zi) for zi in z),
            mi=tuple(float(m) for m in mi),
            t_stab_ms=float(t_stab_ms),
            t_flash_ms=float(t_flash_ms),
            t_total_ms=float((time.perf_counter() - t_start) * 1000.0),
        )

    # ── Soft fallback ─────────────────────────────────────────────────
    if stab.tpd_min > tpd_soft_threshold:
        return StabilityFlashResult(
            phases=1,
            tpd_min=float(stab.tpd_min),
            stable=True,
            flash_result=None,
            stability=stab,
            attempts=attempts + [
                f"soft-fallback single-phase: tpd_min={stab.tpd_min:.3e} > "
                f"{tpd_soft_threshold} and both flash attempts failed"
            ],
            route=8,
            message=f"single-phase (soft fallback; tpd_min={stab.tpd_min:.3e})",
            T=float(T), P=float(P),
            z=tuple(float(zi) for zi in z),
            mi=tuple(float(m) for m in mi),
            t_stab_ms=float(t_stab_ms),
            t_flash_ms=float(t_flash_ms),
            t_total_ms=float((time.perf_counter() - t_start) * 1000.0),
        )

    # ── Hard failure ──────────────────────────────────────────────────
    return StabilityFlashResult(
        phases=2,
        tpd_min=float(stab.tpd_min),
        stable=False,
        flash_result=last_fr,
        stability=stab,
        attempts=attempts,
        route=9,
        message=(f"flash did not converge "
                 f"(tpd_min={stab.tpd_min:.3e}, both K-init attempts failed)"),
        T=float(T), P=float(P),
        z=tuple(float(zi) for zi in z),
        mi=tuple(float(m) for m in mi),
        t_stab_ms=float(t_stab_ms),
        t_flash_ms=float(t_flash_ms),
        t_total_ms=float((time.perf_counter() - t_start) * 1000.0),
    )
