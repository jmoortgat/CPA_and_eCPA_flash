"""Saturation pressure from water isofugacity, on a verified bracket.

The saturation pressure of brine (or pure water) is the root in ``P`` of

    f(ln P) = [ln x_w + lnphi_w^aq(T, P)] - lnphi_w^gas(T, P) = 0,

that is, equal water fugacity in the brine and in pure steam. A plain
``scipy.optimize.brentq`` call on the fixed bracket ``[0.50 p0, 1.10 p0]``
around the IAPWS pure-water value ``p0`` is not reliable below about 300 K,
for two reasons.

1.  The root is not always inside the bracket.

2.  At low ``T`` and low ``P`` the aqueous Newton iteration in
    ``ms_ecpa.lnphi.lnphi_aq`` can stop just above its ``1e-10`` tolerance.
    ``lnphi_aq`` then uses its next initial guess and can converge to the
    vapour root of the aqueous mixture instead of the liquid root. At such a
    point both sides of the isofugacity balance describe the same phase, so
    ``|f|`` is between 1e-19 and 1e-12 with an arbitrary sign (10 of 46
    log-spaced samples at 283.15 K and 0 mol/kg). If such a point is a bracket
    end, ``brentq`` converges to it.

On the liquid branch the residual is smooth and monotone, with
``|df/d ln P|`` of order one, and has exactly one root, including at 283.15
and 293.15 K.

This module therefore:

* verifies the bracket (ends evaluable, opposite signs, exactly one sign
  change among the valid interior samples) before calling ``brentq``. Where
  the default bracket passes, the root is identical to that of the plain
  ``brentq`` call;
* discards samples with ``|f| <= _DEGENERATE`` and never accepts one as a
  root;
* raises ``PsatBracketError`` when no single root can be established.

The Fortran routine ``psat_brine`` (``fortran/src/ecpa_colligative.f90``)
uses the fixed bracket without these checks. Its reference values cover
323.15 to 573.15 K, where the default bracket passes and both give the same
root. Below about 300 K, use this module.
"""
from __future__ import annotations

import numpy as np

__all__ = ["PsatBracketError", "solve_psat_isofugacity"]


class PsatBracketError(RuntimeError):
    """No single, clean isofugacity root could be established."""


#: Residual magnitude at or below which a sample is treated as CONTAMINATED
#: (the aqueous side has collapsed onto the vapour root) rather than as a
#: zero.  The measured contaminated values span 1e-19 to 2.6e-12; the
#: physical residual has |df/d ln P| ~ 0.9-1.0, so a genuine sample only
#: gets this close to zero within 1e-11 of the root in ln P -- two orders
#: inside the refinement stop width below.
_DEGENERATE = 1e-11

#: Stop width for the adaptive refinement, in ln P (i.e. relative in P).
_REFINE_WIDTH = 1e-9

#: Half-step used by the final slope check on a candidate root.
_ROOT_CHECK_H = 1e-6

#: Default bracket: the historical one.  Physically motivated for a
#: water-rich brine, whose saturation pressure is a fraction of the pure
#: water value at the same T.  Used FIRST, and only when it verifies.
DEFAULT_LO_FAC = 0.50
DEFAULT_HI_FAC = 1.10

#: Adaptive scan window, used when the default bracket does not verify.
#: The lower edge admits P/P0 down to ~0.35 (well past 6 m NaCl, which
#: sits near 0.72-0.78); the upper edge covers the cold end, where the
#: model's pure-water root sits slightly ABOVE the IAPWS anchor.
SCAN_LO_FAC = 0.35
SCAN_HI_FAC = 1.30

#: One widening, tried if the first scan window holds no clean root.
WIDE_LO_FAC = 0.12
WIDE_HI_FAC = 1.80


def _probe(f, lnP):
    """Evaluate the residual and classify the sample.

    Returns ``(value, valid)``.  ``valid`` is False for a non-finite
    residual and for a contaminated one (see ``_DEGENERATE``).
    """
    try:
        v = float(f(lnP))
    except Exception:
        return np.nan, False
    if not np.isfinite(v):
        return np.nan, False
    return v, abs(v) > _DEGENERATE


def _sign_changes(samples):
    """Sign changes between CONSECUTIVE VALID samples.

    ``samples`` is a list of ``(lnP, value, valid)``.  Invalid samples are
    skipped rather than counted as sign flips, so an isolated contaminated
    point does not manufacture two spurious roots out of one real one.
    Returns a list of ``(lnP_lo, f_lo, lnP_hi, f_hi)``.
    """
    valid = [(x, v) for x, v, ok in samples if ok]
    out = []
    for (xa, va), (xb, vb) in zip(valid[:-1], valid[1:]):
        if (va > 0.0) != (vb > 0.0):
            out.append((xa, va, xb, vb))
    return out


def _is_true_crossing(f, lnP, h=_ROOT_CHECK_H):
    """Confirm a candidate root by stepping off it on both sides.

    On the physical branch the residual has ``|df/d ln P| ~ 1``, so at
    ``lnP -/+ h`` it must be evaluable, of magnitude ~``h`` (far above the
    contamination floor), and of OPPOSITE sign either side.  A bracket edge
    fails this test, and so does a contaminated sample (whose neighbours
    are both on the same side of zero, at O(0.1)).  This is the check that
    makes it structurally impossible to return the artefacts this module
    exists to reject.
    """
    fm, ok_m = _probe(f, lnP - h)
    fp, ok_p = _probe(f, lnP + h)
    if not (ok_m and ok_p):
        return False
    return (fm > 0.0) != (fp > 0.0)


def _bisect_clean(f, lo, flo, hi, fhi, xtol, rtol, maxiter=200):
    """Bisection that refuses to land on a contaminated sample.

    When the midpoint is contaminated the step is retried at a few nearby
    interior fractions; the residual is continuous on the physical branch,
    so any clean neighbour carries the same sign information.  If no clean
    sample can be found the current bracket is returned as-is, and the
    caller decides whether it is tight enough to be an answer.
    """
    for _ in range(maxiter):
        width = hi - lo
        if width <= max(_REFINE_WIDTH, xtol + rtol * abs(0.5 * (lo + hi))):
            break
        got = False
        for frac in (0.5, 0.45, 0.55, 0.40, 0.60, 0.35, 0.65, 0.30, 0.70):
            mid = lo + frac * width
            fm, ok = _probe(f, mid)
            if ok:
                got = True
                break
        if not got:
            break
        if (fm > 0.0) == (flo > 0.0):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return 0.5 * (lo + hi), hi - lo


def _scan(f, p0_Pa, lo_fac, hi_fac, n):
    lo, hi = np.log(lo_fac * p0_Pa), np.log(hi_fac * p0_Pa)
    xs = np.linspace(lo, hi, int(n))
    out = []
    for x in xs:
        v, ok = _probe(f, float(x))
        out.append((float(x), v, ok))
    return out


def solve_psat_isofugacity(f, p0_Pa, *, lo_fac=DEFAULT_LO_FAC,
                           hi_fac=DEFAULT_HI_FAC, xtol=1e-12, rtol=1e-10,
                           n_verify=25, n_scan=97, context=""):
    """Return the saturation pressure [Pa] as the root of an isofugacity residual.

    The root is first sought with ``brentq`` on the bracket
    ``[lo_fac p0, hi_fac p0]``, which is used only if both ends can be
    evaluated, have opposite signs, and ``n_verify`` samples show exactly
    one sign change.  Otherwise the windows ``[SCAN_LO_FAC, SCAN_HI_FAC]``
    and then ``[WIDE_LO_FAC, WIDE_HI_FAC]`` (times ``p0``) are scanned and
    the single sign change found is refined by bisection.  Samples with
    ``|f| <= 1e-11`` are skipped: at such points the aqueous kernel has
    returned the vapour root, so the residual is zero for the wrong reason.
    A root is returned only if it is inside the bracket and ``f`` changes
    sign across it.

    Parameters
    ----------
    f : callable
        ``f(lnP) -> float``, the water isofugacity residual as a function of
        the natural log of the pressure in Pa:
        ``[ln x_w + lnphi_w^aq(T, P)] - lnphi_w^gas(T, P)``.
    p0_Pa : float
        Saturation pressure of pure water at the same temperature (from
        IAPWS or CoolProp) [Pa].  All brackets are multiples of it.
    lo_fac : float
        Lower end of the first bracket, as a multiple of ``p0_Pa``.
    hi_fac : float
        Upper end of the first bracket, as a multiple of ``p0_Pa``.
    xtol : float
        Absolute tolerance in ln P for ``brentq`` and for the bisection.
    rtol : float
        Relative tolerance in ln P for ``brentq`` and for the bisection.
    n_verify : int
        Number of samples used to check the first bracket.
    n_scan : int
        Number of samples in each scanned window.
    context : str
        Text included in error messages, for example ``"T=283.15 K, m=6"``.

    Returns
    -------
    float
        Saturation pressure [Pa].

    Raises
    ------
    PsatBracketError
        If no single root with a sign change can be found, if the residual
        has more than one sign change in a window, or if the refinement
        cannot narrow the bracket.  A bracket end or NaN is never returned.
    """
    from scipy.optimize import brentq

    p0_Pa = float(p0_Pa)
    if not (np.isfinite(p0_Pa) and p0_Pa > 0.0):
        raise PsatBracketError(
            f"psat: non-physical anchor p0={p0_Pa!r} ({context})")

    # ---- 1. the historical bracket, but only if it verifies -----------
    # Verified means: both endpoints evaluable and uncontaminated, opposite
    # signs, and exactly one sign change among the clean interior samples.
    # When it verifies, the brentq call below is the identical call the old
    # code made, so every well-posed state keeps a bit-identical root.
    lo, hi = np.log(lo_fac * p0_Pa), np.log(hi_fac * p0_Pa)
    flo, ok_lo = _probe(f, lo)
    fhi, ok_hi = _probe(f, hi)
    if ok_lo and ok_hi and ((flo > 0.0) != (fhi > 0.0)):
        if len(_sign_changes(_scan(f, p0_Pa, lo_fac, hi_fac,
                                   n_verify))) == 1:
            try:
                lnP = float(brentq(f, lo, hi, xtol=xtol, rtol=rtol))
            except (ValueError, RuntimeError):
                lnP = None
            # A bracket edge is never an answer: the root must be strictly
            # interior and must survive the two-sided crossing check.
            if lnP is not None:
                margin = max(xtol, rtol * abs(lnP))
                if (lo + margin < lnP < hi - margin
                        and _is_true_crossing(f, lnP)):
                    return float(np.exp(lnP))

    # ---- 2. adaptive scan --------------------------------------------
    for wlo, whi in ((SCAN_LO_FAC, SCAN_HI_FAC), (WIDE_LO_FAC, WIDE_HI_FAC)):
        samples = _scan(f, p0_Pa, wlo, whi, n_scan)
        changes = _sign_changes(samples)
        if len(changes) > 1:
            facs = [f"{np.exp(c[0]) / p0_Pa:.4f}-{np.exp(c[2]) / p0_Pa:.4f}"
                    for c in changes]
            raise PsatBracketError(
                f"psat: the isofugacity residual is not single-rooted "
                f"({context}): {len(changes)} sign changes among the clean "
                f"samples over [{wlo}, {whi}]*p0, at P/p0 in "
                f"{', '.join(facs)}. No single saturation pressure is "
                f"defined for this state")
        if len(changes) == 1:
            a, fa, b, fb = changes[0]
            lnP, width = _bisect_clean(f, a, fa, b, fb, xtol, rtol)
            if width > 1e-6 * max(1.0, abs(lnP)):
                raise PsatBracketError(
                    f"psat: the isofugacity root could not be refined "
                    f"({context}): the bracket stalled at a width of "
                    f"{width:.3e} in ln P because every further interior "
                    f"sample was contaminated (the aqueous kernel returns "
                    f"the vapour root there)")
            if not _is_true_crossing(f, lnP):
                raise PsatBracketError(
                    f"psat: the candidate root at P/p0="
                    f"{np.exp(lnP) / p0_Pa:.6f} ({context}) does not "
                    f"behave like a sign crossing when stepped off by "
                    f"{_ROOT_CHECK_H:g} in ln P; it is an artefact of the "
                    f"contaminated residual, not a saturation pressure")
            return float(np.exp(lnP))

    raise PsatBracketError(
        f"psat: no isofugacity sign change found ({context}) among the "
        f"clean samples over [{WIDE_LO_FAC}, {WIDE_HI_FAC}]*p0 with "
        f"p0={p0_Pa:.6g} Pa; no saturation pressure is returned")
