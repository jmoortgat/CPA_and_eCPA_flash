"""Saturation-pressure solve by water isofugacity, on a VERIFIED bracket.

The brine (and pure-water) saturation pressure is the root in ``P`` of

    f(ln P) = [ln x_w + lnphi_w^aq(T, P)] - lnphi_w^gas(T, P) = 0,

i.e. equal water fugacity in the brine liquid and in pure steam.  Every
caller of that root -- the vapor-pressure scripts and the
benchmark's ``psat_brine`` -- used to hand ``scipy.optimize.brentq`` the
fixed bracket ``[0.50 p0, 1.10 p0]`` around the IAPWS pure-water anchor
``p0`` and trust whatever came back.  Two things go wrong with that below
about 300 K.

1.  **The root is not always inside the bracket**, and ``brentq`` raises
    (the callers turned that into a silent NaN).

2.  **Worse, the residual is not single-valued on the sampled grid.**  At
    low ``T`` and low ``P`` the aqueous inner Newton in
    :func:`ms_ecpa.lnphi.lnphi_aq` occasionally stalls just above its
    ``1e-10`` tolerance; ``lnphi_aq`` then falls through its seed ladder
    and returns the **vapour** root of the aqueous mixture instead of the
    liquid one.  At such a point the two sides of the isofugacity balance
    are the same phase, so the residual collapses to zero to machine
    precision -- ``|f| ~ 1e-19 .. 1e-12`` -- with an arbitrary sign.
    These spurious zeros are sprinkled unpredictably through the bracket
    (10 of 46 log-spaced samples at 283.15 K, 0 m).  They are what makes
    the residual appear to "change sign 3-20 times" at the cold end, and
    when one of them lands on a bracket **edge** ``brentq`` converges onto
    that edge and reports it as a saturation pressure.  That is how
    283.15 K, m = 0 came to be published as exactly ``0.5 * p0``.

    On the physical liquid branch the same residual is perfectly smooth
    and monotone with ``|df/d ln P| = O(1)``, and has exactly ONE root --
    including at 283.15 and 293.15 K.

This module therefore does three things the bare ``brentq`` call did not:

* it **verifies** the bracket (endpoints evaluable, opposite signs, and
  exactly one sign change among the valid interior samples) before
  trusting it, and only then runs the identical ``brentq`` call -- so
  every state that was already well posed returns a bit-identical root;
* it **discards contaminated samples** (``|f| <= _DEGENERATE``) instead of
  mistaking them for roots, and never lets the refinement land on one;
* it **fails loudly** -- :class:`PsatBracketError` -- when no single clean
  root can be established, instead of returning a bracket edge or a NaN
  that looks like a number.

Not mirrored in the Fortran port.  ``fortran/src/ecpa_colligative.f90``
``psat_brine`` is a deliberate byte-faithful mirror of the OLD reference
(fixed ``lo_fac``/``hi_fac``, bare Brent).  Its golden vectors cover
T = 323.15-573.15 K only, where the default bracket verifies and this
module reproduces the old root exactly, so the gate is unaffected; below
about 300 K the Fortran routine still carries the original defect.
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
    """Saturation pressure [Pa] from the isofugacity residual ``f(ln P)``.

    Parameters
    ----------
    f : callable
        ``f(lnP) -> float``; the water isofugacity residual.
    p0_Pa : float
        Pure-water saturation pressure at the same temperature (IAPWS or
        CoolProp); the physical anchor the brackets are scaled from.
    lo_fac, hi_fac : float
        The bracket to TRY FIRST, as multiples of ``p0_Pa``.  It is used
        only if it verifies; the historical defaults reproduce the old
        root bit-for-bit wherever the old root was meaningful.
    xtol, rtol : float
        Passed to ``brentq`` on the verified default bracket, and used as
        the stopping width for the adaptive refinement.
    n_verify, n_scan : int
        Samples used to verify the default bracket and, failing that, to
        scan the adaptive window.
    context : str
        Free text (``"T=283.15 K, m=6"``) folded into error messages.

    Raises
    ------
    PsatBracketError
        When no single clean root can be established.  A bracket edge is
        never returned, and neither is NaN.
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
