"""Ground-truth verification of the association/permittivity chi-chain derivatives.

The aqueous inner solve carries three association fixed points

    F(chi_w, chi_c, chi_h; delta, u1, u2, u3) = chi_w - 1/(1 + 2*delta*(
        u1*chi_w + u3*chi_c*S13 + u2*chi_h*S12A))
    G(...) = chi_c - 1/(1 + 2*delta*u1*chi_w*S13)
    H(...) = chi_h - 1/(1 + 2*delta*u1*chi_w*S12A)

with u_k = rho*x_k = N_k/V (mol/m^3).  ``_aq_chi_derivatives`` hard-codes the
partials of F/G/H and solves the implicit-function-theorem system for
d(chi_w)/d(V, N_w, N_c, N_h).  Those partials feed the *residual* (through
ZPerm), not just the Jacobian, so an error in them moves the converged state.

This module re-derives every partial by complex-step differentiation of the
defining equations above -- written out independently of ``lnphi.py`` -- and
asserts agreement to ~1e-14 relative.  It is the regression guard for the
2026-09 chi-chain correction:

  * ``dFdchic`` / ``dGdchiw`` (and the S12A analogue ``dHdchiw``) must carry
    the chi of the row's OWN residual, not the other one.
  * ``dFddelta`` / ``dGddelta`` / ``dHddelta`` are ``+chi**2 * Q`` with
    ``Q = dQ/ddelta``; the old ``-chi**2*(1 + (delta-1)*Q)`` grouping is
    dimensionally incoherent (delta has units m^3/mol) and differs from the
    truth by exactly ``-chi``.
"""
from __future__ import annotations

import numpy as np
import pytest

from ms_ecpa import lnphi as _lnphi
from ms_ecpa import parameters as _prm


CS = 1e-200  # complex step; exact to machine precision, no subtraction error


def _residuals(chi_w, chi_c, chi_h, delta, u1, u2, u3, S13, S12A):
    """The three defining association fixed points, as residuals."""
    F = chi_w - 1.0 / (1.0 + 2.0 * delta * (u1 * chi_w
                                            + u3 * chi_c * S13
                                            + u2 * chi_h * S12A))
    G = chi_c - 1.0 / (1.0 + 2.0 * delta * u1 * chi_w * S13)
    H = chi_h - 1.0 / (1.0 + 2.0 * delta * u1 * chi_w * S12A)
    return F, G, H


def _cs_partial(args, idx, S13, S12A):
    """Complex-step d(F,G,H)/d(args[idx])."""
    a = [complex(v) for v in args]
    a[idx] = a[idx] + 1j * CS
    F, G, H = _residuals(*a, S13, S12A)
    return (F.imag / CS, G.imag / CS, H.imag / CS)


def _ground_truth(chi_w, chi_c, chi_h, delta, rho, x1w, x2w, x3w, S13, S12A):
    """All partials of F/G/H in the kernel's own derivative convention.

    Convention (matches ``_aq_chi_derivatives``):
      * ``dXdchi*`` and ``dXddelta`` are plain partials;
      * ``dXdV``  is ``V * dX/dV`` at fixed N and fixed delta, i.e. minus the
        Euler derivative in the u_k (which all scale as 1/V);
      * ``dXdN*`` is ``N * dX/dN_k`` at fixed V and fixed delta, i.e.
        ``rho * dX/du_k``.
    """
    u1, u2, u3 = rho * x1w, rho * x2w, rho * x3w
    args = (chi_w, chi_c, chi_h, delta, u1, u2, u3)

    dchiw = _cs_partial(args, 0, S13, S12A)
    dchic = _cs_partial(args, 1, S13, S12A)
    dchih = _cs_partial(args, 2, S13, S12A)
    dddel = _cs_partial(args, 3, S13, S12A)
    du1 = _cs_partial(args, 4, S13, S12A)
    du2 = _cs_partial(args, 5, S13, S12A)
    du3 = _cs_partial(args, 6, S13, S12A)

    out = {}
    for row, name in enumerate("FGH"):
        out[f"d{name}dchiw"] = dchiw[row]
        out[f"d{name}dchic"] = dchic[row]
        out[f"d{name}dchih"] = dchih[row]
        out[f"d{name}ddelta"] = dddel[row]
        # V * dX/dV at fixed N: u_k = N_k/V  ->  V du_k/dV = -u_k
        out[f"d{name}dV"] = -(u1 * du1[row] + u2 * du2[row] + u3 * du3[row])
        # N * dX/dN_k at fixed V: du_k/dN_k = 1/V  ->  N/V = rho
        out[f"d{name}dNw"] = rho * du1[row]
        out[f"d{name}dNh"] = rho * du2[row]
        out[f"d{name}dNc"] = rho * du3[row]
    return out


def _kernel_expressions(chi1w, chi3w, chi2w, delta, rho, x1w, x2w, x3w,
                        S13, S12A_):
    """Verbatim transcription of the expressions in ``_aq_chi_derivatives``.

    Kept as a literal copy so this test fails loudly if the kernel algebra is
    edited without updating the ground truth (or vice versa).
    """
    return {
        "dFdchiw": 1 + 2 * x1w * rho * delta * chi1w ** 2,
        "dFdchic": 2 * x3w * rho * S13 * delta * chi1w ** 2,
        "dFdchih": 2 * x2w * rho * S12A_ * delta * chi1w ** 2,
        "dGdchiw": 2 * x1w * rho * S13 * delta * chi3w ** 2,
        "dGdchic": 1.0,
        "dHdchiw": 2 * x1w * rho * S12A_ * delta * chi2w ** 2,
        "dHdchih": 1.0,
        "dFddelta": chi1w ** 2 * (2 * rho * x1w * chi1w
                                  + 2 * rho * x3w * chi3w * S13
                                  + 2 * rho * x2w * chi2w * S12A_),
        "dGddelta": chi3w ** 2 * (2 * rho * x1w * chi1w * S13),
        "dHddelta": chi2w ** 2 * (2 * rho * x1w * chi1w * S12A_),
        "dFdV": -(2 * rho * x1w * delta * chi1w
                  + 2 * rho * x3w * S13 * delta * chi3w
                  + 2 * rho * x2w * S12A_ * delta * chi2w) * chi1w ** 2,
        "dGdV": -(2 * rho * x1w * S13 * delta * chi1w) * chi3w ** 2,
        "dHdV": -(2 * rho * x1w * S12A_ * delta * chi1w) * chi2w ** 2,
        "dFdNw": 2 * rho * delta * chi1w ** 3,
        "dGdNw": 2 * rho * S13 * delta * chi1w * chi3w ** 2,
        "dHdNw": 2 * rho * S12A_ * delta * chi1w * chi2w ** 2,
        "dFdNc": 2 * rho * S13 * delta * chi1w ** 2 * chi3w,
        "dFdNh": 2 * rho * S12A_ * delta * chi1w ** 2 * chi2w,
        "dGdNc": 0.0,
        "dGdNh": 0.0,
        "dHdNc": 0.0,
        "dHdNh": 0.0,
    }


def _solve_fixed_point(delta, rho, x1w, x2w, x3w, S13, S12A):
    """Converge (chi_w, chi_c, chi_h) on the defining fixed point.

    The kernel writes its partials with ``chi**2`` where the algebra calls for
    ``chi_new**2``; the two coincide only ON the fixed point, which is where
    ``_aq_chi_derivatives`` is always evaluated.  Test states must therefore be
    genuine fixed points, not arbitrary triples.
    """
    u1, u2, u3 = rho * x1w, rho * x2w, rho * x3w
    chi_w = 0.5
    for _ in range(2000):
        chi_c = 1.0 / (1.0 + 2.0 * delta * u1 * chi_w * S13)
        chi_h = 1.0 / (1.0 + 2.0 * delta * u1 * chi_w * S12A)
        new = 1.0 / (1.0 + 2.0 * delta * (u1 * chi_w + u3 * chi_c * S13
                                          + u2 * chi_h * S12A))
        if abs(new - chi_w) < 1e-15:
            chi_w = new
            break
        chi_w = 0.5 * (chi_w + new)   # damped, always converges here
    chi_c = 1.0 / (1.0 + 2.0 * delta * u1 * chi_w * S13)
    chi_h = 1.0 / (1.0 + 2.0 * delta * u1 * chi_w * S12A)
    F, G, H = _residuals(chi_w, chi_c, chi_h, delta, u1, u2, u3, S13, S12A)
    assert max(abs(F), abs(G), abs(H)) < 1e-13, "test state is not a fixed point"
    return chi_w, chi_c, chi_h


# (delta [m^3/mol], rho [mol/m^3], x1w, x2w, x3w) -- spans chi_w ~ 0.05-0.95,
# the range the aqueous inner Newton actually visits.
_STATES = [
    (1.0e-6, 5.0e4, 0.980, 0.000, 0.020),
    (1.4e-5, 5.0e4, 0.980, 0.000, 0.020),
    (2.6e-5, 4.2e4, 0.930, 0.010, 0.060),
    (8.0e-6, 5.5e4, 0.999, 0.000, 0.001),
    (3.0e-4, 3.0e4, 0.900, 0.050, 0.050),
]
_SPARAMS = [(0.30, 0.0), (0.30, 0.12), (0.05, 0.40)]


@pytest.mark.parametrize("state", _STATES)
@pytest.mark.parametrize("sp", _SPARAMS)
def test_kernel_partials_match_complex_step(state, sp):
    delta, rho, x1w, x2w, x3w = state
    S13, S12A_ = sp
    chi1w, chi3w, chi2w = _solve_fixed_point(delta, rho, x1w, x2w, x3w,
                                             S13, S12A_)

    truth = _ground_truth(chi1w, chi3w, chi2w, delta, rho,
                          x1w, x2w, x3w, S13, S12A_)
    got = _kernel_expressions(chi1w, chi3w, chi2w, delta, rho,
                              x1w, x2w, x3w, S13, S12A_)

    worst = 0.0
    bad = []
    for key, ref in truth.items():
        if key not in got:
            continue
        val = got[key]
        scale = max(abs(ref), abs(val), 1e-30)
        rel = abs(val - ref) / scale
        worst = max(worst, rel)
        if rel > 1e-12:
            bad.append(f"{key}: kernel={val!r} truth={ref!r} rel={rel:.3e}")
    assert not bad, "chi-chain partials disagree with complex-step truth:\n" + \
        "\n".join(bad)
    assert worst < 1e-12


@pytest.mark.parametrize("state", _STATES)
@pytest.mark.parametrize("sp", _SPARAMS)
def test_live_kernel_matches_transcription(state, sp, monkeypatch):
    """The live ``_aq_chi_derivatives`` must reproduce the solved chain.

    Builds the implicit-function-theorem solution from the ground-truth
    partials and compares against the kernel's alpha/beta fixed-point form,
    driving the kernel through a synthetic ``state`` dict.
    """
    delta, rho, x1w, x2w, x3w = state
    S13, S12A_ = sp
    chi1w, chi3w, chi2w = _solve_fixed_point(delta, rho, x1w, x2w, x3w,
                                             S13, S12A_)
    # eta/g_eta/dg_deta/b only enter through VddeltadV and NddeltadN*, which
    # are common to both sides; any self-consistent set exercises the solve.
    eta, g_eta, dg_deta, b = 0.30, 1.0 / (1 - 1.9 * 0.30), \
        1.9 / (1 - 1.9 * 0.30) ** 2, 2.0e-5

    st = dict(delta=delta, chi3w=chi3w, chi2w=chi2w, g_eta=g_eta,
              dg_deta=dg_deta, eta=eta, b=b, rho=rho, S13=S13, S12A=S12A_)
    monkeypatch.setattr(_prm, "S12A", S12A_, raising=False)

    got = _lnphi._aq_chi_derivatives(st, 1.0, 70.0, chi1w,
                                     (x1w, x2w, x3w), np.zeros(6), 400.0, 1e7)

    truth = _ground_truth(chi1w, chi3w, chi2w, delta, rho,
                          x1w, x2w, x3w, S13, S12A_)
    VddeltadV = -delta * eta * dg_deta / g_eta
    b1, b2, b3 = _prm.b1, _prm.b2, _prm.b3
    NddeltadN = {
        "Nw": delta * eta * b1 * dg_deta / (g_eta * b),
        "Nh": delta * eta * b2 * dg_deta / (g_eta * b),
        "Nc": delta * eta * b3 * dg_deta / (g_eta * b),
    }

    def solve(tag, dtot_extra):
        rhs = np.array([
            -(truth[f"d{n}d{tag}"] + truth[f"d{n}ddelta"] * dtot_extra)
            for n in "FGH"
        ])
        J = np.array([[truth[f"d{n}dchi{c}"] for c in ("w", "c", "h")]
                      for n in "FGH"])
        J[1, 1] = 1.0
        J[2, 2] = 1.0
        return np.linalg.solve(J, rhs)[0]

    ref = (
        solve("V", VddeltadV),
        solve("Nw", NddeltadN["Nw"]),
        solve("Nc", NddeltadN["Nc"]),
        solve("Nh", NddeltadN["Nh"]),
    )
    names = ("Vdchi1WdV", "Ndchi1WdNw", "Ndchi1WdNc", "Ndchi1WdNh")
    for name, g, r in zip(names, got, ref):
        if name in ("Ndchi1WdNc",) and x3w == 0:
            continue
        if name in ("Ndchi1WdNh",) and x2w == 0:
            continue
        scale = max(abs(g), abs(r), 1e-30)
        assert abs(g - r) / scale < 1e-11, f"{name}: kernel={g!r} truth={r!r}"
