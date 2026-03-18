"""
CPA2.py — two-component CPA flash for CO₂ + H₂O, salt-free.

This revision replaces the original van der Waals one-fluid (vdW1f) mixing
rule with the Huron–Vidal (HV) mixing rule used in the eCPA notebook
(Coelho, Franco & Firoozabadi, IECR 2025), and replaces the Carnahan–Starling
radial distribution function with the simplified form g(η) = 1/(1−1.9η) used
in the same reference.  With these changes and the eCPA parameter set the
salt-free flash reproduces the eCPA ELV solution at ms→0.

Key differences from the previous CPA2.py
------------------------------------------
MIXING RULE
  Old:  vdW1f — a_mix = ΣᵢΣⱼ xᵢxⱼ √(aᵢaⱼ)(1−kᵢⱼ)   (symmetric quadratic)
  New:  Huron–Vidal for the AQUEOUS phase (water-rich); vdW1f retained for
        the CO₂-RICH phase — matching exactly what the eCPA notebook does.
        The aqueous HV mixing rule:
            b   = Σᵢ xᵢ bᵢ
            U₁₄ = ln2·[a₄/b₄ − 2√(a₁a₄)(1−k₁₄)/(b₁+b₄)]
            U₄₁ = ln2·[a₁/b₁ − 2√(a₁a₄)(1−k₁₄)/(b₄+b₁)]
            gᴱ  = (1/b)·x₁x₄(b₁U₁₄ + b₄U₄₁)
            a   = b·[Σᵢ xᵢ aᵢ/bᵢ − gᴱ/ln2]

FUGACITY COEFFICIENTS
  Old:  PR-form log term:  ln[(Z+(1+√2)B)/(Z+(1−√2)B)]  (Peng-Robinson)
  New:  SRK-form log term: ln[(Z+B)/Z]  for aqueous phase (matches eCPA ELV)
        SRK-form:           ln[1+B/Z]   for CO₂-rich phase (same, rearranged)
        The HV correction to ∂a/∂nᵢ replaces the vdW1f dAdn.

RADIAL DISTRIBUTION FUNCTION
  Old:  Carnahan–Starling: g = (1−0.5η)/(1−η)³
  New:  Simplified form:   g = 1/(1−1.9η)   with  dg/dη = 1.9/(1−1.9η)²

ASSOCIATION FUGACITY
  Matches eCPA notebook exactly:
    lnφᵢ_ass = 4·ln(χᵢ) + Bᵢ/(8·g·Z)·(dg/dη)·Σⱼ 4xⱼ(χⱼ−1)

PARAMETER SETS
  "CPA2"  — original MATLAB parameters.  Use MIXING="vdW1f", G_ETA="CS",
             kij12=0.0 for the best standalone CPA2 results.
  "eCPA"  — Coelho et al. 2025.  Use MIXING="HV", G_ETA="MOD" (defaults),
             kij12=kij_ecpa(T).  Reproduces eCPA ELV at ms→0.

Units
-----
  T   : K          P   : bar
  R   : 0.083145 L·bar/(mol·K)
  b   : L/mol      a   : bar·L²/mol²
  Pc  : bar        Mw  : kg/mol

Notebook variable mapping (eCPA → this module)
  x1  = H₂O mole fraction  →  x[1]  (component 1, water)
  x4  = CO₂ mole fraction  →  x[0]  (component 0, solvent)
  chi1w / chi1c  = χ_H₂O  →  Chi   (water association fraction)
  chi4w / chi4c  = χ_CO₂  →  Chi1  (CO₂ cross-association fraction; =1 when swc=0)
  Zw  = aqueous Z          →  Zx
  Zc  = CO₂-rich Z         →  Zy

Author: derived from CPA2_040926.py and eCPA_VLE_04102026.ipynb by Claude.
"""

from __future__ import annotations
import numpy as np

# =============================================================================
# PARAMETER SET SELECTION
# =============================================================================
PARAM_SET: str = "eCPA"   # "CPA2" | "eCPA"

# =============================================================================
# ALGORITHM FLAGS
# =============================================================================
MIXING: str = "HV"    # "HV" | "vdW1f"
G_ETA:  str = "MOD"   # "MOD" | "CS"

# =============================================================================
# Constants
# =============================================================================
R_BAR_L    = 0.083145   # L·bar/(mol·K)
_LN2       = np.log(2.0)
_LOG_GUARD = 1e-300

# =============================================================================
# CPA2 parameter set  (original MATLAB / ParaCompEOS.m)
# =============================================================================
_PARAMS_CPA2 = {
    "Tc_co2":    304.1282,
    "Pc_co2":    73.773,
    "Omega_co2": 0.22394,
    "b_co2":     None,                      # derived from PR formula
    "alpha_co2": "PR",

    "Tc_h2o":    647.096,
    "Pc_h2o":    220.64,
    "Omega_h2o": 0.34400,
    "b_h2o":     0.01458431489141052,       # L/mol  (fitted)
    "alpha_h2o": "poly3",
    "ai0_h2o":   0.9627316625476,
    "c1_h2o":    1.75573246325004,
    "c2_h2o":    0.00351802110081,
    "c3_h2o":   -0.27463687473246,

    "eps_over_T": 1738.393603227767,        # K
    "kappa":      0.001801506043021089,     # L/mol

    "Mw_co2": 44.0095e-3,
    "Mw_h2o": 18.01528e-3,
}

# =============================================================================
# eCPA parameter set  (Coelho, Franco & Firoozabadi, IECR 2025)
#
# Notebook uses SI (m³/mol, J/(mol·K)).  Converted to L·bar units here:
#   b [L/mol]     = b_SI [m³/mol] × 1000
#   kappa [L/mol] = kappa_SI [m³/mol] × 1000
# The MC1 alpha prefactor 0.45724·R²Tc²/Pc is equivalent to the notebook's
# a0i = (const)·R·bi after unit conversion; we use the standard formula.
# =============================================================================
_b1_ecpa  = 14.515e-6 * 1e3          # 0.014515 L/mol   H₂O
_b4_ecpa  = 27.2e-6   * 1e3          # 0.027200 L/mol   CO₂
_bettaW   = 69.2e-3
_kappa_W  = _bettaW * _b1_ecpa       # 0.0010044 L/mol

_R_si   = 8.314          # J/(mol·K) — used only to compute a0 from notebook formula
_a01_nb = 1017.3 * _R_si * _b1_ecpa / 1000  # J·m³/mol² → bar·L²/mol² (* 10 / 1000? no)
# Unit conversion: a01_notebook [J·m³/mol²] × 10 = [bar·L²/mol²]
# a01 = 1017.3 * R_si [J/mol/K] * b1 [m³/mol] = J·m³/mol² × K⁻¹ ... wait, units:
# R_si [J/(mol·K)], b1 [m³/mol]: product = J·m³/(mol²·K) — needs ×T to get J·m³/mol²
# BUT in the notebook a01 appears as the PREFACTOR before the alpha function, so
# a1 = a01*(1+c11*(1-sqrt(T/Tc)))^2 where a01 must have units [J·m³/mol²].
# The notebook sets a01=1017.3*R*b1 which has units J·m³/(mol²·K)·...
# Actually R [J/mol/K] * b1 [m³/mol] = J·m³/mol²/K — that's not right dimensionally.
# Looking more carefully: the notebook uses R=8.314 J/mol/K and b1=14.515e-6 m³/mol.
# a01 = 1017.3 * 8.314 * 14.515e-6 = 0.12277 J·m³/mol².  This is dimensionally:
# [dimensionless] × [J/(mol·K)] × [m³/mol] = J·m³/(mol²·K).
# For a to have correct units [J·m³/mol²], we need a01 to be [J·m³/mol²] (not per K).
# The factor 1017.3 must carry units of K implicitly — it's a fitted constant such that
# a01 = 1017.3[K] × R[J/mol/K] × b1[m³/mol] = 1017.3 × R × b1 [J·m³/mol²]. ✓
# This is the CPA convention: a0 = Ω_a * R * Tc * b  (not the PR formula a0 = 0.45724*R²Tc²/Pc).
# Convert to bar·L²/mol²: 1 J·m³/mol² = 10 bar·L²/mol²
_a01_ecpa = 1017.3 * _R_si * (_b1_ecpa/1000) * 10   # bar·L²/mol²  (1017.3 K × R × b1 → ×10)
_a04_ecpa = 1551.2 * _R_si * (_b4_ecpa/1000) * 10   # bar·L²/mol²

_PARAMS_ECPA = {
    "Tc_co2":    304.4,
    "Pc_co2":    73.80,
    "Omega_co2": 0.22394,
    "b_co2":     _b4_ecpa,
    "alpha_co2": "MC1",
    "c1_co2":    0.7602,
    "a0_co2":    _a04_ecpa,   # fitted prefactor [bar·L²/mol²]; not from PR formula

    "Tc_h2o":    647.29,
    "Pc_h2o":    220.60,
    "Omega_h2o": 0.34400,
    "b_h2o":     _b1_ecpa,
    "alpha_h2o": "MC1",
    "c1_h2o":    0.6736,
    "a0_h2o":    _a01_ecpa,   # fitted prefactor [bar·L²/mol²]

    "eps_over_T": 2003.25,              # K
    "kappa":      _kappa_W,             # L/mol

    "Mw_co2": 44.0095e-3,
    "Mw_h2o": 18.01528e-3,
}

# =============================================================================
# Active parameter set
# =============================================================================
_VALID_PARAM = {"CPA2", "eCPA"}
_VALID_MIX   = {"HV", "vdW1f"}
_VALID_G     = {"MOD", "CS"}

if PARAM_SET not in _VALID_PARAM:
    raise ValueError(f"PARAM_SET must be one of {_VALID_PARAM}")
if MIXING not in _VALID_MIX:
    raise ValueError(f"MIXING must be one of {_VALID_MIX}")
if G_ETA not in _VALID_G:
    raise ValueError(f"G_ETA must be one of {_VALID_G}")

_P = _PARAMS_CPA2 if PARAM_SET == "CPA2" else _PARAMS_ECPA

_DEFAULT_CO2_H2O = {
    "Tc":    np.array([_P["Tc_co2"],    _P["Tc_h2o"]],    dtype=float),
    "Pc":    np.array([_P["Pc_co2"],    _P["Pc_h2o"]],    dtype=float),
    "Omega": np.array([_P["Omega_co2"], _P["Omega_h2o"]], dtype=float),
    "Mw":    np.array([_P["Mw_co2"],    _P["Mw_h2o"]],    dtype=float),
}


# =============================================================================
# Radial distribution function
# =============================================================================
def _g(eta: float) -> float:
    if G_ETA == "MOD":
        return 1.0 / (1.0 - 1.9 * eta)
    return (1.0 - 0.5 * eta) / (1.0 - eta) ** 3


def _dgdeta(eta: float) -> float:
    if G_ETA == "MOD":
        return 1.9 / (1.0 - 1.9 * eta) ** 2
    return (2.5 - eta) / (1.0 - eta) ** 4


# =============================================================================
# Wilson K initial guess
# =============================================================================
def wilson_K_init(
    T: float, P_bar: float,
    Omega: np.ndarray, Tc: np.ndarray, Pc: np.ndarray,
) -> np.ndarray:
    """Wilson Kᵢ = (Pcᵢ/P)·exp[5.373(1+ωᵢ)(1−Tcᵢ/T)]."""
    return (np.asarray(Pc) / P_bar) * np.exp(
        5.373 * (1.0 + np.asarray(Omega)) * (1.0 - np.asarray(Tc) / float(T))
    )


# =============================================================================
# kij and S14 polynomials (eCPA only)
# =============================================================================
def kij_ecpa(T: float) -> float:
    """k₁₄(T) = Akij·(T/Tc4)² + Bkij·(T/Tc4) + Ckij  for eCPA/HV."""
    Tc4  = _PARAMS_ECPA["Tc_co2"]
    Akij = -0.49206; Bkij = 2.10136; Ckij = -1.57135
    tr = float(T) / Tc4
    return float(Akij * tr**2 + Bkij * tr + Ckij)


def s14_ecpa(T: float) -> float:
    """
    Cross-association strength S₁₄(T) = ASij·(T/Tc4)² + BSij·(T/Tc4) + CSij.

    This is the CO₂–H₂O cross-association parameter used in the eCPA notebook
    (Coelho, Franco & Firoozabadi, IECR 2025).  It maps directly to the `swc`
    argument of tie_line_two_comp / flash_tpz_two_comp.

    ASij =  0.19173,  BSij = -0.17299,  CSij = -0.00909

    The polynomial changes sign at T≈290 K.  A negative value means
    anti-cross-association (χ_CO₂ > 1); this is handled correctly by ChiChi.
    When |S14| is very small (T≈288–292 K) the SSI loop can oscillate; the
    flash wrapper flash_co2_h2o_tpz retries with swc=0 if convergence fails.
    """
    Tc4  = _PARAMS_ECPA["Tc_co2"]
    ASij =  0.19173; BSij = -0.17299; CSij = -0.00909
    tr = float(T) / Tc4
    return float(ASij * tr**2 + BSij * tr + CSij)


# =============================================================================
# Pure-component a(T), b
# =============================================================================
def _pure_ab(T: float, R: float = R_BAR_L) -> tuple[float, float, float, float]:
    """Return (a_CO2, b_CO2, a_H2O, b_H2O) in bar·L²/mol² and L/mol."""
    T = float(T)
    Tc4 = _P["Tc_co2"]; Pc4 = _P["Pc_co2"]
    Tc1 = _P["Tc_h2o"]; Pc1 = _P["Pc_h2o"]

    # CO₂
    if _P["alpha_co2"] == "PR":
        b4  = 0.07780 * R * Tc4 / Pc4
        ci  = 0.37464 + 1.54226 * _P["Omega_co2"] - 0.26992 * _P["Omega_co2"]**2
        ai0 = 0.45724 * R**2 * Tc4**2 / Pc4
        a4  = ai0 * (1.0 + ci * (1.0 - np.sqrt(T / Tc4)))**2
    else:  # MC1 — use the fitted a0 prefactor stored in the parameter set
        b4  = _P["b_co2"]
        ai0 = _P["a0_co2"]   # [bar·L²/mol²] — fitted, NOT from PR formula
        a4  = ai0 * (1.0 + _P["c1_co2"] * (1.0 - np.sqrt(T / Tc4)))**2

    # H₂O
    b1 = _P["b_h2o"]
    if _P["alpha_h2o"] == "poly3":
        theta = 1.0 - np.sqrt(T / Tc1)
        a1 = _P["ai0_h2o"] * (
            1.0 + _P["c1_h2o"] * theta
                + _P["c2_h2o"] * theta**2
                + _P["c3_h2o"] * theta**3
        )**2
    else:  # MC1 — use the fitted a0 prefactor stored in the parameter set
        ai0 = _P["a0_h2o"]   # [bar·L²/mol²] — fitted, NOT from PR formula
        a1  = ai0 * (1.0 + _P["c1_h2o"] * (1.0 - np.sqrt(T / Tc1)))**2

    return float(a4), float(b4), float(a1), float(b1)


# =============================================================================
# EOS parameters — aqueous phase
# =============================================================================
def _eos_aq(
    T: float, P_bar: float,
    x: np.ndarray,           # x[0]=CO₂, x[1]=H₂O
    kij12: float,
    R: float = R_BAR_L,
) -> dict:
    """
    EOS parameters for the aqueous (water-rich) phase.

    HV mixing rule (MIXING="HV"):
        b    = b₁x₁ + b₄x₄
        U₁₄  = ln2·(a₄/b₄ − 2√(a₁a₄)(1−k)/( b₁+b₄))
        U₄₁  = ln2·(a₁/b₁ − 2√(a₁a₄)(1−k)/(b₄+b₁))
        gᴱ   = (x₁x₄/b)·(b₁U₁₄ + b₄U₄₁)
        a    = b·(x₁a₁/b₁ + x₄a₄/b₄ − gᴱ/ln2)

    vdW1f fallback (MIXING="vdW1f"):
        a    = x₁²a₁ + 2x₁x₄√(a₁a₄)(1−k) + x₄²a₄

    Returns a dict used by _lnphi_aq and ZChi.
    """
    T = float(T); P_bar = float(P_bar); kij12 = float(kij12)
    x4 = float(x[0]); x1 = float(x[1])   # CO₂=4, H₂O=1 (notebook notation)
    a4, b4, a1, b1 = _pure_ab(T, R)

    b = b1 * x1 + b4 * x4

    if MIXING == "HV":
        U14 = _LN2 * (a4/b4 - 2.0*np.sqrt(a1*a4)*(1.0-kij12)/(b1+b4))
        U41 = _LN2 * (a1/b1 - 2.0*np.sqrt(a1*a4)*(1.0-kij12)/(b4+b1))
        gE  = (x1 * x4 / b) * (b1*U14 + b4*U41)
        a   = b * (x1*a1/b1 + x4*a4/b4 - gE/_LN2)
        hv  = {"U14": U14, "U41": U41, "gE": gE}
    else:
        a14 = np.sqrt(a1*a4) * (1.0 - kij12)
        a   = x1**2*a1 + 2.0*x1*x4*a14 + x4**2*a4
        hv  = None

    A   = a  * P_bar / (R*T)**2
    B   = b  * P_bar / (R*T)
    A1  = a1 * P_bar / (R*T)**2
    B1  = b1 * P_bar / (R*T)
    A4  = a4 * P_bar / (R*T)**2
    B4  = b4 * P_bar / (R*T)
    Kapa = _P["kappa"]      * P_bar / (R*T)
    Eps  = _P["eps_over_T"] / T

    return dict(A=A, B=B, A1=A1, B1=B1, A4=A4, B4=B4,
                a=a, b=b, a1=a1, b1=b1, a4=a4, b4=b4,
                x1=x1, x4=x4, R=R, T=T, P_bar=P_bar,
                kij12=kij12, hv=hv, Kapa=Kapa, Eps=Eps)


# =============================================================================
# EOS parameters — CO₂-rich phase  (always vdW1f, SRK)
# =============================================================================
def _eos_vap(
    T: float, P_bar: float,
    y: np.ndarray,           # y[0]=CO₂, y[1]=H₂O
    kij12: float,
    R: float = R_BAR_L,
) -> dict:
    """
    EOS parameters for the CO₂-rich phase.

    Always vdW1f (as in eCPA ELV CO₂-rich block):
        a = x₁²a₁ + 2x₁x₄a₁₄ + x₄²a₄    a₁₄ = √(a₁a₄)(1−k)
    """
    T = float(T); P_bar = float(P_bar); kij12 = float(kij12)
    x4 = float(y[0]); x1 = float(y[1])
    a4, b4, a1, b1 = _pure_ab(T, R)

    a14 = np.sqrt(a1*a4) * (1.0 - kij12)
    a   = x1**2*a1 + 2.0*x1*x4*a14 + x4**2*a4
    b   = b1*x1 + b4*x4

    A   = a   * P_bar / (R*T)**2
    B   = b   * P_bar / (R*T)
    A1  = a1  * P_bar / (R*T)**2
    B1  = b1  * P_bar / (R*T)
    A4  = a4  * P_bar / (R*T)**2
    B4  = b4  * P_bar / (R*T)
    A14 = a14 * P_bar / (R*T)**2
    Kapa = _P["kappa"]      * P_bar / (R*T)
    Eps  = _P["eps_over_T"] / T

    return dict(A=A, B=B, A1=A1, B1=B1, A4=A4, B4=B4, A14=A14,
                a=a, b=b, a1=a1, b1=b1, a4=a4, b4=b4,
                x1=x1, x4=x4, R=R, T=T, P_bar=P_bar,
                kij12=kij12, Kapa=Kapa, Eps=Eps)


# =============================================================================
# Cubic solver (analytic)
# =============================================================================
def _solve_cubic_real(a3, a2, a1, a0):
    """Real roots of a3·x³ + a2·x² + a1·x + a0 = 0."""
    if abs(a3) < 1e-30:
        if abs(a2) < 1e-30:
            return np.array([] if abs(a1) < 1e-30 else [-a0/a1])
        disc = a1**2 - 4*a2*a0
        if disc < 0:
            return np.array([])
        sq = np.sqrt(disc)
        return np.array([(-a1+sq)/(2*a2), (-a1-sq)/(2*a2)])

    p = a2/a3; q = a1/a3; r = a0/a3
    P3 = q - p**2/3
    Q2 = 2*p**3/27 - p*q/3 + r
    disc = (Q2/2)**2 + (P3/3)**3
    sh = p/3

    if disc > 1e-14:
        sq = np.sqrt(disc)
        return np.array([np.cbrt(-Q2/2+sq) + np.cbrt(-Q2/2-sq) - sh])
    elif disc < -1e-14:
        rho = np.sqrt(-(P3/3)**3)
        th  = np.arccos(np.clip(-Q2/(2*rho), -1, 1))
        return 2*np.cbrt(rho)*np.cos((th + 2*np.pi*np.array([0,1,2]))/3) - sh
    else:
        r_ = np.roots([a3, a2, a1, a0])
        return np.real(r_[np.abs(np.imag(r_)) < 1e-8])


# =============================================================================
# ChiChi — association fraction solver
# =============================================================================
def ChiChi(
    Z: float, B: float, n: np.ndarray,
    Kapa: float, Eps: float, swc: float,
) -> tuple[float, float]:
    """
    Solve for (χ_H₂O, χ_CO₂).

    n[0]=CO₂ moles, n[1]=H₂O moles (or fractions — only ratio matters).
    Uses g(η) from G_ETA flag.
    """
    Z = float(Z); B = float(B)
    Kapa = float(Kapa); Eps = float(Eps); swc = float(swc)
    n = np.asarray(n, dtype=float).reshape(2,)
    nc = n[0]; nw = n[1]

    eta   = B / (4.0*Z)
    g     = _g(eta)
    delta = g * Kapa * np.expm1(Eps)
    delta1 = swc * delta
    tol = 1e-10

    if abs(delta1) < 1e-30:
        if abs(delta) < 1e-30:
            return 1.0, 1.0
        # Quadratic: 2·nw·Z·δ·χ² + Z²·χ − Z² = 0
        a2 = 2.0*nw*Z*delta; a1c = Z**2; a0 = -(Z**2)
        disc = a1c**2 - 4*a2*a0
        if disc < 0:
            raise ValueError("No real Chi root (swc=0 quadratic).")
        sq = np.sqrt(disc)
        roots_q = [(-a1c+sq)/(2*a2), (-a1c-sq)/(2*a2)]
        valid = []
        for Chi in roots_q:
            if not np.isfinite(Chi) or Chi < -tol or Chi > 1+tol:
                continue
            valid.append((float(np.clip(Chi, 0, 1)), 1.0))
        if not valid:
            raise ValueError("No valid Chi root (swc=0).")
        return sorted(valid)[0]

    coeff = np.array([
        4*nw**2*delta*delta1,
        2*nw*Z*(delta+delta1),
        2*delta1*Z*(nc-nw) + Z**2,
        -(Z**2),
    ])
    if not np.all(np.isfinite(coeff)):
        raise ValueError("Non-finite Chi polynomial coefficients.")

    # When delta1 < 0 (S14 < 0, anti-association), chi_CO2 = Z/(Z+2·nw·χ·δ₁)
    # can exceed 1 because the denominator < Z.  The eCPA ELV notebook allows
    # this naturally (no [0,1] constraint on chi).  We replicate that here:
    # chi_H₂O stays in [0,1]; chi_CO2 is allowed up to _chi1_hi > 1 when δ₁ < 0.
    _chi1_hi = 10.0 if delta1 < 0 else 1.0 + tol

    roots = _solve_cubic_real(*coeff)
    valid = []
    for Chi in roots:
        if not np.isfinite(Chi) or Chi < -tol or Chi > 1+tol:
            continue
        Chi = float(np.clip(Chi, 0, 1))
        denom = Z + 2*nw*Chi*delta1
        if not np.isfinite(denom) or denom <= 0:
            continue
        Chi1 = float(Z / denom)
        if not np.isfinite(Chi1) or Chi1 < -tol or Chi1 > _chi1_hi:
            continue
        # Clip chi_H₂O to [0,1]; chi_CO2: clip to [0,1] only when δ₁ ≥ 0
        Chi1_clipped = float(np.clip(Chi1, 0.0, 1.0)) if delta1 >= 0 else float(max(Chi1, 0.0))
        valid.append((Chi, Chi1_clipped))

    if not valid and delta1 < 0:
        # Fallback for ill-conditioned cubic (e.g., very small nw in the CO₂-rich
        # phase where a3 ∝ nw² → 0).  Cross-association is negligible there, so
        # solve with delta1=0 (swc=0 quadratic) and set chi_CO2=1.
        a2_q = 2.0*nw*Z*delta; a1_q = Z**2; a0_q = -(Z**2)
        disc_q = a1_q**2 - 4*a2_q*a0_q
        if disc_q >= 0:
            sq_q = np.sqrt(max(disc_q, 0.0))
            for Chi_q in [(-a1_q + sq_q)/(2*a2_q), (-a1_q - sq_q)/(2*a2_q)]:
                if not np.isfinite(Chi_q) or Chi_q < -tol or Chi_q > 1+tol:
                    continue
                valid.append((float(np.clip(Chi_q, 0.0, 1.0)), 1.0))

    if not valid:
        raise ValueError("No valid (Chi, Chi1) in [0,1].")
    return sorted(valid)[0]


# =============================================================================
# FunZ — compressibility residual  (SRK form)
# =============================================================================
def FunZ(
    Z: float, A: float, B: float,
    n: np.ndarray, Chi: float, Chi1: float,
) -> float:
    """
    Compressibility residual F(Z) = 0 for SRK EOS + association.

    SRK physical Z:   Zphys = Z/(Z−B) − A/(Z+B)
    At solution:      Z = Zphys + Zassoc
    Rearranged:       F = Z − Z/(Z−B) + A/(Z+B) + Zassoc = 0

    Zassoc = −2(1 + η·dg/dη/g)·Σᵢ xᵢ(χᵢ−1)

    NOTE: SRK denominator is (Z+B), PR would be (Z²+2BZ−B²).
    The A term has NO extra factor of Z — that was the bug in the previous version.
    """
    Z = float(Z); A = float(A); B = float(B)
    n = np.asarray(n, dtype=float).reshape(2,)
    if Z - B <= 0 or abs(Z + B) < 1e-30:
        return np.nan
    nc = n[0]; nw = n[1]
    eta = B / (4*Z)
    g   = _g(eta); dg = _dgdeta(eta)
    return (Z - Z/(Z - B) + A/(Z + B)
            - 2.0*(1.0 + eta*dg/g)*(nw*(Chi - 1.0) + nc*(Chi1 - 1.0)))


# =============================================================================
# ExcessGibbs — root selection  (SRK form)
# =============================================================================
def ExcessGibbs(A, B, Z, n, Chi, Chi1):
    """Dimensionless residual Gibbs for root selection (SRK log term)."""
    A = float(A); B = float(B)
    Z = np.asarray(Z, dtype=float)
    Chi  = np.asarray(Chi,  dtype=float)
    Chi1 = np.asarray(Chi1, dtype=float)
    n = np.asarray(n, dtype=float).reshape(2,)
    nc = n[0]; nw = n[1]
    eta = B/(4*Z); g = _g(eta); dg = _dgdeta(eta)
    sC  = np.maximum(Chi,  _LOG_GUARD)
    sC1 = np.maximum(Chi1, _LOG_GUARD)
    return (
        -np.log(Z - B)
        + (B/(Z-B) - A/(Z+B))
        - A/B * np.log(1 + B/Z)
        + 4*(nw*np.log(sC) + nc*np.log(sC1))
        + 2*(nw*(Chi-1) + nc*(Chi1-1)) * eta * dg / g
    )


# =============================================================================
# ZChi — find Z and χ via coarse scan + Brent
# =============================================================================
def _funz_combined(Z, A, B, n, Kapa, Eps, swc):
    Chi, Chi1 = ChiChi(Z, B, n, Kapa, Eps, swc)
    f = FunZ(Z, A, B, n, Chi, Chi1)
    return f, Chi, Chi1


def ZChi(A, B, n, Kapa, Eps, swc):
    """Find Z and (χ_H₂O, χ_CO₂) by coarse log-spaced scan + Brent refinement."""
    N_SCAN = 12; Z_TOL = 1e-10
    n = np.asarray(n, dtype=float).reshape(2,)
    z_lo = B + 1e-10
    z_hi = max(5.0, A / max(B, 1e-10) + 1.0)

    z_grid    = np.geomspace(z_lo, z_hi, N_SCAN)
    f_grid    = np.empty(N_SCAN)
    Chi_grid  = np.empty(N_SCAN)
    Chi1_grid = np.empty(N_SCAN)

    for i, z in enumerate(z_grid):
        try:
            f, Chi, Chi1 = _funz_combined(z, A, B, n, Kapa, Eps, swc)
        except Exception:
            f, Chi, Chi1 = np.nan, np.nan, np.nan
        f_grid[i] = f; Chi_grid[i] = Chi; Chi1_grid[i] = Chi1

    brackets = [
        (z_grid[i], f_grid[i], Chi_grid[i], Chi1_grid[i],
         z_grid[i+1], f_grid[i+1], Chi_grid[i+1], Chi1_grid[i+1])
        for i in range(N_SCAN-1)
        if np.isfinite(f_grid[i]) and np.isfinite(f_grid[i+1])
           and f_grid[i]*f_grid[i+1] < 0
    ]

    if not brackets:
        return 0.0, np.nan, np.nan

    Zi = []; Chii = []; Chi1i = []

    for (za, fa, Ca, C1a, zb, fb, Cb, C1b) in brackets:
        a, fa_c = za, fa
        b, fb_c = zb, fb
        Chi_c = Ca; Chi1_c = C1a
        c, fc = a, fa_c
        mflag = True

        for _ in range(60):
            if abs(b - a) < Z_TOL:
                break
            if fa_c != fc and fb_c != fc:
                s = (a*fb_c*fc/((fa_c-fb_c)*(fa_c-fc))
                   + b*fa_c*fc/((fb_c-fa_c)*(fb_c-fc))
                   + c*fa_c*fb_c/((fc-fa_c)*(fc-fb_c)))
            else:
                s = b - fb_c*(b-a)/(fb_c-fa_c)

            cond = (not (min(a,b) < s < max(a,b))
                    or (mflag and abs(s-b) >= 0.5*abs(b-c))
                    or (not mflag and abs(s-b) >= 0.5*abs(c-b))
                    or (mflag and abs(b-c) < Z_TOL)
                    or (not mflag and abs(c-b) < Z_TOL))
            if cond:
                s = 0.5*(a+b); mflag = True
            else:
                mflag = False

            try:
                fs, Cs, C1s = _funz_combined(s, A, B, n, Kapa, Eps, swc)
            except Exception:
                s = 0.5*(a+b)
                try:
                    fs, Cs, C1s = _funz_combined(s, A, B, n, Kapa, Eps, swc)
                except Exception:
                    break

            c, fc = b, fb_c
            if fa_c*fs < 0:
                b, fb_c = s, fs; Chi_c, Chi1_c = Cs, C1s
            else:
                a, fa_c = s, fs
            if abs(fa_c) < abs(fb_c):
                a, fa_c, b, fb_c = b, fb_c, a, fa_c
                Chi_c, Chi1_c = Cs, C1s

        Zi.append(b); Chii.append(Chi_c); Chi1i.append(Chi1_c)

    if not Zi:
        return 0.0, np.nan, np.nan

    Zi    = np.array(Zi);    Chii  = np.array(Chii);    Chi1i = np.array(Chi1i)
    if Zi.size > 1:
        ge  = ExcessGibbs(A, B, Zi, n, Chii, Chi1i)
        idx = int(np.argsort(ge)[0])
        return float(Zi[idx]), float(Chii[idx]), float(Chi1i[idx])
    return float(Zi[0]), float(Chii[0]), float(Chi1i[0])


# =============================================================================
# Fugacity coefficients
# =============================================================================
def _lnphi_aq(ep: dict, Z: float, Chi: float, Chi1: float) -> np.ndarray:
    """
    ln(φ) for [CO₂, H₂O] in the aqueous phase.

    Physical part — SRK log term ln((Z+B)/Z):
      lnφᵢ_phys = −ln(Z−B) + Bᵢ/B·[B/(Z−B) − A/(Z+B)]
                  − ln((Z+B)/Z) · bracketᵢ

    HV bracketᵢ (matches eCPA ELV lnPHI4phys / lnPHI1phys):
      bracket₄ = A₄/B₄ − 1/(B·ln2)·[x₁(B₁U₁₄+B₄U₄₁)/RT − B₄·gᴱ/RT]
      bracket₁ = A₁/B₁ − 1/(B·ln2)·[x₄(B₁U₁₄+B₄U₄₁)/RT − B₁·gᴱ/RT]

    vdW1f bracketᵢ:
      bracketᵢ = dA/dnᵢ / A − Bᵢ/B     (standard expression)

    Association part (matches eCPA lnPHI1assoc / lnPHI4assoc):
      lnφᵢ_ass = 4·ln(χᵢ) + Bᵢ/(8·g·Z)·(dg/dη)·Σⱼ 4xⱼ(χⱼ−1)
    """
    A  = ep["A"];  B  = ep["B"]
    A1 = ep["A1"]; B1 = ep["B1"]
    A4 = ep["A4"]; B4 = ep["B4"]
    x1 = ep["x1"]; x4 = ep["x4"]
    R  = ep["R"];  T  = ep["T"]; P_bar = ep["P_bar"]
    kij12 = ep["kij12"]
    Z = float(Z); Chi = float(Chi); Chi1 = float(Chi1)

    eta = B/(4*Z); g = _g(eta); dg = _dgdeta(eta)
    lnZmB = np.log(max(Z-B, _LOG_GUARD))
    # ln((Z+B)/Z) — the SRK log argument for aqueous phase
    lnZpBZ = np.log(max((Z+B)/max(Z, _LOG_GUARD), _LOG_GUARD))
    phys_c = B/(Z-B) - A/(Z+B)   # common bracket

    if MIXING == "HV":
        hv  = ep["hv"]
        U14 = hv["U14"]; U41 = hv["U41"]; gE = hv["gE"]
        RT  = R * T
        # In eCPA ELV all Bᵢ and B are dimensionless (=bᵢ·P/(RT));
        # U₁₄ and gᴱ are in bar·L/mol; dividing by RT [bar·L/mol] is correct.
        cross_BU = B1*U14 + B4*U41    # B₁·U₁₄ + B₄·U₄₁  [dimensionless × bar·L/mol]
        # Note: Bᵢ is dimensionless, Uᵢⱼ in bar·L/mol → product has units bar·L/mol
        # Dividing by RT (bar·L/mol) gives dimensionless.
        bracket4 = A4/B4 - 1/(B*_LN2)*(x1*cross_BU/RT - B4*gE/RT)
        bracket1 = A1/B1 - 1/(B*_LN2)*(x4*cross_BU/RT - B1*gE/RT)
    else:  # vdW1f
        a14_r = np.sqrt(A1*A4) * (1.0 - kij12)   # reduced cross energy
        dA4 = 2*(x4*A4 + x1*a14_r)
        dA1 = 2*(x1*A1 + x4*a14_r)
        bracket4 = dA4/A - B4/B
        bracket1 = dA1/A - B1/B

    lnphi_ph = np.array([
        -lnZmB + B4/B*phys_c - lnZpBZ*bracket4,
        -lnZmB + B1/B*phys_c - lnZpBZ*bracket1,
    ])

    # Association — 4 sites on H₂O (Wertheim 4B scheme); CO₂ cross-assoc only if swc>0
    assoc_sum = 4.0*(x1*(Chi-1) + x4*(Chi1-1))
    sC  = max(Chi,  _LOG_GUARD)
    sC1 = max(Chi1, _LOG_GUARD)
    lnphi_ass = np.array([
        4*np.log(sC1) + B4/(8*g*Z)*dg*assoc_sum,
        4*np.log(sC)  + B1/(8*g*Z)*dg*assoc_sum,
    ])

    return lnphi_ph + lnphi_ass


def _lnphi_vap(ep: dict, Z: float, Chi: float, Chi1: float) -> np.ndarray:
    """
    ln(φ) for [CO₂, H₂O] in the CO₂-rich phase.

    Always vdW1f + SRK log term ln(1+B/Z) (matches eCPA ELV CO₂-rich block):
      lnφᵢ_phys = −ln(Z−B) + Bᵢ/B·[B/(Z−B)−A/(Z+B)]
                  + A/B·[Bᵢ/B − 2(Σⱼ xⱼ Aᵢⱼ)/A]·ln(1+B/Z)
    """
    A   = ep["A"];  B  = ep["B"]
    A1  = ep["A1"]; B1 = ep["B1"]
    A4  = ep["A4"]; B4 = ep["B4"]
    A14 = ep["A14"]
    x1  = ep["x1"]; x4 = ep["x4"]
    Z = float(Z); Chi = float(Chi); Chi1 = float(Chi1)

    eta = B/(4*Z); g = _g(eta); dg = _dgdeta(eta)
    lnZmB  = np.log(max(Z-B, _LOG_GUARD))
    ln1pBZ = np.log(max(1 + B/max(Z, _LOG_GUARD), _LOG_GUARD))
    phys_c = B/(Z-B) - A/(Z+B)

    # vdW1f: Σⱼ xⱼ Aᵢⱼ
    dA4_sum = x1*A14 + x4*A4
    dA1_sum = x1*A1  + x4*A14

    lnphi_ph = np.array([
        -lnZmB + B4/B*phys_c + A/B*(B4/B - 2*dA4_sum/A)*ln1pBZ,
        -lnZmB + B1/B*phys_c + A/B*(B1/B - 2*dA1_sum/A)*ln1pBZ,
    ])

    assoc_sum = 4.0*(x1*(Chi-1) + x4*(Chi1-1))
    sC  = max(Chi,  _LOG_GUARD)
    sC1 = max(Chi1, _LOG_GUARD)
    lnphi_ass = np.array([
        4*np.log(sC1) + B4/(8*g*Z)*dg*assoc_sum,
        4*np.log(sC)  + B1/(8*g*Z)*dg*assoc_sum,
    ])

    return lnphi_ph + lnphi_ass


# =============================================================================
# Tie-line solver
# =============================================================================
def tie_line_two_comp(
    T: float, P_bar: float,
    Omega: np.ndarray, Tc: np.ndarray, Pc: np.ndarray, Mw: np.ndarray,
    kij12: float = 0.0, swc: float = 0.0,
    *,
    K_init: np.ndarray | None = None,
    tol: float = 1e-10,
    maxiter: int = 1000,
) -> dict:
    """
    SSI tie-line solver: CO₂ (component 0) + H₂O (component 1).

    Aqueous  phase: MIXING flag (HV or vdW1f) + G_ETA flag.
    CO₂-rich phase: always vdW1f + MOD g(η) (when G_ETA="MOD").

    Returns dict: converged, iterations, K, x [aq], y [vap], and if converged:
      Z[0,1], rho_mass[0,1] [kg/L], assoc_t[0,1], chi{liq,vap}.
    """
    T = float(T); P_bar = float(P_bar); kij12 = float(kij12); swc = float(swc)
    Tc    = np.asarray(Tc,    dtype=float).reshape(2,)
    Pc    = np.asarray(Pc,    dtype=float).reshape(2,)
    Mw    = np.asarray(Mw,    dtype=float).reshape(2,)
    Omega = np.asarray(Omega, dtype=float).reshape(2,)

    if Tc[0] > Tc[1]:
        raise ValueError(
            f"Water (Tc≈647 K) must be component 1 (last). Got Tc={Tc}.")

    if K_init is None:
        K = wilson_K_init(T, P_bar, Omega, Tc, Pc)
        # Wilson K for H₂O (high-Tc component) can exceed 1 at low pressure near
        # boiling (large Pc/P factor overcomes the negative exp term).  The physical
        # K_H₂O < 1 for T < Tc_H₂O, so cap it to prevent the tie-line formula
        # x[0]=(K[1]−1)/(K[1]−K[0]) from yielding negative x_CO₂.
        if float(K[1]) > 1.0:
            K[1] = 0.95
        # Wilson K for CO₂ (low-Tc component) is unreliable when P exceeds Pc,
        # regardless of whether T is sub- or supercritical.  The Pc/P factor
        # drags K[0] below 1 (or barely above 1), making x[0] = (K[1]−1)/(K[1]−K[0])
        # land outside (0, 0.5): a degenerate starting composition from which SSI
        # drifts to the trivial K→1 solution.
        # Check: compute candidate x[0] and, if it is unphysical, replace K[0]
        # with a value that targets x_CO₂ ≈ 0.05 in the aqueous phase.
        _denom_K = float(K[1]) - float(K[0])
        _x0 = (float(K[1]) - 1.0) / _denom_K if abs(_denom_K) > 1e-12 else 2.0
        if not (0.0 < _x0 < 0.5):
            # Solve x[0]=0.05 for K[0]: K[0] = (1 − 0.95·K[1]) / 0.05
            K[0] = max((1.0 - 0.95 * float(K[1])) / 0.05, 5.0)
    else:
        K = np.asarray(K_init, dtype=float).reshape(2,)

    x = np.zeros(2); y = np.zeros(2)
    converged = False; it = 0

    for it in range(int(maxiter)):
        if np.linalg.norm(K - 1.0) < 1e-5:
            break

        denom = K[1] - K[0]
        if abs(denom) < 1e-14:
            break
        x[0] = (K[1] - 1.0) / denom
        x[1] = 1.0 - x[0]
        y[:] = K * x

        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))
                and np.all(x >= -1e-12) and np.all(y >= -1e-12)):
            x[:] = 0; y[:] = 0; break

        x = np.clip(x, 0, 1); y = np.clip(y, 0, 1)

        ep_aq  = _eos_aq (T, P_bar, x, kij12)
        ep_vap = _eos_vap(T, P_bar, y, kij12)

        Zx, Chix, Chi1x = ZChi(ep_aq["A"],  ep_aq["B"],  x,
                                ep_aq["Kapa"],  ep_aq["Eps"],  swc)
        Zy, Chiy, Chi1y = ZChi(ep_vap["A"], ep_vap["B"], y,
                                ep_vap["Kapa"], ep_vap["Eps"], swc)

        if Zx == 0.0 or Zy == 0.0:
            x[:] = 0; y[:] = 0; break

        lnphi_x = _lnphi_aq (ep_aq,  Zx, Chix,  Chi1x)
        lnphi_y = _lnphi_vap(ep_vap, Zy, Chiy,  Chi1y)

        lnK_new  = lnphi_x - lnphi_y
        lnK_old  = np.log(np.maximum(K, 1e-300))
        lnK_step = np.clip(lnK_new - lnK_old, -5.0, 5.0)
        if np.all(np.abs(lnK_old) < 0.05):
            lnK_step *= 0.5

        K = np.exp(lnK_old + lnK_step)
        e = -lnK_step

        if np.linalg.norm(e) < float(tol):
            converged = True; break

    out = {
        "converged": bool(converged),
        "iterations": int(it + 1),
        "K": K.copy(), "x": x.copy(), "y": y.copy(),
    }

    if converged:
        denom = K[1] - K[0]
        x[0] = (K[1] - 1.0) / denom; x[1] = 1.0 - x[0]; y[:] = K * x
        x = np.clip(x, 0, 1); y = np.clip(y, 0, 1)

        ep_aq  = _eos_aq (T, P_bar, x, kij12)
        ep_vap = _eos_vap(T, P_bar, y, kij12)

        Zx, Chix,  Chi1x  = ZChi(ep_aq["A"],  ep_aq["B"],  x,
                                  ep_aq["Kapa"],  ep_aq["Eps"],  swc)
        Zy, Chiy,  Chi1y  = ZChi(ep_vap["A"], ep_vap["B"], y,
                                  ep_vap["Kapa"], ep_vap["Eps"], swc)

        rho_liq = float(np.dot(Mw, x) * P_bar / (Zx * R_BAR_L * T))
        rho_vap = float(np.dot(Mw, y) * P_bar / (Zy * R_BAR_L * T))

        denom_t = (1.0 - Chix**4) * x[1]
        tx = float(((1.0 - Chi1x**4) * x[0]) / denom_t) if abs(denom_t) > 1e-30 else np.nan
        denom_t = (1.0 - Chiy**4) * y[1]
        ty = float(((1.0 - Chi1y**4) * y[0]) / denom_t) if abs(denom_t) > 1e-30 else np.nan

        out.update({
            "Z":        np.array([Zx, Zy]),
            "rho_mass": np.array([rho_liq, rho_vap]),
            "assoc_t":  np.array([tx, ty]),
            "chi": {"liq": (Chix, Chi1x), "vap": (Chiy, Chi1y)},
        })

    return out


# =============================================================================
# TPz flash
# =============================================================================
def flash_tpz_two_comp(
    T: float, P_bar: float, z: np.ndarray,
    Omega: np.ndarray, Tc: np.ndarray, Pc: np.ndarray, Mw: np.ndarray,
    kij12: float = 0.0, swc: float = 0.0,
    *,
    K_init: np.ndarray | None = None,
    tol: float = 1e-10,
    maxiter: int = 1000,
) -> dict:
    """
    TPz flash for CO₂(0) + H₂O(1).

    Returns phase ∈ {'two_phase','single_liquid','single_vapor','failed'},
    beta (vapour fraction), x, y, tie.
    """
    z = np.asarray(z, dtype=float).reshape(2,)
    z = z / z.sum()

    tie = tie_line_two_comp(
        T=T, P_bar=P_bar, Omega=Omega, Tc=Tc, Pc=Pc, Mw=Mw,
        kij12=kij12, swc=swc, K_init=K_init, tol=tol, maxiter=maxiter,
    )

    out = {"T": float(T), "P_bar": float(P_bar), "z": z.copy(), "tie": tie}

    if not tie["converged"]:
        out.update({"phase": "failed", "beta": np.nan}); return out

    x0, y0, z0 = float(tie["x"][0]), float(tie["y"][0]), float(z[0])
    denom = y0 - x0
    if abs(denom) < 1e-14:
        out.update({"phase": "failed", "beta": np.nan}); return out

    beta = (z0 - x0) / denom
    if beta <= 0:
        phase = "single_liquid"; beta = 0.0
    elif beta >= 1:
        phase = "single_vapor";  beta = 1.0
    else:
        phase = "two_phase"

    out.update({"phase": phase, "beta": float(beta),
                "x": tie["x"].copy(), "y": tie["y"].copy()})
    return out


# =============================================================================
# Convenience wrappers
# =============================================================================
def make_components_co2_h2o() -> dict:
    """Omega, Tc, Pc, Mw arrays for (CO₂, H₂O) with water last."""
    return {k: v.copy() for k, v in _DEFAULT_CO2_H2O.items()}


def flash_co2_h2o_tpz(
    T: float, P_bar: float, z_co2: float,
    kij12: float | None = None, swc: float | None = None,
    *,
    vshift_co2: float = 0.0,
    vshift_h2o: float = 0.0,
    tol: float = 1e-10,
    maxiter: int = 1000,
) -> dict:
    """
    Convenience wrapper: CO₂(0) + H₂O(1), z_co2 = overall CO₂ mole fraction.

    kij12 : binary interaction parameter.
        Default None → kij_ecpa(T) when PARAM_SET="eCPA", else 0.
    swc   : CO₂–H₂O cross-association strength (maps to S₁₄ in the eCPA notebook).
        Default None → s14_ecpa(T) when PARAM_SET="eCPA", else 0.
    vshift_co2, vshift_h2o : Péneloux volume shifts [m³/mol] for CO₂ and H₂O.
        Applied as Vm_corrected = Vm_EoS + Σ xᵢ·cᵢ; rho_mass updated in-place.
        Default 0.0 (no shift).
    """
    if kij12 is None:
        kij12 = kij_ecpa(T) if PARAM_SET == "eCPA" else 0.0
    if swc is None:
        swc = s14_ecpa(T) if PARAM_SET == "eCPA" else 0.0
    comps = make_components_co2_h2o()
    z = np.array([z_co2, 1.0 - z_co2])
    kw = dict(Omega=comps["Omega"], Tc=comps["Tc"], Pc=comps["Pc"], Mw=comps["Mw"],
              kij12=kij12, tol=tol, maxiter=maxiter)

    result = flash_tpz_two_comp(T=T, P_bar=P_bar, z=z, swc=swc, **kw)

    # Near S₁₄(T)≈0 (T≈288–292 K) the SSI loop can oscillate due to numerical
    # noise from the tiny delta₁ = swc·κ_W·ε_W/b_1.  If convergence failed and
    # |swc| is below a small threshold, retry with swc=0 (physically equivalent
    # in this regime: cross-association is negligible at the sign-change).
    if not result["tie"]["converged"] and 0 < abs(swc) < 0.005:
        result = flash_tpz_two_comp(T=T, P_bar=P_bar, z=z, swc=0.0, **kw)

    # Apply Péneloux volume shift to rho_mass if requested.
    # c [m³/mol] → L/mol (*1000); Vm in tie is implicitly L/mol (R_BAR_L units).
    # rho_mass [g/L]: Vm_corr = M_mix/rho_old + Σ xᵢcᵢ; rho_new = M_mix/Vm_corr
    if (vshift_co2 != 0.0 or vshift_h2o != 0.0) and result["tie"]["converged"]:
        tie = result["tie"]
        Mw_arr = comps["Mw"]                           # [M_CO2, M_H2O] g/mol
        c = np.array([vshift_co2, vshift_h2o]) * 1000  # m³/mol → L/mol
        for i, xi in enumerate([result.get("x", tie["x"]),
                                 result.get("y", tie["y"])]):
            rho_old = float(tie["rho_mass"][i])
            if rho_old <= 0:
                continue
            M_mix = float(np.dot(Mw_arr, xi))
            Vm_corr = M_mix / rho_old + float(np.dot(c, xi))
            tie["rho_mass"][i] = M_mix / Vm_corr if Vm_corr > 0 else rho_old

    return result


__all__ = [
    "PARAM_SET", "MIXING", "G_ETA",
    "R_BAR_L",
    "wilson_K_init",
    "kij_ecpa", "s14_ecpa",
    "ChiChi", "ZChi",
    "tie_line_two_comp",
    "flash_tpz_two_comp",
    "make_components_co2_h2o",
    "flash_co2_h2o_tpz",
]
