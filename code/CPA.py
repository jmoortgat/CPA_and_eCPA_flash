"""
CPA.py — two-component CPA flash for CO₂ + H₂O, salt-free.

This revision replaces the original van der Waals one-fluid (vdW1f) mixing
rule with the Huron–Vidal (HV) mixing rule used in the eCPA notebook
(Coelho, Franco & Firoozabadi, IECR 2025), and replaces the Carnahan–Starling
radial distribution function with the simplified form g(η) = 1/(1−1.9η) used
in the same reference.  With these changes and the eCPA parameter set the
salt-free flash reproduces the eCPA ELV solution at ms→0.

Key differences from the previous CPA.py
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
  "CPA_orig"  — original MATLAB parameters.  Use MIXING="vdW1f", G_ETA="CS",
             kij12=0.0 for the best standalone CPA_orig results.
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

Author: J. Moortgat. Derived from CPA_040926.py and eCPA_VLE_04102026.ipynb.
"""

from __future__ import annotations
import numpy as np

# =============================================================================
# PARAMETER SET SELECTION
# =============================================================================
PARAM_SET: str = "eCPA"   # "CPA_orig" | "eCPA"

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
# CPA original parameter set  (original MATLAB / ParaCompEOS.m)
# =============================================================================
_PARAMS_CPA_ORIG = {
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
_VALID_PARAM = {"CPA_orig", "eCPA"}
_VALID_MIX   = {"HV", "vdW1f"}
_VALID_G     = {"MOD", "CS"}

if PARAM_SET not in _VALID_PARAM:
    raise ValueError(f"PARAM_SET must be one of {_VALID_PARAM}")
if MIXING not in _VALID_MIX:
    raise ValueError(f"MIXING must be one of {_VALID_MIX}")
if G_ETA not in _VALID_G:
    raise ValueError(f"G_ETA must be one of {_VALID_G}")

_P = _PARAMS_CPA_ORIG if PARAM_SET == "CPA_orig" else _PARAMS_ECPA

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


def _d2gdeta2(eta: float) -> float:
    """Second derivative d²g/dη²."""
    if G_ETA == "MOD":
        return 2.0 * 1.9**2 / (1.0 - 1.9 * eta) ** 3
    return 3.0 * (3.0 - eta) / (1.0 - eta) ** 5


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
def _chi1_from_Z_chi(Z, chi_w, B, nw, Kapa, Eps, swc):
    """
    Compute χ_CO₂ from known Z, χ_H₂O and EOS parameters.

    Used to complete a (Z, χ_w) table warm-start into a full (Z, χ_w, χ_c)
    triple before the first SSI ZChi call.
    """
    if abs(swc) < 1e-30:
        return 1.0
    eta    = B / (4.0 * float(Z))
    delta1 = swc * _g(eta) * float(Kapa) * np.expm1(float(Eps))
    D = float(Z) + 2.0 * float(nw) * float(chi_w) * delta1
    if D <= 0.0:
        return 1.0
    _hi = 10.0 if delta1 < 0.0 else 1.0
    return float(np.clip(float(Z) / D, 0.0, _hi))


def _funz_combined(Z, A, B, n, Kapa, Eps, swc):
    Chi, Chi1 = ChiChi(Z, B, n, Kapa, Eps, swc)
    f = FunZ(Z, A, B, n, Chi, Chi1)
    return f, Chi, Chi1


_ZCHI_NEWTON_TOL     = 1e-12
_ZCHI_NEWTON_MAXITER = 20


def _ZChi_newton(A, B, n, Kapa, Eps, swc, Z0, Chi0, Chi10):
    """
    Newton refinement of (Z, χ_H₂O, χ_CO₂) from a warm-start.

    Solves the 3×3 system simultaneously:
      F₁ = FunZ(Z, χ_w, χ_c)                              = 0
      F₂ = c₃·χ_w³ + c₂·χ_w² + c₁·χ_w + c₀              = 0  (χ_H₂O cubic)
      F₃ = χ_c·(Z + 2·n_w·χ_w·δ₁) − Z                    = 0  (χ_CO₂ closure)

    When swc=0: χ_c=1 always; reduces to a 2×2 system (Z, χ_w).

    The Jacobian is the same 3×3 implicit-function system used in
    _dlnphi_dx_phase (validated in _debug_jacobian.py / _test_jacobian.py).

    Raises RuntimeError on failure; caller should fall back to scan+Brent.
    """
    n  = np.asarray(n, dtype=float).reshape(2)
    nc, nw = n[0], n[1]
    Z    = float(Z0);   Chi  = float(Chi0);  Chi1 = float(Chi10)
    _no_cross = abs(swc) < 1e-30
    max_resid = np.inf

    for _ in range(_ZCHI_NEWTON_MAXITER):
        if Z <= B:
            raise RuntimeError("Z ≤ B during Newton")

        eta   = B / (4.0 * Z)
        g     = _g(eta);  dg = _dgdeta(eta);  d2g = _d2gdeta2(eta)
        delta  = g * Kapa * np.expm1(Eps)
        delta1 = swc * delta
        dgog      = dg / g
        dlogd_dZ  = dgog * (-B / (4.0 * Z * Z))
        ZmB = Z - B;  ZpB = Z + B
        h_val = 1.0 + eta * dgog
        dh    = dgog + eta * (d2g / g - dgog * dgog)
        Sigma = nw * (Chi - 1.0) + nc * (Chi1 - 1.0)
        F1    = Z - Z / ZmB + A / ZpB - 2.0 * h_val * Sigma

        if _no_cross:
            # 2×2: χ_c = 1 fixed, quadratic F₂ in χ_w
            c2b = 2.0 * nw * Z * delta
            F2  = c2b * Chi * Chi + Z * Z * Chi - Z * Z
            max_resid = max(abs(F1), abs(F2))
            if max_resid < _ZCHI_NEWTON_TOL:
                break
            dc2b_dZ = 2.0 * nw * delta * (1.0 + Z * dlogd_dZ)
            J11 = 1.0 + B / ZmB**2 - A / ZpB**2 + dh * B / (2.0 * Z * Z) * Sigma
            J12 = -2.0 * h_val * nw
            J21 = dc2b_dZ * Chi * Chi + 2.0 * Z * Chi - 2.0 * Z
            J22 = 2.0 * c2b * Chi + Z * Z
            try:
                dv = np.linalg.solve(
                    np.array([[J11, J12], [J21, J22]]),
                    -np.array([F1, F2]))
            except np.linalg.LinAlgError:
                raise RuntimeError("Singular 2×2 Jacobian")
            Z    = max(Z + float(dv[0]), B + 1e-10)
            Chi  = float(np.clip(Chi + float(dv[1]), 0.0, 1.0))
            Chi1 = 1.0

        else:
            # 3×3 full system
            c3 = 4.0 * nw * nw * delta * delta1
            c2 = 2.0 * nw * Z * (delta + delta1)
            c1 = 2.0 * delta1 * Z * (nc - nw) + Z * Z
            c0 = -(Z * Z)
            F2 = c3 * Chi**3 + c2 * Chi**2 + c1 * Chi + c0
            D  = Z + 2.0 * nw * Chi * delta1
            if D <= 0.0:
                raise RuntimeError("D ≤ 0")
            F3 = Chi1 * D - Z
            max_resid = max(abs(F1), abs(F2), abs(F3))
            if max_resid < _ZCHI_NEWTON_TOL:
                break

            # Row 1
            J11 = 1.0 + B / ZmB**2 - A / ZpB**2 + dh * B / (2.0 * Z * Z) * Sigma
            J12 = -2.0 * h_val * nw
            J13 = -2.0 * h_val * nc
            # Row 2
            dc3_dZ = c3 * 2.0 * dlogd_dZ
            dc2_dZ = 2.0 * nw * (delta + delta1) + c2 * dlogd_dZ
            dc1_dZ = (2.0 * delta1 * (nc - nw)
                      + 2.0 * delta1 * Z * (nc - nw) * dlogd_dZ + 2.0 * Z)
            dc0_dZ = -2.0 * Z
            J21 = dc3_dZ * Chi**3 + dc2_dZ * Chi**2 + dc1_dZ * Chi + dc0_dZ
            J22 = 3.0 * c3 * Chi**2 + 2.0 * c2 * Chi + c1
            # Row 3
            dD_dZ = 1.0 + 2.0 * nw * Chi * delta1 * dlogd_dZ
            J31   = Chi1 * dD_dZ - 1.0
            J32   = Chi1 * 2.0 * nw * delta1
            J33   = D
            try:
                dv = np.linalg.solve(
                    np.array([[J11, J12, J13],
                               [J21, J22, 0.0],
                               [J31, J32, J33]]),
                    -np.array([F1, F2, F3]))
            except np.linalg.LinAlgError:
                raise RuntimeError("Singular 3×3 Jacobian")
            Z    = max(Z + float(dv[0]), B + 1e-10)
            Chi  = float(np.clip(Chi + float(dv[1]), 0.0, 1.0))
            _chi1_hi = 10.0 if delta1 < 0.0 else 1.0
            Chi1 = float(np.clip(Chi1 + float(dv[2]), 0.0, _chi1_hi))

    else:
        raise RuntimeError(
            f"ZChi Newton did not converge in {_ZCHI_NEWTON_MAXITER} iterations "
            f"(max_resid={max_resid:.2e})")

    return Z, Chi, Chi1


def _ZChi_scan_brent(A, B, n, Kapa, Eps, swc):
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


def ZChi(A, B, n, Kapa, Eps, swc, Z0=None, Chi0=None, Chi10=None):
    """
    Find Z and (χ_H₂O, χ_CO₂).

    When a warm-start (Z0, Chi0, Chi10) is provided, attempts 3×3 Newton
    (_ZChi_newton) first — typically 2–4 iterations from a close starting
    point, replacing the ~42 function evaluations of the coarse scan + Brent.
    Falls back to _ZChi_scan_brent on Newton failure or when no warm start
    is available (first SSI iteration).
    """
    if Z0 is not None and Chi0 is not None and Chi10 is not None:
        try:
            Z, Chi, Chi1 = _ZChi_newton(A, B, n, Kapa, Eps, swc, Z0, Chi0, Chi10)
            if Z > B and np.isfinite(Z) and np.isfinite(Chi) and np.isfinite(Chi1):
                f = FunZ(Z, A, B, n, Chi, Chi1)
                if np.isfinite(f) and abs(f) < 1e-7:
                    return Z, Chi, Chi1
        except Exception:
            pass
    return _ZChi_scan_brent(A, B, n, Kapa, Eps, swc)


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
# Analytical composition derivatives of lnφ  (for Newton flash)
# =============================================================================
def _dlnphi_dx_phase(ep, Z, Chi, Chi1, swc, is_aqueous):
    r"""
    Analytical d(lnφ_i)/d(x_CO₂) for i=CO₂,H₂O in one phase.

    Accounts for all implicit dependencies: Z(x), χ(x), χ₁(x)
    through the coupled EOS + association balance equations.
    Uses implicit function theorem on the 3×3 system (F_Z, F_χ, G).

    Parameters
    ----------
    ep : dict from _eos_aq or _eos_vap
    Z, Chi, Chi1 : floats (converged values)
    swc : float (cross-association S14)
    is_aqueous : bool

    Returns
    -------
    dlnphi : (2,) array  [dlnφ_CO₂/dx_CO₂ , dlnφ_H₂O/dx_CO₂]
    """
    A, B  = ep["A"], ep["B"]
    A1, B1 = ep["A1"], ep["B1"]
    A4, B4 = ep["A4"], ep["B4"]
    x4, x1 = ep["x4"], ep["x1"]          # CO₂ , H₂O
    nc, nw = x4, x1
    Kapa, Eps = ep["Kapa"], ep["Eps"]
    kij12 = ep["kij12"]
    Z = float(Z); Chi = float(Chi); Chi1 = float(Chi1)
    Bi = np.array([B4, B1])               # indexed [CO₂, H₂O]

    # ── packing fraction & radial distribution function ──────────────
    eta  = B / (4.0 * Z)
    g    = _g(eta)
    dg   = _dgdeta(eta)
    d2g  = _d2gdeta2(eta)
    dgog = dg / g                          # dg/g
    h    = 1.0 + eta * dgog                # 1 + η·g'/g
    dh   = dgog + eta * (d2g / g - dgog**2)  # dh/dη

    # ── association strengths ────────────────────────────────────────
    expm1_Eps = np.expm1(Eps)
    delta  = g * Kapa * expm1_Eps
    delta1 = swc * delta
    Sigma  = nw * (Chi - 1.0) + nc * (Chi1 - 1.0)

    # ── log-derivatives of δ w.r.t. Z and B ─────────────────────────
    #  δ = g(η)·κ·(eᵉ−1),  η = B/(4Z)
    dlogd_dZ = dgog * (-B / (4.0 * Z**2))   # d(ln δ)/dZ
    dlogd_dB = dgog / (4.0 * Z)             # d(ln δ)/dB

    # ── mixing-rule derivatives  dA/dx₀, dB/dx₀ ────────────────────
    dB_dx = B4 - B1

    if is_aqueous and MIXING == "HV":
        hv   = ep["hv"]
        U14, U41, gE = hv["U14"], hv["U41"], hv["gE"]
        a1d, b1d = ep["a1"], ep["b1"]
        a4d, b4d = ep["a4"], ep["b4"]
        bd       = ep["b"]
        RT       = ep["R"] * ep["T"]
        cross_BU = b1d * U14 + b4d * U41         # dimensional

        # dgE/dx₄:  gE = x₁ x₄ cross_BU / b
        dgE_dx = cross_BU * ((1.0 - 2.0*x4)*bd
                             - x1*x4*(b4d - b1d)) / bd**2
        # da/dx₄:  a = b·S,  S = x₁a₁/b₁ + x₄a₄/b₄ − gE/ln2
        S     = x1*a1d/b1d + x4*a4d/b4d - gE / _LN2
        dS_dx = -a1d/b1d + a4d/b4d - dgE_dx / _LN2
        db_dx = b4d - b1d
        da_dx = db_dx * S + bd * dS_dx
        dA_dx = da_dx * ep["P_bar"] / (RT)**2
    else:
        A14   = ep.get("A14", np.sqrt(A1 * A4) * (1.0 - kij12))
        dA_dx = 2.0 * (nc * (A4 - A14) + nw * (A14 - A1))

    # ── F_Z partials ────────────────────────────────────────────────
    ZmB = Z - B;  ZpB = Z + B
    if ZmB < 1e-30 or abs(ZpB) < 1e-30:
        return np.zeros(2)

    dFZ_dZ    = (1.0 + B / ZmB**2 - A / ZpB**2
                 + dh * B / (2.0 * Z**2) * Sigma)
    dFZ_dChi  = -2.0 * h * nw
    dFZ_dChi1 = -2.0 * h * nc

    # ∂F_Z/∂x₀  (at fixed Z, χ, χ₁; A, B, nw, nc, η all vary)
    dFZ_dx = (dA_dx / ZpB
              + dB_dx * (-Z / ZmB**2 - A / ZpB**2
                         - dh / (2.0 * Z) * Sigma)
              + 2.0 * h * ((Chi - 1.0) - (Chi1 - 1.0)))

    # ── F_χ partials  (cubic association balance) ───────────────────
    c3 = 4.0 * nw**2 * delta * delta1
    c2 = 2.0 * nw * Z * (delta + delta1)
    c1 = 2.0 * delta1 * Z * (nc - nw) + Z**2
    # c0 = -Z**2  (not needed beyond building dF_chi)

    dFchi_dChi = 3.0 * c3 * Chi**2 + 2.0 * c2 * Chi + c1

    # dF_χ/dZ  (Z appears directly in c2, c1, c0 and through δ(η(Z)))
    dc3_dZ = c3 * 2.0 * dlogd_dZ
    dc2_dZ = (2.0 * nw * (delta + delta1)
              + 2.0 * nw * Z * (delta + delta1) * dlogd_dZ)
    dc1_dZ = (2.0 * delta1 * (nc - nw)
              + 2.0 * delta1 * Z * (nc - nw) * dlogd_dZ
              + 2.0 * Z)
    dc0_dZ = -2.0 * Z
    dFchi_dZ = (dc3_dZ * Chi**3 + dc2_dZ * Chi**2
                + dc1_dZ * Chi + dc0_dZ)

    # dF_χ/dx₀  (through nw, nc directly + δ(B(x₀)))
    dc3_dB = c3 * 2.0 * dlogd_dB
    dc2_dB = 2.0 * nw * Z * (delta + delta1) * dlogd_dB
    dc1_dB = 2.0 * delta1 * Z * (nc - nw) * dlogd_dB

    dc3_dn = -8.0 * nw * delta * delta1     # dc3/dx₀ (direct nw change)
    dc2_dn = -2.0 * Z * (delta + delta1)    # dc2/dx₀ (direct nw change)
    dc1_dn = 4.0 * delta1 * Z               # dc1/dx₀ (from d(nc−nw)/dx₀=2)

    dFchi_dx = ((dc3_dn + dc3_dB * dB_dx) * Chi**3
                + (dc2_dn + dc2_dB * dB_dx) * Chi**2
                + (dc1_dn + dc1_dB * dB_dx) * Chi)

    # ── G partials:  G = χ₁ − Z / (Z + 2 nw χ δ₁) ─────────────────
    D = Z + 2.0 * nw * Chi * delta1

    dD_dZ   = 1.0 + 2.0 * nw * Chi * delta1 * dlogd_dZ
    dG_dZ   = -(D - Z * dD_dZ) / D**2

    dD_dChi = 2.0 * nw * delta1
    dG_dChi = Z * dD_dChi / D**2

    dG_dChi1 = 1.0

    # ∂G/∂x₀ (through nw and δ₁(B))
    dD_dx = (-2.0 * Chi * delta1
             + 2.0 * nw * Chi * delta1 * dlogd_dB * dB_dx)
    dG_dx = Z * dD_dx / D**2

    # ── solve 3×3 implicit system  J · d = −rhs ────────────────────
    J_impl = np.array([
        [dFZ_dZ,   dFZ_dChi,   dFZ_dChi1],
        [dFchi_dZ, dFchi_dChi, 0.0       ],
        [dG_dZ,    dG_dChi,    dG_dChi1  ],
    ])
    rhs = -np.array([dFZ_dx, dFchi_dx, dG_dx])

    try:
        d_impl = np.linalg.solve(J_impl, rhs)   # [dZ, dχ, dχ₁]/dx₀
    except np.linalg.LinAlgError:
        return np.zeros(2)

    dZ_dx0, dChi_dx0, dChi1_dx0 = d_impl
    deta_dx0 = -B / (4.0 * Z**2) * dZ_dx0 + dB_dx / (4.0 * Z)

    # ── partial derivatives of lnφ ──────────────────────────────────
    # Physical part
    lnZpBZ = np.log(max(ZpB / max(Z, _LOG_GUARD), _LOG_GUARD))
    phys_c = B / ZmB - A / ZpB
    d_phys_c_dZ = -B / ZmB**2 + A / ZpB**2
    d_lnZpBZ_dZ = -B / (Z * ZpB)
    d_lnZpBZ_dB = 1.0 / ZpB

    if is_aqueous and MIXING == "HV":
        hv   = ep["hv"]
        U14, U41, gE = hv["U14"], hv["U41"], hv["gE"]
        bd   = ep["b"]
        RT   = ep["R"] * ep["T"]
        cross_BU = ep["B1"] * U14 + ep["B4"] * U41   # must match _lnphi_aq (dimensionless Bi)

        bracket = np.array([
            A4/B4 - 1.0/(B*_LN2)*(x1*cross_BU/RT - B4*gE/RT),   # CO₂
            A1/B1 - 1.0/(B*_LN2)*(x4*cross_BU/RT - B1*gE/RT),   # H₂O
        ])

        # ∂lnφ_phys/∂Z  (bracket doesn't depend on Z)
        dlnphi_phys_dZ = np.array([
            -1.0/ZmB + B4/B*d_phys_c_dZ - d_lnZpBZ_dZ * bracket[0],
            -1.0/ZmB + B1/B*d_phys_c_dZ - d_lnZpBZ_dZ * bracket[1],
        ])

        # ∂lnφ_phys/∂x₀  (at fixed Z, χ, χ₁):
        #   d(-ln(Z-B))/dx₀ = dB_dx/(Z-B)
        #   d(Bi/B·phys_c)/dx₀ = Bi/B·[dB_dx/(Z-B)² + dA_dx/(Z+B)²
        #                                - dB_dx·A/(Z+B)² ... ]
        #   Easier: compute numerically at this level?  No — let's do it term by term.
        #
        # term1 = -ln(Z-B)  →  d/dx₀ = dB_dx/(Z-B)
        # term2 = Bi/B·phys_c = Bi/(Z-B) - Bi·A/(B·(Z+B))
        #   d(Bi/(Z-B))/dx₀ = Bi·dB_dx/(Z-B)²
        #   d(-Bi·A/(B·(Z+B)))/dx₀ = -Bi·[dA_dx/(B·ZpB)
        #                              - A·(dB_dx·ZpB + B·dB_dx)/(B·ZpB)²]
        #     = -Bi·[dA_dx/(B·ZpB) - A·dB_dx·(B+ZpB)/(B²·ZpB²)]
        #     = -Bi·dA_dx/(B·ZpB) + Bi·A·dB_dx·(2*B+Z)/(B²·ZpB²)
        #
        # term3 = -lnZpBZ·bracket_i
        #   d/dx₀ = -d_lnZpBZ_dB·dB_dx·bracket_i - lnZpBZ·d_bracket_dx_i

        # d_bracket/dx₀
        term_bkt0 = x1 * cross_BU / RT - B4 * gE / RT
        term_bkt1 = x4 * cross_BU / RT - B1 * gE / RT

        d_bracket_dx = np.array([
            -1.0/(B*_LN2)*(-cross_BU/RT - B4*dgE_dx/RT)
            + dB_dx/(B**2*_LN2) * term_bkt0,
            -1.0/(B*_LN2)*(cross_BU/RT - B1*dgE_dx/RT)
            + dB_dx/(B**2*_LN2) * term_bkt1,
        ])

        d_term1_dx = dB_dx / ZmB
        d_term2_CO2_dx = (B4 * dB_dx / ZmB**2
                          - B4 * dA_dx / (B * ZpB)
                          + B4 * A * dB_dx * (2*B + Z) / (B**2 * ZpB**2))
        d_term2_H2O_dx = (B1 * dB_dx / ZmB**2
                          - B1 * dA_dx / (B * ZpB)
                          + B1 * A * dB_dx * (2*B + Z) / (B**2 * ZpB**2))
        d_term3_dx = np.array([
            -d_lnZpBZ_dB * dB_dx * bracket[0] - lnZpBZ * d_bracket_dx[0],
            -d_lnZpBZ_dB * dB_dx * bracket[1] - lnZpBZ * d_bracket_dx[1],
        ])

        dlnphi_phys_dx = np.array([
            d_term1_dx + d_term2_CO2_dx + d_term3_dx[0],
            d_term1_dx + d_term2_H2O_dx + d_term3_dx[1],
        ])
    else:
        # vdW1f (vapor phase)
        A14_v = ep.get("A14", np.sqrt(A1 * A4) * (1.0 - kij12))
        dA_sum = np.array([x1*A14_v + x4*A4, x1*A1 + x4*A14_v])
        coef_i = A / B * (Bi / B - 2.0 * dA_sum / A)  # A/B·(Bi/B − 2Σ/A)
        ln1pBZ = lnZpBZ  # same quantity

        dlnphi_phys_dZ = np.array([
            -1.0/ZmB + B4/B*d_phys_c_dZ + coef_i[0]*(-B/(Z*ZpB)),
            -1.0/ZmB + B1/B*d_phys_c_dZ + coef_i[1]*(-B/(Z*ZpB)),
        ])

        # ∂lnφ_phys/∂x₀ at fixed Z:
        # d(-ln(Z-B))/dx₀ = dB_dx/(Z-B)
        # d(Bi/B·phys_c)/dx₀ [same as HV term2 above]
        # d(A/B·(Bi/B−2dA_sum_i/A)·ln1pBZ)/dx₀:
        #   the multiplier and ln1pBZ both depend on x₀

        d_term1_dx = dB_dx / ZmB
        d_term2_CO2_dx = (B4 * dB_dx / ZmB**2
                          - B4 * dA_dx / (B * ZpB)
                          + B4 * A * dB_dx * (2*B+Z) / (B**2 * ZpB**2))
        d_term2_H2O_dx = (B1 * dB_dx / ZmB**2
                          - B1 * dA_dx / (B * ZpB)
                          + B1 * A * dB_dx * (2*B+Z) / (B**2 * ZpB**2))

        # dA_sum_i/dx₀:  dA_sum_CO2 = x1·A14+x4·A4 → d/dx₀ = -A14+A4
        #                 dA_sum_H2O = x1·A1+x4·A14 → d/dx₀ = -A1+A14
        d_dAsum_dx = np.array([A4 - A14_v, A14_v - A1])

        # d[A/B·(Bi/B-2S/A)·ln]/dx₀:  product rule on (A/B)·coef_inner·ln
        #   coef_inner_i = Bi/B - 2·dA_sum_i/A
        coef_inner = Bi / B - 2.0 * dA_sum / A

        # d(A/B)/dx₀ = (dA_dx·B - A·dB_dx)/B²
        dAoB_dx = (dA_dx * B - A * dB_dx) / B**2
        # d(coef_inner_i)/dx₀ = -Bi·dB_dx/B² - 2(d_dAsum·A-dA_sum·dA_dx)/A²
        d_coef_inner_dx = (-Bi * dB_dx / B**2
                           - 2.0 * (d_dAsum_dx * A
                                    - dA_sum * dA_dx) / A**2)
        # d(ln1pBZ)/dx₀ = d_lnZpBZ_dB · dB_dx
        d_ln_dx = d_lnZpBZ_dB * dB_dx

        d_term3_dx = (dAoB_dx * coef_inner * ln1pBZ
                      + A / B * d_coef_inner_dx * ln1pBZ
                      + A / B * coef_inner * d_ln_dx)

        dlnphi_phys_dx = np.array([
            d_term1_dx + d_term2_CO2_dx + d_term3_dx[0],
            d_term1_dx + d_term2_H2O_dx + d_term3_dx[1],
        ])

    # ── Association part ────────────────────────────────────────────
    # lnφ_ass_i = 4·ln(χ_self_i) + Bi·dg/(2gZ) · Σ
    # (where 4·Σ/8 = Σ/2, factor comes from 4 sites, 8 in denominator)
    ass_coef = dg / (2.0 * g * Z)          # = dg/(2gZ)

    # ∂lnφ_ass_i/∂Z  (η varies with Z; χ, χ₁ fixed)
    # d[Bi·dg/(2gZ)·Σ]/dZ = Bi·Σ · d[dg/(2gZ)]/dZ
    # d[dg/(2gZ)]/dZ = [d²g/dη·dη/dZ·gZ - dg·(dg/dη·dη/dZ·Z+g)]/(gZ)²
    #  = [(d2g·(-B/(4Z²))·gZ - dg·(dg·(-B/(4Z²))·Z + g)] / (gZ)²
    # Simpler: dg/(2gZ) = dgog/(2Z)
    # d[dgog/(2Z)]/dZ = [(d2g/g-dgog²)·(-B/(4Z²))]/(2Z) - dgog/(2Z²)
    #                 = -[B·(d2g/g-dgog²)/(4Z²) + dgog]/(2Z)
    # Actually just compute d(ass_coef)/dZ directly:
    # ass_coef = dg/(2gZ),  let's diff:
    d_asscoef_deta = (d2g * g - dg**2) / (2.0 * g**2 * Z)  # d[dg/(2gZ)]/dη · (1, not /dZ)
    deta_dZ = -B / (4.0 * Z**2)
    d_asscoef_dZ_from_eta = d_asscoef_deta * deta_dZ
    d_asscoef_dZ_from_Z   = -dg / (2.0 * g * Z**2)
    d_asscoef_dZ = d_asscoef_dZ_from_eta + d_asscoef_dZ_from_Z

    dlnphi_ass_dZ = Bi * Sigma * d_asscoef_dZ

    # ∂lnφ_ass_i/∂χ :  4/χ for H₂O (i=1) + Bi·ass_coef·nw
    dlnphi_ass_dChi = np.array([
        Bi[0] * ass_coef * nw,                      # CO₂
        4.0 / max(Chi, _LOG_GUARD) + Bi[1] * ass_coef * nw,   # H₂O
    ])

    # ∂lnφ_ass_i/∂χ₁ : 4/χ₁ for CO₂ (i=0) + Bi·ass_coef·nc
    dlnphi_ass_dChi1 = np.array([
        4.0 / max(Chi1, _LOG_GUARD) + Bi[0] * ass_coef * nc,  # CO₂
        Bi[1] * ass_coef * nc,                       # H₂O
    ])

    # ∂lnφ_ass_i/∂x₀  (at fixed Z, χ, χ₁):
    #   through Σ: dΣ/dx₀ = -(χ-1)+(χ₁-1)
    #   through ass_coef: depends on η → B → x₀
    deta_dB = 1.0 / (4.0 * Z)
    d_asscoef_deta_val = d_asscoef_deta  # already computed
    d_asscoef_dx = d_asscoef_deta_val * deta_dB * dB_dx
    dSigma_dx = -(Chi - 1.0) + (Chi1 - 1.0)

    dlnphi_ass_dx = Bi * (d_asscoef_dx * Sigma + ass_coef * dSigma_dx)

    # ── Chain rule: total dlnφ_i/dx₀ ───────────────────────────────
    dlnphi_dZ   = dlnphi_phys_dZ   + dlnphi_ass_dZ
    dlnphi_dChi = dlnphi_ass_dChi                    # phys doesn't depend on χ
    dlnphi_dChi1= dlnphi_ass_dChi1                   # phys doesn't depend on χ₁
    dlnphi_dx   = dlnphi_phys_dx   + dlnphi_ass_dx   # direct composition

    return (dlnphi_dZ * dZ_dx0
            + dlnphi_dChi * dChi_dx0
            + dlnphi_dChi1 * dChi1_dx0
            + dlnphi_dx)


def _dlnphi_dx_numerical(T, P_bar, comp, kij12, swc, is_aqueous,
                          h=1e-7):
    """Numerical d(lnφ)/d(x_CO₂) via central differences (for verification)."""
    x0 = float(comp[0])
    results = []
    for sign in (+1, -1):
        xp = np.array([x0 + sign * h, 1.0 - (x0 + sign * h)])
        xp = np.clip(xp, 1e-15, 1.0 - 1e-15)
        if is_aqueous:
            ep_p = _eos_aq(T, P_bar, xp, kij12)
        else:
            ep_p = _eos_vap(T, P_bar, xp, kij12)
        Zp, Chip, Chi1p = ZChi(ep_p["A"], ep_p["B"], xp,
                                ep_p["Kapa"], ep_p["Eps"], swc)
        if Zp == 0.0:
            return np.zeros(2)
        if is_aqueous:
            lnphi_p = _lnphi_aq(ep_p, Zp, Chip, Chi1p)
        else:
            lnphi_p = _lnphi_vap(ep_p, Zp, Chip, Chi1p)
        results.append(lnphi_p)
    return (results[0] - results[1]) / (2.0 * h)


def newton_jacobian(lnK, T, P_bar, kij12, swc,
                    ep_aq, ep_vap,
                    Zx, Chix, Chi1x, Zy, Chiy, Chi1y,
                    x, y):
    """
    Analytical 2×2 Jacobian  J[i,j] = d(lng_i)/d(lnK_j).

    Uses analytical dlnφ/dx for each phase plus algebraic dx/dlnK, dy/dlnK.
    """
    K = np.exp(lnK)
    denom = K[1] - K[0]
    denom2 = denom**2

    # dx₀/dlnK_j  (x₀ = x_CO₂)
    dx0_dlnK = np.array([
        K[0] * (K[1] - 1.0) / denom2,     # d/dlnK_CO₂
        K[1] * (1.0 - K[0]) / denom2,      # d/dlnK_H₂O
    ])

    # dy_i/dlnK_j  (2×2 matrix, [component, lnK index])
    dy_dlnK = np.array([
        [K[0] * dx0_dlnK[0] + y[0],   K[0] * dx0_dlnK[1]       ],  # CO₂
        [-K[1] * dx0_dlnK[0],          -K[1] * dx0_dlnK[1] + y[1]],  # H₂O
    ])

    # dx_i/dlnK_j  (2×2 matrix)
    dx_dlnK = np.array([
        [dx0_dlnK[0],   dx0_dlnK[1]  ],   # CO₂
        [-dx0_dlnK[0], -dx0_dlnK[1]  ],   # H₂O
    ])

    # Analytical dlnφ/dx₀ for each phase  (2-element vectors)
    dphi_x = _dlnphi_dx_phase(ep_aq,  Zx, Chix,  Chi1x,  swc, True)
    dphi_y = _dlnphi_dx_phase(ep_vap, Zy, Chiy,  Chi1y,  swc, False)

    # J[i,j] = dphi_y[i]·dy₀/dlnK_j + (1/y_i)·dy_i/dlnK_j
    #         - dphi_x[i]·dx₀/dlnK_j - (1/x_i)·dx_i/dlnK_j
    J = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            J[i, j] = (dphi_y[i] * dy_dlnK[0, j]       # dy₀/dlnK_j
                        + (1.0 / max(y[i], _LOG_GUARD)) * dy_dlnK[i, j]
                        - dphi_x[i] * dx_dlnK[0, j]     # dx₀/dlnK_j
                        - (1.0 / max(x[i], _LOG_GUARD)) * dx_dlnK[i, j])
    return J


def newton_jacobian_numerical(lnK, T, P_bar, kij12, swc, kw_base,
                              h=1e-7):
    """Numerical 2×2 Jacobian of lng w.r.t. lnK (for verification)."""
    def _eval_lng(lnK_):
        K_ = np.exp(lnK_)
        d_ = K_[1] - K_[0]
        if abs(d_) < 1e-14:
            return np.full(2, np.nan)
        x_ = np.array([(K_[1]-1.0)/d_, (K_[0]-1.0)/(-d_)])  # CO₂, H₂O
        x_ = np.clip(x_, 1e-15, 1.0-1e-15)
        y_ = K_ * x_
        y_ = np.clip(y_, 1e-15, 1.0-1e-15)

        ep_a = _eos_aq(T, P_bar, x_, kij12)
        ep_v = _eos_vap(T, P_bar, y_, kij12)
        Zx_, Cx_, C1x_ = ZChi(ep_a["A"], ep_a["B"], x_,
                               ep_a["Kapa"], ep_a["Eps"], swc)
        Zy_, Cy_, C1y_ = ZChi(ep_v["A"], ep_v["B"], y_,
                               ep_v["Kapa"], ep_v["Eps"], swc)
        if Zx_ == 0.0 or Zy_ == 0.0:
            return np.full(2, np.nan)
        lnphi_x_ = _lnphi_aq(ep_a, Zx_, Cx_, C1x_)
        lnphi_y_ = _lnphi_vap(ep_v, Zy_, Cy_, C1y_)
        return (lnphi_y_ + np.log(np.maximum(y_, _LOG_GUARD))
                - lnphi_x_ - np.log(np.maximum(x_, _LOG_GUARD)))

    lng0 = _eval_lng(lnK)
    J = np.zeros((2, 2))
    for j in range(2):
        lnK_p = lnK.copy(); lnK_p[j] += h
        lnK_m = lnK.copy(); lnK_m[j] -= h
        J[:, j] = (_eval_lng(lnK_p) - _eval_lng(lnK_m)) / (2.0 * h)
    return J


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
    accelerated: bool = True,
    accel_method: str = "jex",
    diis_depth: int = 4,
    use_newton: bool = True,
    newton_tol: float = 1e-4,
    max_newton: int = 5,
    ZChi_aq_init: tuple | None = None,
    ZChi_vap_init: tuple | None = None,
) -> dict:
    """
    SSI + Newton tie-line solver: CO₂ (component 0) + H₂O (component 1).

    Aqueous  phase: MIXING flag (HV or vdW1f) + G_ETA flag.
    CO₂-rich phase: always vdW1f + MOD g(η) (when G_ETA="MOD").

    SSI runs until ‖lng‖ < newton_tol (or tol if use_newton=False), then
    Newton polish (analytical Jacobian, max_newton steps) brings it to tol.

    Returns dict: converged, iterations, ssi_iterations, newton_iterations,
      K, x [aq], y [vap], and if converged: Z[0,1], rho_mass[0,1] [kg/L],
      assoc_t[0,1], chi{liq,vap}.
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
    _resid_norm = np.nan   # final ||lng|| at convergence

    # Resolve acceleration method
    _use_jex = False
    _use_diis = False
    if accelerated:
        if accel_method == "diis":
            _use_diis = True
        elif accel_method == "jex":
            _use_jex = True
        # "none" or unrecognised → standard SSI

    # Jex state (Jex et al. 2024, Eq. 12–13)
    _m = 1.0           # step-size factor; starts at 1 (conventional SS)
    _g_prev = None      # fugacity-ratio vector from previous iteration
    _M_CAP_HI = 10.0   # upper cap  (= 1/L with L = 0.1 for 2-phase)

    # DIIS state (Pulay 1980)
    _diis_lnK = []      # history of lnK iterates
    _diis_resid = []    # history of residual vectors (-lng)

    # ZChi warm-start state: (Z, Chi, Chi1) from previous SSI iteration.
    # First iteration uses scan+Brent (no warm start); thereafter Newton.
    _Zx_warm = _Chix_warm = _Chi1x_warm = None
    _Zy_warm = _Chiy_warm = _Chi1y_warm = None

    # Table-provided (Z, χ_w) for the first SSI iteration.  χ_c is computed
    # on the fly after the first _eos_* call (needs Kapa, Eps, B).
    _Zx_tbl   = float(ZChi_aq_init[0])  if ZChi_aq_init  is not None else None
    _Chix_tbl = float(ZChi_aq_init[1])  if ZChi_aq_init  is not None else None
    _Zy_tbl   = float(ZChi_vap_init[0]) if ZChi_vap_init is not None else None
    _Chiy_tbl = float(ZChi_vap_init[1]) if ZChi_vap_init is not None else None

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

        # First iteration only: complete table warm start with χ_c
        if _Zx_warm is None and _Zx_tbl is not None:
            _Chi1x_tbl = _chi1_from_Z_chi(
                _Zx_tbl, _Chix_tbl, ep_aq["B"], x[1],
                ep_aq["Kapa"], ep_aq["Eps"], swc)
            _Zx_warm = _Zx_tbl; _Chix_warm = _Chix_tbl; _Chi1x_warm = _Chi1x_tbl
        if _Zy_warm is None and _Zy_tbl is not None:
            _Chi1y_tbl = _chi1_from_Z_chi(
                _Zy_tbl, _Chiy_tbl, ep_vap["B"], y[1],
                ep_vap["Kapa"], ep_vap["Eps"], swc)
            _Zy_warm = _Zy_tbl; _Chiy_warm = _Chiy_tbl; _Chi1y_warm = _Chi1y_tbl

        Zx, Chix, Chi1x = ZChi(ep_aq["A"],  ep_aq["B"],  x,
                                ep_aq["Kapa"],  ep_aq["Eps"],  swc,
                                Z0=_Zx_warm, Chi0=_Chix_warm, Chi10=_Chi1x_warm)
        Zy, Chiy, Chi1y = ZChi(ep_vap["A"], ep_vap["B"], y,
                                ep_vap["Kapa"], ep_vap["Eps"], swc,
                                Z0=_Zy_warm, Chi0=_Chiy_warm, Chi10=_Chi1y_warm)

        if Zx == 0.0 or Zy == 0.0:
            x[:] = 0; y[:] = 0; break
        _Zx_warm = Zx;  _Chix_warm = Chix;  _Chi1x_warm = Chi1x
        _Zy_warm = Zy;  _Chiy_warm = Chiy;  _Chi1y_warm = Chi1y

        lnphi_x = _lnphi_aq (ep_aq,  Zx, Chix,  Chi1x)
        lnphi_y = _lnphi_vap(ep_vap, Zy, Chiy,  Chi1y)

        # Fugacity ratios: g_i = f_{vap,i} / f_{aq,i}
        # ln g_i = lnphi_y_i + ln(y_i) - lnphi_x_i - ln(x_i)
        lng = lnphi_y + np.log(np.maximum(y, _LOG_GUARD)) \
            - lnphi_x - np.log(np.maximum(x, _LOG_GUARD))

        lnK_old = np.log(np.maximum(K, _LOG_GUARD))

        # --- DIIS acceleration (Pulay 1980) ---
        if _use_diis:
            resid = -lng  # residual: zero at convergence
            _diis_lnK.append(lnK_old.copy())
            _diis_resid.append(resid.copy())
            if len(_diis_lnK) > diis_depth:
                _diis_lnK.pop(0)
                _diis_resid.pop(0)

            _did_diis = False
            n_buf = len(_diis_resid)
            if n_buf >= 2:
                # Build DIIS error matrix B[i,j] = r_i · r_j
                R = np.array(_diis_resid)         # (n_buf, 2)
                B_core = R @ R.T                   # (n_buf, n_buf)
                B = np.zeros((n_buf + 1, n_buf + 1))
                B[:n_buf, :n_buf] = B_core
                B[:n_buf, n_buf] = -1.0
                B[n_buf, :n_buf] = -1.0
                rhs = np.zeros(n_buf + 1)
                rhs[n_buf] = -1.0
                try:
                    sol = np.linalg.solve(B, rhs)
                    c = sol[:n_buf]
                    if np.all(np.isfinite(c)) and np.max(np.abs(c)) < 10.0:
                        X = np.array(_diis_lnK)   # (n_buf, 2)
                        lnK_new = c @ (X + R)      # (2,)
                        lnK_step_diis = np.clip(lnK_new - lnK_old, -5.0, 5.0)
                        # Accept only if DIIS step doesn't blow up residual
                        # (prospective check: estimate new residual direction)
                        lnK_cand = lnK_old + lnK_step_diis
                        step_norm = np.linalg.norm(lnK_step_diis)
                        resid_norm_cur = np.linalg.norm(lng)
                        if step_norm < 10.0 * max(resid_norm_cur, 0.1):
                            lnK_step_acc = lnK_step_diis
                            _did_diis = True
                except np.linalg.LinAlgError:
                    pass

            if not _did_diis:
                # Fallback: standard SSI step
                lnK_step_acc = np.clip(-lng, -5.0, 5.0)

        # --- Jex acceleration (Jex et al. 2024, Eq. 12–13) ---
        elif _use_jex:
            if _g_prev is not None:
                dg = _g_prev - lng
                num_a   = np.dot(_g_prev, _g_prev)
                denom_a = np.dot(_g_prev, dg)
                if abs(denom_a) > 1e-30:
                    _m = abs(num_a / denom_a * _m)
                    _m = float(np.clip(_m, 1.0, _M_CAP_HI))
                else:
                    _m = 1.0
            _g_prev = lng.copy()
            lnK_step_acc = np.clip(_m * (-lng), -5.0, 5.0)

        # --- Standard SSI (no acceleration) ---
        else:
            lnK_step_acc = np.clip(-lng, -5.0, 5.0)
            if np.all(np.abs(lnK_old) < 0.05):
                lnK_step_acc *= 0.5

        K = np.exp(lnK_old + lnK_step_acc)

        # Convergence based on fugacity equilibrium: ||lng|| < tol
        _resid_norm = float(np.linalg.norm(lng))
        if _resid_norm < float(tol):
            converged = True; break
        if use_newton and _resid_norm < float(newton_tol):
            break  # close enough — hand off to Newton polish

    _ssi_iters = int(it + 1)
    _newton_iters = 0

    # ── Newton polish ───────────────────────────────────────────────────────
    # Switch from linear SSI to quadratically-convergent Newton once ‖lng‖
    # drops below newton_tol (default 1e-4).  Uses the analytical 2×2 Jacobian
    # d(lng)/d(lnK) from newton_jacobian().  If Newton diverges or fails, the
    # solver reverts to the pre-Newton state and continues with SSI.
    if use_newton and not converged and np.isfinite(_resid_norm) and _resid_norm < float(newton_tol):
        # Save state so we can revert if Newton diverges
        _K_pre      = K.copy()
        _x_pre      = x.copy()
        _y_pre      = y.copy()
        _resid_pre  = _resid_norm
        _ep_aq_pre  = ep_aq
        _ep_vap_pre = ep_vap
        _Zx_pre     = Zx;  _Chix_pre  = Chix;  _Chi1x_pre = Chi1x
        _Zy_pre     = Zy;  _Chiy_pre  = Chiy;  _Chi1y_pre = Chi1y
        _lng_pre    = lng.copy()

        _newton_ok = False
        lnK = np.log(np.maximum(K, _LOG_GUARD))
        for _nit in range(int(max_newton)):
            J = newton_jacobian(lnK, T, P_bar, kij12, swc,
                                ep_aq, ep_vap, Zx, Chix, Chi1x, Zy, Chiy, Chi1y,
                                x, y)
            try:
                dlnK = np.linalg.solve(J, -lng)
            except np.linalg.LinAlgError:
                break
            dlnK = np.clip(dlnK, -5.0, 5.0)
            lnK = lnK + dlnK
            K = np.exp(lnK)

            denom = K[1] - K[0]
            if abs(denom) < 1e-14:
                break
            x[0] = (K[1] - 1.0) / denom
            x[1] = 1.0 - x[0]
            y[:] = K * x
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))
                    and np.all(x >= -1e-12) and np.all(y >= -1e-12)):
                break
            x = np.clip(x, 0, 1); y = np.clip(y, 0, 1)

            ep_aq  = _eos_aq (T, P_bar, x, kij12)
            ep_vap = _eos_vap(T, P_bar, y, kij12)
            Zx, Chix, Chi1x = ZChi(ep_aq["A"],  ep_aq["B"],  x,
                                    ep_aq["Kapa"],  ep_aq["Eps"],  swc,
                                    Z0=_Zx_warm, Chi0=_Chix_warm, Chi10=_Chi1x_warm)
            Zy, Chiy, Chi1y = ZChi(ep_vap["A"], ep_vap["B"], y,
                                    ep_vap["Kapa"], ep_vap["Eps"], swc,
                                    Z0=_Zy_warm, Chi0=_Chiy_warm, Chi10=_Chi1y_warm)
            if Zx == 0.0 or Zy == 0.0:
                break
            _Zx_warm = Zx;  _Chix_warm = Chix;  _Chi1x_warm = Chi1x
            _Zy_warm = Zy;  _Chiy_warm = Chiy;  _Chi1y_warm = Chi1y

            lnphi_x = _lnphi_aq (ep_aq,  Zx, Chix,  Chi1x)
            lnphi_y = _lnphi_vap(ep_vap, Zy, Chiy,  Chi1y)
            lng = (lnphi_y + np.log(np.maximum(y, _LOG_GUARD))
                   - lnphi_x - np.log(np.maximum(x, _LOG_GUARD)))

            _newton_iters += 1
            _resid_norm = float(np.linalg.norm(lng))
            if _resid_norm < float(tol):
                converged = True
                _newton_ok = True
                break

        # If Newton failed or diverged, revert to pre-Newton state and
        # continue with SSI until convergence (or maxiter).
        if not converged:
            if not _newton_ok or _resid_norm > _resid_pre:
                # Revert
                K = _K_pre; x = _x_pre; y = _y_pre
                ep_aq = _ep_aq_pre; ep_vap = _ep_vap_pre
                Zx = _Zx_pre; Chix = _Chix_pre; Chi1x = _Chi1x_pre
                Zy = _Zy_pre; Chiy = _Chiy_pre; Chi1y = _Chi1y_pre
                lng = _lng_pre; _resid_norm = _resid_pre
                _newton_iters = 0
            _Zx_warm = Zx;  _Chix_warm = Chix;  _Chi1x_warm = Chi1x
            _Zy_warm = Zy;  _Chiy_warm = Chiy;  _Chi1y_warm = Chi1y

            # Continue SSI from current state (re-use same acceleration)
            lnK = np.log(np.maximum(K, _LOG_GUARD))
            lnphi_x = _lnphi_aq (ep_aq,  Zx, Chix,  Chi1x)
            lnphi_y = _lnphi_vap(ep_vap, Zy, Chiy,  Chi1y)
            lng = (lnphi_y + np.log(np.maximum(y, _LOG_GUARD))
                   - lnphi_x - np.log(np.maximum(x, _LOG_GUARD)))
            _fb_m = 1.0; _fb_g_prev = None   # fresh Jex state for fallback
            _fb_diis_lnK = []; _fb_diis_resid = []
            for _sit in range(int(maxiter) - _ssi_iters):
                if _use_jex:
                    if _fb_g_prev is not None:
                        dg = _fb_g_prev - lng
                        num_a   = np.dot(_fb_g_prev, _fb_g_prev)
                        denom_a = np.dot(_fb_g_prev, dg)
                        if abs(denom_a) > 1e-30:
                            _fb_m = abs(num_a / denom_a * _fb_m)
                            _fb_m = float(np.clip(_fb_m, 1.0, _M_CAP_HI))
                        else:
                            _fb_m = 1.0
                    _fb_g_prev = lng.copy()
                    lnK_step = np.clip(_fb_m * (-lng), -5.0, 5.0)
                elif _use_diis:
                    resid = -lng
                    _fb_diis_lnK.append(lnK.copy())
                    _fb_diis_resid.append(resid.copy())
                    if len(_fb_diis_lnK) > diis_depth:
                        _fb_diis_lnK.pop(0); _fb_diis_resid.pop(0)
                    lnK_step = np.clip(-lng, -5.0, 5.0)
                    n_buf = len(_fb_diis_resid)
                    if n_buf >= 2:
                        R = np.array(_fb_diis_resid); X = np.array(_fb_diis_lnK)
                        B_c = R @ R.T
                        B = np.zeros((n_buf+1, n_buf+1))
                        B[:n_buf,:n_buf] = B_c; B[:n_buf,n_buf] = -1; B[n_buf,:n_buf] = -1
                        rhs2 = np.zeros(n_buf+1); rhs2[n_buf] = -1
                        try:
                            sol = np.linalg.solve(B, rhs2); c = sol[:n_buf]
                            if np.all(np.isfinite(c)) and np.max(np.abs(c)) < 10.0:
                                lnK_new = c @ (X + R)
                                step_d = np.clip(lnK_new - lnK, -5.0, 5.0)
                                if np.linalg.norm(step_d) < 10.0 * max(np.linalg.norm(lng), 0.1):
                                    lnK_step = step_d
                        except np.linalg.LinAlgError:
                            pass
                else:
                    lnK_step = np.clip(-lng, -5.0, 5.0)
                lnK = lnK + lnK_step
                K = np.exp(lnK)

                denom = K[1] - K[0]
                if abs(denom) < 1e-14:
                    break
                x[0] = (K[1] - 1.0) / denom
                x[1] = 1.0 - x[0]
                y[:] = K * x
                if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))
                        and np.all(x >= -1e-12) and np.all(y >= -1e-12)):
                    break
                x = np.clip(x, 0, 1); y = np.clip(y, 0, 1)

                ep_aq  = _eos_aq (T, P_bar, x, kij12)
                ep_vap = _eos_vap(T, P_bar, y, kij12)
                Zx, Chix, Chi1x = ZChi(ep_aq["A"],  ep_aq["B"],  x,
                                        ep_aq["Kapa"],  ep_aq["Eps"],  swc,
                                        Z0=_Zx_warm, Chi0=_Chix_warm, Chi10=_Chi1x_warm)
                Zy, Chiy, Chi1y = ZChi(ep_vap["A"], ep_vap["B"], y,
                                        ep_vap["Kapa"], ep_vap["Eps"], swc,
                                        Z0=_Zy_warm, Chi0=_Chiy_warm, Chi10=_Chi1y_warm)
                if Zx == 0.0 or Zy == 0.0:
                    break
                _Zx_warm = Zx;  _Chix_warm = Chix;  _Chi1x_warm = Chi1x
                _Zy_warm = Zy;  _Chiy_warm = Chiy;  _Chi1y_warm = Chi1y

                lnphi_x = _lnphi_aq (ep_aq,  Zx, Chix,  Chi1x)
                lnphi_y = _lnphi_vap(ep_vap, Zy, Chiy,  Chi1y)
                lng = (lnphi_y + np.log(np.maximum(y, _LOG_GUARD))
                       - lnphi_x - np.log(np.maximum(x, _LOG_GUARD)))
                _ssi_iters += 1
                _resid_norm = float(np.linalg.norm(lng))
                if _resid_norm < float(tol):
                    converged = True
                    break

    out = {
        "converged": bool(converged),
        "iterations": _ssi_iters + _newton_iters,
        "ssi_iterations": _ssi_iters,
        "newton_iterations": _newton_iters,
        "K": K.copy(), "x": x.copy(), "y": y.copy(),
        "residual_norm": _resid_norm,
        "final_m": float(_m),
    }

    if converged:
        denom = K[1] - K[0]
        x[0] = (K[1] - 1.0) / denom; x[1] = 1.0 - x[0]; y[:] = K * x
        x = np.clip(x, 0, 1); y = np.clip(y, 0, 1)

        ep_aq  = _eos_aq (T, P_bar, x, kij12)
        ep_vap = _eos_vap(T, P_bar, y, kij12)

        Zx, Chix,  Chi1x  = ZChi(ep_aq["A"],  ep_aq["B"],  x,
                                  ep_aq["Kapa"],  ep_aq["Eps"],  swc,
                                  Z0=_Zx_warm, Chi0=_Chix_warm, Chi10=_Chi1x_warm)
        Zy, Chiy,  Chi1y  = ZChi(ep_vap["A"], ep_vap["B"], y,
                                  ep_vap["Kapa"], ep_vap["Eps"], swc,
                                  Z0=_Zy_warm, Chi0=_Chiy_warm, Chi10=_Chi1y_warm)

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
# Single-phase lnphi helper (for stability test)
# =============================================================================
def _lnphi_single_phase(
    T: float, P_bar: float, w: np.ndarray,
    kij12: float, swc: float,
) -> np.ndarray:
    """
    Compute ln(φ_i(w)) for a composition w using the lowest-Gibbs root.

    Tries both aqueous (HV/SRK) and vapour (vdW1f/SRK) models and picks the
    root with the lower excess Gibbs energy.  This is the correct thing to do
    when we don't know a priori whether w is liquid-like or vapour-like.
    """
    w = np.asarray(w, dtype=float).reshape(2,)
    w = np.clip(w, _LOG_GUARD, None)
    w = w / w.sum()

    results = []
    for phase_fn, eos_fn in [(_lnphi_aq, _eos_aq), (_lnphi_vap, _eos_vap)]:
        try:
            ep = eos_fn(T, P_bar, w, kij12)
            Z, Chi, Chi1 = ZChi(ep["A"], ep["B"], w,
                                ep["Kapa"], ep["Eps"], swc)
            if Z == 0.0 or not np.isfinite(Z):
                continue
            lnphi = phase_fn(ep, Z, Chi, Chi1)
            if not np.all(np.isfinite(lnphi)):
                continue
            ge = ExcessGibbs(ep["A"], ep["B"], np.array([Z]), w,
                             np.array([Chi]), np.array([Chi1]))
            results.append((float(ge[0]), lnphi))
        except Exception:
            continue

    if not results:
        return np.full(2, np.nan)
    results.sort(key=lambda r: r[0])
    return results[0][1]


# =============================================================================
# Michelsen TPD stability test (Jex et al. 2024)
# =============================================================================
def stability_test(
    T: float, P_bar: float, z: np.ndarray,
    Omega: np.ndarray, Tc: np.ndarray, Pc: np.ndarray, Mw: np.ndarray,
    kij12: float = 0.0, swc: float = 0.0,
    *,
    accelerated: bool = True,
    tol: float = 1e-10,
    maxiter: int = 200,
) -> dict:
    """
    Michelsen tangent-plane distance (TPD) stability test with multiple
    initial guesses (Jex et al. 2024, Eq. 1 & 7).

    For each trial, runs (accelerated) successive substitution on the
    TPD stationary conditions:
        W_i = exp(d_i - lnphi_i(w))
    where d_i = ln(z_i) + lnphi_i(z) is evaluated once for the feed.

    Returns dict with:
        stable    : bool — True if all TPD ≥ -1e-7
        tpd_min   : float — most negative TPD found
        K_unstable: np.ndarray or None — K = W/z from most-negative-TPD trial
        trials    : list of (label, tpd, converged, iterations)
    """
    T = float(T); P_bar = float(P_bar)
    z = np.asarray(z, dtype=float).reshape(2,)
    z = np.clip(z, _LOG_GUARD, None); z = z / z.sum()
    Tc    = np.asarray(Tc,    dtype=float).reshape(2,)
    Pc    = np.asarray(Pc,    dtype=float).reshape(2,)
    Omega = np.asarray(Omega, dtype=float).reshape(2,)

    # Feed fugacity: d_i = ln(z_i) + lnphi_i(z)
    lnphi_z = _lnphi_single_phase(T, P_bar, z, kij12, swc)
    if not np.all(np.isfinite(lnphi_z)):
        return {"stable": True, "tpd_min": 0.0, "K_unstable": None, "trials": []}
    d = np.log(z) + lnphi_z

    # Build trial K-value sets
    K_wil = wilson_K_init(T, P_bar, Omega, Tc, Pc)
    K_wil = np.clip(K_wil, 1e-12, 1e12)
    trials_K = [
        ("Wilson",       K_wil.copy()),
        ("1/Wilson",     1.0 / K_wil),
        ("Wilson^1/3",   np.cbrt(K_wil)),
        ("1/Wilson^1/3", np.cbrt(1.0 / K_wil)),
        ("CO2-rich",     np.array([0.95 / z[0], 0.05 / z[1]])),
        ("H2O-rich",     np.array([1e-15 / z[0], (1.0 - 1e-15) / z[1]])),
    ]

    _M_CAP_HI = 10.0
    best_tpd = 0.0
    best_K = None
    all_trials = []

    for label, K_trial in trials_K:
        # Initial W = K * z
        W = np.clip(K_trial * z, _LOG_GUARD, None)
        _m = 1.0; _g_prev = None
        conv = False; n_it = 0

        for n_it in range(int(maxiter)):
            w = W / W.sum()
            lnphi_w = _lnphi_single_phase(T, P_bar, w, kij12, swc)
            if not np.all(np.isfinite(lnphi_w)):
                break

            # Stationary condition residual: ln(W_i) + lnphi_i(w) - d_i = 0
            lnW = np.log(np.maximum(W, _LOG_GUARD))
            g_vec = lnW + lnphi_w - d   # should → 0 at convergence

            # Accelerated step-size
            if accelerated and _g_prev is not None:
                num_a   = np.dot(_g_prev, _g_prev)
                denom_a = np.dot(_g_prev, _g_prev - g_vec)
                if abs(denom_a) > 1e-30:
                    _m = abs(num_a / denom_a * _m)
                    _m = np.clip(_m, 1.0, _M_CAP_HI)
                else:
                    _m = 1.0

            _g_prev = g_vec.copy()

            # Update: W_i = exp(d_i - lnphi_i(w))  (direct substitution)
            W_new = np.exp(np.clip(d - lnphi_w, -50, 50))

            # Apply acceleration: blend old and new in log-space
            lnW_new = d - lnphi_w
            lnW_step = np.clip(lnW_new - lnW, -5.0, 5.0)
            W = np.exp(lnW + _m * lnW_step)
            W = np.clip(W, _LOG_GUARD, 1e10)

            if np.linalg.norm(lnW_step) < float(tol):
                conv = True; break

        # Compute TPD at converged W
        sumW = W.sum()
        tpd = 1.0 + float(np.sum(W * (np.log(np.maximum(W, _LOG_GUARD))
                                        + lnphi_w - d - 1.0)))
        # Michelsen form: TPD = Σ W_i (ln W_i + ln φ_i(w) - d_i - 1) + 1
        # Unstable if TPD < 0 (or equivalently, sumW > 1 at a stationary point)

        all_trials.append((label, float(tpd), bool(conv), int(n_it + 1)))

        if tpd < best_tpd:
            best_tpd = tpd
            best_K = W / z   # K = W_i / z_i

    stable = best_tpd >= -1e-7
    return {
        "stable":     stable,
        "tpd_min":    best_tpd,
        "K_unstable": best_K,
        "trials":     all_trials,
    }


# =============================================================================
# Hierarchical flash: stability → flash (Jex et al. 2024, Fig. 1)
# =============================================================================
def flash_co2_h2o_tpz_robust(
    T: float, P_bar: float, z_co2: float,
    kij12: float | None = None, swc: float | None = None,
    *,
    vshift_co2: float = 0.0,
    vshift_h2o: float = 0.0,
    accelerated: bool = True,
    tol: float = 1e-10,
    maxiter: int = 1000,
) -> dict:
    """
    Hierarchical TPz flash for CO₂(0) + H₂O(1):
      1. Run stability_test → if stable, return single-phase
      2. Use K from lowest TPD as initial guess for tie_line_two_comp
      3. If flash fails, fall back to Wilson K

    Returns same dict as flash_co2_h2o_tpz, with added 'stability' key.
    """
    if kij12 is None:
        kij12 = kij_ecpa(T) if PARAM_SET == "eCPA" else 0.0
    if swc is None:
        swc = s14_ecpa(T) if PARAM_SET == "eCPA" else 0.0

    comps = make_components_co2_h2o()
    z = np.array([z_co2, 1.0 - z_co2])

    # Step 1: Stability test
    stab = stability_test(
        T, P_bar, z,
        Omega=comps["Omega"], Tc=comps["Tc"], Pc=comps["Pc"], Mw=comps["Mw"],
        kij12=kij12, swc=swc, accelerated=accelerated,
        tol=tol, maxiter=min(maxiter, 200),
    )

    base = {"T": float(T), "P_bar": float(P_bar), "z": z.copy(),
            "stability": stab}

    if stab["stable"]:
        # Single phase — determine liquid or vapour from feed density
        base.update({"phase": "single_phase", "beta": np.nan,
                     "x": z.copy(), "y": z.copy(), "tie": None})
        return base

    # Step 2: Flash with K from stability test
    kw = dict(Omega=comps["Omega"], Tc=comps["Tc"], Pc=comps["Pc"],
              Mw=comps["Mw"], kij12=kij12, swc=swc, tol=tol, maxiter=maxiter)

    K_stab = stab["K_unstable"]
    result = None

    if K_stab is not None and np.all(np.isfinite(K_stab)) and np.all(K_stab > 0):
        tie = tie_line_two_comp(T=T, P_bar=P_bar, K_init=K_stab,
                                accelerated=accelerated, **kw)
        if tie["converged"]:
            result = _build_flash_result(T, P_bar, z, tie, comps, base,
                                         vshift_co2, vshift_h2o)

    # Step 3: Fallback — Wilson K (standard)
    if result is None:
        tie = tie_line_two_comp(T=T, P_bar=P_bar, accelerated=accelerated, **kw)
        if tie["converged"]:
            result = _build_flash_result(T, P_bar, z, tie, comps, base,
                                         vshift_co2, vshift_h2o)

    # Step 4: Near S₁₄≈0 (T≈288–292 K) retry with swc=0 (same as flash_co2_h2o_tpz)
    if result is None and 0 < abs(swc) < 0.005:
        kw_s0 = dict(kw, swc=0.0)
        if K_stab is not None and np.all(np.isfinite(K_stab)) and np.all(K_stab > 0):
            tie = tie_line_two_comp(T=T, P_bar=P_bar, K_init=K_stab,
                                    accelerated=accelerated, **kw_s0)
            if tie["converged"]:
                result = _build_flash_result(T, P_bar, z, tie, comps, base,
                                             vshift_co2, vshift_h2o)
        if result is None:
            tie = tie_line_two_comp(T=T, P_bar=P_bar, accelerated=accelerated,
                                    **kw_s0)
            if tie["converged"]:
                result = _build_flash_result(T, P_bar, z, tie, comps, base,
                                             vshift_co2, vshift_h2o)

    if result is None:
        base.update({"phase": "failed", "beta": np.nan, "tie": None})
        return base

    return result


def flash_co2_h2o_tpz_warmstart(
    T: float, P_bar: float, z_co2: float,
    solution_guess_fn,
    kij12: float | None = None, swc: float | None = None,
    *,
    tol: float = 1e-10,
    maxiter: int = 1000,
) -> dict:
    """
    CPA two-phase flash warm-started from the eCPA solution table.

    Queries the solution table at ms=0 for K-value initial guesses, skipping
    the expensive Michelsen stability test when the table indicates two-phase.
    Falls back to flash_co2_h2o_tpz_robust on any failure.

    Parameters
    ----------
    solution_guess_fn : callable
        As returned by ecpa.solution_table.make_solution_guess_fn().
        Signature: (T, P_bar, z_co2, ms) → (sol_10, ms_aq, is_two_phase_hint)
        Queried at ms=0 for the salt-free CPA limit.

    Returns
    -------
    Same dict format as flash_co2_h2o_tpz_robust, plus:
        "n_iter" : int   — SSI+Newton iterations from tie_line_two_comp
        "warmstarted" : bool — True if table guess was used (False = fell back)
    """
    if kij12 is None:
        kij12 = kij_ecpa(T) if PARAM_SET == "eCPA" else 0.0
    if swc is None:
        swc = s14_ecpa(T) if PARAM_SET == "eCPA" else 0.0

    comps = make_components_co2_h2o()
    z = np.array([z_co2, 1.0 - z_co2])
    base = {"T": float(T), "P_bar": float(P_bar), "z": z.copy()}
    kw = dict(Omega=comps["Omega"], Tc=comps["Tc"], Pc=comps["Pc"],
              Mw=comps["Mw"], kij12=kij12, swc=swc, tol=tol, maxiter=maxiter)

    # ── Step 1: table lookup at ms=0 ─────────────────────────────────────────
    try:
        sol_10, _ms_aq, is_two_phase_hint = solution_guess_fn(
            float(T), float(P_bar), float(z_co2), 0.0)
        sol_10 = np.asarray(sol_10, dtype=float)
        x1w = float(sol_10[1])   # H2O in aqueous phase
        x1c = float(sol_10[4])   # H2O in CO2-rich phase
        # K[0]=K_CO2, K[1]=K_H2O  (K_i = y_i/x_i, y=CO2-rich, x=aqueous)
        denom_co2 = max(1.0 - x1w, 1e-15)
        denom_h2o = max(x1w, 1e-15)
        K_table = np.array([(1.0 - x1c) / denom_co2, x1c / denom_h2o])
        table_ok = (np.all(np.isfinite(K_table)) and np.all(K_table > 0)
                    and bool(is_two_phase_hint))
    except Exception:
        table_ok = False
        K_table = None
        is_two_phase_hint = None

    # ── Step 2: warm-started flash (skip stability test) ─────────────────────
    result = None
    warmstarted = False
    if table_ok:
        try:
            tie = tie_line_two_comp(T=T, P_bar=P_bar, K_init=K_table,
                                    accelerated=True, **kw)
            if tie["converged"]:
                result = _build_flash_result(T, P_bar, z, tie, comps,
                                             base, 0.0, 0.0)
                if result is not None:
                    result["n_iter"] = int(tie.get("iterations", -1))
                    result["warmstarted"] = True
                    warmstarted = True
        except Exception:
            pass

    # ── Step 3: fallback to full robust solver ────────────────────────────────
    # Used when: table says single-phase, warm-started tie-line failed, or
    # table lookup itself raised an exception.
    if result is None:
        r = flash_co2_h2o_tpz_robust(T=T, P_bar=P_bar, z_co2=z_co2,
                                     kij12=kij12, swc=swc,
                                     tol=tol, maxiter=maxiter)
        r["n_iter"] = int((r.get("tie") or {}).get("iterations", -1))
        r["warmstarted"] = False
        return r

    return result


def _build_flash_result(T, P_bar, z, tie, comps, base, vshift_co2, vshift_h2o):
    """Build flash result dict from a converged tie-line."""
    x0, y0, z0 = float(tie["x"][0]), float(tie["y"][0]), float(z[0])
    denom = y0 - x0
    if abs(denom) < 1e-14:
        return None

    beta = (z0 - x0) / denom
    if beta <= 0:
        phase = "single_liquid"; beta = 0.0
    elif beta >= 1:
        phase = "single_vapor";  beta = 1.0
    else:
        phase = "two_phase"

    out = dict(base)
    out.update({"phase": phase, "beta": float(beta), "tie": tie,
                "x": tie["x"].copy(), "y": tie["y"].copy()})

    # Volume shift (same logic as flash_co2_h2o_tpz)
    if (vshift_co2 != 0.0 or vshift_h2o != 0.0) and tie.get("rho_mass") is not None:
        Mw_arr = comps["Mw"]
        c = np.array([vshift_co2, vshift_h2o]) * 1000  # m³/mol → L/mol
        for i, xi in enumerate([out["x"], out["y"]]):
            rho_old = float(tie["rho_mass"][i])
            if rho_old <= 0:
                continue
            M_mix = float(np.dot(Mw_arr, xi))
            Vm_corr = M_mix / rho_old + float(np.dot(c, xi))
            tie["rho_mass"][i] = M_mix / Vm_corr if Vm_corr > 0 else rho_old

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
    "stability_test",
    "flash_tpz_two_comp",
    "make_components_co2_h2o",
    "flash_co2_h2o_tpz",
    "flash_co2_h2o_tpz_robust",
]
