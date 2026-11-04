"""eCPA model parameters -- production set.

The values in this module are the **single source of truth** for every
figure, validation, and flash call in the project. Every fig_*.py script
imports parameters from here -- there is intentionally no mechanism to
swap in alternate parameter sets at runtime, because mixing fits across
figures was a known failure mode of the legacy ad hoc scripts.

Parameter provenance (three independent fits by F. Mourao Coelho,
verified against F. M. Coelho's AAD_*.dat files on 2026-06-24):

  - CO2-ion interaction energies  (A_Uc4..9, B_Uc4..9):
      production fit ~ F. M. Coelho's MIXED_SALTS/OPT/OPT7 (excludes BANDO,
      WANG, JACOB, LIU 2021, NIGHSWANDER per F. M. Coelho's fitting notes).

  - CH4-ion interaction energies  (A_Uh4..9, B_Uh4..9):
      production fit ~ F. M. Coelho's CH4-Brine/OPT/OPT6 (refinement of OPT1
      dropping T > 550 K and making Mg/SO4 temperature-independent).

  - Peneloux ion-specific volume shifts  (A_Pen4..9, B_Pen4..9):
      production fit ~ F. M. Coelho's DENSITY/OPT/OPT_PEN6 (T-linear shift
      seeded from OPT_PERT; AAD = 1.16% on Al-Ghafri density data).

The numeric "OPT=N" labels above are F. M. Coelho's per-axis bookkeeping
(separate iteration sequences within each physics term) -- they are
not a single global parameter generation. Don't be confused if a CO2
"OPT=7" and a CH4 "OPT=6" appear together: they refer to independent
fits of unrelated physics terms.
"""

# Provenance identifiers
# ----------------------
# Names like `mg_row_refit_2026_07` below identify
# the study a value came from. Those studies live in the authors' research
# tree, not in this repository: they are incremental fits and checks, many
# superseded, and shipping them would bury the code they support. The
# identifier is kept so a value can still be traced to the work that produced
# it; the text beside each one is written to stand on its own without it.

import numpy as np
import numpy as _np

# Component indexing convention (used throughout the package):
#   1 -> H2O (w)
#   2 -> CH4 (h)
#   3 -> CO2 (c)
#   4 -> Na+
#   5 -> Cl-
#   6 -> K+
#   7 -> Ca2+
#   8 -> Mg2+
#   9 -> SO4(2-)


# =============================================================
# UNIVERSAL CONSTANTS
# =============================================================
R = 8.314 # J/mol.K
Na = 6.02e23 # 1/mol
kb = 1.38e-23 # J/K
e = 1.6e-19 # C
eps0 = 8.854e-12 # F/m


# =============================================================
# COMPOUND PARAMETERS (example: extend to dicts or classes later)
# =============================================================
# Water molar mass [kg/mol]. 0.01801528 -> n_w = 55.5084 mol per kg water.
#
# This was the rounded 0.018 (n_w = 55.5556) until 2026-09-18, which is
# 0.085 % low and disagreed with the conversion layers that surround this
# library: scripts/total_mole_basis.py and GasBrineBench's
# bench/core/conventions.py both use 0.01801528. The model constant and
# the molality conversions applied to the data it is fitted against must
# not differ, so the model moved to the correct value.
#
# The published Paper I code uses the rounded 0.018 in its EoS
# mole-fraction algebra, so paper1_config binds 0.018 back for the
# duration of apply_paper1() and Paper I reproduction is unaffected.
# Paper I's pure-water DENSITY route used a third value, 18.015e-3;
# that one is a reproduction constant and stays where it is, in
# paper1_config.M_H2O_DENSITY.
Mw = 0.01801528 # kg/mol, water
Mh = 0.016 # kg/mol, methane
Mc = 0.044 # kg/mol, CO2


# Critical properties
Tc1, Pc1 = 647.29, 22060000 # Water
Tc2, Pc2 = 190.56, 45.992e5 # CH4
Tc3, Pc3 = 304.4, 7380000 # CO2


# Physical parameters (example subset)
b1, c11, a01 = 14.515e-6, 0.6736, 1017.3*R*14.515e-6
b2, w2 = 0.08664*R*Tc2/Pc2, 0.01142
a02 = 0.42748*R**2*Tc2**2/Pc2
c12 = 0.480 + 1.574*w2 - 0.176*w2**2
b3, c13, a03 = 27.2e-6, 0.7602, 1551.2*R*27.2e-6


# =============================================================
# ASSOCIATION PARAMETERS
# =============================================================
epsW = 2003.25 # K (/kb)
bettaW = 69.2e-3
kappaW = bettaW*b1 # m³/mol


# =============================================================
# PERMITTIVITY PARAMETERS
# =============================================================
dip01 = 1.8546*3.335e-30 # C.m
pol1 = 1.6133e-40 # Cm²/J
pol2 = 2.6155e-40 # Cm²/J
pol3 = 2.6946e-40 # Cm²/J
GAMMA1 = 63.4715*np.pi/180 # rad
THETA1 = 94.7939*np.pi/180 # rad
zww = 4



# =============================================================
# ION PARAMETERS
# =============================================================
Tref = 298.15
# Ion 4: Na⁺
Sg4 = 2.63e-10              #m
b4 = 2/3*np.pi*Sg4**3*Na    #m³/mol 
Rb4 = 1.54e-10              #m
a04 = 959.54*R*b4           #J.m³/mol² 
c14 = 0
Uref14 = 91.46*R            #J/mol
Talfa14 = 260.75            #K
alfa14 = 1449.01            #K
Z4 = 1
pol4 = 2.221e-40            #Cm²/J
MM4 = 0.023                 #kg/mol

# Ion 5: Cl⁻
Sg5 = 3.571e-10             #m
b5 = 2/3*np.pi*Sg5**3*Na    #m³/mol 
Rb5 = 1.99e-10              #m
a05 = 1878.78*R*b5          #J.m³/mol²
c15 = 0
Uref15 = -220.22*R          #J/mol
Talfa15 = 413.07            #K
alfa15 = 1579.19            #K
Z5 = -1
pol5 = 3.557e-40            #Cm²/J
MM5 = 0.0355                #kg/mol

# Ion 6: K⁺
Sg6 = 2.90e-10             #m
b6 = 2/3*np.pi*Sg6**3*Na   #m³/mol 
Rb6 = 2.04e-10             #m
a06 = 571.37*R*b6          #J.m³/mol² 
c16 = 0
Uref16 = 53.57*R           #J/mol
Talfa16 = 999.89           #K
alfa16 = -2770.48          #K
Z6 = 1
pol6 = 3.166e-40           #Cm²/J
MM6 = 0.0391               #kg/mol

# Ion 7: Ca²⁺
# sigma/R_B/Uref refit 2026-07 within physical brackets (Shannon-Held for
# sigma, Rashin-Honig anchor for R_B) against Pitzer phi_osm at 298.15 K:
# the printed Schlaikjer 2018 Table-4 row (1.17e-10 / 1.73e-10 / 595.83R)
# cannot reproduce that paper's own reported CaCl2 accuracy (RAD 33% vs
# their claimed 5.0%); the refit row gives 1.65%. Paper II SI, Section S8.3.
Sg7 = 2.7499349395317134e-10  #m
b7 = 2/3*np.pi*Sg7**3*Na   #m³/mol
Rb7 = 1.6007056707623875e-10  #m
a07 = 6261.18*R*b7         #J.m³/mol²
c17 = 0
Uref17 = 117.18712325857294*R  #J/mol
Talfa17 = 492.90           #K
alfa17 = 376.77            #K
Z7 = 2
pol7 = 5.138e-40           #Cm²/J
MM7 = 0.040                #kg/mol  

# Ion 8: Mg²⁺
Sg8 = 3.42e-10             #m
b8 = 2/3*np.pi*Sg8**3*Na   #m³/mol 
Rb8 = 1.46e-10             #m
a08 = 4935.39*R*b8         #J.m³/mol² 
c18 = 0
Uref18 = 440.04*R          #J/mol
Talfa18 = 102.49           #K
alfa18 = 207.42            #K
Z8 = 2
# pol8 refit 2026-07 against Pitzer MgCl2 phi_osm (geometry untouched);
# the prior 5.138e-40 was a copy of the Ca2+ value, untraceable in the
# primary sources and ~60x the Pyper 1992 ab-initio free-ion value
# (0.080e-40). Paper II SI, Section S8.4; provenance: mg_row_refit_2026_07.
pol8 = 1.6093035745986959e-40  #Cm²/J
MM8 = 0.0243               #kg/mol  

# Ion 9: SO₄²⁻
Sg9 = 3.14e-10             #m
b9 = 2/3*np.pi*Sg9**3*Na   #m³/mol 
Rb9 = 2.40e-10             #m
a09 = 2903.63*R*b9         #J.m³/mol²
c19 = 0
Uref19 = 0.67*R            #J/mol
Talfa19 = 433.55           #K
alfa19 = 5391.99           #K
Z9 = -2
pol9 = 5.775e-40           #Cm²/J
MM9 = 0.096                #kg/mol


# =============================================================
# MIXTURE PARAMETERS
# =============================================================
# CH4-H2O
Ak12 = -4.0935
Bk12 = 5.6634
Ck12 = -1.7845

# CH4-Ions
A_Uh4 = 16060.77 ; B_Uh4 = -1585.83 # CH4-Na
A_Uh5 = 1370.72 ; B_Uh5 = 1510.16 # CH4-Cl
A_Uh6 = 8355.29 ; B_Uh6 = -395.72 # CH4-K
# B_Uh7 co-refit 2026-07 with the repaired Ca2+ row (33058.23 was fitted
# against the printed row and compensated for it); CH4-CaCl2 AAD 4.72%
# vs 4.93% printed-row baseline. ca_ch4_corefit_2026_07/.
A_Uh7 = 2522.69 ; B_Uh7 = 17458.124121860732 # CH4-Ca
A_Uh8 = 0.0 ; B_Uh8 = 20336.87 # CH4-Mg
A_Uh9 = 0.0 ; B_Uh9 = 4810.31 # CH4-SO4

# CO2-H2O
Ak13 = -0.4921
Bk13 = 2.1014
Ck13 = -1.5714
#: m-CR1 combining rule for the CO2-H2O cross-association ENERGY.
#:
#: OFF (default, and what Papers I-III were computed with): the cross strength
#: is S13(T) * Delta_water, so it inherits water's full association energy and
#: exp(epsW/T) varies 68x across 273-633 K. A single quadratic S13(T) must
#: flatten that swing, and the fitted one varies 46x AND CHANGES SIGN below
#: 289.8 K -- a negative bond strength, which pushes water out of the CO2-rich
#: phase and is why y_H2O collapses there (enhancement factor 0.147 of
#: experiment at 273-304 K, 100-300 bar).
#:
#: ON: eps_cross = eps_water / 2, the standard CPA solvation rule (CR-1 /
#: m-CR1) and what Chabab 2019's e-PR-CPA uses. exp(epsW/2T) varies 10x
#: instead of 68x, so the remaining quadratic is only a mild positive
#: correction rather than a 46x sign-changing one.
#:
#: The two forms span the SAME set of cross-strength curves for an arbitrary
#: S13(T); what changes is which curves a QUADRATIC can reach. With the flag
#: ON, AS13/BS13/CS13 parameterise that milder factor, so the published values
#: do not carry over -- see fitting.blocks._solvation_start, which projects the
#: equivalent curve.
MCR1_CROSS_ENERGY = False


def s13_of_T(T):
    """CO2-H2O solvation strength entering Delta_13 = S13(T) * Delta_water.

    One definition, called from all six places that used to inline it --
    ms_ecpa/eos.py (2) and ms_ecpa/lnphi.py (4). Reads this module's globals,
    so a rebind reaches it without the star-import staleness that has bitten
    this codebase before. Analytic in T and T is never complex-step perturbed,
    so it is safe inside the inner Newton.
    """
    S = AS13 * (T / Tc3) ** 2 + BS13 * (T / Tc3) + CS13
    if MCR1_CROSS_ENERGY:
        S = S * (_np.exp(0.5 * epsW / T) - 1.0) / (_np.exp(epsW / T) - 1.0)
    return S


AS13 = 0.191728
BS13 = -0.172991
CS13 = -0.009094

# CO2-Ions
A_Uc4 = 5512.23 ; B_Uc4 = 5986.83 # CO2-Na
A_Uc5 = 8081.74 ; B_Uc5 = -3107.76 # CO2-Cl
A_Uc6 = 12056.43 ; B_Uc6 = -3545.23 # CO2-K
# B_Uc7 co-refit 2026-07 with the repaired Ca2+ row (24425.62 was fitted
# against the printed row and compensated for it); CO2-CaCl2 AAD 12.34%
# vs 13.10% printed-row baseline. ca_gas_corefit_2026_07/.
A_Uc7 = -2772.70 ; B_Uc7 = 6049.666862771973 # CO2-Ca
A_Uc8 = -20.12 ; B_Uc8 = 11538.44 # CO2-Mg
A_Uc9 = -14725.94 ; B_Uc9 = 27434.52 # CO2-SO4

# CH4-CO2
k23 = 0.079734


# =============================================================
# PENELOUX VOLUME TRANSLATION
# =============================================================
#Na
A_Pen4 = -19.8574; B_Pen4 = -27.7728
#Cl
A_Pen5 = -15.7929; B_Pen5 = -30.5925
#K
A_Pen6 = -63.6500; B_Pen6 = -7.07101
#Ca — B_Pen7 shifted -19.3466163811085 cm3/mol with the repaired Ca2+ row
# (larger co-volume needs a different translation to keep the Al Ghafri
# CaCl2 densities at paper quality; |err| <= 1.3% at m = 1/3/6).
A_Pen7 = 30.4741; B_Pen7 = -58.8327163811085
#Mg
A_Pen8 = 46.7469; B_Pen8 = -105.7743
#SO4
A_Pen9 = 0.7952; B_Pen9 = -1.2678


# =============================================================
# TOTAL ION
# =============================================================
Sgi = np.array([Sg4, Sg5, Sg6, Sg7, Sg8, Sg9])
bi = np.array([b4, b5, b6, b7, b8, b9])
Rbi = np.array([Rb4, Rb5, Rb6, Rb7, Rb8, Rb9])
a0i = np.array([a04, a05, a06, a07, a08, a09])
c1i = np.array([c14, c15, c16, c17, c18, c19])
Uref1i = np.array([Uref14, Uref15, Uref16, Uref17, Uref18, Uref19])
Talfa1i = np.array([Talfa14, Talfa15, Talfa16, Talfa17, Talfa18, Talfa19])
alfa1i = np.array([alfa14, alfa15, alfa16, alfa17, alfa18, alfa19])
Zi = np.array([Z4, Z5, Z6, Z7, Z8, Z9])
poli = np.array([pol4, pol5, pol6, pol7, pol8, pol9])
MMi = np.array([MM4, MM5, MM6, MM7, MM8, MM9])
A_U2i = np.array([A_Uh4, A_Uh5, A_Uh6, A_Uh7, A_Uh8, A_Uh9])
B_U2i = np.array([B_Uh4, B_Uh5, B_Uh6, B_Uh7, B_Uh8, B_Uh9])
A_U3i = np.array([A_Uc4, A_Uc5, A_Uc6, A_Uc7, A_Uc8, A_Uc9])
B_U3i = np.array([B_Uc4, B_Uc5, B_Uc6, B_Uc7, B_Uc8, B_Uc9])
A_Peni = np.array([A_Pen4, A_Pen5, A_Pen6, A_Pen7, A_Pen8, A_Pen9])
B_Peni = np.array([B_Pen4, B_Pen5, B_Pen6, B_Pen7, B_Pen8, B_Pen9])


# =============================================================
# ION-ION HURON-VIDAL REFERENCE ENERGIES
# =============================================================
# Symmetric 6x6 matrix of ion-ion Huron-Vidal/NRTL reference energies
# DU_ij [J/mol], ion order (Na+, Cl-, K+, Ca2+, Mg2+, SO4^2-) -- the same
# order as every other 6-long ion array in this module.  The published
# HV/NRTL mixing rule (Schlaikjer 2018 eqs 4-5 == Maribo-Mogensen 2015
# eq 15) is a full double sum over components, g^E = (1/b) sum_ij
# x_i x_j b_j DU_ij with non-randomness alpha = 0 and symmetric DU; the
# printed parameter tables carry every ion-ion entry at ZERO, and so does
# the production model: this matrix is ALL ZEROS by default, which makes
# the extra g^E term analytically (not merely numerically) zero -- the
# evaluation is skipped entirely in lnphi.py/eos.py, so the baseline is
# bit-exact.  T-independent for now; a Schlaikjer-eq-6-style temperature
# dependence can be added when data demands it.
#
# Set entries with set_ion_ion_du() (enforces symmetry); direct writes
# must keep the matrix symmetric -- lnphi_aq/solve_elv raise ValueError
# on an active asymmetric matrix.
ION_INDEX = {"Na": 0, "Cl": 1, "K": 2, "Ca": 3, "Mg": 4, "SO4": 5}
Uref_ii = np.zeros((6, 6))

# Slot-2 (CH4 slot) water cross-association strength, mirroring the
# CO2 (slot-3) solvation pattern: the slot-2 species solvates on water
# with strength S12A * DELTA_water, exactly as CO2 does with S13(T).
# Production value 0.0 is an ANALYTIC no-op (chi2 = 1 identically, all
# added terms vanish bit-exactly). Runtime-togglable via a live module
# reference (lnphi.py reads _prm.S12A each call), so gas-swap pilots
# (e.g. C3H8) can activate water-alkane solvation without touching the
# production baseline. Constant (not T-dependent) by design at this
# stage.
#
# ONLY the lnphi kernels honour this. ms_ecpa/eos.py (residual_elv,
# solve_elv, gas_fugacities) contains no chi2w terms whatsoever, so a
# nonzero S12A would be silently ignored there; solve_elv raises rather
# than let that happen. flash_kv and stability_flash route through
# lnphi_aq/lnphi_gas and are unaffected.
S12A = 0.0


def set_ion_ion_du(ion_a, ion_b, dU):
    """Set the symmetric ion-ion HV reference energy DU_ab = DU_ba [J/mol].

    ``ion_a``/``ion_b``: ion name in ``ION_INDEX`` ("Na", "Cl", "K",
    "Ca", "Mg", "SO4") or integer index 0-5.  ``dU`` in J/mol (multiply a
    literature DU/R [K] by ``R``).  Writes both (a, b) and (b, a) so
    the matrix stays symmetric.  In-place update: modules that read
    ``parameters.Uref_ii`` through a live module reference (lnphi.py,
    eos.py) see the change immediately; snapshotting consumers (e.g.
    scripts/total_mole_basis.py) need their reload hook.
    """
    ia = ION_INDEX[ion_a] if isinstance(ion_a, str) else int(ion_a)
    ib = ION_INDEX[ion_b] if isinstance(ion_b, str) else int(ion_b)
    Uref_ii[ia, ib] = dU
    Uref_ii[ib, ia] = dU


