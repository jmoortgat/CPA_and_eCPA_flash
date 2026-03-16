"""
Physical constants and eCPA EoS parameters.
All values are scalars (float or derived float).
"""
import numpy as np

# ── Universal constants ────────────────────────────────────────────────────────
R    = 8.314          # J / (mol·K)
Na   = 6.02e23        # 1/mol
kb   = 1.38e-23       # J/K
e    = 1.6e-19        # C
eps0 = 8.854e-12      # F/m

# ── Molecular weights ─────────────────────────────────────────────────────────
Mw   = 0.018          # kg/mol  (H₂O)
Ms   = 0.0585         # kg/mol  (NaCl)
Mc   = 0.044          # kg/mol  (CO₂)

# ── Critical properties ───────────────────────────────────────────────────────
Tc1  = 647.29         # K   (H₂O)
Pc1  = 22_060_000     # Pa  (H₂O)
Tc4  = 304.4          # K   (CO₂)
Pc4  = 7_380_000      # Pa  (CO₂)

# ── CPA/eCPA EoS parameters ───────────────────────────────────────────────────
# Co-volumes  [m³/mol]
b1 = 14.515e-6        # H₂O
b2 = 16.49e-6         # Na⁺
b3 = 40.83e-6         # Cl⁻
b4 = 27.2e-6          # CO₂

# Soave attractive parameters  [J·m³/mol²  or dimensionless]
c11  = 0.6736
a01  = 1017.3 * R * b1   # J·m³/mol²
a02  = 0.0
a03  = 0.0
c14  = 0.7602
a04  = 1551.2 * R * b4   # J·m³/mol²

# Reference temperature
Tref = 298.15         # K

# Binary interaction parameters (H₂O–CO₂)
Akij = -0.49206
Bkij =  2.10136
Ckij = -1.57135
ASij =  0.19173
BSij = -0.17299
CSij = -0.00909

# ── Association parameters (H₂O) ──────────────────────────────────────────────
epsW   = 2003.25      # K  (ε/k_B)
bettaW = 69.2e-3      # dimensionless
kappaW = bettaW * b1  # m³/mol  (association volume)

# ── Ion parameters ────────────────────────────────────────────────────────────
Z2 = 1                # charge Na⁺
Z3 = -1               # charge Cl⁻
Sg2 = 2.356e-10       # m  (Debye–Hückel distance, Na⁺)
Sg3 = 3.187e-10       # m  (Debye–Hückel distance, Cl⁻)
Rb2 = 1.665e-10       # m  (Born radius, Na⁺)
Rb3 = 1.828e-10       # m  (Born radius, Cl⁻)

Penelouxs = -53.5e-6  # m³/mol  (Péneloux volume shift)

# Solvation energy parameters  [J/mol or K]
Uref1s  = -223.5 * R  # J/mol
Talfa1s = 340.0        # K
alfa1s  = 1573.0       # K
Uref4s  = 6056.13852   # J/mol
Talfa4s = 243.79352    # K
alfa4s  = 691.85326    # K

# ── Permittivity / dipole parameters ──────────────────────────────────────────
dip01  = 1.8546 * 3.335e-30    # C·m   (H₂O dipole moment)
pol1   = 1.6133e-40            # C·m²/J  (H₂O polarisability)
pol2   = 2.221e-40             # C·m²/J  (Na⁺)
pol3   = 3.557e-40             # C·m²/J  (Cl⁻)
pol4   = 2.6946e-40            # C·m²/J  (CO₂)
GAMMA1 = 63.4715 * np.pi / 180 # rad
THETA1 = 94.7939 * np.pi / 180 # rad
zww    = 4                     # H-bond coordination number
