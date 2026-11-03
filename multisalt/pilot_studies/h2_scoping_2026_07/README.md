# H2 data and the H2 parameters of Coelho et al. (2025)

This folder holds the experimental H2 data and documents the H2
parameters used for the H2 results of Paper III (the H2 solubility
figure in water and NaCl brines). The parameters are those published in

> F. M. Coelho, L. F. Mercier Franco, A. Firoozabadi. Modeling of phase
> equilibria of H2 and H2–CO2 in NaCl brine mixtures from eCPA equation
> of state. *Int. J. Hydrogen Energy* **2025**, 178, 151361.
> doi:10.1016/j.ijhydene.2025.151361

They are used without refitting by
`pilot_studies/h2_pilot_2026_07/h2_pilot.py`.

## Data files

The CSVs were extracted by `extract_h2_data.py` from the data files
compiled by F. M. Coelho for Coelho et al. (2025). Values were copied
without change; `X` (not measured) became an empty cell. The original
files are not distributed. To regenerate the CSVs, set
`ECPA_SOURCE_MATERIALS` to the directory that contains `EoS/H2/` and run
`python3 pilot_studies/h2_scoping_2026_07/extract_h2_data.py`. Each CSV has
the columns given below plus `author` and `source_file`.

| File | Columns | Content |
|---|---|---|
| `h2_water_binaries.csv` | `T_K, P_bar, x_H2_aq, y_H2O_gas` | 216 data points, 19 isotherms, 273–589 K, 3.4–1013 bar; 163 values of x_H2 and 78 of y_H2O |
| `h2_nacl_brine.csv` | `T_K, P_bar, m_NaCl_molkg, m_H2_molkg` | 71 data points, 298–423 K, 20–458 bar, 1–5 mol/kg NaCl |
| `h2_water_isobaric.csv` | `T_K, P_bar, x_H2_aq, y_H2O_gas` | 58 data points of x_H2, 286–617 K, 1–207 bar |
| `h2_co2_elv.csv` | `T_K, P_bar, x_H2_liqCO2, y_H2_gas` | 141 H2–CO2 data points, 220–290 K |

Sources (number of data points):

- Water: Wiebe (1934) 40, Jung (1968) 38, Ipatev (1934) 21,
  Gillespie (1980) 18, Chabab (2024) 9, Kling (1991) 9, Devaney (1978) 9,
  Pray (1957) 9, Chabab (2020) 6, Torín-Ollarves (2022) 5, Dohrn (1986) 1;
  y_H2O only: Maslennikova (1976) 46, Bartlett (1927) 5.
- NaCl brine: Chabab (2020) 31, Chabab (2024) 30, Torín-Ollarves (2022) 10.
- Isobaric water data: Zoss (1952) 25, Alvarez (1988) 21,
  Morrison (1952) 12.
- H2–CO2: Tsang (1981) 141.

Full references are given in Coelho et al. (2025). Only
`h2_water_binaries.csv` and `h2_nacl_brine.csv` are read by the H2 pilot
and by `scripts/fig_additional_gases.py`.

## Parameters of Coelho et al. (2025)

Pure H2: SRK with the Soave alpha function and no association sites.
Tc = 33.2 K, Pc = 13.0 bar, ω = −0.220, M = 0.002 kg/mol. Then
b = 0.08664 RTc/Pc = 18.396 cm³/mol, a0 = 0.42748 R²Tc²/Pc =
0.02505 J m³/mol² and c1 = 0.480 + 1.574ω − 0.176ω² = 0.1252. With the
negative ω, the alpha function stays positive up to about 2680 K.
Polarizability 0.787 Å³ = 8.756e-41 C m²/J.

Water–H2 kij, no cross-association (Tc1 = 647.29 K):

    k12(T) = A (T/Tc1)^2 + B (T/Tc1) + C
    A = -7.766171507123477, B = 12.307804628800731, C = -4.590647844067419

This gives k12 = −0.570 at 298 K, −0.077 at 373 K and +0.136 at 423 K.
The water CPA parameters of Coelho et al. (2025) are the same as those of
ms-eCPA, so this kij applies to ms-eCPA without change.

H2–ion energy: one salt-specific value for NaCl, used for both Na+ and
Cl−:

    U_hs(T) = U_ref + α R [(1 − T/Tα)^2 − (1 − Tref/Tα)^2],  Tref = 298.15 K
    U_ref = 9992.893 J/mol, α = −9889.080 K, Tα = 364.487 K

U_hs is 9993 J/mol at 298.15 K, has a maximum of 12716 J/mol at Tα and
is about 10600 J/mol at 423 K. A positive U_hs means salting-out. No
H2 energies exist for K+, Ca2+, Mg2+ or SO4 2−.

H2–CO2 kij (T-independent, from second virial coefficients): Coelho et
al. (2025) report 0.145; their H2–CO2–brine model curves were computed
with 0.1277. The pilot uses 0.1277 and reports the effect of 0.145.

## Differences from ms-eCPA

The gas–ion energy of ms-eCPA has the linear form A (T/Tc2) + B. A
linear form cannot represent the maximum of U_hs(T). Because a flash is
computed at one temperature, the pilot sets A = 0 and B = U_hs(T) for
each data point, which reproduces U_hs exactly.

Coelho et al. (2025) fitted U_hs with a salt-specific NaCl model: one
water–salt energy, zero ion dispersion parameters, a constant volume
shift per mole of salt, and ion diameters and Born radii that differ
from those of ms-eCPA. ms-eCPA is ion-specific: it has separate
water–ion energies, nonzero dispersion parameters and temperature-
dependent volume shifts for Na+ and Cl−. Using U_hs in ms-eCPA is
therefore a prediction, not a reproduction of the published curves.
The volume shifts do not affect phase compositions.
