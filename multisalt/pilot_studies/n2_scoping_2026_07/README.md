# N2 solubility data in water and NaCl brines

This folder holds the experimental N2 solubility data used for the N2
results of Paper III (the N2 solubility figure in water and NaCl brines).
The data are read by `pilot_studies/n2_pilot_2026_07/n2_pilot.py` and by
`scripts/fig_additional_gases.py`. There is no script in this folder.

## Source

T. D. O'Sullivan, N. O. Smith. The Solubility and Partial Molar Volume of
Nitrogen and Methane in Water and in Aqueous Sodium Chloride from 50 to
125 °C and 100 to 600 atm. *J. Phys. Chem.* **1970**, 74 (7), 1460–1466.
Table I, p. 1461.

The values were transcribed from the text layer of the article and every
entry was checked against an image of the printed page. The source PDF is
not distributed with the repository.

## File

`n2_nacl_osullivan1970.csv` (comment lines start with `#`), columns:

| Column | Meaning |
|---|---|
| `gas` | `N2` or `CH4` |
| `T_C`, `T_K` | temperature; `T_K = T_C + 273.15` |
| `P_atm` | total pressure [atm], as printed |
| `m_NaCl_mol_kg` | NaCl molality of the solvent [mol/kg]: 0, 1.000 or 4.00 |
| `x1e4_printed` | mole fraction of dissolved gas × 10^4, as printed |
| `x_molefrac` | `x1e4_printed × 1e-4` |

Content: 51 N2 data points (18 in water, 18 in 1.000 mol/kg NaCl, 15 in
4.00 mol/kg NaCl) at 51.5, 102.5 and 125.0 °C and 100–608 atm. The table
has no 4 mol/kg N2 measurement at the lowest pressure of each isotherm.
The 50 CH4 data points of the same table are included for comparison
with the CH4 data used in Paper II; they are not used in the N2
calculations.

## Conventions of the source

- Pressure is the total pressure. The authors state a maximum error of
  0.5 atm below 100 atm and 1 atm at 600 atm.
- The mole fraction X2 was calculated with NaCl treated as one
  undissociated component. Model mole fractions on the dissociated,
  ion-inclusive basis must be converted before comparison; the
  conversion is given in `pilot_studies/n2_pilot_2026_07/README.md`.
- O'Sullivan and Smith report that the salting-out coefficients of N2
  and CH4 in NaCl solutions are nearly equal. The calculations in
  `pilot_studies/n2_pilot_2026_07` test this observation by using the
  CH4–ion interaction energies of ms-eCPA for N2.
