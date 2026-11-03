# H2 in water and NaCl brines

This folder computes the H2 solubilities of Paper III. The H2
parameters of Coelho et al. (2025) are used in ms-eCPA without any
refitting, so all results are predictions. The calculations give the
H2–water and H2–NaCl-brine AAREs reported in Paper III and the model
values for the H2 solubility figure in water and NaCl brines. They also
compare H2–CO2–brine results with the model curves of Coelho et al.

> F. M. Coelho, L. F. Mercier Franco, A. Firoozabadi. Modeling of phase
> equilibria of H2 and H2–CO2 in NaCl brine mixtures from eCPA equation
> of state. *Int. J. Hydrogen Energy* **2025**, 178, 151361.
> doi:10.1016/j.ijhydene.2025.151361

## How to run

From the `multisalt/` directory, one stage per call:

```bash
for s in refs v1water v1brine v3 curves figure finalize; do
    python3 pilot_studies/h2_pilot_2026_07/h2_pilot.py --stage $s
done
```

| Stage | Computation |
|---|---|
| `refs` | CH4–NaCl and CO2–NaCl reference flashes with the unmodified parameters |
| `v1water` | x_H2 and y_H2O at the 216 water data points |
| `v1brine` | m_H2 at the 71 NaCl data points; check that a KCl feed is rejected |
| `v3` | H2–CO2–NaCl flashes and the effect of the H2–CO2 kij |
| `curves`, `figure` | model curves and `fig_h2_brine_pilot.png` |
| `finalize` | reference-flash check and `h2_results.json` |

Stages `v2fit` and `v2eval` (a refit of U_ref to the brine data) exist
in the script but were not used; Paper III uses no fitted H2 parameter.
Stage `v3` compares with the model curves of Coelho et al., which are
read from `$ECPA_SOURCE_MATERIALS/EoS/H2/HCB/` and are not distributed;
without them the comparison is skipped. The Paper III figure
(`h2_brine_main.png`) is produced by `scripts/fig_additional_gases.py`.

## Inputs and outputs

Inputs: `pilot_studies/h2_scoping_2026_07/h2_water_binaries.csv` and
`h2_nacl_brine.csv` (sources listed in the README of that folder).

Outputs: `h2_results.json` (parameters, AAREs by subset and by source,
model value at every data point, ternary comparison, checks),
`work_state.json`, `curves_cache.csv`, `fig_h2_brine_pilot.png`.

## Model

H2 replaces CH4 as component 2. The script sets the component-2
parameters of `ms_ecpa` at run time and restores the CH4 values after each
calculation; `ms_ecpa/` is not modified. CH4–NaCl and CO2–NaCl flashes at
380 K, 200 bar and 1 mol/kg NaCl are bit-identical before and after the
H2 calculations.

Parameters (README of `h2_scoping_2026_07`): Tc = 33.2 K, Pc = 13.0 bar,
ω = −0.220, M = 0.002 kg/mol, polarizability 8.756e-41 C m²/J; water–H2
kij (A, B, C) = (−7.766172, 12.307805, −4.590648); H2–ion energy
U_hs(T) with U_ref = 9992.89 J/mol, α = −9889.08 K, Tα = 364.487 K. At
each data point the Na+ and Cl− energies are set to A_U2i = 0 and
B_U2i = U_hs(T), which reproduces U_hs(T) exactly. Feeds containing K+,
Ca2+, Mg2+ or SO4 2− are rejected, because no H2 energies exist for these
ions. The H2–CO2 kij is 0.1277. Molalities are m_H2 = x2w/(x1w Mw).

A flash result is accepted when two feeds give x_H2 within 0.5 % and both
vapor fractions are below 0.995. At high temperature and low pressure,
`stability_flash` can return a solution whose aqueous H2 content changes
with the feed. Such points are recomputed with feeds at 0.95 and 0.99
times the gas-phase H2 fraction, extrapolated linearly to the dew point.
This was needed for 29 of the 216 water data points and none of the
brine data points.

## Results

AARE [%] (from `h2_results.json`); the values of Coelho et al. (2025)
are for their own model:

| Data | n | ms-eCPA | Coelho et al. |
|---|---|---|---|
| water, x_H2 | 163 (153 converged) | 10.98 | 8.97 |
| water, x_H2, without Pray (1957) and Dohrn (1986) | 149 | 9.30 | 8.97 |
| water, y_H2O | 78 (77 converged) | 12.37 | 11.45 |
| NaCl brine, m_H2 | 71 (71 converged) | 9.13 | 8.64 |

Pray (1957) and Dohrn (1986) were not used in the fit of Coelho et al.
The 10 water data points without a converged two-phase result are at
523–589 K; 6 of them (Pray 1957) are below the vapor pressure of water.

Brine AARE by molality: 1 mol/kg 3.48, 2 mol/kg 7.79, 2.5 mol/kg 19.99
(all Torín-Ollarves 2022), 3 mol/kg 10.69, 4 mol/kg 11.96,
5 mol/kg 11.82 %. The mean deviation at 2, 3, 4 and 5 mol/kg is −7.1,
+10.7, −6.7 and +11.8 %, so it does not increase with molality.

H2–CO2–NaCl ternary at a gas feed fraction of 0.30 and a water-free CO2
feed fraction of 0.05–0.95. All 36 flashes converged. Mean absolute
deviation from the model curves of Coelho et al. at equal water-free
gas CO2 fraction:

| T [K], P [bar] | m_H2, 1 / 4 mol/kg | m_CO2, 1 / 4 mol/kg |
|---|---|---|
| 323, 200 | 0.23 / 1.54 % | 0.60 / 3.20 % |
| 323, 400 | 0.25 / 1.37 % | 0.63 / 3.11 % |
| 423, 200 | 2.31 / 3.06 % | 3.11 / 3.30 % |

This is a comparison between two models, not with measurements. With
an H2–CO2 kij of 0.145 instead of 0.1277, m_H2 changes by at most 0.10 % and m_CO2
by at most 0.24 % at the three points tested.
