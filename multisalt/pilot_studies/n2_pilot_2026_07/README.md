# N2 in water and NaCl brines

This folder computes the N2 parameters and N2 solubilities of Paper III.
The water–N2 binary interaction parameter kij(T) is fitted to salt-free
data only. The N2–ion interaction energies are taken from the CH4–ion
energies of ms-eCPA (Paper II) without refitting, so that all NaCl-brine
results are predictions. A one-parameter brine fit is computed for
comparison. The results support the N2 solubility figure in water and
NaCl brines and the N2 parameter values and AAREs reported in Paper III.

## How to run

From the `multisalt/` directory:

```bash
python3 pilot_studies/n2_pilot_2026_07/n2_pilot.py --stage all
```

or one stage at a time, in this order (`--stage` choices):

| Stage | Computation |
|---|---|
| `refs` | CH4–NaCl and CO2–NaCl reference flashes with the unmodified parameters |
| `b0` | kij from the literature (case B0, below) and predictions at all 51 data points |
| `fitB` | fit of kij(T) to the 18 water data points, three starting points |
| `evalB` | case B predictions at all 51 data points |
| `fitA` | fit of the brine scaling factor s to the 33 brine data points, three starting points |
| `evalA` | case A predictions; sensitivity to the N2 polarizability |
| `curves` | model curves for the figure |
| `figure` | `fig_n2_brine_pilot.png` |
| `finalize` | reference-flash check, AARE table, `n2_results.json` |

Intermediate results are stored in `work_state.json` between stages. The
fit stages use 8 worker processes. The Paper III figure itself
(`n2_brine_main.png`) is produced by `scripts/fig_additional_gases.py`,
which reads the fitted values from `n2_results.json`.

## Inputs and outputs

Input: `pilot_studies/n2_scoping_2026_07/n2_nacl_osullivan1970.csv`,
51 N2 data points from O'Sullivan and Smith, *J. Phys. Chem.* **1970**, 74,
1460–1466 (see the README in that folder).

Outputs: `n2_results.json` (constants, fits, AAREs, model value at every
data point, reference-flash check), `work_state.json`, `curves_cache.csv`
(model curves), `fig_n2_brine_pilot.png`.

## Model

N2 replaces CH4 as component 2. The script sets the component-2
parameters of `ms_ecpa` at run time and restores the CH4 values after each
calculation; `ms_ecpa/` is not modified. The `finalize` stage checks that
CH4–NaCl and CO2–NaCl flashes at 380 K, 200 bar and 1 mol/kg NaCl give
bit-identical results before and after the N2 calculations.

Pure N2 (SRK with the Soave alpha function, no association sites):
Tc = 126.192 K, Pc = 33.958 bar, ω = 0.0372, M = 0.0280134 kg/mol,
b = 0.08664 RTc/Pc, a0 = 0.42748 R²Tc²/Pc,
c1 = 0.480 + 1.574ω − 0.176ω². The polarizability is
α = 1.7403 Å³ (1.9360e-40 C m²/J). The CO2–N2 kij is set to zero; it has
no effect because the feeds contain no CO2.

Water–N2 kij, in the form used for CH4 in Paper II (Tc1 = 647.29 K):

    k12(T) = Ak (T/Tc1)^2 + Bk (T/Tc1) + Ck

Gas–ion energies have the form U2i = A_U2i (T/Tc2) + B_U2i. To keep
U2i(T) unchanged when Tc2 changes from CH4 to N2, the A coefficients are
multiplied by Tc(N2)/Tc(CH4) = 126.192/190.56 = 0.66222 and the B
coefficients are kept.

Feeds are z = (1 − zg, zg, 0) with zg = 0.05; if the flash does not give
two phases, zg = 0.10, 0.02, 0.20 and 0.30 are used in turn. All 51 data
points converged in all three cases. The model aqueous mole fractions
(dissociated basis) are converted to the basis of O'Sullivan and Smith
(NaCl undissociated) with

    X2 = x2w / (x1w + x2w + Mw x1w m_NaCl)

which reduces to x2w in pure water. Pressures are converted from atm to
bar (× 1.01325).

## Cases and results

- **B0.** kij(T) = 1.0741 − 368.3/T from B. Maribo-Mogensen, PhD thesis,
  Technical University of Denmark (2014), Table 7.1, re-expressed in the
  quadratic form above by least squares over 315–410 K (maximum
  deviation 9.9e-4 in kij): (Ak, Bk, Ck) = (−3.2883, 5.5164, −1.9999).
  No N2 data are used. That kij was determined with a different CPA
  parameter set for water.
- **B.** (Ak, Bk, Ck) fitted to the 18 water data
  points: **Ak = 5.51592, Bk = −3.96787, Ck = 0.46600**. This gives
  k12 = −0.136 at 324.65 K and +0.112 at 398.15 K. The three starting
  points converged to the same minimum (cost 0.0016387); the RMS relative
  deviation is 1.35 %. With three coefficients and three isotherms, the
  fit should not be extrapolated far outside 324.65–398.15 K.
- **A.** Case B kij plus one factor s that multiplies the A and B
  coefficients of both Na+ and Cl−. NaCl data alone cannot separate the
  Na+ and Cl− energies, so the ratio between them is kept from CH4. The
  fit gives **s = 1.0084** from all three starting points.

AARE [%] of x_N2 (from `n2_results.json`):

| Case | water (18) | 1 mol/kg (18) | 4 mol/kg (15) | brine (33) | all (51) |
|---|---|---|---|---|---|
| B0 | 15.11 | 16.04 | 9.59 | 13.11 | 13.81 |
| B | 1.09 | 3.51 | 3.84 | 3.66 | 2.75 |
| A | 1.09 | 3.77 | 3.10 | 3.46 | 2.63 |

In case B the bias is −2.7 % at 1 mol/kg and +2.6 % at 4 mol/kg. By
isotherm, the brine AAREs are 4.41/3.09 % (324.65 K, 1/4 mol/kg),
4.35/2.14 % (375.65 K) and 1.78/6.28 % (398.15 K). The brine fit (case A)
changes the transferred energies by 0.84 % and lowers the brine AARE by
0.20 percentage points. In case B0 the mean deviation is negative at every isotherm:
−20 to −22 % at 324.65 K and −3 to −7 % at 398.15 K.

Polarizability sensitivity: replacing the N2 value by the CH4 value
(2.6155e-40 C m²/J) increases x_N2 by 1.2 % at 324.65 K, 100 atm,
1 mol/kg and by 4.3–4.6 % at 600 atm, 4 mol/kg.

The Paper III figure shows cases B and A.

Limitations: NaCl is the only salt with N2 data here, so the separate
Na+ and Cl− energies of N2 are not determined by data.
