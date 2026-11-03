# Three-phase equilibrium: vapor, aqueous liquid and CO2-rich liquid

This folder computes the three-phase (vapor + aqueous liquid + CO2-rich
liquid) results of Paper III. The first is the three-phase pressure
P3(T) of the CO2–H2O binary, shown with the CO2 saturation pressure in
the three-phase locus figure. The second is a set of three-phase
stability and flash calculations for CO2–CH4–H2O feeds with and without
salt. The method is
implemented in `ms_ecpa/flash3.py` (entry point `stability_flash3`).

## How to run

From the `multisalt/` directory:

```bash
python3 pilot_studies/three_phase_2026_08/vlle_locus.py
python3 pilot_studies/three_phase_2026_08/sweep3.py
python3 pilot_studies/three_phase_2026_08/timing_probe.py
```

The scripts have no command-line options. `vlle_locus.py` needs CoolProp.
The unit tests are in `tests/test_flash3.py` (`python3 -m pytest
tests/test_flash3.py`).

## Outputs

| Script | Output | Content |
|---|---|---|
| `vlle_locus.py` | `vlle_locus.csv`, `fig_vlle_locus.png` | P3(T) of the CO2–H2O binary at 283.15–302.15 K and the CO2 saturation pressure |
| `sweep3.py` | `sweep3.parquet` | phase count, phase fractions, stage-2 TPD, fugacity residuals and times at 2,625 states |
| `timing_probe.py` | `timing_probe.csv` | time of each stage of `stability_flash3` at 495 states |

`sweep3.parquet` is not distributed; `sweep3.py` writes it.

## Method

The aqueous phase is the reference phase. The vapor and the CO2-rich
liquid have the K-values Kg = y/x and Kl = xl/x on the (H2O, CH4, CO2)
basis. Both non-aqueous phases use the gas-phase equation of state; the
CO2-rich liquid is its dense root. The phase fractions solve the
three-phase Rachford–Rice equations as the minimum of Michelsen's convex
function −Σ zi ln Di, with Di = 1 + βg(Kgi − 1) + βl(Kli − 1). A damped
Newton method keeps the fractions within [−0.5, 1.5]. A minimum outside
the physical range means that one phase is absent. Ions stay in the
aqueous phase and do not enter the Rachford–Rice equations. The flash
uses successive substitution in ln K, with acceleration, followed by
Newton iterations.

`stability_flash3` has three stages: (1) the two-phase `stability_flash` of
ms-eCPA; (2) a tangent-plane-distance test of the converged
two-phase state with dense (liquid CO2) and vapor trial phases; (3) the
three-phase flash, seeded from the two-phase K-values and the trial
phase. If a phase vanishes or the flash fails, the two-phase result is
returned.

Three numerical points determine the design:

1. In the CO2–H2O binary the vapor and the CO2-rich liquid have nearly
   the same composition and differ mainly in density. Trial phases are
   therefore compared by compressibility factor Z, not by composition.
2. A Newton iteration started from a dense seed can converge to the
   middle, mechanically unstable root at low temperature (observed at
   274 K, 44 bar). The gas-phase solver therefore runs a set of seeds
   and takes the smallest Z for a liquid and the largest Z for a vapor.
3. K-values for the new phase are computed from fugacity coefficients,
   Ki = scale · exp(ln φi,aq − ln φi,trial). Composition-ratio K-values
   are nearly parallel to Kg in the binary and make the Rachford–Rice
   Hessian singular.

## Results

P3(T) is the pressure at which the stage-2 TPD changes sign. It is
compared with the pure-CO2 saturation pressure of Span and Wagner,
computed with CoolProp. The measured three-phase line of Wendland,
Hasse and Maurer (*J. Chem. Eng. Data* **1999**, 44, 901–906) follows
the CO2 saturation pressure to about 0.3 bar. The temperatures start at
283.15 K, the lowest temperature of the CO2–H2O parameter fit.

| T [K] | P3, model [bar] | Psat CO2 [bar] | ΔP [%] |
|---|---|---|---|
| 283.15 | 44.84 | 45.02 | −0.40 |
| 286.15 | 48.23 | 48.47 | −0.48 |
| 289.15 | 51.81 | 52.11 | −0.58 |
| 292.15 | 55.56 | 55.96 | −0.71 |
| 295.15 | 59.50 | 60.03 | −0.89 |
| 298.15 | 63.63 | 64.34 | −1.11 |
| 300.15 | 66.49 | 67.36 | −1.29 |
| 302.15 | 70.77 | 70.51 | +0.38 |

Values from `vlle_locus.csv`. From 283.15 to 300.15 K the model pressure
is below the saturation pressure, and the difference increases toward
the CO2 critical temperature (304.1 K).

The grid of `sweep3.py` is T = 283.15–303.15 K (5 values) and
P = 40–80 bar (step 2 bar). It has five feeds z = (0.50, zCH4,
0.50 − zCH4), with zCH4 = 0, 0.01, 0.03, 0.06 and 0.10. It has five
brines: no salt, NaCl 1 and 3 mol/kg, KCl 1 mol/kg and CaCl2 1 mol/kg. With CH4 in the feed the
three-phase states form a finite region in (T, P) instead of a line.

`timing_probe.csv` contains 495 states (3 feeds, 3 brines, 5
temperatures, 11 pressures): 427 two-phase and 68 three-phase states.
The three-phase flashes took a median of 4 and at most 6
successive-substitution iterations. The times in the file depend on the
computer used.
