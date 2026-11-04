# Published and released numbers in Papers II and III

## Paper II: CH4 tables

The code in this directory reproduces the CO2 results of Paper II: all 52
rows of the CO2–brine table of the Supporting Information and the AAD per
salt in Section 3.4 (5.4, 7.2, 11.1, 7.1 and 4.3 %).

It does not reproduce two CH4 tables of the Paper II Supporting Information,
the CH4–H2O table and the CH4–brine table. Those tables were computed with the
earlier code from which this package was restructured, and the solver has
been corrected since. The difference has not been traced to a single change.
With the released code, the CH4 solubility in
water has a lower AAD for most sources (overall 7.6 % instead of 11.1 %). The model
parameters are the same in both versions.

Some rows also differ because the CH4 data set of GasBrineBench v1.1.1,
which the released code reads, is not identical to the one used for the
tables: it adds
Duffy et al. (1961) for CH4 in brines and Mohammadi et al. (2004) and
Tabasinejad et al. (2011) for the water content, and it does not contain the
CH4–NaCl data of O'Sullivan and Smith (1970). The added water-content data
account for the 262 instead of 196 points (17 + 38 + 11 more for Olds et
al., 1942).

The tables below list only the rows that differ. They are produced by
`scripts/build_paper_tables.py` (output in `paper_tables/tables/`).
"-" means that the row is not in that table.

### CH4 in water (CH4–H2O table)

| Reference | N (solubility) published / released | AAD solubility (%) published / released | N (water content) published / released | AAD water content (%) published / released |
|---|---|---|---|---|
| OU2015 | 204 / 204 | 8.0 / 6.6 | 0 / 0 | --- / --- |
| Carroll1998 | 43 / 43 | 18.6 / 4.3 | 0 / 0 | --- / --- |
| Culberson1950 | 72 / 72 | 6.8 / 2.9 | 0 / 0 | --- / --- |
| DUFFY1961 | 17 / 17 | 13.0 / 8.5 | 0 / 0 | --- / --- |
| Kim2003 | 6 / 5 | 3.5 / 3.9 | 0 / 0 | --- / --- |
| Susak1980 | 87 / 87 | 8.5 / 6.9 | 0 / 0 | --- / --- |
| Campos2010 | 15 / 15 | 84.9 / 35.1 | 0 / 0 | --- / --- |
| Olds1942 | 0 / 0 | --- / --- | 65 / 76 | 8.1 / 7.4 |
| Kiepe2003 | 26 / 26 | 27.5 / 5.4 | 0 / 0 | --- / --- |
| OSullivan1970 | 18 / 18 | 5.0 / 4.9 | 0 / 0 | --- / --- |
| Sultanov1972 | 71 / 71 | 9.4 / 8.1 | 0 / 0 | --- / --- |
| Price1979 | 71 / 71 | 16.3 / 14.5 | 0 / 0 | --- / --- |
| Global | 836 / 835 | 11.1 / 7.6 | 196 / 262 | 19.4 / 16.0 |
| Mohammadi2004 | - / 0 | - / --- | - / 17 | - / 2.0 |
| Tabasinejad2011 | - / 0 | - / --- | - / 38 | - / 8.7 |

### CH4 in brines (CH4–brine table)

| Salt | Reference | N published / released | AAD (%) published / released |
|---|---|---|---|
| NaCl | STOESSELL1982 | 12 / 12 | 6.3 / 6.4 |
| NaCl | OSullivan1970 | 32 / - | 7.4 / - |
| NaCl | Global | 372 / 362 | 9.6 / 10.4 |
| KCl | STOESSELL1982 | 12 / 12 | 38.2 / 38.1 |
| CaCl2 | Blanco1978 | 30 / 30 | 4.3 / 4.8 |
| CaCl2 | STOESSELL1982 | 9 / 9 | 6.9 / 4.5 |
| CaCl2 | Global | 39 / 65 | 4.9 / 8.4 |
| Na2SO4 | STOESSELL1982 | 6 / 6 | 1.8 / 1.9 |
| Na2SO4 | Global | 6 / 6 | 1.8 / 1.9 |
| MgSO4 | STOESSELL1982 | 9 / 9 | 1.8 / 1.7 |
| MgSO4 | Global | 9 / 9 | 1.8 / 1.7 |
| NaCl | DUFFY1961 | - / 22 | - / 19.5 |
| CaCl2 | DUFFY1961 | - / 26 | - / 13.8 |

## Paper II: salinity-coverage table (Table S8)

Table S8 of the Paper II Supporting Information states that its molality
ranges are those of Tables S5 and S7. For the CO2–brine column, the published
lower ends for NaCl, CaCl2 and MgCl2 (1, 0.67 and 0.67 mol/kg) do not agree
with the published Table S5, which contains data points down to 0.17, 0.09
and 0.11 mol/kg. The upper ends, the density column and the maximum ionic
strengths agree. The released Table S5 is identical to the published one, so
the ranges computed from it give the Table S5 values. The CH4–brine column
also differs because of the added data described above (NaCl 0.5–6 instead
of 0.5–4.3, CaCl2 0.25–5.6 instead of 0.5–2 mol/kg).

## Paper III: CO2–H2O three-phase locus

`pilot_studies/three_phase_2026_08/vlle_locus.py` computes the three-phase
(vapor + aqueous + liquid CO2) pressure P3(T) of the CO2–H2O binary and
compares it with the CO2 saturation pressure of the Span–Wagner equation of
state. The released code reproduces Figure 6 of Paper III, including the
point at 302.15 K, 2 K below the CO2 critical temperature, which lies
0.27 bar (0.38 %) above the reference curve (P3 = 70.77 bar against
70.51 bar).

The text of Paper III (Section 3.4 and the abstract) states that the
predicted locus lies 0.2–1.1 bar (0.4–1.5 %) below the reference curve over
283–302 K. That range agrees with the released values at 283.15–300.15 K
(0.18–0.87 bar, 0.40–1.29 % below) but not with the point at 302.15 K shown
in the same figure. The released values are in
`pilot_studies/three_phase_2026_08/vlle_locus.csv`.

## Paper III: other differences

* **Table 3, CH4 entries.** The CH4 accuracies of Table 3 come from the same
  earlier code as the Paper II CH4 tables. The released code gives 7.6 % for
  CH4 in water (835 data points; published 11.1 %, 836 points) and 8.16 % for
  CH4 in brines (485 points; published 7.25 %, 469 points). The other rows of
  Table 3 are reproduced.
* **Table S10.** At 0.5 times the tabulated k_hc, the released code gives an
  AARE of 32.9 % for the gas-phase water content (32.95 % before rounding);
  the published value is 33.0 %. The other eleven entries are reproduced.
* **Table S11.** The 95th percentile of the wall time per call for the binary
  CO2–H2O feed is 12,705.5 ms, which rounds to 12,706 ms; the published value
  is 12,705 ms.

The other tables that the notebook computes (Tables 1, 4, S8, S9, S11–S13
and S15) agree with the published values. Tables S7 (Born energies) and S14
(SSI with and without the Newton polish) come from computations that are not
part of the release; the notebook shows them as published, as it does the
parameter and tolerance tables (Tables 2, S1 and S6).

## Stability test of water-rich brines (fixed in the release)

The stability test of the published code compares trial phases with the gas-side
equation of state at the feed composition, which has no ions. For a brine feed
the gas trials therefore did not depend on salinity. A brine with more dissolved
CO2 than its saturation value, but less than the salt-free saturation value, was
reported single-phase. The released code tests such a feed again against the
brine at the feed composition (`BRINE_REFERENCE_FOR_WATER_RICH` in
`ms_ecpa/stability.py`); setting it to False recovers the published test.

No published number changes:

* The flash grid of Paper III has z_H2O <= 0.75. The second test applies only
  for z_H2O >= 0.9.
* The CO2–brine predictions of both papers use the feed z_CO2 = 0.05 first; it
  gave a split at every data point with both versions. The ternary figures use
  z_gas = 0.20 at every point.
* The CH4 predictions that use the stability test (the CH4–H2O table and the
  Susak (1980) points above 532 K) are identical with and without the second
  test (1,063 and 64 data points).
* Of the 1,920 states in `reproduce/reference_states.json`, one changes: at
  523.15 K, 500 bar, 1 mol/kg MgSO4 and z_CO2 = 0.05 (saturation 0.0449) the
  published test gives one phase and the released test a split with a gas
  fraction of 0.008.
