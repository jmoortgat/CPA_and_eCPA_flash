# Extraction notes: propane solubility (IUPAC SDS Vol. 24, Chapoy 2004, Morrison & Billett 1952)

Date: 2026-08-12. All numeric values were confirmed against visual renderings of the scanned
pages (Read tool, page images), not just the OCR text layer. Setschenow constants recomputable
from the extracted solubilities were recomputed and match the evaluator's printed values
(all 16 Yano-derived kscc exact; Wen & Hung per-concentration ksmm to +-0.001).

Book page = printed page number; PDF page = book page + 21 for the IUPAC volume.

---

## Source 1: IUPAC Solubility Data Series Vol. 24 (Hayduk ed., Pergamon 1986)

### What exists for propane (aqueous systems)

| Section | Book pp. | PDF pp. | Extracted to |
|---|---|---|---|
| Propane in water: critical evaluation (Battino) | 1-2 | 22-23 | `iupac24_propane_water_evaluated.csv` (Table 1) |
| Propane in water: primary sheets (13 sheets) | 3-15 | 24-36 | `iupac24_propane_water_primary.csv` |
| Propane in water, high pressure: evaluation (Young) | 39 | 60 | notes only (no numeric table) |
| Propane in water, high pressure: primary sheets | 40-47 | 61-68 | `iupac24_propane_water_highP_primary.csv`, `_highP_smoothed.csv`, `_threephase_kobayashikatz.csv` |
| Propane+butane in heavy water (D2O) | 54-57 | 75-78 | NOT extracted (out of scope) |
| Aqueous electrolytes: critical evaluation (Clever) | 58-71 | 79-92 | `iupac24_propane_salts_evaluated.csv` |
| Aqueous electrolytes: primary sheets | 73-122 | 94-143 | `iupac24_propane_salts_primary.csv`, `_umano_nacl.csv`, `_setschenow_primary.csv` |

### Propane + water (low pressure)

- Evaluation (Battino, Nov 1983): smoothed x1 at 101.325 kPa **partial** pressure, 273.15-348.15 K
  in 5 K steps, with Ostwald L. Smoothing eq. (273-347 K):
  `ln x1 = -102.044 + 144.345/tau + 39.4740 ln tau`, tau = T/100 K; sd 0.012 in ln x1 (~2%).
  Built on refs (1)-(6); refs (7)-(13) rejected in fitting. Evaluator warns values below 288 K
  may be 2-6% low relative to his own (unpublished) measurements.
- Primary water sheets compiled (authors -> our CSV): Wen & Hung 1970; Kresheck, Schneider &
  Scheraga 1965; Umano & Nakano 1958; Barone et al. 1966; Morrison & Billett 1952; Claussen &
  Polglase 1952; Yano et al. 1974; Wetlaufer et al. 1964; McAuliffe 1966/1963; Wishnia 1963
  (graph-read); Namiot 1961 (Henry constants); Rudakov & Lutsyk 1979; Lebeau 1905.
- Pressure/solubility bases differ per sheet and are preserved in `s_orig_units` (partial- vs
  total-pressure basis flagged; Morrison & Billett, McAuliffe, Umano & Nakano, Lebeau are
  total-pressure data).

### Propane + water (high pressure) - useful for the three-phase paper

- Azarnoosh & McKetta 1958: experimental (288.7-410.9 K, up to ~3.4 MPa) + authors' smoothed
  table (9 isotherms 288.7-410.9 K with LLV-boundary flags). Kobayashi & Katz 1953: 2-phase
  points 310.9-422.0 K + the **three-phase (Lw-Lhc-V) boundary solubility** 278.9-369.5 K
  (16 points, in `iupac24_propane_water_threephase_kobayashikatz.csv`). Wehe & McKetta 1961:
  344.3 K, 0.51-1.25 MPa.
- Evaluation: Azarnoosh & McKetta classed tentative/most extensive; **no recommended values**.
  De Loos 1980 and Sanchez & Coll 1978 (very high P, gas-gas immiscibility) are cited but not
  compiled as solubility sheets.

### Propane + electrolytes (the Setschenow calibration target)

**No propane + seawater data exist in Vol. 24.** The synthetic-sea-salt sheet (book p. 120,
Rice, Gale & Barduhn 1976) is n-BUTANE only (sea salt: 24,067 ppm NaCl, 5,107 ppm MgCl2,
4,016 ppm Na2SO4, 1,130 ppm CaCl2, 680 ppm KCl by weight). No CaCl2 or MgCl2 propane data
exist either (BaCl2, LaCl3 sheets are butane only).

Propane electrolyte systems compiled (with primary authors):

| Electrolyte | Primary source(s) | T range | Conc basis | Our file(s) |
|---|---|---|---|---|
| HCl | Yano et al. 1974 | 298.2 K | mol/dm3, 0-1.0 | salts_primary + salts_evaluated |
| H2SO4 | Rudakov & Lutsyk 1979 | 298.15, 363.15 K | wt%, 0-97.7 | salts_primary + salts_evaluated |
| NH4Br | Wen & Hung 1970 | 278.15-308.15 K | mol/kg, 0-0.94 | salts_primary + salts_evaluated |
| 1-dodecanamine HCl | Lin & Metzer 1971 | 298.2 K | 1e-8-1e-3 M | salts_primary (micellar; no k defined) |
| Guanidine HCl | Wetlaufer et al. 1964 | 278-318 K | 4.87 M | salts_primary + salts_evaluated |
| (CH3)4NBr, (C2H5)4NBr, (C3H7)4NBr, (C4H9)4NBr, (C2H4OH)4NBr | Wen & Hung 1970 | 278.15-308.15 K | mol/kg | salts_primary + salts_evaluated (all salt IN except NH4Br) |
| LiCl | Morrison & Billett 1952 | 285.75-344.85 K | 1.0 mol/kg | setschenow_primary + salts_evaluated |
| NaCl | Morrison & Billett 1952; Umano & Nakano 1958; Yano et al. 1974 | 264.7-344.85 K | m up to 5.315; c up to 1.5 | setschenow_primary, umano_nacl, salts_primary, salts_evaluated |
| NaBr | Yano et al. 1974 | 298.2 K | 0-1.0 M | salts_primary + salts_evaluated |
| Na2CO3 | Yano et al. 1974 | 298.2 K | 0-1.0 M | salts_primary + salts_evaluated |
| KCl | Yano et al. 1974 | 298.2 K | 0-1.5 M | salts_primary + salts_evaluated |
| KBr | Yano et al. 1974 | 298.2 K | 0-0.5 M | salts_primary + salts_evaluated |
| KI | Morrison & Billett 1952 | 285.75-344.85 K | 1.0 mol/kg | setschenow_primary + salts_evaluated |
| Urea (nonelectrolyte) | Wen & Hung 1970; Wetlaufer et al. 1964 | 278-318 K | 0.495 m; 6.96 M | salts_primary + salts_evaluated |

Complete primary-literature inventory for propane + salt solutions (the papers Vol. 24 compiles):

1. Morrison, T.J.; Billett, F. *J. Chem. Soc.* **1952**, 3819 (NaCl, LiCl, KI; also water).
2. Morrison, T.J.; Johnstone, N.B.B. *J. Chem. Soc.* **1955**, 3655 (butane only for salts of interest).
3. Umano, S.; Nakano, Y. *Kogyo Kagaku Zasshi* **1958**, 61, 536 (NaCl 0.52-5.32 m, 264.7-298.2 K; also water).
4. Wetlaufer, D.B.; Malik, S.K.; Stoller, L.; Coffin, R.L. *J. Am. Chem. Soc.* **1964**, 86, 508 (guanidine HCl, urea).
5. Wen, W.-Y.; Hung, J.H. *J. Phys. Chem.* **1970**, 74, 170 (NH4Br, R4NBr series, tetraethanolammonium bromide, urea).
6. Lin, I.J.; Metzer, A. *J. Phys. Chem.* **1971**, 75, 3000 (dodecylamine hydrochloride).
7. Denton, Smith, Klaschka, Forgan, *4th Int. Symp. Fresh Water Sea* **1973**, 3, 291 (butane + NaCl only).
8. Yano, T.; Suetaka, T.; Umehara, T.; Horiuchi, A. *Kagaku Kogaku* **1974**, 38, 320 (HCl, NaCl, NaBr, Na2CO3, KCl, KBr; also water).
9. Rice, P.A.; Gale, R.P.; Barduhn, A.J. *J. Chem. Eng. Data* **1976**, 21, 204 (butane + NaCl, + sea salt only).
10. Zimmels, Y.; Metzer, A. *J. Coll. Interface Sci.* **1976**, 57, 75 (butane + Na oleate only).
11. Rudakov, E.S.; Lutsyk, A.I. *Zh. Fiz. Khim.* **1979**, 53, 1298 (H2SO4).

Primary water-only literature compiled (evaluation refs 1-13): Kresheck/Schneider/Scheraga 1965;
Morrison & Billett 1952; Claussen & Polglase 1952; Wetlaufer et al. 1964; Barone et al. 1966;
Namiot 1961; Wen & Hung 1970; McAuliffe 1966/1963; Wishnia 1963; Rudakov & Lutsyk 1979;
Lebeau 1905; Umano & Nakano 1958; Yano et al. 1974. High-pressure: Kobayashi & Katz 1953;
Azarnoosh & McKetta 1958; Wehe & McKetta 1961 (+ De Loos 1980, Sanchez & Coll 1978, not compiled).

### Setschenow parameter conventions (all log base 10, at 101.325 kPa gas partial pressure)

- `ksmm /kg mol-1 = (1/m2) log10(m1_0/m1)` - gas molality, salt molality (most propane data).
- `ksmx /kg mol-1 = (1/m2) log10(x1_0/x1)` - mole-fraction form; for these dilute systems
  ksmx = ksmm + 0.015 (evaluator's rule, exact within rounding on all M&B sheets).
- `kscc /dm3 mol-1 = (1/c2) log10(c1_0/c1)` - molarity form (Yano, Wetlaufer data).
- `ksmc` - mixed (salt molality, gas molarity), used only in the H2SO4 evaluation.
- Morrison & Billett's own `k = (log s0/s)/c` uses c in g-equiv per kg water (= molality for 1-1
  salts; the IUPAC compiler divided by 2, 3 for BaCl2, LaCl3 - butane systems).
- Wen & Hung's sheet constants are `(1/m3) log10(S1_0/S1)`, S in cm3(STP)/kg == ksmm; the authors
  state their smoothed values apply at m = 0.1 mol/kg.

Evaluation quality labels: everything is "tentative" (or "doubtful": Yano HCl 0.25 M, Yano KCl
0.250 M); **no recommended salt-effect values**. Data from more than one lab exist only for
NaCl and urea. For NaCl the evaluator prefers Morrison & Billett (self-consistent, 60 K span);
inter-laboratory agreement for NaCl is poor (see book Fig. 1, p. 66: M&B 0.216-0.165 vs
Umano-Nakano ~0.11-0.23 with nonmonotonic T-dependence vs Yano 0.233 at 298 K).

### Misprints / items needing a human double-check (all flagged in the CSV `note` columns)

1. Water sheet (book p.5) prints Umano-Nakano 273.2 K first-row p1 = "0.99747" atm; the NaCl
   sheet (p.101) prints 0.09747 for the same point. Book misprint; 0.09747 used, flagged.
2. Umano-Nakano 278.2 K water row: p1 printed 0.09621 (NaCl sheet) vs 0.09601 (water sheet) -
   two printings of the same datum disagree in the 4th digit. Both recorded, flagged.
3. Umano-Nakano 288.2 K / 2.96 wt% first row prints p1 = 0.9943 > p_total = 0.1168 (leading-zero
   misprint for 0.09943). Kept as printed with correction in note.
4. Umano-Nakano 278.2 K / 16.36 wt% third row prints p1 = p_total = 0.5069 (impossible;
   expected ~0.4994). Kept as printed, flagged.
5. Umano-Nakano 293.2 K water first row mol ratio 0.1309 looks low vs the isotherm trend
   (compare 0.2686 at 288.2 K); printed identically in both sheets - original-paper issue.
6. Azarnoosh & McKetta experimental 377.6 K: partial-pressure column appears shifted down one
   row at the last three entries (447.2 psia | 3309 kPa etc.); expected ~2965/3310/3410 kPa.
7. Azarnoosh smoothed 377.6 K, 450 psia: printed p1 = 2904 kPa, arithmetic gives 2984 kPa.
8. NH4Br evaluation row label printed "0.210-9.224" - misprint for 0.210-0.224.
9. Wen & Hung urea sheet (book p.121) Setschenow constants at 298.15 K (-0.10) and 308.15 K
   (-0.012) are inconsistent with the tabulated solubilities (recomputed: -0.009, -0.029) and
   with the evaluation (p.71). Probable misprints in the sheet. Also at 288.15 K the evaluator
   prints -0.002 while recomputation from the sheet solubilities gives -0.006.
10. Guanidine-HCl / Wetlaufer-urea temperatures: evaluation says 278.2/293.2/318.2 K, the data
   sheets (and Wetlaufer's paper) say 278.15/298.15/318.15 K - the evaluator's 293.2 is almost
   surely a typo for 298.2. Both recorded as printed, flagged.
11. Morrison & Billett NaCl sheet (book p.99): mole-fraction k at 303.15 K printed 0.224, which
   breaks the ksmx = ksmm + 0.015 rule obeyed by every other row (expected 0.209). Flagged.
12. Tetraethanolammonium sheet: S0(308.15 K) uncertainty printed "+/-0.007" (likely +/-0.07);
   Pr4NBr propane sheet (book p.89) column header misprinted "Butane Solubility";
   book prints KCl CAS as 7747-40-7 (correct: 7447-40-7); Et4NBr sheet prints S0(288.15)=45.74
   vs 45.75 on all other Wen & Hung sheets.

No entries were illegible; nothing had to be marked MISSING.

---

## Source 2: Chapoy et al., Fluid Phase Equilib. 226 (2004) 213-220

- Extracted: Table 4 (p. 217), all 61 experimental points -> `chapoy2004_solubility.csv`.
- T = 277.62-368.16 K (12 isotherms), P = 0.357-3.915 MPa total, x_C3H8 = 3.21e-5 to 2.70e-4
  in the aqueous (Lw) phase, Lw-V region below the hydrate/LLV boundaries.
- Uncertainty info: no per-point error bars; the AAD(exp) column (kept) is the scatter between
  repeated GC samples at each point. Instrument accuracy from the text: dT <= +-0.04 K,
  dP +-0.1 kPa (low-range transducer) / +-1 kPa (high-range), propane FID calibration ~+-1%.
- The calculated columns (VPT-EoS/NDD xcal, AD%) were NOT extracted (model, not experiment).
  The paper also gives a Henry's-constant correlation (recorded in the CSV header) and water
  content of the gas phase (Table 2 - not extracted; propane-in-water was the assignment).

## Source 3: Morrison & Billett, J. Chem. Soc. 1952, 3819-3822

- Table 2 (p. 3821) propane column -> `morrison1952_propane_water.csv` (12 points, 12.3-74.1 C,
  cc(NTP)/kg water at 1 atm TOTAL pressure).
- Table 3 (p. 3822) propane columns -> `morrison1952_setschenow.csv`: k = (log10 s0/s)/c for
  NaCl, LiCl, KI at 1/T = 0.0035, 0.0033, 0.0031, 0.0029 (12.6/30.0/49.4/71.7 C). Definition:
  log10, c in g-equiv per 1000 g water (= molality here), gas-solubility (not mole-fraction)
  ratio, measured at c = 1; salt-solution solubilities themselves were never published.
- The KCl, HCl, 1/2BaCl2, 1/3LaCl3 columns of Table 3 belong to n-BUTANE, not propane.
- IUPAC Vol. 24 estimated dk = 0.010 for these constants.

---

## File inventory (this extraction; prefixes iupac24_/chapoy2004_/morrison1952_ only)

| File | Rows | Content |
|---|---|---|
| iupac24_propane_water_evaluated.csv | 16 | Battino smoothed x1(T), L(T), 273-348 K |
| iupac24_propane_water_primary.csv | 84 | 13 primary water sheets, original units |
| iupac24_propane_water_highP_primary.csv | 102 | Azarnoosh, Wehe, Kobayashi-Katz experimental |
| iupac24_propane_water_highP_smoothed.csv | 95 | Azarnoosh smoothed isotherms w/ LLV flags |
| iupac24_propane_water_threephase_kobayashikatz.csv | 16 | x_C3 in water along LLV boundary |
| iupac24_propane_salts_evaluated.csv | 190 | Clever's Setschenow tables, propane rows |
| iupac24_propane_salts_primary.csv | 153 | solubility-vs-conc sheets (Wen&Hung, Yano, Rudakov, Lin, Wetlaufer) |
| iupac24_propane_salts_umano_nacl.csv | 151 | Umano-Nakano NaCl P-x, 264.7-298.2 K, 0-5.3 m |
| iupac24_propane_salts_setschenow_primary.csv | 12 | Morrison-Billett k-only sheets (NaCl/LiCl/KI) |
| chapoy2004_solubility.csv | 61 | propane-in-water, 277.6-368.2 K, to 3.9 MPa |
| morrison1952_propane_water.csv | 12 | original water solubilities |
| morrison1952_setschenow.csv | 12 | original salting-out constants |

(The kobayashi1951_* files and extraction_notes.md in this directory are from a concurrent
extraction of a different source PDF and are not part of this set.)
