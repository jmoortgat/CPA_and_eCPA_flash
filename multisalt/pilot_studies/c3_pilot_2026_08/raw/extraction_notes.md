# Kobayashi (1951) propane-water data extraction notes

Source: a scan of the dissertation (144 PDF pages, with an OCR text layer). R. Kobayashi, "Vapor-Liquid Equilibria in Binary
Hydrocarbon-Water Systems", PhD dissertation, University of Michigan, 1951.
Published companion: Kobayashi & Katz, Ind. Eng. Chem. 1953, 45, 440.

Page bookkeeping: dissertation (printed) page N = PDF page N + 16
(e.g. diss. p.14 = PDF p.30).

## Method

The OCR text layer was used only to locate tables (List of Tables on PDF
pp.7-8). Every numeric value was transcribed by visually reading 300-dpi
`pdftoppm` renders of the pages. The two landscape tables (VII, VIII) and
Table XI were rotated 90 degrees and read in magnified halves; their OCR text
is complete garbage. Unit conversions (T_K = (F-32)/1.8 + 273.15,
P_bar = psia * 0.0689476) were computed programmatically, never by hand.

## Composition basis

All compositions are **mole fractions as printed**. Every table header in the
source reads "Mol Fraction Water/Propane ..."; there is no mole-ratio or
scaled (x 10^4) basis anywhere in these tables. Values are transcribed
verbatim (leading 0 added before the decimal point).

## Tables extracted -> CSV files

| CSV | Source table | Diss. page (PDF page) | Rows | Content |
|---|---|---|---|---|
| `kobayashi1951_3phase_yH2O_vapor_exp.csv` | Table I | p.14 (30) | 21 | Experimental, 3-phase region: water in vapor, 63.0-201.7 F, 113.8-613 psia |
| `kobayashi1951_3phase_xH2O_liqC3_exp.csv` | Table II | p.15 (31) | 18 | Experimental, 3-phase region: water in propane-rich liquid, 58.4-206.1 F, 107.1-638 psia (last 2 rows starred "Critical Region") |
| `kobayashi1951_3phase_xC3_water_exp.csv` | Table III | p.16 (32) | 16 | Experimental, 3-phase region: propane in water-rich liquid, 42.3-205.4 F, 82.2-632 psia (first row = quadruple point by extrapolation) |
| `kobayashi1951_3phase_smoothed.csv` | Table IV | p.17 (33) | 13 | SMOOTHED 3-phase locus with all three phase compositions, 42.3-205.7 F, 82.2-637 psia. This is the three-phase P-T locus file. |
| `kobayashi1951_2phase_yH2O_propane_rich_exp.csv` | Table V | pp.18-19 (34-35) | 53 | Experimental, 2-phase: water in propane-rich phase(s), isotherms 100/150/190/205.7/230/260/300 F, 102-2803 psia |
| `kobayashi1951_2phase_xC3_water_exp.csv` | Table VI | pp.20-21 (36-37) | 50 | Experimental, 2-phase: propane in water-rich liquid, isotherms 54/100/133/170/190/205.7/230/260/300 F, 72-2787 psia |
| `kobayashi1951_2phase_yH2O_propane_rich_smoothed.csv` | Table VII | p.22 (38, landscape) | 135 | GRAPHICALLY SMOOTHED: water in propane-rich phase(s); 15 P (100-3000 psia) x 9 T (100-310 F) grid |
| `kobayashi1951_2phase_xC3_water_smoothed.csv` | Table VIII | p.23 (39, landscape) | 150 | GRAPHICALLY SMOOTHED: propane in water-rich phase; 15 P (100-3000 psia) x 10 T (60-310 F) grid |
| `kobayashi1951_xC3_water_smoothed_comparison.csv` | Table XI | pp.65-66 (82-83, landscape) | 104 | Propane solubility in water: graphically smoothed (to 3000 psia) vs THERMODYNAMICALLY smoothed (extrapolated to 10000 psia); 13 P x 8 T (100-310 F) |

Raw (Tables I-III, V, VI) and smoothed (Tables IV, VII, VIII, XI) data are in
separate files as tabulated by the source. **The smoothed tables (VII, VIII,
and IV) are the cleaner fit targets**; the experimental tables carry the run
scatter. Table XI's thermodynamically smoothed values above 3000 psia are
model extrapolations by the author (modified Henry's law + partial molal
volume), not measurements — the dissertation text (diss. p.62) states
graphical smoothing is preferred except at high pressures.

Not extracted (out of scope): Tables IX/X (methane/ethane analogues of XI,
diss. pp.63-64), Table XII (modified Henry's-law constants and partial molal
volumes, diss. p.67 — contains a propane column: log K' at 122-310 F,
v2 ~ 1.22-1.32 cu ft/lb-mol), Table XIII (natural-gas A solubility, p.77).

## Suspected misprints in the source (values kept verbatim, flagged in `note` columns)

1. **Table III, 100.2 F / 191 psia**: printed `.00002045`. One zero too many —
   the duplicate run at 100.6 F/191 psia gives `.0002001` and smoothed
   Table IV gives `.000203` at 100 F. True value almost certainly `.0002045`.
2. **Table V, 100 F / 533 psia**: printed `.00492`. This point is on the
   liquid-propane branch, where neighbors read .000540-.000673 and smoothed
   Table VII gives .000604; almost certainly a dropped zero (`.000492`).
3. **Table VIII, 250 F / 100 psia**: printed `.000513`. The 100-psia isobar
   decreases monotonically across T (.0000562 at 220 F, .0000415 at 280 F);
   almost certainly a dropped zero (`.0000513`).

Filter rows with non-empty `note` fields before fitting, or apply the
suggested corrections explicitly.

## Other observations

- No entries were illegible; every cell in the nine extracted tables was
  read successfully from the 300-dpi renders. No MISSING placeholders were
  needed. (Empty `xC3_graph` cells in the Table XI CSV reflect the source:
  graphical smoothing stops at 3000 psia.)
- Table IV types the final digit of five entries as a subscript
  (.000105, .000187, .000344, .000615, .001095), presumably marking a
  less-certain digit; typed inline in the CSV.
- Table II shows scatter at 191.4 F (.00605) vs 191.6 F (.00544) — verified
  against the scan at high zoom; both as printed.
- Internal consistency checks passed:
  - Table VII at 205.7 F / 637 psia = .00998 = Table IV 3-phase critical
    composition (vapor = propane-rich liquid there).
  - Table XI graphical values reproduce Table VIII at all 72 overlapping
    grid points to within 1e-6 (the single 1e-6 difference, 130 F/600 psia
    .0002090 vs .000210, is the source's own rounding; both readings
    re-verified at high zoom).
- Physics sanity checks passed:
  - Smoothed 3-phase locus: 9.7 bar at 298 K, 13.2 bar at 311 K (slightly
    above pure-propane Psat, as expected).
  - Propane solubility in water ~ 0.8-7 x 10^-4 mole fraction; water in
    liquid propane ~ 1x10^-4 - 1x10^-2 (rising toward the 3-phase critical
    at 205.7 F).
- The `branch` column in the Table V CSV (vapor/liquid/dense/single-phase)
  is DERIVED here by comparison with the Table IV locus; it is not printed
  in the source.

## Pages a human may want to double-check

- The three flagged misprints above (PDF pp.32, 34, 39) — decide whether to
  drop or correct those rows.
- Table II critical-region rows (205.4/206.1 F, PDF p.31): compositions
  .01001/.00995 sit right at the 3-phase critical; fine as printed but
  near-critical data may deserve reduced fit weight.
- If higher-pressure propane solubility is wanted, Table XII (PDF p.84)
  provides the Henry's-law representation used for the >3000 psia
  extrapolations in Table XI.
