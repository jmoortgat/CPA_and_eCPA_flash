"""Extract F. M. Coelho's on-disk H2 experimental data files to tidy CSVs.

Sources (all machine-readable, '#AUTHOR' header + whitespace-separated):
  $ECPA_SOURCE_MATERIALS/EoS/H2/H2-Water/<T>K/EXP<k>_T<T>K.txt   (P, xh_W, yw_H)
  $ECPA_SOURCE_MATERIALS/EoS/H2/H2-Brine/<T>K/EXP<k>_T<T>K.txt   (P, ms, mH2)
  $ECPA_SOURCE_MATERIALS/EoS/H2/H2-CO2/ELV_H2-CO2/<T>K/EXP1_...  (P, xh_c, yh_H)

'X'/'x' placeholders -> empty cells. No digitization, no value edits.
Run from anywhere: the source tree is $ECPA_SOURCE_MATERIALS (default ./source_materials); outputs are written beside this script.
"""
import os
import csv
import re
from pathlib import Path

# Source files (not distributed); the extracted CSVs beside this script are.
ROOT = Path(os.environ.get("ECPA_SOURCE_MATERIALS", "source_materials")) / "EoS/H2"
OUT = Path(__file__).resolve().parent


def parse_exp_file(path):
    """Return (author, rows) where rows are lists of raw column strings."""
    lines = path.read_text().strip().splitlines()
    author = lines[0].lstrip("#").split()[0]
    rows = []
    for ln in lines[2:]:                    # line 2 is the column header
        parts = ln.split()
        if parts:
            rows.append(parts)
    return author, rows


def val(tok):
    return "" if tok.upper() == "X" else tok


def tdirs(base):
    out = []
    for d in sorted(base.iterdir()):
        m = re.fullmatch(r"(\d+(?:\.\d+)?)K", d.name)
        if m and d.is_dir():
            out.append((float(m.group(1)), d))
    return sorted(out)


def extract(base, colnames, outname, extra_T=True):
    n = 0
    with open(OUT / outname, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["T_K"] + colnames + ["author", "source_file"])
        for T, d in tdirs(base):
            for f in sorted(d.glob("EXP*.txt")):
                author, rows = parse_exp_file(f)
                rel = f.relative_to(ROOT.parent.parent)
                for r in rows:
                    w.writerow([T] + [val(x) for x in r[:len(colnames)]]
                               + [author, str(rel)])
                    n += 1
    print(f"{outname}: {n} rows")


extract(ROOT / "H2-Water", ["P_bar", "x_H2_aq", "y_H2O_gas"],
        "h2_water_binaries.csv")


def extract_water_isobaric():
    """H2-Water/P/: isobaric files (T, xh_W, yw_H at fixed P) plus one
    free-form (T, P, xh_W, yw_H) file. '(copy)' duplicates skipped."""
    base = ROOT / "H2-Water" / "P"
    n = 0
    with open(OUT / "h2_water_isobaric.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["T_K", "P_bar", "x_H2_aq", "y_H2O_gas",
                    "author", "source_file"])
        for f in sorted(base.rglob("EXP*.txt")):
            if "(copy)" in f.name:
                continue
            author, rows = parse_exp_file(f)
            rel = f.relative_to(ROOT.parent.parent)
            m = re.search(r"P(\d+(?:\.\d+)?)bar", f.name)
            for r in rows:
                if m:                      # (T, x, y) at fixed P from name
                    w.writerow([r[0], m.group(1), val(r[1]), val(r[2]),
                                author, str(rel)])
                else:                      # (T, P, x, y)
                    w.writerow([r[0], r[1], val(r[2]), val(r[3]),
                                author, str(rel)])
                n += 1
    print(f"h2_water_isobaric.csv: {n} rows")


extract_water_isobaric()
extract(ROOT / "H2-Brine", ["P_bar", "m_NaCl_molkg", "m_H2_molkg"],
        "h2_nacl_brine.csv")
extract(ROOT / "H2-CO2" / "ELV_H2-CO2", ["P_bar", "x_H2_liqCO2", "y_H2_gas"],
        "h2_co2_elv.csv")
