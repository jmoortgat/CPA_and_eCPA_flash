#!/usr/bin/env python3
"""Auto-generate the three per-author data-inventory + AAD tables:

  tables/exp_co2_brine.tex      Paper II Table S5  (\ce{CO2}-brine per author + AAD)
  tables/exp_ch4_h2o.tex        Paper II Table S6  (\ce{CH4}-\ce{H2O} binary per author + AAD_W, AAD_H)
  tables/exp_ch4_brine.tex      Paper II Table S7  (\ce{CH4}-brine per author + AAD)

Each output is a complete LaTeX table. The tables are written to
``paper_tables/tables/`` (or ``$ECPA_PAPER_DIR/tables/``). Tables S6 and S7
differ from the published values; see PUBLISHED_VS_RELEASED.md.

Compute strategy per table:

  - **Table 3 (CO2-brine)** reuses the already-cached
    ``results_cache/fig_co2_singlesalt_data.parquet`` which contains
    predictions for every experimental row (converged column filters
    failed flashes). Per-(salt, author) AAD is a groupby.

  - **Table 5 (CH4-brine)** reuses ``fig_ch4_singlesalt_data.parquet``
    for authors within the ``solve_elv`` domain (T <= 532 K). For
    authors that extend above 532 K -- currently only Susak (1980) at
    548, 573, 598 K -- we run a supplementary ``stability_flash`` pass
    directly here (with a z_CH4 retry ladder for the challenging
    near-critical points). Those extra predictions are merged into the
    per-author AAD before emission.

  - **Table 4 (CH4-H2O binary)** doesn't have a per-author cache in the
    figure pipeline, so this script runs ``stability_flash`` on every
    experimental row of ``data/binaries_ch4_water.parquet`` and caches
    the result to ``results_cache/binaries_ch4_water_per_author.parquet``.
    Groupby author -> AAD^W (aqueous CH4 mole fraction) and AAD^H
    (water in the CH4-rich gas phase) with N counts per column.

Usage:
    python3 scripts/build_paper_tables.py             # use per-author caches
    python3 scripts/build_paper_tables.py --force     # recompute the compute-heavy caches

Called automatically by ``scripts/make_all_paper_figures.py`` and the
top-level ``Makefile``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "scripts"))

from ms_ecpa import stability_flash                # noqa: E402
from data_loader import (                          # noqa: E402
    load_experimental, load_or_compute, add_cache_flag, RESULTS_CACHE,
    save_stats, DATA_DIR, _use_gasbrinebench,
)

# Output folder: $ECPA_PAPER_DIR, else <package>/paper_tables.
def _resolve_paper_dir():
    env = os.environ.get("ECPA_PAPER_DIR")
    return Path(env) if env else _CODE / "paper_tables"


_PAPER = _resolve_paper_dir()
_TABLES = _PAPER / "tables"

MW_H2O = 0.018015    # kg/mol
MW_CH4 = 0.016043
MW_CO2 = 0.044010

# Retry ladders for stability_flash (initial guess for gas mole fraction)
Z_CO2_LADDER = (0.05, 0.10, 0.02, 0.20, 0.30)
Z_CH4_LADDER = (0.02, 0.05, 0.10, 0.20, 0.40)


# ---------------------------------------------------------------------------
# Formatting helpers -- keep table cells consistent
# ---------------------------------------------------------------------------

# LaTeX-safe salt names (order matches the existing paper table).
_SALT_ORDER = ("NaCl", "KCl", "CaCl", "MgCl", "NaSO", "MgSO")
_SALT_TEX = {
    "NaCl": r"\ce{NaCl}",
    "KCl":  r"\ce{KCl}",
    "CaCl": r"\ce{CaCl2}",
    "MgCl": r"\ce{MgCl2}",
    "NaSO": r"\ce{Na2SO4}",
    "MgSO": r"\ce{MgSO4}",
}


def _range_int(vals) -> str:
    """`298--433` from a series of numbers, ints only."""
    v = pd.Series(vals).dropna()
    if v.empty:
        return "---"
    lo, hi = int(round(v.min())), int(round(v.max()))
    return f"{lo}" if lo == hi else f"{lo}--{hi}"


def _range_float(vals, spec: str = ".2g") -> str:
    """`0.5--4` or `1` from a series of floats."""
    v = pd.Series(vals).dropna()
    if v.empty:
        return "---"
    lo, hi = float(v.min()), float(v.max())
    fmt = lambda x: format(x, spec).rstrip("0").rstrip(".") if "." in format(x, spec) else format(x, spec)
    return fmt(lo) if lo == hi else f"{fmt(lo)}--{fmt(hi)}"


# Explicit raw-author-string -> bib key mapping. Necessary because:
#   (i)   The `AUTHOR(YEAR)` format in the parquet data files uses the raw
#         publication year while the bibliography sometimes uses the
#         adjusted/received year (BLANCO(1977) -> Blanco1978).
#   (ii)  Some entries in the bib use uppercase keys (STOESSELL1982,
#         CHAPOY2003) that a case-folding heuristic can't guess.
#   (iii) A handful of typos exist in the data files (Takenoushi/
#         Takenouchi, Botger/Bottger) that need explicit correction.
#   (iv)  MESSABEB(2017) is treated as a redundant tag for Messabeb2016
#         because no separate 2017 entry exists in the bib yet.
_AUTHOR_TO_BIB: dict[str, str] = {
    # Source labels that name one TABLE inside a paper. They are the same
    # reference as the base key; without these they emit \citet{Chabab2021_
    # JCED66_T2} and compile as [?]. Added when the CO2-brine tables moved
    # from the 511-point calibration set to the full 1,931-point pool, which
    # brought these sources in for the first time.
    "Chabab2019_IJGGC91_T2": "Chabab2019",
    "Chabab2021_JCED66_T2":  "Chabab2021",
    "Hou2013_JSCF78_T2":     "Hou2013",
    "Hou2013_JSCF78_T3":     "Hou2013b",
    # -------- CH4-brine data (data/ch4_brines.parquet) --------
    "BLANCO(1977)":     "Blanco1978",
    "KIEPE(2003)":      "Kiepe2003",
    "OSULLIVAN(1969)":  "OSullivan1970",
    "STOESSELL(1982)":  "STOESSELL1982",
    "SUSAK(1980)":      "Susak1980",
    # -------- CO2-brine data (data/co2_brines.parquet) --------
    "BASTAMI(2014)":    "Bastami2014",
    "CHABAB(2020)":     "Chabab2019",
    "CORTI(1990)":      "Corti1990",
    "GUO(2015)":        "Guo2016",
    "KAMP(2007)":       "Kamps2007",
    "KOSCHEL(2006)":    "Koschel2006",
    "MESSABEB(2016)":   "Messabeb2016",
    "MESSABEB(2017)":   "Messabeb2016",       # see note (iv)
    "PRUTTON(1945)":    "Prutton1945",
    "RUMPFb(1993)":     "Rumpf1993b",
    "SANTOS(2020)":     "Santos2020",
    "SANTOS(2021)":     "Santos2021",
    "SAVARY(2012)":     "Savary2012",
    "TAKENOUSHI(1965)": "Takenouchi1965",
    "TEYMOURI(2017)":   "Teymouri2017",
    "TONG(2013)":       "Tong2013",
    "YAN(2011)":        "Yan2011",
    "ZHAO(2015)":       "Zhao2015",
    "ZHAOb(2015)":      "Zhao2015b",
    # -------- CH4-H2O binary data (data/binaries_ch4_water.parquet) --------
    "ADDICKS(2002)":    "Addicks2002",
    "AMIRIJAFARI(1972)":"Amirijafari1972",
    "AWAN(2010)":       "Awan2010",
    "BOTGER(2016)":     "BOTTGER2016",
    "CAMPOS(2010)":     "Campos2010",
    "CARROLL(1998)":    "Carroll1998",
    "CHAPOY(2003)":     "CHAPOY2003",
    "CHAPOY(2004)":     "CHAPOY2004",
    "CHAPOY(2005)":     "Chapoy2005",
    "CULBERSON(1951)":  "Culberson1950",
    "DUFFY(1961)":      "DUFFY1961",
    "FROST(2013)":      "Frost2014",
    "GAO(1997)":        "Gao1997",
    "KIM(2003)":        "Kim2003",
    "LEKVAM(1997)":     "LEKVAM1997",
    "MICHELS(1936)":    "MICHELS1936",
    "OLDS(1942)":       "Olds1942",
    "OU(2015)":         "OU2015",
    "PRICE(1979)":      "Price1979",
    "QIN(2008)":        "Qin2008",
    "SACHS(1995)":      "SACHS1995",
    "SULTANOV(1972)":   "Sultanov1972",
    "WANG(1995)":       "WANG1995",
    "WANG(2003)":       "WANG2003",
    "YANG(2001)":       "YANG2001",
    "YARRISON(2006)":   "Yarrison2006",
    "YOKOYAMA(1988)":   "Yokoyama1988",
}


def _cite(author: str) -> str:
    """Return `\textcite{...}` for a raw author string via the
    explicit ``_AUTHOR_TO_BIB`` mapping. Falls back to a
    surname-capitalised guess (with a printed WARNING) for unknown
    authors -- that way the table still emits something and the user
    sees exactly which entries need to be added to the mapping."""
    a = author.strip()
    if a in _AUTHOR_TO_BIB:
        return rf"\textcite{{{_AUTHOR_TO_BIB[a]}}}"
    if "(" in a and a.endswith(")"):
        surname = a[: a.index("(")].strip()
        year = a[a.index("(")+1: -1].strip()
        guess = f"{surname.capitalize() if surname.isupper() else surname}{year}"
        print(f"  WARN: no bibkey mapping for '{a}' -- guessing '\\textcite{{{guess}}}'")
        return rf"\textcite{{{guess}}}"
    # No `AUTHOR(YEAR)` structure to guess from -- the GasBrineBench source
    # labels rows with the database's own source tag (e.g.
    # `Hou2013_JSCF78_T2`), which is not a bib key. Emitting it unchanged
    # compiles to a `[?]` citation, so warn as loudly as the guess branch
    # rather than letting it reach the .tex silently.
    print(f"  WARN: no bibkey mapping for '{a}' and no '(YEAR)' to guess "
          f"from -- emitting '\\textcite{{{a}}}' verbatim; add an "
          f"_AUTHOR_TO_BIB entry or the citation will compile as [?]")
    return rf"\textcite{{{a}}}"


def _exp_source_paths(kind: str) -> list[Path]:
    """Files ``load_experimental(kind)`` actually reads -- the
    GasBrineBench CSVs when that source is active, else the local
    parquet. Used only by :func:`_warn_if_stale`."""
    if _use_gasbrinebench(kind):
        try:
            import gasbrinebench_source as _gbb
            gbb_data = _gbb.gbb_root() / "data"
        except Exception:
            return []
        names = {"ch4_brines": ["solubility.csv"],
                 "binaries_ch4_water": ["solubility.csv", "y_h2o.csv"]}
        return [gbb_data / n for n in names.get(kind, [])]
    return [DATA_DIR / f"{kind}.parquet"]


def _warn_if_stale(cache: Path, inputs: list[Path]) -> None:
    """Warn when a derived cache is older than a file it was built from.

    ``make_all_paper_figures.py``'s freshness check compares these two
    per-author caches only against ``ms_ecpa/parameters.py``, so a cache
    that survived a direct ``fig_ch4_singlesalt.py --force`` or a
    refreshed experimental source -- neither of which touches
    parameters.py -- is otherwise reported fresh and reused silently,
    putting pre-change numbers in the table next to post-change figures.
    Warn rather than recompute: rebuilding here would move published
    numbers without the user asking for it."""
    try:
        cache_mtime = cache.stat().st_mtime
    except OSError:
        return
    newer = [p for p in inputs if p.exists() and p.stat().st_mtime > cache_mtime]
    if newer:
        print(f"  WARN: {cache.name} is older than "
              f"{', '.join(p.name for p in newer)} -- it was built from an "
              f"earlier version of that input. Re-run with --force to rebuild.")


def _fmt_aad(x, spec: str = ".1f") -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "---"
    return format(float(x), spec)


# ---------------------------------------------------------------------------
# Table 3 -- CO2-brine per author
# ---------------------------------------------------------------------------

def _co2_per_author_from_cache() -> pd.DataFrame:
    """Return exp-row-level DataFrame with predictions for the CO2-brine
    dataset. Reuses ``fig_co2_singlesalt_data.parquet``."""
    cache = RESULTS_CACHE / "fig_co2_singlesalt_data.parquet"
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache.name} is missing. Run `python3 scripts/fig_co2_singlesalt.py --force` first."
        )
    df = pd.read_parquet(cache)
    exp = df[df["kind"] == "exp"].copy()
    exp = exp[exp["converged"] & (exp["xc_W"] > 0) &
              exp["xc_W_pred"].notna()]
    # We compare in mole-fraction space; the paper's AAD definition uses
    # molality ratio, but for x = m / (n_H2O + m) with n_H2O constant,
    # |x_pred/x_exp - 1| == |m_pred/m_exp - 1| to first order. Use the
    # mole-fraction directly, which is what the fig cache stores.
    # The caption defines AAD on MOLALITY,
    #     AAD = N_p^-1 sum |m_calc - m_exp| / m_exp,
    # and the main text quotes the molality figure. This used the mole
    # fraction instead, on the argument that the two agree "to first order";
    # they agree to about a part in a thousand, which is enough to print
    # 11.0 where the text says 11.1. Compute what the caption says.
    exp["rel_err"] = (exp["mc_pred"] - exp["mc_exp"]).abs() / exp["mc_exp"]
    return exp


def _per_salt_rows(exp: pd.DataFrame, rel_col: str = "rel_err") -> list[str]:
    """Generate the shared table body rows for the CO2- and CH4-brine
    per-author tables. Grouped by salt, with per-salt Global rows and
    `\\midrule` separators between salt groups."""
    lines: list[str] = []
    for salt in _SALT_ORDER:
        salt_sub = exp[exp["salt"] == salt]
        if salt_sub.empty:
            continue
        lines.append(rf"\multicolumn{{6}}{{c}}{{\textit{{{_SALT_TEX[salt]}}}}} \\")
        for author, g in sorted(salt_sub.groupby("author"),
                                key=lambda kv: (kv[1]["T_K"].min(), kv[0])):
            aad = 100 * float(g[rel_col].mean())
            lines.append(
                f"{_cite(author):<28s} & "
                f"{_range_int(g['T_K']):>10s} & "
                f"{_range_int(g['P_bar']):>10s} & "
                f"{_range_float(g['ms_mol_kg']):>10s} & "
                f"{int(len(g)):>4d} & "
                f"{_fmt_aad(aad):>5s} \\\\"
            )
        aad_global = 100 * float(salt_sub[rel_col].mean())
        lines.append(
            rf"\textbf{{Global}} & "
            rf"\textbf{{{_range_int(salt_sub['T_K'])}}} & "
            rf"\textbf{{{_range_int(salt_sub['P_bar'])}}} & "
            rf"\textbf{{{_range_float(salt_sub['ms_mol_kg'])}}} & "
            rf"\textbf{{{int(len(salt_sub))}}} & "
            rf"\textbf{{{_fmt_aad(aad_global)}}} \\"
        )
        lines.append(r"\midrule")
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    return lines


# LaTeX longtable prologue used by all three per-author tables in
# the Paper II SI. Includes caption, label, \endfirsthead / \endhead /
# \endfoot / \endlastfoot boilerplate so the table paginates cleanly.
def _longtable_header(caption_body: str, label: str, colspec: str,
                      header_row: str, header_ncol: int) -> list[str]:
    return [
        rf"\begin{{longtable}}{{{colspec}}}",
        rf"\caption{{{caption_body}}}",
        rf"\label{{{label}}} \\",
        r"\toprule",
        f"{header_row} \\\\",
        r"\midrule",
        r"\endfirsthead",
        rf"\multicolumn{{{header_ncol}}}{{l}}{{\small\textit{{Table~\thetable\ "
        r"continued from previous page.}}\\",
        r"\toprule",
        f"{header_row} \\\\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        rf"\multicolumn{{{header_ncol}}}{{r}}{{\small\textit{{(continued on next page)}}}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]


def emit_table_co2(exp: pd.DataFrame) -> Path:
    """Full \begin{longtable}...\end{longtable} for Table S? / tab:exp_CO2."""
    out = _TABLES / "exp_co2_brine.tex"
    _TABLES.mkdir(parents=True, exist_ok=True)
    caption = (
        r"Experimental \ce{CO2}--brine VLE compared with the model, "
        r"with the AAD of the predicted "
        r"\ce{CO2} solubility in brine (AAD $= N_p^{-1}\sum "
        r"|m_c^\text{calc}-m_c^\text{exp}|/m_c^\text{exp}$). Data are "
        r"grouped by salt; the per-salt ``Global'' row summarises the "
        r"cumulative $N_p$ and the point-weighted AAD for that salt. "
        r"The gas--ion energies were fitted to the $511$ points at "
        r"$323$, $373$ and $423$\,K; the remaining points are out of "
        r"sample."
    )
    header_row = r"Reference & $T$ (K) & $P$ (bar) & $m_s$ (m) & $N_p$ & AAD (\%)"
    lines = [
        "% AUTO-GENERATED by scripts/build_paper_tables.py.",
        "% Table tab:exp_CO2 (Paper II SI). "
        "Refit -> `make paper` -> updates automatically.",
    ]
    lines += _longtable_header(caption, "tab:exp_CO2", "lccccc", header_row, 6)
    lines += _per_salt_rows(exp)
    lines.append(r"\end{longtable}")
    out.write_text("\n".join(lines) + "\n")
    return out


# ---------------------------------------------------------------------------
# Table 5 -- CH4-brine per author (with Susak high-T augmentation)
# ---------------------------------------------------------------------------

def _ch4_per_author_from_cache() -> pd.DataFrame:
    """Read fig_ch4_singlesalt_data.parquet and join predictions onto
    experimental rows via nearest-P interpolation at each (salt, T, ms)
    combination -- which matches how fig_ch4_singlesalt's own AARE
    table is computed."""
    cache = RESULTS_CACHE / "fig_ch4_singlesalt_data.parquet"
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache.name} is missing. Run `python3 scripts/fig_ch4_singlesalt.py --force` first."
        )
    pred = pd.read_parquet(cache)
    exp = load_experimental("ch4_brines")
    # For each experimental row, look up the model prediction at that P.
    # Rows that find no prediction are dropped, which silently shrinks the
    # N_p column and narrows the reported P range of Table 5; count them by
    # cause so the console says how much of the measured set the table
    # actually covers.
    n_no_group, n_out_of_span, n_bad_value = 0, 0, 0
    out_rows: list[dict] = []
    for (salt, T, ms), g_exp in exp.groupby(["salt", "T_K", "ms_mol_kg"]):
        g_pred = pred[(pred["salt"] == salt) & (pred["T_K"] == T) &
                      np.isclose(pred["ms_mol_kg"], ms) & pred["converged"]]
        if g_pred.empty:
            n_no_group += len(g_exp)
            continue
        g_pred = g_pred.sort_values("P_bar")
        P_ref, y_ref = g_pred["P_bar"].to_numpy(), g_pred["mh_W_pred"].to_numpy()
        for _, r in g_exp.iterrows():
            # Log-log interp -- matches print_aare_table in fig_ch4_singlesalt.
            P_e = float(r["P_bar"])
            if P_e < P_ref.min() or P_e > P_ref.max():
                n_out_of_span += 1
                continue
            y_p = float(np.exp(np.interp(np.log(P_e), np.log(P_ref), np.log(y_ref))))
            if not (np.isfinite(y_p) and r["mh_W"] > 0):
                n_bad_value += 1
                continue
            out_rows.append({
                "salt": salt, "T_K": int(T), "P_bar": P_e,
                "ms_mol_kg": float(ms), "author": r["author"],
                "mh_W": float(r["mh_W"]), "mh_W_pred": y_p,
                "rel_err": abs(y_p - r["mh_W"]) / r["mh_W"],
            })
    n_dropped = n_no_group + n_out_of_span + n_bad_value
    if n_dropped:
        print(f"  NOTE: {n_dropped}/{len(exp)} experimental CH4-brine rows have "
              f"no prediction in {cache.name} and are absent from Table 5's "
              f"N_p and P range -- {n_no_group} in (salt, T, ms) groups the "
              f"figure cache does not cover or where no grid point converged, "
              f"{n_out_of_span} outside the converged prediction's P span, "
              f"{n_bad_value} non-finite or non-positive.")
    return pd.DataFrame(out_rows)


def _susak_high_T_augment() -> pd.DataFrame:
    """Directly run stability_flash on Susak(1980) NaCl points that fall
    outside fig_ch4_singlesalt's T_MAX = 532 K window. Uses a z_CH4
    retry ladder because the near-critical high-T points are sensitive
    to the initial-guess mole fraction of methane."""
    exp = load_experimental("ch4_brines")
    hi_T = exp[(exp["salt"] == "NaCl") &
               (exp["author"] == "SUSAK(1980)") &
               (exp["T_K"] > 532)].copy()
    if hi_T.empty:
        return pd.DataFrame(columns=[
            "salt","T_K","P_bar","ms_mol_kg","author","mh_W","mh_W_pred","rel_err"
        ])
    print(f"  augmenting: {len(hi_T)} SUSAK(1980) NaCl points at T > 532 K "
          f"(stability_flash with z_CH4 retry ladder)...")
    ion_cols = ["m_Na", "m_Cl", "m_K", "m_Ca", "m_Mg", "m_SO4"]
    out_rows: list[dict] = []
    t0 = time.perf_counter()
    for _, r in hi_T.iterrows():
        mi = np.array([r[c] for c in ion_cols], dtype=float)
        mh_pred = np.nan
        for yh in Z_CH4_LADDER:
            z = np.array([1.0 - yh, yh, 0.0])
            try:
                res = stability_flash(float(r["T_K"]), float(r["P_bar"]),
                                      z, mi)
                if res.phases == 2 and res.flash_result is not None:
                    # Same two rejections as _compute_ch4_h2o_per_author
                    # below: stability_flash's terminal hard-failure branch
                    # returns phases=2 carrying the last (non-converged) SSI
                    # iterate, whose x_aq is finite and would read as a
                    # plausible molality; and a feed-copy iterate
                    # (x_CH4/(x_H2O+x_CH4) == the assumed z_CH4) is the
                    # degenerate solution, not a solved one. Measured: 0 of
                    # the 64 Susak T > 532 K rows are rejected by either
                    # test, so the table does not move -- this keeps it
                    # that way under a refit.
                    if not bool(res.flash_result.converged):
                        continue
                    xH, xC, _ = res.flash_result.x_aq
                    if xH + xC > 0 and abs(
                            xC / (xH + xC) - yh) < 1e-3 * max(yh, 1e-6):
                        continue
                    if xH > 0 and xC > 0:
                        # CH4 molality (mol per kg H2O): mh = xC / xH * (1/MW_H2O)
                        mh_pred = float(xC / xH / MW_H2O)
                        break
            except Exception:
                pass
        if not (np.isfinite(mh_pred) and r["mh_W"] > 0):
            continue
        out_rows.append({
            "salt": "NaCl", "T_K": int(r["T_K"]),
            "P_bar": float(r["P_bar"]), "ms_mol_kg": float(r["ms_mol_kg"]),
            "author": r["author"], "mh_W": float(r["mh_W"]),
            "mh_W_pred": mh_pred,
            "rel_err": abs(mh_pred - r["mh_W"]) / r["mh_W"],
        })
    print(f"  augmenting: {len(out_rows)}/{len(hi_T)} accepted "
          f"(converged, non-degenerate flash with finite molality) "
          f"({time.perf_counter()-t0:.1f} s)")
    return pd.DataFrame(out_rows)


def _build_ch4_brine_dataset(force: bool = False) -> pd.DataFrame:
    """Combined per-author dataset for Table 5: fig_ch4_singlesalt cache
    + Susak high-T augmentation. Cached in results_cache so we don't
    re-run stability_flash every `make stats`."""
    cache = RESULTS_CACHE / "ch4_brine_per_author.parquet"
    if cache.exists() and not force:
        _warn_if_stale(cache, [RESULTS_CACHE / "fig_ch4_singlesalt_data.parquet"]
                       + _exp_source_paths("ch4_brines"))
        print(f"  [cache] {cache.name}")
        return pd.read_parquet(cache)
    base = _ch4_per_author_from_cache()
    aug = _susak_high_T_augment()
    combined = pd.concat([base, aug], ignore_index=True)
    combined.to_parquet(cache)
    print(f"  [saved] {cache.name}  ({len(combined)} rows)")
    return combined


def emit_table_ch4_brine(df: pd.DataFrame) -> Path:
    """Full \begin{longtable}...\end{longtable} for tab:exp_CH4_brine."""
    out = _TABLES / "exp_ch4_brine.tex"
    _TABLES.mkdir(parents=True, exist_ok=True)
    caption = (
        r"Experimental \ce{CH4}--brine VLE used in the eCPA "
        r"parameterization, with the resulting AAD of the predicted "
        r"\ce{CH4} solubility in brine "
        r"(AAD $= N_p^{-1}\sum |m_h^\text{calc}-m_h^\text{exp}|/m_h^\text{exp}$). "
        r"Data are grouped by salt; the per-salt ``Global'' row summarises "
        r"the cumulative $N_p$ and the AAD averaged over all sources for that "
        r"salt. \textcite{Susak1980} coverage extends up to $T = 598$\,K via "
        r"the ``stability\_flash'' compute path in "
        r"\texttt{scripts/build\_paper\_tables.py} (bypassing the "
        r"\texttt{solve\_elv} T = 533\,K guess-table validity window)."
    )
    header_row = r"Reference & $T$ (K) & $P$ (bar) & $m_s$ (m) & $N_p$ & AAD (\%)"
    lines = [
        "% AUTO-GENERATED by scripts/build_paper_tables.py.",
        "% Table tab:exp_CH4_brine (Paper II SI). "
        "Refit -> `make paper` -> updates automatically.",
    ]
    lines += _longtable_header(caption, "tab:exp_CH4_brine", "lccccc",
                               header_row, 6)
    lines += _per_salt_rows(df)
    lines.append(r"\end{longtable}")
    out.write_text("\n".join(lines) + "\n")
    return out


# ---------------------------------------------------------------------------
# Table 4 -- CH4-H2O binary per author (AAD_W + AAD_H)
# ---------------------------------------------------------------------------

def _compute_ch4_h2o_per_author() -> pd.DataFrame:
    """Run stability_flash on every CH4-H2O experimental row and store
    predictions for both phases (x_CH4_aq and y_H2O_gas). Salt-free
    (mi = 0). Cached in results_cache."""
    exp = load_experimental("binaries_ch4_water")
    print(f"  flashing {len(exp)} CH4-H2O experimental points...")
    mi = np.zeros(6)
    out_rows: list[dict] = []
    t0 = time.perf_counter()
    for i, r in exp.iterrows():
        # Retry ladder over z_CH4 initial guess
        x_pred, y_pred = np.nan, np.nan
        for yh in Z_CH4_LADDER:
            z = np.array([1.0 - yh, yh, 0.0])
            try:
                res = stability_flash(float(r["T_K"]), float(r["P_bar"]),
                                      z, mi)
                if res.phases == 2 and res.flash_result is not None:
                    # Measured: 0 of 1009 accepted rows are non-converged or
                    # degenerate, so Table 4 does not move -- this keeps it
                    # that way.
                    if not bool(res.flash_result.converged):
                        continue
                    xH_aq, xC_aq, _ = res.flash_result.x_aq
                    if xH_aq + xC_aq > 0 and abs(
                            xC_aq / (xH_aq + xC_aq) - yh) < 1e-3 * max(yh, 1e-6):
                        continue
                    yH, _, _ = res.flash_result.y_gas
                    x_pred = float(xC_aq)
                    y_pred = float(yH)
                    break
            except Exception:
                pass
        out_rows.append({
            # Whether any rung of the ladder produced a prediction.
            # _aad_col drops NaN predictions silently, so without this
            # column a row the whole ladder failed on is indistinguishable
            # in the cache from a row never measured for that phase.
            "converged": bool(np.isfinite(x_pred) or np.isfinite(y_pred)),
            "T_K": int(r["T_K"]),
            "P_bar": float(r["P_bar"]),
            "author": r["author"],
            "x_CH4_aq_exp": float(r["x_CH4_aq_exp"])
                if not pd.isna(r["x_CH4_aq_exp"]) else np.nan,
            "y_H2O_gas_exp": float(r["y_H2O_gas_exp"])
                if not pd.isna(r["y_H2O_gas_exp"]) else np.nan,
            "x_CH4_aq_pred": x_pred,
            "y_H2O_gas_pred": y_pred,
        })
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(exp)}  ({time.perf_counter()-t0:.0f} s)")
    out = pd.DataFrame(out_rows)
    n_ok = int(out["converged"].sum()) if not out.empty else 0
    print(f"  finished CH4-H2O compute in {time.perf_counter()-t0:.1f} s "
          f"({n_ok}/{len(out)} flashed; {len(out) - n_ok} produced no "
          f"prediction and are absent from Table 4's N_p^W / N_p^H)")
    return out


def _load_or_compute_ch4_h2o(force: bool = False) -> pd.DataFrame:
    cache = RESULTS_CACHE / "binaries_ch4_water_per_author.parquet"
    if cache.exists() and not force:
        _warn_if_stale(cache, _exp_source_paths("binaries_ch4_water"))
        print(f"  [cache] {cache.name}")
        return pd.read_parquet(cache)
    df = _compute_ch4_h2o_per_author()
    df.to_parquet(cache)
    print(f"  [saved] {cache.name}  ({len(df)} rows)")
    return df


def emit_table_ch4_h2o(df: pd.DataFrame) -> Path:
    """Full \begin{longtable}...\end{longtable} for tab:exp_CH4_H2O."""
    out = _TABLES / "exp_ch4_h2o.tex"
    _TABLES.mkdir(parents=True, exist_ok=True)
    caption = (
        r"Experimental \ce{CH4}--\ce{H2O} VLE used in the CPA "
        r"parameterization, with the resulting AAD of the predicted mutual "
        r"solubilities of the aqueous ($W$) and \ce{CH4}-rich ($H$) phases. "
        r"$N_p^W$ and $N_p^H$ denote the number of points used for each "
        r"phase."
    )
    header_row = (r"Reference & $T$ (K) & $P$ (bar) & $N_p^W$ "
                  r"& AAD$^W$ (\%) & $N_p^H$ & AAD$^H$ (\%)")
    lines = [
        "% AUTO-GENERATED by scripts/build_paper_tables.py.",
        "% Table tab:exp_CH4_H2O (Paper II SI). "
        "Refit -> `make paper` -> updates automatically.",
    ]
    lines += _longtable_header(caption, "tab:exp_CH4_H2O", "lcccccc",
                               header_row, 7)

    def _aad_col(g: pd.DataFrame, exp_col: str, pred_col: str) -> tuple[int, float]:
        mask = (g[exp_col].notna() & g[pred_col].notna() &
                (g[exp_col] > 0) & np.isfinite(g[pred_col]))
        n = int(mask.sum())
        if n == 0:
            return 0, float("nan")
        rel = (g.loc[mask, pred_col] - g.loc[mask, exp_col]).abs() / \
               g.loc[mask, exp_col]
        return n, 100 * float(rel.mean())

    for author, g in sorted(df.groupby("author"),
                            key=lambda kv: (kv[1]["T_K"].min(), kv[0])):
        n_W, aad_W = _aad_col(g, "x_CH4_aq_exp", "x_CH4_aq_pred")
        n_H, aad_H = _aad_col(g, "y_H2O_gas_exp", "y_H2O_gas_pred")
        lines.append(
            f"{_cite(author):<28s} & "
            f"{_range_int(g['T_K']):>10s} & "
            f"{_range_int(g['P_bar']):>10s} & "
            f"{n_W:>3d} & {_fmt_aad(aad_W):>5s} & "
            f"{n_H:>3d} & {_fmt_aad(aad_H):>5s} \\\\"
        )
    lines.append(r"\midrule")
    n_W_tot, aad_W_tot = _aad_col(df, "x_CH4_aq_exp", "x_CH4_aq_pred")
    n_H_tot, aad_H_tot = _aad_col(df, "y_H2O_gas_exp", "y_H2O_gas_pred")
    lines.append(
        rf"\textbf{{Global}} & "
        rf"\textbf{{{_range_int(df['T_K'])}}} & "
        rf"\textbf{{{_range_int(df['P_bar'])}}} & "
        rf"\textbf{{{n_W_tot}}} & \textbf{{{_fmt_aad(aad_W_tot)}}} & "
        rf"\textbf{{{n_H_tot}}} & \textbf{{{_fmt_aad(aad_H_tot)}}} \\"
    )
    lines.append(r"\end{longtable}")
    out.write_text("\n".join(lines) + "\n")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cache_flag(p)
    args = p.parse_args()

    print("Building Table 3 (CO2-brine per author)...")
    co2 = _co2_per_author_from_cache()
    out = emit_table_co2(co2)
    print(f"  wrote {out}  "
          f"({int(co2['salt'].nunique())} salts, {int(len(co2))} rows aggregated)")

    print("Building Table 5 (CH4-brine per author, Susak high-T augmented)...")
    ch4b = _build_ch4_brine_dataset(force=args.force)
    out = emit_table_ch4_brine(ch4b)
    print(f"  wrote {out}  "
          f"({int(ch4b['salt'].nunique())} salts, {int(len(ch4b))} rows aggregated)")
    # Emit specific numbers cited in the paper text as sidecar stats so
    # build_paper_stats_tex.py can turn them into \newcommand macros
    # (e.g. \aareCHfourSusakNaClHiT for §3.5).
    susak_nacl = ch4b[(ch4b["salt"] == "NaCl") &
                      (ch4b["author"] == "SUSAK(1980)")]
    ch4b_stats = {
        "susak_nacl_full_T_aad": round(100 * float(susak_nacl["rel_err"].mean()), 2)
            if not susak_nacl.empty else None,
        "susak_nacl_full_T_N": int(len(susak_nacl)),
        "susak_nacl_full_T_Tmin": int(susak_nacl["T_K"].min()) if not susak_nacl.empty else None,
        "susak_nacl_full_T_Tmax": int(susak_nacl["T_K"].max()) if not susak_nacl.empty else None,
    }
    save_stats("build_paper_tables", ch4b_stats)

    print("Building Table 4 (CH4-H2O binary per author)...")
    ch4h = _load_or_compute_ch4_h2o(force=args.force)
    out = emit_table_ch4_h2o(ch4h)
    print(f"  wrote {out}  "
          f"({int(ch4h['author'].nunique())} authors)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
