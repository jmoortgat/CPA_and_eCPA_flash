#!/usr/bin/env python3
"""CO2-H2O and CH4-H2O binary figures from the Phase-1 cache pipeline.

Two figures, both two-panel:

  - ``binaries_main.png``: (a) CO2-H2O, (b) CH4-H2O. Each panel shows
    the aqueous gas mole fraction in per-mil (x * 1000, axis labelled in ‰) vs.
    pressure at two representative isotherms (298 and 423 K). Solid
    eCPA-binary model curves overlay scatter from multiple authors.

  - ``binaries_SI.png``: 2 x 2. Top row = aqueous gas solubility,
    bottom row = water in the gas phase, at four isotherms (298, 373,
    423, 473 K).

Both figures share one cache:
``results_cache/fig_binaries_data.parquet``. Plotting is fast (<1 s);
the cold compute (model + per-author error stats) runs salt-free
``solve_elv`` once per (T, P) on a smooth pressure grid for each
isotherm in ``ISOTHERMS_SI`` (~250 model points total).

Usage:
    python3 scripts/fig_binaries.py            # use cache
    python3 scripts/fig_binaries.py --force    # recompute
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "scripts"))

from ms_ecpa import solve_elv                          # noqa: E402
from paper_style import (                              # noqa: E402
    apply_style, bold_ticks, panel_label,
    PAPER_FIGURES, WONG, LINESTYLES,
)
from data_loader import (                              # noqa: E402
    load_experimental, load_or_compute, add_cache_flag,
)

# Isotherms used for the SI grid (one model curve per T per binary).
# Constrained to the solve_elv validity window 283 < T < 533 K.
ISOTHERMS_SI = (298, 373, 423, 473)
# Main figure uses a subset (one low-T, one high-T per binary).
ISOTHERMS_MAIN = (298, 423)

# Pressure grid per isotherm. The eCPA binary solver is fast at salt = 0,
# so we use a dense log grid here; cold compute is still well under a minute.
P_GRID = np.geomspace(2.0, 1200.0, 40)

# Tolerance for matching an experimental T to a model isotherm.
T_MATCH_TOL = 5.0  # K
# Salt-free ion vector.
MI_ZERO = np.zeros(6)


# ---------------------------------------------------------------------------
# Compute (slow path).
# ---------------------------------------------------------------------------
def _solve_one(T_K: float, P_bar: float, YC: int) -> dict:
    """Return one row of model predictions for a single (T, P, gas) point.

    YC=1 -> CO2-H2O (read x3w = CO2 in aq).
    YC=0 -> CH4-H2O (read x2w = CH4 in aq).
    Returns NaN-filled row on solver failure.
    """
    try:
        res = solve_elv(float(T_K), float(P_bar), YC=YC, mi=MI_ZERO)
        if not res.converged:
            raise RuntimeError(f"ier={res.ier}")
        x_aq = float(res.x3w if YC == 1 else res.x2w)
        # H2O mole fraction in the gas (CH4-rich or CO2-rich) phase = x1h.
        y_h2o = float(res.x1h)
        return {"T_K": float(T_K), "P_bar": float(P_bar), "YC": int(YC),
                "x_gas_aq": x_aq, "y_H2O_gas": y_h2o, "ok": True}
    except Exception:
        return {"T_K": float(T_K), "P_bar": float(P_bar), "YC": int(YC),
                "x_gas_aq": float("nan"), "y_H2O_gas": float("nan"),
                "ok": False}


def compute_binary_predictions() -> pd.DataFrame:
    """Run salt-free solve_elv on a smooth (T, P) grid for each binary."""
    rows = []
    print(f"  computing binary predictions on "
          f"{len(ISOTHERMS_SI)} isotherms x {len(P_GRID)} P points x 2 binaries ...")
    t0 = time.perf_counter()
    nfail = 0
    # Suppress harmless overflow warnings deep in the EoS solver tail.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for T in ISOTHERMS_SI:
            for P in P_GRID:
                for YC in (1, 0):    # 1 = CO2, 0 = CH4
                    r = _solve_one(T, P, YC)
                    rows.append(r)
                    if not r["ok"]:
                        nfail += 1
    print(f"  ...{time.perf_counter()-t0:.1f}s; "
          f"{len(rows)-nfail}/{len(rows)} converged "
          f"({nfail} failures kept as NaN for plotting).")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot helpers.
# ---------------------------------------------------------------------------
def _exp_at_T(exp: pd.DataFrame, T_target: float) -> pd.DataFrame:
    """Return experimental rows within +-T_MATCH_TOL of T_target."""
    return exp[(exp["T_K"] - T_target).abs() <= T_MATCH_TOL]


def _scatter_authors(ax, sub: pd.DataFrame, x_col: str, y_col: str,
                     color, *, scale: float = 1.0,
                     show_labels: bool = False) -> None:
    """Overlay one marker style per author at this (T, panel)."""
    yvals = sub[y_col]
    mask = yvals.notna()
    if not mask.any():
        return
    sub = sub[mask]
    # Sort authors by point count so common ones get distinct markers.
    counts = sub["author"].value_counts()
    markers = ["o", "s", "^", "v", "D", "P", "X", ">", "<", "p", "*", "h", "H"]
    for k, author in enumerate(counts.index):
        rows = sub[sub["author"] == author]
        m = markers[k % len(markers)]
        label = author if show_labels else None
        ax.scatter(rows[x_col], rows[y_col] * scale,
                   marker=m, s=42, facecolors="none",
                   edgecolors=color, linewidths=1.4, alpha=0.9,
                   label=label)


def _curve_at_T(model: pd.DataFrame, T_target: int, y_col: str,
                YC: int) -> tuple[np.ndarray, np.ndarray]:
    sub = model[(model["T_K"] == T_target) & (model["YC"] == YC) & model["ok"]]
    sub = sub.sort_values("P_bar")
    return sub["P_bar"].to_numpy(), sub[y_col].to_numpy()


# ---------------------------------------------------------------------------
# Main paper figure.
#
# Recreates the original Mourão (2025) layout for the salt-free CH4-H2O
# binary in the new style (rcParams, WONG palette, line styles, per-mil
# convention):
#   (a) x_CH4^aq vs P at 298 K, log-log scale, across the dense literature
#       compilation (~20 authors). Single CPA model curve.
#   (b) x_CH4^aq vs P on linear axes at four isotherms (298, 373, 423,
#       473 K), per-mil y-axis. One CPA curve per T.
# CO2-H2O is handled separately in the prose and the companion paper
# (Coelho 2025 / Moortgat 2025); the SI figure still shows both binaries
# (top row aqueous, bottom row water-in-gas) at four isotherms.
# ---------------------------------------------------------------------------
ISOTHERMS_MAIN_PANEL_B = (298, 373, 423, 473)
PANEL_A_MARKERS = (
    "o", "s", "^", "v", "D", "P", "X", ">", "<", "p", "*",
    "h", "H", "8", "d", "x", "+", "1", "2", "3", "4",
)


def _use_permille_font() -> None:
    """Switch to a font that renders ‰ cleanly.

    Both figures in this module carry per-mil y-axes, and the paper's
    default DejaVu Sans draws per-mille as a stacked "%o" that reads
    poorly at print size. The override is process-wide and deliberately
    left in place rather than restored after each figure: every figure
    this script produces wants it, and ``make_all_paper_figures.py`` runs
    each figure script in its own subprocess, so it cannot reach the
    other paper figures.
    """
    plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]


def make_main_figure(exp_co2: pd.DataFrame, exp_ch4: pd.DataFrame,
                     model: pd.DataFrame, out_path: Path) -> None:
    _use_permille_font()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8),
                             gridspec_kw=dict(wspace=0.30))

    # ---- Panel (a) -- CH4 at 298 K, log-log, per-mil ----
    # Panels (a) and (b) now use the same y-unit (aqueous CH4 mole
    # fraction in per-mil, i.e. mole fraction * 1000).  The two panels
    # keep their independent axes because (a) is log-log and covers 3
    # decades of P at a single T while (b) is linear-linear and covers
    # four isotherms at moderate P — sharing a y-axis would waste ~80%
    # of panel (b).
    ax = axes[0]
    Ps, xs = _curve_at_T(model, 298, "x_gas_aq", YC=0)
    ax.plot(Ps, xs * 1000.0, color="k", linestyle="-", linewidth=2.4,
            label="CPA, 298 K", zorder=4)

    sub_298 = _exp_at_T(exp_ch4, 298)
    counts_298 = sub_298["author"].value_counts()
    for k, author in enumerate(counts_298.index):
        rows = sub_298[sub_298["author"] == author]
        m_idx = k % len(PANEL_A_MARKERS)
        c_idx = k % len(WONG)
        ax.scatter(rows["P_bar"], rows["x_CH4_aq_exp"] * 1000.0,
                   marker=PANEL_A_MARKERS[m_idx], s=44,
                   color=WONG[c_idx], edgecolors="k", linewidths=0.4,
                   alpha=0.85, label=author, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$P$ [bar]")
    ax.set_ylabel(r"$x_{\mathrm{CH_4}}^{\mathrm{aq}}$  [‰]")
    ax.set_xlim(5.0, 2000.0)
    ax.set_ylim(0.1, 12.0)
    ax.grid(alpha=0.25, which="both")
    panel_label(ax, "a")
    bold_ticks(ax)
    # Stash handles/labels for the shared top legend (set below, after
    # both axes are drawn). Both panels show the SAME literature cohort
    # (panel b just spreads it across four isotherms), so a single wide
    # legend centred above both panels is far more readable than two
    # tiny per-panel legends.
    author_handles, author_labels = ax.get_legend_handles_labels()

    # ---- Panel (b) -- CH4 at 4 isotherms, linear, per-mil ----
    # Cool->warm progression on T from WONG palette: blue, bluish-green,
    # orange, vermillion. All four are colorblind-distinguishable.
    PANEL_B_COLOR = (WONG[5], WONG[3], WONG[1], WONG[6])
    ax = axes[1]
    for j, T in enumerate(ISOTHERMS_MAIN_PANEL_B):
        color = PANEL_B_COLOR[j]
        ls = LINESTYLES[j % len(LINESTYLES)]
        Ps, xs = _curve_at_T(model, T, "x_gas_aq", YC=0)
        ax.plot(Ps, xs * 1000.0, color=color, linestyle=ls, linewidth=2.4,
                label=f"{T} K")
        sub = _exp_at_T(exp_ch4, T)
        _scatter_authors(ax, sub, "P_bar", "x_CH4_aq_exp",
                         color=color, scale=1000.0)
    ax.set_xlabel(r"$P$ [bar]")
    ax.set_ylabel(r"$x_{\mathrm{CH_4}}^{\mathrm{aq}}$  [‰]")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 15.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=11)
    panel_label(ax, "b")
    bold_ticks(ax)

    # ---- Shared literature legend spanning both panels ----
    # Pick a column count that gives the squattest legend possible
    # without crowding labels: figure out how many rows result for
    # several ncol choices and pick the one that minimises rows. With
    # ~22 entries this lands at ncol=7 (3-4 rows).
    n_entries = len(author_handles)
    ncol_choices = list(range(5, min(n_entries, 11) + 1))
    ncol = min(ncol_choices, key=lambda c: (
        -((n_entries + c - 1) // c == max(1, n_entries // c)), c
    )) if ncol_choices else max(1, n_entries)
    # Simpler / more predictable: just pick ncol=7 which fits ~22 author
    # cohorts in 4 rows at the chosen fontsize. Override the heuristic
    # above unless n_entries < 7.
    ncol = min(n_entries, 7)
    # Push the panel tops close to the figure top edge, then anchor the
    # legend's bottom just above them. bbox_inches='tight' will then add
    # only a small padding above the legend, so the gap between legend
    # and panels stays narrow without overlapping the panel labels.
    fig.subplots_adjust(top=0.93)
    fig.legend(author_handles, author_labels,
               loc="lower center", bbox_to_anchor=(0.5, 0.945),
               ncol=ncol, fontsize=12, markerscale=1.3,
               columnspacing=1.4, handletextpad=0.45,
               labelspacing=0.40, frameon=True, borderaxespad=0.0)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# SI figure: 2 x 2.
# ---------------------------------------------------------------------------
def make_SI_figure(exp_co2: pd.DataFrame, exp_ch4: pd.DataFrame,
                   model: pd.DataFrame, out_path: Path) -> None:
    """Top row: x_gas_aq vs P (CO2 left, CH4 right). Bottom row: y_H2O_gas
    vs P (CO2 left, CH4 right). Four isotherms per panel."""
    # Per-mil y-axes on the top row; set the font here too so this figure
    # is not dependent on make_main_figure having run first in the process.
    _use_permille_font()

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5),
                             gridspec_kw=dict(wspace=0.28, hspace=0.32))
    binaries = [
        ("CO_2", exp_co2, "x_CO2_aq_exp", 1),
        ("CH_4", exp_ch4, "x_CH4_aq_exp", 0),
    ]
    ymax_aq = {"CO_2": 65.0, "CH_4": 8.5}        # per-mil
    panels = (("a", "b"), ("c", "d"))

    for col, (binary, exp, x_col, YC) in enumerate(binaries):
        # ---- top row: aqueous gas solubility in per-mil ----
        ax = axes[0, col]
        for j, T in enumerate(ISOTHERMS_SI):
            color = WONG[(j % 6) + 1]            # skip black (used elsewhere)
            ls = LINESTYLES[j % len(LINESTYLES)]
            Ps, xs = _curve_at_T(model, T, "x_gas_aq", YC)
            ax.plot(Ps, xs * 1000.0, color=color, linestyle=ls, linewidth=2.2,
                    label=f"{T} K")
            sub = _exp_at_T(exp, T)
            _scatter_authors(ax, sub, "P_bar", x_col,
                             color=color, scale=1000.0)
        ax.set_xlabel(r"$P$ [bar]")
        ax.set_ylabel(
            "$x_{\\mathrm{" + binary + "}}^{\\mathrm{aq}}$  [‰]"
        )
        ax.set_xlim(0, 1200)
        ax.set_ylim(0, ymax_aq[binary])
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=10, ncol=2)
        panel_label(ax, panels[0][col])
        bold_ticks(ax)

        # ---- bottom row: water in gas phase (percent) ----
        ax = axes[1, col]
        for j, T in enumerate(ISOTHERMS_SI):
            color = WONG[(j % 6) + 1]
            ls = LINESTYLES[j % len(LINESTYLES)]
            Ps, ys = _curve_at_T(model, T, "y_H2O_gas", YC)
            ax.plot(Ps, ys * 100.0, color=color, linestyle=ls, linewidth=2.2,
                    label=f"{T} K")
            sub = _exp_at_T(exp, T)
            _scatter_authors(ax, sub, "P_bar", "y_H2O_gas_exp",
                             color=color, scale=100.0)
        ax.set_xlabel(r"$P$ [bar]")
        ax.set_ylabel(
            r"$y_{\mathrm{H_2O}}^{\mathrm{" + binary + r"-rich}}$  [%]"
        )
        ax.set_xlim(0, 1200)
        ax.set_yscale("log")
        ax.set_ylim(0.02, 100.0)
        ax.grid(alpha=0.25, which="both")
        ax.legend(loc="upper right", fontsize=10, ncol=2)
        panel_label(ax, panels[1][col])
        bold_ticks(ax)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# AARE table to console (informational; not a paper artifact).
# ---------------------------------------------------------------------------
def _aare(exp_pts: np.ndarray, mod_pts: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(exp_pts) & np.isfinite(mod_pts) & (exp_pts > 0)
    if mask.sum() == 0:
        return float("nan"), 0
    return float(np.mean(np.abs(mod_pts[mask] - exp_pts[mask]) /
                         exp_pts[mask] * 100)), int(mask.sum())


def _interp_model(model: pd.DataFrame, T: float, y_col: str,
                  YC: int, P_query: np.ndarray) -> np.ndarray:
    sub = model[(model["T_K"] == T) & (model["YC"] == YC) & model["ok"]]
    sub = sub.sort_values("P_bar")
    if len(sub) < 2:
        return np.full_like(P_query, np.nan, dtype=float)
    return np.interp(P_query, sub["P_bar"].to_numpy(), sub[y_col].to_numpy(),
                     left=np.nan, right=np.nan)


def print_summary_aare(exp_co2: pd.DataFrame, exp_ch4: pd.DataFrame,
                       model: pd.DataFrame) -> None:
    print()
    print("Salt-free binary AARE summary (interpolated to experimental P at "
          "nearest model isotherm):")
    print(f"  {'binary':>8}  {'T (K)':>6}  {'col':>6}  {'AARE [%]':>10}  {'N':>5}")
    for binary, exp, x_col, YC in [
        ("CO2", exp_co2, "x_CO2_aq_exp", 1),
        ("CH4", exp_ch4, "x_CH4_aq_exp", 0),
    ]:
        for T in ISOTHERMS_SI:
            sub = _exp_at_T(exp, T)
            for col in (x_col, "y_H2O_gas_exp"):
                e = sub[col].to_numpy()
                P = sub["P_bar"].to_numpy()
                yc = "x_gas_aq" if col == x_col else "y_H2O_gas"
                m = _interp_model(model, T, yc, YC, P)
                aare, n = _aare(e, m)
                tag = "x_aq" if col == x_col else "y_H2O"
                if n:
                    print(f"  {binary:>8}  {T:>6}  {tag:>6}  "
                          f"{aare:>10.2f}  {n:>5}")


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cache_flag(p)
    args = p.parse_args()
    apply_style()

    exp_co2 = load_experimental("binaries_co2_water")
    exp_ch4 = load_experimental("binaries_ch4_water")
    model = load_or_compute("fig_binaries", compute_binary_predictions,
                            force=args.force)

    print_summary_aare(exp_co2, exp_ch4, model)
    print()
    print("Making binary figures from cached predictions ...")
    main_path = PAPER_FIGURES / "binaries_main.png"
    si_path = PAPER_FIGURES / "binaries_SI.png"
    make_main_figure(exp_co2, exp_ch4, model, main_path)
    print(f"  {main_path.name}")
    make_SI_figure(exp_co2, exp_ch4, model, si_path)
    print(f"  {si_path.name}")


if __name__ == "__main__":
    raise SystemExit(main())
