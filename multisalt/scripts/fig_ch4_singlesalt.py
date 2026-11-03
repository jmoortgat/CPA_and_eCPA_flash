#!/usr/bin/env python3
"""Generate the CH4-brine single-salt paper figures from ONE shared cache:

  - ch4_singlesalt_main.png   (main paper; NaCl @ 298 K + CaCl2 @ 343 K)
  - ch4_singlesalt_SI.png     (SI; multi-panel grid covering all 5 salts x all T)

Compute path (slow): runs ``solve_elv`` on a smooth (P, ms) grid for every
(salt, T) combination present in the experimental parquet, then re-uses
the experimental points for the scatter overlay.

Plot path (fast): re-reads the cache and re-plots the figures.

Usage:
    python3 scripts/fig_ch4_singlesalt.py            # use cache
    python3 scripts/fig_ch4_singlesalt.py --force    # recompute model curves
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "scripts"))

from ms_ecpa import solve_elv                        # noqa: E402
from paper_style import (                            # noqa: E402
    apply_style, bold_ticks, panel_label,
    PAPER_FIGURES, WONG, LINESTYLES,
    FIGSIZE_2COL, FIGSIZE_PANELS_5,
)
from data_loader import (                            # noqa: E402
    load_experimental, load_or_compute, add_cache_flag, save_stats,
)

# solve_elv physical domain.
T_MIN, T_MAX = 284, 532          # K -- strict (T0 < T < Tf)
P_MIN, P_MAX = 1.5, 1499         # bar

# Smooth-curve P sampling per isotherm. Use log-spaced because the
# experimental range spans 5 to 1500 bar.
N_CURVE_P = 30

# Five canonical salts (drop MgSO4 and MIX from the single-salt SI grid:
# we have only the 298 K single point for each, kept in the parquet but
# not plotted as an isotherm grid panel).
SALTS_ORDER = ("NaCl", "KCl", "CaCl", "MgCl", "NaSO", "MgSO")

# ----------------------------------------------------------------------
# Ion composition helper
# ----------------------------------------------------------------------

_MOL_H2O_PER_KG = 1000.0 / 18.015


def _to_permil_mod(m):
    """CH4 molality [mol/kg-H2O] -> aqueous CH4 mole fraction [permil]."""
    m = np.asarray(m, dtype=float)
    return 1000.0 * m / (_MOL_H2O_PER_KG + m)


def _ions_for(salt: str, ms: float) -> np.ndarray:
    """Return the (Na, Cl, K, Ca, Mg, SO4) molality vector for ``salt``
    at salt molality ``ms`` (mol/kg water).

    Stoichiometry follows ``OPT_ALLIONS.py`` ``vion_SALT``::

        NaCl -> (Na+, Cl-)              ms each
        KCl  -> (K+, Cl-)               ms each
        CaCl -> (Ca2+, 2 Cl-)           Ca=ms,  Cl=2 ms
        MgCl -> (Mg2+, 2 Cl-)           Mg=ms,  Cl=2 ms
        NaSO -> (2 Na+, SO4^2-)         Na=2 ms, SO4=ms
        MgSO -> (Mg2+, SO4^2-)          Mg=ms,  SO4=ms
    """
    mNa, mCl, mK, mCa, mMg, mSO4 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if salt == "NaCl":
        mNa = ms; mCl = ms
    elif salt == "KCl":
        mK  = ms; mCl = ms
    elif salt == "CaCl":
        mCa = ms; mCl = 2 * ms
    elif salt == "MgCl":
        mMg = ms; mCl = 2 * ms
    elif salt == "NaSO":
        mNa = 2 * ms; mSO4 = ms
    elif salt == "MgSO":
        mMg = ms; mSO4 = ms
    else:
        raise ValueError(f"unknown single-salt label: {salt}")
    return np.array([mNa, mCl, mK, mCa, mMg, mSO4], dtype=float)


# ----------------------------------------------------------------------
# Compute step
# ----------------------------------------------------------------------
def compute_ch4_singlesalt() -> pd.DataFrame:
    """For each (salt, T, ms) combination present in the parquet, evaluate
    the eCPA model on a smooth log-spaced P grid spanning the experimental
    pressure range for that combination.

    Returns a long-form DataFrame with columns
    ``salt, T_K, ms_mol_kg, P_bar, mh_W_pred, converged``. Failed solves
    keep their row with ``mh_W_pred=NaN`` and ``converged=False`` so the
    cache records the attempt.
    """
    exp = load_experimental("ch4_brines")
    # We only model points that fit the solver's domain.
    exp = exp[(exp["T_K"] >= T_MIN) & (exp["T_K"] <= T_MAX)].copy()
    out_rows = []
    keys = exp.groupby(["salt", "T_K", "ms_mol_kg"]).groups.keys()
    keys = sorted(keys)
    t0 = time.perf_counter()
    n_keys = len(keys)
    for ik, (salt, T, ms) in enumerate(keys):
        sub = exp[(exp["salt"] == salt) & (exp["T_K"] == T) &
                  (np.isclose(exp["ms_mol_kg"], ms))]
        P_lo = max(P_MIN, float(sub["P_bar"].min()) * 0.9)
        P_hi = min(P_MAX, float(sub["P_bar"].max()) * 1.1)
        if P_hi <= P_lo:
            continue
        P_grid = np.geomspace(P_lo, P_hi, N_CURVE_P)
        mi = _ions_for(salt, float(ms))
        ier_ok = 0
        for P in P_grid:
            try:
                r = solve_elv(float(T), float(P), YC=0, mi=mi)
                if r.converged and np.isfinite(r.mh):
                    out_rows.append({
                        "salt": salt, "T_K": int(T), "ms_mol_kg": float(ms),
                        "P_bar": float(P), "mh_W_pred": float(r.mh),
                        "converged": True,
                    })
                    ier_ok += 1
                    continue
            except Exception:
                pass
            out_rows.append({
                "salt": salt, "T_K": int(T), "ms_mol_kg": float(ms),
                "P_bar": float(P), "mh_W_pred": float("nan"),
                "converged": False,
            })
        if (ik + 1) % 10 == 0 or ik == n_keys - 1:
            print(f"  [{ik+1}/{n_keys}] {salt} T={T}K ms={ms:.2f}  "
                  f"OK {ier_ok}/{len(P_grid)}   "
                  f"({time.perf_counter()-t0:.1f}s elapsed)")
    return pd.DataFrame(out_rows)


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------
def _panel_isotherm(ax, exp_sub: pd.DataFrame, pred_sub: pd.DataFrame,
                    salt: str, T: int) -> None:
    """Plot a single salt/T isotherm panel: predicted curves (one per ms)
    + experimental points (matching ms).

    Axes: x = P [bar], y = mh [mol/kg water] (CH4 molality). Both axes
    log-scaled because the data spans 3-4 decades.
    """
    ms_values = sorted(exp_sub["ms_mol_kg"].unique())
    for i, ms in enumerate(ms_values):
        color = WONG[(i + 1) % len(WONG)]
        ls = LINESTYLES[i % len(LINESTYLES)]
        ep = exp_sub[np.isclose(exp_sub["ms_mol_kg"], ms)]
        pp = pred_sub[np.isclose(pred_sub["ms_mol_kg"], ms) &
                       (pred_sub["converged"])]
        if len(pp):
            pp_sorted = pp.sort_values("P_bar")
            ax.plot(pp_sorted["P_bar"], pp_sorted["mh_W_pred"],
                    color=color, linestyle=ls, lw=2.2,
                    label=f"{ms:g} m")
        # Per-author markers help distinguish overlapping datasets.
        authors = ep["author"].unique()
        marker_cycle = ("o", "s", "^", "D", "v", ">", "<", "P", "X", "*")
        for j, auth in enumerate(authors):
            ea = ep[ep["author"] == auth]
            ax.scatter(ea["P_bar"], ea["mh_W"],
                       color=color, marker=marker_cycle[j % len(marker_cycle)],
                       s=42, edgecolors="black", linewidths=0.8,
                       zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$P$ [bar]")
    ax.set_ylabel(r"$m_{\mathrm{CH_4}}$ [mol kg$^{-1}$]")
    ax.grid(which="both", alpha=0.25)
    bold_ticks(ax)


def _salt_label(salt: str) -> str:
    """Pretty LaTeX label for a salt code."""
    return {
        "NaCl": r"NaCl",
        "KCl":  r"KCl",
        "CaCl": r"CaCl$_2$",
        "MgCl": r"MgCl$_2$",
        "NaSO": r"Na$_2$SO$_4$",
        "MgSO": r"MgSO$_4$",
    }.get(salt, salt)


# ----------------------------------------------------------------------
# Main paper figure: NaCl 298 K + CaCl2 343 K
# ----------------------------------------------------------------------
MAIN_PAPER_PANELS = [
    ("NaCl", 298),
    ("CaCl", 343),
]


def make_main_paper(exp: pd.DataFrame, pred: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D
    # Real LaTeX text rendering so \textperthousand renders as ‰ in the
    # y-axis label.  Enabled locally so subsequent SI figures still use
    # matplotlib's native mathtext.
    # Use a font that renders ‰ cleanly (default DejaVu Sans stacks
    # per-mille as a small "%o" that reads poorly).  Helvetica is
    # available on macOS and most journal-figure LaTeX setups.
    prev_family = plt.rcParams["font.family"]
    plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]

    # y-axis: aqueous CH4 mole fraction in per-mil, matching the convention
    # of §S12 and the canonical solubility table. Universal one-to-one map
    # from molality (mol per kg H2O):
    #     x_CH4 = m_CH4 / (n_H2O + m_CH4),  n_H2O = 1000 / M_H2O ≈ 55.508 mol/kg
    # (does NOT depend on salt loading; per-kg-water normalises that out).
    _MOL_H2O_PER_KG = 1000.0 / 18.015

    def _to_permil(m):
        m = np.asarray(m, dtype=float)
        return 1000.0 * m / (_MOL_H2O_PER_KG + m)

    # Wider, less elongated panels (matches co2_singlesalt_main style).
    # Share y-axis across panels for direct comparison, but let each panel
    # set its own x range -- the two isotherms cover quite different
    # pressure windows (NaCl 298 K: 24..1379 bar; CaCl2 343 K: 101..608),
    # and a shared x leaves most of the right panel empty.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)

    # Global y limits from the union of both panels' data; per-panel x
    # limits computed inside the loop below.
    panel_exp = pd.concat([exp[(exp["salt"] == s) & (exp["T_K"] == T)]
                           for s, T in MAIN_PAPER_PANELS])
    panel_pred = pd.concat([pred[(pred["salt"] == s) & (pred["T_K"] == T)
                                  & pred["converged"]]
                            for s, T in MAIN_PAPER_PANELS])
    mh_all = np.concatenate([_to_permil(panel_exp["mh_W"].values),
                             _to_permil(panel_pred["mh_W_pred"].values)])
    mh_all = mh_all[(mh_all > 0) & np.isfinite(mh_all)]
    # Multiplicative margin in log space (~10% padding each side).
    y_lim = (mh_all.min() / 1.10, mh_all.max() * 1.10)

    # Canonical molality list (sorted) -> consistent color & linestyle per ms
    # so the right panel's "1 m" curve shares the color/style of the left
    # panel's "1 m" curve, letting a single shared legend cover both.
    ms_canon = sorted({float(ms)
                       for s, T in MAIN_PAPER_PANELS
                       for ms in exp[(exp["salt"] == s) & (exp["T_K"] == T)]
                                    ["ms_mol_kg"].unique()})
    ms_color = {ms: WONG[(i + 1) % len(WONG)] for i, ms in enumerate(ms_canon)}
    ms_style = {ms: LINESTYLES[i % len(LINESTYLES)] for i, ms in enumerate(ms_canon)}

    for i, (salt, T) in enumerate(MAIN_PAPER_PANELS):
        ax = axes[i]
        ep = exp[(exp["salt"] == salt) & (exp["T_K"] == T)]
        pp = pred[(pred["salt"] == salt) & (pred["T_K"] == T) & pred["converged"]]
        ms_values = sorted(ep["ms_mol_kg"].unique())
        marker_cycle = ("o", "s", "^", "D", "v", ">", "<", "P", "X", "*")
        for ms in ms_values:
            color = ms_color[ms]
            ls = ms_style[ms]
            pp_sub = pp[np.isclose(pp["ms_mol_kg"], ms)].sort_values("P_bar")
            if len(pp_sub):
                ax.plot(pp_sub["P_bar"], _to_permil(pp_sub["mh_W_pred"]),
                        color=color, linestyle=ls, lw=2.2)
            ep_sub = ep[np.isclose(ep["ms_mol_kg"], ms)]
            for j, auth in enumerate(ep_sub["author"].unique()):
                ea = ep_sub[ep_sub["author"] == auth]
                ax.scatter(ea["P_bar"], _to_permil(ea["mh_W"]),
                           color=color,
                           marker=marker_cycle[j % len(marker_cycle)],
                           s=42, edgecolors="black", linewidths=0.8,
                           zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Per-panel x range from this panel's own data.
        P_panel = np.concatenate([ep["P_bar"].values, pp["P_bar"].values])
        P_panel = P_panel[(P_panel > 0) & np.isfinite(P_panel)]
        x_lim = (P_panel.min() / 1.10, P_panel.max() * 1.10)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel(r"$P$ [bar]")
        if i == 0:
            ax.set_ylabel(r"$x_{\mathrm{CH_4}}^{\mathrm{aq}}$ [‰]")
        # Log x-axis with plain-number tick labels at sensible round
        # positions inside this panel's range.
        x_candidates = (10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000)
        x_majors = [t for t in x_candidates if x_lim[0] <= t <= x_lim[1]]
        ax.xaxis.set_major_locator(mticker.FixedLocator(x_majors))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(t) for t in x_majors]))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        # Y axis: decade ticks only (label 10^-2, 10^-1; minor ticks blank).
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.grid(which="both", alpha=0.25)
        panel_label(ax, chr(ord("a") + i))
        # Compact "<salt>, <T> K" annotation (no "T =" prefix).
        ax.text(0.96, 0.06, f"{_salt_label(salt)}, {T} K",
                transform=ax.transAxes, ha="right", va="bottom",
                fontweight="bold", fontsize=12,
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="none", pad=2))
        bold_ticks(ax)

    # Single shared legend across the top of the figure, with an inline
    # ``m_s:`` prefix (matching the SI per-salt legends).
    title_handle = Line2D([0], [0], linestyle="None", marker="",
                          label=r"$m_s$:")
    ms_handles = [Line2D([0], [0], color=ms_color[ms], linestyle=ms_style[ms],
                         linewidth=2.4, label=f"{ms:g} m")
                  for ms in ms_canon]
    handles = [title_handle] + ms_handles
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.005),
               ncol=len(handles), frameon=True, fontsize=11,
               handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    # Restore previous font family.
    plt.rcParams["font.family"] = prev_family


# ----------------------------------------------------------------------
# SI figures: one figure per salt-family
#
# The original single-SI-grid was 25 panels, ungainly and inconsistent with
# the main figure. We split into four figures matching the natural
# isotherm count per salt:
#   - NaCl  : 13 isotherms      (4 cols x 4 rows)
#   - KCl   :  4 isotherms      (2 cols x 2 rows)
#   - CaCl2 :  5 isotherms      (3 cols x 2 rows)
#   - other : 3 single-T panels (MgCl2, MgSO4, Na2SO4)  (3 cols x 1 row)
# Each SI figure uses the same conventions as the main figure: shared y,
# per-panel x, plain-number ticks on log axes, decade-only y labels,
# compact "<salt>, <T> K" annotation, single shared top legend.
# ----------------------------------------------------------------------
def _make_SI_panels(exp_in: pd.DataFrame, pred: pd.DataFrame,
                    pairs: list, ncol: int, out_path: Path) -> None:
    """Plot a grid of (salt, T) panels in main-figure style.

    Parameters
    ----------
    pairs : list of (salt, T_K) tuples
        Panel selection in order.
    ncol : int
        Number of columns; the row count is computed to fit.
    """
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D

    if not pairs:
        return
    n = len(pairs)
    nrow = (n + ncol - 1) // ncol

    # Global y limits from the union of all panel data.
    panel_exp = pd.concat([exp_in[(exp_in["salt"] == s) & (exp_in["T_K"] == T)]
                           for s, T in pairs])
    panel_pred = pd.concat([pred[(pred["salt"] == s) & (pred["T_K"] == T)
                                  & pred["converged"]] for s, T in pairs])
    mh_all = np.concatenate([_to_permil_mod(panel_exp["mh_W"].values),
                             _to_permil_mod(panel_pred["mh_W_pred"].values)])
    mh_all = mh_all[(mh_all > 0) & np.isfinite(mh_all)]
    y_lim = (mh_all.min() / 1.10, mh_all.max() * 1.10) if len(mh_all) else (1e-2, 10.0)

    # Canonical molality list across all panels in this figure -> consistent
    # color & linestyle per ms.
    ms_canon = sorted({float(ms)
                       for s, T in pairs
                       for ms in exp_in[(exp_in["salt"] == s)
                                        & (exp_in["T_K"] == T)]
                                ["ms_mol_kg"].unique()})
    ms_color = {ms: WONG[(i + 1) % len(WONG)] for i, ms in enumerate(ms_canon)}
    ms_style = {ms: LINESTYLES[i % len(LINESTYLES)] for i, ms in enumerate(ms_canon)}
    marker_cycle = ("o", "s", "^", "D", "v", ">", "<", "P", "X", "*")

    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(3.6 * ncol, 3.2 * nrow),
                             squeeze=False, sharey=True)
    x_candidates = (10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000)
    for i, (salt, T) in enumerate(pairs):
        r, c = divmod(i, ncol)
        ax = axes[r][c]
        ep = exp_in[(exp_in["salt"] == salt) & (exp_in["T_K"] == T)]
        pp = pred[(pred["salt"] == salt) & (pred["T_K"] == T) & pred["converged"]]
        ms_values = sorted(ep["ms_mol_kg"].unique())
        for ms in ms_values:
            color = ms_color[ms]
            ls = ms_style[ms]
            pp_sub = pp[np.isclose(pp["ms_mol_kg"], ms)].sort_values("P_bar")
            if len(pp_sub):
                ax.plot(pp_sub["P_bar"], _to_permil_mod(pp_sub["mh_W_pred"]),
                        color=color, linestyle=ls, lw=2.2)
            ep_sub = ep[np.isclose(ep["ms_mol_kg"], ms)]
            for j, auth in enumerate(ep_sub["author"].unique()):
                ea = ep_sub[ep_sub["author"] == auth]
                ax.scatter(ea["P_bar"], _to_permil_mod(ea["mh_W"]),
                           color=color,
                           marker=marker_cycle[j % len(marker_cycle)],
                           s=36, edgecolors="black", linewidths=0.7,
                           zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(y_lim)
        # Per-panel x range.
        P_panel = np.concatenate([ep["P_bar"].values, pp["P_bar"].values])
        P_panel = P_panel[(P_panel > 0) & np.isfinite(P_panel)]
        if len(P_panel):
            x_lim = (P_panel.min() / 1.10, P_panel.max() * 1.10)
            ax.set_xlim(x_lim)
        else:
            x_lim = ax.get_xlim()
        # X label on bottom row only, Y label on leftmost column only.
        if r == nrow - 1 or i + ncol >= n:
            ax.set_xlabel(r"$P$ [bar]")
        if c == 0:
            ax.set_ylabel(r"$x_{\mathrm{CH_4}}^{\mathrm{aq}}$ [‰]")
        # Plain-number x-ticks inside the data range.
        x_majors = [t for t in x_candidates if x_lim[0] <= t <= x_lim[1]]
        ax.xaxis.set_major_locator(mticker.FixedLocator(x_majors))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(t) for t in x_majors]))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.grid(which="both", alpha=0.25)
        panel_label(ax, chr(ord("a") + i))
        ax.text(0.96, 0.06, f"{_salt_label(salt)}, {T} K",
                transform=ax.transAxes, ha="right", va="bottom",
                fontweight="bold", fontsize=11,
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="none", pad=2))
        bold_ticks(ax)
    # Blank any leftover panels in the last row.
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    # Single shared legend at the top.
    handles = [Line2D([0], [0], color=ms_color[ms], linestyle=ms_style[ms],
                      linewidth=2.4, label=f"{ms:g} m")
               for ms in ms_canon]
    legend_ncol = min(len(handles), 8)
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.003),
               ncol=legend_ncol, frameon=True, fontsize=10)
    top_rect = 0.98 - 0.03 * ((len(handles) + legend_ncol - 1) // legend_ncol)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top_rect))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_SI_figures(exp: pd.DataFrame, pred: pd.DataFrame,
                    out_dir: Path) -> list:
    """Produce one SI figure per salt-family. Returns the list of paths."""
    exp_in = exp[(exp["T_K"] >= T_MIN) & (exp["T_K"] <= T_MAX)]

    def pairs_for(salts):
        return sorted(
            exp_in[exp_in["salt"].isin(salts)]
                  .groupby(["salt", "T_K"]).groups.keys(),
            key=lambda st: (SALTS_ORDER.index(st[0]) if st[0] in SALTS_ORDER else 99,
                            st[1])
        )

    # Per-salt panel-drop overrides (T_K values to skip from the SI grid).
    # NaCl 375 K has only 2 curves and was the 13th panel in an otherwise
    # natural 12-panel grid; drop it so we land on a clean 3 cols x 4 rows.
    DROP_T = {"NaCl": {375}}

    out_paths = []
    plan = [
        # (salts, ncol, output filename)
        (["NaCl"],                 3, "ch4_singlesalt_SI_NaCl.png"),
        (["KCl"],                  2, "ch4_singlesalt_SI_KCl.png"),
        (["CaCl"],                 3, "ch4_singlesalt_SI_CaCl2.png"),
        (["MgCl", "MgSO", "NaSO"], 3, "ch4_singlesalt_SI_other.png"),
    ]
    for salts, ncol, fname in plan:
        pairs = [(s, T) for (s, T) in pairs_for(salts)
                 if T not in DROP_T.get(s, set())]
        if not pairs:
            continue
        out_path = out_dir / fname
        _make_SI_panels(exp_in, pred, pairs, ncol, out_path)
        out_paths.append(out_path)
        print(f"  {fname}  ({len(pairs)} panels)")
    return out_paths


# ----------------------------------------------------------------------
# Headline AARE summary (printed only; not on the figure)
# ----------------------------------------------------------------------
def _interp_pred_at_exp_P(pred_iso: pd.DataFrame, P_exp: np.ndarray) -> np.ndarray:
    """Log-interpolate the smooth predicted mh_W curve at experimental P."""
    pp = pred_iso[pred_iso["converged"]].sort_values("P_bar")
    if len(pp) < 2:
        return np.full_like(P_exp, np.nan, dtype=float)
    return np.exp(np.interp(np.log(np.asarray(P_exp, dtype=float)),
                            np.log(pp["P_bar"].to_numpy()),
                            np.log(pp["mh_W_pred"].to_numpy()),
                            left=np.nan, right=np.nan))


def print_aare_table(exp: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Print per-(salt, T) AARE table for CH4 in single-salt brines AND
    return a JSON-serializable dict of headline numbers (Susak NaCl 548 K
    special-case, per-salt totals, overall total) for the paper stats.

    Section 3.5 of the paper text cites the Susak1980 CH4-NaCl AAD
    (line 1085) and a small-salt range (1.5-6.9%, line 1088). Both
    become macros via this stats sidecar.
    """
    print()
    print("=" * 70)
    print(f"{'salt':>6}  {'T [K]':>6}  {'N':>4}  {'AARE [%]':>10}")
    print("-" * 70)
    exp_in = exp[(exp["T_K"] >= T_MIN) & (exp["T_K"] <= T_MAX)]
    total_err, total_n = 0.0, 0
    per_iso: list[dict] = []
    per_salt_err: dict[str, dict[str, float]] = {}
    for (salt, T), g in exp_in.groupby(["salt", "T_K"]):
        pp = pred[(pred["salt"] == salt) & (pred["T_K"] == T)]
        # Group by ms then interp
        errs = []
        for ms, gm in g.groupby("ms_mol_kg"):
            pp_ms = pp[np.isclose(pp["ms_mol_kg"], ms)]
            pred_at = _interp_pred_at_exp_P(pp_ms, gm["P_bar"].to_numpy())
            mask = np.isfinite(pred_at) & (gm["mh_W"].to_numpy() > 0)
            if mask.any():
                rel = np.abs(pred_at[mask] - gm["mh_W"].to_numpy()[mask]) / \
                       gm["mh_W"].to_numpy()[mask]
                errs.extend(rel.tolist())
        if errs:
            aare_pct = 100 * float(np.mean(errs))
            n = len(errs)
            # int(T), not {T:d}: GasBrineBench carries T_K as float (298.0)
            # where the legacy parquet carried it as int, and a 'd' format
            # code raises on a float. The value is an isotherm label, so it
            # rounds cleanly; the stored per_iso entry below already did this.
            print(f"{salt:>6}  {int(round(T)):>6d}  {n:>4d}  {aare_pct:>10.2f}")
            total_err += float(np.sum(errs))
            total_n += n
            per_iso.append({"salt": salt, "T_K": int(round(T)), "N": n,
                            "aare": round(aare_pct, 2)})
            bucket = per_salt_err.setdefault(salt, {"sum_err": 0.0, "N": 0})
            bucket["sum_err"] += float(np.sum(errs))
            bucket["N"] += n
    print("-" * 70)
    total_aare = 100 * total_err / total_n if total_n else float("nan")
    if total_n:
        print(f"{'TOTAL':>6}  {'':>6}  {total_n:>4d}  {total_aare:>10.2f}")
    print("=" * 70)

    per_salt = {
        salt: {"N": v["N"], "aare": round(100 * v["sum_err"] / v["N"], 2)}
        for salt, v in per_salt_err.items()
    }
    # Susak's high-T NaCl isotherms (548/573/598 K) cannot be scored here:
    # solve_elv's domain stops at T_MAX = 532 K, so exp_in -- and therefore
    # per_iso -- never contains them. The number §3.5 quotes comes from
    # build_paper_tables.py::_susak_high_T_augment(), which runs
    # stability_flash on those points and writes susak_nacl_full_T_aad to
    # its own stats sidecar. The key is kept (always None) so the
    # build_paper_stats_tex.py fallback keeps a defined shape to read.
    susak_nacl_high_T = None
    # Range across smaller-window salts (KCl, MgCl, NaSO, MgSO) --
    # use per-salt totals (across all isotherms of that salt), not raw
    # per-isotherm values, so the "1.5-6.9%" style bracket in §3.5 maps
    # to overall salt AAREs rather than the worst individual isotherm.
    small_totals = [per_salt[s]["aare"] for s in ("KCl", "MgCl", "NaSO", "MgSO")
                    if s in per_salt]
    range_low = round(min(small_totals), 2) if small_totals else None
    range_high = round(max(small_totals), 2) if small_totals else None
    stats = {
        "N_total": total_n,
        "aare_overall": round(total_aare, 2),
        "susak_nacl_high_T_aare": susak_nacl_high_T,
        "small_salt_range_low": range_low,
        "small_salt_range_high": range_high,
        "per_salt": per_salt,
        "per_iso": per_iso,
    }
    return stats


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cache_flag(p)
    args = p.parse_args()
    apply_style()
    exp = load_experimental("ch4_brines")
    pred = load_or_compute("fig_ch4_singlesalt", compute_ch4_singlesalt,
                           force=args.force)
    stats = print_aare_table(exp, pred)
    save_stats("fig_ch4_singlesalt", stats)
    print()
    print("Making 2 figures from cached predictions ...")
    make_main_paper(exp, pred, PAPER_FIGURES / "ch4_singlesalt_main.png")
    print("  ch4_singlesalt_main.png")
    make_SI_figures(exp, pred, PAPER_FIGURES)


if __name__ == "__main__":
    raise SystemExit(main())
