#!/usr/bin/env python3
"""Generate the CO2 single-salt brine paper figures from the Phase-1 cache.

  - co2_singlesalt_main.png   2 panels: NaCl @ 323 K + CaCl2 @ 423 K
  - co2_singlesalt_SI.png     5x3 grid: {NaCl, KCl, CaCl, MgCl, NaSO} x {323, 373, 423} K

Each panel plots CO2 molality :math:`m_{\mathrm{CO_2}}` (mol/kg H2O) as a
function of pressure, with one model curve per salinity ``ms`` and per-author
experimental scatter overlaid. The model prediction is converted from the
salt-free CO2 mole fraction via ``mc = xcW / (1 - xcW) / Mw``.

Compute path (slow, ~3 minutes): runs ``stability_flash`` at every
experimental (T, P, salt, ions) point. To get smooth model curves we use
the predicted mc values directly at the experimental P grid (this is the
same prediction the parity figures already use; it matches the experimental
sweep without re-running a separate pressure mesh).

Plot path (fast, ~3 s): re-reads the cache and re-plots both figures.

Usage:
    python3 scripts/fig_co2_singlesalt.py            # use cache
    python3 scripts/fig_co2_singlesalt.py --force    # recompute
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "scripts"))

from ms_ecpa import stability_flash                    # noqa: E402
from ms_ecpa.parameters import Mw as _Mw               # noqa: E402
from paper_style import (                              # noqa: E402
    apply_style, bold_ticks, panel_label,
    PAPER_FIGURES, WONG, LINESTYLES,
)
from data_loader import (                              # noqa: E402
    load_experimental, load_or_compute, add_cache_flag, save_stats,
)

# Water molar mass, taken from the library rather than re-rounded here.
# The per-mil conversion further down inverts this same constant, so a
# local rounding makes the molality round trip a non-identity: 0.018
# against the library's 0.01801528 is 0.085 %, and parameters.py records
# that the model constant and the molality conversions applied to the
# data it is fitted against must not differ.
MW = _Mw  # kg/mol H2O
Z_CO2_RETRIES = (0.05, 0.10, 0.02, 0.20, 0.30)

# Salts present in co2_brines.parquet and the order they appear in the SI
# grid. Names follow the source directory convention.
SALTS = ("NaCl", "KCl", "CaCl", "MgCl", "NaSO")
SALT_LABEL = {
    "NaCl": r"NaCl",
    "KCl":  r"KCl",
    "CaCl": r"CaCl$_2$",
    "MgCl": r"MgCl$_2$",
    "NaSO": r"Na$_2$SO$_4$",
}
TEMPS = (323, 373, 423)

# Markers cycled per author within a panel.
AUTHOR_MARKERS = ("o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h")

FIGSIZE_TWO_PANEL = (10.5, 4.4)
FIGSIZE_SI_GRID  = (15, 18)


# ---------------------------------------------------------------------------
# Compute step
# ---------------------------------------------------------------------------
# Smooth model-curve pressure grid: 5 to 600 bar, 16 points (~40 bar
# spacing). Dense enough for the gentle mc(P) curvature without doubling
# the compute time. Skip 0 bar (vapor degenerate, single-phase trivially).
MODEL_P_GRID = np.linspace(5.0, 600.0, 16)


def _flash_one(T: float, P: float, mi: np.ndarray) -> tuple[float, float]:
    """One flash -> ``(salt-free xc_W, z_CO2 seed that produced it)``, or
    ``(NaN, NaN)``. Uses the adaptive z_CO2 ladder pattern from
    fig_flash_predictions to rescue tricky low-P / high-ms points.

    The seed is returned so it can be stored alongside the prediction: the
    degenerate-2-phase test below is only meaningful against the seed
    actually in play, and a cache that does not record it forces the
    cache-side re-test to guess (see :func:`_is_seed_degenerate`).

    Guards against a degenerate-2-phase artefact: at some (T, P, mi)
    combinations stability_flash returns ``phases == 2`` but with the
    aqueous CO2 mole fraction essentially equal to the feed ``zC`` -- a
    fake split where no mass actually moved between phases. Those points
    produce a spike at ``mc_pred = zC / (1 - zC) / Mw`` (e.g. 2.92 mol/kg
    for zC=0.05) that contaminates the smooth curves. We reject any
    result with ``xC - zC`` below a tight tolerance and continue the
    retry ladder.
    """
    for zC in Z_CO2_RETRIES:
        z = np.array([1.0 - zC, 0.0, zC])
        try:
            res = stability_flash(float(T), float(P), z, mi)
            if res.phases == 2 and res.flash_result is not None:
                if not bool(res.flash_result.converged):
                    continue          # the flash's own verdict, never read
                xH, _, xC = res.flash_result.x_aq
                xc_W = float(xC / (xH + xC))
                # Degenerate-2-phase guard: the salt-free CO2 mole fraction
                # in the aqueous phase numerically equal to the feed CO2
                # mole fraction means the flash preserved the feed
                # H2O:CO2 ratio in the aqueous phase -- i.e. no selective
                # partitioning happened. The CO2 mole fraction xC itself
                # is not equal to zC in that case (ions take up the
                # difference), so we check xc_W rather than xC.
                if abs(xc_W - zC) < 1e-3 * max(zC, 1e-6):
                    continue
                return xc_W, zC
        except Exception:
            continue
    return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# Cache-side counterpart of the degenerate-2-phase guard in _flash_one
# ---------------------------------------------------------------------------
# The guard inside _flash_one keeps the artefact out of any cache written by
# this version of the script. Caches written before it existed still carry
# the spikes ([[feedback_degenerate_flash_artifact]] counted 140+ in this
# very file), and ``make paper`` reuses a cache without recomputing, so the
# same test is re-applied when the cache is read. Both the figure and the
# per-salt AAD go through this one function, so the curve and the number
# beside it are always computed over the same rows.


def _is_seed_degenerate(xcw: float, z_used: float = float("nan")) -> bool:
    """Whether ``xcw`` is the feed-copy artefact described in _flash_one.

    Mirrors the compute-side test exactly: the same RELATIVE tolerance, and,
    where the cache records the seed that produced the value, the same
    single seed. An absolute window is the wrong shape here -- 1e-4 is five
    times wider than the compute-side test at zC=0.02, and aqueous CO2 mole
    fractions in this dataset run straight through 0.02, so it deletes
    ordinary converged predictions along a horizontal slice of every curve
    that crosses that level.
    """
    if not np.isfinite(xcw):
        return False
    seeds = (z_used,) if np.isfinite(z_used) else Z_CO2_RETRIES
    return any(abs(xcw - z) < 1e-3 * max(z, 1e-6) for z in seeds)


def drop_seed_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` without the degenerate-2-phase rows.

    Caches predating the ``z_CO2_used`` column carry no record of the seed,
    so every ladder entry has to be tested there; that is looser than the
    compute-side guard, which is unavoidable without a recompute.
    """
    if not len(df):
        return df
    if "z_CO2_used" in df.columns:
        bad = [_is_seed_degenerate(x, z)
               for x, z in zip(df["xc_W_pred"], df["z_CO2_used"])]
    else:
        bad = [_is_seed_degenerate(x) for x in df["xc_W_pred"]]
    return df[~np.asarray(bad, dtype=bool)]


def compute_singlesalt() -> pd.DataFrame:
    """Run flash at every experimental row AND on a smooth P grid per recipe.

    Two row kinds in the output (``kind`` column):
      ``exp``   one per experimental observation -- carries xc_W, mc_exp, author.
      ``model`` smooth model curve points on MODEL_P_GRID per
                (salt, T, ms_mol_kg, recipe) combination -- no exp values.
    """
    exp = load_experimental("co2_brines")
    print(f"  experimental points: {len(exp)}")
    ion_cols = ["m_Na", "m_Cl", "m_K", "m_Ca", "m_Mg", "m_SO4"]
    out_rows: list[dict] = []
    t0 = time.perf_counter()

    # Pass 1: experimental observations (each row stores both exp and pred).
    for _, r in exp.iterrows():
        mi = np.array([float(r[c]) for c in ion_cols])
        xcW, zC_used = _flash_one(r["T_K"], r["P_bar"], mi)
        mc_pred = (xcW / (1.0 - xcW) / MW
                   if np.isfinite(xcW) and 0.0 < xcW < 1.0 else np.nan)
        mc_exp = r["xc_W"] / (1.0 - r["xc_W"]) / MW
        out_rows.append({
            "kind":      "exp",
            "salt":      r["salt"],
            "T_K":       int(r["T_K"]),
            "P_bar":     float(r["P_bar"]),
            "ms_mol_kg": float(r["ms_mol_kg"]),
            **{c: float(r[c]) for c in ion_cols},
            "author":    r["author"],
            "xc_W":      float(r["xc_W"]),
            "mc_exp":    float(mc_exp),
            "xc_W_pred": xcW,
            "mc_pred":   mc_pred,
            "z_CO2_used": zC_used,
            "converged": bool(np.isfinite(xcW)),
        })
    n_conv_exp = sum(1 for x in out_rows if x["converged"])
    print(f"  experimental pass: {n_conv_exp}/{len(out_rows)} converged "
          f"({time.perf_counter()-t0:.1f} s)")

    # Pass 2: smooth model curves on MODEL_P_GRID per unique recipe.
    unique_keys: dict[tuple, dict] = {}
    for _, r in exp.iterrows():
        key = (r["salt"], int(r["T_K"]), round(float(r["ms_mol_kg"]), 4),
               tuple(round(float(r[c]), 6) for c in ion_cols))
        if key not in unique_keys:
            unique_keys[key] = {
                "salt": r["salt"], "T_K": int(r["T_K"]),
                "ms_mol_kg": float(r["ms_mol_kg"]),
                **{c: float(r[c]) for c in ion_cols},
            }
    print(f"  model-grid pass: {len(unique_keys)} recipes x "
          f"{len(MODEL_P_GRID)} P -> {len(unique_keys)*len(MODEL_P_GRID)} flashes")
    t1 = time.perf_counter()
    for meta in unique_keys.values():
        mi = np.array([meta[c] for c in ion_cols])
        for P in MODEL_P_GRID:
            xcW, zC_used = _flash_one(meta["T_K"], float(P), mi)
            mc_pred = (xcW / (1.0 - xcW) / MW
                       if np.isfinite(xcW) and 0.0 < xcW < 1.0 else np.nan)
            out_rows.append({
                "kind":      "model",
                "salt":      meta["salt"],
                "T_K":       meta["T_K"],
                "P_bar":     float(P),
                "ms_mol_kg": meta["ms_mol_kg"],
                **{c: meta[c] for c in ion_cols},
                "author":    "",
                "xc_W":      float("nan"),
                "mc_exp":    float("nan"),
                "xc_W_pred": xcW,
                "mc_pred":   mc_pred,
                "z_CO2_used": zC_used,
                "converged": bool(np.isfinite(xcW)),
            })
    print(f"  ...total {time.perf_counter()-t0:.1f} s")
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
# Water molar mass, used to convert CO2 molality (mol per kg H2O) to CO2
# mole fraction on the partitioning basis (ions excluded, matches §S12
# and the canonical solubility table). The conversion is
#     x_CO2 = m_CO2 / (n_H2O + m_CO2),   n_H2O = 1000 / M_H2O ≈ 55.508 mol/kg
# which is a universal one-to-one function of m_CO2 alone — it does NOT
# depend on the salt loading, because per-kg-water molality already
# normalises out the salt-mass contribution.  Built from the same MW that
# made the molality, so x -> m -> x is the identity.
_MOL_H2O_PER_KG = 1.0 / MW

def _molality_to_permil(m):
    """CO2 molality [mol/kg-H2O] → aqueous CO2 mole fraction [‰]."""
    m = np.asarray(m, dtype=float)
    return 1000.0 * m / (_MOL_H2O_PER_KG + m)


def _plot_panel(ax, df_panel: pd.DataFrame, *, show_legend: bool = True,
                legend_fontsize: int = 9, legend_ncol: int = 1,
                color_for_ms: dict | None = None,
                style_for_ms: dict | None = None,
                y_unit: str = "molality") -> None:
    """Plot one (salt, T) panel: mc vs P. One model curve and one author-
    cycled scatter cohort per ms_mol_kg salinity level.

    ``color_for_ms``: optional ``{ms_value: color}`` mapping used to assign
    the same color to a given molality across multiple panels in the same
    figure. When ``None``, the local viridis cmap is used (per-panel
    colors), which is the standalone-panel default.
    ``style_for_ms``: optional ``{ms_value: linestyle}`` mapping. When
    supplied, lines are drawn with these styles -- essential for
    colorblind / grayscale readability when there are many series.
    """
    exp_rows = df_panel[df_panel["kind"] == "exp"]
    # The "model" cache rows are flash predictions on the smooth MODEL_P_GRID
    # (5..600 bar). The "exp" rows ALSO carry a flash prediction (mc_pred)
    # at the experimental P -- which often extends well past 600 bar and
    # down to ~20 bar. Use BOTH so the model curve covers the same P range
    # as the data instead of clipping at the smooth-grid endpoints.
    model_rows = df_panel[df_panel["kind"].isin(("model", "exp"))]
    ms_levels = sorted(exp_rows["ms_mol_kg"].unique())
    if not ms_levels:
        return
    if color_for_ms is None:
        cmap_local = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(ms_levels), 2)))
        color_lookup = lambda k, ms: cmap_local[k]
    else:
        color_lookup = lambda k, ms: color_for_ms[ms]
    style_lookup = (lambda ms: "-") if style_for_ms is None else (lambda ms: style_for_ms[ms])
    authors = sorted(exp_rows["author"].unique())
    author_marker = {a: AUTHOR_MARKERS[i % len(AUTHOR_MARKERS)]
                     for i, a in enumerate(authors)}

    # y-unit transform: identity (molality) or CO2 mole fraction in per-mil.
    _y_of = _molality_to_permil if y_unit == "permil" else (lambda m: m)

    # Pre-clean model_rows ONCE so both the per-ms curves and the panel y-axis
    # range use the same artefact-free dataset. Order:
    #   (a) drop NaNs
    #   (b) drop seed-z degenerate spikes (same predicate main() applies to
    #       the AAD rows, so curve and number never disagree)
    #   (c) drop pure-water recipes mis-tagged with a brine salinity
    ION_COLS = ["m_Na", "m_Cl", "m_K", "m_Ca", "m_Mg", "m_SO4"]
    model_clean = drop_seed_artifacts(
        model_rows.dropna(subset=["mc_pred"])).copy()
    if len(model_clean):
        ion_total = model_clean[ION_COLS].abs().sum(axis=1)
        # Keep all rows with zero label (pure water) OR matching ions.
        model_clean = model_clean[(model_clean["ms_mol_kg"] == 0) |
                                  (ion_total > 1e-6)]

    for k, ms in enumerate(ms_levels):
        color = color_lookup(k, ms)
        # Model curve on dense grid (from pre-cleaned dataframe).
        mc_curve = (model_clean[model_clean["ms_mol_kg"] == ms]
                    .sort_values("P_bar"))
        if len(mc_curve):
            # Median-dedup by P: some experimental sources record the same
            # nominal m_s with slightly different ion compositions, so several
            # model curves can sit at the same nominal m_s after the recipe
            # loop in compute_singlesalt.
            mc_curve = (mc_curve.groupby("P_bar", as_index=False)["mc_pred"]
                                .median()
                                .sort_values("P_bar")
                                .reset_index(drop=True))
            ax.plot(mc_curve["P_bar"], _y_of(mc_curve["mc_pred"]),
                    linestyle=style_lookup(ms),
                    color=color, lw=2.0, label=f"{ms:g} m")
        # Experimental scatter, marker per author.
        sub_exp = exp_rows[exp_rows["ms_mol_kg"] == ms]
        for a in authors:
            asub = sub_exp[sub_exp["author"] == a]
            if not len(asub):
                continue
            ax.scatter(asub["P_bar"], _y_of(asub["mc_exp"]),
                       marker=author_marker[a],
                       color=color, edgecolors="k", linewidths=0.4,
                       s=36, zorder=3)

    ax.set_xlabel(r"$P$ [bar]")
    if y_unit == "permil":
        ax.set_ylabel(r"$x_{\mathrm{CO_2}}^{\mathrm{aq}}$ [‰]")
    else:
        ax.set_ylabel(r"$m_{\mathrm{CO_2}}$ [mol/(kg H$_2$O)]")
    ax.set_xlim(0, max(600, float(exp_rows["P_bar"].max()) * 1.05))
    # Cap y from the cleaned model + experimental data only. Use the model
    # max (not 95th-pctile -- the cache is small enough that any quantile
    # close to 1 still ignores legitimate end-of-range points).
    y_top = max(
        float(_y_of(exp_rows["mc_exp"]).max()) if len(exp_rows) else 0.0,
        float(_y_of(model_clean["mc_pred"]).max()) if len(model_clean) else 0.0,
    ) * 1.15
    ax.set_ylim(0, y_top if y_top > 0 else None)
    ax.grid(alpha=0.25)
    if show_legend and ms_levels:
        ax.legend(loc="lower right", fontsize=legend_fontsize, ncol=legend_ncol,
                  frameon=True, title=r"$m_s$", title_fontsize=legend_fontsize)
    bold_ticks(ax)


# ---------------------------------------------------------------------------
# Figure 1: 2-panel main
# ---------------------------------------------------------------------------
MAIN_PANELS = (("NaCl", 323), ("CaCl", 423))


def make_main(df: pd.DataFrame, out_path: Path) -> None:
    from matplotlib.lines import Line2D
    # Real LaTeX text rendering so \textperthousand renders as ‰ in the
    # y-axis label (the default matplotlib mathtext font renders ‰ as a
    # broken glyph).  Enabled locally so it doesn't affect other figures.
    # Use a font that renders ‰ cleanly (matplotlib's default DejaVu Sans
    # renders per-mille as a stacked "%o" that reads poorly at figure
    # scale).  Helvetica is available on macOS and on most journal-style
    # LaTeX setups; the math parts of the label still route through
    # matplotlib's mathtext, only surrounding text picks up the family.
    prev_family = plt.rcParams["font.family"]
    plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]

    # Union of molality values across BOTH main panels, so a given m_s gets
    # the same color and linestyle in panel (a) and panel (b).
    ms_canon = sorted({
        float(ms)
        for s, T in MAIN_PANELS
        for ms in df[(df["kind"] == "exp")
                     & (df["salt"] == s) & (df["T_K"] == T)]["ms_mol_kg"].unique()
    })
    cmap_vals = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(ms_canon), 2)))
    color_for_ms = {ms: cmap_vals[k] for k, ms in enumerate(ms_canon)}
    style_for_ms = {ms: LINESTYLES[k % len(LINESTYLES)]
                    for k, ms in enumerate(ms_canon)}

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_TWO_PANEL,
                             sharey=True)
    panel_data = []
    for ax, (salt, T) in zip(axes, MAIN_PANELS):
        sub = df[(df["salt"] == salt) & (df["T_K"] == T)]
        _plot_panel(ax, sub, show_legend=False,
                    color_for_ms=color_for_ms,
                    style_for_ms=style_for_ms,
                    y_unit="permil")
        # Salt/T annotation in bottom-right (matches the convention used
        # throughout the rest of the paper figures).
        ax.text(
            0.96, 0.06,
            f"{SALT_LABEL[salt]}, {T} K",
            transform=ax.transAxes, ha="right", va="bottom",
            fontweight="bold", fontsize=14,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2),
        )
        panel_data.append(ax.get_ylim()[1])
    # Override per-panel y_top with the union (sharey keeps axes locked).
    y_top = max(panel_data) if panel_data else None
    if y_top:
        axes[0].set_ylim(0, y_top)
    axes[1].set_ylabel("")
    # Panel labels (a)/(b) in the top-left, default position.
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")

    # Single shared top legend with inline ``m_s:`` prefix. Wrapping to two
    # rows keeps the legend roughly aligned with the underlying panel pair
    # instead of overflowing past their combined width.
    title_handle = Line2D([0], [0], linestyle="None", marker="",
                          label=r"$m_s$:")
    ms_handles = [Line2D([0], [0], color=color_for_ms[ms],
                         linestyle=style_for_ms[ms],
                         linewidth=2.4, label=f"{ms:g} m")
                  for ms in ms_canon]
    handles = [title_handle] + ms_handles
    legend_ncol = min(len(handles), 6)        # 2 rows when there are ~11 entries
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.005),
               ncol=legend_ncol, frameon=True, fontsize=11,
               handletextpad=0.4, columnspacing=1.0)
    legend_rows = (len(handles) + legend_ncol - 1) // legend_ncol
    top_rect = 1.0 - 0.05 * legend_rows - 0.02
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top_rect))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    # Restore the previous font family so SI figures render with the
    # paper's default matplotlib font.
    plt.rcParams["font.family"] = prev_family


# ---------------------------------------------------------------------------
# Figure 2: 5x3 SI grid (rows: salts, cols: 323 / 373 / 423 K)
# ---------------------------------------------------------------------------
def _make_SI_per_salt(df: pd.DataFrame, salt: str, out_path: Path) -> None:
    """One SI figure per salt: 1 row x 3 isotherms (323, 373, 423 K).

    Builds the UNION of molality levels across the 3 panels and assigns a
    consistent color per molality so the same m_s gets the same color in
    every panel. A single shared legend on top covers all molalities; per
    panel only those molalities that have data are actually drawn.
    """
    from matplotlib.lines import Line2D

    salt_df = df[df["salt"] == salt]
    ms_canon = sorted(
        salt_df[(salt_df["kind"] == "exp") & salt_df["T_K"].isin(TEMPS)]
            ["ms_mol_kg"].unique()
    )
    if not ms_canon:
        return
    cmap_vals = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(ms_canon), 2)))
    color_for_ms = {ms: cmap_vals[k] for k, ms in enumerate(ms_canon)}
    # Cycle linestyles in addition to color, so curves remain distinguishable
    # in B&W / for colorblind readers.
    style_for_ms = {ms: LINESTYLES[k % len(LINESTYLES)]
                    for k, ms in enumerate(ms_canon)}

    fig, axes = plt.subplots(1, len(TEMPS), figsize=(13.5, 4.5),
                             squeeze=False, sharey=True)
    for j, T in enumerate(TEMPS):
        ax = axes[0][j]
        sub = salt_df[salt_df["T_K"] == T]
        _plot_panel(ax, sub, show_legend=False,
                    color_for_ms=color_for_ms,
                    style_for_ms=style_for_ms, y_unit="permil")
        ax.text(
            0.96, 0.06,
            f"{SALT_LABEL[salt]}, {T} K",
            transform=ax.transAxes, ha="right", va="bottom",
            fontweight="bold", fontsize=12,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2),
        )
        panel_label(ax, chr(ord("a") + j))
        if j > 0:
            ax.set_ylabel("")

    # Single shared legend on top, listing the UNION of molality levels in
    # the canonical sorted order. Per panel, only the molalities actually
    # present get drawn -- but the legend shows what's available across all
    # 3 panels so the reader can map a color to an m_s anywhere. The
    # ``m_s:`` title is rendered as the first inline entry (zero-length
    # invisible handle) so it sits on the same row as the molalities and
    # saves vertical space.
    title_handle = Line2D([0], [0], linestyle="None", marker="",
                          label=r"$m_s$:")
    ms_handles = [Line2D([0], [0], color=color_for_ms[ms],
                         linestyle=style_for_ms[ms],
                         linewidth=2.4, label=f"{ms:g} m")
                  for ms in ms_canon]
    handles = [title_handle] + ms_handles
    # Try to keep everything on a single line; cap at 12 columns so very
    # crowded sets (e.g. CaCl2 with ~13 molalities) wrap to a second row.
    legend_ncol = min(len(handles), 12)
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.005),
               ncol=legend_ncol, frameon=True, fontsize=11,
               handletextpad=0.4, columnspacing=1.0)
    legend_rows = (len(handles) + legend_ncol - 1) // legend_ncol
    top_rect = 1.0 - 0.04 * legend_rows - 0.02
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top_rect))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_SI(df: pd.DataFrame, out_dir: Path) -> list:
    """Produce one SI figure per salt. Returns list of output paths."""
    out_paths = []
    for salt in SALTS:
        fname = f"co2_singlesalt_SI_{salt}.png"
        out_path = out_dir / fname
        _make_SI_per_salt(df, salt, out_path)
        out_paths.append(out_path)
        print(f"  {fname}")
    return out_paths


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cache_flag(p)
    args = p.parse_args()
    apply_style()
    df = load_or_compute("fig_co2_singlesalt", compute_singlesalt,
                         force=args.force)
    # Diagnostic: convergence reported on the experimental rows only
    # (the model-grid rows are not anchored to a measurement, so a NaN
    # there just means the model curve has a hole at that P).
    exp_df = df[df["kind"] == "exp"]
    n_total = len(exp_df)
    n_conv = int(exp_df["converged"].sum())
    print(f"  experimental converged: {n_conv}/{n_total} "
          f"({100*n_conv/max(n_total,1):.1f}%)")
    by = exp_df.groupby(["salt", "T_K"])["converged"].agg(["size", "sum"])
    by["fail"] = by["size"] - by["sum"]
    print(by)

    # Per-salt AAD across the full calibration set of converged experimental
    # rows. §3.4 of the paper cites five per-salt numbers here (NaCl, KCl,
    # CaCl2, MgCl2, Na2SO4). Range across salts feeds the abstract range.
    # Same degenerate-2-phase predicate the figure applies, so the cited
    # AAD and the plotted curve are computed over one row set. On a cache
    # written by this script it removes nothing -- _flash_one already
    # rejected those seeds -- and it only bites on a legacy cache.
    exp_ok = drop_seed_artifacts(
        exp_df[exp_df["converged"] & (exp_df["xc_W"] > 0)]).copy()
    exp_ok["rel_err"] = (exp_ok["xc_W_pred"] - exp_ok["xc_W"]).abs() / exp_ok["xc_W"]
    per_salt: dict[str, dict[str, float]] = {}
    print()
    print(f"  {'salt':>6}  {'N':>4}  {'AAD [%]':>10}")
    for salt, g in exp_ok.groupby("salt"):
        aad = round(100 * float(g["rel_err"].mean()), 2)
        per_salt[salt] = {"N": int(len(g)), "aad": aad}
        print(f"  {salt:>6}  {len(g):>4d}  {aad:>10.2f}")
    salt_aads = [v["aad"] for v in per_salt.values()]
    stats = {
        "N_converged": int(len(exp_ok)),
        "N_total": int(n_total),
        "per_salt": per_salt,
        "range_low":  round(min(salt_aads), 2) if salt_aads else None,
        "range_high": round(max(salt_aads), 2) if salt_aads else None,
    }
    save_stats("fig_co2_singlesalt", stats)

    print("\nMaking figures from cached predictions ...")
    make_main(df, PAPER_FIGURES / "co2_singlesalt_main.png")
    print(f"  co2_singlesalt_main.png")
    make_SI(df, PAPER_FIGURES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
