#!/usr/bin/env python3
"""CH4-CO2-H2O (CCW) ternary equilibrium figures (main paper + SI).

  - ccw_ternary_main.png  -- 2 panels at T = 344 K + 423 K (x_CH4_aq view)
  - ccw_ternary_SI.png    -- 9 panels: 3 temperatures x 3 views

Views (one per column in the SI):
    x_CH4_aq      aqueous CH4 mole fraction       vs Yc_gas
    x_CO2_aq      aqueous CO2 mole fraction       vs Yc_gas
    y_H2O_gas     gas-phase water mole fraction   vs Yc_gas

Compute path (slow, ~1-2 minutes per T): runs ``stability_flash`` over a
smooth grid of Yc_gas at each (T, P), with the adaptive z_CO2 retry pattern
borrowed from ``fig_flash_predictions.py``.

Plot path (fast, ~3 s): re-reads ``results_cache/fig_ccw_ternary_data.parquet``
and re-plots both figures.

Yc_gas is the *free-water* CO2 fraction in the gas phase:

    Yc_gas = y_CO2_gas / (y_CH4_gas + y_CO2_gas)

so the feed at fixed retry z_gas_total satisfies
``z_CO2 = z_gas_total * Yc_gas`` and ``z_CH4 = z_gas_total * (1 - Yc_gas)``.

Usage:
    python3 scripts/fig_ccw_ternary.py            # use cache
    python3 scripts/fig_ccw_ternary.py --force    # recompute
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

from ms_ecpa import stability_flash                      # noqa: E402
from paper_style import (                                # noqa: E402
    apply_style, bold_ticks, panel_label,
    PAPER_FIGURES, WONG, LINESTYLES,
    FIGSIZE_2COL, FIGSIZE_PANELS_5,
)
from data_loader import (                                # noqa: E402
    load_experimental, load_or_compute, add_cache_flag,
)

# ---------------------------------------------------------------------------
# Compute-side configuration
# ---------------------------------------------------------------------------

# Pressure grid per author / temperature -- matches the source script.
PRESSURES_BAR = {
    323: [20.0, 60.0, 100.0, 140.0, 180.0],     # ALGHAFRI
    344: [100.0, 200.0, 500.0, 750.0, 1000.0],  # DHIMA
    376: [105.0, 205.0, 303.0, 402.0, 500.0],   # QIN
    423: [20.0, 60.0, 100.0, 140.0, 180.0],     # ALGHAFRI
}

# Smooth Yc grid for the model curves.  Endpoints clipped from 1 / 0 to avoid
# degenerate single-component feeds.  30 points is a good compromise between
# resolution and compute time (3-4 minutes cold for the full grid).
YC_GRID = np.linspace(0.99, 0.01, 30)

# Adaptive total-gas-feed ladder (same idea as fig_flash_predictions.py).
# 0.20 is the most robust first choice (interior beta at most (T, P, Yc)
# points); 0.30 and 0.10 fan out from there; 0.05/0.50 are last-resort.
Z_GAS_RETRIES = (0.20, 0.30, 0.10, 0.50, 0.05)

# Beta-degeneracy thresholds, kept here so the retry ladder, the plot filter
# and the stdout diagnostics cannot drift apart.
#   ACCEPT -- the window the retry ladder demands before it stops early. A
#     beta outside it is a saturated split: the flash converged, and the
#     aqueous side is still the equilibrium composition, but no mass moved,
#     so the reported gas side is a copy of the feed. The ladder keeps
#     looking for a real split and only falls back on such a result if every
#     rung gives one.
#   PLOT -- the stricter window the drawn curves use. Within a few 1e-3 of
#     the phase boundary the gas-side composition is ill-conditioned enough
#     to kink the curve, so those points are held back even though the flash
#     converged. This is also what the "interior beta" counts below report,
#     so the printed coverage matches what is actually drawn.
BETA_INTERIOR_ACCEPT = 1e-3
BETA_INTERIOR_PLOT = 5e-3

# Authors -- used for plotting marker colour / shape.
T_AUTHOR = {323: "ALGHAFRI", 344: "DHIMA", 376: "QIN", 423: "ALGHAFRI"}

# Wong palette indices used to colour each pressure (kept consistent main / SI).
P_COLOUR_IDX = (1, 3, 2, 5, 6)


# ---------------------------------------------------------------------------
# Compute step (slow)
# ---------------------------------------------------------------------------
def _flash_one(T_K: float, P_bar: float, Yc: float) -> dict:
    """Run stability_flash with the z_gas_total retry ladder.

    Accepts the first retry where the flash converges *and* the vapor
    fraction ``beta`` is strictly interior (``BETA_INTERIOR_ACCEPT < beta <
    1 - BETA_INTERIOR_ACCEPT``): a saturated beta=0 or beta=1 means the feed
    sits entirely in a single phase, so the reported ``x_aq`` is the
    *equilibrium* aqueous composition but the gas-side ``y`` equals the feed
    (no mass actually transferred) -- we accept those interior-Yc points
    anyway because the aqueous side is still physically meaningful.

    ``converged`` therefore means "the flash converged", not "a real
    two-phase split was found"; ``beta`` is what separates the two, and the
    curves filter on it (``BETA_INTERIOR_PLOT``) rather than on ``converged``.

    Returns NaNs if every retry fails.
    """
    out = {
        "x_H2O_aq_pred": np.nan,
        "x_CH4_aq_pred": np.nan,
        "x_CO2_aq_pred": np.nan,
        "y_H2O_gas_pred": np.nan,
        "y_CH4_gas_pred": np.nan,
        "y_CO2_gas_pred": np.nan,
        "beta": np.nan,
        "z_gas_total": np.nan,
        "tpd_min": np.nan,
        "converged": False,
    }
    mi = np.zeros(6)
    best = None  # fallback: degenerate-beta but converged result
    for k, zg in enumerate(Z_GAS_RETRIES):
        z_CO2 = zg * Yc
        z_CH4 = zg * (1.0 - Yc)
        z_H2O = 1.0 - z_CH4 - z_CO2
        z = np.array([z_H2O, z_CH4, z_CO2])
        try:
            res = stability_flash(float(T_K), float(P_bar), z, mi)
        except Exception:
            continue
        # Fast-fail: if the first retry returns a clearly single-phase
        # tpd_min, the (T, P, Yc) point is genuinely single-phase --
        # don't waste 30+ seconds running the rest of the retry ladder.
        if (k == 0 and res.phases == 1
                and res.tpd_min > -0.01):
            return out
        if res.phases == 2 and res.flash_result is not None and res.flash_result.converged:
            fr = res.flash_result
            xH, xC, xCO2 = fr.x_aq
            yH, yC, yCO2 = fr.y_gas
            beta = float(fr.beta)
            rec = {
                "x_H2O_aq_pred": float(xH),
                "x_CH4_aq_pred": float(xC),
                "x_CO2_aq_pred": float(xCO2),
                "y_H2O_gas_pred": float(yH),
                "y_CH4_gas_pred": float(yC),
                "y_CO2_gas_pred": float(yCO2),
                "beta": beta,
                "z_gas_total": float(zg),
                "tpd_min": float(res.tpd_min),
                "converged": True,
            }
            # Strictly interior beta is preferred (real two-phase split).
            if BETA_INTERIOR_ACCEPT < beta < 1.0 - BETA_INTERIOR_ACCEPT:
                out.update(rec)
                return out
            # Otherwise stash and keep trying.
            if best is None:
                best = rec
    if best is not None:
        out.update(best)
    return out


def _beta_interior(beta: float) -> bool:
    """True when ``beta`` is far enough from 0/1 that the curves use the row.

    Rows where every retry failed carry ``beta = nan``; the comparisons are
    False for nan, so those are excluded without a separate test.
    """
    return bool(BETA_INTERIOR_PLOT < beta < 1.0 - BETA_INTERIOR_PLOT)


def compute_ccw_curves() -> pd.DataFrame:
    """Sweep stability_flash over the (T, P, Yc) grid for the smooth curves.

    No experimental points are flashed here; the experimental data are
    overlaid as scatter at plot time (already in ``data/ccw_ternary.parquet``).
    """
    rows: list[dict] = []
    t0 = time.perf_counter()
    total = sum(len(ps) for ps in PRESSURES_BAR.values()) * len(YC_GRID)
    print(f"  flashing {total} (T, P, Yc) points (z_gas retry ladder) ...",
          flush=True)
    for T_K, Ps in PRESSURES_BAR.items():
        for P_bar in Ps:
            for Yc in YC_GRID:
                rec = {"T_K": int(T_K), "P_bar": float(P_bar), "Yc_gas": float(Yc)}
                rec.update(_flash_one(T_K, P_bar, Yc))
                rows.append(rec)
            here = [r for r in rows
                    if r["T_K"] == T_K and r["P_bar"] == P_bar]
            n_conv = sum(1 for r in here if r["converged"])
            n_int = sum(1 for r in here if _beta_interior(r["beta"]))
            print(f"    [T={T_K} K, P={int(P_bar)} bar] done "
                  f"({n_conv} / {len(YC_GRID)} converged, "
                  f"{n_int} interior beta)  "
                  f"elapsed {time.perf_counter()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    n_int = int(df["beta"].map(_beta_interior).sum())
    print(f"  ...{time.perf_counter()-t0:.1f}s, "
          f"{int(df['converged'].sum())}/{len(df)} converged points, "
          f"{n_int} with interior beta (the rest converged onto a saturated "
          f"split whose gas side is a copy of the feed)")
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

VIEWS = {
    "x_CH4_aq": dict(
        model_col="x_CH4_aq_pred",
        exp_col="x_CH4_aq_exp",
        ylabel=r"$x^{\mathrm{aq}}_{\mathrm{CH_4}}$",
        marker="o",
    ),
    "x_CO2_aq": dict(
        model_col="x_CO2_aq_pred",
        exp_col="x_CO2_aq_exp",
        ylabel=r"$x^{\mathrm{aq}}_{\mathrm{CO_2}}$",
        marker="o",
    ),
    "y_H2O_gas": dict(
        model_col="y_H2O_gas_pred",
        exp_col="y_H2O_gas_exp",
        ylabel=r"$y^{\mathrm{gas}}_{\mathrm{H_2O}}$",
        marker="o",
    ),
}


def _plot_one_panel(ax, model: pd.DataFrame, exp: pd.DataFrame,
                    T_K: int, view: str, *,
                    show_legend: bool = True,
                    legend_loc: str = "upper left",
                    y_scale: float = 1.0,
                    y_unit_label: str = "") -> None:
    """Plot one T-view panel: model curves per pressure + experimental overlay."""
    cfg = VIEWS[view]
    Ps = PRESSURES_BAR[T_K]

    # Model curves -- one per pressure. Plot only points with strictly
    # interior beta; saturated beta=0 or beta=1 means the feed is single-phase
    # and the reported gas-side composition equals the feed (not a real flash).
    #
    # x-axis fix: ``Yc_gas`` in the cache is the FEED Yc (CO2 mole fraction
    # of the gas-only component of the feed). When the z_gas_total retry
    # ladder picks different seeds for different Yc points, the resulting
    # equilibrium gas-phase Yc bounces around even though Yc_feed marches
    # monotonically -- that's the zigzag. Recompute Yc as the EQUILIBRIUM
    # gas-phase ratio y_CO2 / (y_CO2 + y_CH4); this is what the experiments
    # report anyway (it's the measured gas composition at equilibrium).
    sub_model = model[model["T_K"] == T_K].copy()
    denom = sub_model["y_CO2_gas_pred"] + sub_model["y_CH4_gas_pred"]
    sub_model["Yc_eq"] = sub_model["y_CO2_gas_pred"] / denom
    for i, P in enumerate(Ps):
        sub_P = sub_model[(sub_model["P_bar"] == P)
                          & sub_model["converged"]
                          & (sub_model["beta"] > BETA_INTERIOR_PLOT)
                          & (sub_model["beta"] < 1.0 - BETA_INTERIOR_PLOT)]
        if sub_P.empty:
            continue
        sub_P = sub_P.sort_values("Yc_eq")
        color = WONG[P_COLOUR_IDX[i % len(P_COLOUR_IDX)]]
        ls = LINESTYLES[i % len(LINESTYLES)]
        ax.plot(sub_P["Yc_eq"], sub_P[cfg["model_col"]] * y_scale,
                color=color, linestyle=ls, label=f"{int(P)} bar")

    # Experimental overlay.
    sub_exp = exp[exp["T_K"] == T_K].copy()
    if not sub_exp.empty and cfg["exp_col"] in sub_exp.columns:
        for i, P in enumerate(Ps):
            sub_PE = sub_exp[sub_exp["P_bar"] == P]
            if sub_PE.empty:
                continue
            vals = sub_PE[cfg["exp_col"]].to_numpy()
            yc = sub_PE["Yc_gas"].to_numpy()
            mask = np.isfinite(vals)
            if not mask.any():
                continue
            ax.scatter(yc[mask], vals[mask] * y_scale,
                       marker=cfg["marker"], s=70,
                       color=WONG[P_COLOUR_IDX[i % len(P_COLOUR_IDX)]],
                       edgecolors="black", linewidths=0.5, zorder=3)

    ax.set_xlabel(r"$Y_{\mathrm{CO_2}}^{\mathrm{gas, fw}}$")
    ax.set_ylabel(cfg["ylabel"] + (f"  {y_unit_label}" if y_unit_label else ""))
    ax.set_xlim(0, 1)
    if show_legend:
        ax.legend(loc=legend_loc, fontsize=9, ncol=2, frameon=True)
    bold_ticks(ax)


# ---------------------------------------------------------------------------
# Main-paper figure: 2 panels at T = 344 K + 423 K (x_CH4_aq view).
# ---------------------------------------------------------------------------
def make_main(model: pd.DataFrame, exp: pd.DataFrame, out_path: Path) -> None:
    """2-panel main figure: x_CH4_aq vs Yc_eq at T = 344 K (DHIMA) + 423 K
    (ALGHAFRI), with shared y-axis, per-mil scaling, and per-column legends
    lifted above the panels (each author uses a different pressure list so
    the legends can't be merged)."""
    Ts = (344, 423)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    Y_SCALE = 1000.0
    for k, T_K in enumerate(Ts):
        ax = axes[k]
        _plot_one_panel(ax, model, exp, T_K, "x_CH4_aq",
                        show_legend=False,
                        y_scale=Y_SCALE, y_unit_label="[‰]")
        if k > 0:
            ax.set_ylabel("")
        panel_label(ax, chr(ord("a") + k))
        ax.text(0.96, 0.94, f"T = {T_K} K  ({T_AUTHOR[T_K]})",
                transform=ax.transAxes, ha="right", va="top",
                fontweight="bold", fontsize=12,
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="none", pad=2))

    # Reserve top margin for the per-column legends.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    for k, T_K in enumerate(Ts):
        h, l = axes[k].get_legend_handles_labels()
        bbox = axes[k].get_position()
        cx = bbox.x0 + bbox.width / 2.0
        fig.legend(handles=h, labels=l, loc="lower center",
                   bbox_to_anchor=(cx, 0.89),
                   ncol=len(h), frameon=True, fontsize=10,
                   handletextpad=0.4, columnspacing=1.0,
                   handlelength=2.0)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# SI figure: 3 views x 3 T (9 panels) -- "transposed" relative to the legacy
# layout so each ROW corresponds to one view (same y-axis quantity / range),
# letting us share y-axis labels/ticks across the row and share x across all
# rows (every panel has Y_CO2^{gas,fw} on the x). A single legend sits above
# the panels.
# ---------------------------------------------------------------------------
def make_SI(model: pd.DataFrame, exp: pd.DataFrame, out_path: Path) -> None:
    from matplotlib.lines import Line2D

    Ts = (344, 376, 423)
    views = ("x_CH4_aq", "x_CO2_aq", "y_H2O_gas")
    fig, axes = plt.subplots(len(views), len(Ts),
                             figsize=(12.0, 10.5), squeeze=False,
                             sharex=True, sharey="row")
    for r, v in enumerate(views):
        for c, T_K in enumerate(Ts):
            ax = axes[r][c]
            # Per-panel legends are suppressed; we place one legend per
            # column above the top row of the figure (each column's
            # author uses a different pressure list, so the legends
            # cannot be combined into one).
            _plot_one_panel(ax, model, exp, T_K, v, show_legend=False)
            if c > 0:
                ax.set_ylabel("")
            if r < len(views) - 1:
                ax.set_xlabel("")
            panel_label(ax, chr(ord("a") + r * len(Ts) + c))
            # T + author tag back in the top-right of each panel.
            ax.text(0.96, 0.94, f"T = {T_K} K  ({T_AUTHOR[T_K]})",
                    transform=ax.transAxes, ha="right", va="top",
                    fontweight="bold", fontsize=11,
                    bbox=dict(facecolor="white", alpha=0.85,
                              edgecolor="none", pad=2))

    # Reserve top margin for the three column legends.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    # Place one fig-level legend per top-row column, centred above that
    # column. Anchor by reading each top-row axes' bbox in figure
    # coordinates after the layout is finalised.
    for c, T_K in enumerate(Ts):
        Ps = PRESSURES_BAR[T_K]
        handles = [Line2D([0], [0],
                          color=WONG[P_COLOUR_IDX[i % len(P_COLOUR_IDX)]],
                          linestyle=LINESTYLES[i % len(LINESTYLES)],
                          linewidth=2.4,
                          label=f"{int(P)} bar")
                   for i, P in enumerate(Ps)]
        bbox = axes[0][c].get_position()
        cx = bbox.x0 + bbox.width / 2.0
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(cx, 0.95),
                   ncol=2, frameon=True, fontsize=10,
                   handletextpad=0.4, columnspacing=1.0,
                   handlelength=2.2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_cache_flag(p)
    args = p.parse_args()
    apply_style()
    model = load_or_compute("fig_ccw_ternary", compute_ccw_curves,
                            force=args.force)
    exp = load_experimental("ccw_ternary")

    n_ok = int(model["converged"].sum())
    n_tot = len(model)
    n_int = int(model["beta"].map(_beta_interior).sum())
    print(f"  cache: {n_ok}/{n_tot} converged points "
          f"({100*n_ok/n_tot:.1f}%), of which {n_int} have interior beta and "
          f"reach the drawn curves; experimental points: {len(exp)}")

    print("Making CCW ternary figures from cached predictions ...")
    make_main(model, exp, PAPER_FIGURES / "ccw_ternary_main.png")
    print("  ccw_ternary_main.png")
    make_SI(model, exp, PAPER_FIGURES / "ccw_ternary_SI.png")
    print("  ccw_ternary_SI.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
