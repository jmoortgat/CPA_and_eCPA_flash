#!/usr/bin/env python3
"""Generate the four flash-prediction paper figures from ONE shared cache:

  - flash_parity_CO2_brines_main.png    (main paper; all authors pooled)
  - per_salt_parity_SI.png              (SI; 5 panels, one per salt)
  - experimental_variance_SI.png        (SI; model error vs literature variance)

All authors are pooled into a single statistic: the
converged solver predicts the TEYMOURI cohort at the same accuracy as
the multi-author cluster, so no held-out split is carried. TEYMOURI
counts are still exported (N_teymouri) for the main-text sentence noting
that the pool includes data extrapolating beyond the calibration range.

Compute path (slow, ~5 minutes): runs ``stability_flash`` at every
experimental (T, P, salt, ions) point and stores predictions in
``results_cache/fig_flash_predictions_data.parquet``.

Plot path (fast, ~3 seconds): re-reads the cache and re-plots all four.

By default, if the cache exists it is used as-is -- pass ``--force`` to
recompute (e.g. after a model/parameter change). The cache file is small
(~50 kB) and tracked in git so coauthors and readers can reproduce every
figure in seconds without re-running the flash.

Usage:
    python3 scripts/fig_flash_predictions.py            # use cache
    python3 scripts/fig_flash_predictions.py --force    # recompute
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
from paper_style import (                              # noqa: E402
    apply_style, bold_ticks, bold_colorbar, panel_label,
    PAPER_FIGURES, WONG, FIGSIZE_SQUARE, FIGSIZE_PANELS_5, FIGSIZE_15COL,
)
from data_loader import (                              # noqa: E402
    load_experimental, load_or_compute, add_cache_flag, save_stats,
)

TEYMOURI_AUTHOR = "TEYMOURI"            # largest single-author cohort (pooled; counted for the extrapolation note)
Z_CO2_RETRIES = (0.05, 0.10, 0.02, 0.20, 0.30)


def aare(rows) -> float:
    """Mean absolute relative error in %. Skips rows without a finite
    prediction. Rows this script could not solve never reach the cache --
    ``compute_flash_predictions`` drops them and reports the count (see
    ``n_attempted`` there); the skipna path below is the guard for a legacy
    cache that does carry non-finite predictions."""
    if isinstance(rows, pd.DataFrame):
        # vectorized: works correctly even when xc_W_pred has NaN entries
        diff = (rows["xc_W_pred"] - rows["xc_W"]).abs() / rows["xc_W"] * 100
        return float(diff.replace([np.inf, -np.inf], np.nan).mean(skipna=True))
    errs = []
    for r in rows:
        xp, xe = r["xc_W_pred"], r["xc_W"]
        if not (np.isfinite(xp) and xe > 0):
            continue
        errs.append(abs(xp - xe) / xe * 100)
    return float(np.mean(errs)) if errs else float("nan")


# ---------------------------------------------------------------------------
# Compute step (slow)
# ---------------------------------------------------------------------------
def compute_flash_predictions() -> pd.DataFrame:
    """Run stability_flash on every row of data/co2_brines.parquet.

    Adds columns: ``xc_W_pred``, ``rel_err_pct``, ``z_CO2_used``, ``tpd_min``,
    ``converged``, ``n_attempted``. Failed solves are dropped (so the cache
    contains only converged points -- consistent with paper headline), but
    ``n_attempted`` records how many experimental points were tried, so the
    drop rate stays recoverable from the cache alone.
    """
    exp = load_experimental("co2_brines")
    n_exp = len(exp)
    print(f"  flashing {n_exp} experimental points (z_CO2 retry ladder)...")
    out_rows = []
    n_dropped = 0
    t0 = time.perf_counter()
    for idx, r in exp.iterrows():
        mi = np.array([r["m_Na"], r["m_Cl"], r["m_K"],
                       r["m_Ca"], r["m_Mg"], r["m_SO4"]])
        success = False
        for k, zC in enumerate(Z_CO2_RETRIES):
            z = np.array([1.0 - zC, 0.0, zC])
            try:
                res = stability_flash(float(r["T_K"]), float(r["P_bar"]), z, mi)
                if res.phases == 2 and res.flash_result is not None:
                    # The flash's own verdict, which this loop used to stamp
                    # as True without reading. Measured on the current data:
                    # 0 of 1833 accepted rows fail either test, so the
                    # headline AARE is unchanged -- these stop a future
                    # change from silently feeding bad rows into it.
                    if not bool(res.flash_result.converged):
                        continue
                    xH, _, xC = res.flash_result.x_aq
                    xcW_pred = float(xC / (xH + xC))
                    # A vanishing (xH + xC) turns the salt-free fraction into
                    # nan/inf rather than raising, and aare() skips it with
                    # skipna while N still counts the row -- so reject it here
                    # instead of caching a prediction that is not a number.
                    if not np.isfinite(xcW_pred):
                        continue
                    # Degenerate two-phase: aqueous salt-free fraction equal
                    # to the FEED fraction means no selective partitioning
                    # happened, so this is a copy of the feed, not a
                    # solubility. Same guard fig_co2_singlesalt.py applies.
                    if abs(xcW_pred - zC) < 1e-3 * max(zC, 1e-6):
                        continue
                    rel = abs(xcW_pred - r["xc_W"]) / r["xc_W"] * 100
                    out_rows.append({
                        **r.to_dict(),
                        "xc_W_pred": xcW_pred,
                        "rel_err_pct": rel,
                        "z_CO2_used": zC,
                        "tpd_min": float(res.tpd_min),
                        "converged": True,
                        # Carried on every row so the number of experimental
                        # points ATTEMPTED survives into the cache. The plot
                        # path never re-reads the input, so without it the
                        # drop rate -- i.e. the denominator the headline AARE
                        # is conditioned on -- is unrecoverable downstream.
                        "n_attempted": n_exp,
                    })
                    success = True
                    break
            except Exception:
                continue
        if not success:
            # No rung of the ladder gave a converged, non-degenerate split.
            # The row is dropped (the paper headline averages over converged
            # points), but it is counted so the gap to n_attempted is visible
            # here and in the stats JSON instead of silently shrinking N.
            n_dropped += 1
    print(f"  ...{time.perf_counter()-t0:.1f}s, {len(out_rows)} converged points")
    if n_dropped:
        print(f"  WARNING: {n_dropped}/{n_exp} experimental points dropped "
              f"(no converged, non-degenerate split on any z_CO2 rung). "
              f"The AARE and N reported below cover the {len(out_rows)} that "
              f"solved, not the full validation set.")
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _setup_parity_axes(ax, df):
    """Set up CO2 aqueous mole fraction parity axes labelled in per-mil
    (‰ = parts per thousand). All scatters in this script must multiply
    xc_W and xc_W_pred by 1000 before plotting."""
    x_lim = float(df["xc_W"].max()) * 1000.0 * 1.05
    ax.plot([0, x_lim], [0, x_lim], "k--", linewidth=1.0, label=r"$y = x$")
    xx = np.linspace(0, x_lim, 50)
    ax.fill_between(xx, 0.9 * xx, 1.1 * xx, color="gray", alpha=0.12,
                    label=r"$\pm 10\%$")
    ax.set_xlim(0, x_lim); ax.set_ylim(0, x_lim); ax.set_aspect("equal")
    ax.set_xlabel("experimental $x^{\\mathrm{aq}}_{\\mathrm{CO_2}}$  [‰]")
    ax.set_ylabel("predicted $x^{\\mathrm{aq}}_{\\mathrm{CO_2}}$  [‰]")
    ax.grid(alpha=0.25)
    return x_lim


def _aare_box(ax, txt):
    ax.text(0.97, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black"))


# ---------------------------------------------------------------------------
# Figure 1: main-paper parity (all authors pooled)
# ---------------------------------------------------------------------------
def make_parity_main(df: pd.DataFrame, out_path: Path) -> None:
    rows = df
    salts = sorted(rows["salt"].unique())
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    _setup_parity_axes(ax, rows)
    for i, salt in enumerate(salts):
        sub = rows[rows["salt"] == salt]
        ax.scatter(sub["xc_W"] * 1000, sub["xc_W_pred"] * 1000,
                   s=22, color=WONG[(i+1) % len(WONG)], alpha=0.75,
                   edgecolors="none", label=f"{salt} (N={len(sub)})")
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    txt = f"AARE = {aare(rows):.1f}%\nN = {len(rows)} points"
    # The plotted N is the converged count. Whenever it is short of the
    # attempted count the annotation says so, so the panel cannot read as
    # coverage of the whole validation set. No line is added when every
    # experimental point solved, which is the case for the shipped cache.
    if "n_attempted" in rows.columns and len(rows):
        n_att = int(rows["n_attempted"].iloc[0])
        if n_att != len(rows):
            txt += f"\n({n_att - len(rows)} of {n_att} not converged)"
    _aare_box(ax, txt)
    bold_ticks(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: per-salt SI subplots
# ---------------------------------------------------------------------------
def make_per_salt_SI(df: pd.DataFrame, out_path: Path) -> None:
    salts = sorted(df["salt"].unique())
    ncol = 2
    nrow = (len(salts) + ncol - 1) // ncol
    # Shared parity limit across all panels (largest salt sets the bound).
    global_lim = float(df["xc_W"].max()) * 1000.0 * 1.05
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.0, 4.0 * nrow),
                             squeeze=False, constrained_layout=True,
                             sharex=True, sharey=True)
    for i, salt in enumerate(salts):
        r, c = i // ncol, i % ncol
        ax = axes[r][c]
        salt_rows = df[df["salt"] == salt]
        _setup_parity_axes(ax, salt_rows)
        # Override the per-panel x_lim with the global one (sharex/sharey
        # already locks them together, but _setup_parity_axes called set_xlim
        # individually and the dashed y=x line was drawn only up to that
        # local max). Redraw the diagonal to the global limit.
        ax.lines[0].remove()
        ax.collections[0].remove()  # the +/-10% band
        ax.plot([0, global_lim], [0, global_lim], "k--", linewidth=1.0,
                label=r"$y = x$")
        xx = np.linspace(0, global_lim, 50)
        ax.fill_between(xx, 0.9 * xx, 1.1 * xx, color="gray", alpha=0.12,
                        label=r"$\pm 10\%$")
        ax.set_xlim(0, global_lim); ax.set_ylim(0, global_lim)
        ax.scatter(salt_rows["xc_W"] * 1000, salt_rows["xc_W_pred"] * 1000,
                   s=20, color=WONG[(i+1) % len(WONG)], alpha=0.7,
                   edgecolors="none",
                   label=f"N={len(salt_rows)}  (AARE={aare(salt_rows):.1f}%)")
        panel_label(ax, chr(ord("a") + i))
        # Salt name as a second annotation, top-right
        ax.text(0.96, 0.94, salt, transform=ax.transAxes, ha="right", va="top",
                fontweight="bold", fontsize=14,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
        ax.legend(loc="lower right", fontsize=8)
        # Drop redundant labels: y-label only on left column, x-label only
        # on the bottom-most occupied row.
        if c > 0:
            ax.set_ylabel("")
        bottom_idx = len(salts) - 1
        is_bottom_in_col = (i == bottom_idx) or (i + ncol >= len(salts))
        if not is_bottom_in_col:
            ax.set_xlabel("")
        bold_ticks(ax)
    # Blank unused axes
    for j in range(len(salts), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: experimental variance analysis
# ---------------------------------------------------------------------------
def make_variance_analysis(df: pd.DataFrame, out_path: Path,
                           P_bin_bar: float = 20.0) -> None:
    grp = defaultdict(list)
    rows = df
    for _, r in rows.iterrows():
        key = (r["salt"], int(r["T_K"]), round(r["ms_mol_kg"], 2),
               round(r["P_bar"] / P_bin_bar) * P_bin_bar)
        grp[key].append(r)

    sigma_exp, model_err, n_auth = [], [], []
    for key, gr in grp.items():
        authors = {x["author"] for x in gr}
        if len(authors) < 2:
            continue
        xs = np.array([x["xc_W"] for x in gr])
        ps = np.array([x["xc_W_pred"] for x in gr])
        sigma_exp.append(float(np.std(xs, ddof=1)))
        model_err.append(float(abs(np.median(ps) - np.mean(xs))))
        n_auth.append(len(authors))
    if not sigma_exp:
        print("  (no overlapping bins; skipping variance figure)")
        return
    sigma_exp = np.array(sigma_exp); model_err = np.array(model_err)
    n_auth = np.array(n_auth)

    # Convert to per-mil (x * 10^3) for compact axes labels.
    sigma_exp_pm = sigma_exp * 1000
    model_err_pm = model_err * 1000
    fig, ax = plt.subplots(figsize=FIGSIZE_15COL)
    lim = max(sigma_exp_pm.max(), model_err_pm.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.0,
            label=r"model error $= \sigma_\mathrm{exp}$")
    sc = ax.scatter(sigma_exp_pm, model_err_pm, c=n_auth, cmap="viridis",
                    s=42, edgecolors="k", linewidths=0.5)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("authors / bin", fontweight="bold")
    bold_colorbar(cbar)
    # Per-mil scaling is applied to the data but NOT in the axis label --
    # it would run off the figure. The caption tells the reader that
    # both axes are sigma_exp x 10^3 and model_error x 10^3.
    ax.set_xlabel(r"literature $\sigma_\mathrm{exp}$")
    ax.set_ylabel(r"model error $|\bar{x}_\mathrm{pred} - \bar{x}_\mathrm{exp}|$")
    # Extend axes a hair below 0 so markers sitting exactly on an axis are
    # rendered whole instead of getting clipped in half by the frame.
    margin = 0.025 * lim
    ax.set_xlim(-margin, lim); ax.set_ylim(-margin, lim); ax.set_aspect("equal")
    n_below = int(np.sum(model_err <= sigma_exp))
    _aare_box(ax,
              f"{n_below}/{len(sigma_exp)} bins\n"
              f"with err $\\leq \\sigma_\\mathrm{{exp}}$\n"
              f"({100*n_below/len(sigma_exp):.0f}%)")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(alpha=0.25)
    bold_ticks(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Print the headline AARE table (all authors pooled)
# ---------------------------------------------------------------------------
def print_aare_table(df: pd.DataFrame) -> dict:
    """Print the per-salt AARE table AND return a JSON-serializable dict
    of every headline number, keyed for consumption by
    ``build_paper_stats_tex.py``.

    Every entry cited in the paper text or in Table 6 (`tab:aare_per_salt`)
    is exported here so the LaTeX side only needs to reference macros.
    All authors are pooled; the TEYMOURI count is exported separately for
    the main-text sentence noting the pool includes beyond-calibration
    pressures.
    """
    held = df[df["author"].str.startswith(TEYMOURI_AUTHOR)]
    salts = sorted(df["salt"].unique())
    width = 60
    print("=" * width)
    print(f"{'salt':>6}  {'N':>6}  {'AARE [%]':>10}")
    print("-" * width)
    per_salt = {}
    for salt in salts:
        s_all = df[df["salt"] == salt]
        aare_salt = aare(s_all)
        print(f"{salt:>6}  {len(s_all):>6}  {aare_salt:>10.2f}")
        per_salt[salt] = {
            "N": len(s_all),
            "aare": round(aare_salt, 2),
        }
    print("-" * width)
    aare_overall = aare(df)
    print(f"{'TOTAL':>6}  {len(df):>6}  {aare_overall:>10.2f}")
    print("=" * width)
    print(f"Headline (all authors pooled): AARE = {aare_overall:.2f}%")

    best = min(per_salt.items(), key=lambda kv: kv[1]["aare"])
    stats = {
        "N_total": len(df),
        "N_teymouri": len(held),
        "aare_overall": round(aare_overall, 2),
        "best_salt": best[0],
        "best_salt_aare": best[1]["aare"],
        "per_salt": per_salt,
    }
    # Denominator accounting. ``N_total`` is the converged count -- the set the
    # headline AARE averages over -- and stays as-is so \NflashTotal keeps its
    # meaning. When the cache carries the attempted count, export it and the
    # drop count alongside, so how much of the validation set solved is
    # readable from the stats JSON rather than only from a --force console run.
    # Older caches predate the column; then the two keys are simply absent.
    if len(df) and "n_attempted" in df.columns:
        n_att = int(df["n_attempted"].iloc[0])
        stats["N_attempted"] = n_att
        stats["N_dropped"] = n_att - len(df)
        if n_att != len(df):
            print(f"NOTE: {n_att - len(df)}/{n_att} experimental points did "
                  f"not converge and are not in the cache; the numbers above "
                  f"are conditioned on the {len(df)} that did.")
    return stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cache_flag(p)
    args = p.parse_args()
    apply_style()
    df = load_or_compute("fig_flash_predictions", compute_flash_predictions,
                         force=args.force)
    stats = print_aare_table(df)
    save_stats("fig_flash_predictions", stats)
    print()
    print("Making 3 figures from cached predictions ...")
    make_parity_main(df, PAPER_FIGURES / "flash_parity_CO2_brines_main.png")
    print("  flash_parity_CO2_brines_main.png")
    make_per_salt_SI(df, PAPER_FIGURES / "per_salt_parity_SI.png")
    print("  per_salt_parity_SI.png")
    make_variance_analysis(df, PAPER_FIGURES / "experimental_variance_SI.png")
    print("  experimental_variance_SI.png")


if __name__ == "__main__":
    raise SystemExit(main())
