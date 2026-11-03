#!/usr/bin/env python3
"""Generate CO2 mixed-salt brine paper figures from the Phase-1 cache.

  - co2_mixed_brines_main.png   1 panel: LIU(2011) B1 + WANG(2014) B1 overlaid
  - co2_mixed_brines_SI.png     ~19-panel grid: one panel per mix_id

Each panel shows CO2 molality :math:`m_{\mathrm{CO_2}}` (mol/kg H2O) vs
pressure, with one model curve per temperature (colour-coded blue->red) and
experimental scatter at matching markers. The mixed-brine recipe (Na, Cl,
K, Ca, Mg, SO4 molalities) is annotated in the panel via an ionic-strength
caption.

Compute path: runs ``stability_flash`` at every (T, P, recipe) point served
by ``load_experimental("co2_mixed_brines")`` -- GasBrineBench since
2026-09-18, the legacy parquet under ``ECPA_EXP_SOURCE=legacy`` -- plus a
smooth P grid per (recipe, T). ~1250 flashes, ~8 min.

Usage:
    python3 scripts/fig_co2_mixed_brines.py            # use cache
    python3 scripts/fig_co2_mixed_brines.py --force    # recompute
"""
from __future__ import annotations

import argparse

# Water molar mass, to convert CO2 molality (mol per kg H2O) to CO2 mole
# fraction in per-mil on the partitioning basis (ions excluded) --- the
# same convention as fig_co2_singlesalt.py:
#     x_CO2 = m_CO2 / (n_H2O + m_CO2),  n_H2O = 1000/M_H2O ~ 55.508 mol/kg
_MOL_H2O_PER_KG = 1000.0 / 18.015


def _molality_to_permil(m):
    """CO2 molality [mol/kg-H2O] -> aqueous CO2 mole fraction [permil]."""
    import numpy as _np
    m = _np.asarray(m, dtype=float)
    return 1000.0 * m / (_MOL_H2O_PER_KG + m)
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "scripts"))

from ms_ecpa import stability_flash                    # noqa: E402
from paper_style import (                              # noqa: E402
    apply_style, bold_ticks, panel_label,
    PAPER_FIGURES, WONG,
)
from data_loader import (                              # noqa: E402
    load_experimental, load_or_compute, add_cache_flag,
)

MW = 0.018  # kg/mol H2O
Z_CO2_RETRIES = (0.05, 0.10, 0.02, 0.20, 0.30)
ION_LABELS = (r"Na$^+$", r"Cl$^-$", r"K$^+$",
              r"Ca$^{2+}$", r"Mg$^{2+}$", r"SO$_4^{2-}$")
ION_COLS = ("m_Na", "m_Cl", "m_K", "m_Ca", "m_Mg", "m_SO4")
ZI = np.array([+1, -1, +1, +2, +2, -2], dtype=float)

# Author cohorts featured in the main 1-panel comparison.
MAIN_MIX_IDS = ("LIU_2011_B1", "WANG_2014_B1")

FIGSIZE_MAIN = (6.5, 5.0)


# ---------------------------------------------------------------------------
# Compute step
# ---------------------------------------------------------------------------
# Smooth model-curve pressure grid: 5 to 600 bar, 16 points (~40 bar
# spacing). Dense enough for the smooth mc(P) curvature, fast enough that
# the model-grid pass adds ~10 min for ~50 (mix_id, T) keys.
MODEL_P_GRID = np.linspace(5.0, 600.0, 16)


def _flash_one(T: float, P: float, mi: np.ndarray) -> tuple[float, float]:
    """One flash -> ``(salt-free xc_W, z_CO2 seed that produced it)``, or
    ``(NaN, NaN)``, with the z_CO2 retry ladder.

    Two independent rejections, both of which continue the ladder:

    * ``flash_result.converged`` false. ``stability_flash`` labels its
      terminal hard-failure return ``phases == 2`` and hands back the last
      SSI iterate, which carries a finite ``x_aq``; without reading the
      flag that mid-iteration composition is indistinguishable from a
      solved one. Same guard as fig_co2_singlesalt.py.
    * the degenerate-2-phase artefact (see
      [[feedback_degenerate_flash_artifact]]): ``phases == 2`` but the
      aqueous phase preserves the feed H2O:CO2 ratio (xc_W ≈ zC to ~5 sig
      fig), i.e. no selective partitioning occurred.

    The seed is returned so it can be stored alongside the prediction: the
    degenerate test is only meaningful against the seed actually in play,
    and a cache that does not record it forces the cache-side re-test to
    guess (see :func:`_is_seed_artifact`).
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
                if abs(xc_W - zC) < 1e-3 * max(zC, 1e-6):
                    continue
                return xc_W, zC
        except Exception:
            continue
    return float("nan"), float("nan")


def recipe_mix_ids(df: pd.DataFrame) -> pd.Series:
    """``{AUTHOR}_B{N}`` per distinct ion recipe within each author.

    The ``mix_id`` the loader supplies cannot be used as a recipe label:
    GasBrineBench carries one ``dataset_id`` -- the literal string
    "co2_mixed_brines" -- for all 314 mixed-brine rows, so grouping on it
    collapses every recipe at a given T onto whichever row comes first,
    and the ``*_B<N>`` panel selectors below match nothing at all. Deriving
    the label from (author, ion vector) makes it independent of how the
    upstream database names its datasets.

    Numbering follows first appearance within an author, the convention
    scripts/build_co2_mixed_brines.py used for the legacy parquet; on the
    314 rows the two sources share, the labels agree row for row, so
    LIU_2011_B1 and WANG_2014_B1 still name the same two brines.
    """
    seen: dict[str, dict[tuple, int]] = {}
    ids = []
    for _, r in df.iterrows():
        au = str(r["author"])
        key = tuple(round(float(r[c]), 6) for c in ION_COLS)
        per_author = seen.setdefault(au, {})
        if key not in per_author:
            per_author[key] = len(per_author) + 1
        ids.append(f"{au}_B{per_author[key]}")
    return pd.Series(ids, index=df.index, dtype=object)


def compute_mixed() -> pd.DataFrame:
    """Flash every experimental row AND a smooth P grid per (mix_id, T_K)
    combination. Returns a long-format DataFrame with a ``kind`` column
    distinguishing 'exp' rows from 'model' rows."""
    exp = load_experimental("co2_mixed_brines").copy()
    exp["mix_id"] = recipe_mix_ids(exp)
    print(f"  experimental rows: {len(exp)} "
          f"({exp['mix_id'].nunique()} recipes)")
    out_rows: list[dict] = []
    t0 = time.perf_counter()

    # Pass 1: experimental observations.
    for _, r in exp.iterrows():
        mi = np.array([float(r[c]) for c in ION_COLS])
        xcW, zC_used = _flash_one(r["T_K"], r["P_bar"], mi)
        mc_pred = (xcW / (1.0 - xcW) / MW
                   if np.isfinite(xcW) and 0.0 < xcW < 1.0 else np.nan)
        mc_exp = r["xc_W"] / (1.0 - r["xc_W"]) / MW
        out_rows.append({
            "kind":      "exp",
            "T_K":       int(r["T_K"]),
            "P_bar":     float(r["P_bar"]),
            **{c: float(r[c]) for c in ION_COLS},
            "mix_id":    r["mix_id"],
            "author":    r["author"],
            "xc_W":      float(r["xc_W"]),
            "mc_exp":    float(mc_exp),
            "xc_W_pred": xcW,
            "mc_pred":   mc_pred,
            "z_CO2_used": zC_used,
            # Finite only when a rung of the ladder cleared both of
            # _flash_one's tests, the solver's own convergence flag
            # included -- so this is convergence, not just finiteness.
            "converged": bool(np.isfinite(xcW)),
        })
    print(f"  experimental pass: "
          f"{sum(1 for x in out_rows if x['converged'])}/{len(out_rows)} "
          f"converged ({time.perf_counter()-t0:.1f} s)")

    # Pass 2: smooth model grid per (mix_id, T_K). Taking the first row's
    # ion vector per key is exact because mix_id is now one recipe by
    # construction -- every row sharing a key shares the ion vector.
    keys: dict[tuple, dict] = {}
    for _, r in exp.iterrows():
        key = (r["mix_id"], int(r["T_K"]))
        if key not in keys:
            keys[key] = {
                "mix_id": r["mix_id"],
                "T_K":    int(r["T_K"]),
                "author": r["author"],
                **{c: float(r[c]) for c in ION_COLS},
            }
    print(f"  model-grid pass: {len(keys)} (mix_id,T) x "
          f"{len(MODEL_P_GRID)} P -> {len(keys)*len(MODEL_P_GRID)} flashes")
    for meta in keys.values():
        mi = np.array([meta[c] for c in ION_COLS])
        for P in MODEL_P_GRID:
            xcW, zC_used = _flash_one(meta["T_K"], float(P), mi)
            mc_pred = (xcW / (1.0 - xcW) / MW
                       if np.isfinite(xcW) and 0.0 < xcW < 1.0 else np.nan)
            out_rows.append({
                "kind":      "model",
                "T_K":       meta["T_K"],
                "P_bar":     float(P),
                **{c: meta[c] for c in ION_COLS},
                "mix_id":    meta["mix_id"],
                "author":    meta["author"],
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
def _ion_caption(row, wrap_every: int = 3) -> str:
    """Build a short string like '0.30 Na  0.85 Cl  0.24 K  0.16 Ca'
    listing only the non-trivially-non-zero ions of the recipe (drop ions
    whose molality rounds to 0.00, so the caption isn't littered with
    ``0.00 K`` etc.). For long recipes, wrap every ``wrap_every`` ions
    onto a new line so the caption doesn't stretch past the panel width
    and squash the axes via constrained_layout."""
    pieces = []
    for c, lab in zip(ION_COLS, ION_LABELS):
        v = float(row[c])
        if v >= 0.005:                       # rounds to >= 0.01 -> keep
            pieces.append(f"{v:.2f} {lab}")
    lines = ["  ".join(pieces[i:i + wrap_every])
             for i in range(0, len(pieces), wrap_every)]
    return "\n".join(lines)


def _ionic_strength(row) -> float:
    mi = np.array([row[c] for c in ION_COLS], dtype=float)
    return float(0.5 * np.sum(mi * ZI**2))


def _T_colors(temps) -> dict:
    """blue (cool) -> red (hot) palette for the temperature axis."""
    if len(temps) == 1:
        return {temps[0]: (0.0, 0.0, 0.7)}
    t_min, t_max = min(temps), max(temps)
    span = t_max - t_min if t_max > t_min else 1.0
    return {T: plt.cm.coolwarm((T - t_min) / span) for T in temps}


def _is_seed_artifact(xcw: float, z_used: float = float("nan")) -> bool:
    """Cache-side counterpart of the degenerate-2-phase guard in _flash_one:
    xc_W equal to the z_CO2 seed that produced it means the aqueous phase
    kept the feed H2O:CO2 ratio.

    Mirrors the compute-side test exactly -- same RELATIVE tolerance, and,
    where the cache records the seed, the same single seed. An absolute
    window is the wrong shape here: 1e-4 is five times wider than the
    compute-side test at zC = 0.02, and aqueous CO2 mole fractions in this
    dataset run straight through 0.02 (experiment tops out near 0.024), so
    it deletes ordinary converged predictions wherever a curve crosses that
    level. Caches predating the ``z_CO2_used`` column carry no record of the
    seed, so every ladder entry has to be tested there.
    """
    if not np.isfinite(xcw):
        return False
    seeds = (z_used,) if np.isfinite(z_used) else Z_CO2_RETRIES
    return any(abs(xcw - z) < 1e-3 * max(z, 1e-6) for z in seeds)


def _clean_model_curve(df_mix: pd.DataFrame, T: float) -> pd.DataFrame:
    """Return a (P, mc_pred) DataFrame for the model curve at temperature T,
    filtered for the seed-z degenerate artefact and merged across "model" +
    "exp" rows so the curve covers the full experimental P range (not just
    the MODEL_P_GRID 5..600 bar window)."""
    rows = df_mix[df_mix["kind"].isin(("model", "exp"))
                  & (df_mix["T_K"] == T)].copy()
    rows = rows.dropna(subset=["mc_pred"])
    if len(rows) and "xc_W_pred" in rows.columns:
        if "z_CO2_used" in rows.columns:
            bad = [_is_seed_artifact(x, z) for x, z
                   in zip(rows["xc_W_pred"], rows["z_CO2_used"])]
        else:
            bad = [_is_seed_artifact(x) for x in rows["xc_W_pred"]]
        rows = rows[~np.asarray(bad)]
    if not len(rows):
        return rows
    return (rows.groupby("P_bar", as_index=False)["mc_pred"]
                .median()
                .sort_values("P_bar")
                .reset_index(drop=True))


def _plot_mix_panel(ax, df_mix: pd.DataFrame, *, show_recipe: bool = True,
                    show_legend: bool = True,
                    legend_fontsize: int = 9) -> None:
    """One panel for a single mix_id: mc vs P, one colour per T."""
    if not len(df_mix):
        ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes,
                ha="center", va="center", fontsize=12)
        return
    exp_rows = df_mix[df_mix["kind"] == "exp"]
    temps = sorted(exp_rows["T_K"].unique())
    Tcol = _T_colors(temps)
    panel_pred_max = 0.0
    for T in temps:
        m_T = _clean_model_curve(df_mix, T)
        if len(m_T):
            ax.plot(m_T["P_bar"], _molality_to_permil(m_T["mc_pred"]), "-",
                    color=Tcol[T], lw=2.0, label=f"{T} K")
            panel_pred_max = max(panel_pred_max,
                                 float(_molality_to_permil(m_T["mc_pred"]).max()))
        # Experimental scatter.
        e_T = exp_rows[exp_rows["T_K"] == T].sort_values("P_bar")
        ax.scatter(e_T["P_bar"], _molality_to_permil(e_T["mc_exp"]),
                   color=Tcol[T], edgecolors="k", linewidths=0.4,
                   s=42, marker="o", zorder=3)

    ax.set_xlabel(r"$P$ [bar]")
    ax.set_ylabel(r"$x_{\mathrm{CO_2}}^{\mathrm{aq}}$ [‰]")
    ax.set_xlim(0, max(600, float(exp_rows["P_bar"].max()) * 1.05))
    # Cap y from cleaned data only (the legacy cache still contains spikes,
    # so an uncleaned quantile would inflate the range).
    y_top = max(
        float(_molality_to_permil(exp_rows["mc_exp"]).max())
        if len(exp_rows) else 0.0,
        panel_pred_max,
    ) * 1.15
    ax.set_ylim(0, y_top if y_top > 0 else None)
    ax.grid(alpha=0.25)
    if show_recipe:
        first = exp_rows.iloc[0]
        Ic = _ionic_strength(first)
        caption = (f"{first['mix_id']}  "
                   r"($I_c$ = " + f"{Ic:.2f} m)\n"
                   + _ion_caption(first))
        ax.text(0.5, 1.02, caption, transform=ax.transAxes,
                ha="center", va="bottom",
                fontweight="bold", fontsize=9)
    if show_legend:
        ax.legend(loc="lower right", fontsize=legend_fontsize, frameon=True)
    bold_ticks(ax)


# ---------------------------------------------------------------------------
# Figure 1: main (LIU_2011_B1 + WANG_2014_B1 overlaid in one axes)
# ---------------------------------------------------------------------------
def make_main(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_MAIN, constrained_layout=True)
    cohort_colors = {MAIN_MIX_IDS[0]: WONG[5],  # blue
                     MAIN_MIX_IDS[1]: WONG[6]}  # vermillion
    markers = ("o", "s", "^", "v", "D", "P")
    for cohort in MAIN_MIX_IDS:
        sub = df[df["mix_id"] == cohort]
        if not len(sub):
            # Say so: an empty cohort otherwise saves a blank axes over the
            # previous build, and the only trace is matplotlib's "no artists
            # with labels" warning buried in a slurm log.
            print(f"  [warn] main panel: no rows for mix_id {cohort!r}")
            continue
        col = cohort_colors[cohort]
        exp_rows = sub[sub["kind"] == "exp"]
        temps = sorted(exp_rows["T_K"].unique())
        ls = "-" if cohort == MAIN_MIX_IDS[0] else "--"
        for i, T in enumerate(temps):
            m_T = _clean_model_curve(sub, T)
            if len(m_T):
                ax.plot(m_T["P_bar"], _molality_to_permil(m_T["mc_pred"]),
                        ls, color=col, lw=2.0,
                        alpha=0.55 + 0.45 * i / max(len(temps) - 1, 1))
            # Experimental scatter.
            e_T = exp_rows[exp_rows["T_K"] == T].sort_values("P_bar")
            ax.scatter(e_T["P_bar"], _molality_to_permil(e_T["mc_exp"]),
                       marker=markers[i % len(markers)],
                       color=col, edgecolors="k", linewidths=0.4,
                       s=48, zorder=3,
                       label=f"{cohort.replace('_', ' ')}, {T} K")

    ax.set_xlabel(r"$P$ [bar]")
    ax.set_ylabel(r"$x_{\mathrm{CO_2}}^{\mathrm{aq}}$ [‰]")
    ax.set_xlim(0, 250)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, ncol=2, frameon=True)
    bold_ticks(ax)

    # Recipe annotations were previously rendered in-panel here (small
    # colour-matched text blocks, upper-left corner). Moved to the LaTeX
    # caption instead (Paper II Figure 6) to keep the figure itself
    # to just the data; if the underlying recipes ever change, regenerate
    # the caption text with the same _ion_caption()/_ionic_strength()
    # helpers used above rather than hand-editing the LaTeX.

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: SI grid -- one panel per mix_id
# ---------------------------------------------------------------------------
def _make_SI_subset(df: pd.DataFrame, mix_ids: list, ncols: int,
                    out_path: Path) -> None:
    """Render ``mix_ids`` in a grid (at most ``ncols`` wide), shared y-axis,
    panel labels in the top-left, per-panel temperature legend."""
    if not mix_ids:
        return
    n = len(mix_ids)
    nrows = (n + ncols - 1) // ncols

    # Global y-cap from all cleaned predictions + experiments in the
    # subset, in the plotted (per-mil mole fraction) unit.
    panel_dfs = [df[df["mix_id"] == m] for m in mix_ids]
    y_candidates = []
    for sub in panel_dfs:
        exp_max = sub[sub["kind"] == "exp"]["mc_exp"].max()
        if pd.notna(exp_max):
            y_candidates.append(float(_molality_to_permil(exp_max)))
        for T in sub[sub["kind"] == "exp"]["T_K"].unique():
            m_T = _clean_model_curve(sub, T)
            if len(m_T):
                y_candidates.append(
                    float(_molality_to_permil(m_T["mc_pred"]).max()))
    y_top = (max(y_candidates) * 1.15) if y_candidates else None

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.0 * ncols, 3.6 * nrows),
                             squeeze=False, constrained_layout=True,
                             sharey=True)
    letter = iter([chr(ord("a") + k) for k in range(nrows * ncols)])
    for i, sub in enumerate(panel_dfs):
        ax = axes[i // ncols][i % ncols]
        _plot_mix_panel(ax, sub, show_recipe=True, show_legend=True,
                        legend_fontsize=8)
        # Panel letter top-left (default position of panel_label).
        panel_label(ax, next(letter))
    # Apply the global y-cap (sharey already locks them; override).
    if y_top:
        axes[0][0].set_ylim(0, y_top)
    # Y-label only on the leftmost column.
    for r in range(nrows):
        for c in range(ncols):
            if c > 0:
                axes[r][c].set_ylabel("")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# Split plan: group by reference/author. Each group becomes one SI figure
# with at most ``ncols`` columns. Singletons / few-panel authors are
# collected into an "other" figure.
SI_SPLIT_PLAN = [
    # (suffix, mix_id_prefixes, ncols)
    ("LIU_2011",    ("LIU_2011",),    3),
    ("WANG_2014",   ("WANG_2014",),   2),
    ("POULANI_2019", ("POULANI_2019",), 2),
    ("ZHAOc_2015",  ("ZHAOc_2015",),  2),
    ("other",       ("ELMAGHRABY_2012", "LI_2004", "PORTIER_2005",
                     "TEYMOURI_2017", "TONG_2013"), 3),
]


def make_SI(df: pd.DataFrame, out_dir: Path) -> list:
    """Produce one SI figure per author group (LIU 2011, WANG 2014,
    Poulain 2019, Zhao 2015, and an 'other' figure for the 5 singletons)."""
    out_paths = []
    for suffix, prefixes, ncols in SI_SPLIT_PLAN:
        mix_ids = [m for m in sorted(df["mix_id"].unique(), key=_mix_sort_key)
                   if any(m.startswith(p + "_B") for p in prefixes)]
        if not mix_ids:
            # Say so: a group that matches nothing simply leaves the figure
            # from the previous build on disk, and the paper then compiles
            # against a stale PNG with nothing printed and exit code 0.
            print(f"  [warn] SI group {suffix!r}: no mix_id starts with "
                  f"any of {prefixes}")
            continue
        fname = f"co2_mixed_brines_SI_{suffix}.png"
        out_path = out_dir / fname
        _make_SI_subset(df, mix_ids, ncols, out_path)
        out_paths.append(out_path)
        print(f"  {fname}  ({len(mix_ids)} panels)")
    return out_paths


def _mix_sort_key(mid: str):
    """Group by author then by 'B<N>' index numerically (so LIU_2011_B10
    sorts after LIU_2011_B2). Currently we top out at B6 but keep the
    parser robust."""
    parts = mid.rsplit("_B", 1)
    author = parts[0]
    try:
        n = int(parts[1])
    except (IndexError, ValueError):
        n = 0
    return (author, n)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cache_flag(p)
    args = p.parse_args()
    apply_style()
    df = load_or_compute("fig_co2_mixed_brines", compute_mixed,
                         force=args.force)
    # Convergence reported on the experimental rows only.
    exp_df = df[df["kind"] == "exp"]
    # A cache written before mix_id was derived from the recipe keys its
    # model curves on the collapsed upstream id, so those curves belong to
    # the wrong ion vectors. Relabelling cannot repair that; only a
    # recompute can, so fail loudly rather than plot them.
    if (exp_df["mix_id"].values != recipe_mix_ids(exp_df).values).any():
        print("  ERROR: cached mix_id does not match the ion recipes "
              "(stale cache). Rerun with --force.")
        return 1
    n_total = len(exp_df)
    n_conv = int(exp_df["converged"].sum())
    print(f"  experimental converged: {n_conv}/{n_total} "
          f"({100*n_conv/max(n_total,1):.1f}%)")
    print(f"  mix_ids:  {exp_df['mix_id'].nunique()}")
    print(exp_df.groupby("mix_id").size().to_string())

    print("\nMaking figures from cached predictions ...")
    make_main(df, PAPER_FIGURES / "co2_mixed_brines_main.png")
    print("  co2_mixed_brines_main.png")
    make_SI(df, PAPER_FIGURES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
