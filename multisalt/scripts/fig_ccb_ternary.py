#!/usr/bin/env python3
"""CH4-CO2-brine (CCB) solubility figure (SI only).

  - ccb_ternary_SI.png -- 6 panels, one per salt (NaCl, KCl, CaCl2, MgCl2,
                          Na2SO4, MgSO4), each panel showing CH4 and CO2
                          aqueous molality vs free-water gas-phase CO2
                          fraction Yc_gas, with curves at salinity
                          ms = 0, 1, 2, 3, 4 mol/kg.

Replicates the layout of the legacy ``MC_SOLUBILITY_*.png`` figures (one
per salt, from ``$ECPA_SOURCE_MATERIALS/EoS/CH4/CCB/eCPA_ELV.py``). The original
shows: solid = CH4, dashed = CO2; curve colour gradient = salinity.

The CCB plot is NOT a ternary triangle -- it is m_i (aqueous molality) vs
Yc_gas, despite the "ternary" label in the legacy filename. The system is
3-component (H2O + CH4 + CO2) plus brine, but the natural plot axes are
aqueous molality vs gas composition; the salinity dependence is the story.

Conditions: T = 323 K, P = 300 bar (matches the legacy figure).

The CCB system has no embedded experimental data in the upstream tree
(``build_ccb.py`` writes a schema-only parquet); this figure is model-only.

Compute path (slow, ~3-5 minutes): runs ``stability_flash`` over the
(salt, salinity, Yc) grid with the multi-gas adaptive z_gas retry pattern.

Plot path (fast, ~3 s): re-reads ``results_cache/fig_ccb_ternary_data.parquet``
and re-plots.

Usage:
    python3 scripts/fig_ccb_ternary.py            # use cache
    python3 scripts/fig_ccb_ternary.py --force    # recompute
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
)
from data_loader import (                                # noqa: E402
    load_experimental, load_or_compute, add_cache_flag,
)

Mw = 0.018  # kg/mol (water)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

T_K = 323
P_BAR = 300.0

SALTS = ("NaCl", "KCl", "CaCl", "MgCl", "NaSO", "MgSO")
SALT_LABELS = {
    "NaCl": "NaCl",
    "KCl":  "KCl",
    "CaCl": r"CaCl$_2$",
    "MgCl": r"MgCl$_2$",
    "NaSO": r"Na$_2$SO$_4$",
    "MgSO": r"MgSO$_4$",
}

# Stoichiometry: ions per formula unit, in (Na, Cl, K, Ca, Mg, SO4) order.
SALT_STOICH = {
    "NaCl": np.array([1, 1, 0, 0, 0, 0], dtype=float),
    "KCl":  np.array([0, 1, 1, 0, 0, 0], dtype=float),
    "CaCl": np.array([0, 2, 0, 1, 0, 0], dtype=float),
    "MgCl": np.array([0, 2, 0, 0, 1, 0], dtype=float),
    "NaSO": np.array([2, 0, 0, 0, 0, 1], dtype=float),
    "MgSO": np.array([0, 0, 0, 0, 1, 1], dtype=float),
}

MS_LIST = (0.0, 1.0, 2.0, 3.0, 4.0)   # mol-salt / kg-water
YC_GRID = np.linspace(0.99, 0.01, 30)

# Adaptive total-gas-feed ladder (mirrors fig_ccw_ternary.py).
Z_GAS_RETRIES = (0.20, 0.30, 0.10, 0.50, 0.05)

# Wong palette indices for the 5 salinities (kept consistent across panels).
MS_COLOUR_IDX = (1, 3, 2, 5, 6)


# ---------------------------------------------------------------------------
# Compute step (slow)
# ---------------------------------------------------------------------------
def _flash_one(T: float, P: float, Yc: float, mi: np.ndarray) -> dict:
    """Run stability_flash with the z_gas_total retry ladder; prefer interior beta."""
    out = {
        "x_H2O_aq_pred": np.nan,
        "x_CH4_aq_pred": np.nan,
        "x_CO2_aq_pred": np.nan,
        "y_H2O_gas_pred": np.nan,
        "beta": np.nan,
        "z_gas_total": np.nan,
        "tpd_min": np.nan,
        "converged": False,
    }
    best = None
    for k, zg in enumerate(Z_GAS_RETRIES):
        z_CO2 = zg * Yc
        z_CH4 = zg * (1.0 - Yc)
        z_H2O = 1.0 - z_CH4 - z_CO2
        z = np.array([z_H2O, z_CH4, z_CO2])
        try:
            res = stability_flash(float(T), float(P), z, mi)
        except Exception:
            continue
        # Fast-fail: first retry single-phase => skip remaining retries.
        if k == 0 and res.phases == 1 and res.tpd_min > -0.01:
            return out
        if res.phases == 2 and res.flash_result is not None and res.flash_result.converged:
            fr = res.flash_result
            beta = float(fr.beta)
            rec = {
                "x_H2O_aq_pred": float(fr.x_aq[0]),
                "x_CH4_aq_pred": float(fr.x_aq[1]),
                "x_CO2_aq_pred": float(fr.x_aq[2]),
                "y_H2O_gas_pred": float(fr.y_gas[0]),
                "beta": beta,
                "z_gas_total": float(zg),
                "tpd_min": float(res.tpd_min),
                "converged": True,
            }
            if 1e-3 < beta < 1.0 - 1e-3:
                out.update(rec)
                return out
            if best is None:
                best = rec
    if best is not None:
        out.update(best)
    return out


def compute_ccb_curves() -> pd.DataFrame:
    """Sweep stability_flash over (salt, ms, Yc) at fixed (T, P)."""
    rows: list[dict] = []
    total = len(SALTS) * len(MS_LIST) * len(YC_GRID)
    print(f"  flashing {total} (salt, ms, Yc) points at T={T_K}K, P={int(P_BAR)} bar ...",
          flush=True)
    t0 = time.perf_counter()
    for salt in SALTS:
        stoich = SALT_STOICH[salt]
        for ms in MS_LIST:
            mi = stoich * ms
            n_ok = 0
            for Yc in YC_GRID:
                rec = {
                    "salt": salt,
                    "ms_mol_kg": float(ms),
                    "Yc_gas": float(Yc),
                    "T_K": int(T_K),
                    "P_bar": float(P_BAR),
                }
                rec.update(_flash_one(T_K, P_BAR, Yc, mi))
                if rec["converged"]:
                    # Convert aqueous mole fractions to per-kg-water molality
                    # m_i = x_i^aq / (Mw * x_H2O^aq), matching the legacy figure.
                    xH = rec["x_H2O_aq_pred"]
                    if xH > 0:
                        rec["m_CH4_aq"] = rec["x_CH4_aq_pred"] / (Mw * xH)
                        rec["m_CO2_aq"] = rec["x_CO2_aq_pred"] / (Mw * xH)
                    else:
                        rec["m_CH4_aq"] = np.nan
                        rec["m_CO2_aq"] = np.nan
                    n_ok += 1
                else:
                    rec["m_CH4_aq"] = np.nan
                    rec["m_CO2_aq"] = np.nan
                rows.append(rec)
            print(f"    [{salt}, ms={ms} m] done ({n_ok}/{len(YC_GRID)} converged)  "
                  f"elapsed {time.perf_counter()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    print(f"  ...{time.perf_counter()-t0:.1f}s, "
          f"{int(df['converged'].sum())}/{len(df)} converged points")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_SI(model: pd.DataFrame, out_path: Path) -> None:
    """6-panel SI figure: one panel per salt, m_CH4 (solid) and m_CO2 (dashed)
    vs Yc_eq, with one curve per salinity.

    Layout: 3 rows x 2 cols, shared x and y across all panels; one common
    legend above the panel grid.
    """
    from matplotlib.lines import Line2D

    nrow, ncol = 3, 2
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(9.0, 9.5),
                             squeeze=False,
                             sharex=True, sharey=True)
    for k, salt in enumerate(SALTS):
        r, c = k // ncol, k % ncol
        ax = axes[r][c]
        # Keep only converged, strictly interior beta -- saturated beta = 0
        # or 1 means the feed is single-phase and the reported aqueous side
        # is not a real equilibrium split.
        sub = model[(model["salt"] == salt)
                    & model["converged"]
                    & (model["beta"] > 5e-3)
                    & (model["beta"] < 1.0 - 5e-3)].copy()
        # x-axis fix (same artefact as fig_ccw_ternary): ``Yc_gas`` is the
        # FEED gas-phase CO2 fraction. The z_gas_total retry ladder picks
        # different seeds per Yc point, which makes Yc_feed an unstable
        # parameterization -- adjacent points sample different feeds.
        # Recover the EQUILIBRIUM gas-phase Yc via mass balance:
        #   z_CO2 = Yc_gas * z_gas_total
        #   y_CO2 = (z_CO2 - x_CO2_aq * (1 - beta)) / beta
        # and similarly for CH4; then Yc_eq = y_CO2 / (y_CO2 + y_CH4).
        zg = sub["z_gas_total"].to_numpy()
        Yf = sub["Yc_gas"].to_numpy()
        beta = sub["beta"].to_numpy()
        x_CO2 = sub["x_CO2_aq_pred"].to_numpy()
        x_CH4 = sub["x_CH4_aq_pred"].to_numpy()
        y_CO2 = (Yf * zg - x_CO2 * (1.0 - beta)) / beta
        y_CH4 = ((1.0 - Yf) * zg - x_CH4 * (1.0 - beta)) / beta
        sub["Yc_eq"] = y_CO2 / (y_CO2 + y_CH4)
        for j, ms in enumerate(MS_LIST):
            sub_m = sub[sub["ms_mol_kg"] == ms].sort_values("Yc_eq")
            if sub_m.empty:
                continue
            color = WONG[MS_COLOUR_IDX[j % len(MS_COLOUR_IDX)]]
            ax.plot(sub_m["Yc_eq"], sub_m["m_CH4_aq"],
                    color=color, linestyle="-")
            ax.plot(sub_m["Yc_eq"], sub_m["m_CO2_aq"],
                    color=color, linestyle="--")
        if c == 0:
            ax.set_ylabel(r"$m_i$  [mol/kg]")
        if r == nrow - 1:
            ax.set_xlabel(r"$Y_{\mathrm{CO_2}}^{\mathrm{gas, fw}}$")
        ax.set_xlim(0, 1)
        # Salt-name tag, top-right
        ax.text(0.96, 0.94, SALT_LABELS[salt],
                transform=ax.transAxes, ha="right", va="top",
                fontweight="bold", fontsize=12,
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="none", pad=2))
        panel_label(ax, chr(ord("a") + k))
        bold_ticks(ax)
    # Blank any extra axes
    for j in range(len(SALTS), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    # Common legend above the grid, laid out as a 2-row x 6-column block:
    #   row 1: "CH4:"  +  one solid line per molality (color = molality)
    #   row 2: "CO2:"  +  one dashed line per molality (color = molality)
    # matplotlib's legend fills column-major by default, so to get the
    # rows we want we INTERLEAVE solid and dashed handles per column.
    species_tag = lambda txt: Line2D([0], [0], linestyle="None", marker="",
                                      label=txt)
    ms_solid = [Line2D([0], [0],
                       color=WONG[MS_COLOUR_IDX[j % len(MS_COLOUR_IDX)]],
                       linestyle="-", linewidth=2.4,
                       label=f"{int(ms)} m")
                for j, ms in enumerate(MS_LIST)]
    ms_dashed = [Line2D([0], [0],
                        color=WONG[MS_COLOUR_IDX[j % len(MS_COLOUR_IDX)]],
                        linestyle="--", linewidth=2.4,
                        label=f"{int(ms)} m")
                 for j, ms in enumerate(MS_LIST)]
    # Column-major order for ncol = len(MS_LIST) + 1, nrow = 2:
    # column 0: [CH4: , CO2:]; column k>0: [ms_solid[k-1], ms_dashed[k-1]].
    handles = [species_tag(r"$\mathbf{CH_4}$:"),
               species_tag(r"$\mathbf{CO_2}$:")]
    for s, d in zip(ms_solid, ms_dashed):
        handles += [s, d]
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.005),
               ncol=len(MS_LIST) + 1, frameon=True, fontsize=11,
               handletextpad=0.4, columnspacing=1.0,
               handlelength=2.4)
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
    model = load_or_compute("fig_ccb_ternary", compute_ccb_curves,
                            force=args.force)
    # Loaded only for API symmetry / future experimental overlays; CCB has
    # no experimental data shipped today.
    _ = load_experimental("ccb_ternary")

    n_ok = int(model["converged"].sum())
    n_tot = len(model)
    print(f"  cache: {n_ok}/{n_tot} converged points "
          f"({100*n_ok/n_tot:.1f}%)")

    print("Making CCB SI figure from cached predictions ...")
    make_SI(model, PAPER_FIGURES / "ccb_ternary_SI.png")
    print("  ccb_ternary_SI.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
