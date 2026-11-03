#!/usr/bin/env python3
"""Generate the CH4 Henry's-constant SI figure from ONE shared cache.

  - henry_water_brine_SI.png : 2 subpanels:
        (a) H(T) for CH4 in pure water (model + 6 experimental sources)
        (b) H(T) for CH4 in NaCl brine at several molalities

Compute path (slow): for each (salt, ms, T) in a smooth grid, evaluate
``lnphi_aq`` for CH4 at low pressures and extrapolate H = lim_{P->0}
phi_CH4 * P via polyfit (matching the original ``eCPA_HENRY.py:425``
recipe).

Plot path (fast): re-reads the cache and overlays experimental scatter
from ``data/ch4_henry.parquet``.

Usage:
    python3 scripts/fig_ch4_henry.py            # use cache
    python3 scripts/fig_ch4_henry.py --force    # recompute
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
    FIGSIZE_2COL,
)
from data_loader import (                            # noqa: E402
    load_experimental, load_or_compute, add_cache_flag,
)

# ----------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------
# Solver domain caps at T < 533 K and P > 0; we cap at 525 K for safety.
# 30 points is enough for smooth visual curves and keeps the total
# (5 salinities x 30 T x 6 P) under a few minutes per cold run.
T_PURE = np.linspace(285, 525, 30)        # pure-water Henry curve
T_BRINE = np.linspace(285, 525, 30)       # brine Henry curve (same range)

# Source script uses P in [1, 5] bar to extrapolate H = lim_{P->0} phi*P.
# We extrapolate over a slightly wider range (and use solve_elv, not the
# bare aqueous lnphi kernel, because the latter loses convergence at very
# low P / very dilute CH4 -- see fig_ch4_henry.py development notes).
# H is computed as the *low-P limit* of f_CH4 / x_CH4 (Henry's law), which
# is independent of the choice of P in the dilute-gas regime.
N_PRESSURE = 6
P_GRID_BAR = np.linspace(2.0, 20.0, N_PRESSURE)

NACL_BRINE_CONCS = [0.0, 0.81, 1.95, 3.18, 4.70]   # match parquet authors


def _ions_for(salt: str, ms: float) -> np.ndarray:
    """Same stoichiometry helper as the brine-figure script."""
    out = np.zeros(6)
    if salt == "NaCl":
        out[0] = ms; out[1] = ms
    elif salt == "KCl":
        out[2] = ms; out[1] = ms
    elif salt == "H2O":
        pass
    else:
        raise ValueError(f"unknown Henry salt label: {salt}")
    return out


# ----------------------------------------------------------------------
# Henry computation
# ----------------------------------------------------------------------
def _henry_at_T(T: float, mi: np.ndarray) -> float:
    """Henry's constant of CH4 in the (water + ions) solvent at ``T``.

    Recipe (mirrors the original ``eCPA_HENRY.py:391-401`` extrapolation):
    solve full VLE at several low pressures with YC=0 (pure CH4 gas), then
    compute the local Henry proxy ``H_local = f_CH4 / x_CH4`` (gas fugacity
    divided by aqueous CH4 mole fraction). This is exactly Henry's law in
    the dilute-aqueous limit. To remove the residual P-dependence, fit
    ``H_local(P)`` to a line in P and take the intercept ``H(P->0)``.

    Returns H in GPa, or NaN if too few low-P points converged.
    """
    H_local = np.full(N_PRESSURE, np.nan)
    for i, P_bar in enumerate(P_GRID_BAR):
        try:
            r = solve_elv(float(T), float(P_bar), YC=0, mi=mi)
            if r.converged and r.x2w > 0:
                # f_ch4 is in bar; convert to Pa. x2w is CH4 mole fraction in brine.
                H_local[i] = r.f_ch4 * 1e5 / r.x2w
        except FileNotFoundError:
            # A missing warm-start table is a broken installation, not a state
            # that failed to converge, and it fails EVERY point identically.
            # It used to be swallowed here with everything else: all 1,080
            # solves raised, every H came back NaN, the run reported "done in
            # 0.5s", and the first visible symptom was a zero-size-array error
            # nine frames away in np.nanmax. Let it out.
            raise
        except Exception:
            # A genuine non-convergence at one pressure. The caller needs
            # three of six to fit the P -> 0 intercept, so one is survivable.
            pass
    mask = np.isfinite(H_local)
    if mask.sum() < 3:
        return float("nan")
    # Linear fit H_local(P) -> intercept at P=0 is the Henry constant.
    P_Pa = P_GRID_BAR[mask] * 1e5
    slope, intercept = np.polyfit(P_Pa, H_local[mask], 1)
    return float(intercept) * 1e-9   # GPa


def compute_ch4_henry() -> pd.DataFrame:
    """Compute H(T, m_salt) for pure water and NaCl brines."""
    rows = []
    t0 = time.perf_counter()

    # Pure water
    print("  pure water ...")
    for T in T_PURE:
        H = _henry_at_T(float(T), _ions_for("H2O", 0.0))
        rows.append({"salt": "H2O", "m_salt_mol_kg": 0.0,
                     "T_K": float(T), "H_GPa": H})

    # NaCl brine at several molalities (matching the experimental dataset)
    for ms in NACL_BRINE_CONCS:
        print(f"  NaCl ms={ms:.2f} ...   ({time.perf_counter()-t0:.1f}s)")
        for T in T_BRINE:
            H = _henry_at_T(float(T), _ions_for("NaCl", ms))
            rows.append({"salt": "NaCl", "m_salt_mol_kg": float(ms),
                         "T_K": float(T), "H_GPa": H})
    print(f"  done in {time.perf_counter()-t0:.1f}s")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def make_henry_SI(exp: pd.DataFrame, pred: pd.DataFrame,
                  out_path: Path) -> None:
    """Two-panel: (a) pure water + 6 EXP sources, (b) NaCl brine
    at various salinities + EXP overlays.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_2COL)

    # ---- Panel (a): pure water ---------------------------------------
    ax = axes[0]
    pw = pred[(pred["salt"] == "H2O") & np.isfinite(pred["H_GPa"])]
    ax.plot(pw["T_K"], pw["H_GPa"], color=WONG[0],
            linestyle="-", lw=2.4, label="eCPA")
    ew = exp[exp["salt"] == "H2O"]
    authors = sorted(ew["author"].unique())
    markers = ("o", "s", "^", "D", "v", ">", "<", "P", "X", "*")
    for i, auth in enumerate(authors):
        ea = ew[ew["author"] == auth]
        ax.scatter(ea["T_K"], ea["H_GPa"],
                   color=WONG[(i + 1) % len(WONG)],
                   marker=markers[i % len(markers)],
                   s=42, edgecolors="black", linewidths=0.6,
                   label=auth, zorder=3)
    ax.set_xlabel(r"$T$ [K]")
    ax.set_ylabel(r"$H$ [GPa]")
    ax.set_xlim(270, 530)
    ax.set_ylim(0, max(8, float(np.nanmax(pw["H_GPa"])) * 1.1))
    ax.grid(alpha=0.25)
    bold_ticks(ax)
    panel_label(ax, "a")
    ax.text(0.96, 0.06, "pure water",
            transform=ax.transAxes, ha="right", va="bottom",
            fontweight="bold", fontsize=12,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
    # Per-column legend lifted ABOVE the panel.
    handles_a, labels_a = ax.get_legend_handles_labels()

    # ---- Panel (b): NaCl brine ---------------------------------------
    ax = axes[1]
    salinities = sorted(set(pred[pred["salt"] == "NaCl"]["m_salt_mol_kg"]))
    en = exp[exp["salt"] == "NaCl"]
    # Map experimental ms to nearest model ms for color matching.
    exp_ms_uniq = sorted(en["m_salt_mol_kg"].unique())
    for i, ms in enumerate(salinities):
        color = WONG[(i + 1) % len(WONG)]
        ls = LINESTYLES[i % len(LINESTYLES)]
        sm = pred[(pred["salt"] == "NaCl") &
                  np.isclose(pred["m_salt_mol_kg"], ms) &
                  np.isfinite(pred["H_GPa"])].sort_values("T_K")
        ax.plot(sm["T_K"], sm["H_GPa"], color=color, linestyle=ls, lw=2.2,
                label=f"{ms:g} m")
        # Find nearest experimental salinity and overlay scatter.
        if exp_ms_uniq:
            ms_exp_match = min(exp_ms_uniq,
                               key=lambda x: abs(x - ms))
            if abs(ms_exp_match - ms) < 0.1:
                em = en[np.isclose(en["m_salt_mol_kg"], ms_exp_match)]
                ax.scatter(em["T_K"], em["H_GPa"],
                           color=color, marker="o",
                           s=44, edgecolors="black", linewidths=0.7,
                           zorder=3)
    ax.set_xlabel(r"$T$ [K]")
    ax.set_ylabel(r"$H$ [GPa]")
    ax.set_xlim(270, 530)
    # Brine H can climb above 30 GPa at the highest salinity (see the SUSAK
    # 4.27 m dataset). Use the data + model max to set the cap.
    pm = pred[pred["salt"] == "NaCl"]["H_GPa"]
    em = exp[exp["salt"] == "NaCl"]["H_GPa"]
    ymax = max(8.0, float(np.nanmax(pm)) * 1.1,
               float(em.max()) * 1.1 if len(em) else 0.0)
    ax.set_ylim(0, ymax)
    ax.grid(alpha=0.25)
    bold_ticks(ax)
    panel_label(ax, "b")
    ax.text(0.96, 0.06, "NaCl brine",
            transform=ax.transAxes, ha="right", va="bottom",
            fontweight="bold", fontsize=12,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
    handles_b, labels_b = ax.get_legend_handles_labels()

    # Place one fig-level legend per column, centred above that column.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    for col_axes, handles, labels, ncol, title in (
        (axes[0], handles_a, labels_a, 2, None),
        (axes[1], handles_b, labels_b, 2, r"$m_{\mathrm{salt}}$"),
    ):
        bbox = col_axes.get_position()
        cx = bbox.x0 + bbox.width / 2.0
        fig.legend(handles=handles, labels=labels,
                   loc="lower center",
                   bbox_to_anchor=(cx, 0.87),
                   ncol=ncol, frameon=True, fontsize=9,
                   title=title, title_fontsize=10,
                   handletextpad=0.4, columnspacing=1.0,
                   handlelength=2.0)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cache_flag(p)
    args = p.parse_args()
    apply_style()
    exp = load_experimental("ch4_henry")
    pred = load_or_compute("fig_ch4_henry", compute_ch4_henry, force=args.force)
    print()
    print("Making 1 figure from cached predictions ...")
    make_henry_SI(exp, pred, PAPER_FIGURES / "henry_water_brine_SI.png")
    print("  henry_water_brine_SI.png")


if __name__ == "__main__":
    raise SystemExit(main())
