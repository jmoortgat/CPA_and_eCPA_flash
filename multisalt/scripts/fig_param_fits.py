#!/usr/bin/env python3
"""Parameter-fit figures: kij(T), CH4-ion DeltaU(T), CO2-ion DeltaU(T), Peneloux(T).

These four figures are analytic plots of the temperature dependence of the
production parameter set imported from ``ms_ecpa.parameters``. Each is a
single 1-panel figure (Paper II Figures S1-S4).

Outputs (in figures/, or $ECPA_PAPER_FIGURES):
    kij_h2o_ch4.png           CH4-H2O binary interaction parameter vs T
    delta_U_ch4_ions.png      DeltaU_{CH4-ion}/R vs T for the 6 ions
    delta_U_co2_ions.png      DeltaU_{CO2-ion}/R vs T for the 6 ions
    peneloux_ions.png         Peneloux ion shift Upsilon_i vs T for the 6 ions

There is no slow compute step here -- the curves are closed-form in T, so
the script just evaluates the formulas on a smooth grid and plots. A parquet
cache is still written for consistency with the other fig_*.py scripts.

Usage:
    python3 scripts/fig_param_fits.py            # use cache
    python3 scripts/fig_param_fits.py --force    # recompute (instant)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "scripts"))

from ms_ecpa.parameters import (                       # noqa: E402
    R, Tc1, Tc2, Tc3,
    Ak12, Bk12, Ck12,
    A_U2i, B_U2i,         # CH4-ion (six ions in order Na, Cl, K, Ca, Mg, SO4)
    A_U3i, B_U3i,         # CO2-ion
    A_Peni, B_Peni,       # Peneloux ion shifts
)
from paper_style import (                              # noqa: E402
    apply_style, bold_ticks, WONG, LINESTYLES, PAPER_FIGURES,
)
from data_loader import (                              # noqa: E402
    load_or_compute, add_cache_flag,
)

# Ion order matches the package convention (ions 4..9).
ION_LABELS = [r"Na$^{+}$", r"Cl$^{-}$", r"K$^{+}$",
              r"Ca$^{2+}$", r"Mg$^{2+}$", r"SO$_4^{2-}$"]
# Fixed colorblind-safe assignment shared across all three ion figures, so
# Na is the same color/style in delta_U_ch4 as in delta_U_co2 as in peneloux.
ION_COLORS = [WONG[5], WONG[6], WONG[7], WONG[3], WONG[2], WONG[1]]
ION_STYLES = [LINESTYLES[i % len(LINESTYLES)] for i in range(6)]

# Temperature grids (match the calibration windows of the underlying data).
T_KIJ      = np.linspace(273.0, 640.0, 400)   # CH4-H2O kij: water Tc upper bound
T_ION_CH4  = np.linspace(273.0, 550.0, 400)   # CH4-brine calibration window
T_ION_CO2  = np.linspace(273.0, 425.0, 400)   # CO2-brine calibration window
T_PENELOUX = np.linspace(273.0, 400.0, 400)   # density calibration window


# ---------------------------------------------------------------------------
# Closed-form parameter formulas (single source of truth: ms_ecpa.eos.py)
# ---------------------------------------------------------------------------
def kij_h2o_ch4(T: np.ndarray) -> np.ndarray:
    """k_wh(T) = A (T/Tc1)^2 + B (T/Tc1) + C   (ms_ecpa.eos:34)."""
    Tr = T / Tc1
    return Ak12 * Tr ** 2 + Bk12 * Tr + Ck12


def delta_U_ch4_ions(T: np.ndarray) -> np.ndarray:
    """Returns shape (n_T, 6): DeltaU_{CH4-ion}(T) = A_U2i (T/Tc2) + B_U2i  (J/mol)."""
    Tr = (T / Tc2)[:, None]
    return A_U2i[None, :] * Tr + B_U2i[None, :]


def delta_U_co2_ions(T: np.ndarray) -> np.ndarray:
    """Returns shape (n_T, 6): DeltaU_{CO2-ion}(T) = A_U3i (T/Tc3) + B_U3i  (J/mol)."""
    Tr = (T / Tc3)[:, None]
    return A_U3i[None, :] * Tr + B_U3i[None, :]


def peneloux_ions(T: np.ndarray) -> np.ndarray:
    """Returns shape (n_T, 6): Upsilon_i(T) = A_Peni (T/Tc1) + B_Peni   (cm^3/mol)."""
    Tr = (T / Tc1)[:, None]
    return A_Peni[None, :] * Tr + B_Peni[None, :]


# ---------------------------------------------------------------------------
# Compute step (instant) -- still goes through load_or_compute for consistency.
# ---------------------------------------------------------------------------
def compute_param_fits() -> pd.DataFrame:
    """Evaluate the four parameter-fit families on their respective T grids.

    Returns a long-format DataFrame with columns (kind, T_K, ion, value).
    Kinds: ``kij`` (no ion), ``deltaU_ch4``, ``deltaU_co2``, ``peneloux``.
    """
    rows = []
    # 1) CH4-H2O kij
    for T, k in zip(T_KIJ, kij_h2o_ch4(T_KIJ)):
        rows.append({"kind": "kij", "T_K": float(T), "ion": "", "value": float(k)})
    # 2) CH4-ion DeltaU/R (K)  -- store as DeltaU/R so the plot is unitless K
    U_h = delta_U_ch4_ions(T_ION_CH4) / R
    for j, ion in enumerate(ION_LABELS):
        for T, val in zip(T_ION_CH4, U_h[:, j]):
            rows.append({"kind": "deltaU_ch4", "T_K": float(T),
                         "ion": ion, "value": float(val)})
    # 3) CO2-ion DeltaU/R (K)
    U_c = delta_U_co2_ions(T_ION_CO2) / R
    for j, ion in enumerate(ION_LABELS):
        for T, val in zip(T_ION_CO2, U_c[:, j]):
            rows.append({"kind": "deltaU_co2", "T_K": float(T),
                         "ion": ion, "value": float(val)})
    # 4) Peneloux ion shift Upsilon_i (cm^3/mol)
    Pen = peneloux_ions(T_PENELOUX)
    for j, ion in enumerate(ION_LABELS):
        for T, val in zip(T_PENELOUX, Pen[:, j]):
            rows.append({"kind": "peneloux", "T_K": float(T),
                         "ion": ion, "value": float(val)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _new_axes(figsize=(7.5, 5.0)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# LaTeX-like math font (STIX) test settings, applied per-figure via
# plt.rc_context. STIX is widely supported by journal publishers and has solid
# bold math glyphs -- safer than mathtext.fontset = "cm" (Computer Modern),
# where bold math support is patchy. STIX renders perceptually smaller than
# DejaVu Sans, so every size is bumped up to compensate. To roll out across
# the rest of the paper, lift these into apply_style() in paper_style.py.
STIX_RC = {
    "mathtext.fontset": "stix",
    "font.family":      "STIXGeneral",
    "font.weight":      "bold",
    "axes.labelweight": "bold",
    "font.size":         19,
    "axes.labelsize":    22,
    "xtick.labelsize":   18,
    "ytick.labelsize":   18,
    "legend.fontsize":   17,
}


def make_kij_figure(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["kind"] == "kij"].sort_values("T_K")
    with plt.rc_context(STIX_RC):
        fig, ax = _new_axes()
        ax.plot(sub["T_K"], sub["value"], color=WONG[6], linestyle="-",
                linewidth=2.4)
        # Annotate the closed-form expression at the bottom-center; curve
        # peaks at T ~ 450 K above y = 0, so the bottom of the panel is
        # free of data here.
        ax.text(0.5, 0.06,
                (rf"$k_{{wh}} = {Ak12:+.2f}\,T_r^2 {Bk12:+.2f}\,T_r {Ck12:+.2f}$"
                 "\n" rf"with $T_r = T/T_{{c,\mathrm{{H_2O}}}}$"),
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=18, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="black",
                          pad=5))
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
        ax.set_xlabel(r"$T$ [K]")
        ax.set_ylabel(r"$k_{wh}$")
        ax.grid(alpha=0.25)
        bold_ticks(ax)
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)


def _ion_curve_figure(df: pd.DataFrame, kind: str, ylabel: str,
                      out: Path, *, stix: bool = False) -> None:
    rc = STIX_RC if stix else {}
    with plt.rc_context(rc):
        fig, ax = _new_axes(figsize=(7.5, 5.0))
        handles = []
        for j, ion in enumerate(ION_LABELS):
            sub = df[(df["kind"] == kind) & (df["ion"] == ion)].sort_values("T_K")
            line, = ax.plot(sub["T_K"], sub["value"],
                            color=ION_COLORS[j], linestyle=ION_STYLES[j],
                            linewidth=2.4, label=ion)
            handles.append(line)
        ax.set_xlabel(r"$T$ [K]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        # Legend OUTSIDE the axes, to the right of the panel (1 column x 6 rows).
        # bbox_inches="tight" on savefig expands the canvas to include it, so
        # it can never overlap any curve.
        ax.legend(handles=handles, ncol=1, frameon=True,
                  bbox_to_anchor=(1.02, 0.5), loc="center left",
                  borderaxespad=0.0)
        bold_ticks(ax)
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)


def make_delta_U_ch4_figure(df: pd.DataFrame, out: Path) -> None:
    _ion_curve_figure(df, "deltaU_ch4",
                      r"$\Delta U_{\mathrm{CH_4-ion}} / R$ [K]", out,
                      stix=True)


def make_delta_U_co2_figure(df: pd.DataFrame, out: Path) -> None:
    _ion_curve_figure(df, "deltaU_co2",
                      r"$\Delta U_{\mathrm{CO_2-ion}} / R$ [K]", out,
                      stix=True)


def make_peneloux_figure(df: pd.DataFrame, out: Path) -> None:
    _ion_curve_figure(df, "peneloux",
                      r"$\Upsilon_i$ [cm$^{3}$ mol$^{-1}$]", out,
                      stix=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cache_flag(p)
    args = p.parse_args()
    apply_style()
    df = load_or_compute("fig_param_fits", compute_param_fits,
                         force=args.force)
    print("Making 4 figures from cached parameter fits ...")
    make_kij_figure(df,           PAPER_FIGURES / "kij_h2o_ch4.png")
    print("  kij_h2o_ch4.png")
    make_delta_U_ch4_figure(df,   PAPER_FIGURES / "delta_U_ch4_ions.png")
    print("  delta_U_ch4_ions.png")
    make_delta_U_co2_figure(df,   PAPER_FIGURES / "delta_U_co2_ions.png")
    print("  delta_U_co2_ions.png")
    make_peneloux_figure(df,      PAPER_FIGURES / "peneloux_ions.png")
    print("  peneloux_ions.png")


if __name__ == "__main__":
    raise SystemExit(main())
