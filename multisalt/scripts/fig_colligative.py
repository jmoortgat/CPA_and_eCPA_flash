#!/usr/bin/env python3
"""Generate the osmotic-coefficient (colligative) paper figures (main + SI).

  - colligative_main.png  (main paper; (a) NaCl, (b) CaCl2 with the
                           printed Schlaikjer 2018 Ca2+ row overlaid)
  - colligative_SI.png    (SI; all five salts + legend panel)

Physics: phi_osm(m) at T = 298.15 K, P = 1 bar for the five paper salts,
computed from the total-mole water activity of the published eCPA
formulation (``scripts/total_mole_basis.py``; audit-validated against
independent re-implementations and against Schlaikjer 2018's own
published Na2SO4 curve).  The reference is the Pitzer ion-interaction
correlation with the verified Pitzer & Mayorga (1973) constants
(``scripts/pitzer_constants.py``).

The model curves use the CURRENT production parameter set in
``ms_ecpa/parameters.py`` -- nothing is hardcoded, so a parameter
adoption propagates on the next ``--force`` regeneration.  For the
CaCl2 panels only, an overlay curve is computed with the superseded
*printed* Schlaikjer 2018 Ca2+ row (sigma = 1.17 A, R_B = 1.73 A,
dU_ref = 595.83 R, printed Peneloux) via runtime rebinding
(``total_mole_basis.printed_ca_row``), so the provenance comparison
stays correct before and after the Ca2+ row adoption.

Pitzer curves are drawn solid within the molality range anchored to
experimental data by Pitzer & Mayorga 1973 (max-m: NaCl 6, KCl 4.8,
CaCl2 2.5, MgCl2 4.5, Na2SO4 4) and dotted beyond; gray shading marks
molalities beyond the Schlaikjer 2018 fit range for that salt.

Stats sidecar (``results_cache/fig_colligative_stats.json``): per-salt
RAD vs Pitzer in the Schlaikjer fit range and over the full plotted
range (each with the number of molalities that entered it), the
printed-Ca2+-row RAD and spot deviations, and the Debye-Hueckel-limit
check value.  Consumed by
``scripts/build_paper_stats_tex.py`` (macros \\radColl<salt> etc.).

Usage:
    python3 scripts/fig_colligative.py            # use cache
    python3 scripts/fig_colligative.py --force    # recompute
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

import pitzer_constants                                  # noqa: E402
from total_mole_basis import phi_osm_total, printed_ca_row  # noqa: E402
from paper_style import (                                # noqa: E402
    apply_style, bold_ticks, panel_label, PAPER_FIGURES, WONG,
)
from data_loader import (                                # noqa: E402
    load_or_compute, save_stats, add_cache_flag,
)

T_K = 298.15
P_Pa = 1.0e5
N_PTS = 25
M_DH = 1.0e-3        # Debye-Hueckel-limit check molality (NaCl)

# salt -> (m_max plotted, Schlaikjer-2018 fit-range m_max, display label)
SALTS = {
    "NaCl":   (6.0, 6.0, "NaCl(aq)"),
    "KCl":    (4.5, 6.0, "KCl(aq)"),
    "CaCl2":  (5.0, 2.0, "CaCl$_2$(aq)"),
    "MgCl2":  (4.5, 2.0, "MgCl$_2$(aq)"),
    "Na2SO4": (2.0, 2.0, "Na$_2$SO$_4$(aq)"),
}

# Molality up to which the Pitzer & Mayorga 1973 parameterization of each
# salt is anchored to experimental data (their Tables I/VI "max m").
PITZER_ANCHOR_MMAX = {
    "NaCl": 6.0, "KCl": 4.8, "CaCl2": 2.5, "MgCl2": 4.5, "Na2SO4": 4.0,
}

# Stats-sidecar salt keys (match build_paper_stats_tex.py macro suffixes).
SALT_MACRO_KEY = {
    "NaCl": "NaCl", "KCl": "KCl", "CaCl2": "CaCl",
    "MgCl2": "MgCl", "Na2SO4": "NaSO",
}

# per-salt in-panel label anchor (x, y, va) in axes coords, clear of curves
LABEL_POS = {
    "NaCl":   (0.96, 0.05, "bottom"),
    "KCl":    (0.96, 0.05, "bottom"),
    "CaCl2":  (0.96, 0.45, "center"),
    "MgCl2":  (0.96, 0.05, "bottom"),
    "Na2SO4": (0.96, 0.95, "top"),
}

# Spot molalities carried into the stats sidecar for the printed-row
# deviation narrative in the paper.
SPOT_M = (0.5, 1.0, 2.0)

# Curve styles (colorblind-safe; distinguishable in grayscale).
STYLE_PITZER = dict(color=WONG[0], linestyle="-")
STYLE_PITZER_EXT = dict(color=WONG[0], linestyle=":", linewidth=1.8)
STYLE_MODEL = dict(color=WONG[6], linestyle="--")
STYLE_PRINTED = dict(color=WONG[5], linestyle="-.")

LBL_PITZER = "Pitzer correlation (data-anchored)"
LBL_PITZER_EXT = "Pitzer correlation (extrapolated)"
LBL_MODEL = "eCPA, this work"
LBL_PRINTED = "printed Ca$^{2+}$ row"
LBL_SHADE = "beyond Schlaikjer 2018 fit range"


# ---------------------------------------------------------------------------
# Self-checks (parameter-stable: constants + the Na+/Cl- literature rows)
# ---------------------------------------------------------------------------
def run_checks() -> None:
    """Abort unless the Pitzer constants and the DH limit reproduce the
    2026-07-08 audit anchors. Both checks are invariant under the Ca2+ /
    Mg2+ parameter corrections (Na+ and Cl- rows are untouched)."""
    failures = []
    pz_anchor = {"NaCl": 0.9356, "KCl": 0.8980, "CaCl2": 1.0468,
                 "MgCl2": 1.1087, "Na2SO4": 0.6408}
    for salt, ref in pz_anchor.items():
        got = float(pitzer_constants.phi_osm(1.0, salt))
        # ``not ... <`` rather than ``>=``: every ordering comparison
        # against nan is False, so the ``>=`` form would pass a value the
        # solver never produced (phi_osm_total returns nan when the volume
        # root cannot be bracketed) straight through the gate.
        if not abs(got - ref) < 5e-4:
            failures.append(f"Pitzer anchor {salt}: {got:.4f} != {ref:.4f}")
    phi_dh = phi_osm_total("NaCl", M_DH, T_K, P_Pa)
    if not abs(phi_dh - 0.9884) < 2e-3:
        failures.append(f"DH limit: phi(0.001m NaCl) = {phi_dh:.4f}")
    if failures:
        raise SystemExit(f"fig_colligative self-checks FAILED: {failures}")
    print("  self-checks passed (Pitzer anchors + DH limit)")


# ---------------------------------------------------------------------------
# Compute step (slow): one row per (salt, row-variant, m)
# ---------------------------------------------------------------------------
def compute_phi_grid() -> pd.DataFrame:
    run_checks()
    rows = []
    t0 = time.perf_counter()

    def add(salt: str, variant: str, m_vals: np.ndarray) -> None:
        for m in m_vals:
            phi = phi_osm_total(salt, float(m), T_K, P_Pa)
            rows.append({
                "salt": salt, "variant": variant, "m": float(m),
                "phi_model": float(phi),
                "phi_pitzer": float(pitzer_constants.phi_osm(float(m), salt)),
            })

    for salt, (m_max, _m_fit, _lbl) in SALTS.items():
        m_vals = np.unique(np.concatenate(
            [np.linspace(0.05, m_max, N_PTS), np.asarray(SPOT_M)]))
        m_vals = m_vals[m_vals <= m_max + 1e-12]
        add(salt, "current", m_vals)
        n_fail = sum(1 for r in rows
                     if r["salt"] == salt and np.isnan(r["phi_model"]))
        if n_fail:
            print(f"  WARNING: {salt}: {n_fail} non-converged points")

    # Printed Schlaikjer 2018 Ca2+ row overlay (runtime rebinding).
    m_ca = np.unique(np.concatenate(
        [np.linspace(0.05, SALTS["CaCl2"][0], N_PTS), np.asarray(SPOT_M)]))
    with printed_ca_row():
        add("CaCl2", "printed", m_ca)

    # DH-limit point (NaCl, current parameters).
    add("NaCl", "dh", np.array([M_DH]))

    print(f"  ...{time.perf_counter() - t0:.1f}s for {len(rows)} points")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stats sidecar
# ---------------------------------------------------------------------------
def _rad_pct(sub: pd.DataFrame, m_hi: float) -> tuple[float, int, int]:
    """RAD [%] vs Pitzer for m <= m_hi, with (n_used, n_in_window).

    The mean is a nanmean, so molalities where phi_osm_total returned nan
    (volume root not bracketed) drop out of the average instead of out of
    the number: the counts are returned so the caller can refuse to
    publish a RAD taken over an incomplete window."""
    s = sub[sub["m"] <= m_hi + 1e-12]
    rel = (s["phi_model"] - s["phi_pitzer"]).abs() / s["phi_pitzer"].abs()
    n_tot = int(len(rel))
    n_used = int(np.isfinite(rel).sum())
    if n_used == 0:
        return float("nan"), 0, n_tot
    return float(100.0 * np.nanmean(rel)), n_used, n_tot


def build_stats(df: pd.DataFrame) -> dict:
    stats: dict = {
        "T_K": T_K, "P_Pa": P_Pa, "n_pts_per_curve": N_PTS,
        "basis": ("total-mole water activity of the published formulation; "
                  "target = Pitzer & Mayorga 1973 constants"),
        "per_salt": {},
    }
    # Windows whose RAD would be averaged over only part of its molality
    # range. The \radColl* macros carry no N, so a partial loss of
    # high-m points -- where the deviation from Pitzer is largest -- would
    # read in the paper as an improvement rather than as missing data.
    gaps: list[str] = []

    def _rad(sub: pd.DataFrame, m_hi: float, tag: str) -> tuple[float, int, int]:
        rad, n_used, n_tot = _rad_pct(sub, m_hi)
        if n_used < n_tot:
            gaps.append(f"{tag} (m <= {m_hi:g}): {n_tot - n_used} of "
                        f"{n_tot} points non-converged")
        return rad, n_used, n_tot

    for salt, (m_max, m_fit, _lbl) in SALTS.items():
        sub = df[(df["salt"] == salt) & (df["variant"] == "current")]
        key = SALT_MACRO_KEY[salt]
        rad_fit, n_fit, n_fit_tot = _rad(sub, min(m_fit, m_max), f"{salt} fit")
        rad_full, n_full, n_full_tot = _rad(sub, m_max, f"{salt} full")
        stats["per_salt"][key] = {
            "rad_fit": round(rad_fit, 2),
            "rad_full": round(rad_full, 2),
            "m_fit": min(m_fit, m_max), "m_max": m_max,
            # points entering each RAD / points in that window
            "n_rad_fit": n_fit, "n_rad_fit_total": n_fit_tot,
            "n_rad_full": n_full, "n_rad_full_total": n_full_tot,
            "n_nonconverged": int(sub["phi_model"].isna().sum()),
        }
        print(f"  {salt:7s} RAD(fit, m<={min(m_fit, m_max):g}) = "
              f"{stats['per_salt'][key]['rad_fit']:6.2f}%   "
              f"RAD(full, m<={m_max:g}) = "
              f"{stats['per_salt'][key]['rad_full']:6.2f}%")

    pr = df[(df["salt"] == "CaCl2") & (df["variant"] == "printed")]
    spot_dev = {}
    for m in SPOT_M:
        r = pr[np.isclose(pr["m"], m)]
        if len(r):
            dev = 100.0 * (r["phi_model"].iloc[0] - r["phi_pitzer"].iloc[0]) \
                / r["phi_pitzer"].iloc[0]
            # left out when non-finite: these go to the .tex pre-formatted,
            # and build_paper_stats_tex._new emits a str verbatim, so a
            # "nan" here would be typeset as "nan"; absent renders "---".
            if np.isfinite(dev):
                spot_dev[m] = f"{dev:.1f}"
    rad_p_fit, n_p_fit, n_p_fit_tot = _rad(
        pr, SALTS["CaCl2"][1], "CaCl2 printed row fit")
    rad_p_full, n_p_full, n_p_full_tot = _rad(
        pr, SALTS["CaCl2"][0], "CaCl2 printed row full")
    stats["cacl2_printed_row"] = {
        "rad_fit": round(rad_p_fit, 2),
        "rad_full": round(rad_p_full, 2),
        "n_rad_fit": n_p_fit, "n_rad_fit_total": n_p_fit_tot,
        "n_rad_full": n_p_full, "n_rad_full_total": n_p_full_tot,
        "dev_pct_m0p5": spot_dev.get(0.5),
        "dev_pct_m1": spot_dev.get(1.0),
        "dev_pct_m2": spot_dev.get(2.0),
    }
    print(f"  CaCl2 printed row: RAD(fit) = "
          f"{stats['cacl2_printed_row']['rad_fit']:.2f}%, spot devs "
          f"{spot_dev}")

    dh = df[(df["salt"] == "NaCl") & (df["variant"] == "dh")]
    if len(dh):
        phi_dh = float(dh["phi_model"].iloc[0])
        # None, not f"{nan:.4f}": _new emits a str verbatim and would put
        # a literal "nan" in \collPhiDHNaCl, while None renders "---".
        stats["phi_dh_nacl"] = f"{phi_dh:.4f}" if np.isfinite(phi_dh) else None
        stats["phi_dh_pitzer"] = f"{dh['phi_pitzer'].iloc[0]:.4f}"

    if gaps:
        raise SystemExit(
            "fig_colligative: RAD windows are incomplete, so the published "
            "averages would be taken over only the molalities that solved: "
            + "; ".join(gaps))
    return stats


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _plot_salt_panel(ax, df: pd.DataFrame, salt: str, *,
                     overlay_printed: bool = False) -> dict:
    """Draw one salt panel; return dict of legend handles by role."""
    m_max, m_fit, label = SALTS[salt]
    handles: dict = {}
    cur = df[(df["salt"] == salt) & (df["variant"] == "current")
             & (df["m"] >= 0.049)].sort_values("m")
    m = cur["m"].to_numpy()
    phi_p = cur["phi_pitzer"].to_numpy()
    phi_e = cur["phi_model"].to_numpy()

    # Pitzer: solid where data-anchored, dotted where extrapolated.
    m_anchor = PITZER_ANCHOR_MMAX[salt]
    in_anchor = m <= m_anchor + 1e-12
    handles["pitzer"], = ax.plot(m[in_anchor], phi_p[in_anchor],
                                 **STYLE_PITZER, label=LBL_PITZER)
    if (~in_anchor).any():
        m_ext = np.concatenate(([m_anchor], m[~in_anchor]))
        phi_ext = np.concatenate(
            ([float(pitzer_constants.phi_osm(m_anchor, salt))],
             phi_p[~in_anchor]))
        handles["pitzer_ext"], = ax.plot(m_ext, phi_ext, **STYLE_PITZER_EXT,
                                         label=LBL_PITZER_EXT)

    ok = np.isfinite(phi_e)
    handles["model"], = ax.plot(m[ok], phi_e[ok], **STYLE_MODEL,
                                label=LBL_MODEL)

    if overlay_printed:
        pr = df[(df["salt"] == salt) & (df["variant"] == "printed")
                & (df["m"] >= 0.049)].sort_values("m")
        okp = np.isfinite(pr["phi_model"])
        handles["printed"], = ax.plot(pr["m"][okp], pr["phi_model"][okp],
                                      **STYLE_PRINTED, label=LBL_PRINTED)

    if m_fit < m_max:      # beyond the Schlaikjer 2018 fit range
        ax.axvline(m_fit, color="0.45", linewidth=1.4, linestyle=":")
        handles["shade"] = ax.axvspan(m_fit, m_max, color="0.75",
                                      alpha=0.25, zorder=0, label=LBL_SHADE)

    ax.set_xlim(0, m_max)
    ax.set_xlabel(r"$m$ [mol/kg]")
    lx, ly, lva = LABEL_POS[salt]
    ax.text(lx, ly, label, transform=ax.transAxes,
            ha="right", va=lva, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
    ax.grid(alpha=0.25)
    return handles


# ---------------------------------------------------------------------------
# Figure 1: main paper -- (a) NaCl, (b) CaCl2 with printed-row overlay
# ---------------------------------------------------------------------------
def make_main_figure(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    h_a = _plot_salt_panel(axes[0], df, "NaCl")
    h_b = _plot_salt_panel(axes[1], df, "CaCl2", overlay_printed=True)
    for j, ax in enumerate(axes):
        panel_label(ax, chr(ord("a") + j))
        bold_ticks(ax)
    axes[0].set_ylabel(r"$\phi_{\mathrm{osm}}$")

    handles = [h_a["pitzer"], h_b["pitzer_ext"], h_a["model"],
               h_b["printed"], h_b["shade"]]
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=True, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: SI -- all five salts + legend cell
# ---------------------------------------------------------------------------
def make_SI_figure(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    axes = axes.ravel()
    legend_handles: dict = {}
    for i, salt in enumerate(SALTS):
        h = _plot_salt_panel(axes[i], df, salt,
                             overlay_printed=(salt == "CaCl2"))
        panel_label(axes[i], "abcde"[i], x=0.04, y=0.95)
        if i % 3 == 0:
            axes[i].set_ylabel(r"$\phi_{\mathrm{osm}}$")
        bold_ticks(axes[i])
        for k, v in h.items():
            legend_handles.setdefault(k, v)

    axes[5].axis("off")
    order = ["pitzer", "pitzer_ext", "model", "printed", "shade"]
    handles = [legend_handles[k] for k in order if k in legend_handles]
    labels = {
        "pitzer": LBL_PITZER,
        "pitzer_ext": LBL_PITZER_EXT,
        "model": LBL_MODEL,
        "printed": "Schlaikjer 2018 printed Ca$^{2+}$ row",
        "shade": LBL_SHADE,
    }
    axes[5].legend(handles=handles,
                   labels=[labels[k] for k in order if k in legend_handles],
                   loc="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cache_flag(p)
    args = p.parse_args()
    apply_style()
    df = load_or_compute("fig_colligative", compute_phi_grid,
                         force=args.force)
    stats = build_stats(df)
    save_stats("fig_colligative", stats)
    print("Making 2 figures from cached predictions ...")
    make_main_figure(df, PAPER_FIGURES / "colligative_main.png")
    print("  colligative_main.png")
    make_SI_figure(df, PAPER_FIGURES / "colligative_SI.png")
    print("  colligative_SI.png")


if __name__ == "__main__":
    raise SystemExit(main())
