#!/usr/bin/env python3
"""Generate the brine-density paper figures (main + SI) from one shared cache.

  - brine_density_main.png   (main paper; NaCl at T = 298, 348, 398 K)
  - brine_density_SI.png     (SI; KCl, CaCl2, MgCl2, Na2SO4 x T = 298, 348, 398 K)

Compute path (slow, ~10 s): runs the aqueous-phase eCPA solver on each
experimental (salt, T, m_salt, P) point and stores the model prediction
in ``results_cache/fig_brine_density_data.parquet``.

Plot path (fast, ~3 s): re-reads the cache and re-plots both figures.

Each panel: x = P [bar], y = rho [kg/m^3], with a model line + experimental
scatter for each available molality. This mirrors the layout of the original
``OPT_PENELOUX.py`` density figures of Coelho et al.

Usage:
    python3 scripts/fig_brine_density.py            # use cache
    python3 scripts/fig_brine_density.py --force    # recompute
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

# Internal helpers (not re-exported by ms_ecpa.__init__ -- we use them to get
# Zw directly without computing the full lnphi).
from ms_ecpa.lnphi import _aq_residual_3var            # noqa: E402
from ms_ecpa.newton import newton_solve                # noqa: E402
from ms_ecpa.roots import is_liquid_root              # noqa: E402
from ms_ecpa.parameters import (                       # noqa: E402
    Mc, R, Tc1,
    MMi, A_Peni, B_Peni,
)
import ms_ecpa.parameters as _P   # Mw read LIVE: paper1_config rebinds it
from paper_style import (                              # noqa: E402
    apply_style, bold_ticks, panel_label,
    PAPER_FIGURES, WONG, LINESTYLES,
)
from data_loader import (                              # noqa: E402
    load_experimental, load_or_compute, add_cache_flag,
)

# Pressure grid for the smooth model curves (matches OPT_PENELOUX.py; 5 -> 700 bar).
P_GRID_BAR = np.linspace(5.0, 700.0, 200)

# Main-paper / SI panel selections (kept here as data so the layout can be
# tweaked without touching the plotting logic).
MAIN_SALT = "NaCl"
MAIN_TEMPERATURES = (298, 348, 398)
SI_SALTS = ("KCl", "CaCl", "MgCl", "NaSO")
SI_TEMPERATURES = (298, 348, 398)

# Cosmetic label per salt (for axis annotations and legends).
SALT_LABELS = {
    "NaCl": "NaCl",
    "KCl":  "KCl",
    "CaCl": r"CaCl$_2$",
    "MgCl": r"MgCl$_2$",
    "NaSO": r"Na$_2$SO$_4$",
}


# ---------------------------------------------------------------------------
# Single-point brine density helper
# ---------------------------------------------------------------------------

#: Fixed-Zw seeds tried after the warm start.
_ZW_SEEDS_FIXED = (np.array([0.07, 60.0, 0.13]),
                   np.array([0.10, 30.0, 0.05]),
                   np.array([0.30, 70.0, 0.20]),
                   np.array([0.20, 40.0, 0.10]),
                   np.array([0.50, 50.0, 0.10]),
                   np.array([0.03, 78.0, 0.07]))


def _zw_seeds(T, P_Pa):
    """Seed ladder for the aqueous (Zw, epsr, chi1w) solve.

    Every rung used to be a FIXED Zw, but a liquid's compressibility factor is
    Zw = P v / (R T) with v nearly constant, so the liquid root MOVES with
    pressure: about 0.0065 at 9 bar and 0.05 at 70 bar. Below ~20 bar the
    whole fixed ladder -- smallest rung 0.03 -- sits three to forty times
    above the liquid basin and never finds it. That is 100 of the 105
    brine-density states the model could not solve, all of them under 20 bar
    and spread across every temperature.

    lnphi_aq already knew this: it seeds its own inner Newton with exactly
    P*Mw/(R*T) (lnphi.py). The density solver simply never used it. The
    physically scaled rung goes FIRST, with a couple of multiples for the
    salt-loaded cases where the brine is denser than pure water.
    """
    Zw_liq = float(P_Pa) * 18.015e-6 / (8.314 * float(T))
    scaled = [np.array([Zw_liq, 50.0, 0.10]),
              np.array([0.85 * Zw_liq, 60.0, 0.13]),
              np.array([1.25 * Zw_liq, 40.0, 0.07]),
              # Low-permittivity rungs, for concentrated brine. A 5 molal
              # MgCl2 solution has its permittivity suppressed to roughly 30
              # by the ions, so a rung pairing a scaled Zw with eps_r near 78
              # starts in the wrong basin twice over. These two are what solve
              # the last states in the set (323 and 348 K, 10.5 bar, MgCl2 5 m).
              np.array([1.5 * Zw_liq, 30.0, 0.10]),
              np.array([0.4 * Zw_liq, 30.0, 0.05])]
    return tuple(scaled) + _ZW_SEEDS_FIXED


def _solve_Zw(x_aq, mi, T, P_Pa, *, v_init=None, accept=None):
    """Return the converged (Zw, epsr, chi1w) triple for the aqueous phase.

    Reuses ``_aq_residual_3var`` from ``ms_ecpa.lnphi`` (the same three-equation
    closure that ``lnphi_aq`` solves internally) plus the project's damped
    Newton with complex-step Jacobian.

    ``accept`` decides whether a CONVERGED root is the one that was asked for.
    Without it, this function used to return the first root the Newton settled
    on, and the seed ladder below ran only when the Newton FAILED. That is the
    wrong condition: the closure has several roots, and on four brine-density
    states the Newton succeeds onto Zw ~ 350, which reports a density near
    1 kg/m3 at 500 bar. Nothing failed, so nothing retried, and a meaningless
    number went into the figure. The ladder now runs on "converged to the
    wrong branch" as well.
    """
    def _attempt(seed):
        v, ok, _ = newton_solve(_aq_residual_3var, np.asarray(seed, dtype=float),
                                args=(x_aq, mi, T, P_Pa), tol=1e-10, maxiter=80)
        if not ok:
            return None
        # chi1w is a monomer fraction: it is never negative. The 3x3 Newton
        # can settle on the negative branch of the association closure --
        # it satisfies the residual to machine precision, so the solver
        # reports success -- and the density that root implies can still
        # land inside the liquid bounds, so the branch test below does not
        # catch it. lnphi_aq has rejected chi1w <= 0 for the same reason
        # since the same failure was found there; this is that guard.
        if not (float(v[2]) > 0.0):
            return None
        if accept is not None and not accept(v):
            return None
        return v

    first = (v_init if v_init is not None
             else np.array([float(P_Pa) * 18.015e-6 / (8.314 * float(T)),
                            50.0, 0.10]))
    for seed in (first, *_zw_seeds(T, P_Pa)):
        v = _attempt(seed)
        if v is not None:
            return v
    # Every seed either failed or landed off-branch. NaN is the honest answer;
    # a wrong density is not.
    return None


def predict_rho_brine(T_K: float, P_bar: float, mi: np.ndarray,
                      *, v_init=None) -> tuple[float, np.ndarray]:
    """Predict the aqueous-phase mass density of a CO2-free brine.

    Uses x_aq = (1, 0, 0) (pure water + ions, no dissolved CO2/CH4). The
    Peneloux ion shifts (T-dependent through ``A_Peni*T/Tc1 + B_Peni``,
    units of 1e-6 m^3/mol) are applied to the molar volume:

        v = Zw * R * T / P + sum(xiw * Pen_i)        [m^3/mol]
        rho = MMw / v                                  [kg/m^3]

    Returns ``(rho, v_state)`` where ``v_state = (Zw, epsr, chi1w)`` may be
    fed back in as ``v_init`` to accelerate the next call on the same isotherm.
    """
    # x_aq convention in ms_ecpa.lnphi._aq_core_state: the three components
    # are taken directly as mole fractions of (H2O, CH4, CO2) in the aqueous
    # phase, and the ion fractions xiw = x1w * Mw * mi are added on top. For
    # the totals to sum to 1, the caller must rescale x1w by the ion factor
    #     x1w = (1 - x_ch4 - x_co2) / (1 + Mw * sum(mi))
    # (cf. ms_ecpa.eos.residual_elv:line 59 and OPT_PENELOUX.py:line 263).
    x1w = 1.0 / (1.0 + _P.Mw * float(np.sum(mi)))
    x_aq = np.array([x1w, 0.0, 0.0])
    P_Pa = float(P_bar) * 1e5

    def _rho_of(state):
        """Density implied by a candidate state. Single source of truth."""
        Zw = float(np.real(state[0]))
        # Peneloux shift (matches OPT_PENELOUX.py line 578).
        Pen_i = (A_Peni * T_K / Tc1 + B_Peni) * 1e-6
        xiw = x1w * _P.Mw * mi
        MMw = _P.Mw * x1w + Mc * 0.0 + float(np.sum(MMi * xiw))
        v = Zw * R * T_K / P_Pa + float(np.sum(xiw * Pen_i))
        if v <= 0:
            return float("nan")
        return MMw / v

    state = _solve_Zw(x_aq, mi, float(T_K), P_Pa, v_init=v_init,
                      accept=lambda st: is_liquid_root(_rho_of(st), float(T_K)))
    if state is None:
        return float("nan"), None
    return _rho_of(state), state


# ---------------------------------------------------------------------------
# Compute step (slow): one row per (salt, T, m_salt, P_bar) on a smooth grid,
# PLUS one row per experimental point with the model prediction at exactly
# the experimental P (for AAD reporting and scatter overlays).
# ---------------------------------------------------------------------------
def compute_brine_density_grid() -> pd.DataFrame:
    """Compute model brine densities on (i) the smooth (T, salt, m_salt) iso-T
    pressure grid (for the line curves) and (ii) at each experimental P
    (for AAD calculation).

    Returns a single DataFrame with a ``kind`` column distinguishing
    ``"model_curve"`` rows (P from ``P_GRID_BAR``) from ``"model_at_exp"``
    rows (P from the experimental parquet).
    """
    exp = load_experimental("brine_density")
    print(f"  loaded {len(exp)} experimental rows; computing model on smooth "
          f"P grid + at experimental P ...")
    rows = []
    t0 = time.perf_counter()

    # The unique (salt, T, m_salt) combinations drive both the smooth curves
    # and the per-experimental-point predictions.
    keys = exp[["salt", "T_K", "m_salt_mol_kg"]].drop_duplicates().sort_values(
        ["salt", "T_K", "m_salt_mol_kg"]
    ).reset_index(drop=True)

    for _, k in keys.iterrows():
        salt = k["salt"]; T_K = float(k["T_K"]); m_salt = float(k["m_salt_mol_kg"])
        sub_exp = exp[(exp["salt"] == salt) &
                      (exp["T_K"] == int(T_K)) &
                      (exp["m_salt_mol_kg"] == m_salt)]
        if sub_exp.empty:
            continue
        # Ion molalities are constant within a (salt, T, m_salt) row; pull from
        # the first experimental row.
        first = sub_exp.iloc[0]
        mi = np.array([first["m_Na"], first["m_Cl"], first["m_K"],
                       first["m_Ca"], first["m_Mg"], first["m_SO4"]])

        # (i) Smooth pressure curve at this (salt, T, m_salt).
        state = None
        for P_bar in P_GRID_BAR:
            rho, state = predict_rho_brine(T_K, float(P_bar), mi, v_init=state)
            rows.append({
                "kind": "model_curve",
                "salt": salt,
                "T_K": int(T_K),
                "m_salt_mol_kg": m_salt,
                "P_bar": float(P_bar),
                "rho_pred_kg_m3": float(rho),
                "rho_exp_kg_m3": float("nan"),
                "author": "",
            })

        # (ii) Predictions exactly at experimental pressures (warm-started
        # from the last smooth-grid state for that isotherm).
        for _, er in sub_exp.iterrows():
            rho, _ = predict_rho_brine(T_K, float(er["P_bar"]), mi,
                                       v_init=state)
            rows.append({
                "kind": "model_at_exp",
                "salt": salt,
                "T_K": int(T_K),
                "m_salt_mol_kg": m_salt,
                "P_bar": float(er["P_bar"]),
                "rho_pred_kg_m3": float(rho),
                "rho_exp_kg_m3": float(er["rho_exp_kg_m3"]),
                "author": str(er["author"]),
            })

    dt = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    at_exp = df[df["kind"] == "model_at_exp"].copy()
    n_ok = int(np.sum(np.isfinite(df["rho_pred_kg_m3"])))
    # The pooled finite count mixes the 200-point smooth curve grid in with the
    # experimental states, and only the latter feed the AARE, so report the two
    # separately: a state the solver cannot resolve has to be visible as a
    # missing measurement, not diluted by 23k grid points.
    n_ok_exp = int(np.sum(np.isfinite(at_exp["rho_pred_kg_m3"])))
    print(f"  ...{dt:.1f}s; {n_ok} / {len(df)} predictions finite "
          f"({n_ok / len(df) * 100:.1f}%); at experimental P: "
          f"{n_ok_exp} / {len(at_exp)}")
    # Headline AAD across experimental points.
    err = (at_exp["rho_pred_kg_m3"] - at_exp["rho_exp_kg_m3"]).abs() \
        / at_exp["rho_exp_kg_m3"] * 100.0
    err = err.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  headline AARE = {err.mean():.2f}%  "
          f"(N = {len(err)} of {len(at_exp)})")
    return df


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _ms_colors(n_ms: int):
    """Return n_ms distinct colorblind-safe colors from the WONG palette.

    Skips index 0 (black -- reserved for the y=x guide / frame) and index 4
    (yellow -- low contrast on white). Falls back to repetition for n_ms > 6.
    """
    pool = [WONG[i] for i in (5, 2, 3, 6, 7, 1)]  # blue, sky, green, vermillion, purple, orange
    return [pool[i % len(pool)] for i in range(n_ms)]


def _ms_linestyles(n_ms: int):
    return [LINESTYLES[i % len(LINESTYLES)] for i in range(n_ms)]


def _plot_one_panel(ax, df, salt: str, T_K: int, *, show_salt: bool = True) -> int:
    """Plot one (salt, T_K) panel and return the number of molality series drawn.

    The bottom-right corner annotation is "<salt>, <T> K" (matching the
    convention in fig_co2_singlesalt etc.) when ``show_salt=True``, or just
    "<T> K" when False -- use False on figures where the salt is named in
    the caption (e.g. the NaCl-only main figure).
    """
    salt_label = SALT_LABELS.get(salt, salt)
    curves = df[(df["kind"] == "model_curve") &
                (df["salt"] == salt) &
                (df["T_K"] == T_K)]
    pts = df[(df["kind"] == "model_at_exp") &
             (df["salt"] == salt) &
             (df["T_K"] == T_K)]
    ms_values = sorted(curves["m_salt_mol_kg"].unique())
    cols = _ms_colors(len(ms_values))
    lss = _ms_linestyles(len(ms_values))

    for i, m_salt in enumerate(ms_values):
        c_sub = curves[curves["m_salt_mol_kg"] == m_salt].sort_values("P_bar")
        p_sub = pts[pts["m_salt_mol_kg"] == m_salt]
        ok = np.isfinite(c_sub["rho_pred_kg_m3"])
        ax.plot(c_sub["P_bar"][ok], c_sub["rho_pred_kg_m3"][ok],
                color=cols[i], linestyle=lss[i],
                label=f"{m_salt:g} mol/kg")
        if len(p_sub):
            ax.scatter(p_sub["P_bar"], p_sub["rho_exp_kg_m3"],
                       marker="x", s=42, color=cols[i],
                       linewidths=1.6, zorder=3)

    ax.set_xlabel(r"$P$ [bar]")
    ax.set_ylabel(r"$\rho_{\mathrm{aq}}$ [kg m$^{-3}$]")
    # T (+ salt) annotation bottom-right (no plot title, per style guide).
    annot = f"{salt_label}, {T_K} K" if show_salt else f"{T_K} K"
    ax.text(0.96, 0.06, annot,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=12, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
    ax.grid(alpha=0.25)
    return len(ms_values)


# ---------------------------------------------------------------------------
# Figure 1: main paper -- NaCl x 3 temperatures (1 x 3)
#
# Shared y/x axes for direct visual comparison across isotherms; y-label
# only on the leftmost panel; single common legend above the panels (the
# three panels share the same set of NaCl molalities, so one legend covers
# all three).
# ---------------------------------------------------------------------------
def make_main_figure(df: pd.DataFrame, out_path: Path) -> None:
    from matplotlib.lines import Line2D

    # Global axis ranges across the 3 NaCl panels.
    main = df[(df["salt"] == MAIN_SALT) & df["T_K"].isin(MAIN_TEMPERATURES)]
    rho_all = pd.concat([
        main[main["kind"] == "model_curve"]["rho_pred_kg_m3"],
        main[main["kind"] == "model_at_exp"]["rho_exp_kg_m3"],
    ]).dropna()
    y_min, y_max = float(rho_all.min()), float(rho_all.max())
    y_margin = 0.04 * (y_max - y_min)
    y_lim = (y_min - y_margin, y_max + y_margin)
    x_lim = (float(P_GRID_BAR.min()), float(P_GRID_BAR.max()))

    # Pre-compute the canonical molality list + color/style mapping so the
    # legend handles match what each panel draws.
    ms_values = sorted(
        main[main["kind"] == "model_curve"]["m_salt_mol_kg"].unique()
    )
    cols = _ms_colors(len(ms_values))
    lss = _ms_linestyles(len(ms_values))

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.6), squeeze=False,
                             sharex=True, sharey=True)
    for j, T_K in enumerate(MAIN_TEMPERATURES):
        ax = axes[0][j]
        # Main figure is NaCl-only (named in the caption) -- skip the salt
        # name in the panel annotation.
        _plot_one_panel(ax, df, MAIN_SALT, T_K, show_salt=False)
        panel_label(ax, chr(ord("a") + j))
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        # Y-label only on the leftmost panel.
        if j == 0:
            ax.set_ylabel(r"$\rho_{\mathrm{aq}}$ [kg m$^{-3}$]")
        else:
            ax.set_ylabel("")
        ax.set_xlabel(r"$P$ [bar]")
        # Drop the per-panel legend that _plot_one_panel auto-creates; we'll
        # add a single legend on the rightmost panel below.
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        bold_ticks(ax)

    # Single common legend above the panels.
    handles = [
        Line2D([0], [0], color=cols[i], linestyle=lss[i], linewidth=2.4,
               label=f"{m:g} mol/kg")
        for i, m in enumerate(ms_values)
    ] + [
        Line2D([0], [0], marker="x", color="black", linestyle="",
               markersize=9, markeredgewidth=1.8, label="experiment"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.005),
               ncol=len(handles), frameon=True, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# SI figures: one figure per salt
#
# Each per-salt SI figure mirrors the NaCl main figure: 1 row x 3 T panels,
# shared y/x axes across panels, single shared top legend, compact
# "<salt>, <T> K" annotation in the bottom-right.
# ---------------------------------------------------------------------------
def _make_SI_per_salt(df: pd.DataFrame, salt: str, out_path: Path) -> None:
    from matplotlib.lines import Line2D
    sub = df[(df["salt"] == salt) & df["T_K"].isin(SI_TEMPERATURES)]
    if sub.empty:
        return
    rho_all = pd.concat([
        sub[sub["kind"] == "model_curve"]["rho_pred_kg_m3"],
        sub[sub["kind"] == "model_at_exp"]["rho_exp_kg_m3"],
    ]).dropna()
    if rho_all.empty:
        return
    y_min, y_max = float(rho_all.min()), float(rho_all.max())
    y_margin = 0.04 * (y_max - y_min)
    y_lim = (y_min - y_margin, y_max + y_margin)
    x_lim = (float(P_GRID_BAR.min()), float(P_GRID_BAR.max()))

    ms_values = sorted(
        sub[sub["kind"] == "model_curve"]["m_salt_mol_kg"].unique()
    )
    cols = _ms_colors(len(ms_values))
    lss = _ms_linestyles(len(ms_values))

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.6), squeeze=False,
                             sharex=True, sharey=True)
    for j, T_K in enumerate(SI_TEMPERATURES):
        ax = axes[0][j]
        _plot_one_panel(ax, df, salt, T_K, show_salt=True)
        panel_label(ax, chr(ord("a") + j))
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        if j == 0:
            ax.set_ylabel(r"$\rho_{\mathrm{aq}}$ [kg m$^{-3}$]")
        else:
            ax.set_ylabel("")
        ax.set_xlabel(r"$P$ [bar]")
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        bold_ticks(ax)

    handles = [
        Line2D([0], [0], color=cols[i], linestyle=lss[i], linewidth=2.4,
               label=f"{m:g} mol/kg")
        for i, m in enumerate(ms_values)
    ] + [
        Line2D([0], [0], marker="x", color="black", linestyle="",
               markersize=9, markeredgewidth=1.8, label="experiment"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.005),
               ncol=len(handles), frameon=True, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_SI_figures(df: pd.DataFrame, out_dir: Path) -> list:
    """Produce one SI figure per salt (KCl, CaCl2, MgCl2, Na2SO4)."""
    from matplotlib.lines import Line2D  # noqa: F401  (already imported above)
    out_paths = []
    for salt in SI_SALTS:
        fname = f"brine_density_SI_{salt}.png"
        out_path = out_dir / fname
        _make_SI_per_salt(df, salt, out_path)
        out_paths.append(out_path)
        print(f"  {fname}")
    return out_paths


# ---------------------------------------------------------------------------
# AAD report (printed alongside the figures)
# ---------------------------------------------------------------------------
def print_aad_summary(df: pd.DataFrame) -> None:
    at_exp = df[df["kind"] == "model_at_exp"].copy()
    at_exp = at_exp.assign(
        rel_err_pct=(at_exp["rho_pred_kg_m3"] - at_exp["rho_exp_kg_m3"]).abs()
                    / at_exp["rho_exp_kg_m3"] * 100.0
    )
    # A state whose prediction is non-finite -- predict_rho_brine returns NaN
    # when no seed lands on a liquid root -- carries no error and cannot enter
    # the mean. It must still be counted: without N_exp beside it, N reads as
    # "the number of measurements" when it is "the number that converged", and
    # unsolved states leave the average silently.
    n_meas = at_exp.groupby("salt").size()
    scored = at_exp.replace([np.inf, -np.inf], np.nan).dropna(subset=["rel_err_pct"])
    print("-" * 60)
    print(f"{'salt':>6}  {'N':>5}  {'N_exp':>6}  {'AARE [%]':>10}")
    print("-" * 60)
    for salt in n_meas.index:
        sub = scored[scored["salt"] == salt]
        aare = float(sub["rel_err_pct"].mean()) if len(sub) else float("nan")
        print(f"{salt:>6}  {len(sub):>5}  {int(n_meas[salt]):>6}  {aare:>10.2f}")
    print("-" * 60)
    print(f"{'TOTAL':>6}  {len(scored):>5}  {len(at_exp):>6}  "
          f"{scored['rel_err_pct'].mean():>10.2f}")
    print("-" * 60)
    n_drop = len(at_exp) - len(scored)
    if n_drop:
        missing = ", ".join(
            f"{salt} {int(n_meas[salt]) - len(scored[scored['salt'] == salt])}"
            for salt in n_meas.index
            if int(n_meas[salt]) - len(scored[scored["salt"] == salt]) > 0
        )
        print(f"  NOT SCORED: {n_drop} of {len(at_exp)} experimental states have "
              f"no finite prediction and are excluded from every AARE above "
              f"({missing})")
        print("-" * 60)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cache_flag(p)
    args = p.parse_args()
    apply_style()
    df = load_or_compute("fig_brine_density", compute_brine_density_grid,
                         force=args.force)
    print_aad_summary(df)
    print()
    print("Making 2 figures from cached predictions ...")
    make_main_figure(df, PAPER_FIGURES / "brine_density_main.png")
    print("  brine_density_main.png")
    make_SI_figures(df, PAPER_FIGURES)


if __name__ == "__main__":
    raise SystemExit(main())
