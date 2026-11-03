#!/usr/bin/env python3
"""Quantitative comparison of the two salinity conventions.

Both conventions are implemented in ``ms_ecpa`` and selected by the
``salinity_convention`` keyword of the production entry points
(:func:`stability_flash` / :func:`flash_kv`):

``"as-aqueous"`` (library default)
    the input ``mi`` **is** the equilibrium aqueous molality; no
    salt-conservation amplification is applied.

``"chain"``
    the input ``mi`` is the prepared-brine **feed** molality and the
    equilibrium aqueous molality rises as water leaves the brine,

        m_aq = m_feed * z_w / [(1 - beta) * x_part_w]

    with ``x_part_w`` the salt-free aqueous water mole fraction.

This script measures the difference between them.  It produces three
cached data sets and two figures.

Parts
-----
``validation``
    Re-scores the experimental validation sets under both conventions.
    CO2 single-salt brines (``data/co2_brines.parquet``) and CO2 mixed
    brines (``data/co2_mixed_brines.parquet``) are scored through
    :func:`stability_flash`, reusing the adaptive ``z_CO2`` retry ladder
    and the degenerate-two-phase guard of
    ``scripts/fig_co2_singlesalt.py``.  CH4 brines
    (``data/ch4_brines.parquet``) are scored the same way inside the
    solver's temperature window.

    Because the "chain" prediction depends on the gas-to-brine ratio of
    the cell -- which the source papers do not report -- the chain is
    scored at three fixed feed loadings, z_CO2 in {0.05, 0.5, 0.9}.
    The "as-aqueous" prediction is independent of that loading.

``regime``
    Structured sweep of gas-to-brine ratio x temperature x pressure x
    feed salinity, both conventions at every point.

``gasrich``
    Fine scan into the gas-dominated (reservoir/injection-front) end of
    the range, out to the dry-out boundary where the aqueous phase
    disappears.

Figures
-------
``convention_regime.pdf``    from the ``regime`` cache
``convention_gasrich.pdf``   from the ``gasrich`` cache

Usage
-----
    python3 scripts/fig_convention_comparison.py --part all --force
    python3 scripts/fig_convention_comparison.py            # plot from cache

The library is never modified and the default convention is never
changed: every call passes ``salinity_convention`` explicitly.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Keep BLAS single-threaded: the inner Newton/SSI loops are scalar and
# the parallelism here is at the process level.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "scripts"))

from ms_ecpa import stability_flash                    # noqa: E402
from paper_style import (                              # noqa: E402
    apply_style, bold_ticks, panel_label, save_figure,
    PAPER_FIGURES, WONG, LINESTYLES,
)
from data_loader import (                              # noqa: E402
    load_experimental, load_or_compute, add_cache_flag, save_stats,
    RESULTS_CACHE,
)

MW = 0.018                       # kg/mol H2O, matches fig_co2_singlesalt
ION_COLS = ["m_Na", "m_Cl", "m_K", "m_Ca", "m_Mg", "m_SO4"]

# Adaptive feed ladder, identical to fig_co2_singlesalt.Z_CO2_RETRIES.
Z_GAS_RETRIES = (0.05, 0.10, 0.02, 0.20, 0.30)
# Fixed gas loadings at which the chain is additionally scored.
CHAIN_LOADINGS = (0.05, 0.5, 0.9)

CONVENTIONS = ("as-aqueous", "chain")

# A phase fraction this close to 0 or 1 means the "split" carries no
# material; such states are rejected rather than compared (see
# ``_flash_record``).
BETA_EPS = 1e-6

# stability_flash / solve_elv physical domain (fig_ch4_singlesalt).
T_MIN, T_MAX = 284.0, 532.0
P_MIN, P_MAX = 1.5, 1499.0

N_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 2))

# Wong entries chosen for contrast on white: the palette's yellow is too
# light for thin lines, so it is skipped.
SERIES_COLORS = [WONG[0], WONG[1], WONG[2], WONG[3], WONG[6], WONG[5]]


# ---------------------------------------------------------------------------
# Single flash -> flat record
# ---------------------------------------------------------------------------
def _flash_record(T, P, mi, z_gas, convention, gas_index=2):
    """One :func:`stability_flash` call -> flat dict of observables.

    ``gas_index`` is 2 for CO2 and 1 for CH4 in the fixed component
    ordering ``(H2O, CH4, CO2)``.  Returns ``ok=False`` with a reason on
    any non-two-phase or non-converged outcome; never raises.
    """
    z = np.zeros(3)
    z[0] = 1.0 - z_gas
    z[gas_index] = z_gas
    out = {
        "T_K": float(T), "P_bar": float(P), "z_gas": float(z_gas),
        "convention": convention, "ok": False, "phases": np.nan,
        "reason": "", "x_gas_W": np.nan, "m_gas": np.nan, "beta": np.nan,
        "y_H2O": np.nan, "m_aq_tot": np.nan, "ampl": np.nan,
        "n_iter_ssi": np.nan, "max_residual": np.nan, "t_ms": np.nan,
    }
    t0 = time.perf_counter()
    try:
        res = stability_flash(float(T), float(P), z, np.asarray(mi, float),
                              salinity_convention=convention)
    except Exception as exc:                                  # noqa: BLE001
        out["reason"] = f"exception:{type(exc).__name__}"
        out["t_ms"] = (time.perf_counter() - t0) * 1e3
        return out
    out["t_ms"] = (time.perf_counter() - t0) * 1e3
    out["phases"] = int(res.phases)
    if res.phases != 2 or res.flash_result is None:
        out["reason"] = "single-phase" if res.phases == 1 else "flash-failed"
        return out
    fr = res.flash_result
    # ``stability_flash`` can return phases == 2 carrying a flash result
    # that did NOT converge.  Using such a state as a number is the
    # silent-non-convergence pattern this project audits for; reject it.
    if not bool(fr.converged):
        out["reason"] = "flash-not-converged"
        out["max_residual"] = float(fr.max_residual)
        out["beta"] = float(fr.beta)
        return out
    xH = float(fr.x_aq[0])
    xG = float(fr.x_aq[gas_index])
    if not (np.isfinite(xH) and np.isfinite(xG)) or (xH + xG) <= 0:
        out["reason"] = "non-finite"
        return out
    x_gas_W = xG / (xH + xG)
    # Degenerate-two-phase guard of fig_co2_singlesalt._flash_one: a
    # salt-free aqueous gas fraction numerically equal to the feed means
    # no selective partitioning actually happened.
    if abs(x_gas_W - z_gas) < 1e-3 * max(z_gas, 1e-6):
        out["reason"] = "degenerate-split"
        return out
    # Vanishing-phase guard.  Near the dry-out boundary the flash can
    # return beta == 1 (or 0) with a nominal "aqueous" composition that
    # carries no material.  Such a state has no equilibrium aqueous
    # molality and must not enter a convention comparison.
    if not (BETA_EPS < float(fr.beta) < 1.0 - BETA_EPS):
        out["beta"] = float(fr.beta)
        out["reason"] = "beta-degenerate"
        return out
    m_feed_tot = float(np.sum(mi))
    m_aq_tot = float(np.sum(fr.mi_aq))
    out.update({
        "ok": True,
        "x_gas_W": x_gas_W,
        "m_gas": x_gas_W / (1.0 - x_gas_W) / MW,
        "beta": float(fr.beta),
        "y_H2O": float(fr.y_gas[0]),
        "m_aq_tot": m_aq_tot,
        "ampl": (m_aq_tot / m_feed_tot) if m_feed_tot > 0 else 1.0,
        "n_iter_ssi": int(fr.n_iter_ssi),
        "max_residual": float(fr.max_residual),
    })
    return out


# ---------------------------------------------------------------------------
# Part 1 -- validation re-scoring
# ---------------------------------------------------------------------------
def _validation_task(job):
    """Worker: one experimental row, both conventions.

    ``as-aqueous`` walks the retry ladder; the ladder loading that
    succeeds is then re-used for the paired chain run, so any difference
    between the two rows is attributable to the convention alone.  The
    chain is additionally run at the fixed loadings in
    ``CHAIN_LOADINGS``.
    """
    meta, mi, gas_index = job
    rows = []
    z_used = np.nan
    for z_gas in Z_GAS_RETRIES:
        rec = _flash_record(meta["T_K"], meta["P_bar"], mi, z_gas,
                            "as-aqueous", gas_index)
        if rec["ok"]:
            z_used = z_gas
            rows.append({**meta, **rec, "variant": "as-aqueous"})
            break
    if not np.isfinite(z_used):
        rows.append({**meta, **rec, "variant": "as-aqueous"})
    else:
        rec = _flash_record(meta["T_K"], meta["P_bar"], mi, z_used,
                            "chain", gas_index)
        rows.append({**meta, **rec, "variant": "chain-paired"})
    for z_gas in CHAIN_LOADINGS:
        rec = _flash_record(meta["T_K"], meta["P_bar"], mi, z_gas,
                            "chain", gas_index)
        rows.append({**meta, **rec, "variant": f"chain-z{z_gas:g}"})
    return rows


def _validation_jobs():
    jobs = []
    # CO2, single salt
    exp = load_experimental("co2_brines")
    for i, r in exp.iterrows():
        meta = {"dataset": "co2_singlesalt", "row_id": int(i),
                "salt": str(r["salt"]), "author": str(r["author"]),
                "ms_mol_kg": float(r["ms_mol_kg"]),
                "obs_x_gas_W": float(r["xc_W"]),
                "T_K": float(r["T_K"]), "P_bar": float(r["P_bar"])}
        mi = np.array([float(r[c]) for c in ION_COLS])
        jobs.append((meta, mi, 2))
    # CO2, mixed brines
    exm = load_experimental("co2_mixed_brines")
    for i, r in exm.iterrows():
        mi = np.array([float(r[c]) for c in ION_COLS])
        meta = {"dataset": "co2_mixed", "row_id": int(i),
                "salt": str(r["mix_id"]), "author": str(r["author"]),
                "ms_mol_kg": float(np.sum(mi)),
                "obs_x_gas_W": float(r["xc_W"]),
                "T_K": float(r["T_K"]), "P_bar": float(r["P_bar"])}
        jobs.append((meta, mi, 2))
    # CH4, single salt, inside the solver window
    exh = load_experimental("ch4_brines")
    exh = exh[(exh["T_K"] >= T_MIN) & (exh["T_K"] <= T_MAX) &
              (exh["P_bar"] >= P_MIN) & (exh["P_bar"] <= P_MAX)]
    for i, r in exh.iterrows():
        mh = float(r["mh_W"])                      # mol CH4 / kg H2O
        x_h_W = mh * MW / (1.0 + mh * MW)          # salt-free mole fraction
        meta = {"dataset": "ch4_singlesalt", "row_id": int(i),
                "salt": str(r["salt"]), "author": str(r["author"]),
                "ms_mol_kg": float(r["ms_mol_kg"]),
                "obs_x_gas_W": x_h_W,
                "T_K": float(r["T_K"]), "P_bar": float(r["P_bar"])}
        mi = np.array([float(r[c]) for c in ION_COLS])
        jobs.append((meta, mi, 1))
    return jobs


def compute_validation() -> pd.DataFrame:
    jobs = _validation_jobs()
    print(f"  validation: {len(jobs)} experimental rows x "
          f"{2 + len(CHAIN_LOADINGS)} flashes")
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for k, res in enumerate(ex.map(_validation_task, jobs, chunksize=4)):
            rows.extend(res)
            if (k + 1) % 100 == 0:
                print(f"    [{k+1}/{len(jobs)}] "
                      f"{time.perf_counter()-t0:.0f}s")
    print(f"  ...validation done in {time.perf_counter()-t0:.0f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 2 -- regime sweep
# ---------------------------------------------------------------------------
REGIME_T = (298.0, 323.0, 373.0, 423.0, 473.0)
REGIME_P = (50.0, 100.0, 200.0, 400.0)
REGIME_M = (0.0, 1.0, 3.0, 5.0)                 # NaCl feed molality
# Gas-to-brine molar ratio G = z_gas / z_H2O, log-spaced plus the
# near-dry-out tail.
REGIME_G = (0.01, 0.0526, 0.111, 0.25, 0.4286, 1.0, 2.333, 4.0,
            9.0, 19.0, 32.33, 49.0, 99.0)


def _sweep_task(job):
    T, P, m, G, gas_index = job
    z_gas = G / (1.0 + G)
    mi = np.array([m, m, 0.0, 0.0, 0.0, 0.0])
    out = []
    for conv in CONVENTIONS:
        rec = _flash_record(T, P, mi, z_gas, conv, gas_index)
        rec.update({"m_feed": float(m), "G": float(G)})
        out.append(rec)
    return out


def compute_regime() -> pd.DataFrame:
    jobs = [(T, P, m, G, 2)
            for T in REGIME_T for P in REGIME_P
            for m in REGIME_M for G in REGIME_G]
    print(f"  regime: {len(jobs)} states x 2 conventions")
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for k, res in enumerate(ex.map(_sweep_task, jobs, chunksize=4)):
            rows.extend(res)
            if (k + 1) % 100 == 0:
                print(f"    [{k+1}/{len(jobs)}] "
                      f"{time.perf_counter()-t0:.0f}s")
    print(f"  ...regime done in {time.perf_counter()-t0:.0f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 3 -- gas-rich / reservoir end
# ---------------------------------------------------------------------------
# (T, P, m_NaCl) triples: a benchtop cell, a storage-reservoir state,
# a hot deep state, and a geothermal state.
GASRICH_STATES = (
    (298.0, 100.0, 1.0),
    (323.0, 150.0, 1.0),
    (373.0, 200.0, 1.0),
    (373.0, 200.0, 5.0),
    (423.0, 250.0, 1.0),
    (473.0, 300.0, 1.0),
)
GASRICH_G = tuple(np.geomspace(0.01, 3000.0, 40))


def compute_gasrich() -> pd.DataFrame:
    jobs = [(T, P, m, float(G), 2)
            for (T, P, m) in GASRICH_STATES for G in GASRICH_G]
    print(f"  gasrich: {len(jobs)} states x 2 conventions")
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for k, res in enumerate(ex.map(_sweep_task, jobs, chunksize=2)):
            rows.extend(res)
            if (k + 1) % 50 == 0:
                print(f"    [{k+1}/{len(jobs)}] "
                      f"{time.perf_counter()-t0:.0f}s")
    print(f"  ...gasrich done in {time.perf_counter()-t0:.0f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 4 -- three-phase (vapor + aqueous + CO2-rich liquid)
# ---------------------------------------------------------------------------
# The three-phase flash applies the same closure with beta_g + beta_l in
# place of the single beta.  States taken from tests/test_flash3.py, which
# are known to give a genuine three-phase split.
THREE_PHASE_STATES = (
    (283.15, 50.0, (0.50, 0.03, 0.47)),
    (283.15, 53.0, (0.50, 0.03, 0.47)),
    (283.15, 56.0, (0.50, 0.03, 0.47)),
    (283.15, 50.0, (0.20, 0.03, 0.77)),
    (283.15, 50.0, (0.05, 0.03, 0.92)),
)
THREE_PHASE_M = (1.0, 3.0, 5.0)


def compute_three_phase() -> pd.DataFrame:
    from ms_ecpa import stability_flash3
    rows = []
    for (T, P, zt) in THREE_PHASE_STATES:
        for m in THREE_PHASE_M:
            z = np.array(zt, dtype=float)
            mi = np.array([m, m, 0.0, 0.0, 0.0, 0.0])
            for conv in CONVENTIONS:
                rec = {"T_K": T, "P_bar": P, "z_H2O": zt[0], "z_CO2": zt[2],
                       "m_feed": m, "convention": conv, "ok": False,
                       "phases": np.nan, "reason": ""}
                try:
                    r = stability_flash3(T, P, z, mi,
                                         salinity_convention=conv)
                except Exception as exc:                      # noqa: BLE001
                    rec["reason"] = f"exception:{type(exc).__name__}"
                    rows.append(rec)
                    continue
                rec["phases"] = int(r.phases)
                if r.phases != 3 or r.flash3 is None or not r.flash3.converged:
                    rec["reason"] = f"phases={r.phases}"
                    rows.append(rec)
                    continue
                f3 = r.flash3
                rec.update({
                    "ok": True,
                    "beta_g": float(f3.beta_g), "beta_l": float(f3.beta_l),
                    "x_CO2_aq": float(f3.x_aq[2]),
                    "x_H2O_aq": float(f3.x_aq[0]),
                    "y_H2O": float(f3.y_gas[0]),
                    "m_aq_tot": float(np.sum(f3.mi_aq)),
                    "ampl": float(np.sum(f3.mi_aq)) / (2.0 * m),
                })
                rows.append(rec)
    return pd.DataFrame(rows)


def report_three_phase(df: pd.DataFrame) -> None:
    print("\n" + "=" * 96)
    print("THREE-PHASE (V + Aq + L_CO2): chain vs as-aqueous")
    print("=" * 96)
    keys = ["T_K", "P_bar", "z_CO2", "m_feed"]
    a = df[df["convention"] == "as-aqueous"].set_index(keys)
    c = df[df["convention"] == "chain"].set_index(keys)
    print(f"  {'T':>7} {'P':>6} {'zCO2':>6} {'m':>4} "
          f"{'ampl':>8} {'d_xCO2aq%':>11} {'d_bg%':>8} {'d_bl%':>8} "
          f"{'d_yw%':>8}")
    for k in a.index:
        if k not in c.index:
            continue
        ra, rc = a.loc[k], c.loc[k]
        if not (bool(ra["ok"]) and bool(rc["ok"])):
            print(f"  {k}  as-aq ok={ra['ok']} chain ok={rc['ok']}")
            continue
        d = lambda f: 100.0 * (rc[f] - ra[f]) / ra[f]     # noqa: E731
        print(f"  {k[0]:>7.2f} {k[1]:>6.1f} {k[2]:>6.2f} {k[3]:>4.1f} "
              f"{rc['ampl']:>8.4f} {d('x_CO2_aq'):>11.3f} "
              f"{d('beta_g'):>8.3f} {d('beta_l'):>8.3f} {d('y_H2O'):>8.3f}")


# ---------------------------------------------------------------------------
# Pairing helper: wide table of (as-aqueous, chain) at identical states
# ---------------------------------------------------------------------------
_PAIR_KEYS = ["T_K", "P_bar", "m_feed", "G"]
_PAIR_VALS = ["x_gas_W", "m_gas", "beta", "y_H2O", "ampl", "m_aq_tot",
              "phases", "ok", "n_iter_ssi", "reason"]


def pair_conventions(df: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per state, ``*_a`` (as-aqueous) and ``*_c``
    (chain) columns plus signed relative differences in percent."""
    df = df.copy()
    # Apply the vanishing-phase guard here as well, so caches written
    # before the guard existed are filtered identically.
    degen = df["ok"] & ~df["beta"].between(BETA_EPS, 1.0 - BETA_EPS)
    df.loc[degen, "ok"] = False
    df.loc[degen, "reason"] = "beta-degenerate"
    a = df[df["convention"] == "as-aqueous"].set_index(_PAIR_KEYS)[_PAIR_VALS]
    c = df[df["convention"] == "chain"].set_index(_PAIR_KEYS)[_PAIR_VALS]
    w = a.join(c, lsuffix="_a", rsuffix="_c", how="inner").reset_index()
    both = w["ok_a"] & w["ok_c"]
    for v in ("x_gas_W", "beta", "y_H2O", "m_gas"):
        w[f"d_{v}"] = np.where(
            both, 100.0 * (w[f"{v}_c"] - w[f"{v}_a"]) / w[f"{v}_a"], np.nan)
    w["ampl_c"] = np.where(both, w["ampl_c"], np.nan)
    w["d_m_aq"] = np.where(both, 100.0 * (w["ampl_c"] - 1.0), np.nan)
    return w


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_regime_figure(df: pd.DataFrame, out_path: Path,
                       P_fix: float = 200.0, m_fix: float = 1.0) -> None:
    """2x2: relative chain-minus-as-aqueous difference against the
    gas-to-brine molar ratio, one curve per temperature."""
    w = pair_conventions(df)
    w = w[(w["P_bar"] == P_fix) & (w["m_feed"] == m_fix)]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6), sharex=True)
    # (column, y-label, panel letter, log-|.| scale?)
    quantities = [
        ("d_m_aq",    r"$\Delta m_{\rm aq}$ [%]",        "a", True),
        ("d_x_gas_W", r"$|\Delta x_{\rm CO_2}|$ [%]",    "b", True),
        ("d_beta",    r"$\Delta \beta$ [%]",             "c", False),
        ("d_y_H2O",   r"$|\Delta y_{\rm H_2O}|$ [%]",    "d", True),
    ]
    temps = sorted(w["T_K"].unique())
    for ax, (col, ylab, letter, logy) in zip(axes.ravel(), quantities):
        for i, T in enumerate(temps):
            g = w[w["T_K"] == T].sort_values("G")
            g = g[np.isfinite(g[col])]
            if g.empty:
                continue
            y = g[col].abs() if logy else g[col]
            ax.plot(g["G"], y, color=SERIES_COLORS[i % len(SERIES_COLORS)],
                    linestyle=LINESTYLES[i % len(LINESTYLES)],
                    marker="o", markersize=4.5,
                    label=rf"{T:.0f} K")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        else:
            ax.axhline(0.0, color="0.4", lw=1.0, ls=":")
        ax.set_ylabel(ylab)
        panel_label(ax, letter, x=0.04, y=0.97)
        bold_ticks(ax)
    for ax in axes[1]:
        ax.set_xlabel(r"$z_{\rm CO_2}/z_{\rm H_2O}$")
    axes[0, 0].legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    for ax in axes.ravel():
        bold_ticks(ax)
    save_figure(fig, out_path)
    plt.close(fig)


def make_gasrich_figure(df: pd.DataFrame, out_path: Path) -> None:
    """1x2: (a) molality amplification and (b) aqueous-CO2 difference,
    across the full gas-to-brine range out to dry-out."""
    w = pair_conventions(df)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    states = sorted({(float(r.T_K), float(r.P_bar), float(r.m_feed))
                     for r in w.itertuples()})
    for i, (T, P, m) in enumerate(states):
        g = w[(w["T_K"] == T) & (w["P_bar"] == P) &
              (w["m_feed"] == m)].sort_values("G")
        lab = rf"{T:.0f} K, {P:.0f} bar, {m:.0f} m"
        col = SERIES_COLORS[i % len(SERIES_COLORS)]
        ls = LINESTYLES[i % len(LINESTYLES)]
        gg = g[np.isfinite(g["ampl_c"])]
        if not gg.empty:
            axes[0].plot(gg["G"], gg["ampl_c"], color=col, linestyle=ls,
                         marker="o", markersize=4.0, label=lab)
        gg = g[np.isfinite(g["d_x_gas_W"])]
        if not gg.empty:
            axes[1].plot(gg["G"], np.abs(gg["d_x_gas_W"]), color=col,
                         linestyle=ls, marker="o", markersize=4.0, label=lab)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$z_{\rm CO_2}/z_{\rm H_2O}$")
        bold_ticks(ax)
    axes[0].set_ylabel(r"$m_{\rm aq}/m_{\rm feed}$")
    axes[0].axhline(1.0, color="0.4", lw=1.0, ls=":")
    axes[0].legend(loc="upper left", fontsize=9.5)
    panel_label(axes[0], "a", x=0.04, y=0.60)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$|\Delta x_{\rm CO_2}|$ [%]")
    for thr, sty in ((1.0, "--"), (5.0, "-.")):
        axes[1].axhline(thr, color="0.35", lw=1.4, ls=sty)
        axes[1].annotate(f"{thr:g} %", xy=(0.015, thr), xycoords=("axes fraction",
                         "data"), va="bottom", fontsize=11, fontweight="bold",
                         color="0.25")
    panel_label(axes[1], "b", x=0.04, y=0.97)
    fig.tight_layout()
    for ax in axes:
        bold_ticks(ax)
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def apply_beta_guard(df: pd.DataFrame) -> pd.DataFrame:
    """Reject accepted-but-vanishing splits in a cached frame.

    Idempotent, and a no-op on frames produced after the guard was added
    to :func:`_flash_record`.
    """
    df = df.copy()
    degen = df["ok"] & ~df["beta"].between(BETA_EPS, 1.0 - BETA_EPS)
    n = int(degen.sum())
    if n:
        print(f"  [beta-guard] rejected {n} vanishing-phase rows")
    df.loc[degen, "ok"] = False
    df.loc[degen, "reason"] = "beta-degenerate"
    return df


def _aare(pred, obs):
    pred = np.asarray(pred, float)
    obs = np.asarray(obs, float)
    ok = np.isfinite(pred) & np.isfinite(obs) & (obs > 0)
    if not ok.any():
        return float("nan"), 0
    return 100.0 * float(np.mean(np.abs(pred[ok] - obs[ok]) / obs[ok])), int(ok.sum())


def report_validation(df: pd.DataFrame) -> dict:
    """Print per-dataset, per-salt AARE under every variant."""
    stats: dict = {}
    variants = ["as-aqueous", "chain-paired"] + \
               [f"chain-z{z:g}" for z in CHAIN_LOADINGS]
    print("\n" + "=" * 96)
    print("VALIDATION AARE [%] on the salt-free aqueous gas mole fraction")
    print("=" * 96)
    for ds, gds in df.groupby("dataset"):
        print(f"\n{ds}")
        header = f"  {'salt':>18} {'Nexp':>5}"
        for v in variants:
            header += f" {v:>14}"
        print(header)
        per_salt = {}
        for salt, g in gds.groupby("salt"):
            n_exp = int(g[g["variant"] == "as-aqueous"].shape[0])
            line = f"  {salt:>18} {n_exp:>5}"
            entry = {"N_exp": n_exp}
            for v in variants:
                gv = g[(g["variant"] == v) & g["ok"]]
                a, n = _aare(gv["x_gas_W"], gv["obs_x_gas_W"])
                line += f" {a:>8.2f}({n:>3d})"
                entry[v] = {"aare": None if not np.isfinite(a) else round(a, 2),
                            "N": n}
            per_salt[salt] = entry
            print(line)
        line = f"  {'ALL':>18} {int(gds[gds['variant']=='as-aqueous'].shape[0]):>5}"
        overall = {}
        for v in variants:
            gv = gds[(gds["variant"] == v) & gds["ok"]]
            a, n = _aare(gv["x_gas_W"], gv["obs_x_gas_W"])
            line += f" {a:>8.2f}({n:>3d})"
            overall[v] = {"aare": None if not np.isfinite(a) else round(a, 2),
                          "N": n}
        print(line)
        # Common-subset score: only rows where every variant converged.
        keys = None
        for v in variants:
            s = set(gds[(gds["variant"] == v) & gds["ok"]]["row_id"])
            keys = s if keys is None else (keys & s)
        common = {}
        line = f"  {'COMMON SUBSET':>18} {len(keys):>5}"
        for v in variants:
            gv = gds[(gds["variant"] == v) & gds["ok"] &
                     gds["row_id"].isin(keys)]
            a, n = _aare(gv["x_gas_W"], gv["obs_x_gas_W"])
            line += f" {a:>8.2f}({n:>3d})"
            common[v] = {"aare": None if not np.isfinite(a) else round(a, 2),
                         "N": n}
        print(line)
        stats[ds] = {"per_salt": per_salt, "overall": overall,
                     "common_subset": common}
    return stats


def report_validation_by_T(df: pd.DataFrame) -> dict:
    """Per-isotherm AARE: the divergence between conventions grows with
    the gas-phase water content, hence with temperature."""
    variants = ["as-aqueous", "chain-paired"] + \
               [f"chain-z{z:g}" for z in CHAIN_LOADINGS]
    print("\n" + "=" * 96)
    print("VALIDATION AARE [%] BY ISOTHERM")
    print("=" * 96)
    out: dict = {}
    for ds, gds in df.groupby("dataset"):
        print(f"\n{ds}")
        head = f"  {'T [K]':>7} {'Nexp':>5}"
        for v in variants:
            head += f" {v:>14}"
        print(head)
        for T, g in gds.groupby("T_K"):
            n_exp = int(g[g["variant"] == "as-aqueous"].shape[0])
            line = f"  {T:>7.0f} {n_exp:>5}"
            entry = {}
            for v in variants:
                gv = g[(g["variant"] == v) & g["ok"]]
                a, n = _aare(gv["x_gas_W"], gv["obs_x_gas_W"])
                line += f" {a:>8.2f}({n:>3d})"
                entry[v] = {"aare": None if not np.isfinite(a) else round(a, 2),
                            "N": n}
            out[f"{ds}@{T:.0f}"] = entry
            print(line)
    return out


def report_convergence(df: pd.DataFrame, label: str) -> dict:
    print(f"\nconvergence -- {label}")
    out = {}
    for v, g in df.groupby(df.get("variant", df.get("convention"))):
        n = len(g)
        ok = int(g["ok"].sum())
        reasons = g[~g["ok"]]["reason"].value_counts().to_dict()
        print(f"  {v:>16}  {ok}/{n} ({100*ok/max(n,1):.1f} %)   {reasons}")
        out[str(v)] = {"N": n, "ok": ok, "reasons": reasons}
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cache_flag(p)
    p.add_argument("--part", default="all",
                   choices=("all", "validation", "regime", "gasrich",
                            "three", "plot"))
    args = p.parse_args()
    apply_style()

    do = args.part
    if do in ("all", "validation"):
        dv = apply_beta_guard(load_or_compute("conv_validation",
                                              compute_validation,
                                              force=args.force))
        s = report_validation(dv)
        t = report_validation_by_T(dv)
        c = report_convergence(dv, "validation")
        save_stats("conv_validation",
                   {"aare": s, "aare_by_T": t, "convergence": c})

    if do in ("all", "regime", "plot"):
        dr = apply_beta_guard(load_or_compute(
            "conv_regime", compute_regime,
            force=args.force and do != "plot"))
        report_convergence(dr, "regime sweep")
        make_regime_figure(dr, PAPER_FIGURES / "convention_regime.pdf")
        print(f"  wrote {PAPER_FIGURES / 'convention_regime.pdf'}")
        pair_conventions(dr).to_csv(RESULTS_CACHE / "conv_regime_paired.csv",
                                    index=False)

    if do in ("all", "gasrich", "plot"):
        dg = apply_beta_guard(load_or_compute(
            "conv_gasrich", compute_gasrich,
            force=args.force and do != "plot"))
        report_convergence(dg, "gas-rich scan")
        make_gasrich_figure(dg, PAPER_FIGURES / "convention_gasrich.pdf")
        print(f"  wrote {PAPER_FIGURES / 'convention_gasrich.pdf'}")
        pair_conventions(dg).to_csv(RESULTS_CACHE / "conv_gasrich_paired.csv",
                                    index=False)

    if do in ("all", "three"):
        d3 = load_or_compute("conv_three_phase", compute_three_phase,
                             force=args.force)
        report_three_phase(d3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
