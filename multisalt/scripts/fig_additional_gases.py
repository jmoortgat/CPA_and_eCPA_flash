#!/usr/bin/env python3
"""Generate the additional-gases (N2 + H2) paper figures (main text).

  - n2_brine_main.png  (nine panels: H2O / 1 m / 4 m NaCl x three
                        isotherms; O'Sullivan & Smith 1970 markers; the
                        zero-brine-tuning CH4->N2 transfer prediction and
                        the brine-refit curve, which are visually
                        coincident -- that coincidence is the point)
  - h2_brine_main.png  (nine-panel H2 analogue: the Coelho 2025 H2
                        parameterization adopted verbatim -- zero fitting
                        of any kind -- run through the multi-salt flash)

Physics/provenance: the curves are recomputed through the pilot rebind
machinery (``pilot_studies/n2_pilot_2026_07/n2_pilot.py`` and
``pilot_studies/h2_pilot_2026_07/h2_pilot.py``), imported here as
modules.  Both pilots swap gas slot 2 (CH4) at runtime in the four
consuming namespaces (parameters + the lnphi/eos star-import snapshots +
the flash by-name imports, plus the in-place Wilson arrays) and restore
the production CH4 slot in ``finally`` blocks inside every
``_predict_point`` call, so ``ms_ecpa/`` is never modified and the
production slot is restored inside the worker that rebound it.  That
rebinding happens only in the spawn-pool workers, so the hygiene check
below probes the workers themselves (live parameter state against each
pilot's import-time snapshot) and additionally spot-flashes the parent.

  N2:  pure-component constants from critical data; water-N2 kij(T)
       (production quadratic-in-T/Tc1 form) fitted by the pilot to the
       18 SALT-FREE O'Sullivan points only; gas-ion energies transferred
       from CH4 with the function-preserving Tc rescale
       A_N2,i = A_CH4,i * Tc(N2)/Tc(CH4) (B unchanged) -- "transfer".
       The "fitted" variant additionally scales the transferred
       (Na+, Cl-) energy pair by the pilot's brine-fit s = 1.0084.
       Coefficients are read from the pilot deliverable
       ``n2_results.json`` (single source of truth), not re-derived.
  H2:  the Coelho, Mercier Franco, Firoozabadi (2025, IJHE 178, 151361)
       H2 e-CPA parameterization adopted VERBATIM (kij(T) exact in our
       water; the quadratic-Talfa H2-ion energy represented exactly per
       point via the A=0 / B=U_hs(T) rebind); NaCl-only by construction.

Stats sidecar (``results_cache/fig_additional_gases_stats.json``):
headline AAREs read from the pilot deliverable JSONs -- N2 water /
brine-transfer / brine-fitted + the fitted scaling s; H2 water (on the
Coelho 2025 fit set) / brine + Coelho's own reference AADs + the
model-to-model ternary deviation bracket.  Consumed by
``scripts/build_paper_stats_tex.py`` (macros \\aareNtwo*, \\aareHtwo*).

Usage:
    python3 scripts/fig_additional_gases.py            # use cache
    python3 scripts/fig_additional_gases.py --force    # recompute
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "scripts"))
sys.path.insert(0, str(_CODE / "pilot_studies" / "n2_pilot_2026_07"))
sys.path.insert(0, str(_CODE / "pilot_studies" / "h2_pilot_2026_07"))

import n2_pilot                                          # noqa: E402
import h2_pilot                                          # noqa: E402
from paper_style import (                                # noqa: E402
    apply_style, bold_ticks, PAPER_FIGURES, WONG,
)
from data_loader import (                                # noqa: E402
    load_or_compute, save_stats, add_cache_flag,
)

N2_RESULTS = _CODE / "pilot_studies" / "n2_pilot_2026_07" / "n2_results.json"
H2_RESULTS = _CODE / "pilot_studies" / "h2_pilot_2026_07" / "h2_results.json"

N_WORKERS = 8

# Figure grids (identical to the pilot validation figures).
N2_TS = (324.65, 375.65, 398.15)          # K   (O'Sullivan isotherms)
N2_MS = (0.0, 1.0, 4.0)                   # mol/kg NaCl
N2_P_ATM = np.geomspace(85.0, 650.0, 14)  # atm (total pressure)

H2_TS = (298.0, 323.0, 373.0)             # K
H2_MS = (0.0, 1.0, 4.0)                   # mol/kg NaCl
H2_P_BAR = np.geomspace(20.0, 500.0, 14)  # bar

# Coelho 2025 fit-set restriction: PRAY(1957) and DOHRN(1986) were
# excluded from his adopted OPT4 kij fit (scoping README section 3), so
# the apples-to-apples water AARE excludes them too.
H2_NON_FITSET_AUTHORS = ("PRAY(1957)", "DOHRN(1986)")

STYLE_TRANSFER = dict(color=WONG[5], linestyle="-")     # blue solid
STYLE_FITTED = dict(color=WONG[6], linestyle="-.")      # vermillion dash-dot
STYLE_VERBATIM = dict(color=WONG[5], linestyle="-")     # blue solid
LBL_TRANSFER = "ΔU transferred from CH$_4$ (prediction)"
LBL_FITTED = "brine-refit ($s = 1.0084$)"
LBL_VERBATIM = "Coelho 2025 parameters, verbatim"
MLAB = {0.0: "H$_2$O", 1.0: "1 m NaCl", 4.0: "4 m NaCl"}


def _pilot_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"Pilot result file {path} is missing. Run the corresponding "
            f"pilot first (see the README.md in {path.parent}).")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Compute step (slow): recompute every curve through the pilot machinery
# ---------------------------------------------------------------------------
def _slot_state_probe(_) -> tuple:
    """Worker-side slot-hygiene probe -> (pid, [names of unrestored entries]).

    The pilots rebind gas slot 2 only inside ``_predict_point``, which here
    runs exclusively in the spawn-pool workers; the parent process is never
    rebound.  A check in the parent therefore cannot observe a restore that
    left an entry behind (which would leak N2/H2 parameters into every later
    point handed to that reused worker), so the probe runs in the worker and
    compares the live module state against the pilot's own import-time
    snapshot, over exactly the set the rebind writes.  It compares stored
    parameters rather than flash results, so it cannot trip on cross-process
    float reproducibility.

    The brief sleep spreads the probe tasks over distinct workers, since map
    hands them out one at a time; coverage is best-effort (a worker that got
    no sweep task would be clean trivially), and the caller reports how many
    distinct processes answered.
    """
    time.sleep(0.05)
    leaked: list[str] = []
    for pilot in (n2_pilot, h2_pilot):
        tag = pilot.__name__
        for ns in pilot.NAMESPACES:
            for name, orig in pilot._ORIG.items():
                if hasattr(ns, name) and getattr(ns, name) != orig:
                    leaked.append(f"{tag}:{ns.__name__}.{name}")
        if not np.array_equal(pilot.prm.A_U2i, pilot._ORIG_A_U2i):
            leaked.append(f"{tag}:prm.A_U2i")
        if not np.array_equal(pilot.prm.B_U2i, pilot._ORIG_B_U2i):
            leaked.append(f"{tag}:prm.B_U2i")
        wilson = (float(pilot.flash_mod._TC[1]),
                  float(pilot.flash_mod._PC_BAR[1]),
                  float(pilot.flash_mod._OMEGA[1]))
        if wilson != pilot._ORIG_WILSON:
            leaked.append(f"{tag}:flash_mod.Wilson[1]")
    return os.getpid(), leaked


def compute_curves() -> pd.DataFrame:
    n2 = _pilot_json(N2_RESULTS)
    kij = n2["fitB"]["best"]["x"]              # water-fit kij (framing B)
    s_fit = n2["fitA"]["best"]["x"][0]         # brine-fit scaling (framing A)

    # Parent-process spot flashes, taken before the sweep and repeated
    # after it: importing the pilots must not disturb the production slot
    # here either.  This pair covers the parent only -- the workers, where
    # the rebinding actually happens, are probed by _slot_state_probe.
    guard_pre = {c: n2_pilot.guard_spot(c) for c in n2_pilot.GUARD_CASES}

    rows: list[dict] = []
    t0 = time.perf_counter()
    pool = mp.get_context("spawn").Pool(N_WORKERS)
    try:
        # --- N2: transfer (s = 1) + fitted (s = s_fit) ---
        for variant, s in (("transfer", 1.0), ("fitted", s_fit)):
            for T in N2_TS:
                for m in N2_MS:
                    tasks = [(kij[0], kij[1], kij[2], s, n2_pilot.POL_N2,
                              T, p * n2_pilot.ATM_TO_BAR, m)
                             for p in N2_P_ATM]
                    preds = pool.map(n2_pilot._predict_point, tasks)
                    rows += [dict(gas="N2", T_K=T, m_salt=m, variant=variant,
                                  P_bar=p * n2_pilot.ATM_TO_BAR,
                                  value=float(x))
                             for p, x in zip(N2_P_ATM, preds)]
                print(f"  N2 {variant} T={T} done "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

        # --- H2: verbatim Coelho 2025 (zero fitting) ---
        for T in H2_TS:
            for m in H2_MS:
                tasks = [(h2_pilot.DU_REF, h2_pilot.K23_DISK, T, float(p), m)
                         for p in H2_P_BAR]
                out = pool.map(h2_pilot._predict_point, tasks)
                rows += [dict(gas="H2", T_K=T, m_salt=m, variant="verbatim",
                              P_bar=float(p), value=float(o[2]))
                         for p, o in zip(H2_P_BAR, out)]
            print(f"  H2 verbatim T={T} done "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

        probes = pool.map(_slot_state_probe, range(2 * N_WORKERS),
                          chunksize=1)
    finally:
        pool.close()
        pool.join()

    leaked = sorted({name for _, names in probes for name in names})
    if leaked:
        raise SystemExit(
            "slot-hygiene guard FAILED in the pool workers: the production "
            f"CH4 slot was left rebound for {leaked}")
    n_probed = len({pid for pid, _ in probes})

    guard_post = {c: n2_pilot.guard_spot(c) for c in n2_pilot.GUARD_CASES}
    for c in guard_pre:
        if not n2_pilot.tuples_bit_identical(guard_pre[c], guard_post[c]):
            raise SystemExit(
                f"slot-hygiene guard FAILED for {c}: production CH4/CO2 "
                f"flash changed across the curve sweep")
    print(f"  slot-hygiene guards passed ({n_probed}/{N_WORKERS} worker "
          f"processes probed, plus the parent); "
          f"{len(rows)} curve points in {time.perf_counter() - t0:.0f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stats sidecar (headline numbers come from the pilot deliverable JSONs)
# ---------------------------------------------------------------------------
def build_stats() -> dict:
    n2 = _pilot_json(N2_RESULTS)
    h2 = _pilot_json(H2_RESULTS)

    tbl = n2["comparison_table_aare_pct"]
    aareB = n2["framings"]["B"]["aare"]
    n2_stats = {
        "aare_water": round(tbl["B"]["water"], 2),
        "N_water": aareB["water"]["n_points"],
        "aare_brine_transfer": round(tbl["B"]["brine_all"], 2),
        "aare_brine_1m": round(tbl["B"]["1m"], 2),
        "aare_brine_4m": round(tbl["B"]["4m"], 2),
        "N_brine": aareB["brine_all"]["n_points"],
        "aare_brine_fitted": round(tbl["A"]["brine_all"], 2),
        "s_fit": f"{n2['fitA']['best']['x'][0]:.4f}",
        "provenance": "kij(T) fitted to the 18 salt-free O'Sullivan 1970 "
                      "points; (Na,Cl) energies = Tc-rescaled CH4 transfer",
    }

    # H2 water AARE restricted to the Coelho 2025 fit set (drop the two
    # sources he rejected), n_converged-weighted over the per-author AAREs.
    wrep = h2["V1_water"]["report"]
    num = den = 0.0
    for key, r in wrep.items():
        if not key.startswith("x_by_author/"):
            continue
        if key.split("/", 1)[1] in H2_NON_FITSET_AUTHORS:
            continue
        num += r["aare_pct"] * r["n_converged"]
        den += r["n_converged"]
    fel = h2["coelho2025_reference_aad_pct"]
    v3 = [c[k] for c in h2["V3"]["curves"]
          for k in ("mean_abs_dmh_pct", "mean_abs_dmc_pct")
          if c.get(k) is not None]
    h2_stats = {
        "aare_water_fitset": round(num / den, 2),
        "N_water_fitset": int(den),
        "aare_water_coelho": fel["water_x"],
        "aare_brine": round(h2["V1_brine"]["report"]["all"]["aare_pct"], 2),
        "N_brine": h2["V1_brine"]["report"]["all"]["n_points"],
        "aare_brine_coelho": fel["brine"],
        "ternary_dev_low": f"{min(v3):.2f}",
        "ternary_dev_high": f"{max(v3):.1f}",
        "provenance": "Coelho 2025 (IJHE 178, 151361) parameterization "
                      "adopted verbatim; zero fitting; NaCl-only",
    }
    print(f"  N2: water {n2_stats['aare_water']}% (N={n2_stats['N_water']}), "
          f"brine transfer {n2_stats['aare_brine_transfer']}% "
          f"(N={n2_stats['N_brine']}), fitted "
          f"{n2_stats['aare_brine_fitted']}% (s={n2_stats['s_fit']})")
    print(f"  H2: water {h2_stats['aare_water_fitset']}% vs Coelho "
          f"{h2_stats['aare_water_coelho']}%; brine {h2_stats['aare_brine']}%"
          f" vs {h2_stats['aare_brine_coelho']}% (N={h2_stats['N_brine']}); "
          f"ternary {h2_stats['ternary_dev_low']}-"
          f"{h2_stats['ternary_dev_high']}%")
    return {"n2": n2_stats, "h2": h2_stats}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_n2_figure(df: pd.DataFrame, out_path: Path) -> None:
    exp = n2_pilot.load_n2()
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11.0), sharex=True)
    for i, m in enumerate(N2_MS):
        for j, T in enumerate(N2_TS):
            ax = axes[i, j]
            sub = exp[np.isclose(exp.m_NaCl_mol_kg, m)
                      & np.isclose(exp.T_K, T)]
            ax.plot(sub.P_atm * n2_pilot.ATM_TO_BAR, sub.x_molefrac * 1e4,
                    "o", color="black", mfc="white", mew=1.8, zorder=5,
                    linestyle="none", label="O'Sullivan & Smith 1970")
            for variant, style, lbl in (
                    ("transfer", STYLE_TRANSFER, LBL_TRANSFER),
                    ("fitted", STYLE_FITTED, LBL_FITTED)):
                c = df[(df.gas == "N2") & (df.variant == variant)
                       & np.isclose(df.T_K, T)
                       & np.isclose(df.m_salt, m)].sort_values("P_bar")
                ok = np.isfinite(c.value)
                ax.plot(c.P_bar[ok], c.value[ok] * 1e4, label=lbl, **style)
            ax.text(0.05, 0.95, f"{MLAB[m]}\n{T:.1f} K",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=12, fontweight="bold", color="black",
                    bbox=dict(facecolor="white", alpha=0.85,
                              edgecolor="none", pad=2))
            if i == 2:
                ax.set_xlabel(r"$P$ [bar]")
            if j == 0:
                ax.set_ylabel(r"$10^4\,x_{\mathrm{N_2}}$")
            bold_ticks(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Legend outside and above the panels (house convention for
    # multi-panel figures; keeps every curve unobstructed).
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=True,
               fontsize=12, handletextpad=0.5, columnspacing=1.4)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_h2_figure(df: pd.DataFrame, out_path: Path) -> None:
    import ms_ecpa.parameters as prm
    wdf = h2_pilot.load_water()
    bdf = h2_pilot.load_brine()
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11.0), sharex=True)
    for i, m in enumerate(H2_MS):
        for j, T in enumerate(H2_TS):
            ax = axes[i, j]
            if m == 0.0:
                sub = wdf[np.isclose(wdf.T_K, T) & wdf.x_H2_aq.notna()
                          & (wdf.P_bar <= 510.0)]
                Pd = sub.P_bar.to_numpy()
                md = (sub.x_H2_aq / ((1.0 - sub.x_H2_aq) * prm.Mw)).to_numpy()
            else:
                sub = bdf[np.isclose(bdf.T_K, T)
                          & np.isclose(bdf.m_NaCl_molkg, m)
                          & (bdf.P_bar <= 510.0)]
                Pd = sub.P_bar.to_numpy()
                md = sub.m_H2_molkg.to_numpy()
            ax.plot(Pd, md, "o", color="black", mfc="white", mew=1.8,
                    zorder=5, linestyle="none", label="experiments")
            c = df[(df.gas == "H2") & (df.variant == "verbatim")
                   & np.isclose(df.T_K, T)
                   & np.isclose(df.m_salt, m)].sort_values("P_bar")
            ok = np.isfinite(c.value)
            ax.plot(c.P_bar[ok], c.value[ok], label=LBL_VERBATIM,
                    **STYLE_VERBATIM)
            ax.text(0.05, 0.95, f"{MLAB[m]}\n{T:.0f} K",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=12, fontweight="bold", color="black",
                    bbox=dict(facecolor="white", alpha=0.85,
                              edgecolor="none", pad=2))
            ax.set_xlim(0, 510)
            if i == 2:
                ax.set_xlabel(r"$P$ [bar]")
            if j == 0:
                ax.set_ylabel(r"$m_{\mathrm{H_2}}$ [mol/kg]")
            bold_ticks(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Legend outside and above the panels (house convention).
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=True,
               fontsize=12, handletextpad=0.5, columnspacing=1.4)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cache_flag(p)
    args = p.parse_args()
    apply_style()
    df = load_or_compute("fig_additional_gases", compute_curves,
                         force=args.force)
    stats = build_stats()
    save_stats("fig_additional_gases", stats)
    print("Making 2 figures from cached predictions ...")
    make_n2_figure(df, PAPER_FIGURES / "n2_brine_main.png")
    print("  n2_brine_main.png")
    make_h2_figure(df, PAPER_FIGURES / "h2_brine_main.png")
    print("  h2_brine_main.png")


if __name__ == "__main__":
    raise SystemExit(main())
