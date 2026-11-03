#!/usr/bin/env python3
"""Validate ``ms_ecpa.stability_flash`` against CO2-brine experimental
solubility data for multiple salts.

For each (T, salt, file) experimental data point, this script:
  1. Reads the full ion-molality vector from the data file (cols 4-9 are
     Na, Cl, K, Ca, Mg, SO4 in mol/kg-water). This is salt-stoichiometry-
     agnostic: NaCl, KCl, CaCl2, MgCl2, Na2SO4, MgSO4 all use the same
     code path.
  2. Constructs the partitioning-component feed z = (1-z_CO2, 0, z_CO2)
     with adaptive z_CO2 (retries 0.05 -> 0.10 -> 0.02 -> 0.20 if the
     reference single-phase EoS in michelsen_tpd fails at a tricky
     low-P + high-salinity point).
  3. Calls stability_flash(T, P, z, mi).
  4. Reads x_CO2 in the aqueous phase from result.flash_result.x_aq.
  5. Converts to the salt-free CO2 mole fraction the experiments report
     (xcW_pred = x_CO2 / (x_H2O + x_CO2)) and compares to experiment.

Outputs:
  - AARE summary tables to stdout: per (salt, T, author) and (salt, T, salinity).
  - Comparison figure: per-salt panels with model curves vs experimental
    points, saved to ``results/validate_co2_brines.png``.

Usage:
    python3 scripts/validate_co2_brines.py [--salts NaCl KCl CaCl MgCl NaSO MgSO]
                                            [--temperatures 323 373 423]

Salt name conventions follow F. M. Coelho's directory structure (CaCl = CaCl2,
MgCl = MgCl2, NaSO = Na2SO4, MgSO = MgSO4).
"""
from __future__ import annotations

import os
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Import the package -- script lives at code/scripts/, package at code/ms_ecpa/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ms_ecpa import stability_flash  # noqa: E402

# Original model-output files of the earlier multisalt code (not distributed).
EOS_DATA_ROOT = Path(os.environ.get("ECPA_SOURCE_MATERIALS", "source_materials")) / "EoS/CO2/eCPA/MIXED_SALTS"
SALT_FOLDERS = ("NaCl", "KCl", "CaCl", "MgCl", "NaSO", "MgSO")
DEFAULT_T = (323, 373, 423)
MW = 0.018  # kg/mol (water)

# z_CO2 retry ladder for the adaptive feed (avoids unphysical single-phase
# reference at the most extreme conditions).
Z_CO2_RETRIES = (0.05, 0.10, 0.02, 0.20, 0.30)


def parse_exp_file(path: Path) -> tuple[str, list[dict]]:
    """Read an EXP_<salt><k>_T<T>K.txt file.

    Returns (author, list of row dicts), each row:
        {P [bar], ms [mol/kg-total], xcW [salt-free CO2 mole fraction],
         mi [ndarray(6): Na, Cl, K, Ca, Mg, SO4 molalities], src, col3_units}.

    Third column is sometimes ``xc_W [SALTfree]`` (mole fraction) and
    sometimes ``mc [mol/kg]`` (molality) -- header line tells us which.
    """
    lines = path.read_text().splitlines()
    author = lines[0].lstrip("#").strip()
    header = lines[1].lower() if len(lines) > 1 else ""
    toks = header.split()
    # toks roughly: 'p', '[bar]', 'ms', '[mol/kg]', '<col3-name>', '[<unit>]', ...
    col3_token = toks[4] if len(toks) >= 6 else ""
    col3_is_molality = col3_token in ("mc", "m_co2", "m_c") or "mol/kg" in toks[5:6]

    rows = []
    for ln in lines[2:]:
        if not ln.strip() or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 9:
            continue
        try:
            P = float(parts[0])
            ms = float(parts[1])
            v3 = float(parts[2])
            mi = np.array([float(parts[3 + k]) for k in range(6)])
        except ValueError:
            continue
        if col3_is_molality:
            xcW = v3 * MW / (1.0 + v3 * MW)
        else:
            xcW = v3
        # Quick charge-balance sanity check.
        Zi = np.array([+1, -1, +1, +2, +2, -2], dtype=float)
        Zsum = float(np.sum(mi * Zi))
        if abs(Zsum) > 1e-6 * max(1.0, float(np.sum(np.abs(mi)))):
            continue  # silently skip non-neutral rows (data error)
        rows.append({
            "P": P, "ms": ms, "xcW": xcW, "mi": mi, "src": path.name,
            "col3_units": "mc" if col3_is_molality else "xcW",
        })
    return author, rows


def collect_experimental_data(salt: str, temperatures: list[int]) -> dict:
    """Walk EOS_DATA_ROOT and aggregate {T: {author: [rows...]}} for one salt."""
    by_T = {}
    for T in temperatures:
        Tdir = EOS_DATA_ROOT / f"T{T}K"
        by_author: dict[str, list] = defaultdict(list)
        files = sorted(Tdir.glob(f"EXP_{salt}*_T*K.txt"))
        # Skip "_Xauthor"-suffixed redundant files.
        files = [f for f in files if "_X" not in f.stem]
        for f in files:
            author, rows = parse_exp_file(f)
            by_author[author].extend(rows)
        by_T[T] = dict(by_author)
    return by_T


def model_xcW(T: float, P_bar: float, mi: np.ndarray) -> dict:
    """Predict xc_W with an adaptive-feed retry ladder.

    Returns a dict with keys: xcW, x_aq, phases, converged, tpd_min,
    t_solve, z_CO2_used, n_retries, error.

    A rung counts only if the flash itself converged; ``converged`` in the
    returned dict is therefore the flash's own verdict, not merely "two
    phases came back".
    """
    last_exc = None
    for k, zC in enumerate(Z_CO2_RETRIES):
        z = np.array([1.0 - zC, 0.0, zC])
        t0 = time.perf_counter()
        try:
            res = stability_flash(float(T), float(P_bar), z, mi)
            dt = time.perf_counter() - t0
            if res.phases == 2 and res.flash_result is not None:
                if not bool(res.flash_result.converged):
                    # stability_flash's hard-failure tail returns phases == 2
                    # carrying the last, non-converged flash iterate. Its x_aq
                    # is finite and positive, so it would clear the only filter
                    # run_validation applies (isfinite and > 0) and be averaged
                    # into the AARE tables as though an equation had been
                    # solved. Advance the ladder instead -- a larger or smaller
                    # z_CO2 may converge where this one did not.
                    last_exc = "flash did not converge"
                    continue
                xH, xC4, xC = res.flash_result.x_aq
                xcW = float(xC / (xH + xC))
                return dict(
                    xcW=xcW, x_aq=res.flash_result.x_aq, phases=2,
                    converged=True, tpd_min=res.tpd_min, t_solve=dt,
                    z_CO2_used=zC, n_retries=k, error=None,
                )
            # Got a result but not 2-phase -- try a larger z_CO2.
            last_exc = f"phases={res.phases}"
        except Exception as exc:
            last_exc = str(exc).splitlines()[0][:120]
    return dict(
        xcW=float("nan"), x_aq=None, phases=0,
        converged=False, tpd_min=float("nan"),
        t_solve=float("nan"), z_CO2_used=None, n_retries=len(Z_CO2_RETRIES),
        error=last_exc,
    )


def run_validation(salts: list[str], temperatures: list[int], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    AARE_by_salt_T_author = defaultdict(list)
    AARE_by_salt_T = defaultdict(list)
    fail_by_salt_T = defaultdict(int)

    for salt in salts:
        exp_by_T = collect_experimental_data(salt, temperatures)
        if not any(by_a for by_a in exp_by_T.values()):
            print(f"  (no data for salt={salt}; skipping)")
            continue
        for T, by_author in exp_by_T.items():
            for author, exp_rows in by_author.items():
                for r in exp_rows:
                    pred = model_xcW(T, r["P"], r["mi"])
                    if not np.isfinite(pred["xcW"]) or pred["xcW"] <= 0:
                        fail_by_salt_T[(salt, T)] += 1
                        continue
                    rel = abs(pred["xcW"] - r["xcW"]) / max(r["xcW"], 1e-9) * 100.0
                    all_rows.append({
                        "salt": salt, "T": T, "P": r["P"], "ms": r["ms"],
                        "mi": r["mi"], "author": author,
                        "xcW_exp": r["xcW"], "xcW_pred": pred["xcW"],
                        "rel_err_pct": rel, "tpd_min": pred["tpd_min"],
                        "z_CO2_used": pred["z_CO2_used"], "n_retries": pred["n_retries"],
                    })
                    AARE_by_salt_T_author[(salt, T, author)].append(rel)
                    AARE_by_salt_T[(salt, T)].append(rel)

    # --- Print summary ---------------------------------------------------------
    width = 110
    print("=" * width)
    print(f"{'salt':>6}  {'T [K]':>6}  {'author':<28} {'N':>4}  {'AARE [%]':>10}  {'max [%]':>10}")
    print("-" * width)
    for (salt, T, author), errs in sorted(AARE_by_salt_T_author.items()):
        print(f"{salt:>6}  {T:>6}  {author:<28} {len(errs):>4}  "
              f"{np.mean(errs):>10.2f}  {np.max(errs):>10.2f}")
    print("-" * width)
    print(f"{'salt':>6}  {'T [K]':>6}  {'aggregate over authors':<28} {'N':>4}  {'AARE [%]':>10}  {'fail':>10}")
    print("-" * width)
    for (salt, T), errs in sorted(AARE_by_salt_T.items()):
        nfail = fail_by_salt_T[(salt, T)]
        print(f"{salt:>6}  {T:>6}  {'<all authors>':<28} {len(errs):>4}  "
              f"{np.mean(errs):>10.2f}  {nfail:>10}")
    print("=" * width)
    overall = [r["rel_err_pct"] for r in all_rows]
    n_fail = sum(fail_by_salt_T.values())
    if overall:
        print(f"{'TOTAL':>6}  {'  ':>6}  {'all (salt, T, author)':<28} {len(overall):>4}  "
              f"{np.mean(overall):>10.2f}  {'fails:':>4} {n_fail}")
    print()

    # --- One figure per salt: 1 row × 3 columns (per T) --------------------
    for salt in salts:
        salt_rows = [r for r in all_rows if r["salt"] == salt]
        if not salt_rows:
            continue
        fig, axes = plt.subplots(1, len(temperatures),
                                 figsize=(4.0 * len(temperatures), 4.0),
                                 sharey=False)
        if len(temperatures) == 1:
            axes = [axes]
        for ax, T in zip(axes, temperatures):
            T_rows = [r for r in salt_rows if r["T"] == T]
            ms_levels = sorted({round(r["ms"], 3) for r in T_rows})
            colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(ms_levels), 2)))
            for k, ms in enumerate(ms_levels):
                sub = [r for r in T_rows if round(r["ms"], 3) == ms]
                P_exp = np.array([r["P"] for r in sub])
                mc_exp = np.array([r["xcW_exp"] / (1 - r["xcW_exp"]) / MW for r in sub])
                mc_pred = np.array([r["xcW_pred"] / (1 - r["xcW_pred"]) / MW for r in sub])
                order = np.argsort(P_exp)
                ax.plot(P_exp[order], mc_exp[order], 'o', color=colors[k],
                        markersize=4, label=f"exp m_s={ms:g}")
                ax.plot(P_exp[order], mc_pred[order], '-', color=colors[k],
                        linewidth=1.2)
            ax.set_xlabel("Pressure [bar]")
            ax.set_ylabel(r"$m_{\mathrm{CO_2}}$ [mol/kg H$_2$O]")
            ax.set_title(f"{salt} at T = {T} K")
            ax.legend(fontsize=6, ncol=2, loc="best")
            ax.grid(alpha=0.3)
        fig.suptitle(f"CO$_2$ in {salt} brine: ms_ecpa.stability_flash vs experiment",
                     y=1.02)
        fig.tight_layout()
        fig_path = out_dir / f"validate_co2_{salt}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path.relative_to(out_dir.parent)}")
        plt.close(fig)

    return all_rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--salts", nargs="+", default=list(SALT_FOLDERS),
                   help="Salt folder names (default: all).")
    p.add_argument("--temperatures", "-T", nargs="+", type=int, default=list(DEFAULT_T),
                   help="Temperatures (K) at which to validate.")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = p.parse_args()
    run_validation(args.salts, args.temperatures, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
