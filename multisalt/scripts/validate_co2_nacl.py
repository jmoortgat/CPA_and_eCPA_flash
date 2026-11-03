#!/usr/bin/env python3
"""Validate ``ms_ecpa.stability_flash`` against CO2-NaCl experimental solubility data.

For each (T, m_NaCl, P) point in the experimental files, this script:
  1. Constructs the partitioning-component feed z = (z_H2O, 0, z_CO2) with
     enough CO2 to ensure 2-phase equilibrium (any z_CO2 above saturation
     gives the same aqueous-phase mole fraction).
  2. Sets ion molalities mi = (m_NaCl, m_NaCl, 0, 0, 0, 0).
  3. Calls stability_flash(T, P, z, mi).
  4. Reads x_CO2 in the aqueous phase from result.flash_result.x_aq.
  5. Converts to the salt-free CO2 mole fraction the experiments report
     (xc_W_pred = x_CO2 / (x_H2O + x_CO2)) and compares to experiment.

Outputs:
  - AARE table to stdout, grouped by (T, salinity, author)
  - Comparison plot: model curves vs experimental points, saved to
    ``results/validate_co2_nacl_AAREs.png``

Usage:
    python3 scripts/validate_co2_nacl.py [--temperatures 323 373 423]

This is the first validation of the new flash/stability framework against
paper-relevant experimental data. The published paper's reported AAREs for
CO2-NaCl with the production parameter set (now consolidated in
``ms_ecpa.parameters`` -- see its module docstring for the provenance of
each physics term) should match these numbers within a few percent. Any
discrepancy is interesting and worth investigating.
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


MW = 0.018  # kg/mol  (water molar mass)


def parse_exp_file(path: Path) -> tuple[str, list[dict]]:
    """Read an EXP_<salt><k>_T<T>K.txt file.

    The third column varies between files: some report CO2 mole fraction
    on a salt-free basis (``xc_W``), others report CO2 molality (``mc``
    in mol/kg water). The header line tells us which. We always return
    rows with both ``xcW`` (CO2 salt-free mole fraction) populated, doing
    the molality -> mole fraction conversion when needed:

        xcW = m_CO2 * Mw / (1 + m_CO2 * Mw)

    Tabs-separated; first line is ``#AUTHOR(YEAR)``, second is a column
    header, remaining are data.
    """
    lines = path.read_text().splitlines()
    author = lines[0].lstrip("#").strip()
    # Detect column-3 convention from the header line.
    header = lines[1].lower() if len(lines) > 1 else ""
    # Look for "mc" or "molal" before the next token; otherwise assume mole fraction.
    col3_is_molality = (
        "mc" in header.split()[2:4] or "mol/kg" in header.split()[2:4]
    )
    # Heuristic backup: if header contains "mc [mol/kg]" anywhere in the third
    # field, treat as molality.
    if not col3_is_molality:
        toks = header.split()
        # toks roughly like: 'p', '[bar]', 'ms', '[mol/kg]', '<col3-name>', '[<unit>]', ...
        if len(toks) >= 6 and toks[4] in ("mc", "m_co2", "m_c"):
            col3_is_molality = True
    rows = []
    for ln in lines[2:]:
        if not ln.strip() or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            P = float(parts[0])
            ms = float(parts[1])
            v3 = float(parts[2])
        except ValueError:
            continue
        if col3_is_molality:
            # m_CO2 -> xc_W (salt-free mole fraction)
            xcW = v3 * MW / (1.0 + v3 * MW)
        else:
            xcW = v3
        rows.append({"P": P, "ms": ms, "xcW": xcW, "src": path.name,
                     "col3_units": "mc" if col3_is_molality else "xcW"})
    return author, rows


def collect_experimental_data(temperatures: list[int]) -> dict:
    """Walk EOS_DATA_ROOT and aggregate {(T): {author: [rows...]}}."""
    by_T = {}
    for T in temperatures:
        Tdir = EOS_DATA_ROOT / f"T{T}K"
        by_author: dict[str, list] = defaultdict(list)
        files = sorted(Tdir.glob("EXP_NaCl*_T*K.txt"))
        # Skip "_Xauthor"-suffixed redundant files; the simply-named files are
        # the canonical ones used by OPT_ALLIONS.py at fitting time.
        files = [f for f in files if "_X" not in f.stem]
        for f in files:
            author, rows = parse_exp_file(f)
            by_author[author].extend(rows)
        by_T[T] = dict(by_author)
    return by_T


def model_xcW(T: float, P_bar: float, m_NaCl: float, z_CO2: float = 0.05) -> dict:
    """Predict xc_W (salt-free CO2 mole fraction) at (T, P, m_NaCl).

    The feed z = (1-z_CO2, 0, z_CO2) is chosen aqueous-dominant: enough CO2
    to be safely 2-phase (typical x_CO2_aq across the validation range is
    0.005-0.025, so 0.05 is 2-5x above saturation), but not so much that the
    reference single-phase EoS becomes unphysical (e.g. z_CO2=0.5 fails the
    inner Newton at low pressure because no real gas-like root exists for a
    50:50 H2O/CO2 composition at 27 bar / 323 K).

    Returns a dict with keys: xcW, x_aq, phases, converged, tpd_min, t_solve,
    error (None or the exception message if the solve failed). ``converged``
    is the flash kernel's own verdict, not a restatement of ``phases == 2``.
    """
    z = np.array([1.0 - z_CO2, 0.0, z_CO2])
    mi = np.array([m_NaCl, m_NaCl, 0.0, 0.0, 0.0, 0.0])
    t0 = time.perf_counter()
    try:
        res = stability_flash(float(T), float(P_bar), z, mi)
    except Exception as exc:
        return dict(
            xcW=float("nan"), x_aq=None, phases=0,
            converged=False, tpd_min=float("nan"),
            t_solve=time.perf_counter() - t0,
            error=str(exc).splitlines()[0][:120],
        )
    dt = time.perf_counter() - t0
    if res.phases == 2 and res.flash_result is not None:
        xH, xC4, xC = res.flash_result.x_aq
        xcW = float(xC / (xH + xC))
        # phases == 2 does not by itself mean the flash succeeded. Every
        # early two-phase exit of stability_flash is gated on a converged
        # flash, but its terminal hard-failure exit also returns phases=2,
        # handing back the last SSI iterate: x_aq is finite and physically
        # plausible, so a diverged solve is indistinguishable from a good
        # one downstream unless the kernel's own flag is carried out here.
        conv = bool(res.flash_result.converged)
        return dict(
            xcW=xcW, x_aq=res.flash_result.x_aq, phases=2,
            converged=conv, tpd_min=res.tpd_min, t_solve=dt,
            error=None if conv else f"flash not converged: {res.message}",
        )
    return dict(
        xcW=float("nan"), x_aq=None, phases=res.phases,
        converged=False, tpd_min=res.tpd_min, t_solve=dt,
        error=f"phases={res.phases} (expected 2)",
    )


def validate(temperatures: list[int], plot_path: Path):
    """Run the validation sweep and produce summary + figure."""
    print(f"Loading experimental CO2-NaCl data for T in {temperatures} K ...")
    exp_by_T = collect_experimental_data(temperatures)

    # Per-(T, author, salinity) accumulators
    rows = []  # flat list for later analysis
    AARE_by_T_author = defaultdict(list)   # {(T, author): [|rel err| × 100]}
    AARE_by_T_ms = defaultdict(list)        # {(T, ms): [|rel err| × 100]}

    n_fail = 0
    n_nonconv = 0   # subset of n_fail rejected by the flash's convergence flag
    for T, by_author in exp_by_T.items():
        for author, exp_rows in by_author.items():
            for r in exp_rows:
                pred = model_xcW(T, r["P"], r["ms"])
                if not np.isfinite(pred["xcW"]) or pred["xcW"] <= 0:
                    n_fail += 1
                    continue
                # A two-phase result the flash did not converge on still
                # carries a finite, positive xcW, so the screen above lets
                # it through: without this the diverged iterate is averaged
                # into the AARE tables and never reaches the failure count.
                if not pred["converged"]:
                    n_fail += 1
                    n_nonconv += 1
                    continue
                rel = abs(pred["xcW"] - r["xcW"]) / r["xcW"] * 100.0
                rows.append({
                    "T": T, "P": r["P"], "ms": r["ms"],
                    "author": author, "xcW_exp": r["xcW"], "xcW_pred": pred["xcW"],
                    "rel_err_pct": rel, "tpd_min": pred["tpd_min"],
                    "t_solve": pred["t_solve"], "src": r["src"],
                })
                AARE_by_T_author[(T, author)].append(rel)
                AARE_by_T_ms[(T, round(r["ms"], 2))].append(rel)

    # Print summary
    print("\n" + "=" * 90)
    print(f"{'T [K]':>6}  {'author':<28} {'N':>4}  {'AARE [%]':>10}  {'max [%]':>10}")
    print("-" * 90)
    for (T, author), errs in sorted(AARE_by_T_author.items()):
        print(f"{T:>6}  {author:<28} {len(errs):>4}  {np.mean(errs):>10.2f}  {np.max(errs):>10.2f}")
    print("-" * 90)
    print(f"{'T [K]':>6}  {'ms [mol/kg]':<28} {'N':>4}  {'AARE [%]':>10}  {'max [%]':>10}")
    print("-" * 90)
    for (T, ms), errs in sorted(AARE_by_T_ms.items()):
        print(f"{T:>6}  {f'm_NaCl = {ms:g}':<28} {len(errs):>4}  {np.mean(errs):>10.2f}  {np.max(errs):>10.2f}")
    print("=" * 90)
    overall = [r["rel_err_pct"] for r in rows]
    if overall:
        print(f"{'OVERALL':>6}  {'all (T, salinity, author)':<28} {len(overall):>4}  "
              f"{np.mean(overall):>10.2f}  {np.max(overall):>10.2f}")
    print(f"\nFailures: {n_fail} of {n_fail + len(rows)} experimental points "
          f"({100*n_fail/max(n_fail+len(rows), 1):.1f}%)")
    if n_nonconv:
        print(f"  of which non-converged two-phase flashes: {n_nonconv}")
    print()

    # Comparison figure: m_CO2_aq molality vs P, one panel per T
    Mw = 0.018  # kg/mol H2O
    fig, axes = plt.subplots(1, len(temperatures), figsize=(4.0 * len(temperatures), 4.0),
                             sharey=False)
    if len(temperatures) == 1:
        axes = [axes]
    for ax, T in zip(axes, temperatures):
        T_rows = [r for r in rows if r["T"] == T]
        ms_levels = sorted({round(r["ms"], 3) for r in T_rows})
        colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(ms_levels), 2)))
        for k, ms in enumerate(ms_levels):
            sub = [r for r in T_rows if round(r["ms"], 3) == ms]
            P_exp = np.array([r["P"] for r in sub])
            mc_exp = np.array([
                r["xcW_exp"] / (1 - r["xcW_exp"]) / Mw for r in sub
            ])
            mc_pred = np.array([
                r["xcW_pred"] / (1 - r["xcW_pred"]) / Mw for r in sub
            ])
            order = np.argsort(P_exp)
            ax.plot(P_exp[order], mc_exp[order], 'o', color=colors[k],
                    markersize=4, label=f"exp m={ms:g}")
            ax.plot(P_exp[order], mc_pred[order], '-', color=colors[k],
                    linewidth=1.2)
        ax.set_xlabel("Pressure [bar]")
        ax.set_ylabel(r"$m_{\mathrm{CO_2}}$ [mol/kg H$_2$O]")
        ax.set_title(f"T = {T} K   (lines = stability_flash, points = exp)")
        ax.legend(fontsize=7, ncol=2, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle("CO$_2$ solubility in NaCl brine: ms_ecpa.stability_flash vs experiment",
                 y=1.02)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved: {plot_path}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--temperatures", "-T", nargs="+", type=int,
                   default=[323, 373, 423],
                   help="Temperatures (K) at which to validate.")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                        / "results" / "validate_co2_nacl_AAREs.png"))
    args = p.parse_args()
    validate(args.temperatures, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
