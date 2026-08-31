"""
Benchmark warm-start strategies for flash_co2_h2o_salt_kv.

Compares two initialisation strategies:
  cold  — default cold start (4 K-value candidate pairs)
  table — ScanTableWarmStart interpolated from the precomputed scan table

Metrics reported per strategy:
  - Convergence rate (fraction of points that converge)
  - SSI iteration count distribution (median, 90th pct, max)
  - Wall-clock time per call (median, 90th pct)
  - Composition error vs. cold-start reference (AARE on x4w, x1c)
  - Fallback rate: fraction where warm-start K failed → used cold-start candidates

Usage (from the code/ directory)
--------------------------------
    PYTHONPATH=. python scripts/benchmark_warmstart.py            # 500 random points
    PYTHONPATH=. python scripts/benchmark_warmstart.py --n 2000   # more points
"""
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from ecpa.flash import flash_co2_h2o_salt_kv
from ecpa.parameters import make_params
from ecpa.warmstart import ScanTableWarmStart


# ── Grid bounds ────────────────────────────────────────────────────────────────

T_RANGE  = (293., 628.)
P_RANGE  = (2., 1400.)          # bar
Z_RANGE  = (0.05, 0.90)
MS_RANGE = (1e-5, 6.0)          # mol/kg


def _sample_conditions(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample n random (T, P, z, ms) conditions."""
    T  = rng.uniform(*T_RANGE,  n)
    lP = rng.uniform(np.log10(P_RANGE[0]), np.log10(P_RANGE[1]), n)
    P  = 10.0 ** lP
    z  = rng.uniform(*Z_RANGE,  n)
    ms = 10.0 ** rng.uniform(np.log10(MS_RANGE[0]), np.log10(MS_RANGE[1]), n)
    return pd.DataFrame({"T": T, "P": P, "z": z, "ms": ms})


def _run_strategy(df: pd.DataFrame, params, warm_start=None, label="cold") -> pd.DataFrame:
    """
    Run flash_co2_h2o_salt_kv on every row of df.

    Returns a DataFrame with columns:
        converged, n_iter, t_ms, x4w, x1c, source
    """
    records = []
    for row in df.itertuples(index=False):
        t0 = time.perf_counter()
        try:
            r = flash_co2_h2o_salt_kv(
                T=row.T, P_bar=row.P, z_co2=row.z, m_tot=row.ms,
                warm_start=warm_start, params=params,
            )
            t_ms = (time.perf_counter() - t0) * 1000
            records.append({
                "converged": True,
                "n_iter":    int(r["n_iter_ms"]),
                "t_ms":      t_ms,
                "x4w":       float(r["x_aq"]["x4w"]),
                "x1c":       float(r["x_c"]["x1c"]),
            })
        except Exception:
            t_ms = (time.perf_counter() - t0) * 1000
            records.append({
                "converged": False,
                "n_iter":    -1,
                "t_ms":      t_ms,
                "x4w":       float("nan"),
                "x1c":       float("nan"),
            })
    return pd.DataFrame(records)


def _print_summary(label: str, df: pd.DataFrame, ref: pd.DataFrame | None = None):
    ok  = df["converged"]
    n   = len(df)
    nok = ok.sum()
    print(f"\n{'─'*60}")
    print(f"  Strategy : {label}")
    print(f"  Points   : {n}  (converged: {nok} = {100*nok/n:.1f}%)")
    if nok == 0:
        return
    iters = df.loc[ok, "n_iter"]
    times = df.loc[ok, "t_ms"]
    print(f"  SSI iters: median={iters.median():.1f}  p90={np.percentile(iters,90):.1f}  max={iters.max()}")
    print(f"  Time/call: median={times.median():.2f}ms  p90={np.percentile(times,90):.2f}ms")
    if ref is not None:
        both = ok & ref["converged"]
        if both.sum() > 0:
            dx4w = ((df.loc[both, "x4w"] - ref.loc[both, "x4w"]).abs()
                    / ref.loc[both, "x4w"].clip(1e-10)).dropna()
            dx1c = ((df.loc[both, "x1c"] - ref.loc[both, "x1c"]).abs()
                    / ref.loc[both, "x1c"].clip(1e-10)).dropna()
            print(f"  AARE x4w : {100*dx4w.mean():.4f}%  (vs cold-start reference)")
            print(f"  AARE x1c : {100*dx1c.mean():.4f}%")


def _plot(results: dict, out_path: str):
    labels   = list(results.keys())
    colors   = {"cold": "#555", "table": "#2196F3"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1: SSI iteration CDF
    ax = axes[0]
    for lbl, df in results.items():
        ok = df.loc[df["converged"], "n_iter"]
        vals = np.sort(ok.values)
        ax.step(vals, np.arange(1, len(vals)+1) / len(vals),
                label=lbl, color=colors.get(lbl, "gray"), linewidth=1.8)
    ax.set_xlabel("SSI iterations")
    ax.set_ylabel("CDF")
    ax.set_title("Iteration count CDF")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: wall-clock time distribution (box)
    ax = axes[1]
    data  = [results[lbl].loc[results[lbl]["converged"], "t_ms"].values for lbl in labels]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, lbl in zip(bp["boxes"], labels):
        patch.set_facecolor(colors.get(lbl, "#888"))
    ax.set_ylabel("Wall-clock time (ms)")
    ax.set_title("Time per call")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: iteration savings vs cold
    ax = axes[2]
    ref_iters = results["cold"].loc[results["cold"]["converged"], "n_iter"].values
    for lbl, df in results.items():
        if lbl == "cold":
            continue
        ok  = df["converged"] & results["cold"]["converged"]
        delta = (results["cold"].loc[ok, "n_iter"].values
                 - df.loc[ok, "n_iter"].values)
        counts, edges = np.histogram(delta, bins=np.arange(delta.min()-0.5, delta.max()+1.5))
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centres + (0.2 if lbl == "table" else -0.2),
               counts / counts.sum(),
               width=0.35, alpha=0.8, label=lbl,
               color=colors.get(lbl, "gray"))
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("SSI iters saved vs cold start")
    ax.set_ylabel("Fraction of points")
    ax.set_title("Iteration savings")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Warm-start strategy comparison", fontweight="bold")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",     type=int,  default=500,
                    help="Number of random test points (default 500)")
    ap.add_argument("--seed",  type=int,  default=42)
    ap.add_argument("--table", default="results/scan_v4_table.npz")
    ap.add_argument("--out",   default="figures/warmstart_bench.png")
    args = ap.parse_args()

    rng    = np.random.default_rng(args.seed)
    params = make_params()

    # Sample conditions
    df_all = _sample_conditions(args.n, rng)
    print(f"Sampled {args.n} random conditions")
    print(f"  T  : {df_all['T'].min():.0f}–{df_all['T'].max():.0f} K")
    print(f"  P  : {df_all['P'].min():.1f}–{df_all['P'].max():.0f} bar")
    print(f"  z  : {df_all['z'].min():.3f}–{df_all['z'].max():.3f}")
    print(f"  ms : {df_all['ms'].min():.4f}–{df_all['ms'].max():.2f} mol/kg")

    results = {}

    # ── Cold start ─────────────────────────────────────────────────────────
    print("\nRunning cold-start benchmark …")
    t0 = time.time()
    results["cold"] = _run_strategy(df_all, params, warm_start=None, label="cold")
    print(f"  done in {time.time()-t0:.1f}s")
    _print_summary("cold", results["cold"])

    # ── Table warm-start ───────────────────────────────────────────────────
    print("\nLoading scan table …")
    ws_tab = ScanTableWarmStart.load(args.table)
    print("Running table warm-start benchmark …")
    t0 = time.time()
    results["table"] = _run_strategy(df_all, params, warm_start=ws_tab, label="table")
    print(f"  done in {time.time()-t0:.1f}s")
    _print_summary("table", results["table"], ref=results["cold"])

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'strategy':<10} {'conv%':>6} {'iter_med':>8} {'iter_p90':>8} {'t_med_ms':>9} {'t_p90_ms':>9}")
    for lbl, df in results.items():
        ok = df["converged"]
        if ok.sum() == 0:
            continue
        iters = df.loc[ok, "n_iter"]
        times = df.loc[ok, "t_ms"]
        print(f"{lbl:<10} {100*ok.mean():>6.1f} {iters.median():>8.1f} "
              f"{np.percentile(iters,90):>8.1f} {times.median():>9.2f} "
              f"{np.percentile(times,90):>9.2f}")

    _plot(results, args.out)


if __name__ == "__main__":
    main()
