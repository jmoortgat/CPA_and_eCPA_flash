"""C2 binary refit round: artifact screen + multi-start quadratic + cubic.

Diagnosis (2026-08-10): the single-start quadratic kij(T) fit leaves a
monotone T-trend in the low-P (Henry-regime) residuals, +10% (283 K) to
-52% (444 K), while the high-P plateau fits at 3.6% -- the plateau points
are kij-insensitive and dominated the fit. Also, the 377-444 K low-P
points sit near the dew boundary where the pure-water dew-line-rescue
degeneracy (found at < 15 bar, ~300-350 K) may extend upward in P.

Stages:
  probe   feed-independence screen of ALL water points at T >= 370 K or
          P < 3*Psat_w(T) heuristic band, with the CURRENT fitted kij.
  fitQ    quadratic refit, 4 starts, degenerate points excluded.
  fitC    cubic refit (evaluated pointwise: kij value passed as the
          constant coefficient, Ak = Bk = 0), 4 starts.
  report  per-source and per-isotherm AAREs for both, decision table.

The cubic form: k12(T) = a3*Tr^3 + a2*Tr^2 + a1*Tr + a0, Tr = T/Tc1.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2_pilot as cp                      # noqa: E402

STATE = HERE / "refit_state.json"
N_WORKERS = 8


def _predict_kval(task):
    """(k_value, T, P_bar) -> x2 with constant kij = k_value."""
    k, T, P = task
    return cp._predict_x2((0.0, 0.0, k, 1.0, cp.POL_C2, T, P, cp.MI_WATER))


def _feed_check(task):
    """(kij3, T, P_bar) -> (x2_a, x2_b) at two feeds for degeneracy check."""
    kij, T, P = task
    a = cp._predict_x2((*kij, 1.0, cp.POL_C2, T, P, cp.MI_WATER))
    # second feed: shift the retry ladder by starting from zg=0.2
    import ms_ecpa.parameters as prm
    from ms_ecpa import stability_flash
    try:
        cp.rebind_c2(*kij)
        res = stability_flash(float(T), float(P),
                              np.array([0.8, 0.2, 0.0]), np.zeros(6))
        b = (float(res.flash_result.x_aq[1])
             if res.phases == 2 and res.flash_result is not None
             else float("nan"))
    except Exception:
        b = float("nan")
    finally:
        cp.restore_ch4()
    return (a, b)


def psat_w(T):
    """Rough Antoine water Psat [bar] for the screen heuristic."""
    return 10 ** (5.11564 - 1687.537 / (T - 42.98 + 273.15 - 273.15 + 230.17)) \
        if False else np.exp(11.6834 - 3816.44 / (T - 46.13))


def stage_probe(st):
    df = cp.load_water()
    kij = json.load(open(cp.STATE_JSON))["fitB"]["best"]["x"]
    sus = df[(df.T_K >= 370.0) & (df.P_MPa * 10 < 30.0)]
    print(f"screening {len(sus)} suspicious points (T>=370 K, P<30 bar)")
    bad = []
    for r in sus.itertuples():
        a, b = _feed_check((tuple(kij), r.T_K, r.P_MPa * 10))
        ok = np.isfinite([a, b]).all() and abs(a / b - 1) < 1e-6
        print(f"  T={r.T_K} P={r.P_MPa*10:.1f} bar: ok={ok} "
              f"x2a={a:.3e} x2b={b:.3e}", flush=True)
        if not ok:
            bad.append([float(r.T_K), float(r.P_MPa)])
    st["degenerate_extra"] = bad
    print(f"additional degenerate points: {len(bad)}")


def _masked_df(st):
    df = cp.load_water()
    bad = {(round(t, 2), round(p, 3)) for t, p in st.get("degenerate_extra", [])}
    mask = [(round(r.T_K, 2), round(r.P_MPa, 3)) not in bad
            for r in df.itertuples()]
    return df[mask].reset_index(drop=True)


def _predict_poly(df, coeffs):
    """Predict all points with kij(T) = polyval(coeffs, T/Tc1)."""
    import ms_ecpa.parameters as prm
    Tc1 = prm.Tc1
    tasks = [(float(np.polyval(coeffs, r.T_K / Tc1)), r.T_K, r.P_MPa * 10.0)
             for r in df.itertuples()]
    if cp._POOL is not None:
        out = cp._POOL.map(_predict_kval, tasks)
    else:
        out = [_predict_kval(t) for t in tasks]
    return np.array(out)


def _fit(df, starts, label):
    from scipy.optimize import least_squares
    exp = df["x_molefrac"].to_numpy()

    def resid(x):
        pred = _predict_poly(df, x)
        r = pred / exp - 1.0
        return np.where(np.isfinite(r), r, 5.0)

    best, runs = None, []
    for i, x0 in enumerate(starts):
        t0 = time.perf_counter()
        res = least_squares(resid, x0, diff_step=1e-3, method="trf",
                            max_nfev=60)
        run = {"start": list(map(float, x0)), "x": list(map(float, res.x)),
               "cost": float(res.cost), "nfev": int(res.nfev),
               "status": int(res.status),
               "rms_pct": float(np.sqrt(np.mean(res.fun**2)) * 100),
               "wall_s": round(time.perf_counter() - t0, 1)}
        runs.append(run)
        print(f"  [{label} s{i+1}] {run['start']} -> {run['x']} "
              f"rms={run['rms_pct']:.1f}% nfev={run['nfev']} "
              f"({run['wall_s']}s)", flush=True)
        if best is None or run["cost"] < best["cost"]:
            best = run
    return best, runs


def stage_fitQ(st):
    df = _masked_df(st)
    print(f"fitting quadratic on {len(df)} points")
    ch4 = [-4.0935, 5.6634, -1.7845]
    prev = json.load(open(cp.STATE_JSON))["fitB"]["best"]["x"]
    starts = [prev, ch4, [prev[0]*2.0, prev[1]*2.0, prev[2]*2.0],
              [0.0, 1.0, -0.3]]
    best, runs = _fit(df, starts, "fitQ")
    st["fitQ"] = {"best": best, "runs": runs, "n_points": len(df)}


def stage_fitC(st):
    df = _masked_df(st)
    print(f"fitting cubic on {len(df)} points")
    q = st["fitQ"]["best"]["x"]
    starts = [[0.0] + list(q),
              [5.0, q[0]-3.0, q[1]+1.0, q[2]],
              [-5.0, q[0]+3.0, q[1]-1.0, q[2]],
              [2.0, -8.0, 9.0, -2.9]]
    best, runs = _fit(df, starts, "fitC")
    st["fitC"] = {"best": best, "runs": runs, "n_points": len(df)}


def stage_report(st):
    df = _masked_df(st)
    exp = df["x_molefrac"].to_numpy()
    out = {}
    for tag in ("fitQ", "fitC"):
        coeffs = st[tag]["best"]["x"]
        pred = _predict_poly(df, coeffs)
        rel = pred / exp - 1.0
        ok = np.isfinite(rel)
        res = {"overall_aare_pct":
               float(np.mean(np.abs(rel[ok])) * 100)}
        for src in ("CM1950lo", "CM1950hi", "Mohammadi2004"):
            m = (df.source == src).to_numpy() & ok
            res[src] = float(np.mean(np.abs(rel[m])) * 100)
        iso = {}
        for T, g in df.groupby(df.T_K.round(0)):
            m = df.index.isin(g.index).astype(bool) & ok
            iso[str(int(T))] = round(float(np.mean(rel[m]) * 100), 1)
        res["per_isotherm_bias_pct"] = iso
        out[tag] = res
        print(f"{tag}: {json.dumps(res, indent=2)}", flush=True)
    st["report"] = out


STAGES = {"probe": stage_probe, "fitQ": stage_fitQ,
          "fitC": stage_fitC, "report": stage_report}


def main():
    import argparse
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES) + ["all"])
    args = ap.parse_args()
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    for name in stages:
        if name in ("fitQ", "fitC", "report") and cp._POOL is None:
            cp._POOL = mp.get_context("spawn").Pool(N_WORKERS)
        t0 = time.perf_counter()
        print(f"=== {name} ===", flush=True)
        STAGES[name](st)
        STATE.write_text(json.dumps(st, indent=2))
        print(f"=== {name} done ({time.perf_counter()-t0:.1f}s) ===",
              flush=True)
    if cp._POOL is not None:
        cp._POOL.close(); cp._POOL.join()


if __name__ == "__main__":
    main()
