"""
Topup: add fine P-grid points near the lower two-phase boundary for high-T isotherms.

At T >= ~503 K the lower two-phase boundary is the water saturation pressure
P_sat(H2O, T) (ranging from ~33 bar at 513 K to ~165 bar at 623 K).
The existing 200-point log-spaced grid has only 2-4 points in the steep onset
region just above P_sat, causing a visible kink in log-log plots.  This script
adds N_FINE=25 extra points in [P_sat, P_sat*1.25] per isotherm and merges
them into the existing parquets.
"""
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Import from main script ────────────────────────────────────────────────────
from _run_smooth_co2h2o_robust import (
    _robust_T_worker, _make_figures, _ms_tag, MS_RIBBON, N_WORKERS,
)

HIGH_T_MIN = 500    # K: only topup for T >= this
N_FINE     = 25     # extra P points near lower boundary
OUT_DIR    = 'figures/co2h2o_ws'


def psat_h2o(T_K):
    """Water saturation pressure [bar] — IAPWS-IF97 equation, valid 273–647 K."""
    theta = 1.0 - T_K / 647.096          # reduced temperature distance from Tc
    if theta <= 0:
        return np.nan
    logP = (647.096 / T_K) * (
        -7.85951783 * theta
        + 1.84408259 * theta**1.5
        - 11.7866497 * theta**3
        + 22.6807411 * theta**3.5
        - 15.9618719 * theta**4
        + 1.80122502 * theta**7.5
    )
    return np.exp(logP) * 220.64  # P_c=220.64 bar for water


def get_boundary_P(ms0_df, T):
    """Return (P_last_failed, P_first_converged) for temperature T from ms=1e-5 parquet."""
    sub  = ms0_df[np.abs(ms0_df['T_K'] - T) < 0.5].sort_values('P_bar')
    conv = sub[sub['ecpa_converged']]
    fail = sub[~sub['ecpa_converged']]
    P_fc = float(conv['P_bar'].min()) if len(conv) > 0 else np.nan
    P_lf = float(fail[fail['P_bar'] < P_fc]['P_bar'].max()) if (len(fail) > 0 and not np.isnan(P_fc)) else np.nan
    return P_lf, P_fc


if __name__ == '__main__':
    # ── Load reference parquet (ms=1e-5) to identify boundaries ─────────────
    ref_cache = Path('results/ws2_smooth_co2h2o_ms0.0000.parquet')
    if not ref_cache.exists():
        raise FileNotFoundError(f"Missing {ref_cache} — run _run_smooth_co2h2o_robust.py first")
    ref_df = pd.read_parquet(ref_cache)

    exp_df   = pd.read_parquet('CO2_WATER_exp.parquet')
    from ecpa.validate_co2h2o import T_MAX_ECPA
    T_unique = sorted(exp_df['T_K'].unique())
    T_list   = [T for T in T_unique if T <= T_MAX_ECPA]

    # ── Identify topup P grids ────────────────────────────────────────────────
    topup_tasks = []  # list of (T, P_new_array)

    for T in T_list:
        if T < HIGH_T_MIN:
            continue

        P_lf, P_fc = get_boundary_P(ref_df, T)
        if np.isnan(P_fc):
            print(f"T={T:.0f}K: no converged points, skipping")
            continue

        # Estimate P_sat from the parquet boundary; fall back to formula
        P_sat_est = P_lf if not np.isnan(P_lf) else psat_h2o(T)
        if np.isnan(P_sat_est) or P_sat_est <= 0:
            P_sat_est = P_fc * 0.96   # crude fallback

        # Fine grid from just below P_sat to 1.25×P_sat
        P_lo  = max(P_sat_est * 0.97, 0.5)
        P_hi  = P_sat_est * 1.25
        P_new = np.logspace(np.log10(P_lo), np.log10(P_hi), N_FINE)

        # Drop P values already present in the parquet (within 0.05 bar)
        existing_P = ref_df[np.abs(ref_df['T_K'] - T) < 0.5]['P_bar'].values
        P_keep = np.array([
            p for p in P_new
            if not np.any(np.abs(p - existing_P) < 0.05)
        ])
        if len(P_keep) == 0:
            print(f"T={T:.0f}K: all fine P already in parquet, skipping")
            continue

        topup_tasks.append((T, P_keep))
        print(f"T={T:.0f}K: +{len(P_keep)} pts in [{P_keep.min():.1f}, {P_keep.max():.1f}] bar "
              f"(P_sat_est≈{P_sat_est:.1f} bar, P_fc={P_fc:.1f})")

    if not topup_tasks:
        print("Nothing to topup.")
    else:
        print(f"\n=== Running {len(topup_tasks)} topup workers ===")
        import time
        t0 = time.perf_counter()

        new_cpa_rows  = []
        new_ecpa_rows = {ms: [] for ms in MS_RIBBON}

        with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(topup_tasks))) as pool:
            futs = {pool.submit(_robust_T_worker, a): a[0] for a in topup_tasks}
            for fut in as_completed(futs):
                T_done = futs[fut]
                T_val, cpa_recs, ecpa_by_ms = fut.result()
                print(f"  T={T_val:.0f}K done ({len(cpa_recs)} new P pts)", flush=True)
                new_cpa_rows.extend(cpa_recs)
                for ms in MS_RIBBON:
                    new_ecpa_rows[ms].extend(ecpa_by_ms[ms])

        print(f"\nCompute done in {time.perf_counter()-t0:.1f}s")

        # ── Merge new rows into parquets ─────────────────────────────────────
        # CPA parquet
        cpa_cache = Path('results/ws2_cpa_smooth.parquet')
        if new_cpa_rows:
            new_cpa_df = pd.DataFrame(new_cpa_rows)
            if cpa_cache.exists():
                old = pd.read_parquet(cpa_cache)
                # Remove any rows that overlap (same T_K, P_bar)
                key = ['T_K', 'P_bar']
                overlap = old.merge(new_cpa_df[key], on=key, how='inner')
                old = old[~old.set_index(key).index.isin(overlap.set_index(key).index)]
                combined = pd.concat([old, new_cpa_df], ignore_index=True)
            else:
                combined = new_cpa_df
            combined = combined.sort_values(['T_K', 'P_bar'])
            combined.to_parquet(cpa_cache, index=False)

        # eCPA parquets
        for ms in MS_RIBBON:
            if not new_ecpa_rows[ms]:
                continue
            tag   = _ms_tag(ms)
            cache = Path(f'results/ws2_smooth_co2h2o_ms{tag}.parquet')
            new_df = pd.DataFrame(new_ecpa_rows[ms])

            # Merge CPA columns into new eCPA rows
            if new_cpa_rows:
                cpa_full = pd.read_parquet(cpa_cache)
                new_df = new_df.merge(
                    cpa_full[['T_K', 'P_bar', 'cpa_xc_W', 'cpa_yw_C',
                              'cpa_converged', 'cpa_n_iter', 'cpa_t_ms']],
                    on=['T_K', 'P_bar'], how='left'
                )

            if cache.exists():
                old = pd.read_parquet(cache)
                key = ['T_K', 'P_bar']
                overlap = old.merge(new_df[key], on=key, how='inner')
                old = old[~old.set_index(key).index.isin(overlap.set_index(key).index)]
                combined = pd.concat([old, new_df], ignore_index=True)
            else:
                combined = new_df
            combined = combined.sort_values(['T_K', 'P_bar'])
            for drop_col in ['K1', 'K4']:
                if drop_col in combined.columns:
                    combined = combined.drop(columns=[drop_col])
            combined.to_parquet(cache, index=False)
            print(f"  Updated {cache}")

        # ── Regenerate figures ────────────────────────────────────────────────
        print(f"\n=== Regenerating figures → {OUT_DIR}/ ===")
        smooth_data = {}
        for ms in MS_RIBBON:
            tag   = _ms_tag(ms)
            cache = Path(f'results/ws2_smooth_co2h2o_ms{tag}.parquet')
            if cache.exists():
                smooth_data[ms] = pd.read_parquet(cache)

        val_cache  = Path('results/ws_validation_co2h2o.parquet')
        results_df = pd.read_parquet(val_cache) if val_cache.exists() else None

        T_topup = [t for t, _ in topup_tasks]
        _make_figures(smooth_data, results_df, exp_df, T_topup, out_dir=OUT_DIR)
        print("Done.")
