"""
Robust CO2-H2O smooth-curve computation — no post-processing hacks.

Design principles
-----------------
1.  Always run full ecpa_stability (6 trial compositions) to determine phase
    state.  Never skip stability based on a table hint.
2.  If a feed z_co2 is single-phase, cycle through Z_CANDIDATES until a
    two-phase feed is found.  K-values are z-independent, so any two-phase
    z gives the correct equilibrium compositions.
3.  P-scan warm-start: carry (K1, K4, sol_aq_x0, sol_c_x0) from the most
    recent two-phase result at the same ms to initialise the next P.
4.  ms-ladder warm-start: at each P, seed K from the converged solution at
    the previous (lower) ms.  For ms=0 seed use CPA K-values.
5.  Stability trial K-values: when no prior warm-start is available, use
    (x1c_trial, x1w_trial) from ecpa_stability as K-value initial guess.
6.  No outlier removal.  If results look wrong the physics or numerics
    should be fixed, not filtered.

Parallelism
-----------
One worker per temperature value.  Within each worker all ms values are
computed in ladder order so the ms warm-start is handled in-process.
With N_WORKERS=8 and ~35 T values, workers are kept busy throughout.

Output
------
  results/ws2_cpa_smooth.parquet          — CPA (200 P pts, all T)
  results/ws2_smooth_co2h2o_ms*.parquet   — CPA + eCPA per ms (200 P pts)
  figures/co2h2o_ws2/T*K.png             — per-T ribbon figures

The output format is identical to _run_warmstart_co2h2o.py so the ribbon-
plot function can be reused unchanged.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Constants ──────────────────────────────────────────────────────────────────

N_WORKERS = 8
N_P       = 200
P_MIN     = 1.0
P_MAX     = 1500.0

MS_RIBBON   = [1e-5, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
Mw          = 0.018          # kg/mol  H₂O

# Feed compositions to try, in priority order, to find two-phase region.
# Small z values come first: the bubble-point z is always very small
# (~0.001-0.01) so z=0.05 and z=0.1 are almost always well inside the
# two-phase region.  Large z values (z>0.3) risk being above the dew
# curve, which makes the flash ill-conditioned → wrong K-values.
Z_CANDIDATES = [0.05, 0.1, 0.3, 0.5, 0.7, 0.85, 0.01]

# Minimum TPD magnitude to attempt a flash.  z values near the phase
# boundary (|tpd| < threshold) give ill-conditioned Rachford-Rice
# equations and can converge to false stationary points.
TPD_MIN = -0.005


def _ms_tag(ms_val):
    return f'{ms_val:.4f}' if ms_val < 0.1 else f'{ms_val:.1f}'


# ── Worker ─────────────────────────────────────────────────────────────────────

def _robust_T_worker(args):
    """
    For one temperature T, compute CPA and all eCPA ms values over P_grid.

    Returns
    -------
    T : float
    cpa_records : list[dict]   — one entry per P
    ecpa_by_ms  : dict  ms -> list[dict]
    """
    import warnings; warnings.filterwarnings('ignore')
    import time as _time

    T, P_grid = args

    from ecpa.validate_co2h2o import run_cpa_binary, Z_CO2_RETRY
    from ecpa.flash import flash_co2_h2o_salt_kv
    from ecpa.stability import ecpa_stability
    from ecpa.parameters import make_params

    params = make_params()

    # ── Step 1: CPA sweep ──────────────────────────────────────────────────
    cpa_records = []
    cpa_by_P    = {}   # P -> {'K1': float, 'K4': float, 'x4w': float, 'x1c': float}

    for P in P_grid:
        cpa = run_cpa_binary(T, P, z_co2=0.5)
        if np.isnan(cpa['xc_W']):
            for z_r in Z_CO2_RETRY:
                cpa_r = run_cpa_binary(T, P, z_co2=z_r)
                if not np.isnan(cpa_r['xc_W']):
                    cpa = cpa_r
                    break

        x4w = cpa['xc_W']   # CO2 in aqueous
        x1c = cpa['yw_C']   # H2O in CO2-rich
        if not (np.isnan(x4w) or np.isnan(x1c)):
            x1w = 1.0 - x4w          # ms=0 → no salt, x1w + x4w = 1
            x4c = 1.0 - x1c
            K1  = x1c / max(x1w, 1e-12)
            K4  = x4c / max(x4w, 1e-12)
            cpa_by_P[float(P)] = dict(K1=K1, K4=K4, x4w=x4w, x1c=x1c)

        cpa_records.append({
            'T_K': float(T), 'P_bar': float(P),
            'cpa_xc_W':      x4w,
            'cpa_yw_C':      x1c,
            'cpa_converged': cpa.get('converged', not np.isnan(x4w)),
            'cpa_n_iter':    int(cpa.get('n_iter', 0)),
            'cpa_t_ms':      float(cpa.get('t_ms', 0.0)),
        })

    # ── Step 2: eCPA ms-ladder ─────────────────────────────────────────────
    # prev_ms_by_P: at each P, the warm-start state from the PREVIOUS ms.
    # Keys: P (float), Values: dict with K1, K4, sol_aq, sol_c (may be None).
    prev_ms_by_P = {}
    for P in P_grid:
        ws = cpa_by_P.get(float(P))
        if ws is not None:
            prev_ms_by_P[float(P)] = dict(K1=ws['K1'], K4=ws['K4'],
                                           sol_aq=None, sol_c=None)

    ecpa_by_ms = {}

    for ms in MS_RIBBON:
        records = []
        ms_eff  = max(ms, 1e-9)   # avoid exactly zero for eCPA

        # P-scan warm-start: carry state from the most recent two-phase P.
        ws_prev_P = None   # dict(K1, K4, sol_aq, sol_c) or None

        for P in P_grid:
            t0   = _time.perf_counter()
            P_f  = float(P)

            # Build the K warm-start priority list:
            #  1. Previous P (same ms) — highest priority for smooth scan
            #  2. Previous ms at same P — second priority
            #  3. Stability trial — last resort (computed on demand)
            K_from_prev_P  = ws_prev_P
            K_from_prev_ms = prev_ms_by_P.get(P_f)

            result = None
            stab_results_cache = {}   # z -> stab result, to avoid recomputing

            for z in Z_CANDIDATES:
                # Run stability for this z (cached if already done)
                if z not in stab_results_cache:
                    try:
                        stab_results_cache[z] = ecpa_stability(
                            z, ms_eff, T, P, params)
                    except Exception:
                        stab_results_cache[z] = None

                stab = stab_results_cache[z]
                if stab is None or stab['stable']:
                    continue   # single-phase at this z, try next
                if stab['tpd_min'] > TPD_MIN:
                    continue   # barely two-phase → near dew/bubble curve
                               # → ill-conditioned flash → skip

                # Two-phase found at this z.  Build K warm-start.
                if K_from_prev_P is not None:
                    K_init   = (K_from_prev_P['K1'], K_from_prev_P['K4'])
                    sol_aq   = K_from_prev_P['sol_aq']
                    sol_c    = K_from_prev_P['sol_c']
                elif K_from_prev_ms is not None:
                    K_init   = (K_from_prev_ms['K1'], K_from_prev_ms['K4'])
                    sol_aq   = K_from_prev_ms.get('sol_aq')
                    sol_c    = K_from_prev_ms.get('sol_c')
                else:
                    # Use stability trial compositions as K estimate
                    x1c_t = stab['x1c_trial']
                    x1w_t = stab['x1w_trial']
                    x4w_t = max(1.0 - x1w_t * (1.0 + 2.0 * ms_eff * Mw), 1e-9)
                    x4c_t = 1.0 - x1c_t
                    K_init = (x1c_t / max(x1w_t, 1e-12),
                              x4c_t / max(x4w_t, 1e-12))
                    sol_aq = None
                    sol_c  = None

                try:
                    r = flash_co2_h2o_salt_kv(
                        T=T, P_bar=P, z_co2=z, m_tot=ms_eff,
                        K_init=K_init,
                        sol_aq_x0=sol_aq,
                        sol_c_x0=sol_c,
                        params=params,
                    )
                    result = r
                    break   # converged: no need to try other z candidates

                except RuntimeError:
                    # This z + K_init failed.  Clear P warm-start to avoid
                    # carrying bad state, then try next z with fresh K.
                    K_from_prev_P = None
                    continue

            # ── K4 consistency rescue ──────────────────────────────────────
            # At high T/P, z ≥ 0.5 can converge to a spurious SSI fixed point
            # where K4 increases with P (physically wrong: CO2 solubility in
            # water increases with P, so x4w ↑ and K4 ↓).  If K4 jumped up
            # by more than 15% from the previous P, retry with small z values
            # (z=0.10, z=0.05) which converge to the physical fixed point.
            # Use K_init from prev_P but clear sol_c/sol_aq (inner warm-start
            # was computed at a different composition and may misdirect the
            # CO2-rich phase solver to a wrong Z root).
            if (result is not None and ws_prev_P is not None
                    and result['K_vals'][1] > ws_prev_P['K4'] * 1.15):
                K4_suspect = result['K_vals'][1]
                for z_rescue in [0.10, 0.05, 0.30, 0.01]:
                    K_rescue = (ws_prev_P['K1'], ws_prev_P['K4'])
                    try:
                        r_rescue = flash_co2_h2o_salt_kv(
                            T=T, P_bar=P, z_co2=z_rescue, m_tot=ms_eff,
                            K_init=K_rescue,
                            sol_aq_x0=None,
                            sol_c_x0=None,
                            params=params,
                        )
                        K4_rescue = r_rescue['K_vals'][1]
                        if K4_rescue < K4_suspect:
                            result = r_rescue   # prefer lower (physical) K4
                            break
                    except RuntimeError:
                        continue

            t_ms = (_time.perf_counter() - t0) * 1e3

            if result is not None:
                x_aq = result['x_aq']
                x_c  = result['x_c']
                K1_r, K4_r = result['K_vals']

                # Update P warm-start for the next pressure step
                ws_prev_P = dict(
                    K1    = K1_r,
                    K4    = K4_r,
                    sol_aq = result.get('sol_aq_x0'),
                    sol_c  = result.get('sol_c_x0'),
                )

                records.append({
                    'T_K':            float(T),
                    'P_bar':          P_f,
                    'ecpa_xc_W':      float(x_aq['x4w']),
                    'ecpa_yw_C':      float(x_c['x1c']),
                    'ecpa_converged': True,
                    'ecpa_n_iter_ms': int(result.get('n_iter_ms', 0)),
                    'ecpa_t_ms':      t_ms,
                    'ecpa_available': True,
                    'K1':             K1_r,
                    'K4':             K4_r,
                })
            else:
                # All z candidates gave single-phase or flash failed.
                # Do NOT update ws_prev_P: keep the last known two-phase state.
                records.append({
                    'T_K':            float(T),
                    'P_bar':          P_f,
                    'ecpa_xc_W':      np.nan,
                    'ecpa_yw_C':      np.nan,
                    'ecpa_converged': False,
                    'ecpa_n_iter_ms': 0,
                    'ecpa_t_ms':      t_ms,
                    'ecpa_available': True,
                    'K1':             np.nan,
                    'K4':             np.nan,
                })

        ecpa_by_ms[ms] = records

        # Update prev_ms_by_P for the next ms: only two-phase results
        for rec in records:
            if rec['ecpa_converged']:
                prev_ms_by_P[rec['P_bar']] = dict(
                    K1    = rec['K1'],
                    K4    = rec['K4'],
                    sol_aq = None,   # sol_aq_x0 not in compact record
                    sol_c  = None,
                )
            # If not converged, keep whatever was there from the previous ms

    return T, cpa_records, ecpa_by_ms


# ── Figure generation (reused from _run_warmstart_co2h2o.py) ──────────────────

def _make_figures(smooth_data, results_df, exp_df, T_list,
                  out_dir='figures/co2h2o_ws2'):
    """Generate ribbon figures, one per temperature."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import re

    CMAP    = plt.cm.rainbow
    CNORM   = mcolors.Normalize(vmin=0.0, vmax=max(m for m in MS_RIBBON if m > 0.01))
    P_FINE  = np.logspace(np.log10(P_MIN), np.log10(P_MAX), 3000)
    COLORS  = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
               '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

    def _short_ref(ref):
        m = re.search(r'(\w+)\s+(?:et al\.?\s*)?\((\d{4})\)', str(ref))
        return f"{m.group(1)} ({m.group(2)})" if m else str(ref)[:25]

    sorted_ms = sorted(smooth_data.keys())

    def _interp_curve(df, T_K, col, _SLOPE_MAX=4.0):
        sub  = df[np.abs(df['T_K'] - T_K) < 0.5].sort_values('P_bar')
        if len(sub) < 2:
            return np.full(len(P_FINE), np.nan)
        mask = sub[col].notna() & (sub[col] > 0) & (sub['P_bar'] > 0)
        if mask.sum() < 2:
            return np.full(len(P_FINE), np.nan)
        logP  = np.log10(sub.loc[mask, 'P_bar'].values)
        logY  = np.log10(sub.loc[mask, col].values)
        # Trim steep near-boundary onset from the left: near P_sat(H2O,T),
        # xc_W ∝ (P−P_sat) so the log-log slope diverges as P/(P−P_sat).
        # Drop leading points until the slope to the next point is < _SLOPE_MAX,
        # so the plotted curve starts where the grid can represent it faithfully.
        start = 0
        if len(logP) > 2:
            slopes = np.abs(np.diff(logY) / np.clip(np.diff(logP), 1e-9, None))
            while start < len(slopes) and slopes[start] > _SLOPE_MAX:
                start += 1
        logP = logP[start:]
        logY = logY[start:]
        if len(logP) < 2:
            return np.full(len(P_FINE), np.nan)
        logPf = np.log10(P_FINE)
        within = (logPf >= logP.min()) & (logPf <= logP.max())
        out    = np.full(len(P_FINE), np.nan)
        if within.sum() >= 2:
            out[within] = 10.0 ** np.interp(logPf[within], logP, logY)
        return out

    os.makedirs(out_dir, exist_ok=True)

    for T_K in T_list:
        fig = plt.figure(figsize=(12, 5))
        ax0 = fig.add_axes([0.07, 0.12, 0.40, 0.80])
        ax1 = fig.add_axes([0.54, 0.12, 0.40, 0.80])
        cax = fig.add_axes([0.96, 0.12, 0.016, 0.80])

        panels = [
            (ax0, 'ecpa_xc_W', 'cpa_xc_W', 'exp_xc_W',
             r'$x_{\mathrm{CO_2}}$ (aqueous)', 'upper left'),
            (ax1, 'ecpa_yw_C', 'cpa_yw_C', 'exp_yw_C',
             r'$y_{\mathrm{H_2O}}$ (CO$_2$-rich)', 'upper right'),
        ]

        for ax, ecpa_col, cpa_col, exp_col, ylabel, legend_loc in panels:
            curves = {ms_v: _interp_curve(df, T_K, ecpa_col)
                      for ms_v, df in smooth_data.items()}

            # Fill between adjacent ms curves
            for i in range(len(sorted_ms) - 1):
                ms_lo, ms_hi = sorted_ms[i], sorted_ms[i + 1]
                y_lo, y_hi   = curves[ms_lo], curves[ms_hi]
                valid = np.isfinite(y_lo) & np.isfinite(y_hi)
                if valid.sum() < 2:
                    continue
                y1 = np.where(valid, np.maximum(y_lo, y_hi), np.nan)
                y2 = np.where(valid, np.minimum(y_lo, y_hi), np.nan)
                ax.fill_between(P_FINE, y1, y2, where=valid,
                                color=CMAP(CNORM(0.5 * (ms_lo + ms_hi))),
                                alpha=0.45, linewidth=0, zorder=2)

            # Discrete ms lines
            for ms_v in sorted_ms:
                y     = curves[ms_v]
                valid = np.isfinite(y)
                if valid.sum() < 2:
                    continue
                ax.plot(P_FINE[valid], y[valid], '-',
                        color=CMAP(CNORM(ms_v if ms_v > 0.01 else 0.0)),
                        lw=0.9, alpha=0.85, zorder=3)

            # CPA curve
            cpa_y = _interp_curve(smooth_data[sorted_ms[0]], T_K, cpa_col)
            valid = np.isfinite(cpa_y)
            if valid.sum() > 1:
                ax.plot(P_FINE[valid], cpa_y[valid], 'k-', lw=1.4,
                        label='CPA (salt-free)', zorder=4)

            # Experimental scatter
            if results_df is not None:
                sub_exp = results_df[np.abs(results_df['T_K'] - T_K) < 0.5]
                for ci, (ref, grp) in enumerate(sub_exp.groupby('reference')):
                    y_exp = grp[exp_col].dropna()
                    if len(y_exp) == 0:
                        continue
                    p_exp = grp.loc[y_exp.index, 'P_bar']
                    ax.scatter(p_exp, y_exp, s=22, zorder=6,
                               color=COLORS[ci % len(COLORS)],
                               label=_short_ref(ref))

            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel('P [bar]', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, which='both', alpha=0.25)
            ax.legend(fontsize=7, loc=legend_loc)

        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=CNORM)
        sm.set_array([])
        plt.colorbar(sm, cax=cax, label=r'$m_s$ [mol kg$^{-1}$]')

        fig.savefig(f'{out_dir}/T{T_K:.0f}K.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Figure saved: {out_dir}/T{T_K:.0f}K.png", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--recompute', action='store_true',
                    help='Delete cached parquets and recompute from scratch')
    ap.add_argument('--no-figures', action='store_true',
                    help='Skip figure generation')
    ap.add_argument('--workers', type=int, default=N_WORKERS)
    ap.add_argument('--out-dir', default='figures/co2h2o_ws2',
                    help='Output directory for figures')
    args_cli = ap.parse_args()

    n_workers = args_cli.workers
    out_dir   = args_cli.out_dir

    os.makedirs('results', exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    from ecpa.validate_co2h2o import T_MAX_ECPA

    if args_cli.recompute:
        print("--recompute: removing cached parquets …")
        removed = 0
        for tag_ms in [_ms_tag(m) for m in MS_RIBBON]:
            p = Path(f'results/ws2_smooth_co2h2o_ms{tag_ms}.parquet')
            if p.exists(): p.unlink(); removed += 1
        p = Path('results/ws2_cpa_smooth.parquet')
        if p.exists(): p.unlink(); removed += 1
        print(f"  Removed {removed} parquet file(s)")

    # T values from experimental data file
    exp_df    = pd.read_parquet('CO2_WATER_exp.parquet')
    T_unique  = sorted(exp_df['T_K'].unique())
    T_list    = [T for T in T_unique if T <= T_MAX_ECPA]
    P_grid    = np.logspace(np.log10(P_MIN), np.log10(P_MAX), N_P)

    print(f"Temperatures : {T_list[0]:.0f}–{T_list[-1]:.0f} K  ({len(T_list)} values)")
    print(f"P grid       : {P_MIN}–{P_MAX} bar  ({N_P} points, log-spaced)")
    print(f"ms ladder    : {MS_RIBBON}")
    print(f"z candidates : {Z_CANDIDATES}")
    print(f"Workers      : {n_workers}")

    # ── Check which T values still need computing ──────────────────────────
    # A T is "done" only if ALL ms parquets already contain rows for that T.
    all_ms_cached = {}   # ms -> DataFrame (loaded if exists)
    for ms in MS_RIBBON:
        tag   = _ms_tag(ms)
        cache = Path(f'results/ws2_smooth_co2h2o_ms{tag}.parquet')
        if cache.exists():
            all_ms_cached[ms] = pd.read_parquet(cache)
        else:
            all_ms_cached[ms] = None

    def _T_needs_compute(T):
        for ms, df in all_ms_cached.items():
            if df is None:
                return True
            if not (np.abs(df['T_K'] - T) < 0.5).any():
                return True
        return False

    T_todo = [T for T in T_list if _T_needs_compute(T)]

    if not T_todo:
        print("\nAll T values already cached — loading and skipping compute.")
    else:
        print(f"\n=== Computing {len(T_todo)} temperature(s) "
              f"({len(T_list) - len(T_todo)} cached) ===")
        t_total = time.perf_counter()

        args_list = [(T, P_grid) for T in T_todo]

        # Accumulators for new rows
        new_cpa_rows   = []
        new_ecpa_rows  = {ms: [] for ms in MS_RIBBON}

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs  = {pool.submit(_robust_T_worker, a): a[0] for a in args_list}
            done  = 0
            for fut in as_completed(futs):
                T_done, cpa_recs, ecpa_by_ms = fut.result()
                done += 1

                new_cpa_rows.extend(cpa_recs)
                for ms in MS_RIBBON:
                    new_ecpa_rows[ms].extend(ecpa_by_ms[ms])

                n_2ph = sum(r['ecpa_converged']
                            for ms in MS_RIBBON
                            for r in ecpa_by_ms[ms])
                print(f"  T={T_done:.0f}K  [{done}/{len(T_todo)}]  "
                      f"two-phase pts: {n_2ph}/{N_P * len(MS_RIBBON)}",
                      flush=True)

        # ── Merge new rows into (or create) each parquet ───────────────────
        # CPA parquet
        cpa_cache = Path('results/ws2_cpa_smooth.parquet')
        if new_cpa_rows:
            new_cpa_df = pd.DataFrame(new_cpa_rows)
            if cpa_cache.exists():
                old = pd.read_parquet(cpa_cache)
                # Drop old rows for T values we just recomputed
                T_done_set = set(new_cpa_df['T_K'])
                old = old[~old['T_K'].isin(T_done_set)]
                combined = pd.concat([old, new_cpa_df], ignore_index=True)
            else:
                combined = new_cpa_df
            combined = combined.sort_values(['T_K', 'P_bar'])
            combined.to_parquet(cpa_cache, index=False)

        cpa_df_full = pd.read_parquet(cpa_cache) if cpa_cache.exists() else pd.DataFrame()

        # eCPA parquets (one per ms)
        for ms in MS_RIBBON:
            tag   = _ms_tag(ms)
            cache = Path(f'results/ws2_smooth_co2h2o_ms{tag}.parquet')
            if new_ecpa_rows[ms]:
                new_ecpa_df = pd.DataFrame(new_ecpa_rows[ms])
                # Merge CPA columns
                new_ecpa_df = new_ecpa_df.merge(
                    cpa_df_full[['T_K','P_bar','cpa_xc_W','cpa_yw_C',
                                 'cpa_converged','cpa_n_iter','cpa_t_ms']],
                    on=['T_K','P_bar'], how='left')
                if cache.exists():
                    old = pd.read_parquet(cache)
                    T_done_set = set(new_ecpa_df['T_K'])
                    old = old[~old['T_K'].isin(T_done_set)]
                    combined = pd.concat([old, new_ecpa_df], ignore_index=True)
                else:
                    combined = new_ecpa_df
                combined = combined.sort_values(['T_K', 'P_bar'])
                # Drop internal K1/K4 columns (not needed for plotting)
                for drop_col in ['K1', 'K4']:
                    if drop_col in combined.columns:
                        combined = combined.drop(columns=[drop_col])
                combined.to_parquet(cache, index=False)
                all_ms_cached[ms] = combined
            else:
                if cache.exists():
                    all_ms_cached[ms] = pd.read_parquet(cache)

        elapsed = time.perf_counter() - t_total
        print(f"\nCompute done in {elapsed:.1f}s")

    # ── Load smooth_data for figure generation ─────────────────────────────
    smooth_data = {}
    for ms in MS_RIBBON:
        tag   = _ms_tag(ms)
        cache = Path(f'results/ws2_smooth_co2h2o_ms{tag}.parquet')
        if cache.exists():
            smooth_data[ms] = pd.read_parquet(cache)
        else:
            print(f"  WARNING: missing {cache}")

    # ── Load or compute validation scatter points ──────────────────────────
    val_cache = Path('results/ws_validation_co2h2o.parquet')
    results_df = pd.read_parquet(val_cache) if val_cache.exists() else None
    if results_df is None:
        print("  (No validation parquet found — scatter points will be absent)")

    # ── Generate figures ────────────────────────────────────────────────────
    if not args_cli.no_figures and smooth_data:
        print(f"\n=== Generating figures → {out_dir}/ ===")
        _make_figures(smooth_data, results_df, exp_df, T_list, out_dir=out_dir)

    print("\nDone.")
