"""
Validation of eCPA and CPA against CO₂–H₂O binary VLE experimental data.

Experimental data
-----------------
Loaded from CO2_WATER_exp.parquet (T=273–623 K, 631 points).
Columns: T_K, P_bar, xc_W (CO₂ in aqueous), yw_C (H₂O in CO₂-rich),
         rho_W, exp_id, reference, source_file.

Model predictions
-----------------
CPA   : CPA2.flash_co2_h2o_tpz(kij12=0, swc=0)  — old CPA parameters, no
        temperature-dependent cross-association, no salt terms.
eCPA  : flash_co2_h2o_salt_fast at ms=MS_EVAL mol/kg using the solution-table
        interpolant as the initial-guess provider, exactly as a reservoir
        simulator would.  At ms=MS_EVAL (default 0.1), the DH/Born salting-out
        effect is negligible and results are practically identical to binary.

Public API
----------
run_cpa_binary(T, P_bar, z_co2)
run_ecpa_flash(T, P_bar, z_co2, ms, solution_guess_fn, params, ...)
run_validation(exp_df, solution_guess_fn, params, ...)
run_smooth_curves(T_vals, solution_guess_fn, params, ...)
compute_metrics(results_df)
plot_validation_T(T_K, results_df, smooth_df, ...)
metrics_to_latex(metrics_df, path)
"""

import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Minimum ms in solution table — used as "near-zero salt" for eCPA binary runs
MS_EVAL = 0.1          # mol/kg NaCl
Z_CO2_DEFAULT = 0.5    # feed composition for stability/flash


# ── CPA binary flash (old CPA parameters) ─────────────────────────────────────

def run_cpa_binary(T, P_bar, z_co2=Z_CO2_DEFAULT):
    """
    Run CPA2 flash with old CPA parameters (kij=0, swc=0) for binary CO₂–H₂O.

    Returns dict:
        xc_W, yw_C  : float or NaN (NaN if single-phase or failed)
        phase       : 'two_phase' | 'single_phase' | 'error'
        converged   : bool
        n_iter      : int
        t_ms        : float
    """
    import CPA2
    T = float(T); P_bar = float(P_bar); z_co2 = float(z_co2)
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = CPA2.flash_co2_h2o_tpz(T=T, P_bar=P_bar, z_co2=z_co2,
                                        kij12=0.0, swc=0.0)
    except Exception as exc:
        t_ms = (time.perf_counter() - t0) * 1e3
        return dict(xc_W=np.nan, yw_C=np.nan, phase='error',
                    converged=False, n_iter=0, t_ms=t_ms, error=str(exc))

    t_ms = (time.perf_counter() - t0) * 1e3
    phase = r.get('phase', 'unknown')
    n_iter = int(r['tie']['iterations'])
    converged = bool(r['tie']['converged'])

    if phase != 'two_phase':
        return dict(xc_W=np.nan, yw_C=np.nan, phase=phase,
                    converged=converged, n_iter=n_iter, t_ms=t_ms)

    return dict(
        xc_W      = float(r['x'][0]),   # CO₂ mol-frac in aqueous phase
        yw_C      = float(r['y'][1]),   # H₂O mol-frac in CO₂-rich phase
        phase     = phase,
        converged = converged,
        n_iter    = n_iter,
        t_ms      = t_ms,
    )


# ── eCPA flash (reservoir-simulator workflow) ──────────────────────────────────

def run_ecpa_flash(T, P_bar, z_co2, ms, solution_guess_fn, params,
                   fallback_guess_table_fn=None):
    """
    eCPA flash using solution-table warm start + Michelsen stability + SSI.
    This is the same path a reservoir simulator would take.

    Parameters
    ----------
    ms : float
        Salt molality [mol/kg].  Use MS_EVAL (=0.1) for near-zero-salt runs.

    Returns dict:
        xc_W, yw_C      : float or NaN
        phase           : 'two_phase' | 'single_phase' | 'failed'
        converged       : bool
        n_iter_ms       : int   (SSI outer iterations)
        stability_run   : bool  (True if Michelsen TPD was evaluated)
        stable          : bool or None
        tpd_min         : float or None
        t_ms            : float (total wall time [ms])
    """
    from .flash import flash_co2_h2o_salt_fast
    T = float(T); P_bar = float(P_bar)
    z_co2 = float(z_co2); ms = float(ms)
    t0 = time.perf_counter()

    try:
        r = flash_co2_h2o_salt_fast(
            T=T, P_bar=P_bar, z_co2=z_co2, m_tot=ms,
            solution_guess_fn=solution_guess_fn,
            params=params,
            fallback_guess_table_fn=fallback_guess_table_fn,
            force_stability_check=False,
        )
        t_ms = (time.perf_counter() - t0) * 1e3
        phase = r.get('phase', 'unknown')
        stability_run = r.get('stable') is not None  # None means table hint used

        if phase != 'two_phase':
            return dict(xc_W=np.nan, yw_C=np.nan, phase=phase,
                        converged=True, n_iter_ms=0,
                        stability_run=stability_run,
                        stable=r.get('stable'), tpd_min=r.get('tpd_min'),
                        t_ms=t_ms)

        x_aq = r['x_aq']
        x_c  = r['x_c']
        return dict(
            xc_W        = float(x_aq['x4w']),  # CO₂ in aqueous (x4w = CO₂)
            yw_C        = float(x_c['x1c']),   # H₂O in CO₂-rich (x1c = H₂O)
            phase       = phase,
            converged   = True,
            n_iter_ms   = r.get('n_iter_ms', 0),
            stability_run = stability_run,
            stable      = r.get('stable'),
            tpd_min     = r.get('tpd_min'),
            t_ms        = t_ms,
        )

    except RuntimeError as exc:
        t_ms = (time.perf_counter() - t0) * 1e3
        return dict(xc_W=np.nan, yw_C=np.nan, phase='failed',
                    converged=False, n_iter_ms=0,
                    stability_run=False,
                    stable=None, tpd_min=None,
                    t_ms=t_ms, error=str(exc))


# ── Main validation loop (at experimental conditions) ─────────────────────────

def run_validation(exp_df, solution_guess_fn, params,
                   ms=MS_EVAL, z_co2=Z_CO2_DEFAULT,
                   T_max_ecpa=523.0,
                   verbose=True):
    """
    Run CPA and eCPA predictions at every experimental (T, P) condition.

    Parameters
    ----------
    T_max_ecpa : float
        Maximum T for eCPA flash (solution table covers up to 523 K).

    Returns
    -------
    pd.DataFrame with columns:
        T_K, P_bar, reference,
        exp_xc_W, exp_yw_C,
        cpa_xc_W, cpa_yw_C, cpa_converged, cpa_n_iter, cpa_t_ms,
        ecpa_xc_W, ecpa_yw_C, ecpa_converged, ecpa_n_iter_ms,
        ecpa_stability_run, ecpa_stable, ecpa_tpd_min, ecpa_t_ms,
        ecpa_available
    """
    records = []
    n = len(exp_df)

    for i, row in exp_df.iterrows():
        T = float(row['T_K'])
        P = float(row['P_bar'])

        if verbose and i % 100 == 0:
            print(f"  [{i:4d}/{n}] T={T:.0f}K  P={P:.1f} bar")

        rec = {
            'T_K':       T,
            'P_bar':     P,
            'reference': row.get('reference', ''),
            'exp_xc_W':  row.get('xc_W', np.nan),
            'exp_yw_C':  row.get('yw_C', np.nan),
        }

        # CPA prediction
        cpa = run_cpa_binary(T, P, z_co2=z_co2)
        rec['cpa_xc_W']     = cpa['xc_W']
        rec['cpa_yw_C']     = cpa['yw_C']
        rec['cpa_converged']= cpa['converged']
        rec['cpa_n_iter']   = cpa['n_iter']
        rec['cpa_t_ms']     = cpa['t_ms']

        # eCPA prediction (only if T is within solution-table range)
        ecpa_ok = T <= T_max_ecpa
        rec['ecpa_available'] = ecpa_ok

        if ecpa_ok:
            ecpa = run_ecpa_flash(T, P, z_co2=z_co2, ms=ms,
                                  solution_guess_fn=solution_guess_fn,
                                  params=params)
            rec['ecpa_xc_W']         = ecpa['xc_W']
            rec['ecpa_yw_C']         = ecpa['yw_C']
            rec['ecpa_converged']    = ecpa['converged']
            rec['ecpa_n_iter_ms']    = ecpa.get('n_iter_ms', 0)
            rec['ecpa_stability_run']= ecpa.get('stability_run', False)
            rec['ecpa_stable']       = ecpa.get('stable')
            rec['ecpa_tpd_min']      = ecpa.get('tpd_min')
            rec['ecpa_t_ms']         = ecpa['t_ms']
        else:
            for k in ('xc_W', 'yw_C', 'stable', 'tpd_min'):
                rec[f'ecpa_{k}'] = np.nan
            rec['ecpa_converged']     = False
            rec['ecpa_n_iter_ms']     = 0
            rec['ecpa_stability_run'] = False
            rec['ecpa_t_ms']          = 0.0

        records.append(rec)

    return pd.DataFrame(records)


# ── Smooth curves for plotting ─────────────────────────────────────────────────

def run_smooth_curves(T_vals, solution_guess_fn, params,
                      ms=MS_EVAL, z_co2=Z_CO2_DEFAULT,
                      n_P=100, P_min=1.0, P_max=1500.0,
                      T_max_ecpa=523.0,
                      verbose=True):
    """
    Compute CPA and eCPA predictions on a log-spaced P grid for each T.
    Used for drawing smooth lines on the per-T plots.

    Returns pd.DataFrame with same schema as run_validation but with
    a dense P grid per T (no experimental columns).
    """
    P_grid = np.logspace(np.log10(P_min), np.log10(P_max), n_P)
    records = []
    n_T = len(T_vals)

    for i_T, T in enumerate(T_vals):
        if verbose:
            print(f"  Smooth curves T={T:.0f}K  [{i_T+1}/{n_T}]")

        ecpa_ok = float(T) <= T_max_ecpa
        prev_ecpa_sol = None   # not used directly but kept for future warm-start

        for P in P_grid:
            rec = {'T_K': float(T), 'P_bar': float(P), 'ecpa_available': ecpa_ok}

            # CPA
            cpa = run_cpa_binary(T, P, z_co2=z_co2)
            rec['cpa_xc_W']      = cpa['xc_W']
            rec['cpa_yw_C']      = cpa['yw_C']
            rec['cpa_converged'] = cpa['converged']
            rec['cpa_n_iter']    = cpa['n_iter']
            rec['cpa_t_ms']      = cpa['t_ms']

            # eCPA
            if ecpa_ok:
                ecpa = run_ecpa_flash(T, P, z_co2=z_co2, ms=ms,
                                      solution_guess_fn=solution_guess_fn,
                                      params=params)
                rec['ecpa_xc_W']          = ecpa['xc_W']
                rec['ecpa_yw_C']          = ecpa['yw_C']
                rec['ecpa_converged']     = ecpa['converged']
                rec['ecpa_n_iter_ms']     = ecpa.get('n_iter_ms', 0)
                rec['ecpa_stability_run'] = ecpa.get('stability_run', False)
                rec['ecpa_t_ms']          = ecpa['t_ms']
            else:
                rec['ecpa_xc_W']          = np.nan
                rec['ecpa_yw_C']          = np.nan
                rec['ecpa_converged']     = False
                rec['ecpa_n_iter_ms']     = 0
                rec['ecpa_stability_run'] = False
                rec['ecpa_t_ms']          = 0.0

            records.append(rec)

    return pd.DataFrame(records)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _metrics_pair(pred, exp):
    """
    Compute AARE, bias, R² between pred and exp arrays.
    Only uses rows where both are finite and exp > 0.
    Returns (AARE_pct, bias_pct, R2, n).
    """
    mask = np.isfinite(pred) & np.isfinite(exp) & (np.asarray(exp) > 0)
    n = int(mask.sum())
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    p = pred[mask]; e = exp[mask]
    rel  = (p - e) / e
    aare = float(np.abs(rel).mean() * 100.0)
    bias = float(rel.mean() * 100.0)
    ss_res = float(np.sum((p - e) ** 2))
    ss_tot = float(np.sum((e - e.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return aare, bias, r2, n


def compute_metrics(results_df):
    """
    Compute AARE, bias (signed ARE), R² for:
      - CPA  vs experiment
      - eCPA vs experiment  (only rows where ecpa_available)
      - CPA  vs eCPA        (rows where both are finite)

    Grouped by T_K and overall ('All').

    Returns pd.DataFrame with columns:
        T_K, qty,
        n_cpa, AARE_CPA, bias_CPA, R2_CPA,
        n_ecpa, AARE_eCPA, bias_eCPA, R2_eCPA,
        n_cpa_ecpa, AARE_CPA_eCPA, bias_CPA_eCPA
    """
    rows = []

    groups = [(None, results_df)] + [
        (T, g) for T, g in results_df.groupby('T_K')
    ]

    for T_K, grp in groups:
        label = 'All' if T_K is None else T_K
        exp_arr  = grp[['exp_xc_W', 'exp_yw_C']].values
        cpa_arr  = grp[['cpa_xc_W', 'cpa_yw_C']].values
        ecpa_arr = grp[['ecpa_xc_W', 'ecpa_yw_C']].values

        for j, qty in enumerate(['xc_W', 'yw_C']):
            exp_v  = exp_arr[:, j].astype(float)
            cpa_v  = cpa_arr[:, j].astype(float)
            ecpa_v = ecpa_arr[:, j].astype(float)

            aare_c, bias_c, r2_c, n_c = _metrics_pair(cpa_v, exp_v)
            aare_e, bias_e, r2_e, n_e = _metrics_pair(ecpa_v, exp_v)

            # CPA vs eCPA: treat eCPA as "reference"
            mask_ce = np.isfinite(cpa_v) & np.isfinite(ecpa_v) & (ecpa_v > 0)
            n_ce = int(mask_ce.sum())
            if n_ce > 0:
                rel_ce   = (cpa_v[mask_ce] - ecpa_v[mask_ce]) / ecpa_v[mask_ce]
                aare_ce  = float(np.abs(rel_ce).mean() * 100.0)
                bias_ce  = float(rel_ce.mean() * 100.0)
            else:
                aare_ce = bias_ce = np.nan

            rows.append(dict(
                T_K=label, qty=qty,
                n_cpa=n_c, AARE_CPA=aare_c, bias_CPA=bias_c, R2_CPA=r2_c,
                n_ecpa=n_e, AARE_eCPA=aare_e, bias_eCPA=bias_e, R2_eCPA=r2_e,
                n_cpa_ecpa=n_ce, AARE_CPA_eCPA=aare_ce, bias_CPA_eCPA=bias_ce,
            ))

    return pd.DataFrame(rows)


# ── Per-temperature log-log plots ─────────────────────────────────────────────

_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def plot_validation_T(T_K, results_df, smooth_df, save_path=None, ms=MS_EVAL):
    """
    Log-log plot for a single temperature:
    Left panel  : P [bar] vs xc_W (CO₂ in aqueous)
    Right panel : P [bar] vs yw_C (H₂O in CO₂-rich)

    Experimental data are shown as scatter (coloured by reference).
    CPA and eCPA are shown as smooth lines from smooth_df.
    """
    sub_val    = results_df[np.abs(results_df['T_K'] - T_K) < 0.5]
    sub_smooth = smooth_df[np.abs(smooth_df['T_K'] - T_K) < 0.5].sort_values('P_bar')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    panel_specs = [
        ('xc_W', 'exp_xc_W', 'cpa_xc_W', 'ecpa_xc_W',
         r'$x_{\mathrm{CO_2}}^{\mathrm{aq}}$ (CO₂ in aqueous)'),
        ('yw_C', 'exp_yw_C', 'cpa_yw_C', 'ecpa_yw_C',
         r'$y_{\mathrm{H_2O}}^{\mathrm{CO_2\text{-}rich}}$ (H₂O in CO₂-rich)'),
    ]

    for ax, (qty, exp_col, cpa_col, ecpa_col, xlabel) in zip(axes, panel_specs):
        # — Experimental scatter, coloured by reference —
        refs = sub_val['reference'].unique()
        for ci, ref in enumerate(refs):
            grp = sub_val[sub_val['reference'] == ref]
            mask = grp[exp_col].notna() & (grp[exp_col] > 0)
            if mask.sum() == 0:
                continue
            label = ref[:30] if len(ref) <= 30 else ref[:28] + '…'
            ax.scatter(grp.loc[mask, exp_col], grp.loc[mask, 'P_bar'],
                       color=_COLORS[ci % len(_COLORS)],
                       s=25, zorder=5, label=label)

        # — CPA smooth line —
        cpa_mask = sub_smooth[cpa_col].notna() & (sub_smooth[cpa_col] > 0) \
                   & (sub_smooth['P_bar'] > 0)
        if cpa_mask.sum() > 1:
            ax.plot(sub_smooth.loc[cpa_mask, cpa_col],
                    sub_smooth.loc[cpa_mask, 'P_bar'],
                    'k-', lw=1.8, label='CPA', zorder=3)

        # — eCPA smooth line —
        ecpa_mask = sub_smooth[ecpa_col].notna() & (sub_smooth[ecpa_col] > 0) \
                    & (sub_smooth['P_bar'] > 0)
        if ecpa_mask.sum() > 1:
            ax.plot(sub_smooth.loc[ecpa_mask, ecpa_col],
                    sub_smooth.loc[ecpa_mask, 'P_bar'],
                    'r--', lw=1.8, label=f'eCPA (ms={ms})', zorder=3)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('P [bar]', fontsize=10)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, which='both', ls=':', alpha=0.4)

    fig.suptitle(f'CO₂–H₂O VLE   T = {T_K:.0f} K', fontsize=12, fontweight='bold')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ── LaTeX metrics table ────────────────────────────────────────────────────────

def metrics_to_latex(metrics_df, path='results/co2h2o_metrics.tex', ms=MS_EVAL):
    """
    Write a standalone LaTeX document with the metrics table.
    Produces one table per quantity (xc_W, yw_C) plus an Overall block.
    """

    def _fmt(v, d=2):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return r'\text{---}'
        return f'{v:.{d}f}'

    def _n(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return r'\text{---}'
        return str(int(v))

    lines = [
        r'\documentclass[10pt,a4paper]{article}',
        r'\usepackage{booktabs,amsmath,geometry,caption,array,xcolor}',
        r'\geometry{margin=1.5cm,landscape}',
        r'\begin{document}',
        r'\small',
        r'\begin{center}',
        r'\textbf{CO$_2$--H$_2$O VLE validation: CPA vs eCPA accuracy}\\[4pt]',
        rf'\textit{{eCPA at $m_s = {ms}$ mol/kg NaCl; CPA uses $k_{{ij}}=0$, $S_{{14}}=0$.}}\\[2pt]',
        r'\textit{AARE: avg.\ absolute relative error (\%); '
        r'Bias: mean signed relative error (\%); $R^2$: coeff.\ of determination.}',
        r'\end{center}',
        r'\vspace{4pt}',
    ]

    for qty_label, qty_key in [('CO$_2$ in aqueous phase ($x_{\\rm CO_2}^{\\rm aq}$)', 'xc_W'),
                                ('H$_2$O in CO$_2$-rich phase ($y_{\\rm H_2O}^{\\rm CO_2-rich}$)', 'yw_C')]:
        sub = metrics_df[metrics_df['qty'] == qty_key]

        lines += [
            '',
            r'\begin{table}[ht]',
            r'\centering',
            rf'\caption{{{qty_label}}}',
            r'\begin{tabular}{l rrrr rrrr rr}',
            r'\toprule',
            r'$T$ [K] & $n$ & AARE$_{\rm CPA}$ & Bias$_{\rm CPA}$ & $R^2_{\rm CPA}$ & '
            r'$n$ & AARE$_{\rm eCPA}$ & Bias$_{\rm eCPA}$ & $R^2_{\rm eCPA}$ & '
            r'AARE$_{\rm CPA/eCPA}$ & Bias$_{\rm CPA/eCPA}$ \\',
            r' & & (\%) & (\%) & & & (\%) & (\%) & & (\%) & (\%) \\',
            r'\midrule',
        ]

        # Overall row
        overall = sub[sub['T_K'] == 'All']
        for _, row in overall.iterrows():
            lines.append(
                rf"\textbf{{All}} & {_n(row['n_cpa'])} & "
                rf"{_fmt(row['AARE_CPA'])} & {_fmt(row['bias_CPA'])} & {_fmt(row['R2_CPA'])} & "
                rf"{_n(row['n_ecpa'])} & "
                rf"{_fmt(row['AARE_eCPA'])} & {_fmt(row['bias_eCPA'])} & {_fmt(row['R2_eCPA'])} & "
                rf"{_fmt(row['AARE_CPA_eCPA'])} & {_fmt(row['bias_CPA_eCPA'])} \\"
            )

        lines.append(r'\midrule')

        # Per-temperature rows (sorted numerically)
        T_rows = sub[sub['T_K'] != 'All'].copy()
        T_rows['_Tnum'] = pd.to_numeric(T_rows['T_K'], errors='coerce')
        T_rows = T_rows.sort_values('_Tnum')

        for _, row in T_rows.iterrows():
            T_str = f"{int(row['T_K'])}"
            lines.append(
                rf"{T_str} & {_n(row['n_cpa'])} & "
                rf"{_fmt(row['AARE_CPA'])} & {_fmt(row['bias_CPA'])} & {_fmt(row['R2_CPA'])} & "
                rf"{_n(row['n_ecpa'])} & "
                rf"{_fmt(row['AARE_eCPA'])} & {_fmt(row['bias_eCPA'])} & {_fmt(row['R2_eCPA'])} & "
                rf"{_fmt(row['AARE_CPA_eCPA'])} & {_fmt(row['bias_CPA_eCPA'])} \\"
            )

        lines += [
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ]

    lines.append(r'\end{document}')

    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))
    print(f"Wrote {path}")


# ── Computational performance summary ─────────────────────────────────────────

def print_perf_summary(results_df, smooth_df=None):
    """Print a summary of convergence, iteration counts, and timing."""
    print("\n── Validation at experimental conditions ─────────────────────────")
    n_tot = len(results_df)

    # CPA
    n_cpa_2p = (results_df['cpa_xc_W'].notna()).sum()
    cpa_t    = results_df['cpa_t_ms'].mean()
    cpa_iter = results_df['cpa_n_iter'].mean()
    print(f"  CPA  : {n_cpa_2p}/{n_tot} two-phase  "
          f"avg {cpa_iter:.1f} iters  {cpa_t:.1f} ms/call")

    # eCPA
    ecpa_avail = results_df[results_df['ecpa_available']]
    n_ecpa_av  = len(ecpa_avail)
    n_ecpa_2p  = (ecpa_avail['ecpa_xc_W'].notna()).sum()
    n_stab_run = ecpa_avail['ecpa_stability_run'].sum()
    ecpa_t     = ecpa_avail['ecpa_t_ms'].mean()
    ecpa_iter  = ecpa_avail['ecpa_n_iter_ms'].mean()
    print(f"  eCPA : {n_ecpa_2p}/{n_ecpa_av} two-phase (T≤523K)  "
          f"avg {ecpa_iter:.1f} SSI iters  {ecpa_t:.1f} ms/call  "
          f"stability run {n_stab_run}/{n_ecpa_av} times "
          f"({100*n_stab_run/max(n_ecpa_av,1):.0f}%)")

    if smooth_df is not None:
        print("\n── Smooth curve grid ─────────────────────────────────────────────")
        n_sm = len(smooth_df)
        n_sm_2p_cpa  = smooth_df['cpa_converged'].sum()
        n_sm_2p_ecpa = smooth_df['ecpa_converged'].sum()
        sm_cpa_t     = smooth_df['cpa_t_ms'].mean()
        sm_ecpa_t    = smooth_df[smooth_df['ecpa_available']]['ecpa_t_ms'].mean()
        print(f"  CPA  : {n_sm_2p_cpa}/{n_sm} converged  {sm_cpa_t:.1f} ms/call")
        print(f"  eCPA : {n_sm_2p_ecpa}/{n_sm} converged  {sm_ecpa_t:.2f} ms/call")
