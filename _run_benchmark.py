import warnings; warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import time
    import numpy as np
    from ecpa.parameters import make_params
    from ecpa.guess_table import load_cpa_guess_table, make_guess_fn
    from ecpa.solution_table import load_solution_table, make_solution_guess_fn
    from ecpa.flash import flash_co2_h2o_salt_ssi, flash_co2_h2o_salt_fast

    params = make_params()
    CPA_GROUPS, CPA_TEMPS = load_cpa_guess_table()
    guess_table_fn = make_guess_fn(CPA_GROUPS, CPA_TEMPS)
    grid_data = load_solution_table('results/solution_table.npz')
    solution_guess_fn = make_solution_guess_fn(grid_data)

    # Test conditions: 30 pressure points at fixed (T, z, ms)
    T_bm, z_bm, ms_bm = 398.0, 0.5, 1.0
    P_bm = np.logspace(np.log10(20), np.log10(400), 30)

    def run_ssi(label, **kwargs):
        t0 = time.perf_counter()
        results = []
        for P_i in P_bm:
            try:
                out = flash_co2_h2o_salt_ssi(T=T_bm, P_bar=P_i, z_co2=z_bm, m_tot=ms_bm,
                                              params=params, **kwargs)
                results.append(('ok', out['ms_aq'], out.get('n_iter_ms', -1)))
            except Exception as e:
                results.append(('fail', float('nan'), -1))
        t = time.perf_counter() - t0
        ok = sum(1 for r in results if r[0] == 'ok')
        iters = [r[2] for r in results if r[0] == 'ok' and r[2] > 0]
        print(f'{label:<40s} {ok:2d}/{len(P_bm)}  avg_iters={sum(iters)/len(iters):.1f}  {t:.3f}s')
        return results, t

    def run_fast(label, **kwargs):
        t0 = time.perf_counter()
        results = []
        for P_i in P_bm:
            try:
                out = flash_co2_h2o_salt_fast(T=T_bm, P_bar=P_i, z_co2=z_bm, m_tot=ms_bm,
                                               solution_guess_fn=solution_guess_fn, params=params,
                                               fallback_guess_table_fn=guess_table_fn, **kwargs)
                ms_val = out.get('ms_aq', float('nan')) if out.get('phase') != 'single_phase' else float('nan')
                results.append(('ok', ms_val, out.get('n_iter_ms', 0), out.get('phase', '?')))
            except Exception as e:
                print(f"  fail at P={P_i:.1f}: {e}")
                results.append(('fail', float('nan'), -1, 'fail'))
        t = time.perf_counter() - t0
        ok = sum(1 for r in results if r[0] == 'ok')
        iters = [r[2] for r in results if r[0] == 'ok' and r[2] > 0]
        avg_it = sum(iters)/len(iters) if iters else 0
        print(f'{label:<40s} {ok:2d}/{len(P_bm)}  avg_iters={avg_it:.1f}  {t:.3f}s')
        return results, t

    print(f"\nBenchmark: T={T_bm}K  z={z_bm}  ms={ms_bm}  N={len(P_bm)} pts\n")
    print(f"{'Method':<40s} {'conv':>7}  {'avg_it':>9}  {'time':>6}")
    print("-" * 65)

    # Baseline: cold-start SSI
    r_cold, t_cold = run_ssi("Cold SSI (baseline)",
                              guess_table_fn=guess_table_fn)

    # Fast: table hint + undamped warm SSI
    r_fast, t_fast = run_fast("Fast (hint + omega=1.0, no stab)")

    # Fast: table hint + undamped warm SSI + forced stability
    r_safe, t_safe = run_fast("Fast (hint + omega=1.0, force_stab)",
                               force_stability_check=True)

    # Fast: always stability (old behavior)
    r_old, t_old = run_fast("Fast (always stab, omega=0.7, iter=5)",
                              force_stability_check=True, omega_warm=0.7, max_ssi_iter=5)

    print()
    print(f"Speedup fast vs cold:       {t_cold/t_fast:.1f}×")
    print(f"Speedup fast+stab vs cold:  {t_cold/t_safe:.1f}×")

    # Accuracy
    ms_cold = np.array([r[1] for r in r_cold])
    ms_fast = np.array([r[1] for r in r_fast])
    mask = np.isfinite(ms_cold) & np.isfinite(ms_fast)
    if mask.any():
        print(f"\nms_aq accuracy vs cold SSI:")
        print(f"  fast (hint):  max diff = {np.max(np.abs(ms_cold[mask] - ms_fast[mask])):.2e}")

    ms_safe = np.array([r[1] for r in r_safe])
    mask2 = np.isfinite(ms_cold) & np.isfinite(ms_safe)
    if mask2.any():
        print(f"  fast+stab:    max diff = {np.max(np.abs(ms_cold[mask2] - ms_safe[mask2])):.2e}")
