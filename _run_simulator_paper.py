"""
Run the CO2-brine IMPEC simulator and generate paper-quality figures.

Saves arrays at key time steps and produces:
  figures/simulator/fig_simulator.png  –  3-panel Sg evolution (paper figure)
  figures/simulator/fig_perf.png       –  performance figure (paper style)
  results/simulator_timing.npz         –  raw timing arrays for the table
"""
import warnings; warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import os, time
    import numpy as np
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from multiprocessing import get_context

    # ── Import simulation machinery from co2brine_simulator ───────────────
    from co2brine_simulator import (
        Lx, Ly, Lz, Nx, Ny, Nz, N, dx, dy, dz,
        gridV, gridPV, K,
        T_K, P_ref, ms0, year_s, Q, t_max_yr, n_steps,
        z_inject, z_initial, inj_i, inj_j,
        _worker_init, _flash_one,
        run_flash_parallel, beta_to_Sg, compute_Fz,
        TPFA, upwindingmatrix, estimate_maxdFdz,
        dissolved_co2_molality,
    )
    from ecpa.parameters import make_params
    from ecpa.solution_table import load_solution_table, make_solution_guess_fn

    os.makedirs('results', exist_ok=True)
    os.makedirs('figures/simulator', exist_ok=True)

    # ── Load EoS / table ─────────────────────────────────────────────────
    print("Loading eCPA parameters and solution table …")
    params    = make_params()
    grid_data = load_solution_table()
    guess_fn  = make_solution_guess_fn(grid_data)
    print("  done.\n")

    # ── CFL estimate ─────────────────────────────────────────────────────
    print("Estimating max dF/dz …", end=' ', flush=True)
    maxdFdz = estimate_maxdFdz(params, guess_fn, T_K, P_ref, ms0)
    print(f"{maxdFdz:.2f}\n")

    # ── Parallel pool ────────────────────────────────────────────────────
    ctx  = get_context('spawn')
    pool = ctx.Pool(initializer=_worker_init, initargs=(params, grid_data))

    # ── Initial conditions ────────────────────────────────────────────────
    z       = z_initial * np.ones(N)
    inj     = Q.ravel().clip(min=0)
    inj_src = inj * z_inject
    dt_big  = t_max_yr * year_s / n_steps

    # Steps at which to save state for paper figures
    SAVE_STEPS = {12, 36, 100}       # t ≈ 0.36, 1.08, 3.00 yr
    saved_states = {}                # step → {'t_yr', 'z', 'Sg', 'mc', 'ms_aq'}

    timing = []
    snap_every = max(1, n_steps // 8)

    print(f"Grid: {Nx}×{Ny} = {N} cells  |  {t_max_yr:.0f} yr  |  "
          f"{n_steps} steps  |  dt = {dt_big/year_s:.3f} yr/step\n")
    print(f"{'Step':>5}  {'t [yr]':>7}  {'N_sub':>5}  {'N_flash':>7}  "
          f"{'t_flash [s]':>11}  {'t_step [s]':>10}")
    print("─" * 58)

    t0_total = time.perf_counter()

    for step in range(1, n_steps + 1):
        t_yr = step * dt_big / year_s
        t0   = time.perf_counter()

        fr      = run_flash_parallel(z, ms0, T_K, P_ref, pool)
        t_flash = time.perf_counter() - t0

        S_g       = beta_to_Sg(fr, T_K, P_ref)
        Fz, lam_t = compute_Fz(S_g, fr['x4w'], fr['x4c'])

        lam_3d = np.stack([lam_t.reshape(Nx, Ny, Nz, order='F')] * 3, axis=0)
        Keff   = K * lam_3d
        _, Vx, Vy, Vz = TPFA(Lx, Ly, Lz, Nx, Ny, Nz, Q, Keff)

        UPW, CFL = upwindingmatrix(Nx, Ny, Nz, Vx, Vy, Vz, Q, maxdFdz)
        Nt  = int(np.ceil(dt_big / CFL))
        dtx = (dt_big / Nt) / gridPV
        for _ in range(Nt):
            z = z + (UPW.dot(Fz) + inj_src) * dtx
        z = z.clip(0, 1)

        t_step  = time.perf_counter() - t0
        n_flash = fr['n_flash']
        timing.append((step, t_yr, t_flash, t_step, n_flash))
        print(f"{step:5d}  {t_yr:7.2f}  {Nt:5d}  {n_flash:7d}  "
              f"{t_flash:11.2f}  {t_step:10.2f}")

        # Save state at key steps
        if step in SAVE_STEPS:
            mc    = dissolved_co2_molality(fr['x4w'], fr['ms_aq'])
            saved_states[step] = dict(
                t_yr  = t_yr,
                z     = z.copy().reshape(Nx, Ny, order='F'),
                Sg    = S_g.copy().reshape(Nx, Ny, order='F'),
                mc    = mc.copy().reshape(Nx, Ny, order='F'),
                ms_aq = fr['ms_aq'].copy().reshape(Nx, Ny, order='F'),
                x4w   = fr['x4w'].copy().reshape(Nx, Ny, order='F'),
                x4c   = fr['x4c'].copy().reshape(Nx, Ny, order='F'),
            )
            print(f"          → saved state at t = {t_yr:.2f} yr")

    pool.close()
    pool.join()

    wall = time.perf_counter() - t0_total
    t_fl  = np.array([t[2] for t in timing])
    t_st  = np.array([t[3] for t in timing])
    n_fls = np.array([t[4] for t in timing])
    t_yrs = np.array([t[1] for t in timing])

    nf_pos = n_fls[n_fls > 0]
    tf_pos = t_fl[n_fls > 0]
    mean_throughput = float((nf_pos / tf_pos).mean()) if len(nf_pos) else 0.0

    print(f"\nFinished in {wall:.1f} s")
    print(f"  Mean flash calls/step: {n_fls.mean():.0f}  (max {n_fls.max():.0f})")
    print(f"  Mean flash time/step : {t_fl.mean():.2f} s  "
          f"({t_fl.sum()/t_st.sum()*100:.0f}% of runtime)")
    print(f"  Flash throughput     : {mean_throughput:.0f} calls/s")

    np.savez('results/simulator_timing.npz',
             t_yr=t_yrs, t_flash=t_fl, t_step=t_st, n_flash=n_fls,
             mean_throughput=mean_throughput,
             mean_nflash=n_fls.mean(), mean_tflash=t_fl.mean())
    print("  Saved results/simulator_timing.npz")

    # =========================================================================
    # FIGURE 1 — Sg evolution at 3 time snapshots
    # =========================================================================
    XX = np.linspace(dx/2, Lx - dx/2, Nx)
    YY = np.linspace(dx/2, Ly - dx/2, Ny)   # dy = dx for square grid

    steps_ordered = sorted(saved_states.keys())
    labels = ['(a)', '(b)', '(c)']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2),
                             gridspec_kw={'wspace': 0.06, 'right': 0.88})
    cmap = plt.cm.plasma
    for ax, step_k, lbl in zip(axes, steps_ordered, labels):
        st  = saved_states[step_k]
        im  = ax.pcolormesh(XX, YY, st['Sg'].T,
                            cmap=cmap, vmin=0, vmax=1, shading='auto')
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ [m]', fontsize=12, fontweight='bold')
        ax.tick_params(labelsize=10)
        ax.set_title(f'{lbl}  $t$ = {st["t_yr"]:.2f} yr', fontsize=11)
        # well markers
        ax.plot(XX[inj_i], YY[inj_j], '*', color='white', ms=9, zorder=5)
        for ci, cj in [(0,0),(Nx-1,0),(0,Ny-1),(Nx-1,Ny-1)]:
            ax.plot(XX[ci], YY[cj], '^', color='black', ms=6, zorder=5)

    axes[0].set_ylabel('$y$ [m]', fontsize=12, fontweight='bold')
    for ax in axes[1:]:
        ax.set_yticklabels([])

    cax = fig.add_axes([0.90, 0.12, 0.018, 0.76])
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label(r'CO$_2$-rich saturation $S_g$', fontsize=11, fontweight='bold')
    cb.ax.tick_params(labelsize=10)

    plt.savefig('figures/simulator/fig_simulator.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figures/simulator/fig_simulator.png")

    # =========================================================================
    # FIGURE 2 — Performance (clean paper style)
    # =========================================================================
    fig2, ax2 = plt.subplots(2, 1, figsize=(6.5, 5), sharex=True,
                              gridspec_kw={'hspace': 0.08})

    ax2[0].plot(t_yrs, t_fl, lw=1.5, color='steelblue',
                label='Flash (two-phase cells)')
    ax2[0].plot(t_yrs, t_st, lw=1.5, ls='--', color='tomato',
                label='Total step')
    ax2[0].set_ylabel('Wall time [s]', fontsize=12, fontweight='bold')
    ax2[0].legend(fontsize=10)
    ax2[0].tick_params(labelsize=10)
    ax2[0].grid(True, alpha=0.3)

    bar_w = (t_yrs[1] - t_yrs[0]) * 0.85 if len(t_yrs) > 1 else 0.03
    ax2[1].bar(t_yrs, n_fls, width=bar_w, alpha=0.7, color='steelblue',
               label='Two-phase cells flashed')
    ax2[1].axhline(N, color='grey', ls=':', lw=1.2, label=f'Total cells ({N:,})')
    ax2[1].set_xlabel('Simulation time [yr]', fontsize=12, fontweight='bold')
    ax2[1].set_ylabel('Flash calls', fontsize=12, fontweight='bold')
    ax2[1].legend(fontsize=10)
    ax2[1].tick_params(labelsize=10)
    ax2[1].grid(True, alpha=0.3)

    plt.savefig('figures/simulator/fig_perf.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figures/simulator/fig_perf.png")

    # =========================================================================
    # FIGURE 3 — Compositional fields (2×2): Sg + 3 eCPA-specific outputs
    # Top row: t = 1.08 yr (step 36).  Bottom row uses t = 3.00 yr (step 100)
    # to show salting-out at full domain coverage.
    # =========================================================================
    def _wells(ax):
        ax.plot(XX[inj_i], YY[inj_j], '*', color='white', ms=9,
                markeredgecolor='0.4', markeredgewidth=0.5, zorder=5)
        for ci, cj in [(0,0),(Nx-1,0),(0,Ny-1),(Nx-1,Ny-1)]:
            ax.plot(XX[ci], YY[cj], '^', color='black', ms=6, zorder=5)

    def _prep(step_key):
        st   = saved_states[step_key]
        Sg_  = st['Sg']
        x4w_ = st['x4w']
        x4c_ = st['x4c']
        ms_  = st['ms_aq']
        # masks
        y_h2o = np.where(x4c_ > 0.5,         1.0 - x4c_, np.nan)
        x_co2 = np.where(Sg_  < 0.99,         x4w_,       np.nan)
        ms_p  = np.where(ms_  > ms0 * 1.0001, ms_,        ms0)
        return Sg_, x_co2, y_h2o, ms_p

    Sg36,  xc36,  yh36,  ms36  = _prep(36)   # t = 1.08 yr
    Sg100, xc100, yh100, ms100 = _prep(100)  # t = 3.00 yr

    # shared colour limits (same across both time steps for comparability)
    x_vmax  = float(np.nanpercentile(xc36[np.isfinite(xc36)],   99))
    y_vmax  = float(np.nanpercentile(yh36[np.isfinite(yh36)],   99))
    # Clip ms_aq at a moderate percentile so the gradient is visible over
    # a larger area; values above vmax saturate to the darkest colour.
    ms_vmax = float(np.nanpercentile(ms100[np.isfinite(ms100)], 80))

    # 2×2 figure, each row one time step
    fig3, axes3 = plt.subplots(2, 3, figsize=(13.5, 8.5),
                                gridspec_kw={'wspace': 0.08, 'hspace': 0.18})

    rows = [(36, Sg36, xc36, yh36, ms36), (100, Sg100, xc100, yh100, ms100)]
    row_t = [saved_states[36]['t_yr'], saved_states[100]['t_yr']]

    panels_def = [
        (r'$m_s^\mathrm{aq}$ [mol kg$^{-1}$]', 'YlOrRd', ms0,  ms_vmax, True),
        (r'$x_{\mathrm{CO_2}}$ (aqueous)',       'Blues',  0.0,  x_vmax,  False),
        (r'$y_{\mathrm{H_2O}}$ (CO$_2$-rich)',   'YlGn',   0.0,  y_vmax,  False),
    ]
    lbl_rc = [['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)']]

    for r, (step_k, Sg_, xc_, yh_, ms_) in enumerate(rows):
        data_row = [ms_, xc_, yh_]
        for c, (data, (cb_lbl, cmap, vmin, vmax, bold_cb)) in \
                enumerate(zip(data_row, panels_def)):
            ax  = axes3[r, c]
            im  = ax.pcolormesh(XX, YY, data.T, cmap=cmap,
                                vmin=vmin, vmax=vmax, shading='auto')
            ax.set_aspect('equal')
            ax.tick_params(labelsize=9)
            _wells(ax)
            lbl = lbl_rc[r][c]
            ax.set_title(f'{lbl}  {cb_lbl}', fontsize=10)
            # x-axis label only on bottom row
            if r == 1:
                ax.set_xlabel('$x$ [m]', fontsize=11, fontweight='bold')
            else:
                ax.set_xticklabels([])
            # y-axis label only on left column
            if c == 0:
                ax.set_ylabel(f'$t$ = {row_t[r]:.2f} yr\n$y$ [m]',
                              fontsize=11, fontweight='bold')
            else:
                ax.set_yticklabels([])
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=8)
            if bold_cb:
                cb.ax.yaxis.label.set_fontweight('bold')

    plt.savefig('figures/simulator/fig_compositions.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figures/simulator/fig_compositions.png")

    print("\nDone.")
