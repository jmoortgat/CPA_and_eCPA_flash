"""
Paper demonstration: CO₂ injection into fresh vs saline brine.

Two simulations are run sequentially (each uses all CPU cores for flash):

  Case A — Salt-free brine (CPA, ms = 0 mol/kg)
  Case B — Saline brine    (eCPA, ms = 4 mol/kg)

Both use:
  - MFE pressure solver on a 50×50 heterogeneous grid
  - High injection rate (20 % PV/yr) with BHP-controlled producers at 150 bar
  - T = 350 K,  P_inj_ref = 200 bar,  5-spot well pattern
  - 5 years simulation time, 300 pressure steps

After both runs, a side-by-side comparison figure is saved as
  figures/demo/comparison_final.png

Usage
-----
    python _run_demo_simulations.py          # run both sequentially
    python _run_demo_simulations.py cpa      # only case A
    python _run_demo_simulations.py ecpa     # only case B
    python _run_demo_simulations.py figures  # regenerate comparison figure only
"""

from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Simulation parameters (shared between both cases) ────────────────────────
COMMON = dict(
    pressure_solver = 'mfe',
    Nx_             = 50,
    Ny_             = 50,
    Lx_             = 100.0,    # m
    Ly_             = 100.0,    # m
    Lz_             = 10.0,     # m
    K_mean_mD       = 100.0,    # mean permeability [mD]
    K_sigma_ln      = 1.5,      # log-normal σ; 0 = homogeneous
    K_seed          = 42,
    T_K_            = 350.0,    # K  (isothermal)
    P_ref_bar_      = 200.0,    # bar (reference flash pressure)
    phi_            = 0.20,
    mu_co2_         = 5.0e-5,   # Pa·s
    mu_brine_       = 4.5e-4,   # Pa·s
    inj_pvi_yr      = 0.20,     # 20 % PV per year
    bhp_prod_bar    = 150.0,    # BHP at producers [bar]
    r_w_m           = 0.10,     # wellbore radius [m]
    t_max_yr_       = 5.0,
    n_steps_        = 300,
    snap_frac       = 6,        # save 6 snapshots during the run
)

CASE_A = dict(**COMMON,
    flash_model = 'cpa',
    ms0_        = 0.0,
    outdir      = 'figures/demo/cpa_ms0',
)

CASE_B = dict(**COMMON,
    flash_model = 'ecpa',
    ms0_        = 4.0,          # high-salinity brine (Dead Sea level ~saturation)
    outdir      = 'figures/demo/ecpa_ms4',
)


# =============================================================================
# COMPARISON FIGURE
# =============================================================================
def make_comparison_figure(result_a: dict, result_b: dict,
                            outpath: str = 'figures/demo/comparison_final.png'):
    """
    3 × 2 grid showing (top to bottom):
      - CO₂-rich saturation Sg
      - Dissolved CO₂ molality mc
      - Brine molality ms_aq  (0 everywhere for CPA — use pressure instead)
    Columns: Case A (CPA, ms=0) | Case B (eCPA, ms=4)
    """
    import co2brine_simulator as sim

    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)

    def _fields(res):
        T      = res['T_K'];   P = res['P_ref_bar']
        Nx_    = res['Nx'];    Ny_ = res['Ny']
        Lx_    = res['Lx'];   Ly_ = res['Ly']
        fr     = res['fr']
        z_arr  = res['z']
        P_fld  = res['P_field']

        S_g  = sim.beta_to_Sg(fr, T, P).reshape(Nx_, Ny_, order='F')
        mc   = sim.dissolved_co2_molality(fr['x4w'], fr['ms_aq']).reshape(Nx_, Ny_, order='F')
        ms_  = fr['ms_aq'].reshape(Nx_, Ny_, order='F')
        z2d  = z_arr.reshape(Nx_, Ny_, order='F')
        P2d  = (P_fld.reshape(Nx_, Ny_, order='F') / 1e5) if P_fld is not None else None

        dx_ = Lx_ / Nx_;  dy_ = Ly_ / Ny_
        XX  = np.linspace(dx_/2, Lx_ - dx_/2, Nx_)
        YY  = np.linspace(dy_/2, Ly_ - dy_/2, Ny_)
        return dict(S_g=S_g, mc=mc, ms=ms_, z2d=z2d, P2d=P2d, XX=XX, YY=YY,
                    Nx=Nx_, Ny=Ny_, Lx=Lx_, Ly=Ly_)

    fa = _fields(result_a)
    fb = _fields(result_b)

    # Row definitions (field key, label, cmap, shared vlim?)
    rows = [
        ('S_g', r'CO$_2$-rich saturation $S_g$',              'plasma',  True,  (0, 1)),
        ('mc',  r'Dissolved CO$_2$  $m_c$ [mol kg$^{-1}$]',   'Blues',   True,  None),
        ('P2d', r'Pressure [bar]',                              'RdYlGn', False, None),
    ]
    # For CPA the ms field is 0 everywhere; replace row 3 with pressure for CPA too

    fig = plt.figure(figsize=(12, 10))
    gs  = GridSpec(len(rows), 2, figure=fig,
                   hspace=0.35, wspace=0.08,
                   left=0.06, right=0.94, top=0.93, bottom=0.06)

    col_labels = [
        r'CPA (salt-free,  $m_s = 0$)',
        r'eCPA ($m_s = 4$ mol kg$^{-1}$ NaCl)',
    ]

    for col, (fd, res) in enumerate(zip([fa, fb], [result_a, result_b])):
        for row, (key, row_label, cmap, shared_vlim, vlim_fixed) in enumerate(rows):
            ax = fig.add_subplot(gs[row, col])
            data = fd[key]
            if data is None:
                ax.set_visible(False)
                continue

            # Determine colour limits
            if shared_vlim:
                if vlim_fixed:
                    vmin, vmax = vlim_fixed
                else:
                    d_a = fa[key];  d_b = fb[key]
                    if d_a is not None and d_b is not None:
                        vmin = min(d_a.min(), d_b.min())
                        vmax = max(d_a.max(), d_b.max())
                    else:
                        vmin, vmax = None, None
            else:
                vmin, vmax = None, None

            kw = {}
            if vmin is not None:
                kw = dict(vmin=vmin, vmax=vmax)

            im = ax.pcolormesh(fd['XX'], fd['YY'], data.T,
                               cmap=cmap, shading='auto', **kw)
            plt.colorbar(im, ax=ax, pad=0.02)

            # Well markers
            Nx_ = fd['Nx'];  Ny_ = fd['Ny']
            Lx_ = fd['Lx'];  Ly_ = fd['Ly']
            XX_ = fd['XX'];  YY_ = fd['YY']
            ax.plot(XX_[Nx_//2], YY_[Ny_//2], '*', color='white',
                    ms=10, zorder=5, markeredgecolor='grey', markeredgewidth=0.5)
            for ci, cj in [(0,0),(Nx_-1,0),(0,Ny_-1),(Nx_-1,Ny_-1)]:
                ax.plot(XX_[ci], YY_[cj], '^', color='black', ms=6, zorder=5)

            ax.set_xlabel('x [m]', fontsize=8)
            if col == 0:
                ax.set_ylabel('y [m]', fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(col_labels[col], fontsize=10, pad=4)
            if col == 0:
                ax.annotate(row_label, xy=(-0.22, 0.5), xycoords='axes fraction',
                            fontsize=8, ha='right', va='center', rotation=90)

    fig.suptitle(
        r'CO$_2$ injection: 5-spot, MFE, $T=350\,$K, $P_\mathrm{inj}=200\,$bar, '
        r'$\dot{q}=20\%$ PVI/yr, $t=5\,$yr',
        fontsize=10)
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison figure → {outpath}")


def load_result(outdir: str) -> dict | None:
    """Load saved final state from a previous run."""
    npz_path = f'{outdir}/final_state.npz'
    if not os.path.exists(npz_path):
        return None
    npz = np.load(npz_path)
    # Reconstruct minimal result dict from saved arrays
    fr = dict(beta=npz['beta'], x4w=npz['x4w'], x4c=npz['x4c'],
              ms_aq=npz['ms_aq'], Z_aq=npz['Z_aq'], Z_c=npz['Z_c'])
    return dict(z=npz['z'], P_field=npz['P'],
                fr=fr, outdir=outdir,
                # These need to match COMMON parameters:
                T_K=COMMON['T_K_'], P_ref_bar=COMMON['P_ref_bar_'],
                Nx=COMMON['Nx_'], Ny=COMMON['Ny_'],
                Lx=COMMON['Lx_'], Ly=COMMON['Ly_'],
                ms0=CASE_A['ms0_'])


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    from co2brine_simulator import main as run_sim

    mode = sys.argv[1].lower() if len(sys.argv) > 1 else 'both'

    result_a = result_b = None

    if mode in ('both', 'cpa', 'a'):
        print("=" * 70)
        print("CASE A: Salt-free brine (CPA, ms=0)")
        print("=" * 70)
        result_a = run_sim(**CASE_A)

    if mode in ('both', 'ecpa', 'b'):
        print("=" * 70)
        print("CASE B: Saline brine (eCPA, ms=4 mol/kg)")
        print("=" * 70)
        result_b = run_sim(**CASE_B)

    if mode == 'figures':
        # Load from saved state
        result_a = load_result(CASE_A['outdir'])
        result_b = load_result(CASE_B['outdir'])
        if result_a is None or result_b is None:
            print("ERROR: final_state.npz not found in one or both output dirs.")
            print("  Run simulations first:  python _run_demo_simulations.py")
            sys.exit(1)
        # Patch ms0 for Case B
        result_b['ms0'] = CASE_B['ms0_']

    if result_a is not None and result_b is not None:
        make_comparison_figure(result_a, result_b)
    elif result_a is not None or result_b is not None:
        print("(Only one case run — comparison figure requires both.)")
