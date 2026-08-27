"""Generate ecpa_newton_stats.pdf from the scan_v4 table by replaying flash calls."""
import warnings; warnings.filterwarnings('ignore')
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ecpa.parameters import make_params
from ecpa.warmstart import ScanTableWarmStart
from ecpa.flash import flash_co2_h2o_salt_kv
import os

os.makedirs('figures/scan_v4', exist_ok=True)

data   = np.load('results/scan_v4_table.npz')
T_grid = data['T_grid']
P_grid = data['P_grid']
ms_grid = data['ms_grid']
is_2ph = data['is_two_phase']   # (nT, nP, nms)

params = make_params()
ws     = ScanTableWarmStart.load('results/scan_v4_table.npz')

# Sample every 5th T, every 3rd P, and 4 ms values to keep runtime manageable
T_sel  = T_grid[::5]
P_sel  = P_grid[::3]
ms_sel = [0.0, 1.0, 3.0, 6.0]
ms_idx = [np.argmin(np.abs(ms_grid - m)) for m in ms_sel]

# Patch flash to count Newton iters
try:  # optional legacy module; the script falls back to flash outputs
    from ecpa import newton_inner
except ImportError:
    newton_inner = None

aq_iters = []
c_iters  = []

orig_solve = getattr(newton_inner, 'solve_newton_3x3_aq', None)
# Instead, monkey-patch flash to intercept: just call flash and count fallback
# Alternative: run flash and read the 'newton_iters_aq'/'newton_iters_c' keys if present

# Check if flash returns newton iter counts
z_test = 0.5
T_test = 350.0; P_test = 100.0; ms_test = 1.0
guess = ws(T_test, P_test, z_test, ms_test)
out = flash_co2_h2o_salt_kv(T=T_test, P_bar=P_test, z_co2=z_test, m_tot=ms_test,
                             K_init=guess.K_init if guess else None,
                             sol_aq_x0=guess.sol_aq_x0 if guess else None,
                             sol_c_x0=guess.sol_c_x0 if guess else None,
                             params=params, maxiter=60)
print('Flash output keys:', list(out.keys()) if out else 'None')

if out and 'n_newton_aq' in out:
    print('Newton iter counts available in flash output!')
    for T in T_sel:
        for P in P_sel:
            for ms_i, ms in zip(ms_idx, ms_sel):
                iT = np.argmin(np.abs(T_grid - T))
                iP = np.argmin(np.abs(P_grid - P))
                if not is_2ph[iT, iP, ms_i]:
                    continue
                guess = ws(T, P, z_test, ms)
                try:
                    out = flash_co2_h2o_salt_kv(T=float(T), P_bar=float(P), z_co2=z_test,
                                                m_tot=float(ms),
                                                K_init=guess.K_init if guess else None,
                                                sol_aq_x0=guess.sol_aq_x0 if guess else None,
                                                sol_c_x0=guess.sol_c_x0 if guess else None,
                                                params=params, maxiter=60)
                    if out and out.get('phase') != 'single_phase':
                        aq_iters.append(out.get('n_newton_aq', 0))
                        c_iters.append(out.get('n_newton_c', 0))
                except Exception:
                    pass
else:
    print('Newton iter counts NOT in flash output. Using wall-time proxy figure instead.')
    print('Generating simplified figure from wall_time distribution...')
    # Make a simple figure showing median outer iters from a sample run
    import pandas as pd
    met = pd.read_parquet('results/scan_v4_metrics.parquet')
    two_ph = met[met['is_two_phase']]
    wt = two_ph['wall_time_ms'].values
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(wt[wt < 200], bins=60, density=True, color='steelblue', alpha=0.8, label='all eCPA calls')
    ax.axvline(np.median(wt), ls='--', color='k', lw=1.5, label=f'median {np.median(wt):.0f} ms')
    ax.set_xlabel('Wall time per flash call (ms)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('eCPA flash wall-time distribution (two-phase, 3D scan)', fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig('figures/scan_v4/ecpa_newton_stats.pdf', bbox_inches='tight', dpi=150)
    fig.savefig('figures/scan_v4/ecpa_newton_stats.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('  → ecpa_newton_stats.pdf (wall-time proxy)')
    import sys; sys.exit(0)

# If we got iter counts, plot them
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, iters, label, color in zip(
        axes,
        [aq_iters, c_iters],
        [r'Aqueous ($Z_w,\varepsilon_r,\chi_w$)', r'CO$_2$-rich ($Z_c$)'],
        ['steelblue', 'darkorange']):
    ax.hist(iters, bins=range(0, max(iters)+2), density=True, color=color, alpha=0.8)
    med = np.median(iters)
    ax.axvline(med, ls='--', color='k', lw=1.5, label=f'median = {med:.0f}')
    ax.set_xlabel('Newton iterations per call', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig('figures/scan_v4/ecpa_newton_stats.pdf', bbox_inches='tight', dpi=150)
fig.savefig('figures/scan_v4/ecpa_newton_stats.png', bbox_inches='tight', dpi=150)
plt.close()
print('  → ecpa_newton_stats.pdf (iteration counts)')
