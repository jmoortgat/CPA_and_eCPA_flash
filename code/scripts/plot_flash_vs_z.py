"""Regenerate figures/flash_vs_z.png — rainbow-band style, Celsius labels."""
import warnings; warnings.filterwarnings('ignore')
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams.update({
    'font.weight':       'bold',
    'axes.labelweight':  'bold',
    'axes.titleweight':  'bold',
    'axes.labelsize':    11,
    'axes.titlesize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
})

from ecpa.parameters import make_params
from ecpa.warmstart import ScanTableWarmStart
from ecpa.flash import flash_co2_h2o_salt_kv
from ecpa.stability import ecpa_stability_flash

# ── conditions ────────────────────────────────────────────────────────────────
P_bar  = 200.0
m_feed = 1.0          # mol/kg feed NaCl
Mw     = 0.018015     # kg/mol

TEMPS  = np.arange(300.0, 526.0, 25.0)   # 300, 325, …, 525 K  (10 isotherms)
N_Z    = 300
z_arr  = np.linspace(0.005, 0.995, N_Z)

params = make_params()
ws     = ScanTableWarmStart.load("results/scan_v4_table.npz")

# ── colormap ─────────────────────────────────────────────────────────────────
cmap   = plt.cm.rainbow
norm   = mcolors.Normalize(vmin=TEMPS[0], vmax=TEMPS[-1])

def T_color(T):
    return cmap(norm(T))

# ── run flash for each isotherm ───────────────────────────────────────────────
results = {}

for T in TEMPS:
    beta_arr  = np.full(N_Z, np.nan)
    msrat_arr = np.full(N_Z, np.nan)
    mc_arr    = np.full(N_Z, np.nan)
    x1c_arr   = np.full(N_Z, np.nan)

    K_prev = sol_aq_prev = sol_c_prev = None
    co2_ref_x0 = aq_ref_x0 = None

    for iz, z in enumerate(z_arr):
        out = None

        # Attempt 1: warm-started K-value flash from previous z
        if K_prev is not None:
            try:
                out = flash_co2_h2o_salt_kv(
                    T=float(T), P_bar=P_bar, z_co2=float(z), m_tot=m_feed,
                    K_init=K_prev,
                    sol_aq_x0=sol_aq_prev, sol_c_x0=sol_c_prev,
                    params=params, maxiter=80,
                )
            except Exception:
                out = None

        # Attempt 2: full stability + flash (6 guesses)
        if out is None:
            try:
                sf = ecpa_stability_flash(
                    z_co2=float(z), ms=m_feed,
                    T=float(T), P=P_bar,
                    params=params,
                    co2_ref_x0=co2_ref_x0,
                    aq_ref_x0=aq_ref_x0,
                )
                co2_ref_x0 = sf["stability"]["co2_ref_x0"]
                aq_ref_x0  = sf["stability"]["aq_ref_x0"]
                if sf.get("phase") != "single_phase":
                    out = sf
            except Exception:
                pass

        if out is None or out.get("phase") == "single_phase":
            K_prev = sol_aq_prev = sol_c_prev = None
            continue

        K_prev      = out["K_vals"]
        sol_aq_prev = out["sol_aq_x0"]
        sol_c_prev  = out["sol_c_x0"]

        beta  = float(out["beta"])
        ms_aq = float(out["ms_aq"])
        sol   = out["sol"]
        x1w   = float(sol[1])
        x4w   = 1.0 - x1w - 2.0 * x1w * ms_aq * Mw
        x1c   = float(sol[4])

        if not (1e-4 < beta < 1 - 1e-4):
            continue

        beta_arr [iz] = beta
        msrat_arr[iz] = ms_aq / m_feed
        mc_arr   [iz] = x4w / (x1w * Mw) if x1w > 1e-10 else np.nan
        x1c_arr  [iz] = x1c

    results[T] = dict(beta=beta_arr, msrat=msrat_arr, mc=mc_arr, x1c=x1c_arr)
    Tc = T - 273.15
    n2ph = np.isfinite(beta_arr).sum()
    print(f"T={T:.0f}K ({Tc:.0f}°C): {n2ph} two-phase points", flush=True)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
ax_beta, ax_ms, ax_mc, ax_x1c = axes.flat
z_label = r"Overall CO$_2$ mole fraction $z$"

panels = [
    (ax_beta, "beta",  r"CO$_2$-rich phase fraction $\beta$",
     "(a) Phase split", 0, 1, None, None),
    (ax_ms,   "msrat", r"$m_s^\mathrm{aq}\,/\,m_\mathrm{feed}$",
     "(b) NaCl salinity amplification", 0.9, None, None, None),
    (ax_mc,   "mc",    r"CO$_2$ molality $m_c$ [mol kg$^{-1}$]",
     r"(c) CO$_2$ solubility in aqueous phase", None, None, None, None),
    (ax_x1c,  "x1c",  r"H$_2$O mole fraction $y_{\mathrm{H_2O}}$",
     "(d) Water content of CO$_2$-rich phase", 0, None, None, None),
]

for ax, key, ylabel, title, ymin, ymax, _, __ in panels:
    # Collect valid curves (arrays that have any finite values)
    curves = []
    for T in TEMPS:
        arr = results[T][key]
        mask = np.isfinite(arr)
        if mask.sum() < 2:
            continue
        # interpolate to a common fine grid within the valid z range
        z_v = z_arr[mask]
        y_v = arr[mask]
        curves.append((T, z_v, y_v))

    # Rainbow fill between adjacent isotherms
    for i in range(len(curves) - 1):
        T_lo, z_lo, y_lo = curves[i]
        T_hi, z_hi, y_hi = curves[i + 1]
        T_mid = 0.5 * (T_lo + T_hi)
        color = T_color(T_mid)

        # Common z range for fill
        z_min = max(z_lo[0],  z_hi[0])
        z_max = min(z_lo[-1], z_hi[-1])
        if z_min >= z_max:
            continue
        zf = np.linspace(z_min, z_max, 500)
        yf_lo = np.interp(zf, z_lo, y_lo)
        yf_hi = np.interp(zf, z_hi, y_hi)
        ax.fill_between(zf, yf_lo, yf_hi,
                        color=color, alpha=0.35, zorder=1, linewidth=0)

    # Draw individual isotherm lines
    for T, z_v, y_v in curves:
        ax.plot(z_v, y_v, color=T_color(T), lw=1.0, alpha=0.85, zorder=2)

    ax.set_xlabel(z_label, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel,  fontsize=11, fontweight='bold')
    ax.set_title(title,    fontsize=11, fontweight='bold')
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')
    ax.set_xlim(0, 1)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax)

# Reference line in salinity amplification panel (no legend — colorbar covers T)
ax_ms.axhline(1.0, color="k", lw=1.8, ls="--", zorder=3)
ax_ms.text(0.02, 1.07, r"$m_s^\mathrm{aq} = m_\mathrm{feed}$",
           transform=ax_ms.get_yaxis_transform(), fontsize=9, fontweight='bold', va='bottom')

# Shared colorbar (replaces per-panel legends)
fig.tight_layout(rect=[0, 0, 0.88, 1.0])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.90, 0.10, 0.018, 0.78])   # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, label=r"$T$ [°C]")
cbar.set_label(r"$T$ [°C]", fontsize=11, fontweight='bold')
cbar.set_ticks(TEMPS)
cbar.set_ticklabels([f"{T - 273.15:.0f}" for T in TEMPS])
for tick in cbar.ax.get_yticklabels():
    tick.set_fontweight('bold')

out = "figures/flash_vs_z.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
