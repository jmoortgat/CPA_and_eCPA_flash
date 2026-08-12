"""
optimize_peneloux_co2.py
=========================
Optimize a temperature-dependent Peneloux volume shift c(T) for the CO2
component of the CPA/eCPA equation of state (SRK backbone) to minimise
CO2-rich-phase density errors against the Span-Wagner (1996) reference
equation of state for CO2, accessed via CoolProp.

Motivation: reviewer 2 on ms ie-2026-02626f asked why no volume translation
is applied to CO2, given that cubic EoS can misrepresent dense/supercritical
CO2 density at high P/T. `Peneloux_CO2` in ecpa/constants.py is currently 0.0
(off). Peneloux translation is isofugacity-preserving (main.tex:336-337), so
this cannot change any previously reported phase-composition/solubility
result -- only CO2-rich-phase density.

Strategy (mirrors optimize_peneloux_h2o.py)
--------------------------------------------
1. For each temperature T on a grid spanning the paper's range (0-425 C,
   1-1500 bar), perform a 1-D minimisation over c [cm3/mol] of
       AARE(c) = mean_P |rho(T,P,c) - rho_ref(T,P)| / rho_ref(T,P)
   where rho(T,P,c) = M_CO2 / (Vm_EoS(T,P) + c), Vm_EoS = Zc*R*T/P, and Zc is
   obtained from the CO2-rich-phase inner solver `_lnphi_c_inner` at pure CO2
   (x1c=0, i.e. zero H2O), which already selects the thermodynamically stable
   root (handles the liquid-like/gas-like branch choice near the critical
   point).
2. Fit a polynomial c(T_R) = sum_k a_k * T_R^k, T_R = T/Tc_CO2, to the
   per-temperature optimal shifts, same functional form as the existing
   H2O shift.
3. Report AARE for polynomial degrees 2-6, plus a single best constant shift
   for comparison (since Peneloux_CO2 is currently a bare constant = 0.0).
4. Print polynomial coefficients / constant for insertion into constants.py.

Reference data
--------------
Span, R.; Wagner, W. "A New Equation of State for Carbon Dioxide Covering
the Fluid Region from the Triple-Point Temperature to 1100 K at Pressures
up to 800 MPa." J. Phys. Chem. Ref. Data 1996, 25, 1509-1596. Accessed via
CoolProp's HEOS backend (PropsSI), the standard open-source implementation
of this reference equation.

Outputs
-------
- Console report of c_opt(T) values, polynomial fits, and AARE comparison
- figures/density/peneloux_co2_opt_c.png   -- c_opt vs T, polynomial fits
- figures/density/peneloux_co2_opt_err.png -- AARE improvement with shift
- figures/density/peneloux_co2_parity.png  -- parity plot (no-shift vs optimised)
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import CoolProp.CoolProp as CP

os.makedirs('figures/density', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ── Figure style (per user standing rules: bold, thick, colorblind-safe) ───────
mpl_rc = {
    "font.size": 14, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 10,
    "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
    "axes.linewidth": 1.6, "axes.edgecolor": "black",
    "xtick.major.width": 1.6, "ytick.major.width": 1.6,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 2.2, "lines.markersize": 7,
    "legend.frameon": True, "legend.edgecolor": "black",
    "savefig.dpi": 300, "savefig.bbox": "tight",
}
plt.rcParams.update(mpl_rc)
WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# ── Physical constants ─────────────────────────────────────────────────────────
R_bar_cm3 = 83.14          # bar*cm3/(mol*K)
M_CO2_g   = 44.01          # g/mol (matches CoolProp's own CO2 molar mass convention)
Tc_CO2    = 304.4          # K (from ecpa/constants.py Tc4, the EoS's own working value)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ecpa.stability import _lnphi_c_inner

# ============================================================================
# PART 1 -- Build (T, P) grid and compute EoS + reference molar volumes
# ============================================================================
print("=" * 68)
print("Step 1: Build grid, compute eCPA Vm_EoS and Span-Wagner (CoolProp) Vm_ref")
print("=" * 68)

T_C_vals = [5, 10, 15, 20, 25, 30, 32, 35, 40, 50, 60, 75, 100, 125, 150,
            175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425]
P_vals   = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 73.8, 80, 100, 150, 200,
            300, 400, 500, 600, 700, 800, 1000, 1200, 1500]

print(f"Grid: {len(T_C_vals)} T x {len(P_vals)} P = {len(T_C_vals)*len(P_vals)} conditions")
print("Reference: Span-Wagner 1996 CO2 EOS via CoolProp (HEOS backend)")

rows = []
for T_C in T_C_vals:
    T_K = T_C + 273.15
    x0_c = None
    for P_bar in P_vals:
        P_Pa = P_bar * 1e5
        # Reference density from Span-Wagner (CoolProp)
        try:
            rho_ref = CP.PropsSI('D', 'T', T_K, 'P', P_Pa, 'CO2')  # kg/m3
        except Exception:
            continue
        # eCPA pure-CO2 Zc (stable root selection handled inside _lnphi_c_inner)
        try:
            _, _, sol = _lnphi_c_inner(x1c=0.0, T=T_K, P=P_bar, x0=x0_c)
            Zc = float(sol[0])
            if not (np.isfinite(Zc) and 0 < Zc < 20):
                continue
            x0_c = np.asarray(sol, dtype=float)
        except Exception:
            continue
        Vm_EoS = Zc * R_bar_cm3 * T_K / P_bar    # cm3/mol
        Vm_ref = M_CO2_g * 1000.0 / rho_ref       # cm3/mol (rho_ref in kg/m3)
        rows.append(dict(T_K=T_K, T_C=T_C, P_bar=P_bar, Zc=Zc,
                          Vm_EoS=Vm_EoS, Vm_ref=Vm_ref, rho_ref=rho_ref))

df = pd.DataFrame(rows)
print(f"  Converged: {len(df)} / {len(T_C_vals)*len(P_vals)} conditions")
df.to_parquet('results/co2_density_span_wagner.parquet')

# ============================================================================
# PART 2 -- Per-temperature 1-D optimisation of c
# ============================================================================
print("\n" + "=" * 68)
print("Step 2: Optimise c(T) at each temperature (1-D AARE minimisation)")
print("=" * 68)
print(f"  {'T [K]':>7}  {'T [C]':>6}  {'N_P':>4}  "
      f"{'c_opt [cm3/mol]':>16}  {'AARE_no_shift':>14}  {'AARE_opt':>9}")
print("  " + "-"*70)

T_vals_opt = sorted(df['T_K'].unique())
c_opt_vals = []

for T in T_vals_opt:
    sub = df[df['T_K'] == T].copy()
    if len(sub) < 2:
        c_opt_vals.append(np.nan); continue

    Vm_EoS_arr = sub['Vm_EoS'].values
    rho_ref_arr = sub['rho_ref'].values

    def aare_at_c(c_scalar):
        Vm_corr = Vm_EoS_arr + c_scalar
        rho_pred = M_CO2_g * 1000.0 / Vm_corr
        return np.mean(np.abs(rho_pred - rho_ref_arr) / rho_ref_arr) * 100.0

    result = minimize_scalar(aare_at_c, bounds=(-10.0, 10.0), method='bounded',
                              options={'xatol': 1e-8})
    c_opt = float(result.x)
    aare_no_shift = aare_at_c(0.0)
    aare_with_opt = result.fun

    c_opt_vals.append(c_opt)
    print(f"  {T:7.2f}  {T-273.15:6.1f}  {len(sub):4d}  "
          f"{c_opt:16.5f}  {aare_no_shift:13.3f}%  {aare_with_opt:8.4f}%")

mask = np.array([np.isfinite(c) for c in c_opt_vals])
T_fit  = np.array(T_vals_opt)[mask]
c_fit  = np.array(c_opt_vals)[mask]
TR_fit = T_fit / Tc_CO2

# ============================================================================
# PART 3 -- Polynomial fit c(T_R) = sum a_k T_R^k, and best single constant
# ============================================================================
print("\n" + "=" * 68)
print("Step 3: Polynomial fit c(T_R) = sum_k a_k * T_R^k  (and best constant)")
print("=" * 68)
print(f"  T_R range: {TR_fit.min():.4f} - {TR_fit.max():.4f}")

def aare_all(c_func):
    TR_all  = df['T_K'].values / Tc_CO2
    c_all   = c_func(TR_all)
    Vm_corr = df['Vm_EoS'].values + c_all
    rho_pred = M_CO2_g * 1000.0 / Vm_corr
    return np.mean(np.abs(rho_pred - df['rho_ref'].values) / df['rho_ref'].values) * 100.0

aare_noshift_all = aare_all(lambda tr: 0.0 * tr)

# Best single constant (matches the form Peneloux_CO2 currently takes)
res_const = minimize_scalar(lambda c: aare_all(lambda tr: c),
                             bounds=(-10.0, 10.0), method='bounded')
c_best_const = float(res_const.x)
aare_const = res_const.fun
print(f"\n  No shift (current paper):        AARE = {aare_noshift_all:.4f}%")
print(f"  Best single constant c = {c_best_const:.5f} cm3/mol:  AARE = {aare_const:.4f}%")

poly_results = {}
best_deg, best_aare, best_coeffs = None, aare_const, None
for deg in range(2, 7):
    coeffs_np = np.polyfit(TR_fit, c_fit, deg)
    aare_poly = aare_all(lambda tr, cc=coeffs_np: np.polyval(cc, tr))
    poly_results[deg] = {'coeffs': coeffs_np, 'aare': aare_poly}
    print(f"  Poly degree {deg}:  AARE = {aare_poly:.4f}%  ({deg+1} coeffs)")
    if aare_poly < best_aare:
        best_aare, best_deg, best_coeffs = aare_poly, deg, coeffs_np

print(f"\n  Best overall: {'constant' if best_deg is None else f'polynomial degree {best_deg}'}"
      f"  (AARE = {best_aare:.4f}%)")

# ============================================================================
# PART 4 -- Print result for insertion into constants.py
# ============================================================================
print("\n" + "=" * 68)
print("Step 4: Recommended value(s) for ecpa/constants.py")
print("=" * 68)
print(f"\n  Option A (constant, matches current Peneloux_CO2 form):")
print(f"    Peneloux_CO2 = {c_best_const*1e-6:.6e}   # m3/mol  (AARE={aare_const:.3f}%, was 0.0 -> {aare_noshift_all:.3f}%)")
if best_deg is not None:
    print(f"\n  Option B (T-dependent, degree {best_deg}, AARE={best_aare:.3f}%):")
    print(f"    _PENELOUX_CO2_COEFFS = np.array({best_coeffs.tolist()})")
    print(f"    _PENELOUX_CO2_TC     = {Tc_CO2}  # K")

# ============================================================================
# PART 5 -- Diagnostic figures
# ============================================================================
print("\n" + "=" * 68)
print("Step 5: Generating figures ...")
print("=" * 68)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
TR_plot = np.linspace(TR_fit.min() - 0.01, TR_fit.max() + 0.01, 300)
T_C_plot = TR_plot * Tc_CO2 - 273.15

ax = axes[0]
ax.plot(T_fit - 273.15, c_fit, 'o', color=WONG[0], ms=6, zorder=5,
        label='Per-$T$ optimal $c$')
for i, (deg, res) in enumerate(poly_results.items()):
    c_line = np.polyval(res['coeffs'], TR_plot)
    ax.plot(T_C_plot, c_line, color=WONG[(i+1) % len(WONG)],
            linestyle=['--', '-.', ':', '-', (0, (3,1,1,1,1,1))][i % 5],
            label=f'deg {deg} (AARE={res["aare"]:.2f}%)')
ax.axhline(0, color='gray', lw=1.2, ls='--')
ax.axhline(c_best_const, color=WONG[6], lw=1.6, ls=':',
           label=f'Best constant ({c_best_const:.3f})')
ax.set_xlabel('$T$ [$^\\circ$C]')
ax.set_ylabel('$c$ [cm$^3$ mol$^{-1}$]')
ax.legend(fontsize=8, ncol=1, loc='best')
ax.grid(True, ls=':', alpha=0.4)

ax = axes[1]
T_C_arr, aare_noshift_arr, aare_const_arr, aare_bestpoly_arr = [], [], [], []
for T in T_fit:
    sub = df[df['T_K'] == T]
    if sub.empty: continue
    T_C_arr.append(T - 273.15)
    rho_ref_t = sub['rho_ref'].values
    Vm_t = sub['Vm_EoS'].values
    rho_noshift = M_CO2_g * 1000.0 / Vm_t
    rho_const = M_CO2_g * 1000.0 / (Vm_t + c_best_const)
    aare_noshift_arr.append(np.mean(np.abs(rho_noshift - rho_ref_t) / rho_ref_t) * 100)
    aare_const_arr.append(np.mean(np.abs(rho_const - rho_ref_t) / rho_ref_t) * 100)
    if best_deg is not None:
        TR_t = T / Tc_CO2
        c_best = float(np.polyval(best_coeffs, TR_t))
        rho_best = M_CO2_g * 1000.0 / (Vm_t + c_best)
        aare_bestpoly_arr.append(np.mean(np.abs(rho_best - rho_ref_t) / rho_ref_t) * 100)

ax.plot(T_C_arr, aare_noshift_arr, color=WONG[0], linestyle='--', label='No shift (current paper)')
ax.plot(T_C_arr, aare_const_arr, color=WONG[5], linestyle='-.', label='Best constant')
if best_deg is not None:
    ax.plot(T_C_arr, aare_bestpoly_arr, color=WONG[6], linestyle='-', label=f'Poly deg {best_deg}')
ax.set_xlabel('$T$ [$^\\circ$C]')
ax.set_ylabel('AARE in $\\rho$ [%]')
ax.legend(fontsize=9)
ax.set_yscale('log')
ax.grid(True, which='both', ls=':', alpha=0.4)

fig.tight_layout()
fig.savefig('figures/density/peneloux_co2_opt_c.png')
plt.close(fig)
print("  Saved figures/density/peneloux_co2_opt_c.png")

# Parity plot: no-shift vs best-constant vs best-poly
n_panels = 3 if best_deg is not None else 2
fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 5.5))
cmap = plt.cm.viridis
norm_T = plt.Normalize(vmin=df['T_C'].min(), vmax=df['T_C'].max())

rho_noshift_all = M_CO2_g * 1000.0 / df['Vm_EoS'].values
rho_const_all   = M_CO2_g * 1000.0 / (df['Vm_EoS'].values + c_best_const)
panels = [(axes[0], rho_noshift_all, 'No shift', aare_noshift_all),
          (axes[1], rho_const_all, 'Best constant', aare_const)]
if best_deg is not None:
    TR_all = df['T_K'].values / Tc_CO2
    rho_poly_all = M_CO2_g * 1000.0 / (df['Vm_EoS'].values + np.polyval(best_coeffs, TR_all))
    panels.append((axes[2], rho_poly_all, f'Poly deg {best_deg}', best_aare))

for ax, rho_pred_arr, title, aare_val in panels:
    sca = ax.scatter(df['rho_ref'], rho_pred_arr, c=df['T_C'], cmap=cmap,
                      norm=norm_T, s=45, zorder=4, edgecolors='black', linewidths=0.4)
    lo = min(df['rho_ref'].min(), rho_pred_arr.min()) * 0.95
    hi = max(df['rho_ref'].max(), rho_pred_arr.max()) * 1.05
    lv = np.linspace(lo, hi, 300)
    ax.plot(lv, lv, 'k-', lw=1.4)
    ax.fill_between(lv, lv*0.98, lv*1.02, color=WONG[3], alpha=0.20, label='$\\pm$2%')
    ax.fill_between(lv, lv*0.90, lv*1.10, color=WONG[1], alpha=0.12, label='$\\pm$10%')
    ax.set_xlabel(r'Span--Wagner $\rho_{\rm CO_2}$ [kg m$^{-3}$]')
    ax.set_ylabel(r'eCPA $\rho_{\rm CO_2}$ [kg m$^{-3}$]')
    ax.set_title(f'{title} (AARE={aare_val:.2f}%)', fontsize=12)
    ax.legend(fontsize=8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.grid(True, ls=':', alpha=0.35)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_T)
sm.set_array([])
fig.colorbar(sm, ax=axes, label='$T$ [$^\\circ$C]', shrink=0.8, pad=0.02)
fig.savefig('figures/density/peneloux_co2_parity.png')
plt.close(fig)
print("  Saved figures/density/peneloux_co2_parity.png")

print("\nDone.")
