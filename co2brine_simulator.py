"""
CO₂-brine compositional reservoir simulator (IMPES, 2D).

Generalises the immiscible two-phase simulator to include interphase CO₂/H₂O
transfer via eCPA flash computations (flash_co2_h2o_salt_fast).

Physical model
--------------
- Isothermal Darcy flow on a 2D Cartesian grid (Nz = 1)
- Two mobile phases: aqueous (brine) and CO₂-rich
- Interphase CO₂/H₂O transfer via eCPA stability+flash at every cell, every step
- Constant total molar density (∇·u = q):  TPFA pressure equation unchanged
  from the immiscible case; density-driven compressibility is neglected

Transported scalar: overall CO₂ mole fraction  z  per cell.

Flash maps (T, P, z, ms) → (β, S_g, x_CO₂^aq, y_CO₂^CO₂-rich, ms_aq, Z_aq, Z_c)

Molar fractional flow of CO₂:
    F_z = (λ_g · y_CO₂  +  λ_w · x_CO₂) / (λ_g + λ_w)
where λ = k_r / μ,  and compositions come from flash.

NaCl is treated as non-volatile; feed molality ms is held uniform at ms0
(NaCl advection neglected).  The salting-out effect (ms_aq > ms0 in two-phase
cells) is captured correctly by flash.

Usage
-----
    python co2brine_simulator.py
"""

from __future__ import annotations
import os, warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import get_context

# ── eCPA imports ──────────────────────────────────────────────────────────────
from ecpa.parameters import make_params
from ecpa.solution_table import make_solution_guess_fn
from ecpa.flash import flash_co2_h2o_salt_fast

R_GAS  = 8.314      # J/(mol·K)
MW_H2O = 0.018015   # kg/mol

# =============================================================================
# 1.  GRID AND ROCK
# =============================================================================
Lx, Ly, Lz = 100.0, 100.0, 10.0      # domain dimensions [m]
Nx, Ny, Nz  = 50, 50, 1               # grid cells
N  = Nx * Ny * Nz
dx, dy, dz  = Lx/Nx, Ly/Ny, Lz/Nz

phi    = 0.20                           # porosity (uniform)
gridV  = dx * dy * dz                  # cell volume [m³]
gridPV = phi * gridV                   # pore volume per cell [m³]
PVtot  = N * gridPV

# Log-normal permeability field (σ_ln = 0.6, mean = 100 mD)
mD      = 9.869e-16                    # 1 milliDarcy [m²]
rng     = np.random.default_rng(42)
K_field = 100.0 * mD * np.ones((Nx, Ny))
K       = np.ones((3, Nx, Ny, Nz))
K[0, :, :, 0] = K_field   # kx
K[1, :, :, 0] = K_field   # ky
K[2, :, :, 0] = 100.0 * mD

# =============================================================================
# 2.  FLUID PROPERTIES
# =============================================================================
T_K   = 350.0    # K   – isothermal
P_ref = 200.0    # bar – reference pressure for flash (uniform)
ms0   = 1.0      # mol/kg – initial NaCl molality (uniform)

mu_co2   = 5.0e-5   # Pa·s  (CO₂-rich phase, ~0.05 mPa·s at 350 K, 200 bar)
mu_brine = 4.5e-4   # Pa·s  (aqueous phase,  ~0.45 mPa·s at 350 K)

# Corey relative-permeability parameters
Swc, Sgr   = 0.15, 0.05    # connate water, residual gas saturations
krg0, krw0 = 0.80, 1.00    # endpoint relative permeabilities
ng,   nw   = 2.0,  2.5     # Corey exponents

def relperm(S_g: np.ndarray):
    """Corey k_r for CO₂-rich (g) and aqueous (w) phases."""
    Se_g = ((S_g       - Sgr) / (1 - Swc - Sgr)).clip(0, 1)
    Se_w = ((1 - S_g - Swc) / (1 - Swc - Sgr)).clip(0, 1)
    return krg0 * Se_g**ng, krw0 * Se_w**nw

# =============================================================================
# 3.  WELLS AND TIME STEPPING
# =============================================================================
year_s    = 365.25 * 24 * 3600
Q_m3s     = 0.10 * PVtot / year_s          # inject 10 % PV / yr  [m³/s]

# 5-spot: injector at centre, producers at 4 corners
inj_i, inj_j = Nx // 2, Ny // 2
Qijk      = np.zeros((Nx, Ny, Nz))
Qijk[inj_i, inj_j, 0] =  Q_m3s
for ci, cj in [(0, 0), (Nx-1, 0), (0, Ny-1), (Nx-1, Ny-1)]:
    Qijk[ci, cj, 0] = -Q_m3s / 4.0
Q         = Qijk.reshape((N, 1), order='F')

t_max_yr  = 3.0         # years  (breakthrough at ~7 yr in 5-spot)
n_steps   = 100         # outer (pressure + flash) timesteps

z_inject  = 1.0         # pure CO₂ injected
z_initial = 5e-4        # initial z ≈ pure brine

# =============================================================================
# 4.  MULTIPROCESSING WORKER  (module-level so pickling works)
# =============================================================================
_W_params   = None
_W_guess_fn = None

def _worker_init(params, grid_data):
    global _W_params, _W_guess_fn
    _W_params   = params
    _W_guess_fn = make_solution_guess_fn(grid_data)

# z thresholds for fast single-phase bypass (approximate, valid at T=350K, P=200bar)
_Z_LO = 0.015    # below → single-phase brine  (no flash needed)
_Z_HI = 0.97     # above → single-phase CO₂-rich (no flash needed)

def _single_phase_brine(z_co2: float, ms: float) -> dict:
    return dict(phase='single_phase', beta=0.0, x4w=float(z_co2),
                x4c=0.0, Z_aq=0.90, Z_c=0.60, ms_aq=float(ms))

def _single_phase_co2(z_co2: float, ms: float) -> dict:
    return dict(phase='single_phase', beta=1.0, x4w=0.0,
                x4c=float(z_co2), Z_aq=0.90, Z_c=0.60, ms_aq=float(ms))

def _flash_one(args: tuple) -> dict:
    """Flash one cell (called only for potential two-phase cells)."""
    T, P, z_co2, ms = args
    z_co2 = float(np.clip(z_co2, 1e-6, 1 - 1e-6))
    try:
        r = flash_co2_h2o_salt_fast(
            T=T, P_bar=P, z_co2=z_co2, m_tot=ms,
            params=_W_params, solution_guess_fn=_W_guess_fn,
        )
    except Exception:
        # Flash failed: fall back to single-phase based on z
        return (_single_phase_brine(z_co2, ms) if z_co2 < 0.4
                else _single_phase_co2(z_co2, ms))

    if r['phase'] == 'two_phase':
        return dict(
            phase ='two_phase',
            beta  = float(r['beta']),
            x4w   = float(r['x_aq']['x4w']),   # CO₂ mol frac in aqueous
            x4c   = float(r['x_c' ]['x4c']),   # CO₂ mol frac in CO₂-rich
            Z_aq  = float(r['Z_aq']),
            Z_c   = float(r['Z_c']),
            ms_aq = float(r['ms_aq']),
        )
    else:
        brine = (z_co2 < 0.4)
        return (_single_phase_brine(z_co2, ms) if brine
                else _single_phase_co2(z_co2, ms))

def run_flash_parallel(z_arr: np.ndarray, ms: float,
                       T: float, P: float, pool) -> dict:
    """
    Flash all N cells; skip trivially single-phase cells (z < Z_LO or z > Z_HI).
    Two-phase candidates are dispatched in parallel via the worker pool.
    Returns dict of shape-(N,) arrays.
    """
    N_ = len(z_arr)
    # Default: single-phase brine
    out_beta  = np.zeros(N_)
    out_x4w   = np.zeros(N_)
    out_x4c   = np.zeros(N_)
    out_Z_aq  = np.full(N_, 0.90)
    out_Z_c   = np.full(N_, 0.60)
    out_ms_aq = np.full(N_, ms)
    out_phase = ['single_phase'] * N_

    # Mark CO₂-rich single-phase cells
    co2_mask = z_arr > _Z_HI
    out_beta[co2_mask] = 1.0
    out_x4c[co2_mask]  = z_arr[co2_mask]

    # Dispatch potential two-phase cells to pool
    tp_idx = np.where((z_arr >= _Z_LO) & (z_arr <= _Z_HI))[0]
    if len(tp_idx) > 0:
        args = [(T, P, float(z_arr[i]), ms) for i in tp_idx]
        chunk = max(1, len(tp_idx) // (pool._processes or 1))
        raw   = pool.map(_flash_one, args, chunksize=chunk)
        for j, res in zip(tp_idx, raw):
            out_beta[j]  = res['beta']
            out_x4w[j]   = res['x4w']
            out_x4c[j]   = res['x4c']
            out_Z_aq[j]  = res['Z_aq']
            out_Z_c[j]   = res['Z_c']
            out_ms_aq[j] = res['ms_aq']
            out_phase[j] = res['phase']

    return dict(beta=out_beta, x4w=out_x4w, x4c=out_x4c,
                Z_aq=out_Z_aq, Z_c=out_Z_c, ms_aq=out_ms_aq,
                phase=out_phase, n_flash=len(tp_idx))

# =============================================================================
# 5.  BETA → VOLUME SATURATION
# =============================================================================
def beta_to_Sg(fr: dict, T: float, P_bar: float) -> np.ndarray:
    """
    S_g = β·Vm_c / (β·Vm_c + (1−β)·Vm_w)
    where molar volumes Vm = Z·R·T/P come from flash Z-factors.
    """
    P_Pa = P_bar * 1e5
    Vm_c = fr['Z_c']  * R_GAS * T / P_Pa
    Vm_w = fr['Z_aq'] * R_GAS * T / P_Pa
    b    = fr['beta']
    den  = b * Vm_c + (1.0 - b) * Vm_w
    return np.where(den > 0, b * Vm_c / den, 0.0).clip(0, 1)

# =============================================================================
# 6.  FRACTIONAL FLOW OF CO₂  (molar)
# =============================================================================
def compute_Fz(S_g: np.ndarray, x4w: np.ndarray, x4c: np.ndarray):
    """
    Molar fractional flow of CO₂:
        F_z = (λ_g·y_CO₂ + λ_w·x_CO₂) / (λ_g + λ_w)
    Returns F_z and total mobility λ_t (both shape (N,)).
    """
    kr_g, kr_w = relperm(S_g)
    lam_g = kr_g / mu_co2
    lam_w = kr_w / mu_brine
    lam_t = lam_g + lam_w + 1e-40
    F_z   = (lam_g * x4c + lam_w * x4w) / lam_t
    return F_z, lam_t

# =============================================================================
# 7.  TPFA PRESSURE SOLVER  (identical to immiscible simulator)
# =============================================================================
def TPFA(Lx, Ly, Lz, Nx, Ny, Nz, Q, K):
    dx_, dy_, dz_ = Lx/Nx, Ly/Ny, Lz/Nz
    N_ = Nx * Ny * Nz
    iK  = 1.0 / K
    tx, ty, tz = 2*dy_*dz_/dx_, 2*dx_*dz_/dy_, 2*dx_*dy_/dz_

    TX = np.zeros((Nx+1, Ny, Nz));  TY = np.zeros((Nx, Ny+1, Nz));  TZ = np.zeros((Nx, Ny, Nz+1))
    TX[1:Nx, :, :]  = tx / (iK[0, :-1, :,  :] + iK[0, 1:,  :,  :])
    TY[:,  1:Ny, :] = ty / (iK[1,  :, :-1, :] + iK[1,  :, 1:,  :])
    TZ[:,   :, 1:Nz] = tz / (iK[2, :,  :, :-1] + iK[2,  :,  :, 1:])

    x1=TX[:Nx,:,:].reshape(N_,order='F'); x2=TX[1:,:,:].reshape(N_,order='F')
    y1=TY[:,:Ny,:].reshape(N_,order='F'); y2=TY[:,1:,:].reshape(N_,order='F')
    z1=TZ[:,:,:Nz].reshape(N_,order='F'); z2=TZ[:,:,1:].reshape(N_,order='F')

    md    = x1+x2+y1+y2+z1+z2
    md[0] += np.sum(K[:, 0, 0, 0])
    A = sp.diags([-z2[:-Nx*Ny],-y2[:-Nx],-x2[:-1], md,-x1[1:],-y1[Nx:],-z1[Nx*Ny:]],
                 [-Nx*Ny,-Nx,-1,0,1,Nx,Nx*Ny])
    P_ = spla.spsolve(A, Q.ravel()).reshape(Nx, Ny, Nz, order='F')

    Vx=np.zeros((Nx+1,Ny,Nz)); Vy=np.zeros((Nx,Ny+1,Nz)); Vz=np.zeros((Nx,Ny,Nz+1))
    Vx[1:Nx,:,:]  = (P_[:-1,:,:]  - P_[1:,:,:])  * TX[1:Nx,:,:]
    Vy[:,1:Ny,:]  = (P_[:,:-1,:]  - P_[:,1:,:])  * TY[:,1:Ny,:]
    Vz[:,:,1:Nz]  = (P_[:,:,:-1]  - P_[:,:,1:])  * TZ[:,:,1:Nz]
    return P_, Vx, Vy, Vz

# =============================================================================
# 8.  UPWINDING MATRIX  (identical to immiscible simulator)
# =============================================================================
def upwindingmatrix(Nx, Ny, Nz, Vx, Vy, Vz, Q, maxdFdz):
    N_ = Nx * Ny * Nz
    inj  = Q.ravel().clip(min=0)
    prod = Q.ravel().clip(max=0)

    XN=Vx.clip(max=0); XP=Vx.clip(min=0)
    YN=Vy.clip(max=0); YP=Vy.clip(min=0)
    ZN=Vz.clip(max=0); ZP=Vz.clip(min=0)

    x1=XN[:Nx,:,:].reshape(N_,order='F'); x2=XP[1:,:,:].reshape(N_,order='F')
    y1=YN[:,:Ny,:].reshape(N_,order='F'); y2=YP[:,1:,:].reshape(N_,order='F')
    z1=ZN[:,:,:Nz].reshape(N_,order='F'); z2=ZP[:,:,1:].reshape(N_,order='F')

    md = x1-x2+y1-y2+z1-z2 + prod
    UPW = sp.diags([z2[:-Nx*Ny], y2[:-Nx], x2[:-1], md,
                    -x1[1:], -y1[Nx:], -z1[Nx*Ny:]],
                   [-Nx*Ny, -Nx, -1, 0, 1, Nx, Nx*Ny])

    Vin = ((XP[:Nx,:,:] + YP[:,:Ny,:] + ZP[:,:,:Nz]
           -XN[1:,:,:]  - YN[:,1:,:]  - ZN[:,:,1:])
           .reshape(N_, order='F') + inj).clip(min=1e-30)
    CFL = np.min(gridPV / Vin) / maxdFdz
    return UPW, CFL

# =============================================================================
# 9.  STARTUP: estimate max dF_z/dz for CFL condition
# =============================================================================
def estimate_maxdFdz(params, guess_fn, T, P, ms, n_pts=60) -> float:
    """
    Scan z ∈ [0.01, 0.95] and return the maximum slope of F_z(z),
    used in the CFL stability condition for the transport sub-steps.
    """
    z_scan = np.linspace(0.01, 0.95, n_pts)
    F_scan = np.full(n_pts, np.nan)
    for i, zv in enumerate(z_scan):
        try:
            r = flash_co2_h2o_salt_fast(T=T, P_bar=P, z_co2=float(zv),
                                        m_tot=ms, params=params, solution_guess_fn=guess_fn)
            if r['phase'] == 'two_phase':
                b     = float(r['beta'])
                P_Pa  = P * 1e5
                Vm_c  = float(r['Z_c'])  * R_GAS * T / P_Pa
                Vm_w  = float(r['Z_aq']) * R_GAS * T / P_Pa
                den   = b*Vm_c + (1-b)*Vm_w
                Sg_i  = (b*Vm_c / den) if den > 0 else 0.0
                x4w_i = np.array([float(r['x_aq']['x4w'])])
                x4c_i = np.array([float(r['x_c']['x4c'])])
                Fz_i, _ = compute_Fz(np.array([Sg_i]), x4w_i, x4c_i)
                F_scan[i] = float(Fz_i[0])
        except Exception:
            pass
    ok = ~np.isnan(F_scan)
    if ok.sum() < 2:
        return 2.5   # conservative fallback
    dF  = np.abs(np.diff(F_scan[ok]))
    dz  = np.diff(z_scan[ok])
    return float(np.nanmax(dF / (dz + 1e-12))) * 1.2   # 20 % safety margin

# =============================================================================
# 10.  DIAGNOSTICS
# =============================================================================
def dissolved_co2_molality(x4w: np.ndarray, ms_aq: np.ndarray) -> np.ndarray:
    """
    CO₂ molality in the aqueous phase [mol/kg H₂O].

    From phase composition: x1w + 2·x1w·ms_aq·Mw + x4w = 1  (Na⁺, Cl⁻ symmetric)
    → x1w = (1 − x4w) / (1 + 2·ms_aq·Mw)
    → m_c  = x4w / (x1w · Mw)
    """
    x1w = (1.0 - x4w) / (1.0 + 2.0 * ms_aq * MW_H2O + 1e-12)
    x1w = np.maximum(x1w, 1e-9)
    return x4w / (x1w * MW_H2O)

# =============================================================================
# 11.  PLOTTING
# =============================================================================
def plot_snapshot(z: np.ndarray, fr: dict, T: float, P: float,
                  step: int, t_yr: float, outdir: str = 'figures/simulator'):
    os.makedirs(outdir, exist_ok=True)

    S_g   = beta_to_Sg(fr, T, P).reshape(Nx, Ny, order='F')
    mc    = dissolved_co2_molality(fr['x4w'], fr['ms_aq']).reshape(Nx, Ny, order='F')
    ms_aq = fr['ms_aq'].reshape(Nx, Ny, order='F')
    z2d   = z.reshape(Nx, Ny, order='F')

    XX = np.linspace(dx/2, Lx - dx/2, Nx)   # cell centres [m]
    YY = np.linspace(dy/2, Ly - dy/2, Ny)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    fig.suptitle(f't = {t_yr:.2f} yr  (step {step})', fontsize=11)

    panels = [
        (z2d,   r'Overall CO$_2$ mol fraction $z$',             'viridis', None),
        (S_g,   r'CO$_2$-rich saturation $S_g$',                'plasma',  (0, 1)),
        (mc,    r'Dissolved CO$_2$ $m_c$ [mol kg$^{-1}$]',      'Blues',   None),
        (ms_aq, r'Equilibrium brine $m_s^\mathrm{aq}$ [mol kg$^{-1}$]', 'Oranges', None),
    ]
    for ax, (data, title, cmap, vlim) in zip(axes.flat, panels):
        kw = dict(vmin=vlim[0], vmax=vlim[1]) if vlim else {}
        im = ax.pcolormesh(XX, YY, data.T, cmap=cmap, shading='auto', **kw)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('x [m]', fontsize=8)
        ax.set_ylabel('y [m]', fontsize=8)
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax, pad=0.02)

    # Mark wells: injector (star) at centre, producers (triangle) at corners
    xc, yc = XX[inj_i], YY[inj_j]
    for ax in axes.flat:
        ax.plot(xc, yc, '*', color='white', ms=10, zorder=5, label='inj')
        for ci, cj in [(0,0),(Nx-1,0),(0,Ny-1),(Nx-1,Ny-1)]:
            ax.plot(XX[ci], YY[cj], '^', color='black', ms=7, zorder=5)

    plt.tight_layout()
    fname = f'{outdir}/snap_{step:03d}.png'
    plt.savefig(fname, dpi=120)
    plt.close()
    return fname

# =============================================================================
# 12.  MAIN SIMULATION LOOP
# =============================================================================
def main():
    print("Loading eCPA parameters and solution table …")
    params   = make_params()
    npz       = np.load('results/solution_table.npz')
    grid_data = dict(npz)
    guess_fn  = make_solution_guess_fn(grid_data)
    print("  done.\n")

    # ── Estimate CFL parameter (single-threaded 1D scan) ──────────────────
    print("Estimating max dF_z/dz for CFL …", end=' ', flush=True)
    maxdFdz = estimate_maxdFdz(params, guess_fn, T_K, P_ref, ms0)
    print(f"max dF/dz = {maxdFdz:.2f}\n")

    # ── Spawn parallel pool ───────────────────────────────────────────────
    ctx  = get_context('spawn')
    pool = ctx.Pool(initializer=_worker_init, initargs=(params, grid_data))

    # ── Initial conditions ────────────────────────────────────────────────
    z       = z_initial * np.ones(N)
    inj     = Q.ravel().clip(min=0)
    inj_src = inj * z_inject               # CO₂ source at injector [m³/s]
    dt_big  = t_max_yr * year_s / n_steps

    timing     = []
    snap_every = max(1, n_steps // 8)

    print(f"Grid: {Nx}×{Ny} = {N} cells  |  {t_max_yr:.0f} yr  |  "
          f"{n_steps} steps  |  dt = {dt_big/year_s:.3f} yr/step\n")
    print(f"{'Step':>5}  {'t [yr]':>7}  {'N_sub':>5}  {'N_flash':>7}  "
          f"{'t_flash [s]':>11}  {'t_step [s]':>10}  {'flash/s':>8}")
    print("─" * 66)

    t0_total = time.perf_counter()

    for step in range(1, n_steps + 1):
        t_yr = step * dt_big / year_s
        t0   = time.perf_counter()

        # 1. Flash all cells in parallel
        fr      = run_flash_parallel(z, ms0, T_K, P_ref, pool)
        t_flash = time.perf_counter() - t0

        # 2. Phase saturations and molar fractional flow of CO₂
        S_g       = beta_to_Sg(fr, T_K, P_ref)
        Fz, lam_t = compute_Fz(S_g, fr['x4w'], fr['x4c'])

        # 3. Total-mobility effective permeability → pressure solve
        lam_3d = np.stack([lam_t.reshape(Nx, Ny, Nz, order='F')] * 3, axis=0)
        Keff   = K * lam_3d
        _, Vx, Vy, Vz = TPFA(Lx, Ly, Lz, Nx, Ny, Nz, Q, Keff)

        # 4. CFL-limited transport sub-steps
        UPW, CFL = upwindingmatrix(Nx, Ny, Nz, Vx, Vy, Vz, Q, maxdFdz)
        Nt  = int(np.ceil(dt_big / CFL))
        dtx = (dt_big / Nt) / gridPV
        for _ in range(Nt):
            z = z + (UPW.dot(Fz) + inj_src) * dtx
        z = z.clip(0, 1)

        t_step  = time.perf_counter() - t0
        n_flash = fr['n_flash']
        timing.append((step, t_yr, t_flash, t_step, n_flash))
        rate = f"{n_flash/t_flash:.0f}" if t_flash > 1e-4 and n_flash > 0 else "   —"
        print(f"{step:5d}  {t_yr:7.2f}  {Nt:5d}  {n_flash:7d}  "
              f"{t_flash:11.2f}  {t_step:10.2f}  {rate:>8}")

        if step % snap_every == 0 or step == n_steps:
            fname = plot_snapshot(z, fr, T_K, P_ref, step, t_yr)
            print(f"          → saved {fname}")

    pool.close()
    pool.join()

    wall = time.perf_counter() - t0_total
    t_fl  = np.array([t[2] for t in timing])
    t_st  = np.array([t[3] for t in timing])
    n_fls = np.array([t[4] for t in timing])
    t_yrs = np.array([t[1] for t in timing])

    print(f"\n{'='*66}")
    print(f"Finished in {wall:.1f} s")
    print(f"  Mean flash calls / step: {n_fls.mean():.0f}  "
          f"(max {n_fls.max():.0f})")
    print(f"  Mean flash time / step : {t_fl.mean():.3f} s  "
          f"({t_fl.sum()/t_st.sum()*100:.0f}% of runtime)")
    print(f"  Mean total time / step : {t_st.mean():.3f} s")
    nf_pos = n_fls[n_fls > 0]
    tf_pos = t_fl[n_fls > 0]
    if len(nf_pos):
        print(f"  Flash throughput       : {(nf_pos/tf_pos).mean():.0f} calls/s")

    # ── Performance figure ────────────────────────────────────────────────
    os.makedirs('figures/simulator', exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    axes[0].plot(t_yrs, t_fl, lw=1.5, label='Flash (two-phase cells)')
    axes[0].plot(t_yrs, t_st, lw=1.5, ls='--', label='Total step')
    axes[0].set_ylabel('Wall time [s]')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].bar(t_yrs, n_fls, width=t_yrs[1]-t_yrs[0]*0.8, alpha=0.7,
                label='Two-phase cells flashed')
    axes[1].axhline(N, color='grey', ls=':', lw=1, label=f'Total cells ({N})')
    axes[1].set_xlabel('Simulation time [yr]')
    axes[1].set_ylabel('Cells flashed')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f'eCPA flash performance — {Nx}×{Ny} grid, T = {T_K} K, '
                 f'P = {P_ref} bar, ms = {ms0} mol/kg', fontsize=10)
    plt.tight_layout()
    plt.savefig('figures/simulator/performance.png', dpi=150)
    plt.close()
    print("  Saved figures/simulator/performance.png")


if __name__ == '__main__':
    main()
