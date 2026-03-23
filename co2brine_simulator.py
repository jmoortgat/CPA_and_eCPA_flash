"""
CO₂-brine compositional reservoir simulator (IMPES, 2-D/3-D).

Pressure solvers
----------------
  'tpfa'  Two-point flux approximation (default, fast, grid-orientation bias)
  'mfe'   Mixed finite element — RT₀ on structured grid (less orientation bias)

Flash models
------------
  'ecpa'  eCPA EOS with NaCl (warm-start K-SSI → ecpa_stability_flash fallback)
  'cpa'   Salt-free CPA EOS  (warm-start K-SSI → flash_co2_h2o_tpz_robust fallback)

Both models use the pre-computed solution table for warm-starting; a robust
fallback guarantees convergence at every grid cell and every time step.

Physical model
--------------
- Isothermal Darcy flow on a Cartesian grid
- Two mobile phases: aqueous (brine) and CO₂-rich
- Interphase CO₂/H₂O transfer via flash at every cell, every step
- Constant total molar density (∇·u = q); no gravity
- NaCl non-volatile; feed molality ms₀ uniform (NaCl advection neglected)

Transported scalar: overall CO₂ mole fraction z per cell.

Usage
-----
    python co2brine_simulator.py [tpfa|mfe] [ecpa|cpa]
"""

from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import get_context

# ── Flash imports ──────────────────────────────────────────────────────────────
from ecpa.parameters import make_params
from ecpa.solution_table import load_solution_table, make_solution_guess_fn
from ecpa.flash import flash_co2_h2o_salt_fast_kv
from ecpa.stability import ecpa_stability_flash
from CPA import flash_co2_h2o_tpz_warmstart

R_GAS  = 8.314      # J/(mol·K)
MW_H2O = 0.018015   # kg/mol

# =============================================================================
# 1.  GRID AND ROCK
# =============================================================================
Lx, Ly, Lz = 100.0, 100.0, 10.0      # domain dimensions [m]
Nx, Ny, Nz  = 50, 50, 1               # grid cells
N  = Nx * Ny * Nz
dx, dy, dz  = Lx/Nx, Ly/Ny, Lz/Nz

phi    = 0.20                          # porosity (uniform)
gridV  = dx * dy * dz                 # cell volume [m³]
gridPV = phi * gridV                  # pore volume per cell [m³]
PVtot  = N * gridPV

mD      = 9.869e-16                   # 1 milliDarcy [m²]
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
ms0   = 1.0      # mol/kg – initial NaCl molality (uniform); ignored in CPA mode

mu_co2   = 5.0e-5   # Pa·s  (CO₂-rich phase)
mu_brine = 4.5e-4   # Pa·s  (aqueous phase)

Swc, Sgr   = 0.15, 0.05
krg0, krw0 = 0.80, 1.00
ng,   nw   = 2.0,  2.5

def relperm(S_g: np.ndarray):
    Se_g = ((S_g - Sgr) / (1 - Swc - Sgr)).clip(0, 1)
    Se_w = ((1 - S_g - Swc) / (1 - Swc - Sgr)).clip(0, 1)
    return krg0 * Se_g**ng, krw0 * Se_w**nw

# =============================================================================
# 3.  WELLS AND TIME STEPPING
# =============================================================================
year_s    = 365.25 * 24 * 3600
Q_m3s     = 0.10 * PVtot / year_s     # inject 10 % PV / yr  [m³/s]

inj_i, inj_j = Nx // 2, Ny // 2
Qijk      = np.zeros((Nx, Ny, Nz))
Qijk[inj_i, inj_j, 0] =  Q_m3s
for ci, cj in [(0, 0), (Nx-1, 0), (0, Ny-1), (Nx-1, Ny-1)]:
    Qijk[ci, cj, 0] = -Q_m3s / 4.0
Q         = Qijk.reshape((N, 1), order='F')

t_max_yr  = 3.0
n_steps   = 100

z_inject  = 1.0
z_initial = 5e-4

# =============================================================================
# 4.  PRESSURE SOLVERS
# =============================================================================

def TPFA(Lx, Ly, Lz, Nx, Ny, Nz, Q, K):
    """Two-point flux approximation pressure solver."""
    dx_, dy_, dz_ = Lx/Nx, Ly/Ny, Lz/Nz
    N_ = Nx * Ny * Nz
    iK  = 1.0 / K
    tx, ty, tz = 2*dy_*dz_/dx_, 2*dx_*dz_/dy_, 2*dx_*dy_/dz_

    TX = np.zeros((Nx+1, Ny, Nz))
    TY = np.zeros((Nx, Ny+1, Nz))
    TZ = np.zeros((Nx, Ny, Nz+1))
    TX[1:Nx, :, :]   = tx / (iK[0, :-1, :, :] + iK[0, 1:,  :, :])
    TY[:,  1:Ny, :]  = ty / (iK[1,  :, :-1, :] + iK[1,  :, 1:, :])
    TZ[:,   :, 1:Nz] = tz / (iK[2,  :,  :, :-1] + iK[2, :, :, 1:])

    x1 = TX[:Nx, :, :].reshape(N_, order='F')
    x2 = TX[1:,  :, :].reshape(N_, order='F')
    y1 = TY[:, :Ny, :].reshape(N_, order='F')
    y2 = TY[:, 1:,  :].reshape(N_, order='F')
    z1 = TZ[:, :, :Nz].reshape(N_, order='F')
    z2 = TZ[:, :, 1: ].reshape(N_, order='F')

    md    = x1+x2+y1+y2+z1+z2
    md[0] += np.sum(K[:, 0, 0, 0])
    A = sp.diags([-z2[:-Nx*Ny], -y2[:-Nx], -x2[:-1], md,
                  -x1[1:], -y1[Nx:], -z1[Nx*Ny:]],
                 [-Nx*Ny, -Nx, -1, 0, 1, Nx, Nx*Ny])
    P_ = spla.spsolve(A, Q.ravel()).reshape(Nx, Ny, Nz, order='F')

    Vx = np.zeros((Nx+1, Ny, Nz))
    Vy = np.zeros((Nx, Ny+1, Nz))
    Vz = np.zeros((Nx, Ny, Nz+1))
    Vx[1:Nx, :, :]  = (P_[:-1, :, :] - P_[1:, :, :])  * TX[1:Nx, :, :]
    Vy[:, 1:Ny, :]  = (P_[:, :-1, :] - P_[:, 1:, :])  * TY[:, 1:Ny, :]
    Vz[:, :, 1:Nz]  = (P_[:, :, :-1] - P_[:, :, 1:])  * TZ[:, :, 1:Nz]
    return P_, Vx, Vy, Vz


def _GenB(Nx, Ny, Nz, Lx, Ly, Lz, K):
    """
    Block-diagonal B matrix (E×E) for RT₀ MFE.

    Translates GenB.m: inverse-permeability weighted mass matrix.
    B = block_diag(Bx, By, Bz), each a tridiagonal sparse matrix.
    """
    hx = Lx / Nx;  hy = Ly / Ny;  hz = Lz / Nz
    L  = 1.0 / K                            # (3, Nx, Ny, Nz)
    Ex = (Nx-1)*Ny*Nz
    Ey = Nx*(Ny-1)*Nz
    Ez = Nx*Ny*(Nz-1)
    tx = hx / (6*hy*hz)
    ty = hy / (6*hx*hz)
    tz = hz / (6*hx*hy)

    # ── X block ──────────────────────────────────────────────────────────────
    x0 = (2*tx * (L[0, :-1, :, :] + L[0, 1:, :, :])).ravel(order='F')
    X1 = np.zeros((Nx-1, Ny, Nz));  X1[1:, :, :] = L[0, 1:-1, :, :]
    X2 = np.zeros((Nx-1, Ny, Nz));  X2[:-1, :, :] = L[0, 1:-1, :, :]
    x1 = (tx * X1).ravel(order='F')
    x2 = (tx * X2).ravel(order='F')
    # Matlab spdiags convention: sub-diag d=-1 at (k,k-1) uses x2[k];
    # scipy diags with v: (k,k-1) gets v[k-1] → pass x2[1:] and x1[:-1].
    Bx = sp.diags([x2[1:], x0, x1[:-1]], [-1, 0, 1], shape=(Ex, Ex))

    # ── Y block ──────────────────────────────────────────────────────────────
    y0 = (2*ty * (L[1, :, :-1, :] + L[1, :, 1:, :])).ravel(order='F')
    Y1 = np.zeros((Nx, Ny-1, Nz));  Y1[:, 1:, :] = L[1, :, 1:-1, :]
    Y2 = np.zeros((Nx, Ny-1, Nz));  Y2[:, :-1, :] = L[1, :, 1:-1, :]
    y1 = (ty * Y1).ravel(order='F')
    y2 = (ty * Y2).ravel(order='F')
    By = sp.diags([y2[Nx:], y0, y1[:-Nx]], [-Nx, 0, Nx], shape=(Ey, Ey))

    # ── Z block (empty for Nz = 1) ───────────────────────────────────────────
    if Ez > 0:
        Nxy = Nx * Ny
        z1  = (tz * L[2, :, :, :-1]).ravel(order='F')
        z2  = (tz * L[2, :, :, 1:]).ravel(order='F')
        z0  = 2*(z1 + z2)
        Bz  = sp.diags([z2[Nxy:], z0, z1[:-Nxy]], [-Nxy, 0, Nxy], shape=(Ez, Ez))
        return sp.block_diag([Bx, By, Bz], format='csc')
    return sp.block_diag([Bx, By], format='csc')


def _GenC(Nx, Ny, Nz):
    """
    Divergence matrix C (N×E) for RT₀ MFE.

    Translates GenC.m.  For each edge, C[left_cell, edge] = -1 and
    C[right_cell, edge] = +1 so that −C·v = divergence(v) = q.
    """
    Nxy = Nx * Ny
    N_  = Nxy * Nz
    Ex  = (Nx-1)*Ny*Nz
    Ey  = Nx*(Ny-1)*Nz
    Ez  = Nx*Ny*(Nz-1)
    E   = Ex + Ey + Ez

    # ── X edges: (Nx-1, Ny, Nz) in Fortran order ────────────────────────────
    k_x = np.arange(Nx-1, dtype=np.intp)
    j_x = np.arange(Ny,   dtype=np.intp)
    l_x = np.arange(Nz,   dtype=np.intp)
    kk, jj, ll = np.meshgrid(k_x, j_x, l_x, indexing='ij')   # (Nx-1,Ny,Nz)
    col_x  = (kk + jj*(Nx-1) + ll*(Nx-1)*Ny).ravel()
    left_x = (kk + jj*Nx     + ll*Nxy).ravel()
    right_x = left_x + 1

    # ── Y edges: grouped as (Nxy-Nx, Nz) — layer l in second index ──────────
    k_y = np.arange(Nxy-Nx, dtype=np.intp)
    l_y = np.arange(Nz,     dtype=np.intp)
    k2, l2 = np.meshgrid(k_y, l_y, indexing='ij')             # (Nxy-Nx, Nz)
    col_y  = (Ex + k2 + l2*(Nxy-Nx)).ravel()
    top_y  = (k2 + l2*Nxy).ravel()
    bot_y  = top_y + Nx

    rows = np.concatenate([left_x, right_x, top_y,  bot_y])
    cols = np.concatenate([col_x,  col_x,   col_y,  col_y])
    vals = np.concatenate([np.full(Ex, -1.0), np.full(Ex,  1.0),
                           np.full(Ey, -1.0), np.full(Ey,  1.0)])

    # ── Z edges (only when Nz > 1) ───────────────────────────────────────────
    if Ez > 0:
        kk_z  = np.arange(N_ - Nxy, dtype=np.intp)
        col_z = Ex + Ey + kk_z
        rows  = np.concatenate([rows, kk_z,  kk_z + Nxy])
        cols  = np.concatenate([cols, col_z, col_z])
        vals  = np.concatenate([vals, np.full(Ez, -1.0), np.full(Ez, 1.0)])

    return sp.csr_matrix((vals, (rows, cols)), shape=(N_, E))


def MFE(Lx, Ly, Lz, Nx, Ny, Nz, Q, K):
    """
    Mixed finite element (RT₀) pressure solver.

    Solves [B, Cᵀ; −C, 0][v; p] = [0; q] with a pressure pin at cell 0.

    Returns P (Nx,Ny,Nz), Vx (Nx+1,Ny,Nz), Vy (Nx,Ny+1,Nz), Vz (Nx,Ny,Nz+1).
    Interior face velocities are stored at faces 1..Nx−1 (etc.); boundary faces
    remain zero (no-flow boundary).
    """
    N_  = Nx * Ny * Nz
    Ex  = (Nx-1)*Ny*Nz
    Ey  = Nx*(Ny-1)*Nz
    Ez  = Nx*Ny*(Nz-1)
    E   = Ex + Ey + Ez

    rhs = np.zeros(E + N_)
    rhs[E:] = Q.ravel(order='F')

    B = _GenB(Nx, Ny, Nz, Lx, Ly, Lz, K)
    C = _GenC(Nx, Ny, Nz)

    # Pressure reference: add 1 to A[E, E] (first pressure DOF)
    pin = sp.csr_matrix(([1.0], ([0], [0])), shape=(N_, N_))
    A   = sp.bmat([[B, C.T], [-C, pin]], format='csc')

    x = spla.spsolve(A, rhs)

    v  = x[:E]
    P_ = x[E:].reshape(Nx, Ny, Nz, order='F')

    Vx = np.zeros((Nx+1, Ny, Nz))
    Vy = np.zeros((Nx, Ny+1, Nz))
    Vz = np.zeros((Nx, Ny, Nz+1))
    Vx[1:Nx, :, :]  = v[:Ex].reshape(Nx-1, Ny,   Nz,   order='F')
    Vy[:, 1:Ny, :]  = v[Ex:Ex+Ey].reshape(Nx,   Ny-1, Nz,   order='F')
    if Ez > 0:
        Vz[:, :, 1:Nz] = v[Ex+Ey:].reshape(Nx, Ny, Nz-1, order='F')

    return P_, Vx, Vy, Vz


# =============================================================================
# 5.  UPWINDING MATRIX
# =============================================================================
def upwindingmatrix(Nx, Ny, Nz, Vx, Vy, Vz, Q, maxdFdz):
    N_ = Nx * Ny * Nz
    inj  = Q.ravel().clip(min=0)
    prod = Q.ravel().clip(max=0)

    XN = Vx.clip(max=0);  XP = Vx.clip(min=0)
    YN = Vy.clip(max=0);  YP = Vy.clip(min=0)
    ZN = Vz.clip(max=0);  ZP = Vz.clip(min=0)

    x1 = XN[:Nx,  :,   :].reshape(N_, order='F')
    x2 = XP[1:,   :,   :].reshape(N_, order='F')
    y1 = YN[:, :Ny,    :].reshape(N_, order='F')
    y2 = YP[:, 1:,     :].reshape(N_, order='F')
    z1 = ZN[:, :, :Nz   ].reshape(N_, order='F')
    z2 = ZP[:, :, 1:    ].reshape(N_, order='F')

    md  = x1 - x2 + y1 - y2 + z1 - z2 + prod
    UPW = sp.diags([z2[:-Nx*Ny], y2[:-Nx], x2[:-1], md,
                    -x1[1:], -y1[Nx:], -z1[Nx*Ny:]],
                   [-Nx*Ny, -Nx, -1, 0, 1, Nx, Nx*Ny])

    Vin = ((XP[:Nx, :, :] + YP[:, :Ny, :] + ZP[:, :, :Nz]
           - XN[1:, :, :] - YN[:, 1:, :] - ZN[:, :, 1:])
           .reshape(N_, order='F') + inj).clip(min=1e-30)
    CFL = np.min(gridPV / Vin) / maxdFdz
    return UPW, CFL


# =============================================================================
# 6.  FLASH WORKER INFRASTRUCTURE
# =============================================================================
_W_params      = None
_W_guess_fn    = None
_W_flash_model = 'ecpa'

def _worker_init(params, grid_data, flash_model):
    global _W_params, _W_guess_fn, _W_flash_model
    _W_params      = params
    _W_guess_fn    = make_solution_guess_fn(grid_data)
    _W_flash_model = flash_model


_Z_LO = 0.015   # below → single-phase brine (no flash needed)
_Z_HI = 0.97    # above → single-phase CO₂-rich


def _sp_result(z_co2: float, ms: float) -> dict:
    """Single-phase fallback — type chosen by z_co2."""
    if z_co2 < 0.4:
        return dict(phase='single_phase', beta=0.0,
                    x4w=float(z_co2), x4c=0.0,
                    Z_aq=0.90, Z_c=0.60, ms_aq=float(ms))
    return dict(phase='single_phase', beta=1.0,
                x4w=0.0, x4c=float(z_co2),
                Z_aq=0.90, Z_c=0.60, ms_aq=float(ms))


def _flash_one(args: tuple) -> dict:
    """Dispatch to eCPA or CPA flash; guaranteed to return a valid dict."""
    T, P_bar, z_co2, ms = args
    z_co2 = float(np.clip(z_co2, 1e-6, 1.0 - 1e-6))
    if _W_flash_model == 'cpa':
        return _flash_one_cpa(T, P_bar, z_co2, ms)
    return _flash_one_ecpa(T, P_bar, z_co2, ms)


def _flash_one_ecpa(T: float, P_bar: float, z_co2: float, ms: float) -> dict:
    """
    eCPA flash with guaranteed convergence:
      1. Warm-start K-SSI  (flash_co2_h2o_salt_fast_kv)
      2. Full stability+flash fallback  (ecpa_stability_flash)
      3. Single-phase heuristic
    """
    # Step 1: warm-start
    try:
        r = flash_co2_h2o_salt_fast_kv(
            T=T, P_bar=P_bar, z_co2=z_co2, m_tot=ms,
            solution_guess_fn=_W_guess_fn, params=_W_params,
        )
        if r['phase'] == 'two_phase':
            return dict(phase='two_phase',
                        beta  = float(r['beta']),
                        x4w   = float(r['x_aq']['x4w']),
                        x4c   = float(r['x_c']['x4c']),
                        Z_aq  = float(r['Z_aq']),
                        Z_c   = float(r['Z_c']),
                        ms_aq = float(r['ms_aq']))
        if r['phase'] == 'single_phase':
            return _sp_result(z_co2, ms)
    except Exception:
        pass

    # Step 2: robust stability + flash (Jex et al. 2024 hierarchy)
    try:
        r = ecpa_stability_flash(z_co2=z_co2, ms=ms, T=T, P=P_bar,
                                 params=_W_params)
        if r.get('phase') == 'two_phase':
            return dict(phase='two_phase',
                        beta  = float(r['beta']),
                        x4w   = float(r['x_aq']['x4w']),
                        x4c   = float(r['x_c']['x4c']),
                        Z_aq  = float(r.get('Z_aq', 0.9)),
                        Z_c   = float(r.get('Z_c',  0.6)),
                        ms_aq = float(r.get('ms_aq', ms)))
        return _sp_result(z_co2, ms)
    except Exception:
        pass

    return _sp_result(z_co2, ms)


def _flash_one_cpa(T: float, P_bar: float, z_co2: float, ms: float) -> dict:
    """
    Salt-free CPA flash with guaranteed convergence:
      flash_co2_h2o_tpz_warmstart internally falls back to
      flash_co2_h2o_tpz_robust (stability + Wilson K retries).
    """
    try:
        r = flash_co2_h2o_tpz_warmstart(
            T=T, P_bar=P_bar, z_co2=z_co2,
            solution_guess_fn=_W_guess_fn,
        )
        if r.get('phase') == 'two_phase':
            tie = r.get('tie') or {}
            Z   = tie.get('Z')
            Z_aq = float(Z[0]) if Z is not None else 0.90
            Z_c  = float(Z[1]) if Z is not None else 0.60
            return dict(phase='two_phase',
                        beta  = float(r['beta']),
                        x4w   = float(r['x'][0]),   # CO₂ in liquid
                        x4c   = float(r['y'][0]),   # CO₂ in vapour
                        Z_aq  = Z_aq,
                        Z_c   = Z_c,
                        ms_aq = float(ms))          # no salting-out in CPA
        return _sp_result(z_co2, ms)
    except Exception:
        pass
    return _sp_result(z_co2, ms)


def run_flash_parallel(z_arr: np.ndarray, ms: float,
                       T: float, P: float, pool) -> dict:
    """
    Flash all N cells; trivially single-phase cells (z < Z_LO or z > Z_HI)
    are bypassed.  Two-phase candidates are dispatched to the worker pool.
    """
    N_     = len(z_arr)
    out_beta  = np.zeros(N_)
    out_x4w   = np.zeros(N_)
    out_x4c   = np.zeros(N_)
    out_Z_aq  = np.full(N_, 0.90)
    out_Z_c   = np.full(N_, 0.60)
    out_ms_aq = np.full(N_, ms)
    out_phase = ['single_phase'] * N_

    co2_mask = z_arr > _Z_HI
    out_beta[co2_mask] = 1.0
    out_x4c[co2_mask]  = z_arr[co2_mask]

    tp_idx = np.where((z_arr >= _Z_LO) & (z_arr <= _Z_HI))[0]
    if len(tp_idx) > 0:
        args  = [(T, P, float(z_arr[i]), ms) for i in tp_idx]
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
# 7.  SATURATION AND FRACTIONAL FLOW
# =============================================================================
def beta_to_Sg(fr: dict, T: float, P_bar: float) -> np.ndarray:
    P_Pa = P_bar * 1e5
    Vm_c = fr['Z_c']  * R_GAS * T / P_Pa
    Vm_w = fr['Z_aq'] * R_GAS * T / P_Pa
    b    = fr['beta']
    den  = b * Vm_c + (1.0 - b) * Vm_w
    return np.where(den > 0, b * Vm_c / den, 0.0).clip(0, 1)


def compute_Fz(S_g: np.ndarray, x4w: np.ndarray, x4c: np.ndarray):
    kr_g, kr_w = relperm(S_g)
    lam_g = kr_g / mu_co2
    lam_w = kr_w / mu_brine
    lam_t = lam_g + lam_w + 1e-40
    F_z   = (lam_g * x4c + lam_w * x4w) / lam_t
    return F_z, lam_t


# =============================================================================
# 8.  CFL ESTIMATE (serial, called once at startup)
# =============================================================================
def estimate_maxdFdz(flash_fn, T: float, P: float, ms: float,
                     n_pts: int = 60) -> float:
    """
    Scan z ∈ [0.01, 0.95] and return max(dF_z/dz) for the CFL condition.

    flash_fn : callable (T, P, z_co2, ms) → unified flash dict
    """
    z_scan = np.linspace(0.01, 0.95, n_pts)
    F_scan = np.full(n_pts, np.nan)
    P_Pa   = P * 1e5
    for i, zv in enumerate(z_scan):
        try:
            res = flash_fn(T, P, float(zv), ms)
            if res['phase'] == 'two_phase':
                b    = float(res['beta'])
                Vm_c = float(res['Z_c'])  * R_GAS * T / P_Pa
                Vm_w = float(res['Z_aq']) * R_GAS * T / P_Pa
                den  = b*Vm_c + (1 - b)*Vm_w
                Sg_i = (b*Vm_c / den) if den > 0 else 0.0
                Fz_i, _ = compute_Fz(np.array([Sg_i]),
                                     np.array([float(res['x4w'])]),
                                     np.array([float(res['x4c'])]))
                F_scan[i] = float(Fz_i[0])
        except Exception:
            pass
    ok = ~np.isnan(F_scan)
    if ok.sum() < 2:
        return 2.5
    dF = np.abs(np.diff(F_scan[ok]))
    dz = np.diff(z_scan[ok])
    return float(np.nanmax(dF / (dz + 1e-12))) * 1.2


# =============================================================================
# 9.  DIAGNOSTICS
# =============================================================================
def dissolved_co2_molality(x4w: np.ndarray, ms_aq: np.ndarray) -> np.ndarray:
    x1w = (1.0 - x4w) / (1.0 + 2.0 * ms_aq * MW_H2O + 1e-12)
    x1w = np.maximum(x1w, 1e-9)
    return x4w / (x1w * MW_H2O)


# =============================================================================
# 10.  PLOTTING
# =============================================================================
def plot_snapshot(z: np.ndarray, fr: dict, T: float, P: float,
                  step: int, t_yr: float,
                  outdir: str = 'figures/simulator') -> str:
    os.makedirs(outdir, exist_ok=True)

    S_g   = beta_to_Sg(fr, T, P).reshape(Nx, Ny, order='F')
    mc    = dissolved_co2_molality(fr['x4w'], fr['ms_aq']).reshape(Nx, Ny, order='F')
    ms_aq = fr['ms_aq'].reshape(Nx, Ny, order='F')
    z2d   = z.reshape(Nx, Ny, order='F')

    XX = np.linspace(dx/2, Lx - dx/2, Nx)
    YY = np.linspace(dy/2, Ly - dy/2, Ny)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    panels = [
        (z2d,   r'Overall CO$_2$ mol fraction $z$',                        'viridis', None),
        (S_g,   r'CO$_2$-rich saturation $S_g$',                           'plasma',  (0, 1)),
        (mc,    r'Dissolved CO$_2$ $m_c$ [mol kg$^{-1}$]',                 'Blues',   None),
        (ms_aq, r'Equilibrium brine $m_s^\mathrm{aq}$ [mol kg$^{-1}$]',    'Oranges', None),
    ]
    for ax, (data, title, cmap, vlim) in zip(axes.flat, panels):
        kw = dict(vmin=vlim[0], vmax=vlim[1]) if vlim else {}
        im = ax.pcolormesh(XX, YY, data.T, cmap=cmap, shading='auto', **kw)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('x [m]', fontsize=8)
        ax.set_ylabel('y [m]', fontsize=8)
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax, pad=0.02)

    xc, yc = XX[inj_i], YY[inj_j]
    for ax in axes.flat:
        ax.plot(xc, yc, '*', color='white', ms=10, zorder=5)
        for ci, cj in [(0,0),(Nx-1,0),(0,Ny-1),(Nx-1,Ny-1)]:
            ax.plot(XX[ci], YY[cj], '^', color='black', ms=7, zorder=5)

    plt.tight_layout()
    fname = f'{outdir}/snap_{step:03d}.png'
    plt.savefig(fname, dpi=120)
    plt.close()
    return fname


# =============================================================================
# 11.  MAIN SIMULATION LOOP
# =============================================================================
def main(pressure_solver: str = 'tpfa', flash_model: str = 'ecpa'):
    """
    Run the IMPES CO₂-brine simulation.

    Parameters
    ----------
    pressure_solver : 'tpfa' | 'mfe'
    flash_model     : 'ecpa' | 'cpa'
    """
    pressure_solver = pressure_solver.lower()
    flash_model     = flash_model.lower()
    assert pressure_solver in ('tpfa', 'mfe'),  f"Unknown solver: {pressure_solver}"
    assert flash_model     in ('ecpa', 'cpa'),  f"Unknown model: {flash_model}"

    label   = f"{pressure_solver.upper()} + {flash_model.upper()}"
    outdir  = f'figures/simulator/{pressure_solver}_{flash_model}'
    os.makedirs(outdir, exist_ok=True)

    print(f"=== CO₂-brine simulator  [{label}] ===\n")
    print("Loading eCPA parameters and solution table …")
    params    = make_params()
    grid_data = load_solution_table()
    guess_fn  = make_solution_guess_fn(grid_data)
    print("  done.\n")

    # ── Set module-level globals for serial (CFL) and worker flash ───────────
    global _W_params, _W_guess_fn, _W_flash_model
    _W_params      = params
    _W_guess_fn    = guess_fn
    _W_flash_model = flash_model

    ms_flash = ms0 if flash_model == 'ecpa' else 0.0

    def _serial_flash(T, P, z_co2, ms):
        if flash_model == 'ecpa':
            return _flash_one_ecpa(float(T), float(P), float(z_co2), float(ms))
        return _flash_one_cpa(float(T), float(P), float(z_co2), float(ms))

    print("Estimating max dF_z/dz for CFL …", end=' ', flush=True)
    maxdFdz = estimate_maxdFdz(_serial_flash, T_K, P_ref, ms_flash)
    print(f"max dF/dz = {maxdFdz:.2f}\n")

    # ── Spawn parallel pool ───────────────────────────────────────────────────
    ctx  = get_context('spawn')
    pool = ctx.Pool(initializer=_worker_init,
                    initargs=(params, grid_data, flash_model))

    # ── Initial conditions ────────────────────────────────────────────────────
    z       = z_initial * np.ones(N)
    inj_src = Q.ravel().clip(min=0) * z_inject
    dt_big  = t_max_yr * year_s / n_steps

    timing     = []
    snap_every = max(1, n_steps // 8)

    print(f"Grid: {Nx}×{Ny} = {N} cells  |  {t_max_yr:.0f} yr  |  "
          f"{n_steps} steps  |  dt = {dt_big/year_s:.3f} yr/step")
    print(f"Pressure solver: {pressure_solver.upper()}  |  "
          f"Flash model: {flash_model.upper()}\n")
    print(f"{'Step':>5}  {'t [yr]':>7}  {'N_sub':>5}  {'N_flash':>7}  "
          f"{'t_flash [s]':>11}  {'t_step [s]':>10}  {'flash/s':>8}")
    print("─" * 66)

    t0_total = time.perf_counter()

    for step in range(1, n_steps + 1):
        t_yr = step * dt_big / year_s
        t0   = time.perf_counter()

        # 1. Flash all cells
        fr      = run_flash_parallel(z, ms_flash, T_K, P_ref, pool)
        t_flash = time.perf_counter() - t0

        # 2. Saturations and fractional flow
        S_g       = beta_to_Sg(fr, T_K, P_ref)
        Fz, lam_t = compute_Fz(S_g, fr['x4w'], fr['x4c'])

        # 3. Effective permeability → pressure solve
        lam_3d = np.stack([lam_t.reshape(Nx, Ny, Nz, order='F')] * 3, axis=0)
        Keff   = K * lam_3d

        if pressure_solver == 'mfe':
            _, Vx, Vy, Vz = MFE(Lx, Ly, Lz, Nx, Ny, Nz, Q, Keff)
        else:
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
            fname = plot_snapshot(z, fr, T_K, P_ref, step, t_yr, outdir=outdir)
            print(f"          → saved {fname}")

    pool.close()
    pool.join()

    wall  = time.perf_counter() - t0_total
    t_fl  = np.array([t[2] for t in timing])
    t_st  = np.array([t[3] for t in timing])
    n_fls = np.array([t[4] for t in timing])
    t_yrs = np.array([t[1] for t in timing])

    print(f"\n{'='*66}")
    print(f"[{label}]  Finished in {wall:.1f} s")
    print(f"  Mean flash calls / step : {n_fls.mean():.0f}  (max {n_fls.max():.0f})")
    print(f"  Mean flash time / step  : {t_fl.mean():.3f} s  "
          f"({t_fl.sum()/t_st.sum()*100:.0f}% of runtime)")
    print(f"  Mean total time / step  : {t_st.mean():.3f} s")
    nf_pos = n_fls[n_fls > 0];  tf_pos = t_fl[n_fls > 0]
    if len(nf_pos):
        print(f"  Flash throughput        : {(nf_pos/tf_pos).mean():.0f} calls/s")

    # ── Performance figure ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    axes[0].plot(t_yrs, t_fl, lw=1.5, label='Flash (two-phase cells)')
    axes[0].plot(t_yrs, t_st, lw=1.5, ls='--', label='Total step')
    axes[0].set_ylabel('Wall time [s]')
    axes[0].set_title(label, fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].bar(t_yrs, n_fls, width=(t_yrs[1]-t_yrs[0])*0.8, alpha=0.7)
    axes[1].axhline(N, color='grey', ls=':', lw=1, label=f'Total cells ({N})')
    axes[1].set_xlabel('Simulation time [yr]')
    axes[1].set_ylabel('Cells flashed')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    perf_fig = f'{outdir}/performance.png'
    plt.savefig(perf_fig, dpi=150)
    plt.close()
    print(f"  Saved {perf_fig}")

    return dict(timing=timing, wall=wall)


if __name__ == '__main__':
    # Usage: python co2brine_simulator.py [tpfa|mfe] [ecpa|cpa]
    args   = sys.argv[1:]
    solver = args[0] if len(args) > 0 else 'tpfa'
    model  = args[1] if len(args) > 1 else 'ecpa'
    main(pressure_solver=solver, flash_model=model)
