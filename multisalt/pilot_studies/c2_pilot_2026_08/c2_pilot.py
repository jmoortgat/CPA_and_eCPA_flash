"""C2H6 (ethane) pilot: extend the multi-salt eCPA to ethane at the cost of
one salt-free binary fit, N2-pilot style.

Data (both CSVs in this directory; see headers for provenance):
  c2_water_data.csv       binary C2H6-H2O aqueous mole fractions --
                          Culberson-McKetta 1950 low-P + high-P sheets (via
                          the IUPAC SDS Vol 9 compilation, est. err +/-5%)
                          and Mohammadi et al. 2004 Table 4 (T >= 283 K).
                          116 points, 283.1-444.26 K, 0.37-68.5 MPa.
  c2_setschenow_data.csv  IUPAC SDS Vol 9 (Clever) evaluated Setschenow
                          constants k_sca(T) for C2H6 in NaCl (five labs +
                          smoothed recommendation, 273-348 K) and in KCl
                          (Ben-Naim & Yaacobi, 283-303 K). Molar
                          (concentration) basis, dm3/mol, log10.

Framings (mirroring n2_pilot):
  T   pure transfer:  kij(T) fitted to the salt-free binary (fitB); ALL six
                      gas-ion energies transferred from CH4 with the
                      function-preserving Tc rescale
                      A_C2,i = A_CH4,i * Tc(C2H6)/Tc(CH4) = 1.60222 * A_CH4,i
                      (B unchanged), so DU(T) is numerically identical to
                      CH4's. Setschenow prediction expected to UNDERSHOOT
                      (ethane salts out more strongly than methane).
  S   one-parameter:  a single factor s scaling ALL transferred gas-ion
                      (A, B) pairs jointly, fitted to the IUPAC smoothed
                      NaCl k_sca at five temperatures. KCl then remains a
                      same-s prediction (an ion-specific test), unlike the
                      n2_pilot where s touched (Na, Cl) only.

Setschenow computation (stage ks0/fitS/evalS): at each T the model is
flashed at P_TOTAL = 20 bar in pure water and in m = 1.0 mol/kg brine; the
Henry-normalized volume-basis solubility S_hat = c_gas / f_C2H6 (c_gas from
the aqueous molality and the MODEL brine density; f from the flash gas
fugacity) gives
    k_sca = log10(S_hat_w / S_hat_b) / c_salt,
with c_salt = m * rho_soln / (1 + m*M_salt) from the model density as well.
Henry normalization cancels the (tiny) partial-pressure difference between
the water and brine runs; the dilute limit makes the choice P_TOTAL = 20 bar
immaterial (2 bar would be closer to the experiments, but below ~15 bar
the pure-water flash can collapse to the dew-line rescue branch; see
KS_P_TOTAL_BAR note).

Run stages (state accumulates in work_state.json):
    python c2_pilot.py --stage refs      # unpatched CH4/CO2 guard flashes
    python c2_pilot.py --stage fitB     # fit kij to 113 salt-free points
    python c2_pilot.py --stage evalB    # per-source AAREs
    python c2_pilot.py --stage ks0      # transfer (s=1) Setschenow NaCl+KCl
    python c2_pilot.py --stage fitS     # fit s to IUPAC smoothed NaCl ks
    python c2_pilot.py --stage evalS    # fitted-s Setschenow + sensitivity
    python c2_pilot.py --stage curves   # binary isotherms + ks(T) curves
    python c2_pilot.py --stage figure   # 2-panel house-style figure
    python c2_pilot.py --stage finalize # guards + c2_results.json

C2H6 constants: Tc = 305.32 K, Pc = 48.72 bar, omega = 0.0995,
M = 0.0300690 kg/mol. Polarizability alpha(C2H6) = 4.47 A^3 =
4.9736e-40 C^2 m^2 J^-1 (handbook; enters eps_inf only, sensitivity
quantified in stage evalS).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))

import ms_ecpa.parameters as prm            # noqa: E402
import ms_ecpa.lnphi as lnphi_mod           # noqa: E402
import ms_ecpa.eos as eos_mod               # noqa: E402
import ms_ecpa.flash as flash_mod           # noqa: E402
from ms_ecpa import stability_flash         # noqa: E402

HERE = Path(__file__).resolve().parent
WATER_CSV = HERE / "c2_water_data.csv"
KS_CSV = HERE / "c2_setschenow_data.csv"
STATE_JSON = HERE / "work_state.json"
RESULTS_JSON = HERE / "c2_results.json"
CURVES_CSV = HERE / "curves_cache.csv"
FIG_PNG = HERE / "fig_c2_pilot.png"

# ---------------------------------------------------------------------------
# C2H6 constants
# ---------------------------------------------------------------------------
TC_C2 = 305.32           # K
PC_C2 = 48.72e5          # Pa
W_C2 = 0.0995
M_C2 = 0.0300690         # kg/mol
POL_C2 = 4.9736e-40      # C^2 m^2 J^-1; handbook alpha = 4.47 A^3
POL_CH4 = 2.6155e-40     # production CH4 value (sensitivity test)

TC_RATIO = TC_C2 / 190.56        # 1.60222, function-preserving DU(T) transfer
KS_P_TOTAL_BAR = 20.0     # below ~15 bar the PURE-WATER flash can
                          # collapse to the dew-line rescue branch at
                          # T >~ 300 K (probe 2026-08-10); 20 bar is
                          # feed-independence-verified 283-348 K for
                          # water, NaCl, and KCl; still dilute
KS_M_SALT = 1.0                  # mol/kg for the Setschenow runs
KS_FIT_T = (283.15, 298.15, 313.15, 328.15, 348.15)
M_SALT_KG = {"NaCl": 0.0584428, "KCl": 0.0745513}
M_ION = (0.02299, 0.03545, 0.03910, 0.04008, 0.02431, 0.09606)
R_J = 8.31446261815324

Z_GAS_RETRIES = (0.05, 0.10, 0.02, 0.20, 0.30)
N_WORKERS = 8

# ---------------------------------------------------------------------------
# Rebind machinery (Tier-1 recipe, from n2_pilot)
# ---------------------------------------------------------------------------
NAMESPACES = (prm, lnphi_mod, eos_mod, flash_mod)
SCALARS = ("Tc2", "Pc2", "w2", "b2", "a02", "c12", "Mh", "pol2",
           "Ak12", "Bk12", "Ck12", "k23")

_ORIG = {name: getattr(prm, name) for name in SCALARS}
_ORIG_A_U2i = prm.A_U2i.copy()
_ORIG_B_U2i = prm.B_U2i.copy()
_ORIG_WILSON = (float(flash_mod._TC[1]), float(flash_mod._PC_BAR[1]),
                float(flash_mod._OMEGA[1]))


def rebind_c2(Ak, Bk, Ck, s_all=1.0, pol2=POL_C2):
    """Swap slot 2 CH4 -> C2H6 in all four consuming namespaces.

    kij form: k12 = Ak*(T/Tc1)^2 + Bk*(T/Tc1) + Ck (Tc1 = water). Gas-ion
    energies: function-preserving transfer A*TC_RATIO (B unchanged), then
    ALL SIX transferred (A, B) pairs scaled jointly by s_all (framing S;
    s_all = 1 -> pure transfer). Scaling all ions keeps KCl a same-s
    prediction. k23 (C2H6-CO2) inert here (z_CO2 = 0 throughout).
    """
    R = prm.R
    vals = dict(
        Tc2=TC_C2, Pc2=PC_C2, w2=W_C2,
        b2=0.08664 * R * TC_C2 / PC_C2,
        a02=0.42748 * R ** 2 * TC_C2 ** 2 / PC_C2,
        c12=0.480 + 1.574 * W_C2 - 0.176 * W_C2 ** 2,
        Mh=M_C2, pol2=pol2,
        Ak12=float(Ak), Bk12=float(Bk), Ck12=float(Ck), k23=0.0,
    )
    for ns in NAMESPACES:
        for name, v in vals.items():
            if hasattr(ns, name):
                setattr(ns, name, v)
    prm.A_U2i[:] = _ORIG_A_U2i * TC_RATIO * s_all
    prm.B_U2i[:] = _ORIG_B_U2i * s_all
    flash_mod._TC[1] = TC_C2
    flash_mod._PC_BAR[1] = PC_C2 / 1e5
    flash_mod._OMEGA[1] = W_C2


def restore_ch4():
    for ns in NAMESPACES:
        for name, v in _ORIG.items():
            if hasattr(ns, name):
                setattr(ns, name, v)
    prm.A_U2i[:] = _ORIG_A_U2i
    prm.B_U2i[:] = _ORIG_B_U2i
    flash_mod._TC[1], flash_mod._PC_BAR[1], flash_mod._OMEGA[1] = _ORIG_WILSON


# ---------------------------------------------------------------------------
# Prediction machinery
# ---------------------------------------------------------------------------
def _predict_x2(task):
    """(Ak, Bk, Ck, s_all, pol2, T_K, P_bar, mi6) -> aqueous C2H6 mole
    fraction x2w (dissociated ion-inclusive basis), or NaN."""
    Ak, Bk, Ck, s_all, pol2, T, P_bar, mi6 = task
    try:
        rebind_c2(Ak, Bk, Ck, s_all=s_all, pol2=pol2)
        mi = np.asarray(mi6, dtype=float)
        for zg in Z_GAS_RETRIES:
            z = np.array([1.0 - zg, zg, 0.0])
            try:
                res = stability_flash(float(T), float(P_bar), z, mi)
            except Exception:
                continue
            if res.phases == 2 and res.flash_result is not None:
                return float(res.flash_result.x_aq[1])
        return float("nan")
    finally:
        restore_ch4()


def _solub_point(task):
    """(Ak, Bk, Ck, s_all, pol2, T_K, P_bar, mi6) -> (m_gas [mol/kg-w],
    f_C2H6_gas [bar], rho_aq [kg/m3]) or (nan, nan, nan).

    rho_aq follows the sweep_core density back-out, with the slot-2 molar
    mass set to ethane's."""
    Ak, Bk, Ck, s_all, pol2, T, P_bar, mi6 = task
    try:
        rebind_c2(Ak, Bk, Ck, s_all=s_all, pol2=pol2)
        mi = np.asarray(mi6, dtype=float)
        for zg in Z_GAS_RETRIES:
            z = np.array([1.0 - zg, zg, 0.0])
            try:
                res = stability_flash(float(T), float(P_bar), z, mi)
            except Exception:
                continue
            if res.phases == 2 and res.flash_result is not None:
                fr = res.flash_result
                x1w, x2w, x3w = fr.x_aq
                m_gas = x2w / (prm.Mw * x1w)
                f2 = float(fr.fugacities_gas[1])
                P_Pa = float(P_bar) * 1e5
                V = float(fr.Z_aq) * R_J * float(T) / P_Pa
                M_mix = x1w * prm.Mw + x2w * M_C2 + x3w * 0.04401
                for k in range(6):
                    M_mix += prm.Mw * x1w * float(mi[k]) * M_ION[k]
                rho = M_mix / V
                return (float(m_gas), f2, float(rho))
        return (float("nan"),) * 3
    finally:
        restore_ch4()


MI_WATER = (0.0,) * 6


def mi_for(salt, m):
    if salt == "NaCl":
        return (m, m, 0.0, 0.0, 0.0, 0.0)
    if salt == "KCl":
        return (0.0, m, m, 0.0, 0.0, 0.0)
    raise ValueError(salt)


def _shat(sol, m, salt):
    """Henry-normalized volume-basis solubility from a _solub_point triple."""
    m_gas, f2, rho = sol
    kgw_per_L = (rho / 1000.0) / (1.0 + m * M_SALT_KG[salt]) if m > 0 \
        else rho / 1000.0
    return (m_gas * kgw_per_L) / f2


def ks_model(kij, s_all, T, salt, m=KS_M_SALT, pol2=POL_C2,
             water_cache=None):
    """Model Setschenow constant k_sca (dm3/mol, log10) at T."""
    if water_cache is not None and T in water_cache:
        sol_w = water_cache[T]
    else:
        sol_w = _solub_point((*kij, 1.0, pol2, T, KS_P_TOTAL_BAR, MI_WATER))
        if water_cache is not None:
            water_cache[T] = sol_w
    sol_b = _solub_point((*kij, s_all, pol2, T, KS_P_TOTAL_BAR,
                          mi_for(salt, m)))
    if not (np.isfinite(sol_w[0]) and np.isfinite(sol_b[0])):
        return float("nan")
    Shat_w = _shat(sol_w, 0.0, salt)
    Shat_b = _shat(sol_b, m, salt)
    c_salt = m * (sol_b[2] / 1000.0) / (1.0 + m * M_SALT_KG[salt])
    return float(np.log10(Shat_w / Shat_b) / c_salt)


_POOL = None


def predict_water_all(df, Ak, Bk, Ck):
    tasks = [(Ak, Bk, Ck, 1.0, POL_C2, r.T_K, r.P_MPa * 10.0, MI_WATER)
             for r in df.itertuples()]
    if _POOL is not None:
        out = _POOL.map(_predict_x2, tasks)
    else:
        out = [_predict_x2(t) for t in tasks]
    return np.array(out)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
GUARD_MI = (1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
GUARD_CASES = {
    "ch4_nacl": (380.0, 200.0, (0.90, 0.10, 0.0)),
    "co2_nacl": (380.0, 200.0, (0.95, 0.0, 0.05)),
}


def guard_spot(case):
    T, P, z = GUARD_CASES[case]
    r = stability_flash(T, P, np.array(z), np.array(GUARD_MI))
    fr = r.flash_result
    return [fr.beta, *fr.x_aq, *fr.y_gas, *fr.K, fr.eps_r, fr.Z_aq, fr.Z_gas]


def tuples_bit_identical(a, b):
    if len(a) != len(b):
        return False
    return all((x == y) or (np.isnan(x) and np.isnan(y))
               for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# (T_K, P_MPa) pairs where the PURE-WATER flash degenerates into the
# dew-line rescue branch (feed-dependent x2; probe 2026-08-10). Excluded
# from the fit and the AARE reporting; a library-level low-P robustness
# fix is future work.
DEGENERATE_WATER_PTS = {(310.93, 0.41), (303.19, 0.373), (313.17, 0.439),
                        (323.17, 0.397), (343.08, 0.44)}


def load_water():
    import pandas as pd
    df = pd.read_csv(WATER_CSV, comment="#")
    assert len(df) == 116, f"expected 116 rows, got {len(df)}"
    mask = [(round(r.T_K, 2), round(r.P_MPa, 3)) not in DEGENERATE_WATER_PTS
            for r in df.itertuples()]
    df = df[mask].reset_index(drop=True)
    assert len(df) == 111, f"expected 111 usable rows, got {len(df)}"
    return df


def load_ks():
    import pandas as pd
    df = pd.read_csv(KS_CSV, comment="#")
    return df[df.T_K >= 283.15].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _state():
    return json.loads(STATE_JSON.read_text()) if STATE_JSON.exists() else {}


def _save_state(st):
    STATE_JSON.write_text(json.dumps(st, indent=2))


def fit_least_squares(residual_fn, starts, bounds, label):
    from scipy.optimize import least_squares
    runs, best = [], None
    for i, x0 in enumerate(starts):
        t0 = time.perf_counter()
        res = least_squares(residual_fn, x0, bounds=bounds,
                            diff_step=1e-3, method="trf")
        run = {"start": [float(v) for v in np.atleast_1d(x0)],
               "x": [float(v) for v in np.atleast_1d(res.x)],
               "cost": float(res.cost), "status": int(res.status),
               "message": res.message, "nfev": int(res.nfev),
               "optimality": float(res.optimality),
               "rms_rel_resid_pct":
                   float(np.sqrt(np.mean(res.fun ** 2)) * 100),
               "wall_s": round(time.perf_counter() - t0, 1)}
        runs.append(run)
        print(f"  [{label} start {i+1}] x0={run['start']} -> x={run['x']} "
              f"cost={run['cost']:.6g} nfev={run['nfev']} "
              f"({run['wall_s']}s)", flush=True)
        if best is None or run["cost"] < best["cost"]:
            best = run
    return best, runs


NAN_RESIDUAL = 5.0


def make_resid_water(df):
    exp = df["x_molefrac"].to_numpy()

    def resid(x):
        pred = predict_water_all(df, x[0], x[1], x[2])
        r = pred / exp - 1.0
        return np.where(np.isfinite(r), r, NAN_RESIDUAL)
    return resid


def make_resid_ks(kij, targets, water_cache):
    def resid(x):
        s = float(x[0])
        out = []
        for T, ks_t in targets:
            ks = ks_model(kij, s, T, "NaCl", water_cache=water_cache)
            out.append(ks / ks_t - 1.0 if np.isfinite(ks) else NAN_RESIDUAL)
        return np.array(out)
    return resid


def aare_water(df, pred):
    out = {}
    exp = df["x_molefrac"].to_numpy()
    for label, mask in [("CM1950lo", df.source == "CM1950lo"),
                        ("CM1950hi", df.source == "CM1950hi"),
                        ("Mohammadi2004", df.source == "Mohammadi2004"),
                        ("all", df.source.notna())]:
        m = np.asarray(mask)
        p, e = pred[m], exp[m]
        ok = np.isfinite(p)
        out[label] = {
            "aare_pct": float(np.mean(np.abs(p[ok] / e[ok] - 1.0)) * 100.0)
            if ok.any() else float("nan"),
            "bias_pct": float(np.mean(p[ok] / e[ok] - 1.0) * 100.0)
            if ok.any() else float("nan"),
            "n_converged": int(ok.sum()), "n_points": int(m.sum())}
    return out


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_refs(st):
    st["guard_refs"] = {c: guard_spot(c) for c in GUARD_CASES}
    for c, v in st["guard_refs"].items():
        print(f"  {c}: beta={v[0]:.12g}")


def stage_fitB(st):
    df = load_water()
    ch4 = [-4.0935, 5.6634, -1.7845]        # production CH4-H2O values
    perturbed = [ch4[0] * 1.25, ch4[1] * 0.85, ch4[2] + 0.08]
    bounds = ([-60.0, -60.0, -60.0], [60.0, 60.0, 60.0])
    best, runs = fit_least_squares(make_resid_water(df),
                                   [ch4, perturbed], bounds, "fitB")
    st["fitB"] = {"best": best, "runs": runs, "n_points": len(df),
                  "bounds": bounds, "diff_step": 1e-3}


def stage_evalB(st):
    df = load_water()
    kij = st["fitB"]["best"]["x"]
    pred = predict_water_all(df, *kij)
    st["B"] = {"kij": kij, "aare": aare_water(df, pred),
               "pred": [float(p) for p in pred]}
    print(json.dumps(st["B"]["aare"], indent=2))


def _ks_grid(kij, s_all, salt, Ts, water_cache):
    out = []
    for T in Ts:
        ks = ks_model(kij, s_all, float(T), salt, water_cache=water_cache)
        out.append({"T_K": float(T), "ks": ks})
        print(f"  ks({salt}, T={T}, s={s_all:.4g}) = {ks:.4f}", flush=True)
    return out


def stage_ks0(st):
    kij = st["fitB"]["best"]["x"]
    wc = {}
    st["ks0"] = {
        "NaCl": _ks_grid(kij, 1.0, "NaCl", KS_FIT_T, wc),
        "KCl": _ks_grid(kij, 1.0, "KCl", (283.15, 293.15, 303.15), wc),
    }
    ksdf = load_ks()
    sm = ksdf[(ksdf.salt == "NaCl") & (ksdf.lab == "smoothed")]
    tgt = {float(r.T_K): float(r.ks_dm3_mol) for r in sm.itertuples()}
    devs = [d["ks"] / tgt[d["T_K"]] - 1.0 for d in st["ks0"]["NaCl"]
            if d["T_K"] in tgt and np.isfinite(d["ks"])]
    st["ks0"]["NaCl_mean_dev_pct"] = float(np.mean(devs) * 100.0)
    print(f"  transfer (s=1) NaCl mean deviation vs IUPAC smoothed: "
          f"{st['ks0']['NaCl_mean_dev_pct']:+.1f}%")


def stage_fitS(st):
    kij = st["fitB"]["best"]["x"]
    ksdf = load_ks()
    sm = ksdf[(ksdf.salt == "NaCl") & (ksdf.lab == "smoothed")]
    tgt = {float(r.T_K): float(r.ks_dm3_mol) for r in sm.itertuples()}
    targets = [(T, tgt[T]) for T in KS_FIT_T]
    wc = {}
    best, runs = fit_least_squares(make_resid_ks(kij, targets, wc),
                                   [[1.0], [1.4]], ([0.2], [4.0]), "fitS")
    st["fitS"] = {"best": best, "runs": runs,
                  "targets": targets,
                  "parameterization":
                      "single factor s scaling ALL SIX transferred gas-ion "
                      "(A, B) pairs jointly (KCl stays a same-s prediction); "
                      "s=1 = pure transfer"}


def stage_evalS(st):
    kij = st["fitB"]["best"]["x"]
    s = st["fitS"]["best"]["x"][0]
    wc = {}
    Ts_full = tuple(np.arange(283.15, 348.16, 5.0))
    st["S"] = {
        "s_all": s,
        "NaCl": _ks_grid(kij, s, "NaCl", Ts_full, wc),
        "KCl": _ks_grid(kij, s, "KCl", (283.15, 288.15, 293.15, 298.15,
                                        303.15), wc),
    }
    ksdf = load_ks()
    for salt, lab in (("NaCl", "smoothed"), ("KCl", "BenNaimYaacobi")):
        sub = ksdf[(ksdf.salt == salt) & (ksdf.lab == lab)]
        tgt = {float(r.T_K): float(r.ks_dm3_mol) for r in sub.itertuples()}
        devs = [d["ks"] / tgt[d["T_K"]] - 1.0 for d in st["S"][salt]
                if d["T_K"] in tgt and np.isfinite(d["ks"])]
        st["S"][f"{salt}_mean_dev_pct"] = float(np.mean(devs) * 100.0)
        st["S"][f"{salt}_aad_pct"] = float(
            np.mean(np.abs(devs)) * 100.0)
        print(f"  fitted-s {salt} vs {lab}: mean dev "
              f"{st['S'][f'{salt}_mean_dev_pct']:+.1f}%, AAD "
              f"{st['S'][f'{salt}_aad_pct']:.1f}%")
    # polarizability sensitivity on ks at one corner
    ks_a = ks_model(kij, s, 348.15, "NaCl")
    ks_b = ks_model(kij, s, 348.15, "NaCl", pol2=POL_CH4)
    st["pol2_sensitivity"] = {
        "T_K": 348.15, "ks_pol_c2": ks_a, "ks_pol_ch4": ks_b,
        "rel_shift_pct": (ks_b / ks_a - 1.0) * 100.0}
    print(f"  pol2 sensitivity on ks(348 K): "
          f"{st['pol2_sensitivity']['rel_shift_pct']:+.3f}%")


def stage_curves(st):
    import pandas as pd
    kij = st["fitB"]["best"]["x"]
    s = st["fitS"]["best"]["x"][0]
    rows = []
    P_bar = np.geomspace(4.0, 690.0, 14)
    for T in (310.93, 344.26, 377.59, 410.93, 444.26):
        tasks = [(kij[0], kij[1], kij[2], 1.0, POL_C2, T, p, MI_WATER)
                 for p in P_bar]
        preds = (_POOL.map(_predict_x2, tasks) if _POOL is not None
                 else [_predict_x2(t) for t in tasks])
        for p, x in zip(P_bar, preds):
            rows.append(dict(kind="binary", T_K=T, P_bar=p, val=x,
                             variant="B"))
        print(f"  binary curve T={T} done", flush=True)
    Ts = np.arange(283.15, 348.16, 5.0)
    wc = {}
    for salt, variants in (("NaCl", (("transfer", 1.0), ("fitted", s))),
                           ("KCl", (("transfer", 1.0), ("fitted", s)))):
        for name, sv in variants:
            for T in Ts:
                ks = ks_model(kij, sv, float(T), salt, water_cache=wc)
                rows.append(dict(kind="ks", T_K=float(T), P_bar=np.nan,
                                 val=ks, variant=f"{salt}_{name}"))
            print(f"  ks curve {salt} {name} done", flush=True)
    pd.DataFrame(rows).to_csv(CURVES_CSV, index=False)
    st["curves_csv"] = str(CURVES_CSV)


def stage_figure(st):
    import pandas as pd
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.size": 14, "axes.labelsize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 11,
        "font.weight": "bold", "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 1.6, "axes.edgecolor": "black",
        "xtick.major.width": 1.6, "ytick.major.width": 1.6,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.color": "black", "ytick.color": "black",
        "xtick.direction": "in", "ytick.direction": "in",
        "lines.linewidth": 2.2, "lines.markersize": 7,
        "legend.frameon": True, "legend.edgecolor": "black",
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73",
            "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]

    df = load_water()
    cur = pd.read_csv(CURVES_CSV)
    ksdf = load_ks()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.6))

    # (a) binary isotherms, CM data only (Mohammadi isotherms omitted for
    # clarity; its AARE is reported in the results table)
    Ts = [310.93, 344.26, 377.59, 410.93, 444.26]
    for i, T in enumerate(Ts):
        c = cur[(cur.kind == "binary") & np.isclose(cur.T_K, T)]
        ax1.plot(c.P_bar, c.val * 1e4, linestyle=LINESTYLES[i % 5],
                 color=WONG[i % 8], label=f"{T:.0f} K")
        sub = df[np.isclose(df.T_K, T)]
        ax1.plot(sub.P_MPa * 10.0, sub.x_molefrac * 1e4, "o",
                 color=WONG[i % 8], mfc="white", mew=1.8,
                 linestyle="none")
    ax1.set_xscale("log")
    ax1.set_xlabel("P [bar]")
    ax1.set_ylabel(r"$10^4\,x_{\mathrm{C_2H_6}}$")
    ax1.legend(loc="upper left", ncol=1)
    ax1.text(0.97, 0.03, "(a)", transform=ax1.transAxes, ha="right",
             fontsize=15, fontweight="bold")

    # (b) Setschenow constants
    LABMARK = {"EuckenHertzberg": ("s", "Eucken--Hertzberg 1950"),
               "MorrisonBillett": ("D", "Morrison--Billett 1952"),
               "Mishnina": ("o", "Mishnina 1961"),
               "BenNaimYaacobi": ("^", "Ben-Naim--Yaacobi 1974")}
    for lab, (mark, lbl) in LABMARK.items():
        sub = ksdf[(ksdf.salt == "NaCl") & (ksdf.lab == lab)]
        if len(sub):
            ax2.plot(sub.T_K, sub.ks_dm3_mol, mark, color="black",
                     mfc="white", mew=1.6, linestyle="none", label=lbl)
    sm = ksdf[(ksdf.salt == "NaCl") & (ksdf.lab == "smoothed")]
    ax2.plot(sm.T_K, sm.ks_dm3_mol, "-", color="black", lw=1.6,
             label="IUPAC smoothed")
    for name, col, ls, lbl in (
            ("NaCl_transfer", WONG[1], "--",
             r"transfer ($s=1$)"),
            ("NaCl_fitted", WONG[5], "-",
             rf"fitted ($s={st['fitS']['best']['x'][0]:.2f}$)")):
        c = cur[(cur.kind == "ks") & (cur.variant == name)]
        ax2.plot(c.T_K, c.val, ls, color=col, label=lbl)
    subk = ksdf[(ksdf.salt == "KCl")]
    ax2.plot(subk.T_K, subk.ks_dm3_mol, "v", color=WONG[3], mfc="white",
             mew=1.6, linestyle="none", label="KCl (exp)")
    ck = cur[(cur.kind == "ks") & (cur.variant == "KCl_fitted")]
    ax2.plot(ck.T_K, ck.val, "-.", color=WONG[3], label="KCl (model)")
    ax2.set_xlabel("T [K]")
    ax2.set_ylabel(r"$k_S$ [dm$^3$ mol$^{-1}$]")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.text(0.03, 0.03, "(b)", transform=ax2.transAxes, ha="left",
             fontsize=15, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_PNG)
    print("figure ->", FIG_PNG)
    st["figure"] = str(FIG_PNG.resolve().relative_to(Path(__file__).resolve().parents[3]))


def stage_finalize(st):
    pre = {c: guard_spot(c) for c in GUARD_CASES}
    try:
        rebind_c2(*st["fitB"]["best"]["x"],
                  s_all=st["fitS"]["best"]["x"][0])
        _ = _predict_x2((*st["fitB"]["best"]["x"],
                         st["fitS"]["best"]["x"][0], POL_C2,
                         344.26, 100.0, MI_WATER))
    finally:
        restore_ch4()
    post = {c: guard_spot(c) for c in GUARD_CASES}
    guards = {c: {
        "in_process_bit_identical": tuples_bit_identical(pre[c], post[c]),
        "vs_pre_pilot_refs_bit_identical":
            tuples_bit_identical(st["guard_refs"][c], post[c])}
        for c in GUARD_CASES}
    st["guards"] = guards
    print(json.dumps(guards, indent=2))

    final = {
        "pilot": "c2_pilot_2026_08",
        "data": {
            "water_csv": str(WATER_CSV), "ks_csv": str(KS_CSV),
            "binary": "Culberson-McKetta 1950 lo+hi (via IUPAC SDS Vol 9) "
                      "+ Mohammadi 2004 (T>=283 K); 116 pts, "
                      "283.1-444.26 K, 3.7-685 bar",
            "setschenow": "IUPAC SDS Vol 9 evaluation (Clever): NaCl "
                          "smoothed + 4 labs 283-348 K; KCl "
                          "Ben-Naim-Yaacobi 283-303 K",
        },
        "c2_constants": {"Tc_K": TC_C2, "Pc_Pa": PC_C2, "omega": W_C2,
                         "M_kg_mol": M_C2, "pol2_Cm2_J": POL_C2,
                         "pol2_note": "handbook alpha(C2H6) = 4.47 A^3"},
        "transfer": {"Tc_rescale": TC_RATIO,
                     "rule": "A_C2,i = 1.60222*A_CH4,i (all six ions), "
                             "B unchanged; s_all scales all six pairs"},
        "fitB": st["fitB"],
        "B": {k: v for k, v in st["B"].items() if k != "pred"},
        "ks0": st["ks0"], "fitS": st["fitS"], "S": st["S"],
        "pol2_sensitivity": st["pol2_sensitivity"],
        "guards": guards,
        "ks_method": {
            "P_total_bar": KS_P_TOTAL_BAR, "m_salt": KS_M_SALT,
            "basis": "molar (concentration), log10; Henry-normalized "
                     "volume solubility S_hat = c_gas/f_C2H6 with c_gas "
                     "and c_salt from the MODEL brine density",
        },
    }
    RESULTS_JSON.write_text(json.dumps(final, indent=2))
    print("final ->", RESULTS_JSON)


STAGES = {
    "refs": stage_refs, "fitB": stage_fitB, "evalB": stage_evalB,
    "ks0": stage_ks0, "fitS": stage_fitS, "evalS": stage_evalS,
    "curves": stage_curves, "figure": stage_figure,
    "finalize": stage_finalize,
}
POOL_STAGES = {"fitB", "evalB", "curves"}


def main():
    global _POOL
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=list(STAGES) + ["all"])
    args = ap.parse_args()
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    st = _state()
    for name in stages:
        if name in POOL_STAGES and _POOL is None:
            _POOL = mp.get_context("spawn").Pool(N_WORKERS)
        t0 = time.perf_counter()
        print(f"=== stage {name} ===", flush=True)
        STAGES[name](st)
        _save_state(st)
        print(f"=== stage {name} done ({time.perf_counter()-t0:.1f}s) ===",
              flush=True)
    if _POOL is not None:
        _POOL.close()
        _POOL.join()


if __name__ == "__main__":
    main()
