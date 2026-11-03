"""C3H8 (propane) pilot: extend the multi-salt eCPA to propane at the
cost of one salt-free binary fit, mirroring the C2H6 pilot.

Data (see raw/ and its extraction notes for provenance):
  raw/kobayashi1951_2phase_xC3_water_smoothed.csv
      Kobayashi 1951 dissertation Table VIII: graphically smoothed
      propane-in-water mole fractions on a 15 P x 10 T grid,
      289-428 K, 6.9-207 bar (spans the vapor and liquid-propane
      branches across the three-phase locus).
  raw/chapoy2004_*.csv (when present)
      Chapoy et al. 2004 (Fluid Phase Equilib. 226, 213) propane-in-
      water solubilities, 277.62-368.16 K.
  raw/iupac24_propane_salts_*.csv (when present)
      IUPAC SDS Vol 24 salt-solution data -> Setschenow calibration
      (stages ks0/fitS added once the extraction lands).

Framing (identical to the C2H6 pilot):
  fitB: quadratic kij(T) = Ak*(T/Tc1)^2 + Bk*(T/Tc1) + Ck fitted to
        the salt-free binary aqueous solubility; ALL six gas-ion
        energies transferred from CH4 with the function-preserving Tc
        rescale A_C3,i = A_CH4,i * Tc(C3H8)/Tc(CH4) = 1.94076 * A_CH4,i.

Stages (state accumulates in work_state.json):
    python c3_pilot.py --stage data     # assemble c3_water_data.csv
    python c3_pilot.py --stage refs     # unpatched CH4/CO2 guard flashes
    python c3_pilot.py --stage probe    # feed-independence degeneracy scan
    python c3_pilot.py --stage fitB     # fit kij (optionally --warm)
    python c3_pilot.py --stage evalB    # per-source AAREs

C3H8 constants: Tc = 369.83 K, Pc = 42.48 bar, omega = 0.1523,
M = 0.0440956 kg/mol. Polarizability alpha(C3H8) = 6.29 A^3 =
6.9986e-40 C^2 m^2 J^-1 (CRC; enters eps_inf only).
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
RAW = HERE / "raw"
WATER_CSV = HERE / "c3_water_data.csv"
STATE_JSON = HERE / "work_state.json"

# ---------------------------------------------------------------------------
# C3H8 constants
# ---------------------------------------------------------------------------
TC_C3 = 369.83           # K
PC_C3 = 42.48e5          # Pa
W_C3 = 0.1523
M_C3 = 0.0440956         # kg/mol
POL_C3 = 6.9986e-40      # C^2 m^2 J^-1; CRC alpha = 6.29 A^3

TC_RATIO = TC_C3 / 190.56        # 1.94076, function-preserving DU(T) transfer
T_FLOOR = 283.15                 # calibration floor
R_J = 8.31446261815324

# Feed ladder: water-rich feeds converge best on the liquid-propane
# branch (probe 2026-08-12: 50/50 feeds can fail just above the locus).
Z_GAS_RETRIES = (0.05, 0.10, 0.02, 0.20, 0.30)
N_WORKERS = 8

# ---------------------------------------------------------------------------
# Rebind machinery (Tier-1 recipe, from the C2H6/N2 pilots)
# ---------------------------------------------------------------------------
NAMESPACES = (prm, lnphi_mod, eos_mod, flash_mod)
SCALARS = ("Tc2", "Pc2", "w2", "b2", "a02", "c12", "Mh", "pol2",
           "Ak12", "Bk12", "Ck12", "k23")

_ORIG = {name: getattr(prm, name) for name in SCALARS}
_ORIG_A_U2i = prm.A_U2i.copy()
_ORIG_B_U2i = prm.B_U2i.copy()
_ORIG_WILSON = (float(flash_mod._TC[1]), float(flash_mod._PC_BAR[1]),
                float(flash_mod._OMEGA[1]))


def rebind_c3(Ak, Bk, Ck, s_all=1.0, pol2=POL_C3, s12a=0.0):
    """Swap slot 2 CH4 -> C3H8 in all four consuming namespaces.

    ``s12a``: water-C3H8 cross-association strength (parameters.S12A),
    0.0 = no solvation (the pre-solvation pilot baseline)."""
    R = prm.R
    prm.S12A = float(s12a)
    vals = dict(
        Tc2=TC_C3, Pc2=PC_C3, w2=W_C3,
        b2=0.08664 * R * TC_C3 / PC_C3,
        a02=0.42748 * R ** 2 * TC_C3 ** 2 / PC_C3,
        c12=0.480 + 1.574 * W_C3 - 0.176 * W_C3 ** 2,
        Mh=M_C3, pol2=pol2,
        Ak12=float(Ak), Bk12=float(Bk), Ck12=float(Ck), k23=0.0,
    )
    for ns in NAMESPACES:
        for name, v in vals.items():
            if hasattr(ns, name):
                setattr(ns, name, v)
    prm.A_U2i[:] = _ORIG_A_U2i * TC_RATIO * s_all
    prm.B_U2i[:] = _ORIG_B_U2i * s_all
    flash_mod._TC[1] = TC_C3
    flash_mod._PC_BAR[1] = PC_C3 / 1e5
    flash_mod._OMEGA[1] = W_C3


def restore_ch4():
    prm.S12A = 0.0
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
MI_WATER = (0.0,) * 6


def _predict_x2(task):
    """(Ak, Bk, Ck, s_all, pol2, T_K, P_bar, mi6, zg_list) -> aqueous
    C3H8 mole fraction, or NaN."""
    Ak, Bk, Ck, s_all, pol2, T, P_bar, mi6, zgs = task
    try:
        rebind_c3(Ak, Bk, Ck, s_all=s_all, pol2=pol2)
        mi = np.asarray(mi6, dtype=float)
        for zg in zgs:
            z = np.array([1.0 - zg, zg, 0.0])
            try:
                res = stability_flash(float(T), float(P_bar), z, mi)
            except Exception:
                continue
            if res.phases == 2 and res.flash_result is not None \
                    and res.flash_result.converged:
                return float(res.flash_result.x_aq[1])
        return float("nan")
    finally:
        restore_ch4()


_POOL = None


def predict_water_all(df, Ak, Bk, Ck, zgs=Z_GAS_RETRIES):
    tasks = [(Ak, Bk, Ck, 1.0, POL_C3, r.T_K, r.P_bar, MI_WATER, zgs)
             for r in df.itertuples()]
    if _POOL is not None:
        out = _POOL.map(_predict_x2, tasks)
    else:
        out = [_predict_x2(t) for t in tasks]
    return np.array(out)


# ---------------------------------------------------------------------------
# Setschenow machinery (gas-MOLALITY basis, matching IUPAC Vol 24 ksmm:
# ksmm = (1/m2) log10(m1_0/m1) at fixed propane partial pressure, log10,
# kg/mol). Henry normalization by the model gas fugacity cancels the
# partial-pressure difference between the water and brine runs and makes
# the choice of P_TOTAL immaterial in the dilute limit. No density
# needed on this basis (unlike the C2 pilot's volume-basis kscc).
# ---------------------------------------------------------------------------
KS_P_TOTAL_BAR = 20.0     # water flashes can collapse to the dew-line
                          # rescue below ~15 bar at T > 300 K (C2 lesson)
KS_M_SALT = 1.0           # mol/kg; Morrison & Billett's tables are at 1 m


def mi_for(salt, m):
    if salt == "NaCl":
        return (m, m, 0.0, 0.0, 0.0, 0.0)
    if salt == "KCl":
        return (0.0, m, m, 0.0, 0.0, 0.0)
    raise ValueError(salt)


def _mgas_f(task):
    """(Ak,Bk,Ck,s_all,pol2,T,P_bar,mi6,zgs) -> (m_gas [mol/kg-w],
    f_C3H8_gas [bar]) or (nan, nan)."""
    Ak, Bk, Ck, s_all, pol2, T, P_bar, mi6, zgs = task
    try:
        rebind_c3(Ak, Bk, Ck, s_all=s_all, pol2=pol2)
        mi = np.asarray(mi6, dtype=float)
        for zg in zgs:
            z = np.array([1.0 - zg, zg, 0.0])
            try:
                res = stability_flash(float(T), float(P_bar), z, mi)
            except Exception:
                continue
            if res.phases == 2 and res.flash_result is not None \
                    and res.flash_result.converged:
                fr = res.flash_result
                x1w, x2w, _ = fr.x_aq
                m_gas = x2w / (prm.Mw * x1w)
                return (float(m_gas), float(fr.fugacities_gas[1]))
        return (float("nan"),) * 2
    finally:
        restore_ch4()


def ks_model(kij, s_all, T, salt, m=KS_M_SALT, water_cache=None):
    """Model ksmm (kg/mol, log10) at T via Henry-normalized molality."""
    key = float(T)
    if water_cache is not None and key in water_cache:
        sw = water_cache[key]
    else:
        sw = _mgas_f((*kij, 1.0, POL_C3, T, KS_P_TOTAL_BAR, MI_WATER,
                      Z_GAS_RETRIES))
        if water_cache is not None:
            water_cache[key] = sw
    sb = _mgas_f((*kij, s_all, POL_C3, T, KS_P_TOTAL_BAR,
                  mi_for(salt, m), Z_GAS_RETRIES))
    if not (np.isfinite(sw[0]) and np.isfinite(sb[0])):
        return float("nan")
    return float(np.log10((sw[0] / sw[1]) / (sb[0] / sb[1])) / m)


# Calibration targets from raw/iupac24_propane_salts_evaluated.csv:
#   NaCl: Morrison & Billett 1952 ksmm at 1.00 mol/kg -- the evaluator's
#         preferred, self-consistent 60-K set.
#   KCl:  Yano 1974 ksmm at 298.2 K, 'tentative' rows only (the
#         'doubtful' 0.099 row excluded per the evaluation); mean used.
KS_NACL_TARGETS = [(285.75, 0.216), (303.15, 0.194),
                   (322.55, 0.178), (344.85, 0.165)]
KS_KCL_T = 298.2
KS_KCL_TARGET = float(np.mean([0.187, 0.187, 0.199]))


def make_resid_ks(kij, targets, water_cache):
    def resid(x):
        s = float(x[0])
        out = []
        for T, ks_t in targets:
            ks = ks_model(kij, s, T, "NaCl", water_cache=water_cache)
            out.append(ks / ks_t - 1.0 if np.isfinite(ks) else NAN_RESIDUAL)
        return np.array(out)
    return resid


# ---------------------------------------------------------------------------
# Guards (production CH4/CO2 must be bit-identical after restore)
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


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def stage_data(st):
    import pandas as pd
    frames = []

    kob = pd.read_csv(RAW / "kobayashi1951_2phase_xC3_water_smoothed.csv",
                      comment="#")
    n0 = len(kob)
    kob = kob[kob["note"].isna() | (kob["note"].astype(str).str.strip()
                                    .isin(["", "nan"]))]
    kob = kob[kob.T_K >= T_FLOOR]
    frames.append(pd.DataFrame(dict(
        T_K=kob.T_K, P_bar=kob.P_bar, x_molefrac=kob.xC3,
        source="Kobayashi1951sm")))
    print(f"  Kobayashi smoothed: {len(kob)} rows kept of {n0} "
          f"(T >= {T_FLOOR} K, flagged rows dropped)")

    chap_files = sorted(RAW.glob("chapoy2004_*.csv"))
    for f in chap_files:
        ch = pd.read_csv(f, comment="#")
        # Prefer the true mole-fraction column; NEVER a scaled
        # as-printed column like xexp_e4 (10^4 x mole fraction).
        xcol = ("x_C3H8" if "x_C3H8" in ch.columns else
                next(c for c in ch.columns
                     if c.lower().startswith("x") and "_e" not in c))
        pcol = "P_bar" if "P_bar" in ch.columns else next(
            c for c in ch.columns if c.startswith("P_"))
        P_bar = ch[pcol] if pcol == "P_bar" else ch[pcol] * (
            10.0 if "MPa" in pcol else 1.0)
        sub = pd.DataFrame(dict(T_K=ch.T_K, P_bar=P_bar,
                                x_molefrac=ch[xcol],
                                source="Chapoy2004"))
        sub = sub[sub.T_K >= T_FLOOR]
        frames.append(sub)
        print(f"  {f.name}: {len(sub)} rows kept (T >= {T_FLOOR} K)")
    if not chap_files:
        print("  (no chapoy2004_*.csv yet — Kobayashi-only assembly)")

    df = pd.concat(frames, ignore_index=True)
    # Drop rows the degeneracy probe flagged in an earlier run.
    excl = st.get("probe", {}).get("excluded", [])
    if excl:
        keys = {(round(t, 2), round(p, 3)) for t, p in excl}
        mask = [(round(r.T_K, 2), round(r.P_bar, 3)) not in keys
                for r in df.itertuples()]
        df = df[mask]
        print(f"  degeneracy exclusions applied: {len(excl)}")
    df.to_csv(WATER_CSV, index=False)
    st["data"] = {"n_points": len(df),
                  "sources": df.source.value_counts().to_dict()}
    print(f"  -> {WATER_CSV} ({len(df)} points)")


def load_water():
    import pandas as pd
    return pd.read_csv(WATER_CSV)


# ---------------------------------------------------------------------------
# Feed-independence degeneracy probe (C2 lesson: low-P water flashes can
# collapse to the dew-line rescue branch with feed-dependent x2)
# ---------------------------------------------------------------------------
def stage_probe(st):
    df = load_water()
    kij = st.get("fitB", {}).get("best", {}).get(
        "x", [-4.0935, 5.6634, -1.7845])
    lo = predict_water_all(df, *kij, zgs=(0.05,))
    hi = predict_water_all(df, *kij, zgs=(0.30,))
    bad = []
    for i, r in enumerate(df.itertuples()):
        if np.isfinite(lo[i]) and np.isfinite(hi[i]):
            if abs(lo[i] / hi[i] - 1.0) > 0.02:
                bad.append((float(r.T_K), float(r.P_bar)))
        # one-sided convergence is fine (ladder handles it); both-NaN
        # rows fail the fit residual guard instead
    st["probe"] = {"excluded": bad, "n_flagged": len(bad)}
    print(f"  feed-independence probe: {len(bad)} degenerate points "
          f"flagged -> re-run --stage data to apply")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _state():
    return json.loads(STATE_JSON.read_text()) if STATE_JSON.exists() else {}


def _save_state(st):
    STATE_JSON.write_text(json.dumps(st, indent=2))


NAN_RESIDUAL = 5.0


def make_resid_water(df):
    exp = df["x_molefrac"].to_numpy()

    def resid(x):
        pred = predict_water_all(df, x[0], x[1], x[2])
        r = pred / exp - 1.0
        return np.where(np.isfinite(r), r, NAN_RESIDUAL)
    return resid


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
               "nfev": int(res.nfev),
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


def aare_water(df, pred):
    out = {}
    exp = df["x_molefrac"].to_numpy()
    labels = list(df.source.unique()) + ["all"]
    for label in labels:
        m = np.asarray(df.source == label) if label != "all" \
            else np.ones(len(df), dtype=bool)
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


def stage_fitB(st, warm=False):
    df = load_water()
    ch4 = [-4.0935, 5.6634, -1.7845]        # production CH4-H2O values
    if warm and "fitB" in st:
        starts = [st["fitB"]["best"]["x"]]
        label = "fitB-warm"
    else:
        starts = [ch4, [ch4[0] * 1.25, ch4[1] * 0.85, ch4[2] + 0.08]]
        label = "fitB"
    bounds = ([-60.0, -60.0, -60.0], [60.0, 60.0, 60.0])
    best, runs = fit_least_squares(make_resid_water(df), starts, bounds,
                                   label)
    st["fitB"] = {"best": best, "runs": runs, "n_points": len(df),
                  "bounds": bounds, "diff_step": 1e-3}


def stage_evalB(st):
    df = load_water()
    kij = st["fitB"]["best"]["x"]
    pred = predict_water_all(df, *kij)
    st["B"] = {"kij": kij, "aare": aare_water(df, pred),
               "pred": [float(p) for p in pred]}
    print(json.dumps(st["B"]["aare"], indent=2))
    # guard: production untouched
    for c in GUARD_CASES:
        now = guard_spot(c)
        ref = st.get("guard_refs", {}).get(c)
        if ref is not None:
            same = all((x == y) or (np.isnan(x) and np.isnan(y))
                       for x, y in zip(now, ref))
            print(f"  guard {c}: {'bit-identical' if same else 'CHANGED!'}")


def stage_ks0(st):
    """Pure-transfer (s=1) Setschenow prediction, NaCl + KCl."""
    kij = st["fitB"]["best"]["x"]
    wc = {}
    out = {"NaCl": [], "KCl": []}
    for T, tgt in KS_NACL_TARGETS:
        ks = ks_model(kij, 1.0, T, "NaCl", water_cache=wc)
        out["NaCl"].append({"T_K": T, "ks": ks, "target": tgt})
        print(f"  ks(NaCl, {T} K, s=1) = {ks:.4f}  (target {tgt})",
              flush=True)
    ks = ks_model(kij, 1.0, KS_KCL_T, "KCl", water_cache=wc)
    out["KCl"].append({"T_K": KS_KCL_T, "ks": ks, "target": KS_KCL_TARGET})
    print(f"  ks(KCl, {KS_KCL_T} K, s=1) = {ks:.4f}  "
          f"(target {KS_KCL_TARGET:.3f})", flush=True)
    devs = [d["ks"] / d["target"] - 1.0 for d in out["NaCl"]
            if np.isfinite(d["ks"])]
    out["NaCl_mean_dev_pct"] = float(np.mean(devs) * 100.0)
    print(f"  transfer (s=1) NaCl mean deviation: "
          f"{out['NaCl_mean_dev_pct']:+.1f}%")
    st["ks0"] = out


def stage_fitS(st):
    """Fit s_all (all six transferred (A,B) pairs jointly) to the
    Morrison & Billett NaCl ksmm set; KCl stays a same-s prediction."""
    kij = st["fitB"]["best"]["x"]
    wc = {}
    best, runs = fit_least_squares(make_resid_ks(kij, KS_NACL_TARGETS, wc),
                                   [[1.0], [1.5]], ([0.2], [4.0]), "fitS")
    st["fitS"] = {"best": best, "runs": runs, "targets": KS_NACL_TARGETS}


def stage_evalS(st):
    kij = st["fitB"]["best"]["x"]
    s = st["fitS"]["best"]["x"][0]
    wc = {}
    out = {"s_all": s, "NaCl": [], "KCl": []}
    for T, tgt in KS_NACL_TARGETS:
        ks = ks_model(kij, s, T, "NaCl", water_cache=wc)
        out["NaCl"].append({"T_K": T, "ks": ks, "target": tgt})
        print(f"  ks(NaCl, {T} K, s={s:.4f}) = {ks:.4f} (target {tgt})",
              flush=True)
    ks_k = ks_model(kij, s, KS_KCL_T, "KCl", water_cache=wc)
    out["KCl"].append({"T_K": KS_KCL_T, "ks": ks_k,
                       "target": KS_KCL_TARGET})
    print(f"  ks(KCl, {KS_KCL_T} K, same s) = {ks_k:.4f} "
          f"(target {KS_KCL_TARGET:.3f})", flush=True)
    for salt in ("NaCl", "KCl"):
        devs = [d["ks"] / d["target"] - 1.0 for d in out[salt]
                if np.isfinite(d["ks"])]
        out[f"{salt}_aad_pct"] = float(np.mean(np.abs(devs)) * 100.0)
        out[f"{salt}_mean_dev_pct"] = float(np.mean(devs) * 100.0)
        print(f"  {salt}: mean dev {out[f'{salt}_mean_dev_pct']:+.1f}%, "
              f"AAD {out[f'{salt}_aad_pct']:.1f}%")
    st["S"] = out


def stage_fitK(st):
    """Optional second scale on K+ only (two-scale convention of the
    C2H6 study), fitted to the Yano KCl point with (Na,Cl) held at the
    NaCl-fitted scale."""
    kij = st["fitB"]["best"]["x"]
    s_nacl = st["fitS"]["best"]["x"][0]

    def resid(x):
        sK = float(x[0])
        # scale (A,B) of K+ only: rebind applies s_all to ALL pairs, so
        # express per-ion scaling through a custom rebind here.
        try:
            rebind_c3(*kij, s_all=s_nacl)
            prm.A_U2i[2] = _ORIG_A_U2i[2] * TC_RATIO * sK
            prm.B_U2i[2] = _ORIG_B_U2i[2] * sK
            sw = None
            for zg in Z_GAS_RETRIES:
                z = np.array([1.0 - zg, zg, 0.0])
                try:
                    r = stability_flash(KS_KCL_T, KS_P_TOTAL_BAR, z,
                                        np.zeros(6))
                except Exception:
                    continue
                if r.phases == 2 and r.flash_result is not None:
                    fr = r.flash_result
                    sw = (fr.x_aq[1] / (prm.Mw * fr.x_aq[0]),
                          fr.fugacities_gas[1])
                    break
            sb = None
            for zg in Z_GAS_RETRIES:
                z = np.array([1.0 - zg, zg, 0.0])
                try:
                    r = stability_flash(KS_KCL_T, KS_P_TOTAL_BAR, z,
                                        np.array(mi_for("KCl", 1.0)))
                except Exception:
                    continue
                if r.phases == 2 and r.flash_result is not None:
                    fr = r.flash_result
                    sb = (fr.x_aq[1] / (prm.Mw * fr.x_aq[0]),
                          fr.fugacities_gas[1])
                    break
            if sw is None or sb is None:
                return np.array([NAN_RESIDUAL])
            ks = np.log10((sw[0] / sw[1]) / (sb[0] / sb[1])) / 1.0
            return np.array([ks / KS_KCL_TARGET - 1.0])
        finally:
            restore_ch4()

    best, runs = fit_least_squares(resid, [[s_nacl], [1.0]],
                                   ([0.2], [4.0]), "fitK")
    st["fitK"] = {"best": best, "runs": runs,
                  "s_nacl_held": s_nacl, "target": KS_KCL_TARGET}




# ---------------------------------------------------------------------------
# Solvation pilot: joint fit of kij(T) + S12A against the
# aqueous solubility AND the water content of the propane-rich phase
# (Kobayashi Table VII spans the vapor and liquid-propane branches).
# ---------------------------------------------------------------------------
YWATER_CSV = HERE / "c3_ywater_data.csv"


def stage_dataW(st):
    import pandas as pd
    kob = pd.read_csv(RAW / "kobayashi1951_2phase_yH2O_propane_rich_smoothed.csv",
                      comment="#")
    n0 = len(kob)
    ycol = next(c for c in kob.columns if c.lower().startswith("yh2o"))
    if "note" in kob.columns:
        kob = kob[kob["note"].isna() | (kob["note"].astype(str).str.strip()
                                        .isin(["", "nan"]))]
    kob = kob[kob.T_K >= T_FLOOR]
    df = pd.DataFrame(dict(T_K=kob.T_K, P_bar=kob.P_bar,
                           y_H2O=kob[ycol], source="Kobayashi1951sm"))
    df.to_csv(YWATER_CSV, index=False)
    st["dataW"] = {"n_points": len(df), "of": n0, "ycol": ycol}
    print(f"  water-content targets: {len(df)} of {n0} rows "
          f"(T >= {T_FLOOR} K) -> {YWATER_CSV}")


def _predict_pair_3p(task):
    """(Ak,Bk,Ck,s12a,T,P_bar) -> (x_C3_aq, y_H2O_gas) via the
    three-phase driver (replaces metastable vapor splits above the
    locus; the propane-rich phase is fr.y_gas on either branch)."""
    Ak, Bk, Ck, s12a, T, P_bar = task
    from ms_ecpa import stability_flash3
    try:
        rebind_c3(Ak, Bk, Ck, s12a=s12a)
        mi = np.zeros(6)
        for zg in Z_GAS_RETRIES:
            z = np.array([1.0 - zg, zg, 0.0])
            try:
                r = stability_flash3(float(T), float(P_bar), z, mi)
            except Exception:
                continue
            if r.phases == 3 and r.flash3 is not None:
                return (float(r.flash3.x_aq[1]),
                        float(r.flash3.y_gas[0]))
            fr = r.two_phase.flash_result
            if r.phases == 2 and fr is not None and fr.converged:
                return (float(fr.x_aq[1]), float(fr.y_gas[0]))
        return (float("nan"),) * 2
    finally:
        restore_ch4()


def _predict_all_3p(dfx, dfy, Ak, Bk, Ck, s12a):
    """Predict x_C3 at dfx's states and y_H2O at dfy's states."""
    tasks = [(Ak, Bk, Ck, s12a, r.T_K, r.P_bar) for r in dfx.itertuples()]
    tasks += [(Ak, Bk, Ck, s12a, r.T_K, r.P_bar) for r in dfy.itertuples()]
    if _POOL is not None:
        out = _POOL.map(_predict_pair_3p, tasks)
    else:
        out = [_predict_pair_3p(t) for t in tasks]
    nx = len(dfx)
    x_pred = np.array([o[0] for o in out[:nx]])
    y_pred = np.array([o[1] for o in out[nx:]])
    return x_pred, y_pred


def stage_fitA(st):
    """Joint fit (Ak, Bk, Ck, S12A): relative residuals on the aqueous
    solubility + log residuals on the propane-rich-phase water
    content (spans decades across the vapor/liquid branches)."""
    import pandas as pd
    dfx = load_water()
    dfy = pd.read_csv(YWATER_CSV)
    x_exp = dfx["x_molefrac"].to_numpy()
    y_exp = dfy["y_H2O"].to_numpy()

    def resid(v):
        x_pred, y_pred = _predict_all_3p(dfx, dfy, v[0], v[1], v[2], v[3])
        rx = np.where(np.isfinite(x_pred), x_pred / x_exp - 1.0,
                      NAN_RESIDUAL)
        with np.errstate(divide="ignore", invalid="ignore"):
            ry = np.log(y_pred / y_exp)
        ry = np.where(np.isfinite(ry), ry, NAN_RESIDUAL)
        return np.concatenate([rx, ry])

    kij0 = st["fitB"]["best"]["x"]
    starts = [[*kij0, 0.03], [*kij0, 0.15]]
    bounds = ([-60.0, -60.0, -60.0, 0.0], [60.0, 60.0, 60.0, 2.0])
    best, runs = fit_least_squares(resid, starts, bounds, "fitA")
    st["fitA"] = {"best": best, "runs": runs,
                  "n_x": len(dfx), "n_y": len(dfy), "bounds": bounds}


def stage_evalA(st):
    """Evaluate the joint (kij, S12A) optimum against: aqueous AAREs,
    water-content log-AAD split by branch (vapor vs liquid propane via
    the smoothed three-phase locus), and the S12A=0 counterfactual."""
    import pandas as pd
    dfx = load_water()
    dfy = pd.read_csv(YWATER_CSV)
    locus = pd.read_csv(RAW / "kobayashi1951_3phase_smoothed.csv",
                        comment="#")
    P3 = np.interp(dfy.T_K.to_numpy(), locus.T_K.to_numpy(),
                   locus.P_bar.to_numpy(),
                   left=np.nan, right=np.inf)
    is_liq = dfy.P_bar.to_numpy() > P3   # above locus: liquid propane
    x_exp = dfx["x_molefrac"].to_numpy()
    y_exp = dfy["y_H2O"].to_numpy()

    out = {}
    cases = {"fitA": [*st["fitA"]["best"]["x"]],
             "fitB_s0": [*st["fitB"]["best"]["x"], 0.0]}
    for name, v in cases.items():
        x_pred, y_pred = _predict_all_3p(dfx, dfy, *v)
        okx = np.isfinite(x_pred)
        aare_x = float(np.mean(np.abs(x_pred[okx] / x_exp[okx] - 1)) * 100)
        oky = np.isfinite(y_pred)
        dev = np.abs(np.log(y_pred / y_exp))
        aad_vap = float(np.expm1(np.nanmean(dev[oky & ~is_liq])) * 100)
        aad_liq = float(np.expm1(np.nanmean(dev[oky & is_liq])) * 100)
        out[name] = dict(params=[float(q) for q in v],
                         aare_x_pct=aare_x,
                         yH2O_vapor_logAAD_pct=aad_vap,
                         yH2O_liquid_logAAD_pct=aad_liq,
                         n_x_conv=int(okx.sum()),
                         n_y_conv=int(oky.sum()),
                         n_vap=int((~is_liq).sum()),
                         n_liq=int(is_liq.sum()))
        print(f"  [{name}] x_C3 AARE {aare_x:.2f}% | y_H2O logAAD "
              f"vapor {aad_vap:.1f}% / liquid {aad_liq:.1f}% "
              f"(n={int((~is_liq).sum())}/{int(is_liq.sum())})",
              flush=True)
    st["A"] = out
    for c in GUARD_CASES:
        now = guard_spot(c)
        ref = st.get("guard_refs", {}).get(c)
        if ref is not None:
            same = all((a == b) or (np.isnan(a) and np.isnan(b))
                       for a, b in zip(now, ref))
            print(f"  guard {c}: {'bit-identical' if same else 'CHANGED!'}")


STAGES = {"data": stage_data, "refs": stage_refs, "probe": stage_probe,
          "fitB": stage_fitB, "evalB": stage_evalB,
          "ks0": stage_ks0, "fitS": stage_fitS, "evalS": stage_evalS,
          "fitK": stage_fitK,
          "dataW": stage_dataW, "fitA": stage_fitA, "evalA": stage_evalA}


def main():
    global _POOL
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--warm", action="store_true")
    args = ap.parse_args()
    st = _state()
    if args.stage in ("fitB", "evalB", "probe", "fitA", "evalA"):
        _POOL = mp.Pool(N_WORKERS)
    t0 = time.perf_counter()
    if args.stage == "fitB":
        stage_fitB(st, warm=args.warm)
    else:
        STAGES[args.stage](st)
    _save_state(st)
    print(f"[{args.stage}] done in {time.perf_counter() - t0:.1f} s")
    if _POOL is not None:
        _POOL.close()


if __name__ == "__main__":
    main()
