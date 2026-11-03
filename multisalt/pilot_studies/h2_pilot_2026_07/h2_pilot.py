"""H2-brine validation pilot: adopt Coelho 2025 (Coelho, Mercier Franco,
Firoozabadi, IJHE 178 (2025) 151361) H2 e-CPA parameterization into the
multi-salt production framework by runtime slot-swap, and measure the
basis-transfer cost.

F. M. Coelho's fits were made in the Maribo-Mogensen 2015 SALT-SPECIFIC NaCl
basis; our framework is ION-SPECIFIC (different sigma/R_B/b_ion, nonzero
ion a0, per-ion water-ion energies, per-ion Peneloux). Adopting his
kij(H2-H2O) + shared Na/Cl DU_hs verbatim in ms_ecpa is therefore a
basis-transfer PREDICTION, not a reproduction of his curves.

Stages (each a separate foreground process; state in work_state.json):

    python h2_pilot.py --stage refs      # unpatched CH4/CO2 guard spot flashes
    python h2_pilot.py --stage v1water   # V1: 216-row H2-H2O set (163 x + 78 y)
    python h2_pilot.py --stage v1brine   # V1: 71-row NaCl brine set + guard demo
    python h2_pilot.py --stage v2fit     # V2 (conditional): refit DU_ref only
    python h2_pilot.py --stage v2eval    # V2 predictions on the 71 brine points
    python h2_pilot.py --stage v3        # ternary demo vs F. M. Coelho's HCB curves
    python h2_pilot.py --stage curves    # smooth curves for the figure
    python h2_pilot.py --stage figure    # house-style figure
    python h2_pilot.py --stage finalize  # guards + h2_results.json + table

Tier-1 rebind recipe (h2_scoping_2026_07/README.md section 4, verified by
the N2 pilot): patch FOUR module namespaces (parameters + the lnphi/eos
star-import snapshots + the flash by-name imports), plus the flash Wilson
arrays _TC/_PC_BAR/_OMEGA in-place. A_U2i/B_U2i are the SAME ndarray in
all namespaces (star import binds the reference), so in-place [:]
assignment propagates. F. M. Coelho's H2-ion energy is the quadratic Talfa form
U_hs(T) = DU_ref + alpha*R*[(1-T/Talfa)^2 - (1-Tref/Talfa)^2], Tref =
298.15 K, which production's linear A*(T/Tc2)+B cannot represent globally
-- but flash T is fixed per call, so a per-POINT rebind A_U2i = 0,
B_U2i = U_hs(T) is EXACT with zero library edits. NaCl-only guard: no
H2-K/Ca/Mg/SO4 energies exist anywhere in the H2 tree; refuse feeds with
nonzero m_K/m_Ca/m_Mg/m_SO4 while the H2 slot is armed.

Runtime rebinding only; restore in finally blocks; ms_ecpa/ untouched.

H2 constants (scoping README section 1.1, paper Table Pures): Tc = 33.2 K,
Pc = 13.0 bar, omega = -0.220 (standard Soave alpha -- well-behaved, zero
crossing only at ~2680 K), M = 0.002 kg/mol, no association sites,
pol(H2) = 0.787 A^3 (NIST) -> 8.756e-41 C m^2/J.
"""
from __future__ import annotations

import os
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

# Resolve against this repository (scripts/fig_additional_gases.py imports
# this module).
CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))

import ms_ecpa.parameters as prm            # noqa: E402
import ms_ecpa.lnphi as lnphi_mod           # noqa: E402
import ms_ecpa.eos as eos_mod               # noqa: E402
import ms_ecpa.flash as flash_mod           # noqa: E402
from ms_ecpa import stability_flash         # noqa: E402

HERE = Path(__file__).resolve().parent
SCOPING = CODE / "pilot_studies" / "h2_scoping_2026_07"
# Original model-output files of the earlier H2 code (not distributed; --stage v3 only).
HCB = Path(os.environ.get("ECPA_SOURCE_MATERIALS", "source_materials")) / "EoS/H2/HCB"
STATE_JSON = HERE / "work_state.json"
RESULTS_JSON = HERE / "h2_results.json"
CURVES_CSV = HERE / "curves_cache.csv"
FIG_PNG = HERE / "fig_h2_brine_pilot.png"

# ---------------------------------------------------------------------------
# H2 constants (Coelho 2025, scoping README section 1)
# ---------------------------------------------------------------------------
TC_H2 = 33.2             # K
PC_H2 = 13.0e5           # Pa
W_H2 = -0.220
M_H2 = 0.002             # kg/mol
EPS0 = 8.8541878128e-12
POL_H2 = 0.787e-30 * 4.0 * np.pi * EPS0    # = 8.756e-41 C m^2/J

# kij(H2-H2O) = A*(T/Tc1)^2 + B*(T/Tc1) + C, Tc1 = 647.29 K (OPT4, exact
# transfer -- his water CPA set is bit-identical to production)
AK12_H2 = -7.766171507123477
BK12_H2 = 12.307804628800731
CK12_H2 = -4.590647844067419

# Shared Na+/Cl- DU_hs quadratic Talfa form (OPT4 = paper Table BinPar)
DU_REF = 9992.893420263154   # J/mol at Tref
ALPHA_K = -9889.079908981199  # K
TALFA = 364.4872690301035    # K
TREF = 298.15                # K

K23_DISK = 0.1277343750000001  # H2-CO2 kij his ternary script actually used
K23_PAPER = 0.145              # printed in paper Table BinPar; fit file lost

Z_GAS_RETRIES = (0.05, 0.10, 0.02, 0.20, 0.30, 0.01, 0.003)
N_WORKERS = 8
FEED_INDEP_RTOL = 0.005   # binary uniqueness check: two feeds must agree
BETA_INTERIOR = 0.995     # beta above this = boundary/shadow solution

COELHO2025_AAD = {"water_x": 8.97, "water_y": 11.45, "brine": 8.64}


def u_hs(T, du_ref=DU_REF):
    """F. M. Coelho's quadratic Talfa H2-ion energy, exact at any T."""
    return du_ref + ALPHA_K * prm.R * ((1.0 - T / TALFA) ** 2
                                       - (1.0 - TREF / TALFA) ** 2)


# ---------------------------------------------------------------------------
# Rebind machinery (Tier-1 recipe; N2-pilot pattern)
# ---------------------------------------------------------------------------
NAMESPACES = (prm, lnphi_mod, eos_mod, flash_mod)
SCALARS = ("Tc2", "Pc2", "w2", "b2", "a02", "c12", "Mh", "pol2",
           "Ak12", "Bk12", "Ck12", "k23")

_ORIG = {name: getattr(prm, name) for name in SCALARS}
_ORIG_A_U2i = prm.A_U2i.copy()
_ORIG_B_U2i = prm.B_U2i.copy()
_ORIG_WILSON = (float(flash_mod._TC[1]), float(flash_mod._PC_BAR[1]),
                float(flash_mod._OMEGA[1]))


def rebind_h2(T, du_ref=DU_REF, k23=K23_DISK):
    """Swap slot 2 CH4 -> H2 in all four consuming namespaces.

    Per-POINT exact representation of the quadratic U_hs(T): A_U2i = 0,
    B_U2i = u_hs(T) for Na+ and Cl- (production evaluates U2i =
    A_U2i*(T/Tc2) + B_U2i, so with A = 0 the value is exact at this T).
    K/Ca/Mg/SO4 slots get 0 -- NO H2 energies exist for them; callers
    must enforce the NaCl-only guard (assert_nacl_only).
    """
    R = prm.R
    vals = dict(
        Tc2=TC_H2, Pc2=PC_H2, w2=W_H2,
        b2=0.08664 * R * TC_H2 / PC_H2,
        a02=0.42748 * R ** 2 * TC_H2 ** 2 / PC_H2,
        c12=0.480 + 1.574 * W_H2 - 0.176 * W_H2 ** 2,
        Mh=M_H2, pol2=POL_H2,
        Ak12=AK12_H2, Bk12=BK12_H2, Ck12=CK12_H2, k23=float(k23),
    )
    for ns in NAMESPACES:
        for name, v in vals.items():
            if hasattr(ns, name):
                setattr(ns, name, v)
    U = u_hs(float(T), du_ref=du_ref)
    prm.A_U2i[:] = 0.0                       # shared ndarray -> all ns
    prm.B_U2i[:] = np.array([U, U, 0.0, 0.0, 0.0, 0.0])
    flash_mod._TC[1] = TC_H2
    flash_mod._PC_BAR[1] = PC_H2 / 1e5
    flash_mod._OMEGA[1] = W_H2


def restore_ch4():
    """Restore the production CH4 slot exactly (bit-identical)."""
    for ns in NAMESPACES:
        for name, v in _ORIG.items():
            if hasattr(ns, name):
                setattr(ns, name, v)
    prm.A_U2i[:] = _ORIG_A_U2i
    prm.B_U2i[:] = _ORIG_B_U2i
    flash_mod._TC[1], flash_mod._PC_BAR[1], flash_mod._OMEGA[1] = _ORIG_WILSON


def assert_nacl_only(mi):
    """NaCl-only guard: Coelho 2025 has NO H2-K/Ca/Mg/SO4 energies."""
    mi = np.asarray(mi, dtype=float)
    if np.any(mi[2:] != 0.0):
        raise ValueError(
            "H2 slot is NaCl-only: no H2-ion energies exist for "
            "K+/Ca2+/Mg2+/SO4-- in Coelho 2025 (scoping README sec. 1.3); "
            f"got mi={mi.tolist()}")


# ---------------------------------------------------------------------------
# Prediction machinery
# ---------------------------------------------------------------------------
def _flash_once(T, P_bar, mi, zg):
    """One stability_flash probe. Returns (x1w, x2w, y_w, y_h2, beta) or
    None."""
    z = np.array([1.0 - zg, zg, 0.0])
    try:
        res = stability_flash(float(T), float(P_bar), z, mi)
    except Exception:
        return None
    if res.phases == 2 and res.flash_result is not None:
        fr = res.flash_result
        return (float(fr.x_aq[0]), float(fr.x_aq[1]),
                float(fr.y_gas[0]), float(fr.y_gas[1]), float(fr.beta))
    return None


def _predict_point(task):
    """(du_ref, k23, T, P_bar, m_NaCl) -> (x_H2_aq, y_H2O_gas, m_H2_molkg,
    rescued_flag) or NaNs.

    DEGENERATE-FLASH GUARD (documented production artifact, see memory
    feedback_degenerate_flash_artifact): in the steam-rich / low-P-over-
    Psat regime, stability_flash can return a beta-boundary pseudo-
    solution whose aqueous H2 content scales with the FEED (n_iter_ssi=0,
    K echoed from the TPD trial). In a binary at fixed (T,P) the two-
    phase compositions are unique, so we accept a ladder result only if
    a SECOND feed reproduces x2w within FEED_INDEP_RTOL and beta is
    interior. Otherwise: near-dew rescue -- as zg -> y_H2 the gas equals
    the feed exactly and the reported incipient aqueous is the TPD-
    quality dew solution, so probes at zg = 0.95*y_H2 and 0.99*y_H2
    extrapolated linearly in (y_H2 - zg) -> 0 recover the true unique
    solution (validated: 473K/36bar rescued 3.7e-5 -> 5.73e-4 vs exp
    5.74e-4). x_aq is on the ion-inclusive dissociated basis; molality
    m_H2 = x2w/(x1w*Mw) is basis-free."""
    du_ref, k23, T, P_bar, m = task
    try:
        mi = np.array([m, m, 0.0, 0.0, 0.0, 0.0])
        assert_nacl_only(mi)
        rebind_h2(T, du_ref=du_ref, k23=k23)
        probes = []
        for zg in Z_GAS_RETRIES:
            r = _flash_once(T, P_bar, mi, zg)
            if r is not None:
                probes.append(r)
            if len(probes) == 2:
                break
        if not probes:
            return (float("nan"),) * 3 + (0.0,)
        if (len(probes) == 2
                and probes[0][4] < BETA_INTERIOR
                and probes[1][4] < BETA_INTERIOR
                and abs(probes[0][1] / probes[1][1] - 1.0) < FEED_INDEP_RTOL):
            x1w, x2w, y_w, _, _ = probes[0]
            return (x2w, y_w, x2w / (x1w * prm.Mw), 0.0)
        # near-dew rescue
        y_est = probes[0][3]
        r1 = _flash_once(T, P_bar, mi, 0.95 * y_est)
        r2 = _flash_once(T, P_bar, mi, 0.99 * y_est)
        if r1 is not None and r2 is not None:
            d1 = r1[3] - 0.95 * y_est
            d2 = r2[3] - 0.99 * y_est
            if abs(d1 - d2) > 1e-12:
                w = d2 / (d1 - d2)
                x2w = r2[1] + (r2[1] - r1[1]) * w
                y_w = r2[2] + (r2[2] - r1[2]) * w
            else:
                x2w, y_w = r2[1], r2[2]
            x1w = r2[0]
            return (x2w, y_w, x2w / (x1w * prm.Mw), 1.0)
        if r2 is not None or r1 is not None:
            x1w, x2w, y_w, _, _ = r2 if r2 is not None else r1
            return (x2w, y_w, x2w / (x1w * prm.Mw), 1.0)
        return (float("nan"),) * 3 + (1.0,)
    finally:
        restore_ch4()


def _predict_ternary(task):
    """(k23, T, P_bar, m_NaCl, yc_feed) -> dict with our equilibrium
    water-free gas CO2 fraction Yc_fw and aqueous molalities mh, mc.
    Feed: z_gas = 0.30 split (1-yc_feed) H2 / yc_feed CO2."""
    k23, T, P_bar, m, ycf = task
    zg = 0.30
    try:
        mi = np.array([m, m, 0.0, 0.0, 0.0, 0.0])
        assert_nacl_only(mi)
        rebind_h2(T, k23=k23)
        z = np.array([1.0 - zg, zg * (1.0 - ycf), zg * ycf])
        try:
            res = stability_flash(float(T), float(P_bar), z, mi)
        except Exception:
            res = None
        if res is not None and res.phases == 2 and res.flash_result is not None:
            fr = res.flash_result
            x1w, x2w, x3w = fr.x_aq
            yw, yh, yc = fr.y_gas
            return dict(T_K=T, P_bar=P_bar, ms=m, yc_feed=ycf, k23=k23,
                        Yc_fw=float(yc / (yc + yh)),
                        mh=float(x2w / (x1w * prm.Mw)),
                        mc=float(x3w / (x1w * prm.Mw)),
                        y_H2O=float(yw), ok=True)
        return dict(T_K=T, P_bar=P_bar, ms=m, yc_feed=ycf, k23=k23, ok=False)
    finally:
        restore_ch4()


_POOL = None


def _pool_map(fn, tasks):
    if _POOL is not None:
        return _POOL.map(fn, tasks)
    return [fn(t) for t in tasks]


def predict_df(df, du_ref=DU_REF):
    """Predict every row of a dataframe with T_K, P_bar[, m_NaCl_molkg]."""
    m = df["m_NaCl_molkg"] if "m_NaCl_molkg" in df.columns else 0.0 * df.T_K
    tasks = [(du_ref, K23_DISK, r[0], r[1], r[2])
             for r in zip(df.T_K, df.P_bar, m)]
    out = _pool_map(_predict_point, tasks)
    return (np.array([o[0] for o in out]), np.array([o[1] for o in out]),
            np.array([o[2] for o in out]), np.array([o[3] for o in out]))


def aare(pred, exp):
    """AARE% + bias% + convergence counts over one subset."""
    pred, exp = np.asarray(pred, float), np.asarray(exp, float)
    keep = np.isfinite(exp)
    pred, exp = pred[keep], exp[keep]
    ok = np.isfinite(pred)
    return {
        "aare_pct": float(np.mean(np.abs(pred[ok] / exp[ok] - 1.0)) * 100.0)
        if ok.any() else float("nan"),
        "bias_pct": float(np.mean(pred[ok] / exp[ok] - 1.0) * 100.0)
        if ok.any() else float("nan"),
        "n_converged": int(ok.sum()), "n_points": int(keep.sum()),
    }


T_BINS = ((273.0, 348.1, "273-348K"), (348.1, 448.1, "349-448K"),
          (448.1, 590.0, "449-589K"))


def water_report(df, x_pred, y_pred):
    rep = {"x_all": aare(x_pred, df.x_H2_aq),
           "y_all": aare(y_pred, df.y_H2O_gas)}
    for lo, hi, lab in T_BINS:
        m = (df.T_K >= lo) & (df.T_K < hi)
        rep[f"x_{lab}"] = aare(x_pred[m], df.x_H2_aq[m])
    for a in sorted(df.author.unique()):
        m = (df.author == a).to_numpy()
        if np.isfinite(df.x_H2_aq[m]).any():
            rep[f"x_by_author/{a}"] = aare(x_pred[m], df.x_H2_aq[m])
    return rep


def brine_report(df, m_pred):
    rep = {"all": aare(m_pred, df.m_H2_molkg)}
    for a in sorted(df.author.unique()):
        m = (df.author == a).to_numpy()
        rep[f"by_author/{a}"] = aare(m_pred[m], df.m_H2_molkg[m])
    for T in sorted(df.T_K.unique()):
        m = np.isclose(df.T_K, T)
        rep[f"by_T/{T:.0f}K"] = aare(m_pred[m], df.m_H2_molkg[m])
    for ms in sorted(df.m_NaCl_molkg.unique()):
        m = np.isclose(df.m_NaCl_molkg, ms)
        rep[f"by_ms/{ms}m"] = aare(m_pred[m], df.m_H2_molkg[m])
    return rep


# ---------------------------------------------------------------------------
# Guard spot flashes (slot hygiene; identical cases to the N2 pilot)
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
def load_water():
    import pandas as pd
    df = pd.read_csv(SCOPING / "h2_water_binaries.csv")
    assert len(df) == 216, f"expected 216 rows, got {len(df)}"
    return df


def load_brine():
    import pandas as pd
    df = pd.read_csv(SCOPING / "h2_nacl_brine.csv")
    assert len(df) == 71, f"expected 71 rows, got {len(df)}"
    return df


def load_hcb_curve(T, P, ms):
    """F. M. Coelho's on-disk HCB ternary model curve (his eCPA_ELV.py output).
    Columns: Yc_fw xw_W xh_W xc_W mh mc xw_H xh_H xc_H ..."""
    import pandas as pd
    tag = f"{ms:g}m"
    f = HCB / f"T{T:.0f}K" / f"P{P:.0f}b" / f"ELV_CHW_T{T:.0f}K_P{P:.0f}b_ms{tag}.dat"
    if not f.exists():
        return None
    df = pd.read_csv(f, sep=r"\s+", skiprows=1, header=None,
                     usecols=[0, 4, 5], names=["Yc_fw", "mh", "mc"])
    return df.sort_values("Yc_fw").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def _state():
    return json.loads(STATE_JSON.read_text()) if STATE_JSON.exists() else {}


def _save_state(st):
    STATE_JSON.write_text(json.dumps(st, indent=2))


def stage_refs(st):
    st["guard_refs"] = {c: guard_spot(c) for c in GUARD_CASES}
    print("guard refs (unpatched, pre-pilot):")
    for c, v in st["guard_refs"].items():
        print(f"  {c}: beta={v[0]:.12g} x_aq={v[1:4]}")


def stage_v1water(st):
    df = load_water()
    x_pred, y_pred, _, resc = predict_df(df)
    st["V1_water"] = {
        "report": water_report(df, x_pred, y_pred),
        "x_pred": [float(v) for v in x_pred],
        "y_pred": [float(v) for v in y_pred],
        "n_rescued": int(np.nansum(resc)),
        "rescued": [bool(v) for v in resc],
        "coelho2025_aad": {"x": COELHO2025_AAD["water_x"], "y": COELHO2025_AAD["water_y"]},
    }
    print(f"near-dew rescued points: {int(np.nansum(resc))}/{len(df)}")
    print(json.dumps({k: v for k, v in st["V1_water"]["report"].items()
                      if not k.startswith("x_by")}, indent=2))


def stage_v1brine(st):
    df = load_brine()
    _, _, m_pred, resc = predict_df(df)
    st["V1_brine"] = {
        "report": brine_report(df, m_pred),
        "m_pred": [float(v) for v in m_pred],
        "n_rescued": int(np.nansum(resc)),
        "coelho2025_aad": COELHO2025_AAD["brine"],
    }
    print(f"near-dew rescued points: {int(np.nansum(resc))}/{len(df)}")
    print(json.dumps(st["V1_brine"]["report"], indent=2))
    # NaCl-only guard demonstration: a KCl feed must be refused
    try:
        rebind_h2(323.0)
        assert_nacl_only([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        guard_msg = "GUARD FAILED: KCl feed was accepted"
        ok = False
    except ValueError as e:
        guard_msg = str(e)
        ok = True
    finally:
        restore_ch4()
    st["nacl_only_guard"] = {"refused_kcl_feed": ok, "message": guard_msg}
    print("NaCl-only guard:", ok, "-", guard_msg[:90])
    a = st["V1_brine"]["report"]["all"]["aare_pct"]
    print(f"\nV1 brine AARE {a:.2f}% vs F. M. Coelho 8.64% -> "
          f"V2 {'TRIGGERED' if a > 12.0 else 'not needed'} (gate ~12%)")


NAN_RESIDUAL = 5.0


def stage_v2fit(st):
    from scipy.optimize import least_squares
    df = load_brine()
    exp = df.m_H2_molkg.to_numpy()

    def resid(x):
        _, _, pred, _ = predict_df(df, du_ref=float(x[0]))
        r = pred / exp - 1.0
        return np.where(np.isfinite(r), r, NAN_RESIDUAL)

    runs, best = [], None
    for x0 in ([DU_REF], [7000.0], [13000.0]):
        t0 = time.perf_counter()
        res = least_squares(resid, x0, bounds=([0.0], [30000.0]),
                            diff_step=1e-3, method="trf")
        run = {"start": [float(x0[0])], "x": [float(res.x[0])],
               "cost": float(res.cost), "status": int(res.status),
               "nfev": int(res.nfev),
               "rms_rel_resid_pct": float(np.sqrt(np.mean(res.fun ** 2)) * 100),
               "wall_s": round(time.perf_counter() - t0, 1)}
        runs.append(run)
        print(f"  [v2 start {x0[0]:.0f}] -> DU_ref={run['x'][0]:.2f} "
              f"cost={run['cost']:.6g} nfev={run['nfev']} ({run['wall_s']}s)",
              flush=True)
        if best is None or run["cost"] < best["cost"]:
            best = run
    st["V2_fit"] = {
        "best": best, "runs": runs, "n_brine_points": len(df),
        "parameterization": "refit DU_ref ONLY (alpha, Talfa kept at "
                            "F. M. Coelho's -9889.08 K / 364.487 K); shared "
                            "Na+=Cl- energy, NaCl-only data cannot split "
                            "the pair (same degeneracy as the N2 pilot)",
        "coelho2025_du_ref": DU_REF,
        "shift_J_mol": best["x"][0] - DU_REF,
    }


def stage_v2eval(st):
    df = load_brine()
    du = st["V2_fit"]["best"]["x"][0]
    _, _, m_pred, _ = predict_df(df, du_ref=du)
    st["V2_brine"] = {"du_ref": du,
                      "report": brine_report(df, m_pred),
                      "m_pred": [float(v) for v in m_pred]}
    print(json.dumps(st["V2_brine"]["report"], indent=2))


V3_CURVES = [(323.0, 200.0, 1), (323.0, 200.0, 4),
             (323.0, 400.0, 1), (323.0, 400.0, 4),
             (423.0, 200.0, 1), (423.0, 200.0, 4)]
V3_YCF = (0.05, 0.20, 0.40, 0.60, 0.80, 0.95)
V3_SENS_PTS = [(323.0, 200.0, 1, 0.5), (323.0, 200.0, 4, 0.5),
               (423.0, 200.0, 1, 0.5)]


def stage_v3(st):
    tasks = [(K23_DISK, T, P, float(ms), ycf)
             for (T, P, ms) in V3_CURVES for ycf in V3_YCF]
    out = _pool_map(_predict_ternary, tasks)
    comp = []
    for (T, P, ms) in V3_CURVES:
        fel = load_hcb_curve(T, P, ms)
        pts = [o for o in out
               if o["ok"] and o["T_K"] == T and o["P_bar"] == P
               and o["ms"] == ms]
        if fel is None:
            comp.append({"T_K": T, "P_bar": P, "ms": ms,
                         "note": "no F. M. Coelho .dat on disk"})
            continue
        rows = []
        for o in pts:
            mh_f = float(np.interp(o["Yc_fw"], fel.Yc_fw, fel.mh))
            mc_f = float(np.interp(o["Yc_fw"], fel.Yc_fw, fel.mc))
            rows.append({**{k: o[k] for k in
                            ("yc_feed", "Yc_fw", "mh", "mc")},
                         "mh_coelho2025": mh_f, "mc_coelho2025": mc_f,
                         "dmh_pct": (o["mh"] / mh_f - 1.0) * 100.0,
                         "dmc_pct": (o["mc"] / mc_f - 1.0) * 100.0})
        dmh = [abs(r["dmh_pct"]) for r in rows]
        dmc = [abs(r["dmc_pct"]) for r in rows]
        comp.append({"T_K": T, "P_bar": P, "ms": ms,
                     "n_ok": len(rows), "n_tried": len(V3_YCF),
                     "mean_abs_dmh_pct": float(np.mean(dmh)) if dmh else None,
                     "max_abs_dmh_pct": float(np.max(dmh)) if dmh else None,
                     "mean_abs_dmc_pct": float(np.mean(dmc)) if dmc else None,
                     "max_abs_dmc_pct": float(np.max(dmc)) if dmc else None,
                     "points": rows})
        print(f"  V3 T={T:.0f} P={P:.0f} ms={ms}: "
              f"|dmh| mean {comp[-1]['mean_abs_dmh_pct']}, "
              f"|dmc| mean {comp[-1]['mean_abs_dmc_pct']} ({len(rows)} pts)",
              flush=True)
    st["V3"] = {"k23_used": K23_DISK, "zg_feed": 0.30, "curves": comp}
    # k23 sensitivity: disk 0.1277 vs paper 0.145 at 3 points
    sens = []
    for (T, P, ms, ycf) in V3_SENS_PTS:
        a = _predict_ternary((K23_DISK, T, P, float(ms), ycf))
        b = _predict_ternary((K23_PAPER, T, P, float(ms), ycf))
        row = {"T_K": T, "P_bar": P, "ms": ms, "yc_feed": ycf,
               "ok": a["ok"] and b["ok"]}
        if row["ok"]:
            row.update(mh_0p1277=a["mh"], mh_0p145=b["mh"],
                       mc_0p1277=a["mc"], mc_0p145=b["mc"],
                       dmh_pct=(b["mh"] / a["mh"] - 1.0) * 100.0,
                       dmc_pct=(b["mc"] / a["mc"] - 1.0) * 100.0)
            print(f"  k23 sens T={T:.0f} P={P:.0f} ms={ms}: "
                  f"dmh={row['dmh_pct']:+.3f}% dmc={row['dmc_pct']:+.3f}%")
        sens.append(row)
    st["V3"]["k23_sensitivity"] = sens


FIG_TS = (298.0, 323.0, 373.0)
FIG_MS = (0.0, 1.0, 4.0)


def stage_curves(st):
    import pandas as pd
    P_grid = np.geomspace(20.0, 500.0, 14)
    variants = [("V1", DU_REF)]
    if "V2_fit" in st:
        variants.append(("V2", st["V2_fit"]["best"]["x"][0]))
    rows = []
    for T in FIG_TS:
        for ms in FIG_MS:
            for name, du in variants:
                tasks = [(du, K23_DISK, T, float(p), ms) for p in P_grid]
                out = _pool_map(_predict_point, tasks)
                for p, o in zip(P_grid, out):
                    rows.append(dict(T_K=T, ms=ms, variant=name,
                                     P_bar=p, m_H2=o[2]))
            print(f"  curves T={T} ms={ms} done", flush=True)
    pd.DataFrame(rows).to_csv(CURVES_CSV, index=False)
    st["curves_csv"] = str(CURVES_CSV)
    st["curve_variants"] = [v[0] for v in variants]


def stage_figure(st):
    import pandas as pd
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.size": 14, "axes.labelsize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
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
    STYLE = {"V1": (WONG[5], "-"), "V2": (WONG[6], "--")}
    LABEL = {"V1": "V1 verbatim transfer",
             "V2": r"V2 refit $\Delta U_{\mathrm{ref}}$"}
    MARK = ["o", "s", "^", "D", "v"]

    wdf = load_water()
    bdf = load_brine()
    cur = pd.read_csv(CURVES_CSV)
    mlab = {0.0: "H$_2$O", 1.0: "1 m NaCl", 4.0: "4 m NaCl"}

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11.5), sharex=True)
    for i, ms in enumerate(FIG_MS):
        for j, T in enumerate(FIG_TS):
            ax = axes[i, j]
            if ms == 0.0:
                sub = wdf[np.isclose(wdf.T_K, T) & wdf.x_H2_aq.notna()
                          & (wdf.P_bar <= 510.0)]
                Pd = sub.P_bar.to_numpy()
                md = (sub.x_H2_aq / ((1.0 - sub.x_H2_aq) * prm.Mw)).to_numpy()
            else:
                sub = bdf[np.isclose(bdf.T_K, T)
                          & np.isclose(bdf.m_NaCl_molkg, ms)
                          & (bdf.P_bar <= 510.0)]
                Pd = sub.P_bar.to_numpy()
                md = sub.m_H2_molkg.to_numpy()
            ax.plot(Pd, md, "o", color="black",
                    mfc="white", mew=1.8, zorder=5, linestyle="none",
                    label="exp.")
            for name in st["curve_variants"]:
                c = cur[(cur.variant == name) & np.isclose(cur.T_K, T)
                        & np.isclose(cur.ms, ms)].sort_values("P_bar")
                col, ls = STYLE[name]
                ax.plot(c.P_bar, c.m_H2, ls, color=col, label=LABEL[name])
            ax.text(0.05, 0.95, f"{mlab[ms]}\n{T:.0f} K",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=12, fontweight="bold", color="black",
                    bbox=dict(facecolor="white", alpha=0.85,
                              edgecolor="none", pad=2))
            ax.set_xlim(0, 510)
            if i == 2:
                ax.set_xlabel("P [bar]")
            if j == 0:
                ax.set_ylabel(r"$m_{\mathrm{H_2}}$ [mol/kg]")
            hnd, lbl = ax.get_legend_handles_labels()
            if lbl:
                ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_PNG)
    print("figure ->", FIG_PNG)
    st["figure"] = str(FIG_PNG.resolve().relative_to(Path(__file__).resolve().parents[3]))


def stage_finalize(st):
    # In-process guard: rebind -> flash -> restore, then spot-flash and
    # compare against the pre-pilot refs from stage `refs`.
    pre = {c: guard_spot(c) for c in GUARD_CASES}
    try:
        rebind_h2(323.0)
        _ = _predict_point((DU_REF, K23_DISK, 323.0, 200.0, 1.0))
    finally:
        restore_ch4()
    post = {c: guard_spot(c) for c in GUARD_CASES}
    guards = {}
    for c in GUARD_CASES:
        guards[c] = {
            "in_process_bit_identical": tuples_bit_identical(pre[c], post[c]),
            "vs_pre_pilot_refs_bit_identical":
                tuples_bit_identical(st["guard_refs"][c], post[c]),
        }
    st["guards"] = guards
    print(json.dumps(guards, indent=2))

    wdf, bdf = load_water(), load_brine()
    final = {
        "pilot": "h2_pilot_2026_07",
        "parameterization_source":
            "Coelho, Mercier Franco, Firoozabadi, IJHE 178 (2025) 151361 "
            "(group's published H2 e-CPA; extraction: "
            "pilot_studies/h2_scoping_2026_07/README.md)",
        "h2_constants": {"Tc_K": TC_H2, "Pc_Pa": PC_H2, "omega": W_H2,
                         "M_kg_mol": M_H2, "pol2_Cm2_J": POL_H2,
                         "association_sites": 0},
        "kij_h2_h2o": {"Ak12": AK12_H2, "Bk12": BK12_H2, "Ck12": CK12_H2,
                       "note": "OPT4; exact transfer (his water CPA set is "
                               "bit-identical to production)"},
        "du_hs": {"DU_ref_J_mol": DU_REF, "alpha_K": ALPHA_K,
                  "Talfa_K": TALFA, "Tref_K": TREF,
                  "form": "quadratic Talfa; represented EXACTLY per point "
                          "via A_U2i=0, B_U2i=u_hs(T) rebind",
                  "basis_caveat": "fitted in Maribo 2015 salt-specific NaCl "
                                  "basis; used verbatim in our ion-specific "
                                  "basis (V1) -> transfer prediction"},
        "coelho2025_reference_aad_pct": COELHO2025_AAD,
        "V1_water": {"report": st["V1_water"]["report"]},
        "V1_brine": {"report": st["V1_brine"]["report"]},
        "nacl_only_guard": st["nacl_only_guard"],
        "guards": guards,
        "V3": st["V3"],
        "predictions_water": {
            "columns": ["T_K", "P_bar", "x_exp", "x_V1", "y_exp", "y_V1",
                        "author"],
            "rows": [[float(r.T_K), float(r.P_bar),
                      None if np.isnan(r.x_H2_aq) else float(r.x_H2_aq),
                      st["V1_water"]["x_pred"][i],
                      None if np.isnan(r.y_H2O_gas) else float(r.y_H2O_gas),
                      st["V1_water"]["y_pred"][i], r.author]
                     for i, r in enumerate(wdf.itertuples())],
        },
        "predictions_brine": {
            "columns": ["T_K", "P_bar", "m_NaCl", "m_exp", "m_V1"]
            + (["m_V2"] if "V2_brine" in st else []) + ["author"],
            "rows": [[float(r.T_K), float(r.P_bar), float(r.m_NaCl_molkg),
                      float(r.m_H2_molkg), st["V1_brine"]["m_pred"][i]]
                     + ([st["V2_brine"]["m_pred"][i]] if "V2_brine" in st
                        else []) + [r.author]
                     for i, r in enumerate(bdf.itertuples())],
        },
        "notes": [
            "Runtime rebinding only; ms_ecpa/ and tests/ untouched; "
            "restore in finally blocks; per-point A=0/B=U(T) rebind "
            "represents the quadratic U_hs(T) exactly.",
            "NaCl-only: no H2-K/Ca/Mg/SO4 energies exist in Coelho 2025; "
            "guard refuses such feeds.",
            "Peneloux differs (his constant salt shift vs our per-ion "
            "T-linear); volume translation does not affect VLE "
            "compositions, only densities.",
            "V3 is model-to-model: no experimental H2-CO2-brine ternary "
            "data exists on disk; targets are F. M. Coelho's own HCB .dat "
            "model curves, parameterized by the water-free gas CO2 "
            "fraction Yc_fw.",
        ],
    }
    if "V2_fit" in st:
        final["V2_fit"] = {k: v for k, v in st["V2_fit"].items()}
        final["V2_brine"] = {"du_ref": st["V2_brine"]["du_ref"],
                             "report": st["V2_brine"]["report"]}
    RESULTS_JSON.write_text(json.dumps(final, indent=2))
    print("\nfinal ->", RESULTS_JSON)

    print("\nHeadline table (AARE %):")
    print(f"  water x_H2 (163 obs): V1 "
          f"{st['V1_water']['report']['x_all']['aare_pct']:.2f} "
          f"vs F. M. Coelho {COELHO2025_AAD['water_x']}")
    print(f"  water y_H2O (78 obs): V1 "
          f"{st['V1_water']['report']['y_all']['aare_pct']:.2f} "
          f"vs F. M. Coelho {COELHO2025_AAD['water_y']}")
    line = (f"  brine m_H2 (71 pts):  V1 "
            f"{st['V1_brine']['report']['all']['aare_pct']:.2f}")
    if "V2_brine" in st:
        line += (f", V2 {st['V2_brine']['report']['all']['aare_pct']:.2f} "
                 f"(DU_ref {DU_REF:.0f} -> "
                 f"{st['V2_brine']['du_ref']:.0f} J/mol)")
    print(line + f" vs F. M. Coelho {COELHO2025_AAD['brine']}")


STAGES = {
    "refs": stage_refs, "v1water": stage_v1water, "v1brine": stage_v1brine,
    "v2fit": stage_v2fit, "v2eval": stage_v2eval, "v3": stage_v3,
    "curves": stage_curves, "figure": stage_figure,
    "finalize": stage_finalize,
}
POOL_STAGES = {"v1water", "v1brine", "v2fit", "v2eval", "v3", "curves"}


def main():
    global _POOL
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    args = ap.parse_args()
    st = _state()
    if args.stage in POOL_STAGES and _POOL is None:
        _POOL = mp.get_context("spawn").Pool(N_WORKERS)
    t0 = time.perf_counter()
    print(f"=== stage {args.stage} ===", flush=True)
    STAGES[args.stage](st)
    _save_state(st)
    print(f"=== stage {args.stage} done ({time.perf_counter()-t0:.1f}s) ===",
          flush=True)
    if _POOL is not None:
        _POOL.close()
        _POOL.join()


if __name__ == "__main__":
    main()
