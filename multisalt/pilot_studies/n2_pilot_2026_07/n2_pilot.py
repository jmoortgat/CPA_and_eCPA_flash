"""N2-brine pilot: can the multi-salt eCPA describe N2 solubility in NaCl
brines by zero-tuning transfer from the CH4 slot, vs explicit fitting?

Data: O'Sullivan & Smith 1970 Table I (51 N2 points: 18 H2O, 18 in 1.000 m
NaCl, 15 in 4.00 m NaCl; 324.65-398.15 K, 100-608 atm), extracted at
pilot_studies/n2_scoping_2026_07/n2_nacl_osullivan1970.csv.

Three framings (identical prediction machinery: stability_flash,
z = (H2O, N2-in-slot-2, 0), z_gas retry ladder (0.05, 0.10, 0.02, 0.20,
0.30) as in the established pilots):

  B0  zero-fit:     Maribo-Mogensen 2014 thesis Table 7.1 water-N2
                    kij(T) = 1.0741 - 368.3/T (calibrated for HIS CPA, not
                    ours) re-expressed in our quadratic-in-(T/Tc1) form
                    (exact mapping over 315-410 K, NOT a fit to any N2
                    datum) + CH4-ion energies transferred with the
                    function-preserving Tc rescale
                    A_N2,i = A_CH4,i * Tc2(N2)/Tc2(CH4) = 0.66222 * A_CH4,i
                    (B unchanged), because U2i = A_U2i*(T/Tc2) + B_U2i.
  B   recommended:  fit our kij(T) 3 coefficients (Ak12, Bk12, Ck12) to the
                    18 SALT-FREE water points only; ion energies still
                    transferred. All 33 brine points remain a
                    zero-brine-tuning prediction.
  A   full fit:     kij from B + ONE scaling factor s on the transferred
                    (Na+, Cl-) energy pair, fitted to the 33 brine points.
                    NaCl-only data cannot split Na+ from Cl- (the scoping
                    degeneracy), so a single multiplicative factor that
                    preserves the CH4-derived Na:Cl partition and the full
                    T-shape is the honest 1-dof choice; s = 1 recovers the
                    transfer.

Slot-2 rebind (Tier-1 recipe verified by n2_scoping_2026_07): patch FOUR
module namespaces (parameters + the lnphi/eos star-import snapshots + the
flash by-name imports), plus the flash Wilson arrays _TC/_PC_BAR/_OMEGA
in-place. A_U2i/B_U2i are the SAME ndarray object in all namespaces (star
import binds the reference), so in-place [:] assignment propagates.
Runtime rebinding only; restore in finally blocks; ms_ecpa/ untouched.

Run stages (each is a separate foreground process; state accumulates in
work_state.json):

    python n2_pilot.py --stage refs      # unpatched CH4/CO2 guard spot flashes
    python n2_pilot.py --stage b0        # kij mapping + B0 predictions
    python n2_pilot.py --stage fitB      # fit kij to 18 water points (3 starts)
    python n2_pilot.py --stage evalB     # B predictions on all 51 points
    python n2_pilot.py --stage fitA      # fit s to 33 brine points (3 starts)
    python n2_pilot.py --stage evalA     # A predictions + pol2 sensitivity
    python n2_pilot.py --stage curves    # smooth curves for the figure
    python n2_pilot.py --stage figure    # house-style figure
    python n2_pilot.py --stage finalize  # guards + n2_results.json + table

N2 constants: Tc = 126.192 K, Pc = 33.958 bar, omega = 0.0372,
M = 0.0280134 kg/mol. Polarizability alpha(N2) = 1.7403 A^3 = 1.9360e-40
C m^2/J -- handbook value, citation to be added (no N2 polarizability in
any on-disk PDF; it only enters eps_inf via Clausius-Mossotti, sensitivity
quantified in stage evalA).
"""
from __future__ import annotations

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
CSV = CODE / "pilot_studies" / "n2_scoping_2026_07" / "n2_nacl_osullivan1970.csv"
STATE_JSON = HERE / "work_state.json"
RESULTS_JSON = HERE / "n2_results.json"
CURVES_CSV = HERE / "curves_cache.csv"
FIG_PNG = HERE / "fig_n2_brine_pilot.png"

# ---------------------------------------------------------------------------
# N2 constants
# ---------------------------------------------------------------------------
TC_N2 = 126.192          # K
PC_N2 = 33.958e5         # Pa
W_N2 = 0.0372
M_N2 = 0.0280134         # kg/mol
POL_N2 = 1.9360e-40      # C m^2/J; handbook alpha=1.7403 A^3, citation to be added
POL_CH4 = 2.6155e-40     # production CH4 value (for the sensitivity test)

TC_RATIO = TC_N2 / 190.56        # 0.66222, function-preserving DU(T) transfer
ATM_TO_BAR = 1.01325
Z_GAS_RETRIES = (0.05, 0.10, 0.02, 0.20, 0.30)
N_WORKERS = 8

# ---------------------------------------------------------------------------
# Rebind machinery (Tier-1 recipe)
# ---------------------------------------------------------------------------
NAMESPACES = (prm, lnphi_mod, eos_mod, flash_mod)
SCALARS = ("Tc2", "Pc2", "w2", "b2", "a02", "c12", "Mh", "pol2",
           "Ak12", "Bk12", "Ck12", "k23")

_ORIG = {name: getattr(prm, name) for name in SCALARS}
_ORIG_A_U2i = prm.A_U2i.copy()
_ORIG_B_U2i = prm.B_U2i.copy()
_ORIG_WILSON = (float(flash_mod._TC[1]), float(flash_mod._PC_BAR[1]),
                float(flash_mod._OMEGA[1]))


def rebind_n2(Ak, Bk, Ck, s_nacl=1.0, pol2=POL_N2):
    """Swap slot 2 CH4 -> N2 in all four consuming namespaces.

    kij coefficients (Ak, Bk, Ck) are in OUR form
    k12 = Ak*(T/Tc1)^2 + Bk*(T/Tc1) + Ck  (T-scaled by Tc1 = water, so the
    functional form survives the Tc2 swap). Ion energies: function-
    preserving transfer A*TC_RATIO (B unchanged), then the (Na, Cl) pair
    scaled jointly by s_nacl (framing A; s_nacl = 1 -> pure transfer).
    k23 (N2-CO2) is inert here (z_CO2 = 0 throughout); set to 0.0, a
    literature value is only needed for future ternaries.
    """
    R = prm.R
    vals = dict(
        Tc2=TC_N2, Pc2=PC_N2, w2=W_N2,
        b2=0.08664 * R * TC_N2 / PC_N2,
        a02=0.42748 * R ** 2 * TC_N2 ** 2 / PC_N2,
        c12=0.480 + 1.574 * W_N2 - 0.176 * W_N2 ** 2,
        Mh=M_N2, pol2=pol2,
        Ak12=float(Ak), Bk12=float(Bk), Ck12=float(Ck), k23=0.0,
    )
    for ns in NAMESPACES:
        for name, v in vals.items():
            if hasattr(ns, name):
                setattr(ns, name, v)
    A_new = _ORIG_A_U2i * TC_RATIO
    B_new = _ORIG_B_U2i.copy()
    A_new[0] *= s_nacl; B_new[0] *= s_nacl        # Na+
    A_new[1] *= s_nacl; B_new[1] *= s_nacl        # Cl-
    prm.A_U2i[:] = A_new                          # shared ndarray -> all ns
    prm.B_U2i[:] = B_new
    flash_mod._TC[1] = TC_N2
    flash_mod._PC_BAR[1] = PC_N2 / 1e5
    flash_mod._OMEGA[1] = W_N2


def restore_ch4():
    """Restore the production CH4 slot exactly (bit-identical)."""
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
def _predict_point(task):
    """(Ak, Bk, Ck, s_nacl, pol2, T_K, P_bar, m_NaCl) -> X2 on the
    O'Sullivan basis (NaCl undissociated), or NaN.

    Model x_aq = (x1w, x2w, x3w) is on the ion-inclusive dissociated basis
    (x1w + x2w + x3w + sum xiw = 1, xiw = Mw*x1w*mi). O'Sullivan's X2
    treats NaCl as ONE undissociated component:
        X2 = x2w / (x1w + x2w + Mw*x1w*m_NaCl).
    """
    Ak, Bk, Ck, s_nacl, pol2, T, P_bar, m = task
    try:
        rebind_n2(Ak, Bk, Ck, s_nacl=s_nacl, pol2=pol2)
        mi = np.array([m, m, 0.0, 0.0, 0.0, 0.0])
        for zg in Z_GAS_RETRIES:
            z = np.array([1.0 - zg, zg, 0.0])
            try:
                res = stability_flash(float(T), float(P_bar), z, mi)
            except Exception:
                continue
            if res.phases == 2 and res.flash_result is not None:
                x1w, x2w, _ = res.flash_result.x_aq
                return float(x2w / (x1w + x2w + prm.Mw * x1w * m))
        return float("nan")
    finally:
        restore_ch4()


_POOL = None


def predict_all(df, Ak, Bk, Ck, s_nacl=1.0, pol2=POL_N2):
    tasks = [(Ak, Bk, Ck, s_nacl, pol2, r.T_K, r.P_atm * ATM_TO_BAR,
              r.m_NaCl_mol_kg) for r in df.itertuples()]
    if _POOL is not None:
        out = _POOL.map(_predict_point, tasks)
    else:
        out = [_predict_point(t) for t in tasks]
    return np.array(out)


def aare_by_subset(df, pred):
    """AARE% per molality subset + overall; counts of converged points."""
    out = {}
    exp = df["x_molefrac"].to_numpy()
    for label, mask in [("water", df.m_NaCl_mol_kg == 0.0),
                        ("1m", np.isclose(df.m_NaCl_mol_kg, 1.0)),
                        ("4m", np.isclose(df.m_NaCl_mol_kg, 4.0)),
                        ("brine_all", df.m_NaCl_mol_kg > 0.0),
                        ("all", df.m_NaCl_mol_kg >= 0.0)]:
        m = np.asarray(mask)
        p, e = pred[m], exp[m]
        ok = np.isfinite(p)
        out[label] = {
            "aare_pct": float(np.mean(np.abs(p[ok] / e[ok] - 1.0)) * 100.0)
            if ok.any() else float("nan"),
            "n_converged": int(ok.sum()), "n_points": int(m.sum()),
            "bias_pct": float(np.mean(p[ok] / e[ok] - 1.0) * 100.0)
            if ok.any() else float("nan"),
        }
    return out


# ---------------------------------------------------------------------------
# Guard spot flashes (slot hygiene)
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
    """Elementwise identity, NaN == NaN allowed (K of absent species)."""
    if len(a) != len(b):
        return False
    return all((x == y) or (np.isnan(x) and np.isnan(y))
               for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_n2():
    import pandas as pd
    df = pd.read_csv(CSV, comment="#")
    df = df[df.gas == "N2"].reset_index(drop=True)
    assert len(df) == 51, f"expected 51 N2 rows, got {len(df)}"
    return df


# ---------------------------------------------------------------------------
# kij mapping and fitting
# ---------------------------------------------------------------------------
def maribo_kij_mapping():
    """Re-express Maribo thesis Table 7.1 kij(T) = 1.0741 - 368.3/T in our
    quadratic-in-(T/Tc1) form by least squares over 315-410 K (data range
    +/- margin). This is a functional re-expression, not a fit to N2 data.
    """
    Tgrid = np.linspace(315.0, 410.0, 200)
    target = 1.0741 - 368.3 / Tgrid
    Tr = Tgrid / prm.Tc1
    cf = np.polyfit(Tr, target, 2)
    dev = np.max(np.abs(np.polyval(cf, Tr) - target))
    return [float(c) for c in cf], float(dev)


def _state():
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text())
    return {}


def _save_state(st):
    STATE_JSON.write_text(json.dumps(st, indent=2))


def fit_least_squares(residual_fn, starts, bounds, label):
    """2-3 starts of scipy least_squares, diff_step=1e-3; returns best +
    per-start convergence details."""
    from scipy.optimize import least_squares
    runs = []
    best = None
    for i, x0 in enumerate(starts):
        t0 = time.perf_counter()
        res = least_squares(residual_fn, x0, bounds=bounds,
                            diff_step=1e-3, method="trf")
        cost = float(res.cost)
        run = {
            "start": [float(v) for v in np.atleast_1d(x0)],
            "x": [float(v) for v in np.atleast_1d(res.x)],
            "cost": cost, "status": int(res.status),
            "message": res.message, "nfev": int(res.nfev),
            "optimality": float(res.optimality),
            "rms_rel_resid_pct": float(np.sqrt(np.mean(res.fun ** 2)) * 100),
            "wall_s": round(time.perf_counter() - t0, 1),
        }
        runs.append(run)
        print(f"  [{label} start {i+1}] x0={run['start']} -> x={run['x']} "
              f"cost={cost:.6g} nfev={run['nfev']} ({run['wall_s']}s)",
              flush=True)
        if best is None or cost < best["cost"]:
            best = run
    return best, runs


NAN_RESIDUAL = 5.0   # finite penalty for non-converged points during fits


def make_resid_water(df_w):
    def resid(x):
        pred = predict_all(df_w, x[0], x[1], x[2], s_nacl=1.0)
        exp = df_w["x_molefrac"].to_numpy()
        r = pred / exp - 1.0
        return np.where(np.isfinite(r), r, NAN_RESIDUAL)
    return resid


def make_resid_brine(df_b, kij):
    def resid(x):
        pred = predict_all(df_b, *kij, s_nacl=float(x[0]))
        exp = df_b["x_molefrac"].to_numpy()
        r = pred / exp - 1.0
        return np.where(np.isfinite(r), r, NAN_RESIDUAL)
    return resid


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_refs(st):
    st["guard_refs"] = {c: guard_spot(c) for c in GUARD_CASES}
    print("guard refs (unpatched, pre-pilot):")
    for c, v in st["guard_refs"].items():
        print(f"  {c}: beta={v[0]:.12g} x_aq={v[1:4]}")


def stage_b0(st):
    df = load_n2()
    cf, dev = maribo_kij_mapping()
    st["kij_mapping"] = {"AkBkCk": cf, "max_mapping_dev": dev,
                         "source": "Maribo 2014 thesis Table 7.1: "
                                   "kij = 1.0741 - 368.3/T (CPA, his param set)"}
    print(f"kij mapping: Ak,Bk,Ck = {cf}, max dev {dev:.2e}")
    pred = predict_all(df, *cf, s_nacl=1.0)
    st["B0"] = {"kij": cf, "s_nacl": 1.0,
                "aare": aare_by_subset(df, pred),
                "pred": [float(p) for p in pred]}
    print(json.dumps(st["B0"]["aare"], indent=2))


def stage_fitB(st):
    df = load_n2()
    df_w = df[df.m_NaCl_mol_kg == 0.0].reset_index(drop=True)
    assert len(df_w) == 18
    seed = st["kij_mapping"]["AkBkCk"]
    ch4 = [-4.0935, 5.6634, -1.7845]        # production CH4-H2O values
    perturbed = [seed[0] * 1.15, seed[1] * 0.9, seed[2] + 0.05]
    bounds = ([-60.0, -60.0, -60.0], [60.0, 60.0, 60.0])
    best, runs = fit_least_squares(make_resid_water(df_w),
                                   [seed, ch4, perturbed], bounds, "fitB")
    st["fitB"] = {"best": best, "runs": runs,
                  "n_water_points": len(df_w),
                  "bounds": bounds, "diff_step": 1e-3}


def stage_evalB(st):
    df = load_n2()
    kij = st["fitB"]["best"]["x"]
    pred = predict_all(df, *kij, s_nacl=1.0)
    st["B"] = {"kij": kij, "s_nacl": 1.0,
               "aare": aare_by_subset(df, pred),
               "pred": [float(p) for p in pred]}
    print(json.dumps(st["B"]["aare"], indent=2))


def stage_fitA(st):
    df = load_n2()
    df_b = df[df.m_NaCl_mol_kg > 0.0].reset_index(drop=True)
    assert len(df_b) == 33
    kij = st["fitB"]["best"]["x"]
    bounds = ([0.1], [4.0])
    best, runs = fit_least_squares(make_resid_brine(df_b, kij),
                                   [[1.0], [0.6], [1.6]], bounds, "fitA")
    st["fitA"] = {"best": best, "runs": runs, "kij_from_B": kij,
                  "n_brine_points": len(df_b),
                  "bounds": bounds, "diff_step": 1e-3,
                  "parameterization":
                      "single factor s scaling the transferred (Na+, Cl-) "
                      "(A, B) energy pairs jointly; preserves CH4-derived "
                      "Na:Cl partition and T-shape; s=1 = pure transfer"}


def stage_evalA(st):
    df = load_n2()
    kij = st["fitB"]["best"]["x"]
    s = st["fitA"]["best"]["x"][0]
    pred = predict_all(df, *kij, s_nacl=s)
    st["A"] = {"kij": kij, "s_nacl": s,
               "aare": aare_by_subset(df, pred),
               "pred": [float(p) for p in pred]}
    print(json.dumps(st["A"]["aare"], indent=2))
    # pol2 sensitivity: N2 handbook value vs CH4 value at 3 corners
    sens = []
    for (T, Patm, m) in [(324.65, 600.0, 4.0), (398.15, 600.0, 4.0),
                         (324.65, 100.0, 1.0)]:
        x_n2pol = _predict_point((kij[0], kij[1], kij[2], 1.0, POL_N2,
                                  T, Patm * ATM_TO_BAR, m))
        x_ch4pol = _predict_point((kij[0], kij[1], kij[2], 1.0, POL_CH4,
                                   T, Patm * ATM_TO_BAR, m))
        sens.append({"T_K": T, "P_atm": Patm, "m": m,
                     "x_pol_n2": x_n2pol, "x_pol_ch4": x_ch4pol,
                     "rel_shift_pct": (x_ch4pol / x_n2pol - 1.0) * 100.0})
        print(f"  pol2 sens T={T} P={Patm} m={m}: "
              f"{sens[-1]['rel_shift_pct']:.4f}% shift for pol2 {POL_N2} "
              f"-> {POL_CH4}")
    st["pol2_sensitivity"] = sens


def stage_curves(st):
    import pandas as pd
    kijB = st["fitB"]["best"]["x"]
    kij0 = st["kij_mapping"]["AkBkCk"]
    sA = st["fitA"]["best"]["x"][0]
    P_atm = np.geomspace(85.0, 650.0, 12)
    rows = []
    for T in (324.65, 375.65, 398.15):
        for m in (0.0, 1.0, 4.0):
            for framing, kij, s in [("B0", kij0, 1.0), ("B", kijB, 1.0),
                                    ("A", kijB, sA)]:
                tasks = [(kij[0], kij[1], kij[2], s, POL_N2, T,
                          p * ATM_TO_BAR, m) for p in P_atm]
                if _POOL is not None:
                    preds = _POOL.map(_predict_point, tasks)
                else:
                    preds = [_predict_point(t) for t in tasks]
                for p, x in zip(P_atm, preds):
                    rows.append(dict(T_K=T, m=m, framing=framing,
                                     P_atm=p, x_pred=x))
            print(f"  curves T={T} m={m} done", flush=True)
    pd.DataFrame(rows).to_csv(CURVES_CSV, index=False)
    st["curves_csv"] = str(CURVES_CSV)


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
    STYLE = {"B0": (WONG[1], "--"), "B": (WONG[5], "-"), "A": (WONG[3], "-.")}
    LABEL = {"B0": "B0 zero-fit", "B": "B water-fit kij",
             "A": "A + brine-fit ions"}

    df = load_n2()
    cur = pd.read_csv(CURVES_CSV)
    Ts = [324.65, 375.65, 398.15]
    ms = [0.0, 1.0, 4.0]
    mlab = {0.0: "H$_2$O", 1.0: "1 m NaCl", 4.0: "4 m NaCl"}

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 11.5), sharex=True)
    for i, m in enumerate(ms):
        for j, T in enumerate(Ts):
            ax = axes[i, j]
            sub = df[(np.isclose(df.m_NaCl_mol_kg, m)) &
                     (np.isclose(df.T_K, T))]
            ax.plot(sub.P_atm * ATM_TO_BAR, sub.x_molefrac * 1e4, "o",
                    color="black", mfc="white", mew=1.8, zorder=5,
                    linestyle="none", label="O'Sullivan 1970")
            for fr in ("B0", "B", "A"):
                c = cur[(cur.framing == fr) & (np.isclose(cur.T_K, T)) &
                        (np.isclose(cur.m, m))].sort_values("P_atm")
                col, ls = STYLE[fr]
                ax.plot(c.P_atm * ATM_TO_BAR, c.x_pred * 1e4, ls,
                        color=col, label=LABEL[fr])
            ax.text(0.05, 0.95, f"{mlab[m]}\n{T - 273.15:.1f} $^\\circ$C",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=12, fontweight="bold", color="black",
                    bbox=dict(facecolor="white", alpha=0.85,
                              edgecolor="none", pad=2))
            if i == 2:
                ax.set_xlabel("P [bar]")
            if j == 0:
                ax.set_ylabel(r"$10^4\,x_{\mathrm{N_2}}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 2].legend(handles, labels, loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_PNG)
    print("figure ->", FIG_PNG)
    st["figure"] = str(FIG_PNG.resolve().relative_to(Path(__file__).resolve().parents[3]))


def stage_finalize(st):
    # In-process guard: rebind -> restore -> spot flash, plus cross-stage
    # comparison against the pre-pilot refs from stage `refs`.
    pre = {c: guard_spot(c) for c in GUARD_CASES}
    try:
        rebind_n2(*st["fitB"]["best"]["x"], s_nacl=st["fitA"]["best"]["x"][0])
        _ = _predict_point((st["fitB"]["best"]["x"][0],
                            st["fitB"]["best"]["x"][1],
                            st["fitB"]["best"]["x"][2],
                            st["fitA"]["best"]["x"][0], POL_N2,
                            324.65, 100 * ATM_TO_BAR, 1.0))
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

    df = load_n2()
    table = {}
    for fr in ("B0", "B", "A"):
        table[fr] = {k: st[fr]["aare"][k]["aare_pct"]
                     for k in ("water", "1m", "4m", "brine_all", "all")}
    st["comparison_table_aare_pct"] = table
    print("\nAARE%% table (framing x subset):")
    print(f"{'':6s} {'water':>8s} {'1m':>8s} {'4m':>8s} {'brine':>8s} {'all':>8s}")
    for fr in ("B0", "B", "A"):
        r = table[fr]
        print(f"{fr:6s} {r['water']:8.2f} {r['1m']:8.2f} {r['4m']:8.2f} "
              f"{r['brine_all']:8.2f} {r['all']:8.2f}")

    # Tidy final deliverable JSON
    final = {
        "pilot": "n2_pilot_2026_07",
        "data": {
            "source": "O'Sullivan & Smith 1970 Table I "
                      "(pilot_studies/n2_scoping_2026_07/"
                      "n2_nacl_osullivan1970.csv)",
            "n_points": {"water": 18, "1m": 18, "4m": 15},
            "T_K": [324.65, 375.65, 398.15],
            "P_atm": [100, 608],
            "basis": "X2 with NaCl undissociated (authors' convention); "
                     "model converted via X2 = x2w/(x1w + x2w + Mw*x1w*m)",
        },
        "n2_constants": {"Tc_K": TC_N2, "Pc_Pa": PC_N2, "omega": W_N2,
                         "M_kg_mol": M_N2,
                         "pol2_Cm2_J": POL_N2,
                         "pol2_note": "handbook alpha(N2)=1.7403 A^3, "
                                      "citation to be added; no N2 value "
                                      "in on-disk PDFs"},
        "transfer": {"Tc_rescale": TC_RATIO,
                     "rule": "A_N2,i = 0.66222*A_CH4,i, B unchanged "
                             "(function-preserving under U2i = "
                             "A*(T/Tc2) + B)"},
        "kij_mapping": st["kij_mapping"],
        "framings": {fr: {k: v for k, v in st[fr].items() if k != "pred"}
                     for fr in ("B0", "B", "A")},
        "fitB": st["fitB"], "fitA": st["fitA"],
        "comparison_table_aare_pct": table,
        "pol2_sensitivity": st["pol2_sensitivity"],
        "guards": guards,
        "predictions_per_point": {
            "columns": ["T_K", "P_atm", "m_NaCl", "x_exp",
                        "x_B0", "x_B", "x_A"],
            "rows": [[float(r.T_K), float(r.P_atm),
                      float(r.m_NaCl_mol_kg), float(r.x_molefrac),
                      st["B0"]["pred"][i], st["B"]["pred"][i],
                      st["A"]["pred"][i]]
                     for i, r in enumerate(df.itertuples())],
        },
        "notes": [
            "DH/colligative terms untouched by construction: the slot swap "
            "is purely parametric; the Debye-Hueckel and ion-association "
            "terms have zero gas dependence (scoping audit).",
            "k23 (N2-CO2) inert in this pilot (z_CO2 = 0 throughout).",
        ],
    }
    RESULTS_JSON.write_text(json.dumps(final, indent=2))
    print("\nfinal ->", RESULTS_JSON)


STAGES = {
    "refs": stage_refs, "b0": stage_b0, "fitB": stage_fitB,
    "evalB": stage_evalB, "fitA": stage_fitA, "evalA": stage_evalA,
    "curves": stage_curves, "figure": stage_figure,
    "finalize": stage_finalize,
}
POOL_STAGES = {"b0", "fitB", "evalB", "fitA", "evalA", "curves"}


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
